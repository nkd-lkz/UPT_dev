# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Launch an isolated single-GPU planner pilot without touching other jobs."""

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def experiment_overrides(
    arm: str, steps: int, eval_episodes: int, gpu: int = 2
) -> list[str]:
    """Keep both arms matched and reserve distinct evaluation initial states."""
    if arm not in {"none", "planner"} or steps < 1 or not 1 <= eval_episodes <= 20:
        raise ValueError("Require a known arm, positive steps and 1..20 eval episodes")
    if gpu < 0:
        raise ValueError("Require a nonnegative physical GPU index")
    seeds = list(range(12026, 12026 + eval_episodes))
    return [
        "+experiment=rlt_planner_pilot",
        f"env.train.planner_assistance.enable={arm == 'planner'}",
        f"runner.max_epochs={steps}",
        f"runner.max_steps={steps}",
        f"runner.val_check_interval={steps}",
        f"runner.save_interval={steps}",
        # RLinf enumerates physical devices independently of the CUDA mask.
        f"cluster.component_placement.actor={gpu}-{gpu}",
        f"cluster.component_placement.rollout={gpu}-{gpu}",
        f"cluster.component_placement.env={gpu}-{gpu}",
        f"env.eval.rollout_epoch={eval_episodes}",
        f"env.eval.evaluation_reset_seeds={seeds}",
    ]


def validate_storage(output: Path, scratch: Path) -> None:
    """Check the filesystems that actually receive outputs, not the code mount."""
    if output.exists():
        raise ValueError(f"Output must be new: {output}")
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if not scratch.is_dir() or shutil.disk_usage(scratch).free < 4 * 1024**3:
        raise ValueError(
            "Scratch directory needs 4 GiB free for private Ray/temp files"
        )
    if shutil.disk_usage(parent).free < 5 * 1024**3:
        raise ValueError("Output filesystem needs 5 GiB free for checkpoint and logs")


def stop_owned_process(process: subprocess.Popen | None) -> None:
    """Stop only this launcher's process tree, including private Ray workers."""
    import psutil

    if process is None or process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        owned = parent.children(recursive=True) + [parent]
        for child in owned:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned, timeout=10)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.wait(timeout=10)
    except (psutil.NoSuchProcess, subprocess.TimeoutExpired):
        pass


def main() -> None:
    """Compose first; require an idle GPU and use a private Ray head for execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--arm", choices=["none", "planner"], default="planner")
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=6535)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--scratch-root", type=Path, default=Path("/dev/shm"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    for path in [
        args.stage1 / "model_state_dict/full_weights.pt",
        args.dataset / "norm_stats.json",
    ]:
        if not path.is_file():
            parser.error(f"Missing {path}")
    os.environ.update(
        RLT_STAGE1_ACTOR=str(args.stage1),
        RLT_DATASET_DIR=str(args.dataset),
        RLT_PLANNER_OUTPUT=str(args.output),
        RLT_PLANNER_NAME=f"planner_pilot_{args.arm}",
        EMBODIED_PATH=str(root / "examples/embodiment"),
    )
    os.environ.setdefault("RLT_PLANNER_RENDER_DEVICE", "cuda:0")
    from hydra import compose, initialize_config_dir

    overrides = experiment_overrides(args.arm, args.steps, args.eval_episodes, args.gpu)
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="maniskill_rlt_stage2_ac_mlp",
            overrides=overrides,
        )
    from omegaconf import OmegaConf

    OmegaConf.resolve(cfg)
    if (
        cfg.env.eval.planner_assistance.enable
        or cfg.env.eval.rlt_policy_switch.expert_takeover.enable
    ):
        parser.error("Evaluation must never use an expert")
    if len(cfg.env.eval.evaluation_reset_seeds) != cfg.env.eval.rollout_epoch:
        parser.error("Evaluation requires one explicit initial-state seed per episode")
    print(
        "Config/path preflight passed: 1 train env, disjoint eval seed, planner off for eval. No GPU execution tested.",
        flush=True,
    )
    if args.check:
        return
    validate_storage(args.output, args.scratch_root)
    # Hold a cooperative per-GPU lease until our private workers exit.
    lease = (args.scratch_root / f"rlt-planner-gpu{args.gpu}.lock").open("a")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"GPU {args.gpu} already has a planner launch")
    used = int(
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    )
    if used > 512:
        parser.error(f"GPU {args.gpu} is busy ({used} MiB); refusing to start")
    for port in (args.port, args.port + 1, args.port + 2):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    bus = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=pci.bus_id",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .lower()
    )
    domain, bus_id, slot = bus.split(":")
    os.environ["RLT_PLANNER_RENDER_DEVICE"] = (
        f"pci:{int(domain, 16):04x}:{bus_id}:{slot}"
    )
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(root),
        HYDRA_FULL_ERROR="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    for name in ("RAY_ADDRESS", "RLINF_NODE_RANK", "DISPLAY"):
        os.environ.pop(name, None)
    vk = Path(os.environ.get("RLINF_VULKAN_PREFIX", "/home/luokz/.local/rlinf-vulkan"))
    if vk.is_dir():
        os.environ["LD_LIBRARY_PATH"] = f"{vk}/lib:/usr/lib/x86_64-linux-gnu"
        os.environ["SAPIEN_VULKAN_LIBRARY_PATH"] = str(vk / "lib/libvulkan.so.1.4.357")
        os.environ["VK_DRIVER_FILES"] = str(
            vk / "share/vulkan/icd.d/nvidia_headless_icd.json"
        )
        os.environ["VK_ICD_FILENAMES"] = os.environ["VK_DRIVER_FILES"]
    args.output.mkdir(parents=True, exist_ok=False)
    temp = Path(tempfile.mkdtemp(prefix="rlt-planner.", dir=args.scratch_root))
    os.environ["RAY_TMPDIR"] = str(temp)
    os.environ["TMPDIR"] = str(temp)
    os.environ["TRITON_CACHE_DIR"] = str(temp / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(temp / "inductor")
    ray = str(Path(sys.executable).parent / "ray")
    child = None
    status = None
    metadata = {
        "arm": args.arm,
        "steps": args.steps,
        "eval_seeds": list(range(12026, 12026 + args.eval_episodes)),
        "gpu": args.gpu,
        "stage1": str(args.stage1.resolve()),
        "dataset": str(args.dataset.resolve()),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_diff": subprocess.check_output(["git", "diff"], cwd=root, text=True),
        "complete": False,
    }
    (args.output / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (
        (args.output / "ray.log").open("w") as ray_log,
        (args.output / "train.log").open("w") as train_log,
    ):
        head = subprocess.Popen(
            [
                ray,
                "start",
                "--head",
                "--block",
                f"--port={args.port}",
                f"--ray-client-server-port={args.port + 1}",
                f"--dashboard-port={args.port + 2}",
                "--dashboard-host=127.0.0.1",
                "--dashboard-agent-listen-port=0",
                "--num-gpus=1",
                "--num-cpus=6",
                "--object-store-memory=2147483648",
                f"--temp-dir={temp}",
                f"--object-spilling-directory={args.output / 'ray_spill'}",
                "--disable-usage-stats",
            ],
            stdout=ray_log,
            stderr=subprocess.STDOUT,
        )

        def terminate(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        signal.signal(signal.SIGTERM, terminate)
        try:
            for _ in range(120):
                if head.poll() is not None:
                    raise RuntimeError("Private Ray head exited; inspect ray.log")
                address = temp / "ray_current_cluster"
                if address.exists() and address.read_text().strip():
                    os.environ["RAY_ADDRESS"] = address.read_text().strip()
                    if not os.environ["RAY_ADDRESS"].endswith(f":{args.port}"):
                        raise RuntimeError("Unexpected private Ray address")
                    break
                time.sleep(1)
            else:
                raise TimeoutError("Private Ray startup timed out")
            command = [
                sys.executable,
                str(root / "examples/embodiment/train_embodied_agent.py"),
                "--config-name",
                "maniskill_rlt_stage2_ac_mlp",
                *overrides,
            ]
            with (args.output / "resolved.yaml").open("w") as config_file:
                subprocess.run(
                    command + ["--cfg", "job", "--resolve"],
                    stdout=config_file,
                    cwd=temp,
                    check=True,
                )
            child = subprocess.Popen(
                command, cwd=temp, stdout=train_log, stderr=subprocess.STDOUT
            )
            print(f"Training log: {args.output / 'train.log'}", flush=True)
            status = child.wait()
            if status:
                raise RuntimeError(f"Pilot exited with status {status}")
        finally:
            for process in (child, head):
                stop_owned_process(process)
            metadata.update(complete=status == 0, exit_code=status)
            (args.output / "launch.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            print(f"Retained Ray diagnostics: {temp}", flush=True)


if __name__ == "__main__":
    main()
