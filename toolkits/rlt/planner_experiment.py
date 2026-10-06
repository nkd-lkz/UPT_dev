# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Launch an isolated single-GPU planner pilot without touching other jobs."""

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


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

    override = f"env.train.planner_assistance.enable={args.arm == 'planner'}"
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="maniskill_rlt_stage2_ac_mlp",
            overrides=["+experiment=rlt_planner_pilot", override],
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
    if (
        shutil.disk_usage(root).free < 1024**3
        or shutil.disk_usage("/tmp").free < 1024**3
    ):
        parser.error(
            "Need 1 GiB free on code and /tmp filesystems before distributed training"
        )
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
    temp = Path(tempfile.mkdtemp(prefix="rlt-planner.", dir="/dev/shm"))
    os.environ["RAY_TMPDIR"] = str(temp)
    ray = str(Path(sys.executable).parent / "ray")
    child = None
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
                "+experiment=rlt_planner_pilot",
                override,
            ]
            with (args.output / "resolved.yaml").open("w") as config_file:
                subprocess.run(
                    command + ["--cfg", "job", "--resolve"],
                    stdout=config_file,
                    check=True,
                )
            child = subprocess.Popen(
                command, cwd=root, stdout=train_log, stderr=subprocess.STDOUT
            )
            print(f"Training log: {args.output / 'train.log'}", flush=True)
            status = child.wait()
            if status:
                raise RuntimeError(f"Pilot exited with status {status}")
        finally:
            for process in (child, head):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
            print(f"Retained Ray diagnostics: {temp}", flush=True)


if __name__ == "__main__":
    main()
