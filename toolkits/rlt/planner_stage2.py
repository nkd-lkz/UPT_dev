# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run a bounded two-GPU planner-assisted Stage 2 with online W&B logging."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from toolkits.rlt.planner_experiment import stop_owned_process, validate_storage


def training_overrides(
    steps: int, eval_episodes: int, eval_interval: int, save_interval: int
) -> list[str]:
    """Keep one CPU physics environment, fixed evaluation seeds, and two GPUs."""
    if min(steps, eval_interval, save_interval) < 1 or not 1 <= eval_episodes <= 500:
        raise ValueError("Require positive intervals/steps and 1..500 eval episodes")
    return [
        "+experiment=rlt_planner_stage2",
        f"runner.max_epochs={steps}",
        f"runner.max_steps={steps}",
        f"runner.val_check_interval={eval_interval}",
        f"runner.save_interval={save_interval}",
        f"env.eval.rollout_epoch={eval_episodes}",
        f"env.eval.evaluation_reset_seeds={list(range(12026, 12026 + eval_episodes))}",
        "+env.train.planner_assistance.protocol=complete",
    ]


def validate_contract(cfg) -> None:
    """Reject an assisted evaluation, unsupported environment, or hidden resume."""
    if dict(cfg.cluster.component_placement) != {
        "actor": "0-0",
        "rollout": "1-1",
        "env": "1-1",
    }:
        raise ValueError(
            "Only learner GPU 0 and rollout/environment GPU 1 are supported"
        )
    if cfg.runner.resume_dir or not cfg.runner.ckpt_path:
        raise ValueError("Require weights-only initialization, not a full resume")
    if not cfg.env.train.planner_assistance.enable:
        raise ValueError("Training planner must be enabled")
    for env in (cfg.env.train, cfg.env.eval):
        if env.total_num_envs != 1 or env.init_params.sim_backend != "cpu":
            raise ValueError("Planner launch requires one CPU physics environment")
        if env.rlt_policy_switch.expert_takeover.enable:
            raise ValueError("The model expert must be disabled")
    if cfg.env.eval.planner_assistance.enable:
        raise ValueError("Evaluation planner must be disabled")
    seeds = list(cfg.env.eval.evaluation_reset_seeds)
    if len(seeds) != cfg.env.eval.rollout_epoch or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation requires distinct explicit initial-state seeds")


def validate_weights(path: Path, model_cfg) -> str:
    """Check the full small actor/critic state on CPU and return its SHA-256."""
    import torch

    from rlinf.models.embodiment.mlp_policy import get_model

    weights = torch.load(path, map_location="cpu", weights_only=True)
    model = get_model(model_cfg)
    model.load_state_dict(weights, strict=True)
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Initial weights contain non-finite values")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot_source(root: Path, target: Path, archive: Path) -> None:
    """Freeze tracked and untracked runtime sources without modifying a worktree."""
    files = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "rlinf",
                "examples",
                "evaluations",
                "toolkits",
                "pyproject.toml",
            ],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    with tarfile.open(archive, "w:gz") as tar:
        for name in sorted(set(files) - {""}):
            source = root / name
            if source.is_file():
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                tar.add(destination, arcname=name, recursive=False)


def copy_initial_weights(source: Path, target: Path, expected_sha: str) -> None:
    """Copy content without requiring timestamp/chmod support on a CIFS share."""
    shutil.copyfile(source, target)
    with target.open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("Copied initial weights differ from the verified source")


def main() -> None:
    """Validate inputs, lease GPU 0/1, and own a private Ray process tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics-dir", type=Path, help="Local persistent W&B/TensorBoard directory"
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--port", type=int, default=6575)
    parser.add_argument("--wandb-entity", default="c6522513-sustech")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    for name in ("stage1", "dataset", "initial_weights", "output"):
        setattr(args, name, getattr(args, name).resolve())
    metrics_dir = (
        args.metrics_dir or Path.home() / "rlinf_rlt/run_metrics" / args.output.name
    ).resolve()
    for path in (
        args.stage1 / "model_state_dict/full_weights.pt",
        args.dataset / "norm_stats.json",
        args.initial_weights,
    ):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if not 1024 <= args.port <= 65533:
        parser.error("Port must be in 1024..65533")
    os.environ.update(
        RLT_STAGE1_ACTOR=str(args.stage1),
        RLT_DATASET_DIR=str(args.dataset),
        RLT_PLANNER_OUTPUT=str(args.output),
        RLT_PLANNER_NAME=args.output.name,
        RLT_PLANNER_INITIAL_WEIGHTS=str(args.initial_weights),
        RLT_PLANNER_METRICS=str(metrics_dir),
        RLT_PLANNER_RENDER_DEVICE="cuda:0",
        WANDB_ENTITY=args.wandb_entity,
        EMBODIED_PATH=str(root / "examples/embodiment"),
    )
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = training_overrides(
        args.steps, args.eval_episodes, args.eval_interval, args.save_interval
    )
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="maniskill_rlt_stage2_ac_mlp", overrides=overrides)
    OmegaConf.resolve(cfg)
    validate_contract(cfg)
    weights_sha = validate_weights(args.initial_weights, cfg.actor.model)
    print(f"Config and initial weights verified on CPU: {weights_sha}", flush=True)
    if args.check:
        print("No GPU, Ray, or W&B execution tested.", flush=True)
        return
    validate_storage(args.output, Path("/dev/shm"))
    validate_storage(metrics_dir, Path("/dev/shm"))
    with contextlib.ExitStack() as stack:
        for gpu in (0, 1):
            lease = stack.enter_context(
                Path(f"/dev/shm/rlt-planner-gpu{gpu}.lock").open("a")
            )
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            memory = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        str(gpu),
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            )
            if memory > 512:
                raise RuntimeError(f"GPU {gpu} busy ({memory} MiB); refusing to launch")
        for port in (args.port, args.port + 1, args.port + 2):
            with socket.socket() as sock:
                sock.bind(("0.0.0.0", port))
        import wandb

        api = wandb.Api(timeout=30)
        print(
            f"W&B authenticated: {api.viewer.username}; entity={args.wandb_entity}",
            flush=True,
        )
        bus = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    "1",
                    "--query-gpu=pci.bus_id",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .lower()
        )
        domain, bus_id, slot = bus.split(":")
        args.output.mkdir(parents=True, exist_ok=False)
        metrics_dir.mkdir(parents=True, exist_ok=False)
        temp = Path(tempfile.mkdtemp(prefix="rlt-planner-stage2.", dir="/dev/shm"))
        runtime = temp / "source"
        snapshot_source(root, runtime, args.output / "source.tar.gz")
        copy_initial_weights(
            args.initial_weights, args.output / "initial_weights.pt", weights_sha
        )
        copy_initial_weights(
            args.initial_weights, temp / "initial_weights.pt", weights_sha
        )
        os.environ.update(
            CUDA_VISIBLE_DEVICES="0,1",
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
            RLT_PLANNER_RENDER_DEVICE=f"pci:{int(domain, 16):04x}:{bus_id}:{slot}",
            PYTHONPATH=str(runtime),
            EMBODIED_PATH=str(runtime / "examples/embodiment"),
            RLT_PLANNER_INITIAL_WEIGHTS=str(temp / "initial_weights.pt"),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
            HYDRA_FULL_ERROR="1",
            TOKENIZERS_PARALLELISM="false",
            WANDB_MODE="online",
            WANDB__SERVICE_WAIT="120",
            WANDB_INIT_TIMEOUT="180",
            WANDB_RUN_GROUP="planner-assisted-weights-only",
            WANDB_RESUME="never",
            RAY_TMPDIR=str(temp),
            TMPDIR=str(temp),
            TRITON_CACHE_DIR=str(temp / "triton"),
            TORCHINDUCTOR_CACHE_DIR=str(temp / "inductor"),
        )
        for key in (
            "RAY_ADDRESS",
            "RLINF_NODE_RANK",
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "WANDB_RUN_ID",
        ):
            os.environ.pop(key, None)
        vk = Path(
            os.environ.get(
                "RLINF_VULKAN_PREFIX", str(Path.home() / ".local/rlinf-vulkan")
            )
        )
        if vk.is_dir():
            os.environ.update(
                LD_LIBRARY_PATH=f"{vk}/lib:/usr/lib/x86_64-linux-gnu",
                SAPIEN_VULKAN_LIBRARY_PATH=str(vk / "lib/libvulkan.so.1.4.357"),
                VK_DRIVER_FILES=str(vk / "share/vulkan/icd.d/nvidia_headless_icd.json"),
                VK_ICD_FILENAMES=str(
                    vk / "share/vulkan/icd.d/nvidia_headless_icd.json"
                ),
            )
        metadata = {
            "initialization": "weights_only_new_optimizer_replay_and_counters",
            "initial_weights": str(args.initial_weights),
            "initial_weights_sha256": weights_sha,
            "stage1": str(args.stage1),
            "dataset": str(args.dataset),
            "steps": args.steps,
            "eval_seeds": list(range(12026, 12026 + args.eval_episodes)),
            "protocol": "complete",
            "gpus": [0, 1],
            "ray_temp": str(temp),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "source_archive": "source.tar.gz",
            "backend_log_path": str(metrics_dir),
            "complete": False,
        }
        manifest = args.output / "launch.json"
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        head = child = None
        status = None

        def terminate(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        signal.signal(signal.SIGTERM, terminate)
        try:
            ray_log = stack.enter_context((args.output / "ray.log").open("w"))
            train_log = stack.enter_context((args.output / "train.log").open("w"))
            ray = str(Path(sys.executable).parent / "ray")
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
                    "--num-gpus=2",
                    "--num-cpus=8",
                    "--object-store-memory=2147483648",
                    f"--temp-dir={temp}",
                    f"--object-spilling-directory={args.output / 'ray_spill'}",
                    "--disable-usage-stats",
                ],
                stdout=ray_log,
                stderr=subprocess.STDOUT,
            )
            for _ in range(120):
                if head.poll() is not None:
                    raise RuntimeError("Private Ray exited; inspect ray.log")
                address = temp / "ray_current_cluster"
                if address.exists() and address.read_text().strip().endswith(
                    f":{args.port}"
                ):
                    os.environ["RAY_ADDRESS"] = address.read_text().strip()
                    break
                time.sleep(1)
            else:
                raise TimeoutError("Private Ray startup timed out")
            command = [
                sys.executable,
                str(runtime / "examples/embodiment/train_embodied_agent.py"),
                "--config-path",
                str(runtime / "examples/embodiment/config"),
                "--config-name",
                "maniskill_rlt_stage2_ac_mlp",
                *overrides,
            ]
            with (args.output / "resolved.yaml").open("w") as config_file:
                subprocess.run(
                    command + ["--cfg", "job", "--resolve"],
                    cwd=temp,
                    stdout=config_file,
                    check=True,
                )
            child = subprocess.Popen(
                command, cwd=temp, stdout=train_log, stderr=subprocess.STDOUT
            )
            metadata.update(
                launcher_pid=os.getpid(), ray_pid=head.pid, train_pid=child.pid
            )
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Training log: {args.output / 'train.log'}", flush=True)
            status = child.wait()
            if status:
                raise RuntimeError(f"Training exited with code {status}")
        finally:
            for process in (child, head):
                stop_owned_process(process)
            metadata.update(complete=status == 0, exit_code=status)
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Exit={status}; retained diagnostics: {temp}", flush=True)


if __name__ == "__main__":
    main()
