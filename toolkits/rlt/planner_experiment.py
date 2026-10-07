# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Launch an isolated single-GPU planner pilot without touching other jobs."""

import argparse
import fcntl
import hashlib
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
    arm: str,
    steps: int,
    eval_episodes: int,
    gpu: int = 2,
    *,
    seed: int = 1234,
    protocol: str = "complete",
    control_budget: int = 0,
    update_budget: int = 0,
    eval_scope: str = "full_task",
    checkpoint: Path | None = None,
    initial_weights: Path | None = None,
    export_dir: Path | None = None,
    eval_seeds: list[int] | None = None,
) -> list[str]:
    """Keep both arms matched and reserve distinct evaluation initial states."""
    if arm not in {"none", "planner"} or steps < 1 or not 1 <= eval_episodes <= 500:
        raise ValueError("Require a known arm, positive steps and 1..500 eval episodes")
    if gpu < 0:
        raise ValueError("Require a nonnegative physical GPU index")
    if seed < 0 or protocol not in {"complete", "preinsert_handoff"}:
        raise ValueError("Invalid seed or planner protocol")
    if min(control_budget, update_budget) < 0 or (control_budget and update_budget):
        raise ValueError("Choose at most one positive work budget")
    if eval_scope not in {"full_task", "insertion"}:
        raise ValueError("Unknown evaluation scope")
    if checkpoint is not None and (control_budget or update_budget or export_dir):
        raise ValueError("Checkpoint evaluation cannot train or export corrections")
    if checkpoint is not None and initial_weights is not None:
        raise ValueError("Choose evaluation or weights-only training initialization")
    seeds = (
        list(range(12026, 12026 + eval_episodes)) if eval_seeds is None else eval_seeds
    )
    if (
        len(seeds) != eval_episodes
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("Require one distinct nonnegative reset seed per episode")
    overrides = [
        "+experiment=rlt_planner_pilot",
        f"env.train.planner_assistance.enable={arm == 'planner'}",
        f"runner.max_epochs={steps}",
        f"runner.max_steps={steps}",
        f"runner.val_check_interval={steps}",
        f"runner.save_interval={steps}",
        f"actor.seed={seed}",
        f"env.train.seed={seed}",
        f"+env.train.planner_assistance.protocol={protocol}",
        # RLinf enumerates physical devices independently of the CUDA mask.
        f"cluster.component_placement.actor={gpu}-{gpu}",
        f"cluster.component_placement.rollout={gpu}-{gpu}",
        f"cluster.component_placement.env={gpu}-{gpu}",
    ]
    if control_budget or update_budget:
        overrides += [
            f"+runner.rlt_experiment_budget={{control_limit:{control_budget},update_limit:{update_budget}}}",
            f"+env.train.training_control_budget={control_budget}",
            f"+algorithm.rlt_schedule.total_update_limit={update_budget}",
        ]
    if eval_scope == "insertion":
        overrides += [
            "+env.eval.insertion_fixture=True",
            "env.eval.rlt_policy_switch.task_mode=critical_phase",
        ]
    if checkpoint is not None:
        overrides += [
            "runner.only_eval=True",
            f"runner.ckpt_path={checkpoint.resolve()}",
            "algorithm.rlt_schedule.enable=False",
        ]
    if initial_weights is not None:
        overrides += [
            f"runner.ckpt_path={initial_weights.resolve()}",
            "runner.resume_dir=null",
        ]
    if export_dir is not None:
        overrides += [f"+algorithm.correction_export_dir={export_dir.resolve()}"]
    return overrides + [
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


def fixture_population(path: Path, candidates: list[int]) -> dict:
    """Declare a conditional insertion population before inspecting any policy."""
    raw = path.read_bytes()
    report = json.loads(raw)
    rows = report["results"]
    if (
        len(rows) != len(candidates)
        or sorted(row["seed"] for row in rows) != sorted(candidates)
        or any(
            row["case"] != "insertion_fixture"
            or row["source"] != "planner_prefix_then_constant_control"
            or type(row["fixture_ready"]) is not bool
            or type(row["success"]) is not bool
            for row in rows
        )
    ):
        raise ValueError("Fixture probe must cover the exact candidate population")
    if any(row["success"] for row in rows):
        raise ValueError("Fixture constant-command control already solves the task")
    selected = [row["seed"] for row in rows if row["fixture_ready"]]
    if not selected or any(
        row["failure"] != "none" or not 0 < row["prefix_steps"] < 500
        for row in rows
        if row["fixture_ready"]
    ):
        raise ValueError("No valid, nonterminal insertion fixtures")
    return {
        "source": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_seeds": candidates,
        "selected_seeds": selected,
        "excluded": [
            {"seed": row["seed"], "reason": row["failure"]}
            for row in rows
            if not row["fixture_ready"]
        ],
        "coverage": len(selected) / len(candidates),
        "scope": "Conditional on policy-independent fixture feasibility; not full-task success",
    }


def validate_diagnostic_checkpoint(
    checkpoint: Path, model: dict, features: dict
) -> None:
    """Reject a correction-fit head evaluated with different frozen features."""
    metadata = checkpoint.parent.parent / "contract.json"
    if not metadata.is_file():
        return  # Ordinary distributed checkpoints do not use this cache schema.
    contract = json.loads(metadata.read_text())
    if (
        contract.get("format") != "rlt_corrections_v1"
        or contract.get("model") != model
        or contract.get("feature_model") != features
    ):
        raise ValueError(
            "Diagnostic checkpoint requires its original model/features/normalization contract"
        )


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
    parser.add_argument(
        "--fixture-report",
        type=Path,
        help="Select insertion fixtures from a complete policy-independent probe; retain exclusions",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--protocol", choices=["complete", "preinsert_handoff"], default="complete"
    )
    parser.add_argument("--control-budget", type=int, default=0)
    parser.add_argument("--update-budget", type=int, default=0)
    parser.add_argument(
        "--eval-scope", choices=["full_task", "insertion"], default="full_task"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate a weights-only Stage 2 file; never resume training",
    )
    parser.add_argument("--export-corrections", action="store_true")
    parser.add_argument(
        "--checkpoint-run",
        type=Path,
        help="Evaluate the final budget checkpoint of a completed pilot",
    )
    parser.add_argument(
        "--initial-weights",
        type=Path,
        help="Warm-start training weights only; fresh replay, optimizer and counters",
    )
    parser.add_argument("--scratch-root", type=Path, default=Path("/dev/shm"))
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log this pilot to W&B as well as local TensorBoard",
    )
    args = parser.parse_args()
    if args.checkpoint_run is not None:
        if args.checkpoint is not None:
            parser.error("Choose --checkpoint or --checkpoint-run")
        launch = json.loads((args.checkpoint_run / "launch.json").read_text())
        if launch.get("complete") is not True or launch.get("exit_code") != 0:
            parser.error("--checkpoint-run requires completed training")
        from toolkits.rlt.planner_campaign import final_weights

        args.checkpoint = final_weights(args.checkpoint_run)
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("Evaluation checkpoint must be an existing weights file")
    eval_seeds = list(range(12026, 12026 + args.eval_episodes))
    population = None
    if args.fixture_report is not None:
        if args.eval_scope != "insertion" or args.checkpoint is None:
            parser.error("--fixture-report is only for standalone insertion evaluation")
        population = fixture_population(args.fixture_report, eval_seeds)
        eval_seeds = population["selected_seeds"]
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

    overrides = experiment_overrides(
        args.arm,
        args.steps,
        len(eval_seeds),
        args.gpu,
        seed=args.seed,
        protocol=args.protocol,
        control_budget=args.control_budget,
        update_budget=args.update_budget,
        eval_scope=args.eval_scope,
        checkpoint=args.checkpoint,
        initial_weights=args.initial_weights,
        export_dir=args.output / "corrections" if args.export_corrections else None,
        eval_seeds=eval_seeds,
    )
    if args.wandb:
        overrides += ["runner.logger.logger_backends=[tensorboard,wandb]"]
    # Keep metric-service sockets and files off NAS; checkpoints remain in output.
    run_id = hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:12]
    metric_root = Path.home() / "rlinf_rlt/run_metrics" / f"{args.output.name}_{run_id}"
    overrides += [f"+runner.logger.backend_log_path={metric_root}"]
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="maniskill_rlt_stage2_ac_mlp",
            overrides=overrides,
        )
    from omegaconf import OmegaConf

    OmegaConf.resolve(cfg)
    initial_sha = None
    checkpoint_sha = None
    if args.initial_weights is not None:
        from toolkits.rlt.planner_stage2 import validate_weights

        initial_sha = validate_weights(args.initial_weights, cfg.actor.model)
    if args.checkpoint is not None:
        from toolkits.rlt.planner_stage2 import validate_weights

        checkpoint_sha = validate_weights(args.checkpoint, cfg.actor.model)
        validate_diagnostic_checkpoint(
            args.checkpoint,
            OmegaConf.to_container(cfg.actor.model, resolve=True),
            OmegaConf.to_container(cfg.rollout.rlt_feature_model, resolve=True),
        )
    if (
        cfg.env.eval.planner_assistance.enable
        or cfg.env.eval.rlt_policy_switch.expert_takeover.enable
    ):
        parser.error("Evaluation must never use an expert")
    if len(cfg.env.eval.evaluation_reset_seeds) != cfg.env.eval.rollout_epoch:
        parser.error("Evaluation requires one explicit initial-state seed per episode")
    print(
        f"Config/path preflight passed: eval scope={args.eval_scope}, scored states={len(eval_seeds)}, candidate states={args.eval_episodes}; planner off during scored execution. No GPU execution tested.",
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
        "seed": args.seed,
        "protocol": args.protocol,
        "control_budget": args.control_budget,
        "update_budget": args.update_budget,
        "eval_scope": args.eval_scope,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_sha,
        "initial_weights": str(args.initial_weights.resolve())
        if args.initial_weights
        else None,
        "initial_weights_sha256": initial_sha,
        "initialization": "evaluation"
        if args.checkpoint
        else "weights_only_fresh_state"
        if args.initial_weights
        else "seeded_random",
        "steps": args.steps,
        "eval_seeds": eval_seeds,
        "fixture_population": population,
        "gpu": args.gpu,
        "stage1": str(args.stage1.resolve()),
        "dataset": str(args.dataset.resolve()),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_diff": subprocess.check_output(["git", "diff"], cwd=root, text=True),
        "complete": False,
        "source_archive": "source.tar.gz",
    }
    from toolkits.rlt.planner_stage2 import snapshot_source

    runtime = temp / "source"
    snapshot_source(root, runtime, args.output / "source.tar.gz")
    root = runtime
    os.environ["PYTHONPATH"] = str(root)
    os.environ["EMBODIED_PATH"] = str(root / "examples/embodiment")
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
                str(
                    root
                    / (
                        "evaluations/eval_embodied_agent.py"
                        if args.checkpoint
                        else "examples/embodiment/train_embodied_agent.py"
                    )
                ),
                "--config-path",
                str(root / "examples/embodiment/config"),
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
            if args.export_corrections:
                from rlinf.algorithms.rlt.correction_data import finalize_corrections

                # Mark complete only if both training and sealing succeeded.
                status = None
                finalize_corrections(args.output / "corrections")
                status = 0
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
