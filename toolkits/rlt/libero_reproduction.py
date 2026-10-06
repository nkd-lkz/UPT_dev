# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Pinned AlphaBrain RLT_a release evaluation, separate from RLinf RLT.

The two arms use the author's same evaluator, initialization IDs, action
normalization and chunk length. This is a published-checkpoint reproduction;
its benchmark initial states are not claimed to be unseen training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SOURCE_REVISION = "604924beb77b04b0da49326dfae6ea423a27d28a"
TRAINING_REVISION = "f1d27bd06597a15ec1104133fe7f13848a5b2849"
TRAINING_TREE = "1acfcd5ff4e54d0648715ea4387b500d1ee0e85f"
ASSETS = {
    "vla": (
        "AlphaBrainGroup/qwenoft-5traj-libero-goal",
        "91ea0817f556bf50d64008187519de19458f8188",
    ),
    "rlt_a": (
        "AlphaBrainGroup/alphabrain-rlt-5traj-alltasks-libero-goal",
        "b5c0080e46f4b900c049e76bd9f3df68befac12a",
    ),
}


def episode_plan(tasks: list[int], states: list[int]) -> list[tuple[int, int]]:
    """Validate explicit IDs; never silently wrap a 51st episode to state zero."""
    for values, upper, name in [(tasks, 10, "task"), (states, 50, "initial state")]:
        if (
            not values
            or len(values) != len(set(values))
            or any(v < 0 or v >= upper for v in values)
        ):
            raise ValueError(f"Require unique {name} IDs in [0, {upper})")
    return [(task, state) for task in tasks for state in states]


def paired_summary(reference: list[dict], learned: list[dict]) -> dict:
    """Compare outcomes only when every task/state pair occurs once in both arms."""

    def keyed(rows):
        result = {}
        for row in rows:
            key = (row["task"], row["state"])
            if key in result or not isinstance(row["success"], bool):
                raise ValueError("Duplicate identity or non-boolean success")
            result[key] = row["success"]
        return result

    a, b = keyed(reference), keyed(learned)
    if not a or a.keys() != b.keys():
        raise ValueError("Cannot compare incomplete or unmatched episodes")
    return {
        "episodes_per_arm": len(a),
        "reference_success": sum(a.values()) / len(a),
        "rlt_a_success": sum(b.values()) / len(b),
        "rlt_a_only_successes": sum(b[k] and not a[k] for k in a),
        "reference_only_successes": sum(a[k] and not b[k] for k in a),
        "both_successes": sum(a[k] and b[k] for k in a),
        "both_failures": sum(not a[k] and not b[k] for k in a),
        "limitation": "One checkpoint and training seed; published benchmark states, not a generalization claim.",
    }


def check_source(source: Path, *, training: bool = False) -> str:
    """Require the audited upstream revision and a clean tracked source tree."""
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    # Applying the distributed patch with git am changes committer metadata, but
    # not the audited source tree. Evaluation still requires untouched upstream.
    tree = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    matches = tree == TRAINING_TREE if training else actual == SOURCE_REVISION
    expected = f"training tree {TRAINING_TREE}" if training else SOURCE_REVISION
    if not matches or dirty:
        raise ValueError(f"Need clean AlphaBrain revision {expected}; got {actual}")
    return actual


def asset_paths(storage: Path) -> tuple[Path, Path]:
    """Return explicit release paths, avoiding upstream directory-name discovery."""
    return storage / "vla", storage / "rlt_a/rl_offpolicy_iter_00400"


def training_arguments(
    storage: Path, output: Path, task: int, iterations: int
) -> list[str]:
    """Build a bounded single-GPU TD3 pilot with a fresh actor and critic.

    The public action-token encoder stays frozen. This is not optimizer resume,
    and its small budgets are not the author's full benchmark training recipe.
    """
    if task not in range(10) or iterations < 1:
        raise ValueError("Require a LIBERO Goal task and positive iteration budget")
    vla, rlt = asset_paths(storage)
    options = {
        "phase": "rl_offpolicy",
        "encoder_mode": "action_token",
        "ckpt_path": vla,
        "encoder_path": rlt / "encoder.pt",
        "output_dir": output,
        "suite": "libero_goal",
        "task_id": task,
        "rollout_gpus": "0",
        "train_gpu": 0,
        "bottleneck_dim": 256,
        "encoder_layers": 2,
        "encoder_heads": 4,
        "actor_hidden_dim": 512,
        "critic_hidden_dim": 512,
        "ref_dropout": 0.5,
        "fixed_std": 0.1,
        "actor_chunk_len": 8,
        "G_per_task": 4,
        "group_size": 1,
        "num_envs_per_task": 2,
        "buffer_capacity": 20000,
        "buffer_warmup": 128,
        "warmup_iters": 5,
        "td_updates_per_iter": 128,
        "utd_ratio": 2.0,
        "td_batch_size": 128,
        "beta": 1.0,
        "reward_coef": 5.0,
        "max_iter": iterations,
        # Run independent paired evaluation after training, not concurrently.
        "eval_interval": 0,
        "save_interval": iterations,
        "save_video_interval": iterations,
        "log_interval": 1,
    }
    result = [
        item for key, value in options.items() for item in (f"--{key}", str(value))
    ]
    return result + ["--use_steplock"]


def train(args) -> dict:
    """Call the pinned trainer without shell-wide process cleanup or .env loading."""
    from AlphaBrain.training.reinforcement_learning.trainers.train_args import (
        parse_args,
    )
    from AlphaBrain.training.reinforcement_learning.trainers.train_rl_offpolicy import (
        run_rl_offpolicy,
    )

    options = training_arguments(
        args.storage, args.output, args.tasks[0], args.iterations
    )
    original = sys.argv
    try:
        sys.argv = [original[0], *options, "--seed", str(args.seed)]
        config = parse_args()
    finally:
        sys.argv = original
    (args.output / "training_arguments.json").write_text(
        json.dumps(vars(config), indent=2) + "\n"
    )
    run_rl_offpolicy(config)
    history = json.loads((args.output / "metrics.json").read_text())
    validate_training_history(history, args.iterations)
    checkpoint = (
        args.output / "checkpoints" / f"rl_offpolicy_iter_{args.iterations:05d}"
    )
    for name in ("encoder.pt", "actor.pt", "critic.pt"):
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f"Missing final training weights: {checkpoint / name}")
    return {
        "variant": "rlt_a",
        "iterations": args.iterations,
        "environment_steps": sum(row["iter_env_steps"] for row in history),
        "iterations_with_updates": sum("actor_loss" in row for row in history),
        "checkpoint": str(checkpoint),
        "initialization": "Published frozen encoder; fresh actor and critic",
        "autonomous_success": None,
        "next_action": "Evaluate the new checkpoint against the same frozen reference; training completion is not a success-rate result.",
    }


def validate_training_history(history: list[dict], iterations: int) -> None:
    """Reject apparent completion with missing iterations or no new interaction."""
    if [row.get("iter") for row in history] != list(range(1, iterations + 1)):
        raise ValueError("Training did not report every requested iteration")
    stalled = [row["iter"] for row in history if row.get("iter_env_steps", 0) <= 0]
    if stalled:
        raise ValueError(
            f"No new environment steps in iterations {stalled}; training is not accepted"
        )
    for row in history:
        for name, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(
                    f"Nonfinite training metric {name} at iteration {row['iter']}"
                )
    if iterations > 5 and not any("actor_loss" in row for row in history):
        raise ValueError("No actor updates observed after the pilot warmup budget")


def check_assets(storage: Path) -> list[str]:
    """List missing files without importing torch, opening CUDA or downloading."""
    vla, rlt = asset_paths(storage)
    paths = [
        vla / name
        for name in (
            "model.safetensors",
            "framework_config.yaml",
            "dataset_statistics.json",
            "qwen_pretrained/config.json",
            "qwen_pretrained/tokenizer.json",
        )
    ]
    paths += [rlt / name for name in ("encoder.pt", "actor.pt")]
    return [str(path) for path in paths if not path.is_file()]


def runtime_versions() -> dict[str, str | None]:
    """Record installed model/simulator packages without assuming distribution names."""
    result = {}
    for name in (
        "torch",
        "transformers",
        "libero",
        "rlinf-libero",
        "robosuite",
        "mujoco",
        "numpy",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def download_assets(storage: Path) -> None:
    """Download immutable public revisions into the explicitly chosen storage."""
    from huggingface_hub import snapshot_download

    for name, (repo, revision) in ASSETS.items():
        snapshot_download(
            repo, revision=revision, local_dir=storage / name, max_workers=2
        )
    (storage / "asset_revisions.json").write_text(json.dumps(ASSETS, indent=2) + "\n")


class ReferenceEncoder:
    """Keep the author's evaluation loop while bypassing token compression."""

    def eval(self):
        """Match the evaluator's inference lifecycle."""
        return self

    def encode(self, action_queries):
        """No representation is needed when executing the reference itself."""
        return None


class ReferenceActor:
    """Execute the frozen VLA proposal through the identical action pipeline."""

    def eval(self):
        """Match the evaluator's inference lifecycle."""
        return self

    def __call__(self, token, reference, proprio, *, deterministic: bool):
        """Preserve proposal values in the float32 format used by the evaluator."""
        if not deterministic:
            raise ValueError("Release comparison requires deterministic evaluation")
        # NumPy cannot represent bfloat16; widening preserves every proposal value.
        return reference.float(), None


def evaluate(args) -> dict:
    """Run serial paired evaluation with one frozen VLA resident on one GPU."""
    import random

    import numpy as np
    import torch
    from AlphaBrain.model.framework.base_framework import BaseFramework
    from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import (
        ActionTokenActor,
    )
    from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import (
        ActionTokenEncoderDecoder,
    )
    from AlphaBrain.training.reinforcement_learning.envs.libero_env import MAX_STEPS
    from AlphaBrain.training.reinforcement_learning.eval.eval_helpers import (
        _eval_deterministic_local,
    )

    vla_path, rlt_path = asset_paths(args.storage)
    if args.learner_dir is not None:
        rlt_path = args.learner_dir
    vla = (
        BaseFramework.from_pretrained(str(vla_path))
        .to(torch.bfloat16)
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    if len(vla.norm_stats) != 1:
        raise ValueError("Release must have one unambiguous action normalization entry")
    stats = next(iter(vla.norm_stats.values()))["action"]
    dim = vla.config.framework.action_model.action_dim
    if dim != 7 or vla.chunk_len != 8:
        raise ValueError("Audited Qwen release requires 8 x 7 actions")
    encoder = ActionTokenEncoderDecoder(
        input_dim=vla.qwen_vl_interface.model.config.hidden_size,
        bottleneck_dim=256,
        chunk_len=8,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=2,
    ).to("cuda:0")
    actor = ActionTokenActor(
        bottleneck_dim=256,
        action_dim=7,
        chunk_len=8,
        hidden_dim=512,
        ref_dropout=0.5,
        fixed_std=0.1,
        prop_dim=8,
    ).to("cuda:0")
    for model, name in [(encoder, "encoder"), (actor, "actor")]:
        model.load_state_dict(
            torch.load(rlt_path / f"{name}.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        model.eval().requires_grad_(False)
    outcomes = {}
    for arm, enc, act in [
        ("reference", ReferenceEncoder(), ReferenceActor()),
        ("rlt_a", encoder, actor),
    ]:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        rows = []
        for task in args.tasks:
            for state in args.states:
                result = _eval_deterministic_local(
                    frozen_vla=vla,
                    encoder=enc,
                    actor=act,
                    suite_name="libero_goal",
                    task_id=task,
                    action_norm_stats=stats,
                    max_steps=MAX_STEPS["libero_goal"],
                    chunk_len=8,
                    episode_indices=[state],
                    num_steps_wait=10,
                    seed=args.seed,
                    device="cuda:0",
                    video_dir=str(args.output / "videos" / arm / f"task_{task}")
                    if args.video
                    else None,
                )
                if len(result) != 1 or result[0][:2] != (state, state):
                    raise RuntimeError(
                        "Upstream returned an unexpected episode identity"
                    )
                rows.append(
                    {"task": task, "state": state, "success": bool(result[0][2])}
                )
                (args.output / f"{arm}.json").write_text(
                    json.dumps(rows, indent=2) + "\n"
                )
        outcomes[arm] = rows
    return paired_summary(outcomes["reference"], outcomes["rlt_a"])


def main() -> None:
    """Check, download or evaluate the published RLT_a variant explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "download", "evaluate", "train"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0])
    parser.add_argument("--states", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--learner-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--libero-python", default=sys.executable)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    source_revision = check_source(args.source, training=args.mode == "train")
    plan = episode_plan(args.tasks, args.states)
    if args.mode == "download":
        args.storage.mkdir(parents=True, exist_ok=True)
        download_assets(args.storage)
        return
    missing = check_assets(args.storage)
    if args.mode == "check":
        print(
            json.dumps(
                {
                    "source_revision": SOURCE_REVISION,
                    "variant": "AlphaBrain Qwen RLT_a, not RLinf pi0.5 RLT",
                    "episodes_per_arm": len(plan),
                    "missing_assets": missing,
                    "gpu_tested": False,
                },
                indent=2,
            )
        )
        if missing:
            raise SystemExit(2)
        return
    if missing:
        parser.error(f"Missing release files: {missing}")
    if args.mode == "train" and len(args.tasks) != 1:
        parser.error("Training pilot supports exactly one task")
    if args.mode == "train" and args.learner_dir is not None:
        parser.error("Training starts fresh; --learner-dir is an evaluation option")
    if args.iterations < 1 or args.gpu < 0:
        parser.error("Require a positive iteration budget and nonnegative GPU index")
    if args.learner_dir is not None:
        for name in ("actor.pt", "encoder.pt"):
            if not (args.learner_dir / name).is_file():
                parser.error(f"Missing learner file: {args.learner_dir / name}")
    if args.output is None or args.output.exists():
        parser.error("--output must specify a new directory")
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
        parser.error(f"GPU {args.gpu} busy ({used} MiB); no model loaded")
    gpu_uuid = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    from toolkits.rlt.libero_egl import egl_device_index

    egl_index = egl_device_index(gpu_uuid)
    print(f"GPU {args.gpu}: {gpu_uuid}; matching EGL device {egl_index}", flush=True)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=gpu_uuid,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        MUJOCO_GL="egl",
        LIBERO_PYTHON=args.libero_python,
        TOKENIZERS_PARALLELISM="false",
    )
    os.environ["RLT_LIBERO_EGL_DEVICE_ID"] = str(egl_index)
    os.environ["PYTHONPATH"] = (
        str(args.source.resolve()) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    sys.path.insert(0, str(args.source.resolve()))
    from AlphaBrain.training.reinforcement_learning.envs import libero_env

    # The pinned worker mixes print() diagnostics with a binary stdout protocol.
    # Replace only its process entry, preserving the author's environment logic.
    os.environ["RLT_ALPHABRAIN_SOURCE"] = str(args.source.resolve())
    libero_env._WORKER_SCRIPT = str(Path(__file__).with_name("libero_worker.py"))
    if args.mode == "train":
        from AlphaBrain.training.reinforcement_learning.envs import persistent_env_pool

        persistent_env_pool._FAST_WORKER_SCRIPT = libero_env._WORKER_SCRIPT
    args.output.mkdir(parents=True)
    os.environ["RLT_LIBERO_WORKER_LOG_DIR"] = str(
        (args.output / "worker_logs").resolve()
    )
    manifest = {
        "complete": False,
        "source_revision": source_revision,
        "upstream_base_revision": SOURCE_REVISION,
        "assets": ASSETS,
        "tasks": args.tasks,
        "states": args.states,
        "seed": args.seed,
        "gpu": args.gpu,
        "gpu_uuid": gpu_uuid,
        "egl_device_index": egl_index,
        "variant": "rlt_a",
        "assistance": False,
        "mode": args.mode,
        "runtime_versions": runtime_versions(),
        "learner_dir": str(args.learner_dir) if args.learner_dir else None,
        "adapter_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("libero_reproduction.py", "libero_worker.py", "libero_egl.py")
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = train(args) if args.mode == "train" else evaluate(args)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["complete"] = True
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
