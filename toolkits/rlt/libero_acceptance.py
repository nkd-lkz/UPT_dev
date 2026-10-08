# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Collect, warm up, train and evaluate one auditable LIBERO RLT_a protocol.

All modes share observation preprocessing and action execution. The published
VLA/encoder stay frozen. Fresh small heads are accepted by BC before online RL.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import logging
import os
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from rlinf.utils.logging import get_logger
from toolkits.rlt.libero_reproduction import (
    ASSETS,
    ResizedVLA,
    asset_paths,
    check_source,
)

LOG = get_logger()
PROTOCOL = {
    "id": "libero-rlt-a-acceptance-v1",
    "variant": "rlt_a",
    "suite": "libero_goal",
    "image_size": 224,
    "image_resize": "cv2.INTER_AREA",
    "render_size": 256,
    "views": "both axes flipped by upstream parser",
    "chunk_len": 8,
    "max_control_steps": 320,
    "settling_steps": 10,
    "mujoco": "3.3.7",
    "robosuite": "1.4.1",
    "numpy": "1.26.4",
    "transformers": "4.53.2",
    "tokenizers": "0.21.4",
    "action_mapping": "upstream q99 unnormalize and gripper threshold; OSC clips [-1,1]",
    "gamma_per_tick": 0.99,
    "reward_scale": 5.0,
    "assistance": False,
    "encoder": "public frozen action-query encoder",
    "memory": False,
}


def digest(path: Path) -> str:
    """Hash an existing artifact without materializing a large file in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    """Atomically publish a complete status or manifest within its run directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_cases(path: Path) -> list[dict]:
    """Require unique explicit case IDs; generated cases cannot wrap modulo 50."""
    cases = json.loads(path.read_text())
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Require nonempty cases with unique IDs")
    for case in cases:
        if not 0 <= case["task"] < 10 or case["seed"] < 0:
            raise ValueError("Invalid task/seed")
        if case["kind"] == "published":
            if not 0 <= case["state"] < 50:
                raise ValueError("Invalid published initial state")
        elif case["kind"] == "generated":
            if not Path(case["snapshot"]).is_absolute():
                raise ValueError(
                    "Generated initial states need an absolute snapshot path"
                )
        else:
            raise ValueError("Unknown case source")
    return cases


def split_cache(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hold out the last fifth of whole collection episodes for following error."""
    ids = list(dict.fromkeys(r["case_id"] for r in rows))
    if len(ids) < 5:
        raise ValueError("Collect at least five episodes for held-out BC diagnostics")
    holdout = set(ids[-max(1, len(ids) // 5) :])
    return [r for r in rows if r["case_id"] not in holdout], [
        r for r in rows if r["case_id"] in holdout
    ]


def compatible(payload: dict) -> None:
    """Reject cross-protocol caches rather than silently reusing old scores."""
    if payload.get("protocol") != PROTOCOL or payload.get("assets") != ASSETS:
        raise ValueError("Cache/checkpoint protocol or frozen assets differ")


class RemoteEnvironment:
    """Own the upstream IPC proxy and the extended, private reset protocol."""

    def __init__(self) -> None:
        from AlphaBrain.training.reinforcement_learning.envs import libero_env

        libero_env._WORKER_SCRIPT = str(
            Path(__file__).with_name("libero_acceptance_env.py")
        )
        self.proxy = libero_env.LiberoEnv(libero_python=sys.executable)
        self.instruction = ""
        self.reset_hash = ""

    def reset(self, case: dict, *, create_snapshot: bool = False) -> dict:
        """Restore the declared state, returning only policy-visible observations."""
        from AlphaBrain.training.reinforcement_learning.envs.libero_env import (
            _check_resp,
            _parse_obs,
            _read_msg,
            _write_msg,
        )

        _write_msg(
            self.proxy._proc,
            {"cmd": "reset", "case": case, "create_snapshot": create_snapshot},
        )
        response = _read_msg(self.proxy._proc)
        _check_resp(response)
        self.instruction, self.reset_hash = (
            response["task_description"],
            response["state_sha256"],
        )
        return _parse_obs(response["obs"])

    def step(self, action):
        """Execute one real simulator tick through the upstream action interface."""
        return self.proxy.step(action)

    def close(self) -> None:
        """Idempotently reap this environment's worker process."""
        self.proxy.close()


class FrozenFeatures:
    """Extract public action-query embeddings with identical train/eval resizing."""

    def __init__(self, storage: Path) -> None:
        import torch
        from AlphaBrain.model.framework.base_framework import BaseFramework
        from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import (
            ActionTokenEncoderDecoder,
        )

        vla_path, rlt_path = asset_paths(storage)
        self.model = (
            BaseFramework.from_pretrained(str(vla_path))
            .to(torch.bfloat16)
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        if self.model.chunk_len != 8 or len(self.model.norm_stats) != 1:
            raise ValueError("Unexpected release action contract")
        self.vla = ResizedVLA(self.model, PROTOCOL["image_size"])
        self.stats = next(iter(self.model.norm_stats.values()))["action"]
        self.encoder = ActionTokenEncoderDecoder(
            input_dim=self.model.qwen_vl_interface.model.config.hidden_size,
            bottleneck_dim=256,
            chunk_len=8,
            num_heads=4,
            encoder_layers=2,
            decoder_layers=2,
        ).cuda()
        self.encoder.load_state_dict(
            torch.load(rlt_path / "encoder.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        self.encoder.eval().requires_grad_(False)

    def __call__(self, observation: dict, instruction: str) -> dict:
        """Return CPU features before executing any action from this observation."""
        import torch

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                queries, reference = self.vla.get_vla_action(
                    batch_images=[
                        [observation["primary_image"], observation["wrist_image"]]
                    ],
                    instructions=[instruction],
                )
            z = self.encoder.encode(queries)
        if reference.shape != (1, 8, 7) or z.shape != (1, 1, 256):
            raise ValueError("Unexpected RLT_a feature shape")
        row = {
            "z": z[0, 0].float().cpu(),
            "ref": reference[0].float().cpu(),
            "prop": torch.as_tensor(observation["state"], dtype=torch.float32).clone(),
        }
        if not all(torch.isfinite(value).all() for value in row.values()):
            raise FloatingPointError("Nonfinite frozen features")
        return row


def env_commands(action, stats: dict):
    """Apply the official normalization before the controller's final clipping."""
    import numpy as np
    from AlphaBrain.training.reinforcement_learning.common.rollout import (
        _postprocess_action,
        _unnormalize,
    )

    converted = _unnormalize(action.numpy().copy(), stats)
    return (
        np.stack([_postprocess_action(row) for row in converted])
        .clip(-1, 1)
        .astype(np.float32)
    )


class Metrics:
    """Always persist local metrics; mirror them to W&B with bounded startup."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.stream = (args.output / "metrics.jsonl").open("x", buffering=1)
        self.wandb = None
        if args.wandb != "disabled":
            import wandb

            try:
                self.wandb = wandb.init(
                    project="rlt-libero-acceptance",
                    group=args.group,
                    name=args.output.name,
                    dir=str(args.output),
                    mode=args.wandb,
                    config={
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    settings=wandb.Settings(init_timeout=45),
                )
            except Exception as error:
                LOG.warning(
                    "W&B unavailable (%s); continuing with local JSONL",
                    type(error).__name__,
                )
                write_json(
                    args.output / "wandb_unavailable.json",
                    {"error_type": type(error).__name__, "local_metrics": True},
                )
        if self.wandb:
            write_json(
                args.output / "wandb_run.json",
                {"id": self.wandb.id, "url": self.wandb.url, "mode": args.wandb},
            )

    def append(self, row: dict) -> None:
        """Flush each metric row independently of network availability."""
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        if self.wandb:
            self.wandb.log(row)

    def close(self) -> None:
        """Finish the local stream and the optional W&B run."""
        self.stream.close()
        if self.wandb:
            self.wandb.finish()


def checkpoint(path: Path, learner, replay: list[dict], payload: dict) -> None:
    """Atomically save small heads and replay; never duplicate frozen VLA weights."""
    import torch

    temporary = path.with_suffix(".tmp")
    torch.save(payload | {"learner": learner.state_dict(), "replay": replay}, temporary)
    temporary.replace(path)


def run_episodes(args, cases, features, learner, replay, metrics, payload) -> dict:
    """Execute the same closed loop for reference collection, training and eval."""
    import numpy as np
    import torch

    from toolkits.rlt.libero_audit import observation_record

    env = RemoteEnvironment()
    rows, episodes = [], []
    physical_steps = control_steps = settling_steps = 0
    next_save = args.save_steps
    training = args.mode == "train"
    sequence = itertools.cycle(cases) if training else iter(cases)
    start_actor = learner.actor_updates if learner else 0
    start_critic = learner.critic_updates if learner else 0
    trace = (args.output / "actions.jsonl").open("x", buffering=1)
    try:
        for episode, case in enumerate(sequence):
            if training and args.control_budget - physical_steps <= 10:
                break
            observation = env.reset(case)
            initial = observation_record(observation)
            initial_hash = env.reset_hash
            for _ in range(10):
                observation, _, done = env.step(
                    np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
                )
                physical_steps += 1
                settling_steps += 1
                if done:
                    raise RuntimeError("Environment terminated during settling")
            current = features(observation, env.instruction)
            ticks, success, done = 0, False, False
            first_version = learner.actor_updates if learner else -1
            limit = min(320, args.control_budget - physical_steps) if training else 320
            frames = []
            while ticks < limit:
                version = learner.actor_updates if learner else -1
                action = (
                    learner.command(current, deterministic=not training)
                    if learner
                    else current["ref"].clamp(-1, 1)
                )
                commands = env_commands(action, features.stats)
                executed, reward = [], 0.0
                for command in commands[: limit - ticks]:
                    observation, r, done = env.step(command)
                    reward += 5.0 * 0.99 ** len(executed) * r
                    executed.append(command.copy())
                    ticks += 1
                    control_steps += 1
                    physical_steps += 1
                    success = success or r > 0.5
                    if episode < args.video_episodes:
                        frames.append(np.asarray(observation["primary_image"]).copy())
                    if done:
                        break
                following = features(observation, env.instruction)
                row = current | {
                    "next_z": following["z"],
                    "next_ref": following["ref"],
                    "next_prop": following["prop"],
                    "action": action.clone(),
                    "executed": torch.from_numpy(np.stack(executed)),
                    "reward": reward,
                    "ticks": len(executed),
                    "terminal": bool(done),
                    "truncated": bool(not done and ticks == limit),
                    "case_id": case["id"],
                    "episode": episode,
                    "policy_version": version,
                    "source": "actor" if learner else "reference",
                }
                trace.write(
                    json.dumps(
                        {
                            "case_id": case["id"],
                            "episode": episode,
                            "tick": ticks - len(executed),
                            "policy_version": version,
                            "source": row["source"],
                            "normalized": action.tolist(),
                            "executed": row["executed"].tolist(),
                        }
                    )
                    + "\n"
                )
                if args.mode == "collect":
                    rows.append(row)
                if training:
                    replay.append(row)
                    del replay[: -args.replay_capacity]
                    for _ in range(args.updates_per_chunk):
                        update = learner.update(replay)
                        metrics.append(
                            update
                            | {
                                "physical_steps": physical_steps,
                                "control_steps": control_steps,
                                "behavior_policy_version": version,
                                "replay_size": len(replay),
                            }
                        )
                    if physical_steps >= next_save:
                        checkpoint(
                            args.output / f"checkpoint_{physical_steps}.pt",
                            learner,
                            replay,
                            payload | {"physical_steps": physical_steps},
                        )
                        next_save += args.save_steps
                current = following
                if done:
                    break
            result = {
                "case_id": case["id"],
                "success": bool(success),
                "control_steps": ticks,
                "settling_steps": 10,
                "budget_cut": bool(limit < 320 and not done),
                "terminal": bool(done),
                "initial_observation": initial,
                "state_sha256": initial_hash,
                "policy_version_start": first_version,
                "policy_version_last_executed": version,
            }
            episodes.append(result)
            write_json(args.output / "episodes.json", episodes)
            if frames:
                import imageio.v2 as imageio

                imageio.mimwrite(
                    args.output / f"episode_{episode:03d}.mp4", frames, fps=20
                )
            metrics.append(
                {
                    "episode": episode,
                    "success": int(success),
                    "physical_steps": physical_steps,
                    "control_steps": control_steps,
                    "actor_updates": learner.actor_updates if learner else 0,
                }
            )
            LOG.info(
                "%s case=%s success=%s physical_steps=%s actor_updates=%s",
                args.mode,
                case["id"],
                success,
                physical_steps,
                learner.actor_updates if learner else 0,
            )
    finally:
        env.close()
        trace.close()
    if args.mode == "collect":
        torch.save(
            {"protocol": PROTOCOL, "assets": ASSETS, "rows": rows},
            args.output / "cache.pt",
        )
    if training:
        checkpoint(
            args.output / "checkpoint.pt",
            learner,
            replay,
            payload | {"physical_steps": physical_steps},
        )
    complete = [e for e in episodes if not e["budget_cut"]]
    return {
        "episodes": len(episodes),
        "complete_episodes": len(complete),
        "successes": sum(e["success"] for e in complete),
        "success_rate": sum(e["success"] for e in complete) / max(1, len(complete)),
        "physical_steps": physical_steps,
        "control_steps": control_steps,
        "settling_steps": settling_steps,
        "actor_updates": (learner.actor_updates - start_actor) if learner else 0,
        "critic_updates": (learner.critic_updates - start_critic) if learner else 0,
        "actor_version": learner.actor_updates if learner else -1,
        "scope": "full_task",
        "assistance": False,
        "collected_chunks": len(rows),
        "objective": args.objective,
        "evaluation_updates": 0 if not training else None,
    }


def execute(args: argparse.Namespace, metrics: Metrics) -> dict:
    """Dispatch modes after provenance, GPU mapping and dependencies are checked."""
    import numpy as np
    import torch

    from toolkits.rlt.libero_acceptance_learning import AcceptanceLearner

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cases = read_cases(args.cases) if args.cases else []
    if args.mode == "generate":
        env = RemoteEnvironment()
        hashes = []
        try:
            for case in cases:
                if case["kind"] != "generated":
                    raise ValueError("State-bank generation requires generated cases")
                env.reset(case, create_snapshot=True)
                first = env.reset_hash
                env.reset(case)
                if first != env.reset_hash:
                    raise ValueError("Repeated reset changed the initial state")
                hashes.append(first)
        finally:
            env.close()
        if len(set(hashes)) != len(hashes):
            raise ValueError("Duplicate generated initial states")
        return {
            "generated_states": len(hashes),
            "repeat_reset_verified": True,
            "state_hashes": hashes,
        }
    if args.mode == "fit":
        cache = torch.load(args.cache, map_location="cpu", weights_only=True)
        compatible(cache)
        train_rows, validation = split_cache(cache["rows"])
        learner = AcceptanceLearner(
            "cuda:0", args.objective, args.beta, args.ref_dropout
        )
        initial = learner.following(validation)
        payload = {
            "protocol": PROTOCOL,
            "assets": ASSETS,
            "seed": args.seed,
            "ref_dropout": args.ref_dropout,
            "cache_sha256": digest(args.cache),
            "training_case_ids": sorted({r["case_id"] for r in train_rows}),
            "following_case_ids": sorted({r["case_id"] for r in validation}),
        }
        checkpoint(args.output / "initial.pt", learner, train_rows, payload)
        for step in range(args.bc_updates):
            metrics.append(learner.update(train_rows, warmup=True))
            if (step + 1) % 500 == 0:
                LOG.info(
                    "BC update=%s heldout=%s", step + 1, learner.following(validation)
                )
        learner.publish_warmup()
        final = learner.following(validation)
        checkpoint(args.output / "checkpoint.pt", learner, train_rows, payload)
        return {
            "initial_following": initial,
            "final_following": final,
            "actor_updates": learner.actor_updates,
            "critic_updates": learner.critic_updates,
            "train_chunks": len(train_rows),
            "validation_chunks": len(validation),
        }
    payload = {"protocol": PROTOCOL, "assets": ASSETS}
    learner, replay = None, []
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        compatible(payload)
        if args.mode == "train" and {c["id"] for c in cases} - set(
            payload["training_case_ids"]
        ):
            raise ValueError("Online training would use held-out initial states")
        learner = AcceptanceLearner(
            "cuda:0", args.objective, args.beta, payload["ref_dropout"]
        )
        learner.load_state_dict(payload["learner"])
        replay = payload.pop("replay")
        payload.pop("learner")
        payload["parent_checkpoint_sha256"] = digest(args.checkpoint)
        if args.reference:
            learner = None
    features = FrozenFeatures(args.storage)
    return run_episodes(args, cases, features, learner, replay, metrics, payload)


def main() -> None:
    """Require a new output directory, pinned environment and explicit GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["collect", "fit", "train", "evaluate", "generate"]
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective", choices=["bc_only", "q_bc"], default="q_bc")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--ref-dropout", type=float, default=0.5)
    parser.add_argument("--bc-updates", type=int, default=5000)
    parser.add_argument(
        "--control-budget",
        type=int,
        default=32000,
        help="Total simulator ticks INCLUDING settling",
    )
    parser.add_argument("--updates-per-chunk", type=int, default=2)
    parser.add_argument("--replay-capacity", type=int, default=20000)
    parser.add_argument("--save-steps", type=int, default=16000)
    parser.add_argument("--video-episodes", type=int, default=2)
    parser.add_argument(
        "--wandb", choices=["online", "offline", "disabled"], default="offline"
    )
    parser.add_argument("--group", default="rlt-a-acceptance")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; never overwrite an experiment")
    if args.mode == "fit" and args.cache is None:
        parser.error("fit requires --cache")
    if args.mode != "fit" and args.cases is None:
        parser.error("Simulator modes require --cases")
    if args.mode == "train" and (args.checkpoint is None or args.reference):
        parser.error("train requires a warmed checkpoint and actor control")
    if args.mode == "evaluate" and not (args.reference or args.checkpoint):
        parser.error("evaluate requires --reference or --checkpoint")
    if (
        min(
            args.control_budget,
            args.bc_updates,
            args.updates_per_chunk,
            args.save_steps,
            args.replay_capacity,
        )
        < 1
    ):
        parser.error("Budgets must be positive")
    if not 0 <= args.ref_dropout <= 1 or args.beta <= 0:
        parser.error("Invalid objective parameters")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    revision = check_source(args.source)
    for package in ("mujoco", "robosuite", "numpy", "transformers", "tokenizers"):
        if metadata.version(package) != PROTOCOL[package]:
            raise RuntimeError(f"Protocol requires {package}=={PROTOCOL[package]}")
    from toolkits.rlt.libero_environment_audit import dependency_conflicts

    conflicts = dependency_conflicts("rlinf-libero")
    if conflicts:
        raise RuntimeError(f"LIBERO dependency conflicts: {conflicts}")
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
        raise RuntimeError(
            f"GPU {args.gpu} busy ({used} MiB); refusing to share implicitly"
        )
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
    os.environ.update(
        CUDA_VISIBLE_DEVICES=gpu_uuid,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        MUJOCO_GL="egl",
        LIBERO_PYTHON=sys.executable,
        TOKENIZERS_PARALLELISM="false",
        RLT_LIBERO_EGL_DEVICE_ID=str(egl_index),
        RLT_LIBERO_WORKER_LOG_DIR=str(args.output / "worker_logs"),
        PYTHONPATH=str(args.source.resolve())
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    )
    sys.path.insert(0, str(args.source.resolve()))
    module_versions = {
        name: importlib.import_module(name).__version__
        for name in ("torch", "transformers", "tokenizers", "mujoco", "numpy")
    }
    for name, actual in module_versions.items():
        if actual != metadata.version(name):
            raise RuntimeError(
                f"{name}: imported module {actual} differs from package metadata"
            )
    args.output.mkdir(parents=True)
    manifest = {
        "complete": False,
        "protocol": PROTOCOL,
        "assets": ASSETS,
        "source_revision": revision,
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "gpu_uuid": gpu_uuid,
        "egl_device_index": egl_index,
        "python": sys.executable,
        "module_versions": module_versions,
        "versions": {
            p: metadata.version(p)
            for p in (
                "torch",
                "transformers",
                "mujoco",
                "robosuite",
                "numpy",
                "rlinf-libero",
            )
        },
        "code_sha256": {
            p.name: digest(p)
            for p in Path(__file__).parent.glob("libero_acceptance*.py")
        },
        "inputs": {
            k: digest(v)
            for k, v in {
                "cases": args.cases,
                "cache": args.cache,
                "checkpoint": args.checkpoint,
            }.items()
            if v
        },
    }
    write_json(args.output / "manifest.json", manifest)
    metrics = Metrics(args)
    try:
        summary = execute(args, metrics)
        write_json(args.output / "summary.json", summary)
        manifest["complete"] = True
        if (args.output / "checkpoint.pt").is_file():
            manifest["checkpoint_sha256"] = digest(args.output / "checkpoint.pt")
        write_json(args.output / "manifest.json", manifest)
        LOG.info("Completed %s: %s", args.mode, summary)
    except BaseException as error:
        write_json(
            args.output / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        metrics.close()


if __name__ == "__main__":
    main()
