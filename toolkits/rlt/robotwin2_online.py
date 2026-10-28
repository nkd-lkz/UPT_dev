# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Warm up, execute, and update AlphaBrain RLT in native RoboTwin 2.0.

This entry point deliberately keeps the three experimental phases separate:
BC warmup only consumes immutable reference caches, online training only uses
the RoboTwin training seed file, and evaluation only uses its evaluation seed
file.  The frozen Pi0.5 model always produces a 50-step reference while the
learner and environment exchange exactly the first ten actions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from toolkits.rlt.robodojo_learning import make_encoder
from toolkits.rlt.robotwin2_features import FrozenRoboTwinPi05
from toolkits.rlt.robotwin2_learning import PROTOCOL, AlohaLearner, AlohaNormalizer
from toolkits.rlt.robotwin2_reference_rollout import (
    build_env_config,
    classify_failure,
    scalar,
    task_diagnostics,
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish JSON without exposing a partially written status file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_encoder(path: Path, device: str) -> torch.nn.Module:
    """Restore a protocol-pinned Stage-1 encoder and freeze it."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Stage-1 checkpoint has an incompatible protocol")
    encoder = make_encoder(device)
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval().requires_grad_(False)
    return encoder


@torch.no_grad()
def encode_hidden(
    encoder: torch.nn.Module,
    hidden: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    device: str,
) -> np.ndarray:
    """Reduce one validated 768x2048 image-prefix tensor to one RL token."""
    hidden_tensor = torch.as_tensor(hidden, dtype=torch.float32, device=device)
    valid_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if hidden_tensor.shape != (768, 2048) or valid_tensor.shape != (768,):
        raise ValueError("Invalid RoboTwin frozen-feature shape")
    if not valid_tensor.all() or not torch.isfinite(hidden_tensor).all():
        raise ValueError("RoboTwin frozen features must all be finite and valid")
    token = encoder.encode(
        hidden_tensor.unsqueeze(0),
        key_padding_mask=(~valid_tensor).unsqueeze(0),
    )
    result = token[0, 0].float().cpu().numpy()
    if result.shape != (2048,) or not np.isfinite(result).all():
        raise ValueError("Invalid encoded RL token")
    return result


def cache_paths(roots: list[Path]) -> list[Path]:
    """Find immutable reference decisions and retain each physical file once."""
    paths: dict[str, Path] = {}
    for root in roots:
        for episode_path in sorted(root.rglob("episode.json")):
            episode = json.loads(episode_path.read_text())
            if episode.get("benchmark") != "robotwin2":
                continue
            for record in episode.get("records", []):
                path = Path(record["cache_file"]).resolve()
                paths[str(path)] = path
    if not paths:
        raise ValueError("No RoboTwin reference caches found")
    return list(paths.values())


def encode_reference_rows(
    paths: list[Path],
    encoder: torch.nn.Module,
    normalizer: AlohaNormalizer,
    device: str,
) -> list[dict[str, np.ndarray]]:
    """Build compact BC rows from frozen Pi0.5 reference decisions."""
    rows = []
    for path in paths:
        with np.load(path) as payload:
            hidden = np.asarray(payload["hidden"], dtype=np.float32)
            mask = np.asarray(payload["mask"], dtype=np.bool_)
            state = np.asarray(payload["state"], dtype=np.float32)
            reference = np.asarray(payload["reference"], dtype=np.float32)
        if state.shape != (14,) or reference.shape != (50, 14):
            raise ValueError(f"Invalid state/reference cache: {path}")
        rows.append(
            {
                "z": encode_hidden(encoder, hidden, mask, device),
                "ref": normalizer.encode(reference[:10], state),
                "prop": normalizer.state(state),
            }
        )
    return rows


def save_learner(
    path: Path,
    learner: AlohaLearner,
    *,
    stage1: Path,
    metadata: dict[str, Any],
) -> None:
    """Save heads and provenance without duplicating the 3 GiB encoder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "stage1": str(stage1.resolve()),
            "learner": learner.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def restore_learner(
    path: Path,
    normalizer: AlohaNormalizer,
    *,
    device: str,
    objective: str,
    anchor_scale: float,
) -> tuple[AlohaLearner, dict[str, Any]]:
    """Restore one learner while explicitly setting the current run contract."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Incompatible RoboTwin learner checkpoint")
    learner = AlohaLearner(
        normalizer,
        device=device,
        objective=objective,
        reference_anchor_scale=anchor_scale,
    )
    learner.load_state_dict(payload["learner"])
    learner.objective = objective
    learner.configure_reference_anchor(anchor_scale)
    return learner, payload


def warmup(args: argparse.Namespace) -> dict[str, Any]:
    """Train the residual actor to copy held-out frozen-reference chunks."""
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    normalizer = AlohaNormalizer(args.norm_stats.resolve())
    encoder = load_encoder(args.stage1.resolve(), args.device)
    paths = cache_paths([path.resolve() for path in args.reference_root])
    rows = encode_reference_rows(paths, encoder, normalizer, args.device)
    order = np.random.default_rng(args.seed).permutation(len(rows)).tolist()
    heldout_count = max(1, int(round(len(rows) * args.heldout_fraction)))
    heldout_ids = set(order[-heldout_count:])
    train_rows = [row for index, row in enumerate(rows) if index not in heldout_ids]
    heldout_rows = [row for index, row in enumerate(rows) if index in heldout_ids]
    if not train_rows:
        raise ValueError("Warmup split left no training rows")

    learner = AlohaLearner(
        normalizer,
        device=args.device,
        objective="bc_only",
        reference_anchor_scale=args.anchor_scale,
    )
    initial = learner.following(heldout_rows)
    history = []
    for step in range(1, args.steps + 1):
        batch = random.choices(train_rows, k=args.batch_size)
        metrics = learner.update(batch, warmup=True)
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            history.append({"step": step, **metrics})
    learner.publish_warmup()
    final = learner.following(heldout_rows)
    result = {
        "schema_version": 1,
        "phase": "bc_warmup",
        "state": "passed",
        "protocol": PROTOCOL,
        "stage1": str(args.stage1.resolve()),
        "reference_caches": len(paths),
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "anchor_scale": args.anchor_scale,
        "initial_following": initial,
        "final_following": final,
        "history": history,
        "wall_seconds": time.time() - started,
    }
    save_learner(
        args.output.resolve(),
        learner,
        stage1=args.stage1,
        metadata=result,
    )
    atomic_json(args.output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2))
    return result


def observation_row(
    observation: dict[str, Any],
    inference: dict[str, torch.Tensor],
    encoder: torch.nn.Module,
    normalizer: AlohaNormalizer,
    device: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Form the exact learner row paired with a physical pre-chunk state."""
    state = observation["states"][0].detach().cpu().numpy().astype(np.float32)
    reference = inference["reference"][0].detach().cpu().numpy().astype(np.float32)
    row = {
        "z": encode_hidden(
            encoder,
            inference["hidden"][0],
            inference["mask"][0],
            device,
        ),
        "ref": normalizer.encode(reference[:10], state),
        "prop": normalizer.state(state),
    }
    return row, state


def successor(row: dict[str, Any], next_row: dict[str, Any]) -> dict[str, Any]:
    """Attach the successor fields consumed by AlphaBrain's twin-Q target."""
    return row | {
        "next_z": next_row["z"],
        "next_ref": next_row["ref"],
        "next_prop": next_row["prop"],
    }


def create_env(args: argparse.Namespace, output: Path, split: str, seed: int) -> Any:
    """Create one auditable native environment on the requested split."""
    robotwin = args.robotwin.resolve()
    if str(robotwin) not in sys.path:
        sys.path.insert(0, str(robotwin))
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

    cfg = build_env_config(args.repo.resolve(), robotwin, output, split=split)
    cfg.seed = seed
    return RoboTwinEnv(
        cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def run_episode(
    args: argparse.Namespace,
    *,
    policy: FrozenRoboTwinPi05,
    encoder: torch.nn.Module | None,
    normalizer: AlohaNormalizer | None,
    learner: AlohaLearner | None,
    split: str,
    requested_seed: int,
    deterministic: bool,
    replay: deque[dict[str, Any]] | None = None,
    updates_per_decision: int = 0,
    rng: random.Random | None = None,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Execute one reference or RLT episode and optionally update online."""
    output = args.output.resolve()
    env = create_env(args, output, split, requested_seed)
    np.random.seed(requested_seed)
    torch.manual_seed(requested_seed)
    records: list[dict[str, Any]] = []
    update_metrics: list[dict[str, float]] = []
    started = time.monotonic()
    try:
        observation, _ = env.reset()
        environment_seed = int(env.reset_state_ids[0].item())
        initial = task_diagnostics(env)
        cached_inference = None
        while int(env.elapsed_steps[0].item()) < env.cfg.max_episode_steps:
            # Reuse the successor inference computed for TD.  Drawing a second
            # flow-matching noise tensor here would make next_ref differ from
            # the reference that actually conditions the following decision.
            inference = (
                cached_inference
                if cached_inference is not None
                else policy.infer(observation)
            )
            cached_inference = None
            reference_physical = (
                inference["reference"][0, :10].detach().cpu().numpy().astype(np.float32)
            )
            if learner is None:
                current = None
                physical = reference_physical
                normalized = None
            else:
                assert encoder is not None and normalizer is not None
                current, state = observation_row(
                    observation, inference, encoder, normalizer, args.device
                )
                normalized = learner.predict(current, deterministic=deterministic)
                physical = normalizer.decode(normalized, state)

            before = task_diagnostics(env)
            observation, reward, terminated, truncated, infos = env.step(
                physical[None], auto_reset=False
            )
            is_terminated = bool(scalar(terminated))
            is_truncated = bool(scalar(truncated))
            record: dict[str, Any] = {
                "decision": len(records),
                "start_tick": int(env.elapsed_steps[0].item()) - 10,
                "ticks": 10,
                "reward": float(scalar(reward)),
                "terminated": is_terminated,
                "truncated": is_truncated,
                "diagnostics_before": before,
                "diagnostics": task_diagnostics(env),
                "reference_action_rms": float(np.sqrt(np.mean(reference_physical**2))),
            }
            if "success" in infos:
                record["success"] = bool(scalar(infos["success"]))
            if learner is not None:
                assert (
                    current is not None
                    and normalized is not None
                    and normalizer is not None
                )
                issued = current | {
                    "action": normalized,
                    "executed_action": normalized.copy(),
                    "reward": record["reward"],
                    "ticks": 10,
                    "terminated": is_terminated,
                    "truncated": is_truncated,
                }
                if is_terminated:
                    next_row = current
                else:
                    next_inference = policy.infer(observation)
                    next_row, _ = observation_row(
                        observation, next_inference, encoder, normalizer, args.device
                    )
                    if not is_truncated:
                        cached_inference = next_inference
                transition = successor(issued, next_row)
                correction = normalized - current["ref"]
                record.update(
                    normalized_correction_rms=float(np.sqrt(np.mean(correction**2))),
                    normalized_correction_max=float(np.abs(correction).max()),
                )
                if replay is not None:
                    replay.append(transition)
                    if updates_per_decision:
                        assert rng is not None
                        for _ in range(updates_per_decision):
                            size = min(args.batch_size, len(replay))
                            batch = rng.sample(list(replay), size)
                            metrics = learner.update(batch, warmup=False)
                            update_metrics.append(
                                {key: float(value) for key, value in metrics.items()}
                            )
            records.append(record)
            if is_terminated or is_truncated:
                break
    finally:
        env.offload(clear_cache=True)

    success = any(record.get("success", False) for record in records)
    result = {
        "schema_version": 1,
        "benchmark": "robotwin2",
        "task": "adjust_bottle",
        "split": split,
        "policy": "reference" if learner is None else "alphabrain_rlt",
        "reference_horizon": 50,
        "execution_horizon": 10,
        "requested_seed": requested_seed,
        "environment_seed": environment_seed,
        "elapsed_ticks": int(env.elapsed_steps[0].item()),
        "wall_seconds": time.monotonic() - started,
        "success": success,
        "terminated": any(record["terminated"] for record in records),
        "initial_diagnostics": initial,
        "failure_stage": classify_failure(initial, records, success),
        "records": records,
    }
    return result, update_metrics


def load_runtime(args: argparse.Namespace, objective: str) -> tuple[Any, ...]:
    """Load the common frozen policy, Stage-1 encoder, normalizer, and heads."""
    normalizer = AlohaNormalizer(args.norm_stats.resolve())
    learner, payload = restore_learner(
        args.learner.resolve(),
        normalizer,
        device=args.device,
        objective=objective,
        anchor_scale=args.anchor_scale,
    )
    stage1 = Path(payload["stage1"])
    encoder = load_encoder(stage1, args.device)
    policy = FrozenRoboTwinPi05(args.checkpoint.resolve(), device=args.device)
    return policy, encoder, normalizer, learner, stage1


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run matched held-out episodes for the reference or deterministic actor."""
    started = time.time()
    if args.policy == "reference":
        policy = FrozenRoboTwinPi05(args.checkpoint.resolve(), device=args.device)
        encoder = normalizer = learner = None
    else:
        policy, encoder, normalizer, learner, _ = load_runtime(args, "bc_only")
    episodes = []
    for seed in args.seeds:
        episode, _ = run_episode(
            args,
            policy=policy,
            encoder=encoder,
            normalizer=normalizer,
            learner=learner,
            split="eval",
            requested_seed=seed,
            deterministic=True,
        )
        episode_path = args.output / f"episode-seed{seed}.json"
        atomic_json(episode_path, episode)
        episodes.append(episode)
    result = {
        "schema_version": 1,
        "phase": "evaluation",
        "policy": args.policy,
        "seeds": args.seeds,
        "episodes": len(episodes),
        "successes": sum(item["success"] for item in episodes),
        "success_rate": float(np.mean([item["success"] for item in episodes])),
        "mean_ticks": float(np.mean([item["elapsed_ticks"] for item in episodes])),
        "failure_stages": {
            stage: sum(item["failure_stage"] == stage for item in episodes)
            for stage in sorted({item["failure_stage"] for item in episodes})
        },
        "wall_seconds": time.time() - started,
    }
    atomic_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return result


def online(args: argparse.Namespace) -> dict[str, Any]:
    """Collect training-split interaction and perform live Q+BC updates."""
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy, encoder, normalizer, learner, stage1 = load_runtime(args, "q_bc")
    learner.configure_online_noise(
        exploration_std=args.exploration_std,
        target_noise_std=args.target_noise_std,
        target_noise_clip=args.target_noise_clip,
    )
    replay: deque[dict[str, Any]] = deque(maxlen=args.replay_capacity)
    rng = random.Random(args.seed)
    episodes = []
    all_updates = []
    for requested_seed in args.seeds:
        episode, updates = run_episode(
            args,
            policy=policy,
            encoder=encoder,
            normalizer=normalizer,
            learner=learner,
            split="train",
            requested_seed=requested_seed,
            deterministic=False,
            replay=replay,
            updates_per_decision=args.updates_per_decision,
            rng=rng,
        )
        atomic_json(args.output / f"train-episode-seed{requested_seed}.json", episode)
        episodes.append(episode)
        all_updates.extend(updates)
    result = {
        "schema_version": 1,
        "phase": "online_q_bc",
        "state": "passed",
        "split": "train",
        "seeds": args.seeds,
        "episodes": len(episodes),
        "successes": sum(item["success"] for item in episodes),
        "success_rate": float(np.mean([item["success"] for item in episodes])),
        "environment_ticks": sum(item["elapsed_ticks"] for item in episodes),
        "replay_size": len(replay),
        "online_updates": len(all_updates),
        "actor_updates": learner.actor_updates,
        "critic_updates": learner.critic_updates,
        "anchor_scale": args.anchor_scale,
        "exploration_std": args.exploration_std,
        "last_update": all_updates[-1] if all_updates else None,
        "wall_seconds": time.time() - started,
    }
    save_learner(
        args.output / "online.pt",
        learner,
        stage1=stage1,
        metadata=result,
    )
    atomic_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return result


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--robotwin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-scale", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)


def comma_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("Require comma-separated nonnegative seeds")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    warmup_parser = subparsers.add_parser("warmup")
    warmup_parser.add_argument("--stage1", type=Path, required=True)
    warmup_parser.add_argument("--norm-stats", type=Path, required=True)
    warmup_parser.add_argument(
        "--reference-root", type=Path, action="append", required=True
    )
    warmup_parser.add_argument("--output", type=Path, required=True)
    warmup_parser.add_argument("--steps", type=int, default=1000)
    warmup_parser.add_argument("--batch-size", type=int, default=16)
    warmup_parser.add_argument("--heldout-fraction", type=float, default=0.2)
    warmup_parser.add_argument("--anchor-scale", type=float, default=0.1)
    warmup_parser.add_argument("--log-interval", type=int, default=100)
    warmup_parser.add_argument("--seed", type=int, default=0)
    warmup_parser.add_argument("--device", default="cuda")

    evaluate_parser = subparsers.add_parser("evaluate")
    common(evaluate_parser)
    evaluate_parser.add_argument(
        "--policy", choices=("reference", "actor"), required=True
    )
    evaluate_parser.add_argument("--learner", type=Path)
    evaluate_parser.add_argument("--seeds", type=comma_ints, required=True)

    online_parser = subparsers.add_parser("online")
    common(online_parser)
    online_parser.add_argument("--learner", type=Path, required=True)
    online_parser.add_argument("--seeds", type=comma_ints, required=True)
    online_parser.add_argument("--seed", type=int, default=0)
    online_parser.add_argument("--updates-per-decision", type=int, default=1)
    online_parser.add_argument("--replay-capacity", type=int, default=5000)
    online_parser.add_argument("--exploration-std", type=float, default=0.01)
    online_parser.add_argument("--target-noise-std", type=float, default=0.02)
    online_parser.add_argument("--target-noise-clip", type=float, default=0.05)

    args = parser.parse_args()
    if args.anchor_scale < 0 or args.anchor_scale > 1:
        parser.error("anchor-scale must be in [0, 1]")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.command == "warmup":
        if args.steps < 1 or not 0 < args.heldout_fraction < 1:
            parser.error("steps must be positive and heldout-fraction in (0, 1)")
    if args.command == "evaluate" and args.policy == "actor" and args.learner is None:
        parser.error("actor evaluation requires --learner")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "warmup":
        warmup(arguments)
    elif arguments.command == "evaluate":
        evaluate(arguments)
    else:
        online(arguments)
