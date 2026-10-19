# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run bounded new-RLT online learning on isolated RoboDojo train layouts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from toolkits.rlt.experiment_queue import gpu_idle, stop_group
from toolkits.rlt.pi05_baseline import configure_source
from toolkits.rlt.robodojo_boundary import ACTION_SEMANTICS, FIELDS, decode_actions
from toolkits.rlt.robodojo_collect import create_native, save
from toolkits.rlt.robodojo_eval import (
    ROBO_DOJO_REVISION,
    XPOLICYLAB_REVISION,
    _source_check,
)
from toolkits.rlt.robodojo_experience import ExperienceHistory, history_view
from toolkits.rlt.robodojo_grounding import load_grounding
from toolkits.rlt.robodojo_layouts import digest, verify
from toolkits.rlt.robodojo_learning import (
    ArxLearner,
    ArxNormalizer,
    learner_protocol,
    make_encoder,
)
from toolkits.rlt.robodojo_perturb import (
    make_perturber,
    parse_perturb_schedule,
    phase_for_episode,
)
from toolkits.rlt.robodojo_runner import (
    foreign_gpu_processes,
    owns_listener,
    renderer_index,
)
from toolkits.rlt.robodojo_transition import RoboDojoTransitions


def normalization_path(config: dict) -> Path:
    """Resolve the exact native statistics used by the frozen reference."""
    policy = config["policy"]
    return (
        Path(policy["checkpoint_path"])
        / "assets"
        / policy["model_contract"]["normalization_id"]
        / "norm_stats.json"
    )


def load_warmup(
    path: Path, normalizer: ArxNormalizer
) -> tuple[torch.nn.Module, ArxLearner, dict[int, str], str | None]:
    """Restore a frozen encoder and complete Q+BC learner state.

    The warmup's recorded protocol decides the learner shape: a baseline
    protocol restores the 2048-D learner, while a grounded-memory protocol
    restores the widened one and reports which grounding artifact produced
    its tokens, so the online run can refuse a mismatched artifact.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    memory = (payload.get("protocol") or {}).get("memory")
    token_dim = memory["token_dim"] if isinstance(memory, dict) else 0
    if payload.get("protocol") != learner_protocol(token_dim):
        raise ValueError("Warmup checkpoint uses another embodiment protocol")
    encoder = make_encoder().eval().requires_grad_(False)
    encoder.load_state_dict(payload["encoder"], strict=True)
    learner = ArxLearner(normalizer, objective="q_bc", token_dim=token_dim)
    learner.load_state_dict(payload["learner"])
    layouts = {row["id"]: row["sha256"] for row in payload.get("train_layouts", [])}
    if len(layouts) < 2:
        raise ValueError("Warmup must identify at least two validated train layouts")
    return encoder, learner, layouts, payload.get("grounding_sha256")


def prediction_arrays(prediction: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load feature arrays from a training cache or an inline evaluation response."""
    cache_file = prediction.get("cache_file")
    features = prediction.get("features")
    if cache_file is not None:
        with np.load(Path(cache_file)) as payload:
            hidden = np.asarray(payload["hidden"], dtype=np.float32)
            state = np.asarray(payload["state"], dtype=np.float32)
            reference_physical = np.asarray(payload["reference"], dtype=np.float32)
    elif isinstance(features, dict):
        hidden = np.asarray(features["hidden"], dtype=np.float32)
        state = np.asarray(features["state"], dtype=np.float32)
        reference_physical = np.asarray(features["reference"], dtype=np.float32)
    else:
        raise ValueError(
            "Feature response has neither a cache file nor inline features"
        )
    return hidden, state, reference_physical


def encode_prediction(
    encoder: torch.nn.Module,
    normalizer: ArxNormalizer,
    prediction: dict,
) -> tuple[dict, np.ndarray]:
    """Convert one feature-server response into the actor's native normalized space."""
    hidden, state, reference_physical = prediction_arrays(prediction)
    reference_physical[:, [6, 13]] = np.clip(reference_physical[:, [6, 13]], 0, 1)
    reference = normalizer.encode(reference_physical, state)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        token = (
            encoder.encode(torch.from_numpy(hidden)[None].cuda())[0, 0].float().cpu()
        )
    return {
        "z": token,
        "ref": torch.from_numpy(reference),
        "prop": torch.from_numpy(normalizer.state(state)),
    }, state


def attach_successor(row: dict, successor: dict) -> dict:
    """Attach only the next pre-action state required by one TD target."""
    return row | {
        "next_z": successor["z"],
        "next_ref": successor["ref"],
        "next_prop": successor["prop"],
    }


def action_deviation(issued: np.ndarray, reference: np.ndarray, ticks: int) -> dict:
    """Summarize normalized policy deviation over the acknowledged prefix."""
    issued = np.asarray(issued, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if (
        issued.shape != (10, 14)
        or reference.shape != (10, 14)
        or type(ticks) is not int
        or not 1 <= ticks <= 10
    ):
        raise ValueError("Require two 10x14 chunks and one valid executed prefix")
    absolute = np.abs(issued[:ticks] - reference[:ticks])
    return {
        "reference_mean_abs": float(absolute.mean()),
        "reference_max_abs": float(absolute.max()),
    }


def apply_frozen_action_adapter(
    action: np.ndarray, reference: np.ndarray, adapter: str
) -> np.ndarray:
    """Apply an explicitly labelled frozen-evaluation diagnostic adapter."""
    action = np.asarray(action, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if action.shape != (10, 14) or reference.shape != (10, 14):
        raise ValueError("Frozen action adapter requires 10x14 chunks")
    if adapter == "none":
        return action.copy()
    if adapter == "reference_grippers":
        result = action.copy()
        result[:, [6, 13]] = reference[:, [6, 13]]
        return result
    raise ValueError(f"Unknown frozen action adapter: {adapter}")


def select_layout(
    layouts: list[dict],
    invalid_layouts: set[int],
    outcomes: list[dict],
    episode_id: int,
    *,
    frozen_evaluation: bool,
) -> dict:
    """Select each frozen-eval layout once, while cycling valid train layouts."""
    if frozen_evaluation:
        attempted = {
            row["layout_id"] for row in outcomes if type(row.get("layout_id")) is int
        }
        available = [row for row in layouts if row["id"] not in attempted]
        if not available:
            raise RuntimeError("Every declared frozen-evaluation layout was attempted")
        return available[0]
    available = [row for row in layouts if row["id"] not in invalid_layouts]
    if not available:
        raise RuntimeError("Every warmup-validated train layout failed reset")
    return available[episode_id % len(available)]


def checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    learner: ArxLearner,
    meta: dict,
    online_state: dict | None = None,
) -> None:
    """Atomically save frozen encoder and exact optimizer/target state."""
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "protocol": learner_protocol(getattr(learner, "token_dim", 0)),
            "encoder": encoder.state_dict(),
            "learner": learner.state_dict(),
            "meta": meta,
            "online_state": online_state,
        },
        temporary,
    )
    temporary.replace(path)


def restore_online_checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    learner: ArxLearner,
    *,
    layout_digest: str,
    reference_replay_digest: str,
    seed: int,
    exploration_std: float,
    target_noise_std: float,
    target_noise_clip: float,
    reference_anchor_scale: float,
    grounding_sha256: str | None = None,
    history_mode: str = "correct",
    perturb_schedule: str | None = None,
) -> dict:
    """Restore a locally produced learner/replay checkpoint at a fresh episode."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != learner_protocol(getattr(learner, "token_dim", 0)):
        raise ValueError("Online checkpoint uses another embodiment protocol")
    meta, state = payload.get("meta", {}), payload.get("online_state")
    if (
        not isinstance(state, dict)
        or meta.get("layout_digest") != layout_digest
        or meta.get("reference_replay_sha256") != reference_replay_digest
        or meta.get("seed") != seed
        or meta.get("exploration_std") != exploration_std
        or meta.get("target_noise_std") != target_noise_std
        or meta.get("target_noise_clip") != target_noise_clip
        or meta.get("reference_anchor_scale", 1.0) != reference_anchor_scale
        or meta.get("grounding_sha256") != grounding_sha256
        or meta.get("history_mode", "correct") != history_mode
        or meta.get("perturb_schedule") != perturb_schedule
    ):
        raise ValueError("Online checkpoint provenance does not match this run")
    encoder.load_state_dict(payload["encoder"], strict=True)
    learner.load_state_dict(payload["learner"])
    required = {
        "control_steps",
        "updates",
        "episode_id",
        "replay",
        "outcomes",
        "invalid_layouts",
        "python_rng_state",
        "torch_rng_state",
    }
    if not required <= state.keys():
        raise ValueError("Online checkpoint lacks restart state")
    return state


def load_frozen_weights(
    path: Path, encoder: torch.nn.Module, learner: ArxLearner
) -> dict:
    """Load only model weights for deterministic checkpoint evaluation."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != learner_protocol(getattr(learner, "token_dim", 0)):
        raise ValueError("Evaluation checkpoint uses another embodiment protocol")
    encoder.load_state_dict(payload["encoder"], strict=True)
    learner.load_state_dict(payload["learner"])
    return dict(payload.get("meta", {}))


def load_reference_replay(path: Path, accepted: dict[int, str]) -> list[dict]:
    """Load successful and failed train-only reference chunks for initial TD."""
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise ValueError("Reference replay must contain an encoded row list")
    replay = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Reference replay rows must be mappings")
        episode = row.get("episode")
        if row.get("role") != "train" or row.get("td_eligible") is not True:
            continue
        if (
            not isinstance(episode, tuple)
            or len(episode) != 2
            or type(episode[0]) is not int
            or accepted.get(episode[0]) != episode[1]
        ):
            raise ValueError("Reference replay differs from warmup train layouts")
        replay.append(
            {
                key: row[key]
                for key in (
                    "z",
                    "ref",
                    "prop",
                    "action",
                    "executed_action",
                    "reward",
                    "ticks",
                    "terminated",
                    "truncated",
                    "next_z",
                    "next_ref",
                    "next_prop",
                )
            }
        )
    positive = [row for row in replay if float(row["reward"]) > 0]
    if (
        len(replay) < 64
        or len(positive) < 2
        or not all(row["terminated"] for row in positive)
    ):
        raise ValueError(
            "Sparse-reward online RL requires at least 64 reference chunks and "
            "two positive terminal examples"
        )
    return replay


def renderer_failed(path: Path) -> bool:
    """Recognize the fatal RTX allocation failure that otherwise spins forever."""
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 256 * 1024))
            tail = stream.read().decode(errors="replace")
    except OSError:
        return False
    return "Failed to execute RenderGraph" in tail and (
        "Unable to allocate descriptor sets" in tail
        or "Failed to allocate ParameterBlock resources" in tail
    )


def process_details(pids: list[int]) -> list[dict]:
    """Record owner and command for GPU conflicts while procfs still exists."""
    details = []
    for pid in pids:
        result = subprocess.run(
            ["ps", "-o", "user=", "-o", "etimes=", "-o", "args=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        fields = result.stdout.strip().split(maxsplit=2)
        details.append(
            {
                "pid": pid,
                "user": fields[0] if len(fields) == 3 else None,
                "elapsed_seconds": int(fields[1]) if len(fields) == 3 else None,
                "command": fields[2] if len(fields) == 3 else None,
            }
        )
    return details


def disallowed_gpu_processes(pids: list[int], allowed_users: set[str]) -> list[int]:
    """Return GPU PIDs whose resolved owner is not explicitly allowed."""
    if not allowed_users:
        return pids
    return [
        row["pid"] for row in process_details(pids) if row["user"] not in allowed_users
    ]


def gpu_memory_used_mib(gpu: int) -> int:
    """Read total used memory on one physical GPU for coexistence guards."""
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    return int(value)


def client(args: argparse.Namespace, config: dict) -> None:
    """Own Isaac and PyTorch heads while the isolated JAX server owns Pi0.5."""
    configure_source(args.source)
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
    from isaaclab.app import AppLauncher

    launch = argparse.Namespace(
        headless=True,
        enable_cameras=True,
        device="cuda:0",
        kit_args=f"--/renderer/activeGpu={args.renderer_index} --/renderer/multiGpu/enabled=false --/physics/cudaDevice=0",
    )
    add_zero_delay_kit_args(launch)
    app = AppLauncher(launch).app
    native = None
    try:
        split = verify(args.layouts)
        layout_digest = digest(args.layouts)
        normalizer = ArxNormalizer(normalization_path(config))
        encoder, learner, accepted, warmup_grounding = load_warmup(
            args.warmup, normalizer
        )
        tokenizer = grounding_digest = None
        if args.grounding is not None:
            tokenizer, grounding_digest = load_grounding(args.grounding, device="cuda")
            if learner.token_dim != tokenizer.token_dim:
                raise ValueError(
                    "Warmup learner width does not match the grounding token"
                )
            if warmup_grounding is not None and warmup_grounding != grounding_digest:
                raise ValueError(
                    "Warmup was trained with a different grounding artifact"
                )
        elif learner.token_dim:
            raise ValueError("Grounded warmup requires its grounding artifact")
        history = (
            ExperienceHistory(tokenizer.history_length)
            if tokenizer is not None
            else None
        )
        history_rng = np.random.default_rng(args.seed)
        perturb_schedule = (
            parse_perturb_schedule(args.perturb_schedule)
            if args.perturb_schedule is not None
            else None
        )
        learner.configure_online_noise(
            exploration_std=args.exploration_std,
            target_noise_std=args.target_noise_std,
            target_noise_clip=args.target_noise_clip,
        )
        learner.configure_reference_anchor(args.reference_anchor_scale)
        frozen_evaluation = args.frozen_eval_role is not None
        if frozen_evaluation:
            layouts = [
                row for row in split["layouts"] if row["role"] == args.frozen_eval_role
            ][: args.frozen_eval_episodes]
            if len(layouts) != args.frozen_eval_episodes:
                raise ValueError("Not enough declared layouts for frozen evaluation")
            if args.resume is not None:
                load_frozen_weights(args.resume, encoder, learner)
        else:
            layouts = [
                row
                for row in split["layouts"]
                if row["role"] == "train" and accepted.get(row["id"]) == row["sha256"]
            ]
            if len(layouts) != len(accepted):
                raise ValueError(
                    "Warmup train layouts do not match the current manifest"
                )
        rng = random.Random(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        replay: list[dict] = []
        outcomes = []
        native = create_native(config, app, args.port)
        native.seed_manager.seed_info = {
            row["id"]: {"scene_layout": row["path"]} for row in layouts
        }
        protected = frozenset(row["sha256"] for row in split["protected"])
        control_steps = updates = episode_id = 0
        invalid_layouts = set()
        reference_replay_path = args.warmup.parent / "encoded-rollouts.pt"
        reference_replay_digest = hashlib.sha256(
            reference_replay_path.read_bytes()
        ).hexdigest()
        if not frozen_evaluation and args.resume is None:
            replay = load_reference_replay(reference_replay_path, accepted)
        if args.resume is not None and not frozen_evaluation:
            state = restore_online_checkpoint(
                args.resume,
                encoder,
                learner,
                layout_digest=layout_digest,
                reference_replay_digest=reference_replay_digest,
                seed=args.seed,
                exploration_std=args.exploration_std,
                target_noise_std=args.target_noise_std,
                target_noise_clip=args.target_noise_clip,
                reference_anchor_scale=args.reference_anchor_scale,
                grounding_sha256=grounding_digest,
                history_mode=args.history_mode,
                perturb_schedule=args.perturb_schedule,
            )
            control_steps = int(state["control_steps"])
            updates = int(state["updates"])
            episode_id = int(state["episode_id"])
            replay = list(state["replay"])
            outcomes = list(state["outcomes"])
            invalid_layouts = set(state["invalid_layouts"])
            rng.setstate(state["python_rng_state"])
            torch.set_rng_state(state["torch_rng_state"])
            if state.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(state["cuda_rng_state"])

        def restart_state(next_episode_id: int) -> dict:
            return {
                "control_steps": control_steps,
                "updates": updates,
                "episode_id": next_episode_id,
                "replay": replay,
                "outcomes": outcomes,
                "invalid_layouts": sorted(invalid_layouts),
                "python_rng_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(),
                "restart_boundary": "fresh_episode",
            }

        checkpoint_meta = {
            "layout_digest": layout_digest,
            "seed": args.seed,
            "exploration_std": args.exploration_std,
            "target_noise_std": args.target_noise_std,
            "target_noise_clip": args.target_noise_clip,
            "reference_anchor_scale": args.reference_anchor_scale,
            "frozen_eval_role": args.frozen_eval_role,
            "frozen_eval_episodes": args.frozen_eval_episodes,
            "reference_replay_sha256": reference_replay_digest,
            "grounding_sha256": grounding_digest,
            "history_mode": args.history_mode,
            "perturb_schedule": args.perturb_schedule,
            "initial_reference_replay_chunks": (
                len(replay) if not frozen_evaluation and args.resume is None else None
            ),
        }
        while control_steps < args.control_steps and (
            not frozen_evaluation or len(outcomes) < args.frozen_eval_episodes
        ):
            entry = select_layout(
                layouts,
                invalid_layouts,
                outcomes,
                episode_id,
                frozen_evaluation=frozen_evaluation,
            )
            wrapped = RoboDojoTransitions(
                native,
                layouts={entry["id"]: entry["sha256"]},
                protected_evaluation_hashes=protected,
                mode="probe" if frozen_evaluation else "training",
            )
            try:
                wrapped.reset(entry["id"])
            except Exception as exc:
                invalid_layouts.add(entry["id"])
                outcomes.append(
                    {
                        "episode": episode_id,
                        "layout_id": entry["id"],
                        "state": "reset_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                episode_id += 1
                continue
            native.model_client.call(
                func_name="begin_episode",
                obs={
                    "layout_id": entry["id"],
                    "sha256": entry["sha256"],
                    "episode_id": f"online-{episode_id}",
                },
            )
            records = []
            pending = None
            history_pending = None
            perturber = None
            phase = (
                phase_for_episode(perturb_schedule, episode_id)
                if perturb_schedule is not None
                else None
            )
            if history is not None:
                history.reset()
            while control_steps < args.control_steps:
                native.model_client.call(func_name="update_obs", obs=native.get_obs())
                prediction = native.model_client.call(func_name="get_action")
                current, state = encode_prediction(encoder, normalizer, prediction)
                if history is not None and history_pending is not None:
                    history.append(attach_successor(history_pending, current))
                    history_pending = None
                if tokenizer is not None:
                    window, mask = history.features()
                    if args.history_mode != "correct":
                        window, mask = history_view(
                            window, mask, args.history_mode, history_rng
                        )
                    token = tokenizer.token(
                        current["prop"], window, mask, current["ref"]
                    )
                    current = current | {"z": torch.cat([current["z"], token])}
                if pending is not None and not frozen_evaluation:
                    replay.append(attach_successor(pending, current))
                if phase is not None and phase["kind"] != "none" and not records:
                    perturber = make_perturber(phase, state)
                action = learner.predict(current, deterministic=frozen_evaluation)
                action = apply_frozen_action_adapter(
                    action,
                    current["ref"].detach().cpu().numpy(),
                    args.frozen_action_adapter,
                )
                physical = normalizer.decode(action, state)
                issued = normalizer.encode(physical, state)
                commands = decode_actions(
                    physical, semantics=ACTION_SEMANTICS, horizon=10
                )
                executed = []
                applied_commands = []
                reward = 0.0
                transition = None
                remaining = args.control_steps - control_steps
                for tick, command in enumerate(commands[:remaining]):
                    commanded = np.concatenate([command[name] for name, _ in FIELDS])
                    applied = (
                        perturber.step(commanded)
                        if perturber is not None
                        else commanded
                    )
                    applied_commands.append(applied)
                    transition = wrapped.step(applied, source="policy")
                    executed.append(transition.info["executed_action"])
                    reward += 0.99**tick * transition.reward
                    control_steps += 1
                    if transition.terminated or transition.truncated:
                        break
                executed_normalized = normalizer.encode(executed, state)
                # The acknowledged stream must match what this process applied:
                # under a perturbation that is the perturbed stream, otherwise
                # the issued chunk itself.
                applied_normalized = normalizer.encode(
                    np.asarray(applied_commands, dtype=np.float32), state
                )
                if not np.allclose(
                    executed_normalized, applied_normalized, atol=1e-6, rtol=0
                ):
                    raise RuntimeError(
                        "Executed online action differs from applied command stream"
                    )
                if perturber is None and not np.allclose(
                    executed_normalized, issued[: len(executed)], atol=1e-6, rtol=0
                ):
                    raise RuntimeError(
                        "Executed online action differs from issued chunk"
                    )
                deviation = action_deviation(
                    issued,
                    current["ref"].detach().cpu().numpy(),
                    len(executed),
                )
                if not frozen_evaluation:
                    pending = current | {
                        "action": torch.from_numpy(issued),
                        "executed_action": executed_normalized,
                        "reward": reward,
                        "ticks": len(executed),
                        "terminated": bool(transition.terminated),
                        "truncated": bool(transition.truncated),
                    }
                    if perturber is not None:
                        pending["perturbation"] = perturber.describe()
                if history is not None:
                    history_pending = {
                        "prop": current["prop"],
                        "action": torch.from_numpy(issued),
                        "executed_action": executed_normalized,
                        "reward": reward,
                        "ticks": len(executed),
                        "terminated": bool(transition.terminated),
                        "truncated": bool(transition.truncated),
                    }
                records.append(
                    {
                        "chunk": prediction["chunk"],
                        "ticks": len(executed),
                        "reward": reward,
                        "process_score": transition.info["process_score"],
                        "terminated": transition.terminated,
                        "truncated": transition.truncated,
                        "end_reason": transition.info["end_reason"],
                        "control_steps": control_steps,
                        **deviation,
                    }
                )
                if transition.terminated and not frozen_evaluation:
                    replay.append(attach_successor(pending, current))
                    pending = None
                if not frozen_evaluation and len(replay) >= args.batch_size:
                    for _ in range(args.updates_per_chunk):
                        metrics = learner.update(
                            rng.choices(replay, k=args.batch_size), warmup=False
                        )
                        updates += 1
                save(
                    args.output / f"episode-online-{episode_id}" / "online.json",
                    {
                        "layout": entry,
                        "records": records,
                        "perturbation": (
                            perturber.describe() if perturber is not None else None
                        ),
                        "complete": bool(transition.terminated or transition.truncated),
                    },
                )
                if frozen_evaluation:
                    save(
                        args.output / "online-progress.json",
                        {
                            "control_steps": control_steps,
                            "completed_episodes": len(outcomes)
                            + int(transition.terminated or transition.truncated),
                            "latest_reward": reward,
                            "latest_end_reason": transition.info["end_reason"],
                            **deviation,
                            "training": False,
                        },
                    )
                if not frozen_evaluation and control_steps % args.save_interval < len(
                    executed
                ):
                    checkpoint(
                        args.output / "online-latest.pt",
                        encoder,
                        learner,
                        checkpoint_meta
                        | {"control_steps": control_steps, "updates": updates},
                        restart_state(episode_id + 1),
                    )
                    save(
                        args.output / "online-progress.json",
                        {
                            "control_steps": control_steps,
                            "updates": updates,
                            "replay_chunks": len(replay),
                            "last_metrics": metrics
                            if len(replay) >= args.batch_size
                            else None,
                        },
                    )
                if transition.terminated or transition.truncated:
                    outcomes.append(
                        {
                            "episode": episode_id,
                            "layout_id": entry["id"],
                            "state": "complete",
                            "end_reason": transition.info["end_reason"],
                            "ticks": transition.info["control_step"],
                            "reward": sum(record["reward"] for record in records),
                            "max_process_score": max(
                                record["process_score"] for record in records
                            ),
                            "perturbation": (
                                perturber.describe() if perturber is not None else None
                            ),
                        }
                    )
                    break
            episode_id += 1
        if not frozen_evaluation:
            checkpoint(
                args.output / "online-final.pt",
                encoder,
                learner,
                checkpoint_meta | {"control_steps": control_steps, "updates": updates},
                restart_state(episode_id),
            )
        save(
            args.output / "online-result.json",
            {
                "complete": not frozen_evaluation
                or len(outcomes) == args.frozen_eval_episodes,
                "control_steps": control_steps,
                "updates": updates,
                "replay_chunks": len(replay),
                "reference_replay_sha256": reference_replay_digest,
                "outcomes": outcomes,
                "successes": sum(
                    row.get("end_reason") == "success" for row in outcomes
                ),
                "memory": (
                    {
                        "kind": "grounded_v1",
                        "token_dim": tokenizer.token_dim,
                        "history_length": tokenizer.history_length,
                        "history_mode": args.history_mode,
                        "grounding_sha256": grounding_digest,
                    }
                    if tokenizer is not None
                    else False
                ),
                "perturb_schedule": args.perturb_schedule,
                "intervention": False,
                "training": not frozen_evaluation,
                "frozen_eval_role": args.frozen_eval_role,
                "frozen_action_adapter": args.frozen_action_adapter,
                "resumed_from": str(args.resume) if args.resume else None,
                "exploration_std": args.exploration_std,
                "target_noise_std": args.target_noise_std,
                "target_noise_clip": args.target_noise_clip,
                "reference_anchor_scale": args.reference_anchor_scale,
            },
        )
    except BaseException as exc:
        traceback.print_exc()
        if args.output.exists():
            save(
                args.output / "client-failure.json",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        raise
    finally:
        if native is not None:
            try:
                native.model_client.close()
            finally:
                native.close()
        app.close()


def supervise(args: argparse.Namespace, config: dict) -> None:
    """Wait for warmup and one exclusive GPU, then own JAX/Isaac child groups."""
    if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        raise ValueError("Explicit operator EULA acceptance required")
    warmup_deadline = time.monotonic() + args.wait_artifact_hours * 3600
    while True:
        try:
            warmup_status = json.loads(args.warmup_status.read_text())
        except (OSError, ValueError):
            warmup_status = {}
        if warmup_status.get("state") == "complete" and warmup_status.get(
            "gate_passed"
        ):
            break
        if warmup_status.get("state") == "failed":
            raise RuntimeError("Required warmup failed")
        if not args.wait_artifact_hours or time.monotonic() >= warmup_deadline:
            raise TimeoutError("Warmup artifact did not become ready")
        time.sleep(min(30, max(0, warmup_deadline - time.monotonic())))
    root = Path(config["robodojo_root"])
    _source_check(root, ROBO_DOJO_REVISION)
    _source_check(root / "XPolicyLab", XPOLICYLAB_REVISION)
    verify(args.layouts)
    reference_replay = args.warmup.parent / "encoded-rollouts.pt"
    if not reference_replay.is_file():
        raise FileNotFoundError(
            f"Warmup companion artifact is missing: {reference_replay}"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    status = {
        "state": "starting",
        "online_rl": args.frozen_eval_role is None,
        "gpu_index": args.gpu,
    }
    save(
        args.output / "provenance.json",
        {
            "config": config,
            "warmup": str(args.warmup),
            "warmup_status": warmup_status,
            "layout_manifest": str(args.layouts),
            "layout_sha256": digest(args.layouts),
            "gpu": args.gpu,
            "seed": args.seed,
            "resume": str(args.resume) if args.resume else None,
            "exploration_std": args.exploration_std,
            "target_noise_std": args.target_noise_std,
            "target_noise_clip": args.target_noise_clip,
            "reference_anchor_scale": args.reference_anchor_scale,
            "frozen_eval_role": args.frozen_eval_role,
            "frozen_eval_episodes": args.frozen_eval_episodes,
            "frozen_action_adapter": args.frozen_action_adapter,
            "grounding": str(args.grounding) if args.grounding else None,
            "history_mode": args.history_mode,
            "perturb_schedule": args.perturb_schedule,
            "coexist_users": sorted(args.coexist_user),
            "coexist_start_memory_mib": args.coexist_start_memory_mib,
            "coexist_runtime_memory_mib": args.coexist_runtime_memory_mib,
        },
    )
    children = []
    lease = Path(f"/dev/shm/rlt-research-gpu{args.gpu}.lock").open("a")

    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle_deadline = time.monotonic() + args.wait_idle_hours * 3600
        checks = 0
        allowed_users = set(args.coexist_user)
        while checks < 2:
            idle, detail = gpu_idle(args.gpu)
            foreign = foreign_gpu_processes([], args.gpu)
            disallowed = disallowed_gpu_processes(foreign, allowed_users)
            memory_used_mib = gpu_memory_used_mib(args.gpu)
            coexist_ready = bool(allowed_users) and (
                not disallowed and memory_used_mib <= args.coexist_start_memory_mib
            )
            checks = checks + 1 if (idle and not foreign) or coexist_ready else 0
            status.update(
                state="waiting_gpu",
                gpu=detail,
                foreign_pids=foreign,
                foreign_processes=process_details(foreign),
                disallowed_foreign_pids=disallowed,
                coexist_users=sorted(allowed_users),
                memory_used_mib=memory_used_mib,
                idle_checks=checks,
            )
            save(args.output / "status.json", status)
            if checks >= 2:
                break
            if not args.wait_idle_hours or time.monotonic() >= idle_deadline:
                raise RuntimeError(f"GPU {args.gpu} occupied: {detail}; pids={foreign}")
            time.sleep(min(30, max(0, idle_deadline - time.monotonic())))
        uuid = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        renderer = renderer_index(args.renderer_table.read_text(), uuid)
        environment = dict(os.environ)
        environment.pop("LD_PRELOAD", None)
        environment.update(
            CUDA_VISIBLE_DEVICES=uuid,
            LD_LIBRARY_PATH="",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            NO_PROXY="127.0.0.1,localhost",
            no_proxy="127.0.0.1,localhost",
            ROBODOJO_RUN_ID=args.output.name,
            PYTHONPATH=os.pathsep.join(
                map(
                    str,
                    (
                        Path(__file__).resolve().parents[2],
                        args.source,
                        root,
                        root / "XPolicyLab",
                    ),
                )
            ),
        )
        server_args = [
            "-u",
            "-m",
            "toolkits.rlt.robodojo_collect",
            "--config",
            str(args.config),
            "--output",
            str(args.output),
            "--layouts",
            str(args.layouts),
            "--port",
            str(args.port),
            "--server",
        ]
        if args.frozen_eval_role == "reserved":
            server_args.extend(
                (
                    "--train-episodes",
                    "0",
                    "--development-episodes",
                    "0",
                    "--reserved-evaluation-episodes",
                    str(args.frozen_eval_episodes),
                )
            )
        with (
            (args.output / "server.log").open("x") as server_log,
            (args.output / "client.log").open("x") as client_log,
        ):
            server = subprocess.Popen(
                [config["policy_python"], *server_args],
                cwd=root,
                env=environment,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(server)
            deadline = time.monotonic() + 600
            while not owns_listener(server.pid, args.port):
                server_code = server.poll()
                timed_out = time.monotonic() >= deadline
                foreign = foreign_gpu_processes(children, args.gpu)
                disallowed = disallowed_gpu_processes(foreign, allowed_users)
                memory_used_mib = gpu_memory_used_mib(args.gpu)
                memory_exhausted = bool(allowed_users) and (
                    memory_used_mib > args.coexist_runtime_memory_mib
                )
                if (
                    server_code is not None
                    or timed_out
                    or disallowed
                    or memory_exhausted
                ):
                    raise RuntimeError(
                        "Online feature server failed to start: "
                        f"server_returncode={server_code}, "
                        f"timed_out={timed_out}, foreign_pids={foreign}, "
                        f"disallowed_pids={disallowed}, "
                        f"memory_used_mib={memory_used_mib}"
                    )
                time.sleep(2)
            client_args = [
                "-u",
                "-m",
                "toolkits.rlt.robodojo_online",
                "--client",
                "--config",
                str(args.config),
                "--layouts",
                str(args.layouts),
                "--output",
                str(args.output),
                "--warmup",
                str(args.warmup),
                "--source",
                str(args.source),
                "--port",
                str(args.port),
                "--renderer-index",
                str(renderer),
                "--control-steps",
                str(args.control_steps),
                "--batch-size",
                str(args.batch_size),
                "--updates-per-chunk",
                str(args.updates_per_chunk),
                "--save-interval",
                str(args.save_interval),
                "--seed",
                str(args.seed),
            ]
            if args.resume is not None:
                client_args.extend(("--resume", str(args.resume)))
            if args.grounding is not None:
                client_args.extend(("--grounding", str(args.grounding)))
            if args.history_mode != "correct":
                client_args.extend(("--history-mode", args.history_mode))
            if args.perturb_schedule is not None:
                client_args.extend(("--perturb-schedule", args.perturb_schedule))
            client_args.extend(
                (
                    "--exploration-std",
                    str(args.exploration_std),
                    "--target-noise-std",
                    str(args.target_noise_std),
                    "--target-noise-clip",
                    str(args.target_noise_clip),
                    "--reference-anchor-scale",
                    str(args.reference_anchor_scale),
                )
            )
            if args.frozen_eval_role is not None:
                client_args.extend(
                    (
                        "--frozen-eval-role",
                        args.frozen_eval_role,
                        "--frozen-eval-episodes",
                        str(args.frozen_eval_episodes),
                        "--frozen-action-adapter",
                        args.frozen_action_adapter,
                    )
                )
            client_env = environment | {
                "LD_LIBRARY_PATH": str(args.sim_library_path),
                "VK_ICD_FILENAMES": str(args.vulkan_icd),
                "VK_DRIVER_FILES": str(args.vulkan_icd),
            }
            process = subprocess.Popen(
                [config["sim_python"], *client_args],
                cwd=root,
                env=client_env,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(process)
            status.update(
                state="evaluating" if args.frozen_eval_role else "training",
                server_pid=server.pid,
                client_pid=process.pid,
            )
            save(args.output / "status.json", status)
            deadline = time.monotonic() + args.timeout_seconds
            last_progress = time.monotonic()
            progress_mtime = None
            while process.poll() is None:
                server_code = server.poll()
                foreign = foreign_gpu_processes(children, args.gpu)
                disallowed = disallowed_gpu_processes(foreign, allowed_users)
                memory_used_mib = gpu_memory_used_mib(args.gpu)
                if server_code is not None:
                    raise RuntimeError(
                        f"Online feature server exited with code {server_code}"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Online client exceeded {args.timeout_seconds} seconds"
                    )
                if disallowed:
                    status.update(
                        foreign_processes=process_details(foreign),
                        disallowed_foreign_pids=disallowed,
                    )
                    save(args.output / "status.json", status)
                    raise RuntimeError(
                        f"GPU {args.gpu} acquired by disallowed processes: {disallowed}"
                    )
                if allowed_users and memory_used_mib > args.coexist_runtime_memory_mib:
                    status.update(memory_used_mib=memory_used_mib)
                    save(args.output / "status.json", status)
                    raise RuntimeError(
                        f"GPU {args.gpu} memory crossed coexistence limit: "
                        f"{memory_used_mib} MiB"
                    )
                progress = args.output / "online-progress.json"
                current_mtime = progress.stat().st_mtime if progress.exists() else None
                if current_mtime != progress_mtime:
                    progress_mtime = current_mtime
                    last_progress = time.monotonic()
                if renderer_failed(args.output / "client.log"):
                    raise RuntimeError(
                        "Isaac renderer exhausted GPU descriptor resources"
                    )
                if time.monotonic() - last_progress > args.stall_timeout_seconds:
                    raise RuntimeError(
                        f"Online client made no checkpoint progress for "
                        f"{args.stall_timeout_seconds} seconds"
                    )
                time.sleep(2)
            result_path = args.output / "online-result.json"
            if not result_path.is_file():
                raise RuntimeError(
                    "Online client exited without a result: "
                    f"client_returncode={process.returncode}, "
                    f"server_returncode={server.poll()}"
                )
            result = json.loads(result_path.read_text())
            if process.returncode or not result.get("complete"):
                raise RuntimeError("Online client did not publish completion")
            status.update(state="complete", result=result)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for child in reversed(children):
            stop_group(child)
        lease.close()
        status["finished_at"] = time.time()
        if args.output.exists():
            save(args.output / "status.json", status)
        signal.signal(signal.SIGTERM, previous_handler)


def main() -> None:
    """Run the Isaac client or the GPU-owning, warmup-gated supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "layouts", "output", "warmup", "source"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--renderer-index", type=int)
    parser.add_argument("--port", type=int, default=6135)
    parser.add_argument("--control-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-chunk", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--exploration-std", type=float, default=0.01)
    parser.add_argument("--target-noise-std", type=float, default=0.02)
    parser.add_argument("--target-noise-clip", type=float, default=0.05)
    parser.add_argument("--reference-anchor-scale", type=float, default=1.0)
    parser.add_argument("--stall-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--frozen-eval-role", choices=("train", "development", "reserved")
    )
    parser.add_argument("--frozen-eval-episodes", type=int, default=0)
    parser.add_argument(
        "--frozen-action-adapter",
        choices=("none", "reference_grippers"),
        default="none",
    )
    parser.add_argument("--grounding", type=Path)
    parser.add_argument(
        "--history-mode", choices=("correct", "shuffled", "zero"), default="correct"
    )
    parser.add_argument("--perturb-schedule")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--warmup-status", type=Path)
    parser.add_argument("--wait-artifact-hours", type=float, default=0)
    parser.add_argument("--wait-idle-hours", type=float, default=0)
    parser.add_argument("--coexist-user", action="append", default=[])
    parser.add_argument("--coexist-start-memory-mib", type=int, default=0)
    parser.add_argument("--coexist-runtime-memory-mib", type=int, default=0)
    parser.add_argument("--renderer-table", type=Path)
    parser.add_argument("--sim-library-path", type=Path)
    parser.add_argument("--vulkan-icd", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    args = parser.parse_args()
    if (
        not 1 <= args.control_steps <= 100000
        or not 1 <= args.batch_size <= 512
        or not 1 <= args.updates_per_chunk <= 16
        or not 10 <= args.save_interval <= args.control_steps
        or not 0 <= args.wait_artifact_hours <= 72
        or not 0 <= args.wait_idle_hours <= 72
        or not 0 <= args.gpu < 32
        or args.stall_timeout_seconds < 60
        or not 0 <= args.frozen_eval_episodes <= 25
    ):
        parser.error("Invalid bounded online training configuration")
    if not 0 <= args.exploration_std <= 0.1:
        parser.error("Exploration std must be in [0, 0.1]")
    if not 0 <= args.target_noise_std <= args.target_noise_clip <= 0.5:
        parser.error("Invalid target policy smoothing")
    if not 0 <= args.reference_anchor_scale <= 1:
        parser.error("Reference anchor scale must be in [0, 1]")
    if bool(args.coexist_user) != bool(args.coexist_start_memory_mib):
        parser.error("Coexistence requires users and a positive startup memory limit")
    if bool(args.coexist_user) != bool(args.coexist_runtime_memory_mib):
        parser.error("Coexistence requires users and a positive runtime memory limit")
    if args.coexist_user and (
        args.coexist_start_memory_mib >= args.coexist_runtime_memory_mib
        or args.coexist_runtime_memory_mib > 45000
        or any(
            re.fullmatch(r"[a-z_][a-z0-9_-]*", user) is None
            for user in args.coexist_user
        )
    ):
        parser.error("Invalid bounded GPU coexistence configuration")
    if args.frozen_eval_role is None and args.reference_anchor_scale == 0:
        parser.error("Online training requires a positive reference anchor scale")
    if args.resume is not None and not args.resume.is_file():
        parser.error("Resume checkpoint does not exist")
    if (args.frozen_eval_role is None) != (args.frozen_eval_episodes == 0):
        parser.error("Frozen evaluation requires a role and positive episode count")
    if args.frozen_eval_role is None and args.frozen_action_adapter != "none":
        parser.error("Frozen action adapters cannot modify online training")
    if args.grounding is not None and not args.grounding.is_file():
        parser.error("Grounding artifact does not exist")
    if args.history_mode != "correct" and args.grounding is None:
        parser.error("History ablation modes require a grounding artifact")
    if args.perturb_schedule is not None:
        try:
            parse_perturb_schedule(args.perturb_schedule)
        except ValueError as error:
            parser.error(f"Invalid perturbation schedule: {error}")
    config = yaml.safe_load(args.config.read_text())
    if args.client:
        if args.renderer_index is None:
            parser.error("Client requires renderer index")
        client(args, config)
    else:
        if not all(
            (
                args.warmup_status,
                args.renderer_table,
                args.sim_library_path,
                args.vulkan_icd,
            )
        ):
            parser.error("Supervisor requires warmup status and graphics paths")
        supervise(args, config)


if __name__ == "__main__":
    main()
