# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run a small, explicit LIBERO-Pro Pi0.5 reference baseline on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


PERTURBATIONS = {"lan", "object", "swap", "task"}


def atomic_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Publish an experiment ledger atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def model_config(checkpoint: Path) -> Any:
    """Mirror the released RLinf Pi0.5 LIBERO evaluation contract."""
    norm_stats = checkpoint / "physical-intelligence/libero/norm_stats.json"
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError("LIBERO reference requires model.safetensors")
    if not norm_stats.is_file():
        raise FileNotFoundError(f"Missing LIBERO normalization: {norm_stats}")
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": str(checkpoint),
            "precision": "bf16",
            "pi05": True,
            "num_action_chunks": 10,
            "action_dim": 7,
            "num_steps": 5,
            "add_value_head": False,
            "openpi_data": {"norm_stats_path": str(norm_stats)},
            "openpi": {
                "task": "eval",
                "config_name": "pi05_libero",
                "num_images_in_input": 2,
                "action_chunk": 10,
                "action_env_dim": 7,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "torch_compile": False,
            },
        }
    )


def env_config(output: Path, *, reset_id: int, seed: int) -> Any:
    """Create one fixed-state LIBERO-Pro evaluation environment."""
    return OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": "libero_10",
            "total_num_envs": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": 480,
            "max_episode_steps": 480,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": False,
            "use_rel_reward": True,
            "reward_coef": 5.0,
            "reset_gripper_open": True,
            "is_eval": True,
            "seed": seed,
            "group_size": 1,
            "specific_reset_id": reset_id,
            "task_id_filter": None,
            "num_steps_wait": 15,
            "skip_intermediate_renders": False,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": str(output / "video"),
            },
            "init_params": {"camera_heights": 256, "camera_widths": 256},
        }
    )


def parse_case(value: str) -> dict[str, Any]:
    """Parse TASK:STATE:PERTURBATION:SEED without implicit benchmark wrapping."""
    try:
        task_text, state_text, perturbation, seed_text = value.split(":")
        task, state, seed = int(task_text), int(state_text), int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "cases must be TASK:STATE:PERTURBATION:SEED"
        ) from error
    if task not in range(10) or state < 0 or seed < 0:
        raise argparse.ArgumentTypeError("invalid task, state, or seed")
    if perturbation not in PERTURBATIONS:
        raise argparse.ArgumentTypeError(
            f"perturbation must be one of {sorted(PERTURBATIONS)}"
        )
    return {
        "task_id": task,
        "state_id": state,
        "perturbation": perturbation,
        "seed": seed,
    }


def global_reset_id(task_id: int, state_id: int, trial_bins: list[int]) -> int:
    """Convert an explicit task/trial pair to LIBERO's suite-wide state ID."""
    if task_id >= len(trial_bins) or state_id >= trial_bins[task_id]:
        raise ValueError(
            f"task {task_id} has {trial_bins[task_id]} states, got {state_id}"
        )
    return sum(trial_bins[:task_id]) + state_id


def scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Expected one-environment scalar")
    return array.reshape(-1)[0].item()


def close_env(env: Any) -> None:
    """Close the local subprocess vector environment."""
    if hasattr(env, "env"):
        try:
            env.env.close()
        except (BrokenPipeError, ConnectionResetError):
            # A malformed benchmark case may already have killed its worker.
            # Preserve the original reset/step exception in that situation.
            pass


def run_case(
    args: argparse.Namespace,
    model: Any,
    case: dict[str, Any],
    trial_bins: list[int],
) -> dict[str, Any]:
    """Run one fixed task/state/perturbation episode to success or 480 ticks."""
    from rlinf.envs.sim.libero.libero_env import LiberoEnv

    os.environ["LIBERO_PERTURBATION"] = case["perturbation"]
    reset_id = global_reset_id(case["task_id"], case["state_id"], trial_bins)
    cfg = env_config(args.output, reset_id=reset_id, seed=case["seed"])
    np.random.seed(case["seed"])
    torch.manual_seed(case["seed"])
    env = LiberoEnv(
        cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    decisions = []
    started = time.monotonic()
    try:
        observation, _ = env.reset()
        task = env.task_suite.get_task(case["task_id"])
        task_name = str(
            getattr(task, "name", None)
            or getattr(task, "language", None)
            or task.bddl_file
        )
        instruction = str(observation["task_descriptions"][0])
        selected_folder = str(env._pert_init_folders[0])
        success = False
        for decision in range(48):
            with torch.no_grad():
                actions, _ = model.predict_action_batch(observation, mode="eval")
            actions = actions.detach().float().cpu().numpy()
            if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 7:
                raise ValueError(f"Unexpected Pi0.5 action shape: {actions.shape}")
            chunk = actions[0, :10]
            tick_rewards = []
            executed = 0
            for action in chunk:
                observation, reward, terminated, truncated, _ = env.step(
                    action[None], auto_reset=False
                )
                executed += 1
                tick_rewards.append(float(scalar(reward)))
                success = success or bool(scalar(terminated))
                if bool(scalar(terminated)) or bool(scalar(truncated)):
                    break
            decisions.append(
                {
                    "decision": decision,
                    "start_tick": int(env.elapsed_steps[0]) - executed,
                    "ticks": executed,
                    "reward": float(sum(tick_rewards)),
                    "action_rms": float(np.sqrt(np.mean(chunk[:executed] ** 2))),
                }
            )
            if success or int(env.elapsed_steps[0]) >= cfg.max_episode_steps:
                break
    finally:
        close_env(env)
    return {
        "schema_version": 1,
        "state": "complete",
        "benchmark": "libero_pro",
        "suite": "libero_10",
        "policy": "RLinf-Pi05-LIBERO-SFT",
        **case,
        "global_reset_id": reset_id,
        "task_name": task_name,
        "instruction": instruction,
        "selected_perturbation_folder": selected_folder,
        "success": success,
        "elapsed_ticks": int(env.elapsed_steps[0]),
        "decisions": decisions,
        "wall_seconds": time.monotonic() - started,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load Pi0.5 once and incrementally publish all explicit cases."""
    if os.environ.get("LIBERO_TYPE") != "pro":
        raise RuntimeError("Set LIBERO_TYPE=pro; refusing a silent standard fallback")
    from liberopro.liberopro.benchmark import get_benchmark
    from rlinf.models.embodiment.openpi import get_model

    suite = get_benchmark("libero_10")()
    trial_bins = [
        len(suite.get_task_init_states(task_id))
        for task_id in range(suite.get_num_tasks())
    ]
    if len(trial_bins) != 10 or any(count <= 0 for count in trial_bins):
        raise ValueError(f"Unexpected LIBERO-10 trial bins: {trial_bins}")
    model = get_model(model_config(args.checkpoint.resolve())).to(args.device).eval()
    model.requires_grad_(False)
    episodes = []
    for index, case in enumerate(args.cases):
        try:
            episode = run_case(args, model, case, trial_bins)
        except Exception as error:
            # A malformed perturbation BDDL or failed simulator worker is a
            # benchmark-infrastructure result, not a policy failure.  Persist
            # it separately and continue screening the remaining explicit
            # cases instead of silently turning it into a zero reward.
            episode = {
                "schema_version": 1,
                "state": "failed",
                "benchmark": "libero_pro",
                "suite": "libero_10",
                "policy": "RLinf-Pi05-LIBERO-SFT",
                **case,
                "success": None,
                "error": f"{type(error).__name__}: {error}",
            }
        episodes.append(episode)
        atomic_json(args.output / f"episode-{index:03d}.json", episode)
        atomic_json(args.output / "episodes.json", episodes)
    completed = [episode for episode in episodes if episode["state"] == "complete"]
    by_perturbation = {}
    by_task_id = {}
    for key, field in (("by_perturbation", "perturbation"), ("by_task_id", "task_id")):
        target = by_perturbation if key == "by_perturbation" else by_task_id
        for value in sorted({episode[field] for episode in episodes}):
            selected = [episode for episode in episodes if episode[field] == value]
            valid = [episode for episode in selected if episode["state"] == "complete"]
            target[value] = {
                "attempted": len(selected),
                "completed": len(valid),
                "infrastructure_failures": len(selected) - len(valid),
                "successes": sum(episode["success"] for episode in valid),
                "success_rate": float(np.mean([episode["success"] for episode in valid]))
                if valid
                else None,
            }
    summary = {
        "schema_version": 1,
        "benchmark": "libero_pro",
        "suite": "libero_10",
        "policy": "RLinf-Pi05-LIBERO-SFT",
        "checkpoint": str(args.checkpoint.resolve()),
        "protocol": {
            "action_horizon": 10,
            "execution_horizon": 10,
            "flow_steps": 5,
            "max_episode_steps": 480,
            "settling_steps": 15,
            "fixed_explicit_states": True,
        },
        "trial_bins": trial_bins,
        "attempted_cases": len(episodes),
        "completed_episodes": len(completed),
        "infrastructure_failures": len(episodes) - len(completed),
        "successes": sum(episode["success"] for episode in completed),
        "success_rate": float(np.mean([episode["success"] for episode in completed]))
        if completed
        else None,
        "by_perturbation": by_perturbation,
        "by_task_id": by_task_id,
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=parse_case, action="append", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
