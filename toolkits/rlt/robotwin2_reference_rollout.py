# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run pinned RoboTwin 2.0 Pi0.5 reference rollouts on one local GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from toolkits.rlt.robotwin2_features import FrozenRoboTwinPi05


def build_env_config(
    repo: Path,
    robotwin: Path,
    output: Path,
    *,
    split: str = "eval",
) -> Any:
    """Create the published clean ALOHA environment on a declared seed split."""
    if split not in {"train", "eval"}:
        raise ValueError("RoboTwin split must be 'train' or 'eval'")
    cfg = OmegaConf.load(
        repo / "examples/embodiment/config/env/robotwin_adjust_bottle.yaml"
    )
    overrides = OmegaConf.create(
        {
            "auto_reset": False,
            "ignore_terminations": False,
            "reward_coef": 1.0,
            "use_custom_reward": True,
            "use_rel_reward": True,
            "center_crop": False,
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "max_steps_per_rollout_epoch": 200,
            "max_episode_steps": 200,
            "is_eval": split == "eval",
            "assets_path": str(robotwin),
            "seeds_path": str(
                repo / f"rlinf/envs/sim/robotwin/seeds/{split}_seeds.json"
            ),
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": str(output / "video"),
            },
            "task_config": {
                "embodiment": ["aloha-agilex"],
                "camera": {"collect_wrist_camera": True},
                "domain_randomization": {
                    "random_background": False,
                    "cluttered_table": False,
                    "clean_background_rate": 1.0,
                    "random_head_camera_dis": 0,
                    "random_table_height": 0,
                    "random_light": False,
                    "crazy_random_light_rate": 0.0,
                },
            },
        }
    )
    return OmegaConf.merge(cfg, overrides)


def scalar(value: Any) -> Any:
    """Convert a one-environment tensor, array, or list to a JSON scalar."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return array.tolist()


def task_diagnostics(env: Any) -> dict[str, Any]:
    """Measure the task geometry needed to localize a failed episode."""
    task = env.venv.envs[0].task
    arm = "right" if int(task.qpose_tag) == 1 else "left"
    bottle = np.asarray(task.bottle.get_functional_point(0), dtype=np.float64)
    robot = task.robot
    tcp = np.asarray(
        robot.get_right_tcp_pose() if arm == "right" else robot.get_left_tcp_pose(),
        dtype=np.float64,
    )[:3]
    gripper = (
        robot.get_right_gripper_val()
        if arm == "right"
        else robot.get_left_gripper_val()
    )
    return {
        "arm": arm,
        "bottle_xyz": bottle[:3].tolist(),
        "tcp_xyz": tcp.tolist(),
        "tcp_bottle_distance": float(np.linalg.norm(tcp - bottle[:3])),
        "gripper": float(gripper),
    }


def classify_failure(initial: dict, records: list[dict], success: bool) -> str:
    """Assign a geometric Pick/Place failure stage for adjust_bottle."""
    if success:
        return "success"
    diagnostics = [initial, *[record["diagnostics"] for record in records]]
    if min(item["tcp_bottle_distance"] for item in diagnostics) > 0.12:
        return "reach"
    if max(item["bottle_xyz"][2] for item in diagnostics) < 0.82:
        return "grasp_or_lift"
    arm = initial["arm"]
    transported = any(
        item["bottle_xyz"][0] > 0.15
        if arm == "right"
        else item["bottle_xyz"][0] < -0.15
        for item in diagnostics
    )
    if not transported:
        return "transport"
    return "place_height_or_stability"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one episode and cache the pre-action RLT inputs."""
    repo = args.repo.resolve()
    robotwin = args.robotwin.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve() / f"h{args.execution_horizon}"
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(robotwin) not in sys.path:
        sys.path.insert(0, str(robotwin))

    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

    cfg = build_env_config(repo, robotwin, output)
    cfg.seed = args.seed
    env = RoboTwinEnv(
        cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    policy = FrozenRoboTwinPi05(checkpoint, device="cuda")
    records = []
    start = time.monotonic()
    try:
        obs, _ = env.reset()
        initial_diagnostics = task_diagnostics(env)
        while int(env.elapsed_steps[0].item()) < cfg.max_episode_steps:
            inference = policy.infer(obs)
            reference = inference["reference"]
            state = obs["states"]
            record_index = len(records)
            cache_path = output / f"decision-{record_index:03d}.npz"
            np.savez_compressed(
                cache_path,
                hidden=inference["hidden"][0].cpu().numpy(),
                mask=inference["mask"][0].cpu().numpy(),
                state=state[0].detach().cpu().numpy(),
                reference=reference[0].cpu().numpy(),
            )
            issued = reference[:, : args.execution_horizon]
            obs, reward, terminated, truncated, infos = env.step(
                issued, auto_reset=False
            )
            record = {
                "decision": record_index,
                "start_tick": int(env.elapsed_steps[0].item()) - args.execution_horizon,
                "ticks": args.execution_horizon,
                "reward": float(scalar(reward)),
                "terminated": bool(scalar(terminated)),
                "truncated": bool(scalar(truncated)),
                "cache_file": str(cache_path),
                "diagnostics": task_diagnostics(env),
            }
            if "success" in infos:
                record["success"] = bool(scalar(infos["success"]))
            records.append(record)
            if record["terminated"] or record["truncated"]:
                break
    finally:
        env.offload(clear_cache=True)

    success = any(record.get("success", False) for record in records)
    result = {
        "schema_version": 1,
        "benchmark": "robotwin2",
        "task": "adjust_bottle",
        "policy": "RLinf-Pi05-RoboTwin-SFT-adjust_bottle",
        "reference_horizon": 50,
        "execution_horizon": args.execution_horizon,
        "requested_seed": args.seed,
        "environment_seed": int(env.reset_state_ids[0].item()),
        "elapsed_ticks": int(env.elapsed_steps[0].item()),
        "wall_seconds": time.monotonic() - start,
        "success": success,
        "terminated": any(record["terminated"] for record in records),
        "initial_diagnostics": initial_diagnostics,
        "failure_stage": classify_failure(initial_diagnostics, records, success),
        "records": records,
    }
    result_path = output / "episode.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--robotwin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-horizon", type=int, choices=(10, 25, 50), default=50
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
