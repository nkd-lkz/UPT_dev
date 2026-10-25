# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Collect causally aligned LIBERO-Pro action features and physical targets."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from toolkits.rlt.libero_pro_environment import LiberoProEnvironment
from toolkits.rlt.physrlt.data import (
    HORIZONS,
    PhysicalReference,
    check_split,
    measured_snapshot,
    physical_reset_hash,
    transition_targets,
)
from toolkits.rlt.pi05_baseline import (
    configure_gpu,
    configure_source,
    load_model,
    write_json,
)


def save_dataset(
    path: Path, rows: list[dict], episodes: list[dict], meta: dict
) -> None:
    """Publish complete chunks atomically so a failed episode is recoverable."""
    check_split(rows)
    temp = path.with_suffix(".tmp")
    torch.save({"schema": 1, "rows": rows, "episodes": episodes, "meta": meta}, temp)
    temp.replace(path)


def run(args: argparse.Namespace) -> dict:
    """Capture one task with fixed layout splits and explicit physical units."""
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "transitions.pt").exists():
        raise FileExistsError("Refusing to overwrite a collected dataset")
    device = configure_gpu(args.gpu, args.output)
    reservation = torch.empty(1, device="cuda")
    configure_source(args.source)
    torch.set_num_threads(4)
    model, config = load_model(args.assets, "cuda")
    reference = PhysicalReference(model)
    cases = json.loads(args.cases.read_text())
    env = LiberoProEnvironment()
    rows, episodes, parity = [], [], None
    started = time.monotonic()
    meta = {
        "benchmark": "libero_pro",
        "backbone": "AlphaBrain RLT pi05 flow matching",
        "feature": "last action_out_proj input; fixed four-bin projection",
        "prefix_control": "same forward real-view prefix; same projection width",
        "horizons_control_ticks": list(HORIZONS),
        "depth_units": "metres; OpenGL converted using calibrated near/far",
        "response_units": "joint radians, end-effector metres, gripper metres",
        "geometry_inputs_at_inference": False,
        "training_only_targets": ["metric_depth_delta", "object_position_delta"],
        "feature_projection_seed": 25671,
        "noise_std_native_translation": args.noise,
        "device": device,
        "model_config": config,
        "cases": cases,
    }
    write_json(args.output / "manifest.json", meta)
    try:
        for case in cases:
            obs = env.reset(case)
            # The upstream adapter consumes the first of the ten settling steps.
            for _ in range(9):
                obs, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
            digest = physical_reset_hash(env)
            if parity is None:
                plain = reference.reference.predict(obs, env.instruction, seed=715)
                hooked = reference.predict(obs, env.instruction, seed=715)
                error = float(np.max(np.abs(plain["ref"] - hooked["ref"])))
                parity = {"max_absolute_error": error, "passed": error == 0}
                write_json(args.output / "capture_parity.json", parity)
                if error:
                    raise RuntimeError("Capturing action features changed reference")
            success, ticks, episode_chunks = False, 0, 0
            rng = np.random.default_rng(case["seed"])
            while ticks < args.ticks:
                before = measured_snapshot(env)
                prediction = reference.predict(
                    obs, env.instruction, seed=case["seed"] * 1000 + ticks
                )
                commands = prediction["commands"].copy()
                # Reference and accepted commands remain separate in the ledger.
                # Exploration is for transition identification, not RL reward.
                perturbed = episode_chunks % 2 == 1 and args.noise > 0
                if perturbed:
                    commands[:, :3] += rng.normal(0, args.noise, (10, 3))
                    commands[:, :3] = commands[:, :3].clip(-1, 1)
                following = {}
                accepted = np.zeros_like(commands)
                reward_sum = 0.0
                initial_prop = torch.tensor(obs["state"], dtype=torch.float32)
                start_tick = ticks
                for offset, command in enumerate(commands, 1):
                    obs, reward, done = env.step(command)
                    accepted[offset - 1] = command
                    reward_sum += reward
                    ticks += 1
                    if offset in HORIZONS:
                        following[offset] = measured_snapshot(
                            env, before["object_keys"]
                        )
                    success = success or done
                    if done or ticks >= args.ticks:
                        break
                row = {
                    "case_id": case["id"],
                    "suite": case["suite"],
                    "task": case["task"],
                    "state_id": case["state"],
                    "split": case["split"],
                    "seed": case["seed"],
                    "perturbation": case["perturbation"],
                    "reset_hash": digest,
                    "start_tick": start_tick,
                    "end_tick": ticks,
                    "execution_ticks": ticks - start_tick,
                    "action_feature": prediction["action_feature"],
                    "prefix_feature": prediction["prefix_feature"],
                    "ref": torch.from_numpy(prediction["ref"].copy()),
                    "commands": torch.from_numpy(accepted),
                    "prop": initial_prop,
                    "reward": reward_sum,
                    "terminal": success,
                    "perturbed": perturbed,
                    "denoise_forwards": prediction["denoise_forwards"],
                    "object_keys": before["object_keys"],
                    **transition_targets(before, following),
                }
                rows.append(row)
                episode_chunks += 1
                if success:
                    break
            episode = {
                "case": case,
                "reset_hash": digest,
                "success": success,
                "ticks": ticks,
                "chunks": episode_chunks,
                "noisy_collection_not_reference_success_rate": args.noise > 0,
            }
            episodes.append(episode)
            save_dataset(args.output / "transitions.pt", rows, episodes, meta)
            write_json(args.output / "episodes.json", episodes)
            print(json.dumps(episode), flush=True)
    finally:
        env.close()
        del reservation
    result = {
        "state": "complete",
        "episodes": len(episodes),
        "chunks": len(rows),
        "parity": parity,
        "seconds": time.monotonic() - started,
        "role": "development transition collection, not a success benchmark",
    }
    write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.03)
    args = parser.parse_args()
    if os.environ.get("LIBERO_TYPE") != "pro" or not 10 <= args.ticks <= 520:
        parser.error("Require LIBERO_TYPE=pro and 10..520 control ticks")
    if not 0 <= args.noise <= 0.1:
        parser.error("Translation noise must be within the declared pilot budget")
    run(args)


if __name__ == "__main__":
    main()
