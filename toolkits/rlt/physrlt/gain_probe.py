# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Test whether one completed transition identifies an actuation-gain change."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from toolkits.rlt.libero_pro_environment import LiberoProEnvironment
from toolkits.rlt.physrlt.data import (
    HORIZONS,
    measured_snapshot,
    physical_reset_hash,
    transition_targets,
)
from toolkits.rlt.pi05_baseline import configure_gpu, configure_source, write_json


def realized_translation_score(commands: torch.Tensor, response: torch.Tensor) -> float:
    """Project measured end-effector motion onto the issued translation command."""
    issued = commands[:, :3].sum(0).float()
    denominator = issued.square().sum()
    if denominator <= 1e-8:
        return float("nan")
    measured = response[7:10].float()
    return float(torch.dot(measured, issued) / denominator)


def summarize(records: list[dict], low_gain: float) -> dict:
    """Report paired condition ordering without fitting on held-out labels."""
    grouped = defaultdict(dict)
    for row in records:
        key = (row["split"], row["case_id"], row["start_tick"], row["horizon"])
        grouped[key][float(row["execution_gain"])] = row["translation_score"]
    summaries = {}
    for split in sorted({key[0] for key in grouped}):
        summaries[split] = {}
        for horizon in HORIZONS:
            pairs = []
            for key, values in grouped.items():
                if key[0] != split or key[3] != horizon:
                    continue
                nominal = values.get(1.0)
                reduced = values.get(low_gain)
                if nominal is None or reduced is None:
                    continue
                if np.isfinite(nominal) and np.isfinite(reduced):
                    pairs.append((nominal, reduced))
            margins = [nominal - reduced for nominal, reduced in pairs]
            summaries[split][str(horizon)] = {
                "paired_states": len(pairs),
                "nominal_score_mean": float(np.mean([p[0] for p in pairs]))
                if pairs
                else None,
                "reduced_score_mean": float(np.mean([p[1] for p in pairs]))
                if pairs
                else None,
                "paired_order_accuracy": float(np.mean([m > 0 for m in margins]))
                if pairs
                else None,
                "paired_margin_mean": float(np.mean(margins)) if pairs else None,
                "paired_margin_median": float(np.median(margins)) if pairs else None,
            }
    development = summaries.get("development", {})
    passed = bool(development) and all(
        row["paired_states"] >= 6
        and row["paired_order_accuracy"] >= 0.75
        and row["paired_margin_mean"] > 0
        for row in development.values()
    )
    return {
        "low_gain": low_gain,
        "summaries": summaries,
        "development_gate": {
            "passed": passed,
            "minimum_pairs_per_horizon": 6,
            "minimum_order_accuracy": 0.75,
            "require_positive_mean_margin": True,
        },
    }


def replay_prefix(env, case: dict, row: dict, prefix: list[dict]) -> dict:
    """Restore the recorded intervention state and verify exact replay parity."""
    obs = env.reset(case)
    for _ in range(9):
        obs, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
    if physical_reset_hash(env) != row["reset_hash"]:
        raise RuntimeError("Gain-probe reset does not match collection")
    for previous in prefix:
        for command in previous["commands"][: previous["execution_ticks"]].numpy():
            obs, _, done = env.step(command)
            if done:
                raise RuntimeError("Replay terminated before the gain probe")
    np.testing.assert_allclose(obs["state"], row["prop"].numpy(), atol=1e-5, rtol=0)
    return obs


def run(args: argparse.Namespace) -> None:
    """Replay matched commands under nominal and reduced translation gain."""
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Completed gain probes are immutable")
    configure_gpu(args.gpu, args.output)
    reservation = torch.empty(1, device="cuda")
    configure_source(args.source)
    torch.set_num_threads(4)
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
    limits = {"train": args.train_cases, "development": args.development_cases}
    cases = [
        case
        for split, limit in limits.items()
        for case in [row for row in dataset["meta"]["cases"] if row["split"] == split][
            :limit
        ]
    ]
    env = LiberoProEnvironment()
    records, measurements = [], []
    try:
        for case in cases:
            episode = sorted(
                [row for row in dataset["rows"] if row["case_id"] == case["id"]],
                key=lambda row: row["start_tick"],
            )
            for target_tick in args.ticks:
                matches = [row for row in episode if row["start_tick"] == target_tick]
                if not matches:
                    continue
                row = matches[0]
                prefix = [
                    entry for entry in episode if entry["start_tick"] < target_tick
                ]
                if sum(entry["execution_ticks"] for entry in prefix) != target_tick:
                    raise ValueError("Recorded gain-probe prefix is not contiguous")
                branches, origin = {}, None
                issued = row["commands"].numpy().copy()
                for gain in (1.0, args.low_gain):
                    replay_prefix(env, case, row, prefix)
                    before = measured_snapshot(env, row["object_keys"])
                    if origin is not None:
                        for key in ("response", "geometry"):
                            torch.testing.assert_close(
                                before[key], origin[key], atol=1e-5, rtol=0
                            )
                    origin = before
                    executed = issued.copy()
                    executed[:, :3] = (executed[:, :3] * gain).clip(-1, 1)
                    following = {}
                    for tick, command in enumerate(executed, 1):
                        _, _, done = env.step(command)
                        if tick in HORIZONS:
                            following[tick] = measured_snapshot(env, row["object_keys"])
                        if done:
                            break
                    branch = transition_targets(before, following)
                    branches[gain] = branch
                    for index, horizon in enumerate(HORIZONS):
                        if not branch["horizon_mask"][index]:
                            continue
                        records.append(
                            {
                                "case_id": case["id"],
                                "split": case["split"],
                                "start_tick": target_tick,
                                "horizon": horizon,
                                "execution_gain": gain,
                                "translation_score": realized_translation_score(
                                    row["commands"][:horizon],
                                    branch["response"][index],
                                ),
                            }
                        )
                measurements.append(
                    {
                        "case": case,
                        "start_tick": target_tick,
                        "issued_commands": row["commands"],
                        "branches": branches,
                    }
                )
                temporary = args.output / "measurements.tmp"
                torch.save(measurements, temporary)
                temporary.replace(args.output / "measurements.pt")
                write_json(args.output / "records.json", records)
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "tick": target_tick,
                            "paired_states": len(measurements),
                        }
                    ),
                    flush=True,
                )
    finally:
        env.close()
        del reservation
    if not measurements:
        raise RuntimeError("No requested gain-probe state was available")
    result = summarize(records, args.low_gain) | {
        "state": "complete",
        "paired_states": len(measurements),
        "no_training": True,
        "policy_calls": 0,
        "command_contract": "same issued commands and simulator state; only executed xyz translation is scaled",
        "interpretation": "Causal local identifiability screen, not a control-success result",
    }
    write_json(args.output / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--low-gain", type=float, default=0.7)
    parser.add_argument("--train-cases", type=int, default=6)
    parser.add_argument("--development-cases", type=int, default=2)
    parser.add_argument(
        "--ticks", nargs="+", type=int, default=[40, 100, 160, 220, 280]
    )
    args = parser.parse_args()
    if not 0.2 <= args.low_gain < 1.0:
        parser.error("Reduced gain must be in [0.2, 1.0)")
    if not 0 <= args.train_cases <= 6 or not 1 <= args.development_cases <= 4:
        parser.error("Gain probe exceeds the declared small-screen layout budget")
    run(args)


if __name__ == "__main__":
    main()
