# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Test candidate-action effects by replaying the same simulator prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from toolkits.rlt.libero_pro_environment import LiberoProEnvironment
from toolkits.rlt.physrlt.control import FrozenReadout
from toolkits.rlt.physrlt.data import (
    HORIZONS,
    measured_snapshot,
    physical_reset_hash,
    transition_targets,
)
from toolkits.rlt.pi05_baseline import (
    configure_gpu,
    configure_source,
    env_commands,
    write_json,
)


def effect_errors(
    plus: dict, minus: dict, predicted_plus: dict, predicted_minus: dict, stats: dict
) -> list[dict]:
    """Compare paired execution differences, including the zero-effect control."""
    rows = []
    for index, horizon in enumerate(HORIZONS):
        if not (plus["horizon_mask"][index] and minus["horizon_mask"][index]):
            continue
        response = plus["response"][index] - minus["response"][index]
        response_pred = (
            predicted_plus["response"][0, index] - predicted_minus["response"][0, index]
        )
        geometry = plus["geometry"][index, 64:] - minus["geometry"][index, 64:]
        geometry_pred = (
            predicted_plus["geometry"][0, index, 64:]
            - predicted_minus["geometry"][0, index, 64:]
        )
        rows.append(
            {
                "horizon": horizon,
                "response_effect_scaled_mse": float(
                    ((response_pred - response) / stats["response_scale"])
                    .square()
                    .mean()
                ),
                "zero_effect_response_scaled_mse": float(
                    (response / stats["response_scale"]).square().mean()
                ),
                "object_effect_rmse_m": float(
                    (geometry_pred - geometry).square().mean().sqrt()
                ),
                "zero_effect_object_rmse_m": float(geometry.square().mean().sqrt()),
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    """Replay recorded prefixes before two bounded translation interventions."""
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Completed paired-action diagnostics are immutable")
    configure_gpu(args.gpu, args.output)
    reservation = torch.empty(1, device="cuda")
    configure_source(args.source)
    torch.set_num_threads(4)
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
    cases = [case for case in dataset["meta"]["cases"] if case["split"] == args.split][
        : args.cases
    ]
    readouts = {
        name: FrozenReadout(args.probes / f"{name}-seed31/readout.pt")
        for name in ("joint", "shuffled_joint", "state_action_joint")
    }
    env = LiberoProEnvironment()
    records, measurements = [], []
    completed_states = 0
    try:
        for case in cases:
            episode = sorted(
                [r for r in dataset["rows"] if r["case_id"] == case["id"]],
                key=lambda r: r["start_tick"],
            )
            for target_tick in args.ticks:
                matches = [r for r in episode if r["start_tick"] == target_tick]
                if not matches:
                    continue
                row = matches[0]
                prefix = [r for r in episode if r["start_tick"] < target_tick]
                if sum(r["execution_ticks"] for r in prefix) != target_tick:
                    raise ValueError("Recorded prefix is not contiguous")
                branches, predictions = {}, {}
                origin = None
                for sign in (-1, 1):
                    obs = env.reset(case)
                    for _ in range(9):
                        obs, _, _ = env.step(
                            np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
                        )
                    if physical_reset_hash(env) != row["reset_hash"]:
                        raise RuntimeError(
                            "Counterfactual reset does not match collection"
                        )
                    for previous in prefix:
                        for command in previous["commands"][
                            : previous["execution_ticks"]
                        ].numpy():
                            obs, _, done = env.step(command)
                            if done:
                                raise RuntimeError(
                                    "Replay terminated before the intervention state"
                                )
                    np.testing.assert_allclose(
                        obs["state"], row["prop"].numpy(), atol=1e-5, rtol=0
                    )
                    before = measured_snapshot(env, row["object_keys"])
                    if origin is not None:
                        for key in ("response", "geometry"):
                            torch.testing.assert_close(
                                before[key], origin[key], atol=1e-5, rtol=0
                            )
                    origin = before
                    commands = env_commands(row["ref"].numpy())
                    commands[:, args.axis] = (
                        commands[:, args.axis] + sign * args.offset
                    ).clip(-1, 1)
                    predictions[sign] = {}
                    with torch.no_grad():
                        for name, readout in readouts.items():
                            z = readout.encode(row, row["prop"])["z"][None]
                            prediction = readout.model.consequences(
                                z, torch.from_numpy(commands)[None]
                            )
                            predictions[sign][name] = {
                                key: prediction[key]
                                * readout.payload["stats"][key + "_scale"]
                                for key in ("response", "geometry")
                            }
                    following = {}
                    for tick, command in enumerate(commands, 1):
                        _, _, done = env.step(command)
                        if tick in HORIZONS:
                            following[tick] = measured_snapshot(env, row["object_keys"])
                        if done:
                            break
                    branches[sign] = transition_targets(before, following)
                for name, readout in readouts.items():
                    for error in effect_errors(
                        branches[1],
                        branches[-1],
                        predictions[1][name],
                        predictions[-1][name],
                        readout.payload["stats"],
                    ):
                        records.append(
                            {
                                "case_id": case["id"],
                                "start_tick": target_tick,
                                "method": name,
                                **error,
                            }
                        )
                completed_states += 1
                measurements.append(
                    {
                        "case": case,
                        "start_tick": target_tick,
                        "branches": branches,
                        "predictions": predictions,
                    }
                )
                temporary = args.output / "measurements.tmp"
                torch.save(measurements, temporary)
                temporary.replace(args.output / "measurements.pt")
                write_json(args.output / "effects.json", records)
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "tick": target_tick,
                            "paired_states": completed_states,
                        }
                    ),
                    flush=True,
                )
    finally:
        env.close()
        del reservation
    if not completed_states:
        raise RuntimeError("No requested intervention state was available")
    write_json(
        args.output / "result.json",
        {
            "state": "complete",
            "paired_states": completed_states,
            "axis": args.axis,
            "offset": args.offset,
            "states_from": f"{args.split}-layout replay with new paired actions",
            "split": args.split,
            "no_training": True,
            "policy_calls": 0,
            "interpretation": "Direct consequence-decoder fidelity under paired action changes; not a representation-vs-reconstruction or control-success comparison",
            "records": records,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset", "probes", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--ticks", nargs="+", type=int, default=[40, 100, 160])
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--offset", type=float, default=0.1)
    parser.add_argument(
        "--split",
        choices=("train", "development", "confirmation"),
        default="train",
    )
    args = parser.parse_args()
    if not 0 < args.offset <= 0.1 or not 1 <= args.cases <= 6:
        parser.error(
            "Paired action screen is bounded to six cases and 0.1 native translation offset"
        )
    run(args)


if __name__ == "__main__":
    main()
