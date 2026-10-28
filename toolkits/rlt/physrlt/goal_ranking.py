# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Audit whether predicted physical effects rank goal-directed actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

from toolkits.rlt.physrlt.conditioning import load_sparse_motion_conditioner
from toolkits.rlt.physrlt.control import FrozenReadout, file_sha256
from toolkits.rlt.physrlt.effect_probe import paired_records
from toolkits.rlt.physrlt.sparse_motion import OBJECT_SCALE_M, _token
from toolkits.rlt.pi05_baseline import env_commands, write_json


def load_audited_readout(checkpoint: Path) -> FrozenReadout:
    """Load a readout while auditing its training data, not evaluation data."""
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    training_dataset = Path(checkpoint_payload["dataset"])
    return FrozenReadout(
        checkpoint,
        expected_dataset=training_dataset,
    )


def distance_progress(
    source: torch.Tensor,
    target: torch.Tensor,
    source_delta: torch.Tensor,
    target_delta: torch.Tensor,
) -> torch.Tensor:
    """Return the decrease in source-to-target distance after one transition."""
    before = torch.linalg.vector_norm(target - source)
    after = torch.linalg.vector_norm(target + target_delta - source - source_delta)
    return before - after


def pairwise_summary(rows: list[dict], tie_threshold_m: float) -> dict:
    """Summarize predeclared paired ranks without selecting a favorable tick."""
    useful = [
        row
        for row in rows
        if abs(row["actual_progress_difference_m"]) > tie_threshold_m
    ]
    if not useful:
        raise ValueError("No paired action has a resolvable goal-progress difference")
    correct = sum(row["predicted_sign"] == row["oracle_sign"] for row in useful)
    selected = [row["selected_actual_progress_m"] for row in useful]
    oracle = [row["oracle_actual_progress_m"] for row in useful]
    random_baseline = [row["random_actual_progress_m"] for row in useful]
    return {
        "pairs": len(rows),
        "resolvable_pairs": len(useful),
        "rank_accuracy_pct": 100 * correct / len(useful),
        "constant_sign_accuracy_pct": {
            str(sign): 100
            * sum(row["oracle_sign"] == sign for row in useful)
            / len(useful)
            for sign in (-1, 1)
        },
        "selected_actual_progress_mm": 1000 * statistics.mean(selected),
        "oracle_actual_progress_mm": 1000 * statistics.mean(oracle),
        "random_actual_progress_mm": 1000 * statistics.mean(random_baseline),
        "selected_advantage_vs_random_mm": 1000
        * statistics.mean(
            selected_value - random_value
            for selected_value, random_value in zip(selected, random_baseline)
        ),
        "oracle_regret_mm": 1000
        * statistics.mean(
            oracle_value - selected_value
            for oracle_value, selected_value in zip(oracle, selected)
        ),
    }


def candidate_summary(results: list[dict], tie_threshold_m: float) -> dict:
    """Compare action axes only at identical states and retain fixed controls.

    A fixed-candidate envelope is descriptive: it selects the best constant
    using these evaluation outcomes and must never become a deployed policy.
    """
    if len(results) < 2 or not math.isfinite(tie_threshold_m) or tie_threshold_m < 0:
        raise ValueError("Need multiple axes and a finite nonnegative threshold")
    axes = [result["axis"] for result in results]
    if len(set(axes)) != len(axes):
        raise ValueError("Duplicate candidate axis")
    for field in (
        "evaluation_dataset_sha256",
        "evaluation_split",
        "source_object_key",
        "target_object_key",
        "readout_sha256",
        "sparse_arm",
        "offset",
    ):
        if len({result[field] for result in results}) != 1:
            raise ValueError(f"Candidate runs disagree on {field}")
    if len({r.get("score_mode", "absolute") for r in results}) != 1:
        raise ValueError("Candidate runs disagree on score_mode")
    indices = []
    for result in results:
        index = {(row["case_id"], row["start_tick"]): row for row in result["rows"]}
        if len(index) != len(result["rows"]):
            raise ValueError("Duplicate candidate state")
        indices.append(index)
    if any(set(index) != set(indices[0]) for index in indices):
        raise ValueError("Candidate state sets differ; cannot silently drop states")
    if not indices[0]:
        raise ValueError("Candidate state set is empty")
    candidates = [f"{axis}:{sign}" for axis in axes for sign in (-1, 1)]
    rows = []
    for key in sorted(indices[0]):
        if len({index[key]["origin_sha256"] for index in indices}) != 1:
            raise ValueError("Candidate origins differ")
        actual, predicted = {}, {}
        for axis, index in zip(axes, indices, strict=True):
            for sign in (-1, 1):
                name = f"{axis}:{sign}"
                actual[name] = index[key]["actual_progress_m"][str(sign)]
                predicted[name] = index[key]["predicted_progress_m"][str(sign)]
        if not all(math.isfinite(v) for v in [*actual.values(), *predicted.values()]):
            raise ValueError("Nonfinite candidate score")
        ordered = sorted(candidates, key=actual.get, reverse=True)
        selected = max(candidates, key=predicted.get)
        rows.append(
            {
                "case_id": key[0],
                "start_tick": key[1],
                "selected": selected,
                "oracle": ordered[0],
                "resolvable": actual[ordered[0]] - actual[ordered[1]] > tie_threshold_m,
                "actual_progress_m": actual,
                "selected_progress_m": actual[selected],
                "oracle_progress_m": actual[ordered[0]],
            }
        )
    useful = [row for row in rows if row["resolvable"]]
    fixed = {
        name: {
            "mean_progress_mm": 1000
            * statistics.mean(row["actual_progress_m"][name] for row in rows),
            "resolvable_accuracy_pct": (
                100 * sum(row["oracle"] == name for row in useful) / len(useful)
                if useful
                else None
            ),
        }
        for name in candidates
    }
    mean_selected = 1000 * statistics.mean(row["selected_progress_m"] for row in rows)
    return {
        "states": len(rows),
        "layouts": len({row["case_id"] for row in rows}),
        "resolvable_states": len(useful),
        "resolvable_accuracy_pct": (
            100 * sum(row["oracle"] == row["selected"] for row in useful) / len(useful)
            if useful
            else None
        ),
        "mean_progress_mm_all_states": mean_selected,
        "oracle_regret_mm_all_states": 1000
        * statistics.mean(
            row["oracle_progress_m"] - row["selected_progress_m"] for row in rows
        ),
        "fixed_candidates": fixed,
        "advantage_over_best_fixed_envelope_mm": mean_selected
        - max(value["mean_progress_mm"] for value in fixed.values()),
        "interpretation": "diagnostic with simulator goal positions; fixed envelope uses evaluation outcomes",
        "rows": rows,
    }


def sum_object_progress(results: list[dict]) -> dict:
    """Sum distance progress for declared objects at exactly matching origins.

    This diagnostic objective gives every object equal weight. It does not
    encode task success predicates, grasp feasibility or stage dependencies.
    """
    if not results or len({r["source_object_key"] for r in results}) != len(results):
        raise ValueError("Expected distinct source objects")
    for field in (
        "axis",
        "offset",
        "evaluation_dataset_sha256",
        "evaluation_split",
        "target_object_key",
        "readout_sha256",
        "sparse_arm",
    ):
        if len({r[field] for r in results}) != 1:
            raise ValueError(f"Object goal reports disagree on {field}")
    indices = [
        {(r["case_id"], r["start_tick"]): r for r in result["rows"]}
        for result in results
    ]
    if any(
        len(index) != len(result["rows"])
        for index, result in zip(indices, results, strict=True)
    ):
        raise ValueError("Duplicate object goal state")
    if any(set(index) != set(indices[0]) for index in indices):
        raise ValueError("Object goal state sets differ")
    rows = []
    for key in sorted(indices[0]):
        if len({index[key]["origin_sha256"] for index in indices}) != 1:
            raise ValueError("Object goal origins differ")
        row = {
            field: indices[0][key][field]
            for field in ("case_id", "start_tick", "origin_sha256")
        }
        for field in ("actual_progress_m", "predicted_progress_m"):
            row[field] = {
                str(sign): sum(index[key][field][str(sign)] for index in indices)
                for sign in (-1, 1)
            }
        rows.append(row)
    combined = {
        key: results[0][key]
        for key in (
            "axis",
            "offset",
            "evaluation_dataset_sha256",
            "evaluation_split",
            "target_object_key",
            "readout_sha256",
            "sparse_arm",
        )
    }
    combined["source_object_key"] = "+".join(r["source_object_key"] for r in results)
    if len({r.get("score_mode", "absolute") for r in results}) != 1:
        raise ValueError("Object goal reports disagree on score_mode")
    combined["score_mode"] = results[0].get("score_mode", "absolute")
    combined["source_heads_sha256"] = [r["sparse_head_sha256"] for r in results]
    combined["rows"] = rows
    return combined


@torch.no_grad()
def audit(args: argparse.Namespace) -> dict:
    """Rank paired interventions with a frozen sparse physical predictor."""
    if args.output.exists():
        raise FileExistsError("Goal-ranking output must be a new directory")
    if not 0 <= args.tie_threshold_mm <= 2:
        raise ValueError("Tie threshold must stay within 0--2 mm")
    score_mode = getattr(args, "score_mode", "absolute")
    if score_mode not in {"absolute", "reference_centered"}:
        raise ValueError("Unknown goal-ranking score mode")
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
    measurements = torch.load(args.measurements, map_location="cpu", weights_only=True)
    measurement_result = json.loads(args.measurement_result.read_text())
    if measurement_result.get("state") != "complete" or not measurement_result.get(
        "includes_absolute_origin_geometry"
    ):
        raise ValueError("Goal ranking requires completed absolute-origin pairs")
    split = measurement_result["split"]
    records = paired_records(
        dataset,
        measurements,
        measurement_result["axis"],
        measurement_result["offset"],
        expected_split=split,
    )
    readout = load_audited_readout(args.checkpoint)
    conditioner, payload = load_sparse_motion_conditioner(
        args.sparse_head,
        expected_readout_sha256=file_sha256(args.checkpoint),
    )
    for field in ("axis", "offset"):
        if field in payload and payload[field] != measurement_result[field]:
            raise ValueError(f"Sparse head and measurement {field} differ")
    if (
        "training_axes" in payload
        and measurement_result["axis"] not in payload["training_axes"]
    ):
        raise ValueError("Measurement axis was not in sparse-head training")
    object_index = payload.get("object_index_selected_on_train_only")
    if object_index is None:
        raise ValueError("Sparse head lacks its train-selected object identity")
    rows = []
    excluded_horizons = []
    for record, measurement in zip(records, measurements, strict=True):
        if not all(
            bool(branch["horizon_mask"][-1]) for branch in record["branches"].values()
        ):
            excluded_horizons.append([record["case_id"], record["start_tick"]])
            continue
        object_keys = measurement["origin"]["object_keys"]
        if object_index >= len(object_keys):
            raise ValueError("Sparse head object index is outside the current scene")
        geometry = measurement["origin"]["geometry"]
        expected_width = 64 + 3 * len(object_keys)
        if geometry.shape != (expected_width,) or not torch.isfinite(geometry).all():
            raise ValueError("Invalid absolute origin geometry")
        source_key = object_keys[object_index]
        if source_key != record["row"]["object_keys"][object_index]:
            raise ValueError("Sparse head object identity changed across artifacts")
        if args.target_object_key not in object_keys:
            raise ValueError("Goal target object is absent from the current scene")
        target_index = object_keys.index(args.target_object_key)
        positions = geometry[64:].reshape(-1, 3)
        source, target = positions[object_index], positions[target_index]
        token = _token(readout, record)
        reference_progress = 0.0
        if score_mode == "reference_centered":
            commands = torch.from_numpy(
                env_commands(record["row"]["ref"].numpy())
            ).float()
            _, _, reference_effect = conditioner.head(token, commands[None], 10)
            reference_progress = float(
                distance_progress(
                    source, target, reference_effect[0] * OBJECT_SCALE_M, torch.zeros(3)
                )
            )
        actual_progress, predicted_progress = {}, {}
        for sign in (-1, 1):
            branch = record["branches"][sign]
            source_start = 64 + 3 * object_index
            target_start = 64 + 3 * target_index
            source_delta = branch["geometry"][-1, source_start : source_start + 3]
            target_delta = branch["geometry"][-1, target_start : target_start + 3]
            actual_progress[sign] = distance_progress(
                source, target, source_delta, target_delta
            )
            _, _, expected = conditioner.head(token, branch["commands"][None], 10)
            predicted_progress[sign] = (
                distance_progress(
                    source,
                    target,
                    expected[0] * OBJECT_SCALE_M,
                    torch.zeros(3),
                )
                - reference_progress
            )
        predicted_sign = max((-1, 1), key=lambda sign: predicted_progress[sign])
        oracle_sign = max((-1, 1), key=lambda sign: actual_progress[sign])
        rows.append(
            {
                "case_id": record["case_id"],
                "start_tick": record["start_tick"],
                "origin_sha256": hashlib.sha256(geometry.numpy().tobytes()).hexdigest(),
                "actual_progress_m": {
                    str(sign): float(value) for sign, value in actual_progress.items()
                },
                "predicted_progress_m": {
                    str(sign): float(value)
                    for sign, value in predicted_progress.items()
                },
                "predicted_sign": predicted_sign,
                "predicted_reference_progress_m": reference_progress,
                "oracle_sign": oracle_sign,
                "predicted_progress_difference_m": float(
                    predicted_progress[1] - predicted_progress[-1]
                ),
                "actual_progress_difference_m": float(
                    actual_progress[1] - actual_progress[-1]
                ),
                "selected_actual_progress_m": float(actual_progress[predicted_sign]),
                "oracle_actual_progress_m": float(actual_progress[oracle_sign]),
                "random_actual_progress_m": float(
                    (actual_progress[-1] + actual_progress[1]) / 2
                ),
            }
        )
    summary = pairwise_summary(rows, args.tie_threshold_mm / 1000) | {
        "state": "complete",
        "axis": measurement_result["axis"],
        "offset": measurement_result["offset"],
        "evaluation_dataset_sha256": file_sha256(args.dataset),
        "readout_sha256": file_sha256(args.checkpoint),
        "sparse_head_sha256": file_sha256(args.sparse_head),
        "measurements_sha256": file_sha256(args.measurements),
        "excluded_incomplete_horizons": excluded_horizons,
        "evaluation_split": split,
        "readout_method": readout.payload["method"],
        "sparse_arm": payload["arm"],
        "score_mode": score_mode,
        "source_object_key": records[0]["row"]["object_keys"][object_index],
        "target_object_key": args.target_object_key,
        "tie_threshold_mm": args.tie_threshold_mm,
        "goal_source": "privileged simulator object positions, diagnostic only",
        "interpretation": (
            "offline causal action-ranking diagnostic; not a deployable policy "
            "or an online-RL success result"
        ),
        "rows": rows,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset",
        "measurements",
        "measurement-result",
        "checkpoint",
        "sparse-head",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target-object-key", required=True)
    parser.add_argument("--tie-threshold-mm", type=float, default=0.1)
    parser.add_argument(
        "--score-mode", choices=("absolute", "reference_centered"), default="absolute"
    )
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
