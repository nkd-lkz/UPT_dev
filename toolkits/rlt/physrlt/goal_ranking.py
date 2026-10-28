# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Audit whether predicted physical effects rank goal-directed actions."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from toolkits.rlt.physrlt.conditioning import load_sparse_motion_conditioner
from toolkits.rlt.physrlt.control import FrozenReadout, file_sha256
from toolkits.rlt.physrlt.effect_probe import paired_records
from toolkits.rlt.physrlt.sparse_motion import OBJECT_SCALE_M, _token
from toolkits.rlt.pi05_baseline import write_json


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


@torch.no_grad()
def audit(args: argparse.Namespace) -> dict:
    """Rank paired interventions with a frozen sparse physical predictor."""
    if args.output.exists():
        raise FileExistsError("Goal-ranking output must be a new directory")
    if not 0 <= args.tie_threshold_mm <= 2:
        raise ValueError("Tie threshold must stay within 0--2 mm")
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
    object_index = payload.get("object_index_selected_on_train_only")
    if object_index is None:
        raise ValueError("Sparse head lacks its train-selected object identity")
    rows = []
    for record, measurement in zip(records, measurements, strict=True):
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
            predicted_progress[sign] = distance_progress(
                source,
                target,
                expected[0] * OBJECT_SCALE_M,
                torch.zeros(3),
            )
        predicted_sign = max((-1, 1), key=lambda sign: predicted_progress[sign])
        oracle_sign = max((-1, 1), key=lambda sign: actual_progress[sign])
        rows.append(
            {
                "case_id": record["case_id"],
                "start_tick": record["start_tick"],
                "predicted_sign": predicted_sign,
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
        "evaluation_split": split,
        "readout_method": readout.payload["method"],
        "sparse_arm": payload["arm"],
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
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
