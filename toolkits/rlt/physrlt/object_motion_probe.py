# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Test whether action-side features identify sparse object-motion events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toolkits.rlt.physrlt.critic_audit import binary_auc
from toolkits.rlt.physrlt.effect_probe import paired_records
from toolkits.rlt.pi05_baseline import write_json

FEATURES = ("state_action", "final_action", "flow_path", "prefix")
RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0)


def dominant_object(records: list[dict], rank: int = 0) -> int:
    """Select an object by descending paired-effect energy on training layouts."""
    effects = []
    for record in records:
        geometry = (
            record["branches"][1]["geometry"][1, 64:]
            - record["branches"][-1]["geometry"][1, 64:]
        )
        if len(geometry) % 3:
            raise ValueError("Object geometry must contain 3-D world positions")
        effects.append(geometry.reshape(-1, 3).square().sum(-1))
    energy = torch.stack(effects).mean(0)
    if not torch.isfinite(energy).all() or float(energy.max()) <= 0:
        raise ValueError("Training pairs contain no measurable object effect")
    if not 0 <= rank < len(energy):
        raise ValueError("Object motion rank is outside the scene")
    index = int(torch.argsort(energy, descending=True, stable=True)[rank])
    if float(energy[index]) <= 0:
        raise ValueError("Selected training object has no measurable effect")
    return index


def object_effect(record: dict, object_index: int) -> torch.Tensor:
    """Return the 10-tick paired displacement effect for one object."""
    start = 64 + 3 * object_index
    stop = start + 3
    return (
        record["branches"][1]["geometry"][1, start:stop]
        - record["branches"][-1]["geometry"][1, start:stop]
    ).double()


def feature(record: dict, name: str) -> torch.Tensor:
    """Build a pre-execution feature without reading paired outcomes."""
    row = record["row"]
    state_action = torch.cat((row["prop"].flatten(), row["ref"].flatten())).double()
    if name == "state_action":
        return state_action
    key = {
        "final_action": "action_feature",
        "flow_path": "flow_feature",
        "prefix": "prefix_feature",
    }.get(name)
    if key is None or key not in row:
        raise ValueError(f"Unknown or unavailable object-motion feature: {name}")
    return torch.cat((row[key].flatten(), state_action)).double()


def _ridge_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Fit a centred dual ridge model, efficient when features exceed samples."""
    mean = train_x.mean(0)
    scale = train_x.std(0, correction=0).clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    target_mean = train_y.mean(0, keepdim=True)
    centred = train_y - target_mean
    kernel = train_x @ train_x.T
    dual = torch.linalg.solve(
        kernel + ridge * torch.eye(len(train_x), dtype=train_x.dtype), centred
    )
    return test_x @ train_x.T @ dual + target_mean


def _predict(
    train: list[dict],
    test: list[dict],
    name: str,
    object_index: int,
    direct_ridge: float,
    event_ridge: float,
    motion_ridge: float,
    motion_threshold_m: float,
) -> dict[str, torch.Tensor]:
    train_x = torch.stack([feature(row, name) for row in train])
    test_x = torch.stack([feature(row, name) for row in test])
    train_y = torch.stack([object_effect(row, object_index) for row in train])
    test_y = torch.stack([object_effect(row, object_index) for row in test])
    moving = train_y.norm(dim=1) > motion_threshold_m
    if int(moving.sum()) < 2 or int((~moving).sum()) < 2:
        raise ValueError("Motion gating requires moving and stationary training pairs")
    event_target = moving.double()[:, None]
    event_score = _ridge_predict(train_x, event_target, test_x, event_ridge).clamp(0, 1)
    conditional = _ridge_predict(train_x[moving], train_y[moving], test_x, motion_ridge)
    return {
        "target": test_y,
        "direct": _ridge_predict(train_x, train_y, test_x, direct_ridge),
        "event_score": event_score[:, 0],
        "gated": event_score * conditional,
    }


def _rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction - target).square().mean().sqrt())


def fit_feature(
    train: list[dict],
    evaluation: list[dict],
    name: str,
    object_index: int,
    motion_threshold_m: float = 0.002,
) -> dict:
    """Select all ridge strengths by train-layout CV, then evaluate once."""
    cases = sorted({record["case_id"] for record in train})
    if len(cases) < 3:
        raise ValueError("Object-motion probing requires at least three train layouts")

    def cv_error(direct: float, event: float, motion: float, key: str) -> float:
        squared, count = 0.0, 0
        for heldout in cases:
            fit = [record for record in train if record["case_id"] != heldout]
            test = [record for record in train if record["case_id"] == heldout]
            output = _predict(
                fit,
                test,
                name,
                object_index,
                direct,
                event,
                motion,
                motion_threshold_m,
            )
            squared += float((output[key] - output["target"]).square().sum())
            count += output["target"].numel()
        return (squared / count) ** 0.5

    direct_ridge = min(RIDGES, key=lambda value: cv_error(value, 1, 1, "direct"))
    gated = min(
        (
            (cv_error(direct_ridge, event, motion, "gated"), event, motion)
            for event in RIDGES
            for motion in RIDGES
        ),
        key=lambda value: value[0],
    )
    output = _predict(
        train,
        evaluation,
        name,
        object_index,
        direct_ridge,
        gated[1],
        gated[2],
        motion_threshold_m,
    )
    target = output["target"]
    labels = target.norm(dim=1) > motion_threshold_m
    return {
        "feature": name,
        "direct_ridge": direct_ridge,
        "event_ridge": gated[1],
        "motion_ridge": gated[2],
        "train_cv_gated_rmse_m": gated[0],
        "evaluation_pairs": len(evaluation),
        "evaluation_moving_pairs": int(labels.sum()),
        "zero_motion_rmse_m": _rmse(torch.zeros_like(target), target),
        "direct_rmse_m": _rmse(output["direct"], target),
        "gated_rmse_m": _rmse(output["gated"], target),
        "reversed_gated_rmse_m": _rmse(-output["gated"], target),
        "event_auc": binary_auc(output["event_score"].tolist(), labels.tolist()),
        "event_accuracy": float(
            ((output["event_score"] >= 0.5) == labels).float().mean()
        ),
    }


def _load_records(
    dataset: dict,
    measurements_path: Path,
    result_path: Path,
    split: str,
) -> tuple[list[dict], dict]:
    measurements = torch.load(measurements_path, map_location="cpu", weights_only=True)
    metadata = json.loads(result_path.read_text())
    if (
        metadata.get("state") != "complete"
        or not metadata.get("no_training")
        or metadata.get("paired_states") != len(measurements)
    ):
        raise ValueError("Object-motion probe requires complete measurement-only pairs")
    records = paired_records(
        dataset,
        measurements,
        metadata["axis"],
        metadata["offset"],
        expected_split=split,
    )
    return records, metadata


def run(args: argparse.Namespace) -> dict:
    """Fit train-only motion probes and write one immutable result."""
    if args.output.exists():
        raise FileExistsError("Object-motion probe output must be a new file")
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
    train, train_meta = _load_records(
        dataset, args.train_measurements, args.train_result, "train"
    )
    evaluation, evaluation_meta = _load_records(
        dataset,
        args.evaluation_measurements,
        args.evaluation_result,
        "development",
    )
    if {row["case_id"] for row in train} & {row["case_id"] for row in evaluation}:
        raise ValueError("Object-motion train and evaluation layouts overlap")
    object_index = dominant_object(train)
    object_keys = train[0]["row"]["object_keys"]
    result = {
        "state": "complete",
        "object_index_selected_on_train_only": object_index,
        "object_key": object_keys[object_index],
        "motion_threshold_m": args.motion_threshold_m,
        "train_pairs": len(train),
        "evaluation_pairs": len(evaluation),
        "train_axis": train_meta["axis"],
        "evaluation_axis": evaluation_meta["axis"],
        "results": [
            fit_feature(
                train,
                evaluation,
                name,
                object_index,
                args.motion_threshold_m,
            )
            for name in FEATURES
        ],
        "interpretation": (
            "offline sparse object-motion identifiability; not task success or "
            "online-RL evidence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset",
        "train-measurements",
        "train-result",
        "evaluation-measurements",
        "evaluation-result",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--motion-threshold-m", type=float, default=0.002)
    args = parser.parse_args()
    if not 0.0005 <= args.motion_threshold_m <= 0.01:
        parser.error("Motion threshold must stay within 0.5--10 mm")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
