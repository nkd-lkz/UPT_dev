# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fit a sparse object-motion head on frozen action-side Physical Tokens."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from torch import nn

from toolkits.rlt.physrlt.control import FrozenReadout
from toolkits.rlt.physrlt.critic_audit import binary_auc
from toolkits.rlt.physrlt.effect_probe import file_sha256, paired_records
from toolkits.rlt.physrlt.model import SparseObjectMotionHead
from toolkits.rlt.physrlt.object_motion_probe import dominant_object, object_effect
from toolkits.rlt.physrlt.probe import dataset_sha256, inputs
from toolkits.rlt.pi05_baseline import write_json

ARMS = (
    "direct_pointwise",
    "direct_pair",
    "gated_pointwise",
    "gated_pair",
    "reversed_gated_pair",
)
OBJECT_SCALE_M = 0.01


def _token(readout: FrozenReadout, record: dict) -> torch.Tensor:
    row = record["row"]
    data = {
        key: row[key][None]
        for key in (
            "action_feature",
            "prefix_feature",
            "ref",
            "prop",
        )
    }
    if "flow_feature" in row:
        data["flow_feature"] = row["flow_feature"][None]
    return readout.model.encode(
        *inputs(data, readout.payload["stats"], readout.payload["method"])
    )


def _branch_target(
    record: dict, sign: int, horizon_index: int, object_index: int
) -> torch.Tensor:
    start = 64 + 3 * object_index
    return (
        record["branches"][sign]["geometry"][horizon_index, start : start + 3].float()
        / OBJECT_SCALE_M
    )


def fit_head(
    readout: FrozenReadout,
    records: list[dict],
    object_index: int,
    arm: str,
    seed: int,
    steps: int,
    motion_threshold_m: float,
) -> tuple[SparseObjectMotionHead, list[dict]]:
    """Fit one matched sparse-motion arm from immutable training pairs."""
    if arm not in ARMS:
        raise ValueError("Unknown sparse-motion arm")
    gated = "gated" in arm
    paired = "pair" in arm
    reversed_pair = arm == "reversed_gated_pair"
    torch.manual_seed(seed)
    head = SparseObjectMotionHead(gated).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    trace = []
    for step in range(steps):
        record = records[int(torch.randint(len(records), (), generator=generator))]
        with torch.no_grad():
            token = _token(readout, record)
        losses = []
        for horizon_index, horizon in enumerate((3, 10)):
            if not all(
                bool(branch["horizon_mask"][horizon_index])
                for branch in record["branches"].values()
            ):
                continue
            expected = {}
            targets = {}
            for sign in (-1, 1):
                target = _branch_target(record, sign, horizon_index, object_index)
                event_logit, conditional, prediction = head(
                    token, record["branches"][sign]["commands"][None], horizon
                )
                targets[sign] = target
                expected[sign] = prediction[0]
                if gated:
                    moving = (
                        target.norm() * OBJECT_SCALE_M > motion_threshold_m
                    ).float()
                    event_loss = nn.functional.binary_cross_entropy_with_logits(
                        event_logit, moving[None]
                    )
                    losses.append(event_loss)
                    if bool(moving):
                        losses.append((conditional[0] - target).square().mean())
                else:
                    losses.append((prediction[0] - target).square().mean())
            if paired:
                target_effect = targets[1] - targets[-1]
                if reversed_pair:
                    target_effect = -target_effect
                losses.append(
                    ((expected[1] - expected[-1]) - target_effect).square().mean()
                )
        if not losses:
            raise ValueError("Sparse fitting requires a completed paired horizon")
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite sparse-motion objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError("Nonfinite sparse-motion gradient")
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            trace.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "gradient": float(gradient),
                }
            )
    return head.eval().requires_grad_(False), trace


@torch.no_grad()
def evaluate_head(
    readout: FrozenReadout,
    head: SparseObjectMotionHead,
    records: list[dict],
    object_index: int,
    motion_threshold_m: float,
) -> dict:
    """Evaluate paired 10-tick effects on disjoint layouts."""
    predictions, targets, event_scores = [], [], []
    for record in records:
        token = _token(readout, record)
        branch = {}
        probabilities = []
        for sign in (-1, 1):
            event_logit, _, prediction = head(
                token, record["branches"][sign]["commands"][None], 10
            )
            branch[sign] = prediction[0] * OBJECT_SCALE_M
            probability = (
                event_logit.sigmoid()[0]
                if event_logit is not None
                else token.new_tensor(1.0)
            )
            probabilities.append(probability)
        predictions.append(branch[1] - branch[-1])
        targets.append(object_effect(record, object_index))
        event_scores.append(torch.stack(probabilities).max())
    prediction = torch.stack(predictions)
    target = torch.stack(targets)
    event_score = torch.stack(event_scores)
    moved = target.norm(dim=1) > motion_threshold_m
    zero_rmse = float(target.square().mean().sqrt())
    rmse = float((prediction - target).square().mean().sqrt())
    reversed_rmse = float((-prediction - target).square().mean().sqrt())
    cosine_mask = (prediction.norm(dim=1) > 1e-6) & (target.norm(dim=1) > 1e-6)
    cosine = (
        float(
            nn.functional.cosine_similarity(
                prediction[cosine_mask], target[cosine_mask]
            ).mean()
        )
        if cosine_mask.any()
        else None
    )
    return {
        "pairs": len(records),
        "moving_pairs": int(moved.sum()),
        "zero_motion_rmse_m": zero_rmse,
        "rmse_m": rmse,
        "improvement_vs_zero_pct": (
            100 * (zero_rmse - rmse) / zero_rmse if zero_rmse > 0 else None
        ),
        "sign_reversed_prediction_rmse_m": reversed_rmse,
        "effect_cosine": cosine,
        "event_auc": binary_auc(event_score.tolist(), moved.tolist()),
        "event_accuracy": float(((event_score >= 0.5) == moved).float().mean()),
    }


@torch.no_grad()
def evaluate_object_bank(
    readout: FrozenReadout,
    heads: list[SparseObjectMotionHead],
    object_indices: list[int],
    records: list[dict],
    motion_threshold_m: float,
) -> dict:
    """Score all predicted object effects and the largest affected object.

    Labels come from completed paired interventions. The prediction uses only
    pre-execution tokens and candidate commands. Object order must be selected
    on training layouts; index zero is the fixed-dominant-object control.
    """
    if len(heads) != len(object_indices) or len(heads) < 2:
        raise ValueError("An object bank requires at least two matched heads")
    if len(set(object_indices)) != len(object_indices):
        raise ValueError("Object bank contains duplicate objects")
    if not 0 < motion_threshold_m:
        raise ValueError("Motion threshold must be positive")
    predictions, targets, identities = [], [], []
    for record in records:
        if not all(bool(b["horizon_mask"][-1]) for b in record["branches"].values()):
            continue
        token = _token(readout, record)
        predicted_objects, actual_objects = [], []
        for head, object_index in zip(heads, object_indices, strict=True):
            branch_predictions = {}
            for sign in (-1, 1):
                _, _, prediction = head(
                    token, record["branches"][sign]["commands"][None], 10
                )
                branch_predictions[sign] = prediction[0] * OBJECT_SCALE_M
            predicted_objects.append(branch_predictions[1] - branch_predictions[-1])
            actual_objects.append(object_effect(record, object_index).float())
        predictions.append(torch.stack(predicted_objects))
        targets.append(torch.stack(actual_objects))
        identities.append([record["case_id"], record["start_tick"]])
    if not predictions:
        raise ValueError("No complete pairs for the object bank")
    prediction, target = torch.stack(predictions), torch.stack(targets)
    predicted_norms, actual_norms = prediction.norm(dim=-1), target.norm(dim=-1)
    predicted_label, actual_label = predicted_norms.argmax(-1), actual_norms.argmax(-1)
    predicted_label[predicted_norms.max(-1).values <= motion_threshold_m] = -1
    actual_label[actual_norms.max(-1).values <= motion_threshold_m] = -1
    per_class = {}
    for label in (-1, *range(len(heads))):
        mask = actual_label == label
        per_class[str(label)] = {
            "pairs": int(mask.sum()),
            "recall_pct": float((predicted_label[mask] == label).float().mean() * 100)
            if mask.any()
            else None,
        }
    recalls = [
        row["recall_pct"] for row in per_class.values() if row["recall_pct"] is not None
    ]
    return {
        "pairs": len(predictions),
        "object_indices_train_order": object_indices,
        "accuracy_pct": float((predicted_label == actual_label).float().mean() * 100),
        "macro_recall_pct": sum(recalls) / len(recalls),
        "fixed_dominant_accuracy_pct": float((actual_label == 0).float().mean() * 100),
        "fixed_no_motion_accuracy_pct": float(
            (actual_label == -1).float().mean() * 100
        ),
        "per_class": per_class,
        "effect_rmse_mm": float((prediction - target).square().mean().sqrt() * 1000),
        "zero_effect_rmse_mm": float(target.square().mean().sqrt() * 1000),
        "rows": [
            {
                "case_id": key[0],
                "start_tick": key[1],
                "predicted_label": int(pred),
                "actual_label": int(actual),
            }
            for key, pred, actual in zip(
                identities, predicted_label, actual_label, strict=True
            )
        ],
        "interpretation": "offline physical-effect identity prediction, not task success or goal selection",
    }


def _load_records(
    dataset: dict, measurements: Path, result_path: Path, split: str
) -> tuple[list[dict], dict]:
    values = torch.load(measurements, map_location="cpu", weights_only=True)
    metadata = json.loads(result_path.read_text())
    if metadata.get("state") != "complete" or not metadata.get("no_training"):
        raise ValueError("Sparse-motion input must be a complete measurement run")
    records = paired_records(
        dataset,
        values,
        metadata["axis"],
        metadata["offset"],
        expected_split=split,
    )
    valid = [
        record
        for record in records
        if all(
            bool(branch["horizon_mask"][-1]) for branch in record["branches"].values()
        )
    ]
    if not valid:
        raise ValueError("No completed 10-tick paired horizon")
    metadata = metadata | {"excluded_incomplete_horizons": len(records) - len(valid)}
    return valid, metadata


def run(args: argparse.Namespace) -> dict:
    """Train matched heads and retain checkpoints plus disjoint evaluation."""
    torch.set_num_threads(args.cpu_threads)
    if args.output.exists():
        raise FileExistsError("Sparse-motion output must be a new directory")
    if not 1 <= args.steps <= 2000:
        raise ValueError("Sparse-motion pilot is bounded to 1--2000 updates")
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
    evaluation_dataset_path = args.evaluation_dataset or args.dataset
    evaluation_dataset = torch.load(
        evaluation_dataset_path, map_location="cpu", weights_only=True
    )
    train, train_meta = _load_records(
        dataset, args.train_measurements, args.train_result, "train"
    )
    training_axes = [train_meta["axis"]]
    training_measurements = [args.train_measurements]
    for directory in getattr(args, "additional_train_pairs", []):
        extra, extra_meta = _load_records(
            dataset, directory / "measurements.pt", directory / "result.json", "train"
        )
        if extra_meta["axis"] in training_axes:
            raise ValueError("Duplicate training intervention axis")
        if extra_meta["offset"] != train_meta["offset"]:
            raise ValueError("Training intervention offsets differ")
        training_axes.append(extra_meta["axis"])
        training_measurements.append(directory / "measurements.pt")
        train.extend(extra)
        train_meta["excluded_incomplete_horizons"] += extra_meta[
            "excluded_incomplete_horizons"
        ]
    evaluation, evaluation_meta = _load_records(
        evaluation_dataset,
        args.evaluation_measurements,
        args.evaluation_result,
        args.evaluation_split,
    )
    if evaluation_meta["axis"] not in training_axes:
        raise ValueError("Training and evaluation intervention axes differ")
    if {row["case_id"] for row in train} & {row["case_id"] for row in evaluation}:
        raise ValueError("Sparse-motion training and evaluation layouts overlap")
    readout = FrozenReadout(args.checkpoint, expected_dataset=args.dataset)
    if readout.payload["method"] not in {"joint", "flow_joint", "state_action_joint"}:
        raise ValueError("Sparse motion requires a joint physical readout")
    object_rank = getattr(args, "object_rank", 0)
    object_index = dominant_object(train, rank=object_rank)
    object_key = train[0]["row"]["object_keys"][object_index]
    if any(
        len(record["row"]["object_keys"]) <= object_index
        or record["row"]["object_keys"][object_index] != object_key
        for record in train + evaluation
    ):
        raise ValueError("Sparse-motion object identity changed across splits")
    args.output.mkdir(parents=True)
    results = {}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for arm in getattr(args, "arms", ARMS):
        head, trace = fit_head(
            readout,
            train,
            object_index,
            arm,
            args.seed,
            args.steps,
            args.motion_threshold_m,
        )
        results[arm] = evaluate_head(
            readout,
            head,
            evaluation,
            object_index,
            args.motion_threshold_m,
        ) | {"loss_trace": trace}
        torch.save(
            {
                "schema": 1,
                "revision": revision,
                "arm": arm,
                "seed": args.seed,
                "steps": args.steps,
                "motion_threshold_m": args.motion_threshold_m,
                "object_index_selected_on_train_only": object_index,
                "object_rank_selected_on_train_only": object_rank,
                "object_key": object_key,
                **({"axis": training_axes[0]} if len(training_axes) == 1 else {}),
                "training_axes": sorted(training_axes),
                "offset": train_meta["offset"],
                "dataset_sha256": dataset_sha256(args.dataset),
                "readout_sha256": file_sha256(args.checkpoint),
                "train_measurements_sha256": file_sha256(args.train_measurements),
                "all_train_measurements_sha256": [
                    file_sha256(path) for path in training_measurements
                ],
                "model": head.state_dict(),
                "gated": head.gated,
            },
            args.output / f"{arm}.pt",
        )
    result = {
        "state": "complete",
        "revision": revision,
        "readout_method": readout.payload["method"],
        "training_dataset_sha256": dataset_sha256(args.dataset),
        "evaluation_dataset_sha256": dataset_sha256(evaluation_dataset_path),
        "seed": args.seed,
        "steps_per_arm": args.steps,
        "train_pairs": len(train),
        "evaluation_pairs": len(evaluation),
        "axis": evaluation_meta["axis"],
        "training_axes": sorted(training_axes),
        "evaluation_split": args.evaluation_split,
        "object_index_selected_on_train_only": object_index,
        "object_rank_selected_on_train_only": object_rank,
        "excluded_train_incomplete_horizons": train_meta[
            "excluded_incomplete_horizons"
        ],
        "excluded_evaluation_incomplete_horizons": evaluation_meta[
            "excluded_incomplete_horizons"
        ],
        "object_key": object_key,
        "results": results,
        "interpretation": (
            "offline sparse candidate-effect prediction on disjoint "
            f"{args.evaluation_split} layouts; not policy or online-RL evidence"
        ),
    }
    write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset",
        "checkpoint",
        "train-measurements",
        "train-result",
        "evaluation-measurements",
        "evaluation-result",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 9), default=1)
    parser.add_argument(
        "--evaluation-split",
        choices=("development", "confirmation"),
        default="development",
    )
    parser.add_argument("--evaluation-dataset", type=Path)
    parser.add_argument("--additional-train-pairs", nargs="*", type=Path, default=[])
    parser.add_argument("--motion-threshold-m", type=float, default=0.002)
    parser.add_argument("--object-rank", type=int, choices=range(8), default=0)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    if not 0.0005 <= args.motion_threshold_m <= 0.01:
        parser.error("Motion threshold must stay within 0.5--10 mm")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
