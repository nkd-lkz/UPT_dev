# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Cross-fit paired action-effect supervision on frozen-transition artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from toolkits.rlt.physrlt.control import FrozenReadout
from toolkits.rlt.physrlt.counterfactual import effect_errors
from toolkits.rlt.physrlt.data import HORIZONS, masked_loss
from toolkits.rlt.physrlt.probe import dataset_sha256, inputs
from toolkits.rlt.pi05_baseline import env_commands, write_json


def file_sha256(path: Path) -> str:
    """Hash an experiment artifact without loading or rewriting it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_records(
    dataset: dict,
    measurements: list[dict],
    axis: int,
    offset: float,
    *,
    expected_split: str = "train",
) -> list[dict]:
    """Join same-state branches to their pre-execution model inputs.

    The pair artifact contains only completed physical outcomes. Current-state
    features are recovered by exact ``(case_id, start_tick)`` identity from the
    immutable transition dataset. No nearest-neighbour or aggregate matching is
    allowed.
    """
    if axis not in (0, 1, 2) or not 0 < offset <= 0.1:
        raise ValueError("Paired interventions exceed the declared action bounds")
    rows = {(row["case_id"], row["start_tick"]): row for row in dataset["rows"]}
    if len(rows) != len(dataset["rows"]):
        raise ValueError("Transition identities must be unique")
    records, identities = [], set()
    for measurement in measurements:
        key = (measurement["case"]["id"], measurement["start_tick"])
        if key in identities:
            raise ValueError("Duplicate paired-action state")
        identities.add(key)
        if key not in rows:
            raise ValueError("Paired-action state is absent from the transition data")
        row = rows[key]
        if row["split"] != expected_split:
            raise ValueError(f"Expected paired states from the {expected_split} split")
        if set(measurement["branches"]) != {-1, 1}:
            raise ValueError("Each paired state requires exactly two signed branches")
        branches = {}
        for sign in (-1, 1):
            branch = measurement["branches"][sign]
            if branch["horizon_mask"].shape != (len(HORIZONS),):
                raise ValueError("Unexpected paired horizon mask")
            if branch["horizon_mask"].dtype != torch.bool:
                raise ValueError("Paired horizon mask must be boolean")
            for field in ("response", "geometry"):
                expected = (len(HORIZONS), row[field].shape[-1])
                if branch[field].shape != expected:
                    raise ValueError(f"Unexpected paired {field} shape")
                if not torch.isfinite(branch[field]).all():
                    raise ValueError(f"Nonfinite paired {field}")
            commands = torch.from_numpy(env_commands(row["ref"].numpy())).float()
            commands[:, axis] = (commands[:, axis] + sign * offset).clamp(-1, 1)
            branches[sign] = {
                "commands": commands,
                "horizon_mask": branch["horizon_mask"].clone(),
                "response": branch["response"].clone(),
                "geometry": branch["geometry"].clone(),
            }
        separation = (
            branches[1]["commands"][:, axis] - branches[-1]["commands"][:, axis]
        )
        if not torch.allclose(
            separation, torch.full_like(separation, 2 * offset), atol=1e-6, rtol=0
        ):
            raise ValueError("Action clipping changed the paired intervention")
        records.append(
            {"case_id": key[0], "start_tick": key[1], "row": row, "branches": branches}
        )
    if len({record["case_id"] for record in records}) < 2:
        raise ValueError("Paired effects require at least two layouts")
    return records


def _current_inputs(row: dict, stats: dict, method: str) -> tuple[torch.Tensor, ...]:
    data = {
        key: row[key][None]
        for key in ("action_feature", "prefix_feature", "ref", "prop")
    }
    return inputs(data, stats, method)


def _field_loss(
    prediction: dict, target: dict, mask: torch.Tensor, stats: dict
) -> torch.Tensor:
    response = masked_loss(
        prediction["response"],
        target["response"] / stats["response_scale"],
        mask,
    )
    geometry_target = target["geometry"] / stats["geometry_scale"]
    geometry = 0.5 * (
        masked_loss(
            prediction["geometry"][..., :64],
            geometry_target[..., :64],
            mask,
        )
        + masked_loss(
            prediction["geometry"][..., 64:],
            geometry_target[..., 64:],
            mask,
        )
    )
    return response + geometry


def paired_objective(model, stats: dict, method: str, record: dict, mode: str):
    """Return matched pointwise or action-effect supervision for one state."""
    if mode not in {
        "pointwise",
        "effect",
        "reversed_effect",
        "object_effect",
        "reversed_object_effect",
    }:
        raise ValueError("Unknown paired supervision mode")
    token = model.encode(*_current_inputs(record["row"], stats, method))
    predictions = {
        sign: model.consequences(token, branch["commands"][None])
        for sign, branch in record["branches"].items()
    }
    valid = (
        record["branches"][-1]["horizon_mask"] & record["branches"][1]["horizon_mask"]
    )[None]
    if not valid.any():
        raise ValueError("Paired state has no common completed horizon")
    if mode == "pointwise":
        return torch.stack(
            [
                _field_loss(
                    predictions[sign],
                    {
                        field: record["branches"][sign][field][None]
                        for field in ("response", "geometry")
                    },
                    valid,
                    stats,
                )
                for sign in (-1, 1)
            ]
        ).mean()
    difference_prediction = {
        field: predictions[1][field] - predictions[-1][field]
        for field in ("response", "geometry")
    }
    difference_target = {
        field: (record["branches"][1][field] - record["branches"][-1][field])[None]
        for field in ("response", "geometry")
    }
    if mode in {"reversed_effect", "reversed_object_effect"}:
        difference_target = {
            field: -target for field, target in difference_target.items()
        }
    if mode in {"object_effect", "reversed_object_effect"}:
        return masked_loss(
            difference_prediction["geometry"][..., 64:],
            difference_target["geometry"][..., 64:] / stats["geometry_scale"][64:],
            valid,
        )
    return _field_loss(difference_prediction, difference_target, valid, stats)


def fine_tune(
    base,
    stats: dict,
    method: str,
    records: list[dict],
    mode: str,
    seed: int,
    steps: int,
    parameter_scope: str = "all",
):
    """Update an identical base model with one paired supervision objective."""
    if parameter_scope not in {"all", "decoders"}:
        raise ValueError("Paired updates require all or decoders parameter scope")
    torch.manual_seed(seed)
    model = copy.deepcopy(base).train().requires_grad_(parameter_scope == "all")
    if parameter_scope == "decoders":
        model.response.requires_grad_(True)
        model.geometry.requires_grad_(True)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    trace = []
    for step in range(steps):
        selected = records[int(torch.randint(len(records), (), generator=generator))]
        loss = paired_objective(model, stats, method, selected, mode)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {mode} paired objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"Nonfinite {mode} paired gradients")
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            trace.append(
                {"step": step + 1, "loss": float(loss.detach()), "grad": float(grad)}
            )
    return model.eval().requires_grad_(False), trace


@torch.no_grad()
def evaluate_effects(
    model, stats: dict, method: str, records: list[dict]
) -> list[dict]:
    """Evaluate command-conditioned differences on paired labels never fitted."""
    results = []
    for record in records:
        token = model.encode(*_current_inputs(record["row"], stats, method))
        predictions = {}
        for sign in (-1, 1):
            prediction = model.consequences(
                token, record["branches"][sign]["commands"][None]
            )
            predictions[sign] = {
                field: prediction[field] * stats[field + "_scale"]
                for field in ("response", "geometry")
            }
        for result in effect_errors(
            record["branches"][1],
            record["branches"][-1],
            predictions[1],
            predictions[-1],
            stats,
        ):
            results.append(
                {
                    "case_id": record["case_id"],
                    "start_tick": record["start_tick"],
                    **result,
                }
            )
    return results


def summarize(rows: list[dict]) -> dict:
    """Aggregate paired errors without treating horizons as independent seeds."""
    result = {}
    for horizon in HORIZONS:
        selected = [row for row in rows if row["horizon"] == horizon]
        if not selected:
            result[str(horizon)] = {"paired_states": 0}
            continue
        result[str(horizon)] = {
            key: sum(row[key] for row in selected) / len(selected)
            for key in (
                "response_effect_scaled_mse",
                "zero_effect_response_scaled_mse",
                "object_effect_rmse_m",
                "zero_effect_object_rmse_m",
            )
        } | {"paired_states": len(selected)}
    return result


def effect_checkpoint(
    readout: FrozenReadout,
    model: torch.nn.Module,
    *,
    supervision: str,
    measurements: Path,
    seed: int,
    steps: int,
    revision: str,
    parameter_scope: str,
) -> dict:
    """Build a deployable readout with explicit paired-supervision provenance."""
    if supervision not in {
        "pointwise",
        "effect",
        "reversed_effect",
        "object_effect",
        "reversed_object_effect",
    }:
        raise ValueError("Unknown effect checkpoint supervision")
    payload = {
        key: value
        for key, value in readout.payload.items()
        if key not in {"model", "revision", "seed"}
    }
    return payload | {
        "model": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "revision": revision,
        "seed": seed,
        "effect_supervision": supervision,
        "effect_steps": steps,
        "effect_parameter_scope": parameter_scope,
        "effect_measurements": str(measurements.resolve()),
        "effect_measurements_sha256": file_sha256(measurements),
    }


def run(args: argparse.Namespace) -> dict:
    """Cross-fit paired labels by layout and retain all fold-level outcomes."""
    if not 1 <= args.steps <= 400:
        raise ValueError("Paired pilot is bounded to 1-400 updates")
    if args.output.exists():
        raise FileExistsError("Paired-effect output must be a new directory")
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
    measurements = torch.load(args.measurements, map_location="cpu", weights_only=True)
    result_meta = json.loads(args.paired_result.read_text())
    if result_meta.get("state") != "complete" or not result_meta.get("no_training"):
        raise ValueError("Paired artifact must be a completed measurement-only run")
    if result_meta.get("paired_states") != len(measurements):
        raise ValueError("Paired measurement count differs from its completion record")
    records = paired_records(
        dataset, measurements, result_meta["axis"], result_meta["offset"]
    )
    readout = FrozenReadout(
        args.checkpoint,
        expected_dataset=args.dataset,
        expected_method="joint",
    )
    evaluation_measurements = getattr(args, "evaluation_measurements", None)
    evaluation_paired_result = getattr(args, "evaluation_paired_result", None)
    if bool(evaluation_measurements) != bool(evaluation_paired_result):
        raise ValueError("Evaluation measurements and result must be provided together")
    evaluation_records, evaluation_meta = None, None
    parameter_scope = getattr(args, "parameter_scope", "all")
    if parameter_scope not in {"all", "decoders"}:
        raise ValueError("Unknown paired-effect parameter scope")
    if evaluation_measurements:
        evaluation = torch.load(
            evaluation_measurements, map_location="cpu", weights_only=True
        )
        evaluation_meta = json.loads(evaluation_paired_result.read_text())
        if (
            evaluation_meta.get("state") != "complete"
            or not evaluation_meta.get("no_training")
            or evaluation_meta.get("split") != "development"
            or evaluation_meta.get("paired_states") != len(evaluation)
        ):
            raise ValueError("Evaluation artifact is not a complete development run")
        evaluation_records = paired_records(
            dataset,
            evaluation,
            evaluation_meta["axis"],
            evaluation_meta["offset"],
            expected_split="development",
        )
        if {row["case_id"] for row in records} & {
            row["case_id"] for row in evaluation_records
        }:
            raise ValueError("Training and evaluation paired layouts overlap")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    folds = []
    if evaluation_records is None:
        cases = sorted({record["case_id"] for record in records})
        partitions = [
            (
                heldout,
                [record for record in records if record["case_id"] != heldout],
                [record for record in records if record["case_id"] == heldout],
            )
            for heldout in cases
        ]
    else:
        partitions = [("development", records, evaluation_records)]
    for fold, (heldout, train, test) in enumerate(partitions):
        arms = {"base": (readout.model, [])}
        for arm in (
            "pointwise",
            "effect",
            "reversed_effect",
            "object_effect",
            "reversed_object_effect",
        ):
            arms[arm] = fine_tune(
                readout.model,
                readout.payload["stats"],
                readout.payload["method"],
                train,
                arm,
                args.seed + fold,
                args.steps,
                parameter_scope,
            )
        if evaluation_records is not None:
            checkpoint_root = args.output / "checkpoints"
            checkpoint_root.mkdir(exist_ok=True)
            for arm in (
                "pointwise",
                "effect",
                "reversed_effect",
                "object_effect",
                "reversed_object_effect",
            ):
                torch.save(
                    effect_checkpoint(
                        readout,
                        arms[arm][0],
                        supervision=arm,
                        measurements=args.measurements,
                        seed=args.seed + fold,
                        steps=args.steps,
                        revision=revision,
                        parameter_scope=parameter_scope,
                    ),
                    checkpoint_root / f"{arm}.pt",
                )
        folds.append(
            {
                "heldout_paired_case": heldout,
                "training_paired_cases": sorted({row["case_id"] for row in train}),
                "arms": {
                    arm: {
                        "summary": summarize(
                            evaluate_effects(
                                model,
                                readout.payload["stats"],
                                readout.payload["method"],
                                test,
                            )
                        ),
                        "loss_trace": trace,
                    }
                    for arm, (model, trace) in arms.items()
                },
            }
        )
    result = {
        "state": "complete",
        "revision": revision,
        "dataset_sha256": dataset_sha256(args.dataset),
        "measurements_sha256": file_sha256(args.measurements),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "seed": args.seed,
        "steps_per_fold_arm": args.steps,
        "parameter_scope": parameter_scope,
        "paired_states": len(records),
        "evaluation_paired_states": (
            len(evaluation_records) if evaluation_records is not None else None
        ),
        "evaluation_measurements_sha256": (
            file_sha256(evaluation_measurements) if evaluation_measurements else None
        ),
        "folds": folds,
        "seconds": time.monotonic() - started,
        "interpretation": (
            "training-pair fit evaluated on readout-held-out development layouts; not control evidence"
            if evaluation_records is not None
            else "paired-label cross-fit development evidence; not held-out-layout or control evidence"
        ),
    }
    write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "measurements", "paired-result", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evaluation-measurements", type=Path)
    parser.add_argument("--evaluation-paired-result", type=Path)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--parameter-scope", choices=("all", "decoders"), default="all")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
