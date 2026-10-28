# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Test a bounded object/EEF transport prior using measured training effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toolkits.rlt.physrlt.goal_ranking import load_audited_readout
from toolkits.rlt.physrlt.object_motion_probe import (
    RIDGES,
    _ridge_predict,
    dominant_object,
    object_effect,
)
from toolkits.rlt.physrlt.sparse_motion import _load_records, _token
from toolkits.rlt.pi05_baseline import write_json


def transport_coefficient(response: torch.Tensor, effect: torch.Tensor) -> torch.Tensor:
    """Project object effects onto EEF effects with a coefficient in [0, 1].

    This is an optional local transport prior, not a universal contact law.
    Measured responses and object effects supply training labels only.
    """
    if response.shape != effect.shape or response.ndim != 2 or response.shape[-1] != 3:
        raise ValueError("Transport targets must be matching Nx3 tensors")
    if not torch.isfinite(response).all() or not torch.isfinite(effect).all():
        raise ValueError("Transport targets must be finite")
    return (
        (response * effect).sum(-1) / response.square().sum(-1).clamp_min(1e-10)
    ).clamp(0, 1)[:, None]


def crossfit_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    cases: list[str],
    test_x: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Choose regularization using training-layout folds, then fit all training rows."""
    layouts = sorted(set(cases))
    if len(layouts) < 2 or len(cases) != len(x):
        raise ValueError("Cross-fitting requires at least two training layouts")
    scores = {}
    for strength in RIDGES:
        losses = []
        for case in layouts:
            heldout = torch.tensor([value == case for value in cases])
            prediction = _ridge_predict(x[~heldout], y[~heldout], x[heldout], strength)
            losses.append(float((prediction - y[heldout]).square().mean()))
        scores[str(strength)] = sum(losses) / len(losses)
    strength = min(RIDGES, key=lambda value: scores[str(value)])
    return _ridge_predict(x, y, test_x, strength), {
        "ridge": strength,
        "train_layout_cv_mse": scores,
    }


def response_features(records: list[dict]) -> torch.Tensor:
    """Construct current-state and candidate-command inputs without outcomes."""
    return torch.stack(
        [
            torch.cat(
                (
                    r["row"]["prop"].flatten(),
                    r["row"]["ref"].flatten(),
                    (r["branches"][1]["commands"] - r["branches"][-1]["commands"]).mean(
                        0
                    ),
                )
            )
            for r in records
        ]
    ).double()


def response_targets(records: list[dict]) -> torch.Tensor:
    """Read measured 10-tick paired EEF displacements in metres."""
    return torch.stack(
        [
            r["branches"][1]["response"][-1, 7:10]
            - r["branches"][-1]["response"][-1, 7:10]
            for r in records
        ]
    ).double()


def score_prediction(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Compare a vector prediction with zero effect on the same physical pairs."""
    zero = float(target.square().mean().sqrt())
    error = float((prediction - target).square().mean().sqrt())
    return {
        "rmse_mm": error * 1000,
        "zero_rmse_mm": zero * 1000,
        "improvement_vs_zero_pct": 100 * (1 - error / zero) if zero > 0 else None,
    }


def layout_comparison(
    predictions: dict[str, torch.Tensor], target: torch.Tensor, cases: list[str]
) -> dict:
    """Report paired errors with layouts, not correlated transitions, as clusters.

    Intervals are exploratory percentile bootstrap intervals for equal-layout
    mean squared error differences. Positive values favour the coupled arm.
    They do not correct for repeated model selection on development layouts.
    """
    if len(cases) != len(target) or not cases:
        raise ValueError("Every evaluation target must have a layout identity")
    layouts = sorted(set(cases))
    errors = {}
    for arm, prediction in predictions.items():
        if prediction.shape != target.shape:
            raise ValueError("Predictions and targets must have matching shapes")
        errors[arm] = [
            float((prediction[mask] - target[mask]).square().mean()) * 1e6
            for case in layouts
            for mask in [torch.tensor([value == case for value in cases])]
        ]
    generator = torch.Generator().manual_seed(1028)
    draws = torch.randint(len(layouts), (2000, len(layouts)), generator=generator)
    comparisons = {}
    for arm in (
        "direct",
        "constant_coupling",
        "state_action_coupling",
        "shuffled_coupling",
    ):
        differences = torch.tensor(errors[arm]) - torch.tensor(errors["coupled"])
        interval = torch.quantile(
            differences[draws].mean(-1), torch.tensor([0.025, 0.975])
        )
        comparisons[arm] = {
            "mse_improvement_mm2": float(differences.mean()),
            "exploratory_layout_bootstrap_95pct": interval.tolist(),
            "layouts_improved": int((differences > 0).sum()),
        }
    return {"layouts": layouts, "mse_mm2_by_layout": errors, "coupled_vs": comparisons}


def run(args: argparse.Namespace) -> dict:
    """Compare direct regression with predicted response times predicted coupling."""
    torch.set_num_threads(1)
    if args.output.exists():
        raise FileExistsError("Transport output must be a new directory")
    root = args.root
    dataset = torch.load(root / "collection_v1/transitions.pt", weights_only=True)
    train, evaluation = [], []
    evaluation_dataset = (
        dataset
        if args.split == "development"
        else torch.load(
            root / "confirmation_collection_30_35_v1/transitions.pt", weights_only=True
        )
    )
    for axis in (0, 1):
        pair = root / f"paired_train_axis{axis}_multiphase_v1"
        records, _ = _load_records(
            dataset, pair / "measurements.pt", pair / "result.json", "train"
        )
        train.extend(records)
        pair = root / (
            f"paired_development_axis{axis}_multiphase_v1"
            if args.split == "development"
            else f"paired_confirmation_axis{axis}_multiphase_reused_layouts_v1"
        )
        records, _ = _load_records(
            evaluation_dataset,
            pair / "measurements.pt",
            pair / "result.json",
            args.split,
        )
        evaluation.extend(records)
    cases = [r["case_id"] for r in train]
    if set(cases) & {r["case_id"] for r in evaluation}:
        raise ValueError("Training and evaluation layouts overlap")
    x_response, test_response = response_features(train), response_features(evaluation)
    measured_train_response, measured_test_response = (
        response_targets(train),
        response_targets(evaluation),
    )
    predicted_response, response_cv = crossfit_ridge(
        x_response, measured_train_response, cases, test_response
    )
    results = []
    for method in ("joint", "flow_joint", "state_action_joint"):
        for seed in (31, 47, 59):
            readout = load_audited_readout(
                root / f"probes_v1/{method}-seed{seed}/readout.pt"
            )
            with torch.no_grad():
                x = torch.cat([_token(readout, r) for r in train]).double()
                test_x = torch.cat([_token(readout, r) for r in evaluation]).double()
            x = torch.cat((x, x_response), dim=1)
            test_x = torch.cat((test_x, test_response), dim=1)
            for rank in (0, 1):
                index = dominant_object(train, rank)
                key = train[0]["row"]["object_keys"][index]
                if any(
                    r["row"]["object_keys"][index] != key for r in train + evaluation
                ):
                    raise ValueError("Transport object identity changed")
                target = torch.stack([object_effect(r, index) for r in train])
                truth = torch.stack([object_effect(r, index) for r in evaluation])
                coefficient = transport_coefficient(measured_train_response, target)
                direct, direct_cv = crossfit_ridge(x, target, cases, test_x)
                predicted_coefficient, coefficient_cv = crossfit_ridge(
                    x, coefficient, cases, test_x
                )
                generator = torch.Generator().manual_seed(seed)
                shuffled = coefficient[
                    torch.randperm(len(coefficient), generator=generator)
                ]
                wrong_coefficient, wrong_cv = crossfit_ridge(x, shuffled, cases, test_x)
                state_coefficient, state_cv = crossfit_ridge(
                    x_response, coefficient, cases, test_response
                )
                state_direct, state_direct_cv = crossfit_ridge(
                    x_response, target, cases, test_response
                )
                coupled = predicted_coefficient.clamp(0, 1) * predicted_response
                oracle_coefficient = transport_coefficient(
                    measured_test_response, truth
                )
                arms = {
                    "direct": direct,
                    "coupled": coupled,
                    "constant_coupling": coefficient.mean() * predicted_response,
                    "state_action_coupling": state_coefficient.clamp(0, 1)
                    * predicted_response,
                    "state_action_direct": state_direct,
                    "shuffled_coupling": wrong_coefficient.clamp(0, 1)
                    * predicted_response,
                    "predicted_coupling_oracle_response": predicted_coefficient.clamp(
                        0, 1
                    )
                    * measured_test_response,
                    "oracle_coupling_and_response": oracle_coefficient
                    * measured_test_response,
                }
                results.append(
                    {
                        "method": method,
                        "seed": seed,
                        "object": key,
                        "rank": rank,
                        "results": {
                            name: score_prediction(pred, truth)
                            for name, pred in arms.items()
                        },
                        "direct_cv": direct_cv,
                        "coefficient_cv": coefficient_cv,
                        "shuffled_cv": wrong_cv,
                        "state_action_cv": state_cv,
                        "state_action_direct_cv": state_direct_cv,
                        "layout_comparison": layout_comparison(
                            arms, truth, [r["case_id"] for r in evaluation]
                        ),
                    }
                )
    result = {
        "state": "complete",
        "split": args.split,
        "train_pairs": len(train),
        "evaluation_pairs": len(evaluation),
        "train_layouts": len(set(cases)),
        "evaluation_layouts": len({r["case_id"] for r in evaluation}),
        "response_cv": response_cv,
        "predicted_response": score_prediction(
            predicted_response, measured_test_response
        ),
        "results": results,
        "interpretation": "exploratory offline transport prior; oracle arms use measured future quantities and are not deployable",
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("development", "confirmation"), required=True
    )
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
