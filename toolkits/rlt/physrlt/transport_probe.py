# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Test a bounded object/EEF transport prior using measured training effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toolkits.rlt.physrlt.effect_probe import paired_records
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


def transport_jacobian(response: torch.Tensor, effect: torch.Tensor) -> torch.Tensor:
    """Build a minimum-norm local transport Jacobian with gain at most one."""
    if response.shape != effect.shape or response.ndim != 2 or response.shape[-1] != 3:
        raise ValueError("Jacobian targets must be matching Nx3 tensors")
    if not torch.isfinite(response).all() or not torch.isfinite(effect).all():
        raise ValueError("Jacobian targets must be finite")
    denominator = response.square().sum(-1).clamp_min(1e-10)
    matrix = effect[:, :, None] * response[:, None, :] / denominator[:, None, None]
    return project_transport_jacobian(matrix)


def project_transport_jacobian(matrix: torch.Tensor) -> torch.Tensor:
    """Project batched 3-D transport matrices to spectral gain no greater than one."""
    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 3):
        raise ValueError("Transport Jacobians must have shape Nx3x3")
    gain = torch.linalg.matrix_norm(matrix, ord=2).clamp_min(1e-10)
    scale = (1 / gain).clamp_max(1)
    return matrix * scale[:, None, None]


def apply_transport_jacobian(
    matrix: torch.Tensor, response: torch.Tensor
) -> torch.Tensor:
    """Apply a bounded local transport Jacobian to a predicted EEF response."""
    matrix = project_transport_jacobian(matrix.reshape(-1, 3, 3))
    if response.shape != (len(matrix), 3):
        raise ValueError("Transport response does not align with Jacobians")
    return torch.bmm(matrix, response[:, :, None])[:, :, 0]


def transport_vector(response: torch.Tensor, effect: torch.Tensor) -> torch.Tensor:
    """Normalize object motion by response magnitude and bound its vector gain."""
    if response.shape != effect.shape or response.ndim != 2 or response.shape[-1] != 3:
        raise ValueError("Transport vectors require matching Nx3 tensors")
    magnitude = response.norm(dim=1, keepdim=True).clamp_min(1e-5)
    vector = effect / magnitude
    gain = vector.norm(dim=1, keepdim=True).clamp_min(1e-10)
    return vector * (1 / gain).clamp_max(1)


def apply_transport_vector(
    vector: torch.Tensor, response: torch.Tensor
) -> torch.Tensor:
    """Scale a bounded motion direction by predicted response magnitude."""
    if vector.shape != response.shape or vector.ndim != 2 or vector.shape[-1] != 3:
        raise ValueError("Transport vectors and responses must be matching Nx3 tensors")
    gain = vector.norm(dim=1, keepdim=True).clamp_min(1e-10)
    vector = vector * (1 / gain).clamp_max(1)
    return vector * response.norm(dim=1, keepdim=True)


def transport_residual(response: torch.Tensor, effect: torch.Tensor) -> torch.Tensor:
    """Return object motion not explained by bounded collinear transport."""
    coefficient = transport_coefficient(response, effect)
    return effect - coefficient * response


def apply_transport_residual(
    base: torch.Tensor,
    residual: torch.Tensor,
    response: torch.Tensor,
    *,
    max_response_ratio: float = 1.0,
) -> torch.Tensor:
    """Add a bounded geometry residual to a collinear transport prediction."""
    if base.shape != residual.shape or base.shape != response.shape:
        raise ValueError("Transport residual inputs must have matching shapes")
    if base.ndim != 2 or base.shape[-1] != 3:
        raise ValueError("Transport residual inputs must have shape Nx3")
    if max_response_ratio < 0:
        raise ValueError("Residual response ratio must be nonnegative")
    limit = max_response_ratio * response.norm(dim=1, keepdim=True)
    magnitude = residual.norm(dim=1, keepdim=True).clamp_min(1e-10)
    bounded = residual * (limit / magnitude).clamp_max(1)
    return base + bounded


def select_residual_ratio(
    base: torch.Tensor,
    residual: torch.Tensor,
    response: torch.Tensor,
    target: torch.Tensor,
    candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
) -> tuple[float, dict[str, float]]:
    """Select residual strength from training-layout predictions only."""
    if not candidates or any(value < 0 for value in candidates):
        raise ValueError("Residual ratio candidates must be nonnegative")
    losses = {
        str(value): float(
            (
                apply_transport_residual(
                    base,
                    residual,
                    response,
                    max_response_ratio=value,
                )
                - target
            )
            .square()
            .mean()
        )
        for value in candidates
    }
    selected = min(candidates, key=lambda value: losses[str(value)])
    return selected, losses


def select_incremental_ratio(
    coefficient: torch.Tensor,
    correction: torch.Tensor,
    response: torch.Tensor,
    target: torch.Tensor,
    candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
) -> tuple[float, dict[str, float]]:
    """Select action-token correction strength from training predictions only."""
    if coefficient.ndim != 2 or coefficient.shape[-1] != 1:
        raise ValueError("Incremental coefficient must have shape Nx1")
    if correction.shape != coefficient.shape:
        raise ValueError("Incremental correction must match the coefficient")
    if response.shape != target.shape or response.shape != (len(coefficient), 3):
        raise ValueError("Incremental response and target must have shape Nx3")
    if not candidates or any(value < 0 for value in candidates):
        raise ValueError("Incremental ratio candidates must be nonnegative")
    losses = {
        str(value): float(
            ((coefficient + value * correction).clamp(0, 1) * response - target)
            .square()
            .mean()
        )
        for value in candidates
    }
    selected = min(candidates, key=lambda value: losses[str(value)])
    return selected, losses


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


def out_of_layout_predictions(
    x: torch.Tensor, y: torch.Tensor, cases: list[str], ridge: float
) -> torch.Tensor:
    """Predict every training row from models that exclude its entire layout."""
    if len(x) != len(y) or len(x) != len(cases):
        raise ValueError("Out-of-layout rows must align")
    layouts = sorted(set(cases))
    if len(layouts) < 2:
        raise ValueError("Out-of-layout prediction requires multiple layouts")
    prediction = torch.empty_like(y)
    for layout in layouts:
        heldout = torch.tensor([case == layout for case in cases])
        prediction[heldout] = _ridge_predict(
            x[~heldout], y[~heldout], x[heldout], ridge
        )
    return prediction


def gated_coefficient(
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    coefficient: torch.Tensor,
    moving: torch.Tensor,
    cases: list[str],
    *,
    shuffle_seed: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Predict motion probability times a coefficient fitted on motion events."""
    if moving.dtype != torch.bool or moving.shape != (len(train_x),):
        raise ValueError("Motion mask must align with coupling rows")
    moving_rows = int(moving.sum())
    stationary_rows = int((~moving).sum())
    if moving_rows == 0:
        return torch.zeros(len(test_x), 1, dtype=coefficient.dtype), {
            "moving_rows": 0,
            "stationary_rows": stationary_rows,
            "event_cv": {"constant_label": 0},
            "conditional_cv": None,
            "event_labels_shuffled": shuffle_seed is not None,
        }
    if moving_rows < 2:
        raise ValueError("Gated coupling requires at least two moving examples")
    event_target = moving.double()[:, None]
    if shuffle_seed is not None:
        generator = torch.Generator().manual_seed(shuffle_seed)
        event_target = event_target[
            torch.randperm(len(event_target), generator=generator)
        ]
    if stationary_rows == 0:
        event = torch.ones(len(test_x), 1, dtype=coefficient.dtype)
        event_cv = {"constant_label": 1}
    elif stationary_rows < 2:
        raise ValueError("Gated coupling requires at least two stationary examples")
    else:
        event, event_cv = crossfit_ridge(train_x, event_target, cases, test_x)
    moving_cases = [case for case, active in zip(cases, moving, strict=True) if active]
    conditional, conditional_cv = crossfit_ridge(
        train_x[moving], coefficient[moving], moving_cases, test_x
    )
    return event.clamp(0, 1) * conditional.clamp(0, 1), {
        "moving_rows": moving_rows,
        "stationary_rows": stationary_rows,
        "event_cv": event_cv,
        "conditional_cv": conditional_cv,
        "event_labels_shuffled": shuffle_seed is not None,
    }


def out_of_layout_gated_predictions(
    x: torch.Tensor,
    coefficient: torch.Tensor,
    moving: torch.Tensor,
    cases: list[str],
) -> torch.Tensor:
    """Predict gated coefficients while excluding each row's entire layout."""
    if len(x) != len(coefficient) or len(x) != len(moving) or len(x) != len(cases):
        raise ValueError("Out-of-layout gated rows must align")
    layouts = sorted(set(cases))
    if len(layouts) < 3:
        raise ValueError("Gated out-of-layout prediction requires three layouts")
    prediction = torch.empty_like(coefficient)
    for layout in layouts:
        heldout = torch.tensor([case == layout for case in cases])
        remaining_cases = [
            case for case, excluded in zip(cases, heldout, strict=True) if not excluded
        ]
        prediction[heldout], _ = gated_coefficient(
            x[~heldout],
            x[heldout],
            coefficient[~heldout],
            moving[~heldout],
            remaining_cases,
        )
    return prediction


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
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
    cases: list[str],
    *,
    reference: str = "coupled",
    comparators: tuple[str, ...] = (
        "direct",
        "constant_coupling",
        "state_action_coupling",
        "shuffled_coupling",
    ),
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
    if reference not in errors or any(arm not in errors for arm in comparators):
        raise ValueError("Layout comparison arms are missing")
    for arm in comparators:
        differences = torch.tensor(errors[arm]) - torch.tensor(errors[reference])
        interval = torch.quantile(
            differences[draws].mean(-1), torch.tensor([0.025, 0.975])
        )
        comparisons[arm] = {
            "mse_improvement_mm2": float(differences.mean()),
            "exploratory_layout_bootstrap_95pct": interval.tolist(),
            "layouts_improved": int((differences > 0).sum()),
        }
    return {
        "layouts": layouts,
        "mse_mm2_by_layout": errors,
        f"{reference}_vs": comparisons,
    }


def phase_expand(
    features: torch.Tensor,
    records: list[dict],
    *,
    shuffle_seed: int | None = None,
) -> torch.Tensor:
    """Add equal-capacity phase interactions using only pre-execution time."""
    if features.ndim != 2 or len(features) != len(records):
        raise ValueError("Phase features must align with paired records")
    phases = torch.tensor(
        [min(int(record["start_tick"]) // 100, 2) for record in records]
    )
    if shuffle_seed is not None:
        generator = torch.Generator().manual_seed(shuffle_seed)
        phases = phases[torch.randperm(len(phases), generator=generator)]
    one_hot = torch.nn.functional.one_hot(phases, num_classes=3).to(features)
    interactions = (features[:, :, None] * one_hot[:, None, :]).flatten(1)
    return torch.cat((features, one_hot, interactions), dim=1)


def active_object_labels(
    records: list[dict], object_indices: list[int], motion_threshold_m: float
) -> torch.Tensor:
    """Label no motion or the object with the largest paired displacement."""
    if len(object_indices) < 2 or not 0 < motion_threshold_m:
        raise ValueError(
            "Active-object labels require multiple objects and a threshold"
        )
    magnitudes = torch.stack(
        [
            torch.stack(
                [object_effect(record, index).norm() for index in object_indices]
            )
            for record in records
        ]
    )
    largest, labels = magnitudes.max(1)
    return torch.where(largest > motion_threshold_m, labels + 1, 0)


def classification_report(scores: torch.Tensor, labels: torch.Tensor) -> dict:
    """Report accuracy and macro recall without treating rows as independent layouts."""
    if scores.ndim != 2 or len(scores) != len(labels):
        raise ValueError("Classification scores and labels must align")
    prediction = scores.argmax(1)
    classes = sorted(set(labels.tolist()))
    recalls = []
    confusion = torch.zeros(scores.shape[1], scores.shape[1], dtype=torch.int64)
    for expected, actual in zip(labels, prediction, strict=True):
        confusion[int(expected), int(actual)] += 1
    for label in classes:
        mask = labels == label
        recalls.append(float((prediction[mask] == label).float().mean()))
    return {
        "examples": len(labels),
        "accuracy_pct": float((prediction == labels).float().mean() * 100),
        "macro_recall_pct": 100 * sum(recalls) / len(recalls),
        "class_counts": {str(label): int((labels == label).sum()) for label in classes},
        "confusion": confusion.tolist(),
    }


def classification_layout_comparison(
    scores: dict[str, torch.Tensor], labels: torch.Tensor, cases: list[str]
) -> dict:
    """Compare active-object accuracy with layout-clustered uncertainty."""
    if set(scores) != {
        "action_side",
        "state_action",
        "time_phase",
        "shuffled_labels",
        "fixed_train_majority",
    }:
        raise ValueError("Active-object comparison arms are incomplete")
    if len(cases) != len(labels):
        raise ValueError("Active-object layouts must align with labels")
    layouts = sorted(set(cases))
    predictions = {name: value.argmax(1) for name, value in scores.items()}
    accuracy = {}
    for name, prediction in predictions.items():
        accuracy[name] = [
            float((prediction[mask] == labels[mask]).float().mean())
            for layout in layouts
            for mask in [torch.tensor([case == layout for case in cases])]
        ]
    generator = torch.Generator().manual_seed(1029)
    draws = torch.randint(len(layouts), (2000, len(layouts)), generator=generator)
    comparisons = {}
    for name in scores:
        if name == "action_side":
            continue
        difference = torch.tensor(accuracy["action_side"]) - torch.tensor(
            accuracy[name]
        )
        interval = torch.quantile(
            difference[draws].mean(-1), torch.tensor([0.025, 0.975])
        )
        comparisons[name] = {
            "accuracy_advantage_pct": 100 * float(difference.mean()),
            "exploratory_layout_bootstrap_95pct": (100 * interval).tolist(),
            "layouts_improved": int((difference > 0).sum()),
        }
    return {
        "layouts": layouts,
        "accuracy_by_layout": accuracy,
        "action_side_vs": comparisons,
    }


def active_object_probe(
    train: list[dict],
    evaluation: list[dict],
    train_x: torch.Tensor,
    evaluation_x: torch.Tensor,
    train_state_action: torch.Tensor,
    evaluation_state_action: torch.Tensor,
    cases: list[str],
    object_indices: list[int],
    seed: int,
    motion_threshold_m: float = 0.0001,
) -> dict:
    """Compare action-side active-object evidence with deployment-safe controls."""
    train_labels = active_object_labels(train, object_indices, motion_threshold_m)
    labels = active_object_labels(evaluation, object_indices, motion_threshold_m)
    classes = len(object_indices) + 1
    targets = torch.nn.functional.one_hot(train_labels, classes).double()
    action_scores, action_cv = crossfit_ridge(train_x, targets, cases, evaluation_x)
    state_scores, state_cv = crossfit_ridge(
        train_state_action, targets, cases, evaluation_state_action
    )
    train_phase = torch.nn.functional.one_hot(
        torch.tensor([min(int(row["start_tick"]) // 100, 2) for row in train]), 3
    ).double()
    evaluation_phase = torch.nn.functional.one_hot(
        torch.tensor([min(int(row["start_tick"]) // 100, 2) for row in evaluation]),
        3,
    ).double()
    phase_scores, phase_cv = crossfit_ridge(
        train_phase, targets, cases, evaluation_phase
    )
    generator = torch.Generator().manual_seed(seed)
    shuffled_targets = targets[torch.randperm(len(targets), generator=generator)]
    shuffled_scores, shuffled_cv = crossfit_ridge(
        train_x, shuffled_targets, cases, evaluation_x
    )
    majority = int(torch.bincount(train_labels, minlength=classes).argmax())
    majority_scores = torch.zeros(len(labels), classes, dtype=torch.double)
    majority_scores[:, majority] = 1
    reports = {
        "action_side": classification_report(action_scores, labels),
        "state_action": classification_report(state_scores, labels),
        "time_phase": classification_report(phase_scores, labels),
        "shuffled_labels": classification_report(shuffled_scores, labels),
        "fixed_train_majority": classification_report(majority_scores, labels),
    }
    score_map = {
        "action_side": action_scores,
        "state_action": state_scores,
        "time_phase": phase_scores,
        "shuffled_labels": shuffled_scores,
        "fixed_train_majority": majority_scores,
    }
    return {
        "motion_threshold_m": motion_threshold_m,
        "object_indices": object_indices,
        "label_meaning": {"0": "no_motion"}
        | {
            str(rank + 1): train[0]["row"]["object_keys"][index]
            for rank, index in enumerate(object_indices)
        },
        **reports,
        "layout_comparison": classification_layout_comparison(
            score_map, labels, [record["case_id"] for record in evaluation]
        ),
        "cross_validation": {
            "action_side": action_cv,
            "state_action": state_cv,
            "time_phase": phase_cv,
            "shuffled_labels": shuffled_cv,
        },
    }


def assert_disjoint_records(train: list[dict], evaluation: list[dict]) -> dict:
    """Reject case or reset reuse across the fit and evaluation layouts."""
    train_cases = {record["case_id"] for record in train}
    evaluation_cases = {record["case_id"] for record in evaluation}
    if overlap := train_cases & evaluation_cases:
        raise ValueError(f"Training and evaluation layouts overlap: {sorted(overlap)}")
    train_resets = {record["row"].get("reset_hash") for record in train}
    evaluation_resets = {record["row"].get("reset_hash") for record in evaluation}
    train_resets.discard(None)
    evaluation_resets.discard(None)
    if overlap := train_resets & evaluation_resets:
        raise ValueError(
            f"Training and evaluation reset hashes overlap: {sorted(overlap)}"
        )
    return {
        "train_cases": sorted(train_cases),
        "evaluation_cases": sorted(evaluation_cases),
        "train_reset_hashes": sorted(train_resets),
        "evaluation_reset_hashes": sorted(evaluation_resets),
    }


def load_pair_groups(
    dataset_path: Path, pairs: list[dict], expected_split: str
) -> tuple[list[dict], list[dict]]:
    """Load paired groups and explicitly exclude action-clipped interventions."""
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=True)
    rows = {(row["case_id"], row["start_tick"]): row for row in dataset["rows"]}
    records, audit = [], []
    for pair in pairs:
        measurements_path, result_path = (
            Path(pair["measurements"]),
            Path(pair["result"]),
        )
        measurements = torch.load(
            measurements_path, map_location="cpu", weights_only=True
        )
        metadata = json.loads(result_path.read_text())
        if (
            metadata.get("state") != "complete"
            or not metadata.get("no_training")
            or metadata.get("paired_states") != len(measurements)
        ):
            raise ValueError("Transport probe requires complete measurement-only pairs")
        axis, offset = metadata["axis"], metadata["offset"]
        retained = []
        for measurement in measurements:
            key = (measurement["case"]["id"], measurement["start_tick"])
            if key not in rows:
                raise ValueError("Paired-action state is absent from transition data")
            reference = rows[key]["ref"][:, axis].float()
            positive = (reference + offset).clamp(-1, 1)
            negative = (reference - offset).clamp(-1, 1)
            if torch.allclose(
                positive - negative,
                torch.full_like(reference, 2 * offset),
                atol=1e-6,
                rtol=0,
            ):
                retained.append(measurement)
        loaded = paired_records(
            dataset,
            retained,
            axis,
            offset,
            expected_split=expected_split,
        )
        records.extend(loaded)
        audit.append(
            {
                "measurements": str(measurements_path),
                "reported_pairs": len(measurements),
                "retained_pairs": len(retained),
                "action_clipped_pairs": len(measurements) - len(retained),
            }
        )
    if not records:
        raise ValueError("Transport probe requires at least one paired record")
    return records, audit


def evaluate_records(
    train: list[dict],
    evaluation: list[dict],
    probes: Path,
    methods: list[str],
    object_ranks: list[int],
    motion_threshold_m: float = 0.002,
) -> dict:
    """Fit matched transport arms and evaluate them on disjoint layouts."""
    if motion_threshold_m <= 0:
        raise ValueError("Transport motion threshold must be positive")
    provenance = assert_disjoint_records(train, evaluation)
    cases = [record["case_id"] for record in train]
    x_response, test_response = response_features(train), response_features(evaluation)
    measured_train_response, measured_test_response = (
        response_targets(train),
        response_targets(evaluation),
    )
    predicted_response, response_cv = crossfit_ridge(
        x_response, measured_train_response, cases, test_response
    )
    train_predicted_response = out_of_layout_predictions(
        x_response,
        measured_train_response,
        cases,
        response_cv["ridge"],
    )
    results = []
    object_indices = [dominant_object(train, rank) for rank in object_ranks]
    for method in methods:
        for seed in (31, 47, 59):
            readout = load_audited_readout(probes / f"{method}-seed{seed}/readout.pt")
            with torch.no_grad():
                token_x = torch.cat(
                    [_token(readout, record) for record in train]
                ).double()
                test_token_x = torch.cat(
                    [_token(readout, record) for record in evaluation]
                ).double()
            x, test_x = token_x, test_token_x
            x = torch.cat((x, x_response), dim=1)
            test_x = torch.cat((test_x, test_response), dim=1)
            phase_x = phase_expand(x, train)
            test_phase_x = phase_expand(test_x, evaluation)
            shuffled_phase_x = phase_expand(x, train, shuffle_seed=seed)
            shuffled_test_phase_x = phase_expand(
                test_x, evaluation, shuffle_seed=seed + 1000
            )
            phase_response = phase_expand(x_response, train)
            test_phase_response = phase_expand(test_response, evaluation)
            active_object = (
                active_object_probe(
                    train,
                    evaluation,
                    x,
                    test_x,
                    x_response,
                    test_response,
                    cases,
                    object_indices,
                    seed,
                )
                if len(object_indices) > 1
                else None
            )
            for rank in object_ranks:
                index = object_indices[object_ranks.index(rank)]
                key = train[0]["row"]["object_keys"][index]
                if any(
                    record["row"]["object_keys"][index] != key
                    for record in train + evaluation
                ):
                    raise ValueError("Transport object identity changed")
                target = torch.stack([object_effect(record, index) for record in train])
                truth = torch.stack(
                    [object_effect(record, index) for record in evaluation]
                )
                coefficient = transport_coefficient(measured_train_response, target)
                jacobian = transport_jacobian(measured_train_response, target)
                vector = transport_vector(measured_train_response, target)
                residual = transport_residual(measured_train_response, target)
                moving = target.norm(dim=1) > motion_threshold_m
                direct, direct_cv = crossfit_ridge(x, target, cases, test_x)
                predicted_coefficient, coefficient_cv = crossfit_ridge(
                    x, coefficient, cases, test_x
                )
                phase_coefficient, phase_cv = crossfit_ridge(
                    phase_x, coefficient, cases, test_phase_x
                )
                shuffled_phase_coefficient, shuffled_phase_cv = crossfit_ridge(
                    shuffled_phase_x,
                    coefficient,
                    cases,
                    shuffled_test_phase_x,
                )
                generator = torch.Generator().manual_seed(seed)
                shuffled = coefficient[
                    torch.randperm(len(coefficient), generator=generator)
                ]
                wrong_coefficient, wrong_cv = crossfit_ridge(x, shuffled, cases, test_x)
                predicted_jacobian, jacobian_cv = crossfit_ridge(
                    x, jacobian.flatten(1), cases, test_x
                )
                state_jacobian, state_jacobian_cv = crossfit_ridge(
                    x_response, jacobian.flatten(1), cases, test_response
                )
                shuffled_jacobian = jacobian[
                    torch.randperm(len(jacobian), generator=generator)
                ]
                wrong_jacobian, wrong_jacobian_cv = crossfit_ridge(
                    x, shuffled_jacobian.flatten(1), cases, test_x
                )
                predicted_vector, vector_cv = crossfit_ridge(x, vector, cases, test_x)
                state_vector, state_vector_cv = crossfit_ridge(
                    x_response, vector, cases, test_response
                )
                shuffled_vector = vector[
                    torch.randperm(len(vector), generator=generator)
                ]
                wrong_vector, wrong_vector_cv = crossfit_ridge(
                    x, shuffled_vector, cases, test_x
                )
                predicted_residual, transport_residual_cv = crossfit_ridge(
                    x, residual, cases, test_x
                )
                state_residual, state_residual_cv = crossfit_ridge(
                    x_response, residual, cases, test_response
                )
                shuffled_residual_target = residual[
                    torch.randperm(len(residual), generator=generator)
                ]
                wrong_transport_residual, wrong_transport_residual_cv = crossfit_ridge(
                    x, shuffled_residual_target, cases, test_x
                )
                state_coefficient, state_cv = crossfit_ridge(
                    x_response, coefficient, cases, test_response
                )
                train_coefficient = out_of_layout_predictions(
                    x, coefficient, cases, coefficient_cv["ridge"]
                )
                train_residual = out_of_layout_predictions(
                    x, residual, cases, transport_residual_cv["ridge"]
                )
                residual_ratio, residual_ratio_cv = select_residual_ratio(
                    train_coefficient.clamp(0, 1) * train_predicted_response,
                    train_residual,
                    train_predicted_response,
                    target,
                )
                state_train_prediction = out_of_layout_predictions(
                    x_response, coefficient, cases, state_cv["ridge"]
                )
                residual_target = coefficient - state_train_prediction
                residual_correction, residual_cv = crossfit_ridge(
                    token_x, residual_target, cases, test_token_x
                )
                train_residual_correction = out_of_layout_predictions(
                    token_x,
                    residual_target,
                    cases,
                    residual_cv["ridge"],
                )
                incremental_ratio, incremental_ratio_cv = select_incremental_ratio(
                    state_train_prediction,
                    train_residual_correction,
                    train_predicted_response,
                    target,
                )
                shuffled_residual = residual_target[
                    torch.randperm(len(residual_target), generator=generator)
                ]
                shuffled_residual_correction, shuffled_residual_cv = crossfit_ridge(
                    token_x, shuffled_residual, cases, test_token_x
                )
                train_shuffled_residual_correction = out_of_layout_predictions(
                    token_x,
                    shuffled_residual,
                    cases,
                    shuffled_residual_cv["ridge"],
                )
                shuffled_incremental_ratio, shuffled_incremental_ratio_cv = (
                    select_incremental_ratio(
                        state_train_prediction,
                        train_shuffled_residual_correction,
                        train_predicted_response,
                        target,
                    )
                )
                phase_state_coefficient, phase_state_cv = crossfit_ridge(
                    phase_response, coefficient, cases, test_phase_response
                )
                gated_action_coefficient, gated_action_cv = gated_coefficient(
                    x, test_x, coefficient, moving, cases
                )
                gated_state_coefficient, gated_state_cv = gated_coefficient(
                    x_response, test_response, coefficient, moving, cases
                )
                gated_state_train_prediction = out_of_layout_gated_predictions(
                    x_response, coefficient, moving, cases
                )
                gated_residual_target = coefficient - gated_state_train_prediction
                gated_residual_correction, gated_residual_cv = crossfit_ridge(
                    token_x, gated_residual_target, cases, test_token_x
                )
                shuffled_gated_residual = gated_residual_target[
                    torch.randperm(len(gated_residual_target), generator=generator)
                ]
                shuffled_gated_correction, shuffled_gated_cv = crossfit_ridge(
                    token_x, shuffled_gated_residual, cases, test_token_x
                )
                shuffled_event_coefficient, shuffled_event_cv = gated_coefficient(
                    x,
                    test_x,
                    coefficient,
                    moving,
                    cases,
                    shuffle_seed=seed,
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
                    "incremental_action_coupling": (
                        state_coefficient + residual_correction
                    ).clamp(0, 1)
                    * predicted_response,
                    "adaptive_incremental_action_coupling": (
                        state_coefficient + incremental_ratio * residual_correction
                    ).clamp(0, 1)
                    * predicted_response,
                    "shuffled_incremental_coupling": (
                        state_coefficient + shuffled_residual_correction
                    ).clamp(0, 1)
                    * predicted_response,
                    "adaptive_shuffled_incremental_coupling": (
                        state_coefficient
                        + shuffled_incremental_ratio * shuffled_residual_correction
                    ).clamp(0, 1)
                    * predicted_response,
                    "gated_action_coupling": gated_action_coefficient
                    * predicted_response,
                    "gated_state_action_coupling": gated_state_coefficient
                    * predicted_response,
                    "gated_incremental_action_coupling": (
                        gated_state_coefficient + gated_residual_correction
                    ).clamp(0, 1)
                    * predicted_response,
                    "shuffled_gated_incremental_coupling": (
                        gated_state_coefficient + shuffled_gated_correction
                    ).clamp(0, 1)
                    * predicted_response,
                    "shuffled_event_coupling": shuffled_event_coefficient
                    * predicted_response,
                    "jacobian_coupling": apply_transport_jacobian(
                        predicted_jacobian, predicted_response
                    ),
                    "state_action_jacobian_coupling": apply_transport_jacobian(
                        state_jacobian, predicted_response
                    ),
                    "shuffled_jacobian_coupling": apply_transport_jacobian(
                        wrong_jacobian, predicted_response
                    ),
                    "vector_coupling": apply_transport_vector(
                        predicted_vector, predicted_response
                    ),
                    "state_action_vector_coupling": apply_transport_vector(
                        state_vector, predicted_response
                    ),
                    "shuffled_vector_coupling": apply_transport_vector(
                        wrong_vector, predicted_response
                    ),
                    "residual_coupling": apply_transport_residual(
                        coupled, predicted_residual, predicted_response
                    ),
                    "adaptive_residual_coupling": apply_transport_residual(
                        coupled,
                        predicted_residual,
                        predicted_response,
                        max_response_ratio=residual_ratio,
                    ),
                    "state_action_residual_coupling": apply_transport_residual(
                        state_coefficient.clamp(0, 1) * predicted_response,
                        state_residual,
                        predicted_response,
                    ),
                    "shuffled_transport_residual_coupling": apply_transport_residual(
                        coupled, wrong_transport_residual, predicted_response
                    ),
                    "state_action_direct": state_direct,
                    "shuffled_coupling": wrong_coefficient.clamp(0, 1)
                    * predicted_response,
                    "phase_coupling": phase_coefficient.clamp(0, 1)
                    * predicted_response,
                    "phase_state_action_coupling": phase_state_coefficient.clamp(0, 1)
                    * predicted_response,
                    "shuffled_phase_coupling": shuffled_phase_coefficient.clamp(0, 1)
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
                        "active_object_probe": active_object,
                        "results": {
                            name: score_prediction(prediction, truth)
                            for name, prediction in arms.items()
                        },
                        "direct_cv": direct_cv,
                        "coefficient_cv": coefficient_cv,
                        "shuffled_cv": wrong_cv,
                        "phase_cv": phase_cv,
                        "phase_state_action_cv": phase_state_cv,
                        "shuffled_phase_cv": shuffled_phase_cv,
                        "state_action_cv": state_cv,
                        "incremental_action_cv": residual_cv,
                        "shuffled_incremental_cv": shuffled_residual_cv,
                        "adaptive_incremental_ratio": incremental_ratio,
                        "adaptive_incremental_ratio_train_layout_cv_mse": incremental_ratio_cv,
                        "adaptive_shuffled_incremental_ratio": shuffled_incremental_ratio,
                        "adaptive_shuffled_incremental_ratio_train_layout_cv_mse": shuffled_incremental_ratio_cv,
                        "gated_action_cv": gated_action_cv,
                        "gated_state_action_cv": gated_state_cv,
                        "gated_incremental_action_cv": gated_residual_cv,
                        "shuffled_gated_incremental_cv": shuffled_gated_cv,
                        "shuffled_event_cv": shuffled_event_cv,
                        "jacobian_cv": jacobian_cv,
                        "state_action_jacobian_cv": state_jacobian_cv,
                        "shuffled_jacobian_cv": wrong_jacobian_cv,
                        "vector_cv": vector_cv,
                        "state_action_vector_cv": state_vector_cv,
                        "shuffled_vector_cv": wrong_vector_cv,
                        "residual_cv": transport_residual_cv,
                        "state_action_residual_cv": state_residual_cv,
                        "shuffled_transport_residual_cv": wrong_transport_residual_cv,
                        "adaptive_residual_ratio": residual_ratio,
                        "adaptive_residual_ratio_train_layout_cv_mse": residual_ratio_cv,
                        "state_action_direct_cv": state_direct_cv,
                        "layout_comparison": layout_comparison(
                            arms,
                            truth,
                            [record["case_id"] for record in evaluation],
                        ),
                        "phase_layout_comparison": layout_comparison(
                            arms,
                            truth,
                            [record["case_id"] for record in evaluation],
                            reference="phase_coupling",
                            comparators=(
                                "coupled",
                                "phase_state_action_coupling",
                                "shuffled_phase_coupling",
                            ),
                        ),
                        "incremental_layout_comparison": layout_comparison(
                            arms,
                            truth,
                            [record["case_id"] for record in evaluation],
                            reference="incremental_action_coupling",
                            comparators=(
                                "state_action_coupling",
                                "coupled",
                                "shuffled_incremental_coupling",
                            ),
                        ),
                        "adaptive_incremental_layout_comparison": layout_comparison(
                            arms,
                            truth,
                            [record["case_id"] for record in evaluation],
                            reference="adaptive_incremental_action_coupling",
                            comparators=(
                                "state_action_coupling",
                                "incremental_action_coupling",
                                "adaptive_shuffled_incremental_coupling",
                            ),
                        ),
                        "gated_layout_comparison": layout_comparison(
                            arms,
                            truth,
                            [record["case_id"] for record in evaluation],
                            reference="gated_action_coupling",
                            comparators=(
                                "coupled",
                                "gated_state_action_coupling",
                                "shuffled_event_coupling",
                            ),
                        ),
                        "gated_incremental_layout_comparison": layout_comparison(
                            arms,
                            truth,
                            [record["case_id"] for record in evaluation],
                            reference="gated_incremental_action_coupling",
                            comparators=(
                                "gated_state_action_coupling",
                                "gated_action_coupling",
                                "incremental_action_coupling",
                                "shuffled_gated_incremental_coupling",
                            ),
                        ),
                        "jacobian_layout_comparison": layout_comparison(
                            arms,
                            truth,
                            [record["case_id"] for record in evaluation],
                            reference="jacobian_coupling",
                            comparators=(
                                "coupled",
                                "state_action_jacobian_coupling",
                                "shuffled_jacobian_coupling",
                            ),
                        ),
                        "vector_layout_comparison": layout_comparison(
                            arms,
                            truth,
                            [record["case_id"] for record in evaluation],
                            reference="vector_coupling",
                            comparators=(
                                "coupled",
                                "state_action_vector_coupling",
                                "shuffled_vector_coupling",
                            ),
                        ),
                        "residual_layout_comparison": layout_comparison(
                            arms,
                            truth,
                            [record["case_id"] for record in evaluation],
                            reference="residual_coupling",
                            comparators=(
                                "coupled",
                                "direct",
                                "state_action_residual_coupling",
                                "shuffled_transport_residual_coupling",
                            ),
                        ),
                        "adaptive_residual_layout_comparison": layout_comparison(
                            arms,
                            truth,
                            [record["case_id"] for record in evaluation],
                            reference="adaptive_residual_coupling",
                            comparators=(
                                "coupled",
                                "residual_coupling",
                                "direct",
                            ),
                        ),
                    }
                )
    return {
        "train_pairs": len(train),
        "evaluation_pairs": len(evaluation),
        "train_layouts": len(set(cases)),
        "evaluation_layouts": len({record["case_id"] for record in evaluation}),
        "motion_threshold_m": motion_threshold_m,
        "split_audit": provenance,
        "response_cv": response_cv,
        "predicted_response": score_prediction(
            predicted_response, measured_test_response
        ),
        "results": results,
    }


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
    result = evaluate_records(
        train,
        evaluation,
        root / "probes_v1",
        ["joint", "flow_joint", "state_action_joint"],
        [0, 1],
        0.002,
    ) | {
        "state": "complete",
        "split": args.split,
        "interpretation": "exploratory offline transport prior; oracle arms use measured future quantities and are not deployable",
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    return result


def run_manifest(args: argparse.Namespace) -> dict:
    """Run a transport probe from an explicit immutable data manifest."""
    torch.set_num_threads(1)
    if args.output.exists():
        raise FileExistsError("Transport output must be a new directory")
    manifest = json.loads(args.manifest.read_text())
    required = {
        "train_dataset",
        "evaluation_dataset",
        "train_pairs",
        "evaluation_pairs",
        "probes",
        "methods",
        "object_ranks",
        "evaluation_split",
    }
    if missing := required - manifest.keys():
        raise ValueError(f"Transport manifest is missing keys: {sorted(missing)}")
    train, train_pair_audit = load_pair_groups(
        Path(manifest["train_dataset"]), manifest["train_pairs"], "train"
    )
    evaluation, evaluation_pair_audit = load_pair_groups(
        Path(manifest["evaluation_dataset"]),
        manifest["evaluation_pairs"],
        manifest["evaluation_split"],
    )
    result = evaluate_records(
        train,
        evaluation,
        Path(manifest["probes"]),
        manifest["methods"],
        manifest["object_ranks"],
        manifest.get("motion_threshold_m", 0.002),
    ) | {
        "state": "complete",
        "split": manifest["evaluation_split"],
        "manifest": str(args.manifest.resolve()),
        "pair_audit": {
            "train": train_pair_audit,
            "evaluation": evaluation_pair_audit,
        },
        "interpretation": "development-layout transport replication; oracle arms are diagnostic and not deployable",
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "confirmation"))
    args = parser.parse_args()
    if args.root is not None and args.split is None:
        parser.error("--root requires --split")
    if args.manifest is not None and args.split is not None:
        parser.error("--split is encoded in --manifest")
    result = run_manifest(args) if args.manifest is not None else run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
