# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Test whether physical consequences add cross-trace return information."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from toolkits.rlt.physrlt.consequence_critic_gate import (
    audit_replay_set,
    load_replay,
)
from toolkits.rlt.physrlt.control import FrozenReadout
from toolkits.rlt.physrlt.critic_audit import (
    binary_auc,
    discounted_success_returns,
    ranking_concordance,
)
from toolkits.rlt.pi05_baseline import configure_source, write_json

RIDGES = (0.1, 1.0, 10.0, 100.0, 1000.0)


def episode_ids(replay: list[dict], episode_tick_cap: int = 520) -> torch.Tensor:
    """Assign complete and capped replay segments to distinct episode groups."""
    if not replay or episode_tick_cap < 1:
        raise ValueError("Episode grouping requires replay and a positive tick cap")
    groups = []
    group = 0
    elapsed = 0
    for row in replay:
        ticks = int(row["ticks"])
        if ticks < 1:
            raise ValueError("Replay ticks must be positive")
        elapsed += ticks
        groups.append(group)
        if bool(row["terminal"]) or elapsed >= episode_tick_cap:
            group += 1
            elapsed = 0
    return torch.tensor(groups, dtype=torch.long)


def _fit_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float) -> dict:
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2 or ridge <= 0:
        raise ValueError("Ridge fit requires a nontrivial matrix, targets and penalty")
    x = x.double()
    y = y.double()
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    normalized = (x - mean) / scale
    target_mean = y.mean()
    centred = y - target_mean
    gram = normalized @ normalized.T
    dual = torch.linalg.solve(gram + ridge * torch.eye(len(x), dtype=x.dtype), centred)
    return {
        "mean": mean,
        "scale": scale,
        "target_mean": target_mean,
        "weight": normalized.T @ dual,
    }


def _predict(model: dict, x: torch.Tensor) -> torch.Tensor:
    normalized = (x.double() - model["mean"]) / model["scale"]
    return normalized @ model["weight"] + model["target_mean"]


def select_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    groups: torch.Tensor,
    ridges: tuple[float, ...] = RIDGES,
) -> tuple[float, dict[str, float]]:
    """Choose regularization by leave-one-training-episode-out MSE."""
    if len(x) != len(y) or groups.shape != (len(x),):
        raise ValueError("Ridge selection inputs must align")
    unique = groups.unique().tolist()
    if len(unique) < 2:
        raise ValueError("Ridge selection requires at least two replay episodes")
    losses = {}
    for ridge in ridges:
        squared_errors = []
        for group in unique:
            heldout = groups == group
            if int((~heldout).sum()) < 2:
                raise ValueError("Ridge fold lacks training transitions")
            prediction = _predict(
                _fit_ridge(x[~heldout], y[~heldout], ridge), x[heldout]
            )
            squared_errors.extend((prediction - y[heldout]).square().tolist())
        losses[str(ridge)] = sum(squared_errors) / len(squared_errors)
    selected = min(ridges, key=lambda ridge: losses[str(ridge)])
    return selected, losses


@torch.no_grad()
def replay_features(
    readout: FrozenReadout, replay: list[dict], component: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return semantic state and candidate-relative physical consequences."""
    semantic = torch.stack([row["z"] for row in replay]).float()
    physical_token = torch.stack(
        [row.get("critic_z", row["z"]) for row in replay]
    ).float()
    action = torch.stack([row["action"] for row in replay]).float()
    reference = torch.stack([row["ref"] for row in replay]).float()
    prop = torch.stack([row["prop"] for row in replay]).float()
    conditioned = readout.candidate_conditioner("relative_concat", component=component)(
        semantic, action, reference, physical_token
    )
    consequence = conditioned[:, semantic.shape[-1] :]
    return torch.cat((semantic, prop), dim=-1), consequence


def _design(
    base: torch.Tensor, physical: torch.Tensor, family: str, arm: str
) -> torch.Tensor:
    if arm not in {"correct", "reversed", "zero"}:
        raise ValueError("Unknown physical intervention arm")
    signed = {
        "correct": physical,
        "reversed": -physical,
        "zero": torch.zeros_like(physical),
    }[arm]
    if family == "semantic":
        return base
    if family == "physical":
        return signed
    if family == "joint":
        return torch.cat((base, signed), dim=-1)
    raise ValueError("Unknown return-probe feature family")


def evaluate(args: argparse.Namespace) -> dict:
    """Fit on one replay and evaluate frozen linear maps on another trace."""
    if args.output.exists():
        raise FileExistsError("Return-probe output must be new")
    configure_source(args.source)
    train = load_replay(args.train_base_replay, args.train_physical_replay)
    evaluation = load_replay(
        args.evaluation_base_replay, args.evaluation_physical_replay
    )
    trace_sha256, replay_summaries = audit_replay_set(train, {"evaluation": evaluation})
    readout = FrozenReadout(args.base_readout, consequence=args.consequence_readout)
    train_base, train_physical = replay_features(
        readout, train, args.consequence_component
    )
    eval_base, eval_physical = replay_features(
        readout, evaluation, args.consequence_component
    )
    train_returns = torch.tensor(discounted_success_returns(train), dtype=torch.float64)
    eval_returns = torch.tensor(
        discounted_success_returns(evaluation), dtype=torch.float64
    )
    groups = episode_ids(train)
    rows = []
    fit = {}
    for family in ("semantic", "physical", "joint"):
        train_x = _design(train_base, train_physical, family, "correct")
        ridge, cv_mse = select_ridge(train_x, train_returns, groups)
        model = _fit_ridge(train_x, train_returns, ridge)
        fit[family] = {
            "ridge": ridge,
            "train_episode_cv_mse": cv_mse,
            "features": train_x.shape[-1],
        }
        arms = ("correct",) if family == "semantic" else ("correct", "reversed", "zero")
        for arm in arms:
            scores = _predict(
                model, _design(eval_base, eval_physical, family, arm)
            ).tolist()
            rows.append(
                {
                    "family": family,
                    "arm": arm,
                    "return_ranking_auc": binary_auc(
                        scores, [value > 0 for value in eval_returns.tolist()]
                    ),
                    "return_order_concordance": ranking_concordance(
                        scores, eval_returns.tolist()
                    ),
                    "return_mse": float(
                        torch.tensor(scores).sub(eval_returns).square().mean()
                    ),
                }
            )
    by_key = {(row["family"], row["arm"]): row for row in rows}
    semantic = by_key[("semantic", "correct")]
    joint = by_key[("joint", "correct")]
    physical_causal = all(
        joint[metric]
        > max(
            by_key[("joint", "reversed")][metric],
            by_key[("joint", "zero")][metric],
            semantic[metric],
        )
        for metric in ("return_ranking_auc", "return_order_concordance")
    )
    return {
        "state": "complete",
        "role": "cross-trace linear mechanism probe, not closed-loop evidence",
        "selection": "ridge selected by train replay episode CV only",
        "consequence_component": args.consequence_component,
        "trace_sha256": trace_sha256,
        "replay_summaries": replay_summaries,
        "fit": fit,
        "rows": rows,
        "directional_physical_increment_supported": physical_causal,
        "decision": (
            "requires_reverse_trace_replication"
            if physical_causal
            else "revise_physical_value_representation"
        ),
    }


def write_report(output: Path, result: dict) -> None:
    """Write machine-readable and reviewable return-probe artifacts."""
    output.mkdir(parents=True)
    write_json(output / "result.json", result)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    lines = [
        "# Cross-trace consequence return probe",
        "",
        result["role"],
        "",
        "| Features | Arm | Return AUC | Concordance | Return MSE |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['family']} | {row['arm']} | "
            f"{float(row['return_ranking_auc']):.6f} | "
            f"{float(row['return_order_concordance']):.6f} | "
            f"{float(row['return_mse']):.6f} |"
        )
    lines.extend(("", f"Decision: ``{result['decision']}``"))
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-base-replay", type=Path, required=True)
    parser.add_argument("--train-physical-replay", type=Path, required=True)
    parser.add_argument("--evaluation-base-replay", type=Path, required=True)
    parser.add_argument("--evaluation-physical-replay", type=Path, required=True)
    parser.add_argument("--base-readout", type=Path, required=True)
    parser.add_argument("--consequence-readout", type=Path, required=True)
    parser.add_argument(
        "--consequence-component",
        choices=("joint", "response", "geometry"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args)
    write_report(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
