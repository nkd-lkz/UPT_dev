# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Audit sparse-reward critics on immutable held-out replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from toolkits.rlt.libero_acceptance_learning import AcceptanceLearner
from toolkits.rlt.physrlt.control import FrozenReadout, merge_replay_views
from toolkits.rlt.pi05_baseline import configure_source


def discounted_success_returns(
    replay: list[dict], gamma: float = 0.99, episode_tick_cap: int = 520
) -> list[float]:
    """Recover Monte Carlo success returns without crossing episode boundaries."""
    if not 0 < gamma <= 1 or episode_tick_cap < 1:
        raise ValueError("Require a valid discount and positive episode tick cap")
    returns = [0.0] * len(replay)
    start = 0
    elapsed = 0
    for index, row in enumerate(replay):
        ticks = int(row["ticks"])
        if ticks < 1:
            raise ValueError("Replay ticks must be positive")
        elapsed += ticks
        boundary = bool(row["terminal"]) or elapsed >= episode_tick_cap
        if not boundary:
            continue
        if float(row["reward"]) > 0:
            value = 0.0
            for cursor in range(index, start - 1, -1):
                transition = replay[cursor]
                value = (
                    float(transition["reward"])
                    + gamma ** int(transition["ticks"]) * value
                )
                returns[cursor] = value
        start = index + 1
        elapsed = 0
    return returns


def binary_auc(scores: list[float], labels: list[bool]) -> float | None:
    """Return tie-aware pairwise AUC, or None when one class is absent."""
    if len(scores) != len(labels):
        raise ValueError("Scores and labels must have equal length")
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _parse_evaluations(specs: list[str]) -> dict[tuple[str, str], Path]:
    evaluations = {}
    for spec in specs:
        key, separator, raw_path = spec.partition("=")
        label, method_separator, method = key.partition(":")
        if (
            not separator
            or not method_separator
            or not label
            or not method
            or not raw_path
            or (label, method) in evaluations
        ):
            raise ValueError("Evaluations require unique LABEL:METHOD=PATH entries")
        evaluations[(label, method)] = Path(raw_path).resolve()
    return evaluations


def _learner(run: Path, method: str, device: str) -> AcceptanceLearner:
    checkpoint = torch.load(
        run / f"{method}-online.pt", map_location="cpu", weights_only=True
    )
    metadata = json.loads((run / f"{method}-readout.json").read_text())
    conditioning = checkpoint.get("critic_conditioning", "latent")
    readout = FrozenReadout(
        Path(metadata["checkpoint"]),
        consequence=(
            Path(metadata["consequence_checkpoint"])
            if metadata.get("consequence_checkpoint")
            else None
        ),
    )
    conditioner = None
    if conditioning != "latent":
        if not conditioning.startswith("candidate_"):
            raise ValueError(f"Unsupported critic conditioning: {conditioning}")
        conditioner = readout.candidate_conditioner(
            conditioning.removeprefix("candidate_")
        )
    learner = AcceptanceLearner(
        device,
        "q_bc",
        ref_dropout=0,
        bottleneck_dim=128,
        chunk_len=10,
        gripper_threshold=0,
        critic_conditioner=conditioner,
        critic_conditioning=conditioning,
        correction_limit=checkpoint.get("correction_limit"),
        correction_axes=(
            tuple(checkpoint["correction_axes"])
            if checkpoint.get("correction_axes") is not None
            else None
        ),
        residual_output=checkpoint.get("residual_output", False),
        critic_bottleneck_dim=checkpoint.get("critic_bottleneck_dim"),
        candidate_selection_count=checkpoint.get("candidate_selection_count", 0),
        candidate_selection_margin=checkpoint.get("candidate_selection_margin", 0.0),
        candidate_uncertainty_penalty=checkpoint.get(
            "candidate_uncertainty_penalty", 0.0
        ),
    )
    learner.load_state_dict(checkpoint)
    learner.actor.eval()
    learner.critic.eval()
    return learner


@torch.no_grad()
def audit_replay(
    learner: AcceptanceLearner,
    replay: list[dict],
    *,
    gamma: float = 0.99,
    episode_tick_cap: int = 520,
) -> dict:
    """Measure return ranking, calibration and policy optimism on one replay."""
    if not replay:
        raise ValueError("Critic audit requires non-empty replay")
    returns = discounted_success_returns(replay, gamma, episode_tick_cap)
    q1_values, q2_values, policy_advantages = [], [], []
    base_squares, conditioned_squares, action_signal_squares = 0.0, 0.0, 0.0
    base_elements, conditioned_elements = 0, 0
    for start in range(0, len(replay), 128):
        rows = replay[start : start + 128]
        batch = {
            name: torch.stack([row[name] for row in rows]).to(learner.device)
            for name in ("z", "ref", "prop", "action")
        }
        if "critic_z" in rows[0]:
            batch["critic_z"] = torch.stack([row["critic_z"] for row in rows]).to(
                learner.device
            )
        token = learner._critic_token(
            batch["z"], batch["action"], batch["ref"], batch.get("critic_z")
        )
        reference_token = learner._critic_token(
            batch["z"], batch["ref"], batch["ref"], batch.get("critic_z")
        )
        base_squares += float(batch["z"].square().sum())
        conditioned_squares += float(token.square().sum())
        action_signal_squares += float((token - reference_token).square().sum())
        base_elements += batch["z"].numel()
        conditioned_elements += token.numel()
        q1, q2 = learner.critic(token, batch["action"], batch["prop"])
        raw_action, _ = learner.actor(
            batch["z"], batch["ref"], batch["prop"], deterministic=True
        )
        policy_action = learner._policy_action(raw_action, batch["ref"]).clamp(-1, 1)
        policy_q = learner.critic.q1_forward(
            learner._critic_token(
                batch["z"], policy_action, batch["ref"], batch.get("critic_z")
            ),
            policy_action,
            batch["prop"],
        )
        reference_q = learner.critic.q1_forward(
            learner._critic_token(
                batch["z"], batch["ref"], batch["ref"], batch.get("critic_z")
            ),
            batch["ref"],
            batch["prop"],
        )
        q1_values.extend(q1.cpu().tolist())
        q2_values.extend(q2.cpu().tolist())
        policy_advantages.extend((policy_q - reference_q).cpu().tolist())
    labels = [value > 0 for value in returns]
    positive_q = [value for value, label in zip(q1_values, labels) if label]
    negative_q = [value for value, label in zip(q1_values, labels) if not label]
    return {
        "transitions": len(replay),
        "positive_return_transitions": sum(labels),
        "return_ranking_auc": binary_auc(q1_values, labels),
        "positive_q1_mean": (sum(positive_q) / len(positive_q) if positive_q else None),
        "zero_return_q1_mean": sum(negative_q) / len(negative_q),
        "mc_return_mse": sum(
            (prediction - target) ** 2 for prediction, target in zip(q1_values, returns)
        )
        / len(returns),
        "q_disagreement_mean": sum(
            abs(first - second) for first, second in zip(q1_values, q2_values)
        )
        / len(q1_values),
        "policy_minus_reference_q1_mean": sum(policy_advantages)
        / len(policy_advantages),
        "base_token_rms": (base_squares / base_elements) ** 0.5,
        "conditioned_token_rms": (conditioned_squares / conditioned_elements) ** 0.5,
        "action_conditioning_signal_rms": (action_signal_squares / conditioned_elements)
        ** 0.5,
    }


def run(args: argparse.Namespace) -> list[dict]:
    """Load immutable checkpoints and write one row per held-out replay."""
    configure_source(args.source)
    evaluations = _parse_evaluations(args.evaluations)
    physical_evaluations = _parse_evaluations(getattr(args, "physical_evaluations", []))
    if not set(physical_evaluations).issubset(evaluations):
        raise ValueError("Physical evaluations require a matching base evaluation")
    methods = sorted({method for _, method in evaluations})
    rows = []
    for method in methods:
        learner = _learner(args.trained_run, method, args.device)
        for (label, evaluated_method), replay_path in evaluations.items():
            if evaluated_method != method:
                continue
            replay = torch.load(replay_path, map_location="cpu", weights_only=True)
            physical_path = physical_evaluations.get((label, method))
            if physical_path is not None:
                replay = merge_replay_views(
                    replay,
                    torch.load(physical_path, map_location="cpu", weights_only=True),
                )
            rows.append(
                {
                    "trained_run": str(args.trained_run.resolve()),
                    "method": method,
                    "evaluation": label,
                    "replay": str(replay_path),
                    **audit_replay(
                        learner,
                        replay,
                        gamma=args.gamma,
                        episode_tick_cap=args.episode_tick_cap,
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parser() -> argparse.ArgumentParser:
    """Build the critic-audit command line."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, required=True)
    result.add_argument("--trained-run", type=Path, required=True)
    result.add_argument("--evaluations", nargs="+", required=True)
    result.add_argument(
        "--physical-evaluations",
        nargs="*",
        default=[],
        help="Attach matched Physical Token replay views to base evaluations",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument("--gamma", type=float, default=0.99)
    result.add_argument("--episode-tick-cap", type=int, default=520)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
