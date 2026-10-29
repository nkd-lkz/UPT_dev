# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Audit consequence critics with held-out complete replay episodes."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from toolkits.rlt.libero_acceptance_learning import AcceptanceLearner
from toolkits.rlt.physrlt.conditioning import SignedCandidateConsequence
from toolkits.rlt.physrlt.consequence_critic_gate import load_replay
from toolkits.rlt.physrlt.control import (
    FrozenReadout,
    replay_trace_sha256,
    validate_fixed_replay,
)
from toolkits.rlt.physrlt.critic_audit import audit_replay
from toolkits.rlt.pi05_baseline import configure_source, write_json


def complete_positive_episodes(replay: list[dict]) -> list[list[dict]]:
    """Return terminal-delimited successful episodes and ignore an open tail."""
    episodes = []
    start = 0
    for index, row in enumerate(replay):
        if not bool(row["terminal"]):
            continue
        episode = replay[start : index + 1]
        start = index + 1
        if any(float(transition["reward"]) > 0 for transition in episode):
            episodes.append(episode)
    return episodes


def leave_one_episode_out_folds(replay: list[dict]) -> list[dict]:
    """Build folds whose train and evaluation executed traces are disjoint."""
    episodes = complete_positive_episodes(replay)
    if len(episodes) < 3:
        raise ValueError("Require at least three complete successful episodes")
    hashes = [replay_trace_sha256(episode) for episode in episodes]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Complete successful episode traces must be unique")
    folds = []
    for heldout_index, evaluation in enumerate(episodes):
        train = [
            row
            for episode_index, episode in enumerate(episodes)
            if episode_index != heldout_index
            for row in episode
        ]
        train_summary = validate_fixed_replay(train)
        evaluation_summary = validate_fixed_replay(evaluation)
        if train_summary["trace_sha256"] == evaluation_summary["trace_sha256"]:
            raise ValueError("Train and held-out episode traces must differ")
        if (
            train_summary["positive_transitions"] == 0
            or evaluation_summary["positive_transitions"] == 0
        ):
            raise ValueError("Every fold needs positive train and evaluation evidence")
        folds.append(
            {
                "fold": heldout_index,
                "train": train,
                "evaluation": evaluation,
                "train_summary": train_summary,
                "evaluation_summary": evaluation_summary,
            }
        )
    return folds


def _summaries(rows: list[dict]) -> tuple[list[dict], bool]:
    summaries = []
    strict_fold_passes = []
    folds = sorted({int(row["fold"]) for row in rows})
    for fold in folds:
        means = {}
        for arm in ("correct", "reversed", "zero"):
            values = [
                float(row["return_order_concordance"])
                for row in rows
                if row["fold"] == fold
                and row["arm"] == arm
                and row["return_order_concordance"] is not None
            ]
            means[arm] = sum(values) / len(values)
        passes = means["correct"] > max(means["reversed"], means["zero"])
        strict_fold_passes.append(passes)
        summaries.append(
            {"fold": fold, "mean_concordance": means, "correct_best": passes}
        )
    aggregate = {}
    for arm in ("correct", "reversed", "zero"):
        values = [
            float(row["return_order_concordance"])
            for row in rows
            if row["arm"] == arm and row["return_order_concordance"] is not None
        ]
        aggregate[arm] = sum(values) / len(values)
    aggregate_pass = aggregate["correct"] > max(
        aggregate["reversed"], aggregate["zero"]
    )
    summaries.append(
        {
            "fold": "aggregate",
            "mean_concordance": aggregate,
            "correct_best": aggregate_pass,
        }
    )
    return summaries, aggregate_pass and all(strict_fold_passes)


def run(args: argparse.Namespace) -> dict:
    """Fit critic arms on successful episodes and audit the held-out episode."""
    if args.output.exists() or args.steps < 1 or not args.seeds:
        raise ValueError("Require a new output, positive steps and optimizer seeds")
    configure_source(args.source)
    replay = load_replay(args.base_replay, args.physical_replay)
    folds = leave_one_episode_out_folds(replay)
    readout = FrozenReadout(args.base_readout, consequence=args.consequence_readout)
    rows = []
    for fold in folds:
        for seed in args.seeds:
            for arm, sign in (("correct", 1), ("reversed", -1), ("zero", 0)):
                random.seed(seed)
                torch.manual_seed(seed)
                conditioner = SignedCandidateConsequence(
                    readout.candidate_conditioner(
                        "relative_concat", component=args.consequence_component
                    ),
                    sign,
                )
                learner = AcceptanceLearner(
                    "cpu",
                    "q_bc",
                    ref_dropout=0,
                    bottleneck_dim=128,
                    chunk_len=10,
                    gripper_threshold=0,
                    critic_conditioner=conditioner,
                    critic_conditioning="candidate_relative_concat",
                    critic_bottleneck_dim=256,
                    correction_limit=0.1,
                    correction_axes=(0,),
                    residual_output=True,
                    critic_consequence_dueling=True,
                )
                trace = []
                random.seed(seed)
                torch.manual_seed(seed)
                for step in range(args.steps):
                    metrics = learner.update(fold["train"])
                    if step in {0, args.steps - 1}:
                        trace.append({"step": step + 1, **metrics})
                rows.append(
                    {
                        "fold": fold["fold"],
                        "optimizer_seed": seed,
                        "arm": arm,
                        "train_steps": args.steps,
                        "trace": trace,
                        **audit_replay(learner, fold["evaluation"]),
                    }
                )
    summaries, passes_strict_screen = _summaries(rows)
    result = {
        "state": "complete",
        "protocol": (
            "leave one complete successful episode trace out; fit on the other "
            "complete successful episodes from the same collection run; only the "
            "physical consequence sign changes"
        ),
        "limitation": (
            "Held-out episode traces are disjoint but share one collection run; "
            "this screen does not replace an independently collected replay gate."
        ),
        "base_replay": str(args.base_replay),
        "physical_replay": str(args.physical_replay),
        "source_trace_sha256": replay_trace_sha256(replay),
        "complete_successful_episodes": len(folds),
        "ignored_open_tail_transitions": len(replay)
        - sum(len(fold["evaluation"]) for fold in folds),
        "folds": [
            {
                "fold": fold["fold"],
                "train_summary": fold["train_summary"],
                "evaluation_summary": fold["evaluation_summary"],
            }
            for fold in folds
        ],
        "base_readout": str(args.base_readout),
        "consequence_readout": str(args.consequence_readout),
        "consequence_component": args.consequence_component,
        "seeds": args.seeds,
        "strict_screen": (
            "correct mean return-order concordance must exceed reversed and zero "
            "in every held-out "
            "episode fold and in the aggregate"
        ),
        "passes_strict_screen": passes_strict_screen,
        "summaries": summaries,
        "rows": rows,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-replay", type=Path, required=True)
    parser.add_argument("--physical-replay", type=Path, required=True)
    parser.add_argument("--base-readout", type=Path, required=True)
    parser.add_argument("--consequence-readout", type=Path, required=True)
    parser.add_argument(
        "--consequence-component",
        choices=("joint", "response", "geometry"),
        default="joint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=105)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 47, 59])
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
