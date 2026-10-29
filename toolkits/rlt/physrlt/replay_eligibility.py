# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Reject fixed replays that cannot support an independent critic audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toolkits.rlt.physrlt.control import validate_fixed_replay
from toolkits.rlt.physrlt.critic_audit import discounted_success_returns
from toolkits.rlt.pi05_baseline import write_json

MIN_SUCCESS_EPISODES = 2
MIN_FAILURE_EPISODES = 2


def episode_outcomes(replay: list[dict], episode_tick_cap: int = 520) -> dict:
    """Count complete success/failure episodes without promoting a partial tail."""
    if not replay or episode_tick_cap < 1:
        raise ValueError("Episode audit requires replay and a positive tick cap")
    episodes = []
    start = 0
    elapsed = 0
    reward = 0.0
    for index, row in enumerate(replay):
        ticks = int(row["ticks"])
        if ticks < 1:
            raise ValueError("Replay ticks must be positive")
        elapsed += ticks
        reward += float(row["reward"])
        if bool(row["terminal"]) or elapsed >= episode_tick_cap:
            episodes.append(
                {
                    "transitions": index - start + 1,
                    "ticks": elapsed,
                    "success": reward > 0,
                    "terminal": bool(row["terminal"]),
                }
            )
            start = index + 1
            elapsed = 0
            reward = 0.0
    return {
        "episodes": episodes,
        "complete_episodes": len(episodes),
        "successful_episodes": sum(row["success"] for row in episodes),
        "failed_episodes": sum(not row["success"] for row in episodes),
        "incomplete_tail_transitions": len(replay) - start,
        "incomplete_tail_ticks": elapsed,
    }


def evaluate(replay: list[dict]) -> dict:
    """Require both successful and unsuccessful return evidence."""
    summary = validate_fixed_replay(replay)
    returns = discounted_success_returns(replay)
    positive_returns = sum(value > 0 for value in returns)
    zero_returns = sum(value == 0 for value in returns)
    reasons = []
    if summary["positive_transitions"] == 0:
        reasons.append("no_positive_reward_transition")
    if positive_returns == 0:
        reasons.append("no_positive_return_transition")
    if zero_returns == 0:
        reasons.append("no_zero_return_transition")
    episode_summary = episode_outcomes(replay)
    robust_reasons = list(reasons)
    if episode_summary["successful_episodes"] < MIN_SUCCESS_EPISODES:
        robust_reasons.append("fewer_than_two_successful_episodes")
    if episode_summary["failed_episodes"] < MIN_FAILURE_EPISODES:
        robust_reasons.append("fewer_than_two_failed_episodes")
    return {
        "state": "complete",
        **summary,
        **episode_summary,
        "positive_return_transitions": positive_returns,
        "zero_return_transitions": zero_returns,
        "eligible_for_independent_critic_gate": not reasons,
        "rejection_reasons": reasons,
        "eligible_for_robust_independent_critic_gate": not robust_reasons,
        "robust_rejection_reasons": robust_reasons,
    }


def evaluate_pair(train_replay: list[dict], evaluation_replay: list[dict]) -> dict:
    """Require two eligible replays with distinct executed trajectories."""
    train = evaluate(train_replay)
    evaluation = evaluate(evaluation_replay)
    reasons = []
    reasons.extend(f"train:{reason}" for reason in train["rejection_reasons"])
    reasons.extend(f"evaluation:{reason}" for reason in evaluation["rejection_reasons"])
    distinct_traces = train["trace_sha256"] != evaluation["trace_sha256"]
    if not distinct_traces:
        reasons.append("train_and_evaluation_share_execution_trace")
    robust_reasons = [
        *(f"train:{reason}" for reason in train["robust_rejection_reasons"]),
        *(f"evaluation:{reason}" for reason in evaluation["robust_rejection_reasons"]),
    ]
    if not distinct_traces:
        robust_reasons.append("train_and_evaluation_share_execution_trace")
    return {
        "state": "complete",
        "train": train,
        "evaluation": evaluation,
        "distinct_execution_traces": distinct_traces,
        "eligible_for_independent_critic_gate": not reasons,
        "rejection_reasons": reasons,
        "eligible_for_robust_independent_critic_gate": not robust_reasons,
        "robust_rejection_reasons": robust_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--train-replay", type=Path)
    parser.add_argument("--evaluation-replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    single = args.replay is not None
    pair = args.train_replay is not None or args.evaluation_replay is not None
    if single == pair:
        parser.error(
            "provide either --replay or both --train-replay and --evaluation-replay"
        )
    if pair and (args.train_replay is None or args.evaluation_replay is None):
        parser.error("paired audit requires both replay paths")
    if single:
        replay = torch.load(args.replay, map_location="cpu", weights_only=True)
        result = evaluate(replay)
        result["replay"] = str(args.replay.resolve())
    else:
        train_replay = torch.load(
            args.train_replay, map_location="cpu", weights_only=True
        )
        evaluation_replay = torch.load(
            args.evaluation_replay, map_location="cpu", weights_only=True
        )
        result = evaluate_pair(train_replay, evaluation_replay)
        result["train_replay"] = str(args.train_replay.resolve())
        result["evaluation_replay"] = str(args.evaluation_replay.resolve())
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["eligible_for_independent_critic_gate"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
