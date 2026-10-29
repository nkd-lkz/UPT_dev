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
    return {
        "state": "complete",
        **summary,
        "positive_return_transitions": positive_returns,
        "zero_return_transitions": zero_returns,
        "eligible_for_independent_critic_gate": not reasons,
        "rejection_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = torch.load(args.replay, map_location="cpu", weights_only=True)
    result = evaluate(replay)
    result["replay"] = str(args.replay.resolve())
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["eligible_for_independent_critic_gate"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
