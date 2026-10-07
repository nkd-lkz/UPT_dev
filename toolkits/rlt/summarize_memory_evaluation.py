# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate and summarize the four-arm frozen memory evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(status: dict) -> dict:
    """Require complete paired episodes and return a summary without local paths."""
    arms = {}
    for job in status["jobs"]:
        if job["exit_code"] != 0:
            raise ValueError("Every evaluation queue must finish successfully")
        for run in job["runs"]:
            if run["label"] == "smoke":
                continue
            seed, variant = run["label"].split("_", 1)
            audit = run["audit"]
            episodes = run["episodes"]
            if run["exit_code"] != 0 or not audit["weights_unchanged"]:
                raise ValueError("Run failed or model weights changed")
            if len(episodes) != 16 or {e["lane"] for e in episodes} != set(range(16)):
                raise ValueError("Expected exactly one episode per lane")
            if any(e["seed"] != int(seed[4:]) for e in episodes):
                raise ValueError("Episode seed does not match its run")
            if any(e["success_once"] not in (0, 1) for e in episodes):
                raise ValueError("Expected binary episode outcomes")
            if variant == "reference" and audit["actor_slots"] != 0:
                raise ValueError("Reference arm routed learned actions")
            key = f"{job['reader']}_{variant}"
            arms.setdefault(key, []).append(
                {"seed": int(seed[4:]), "audit": audit, "episodes": episodes}
            )
    expected = {
        "zero_native",
        "zero_reference",
        "response_native",
        "response_zero_context",
    }
    if set(arms) != expected:
        raise ValueError("Expected all four evaluation arms")
    results = {}
    reference_hashes = None
    for key, runs in arms.items():
        runs.sort(key=lambda run: run["seed"])
        if [r["seed"] for r in runs] != [4001, 4002, 4003, 4004]:
            raise ValueError("Expected four distinct evaluation seeds")
        hashes = [r["audit"]["initial_observation_sha256"] for r in runs]
        if reference_hashes is not None and hashes != reference_hashes:
            raise ValueError("Initial observations differ across paired arms")
        reference_hashes = hashes
        episodes = [e for r in runs for e in r["episodes"]]
        counter = sum(r["audit"]["action_difference_count"] for r in runs)
        difference = sum(r["audit"]["action_difference_sum"] for r in runs)
        results[key] = {
            "successes": int(sum(e["success_once"] for e in episodes)),
            "episodes": len(episodes),
            "gate_entered_episodes": int(
                sum(e["entered_actor_phase_once"] for e in episodes)
            ),
            "mean_episode_ticks": sum(e["episode_len"] for e in episodes)
            / len(episodes),
            "actor_routed_slots": sum(r["audit"]["actor_slots"] for r in runs),
            "scheduled_slots": sum(r["audit"]["slots"] for r in runs),
            "mean_absolute_action_change_when_history_masked": difference / counter,
            "runs": runs,
        }
    paired = {}
    for left, right in (
        ("response_native", "zero_native"),
        ("response_native", "response_zero_context"),
        ("zero_native", "zero_reference"),
        ("response_native", "zero_reference"),
    ):
        maps = [
            {
                (e["seed"], e["lane"]): int(e["success_once"])
                for r in arms[key]
                for e in r["episodes"]
            }
            for key in (left, right)
        ]
        outcomes = Counter((maps[0][k], maps[1][k]) for k in maps[0])
        paired[f"{left}_vs_{right}"] = {
            "both_success": outcomes[(1, 1)],
            "left_only_success": outcomes[(1, 0)],
            "right_only_success": outcomes[(0, 1)],
            "both_fail": outcomes[(0, 0)],
        }
    return {
        "as_of": status["checked_at"],
        "campaign": status["campaign"],
        "evaluation_code_revision": status["code_revision"],
        "checkpoint_iteration": 275,
        "episode_limit_ticks": 500,
        "all_initial_observations_matched": True,
        "all_weights_unchanged": True,
        "arms": results,
        "paired_outcomes": paired,
        "limitations": [
            "One training seed; four evaluation seeds are not training replications.",
            "Evaluation seeds do not establish unseen physical conditions.",
            "Context removal changes the input distribution of a trained head.",
            "Scheduled slots include completed lanes; action sensitivity also does.",
            "Model-only evaluation does not validate full replay resume.",
        ],
    }


def main() -> None:
    """Read the completed monitor snapshot and write validated public results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.status.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
