# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Report learner updates and declared takeover burden from an online run."""

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    """Count accepted transitions and human segments, not packet retries.

    Human labels attest to the client's action route, not physical device use.
    Old logs without done/reward fields do not establish completed-episode rates.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    episodes = {}
    previous_sequence = -1
    for row in rows:
        if row["sequence"] != previous_sequence + 1:
            raise ValueError("Metrics contain a duplicate or missing sequence")
        previous_sequence = row["sequence"]
        episode = episodes.setdefault(
            row["episode"],
            {
                "steps": 0,
                "human_steps": 0,
                "human_segments": 0,
                "previous_human": False,
                "completed": False,
                "success": False,
            },
        )
        if episode["completed"]:
            raise ValueError("Transition follows terminal state in the same episode")
        human = row["human"]
        episode["steps"] += 1
        episode["human_steps"] += int(human)
        episode["human_segments"] += int(human and not episode["previous_human"])
        episode["previous_human"] = human
        episode["completed"] = row.get("terminated", False) or row.get(
            "truncated", False
        )
        episode["success"] |= row.get("reward", 0) == 1
    completed = [row for row in episodes.values() if row["completed"]]
    human_steps = sum(row["human_steps"] for row in episodes.values())
    last = rows[-1] if rows else {}
    return {
        "scope": "Single-environment h=1 HIL; human flags are client-declared, not a robot benchmark",
        "accepted_this_run": len(rows),
        "declared_human_steps": human_steps,
        "declared_human_fraction": human_steps / len(rows) if rows else None,
        "declared_human_segments": sum(
            row["human_segments"] for row in episodes.values()
        ),
        "completed_episodes": len(completed),
        "success_among_completed": sum(row["success"] for row in completed)
        / len(completed)
        if completed
        else None,
        "learner_updates_total": last.get("update_step", 0),
        "published_version": last.get("policy_version", 0),
        "actor_ready": last.get("actor_ready", False),
        "update_budget_exhausted": last.get("update_budget_exhausted", False),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run / "metrics.jsonl"), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
