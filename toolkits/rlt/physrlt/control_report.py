# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Summarize immutable PhysRLT control screens without hiding failed updates."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from toolkits.rlt.pi05_baseline import write_json


def _episodes(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return payload.get("episodes", []) if isinstance(payload, dict) else payload


def _metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "successes": 0,
            "episodes": 0,
            "success_rate_pct": None,
            "mean_ticks": None,
            "candidate_nonreference_fraction_mean": None,
            "candidate_score_improvement_mean": None,
        }
    successes = sum(bool(row["success"]) for row in rows)
    nonreference = [
        float(row["candidate_nonreference_fraction"])
        for row in rows
        if row.get("candidate_nonreference_fraction") is not None
    ]
    score_improvements = [
        float(row["candidate_score_improvement_mean"])
        for row in rows
        if row.get("candidate_score_improvement_mean") is not None
    ]
    return {
        "successes": successes,
        "episodes": len(rows),
        "success_rate_pct": 100 * successes / len(rows),
        "mean_ticks": statistics.mean(float(row["ticks"]) for row in rows),
        "candidate_nonreference_fraction_mean": (
            statistics.mean(nonreference) if nonreference else None
        ),
        "candidate_score_improvement_mean": (
            statistics.mean(score_improvements) if score_improvements else None
        ),
    }


def summarize(runs: list[Path]) -> dict:
    """Return one auditable row per run and method."""
    if not runs:
        raise ValueError("At least one control run is required")
    output = []
    seen = set()
    for run in runs:
        result_path = run / "result.json"
        episode_path = run / "episodes.json"
        if not result_path.is_file() or not episode_path.is_file():
            raise ValueError(f"Incomplete control run: {run}")
        result = json.loads(result_path.read_text())
        if not result.get("complete") or result.get("state") != "complete":
            raise ValueError(f"Control run is not complete: {run}")
        episodes = _episodes(episode_path)
        methods = sorted(
            {row["method"] for row in episodes if row["method"] != "reference"}
        )
        if not methods:
            raise ValueError(f"Control run has no evaluated method: {run}")
        correction_axes = result.get("correction_axes")
        axis = ";".join(map(str, correction_axes or []))
        for method in methods:
            key = (str(run.resolve()), method)
            if key in seen:
                raise ValueError(f"Duplicate control row: {run}/{method}")
            seen.add(key)
            selected = [row for row in episodes if row["method"] == method]
            stages = {
                name: _metrics(
                    [
                        row
                        for row in selected
                        if row.get("stage") == stage and row.get("split") == split
                    ]
                )
                for name, stage, split in (
                    ("warmup_validation", "warmup", "validation"),
                    ("warmup_development", "warmup", "development"),
                    ("online_train", "online_train", "train"),
                    ("online_validation", "online_frozen_eval", "validation"),
                    ("online_development", "online_frozen_eval", "development"),
                )
            }
            gate_path = run / f"{method}-gate.json"
            deferred_path = run / f"{method}-deferred.json"
            gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
            deferred = (
                json.loads(deferred_path.read_text()) if deferred_path.is_file() else {}
            )
            row = {
                "run": run.name,
                "seed": result["seed"],
                "correction_axes": axis,
                "method": method,
                "gate_passed": gate.get("passed"),
                "online_tick_cap": result.get("online_tick_cap"),
                "positive_replay_transitions": deferred.get("positive_transitions"),
                "updates": deferred.get("updates"),
                "updates_skipped": deferred.get("updates_skipped"),
                "skip_reason": deferred.get("skip_reason"),
            }
            for name, metrics in stages.items():
                for metric, value in metrics.items():
                    row[f"{name}_{metric}"] = value
            output.append(row)
    return {
        "state": "complete",
        "role": "small development control screens; not final-test claims",
        "source_runs": [str(run.resolve()) for run in runs],
        "rows": output,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a stable flat view atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Report output must be a new directory")
    result = summarize(args.runs)
    args.output.mkdir(parents=True)
    write_json(args.output / "summary.json", result)
    write_csv(args.output / "summary.csv", result["rows"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
