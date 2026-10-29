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
            "mean_failure_capped_ticks": None,
            "successful_mean_ticks": None,
            "mean_wall_seconds": None,
            "successful_mean_wall_seconds": None,
            "ticks_per_wall_second": None,
            "candidate_selection_calls_total": None,
            "candidate_nonreference_fraction_mean": None,
            "candidate_scale_mean": None,
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
    candidate_scales = [
        float(row["candidate_scale_mean"])
        for row in rows
        if row.get("candidate_scale_mean") is not None
    ]
    timed_rows = [row for row in rows if row.get("seconds") is not None]
    seconds = [float(row["seconds"]) for row in timed_rows]
    successful_rows = [row for row in rows if row["success"]]
    successful_seconds = [
        float(row["seconds"])
        for row in successful_rows
        if row.get("seconds") is not None
    ]
    selection_calls = [
        int(row["candidate_selection_calls"])
        for row in rows
        if row.get("candidate_selection_calls") is not None
    ]
    mean_ticks = statistics.mean(float(row["ticks"]) for row in rows)
    return {
        "successes": successes,
        "episodes": len(rows),
        "success_rate_pct": 100 * successes / len(rows),
        "mean_ticks": mean_ticks,
        "mean_failure_capped_ticks": mean_ticks,
        "successful_mean_ticks": (
            statistics.mean(float(row["ticks"]) for row in successful_rows)
            if successful_rows
            else None
        ),
        "mean_wall_seconds": statistics.mean(seconds) if seconds else None,
        "successful_mean_wall_seconds": (
            statistics.mean(successful_seconds) if successful_seconds else None
        ),
        "ticks_per_wall_second": (
            sum(float(row["ticks"]) for row in timed_rows) / sum(seconds)
            if seconds and sum(seconds) > 0
            else None
        ),
        "candidate_selection_calls_total": (
            sum(selection_calls) if selection_calls else None
        ),
        "candidate_nonreference_fraction_mean": (
            statistics.mean(nonreference) if nonreference else None
        ),
        "candidate_scale_mean": (
            statistics.mean(candidate_scales) if candidate_scales else None
        ),
        "candidate_score_improvement_mean": (
            statistics.mean(score_improvements) if score_improvements else None
        ),
    }


def _paired_metrics(method_rows: list[dict], reference_rows: list[dict]) -> dict:
    result = {}
    for split in ("validation", "development"):
        method_items = [
            row
            for row in method_rows
            if row.get("stage") == "online_frozen_eval"
            and row.get("split") == split
        ]
        reference_items = [
            row
            for row in reference_rows
            if row.get("stage") == "frozen" and row.get("split") == split
        ]
        method = {str(row["case_id"]): row for row in method_items}
        reference = {str(row["case_id"]): row for row in reference_items}
        if len(method) != len(method_items) or len(reference) != len(reference_items):
            raise ValueError("Control report contains duplicate case identities")
        prefix = f"paired_{split}"
        if not method:
            result.update(
                {
                    f"{prefix}_episodes": 0,
                    f"{prefix}_success_rate_delta_pct": None,
                    f"{prefix}_failure_capped_ticks_delta_mean": None,
                    f"{prefix}_wall_seconds_delta_mean": None,
                }
            )
            continue
        if set(method) != set(reference):
            raise ValueError("Online and reference control cases are not paired")
        for case_id in method:
            method_hash = method[case_id].get("reset_hash")
            reference_hash = reference[case_id].get("reset_hash")
            if method_hash is not None and method_hash != reference_hash:
                raise ValueError("Online and reference reset hashes differ")
        success_differences = [
            float(method[case]["success"]) - float(reference[case]["success"])
            for case in method
        ]
        tick_differences = [
            float(method[case]["ticks"]) - float(reference[case]["ticks"])
            for case in method
        ]
        second_differences = [
            float(method[case]["seconds"]) - float(reference[case]["seconds"])
            for case in method
            if method[case].get("seconds") is not None
            and reference[case].get("seconds") is not None
        ]
        result.update(
            {
                f"{prefix}_episodes": len(method),
                f"{prefix}_success_rate_delta_pct": 100
                * statistics.mean(success_differences),
                f"{prefix}_failure_capped_ticks_delta_mean": statistics.mean(
                    tick_differences
                ),
                f"{prefix}_wall_seconds_delta_mean": (
                    statistics.mean(second_differences)
                    if second_differences
                    else None
                ),
            }
        )
    return result


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
        reference_rows = [row for row in episodes if row["method"] == "reference"]
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
            for name, metrics in (
                (
                    "reference_validation",
                    _metrics(
                        [
                            item
                            for item in reference_rows
                            if item.get("stage") == "frozen"
                            and item.get("split") == "validation"
                        ]
                    ),
                ),
                (
                    "reference_development",
                    _metrics(
                        [
                            item
                            for item in reference_rows
                            if item.get("stage") == "frozen"
                            and item.get("split") == "development"
                        ]
                    ),
                ),
            ):
                for metric, value in metrics.items():
                    row[f"{name}_{metric}"] = value
            row.update(_paired_metrics(selected, reference_rows))
            output.append(row)
    return {
        "state": "complete",
        "role": "small development control screens; not final-test claims",
        "protocol": (
            "failures retain their capped ticks; online and frozen reference "
            "comparisons use identical case IDs and reset hashes"
        ),
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
