# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate immutable sparse-motion runs without selecting a best seed."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from toolkits.rlt.pi05_baseline import write_json


def _mean_optional(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return statistics.mean(available) if available else None


def aggregate(paths: list[Path]) -> dict:
    """Group matched runs and report every arm across all supplied seeds."""
    if not paths:
        raise ValueError("At least one sparse-motion result is required")
    groups = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("state") != "complete":
            raise ValueError(f"Incomplete sparse-motion result: {path}")
        key = (
            payload["axis"],
            payload["readout_method"],
            payload.get("evaluation_split", "development"),
        )
        groups[key].append(payload)
    summary = []
    for (axis, method, split), runs in sorted(groups.items()):
        seeds = [run["seed"] for run in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError("A sparse-motion group repeats an optimizer seed")
        dataset_hashes = {
            run.get("evaluation_dataset_sha256") for run in runs
        }
        if len(dataset_hashes) != 1:
            raise ValueError("Matched seeds must use one evaluation dataset")
        arms = set(runs[0]["results"])
        if any(set(run["results"]) != arms for run in runs[1:]):
            raise ValueError("Matched seeds expose different ablation arms")
        for arm in sorted(arms):
            values = [run["results"][arm] for run in runs]
            rmse = [value["rmse_m"] * 1000 for value in values]
            improvements = [value["improvement_vs_zero_pct"] for value in values]
            auc = [value["event_auc"] for value in values]
            summary.append(
                {
                    "axis": axis,
                    "readout_method": method,
                    "evaluation_split": split,
                    "arm": arm,
                    "seeds": seeds,
                    "runs": len(runs),
                    "zero_motion_rmse_mm": statistics.mean(
                        value["zero_motion_rmse_m"] * 1000 for value in values
                    ),
                    "rmse_mm_mean": statistics.mean(rmse),
                    "rmse_mm_std": statistics.pstdev(rmse),
                    "improvement_pct_mean": statistics.mean(improvements),
                    "improvement_pct_min": min(improvements),
                    "event_auc_mean": _mean_optional(auc),
                    "effect_cosine_mean": _mean_optional(
                        [value["effect_cosine"] for value in values]
                    ),
                    "sign_reversed_prediction_rmse_mm": statistics.mean(
                        value["sign_reversed_prediction_rmse_m"] * 1000
                        for value in values
                    ),
                    "all_seeds_beat_zero": all(value > 0 for value in improvements),
                    "evaluation_dataset_sha256": next(iter(dataset_hashes)),
                }
            )
    indexed = {
        (row["axis"], row["evaluation_split"], row["readout_method"], row["arm"]): row
        for row in summary
    }
    comparisons = []
    for axis, split in sorted({(row["axis"], row["evaluation_split"]) for row in summary}):
        keys = {
            name: indexed.get((axis, split, method, arm))
            for name, method, arm in (
                ("flow", "flow_joint", "gated_pair"),
                ("final", "joint", "gated_pair"),
                ("reversed", "flow_joint", "reversed_gated_pair"),
            )
        }
        if any(value is None for value in keys.values()):
            continue
        flow, final, reversed_row = (
            keys["flow"],
            keys["final"],
            keys["reversed"],
        )
        comparisons.append(
            {
                "axis": axis,
                "evaluation_split": split,
                "flow_vs_final_rmse_reduction_pct": 100
                * (final["rmse_mm_mean"] - flow["rmse_mm_mean"])
                / final["rmse_mm_mean"],
                "flow_improvement_vs_zero_pct": flow["improvement_pct_mean"],
                "flow_min_seed_improvement_pct": flow["improvement_pct_min"],
                "reversed_improvement_vs_zero_pct": reversed_row[
                    "improvement_pct_mean"
                ],
                "passed": (
                    flow["improvement_pct_min"] > 5
                    and flow["rmse_mm_mean"] < final["rmse_mm_mean"]
                    and flow["rmse_mm_mean"] < reversed_row["rmse_mm_mean"]
                ),
            }
        )
    return {
        "state": "complete",
        "source_runs": len(paths),
        "rows": summary,
        "comparisons": comparisons,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the flat paper-table view atomically."""
    columns = [key for key in rows[0] if key != "seeds"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key != "seeds"}
            for row in rows
        )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Report output must be a new directory")
    result = aggregate(args.results)
    args.output.mkdir(parents=True)
    write_json(args.output / "summary.json", result)
    write_csv(args.output / "summary.csv", result["rows"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
