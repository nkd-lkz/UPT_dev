# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Apply a preregistered closed-loop gate to response-matrix controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toolkits.rlt.pi05_baseline import write_json

CORE_METHODS = (
    "reference",
    "adaptive",
    "matrix_adaptive",
    "matrix_reversed",
    "fixed_high",
    "delayed_high",
)


def load_scenario(directory: Path, label: str) -> tuple[list[dict], list[dict]]:
    """Load a complete immutable run and verify its matched-reset contract."""
    result_path = directory / "result.json"
    episodes_path = directory / "episodes.json"
    if not result_path.is_file() or not episodes_path.is_file():
        raise ValueError(f"{label} requires result.json and episodes.json")
    result = json.loads(result_path.read_text())
    episodes = json.loads(episodes_path.read_text())
    if result.get("state") != "complete" or not episodes:
        raise ValueError(f"{label} is not a complete response-control run")
    selected = [row for row in episodes if row["method"] in CORE_METHODS]
    methods = {row["method"] for row in selected}
    if methods != set(CORE_METHODS):
        raise ValueError(f"{label} does not contain every matched control")
    case_ids = {row["case_id"] for row in selected}
    for case_id in case_ids:
        rows = [row for row in selected if row["case_id"] == case_id]
        if len(rows) != len(CORE_METHODS):
            raise ValueError(f"{label} has duplicate or missing method episodes")
        if len({row["reset_hash"] for row in rows}) != 1:
            raise ValueError(f"{label} methods do not share the same reset")
    declared = {case["id"] for case in result["cases"]}
    if case_ids != declared:
        raise ValueError(f"{label} episode cases differ from the declared cases")
    return selected, result["cases"]


def summarize(episodes: list[dict], scenario: str) -> list[dict]:
    """Return success and failure-capped completion metrics per method."""
    rows = []
    for method in CORE_METHODS:
        selected = [row for row in episodes if row["method"] == method]
        successes = sum(bool(row["success"]) for row in selected)
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "successes": successes,
                "episodes": len(selected),
                "success_rate": successes / len(selected),
                "mean_capped_ticks": sum(float(row["ticks"]) for row in selected)
                / len(selected),
            }
        )
    return rows


def _method(rows: list[dict], scenario: str, method: str) -> dict:
    return next(
        row for row in rows if row["scenario"] == scenario and row["method"] == method
    )


def _strictly_better(candidate: dict, control: dict, tick_margin: float) -> bool:
    if candidate["successes"] != control["successes"]:
        return candidate["successes"] > control["successes"]
    return candidate["mean_capped_ticks"] <= control["mean_capped_ticks"] * (
        1 - tick_margin
    )


def _no_worse(candidate: dict, control: dict, tick_tolerance: float) -> bool:
    if candidate["successes"] != control["successes"]:
        return candidate["successes"] > control["successes"]
    return candidate["mean_capped_ticks"] <= control["mean_capped_ticks"] * (
        1 + tick_tolerance
    )


def evaluate(stress: list[dict], nominal: list[dict]) -> tuple[list[dict], bool]:
    """Evaluate the frozen development gate without tuning on its results."""
    rows = summarize(stress, "stress") + summarize(nominal, "nominal")
    matrix_stress = _method(rows, "stress", "matrix_adaptive")
    matrix_nominal = _method(rows, "nominal", "matrix_adaptive")
    checks = [
        {
            "name": "stress_beats_axis_adaptive",
            "pass": _strictly_better(
                matrix_stress, _method(rows, "stress", "adaptive"), 0.05
            ),
        },
        {
            "name": "stress_beats_timing_matched_static_high",
            "pass": _strictly_better(
                matrix_stress, _method(rows, "stress", "delayed_high"), 0.05
            ),
        },
        {
            "name": "stress_beats_reversed_matrix",
            "pass": _strictly_better(
                matrix_stress, _method(rows, "stress", "matrix_reversed"), 0.05
            ),
        },
        {
            "name": "stress_no_worse_than_unconditional_static_high",
            "pass": _no_worse(
                matrix_stress, _method(rows, "stress", "fixed_high"), 0.05
            ),
        },
        {
            "name": "nominal_no_worse_than_reference",
            "pass": _no_worse(
                matrix_nominal, _method(rows, "nominal", "reference"), 0.05
            ),
        },
    ]
    return rows + checks, all(check["pass"] for check in checks)


def markdown_table(rows: list[dict]) -> str:
    """Render the matched control metrics for the experiment ledger."""
    metric_rows = [row for row in rows if "method" in row]
    lines = [
        "| Scenario | Method | Success | Mean capped ticks |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['scenario']} | {row['method']} | "
            f"{row['successes']}/{row['episodes']} "
            f"({100 * row['success_rate']:.1f}%) | "
            f"{row['mean_capped_ticks']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict:
    """Load matched runs, apply the fixed gate and write immutable artifacts."""
    if args.output.exists():
        raise FileExistsError("Response-matrix gate outputs are immutable")
    stress, stress_cases = load_scenario(args.stress, "stress")
    nominal, nominal_cases = load_scenario(args.nominal, "nominal")
    if [case["id"] for case in stress_cases] != [case["id"] for case in nominal_cases]:
        raise ValueError("Stress and nominal runs must use the same ordered cases")
    rows, passed = evaluate(stress, nominal)
    result = {
        "state": "complete",
        "protocol": (
            "same held-out layouts; failure-capped ticks; a success difference "
            "dominates time, otherwise require 5% faster for strict comparisons; "
            "nominal and static-high safety allow 5% tick tolerance"
        ),
        "stress": str(args.stress),
        "nominal": str(args.nominal),
        "cases": [case["id"] for case in stress_cases],
        "passes_development_gate": passed,
        "rows": rows,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    (args.output / "comparison.md").write_text(markdown_table(rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
