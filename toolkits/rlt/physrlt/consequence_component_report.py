# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate matched response and geometry consequence critic gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from toolkits.rlt.pi05_baseline import write_json

COMPONENTS = ("joint", "response", "geometry")


def parse_gates(specs: list[str]) -> dict[str, Path]:
    """Parse exactly one result path for every physical component."""
    result = {}
    for spec in specs:
        component, separator, raw_path = spec.partition("=")
        if (
            not separator
            or component not in COMPONENTS
            or not raw_path
            or component in result
        ):
            raise ValueError(
                "Gates require unique joint|response|geometry=PATH entries"
            )
        result[component] = Path(raw_path).resolve()
    if set(result) != set(COMPONENTS):
        raise ValueError("Component report requires joint, response and geometry gates")
    return result


def summarize(payloads: dict[str, dict]) -> dict:
    """Verify a matched protocol and combine all causal critic cells."""
    if set(payloads) != set(COMPONENTS):
        raise ValueError("Require all three physical components")
    reference = payloads["joint"]
    matched_fields = (
        "trace_sha256",
        "base_readout_sha256",
        "consequence_readout_sha256",
        "seeds",
    )
    rows = []
    component_summary = []
    for component in COMPONENTS:
        payload = payloads[component]
        if payload.get("state") != "complete":
            raise ValueError("Every component gate must be complete")
        if payload.get("consequence_component") != component:
            raise ValueError("Gate payload physical component differs from its label")
        if any(payload.get(field) != reference.get(field) for field in matched_fields):
            raise ValueError("Component gates do not share the same protocol evidence")
        gate_rows = payload.get("gate_summaries", [])
        if not gate_rows:
            raise ValueError("Component gate lacks causal summaries")
        row_pass = all(bool(row.get("correct_best")) for row in gate_rows)
        if row_pass != bool(payload.get("passes_strict_screen")):
            raise ValueError("Component gate decision differs from its causal cells")
        rows.extend({"component": component, **row} for row in gate_rows)
        component_summary.append(
            {
                "component": component,
                "passes_strict_screen": row_pass,
                "cells": len(gate_rows),
            }
        )
    return {
        "state": "complete",
        "report_type": "independent_replay",
        "protocol": (
            "same independent replay traces, readouts, optimizer seeds and critic "
            "capacity; only the retained physical consequence coordinates differ"
        ),
        "trace_sha256": reference["trace_sha256"],
        "base_readout_sha256": reference["base_readout_sha256"],
        "consequence_readout_sha256": reference["consequence_readout_sha256"],
        "seeds": reference["seeds"],
        "component_summary": component_summary,
        "rows": rows,
    }


def summarize_episode_folds(payloads: dict[str, dict]) -> dict:
    """Combine matched same-run episode-fold component screens."""
    if set(payloads) != set(COMPONENTS):
        raise ValueError("Require all three physical components")
    reference = payloads["joint"]
    matched_fields = (
        "source_trace_sha256",
        "base_replay",
        "physical_replay",
        "base_readout",
        "consequence_readout",
        "complete_successful_episodes",
        "folds",
        "seeds",
    )
    rows = []
    component_summary = []
    for component in COMPONENTS:
        payload = payloads[component]
        if payload.get("state") != "complete":
            raise ValueError("Every component gate must be complete")
        if payload.get("consequence_component") != component:
            raise ValueError("Gate payload physical component differs from its label")
        if any(payload.get(field) != reference.get(field) for field in matched_fields):
            raise ValueError("Component gates do not share the same fold evidence")
        summaries = payload.get("summaries", [])
        if not summaries:
            raise ValueError("Component gate lacks episode-fold summaries")
        row_pass = all(bool(row.get("correct_best")) for row in summaries)
        if row_pass != bool(payload.get("passes_strict_screen")):
            raise ValueError("Component gate decision differs from its fold cells")
        for summary in summaries:
            means = summary.get("mean_concordance", {})
            if set(means) != {"correct", "reversed", "zero"}:
                raise ValueError("Episode-fold summary lacks all causal arms")
            rows.append(
                {
                    "component": component,
                    "fold": summary["fold"],
                    **means,
                    "correct_best": bool(summary["correct_best"]),
                }
            )
        component_summary.append(
            {
                "component": component,
                "passes_strict_screen": row_pass,
                "cells": len(summaries),
            }
        )
    return {
        "state": "complete",
        "report_type": "same_run_episode_fold",
        "protocol": (
            "same collection run, episode folds, readouts, optimizer seeds and "
            "critic capacity; only retained consequence coordinates differ"
        ),
        "limitation": reference.get("limitation"),
        "source_trace_sha256": reference["source_trace_sha256"],
        "seeds": reference["seeds"],
        "component_summary": component_summary,
        "rows": rows,
    }


def write_report(output: Path, result: dict) -> None:
    """Write machine-readable and paper-reviewable matched tables."""
    if output.exists():
        raise FileExistsError("Component report output must be new")
    output.mkdir(parents=True)
    write_json(output / "result.json", result)
    fields = list(result["rows"][0])
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])
    lines = ["# Matched consequence component gate", "", result["protocol"], ""]
    if result["report_type"] == "same_run_episode_fold":
        lines.extend(
            [
                "| Component | Fold | Correct | Reversed | Zero | Correct best |",
                "|---|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in result["rows"]:
            lines.append(
                f"| {row['component']} | {row['fold']} | "
                f"{float(row['correct']):.6f} | {float(row['reversed']):.6f} | "
                f"{float(row['zero']):.6f} | "
                f"{'yes' if row['correct_best'] else 'no'} |"
            )
    else:
        lines.extend(
            [
                "| Component | Evaluation | Seed | Metric | Correct | Reversed | Zero | Correct best |",
                "|---|---|---:|---|---:|---:|---:|:---:|",
            ]
        )
        for row in result["rows"]:
            values = [
                "NA" if row[arm] is None else f"{float(row[arm]):.6f}"
                for arm in ("correct", "reversed", "zero")
            ]
            lines.append(
                f"| {row['component']} | {row['evaluation']} | "
                f"{row['optimizer_seed']} | {row['metric']} | {values[0]} | "
                f"{values[1]} | {values[2]} | "
                f"{'yes' if row['correct_best'] else 'no'} |"
            )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gates", nargs="+")
    group.add_argument("--episode-fold-gates", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = parse_gates(args.gates or args.episode_fold_gates)
    payloads = {
        component: json.loads(path.read_text()) for component, path in paths.items()
    }
    result = summarize(payloads) if args.gates else summarize_episode_folds(payloads)
    result["sources"] = {component: str(path) for component, path in paths.items()}
    write_report(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
