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
GATE_METRICS = ("return_ranking_auc", "return_order_concordance")


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


def _paper_rows(payloads: dict[str, dict]) -> list[dict]:
    """Build one concise row per component and independent evaluation replay."""
    result = []
    for component in COMPONENTS:
        gate_rows = payloads[component]["gate_summaries"]
        evaluations = sorted({str(row["evaluation"]) for row in gate_rows})
        for evaluation in evaluations:
            selected = [
                row for row in gate_rows if str(row["evaluation"]) == evaluation
            ]
            aggregate = {
                str(row["metric"]): row
                for row in selected
                if str(row["optimizer_seed"]) == "aggregate"
            }
            if set(aggregate) != set(GATE_METRICS):
                raise ValueError("Component gate lacks both aggregate critic metrics")
            seed_cells = [
                row for row in selected if str(row["optimizer_seed"]) != "aggregate"
            ]
            if not seed_cells:
                raise ValueError("Component gate lacks per-seed critic cells")
            result.append(
                {
                    "component": component,
                    "evaluation": evaluation,
                    "return_auc_correct": aggregate["return_ranking_auc"]["correct"],
                    "return_auc_reversed": aggregate["return_ranking_auc"]["reversed"],
                    "return_auc_zero": aggregate["return_ranking_auc"]["zero"],
                    "concordance_correct": aggregate["return_order_concordance"][
                        "correct"
                    ],
                    "concordance_reversed": aggregate["return_order_concordance"][
                        "reversed"
                    ],
                    "concordance_zero": aggregate["return_order_concordance"]["zero"],
                    "seed_cells_passed": sum(
                        bool(row["correct_best"]) for row in seed_cells
                    ),
                    "seed_cells_total": len(seed_cells),
                    "aggregate_cells_passed": sum(
                        bool(row["correct_best"]) for row in aggregate.values()
                    ),
                    "passes_strict_screen": bool(
                        payloads[component]["passes_strict_screen"]
                    ),
                }
            )
    return result


def summarize(payloads: dict[str, dict]) -> dict:
    """Verify a matched protocol and combine all causal critic cells."""
    if set(payloads) != set(COMPONENTS):
        raise ValueError("Require all three physical components")
    reference = payloads["joint"]
    if not reference.get("revision") or reference.get("worktree_clean") is not True:
        raise ValueError("Component gates require clean revision provenance")
    matched_fields = (
        "revision",
        "worktree_clean",
        "trace_sha256",
        "collection_provenance",
        "base_readout_sha256",
        "consequence_readout_sha256",
        "consequence_conditioning",
        "require_robust_replay",
        "replay_sampling",
        "train_steps",
        "causal_protocol",
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
            "capacity; each critic is trained once on correct consequences, then "
            "frozen while only the input sign and retained coordinates differ"
        ),
        "causal_protocol": reference["causal_protocol"],
        "revision": reference["revision"],
        "worktree_clean": True,
        "trace_sha256": reference["trace_sha256"],
        "collection_provenance": reference["collection_provenance"],
        "base_readout_sha256": reference["base_readout_sha256"],
        "consequence_readout_sha256": reference["consequence_readout_sha256"],
        "consequence_conditioning": reference["consequence_conditioning"],
        "require_robust_replay": reference["require_robust_replay"],
        "replay_sampling": reference["replay_sampling"],
        "train_steps": reference["train_steps"],
        "seeds": reference["seeds"],
        "component_summary": component_summary,
        "paper_rows": _paper_rows(payloads),
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
        "conditioning_mode",
        "causal_protocol",
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
            "critic capacity; each critic is trained once on correct consequences, "
            "then frozen while only the input sign and retained coordinates differ"
        ),
        "causal_protocol": reference["causal_protocol"],
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
        paper_fields = list(result["paper_rows"][0])
        with (output / "paper_table.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=paper_fields)
            writer.writeheader()
            writer.writerows(result["paper_rows"])
        paper_lines = [
            "# Independent replay component summary",
            "",
            "| Component | Evaluation | AUC c/r/z | Concordance c/r/z | Seed cells | Aggregate cells | Pass |",
            "|---|---|---|---|---:|---:|:---:|",
        ]
        for row in result["paper_rows"]:
            auc = "/".join(
                f"{float(row[name]):.6f}"
                for name in (
                    "return_auc_correct",
                    "return_auc_reversed",
                    "return_auc_zero",
                )
            )
            concordance = "/".join(
                f"{float(row[name]):.6f}"
                for name in (
                    "concordance_correct",
                    "concordance_reversed",
                    "concordance_zero",
                )
            )
            paper_lines.append(
                f"| {row['component']} | {row['evaluation']} | {auc} | "
                f"{concordance} | {row['seed_cells_passed']}/"
                f"{row['seed_cells_total']} | {row['aggregate_cells_passed']}/2 | "
                f"{'yes' if row['passes_strict_screen'] else 'no'} |"
            )
        (output / "paper_table.md").write_text("\n".join(paper_lines) + "\n")
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
