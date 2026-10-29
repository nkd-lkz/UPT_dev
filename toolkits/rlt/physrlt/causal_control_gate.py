# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Gate conservative control on correct-versus-causal-arm outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from toolkits.rlt.pi05_baseline import write_json


def parse_arms(specs: list[str]) -> dict[str, str]:
    """Parse explicit correct, reversed and zero method labels."""
    result = {}
    for spec in specs:
        arm, separator, method = spec.partition("=")
        if (
            not separator
            or arm not in {"correct", "reversed", "zero"}
            or not method
            or arm in result
            or method in result.values()
        ):
            raise ValueError("Arms require unique correct|reversed|zero=METHOD entries")
        result[arm] = method
    if set(result) != {"correct", "reversed", "zero"}:
        raise ValueError("Causal control gate requires all three arms")
    return result


def _episode_map(rows: list[dict], method: str, stage: str, split: str) -> dict:
    selected = [
        row
        for row in rows
        if row.get("method") == method
        and row.get("stage") == stage
        and row.get("split") == split
    ]
    mapped = {str(row["case_id"]): row for row in selected}
    if not selected or len(mapped) != len(selected):
        raise ValueError("Every arm needs unique paired control cases")
    return mapped


def _selection_trace(row: dict) -> list[int] | None:
    """Validate one executed candidate-selection trace when it is available."""
    trace = row.get("candidate_selected_indices")
    if trace is None:
        return None
    if not isinstance(trace, list):
        raise ValueError("Candidate selection traces must be lists")
    if any(type(index) is not int or index < 0 for index in trace):
        raise ValueError("Candidate selection traces must contain nonnegative integers")
    calls = row.get("candidate_selection_calls")
    if type(calls) is not int or calls != len(trace):
        raise ValueError("Candidate selection trace length must match recorded calls")
    return trace


def _paired_row(
    seed: int,
    split: str,
    comparator: str,
    correct: dict,
    control: dict,
) -> dict:
    if set(correct) != set(control):
        raise ValueError("Causal control arms use different case identities")
    trace_comparable_cases = 0
    trace_diverged_cases = 0
    for case_id in correct:
        correct_hash = correct[case_id].get("reset_hash")
        control_hash = control[case_id].get("reset_hash")
        if correct_hash is not None and correct_hash != control_hash:
            raise ValueError("Causal control arms use different reset states")
        correct_trace = _selection_trace(correct[case_id])
        control_trace = _selection_trace(control[case_id])
        if correct_trace is None or control_trace is None:
            continue
        trace_comparable_cases += 1
        if any(first != second for first, second in zip(correct_trace, control_trace)):
            trace_diverged_cases += 1
    correct_successes = sum(bool(row["success"]) for row in correct.values())
    control_successes = sum(bool(row["success"]) for row in control.values())
    correct_ticks = statistics.mean(float(row["ticks"]) for row in correct.values())
    control_ticks = statistics.mean(float(row["ticks"]) for row in control.values())
    return {
        "seed": seed,
        "split": split,
        "comparator": comparator,
        "episodes": len(correct),
        "correct_successes": correct_successes,
        "control_successes": control_successes,
        "success_delta": correct_successes - control_successes,
        "correct_failure_capped_ticks_mean": correct_ticks,
        "control_failure_capped_ticks_mean": control_ticks,
        "tick_delta_mean": correct_ticks - control_ticks,
        "selection_trace_comparable_cases": trace_comparable_cases,
        "selection_trace_diverged_cases": trace_diverged_cases,
    }


def _checkpoint_audit(run: Path, arms: dict[str, str], seed: int) -> dict:
    """Require all causal arms to intervene on one correct-trained checkpoint."""
    records = {}
    for arm, method in arms.items():
        path = run / f"{method}-online-checkpoint.json"
        if not path.is_file():
            raise ValueError("Every causal arm needs checkpoint provenance")
        payload = json.loads(path.read_text())
        required = {
            "source",
            "source_sha256",
            "optimizer_seed",
            "arm",
            "training_arm",
            "consequence_component",
            "consequence_conditioning",
            "train_trace_sha256",
            "replay_sampling",
        }
        if required - set(payload):
            raise ValueError("Causal arm checkpoint provenance is incomplete")
        if (
            payload["arm"] != arm
            or payload["training_arm"] != "correct"
            or int(payload["optimizer_seed"]) != seed
        ):
            raise ValueError("Causal arm checkpoint contract differs")
        records[arm] = payload
    shared_fields = (
        "source",
        "source_sha256",
        "training_arm",
        "consequence_component",
        "consequence_conditioning",
        "train_trace_sha256",
        "replay_sampling",
    )
    for field in shared_fields:
        if len({records[arm][field] for arm in records}) != 1:
            raise ValueError("Causal arms do not share one frozen checkpoint")
    reference = records["correct"]
    return {
        "seed": seed,
        **{field: reference[field] for field in shared_fields},
        "evaluation_arms": sorted(records),
    }


def evaluate(runs: list[Path], arms: dict[str, str]) -> dict:
    """Require causal superiority without a split-level success regression."""
    if not runs:
        raise ValueError("Causal control gate requires at least one run")
    rows = []
    correct_method = arms["correct"]
    seen_seeds = set()
    correct_nonreference = []
    checkpoint_audits = []
    for run in runs:
        result = json.loads((run / "result.json").read_text())
        episodes = json.loads((run / "episodes.json").read_text())
        if isinstance(episodes, dict):
            episodes = episodes.get("episodes", [])
        if result.get("state") != "complete" or not result.get("complete"):
            raise ValueError("Causal control run is incomplete")
        seed = int(result["seed"])
        if seed in seen_seeds:
            raise ValueError("Causal control runs require unique optimizer seeds")
        seen_seeds.add(seed)
        checkpoint_audits.append(_checkpoint_audit(run, arms, seed))
        for split in ("validation", "development"):
            correct = _episode_map(
                episodes, correct_method, "online_frozen_eval", split
            )
            reference = _episode_map(episodes, "reference", "frozen", split)
            rows.append(_paired_row(seed, split, "reference", correct, reference))
            for arm in ("reversed", "zero"):
                control = _episode_map(episodes, arms[arm], "online_frozen_eval", split)
                rows.append(_paired_row(seed, split, arm, correct, control))
            for row in correct.values():
                if row.get("candidate_nonreference_fraction") is not None:
                    correct_nonreference.append(
                        float(row["candidate_nonreference_fraction"])
                    )
    comparisons = []
    for comparator in ("reference", "reversed", "zero"):
        selected = [row for row in rows if row["comparator"] == comparator]
        correct_successes = sum(row["correct_successes"] for row in selected)
        control_successes = sum(row["control_successes"] for row in selected)
        episodes = sum(row["episodes"] for row in selected)
        correct_ticks = (
            sum(
                row["correct_failure_capped_ticks_mean"] * row["episodes"]
                for row in selected
            )
            / episodes
        )
        control_ticks = (
            sum(
                row["control_failure_capped_ticks_mean"] * row["episodes"]
                for row in selected
            )
            / episodes
        )
        no_split_success_regression = all(row["success_delta"] >= 0 for row in selected)
        selection_trace_comparable_cases = sum(
            row["selection_trace_comparable_cases"] for row in selected
        )
        selection_trace_diverged_cases = sum(
            row["selection_trace_diverged_cases"] for row in selected
        )
        selection_trace_complete = comparator == "reference" or (
            selection_trace_comparable_cases == episodes
        )
        causal_choices_diverged = comparator == "reference" or (
            selection_trace_complete and selection_trace_diverged_cases > 0
        )
        strict_success_gain = correct_successes > control_successes
        tick_improvement_fraction = (
            (control_ticks - correct_ticks) / control_ticks
            if control_ticks > 0
            else 0.0
        )
        tied_success_efficiency_gain = (
            correct_successes == control_successes and tick_improvement_fraction >= 0.05
        )
        passed = (
            no_split_success_regression
            and causal_choices_diverged
            and (strict_success_gain or tied_success_efficiency_gain)
        )
        comparisons.append(
            {
                "comparator": comparator,
                "episodes": episodes,
                "correct_successes": correct_successes,
                "control_successes": control_successes,
                "success_delta": correct_successes - control_successes,
                "correct_failure_capped_ticks_mean": correct_ticks,
                "control_failure_capped_ticks_mean": control_ticks,
                "tick_improvement_fraction": tick_improvement_fraction,
                "no_split_success_regression": no_split_success_regression,
                "selection_trace_comparable_cases": selection_trace_comparable_cases,
                "selection_trace_diverged_cases": selection_trace_diverged_cases,
                "selection_trace_complete": selection_trace_complete,
                "causal_choices_diverged": causal_choices_diverged,
                "passes_predeclared_rule": passed,
            }
        )
    correct_nonreference_mean = (
        statistics.mean(correct_nonreference) if correct_nonreference else None
    )
    selector_active = (
        correct_nonreference_mean is not None and correct_nonreference_mean > 0
    )
    passes = selector_active and all(
        row["passes_predeclared_rule"] for row in comparisons
    )
    return {
        "state": "complete",
        "role": "development causal screen, not final test evidence",
        "rule": (
            "correct has no validation/development success regression and, pooled "
            "across unique optimizer seeds, either strictly more successes or equal "
            "successes with at least 5% lower failure-capped ticks; the correct arm "
            "must also select at least one non-reference candidate and must differ "
            "from each causal input control on a shared executed selection prefix; "
            "every causal comparison must provide a complete selection trace"
        ),
        "arms": arms,
        "seeds": sorted(seen_seeds),
        "checkpoint_audits": checkpoint_audits,
        "rows": rows,
        "comparisons": comparisons,
        "correct_candidate_nonreference_fraction_mean": correct_nonreference_mean,
        "correct_selector_active": selector_active,
        "passes_causal_closed_loop_gate": passes,
        "decision": "retain_for_expansion" if passes else "stop_after_development",
    }


def write_report(output: Path, result: dict) -> None:
    """Write the predeclared decision and all paired causal contrasts."""
    if output.exists():
        raise FileExistsError("Causal control gate output must be new")
    output.mkdir(parents=True)
    write_json(output / "result.json", result)
    fields = list(result["rows"][0])
    with (output / "paired_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])
    fields = list(result["comparisons"][0])
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["comparisons"])
    lines = [
        "# Conservative causal control gate",
        "",
        result["rule"],
        "",
        "Correct selector active: "
        f"{'yes' if result['correct_selector_active'] else 'no'} "
        f"(mean non-reference fraction: "
        f"{result['correct_candidate_nonreference_fraction_mean']})",
        "",
        "| Comparator | Episodes | Correct successes | Control successes | Tick improvement | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in result["comparisons"]:
        lines.append(
            f"| {row['comparator']} | {row['episodes']} | "
            f"{row['correct_successes']} | {row['control_successes']} | "
            f"{100 * row['tick_improvement_fraction']:.1f}% | "
            f"{'yes' if row['passes_predeclared_rule'] else 'no'} |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.runs, parse_arms(args.arms))
    result["runs"] = [str(run.resolve()) for run in args.runs]
    write_report(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
