# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Test whether a critic uses candidate physical consequences across replays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from pathlib import Path

import torch

from toolkits.rlt.libero_acceptance_learning import AcceptanceLearner
from toolkits.rlt.physrlt.conditioning import SignedCandidateConsequence
from toolkits.rlt.physrlt.control import (
    FrozenReadout,
    validate_fixed_replay,
)
from toolkits.rlt.physrlt.critic_audit import audit_replay
from toolkits.rlt.physrlt.replay_eligibility import evaluate as evaluate_replay
from toolkits.rlt.pi05_baseline import configure_source, write_json


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_evaluations(specs: list[str]) -> dict[str, tuple[Path, Path]]:
    """Parse unique ``LABEL=BASE_REPLAY,PHYSICAL_REPLAY`` specifications."""
    result = {}
    for spec in specs:
        label, separator, raw_paths = spec.partition("=")
        base, path_separator, physical = raw_paths.partition(",")
        if (
            not separator
            or not path_separator
            or not label
            or not base
            or not physical
            or label in result
        ):
            raise ValueError(
                "Evaluations require unique LABEL=BASE_REPLAY,PHYSICAL_REPLAY entries"
            )
        result[label] = (Path(base), Path(physical))
    return result


def load_replay(base_path: Path, physical_path: Path) -> list[dict]:
    """Merge semantic and physical views of one immutable executed trace."""
    replay = torch.load(base_path, map_location="cpu", weights_only=True)
    physical = torch.load(physical_path, map_location="cpu", weights_only=True)
    base_summary = validate_fixed_replay(replay)
    physical_summary = validate_fixed_replay(physical)
    if (
        len(replay) != len(physical)
        or base_summary["trace_sha256"] != physical_summary["trace_sha256"]
    ):
        raise ValueError("Semantic and physical replay traces differ")
    merged = []
    for row, physical_row in zip(replay, physical, strict=True):
        critic_z = physical_row.get("critic_z", physical_row["z"])
        next_critic_z = physical_row.get("next_critic_z", physical_row["next_z"])
        if (
            critic_z.shape != row["z"].shape
            or next_critic_z.shape != row["next_z"].shape
        ):
            raise ValueError("Semantic and physical replay token shapes differ")
        merged.append(row | {"critic_z": critic_z, "next_critic_z": next_critic_z})
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"actor_z", "next_actor_z"}
        }
        for row in merged
    ]


def audit_replay_set(
    train: list[dict],
    evaluations: dict[str, list[dict]],
    *,
    require_robust: bool = False,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Require informative and trace-independent train/evaluation replays."""
    replays = {"train": train, **evaluations}
    summaries = {label: evaluate_replay(replay) for label, replay in replays.items()}
    eligibility_key = (
        "eligible_for_robust_independent_critic_gate"
        if require_robust
        else "eligible_for_independent_critic_gate"
    )
    reasons_key = "robust_rejection_reasons" if require_robust else "rejection_reasons"
    rejected = {
        label: summary[reasons_key]
        for label, summary in summaries.items()
        if not summary[eligibility_key]
    }
    if rejected:
        raise ValueError(f"Critic gate replays are ineligible: {rejected}")
    trace_hashes = {
        label: str(summary["trace_sha256"]) for label, summary in summaries.items()
    }
    if len(set(trace_hashes.values())) != len(trace_hashes):
        raise ValueError("Train and evaluation replays must have distinct traces")
    return trace_hashes, summaries


def summarize_gate(rows: list[dict]) -> tuple[list[dict], bool]:
    """Require the correct consequence to rank returns best in every cell."""
    if not rows:
        raise ValueError("Critic gate requires audit rows")
    evaluations = sorted({str(row["evaluation"]) for row in rows})
    seeds = sorted({int(row["optimizer_seed"]) for row in rows})
    metrics = ("return_ranking_auc", "return_order_concordance")
    summaries = []
    strict_checks = []
    for evaluation in evaluations:
        for seed in seeds:
            cell = [
                row
                for row in rows
                if row["evaluation"] == evaluation
                and int(row["optimizer_seed"]) == seed
            ]
            by_arm = {str(row["arm"]): row for row in cell}
            if set(by_arm) != {"correct", "reversed", "zero"} or len(cell) != 3:
                raise ValueError("Every critic gate cell requires three unique arms")
            for metric in metrics:
                values = {arm: by_arm[arm].get(metric) for arm in by_arm}
                passes = all(value is not None for value in values.values()) and (
                    float(values["correct"])
                    > max(float(values["reversed"]), float(values["zero"]))
                )
                strict_checks.append(passes)
                summaries.append(
                    {
                        "evaluation": evaluation,
                        "optimizer_seed": seed,
                        "metric": metric,
                        **values,
                        "correct_best": passes,
                    }
                )
        for metric in metrics:
            values = {}
            for arm in ("correct", "reversed", "zero"):
                current = [
                    row.get(metric)
                    for row in rows
                    if row["evaluation"] == evaluation and row["arm"] == arm
                ]
                values[arm] = (
                    sum(float(value) for value in current) / len(current)
                    if current and all(value is not None for value in current)
                    else None
                )
            passes = all(value is not None for value in values.values()) and (
                float(values["correct"])
                > max(float(values["reversed"]), float(values["zero"]))
            )
            strict_checks.append(passes)
            summaries.append(
                {
                    "evaluation": evaluation,
                    "optimizer_seed": "aggregate",
                    "metric": metric,
                    **values,
                    "correct_best": passes,
                }
            )
    return summaries, all(strict_checks)


def _write_gate_table(output: Path, summaries: list[dict]) -> None:
    fields = list(summaries[0])
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# Consequence critic gate",
        "",
        "Correct physical direction must strictly outperform reversed and zero "
        "controls in every optimizer seed and in the seed aggregate.",
        "",
        "| Evaluation | Seed | Metric | Correct | Reversed | Zero | Best |",
        "|---|---:|---|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        values = [
            "NA" if row[arm] is None else f"{float(row[arm]):.6f}"
            for arm in ("correct", "reversed", "zero")
        ]
        lines.append(
            f"| {row['evaluation']} | {row['optimizer_seed']} | {row['metric']} | "
            f"{values[0]} | {values[1]} | {values[2]} | "
            f"{'yes' if row['correct_best'] else 'no'} |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict:
    """Fit correct-conditioned critics and audit frozen causal interventions."""
    if (
        args.output.exists()
        or args.steps < 1
        or len(args.seeds) < 3
        or len(set(args.seeds)) != len(args.seeds)
    ):
        raise ValueError(
            "Require a new output, positive steps and three unique optimizer seeds"
        )
    configure_source(args.source)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Critic gate requires a clean Git worktree")
    train = load_replay(args.train_base_replay, args.train_physical_replay)
    evaluations = {
        label: load_replay(*paths)
        for label, paths in parse_evaluations(args.evaluations).items()
    }
    trace_hashes, replay_summaries = audit_replay_set(
        train,
        evaluations,
        require_robust=args.require_robust_replay,
    )
    readout = FrozenReadout(args.base_readout, consequence=args.consequence_readout)
    rows = []
    seed_states = {}
    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        conditioner = SignedCandidateConsequence(
            readout.candidate_conditioner(
                args.consequence_conditioning,
                component=args.consequence_component,
            ),
            1,
        )
        learner = AcceptanceLearner(
            "cpu",
            "q_bc",
            ref_dropout=0,
            bottleneck_dim=128,
            chunk_len=10,
            gripper_threshold=0,
            critic_conditioner=conditioner,
            critic_conditioning=f"candidate_{args.consequence_conditioning}",
            critic_bottleneck_dim=256,
            correction_limit=0.1,
            correction_axes=(0,),
            residual_output=True,
            critic_consequence_dueling=True,
            candidate_selection_count=5,
            candidate_selection_margin=0.0,
            candidate_uncertainty_penalty=1.0,
        )
        trace = []
        random.seed(seed)
        torch.manual_seed(seed)
        for step in range(args.steps):
            metrics = learner.update(train)
            if step in {0, args.steps - 1}:
                trace.append({"step": step + 1, **metrics})
        seed_states[seed] = learner.state_dict()
        for arm, sign in (("correct", 1), ("reversed", -1), ("zero", 0)):
            conditioner.sign = sign
            for label, replay in evaluations.items():
                rows.append(
                    {
                        "optimizer_seed": seed,
                        "arm": arm,
                        "evaluation": label,
                        "train_steps": args.steps,
                        "trace": trace,
                        **audit_replay(learner, replay),
                    }
                )
        conditioner.sign = 1
    summaries, passes_strict_screen = summarize_gate(rows)
    base_readout_sha256 = _file_sha256(args.base_readout)
    consequence_readout_sha256 = _file_sha256(args.consequence_readout)
    result = {
        "state": "complete",
        "revision": revision,
        "worktree_clean": True,
        "protocol": (
            "one correct-conditioned critic is trained per optimizer seed; its "
            "weights remain frozen while correct, reversed and zero physical "
            "consequence inputs are evaluated on distinct held-out replays"
        ),
        "causal_protocol": "train_correct_once_freeze_weights_intervene_input_sign",
        "train_base_replay": str(args.train_base_replay),
        "train_physical_replay": str(args.train_physical_replay),
        "trace_sha256": trace_hashes,
        "replay_summaries": replay_summaries,
        "base_readout": str(args.base_readout),
        "base_readout_sha256": base_readout_sha256,
        "consequence_readout": str(args.consequence_readout),
        "consequence_readout_sha256": consequence_readout_sha256,
        "consequence_component": args.consequence_component,
        "consequence_conditioning": args.consequence_conditioning,
        "require_robust_replay": args.require_robust_replay,
        "replay_sampling": "td_eligible_transition_uniform",
        "seeds": args.seeds,
        "rows": rows,
        "gate_summaries": summaries,
        "passes_strict_screen": passes_strict_screen,
        "decision": (
            "eligible_for_conservative_candidate_control"
            if passes_strict_screen
            else "stop_before_closed_loop"
        ),
        "correct_checkpoints": {},
        "checkpoints": {arm: {} for arm in ("correct", "reversed", "zero")},
    }
    args.output.mkdir(parents=True)
    if passes_strict_screen:
        for seed, state in seed_states.items():
            checkpoint = args.output / f"correct-trained-seed{seed}.pt"
            torch.save(
                {
                    "schema": 4,
                    "state_dict": state,
                    "gate_revision": revision,
                    "gate_passed": True,
                    "training_arm": "correct",
                    "evaluation_arms": ["correct", "reversed", "zero"],
                    "optimizer_seed": seed,
                    "train_trace_sha256": trace_hashes["train"],
                    "base_readout_sha256": base_readout_sha256,
                    "consequence_readout_sha256": consequence_readout_sha256,
                    "consequence_component": args.consequence_component,
                    "consequence_conditioning": args.consequence_conditioning,
                    "replay_sampling": "td_eligible_transition_uniform",
                    "candidate_selection_count": 5,
                    "candidate_selection_margin": 0.0,
                    "candidate_uncertainty_penalty": 1.0,
                },
                checkpoint,
            )
            for arm in ("correct", "reversed", "zero"):
                result["checkpoints"][arm][str(seed)] = str(checkpoint)
        result["correct_checkpoints"] = result["checkpoints"]["correct"]
    write_json(args.output / "result.json", result)
    _write_gate_table(args.output, summaries)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-base-replay", type=Path, required=True)
    parser.add_argument("--train-physical-replay", type=Path, required=True)
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--base-readout", type=Path, required=True)
    parser.add_argument("--consequence-readout", type=Path, required=True)
    parser.add_argument(
        "--consequence-component",
        choices=("joint", "response", "geometry"),
        default="joint",
    )
    parser.add_argument(
        "--consequence-conditioning",
        choices=("relative_concat", "relative_unit_concat"),
        default="relative_concat",
        help=(
            "Keep raw candidate-relative magnitudes or normalize each active "
            "physical direction before critic training."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-robust-replay",
        action="store_true",
        help="Require at least two complete success and failure episodes per replay.",
    )
    parser.add_argument("--steps", type=int, default=105)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 47, 59])
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
