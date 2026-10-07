# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Plan or run serial, three-seed planner comparisons on one explicitly chosen GPU."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from toolkits.rlt.planner_report import aggregate, compare, read_evidence, read_run


def campaign_commands(args: argparse.Namespace) -> list[list[str]]:
    """Use an interaction budget by default; a separate campaign can match updates."""
    if (
        len(args.seeds) < 3
        or len(set(args.seeds)) != len(args.seeds)
        or min(args.seeds) < 0
    ):
        raise ValueError("Require at least three distinct nonnegative training seeds")
    if args.budget < 1 or args.max_episodes < 1 or not 20 <= args.eval_episodes <= 500:
        raise ValueError(
            "Require positive budgets and 20..500 evaluation initial states"
        )
    common = [
        sys.executable,
        "-m",
        "toolkits.rlt.planner_experiment",
        "--stage1",
        str(args.stage1),
        "--dataset",
        str(args.dataset),
        "--gpu",
        str(args.gpu),
        "--port",
        str(args.port),
        "--steps",
        str(args.max_episodes),
        "--eval-episodes",
        str(args.eval_episodes),
        "--protocol",
        args.protocol,
    ]
    initial_weights = getattr(args, "initial_weights", None)
    if initial_weights is not None:
        common += ["--initial-weights", str(initial_weights)]
    if getattr(args, "wandb", False):
        common += ["--wandb"]
    return [
        common
        + [
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--control-budget" if args.budget_basis == "control" else "--update-budget",
            str(args.budget),
            "--export-corrections",
            "--output",
            str(args.output / f"seed_{seed}" / arm),
        ]
        for seed in args.seeds
        for arm in ("none", "planner")
    ]


def final_weights(run: Path) -> Path:
    """Select the latest completed step by budget, never the highest test score."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(run / "resolved.yaml")
    paths = list(
        (run / cfg.runner.logger.experiment_name / "checkpoints").glob(
            "global_step_*/actor/model_state_dict/full_weights.pt"
        )
    )
    if not paths:
        raise ValueError(f"No completed final weights in {run}")
    return max(paths, key=lambda p: int(p.parents[2].name.removeprefix("global_step_")))


def main() -> None:
    """Default to a printable plan; execute only with explicit --execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--port", type=int, default=6535)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 1235, 1236])
    parser.add_argument(
        "--protocol", choices=["complete", "preinsert_handoff"], default="complete"
    )
    parser.add_argument(
        "--budget-basis", choices=["control", "updates"], default="control"
    )
    parser.add_argument("--budget", type=int, default=30000)
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    commands = campaign_commands(args)
    if not args.execute:
        for command in commands:
            print(shlex.join(command))
        print(
            "After each training run, --execute also evaluates its final weights on insertion fixtures. No jobs started."
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(commands, indent=2) + "\n")
    insertion = []
    for command in commands:
        subprocess.run(command, check=True)
        run = Path(command[-1])
        evaluation = [
            sys.executable,
            "-m",
            "toolkits.rlt.planner_experiment",
            "--stage1",
            str(args.stage1),
            "--dataset",
            str(args.dataset),
            "--gpu",
            str(args.gpu),
            "--port",
            str(args.port),
            "--eval-episodes",
            str(args.eval_episodes),
            "--eval-scope",
            "insertion",
            "--checkpoint",
            str(final_weights(run)),
            "--output",
            str(run.parent / f"{run.name}_insertion"),
        ]
        subprocess.run(evaluation, check=True)
        evidence = read_evidence(Path(evaluation[-1]))
        insertion.append(
            {
                "training_run": str(run),
                "scope": "insertion",
                "scalars": evidence["scalars"],
            }
        )
        (args.output / "insertion_results.json").write_text(
            json.dumps(insertion, indent=2) + "\n"
        )
    pairs = [
        compare(
            read_run(args.output / f"seed_{seed}/none"),
            read_run(args.output / f"seed_{seed}/planner"),
        )
        for seed in args.seeds
    ]
    (args.output / "comparison.json").write_text(
        json.dumps(aggregate(pairs), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
