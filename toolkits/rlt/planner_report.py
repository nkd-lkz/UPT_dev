# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Report matched planner pilots without mistaking warmup for actor evaluation."""

import argparse
import json
from pathlib import Path


def summarize_budget(history: dict[str, dict[int, float]]) -> dict[str, float]:
    """Count actual work in this single-environment pilot, not just outer steps."""
    episodes = history.get("env/num_trajectories", {})
    if not episodes or any(value != 1 for value in episodes.values()):
        raise ValueError("Budget accounting requires one training episode per step")
    lengths = history.get("env/episode_len", {})
    successes = history.get("env/success_once", {})
    if lengths.keys() != episodes.keys() or successes.keys() != episodes.keys():
        raise ValueError("Incomplete per-episode training accounting")
    result = {
        "training_episodes": len(episodes),
        "training_control_ticks": sum(lengths.values()),
        "training_successes_including_assistance": sum(successes.values()),
    }
    for label, tag in {
        "actor_updates": "train/rlt/actor_updates_run",
        "critic_updates": "train/rlt/critic_updates_run",
        "planner_ticks": "env/planner_steps",
        "planner_attempts": "env/planner_attempts",
        "planner_failures": "env/planner_failed",
    }.items():
        result[label] = sum(history.get(tag, {}).values())
    return result


def read_run(path: Path) -> dict:
    """Read recorded launch/config and scalar evidence from this run only."""
    from omegaconf import OmegaConf
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    launch = json.loads((path / "launch.json").read_text())
    cfg = OmegaConf.to_container(OmegaConf.load(path / "resolved.yaml"), resolve=True)
    if not launch["complete"] or launch.get("exit_code") != 0:
        raise ValueError(f"Run has not completed successfully: {path}")
    if (
        cfg["env"]["eval"]["planner_assistance"]["enable"]
        or cfg["env"]["eval"]["rlt_policy_switch"]["expert_takeover"]["enable"]
    ):
        raise ValueError("Assisted evaluation is not autonomous evaluation")
    values = {}
    history = {}
    # Checkpoints contain thousands of NAS replay files. They cannot contain
    # MetricLogger events and must not be traversed to summarize scalar history.
    for event_file in (path / "tensorboard").rglob("events.out.tfevents.*"):
        events = EventAccumulator(
            str(event_file), size_guidance={"scalars": 0}
        ).Reload()
        for tag in events.Tags()["scalars"]:
            for point in events.Scalars(tag):
                rows = history.setdefault(tag, {})
                if point.step not in rows or point.wall_time > rows[point.step][0]:
                    rows[point.step] = (point.wall_time, point.value)
                if tag not in values or point.wall_time > values[tag][0]:
                    values[tag] = (point.wall_time, point.value)
    scalars = {key: row[1] for key, row in values.items()}
    for name in ["eval/success_once", "eval/num_trajectories", "train/rlt/update_step"]:
        if name not in scalars:
            raise ValueError(f"Missing required scalar {name} in {path}")
    if scalars["eval/num_trajectories"] != len(launch["eval_seeds"]):
        raise ValueError("Evaluation count does not match the declared initial states")
    # Remove only the declared experimental variable and output destinations.
    cfg["env"]["train"]["planner_assistance"]["enable"] = None
    cfg["runner"]["logger"]["log_path"] = None
    cfg["runner"]["logger"]["experiment_name"] = None
    for split in ("train", "eval"):
        cfg["env"][split]["video_cfg"]["video_base_dir"] = None
    budget = summarize_budget(
        {
            tag: {step: row[1] for step, row in rows.items()}
            for tag, rows in history.items()
        }
    )
    if budget["training_episodes"] != launch["steps"]:
        raise ValueError("Observed training episodes do not match the launch budget")
    return {"launch": launch, "config": cfg, "scalars": scalars, "budget": budget}


def compare(plain: dict, assisted: dict) -> dict:
    """Fail closed on unmatched budgets; state when the actor was still gated."""
    if plain["launch"]["arm"] != "none" or assisted["launch"]["arm"] != "planner":
        raise ValueError("Expected none and planner arms in that order")
    if plain["config"] != assisted["config"]:
        raise ValueError(
            "Resolved configurations differ beyond assistance/output paths"
        )
    if plain["launch"]["eval_seeds"] != assisted["launch"]["eval_seeds"]:
        raise ValueError("Evaluation initial states differ")
    scores = {"none": plain["scalars"], "planner": assisted["scalars"]}
    ready = all(
        row.get("train/rlt/ready_for_online", 0) == 1 for row in scores.values()
    )
    return {
        "matched_configuration": True,
        "both_learners_finished_warmup": ready,
        "results": scores,
        "actual_budgets": {
            "none": plain.get("budget"),
            "planner": assisted.get("budget"),
        },
        "budget_note": "Outer-step budgets match; episode lengths, expert actions and resulting scheduled updates may differ. Report actual work separately.",
        "autonomous_success_difference": scores["planner"]["eval/success_once"]
        - scores["none"]["eval/success_once"],
        "interpretation": (
            "Small single-seed pilot; readiness alone does not prove actor execution or improvement. Inspect rollout control ownership."
            if ready
            else "At least one learner remained gated: do not claim this compares two trained actors."
        ),
    }


def main() -> None:
    """Write a comparison only after both runs finish and their contracts match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--none", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(read_run(args.none), read_run(args.planner))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
