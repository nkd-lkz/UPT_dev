# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Report matched planner pilots without mistaking warmup for actor evaluation."""

import argparse
import copy
import hashlib
import json
import statistics
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
        "planner_handoffs": "env/planner_handoffs",
        "planner_handoff_hold_ticks": "env/planner_handoff_hold_ticks",
    }.items():
        result[label] = sum(history.get(tag, {}).values())
    return result


def read_evidence(path: Path) -> dict:
    """Read one successful run, preserving its declared evaluation scope."""
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
    backend_path = Path(cfg["runner"]["logger"].get("backend_log_path", path))
    for event_file in (backend_path / "tensorboard").rglob("events.out.tfevents.*"):
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
    for name in ["eval/success_once", "eval/num_trajectories"]:
        if name not in scalars:
            raise ValueError(f"Missing required scalar {name} in {path}")
    if scalars["eval/num_trajectories"] != len(launch["eval_seeds"]):
        raise ValueError("Evaluation count does not match the declared initial states")
    scope = launch.get("eval_scope", "full_task")
    if scope not in {"full_task", "insertion"} or (
        (scope == "insertion")
        != bool(cfg["env"]["eval"].get("insertion_fixture", False))
    ):
        raise ValueError("Evaluation scope and fixture configuration disagree")
    if scope == "insertion" and "eval/fixture_prefix_ticks" not in scalars:
        raise ValueError("Missing insertion fixture cost")
    return {
        "launch": launch,
        "config": cfg,
        "scalars": scalars,
        "history": {
            tag: {step: row[1] for step, row in rows.items()}
            for tag, rows in history.items()
        },
    }


def read_run(path: Path) -> dict:
    """Verify online training accounting before a paired comparison."""
    evidence = read_evidence(path)
    launch, cfg, scalars = (evidence[k] for k in ("launch", "config", "scalars"))
    if "train/rlt/update_step" not in scalars or launch.get("checkpoint"):
        raise ValueError("Expected an online training run, not checkpoint evaluation")
    # Remove only the declared experimental variable and output destinations.
    cfg["env"]["train"]["planner_assistance"]["enable"] = None
    cfg["runner"]["logger"]["log_path"] = None
    cfg["runner"]["logger"]["experiment_name"] = None
    cfg["runner"]["logger"].pop("backend_log_path", None)
    cfg["algorithm"].pop("correction_export_dir", None)
    for split in ("train", "eval"):
        cfg["env"][split]["video_cfg"]["video_base_dir"] = None
    budget = summarize_budget(evidence["history"])
    control, updates = launch.get("control_budget", 0), launch.get("update_budget", 0)
    if control and budget["training_control_ticks"] != control:
        raise ValueError("Observed interaction budget differs from the declared limit")
    if updates and budget["critic_updates"] != updates:
        raise ValueError("Observed update budget differs from the declared limit")
    if budget["training_episodes"] > launch["steps"] or (
        not control and not updates and budget["training_episodes"] != launch["steps"]
    ):
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
    for key, default in (
        ("protocol", "complete"),
        ("eval_scope", "full_task"),
        ("seed", 1234),
        ("control_budget", 0),
        ("update_budget", 0),
        ("initial_weights_sha256", None),
    ):
        if plain["launch"].get(key, default) != assisted["launch"].get(key, default):
            raise ValueError(f"Paired {key} differs")
    scores = {"none": plain["scalars"], "planner": assisted["scalars"]}
    ready = all(
        row.get("train/rlt/ready_for_online", 0) == 1 for row in scores.values()
    )
    matched_config = copy.deepcopy(plain["config"])
    matched_config.get("actor", {}).pop("seed", None)
    matched_config.get("env", {}).get("train", {}).pop("seed", None)
    budgets = [run.get("budget", {}) or {} for run in (plain, assisted)]
    return {
        "matched_configuration": True,
        "experiment_config_sha256": hashlib.sha256(
            json.dumps(matched_config, sort_keys=True).encode()
        ).hexdigest(),
        "matched_optimizer_counts": bool(budgets[0].get("critic_updates", 0))
        and all(
            budgets[0].get(k) == budgets[1].get(k)
            for k in ("critic_updates", "actor_updates")
        ),
        "matched_interaction_ticks": bool(budgets[0].get("training_control_ticks", 0))
        and budgets[0].get("training_control_ticks")
        == budgets[1].get("training_control_ticks"),
        "seed": plain["launch"].get("seed", 1234),
        "protocol": plain["launch"].get("protocol", "complete"),
        "eval_scope": plain["launch"].get("eval_scope", "full_task"),
        "eval_seeds": plain["launch"]["eval_seeds"],
        "control_budget": plain["launch"].get("control_budget", 0),
        "update_budget": plain["launch"].get("update_budget", 0),
        "both_learners_finished_warmup": ready,
        "results": scores,
        "actual_budgets": {
            "none": plain.get("budget"),
            "planner": assisted.get("budget"),
        },
        "budget_basis": "control_ticks"
        if plain["launch"].get("control_budget")
        else "critic_updates"
        if plain["launch"].get("update_budget")
        else "episodes",
        "budget_note": "Only the declared budget axis is matched. Report other work and planner cost separately; actor-update counts must also match for an optimizer-matched claim.",
        "autonomous_success_difference": scores["planner"]["eval/success_once"]
        - scores["none"]["eval/success_once"],
        "interpretation": (
            "Small single-seed pilot; readiness alone does not prove actor execution or improvement. Inspect rollout control ownership."
            if ready
            else "At least one learner remained gated: do not claim this compares two trained actors."
        ),
    }


def aggregate(pairs: list[dict]) -> dict:
    """Summarize independent training seeds without pooling them as IID episodes."""
    if len(pairs) < 3 or len({p["seed"] for p in pairs}) != len(pairs):
        raise ValueError("Require at least three distinct paired training seeds")
    for key in (
        "protocol",
        "eval_scope",
        "budget_basis",
        "control_budget",
        "update_budget",
        "experiment_config_sha256",
    ):
        if len({p[key] for p in pairs}) != 1:
            raise ValueError(f"Cannot combine different {key} experiments")
    if len({tuple(p["eval_seeds"]) for p in pairs}) != 1:
        raise ValueError("Cannot combine different evaluation initial states")
    result = {key: pairs[0][key] for key in ("protocol", "eval_scope", "budget_basis")}
    result["per_seed"] = pairs
    result["success"] = {}
    for arm in ("none", "planner"):
        rates = [p["results"][arm]["eval/success_once"] for p in pairs]
        result["success"][arm] = {
            "mean": statistics.mean(rates),
            "training_seed_std": statistics.stdev(rates),
        }
    differences = [p["autonomous_success_difference"] for p in pairs]
    result["paired_difference"] = {
        "mean": statistics.mean(differences),
        "training_seed_std": statistics.stdev(differences),
    }
    result["limitation"] = (
        "Shared fixed evaluation initial states; no generalization or significance claim. Insertion results exclude planner-generated approach."
    )
    return result


def checkpoint_evaluations(paths: list[Path]) -> dict:
    """Report complete-task and insertion scores separately for fixed weights."""
    rows = [read_evidence(path) for path in paths]
    if not rows or any(not row["launch"].get("checkpoint") for row in rows):
        raise ValueError("Require standalone checkpoint evaluation runs")
    features = rows[0]["config"]["rollout"]["rlt_feature_model"]
    model = rows[0]["config"]["actor"]["model"]
    populations = {}
    contracts = {}
    results = []
    for path, row in zip(paths, rows):
        if (
            row["config"]["rollout"]["rlt_feature_model"] != features
            or row["config"]["actor"]["model"] != model
        ):
            raise ValueError("Checkpoint evaluations use different models or features")
        scope = row["launch"].get("eval_scope", "full_task")
        env = copy.deepcopy(row["config"]["env"]["eval"])
        env.get("video_cfg", {}).pop("video_base_dir", None)
        contract = {
            "environment": env,
            "seeds": row["launch"]["eval_seeds"],
            "fixture_population": row["launch"].get("fixture_population"),
        }
        if scope in contracts and contracts[scope] != contract:
            raise ValueError(
                "Within-scope evaluation configuration or population differs"
            )
        contracts[scope] = contract
        populations[scope] = contract["seeds"]
        population = contract["fixture_population"]
        if population and (
            scope != "insertion" or population["selected_seeds"] != contract["seeds"]
        ):
            raise ValueError("Fixture population does not match scored initial states")
        s = row["scalars"]
        results.append(
            {
                "run": str(path),
                "checkpoint": row["launch"]["checkpoint"],
                "scope": scope,
                "fixture_population": population,
                "episodes": s["eval/num_trajectories"],
                "success_rate": s["eval/success_once"],
                "phase_and_cost": {
                    k: v
                    for k, v in s.items()
                    if k.startswith("eval/")
                    and any(
                        part in k
                        for part in (
                            "actor_phase",
                            "fixture_prefix",
                            "insertion_policy",
                            "episode_len",
                        )
                    )
                },
            }
        )
    return {
        "complete": True,
        "evaluation_seeds_by_scope": populations,
        "results": results,
        "limitation": "Insertion starts after a privileged planner prefix and is conditional on any declared fixture coverage, not full-task autonomy. Initial seeds and environment settings are matched within each scope. They do not establish significance or training-seed robustness.",
    }


def main() -> None:
    """Write a comparison only after both runs finish and their contracts match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--none", type=Path)
    parser.add_argument("--planner", type=Path)
    parser.add_argument("--evaluation-runs", nargs="+", type=Path)
    parser.add_argument("--comparisons", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.comparisons:
        if args.evaluation_runs or args.none or args.planner:
            parser.error("Choose one report mode")
        result = aggregate([json.loads(path.read_text()) for path in args.comparisons])
    elif args.evaluation_runs:
        if args.none or args.planner:
            parser.error("Choose training comparison or checkpoint evaluations")
        result = checkpoint_evaluations(args.evaluation_runs)
    else:
        if args.none is None or args.planner is None:
            parser.error("Training comparison requires --none and --planner")
        result = compare(read_run(args.none), read_run(args.planner))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
