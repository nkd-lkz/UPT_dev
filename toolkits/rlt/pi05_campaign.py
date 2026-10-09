# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""A finite, dependency-gated new-RLT Pi0.5 development campaign."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

from toolkits.rlt.experiment_queue import run_plan
from toolkits.rlt.pi05_assets import REVISION, file_hash
from toolkits.rlt.pi05_baseline import PROTOCOL, write_json

FINAL_ARMS = ("reference_final", "warmup_final", "q_bc_eval", "bc_only_eval")


def audit_acceptance(root: Path) -> dict:
    """Check recorded acceptance provenance and costs without loading GPU models.

    Missing or running stages remain incomplete. A completed stage with bad
    provenance fails even when it reports a high success rate. Checkpoint paths
    and update counters are checked; legacy manifests do not hash head weights.
    Elapsed costs include loading and checkpoint I/O, not just GPU computation.
    """
    plan_path = root / "queue/plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    if "task" not in plan or "training_seed" not in plan:
        return {"state": "not_applicable", "reason": "No acceptance plan"}
    errors, pending, costs, reports = [], [], {}, {}
    planned_jobs = {job["name"]: job for job in plan.get("jobs", [])}

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    sets = acceptance_cases(root, plan["task"])
    case_hashes = {}
    for split, expected in sets.items():
        path = root / "cases" / f"{split}.json"
        if not path.exists():
            errors.append(f"{split}: missing declared cases")
            continue
        actual = json.loads(path.read_text())
        identity = ("id", "task", "kind", "seed")
        check(
            len(actual) == len(expected)
            and all(
                all(a.get(k) == b[k] for k in identity)
                for a, b in zip(actual, expected, strict=True)
            ),
            f"{split}: declared cases differ from the fixed acceptance profile",
        )
        case_hashes[split] = file_hash(path)
    stages = {
        "reference_gate": ("development", "reference"),
        "reference_collect": ("collection", "reference"),
        "warmup": (None, "warmup"),
        "actor_gate": ("development", "actor-eval"),
        "q_bc": ("online", "online"),
        "bc_only": ("online", "online"),
        **{
            name: (
                "heldout",
                "reference" if name == "reference_final" else "actor-eval",
            )
            for name in FINAL_ARMS
        },
    }
    by_id, by_hash = {}, {}
    for name, (split, mode) in stages.items():
        path = root / name / "manifest.json"
        if not path.exists():
            pending.append(name)
            continue
        report = json.loads(path.read_text())
        reports[name] = report
        args = report.get("args", {})
        check(report.get("protocol") == PROTOCOL, f"{name}: incompatible protocol")
        check(args.get("task") == plan["task"], f"{name}: task mismatch")
        check(args.get("seed") == plan["training_seed"], f"{name}: seed mismatch")
        check(args.get("mode") == mode, f"{name}: mode mismatch")
        role = (
            "training"
            if name in ("reference_collect", "warmup", "q_bc", "bc_only")
            else split
        )
        check(args.get("evaluation_role") == role, f"{name}: evaluation role mismatch")
        planned_args = planned_jobs.get(name, {}).get("argv", [])
        requested_logging = (
            planned_args[planned_args.index("--wandb") + 1]
            if "--wandb" in planned_args
            else None
        )
        check(
            requested_logging is not None and args.get("wandb") == requested_logging,
            f"{name}: logging mode differs from the declared plan",
        )
        if split:
            check(
                report.get("case_manifest_sha256") == case_hashes.get(split)
                and split in case_hashes,
                f"{name}: case manifest digest mismatch",
            )
        if name in ("q_bc", "bc_only"):
            check(args.get("objective") == name, f"{name}: actor objective mismatch")
            check(
                args.get("interaction_ticks") == 20000,
                f"{name}: requested budget mismatch",
            )
            check(
                args.get("updates_per_chunk") == 5, f"{name}: update schedule mismatch"
            )
        if name in (
            "actor_gate",
            "q_bc",
            "bc_only",
            "warmup_final",
            "q_bc_eval",
            "bc_only_eval",
        ):
            owner = name.removesuffix("_eval") if name.endswith("_eval") else "warmup"
            checkpoint = (
                root / owner / ("warmup.pt" if owner == "warmup" else "online.pt")
            )
            check(
                Path(args.get("checkpoint") or "").resolve() == checkpoint.resolve(),
                f"{name}: checkpoint path mismatch",
            )
        if report.get("error") or report.get("exit_code", 0) != 0:
            errors.append(
                f"{name}: unsuccessful stage ({report.get('error', report.get('exit_code'))})"
            )
        complete = report.get("complete") is True
        cost = {"complete": complete}
        start = report.get("started_at")
        end = (
            report.get("finished_at")
            if complete
            else report.get("finished_at", time.time())
        )
        valid_time = (
            all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in (start, end)
            )
            and end >= start
        )
        if complete:
            check(valid_time, f"{name}: invalid stage timestamps")
        if valid_time:
            cost["stage_wall_seconds"] = end - start
        progress = report
        if not complete:
            pending.append(name)
            live = root / name / "online_status.json"
            if live.exists():
                progress = json.loads(live.read_text())
        for key in (
            "interaction_ticks",
            "settling_ticks",
            "actor_updates",
            "critic_updates",
            "encoder_updates",
        ):
            if key in progress:
                cost[key] = progress[key]
        wandb_path = root / name / "wandb_run.json"
        if wandb_path.exists():
            wandb_run = json.loads(wandb_path.read_text())
            cost["wandb_url"] = wandb_run.get("url")
        else:
            wandb_run = {}
        costs[name] = cost
        if not complete:
            continue
        if requested_logging == "online":
            check(
                wandb_run.get("mode") == "online" and bool(wandb_run.get("url")),
                f"{name}: missing online W&B run record",
            )
        check(report.get("gate_passed") is True, f"{name}: stage gate did not pass")
        if name == "warmup":
            check(
                args.get("encoder_steps") == 1000 and args.get("bc_steps") == 6000,
                "warmup: requested training budget mismatch",
            )
            check(
                report.get("encoder_updates") == 1000, "warmup: encoder budget mismatch"
            )
            check(report.get("actor_updates") == 6000, "warmup: actor budget mismatch")
            check(
                report.get("critic_updates") == 0, "warmup: unexpected critic updates"
            )
            expected_ids = [c["id"] for c in sets["collection"]]
            check(
                set(report.get("train_episodes", [])) == set(expected_ids[:52]),
                "warmup: training episode split mismatch",
            )
            check(
                set(report.get("heldout_episodes", [])) == set(expected_ids[52:]),
                "warmup: validation episode split mismatch",
            )
            continue
        episodes = report.get("episodes", [])
        expected = sets[split]
        count_ok = (
            0 < len(episodes) <= len(expected)
            if split == "online"
            else len(episodes) == len(expected)
        )
        check(count_ok, f"{name}: episode count mismatch")
        for episode, case in zip(episodes, expected):
            check(
                all(episode.get(k) == case[k] for k in ("id", "task", "kind", "seed")),
                f"{name}: episode identity mismatch",
            )
            digest, case_id = episode.get("state_sha256"), episode.get("id")
            check(bool(digest), f"{name}: missing reset hash")
            check(
                by_id.get(case_id, digest) == digest,
                f"{name}: reset changed for {case_id}",
            )
            check(
                by_hash.get(digest, case_id) == case_id,
                f"{name}: different cases share a physical reset",
            )
            by_id[case_id], by_hash[digest] = digest, case_id
        check(
            all(type(e.get("success")) is bool for e in episodes),
            f"{name}: invalid success flags",
        )
        successes = sum(e.get("success") is True for e in episodes)
        check(
            report.get("successes") == successes, f"{name}: success accounting mismatch"
        )
        check(
            bool(episodes) and report.get("success_rate") == successes / len(episodes),
            f"{name}: success rate mismatch",
        )
        valid_ticks = all(
            type(e.get("ticks")) is int
            and 0 < e["ticks"] <= PROTOCOL["max_control_steps"]
            for e in episodes
        )
        check(valid_ticks, f"{name}: invalid episode tick count")
        ticks = sum(e["ticks"] for e in episodes) if valid_ticks else None
        check(
            report.get("interaction_ticks") == ticks,
            f"{name}: interaction accounting mismatch",
        )
        check(
            report.get("settling_ticks") == len(episodes) * PROTOCOL["settling_steps"],
            f"{name}: settling accounting mismatch",
        )
        if split == "online":
            check(ticks == 20000, f"{name}: executed budget mismatch")
            critic_updates = report.get("critic_updates", 0)
            expected_updates = (
                5 * sum(math.ceil(e["ticks"] / PROTOCOL["chunk_len"]) for e in episodes)
                if valid_ticks
                else 0
            )
            check(
                critic_updates == expected_updates > 0
                and report.get("actor_updates") == 6000 + critic_updates // 2,
                f"{name}: learner update accounting mismatch",
            )
        if split == "heldout":
            check(not args.get("collect"), f"{name}: held-out data collection enabled")
            if name == "reference_final":
                check(
                    report.get("actor_updates") == report.get("critic_updates") == 0,
                    f"{name}: reference has learner updates",
                )
            else:
                owner = (
                    "warmup" if name == "warmup_final" else name.removesuffix("_eval")
                )
                origin = reports.get(owner, {})
                check(
                    all(
                        report.get(k) == origin.get(k)
                        for k in ("actor_updates", "critic_updates")
                    ),
                    f"{name}: checkpoint update counters mismatch",
                )
    anchor = reports.get("reference_gate", {})
    actor_gate = reports.get("actor_gate", {})
    if anchor.get("complete") and actor_gate.get("complete"):
        try:
            check(
                paired_gate(anchor, actor_gate)["gate_passed"],
                "paired_gate: warmup lost too many development successes",
            )
        except ValueError as error:
            errors.append(f"paired_gate: {error}")
    for name, report in reports.items():
        for key in ("adapter_sha256", "versions"):
            check(
                bool(report.get(key)) and report.get(key) == anchor.get(key),
                f"{name}: {key} mismatch",
            )
        if name in FINAL_ARMS:
            for source in ("q_bc", "bc_only"):
                training = reports.get(source, {})
                finished, started = (
                    training.get("finished_at"),
                    report.get("started_at"),
                )
                check(
                    training.get("complete") is True
                    and isinstance(finished, (int, float))
                    and isinstance(started, (int, float))
                    and finished <= started,
                    f"{name}: held-out evaluation started before {source} completed",
                )
    arm_costs = {}
    for arm, dependencies in {
        "warmup_final": ("reference_collect", "warmup"),
        "q_bc_eval": ("reference_collect", "warmup", "q_bc"),
        "bc_only_eval": ("reference_collect", "warmup", "bc_only"),
    }.items():
        if all(
            costs.get(n, {}).get("complete") and "stage_wall_seconds" in costs[n]
            for n in dependencies
        ):
            arm_costs[arm] = {
                key: sum(costs[n].get(key, 0) for n in dependencies)
                for key in (
                    "stage_wall_seconds",
                    "interaction_ticks",
                    "settling_ticks",
                )
            }
    return {
        "state": "failed" if errors else "incomplete" if pending else "passed",
        "violations": errors,
        "pending_stages": pending,
        "stage_costs": costs,
        "completed_training_cost_by_arm": arm_costs,
        "cost_scope": "Collection plus warmup plus each arm's online training. Shared preparation is counted once per arm; do not sum arms as total campaign cost. Excludes upstream VLA pretraining, evaluation, queue waits and manual debugging. Manifest wall time includes loading/checkpoint I/O but not imports or W&B finalization; it is not GPU-active time.",
        "checkpoint_identity": "Recorded paths and update counters; head checkpoint bytes were not hashed by legacy manifests.",
        "claim_scope": "Single task and training seed; engineering acceptance, not proof of RL improvement or full-suite reproduction.",
    }


def paired_gate(reference: dict, actor: dict) -> dict:
    """Require identical reset cases and at most one lost success in the smoke gate."""
    for report in (reference, actor):
        if report.get("complete") is not True or report.get("exit_code") != 0:
            raise ValueError("Require completed successful evaluation commands")
    a, b = reference["episodes"], actor["episodes"]
    if len(a) != len(b) or len(a) < 5:
        raise ValueError("Need at least five matched development cases")
    keys = ("id", "task", "kind", "state", "seed", "state_sha256")
    if any(any(x.get(k) != y.get(k) for k in keys) for x, y in zip(a, b, strict=True)):
        raise ValueError("Reference and actor reset cases differ")
    ref_success = sum(x["success"] for x in a)
    actor_success = sum(x["success"] for x in b)
    passed = ref_success >= math.ceil(
        0.6 * len(a)
    ) and actor_success >= ref_success - max(1, len(a) // 10)
    return {
        "complete": True,
        "exit_code": 0 if passed else 2,
        "gate_passed": passed,
        "reference_successes": ref_success,
        "actor_successes": actor_success,
        "episodes": len(a),
        "scope": "development warmup gate, not statistical evidence of RL improvement",
    }


def summarize_campaign(root: Path) -> dict:
    """Report completed evidence without converting absent results to zero scores."""
    path = root / "queue/status.json"
    queue = json.loads(path.read_text()) if path.exists() else {}
    plan_path = root / "queue/plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    results = {}
    for name in (
        "reference_collect",
        "warmup",
        "reference_gate",
        "actor_gate",
        "q_bc",
        "bc_only",
        "warmup_final",
        "reference_final",
        "q_bc_eval",
        "bc_only_eval",
    ):
        row = dict(queue.get("jobs", {}).get(name, {"state": "not_started"}))
        manifest = root / name / "manifest.json"
        if manifest.exists():
            report = json.loads(manifest.read_text())
            row["manifest_complete"] = report.get("complete") is True
            if report.get("error"):
                row["error"] = report["error"]
            if report.get("complete") is True:
                for key in (
                    "gate_passed",
                    "successes",
                    "success_rate",
                    "actor_updates",
                    "critic_updates",
                    "final_following",
                    "interaction_ticks",
                    "settling_ticks",
                    "evaluation_role",
                ):
                    if key in report:
                        row[key] = report[key]
                if "episodes" in report:
                    row["episodes"] = len(report["episodes"])
        results[name] = row
    final_reports = {}
    for name in ("reference_final", "warmup_final", "q_bc_eval", "bc_only_eval"):
        path = root / name / "manifest.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report.get("complete") is True and report.get("exit_code") == 0:
                final_reports[name] = report
    audit = audit_acceptance(root)
    comparison_error = None
    try:
        comparison = paired_comparison(final_reports)
    except ValueError as error:
        comparison, comparison_error = {}, str(error)
    return {
        "queue_state": queue.get("state", "not_started"),
        "gpu_index": queue.get("gpu_index"),
        "jobs": results,
        "engineering_pipeline_complete": comparison_error is None
        and audit["state"] in ("passed", "not_applicable")
        and all(
            results[name]["state"] == "passed"
            and results[name].get("manifest_complete") is True
            for name in ("reference_final", "warmup_final", "q_bc_eval", "bc_only_eval")
        ),
        "paper_reproduction_validated": False,
        "robodojo_closed_loop_validated": False,
        "scope": plan.get(
            "scope", "single-task development pilot; not evidence of an RL improvement"
        ),
        "comparison": comparison,
        "comparison_error": comparison_error,
        "acceptance_audit": audit,
    }


def paired_comparison(reports: dict) -> dict:
    """Report binomial uncertainty and paired outcomes without treating seeds as tasks."""
    result = {}
    reference = reports.get("reference_final", {}).get("episodes", [])
    for name, report in reports.items():
        episodes = report["episodes"]
        n = len(episodes)
        if not n:
            continue
        successes = sum(r["success"] for r in episodes)
        p, z = successes / n, 1.959963984540054
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        row = {
            "episodes": n,
            "successes": successes,
            "success_rate": p,
            "wilson95": [max(0, center - half), min(1, center + half)],
        }
        if reference and name != "reference_final":
            keys = ("id", "task", "kind", "seed", "state_sha256")
            if len(reference) != n or any(
                any(a[k] != b[k] for k in keys)
                for a, b in zip(reference, episodes, strict=True)
            ):
                raise ValueError("Final evaluation cases do not match reference")
            row.update(
                gained=sum(
                    not a["success"] and b["success"]
                    for a, b in zip(reference, episodes, strict=True)
                ),
                lost=sum(
                    a["success"] and not b["success"]
                    for a, b in zip(reference, episodes, strict=True)
                ),
            )
        result[name] = row
    return result


def acceptance_cases(root: Path, task: int) -> dict[str, list[dict]]:
    """Fix disjoint generated-reset seeds before inspecting any held-out outcomes."""
    sets = {}
    for split, count, offset in (
        ("collection", 64, 100000),
        ("online", 2048, 200000),
        ("development", 20, 300000),
        ("heldout", 100, 400000),
    ):
        sets[split] = [
            {
                "id": f"{split}-task{task}-seed{offset + task * 10000 + i}",
                "task": task,
                "kind": "generated",
                "seed": offset + task * 10000 + i,
                "snapshot": str(root.resolve() / "snapshots" / split / f"{i}.json"),
            }
            for i in range(count)
        ]
    return sets


def acceptance_plan(
    root: Path,
    repo: Path,
    assets: Path,
    source: Path,
    gpu: int,
    task: int,
    seed: int,
    wandb: str,
    entity: str | None,
) -> dict:
    """Build a one-seed, four-arm acceptance with 100 paired held-out initial states."""
    base = campaign_plan(root, repo, assets, source, gpu)
    jobs = [base["jobs"][0]]
    common = [
        "--source",
        str(source),
        "--assets",
        str(assets),
        "--gpu",
        str(gpu),
        "--task",
        str(task),
        "--seed",
        str(seed),
        "--wandb",
        wandb,
        "--group",
        f"rlt-pi05-task{task}-seed{seed}",
    ]
    if entity:
        common += ["--wandb-entity", entity]

    def add(name, mode, deps, options, split=None, role="development"):
        case_args = (
            ["--cases", str(root / "cases" / f"{split}.json"), "--create-snapshots"]
            if split
            else []
        )
        jobs.append(
            {
                "name": name,
                "cwd": str(repo),
                "depends_on": deps,
                "argv": [
                    "bash",
                    "run_rlt_pi05.sh",
                    mode,
                    *common,
                    "--output",
                    str(root / name),
                    "--evaluation-role",
                    role,
                    *case_args,
                    *options,
                ],
                "timeout_seconds": 14400,
                "manifest": str(root / name / "manifest.json"),
            }
        )

    add(
        "reference_gate",
        "reference",
        ["assets"],
        ["--min-successes", "12"],
        "development",
    )
    add(
        "reference_collect",
        "reference",
        ["reference_gate"],
        ["--collect", "--min-successes", "0"],
        "collection",
        "training",
    )
    add(
        "warmup",
        "warmup",
        ["reference_collect"],
        [
            "--cache",
            str(root / "reference_collect/features.pt"),
            "--encoder-steps",
            "1000",
            "--bc-steps",
            "6000",
        ],
        role="training",
    )
    checkpoint = str(root / "warmup/warmup.pt")
    add(
        "actor_gate",
        "actor-eval",
        ["warmup"],
        ["--checkpoint", checkpoint, "--min-successes", "0"],
        "development",
    )
    jobs.append(next(job for job in base["jobs"] if job["name"] == "paired_gate"))
    for objective in ("q_bc", "bc_only"):
        add(
            objective,
            "online",
            ["paired_gate"],
            [
                "--checkpoint",
                checkpoint,
                "--objective",
                objective,
                "--interaction-ticks",
                "20000",
                "--min-successes",
                "0",
            ],
            "online",
            "training",
        )
    # No held-out outcomes are exposed until both fixed-budget training jobs pass.
    add(
        "reference_final",
        "reference",
        ["q_bc", "bc_only"],
        ["--min-successes", "0"],
        "heldout",
        "heldout",
    )
    add(
        "warmup_final",
        "actor-eval",
        ["reference_final"],
        ["--checkpoint", checkpoint, "--min-successes", "0"],
        "heldout",
        "heldout",
    )
    for objective in ("q_bc", "bc_only"):
        add(
            f"{objective}_eval",
            "actor-eval",
            ["reference_final"],
            [
                "--checkpoint",
                str(root / objective / "online.pt"),
                "--objective",
                objective,
                "--min-successes",
                "0",
            ],
            "heldout",
            "heldout",
        )
    return {
        "gpu": gpu,
        "hours": 12,
        "task": task,
        "training_seed": seed,
        "scope": "one-task one-training-seed engineering acceptance; not full-suite or multi-seed reproduction",
        "jobs": jobs,
    }


def campaign_plan(root: Path, repo: Path, assets: Path, source: Path, gpu: int) -> dict:
    """Keep collection, warmup acceptance, online training and pilot eval disjoint."""
    common = ["--source", str(source), "--assets", str(assets), "--gpu", str(gpu)]
    jobs = [
        {
            "name": "assets",
            "cwd": str(repo),
            "requires_gpu": False,
            "argv": [
                sys.executable,
                "-m",
                "toolkits.rlt.pi05_campaign",
                "wait-assets",
                "--assets",
                str(assets),
            ],
            "timeout_seconds": 7200,
        }
    ]

    def add(name, mode, depends, options):
        jobs.append(
            {
                "name": name,
                "cwd": str(repo),
                "depends_on": depends,
                "argv": [
                    "bash",
                    "run_rlt_pi05.sh",
                    mode,
                    *common,
                    "--output",
                    str(root / name),
                    *options,
                ],
                "timeout_seconds": 7200,
                "manifest": str(root / name / "manifest.json"),
            }
        )

    add(
        "reference_collect",
        "reference",
        ["assets"],
        ["--episodes", "10", "--collect", "--min-successes", "5"],
    )
    add(
        "warmup",
        "warmup",
        ["reference_collect"],
        [
            "--cache",
            str(root / "reference_collect/features.pt"),
            "--encoder-steps",
            "300",
            "--bc-steps",
            "3000",
        ],
    )
    add(
        "reference_gate",
        "reference",
        ["warmup"],
        ["--state-start", "10", "--episodes", "5", "--min-successes", "3"],
    )
    checkpoint = str(root / "warmup/warmup.pt")
    add(
        "actor_gate",
        "actor-eval",
        ["reference_gate"],
        ["--state-start", "10", "--episodes", "5", "--checkpoint", checkpoint],
    )
    jobs.append(
        {
            "name": "paired_gate",
            "cwd": str(repo),
            "requires_gpu": False,
            "depends_on": ["actor_gate"],
            "timeout_seconds": 60,
            "argv": [
                sys.executable,
                "-m",
                "toolkits.rlt.pi05_campaign",
                "paired-gate",
                "--root",
                str(root),
            ],
            "manifest": str(root / "paired_gate.json"),
        }
    )
    for objective in ("q_bc", "bc_only"):
        add(
            objective,
            "online",
            ["paired_gate"],
            [
                "--state-start",
                "20",
                "--episodes",
                "10",
                "--seed",
                "43000",
                "--min-successes",
                "0",
                "--checkpoint",
                checkpoint,
                "--objective",
                objective,
            ],
        )
        add(
            f"{objective}_eval",
            "actor-eval",
            [objective],
            [
                "--state-start",
                "15",
                "--episodes",
                "5",
                "--seed",
                "44000",
                "--min-successes",
                "0",
                "--checkpoint",
                str(root / objective / "online.pt"),
                "--objective",
                objective,
            ],
        )
    add(
        "reference_final",
        "reference",
        ["paired_gate"],
        [
            "--state-start",
            "15",
            "--episodes",
            "5",
            "--seed",
            "44000",
            "--min-successes",
            "0",
        ],
    )
    add(
        "warmup_final",
        "actor-eval",
        ["paired_gate"],
        [
            "--state-start",
            "15",
            "--episodes",
            "5",
            "--seed",
            "44000",
            "--min-successes",
            "0",
            "--checkpoint",
            checkpoint,
        ],
    )
    return {"gpu": gpu, "hours": 12, "jobs": jobs}


def main() -> None:
    """Wait for checked assets, run a gate, or own a twelve-hour GPU queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["run", "wait-assets", "paired-gate", "status", "audit"]
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--profile", choices=["smoke", "acceptance"], default="smoke")
    parser.add_argument("--task", type=int, choices=[0, 1, 3], default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wandb", choices=["online", "offline", "disabled"], default="disabled"
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--report",
        type=Path,
        help="Save a new audit snapshot without overwriting an existing report",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.report is not None and args.mode != "audit":
        parser.error("--report applies only to audit")
    if args.mode == "audit":
        if args.root is None:
            parser.error("audit requires --root")
        report = audit_acceptance(args.root)
        if args.report is not None:
            if args.report.exists():
                parser.error("Audit report already exists; choose a new snapshot path")
            write_json(args.report, report)
        print(json.dumps(report, indent=2))
        raise SystemExit(
            {"passed": 0, "failed": 2, "incomplete": 3, "not_applicable": 4}[
                report["state"]
            ]
        )
    elif args.mode == "status":
        if args.root is None:
            parser.error("status requires --root")
        print(json.dumps(summarize_campaign(args.root), indent=2))
    elif args.mode == "wait-assets":
        deadline = time.monotonic() + 7100
        while time.monotonic() < deadline:
            path = args.assets / "verified.json"
            if path.exists():
                report = json.loads(path.read_text())
                if (
                    report.get("complete") is not True
                    or report.get("revision") != REVISION
                ):
                    raise ValueError("Asset verification failed")
                return
            time.sleep(15)
        raise TimeoutError("Model asset download did not finish within its budget")
    elif args.mode == "paired-gate":
        report = paired_gate(
            json.loads((args.root / "reference_gate/manifest.json").read_text()),
            json.loads((args.root / "actor_gate/manifest.json").read_text()),
        )
        write_json(args.root / "paired_gate.json", report)
        raise SystemExit(report["exit_code"])
    else:
        if args.root is None or args.assets is None or args.source is None:
            parser.error("run requires --root, --assets and --source")
        args.root.mkdir(parents=True, exist_ok=False)
        repo = Path(__file__).resolve().parents[2]
        if args.profile == "acceptance":
            (args.root / "cases").mkdir()
            for split, cases in acceptance_cases(args.root, args.task).items():
                write_json(args.root / "cases" / f"{split}.json", cases)
            plan = acceptance_plan(
                args.root,
                repo,
                args.assets,
                args.source,
                args.gpu,
                args.task,
                args.seed,
                args.wandb,
                args.wandb_entity,
            )
        else:
            plan = campaign_plan(args.root, repo, args.assets, args.source, args.gpu)
        try:
            run_plan(plan, args.root / "queue")
        finally:
            write_json(args.root / "summary.json", summarize_campaign(args.root))


if __name__ == "__main__":
    main()
