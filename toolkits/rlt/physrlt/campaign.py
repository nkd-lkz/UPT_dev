# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepare and supervise finite, dependency-aware small PhysRLT experiments."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from toolkits.rlt.experiment_queue import LEASE_DIRECTORY, gpu_idle
from toolkits.rlt.pi05_baseline import write_json


def prepare(root: Path, source: Path, assets: Path) -> None:
    """Freeze a two-task pilot plan, not a full benchmark or final-test set."""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("An existing plan must not be silently changed")
    smoke_cases = [
        {
            "id": "physrlt-smoke-task3-layout0",
            "suite": "libero_10",
            "task": 3,
            "state": 0,
            "kind": "published",
            "perturbation": "lan",
            "seed": 72000,
            "split": "train",
        }
    ]
    write_json(root / "smoke_cases.json", smoke_cases)
    jobs = [
        {
            "id": "capture-smoke",
            "kind": "gpu",
            "dependencies": [],
            "output": str(root / "smoke"),
            "timeout_seconds": 900,
            "module": "toolkits.rlt.physrlt.collect",
            "args": [
                "--source",
                str(source),
                "--assets",
                str(assets),
                "--cases",
                str(root / "smoke_cases.json"),
                "--output",
                str(root / "smoke"),
                "--ticks",
                "20",
                "--noise",
                "0.03",
            ],
        }
    ]
    for task, perturbation in ((0, "object"), (3, "lan")):
        task_dir = root / f"task{task}-{perturbation}"
        task_dir.mkdir(exist_ok=True)
        cases = [
            {
                "id": f"physrlt-task{task}-{perturbation}-layout{state}",
                "suite": "libero_10",
                "task": task,
                "state": state,
                "kind": "published",
                "perturbation": perturbation,
                "seed": 74000 + 100 * task + state,
                "split": "train"
                if state < 6
                else "validation"
                if state < 8
                else "development",
            }
            for state in range(10)
        ]
        write_json(task_dir / "cases.json", cases)
        collect_id = f"task{task}-collect"
        collect = task_dir / "collection"
        probes = task_dir / "probes"
        jobs.append(
            {
                "id": collect_id,
                "kind": "gpu",
                "dependencies": ["capture-smoke"],
                "output": str(collect),
                "timeout_seconds": 5400,
                "module": "toolkits.rlt.physrlt.collect",
                "args": [
                    "--source",
                    str(source),
                    "--assets",
                    str(assets),
                    "--cases",
                    str(task_dir / "cases.json"),
                    "--output",
                    str(collect),
                    "--ticks",
                    "320",
                    "--noise",
                    "0.03",
                ],
            }
        )
        jobs.append(
            {
                "id": f"task{task}-probe",
                "kind": "cpu",
                "dependencies": [collect_id],
                "output": str(probes),
                "timeout_seconds": 5400,
                "module": "toolkits.rlt.physrlt.probe",
                "args": [
                    "--dataset",
                    str(collect / "transitions.pt"),
                    "--output",
                    str(probes),
                    "--steps",
                    "400",
                    "--seeds",
                    "31",
                    "47",
                    "59",
                    "--threads",
                    "4",
                ],
            }
        )
        # Seeds and methods are fixed before any development outcome is observed.
        for seed in (31, 47):
            control = task_dir / f"control-seed{seed}"
            jobs.append(
                {
                    "id": f"task{task}-control-{seed}",
                    "kind": "gpu",
                    "dependencies": [f"task{task}-probe"],
                    "output": str(control),
                    "timeout_seconds": 10800,
                    "module": "toolkits.rlt.physrlt.control",
                    "args": [
                        "--source",
                        str(source),
                        "--assets",
                        str(assets),
                        "--dataset",
                        str(collect / "transitions.pt"),
                        "--probes",
                        str(probes),
                        "--output",
                        str(control),
                        "--seed",
                        str(seed),
                        "--warmup-steps",
                        "1500",
                        "--online-ticks",
                        "1040",
                        "--methods",
                        "action_recon",
                        "geometry",
                        "response",
                        "joint",
                    ],
                }
            )
    write_json(
        root / "plan.json",
        {
            "schema": 1,
            "title": "PhysRLT LIBERO-Pro small development campaign",
            "global_wall_time_limit": None,
            "bounded_work": "2 tasks, 20 collection episodes; 7 readouts x 3 seeds; 4 control arms x 2 seeds",
            "no_final_test_access": True,
            "published_layouts_previously_used_by_baseline": True,
            "formal_paper_split_required_later": True,
            "main_hypothesis": "Action-side features grounded by measured response and geometry improve lightweight control",
            "not_claimed": [
                "PrismWAM reproduction",
                "universal physical law",
                "long-term memory",
                "throughput gain",
                "human-intervention reduction",
            ],
            "physics_constraints": "Native action limits and calibrated measurable quantities only; no assumed object/EE rigid coupling",
            "sensor_matching": "All actors get the same RGB/state/reference; depth/object states are training targets only",
            "supervision": "3/10 control-tick metric depth/object displacement and measured joint/EE/gripper displacement",
            "baseline_scope": "Matched small action readout is not the original 2048D RLT encoder; original baseline remains separate",
            "feature_controls": [
                "prefix_joint",
                "state_action_joint",
                "action_recon",
                "shuffled_joint",
            ],
            "selection": "Fixed budgets; ridge selected on validation only; small BC gate uses validation only",
            "jobs": jobs,
        },
    )


def ready_jobs(jobs: list[dict], state: dict) -> tuple[list[dict], list[str]]:
    """Separate runnable jobs from descendants of failed or blocked jobs."""
    ready, blocked = [], []
    for job in jobs:
        if state[job["id"]]["state"] not in {"pending", "waiting_gpu"}:
            continue
        parents = [state[name]["state"] for name in job["dependencies"]]
        if any(value in {"failed", "blocked"} for value in parents):
            blocked.append(job["id"])
        elif all(value == "complete" for value in parents):
            ready.append(job)
    return ready, blocked


def prepare_followup(root: Path, base_root: Path, source: Path, assets: Path) -> None:
    """Freeze targeted followups prompted by the first development probes."""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("An existing plan must not be silently changed")
    methods = [
        "hybrid_recon",
        "hybrid_joint",
        "joint_recon",
        "factorized_joint",
        "residual_joint",
        "state_action_joint",
    ]
    jobs = []
    for task, perturbation in ((0, "object"), (3, "lan")):
        task_dir = root / f"task{task}-{perturbation}"
        dataset = base_root / task_dir.name / "collection/transitions.pt"
        if not dataset.exists():
            raise FileNotFoundError(dataset)
        probes = task_dir / "probes"
        jobs.append(
            {
                "id": f"task{task}-probe",
                "kind": "cpu",
                "dependencies": [],
                "output": str(probes),
                "timeout_seconds": 3600,
                "module": "toolkits.rlt.physrlt.probe",
                "args": [
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(probes),
                    "--steps",
                    "400",
                    "--seeds",
                    "31",
                    "47",
                    "59",
                    "--threads",
                    "4",
                    "--methods",
                    *methods,
                ],
            }
        )
        control = task_dir / "control-seed31"
        jobs.append(
            {
                "id": f"task{task}-control-31",
                "kind": "gpu",
                "dependencies": [f"task{task}-probe"],
                "output": str(control),
                "timeout_seconds": 10800,
                "module": "toolkits.rlt.physrlt.control",
                "args": [
                    "--source",
                    str(source),
                    "--assets",
                    str(assets),
                    "--dataset",
                    str(dataset),
                    "--probes",
                    str(probes),
                    "--output",
                    str(control),
                    "--seed",
                    "31",
                    "--warmup-steps",
                    "1500",
                    "--online-ticks",
                    "1040",
                    "--methods",
                    *methods,
                ],
            }
        )
    write_json(
        root / "plan.json",
        {
            "schema": 1,
            "title": "PhysRLT targeted development followups",
            "base_root": str(base_root),
            "gpu_after_status": str(base_root / "status.json"),
            "global_wall_time_limit": None,
            "no_final_test_access": True,
            "selection_provenance": "Motivated by pilot-v1 development results; these are not independent confirmatory tests",
            "hypotheses": {
                "hybrid_joint": "Prefix provides geometry while action features provide execution intent; compare hybrid_recon",
                "joint_recon": "A 0.1 reconstruction term may preserve control-relevant information during grounding",
                "factorized_joint": "Separate supervision of two latent halves may reduce response/geometry gradient interference",
                "residual_joint": "Cross-fitted state/action response residuals may focus the token on nontrivial execution effects",
                "state_action_joint": "Check whether action-expert features add value beyond current proprioception/reference",
            },
            "jobs": jobs,
        },
    )


def summarize(root: Path) -> None:
    """Build an auditable development table without converting predictions to SR."""
    records = []
    invalidated = (root / "INVALIDATION.json").exists()
    for path in sorted(root.glob("task*/control-seed*/episodes.json")):
        episodes = json.loads(path.read_text())
        groups = {}
        for episode in episodes:
            key = (episode["method"], episode["stage"], episode["split"])
            groups.setdefault(key, []).append(episode)
        for (method, stage, split), group in groups.items():
            records.append(
                {
                    "task": path.parents[1].name,
                    "run": path.parent.name,
                    "method": method,
                    "stage": stage,
                    "split": split,
                    "episodes": len(group),
                    "successes": sum(r["success"] for r in group),
                    "success_rate": sum(r["success"] for r in group) / len(group),
                    "mean_ticks_all_episodes": sum(r["ticks"] for r in group)
                    / len(group),
                    "final_test": False,
                    "unique_layouts": len({r["state_id"] for r in group}),
                    "scientific_status": "invalidated_action_domain"
                    if invalidated and stage.startswith("online_")
                    else "development_only",
                }
            )
    write_json(root / "control_summary.json", records)
    if records:
        with (root / "control_summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def prepare_control_check(
    root: Path,
    base_root: Path,
    after_root: Path,
    source: Path,
    assets: Path,
    *,
    recovery: bool = False,
) -> None:
    """Repeat matched controls after the Q action-domain fix and test outcome tokens."""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("An existing plan must not be silently changed")
    jobs = []
    for task, perturbation in ((0, "object"), (3, "lan")):
        name = f"task{task}-{perturbation}"
        output = root / name / "control-seed31"
        jobs.append(
            {
                "id": f"task{task}-control-31",
                "kind": "gpu",
                "dependencies": [],
                "output": str(output),
                "timeout_seconds": 10800,
                "module": "toolkits.rlt.physrlt.control",
                "args": [
                    "--source",
                    str(source),
                    "--assets",
                    str(assets),
                    "--dataset",
                    str(base_root / name / "collection/transitions.pt"),
                    "--probes",
                    str(base_root / name / "probes"),
                    "--output",
                    str(output),
                    "--seed",
                    "31",
                    "--warmup-steps",
                    "1500",
                    "--online-ticks",
                    "1040",
                    "--methods",
                    *(
                        ["joint_mixed"]
                        if recovery
                        else [
                            "action_recon",
                            "geometry",
                            "response",
                            "joint",
                            "joint_outcomes",
                            "joint_mixed",
                        ]
                    ),
                ],
            }
        )
    stress = root / "reference_stress"
    jobs.append(
        {
            "id": "reference-stress",
            "kind": "gpu",
            "dependencies": []
            if recovery
            else ["task0-control-31", "task3-control-31"],
            "output": str(stress),
            "timeout_seconds": 5400,
            "module": "toolkits.rlt.physrlt.stress",
            "args": [
                "--source",
                str(source),
                "--assets",
                str(assets),
                "--base-root",
                str(base_root),
                "--output",
                str(stress),
            ],
        }
    )
    write_json(
        root / "plan.json",
        {
            "schema": 1,
            "title": "Recover only the incomplete mixed-token screen and blocked stress"
            if recovery
            else "Corrected matched controls, structured physical outputs and reference stress",
            "control_protocol_version": 2,
            "gpu_after_status": str(after_root / "status.json"),
            "global_wall_time_limit": None,
            "base_root": str(base_root),
            "no_final_test_access": True,
            "hypotheses": {
                "joint_outcomes": "Explicit predicted response and geometry may be easier for the small policy to use than a latent alone",
                "joint_mixed": "Preserve half the latent while exposing predicted physical quantities",
                "reference_stress": "Screen software actuation shifts on training layouts before allocating adaptation training",
            },
            "invalidated_prior_online_results": str(base_root / "INVALIDATION.json"),
            "jobs": jobs,
        },
    )


def prepare_anchor_check(
    root: Path,
    base_root: Path,
    after_root: Path,
    source: Path,
    assets: Path,
    *,
    recovery: bool = False,
) -> None:
    """Test physical grounding alongside, rather than replacing, an action prior."""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("An existing plan must not be silently changed")
    jobs = []
    for task, perturbation in (
        ((0, "object"),) if recovery else ((0, "object"), (3, "lan"))
    ):
        name = f"task{task}-{perturbation}"
        output = root / name / "control-seed31"
        jobs.append(
            {
                "id": f"task{task}-control-31",
                "kind": "gpu",
                "dependencies": [],
                "output": str(output),
                "timeout_seconds": 10800,
                "module": "toolkits.rlt.physrlt.control",
                "args": [
                    "--source",
                    str(source),
                    "--assets",
                    str(assets),
                    "--dataset",
                    str(base_root / name / "collection/transitions.pt"),
                    "--probes",
                    str(base_root / name / "probes"),
                    "--output",
                    str(output),
                    "--seed",
                    "31",
                    "--warmup-steps",
                    "1500",
                    "--online-ticks",
                    "1040",
                    "--methods",
                    *(
                        ["response_anchor", "joint_anchor"]
                        if recovery
                        else [
                            "action_recon_anchor",
                            "geometry_anchor",
                            "response_anchor",
                            "joint_anchor",
                        ]
                    ),
                ],
            }
        )
    write_json(
        root / "plan.json",
        {
            "schema": 1,
            "title": "Recover interrupted task0 anchor arms"
            if recovery
            else "Preserve the action prior while adding physical grounding",
            "control_protocol_version": 2,
            "no_final_test_access": True,
            "gpu_after_status": str(after_root / "status.json"),
            "global_wall_time_limit": None,
            "hypothesis": "Preserving a separate frozen action-reconstruction readout may prevent physical supervision from discarding control-relevant cues",
            "matching": "All arms combine two 400-step encoders into a fixed 128D token; control combines reconstruction seeds 59 and 31, physics arms combine reconstruction seed 59 with grounded seed 31",
            "base_root": str(base_root),
            "jobs": jobs,
        },
    )


def prepare_repeat(
    root: Path, base_root: Path, after_root: Path, source: Path, assets: Path
) -> None:
    """Predeclare bounded replication on new development layouts and two seeds."""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("An existing plan must not be silently changed")
    followup = base_root.parent / "pilot-v2"
    if not (followup / "plan.json").exists():
        raise FileNotFoundError(
            "This local replication requires completed pilot-v2 readouts"
        )
    jobs = []
    for task, perturbation in ((0, "object"), (3, "lan")):
        name = f"task{task}-{perturbation}"
        cases_path = root / name / "development_cases.json"
        cases_path.parent.mkdir(parents=True, exist_ok=True)
        cases = [
            {
                "id": f"repeat-task{task}-{perturbation}-layout{state}",
                "suite": "libero_10",
                "task": task,
                "state": state,
                "kind": "published",
                "perturbation": perturbation,
                "seed": 77000 + 100 * task + state,
                "split": "development",
            }
            for state in range(10, 16)
        ]
        write_json(cases_path, cases)
        for gain in (1.0, 0.7):
            for seed in (47, 59):
                for family, probes, methods in (
                    ("action", base_root / name / "probes", ["action_recon", "joint"]),
                    (
                        "fusion",
                        followup / name / "probes",
                        ["hybrid_recon", "hybrid_joint"],
                    ),
                ):
                    output = root / name / f"control-seed{seed}-gain{gain}-{family}"
                    jobs.append(
                        {
                            "id": f"task{task}-seed{seed}-gain{gain}-{family}",
                            "kind": "gpu",
                            "dependencies": [],
                            "output": str(output),
                            "timeout_seconds": 10800,
                            "module": "toolkits.rlt.physrlt.control",
                            "args": [
                                "--source",
                                str(source),
                                "--assets",
                                str(assets),
                                "--dataset",
                                str(base_root / name / "collection/transitions.pt"),
                                "--probes",
                                str(probes),
                                "--output",
                                str(output),
                                "--seed",
                                str(seed),
                                "--warmup-steps",
                                "1500",
                                "--online-ticks",
                                "1040",
                                "--execution-gain",
                                str(gain),
                                "--development-cases",
                                str(cases_path),
                                "--methods",
                                *methods,
                            ],
                        }
                    )
    write_json(
        root / "plan.json",
        {
            "schema": 1,
            "title": "Bounded repeated PhysRLT development with matched reconstruction controls",
            "gpu_after_status": str(after_root / "status.json"),
            "global_wall_time_limit": None,
            "control_protocol_version": 2,
            "no_final_test_access": True,
            "bounded_work": "2 tasks x 2 gains x 2 training seeds x 4 methods, each at most 1040 online ticks; no long training",
            "new_development_layouts": list(range(10, 16)),
            "selection": "No selection using these outcomes; report all arms",
            "hypothesis": "Physical supervision must beat its same-feature reconstruction control across seeds and layouts, not just an easy pilot",
            "stress_scope": "0.7 translation command gain is custom software attenuation, not changed friction or an official benchmark; readouts remain trained on nominal data",
            "limits": "Published layouts may have appeared in prior baseline work; this remains development. Same layouts reused across seeds are not independent episodes.",
            "base_root": str(base_root),
            "jobs": jobs,
        },
    )


def prepare_paired_actions(
    root: Path, base_root: Path, source: Path, *, split: str = "train"
) -> None:
    """Queue independent paired-action diagnostics after verifying frozen inputs."""
    if split not in {"train", "development"}:
        raise ValueError("Paired actions require the train or development split")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("An existing plan must not be silently changed")
    jobs = []
    for task, perturbation in ((0, "object"), (3, "lan")):
        name = f"task{task}-{perturbation}"
        dataset = base_root / name / "collection/transitions.pt"
        probes = base_root / name / "probes"
        if not dataset.exists() or not all(
            (probes / f"{method}-seed31/readout.pt").exists()
            for method in ("joint", "shuffled_joint", "state_action_joint")
        ):
            raise FileNotFoundError("Paired-action inputs must already be complete")
        output = root / name / "paired_actions"
        jobs.append(
            {
                "id": f"task{task}-{split}-paired-actions",
                "kind": "gpu",
                "dependencies": [],
                "output": str(output),
                "timeout_seconds": 1800,
                "module": "toolkits.rlt.physrlt.counterfactual",
                "args": [
                    "--source",
                    str(source),
                    "--dataset",
                    str(dataset),
                    "--probes",
                    str(probes),
                    "--output",
                    str(output),
                    "--cases",
                    "2",
                    "--ticks",
                    "40",
                    "100",
                    "160",
                    "--axis",
                    str(0 if task == 0 else 1),
                    "--offset",
                    "0.1",
                    "--split",
                    split,
                ],
            }
        )
    write_json(
        root / "plan.json",
        {
            "schema": 1,
            "title": "Paired candidate-action effect diagnostic",
            "global_wall_time_limit": None,
            "no_final_test_access": True,
            "paired_split": split,
            "hypothesis": "A grounded consequence decoder should predict the difference between two executed candidate actions from an identical state",
            "limits": (
                "Development layouts with new paired actions; at most 12 states; independent of readout fitting but still development evidence"
                if split == "development"
                else "Training states with new paired actions; at most 12 states; not generalization or control-success evidence"
            ),
            "base_root": str(base_root),
            "jobs": jobs,
        },
    )


def prepare_response_check(
    root: Path,
    base_root: Path,
    probes_root: Path,
    source: Path,
    assets: Path,
) -> None:
    """Screen response targets with matched BC and no online interaction."""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("An existing plan must not be silently changed")
    jobs = []
    for task, perturbation in ((0, "object"), (3, "lan")):
        name = f"task{task}-{perturbation}"
        output = root / name / "control-seed31"
        readouts = probes_root / name / "probes"
        for method in ("joint", "arm_q_gripper_joint", "eef_gripper_joint"):
            if not (readouts / f"{method}-seed31/readout.pt").exists():
                raise FileNotFoundError(f"Missing response ablation: {method}")
        jobs.append(
            {
                "id": f"task{task}-response-control",
                "kind": "gpu",
                "dependencies": [],
                "output": str(output),
                "timeout_seconds": 10800,
                "module": "toolkits.rlt.physrlt.control",
                "args": [
                    "--source",
                    str(source),
                    "--assets",
                    str(assets),
                    "--dataset",
                    str(base_root / name / "collection/transitions.pt"),
                    "--probes",
                    str(readouts),
                    "--output",
                    str(output),
                    "--seed",
                    "31",
                    "--warmup-steps",
                    "1500",
                    "--online-ticks",
                    "0",
                    "--methods",
                    "joint",
                    "arm_q_gripper_joint",
                    "eef_gripper_joint",
                ],
            }
        )
    write_json(
        root / "plan.json",
        {
            "schema": 1,
            "title": "Matched execution-response redundancy screen",
            "control_protocol_version": 2,
            "global_wall_time_limit": None,
            "no_final_test_access": True,
            "base_root": str(base_root),
            "probes_root": str(probes_root),
            "hypothesis": "Arm-joint and end-effector response targets are kinematically redundant; retain the smaller target only if matched BC control does not regress",
            "matching": "Same action features, 128D bottleneck, geometry target, 400 readout updates, 1500 actor BC updates, seed and layouts; no online RL",
            "contact_scope": "No force, tactile, simulator contact-pair, impulse, friction or stiffness supervision",
            "jobs": jobs,
        },
    )


def claim_gpu(gpu: int, directory: Path = LEASE_DIRECTORY):
    """Acquire the shared research lease without waiting or taking another job's card."""
    lease = (directory / f"rlt-research-gpu{gpu}.lock").open("a")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return None
    return lease


def job_stalled(output: Path, now: float, limit: float = 600) -> bool:
    """Treat completed episode/effect artifacts as progress even with quiet stdout."""
    paths = [output / name for name in ("stdout.log", "episodes.json", "effects.json")]
    timestamps = [path.stat().st_mtime for path in paths if path.exists()]
    return bool(timestamps) and now - max(timestamps) > limit


def supervise(args: argparse.Namespace) -> None:
    """Wait for unused authorized cards; isolate per-job failures and keep logs."""
    plan_path = args.root / "plan.json"
    plan = json.loads(plan_path.read_text())
    lock = (args.root / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = args.root / "status.json"
    state = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {job["id"]: {"state": "pending", "attempts": 0} for job in plan["jobs"]}
    )
    # A restart does not assume a previously running job completed successfully.
    for item in state.values():
        if item["state"] == "running":
            raise RuntimeError(
                "Interrupted supervisor needs an explicit process/output audit before restart"
            )
    active, idle_since = {}, {}
    env = os.environ.copy()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(
        args.root / "launcher.json",
        {
            "pid": os.getpid(),
            "revision": revision,
            "cwd": str(Path.cwd()),
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "gpus": args.gpus,
            "start_unix": time.time(),
        },
    )
    try:
        while True:
            for name, running in list(active.items()):
                process, job, log, start, lease = running
                code = process.poll()
                stalled = job_stalled(Path(job["output"]), time.time())
                if code is None and (
                    time.monotonic() - start > job["timeout_seconds"] or stalled
                ):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                    code = process.returncode
                    state[name]["timeout"] = True
                    state[name]["stalled_without_log_progress"] = stalled
                if code is not None:
                    log.close()
                    if lease is not None:
                        lease.close()
                    output = Path(job["output"])
                    artifact = output / (
                        "results.json" if job["kind"] == "cpu" else "result.json"
                    )
                    state[name].update(
                        state="complete"
                        if code == 0 and artifact.exists()
                        else "failed",
                        exit_code=code,
                        end_unix=time.time(),
                    )
                    failure_text = (output / "stdout.log").read_text(errors="replace")[
                        -4000:
                    ]
                    if (
                        state[name]["state"] == "failed"
                        and "is occupied: memory=" in failure_text
                        and state[name]["attempts"] < 5
                    ):
                        state[name].update(
                            state="waiting_gpu",
                            wait_reason="GPU became occupied during launch; bounded retry",
                        )
                    del active[name]
            ready, blocked = ready_jobs(plan["jobs"], state)
            for name in blocked:
                state[name].update(
                    state="blocked", reason="dependency did not complete"
                )
            used = {state[name].get("gpu") for name in active}
            available = []
            for gpu in args.gpus:
                idle, detail = gpu_idle(gpu)
                if idle and gpu not in used:
                    idle_since.setdefault(gpu, time.monotonic())
                    if time.monotonic() - idle_since[gpu] >= args.idle_seconds:
                        available.append(gpu)
                else:
                    idle_since.pop(gpu, None)
            for job in ready:
                name = job["id"]
                predecessor = plan.get("gpu_after_status")
                if job["kind"] == "gpu" and predecessor:
                    previous = json.loads(Path(predecessor).read_text())
                    if not all(
                        value["state"] in {"complete", "failed", "blocked"}
                        for value in previous.values()
                    ):
                        state[name]["state"] = "waiting_gpu"
                        state[name]["wait_reason"] = "prior campaign is still active"
                        continue
                if job["kind"] == "cpu" and any(
                    j[1]["kind"] == "cpu" for j in active.values()
                ):
                    continue
                if job["kind"] == "gpu" and not available:
                    state[name]["state"] = "waiting_gpu"
                    continue
                gpu = available.pop(0) if job["kind"] == "gpu" else None
                lease = claim_gpu(gpu) if gpu is not None else None
                if gpu is not None and (lease is None or not gpu_idle(gpu)[0]):
                    if lease is not None:
                        lease.close()
                    state[name].update(
                        state="waiting_gpu",
                        wait_reason="shared GPU lease or occupancy changed",
                    )
                    continue
                command = [sys.executable, "-u", "-m", job["module"], *job["args"]]
                if gpu is not None:
                    command += ["--gpu", str(gpu)]
                output = Path(job["output"])
                output.mkdir(parents=True, exist_ok=True)
                log = (output / "stdout.log").open("a", buffering=1)
                try:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        start_new_session=True,
                        pass_fds=(lease.fileno(),) if lease is not None else (),
                    )
                except BaseException:
                    log.close()
                    if lease is not None:
                        lease.close()
                    raise
                state[name].update(
                    state="running",
                    gpu=gpu,
                    pid=process.pid,
                    attempts=state[name]["attempts"] + 1,
                    command=command,
                    start_unix=time.time(),
                )
                active[name] = (process, job, log, time.monotonic(), lease)
            write_json(status_path, state)
            summarize(args.root)
            if all(
                item["state"] in {"complete", "blocked", "failed"}
                for item in state.values()
            ):
                break
            time.sleep(10)
    finally:
        for _, _, log, _, lease in active.values():
            log.close()
            if lease is not None:
                # Children inherit the lease, so an interrupted supervisor does
                # not expose a still-running GPU job to another campaign.
                lease.close()
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "prepare-followup",
            "prepare-control-check",
            "prepare-control-recovery",
            "prepare-anchor-check",
            "prepare-anchor-recovery",
            "prepare-repeat",
            "prepare-paired-actions",
            "prepare-response-check",
            "run",
            "summarize",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--after-root", type=Path)
    parser.add_argument("--probes-root", type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--idle-seconds", type=int, default=30)
    parser.add_argument(
        "--paired-split", choices=("train", "development"), default="train"
    )
    args = parser.parse_args()
    if args.mode in {
        "prepare",
        "prepare-followup",
        "prepare-control-check",
        "prepare-control-recovery",
        "prepare-anchor-check",
        "prepare-anchor-recovery",
        "prepare-repeat",
        "prepare-paired-actions",
        "prepare-response-check",
    }:
        if not args.source or not args.assets:
            parser.error("prepare requires --source and --assets")
        if args.mode == "prepare-paired-actions":
            if not args.base_root:
                parser.error("prepare-paired-actions requires --base-root")
            prepare_paired_actions(
                args.root, args.base_root, args.source, split=args.paired_split
            )
        elif args.mode == "prepare-response-check":
            if not args.base_root or not args.probes_root:
                parser.error(
                    "prepare-response-check requires --base-root and --probes-root"
                )
            prepare_response_check(
                args.root,
                args.base_root,
                args.probes_root,
                args.source,
                args.assets,
            )
        elif args.mode in {
            "prepare-control-check",
            "prepare-control-recovery",
            "prepare-anchor-check",
            "prepare-anchor-recovery",
            "prepare-repeat",
        }:
            if not args.base_root or not args.after_root:
                parser.error(
                    "prepare-control-check requires --base-root and --after-root"
                )
            if args.mode == "prepare-control-recovery":
                prepare_control_check(
                    args.root,
                    args.base_root,
                    args.after_root,
                    args.source,
                    args.assets,
                    recovery=True,
                )
                return
            if args.mode == "prepare-anchor-recovery":
                prepare_anchor_check(
                    args.root,
                    args.base_root,
                    args.after_root,
                    args.source,
                    args.assets,
                    recovery=True,
                )
                return
            operation = {
                "prepare-control-check": prepare_control_check,
                "prepare-anchor-check": prepare_anchor_check,
                "prepare-repeat": prepare_repeat,
            }[args.mode]
            operation(
                args.root, args.base_root, args.after_root, args.source, args.assets
            )
        elif args.mode == "prepare-followup":
            if not args.base_root:
                parser.error("prepare-followup requires --base-root")
            prepare_followup(args.root, args.base_root, args.source, args.assets)
        else:
            prepare(args.root, args.source, args.assets)
    elif args.mode == "run":
        supervise(args)
    else:
        summarize(args.root)


if __name__ == "__main__":
    main()
