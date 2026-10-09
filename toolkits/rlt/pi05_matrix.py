# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Predeclare a four-suite, three-seed RLT comparison and run bounded GPU queues."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from toolkits.rlt.experiment_queue import run_plan, validate_plan
from toolkits.rlt.pi05_assets import file_hash
from toolkits.rlt.pi05_baseline import SUITE_HORIZONS, write_json
from toolkits.rlt.pi05_campaign import (
    FINAL_ARMS,
    acceptance_cases,
    acceptance_plan,
    audit_acceptance,
    paired_comparison,
)

TRAINING_SEEDS = (42, 43, 44)
CODE_FILES = (
    "run_rlt_pi05.sh",
    "toolkits/rlt/pi05_baseline.py",
    "toolkits/rlt/pi05_learning.py",
    "toolkits/rlt/pi05_campaign.py",
    "toolkits/rlt/pi05_matrix.py",
    "toolkits/rlt/pi05_assets.py",
    "toolkits/rlt/libero_acceptance.py",
    "toolkits/rlt/libero_acceptance_learning.py",
    "toolkits/rlt/libero_acceptance_env.py",
    "toolkits/rlt/experiment_queue.py",
)


def code_fingerprint(repo: Path) -> dict[str, str]:
    """Identify all local runtime files used by the declared experiment."""
    return {name: file_hash(repo / name) for name in CODE_FILES}


def prepare_matrix(
    root: Path, repo: Path, assets: Path, source: Path, gpus: list[int], entity: str
) -> dict:
    """Declare all 120 cells before any result, refusing to reuse an output root.

    Each cell owns its encoder and both trained heads. Held-out reset seeds are
    shared across training seeds, not counted as independent repeated trials.
    Old exploratory Goal pilots are deliberately outside this matrix.
    """
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError("Require distinct, explicitly authorized physical GPUs")
    root = root.resolve()
    repo, assets, source = repo.resolve(), assets.resolve(), source.resolve()
    fingerprints = code_fingerprint(repo)
    root.mkdir(parents=True, exist_ok=False)
    matrix = {
        "schema": "rlt-pi05-four-suite-v1",
        "repo": str(repo),
        "source": str(source),
        "assets": str(assets),
        "code_sha256": fingerprints,
        "gpus": gpus,
        "training_seeds": list(TRAINING_SEEDS),
        "suites": SUITE_HORIZONS,
        "heldout_episodes_per_arm": 100,
        "online_ticks_per_arm": 20000,
        "collection_episodes": 64,
        "encoder_updates": 1000,
        "bc_warmup_updates": 6000,
        "arms": list(FINAL_ARMS),
        "planned_eval_episodes": 40 * len(TRAINING_SEEDS) * 100 * len(FINAL_ARMS),
        "scope": "Independent task-specific heads, shared public four-suite SFT VLA. Bounded adapted baseline; not an unmodified paper reproduction or unseen-task transfer.",
        "aggregation": "Equal task means within suite, then equal suite means. Publish each training seed and its dispersion. A missing/failed cell blocks the corresponding full aggregate; never discard failed gates.",
        "selection": "Final-budget checkpoints only. Held-out outcomes are not used for hyperparameters, task selection or early stopping.",
        "cells": [],
    }
    plans = {
        gpu: {"gpu": gpu, "hours": 12, "wait_for_lease": True, "jobs": []}
        for gpu in gpus
    }
    for seed in TRAINING_SEEDS:
        for task in range(10):
            for suite_index, suite in enumerate(SUITE_HORIZONS):
                # Rotate assignments by task so each GPU visits all four suites.
                gpu = gpus[(suite_index + task) % len(gpus)]
                name = f"{suite}-task{task}-seed{seed}"
                cell_root = root / "cells" / name
                (cell_root / "cases").mkdir(parents=True)
                for split, cases in acceptance_cases(
                    cell_root, task, suite, seed
                ).items():
                    write_json(cell_root / "cases" / f"{split}.json", cases)
                plan = acceptance_plan(
                    cell_root,
                    repo,
                    assets,
                    source,
                    gpu,
                    task,
                    seed,
                    "online",
                    entity,
                    suite=suite,
                    case_training_seed=seed,
                )
                validate_plan(plan)
                (cell_root / "queue").mkdir()
                write_json(cell_root / "queue/plan.json", plan)
                cell = {
                    "name": name,
                    "suite": suite,
                    "task": task,
                    "seed": seed,
                    "gpu": gpu,
                    "root": str(cell_root),
                }
                matrix["cells"].append(cell)
                for job in plan["jobs"]:
                    plans[gpu]["jobs"].append(
                        {
                            **job,
                            "name": f"{name}__{job['name']}",
                            "depends_on": [
                                f"{name}__{dep}" for dep in job.get("depends_on", [])
                            ],
                        }
                    )
                # Independent audit still runs after a gate failure, preserving it.
                plans[gpu]["jobs"].append(
                    {
                        "name": f"{name}__audit",
                        "requires_gpu": False,
                        "cwd": str(repo),
                        "argv": [
                            sys.executable,
                            "-m",
                            "toolkits.rlt.pi05_campaign",
                            "audit",
                            "--root",
                            str(cell_root),
                            "--report",
                            str(cell_root / "audit.json"),
                        ],
                        "timeout_seconds": 120,
                    }
                )
    (root / "plans").mkdir()
    for gpu, plan in plans.items():
        validate_plan(plan)
        write_json(root / "plans" / f"gpu{gpu}.json", plan)
    # Seal case and job plans so edits cannot silently change a declared run.
    matrix["plan_sha256"] = {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(root.rglob("*.json"))
    }
    write_json(root / "matrix.json", matrix)
    return matrix


def validate_matrix(root: Path) -> dict:
    """Reject changes to runtime code, case sets, or jobs since declaration."""
    matrix = json.loads((root / "matrix.json").read_text())
    if matrix["code_sha256"] != code_fingerprint(Path(matrix["repo"])):
        raise ValueError("Runtime code changed after the matrix was declared")
    for relative, expected in matrix["plan_sha256"].items():
        if file_hash(root / relative) != expected:
            raise ValueError(f"Declared plan changed: {relative}")
    return matrix


def aggregate_cells(cells: list[dict], seeds: list[int]) -> dict:
    """Require the complete task/seed grid before returning suite or overall means."""
    suites = {}
    for suite in SUITE_HORIZONS:
        rows = [c for c in cells if c["suite"] == suite and c.get("state") == "passed"]
        expected = {(task, seed) for task in range(10) for seed in seeds}
        ready = (
            len(rows) == len(expected)
            and {(c["task"], c["seed"]) for c in rows} == expected
        )
        result = {
            "completed_cells": len(rows),
            "expected_cells": len(expected),
            "complete": ready,
            "arms": {},
        }
        if ready:
            for arm in FINAL_ARMS:
                means = [
                    sum(
                        c["comparison"][arm]["success_rate"]
                        for c in rows
                        if c["seed"] == seed
                    )
                    / 10
                    for seed in seeds
                ]
                mean = sum(means) / len(means)
                result["arms"][arm] = {
                    "success_rate": mean,
                    "seed_means": dict(zip(map(str, seeds), means, strict=True)),
                    "seed_std": math.sqrt(
                        sum((x - mean) ** 2 for x in means) / (len(means) - 1)
                    )
                    if len(means) > 1
                    else None,
                }
        suites[suite] = result
    overall = {}
    if all(row["complete"] for row in suites.values()):
        overall = {
            arm: sum(row["arms"][arm]["success_rate"] for row in suites.values()) / 4
            for arm in FINAL_ARMS
        }
    return {"suites": suites, "four_suite_means": overall}


def summarize_matrix(root: Path) -> dict:
    """Audit artifacts and expose missing cells without converting them to scores."""
    matrix = validate_matrix(root)
    cells, heldout_states = [], {}
    for declared in matrix["cells"]:
        cell = dict(declared)
        audit = audit_acceptance(Path(cell["root"]))
        cell.update(
            state=audit["state"],
            violations=audit["violations"],
            pending_stages=audit["pending_stages"],
        )
        cell["comparison"] = {}
        if cell["state"] == "passed":
            reports = {
                arm: json.loads(
                    (Path(cell["root"]) / arm / "manifest.json").read_text()
                )
                for arm in FINAL_ARMS
            }
            try:
                cell["comparison"] = paired_comparison(reports)
                states = [
                    (e["id"], e["state_sha256"])
                    for e in reports["reference_final"]["episodes"]
                ]
                key = (cell["suite"], cell["task"])
                if key in heldout_states and states != heldout_states[key]:
                    raise ValueError(
                        "Held-out physical states differ across training seeds"
                    )
                heldout_states[key] = states
            except ValueError as error:
                cell.update(state="failed", comparison={})
                cell["violations"].append(str(error))
        cells.append(cell)
    return {
        "complete": all(c["state"] == "passed" for c in cells),
        "completed_cells": sum(c["state"] == "passed" for c in cells),
        "expected_cells": len(cells),
        "failed_cells": sum(c["state"] == "failed" for c in cells),
        **aggregate_cells(cells, matrix["training_seeds"]),
        "cells": cells,
        "scope": matrix["scope"],
    }


def preflight_resets(root: Path, output: Path) -> None:
    """Create and restore one declared development reset per task on CPU only."""
    import os

    import numpy as np

    from toolkits.rlt.libero_acceptance import RemoteEnvironment
    from toolkits.rlt.pi05_baseline import configure_source

    matrix = validate_matrix(root)
    configure_source(Path(matrix["source"]))
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        RLT_LIBERO_EGL_DEVICE_ID="0",
        RLT_LIBERO_WORKER_LOG_DIR=str(output / "worker_logs"),
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "control_ticks": 0,
        "policy_inference": False,
        "checks": [],
    }
    env = RemoteEnvironment()
    try:
        for cell in matrix["cells"]:
            if cell["seed"] != TRAINING_SEEDS[0]:
                continue
            case = json.loads(
                (Path(cell["root"]) / "cases/development.json").read_text()
            )[0]
            # Diagnostic snapshots have a separate owner from GPU campaign resets.
            case["snapshot"] = str(
                output / "snapshots" / f"{cell['suite']}-{cell['task']}.json"
            )
            observation = env.reset(case, create_snapshot=True)
            original_hash = env.reset_hash
            restored = env.reset(case)
            if env.reset_hash != original_hash:
                raise ValueError("Reset restore hash mismatch")
            if not np.array_equal(observation["state"], restored["state"]):
                raise ValueError("Reset proprioception mismatch")
            report["checks"].append(
                {
                    "suite": cell["suite"],
                    "task": cell["task"],
                    "state_sha256": env.reset_hash,
                    "instruction": env.instruction,
                }
            )
            write_json(output / "manifest.json", report)
        report["complete"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        env.close()
        write_json(output / "manifest.json", report)


def main() -> None:
    """Prepare, run, or inspect the fixed matrix without overwriting past runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "status", "preflight"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--wandb-entity", default="c6522513-sustech")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.mode == "prepare":
        if args.assets is None or args.source is None:
            parser.error("prepare requires --assets and --source")
        matrix = prepare_matrix(
            args.root,
            Path(__file__).resolve().parents[2],
            args.assets,
            args.source,
            args.gpus,
            args.wandb_entity,
        )
        print(
            json.dumps(
                {
                    "cells": len(matrix["cells"]),
                    "gpus": matrix["gpus"],
                    "eval_episodes": matrix["planned_eval_episodes"],
                }
            )
        )
    elif args.mode == "status":
        print(json.dumps(summarize_matrix(args.root), indent=2))
    elif args.mode == "preflight":
        preflight_resets(args.root, args.root / "reset-preflight")
    else:
        matrix = validate_matrix(args.root)
        if args.gpu not in matrix["gpus"]:
            parser.error("GPU is not in the declared authorized set")
        plan = json.loads((args.root / "plans" / f"gpu{args.gpu}.json").read_text())
        plan["hours"] = args.hours
        try:
            run_plan(plan, args.root / f"queue-gpu{args.gpu}")
        finally:
            write_json(
                args.root / f"summary-gpu{args.gpu}.json", summarize_matrix(args.root)
            )


if __name__ == "__main__":
    main()
