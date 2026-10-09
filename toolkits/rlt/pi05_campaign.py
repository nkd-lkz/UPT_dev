# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""A finite, dependency-gated new-RLT Pi0.5 development campaign."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from toolkits.rlt.experiment_queue import run_plan
from toolkits.rlt.pi05_assets import REVISION
from toolkits.rlt.pi05_baseline import write_json


def paired_gate(reference: dict, actor: dict) -> dict:
    """Require identical reset cases and at most one lost success in the smoke gate."""
    for report in (reference, actor):
        if report.get("complete") is not True or report.get("exit_code") != 0:
            raise ValueError("Require completed successful evaluation commands")
    a, b = reference["episodes"], actor["episodes"]
    if len(a) != len(b) or len(a) < 5:
        raise ValueError("Need at least five matched development cases")
    keys = ("id", "task", "kind", "state", "seed", "state_sha256")
    if any(any(x[k] != y[k] for k in keys) for x, y in zip(a, b, strict=True)):
        raise ValueError("Reference and actor reset cases differ")
    ref_success = sum(x["success"] for x in a)
    actor_success = sum(x["success"] for x in b)
    passed = ref_success >= 3 and actor_success >= ref_success - 1
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
                ):
                    if key in report:
                        row[key] = report[key]
                if "episodes" in report:
                    row["episodes"] = len(report["episodes"])
        results[name] = row
    return {
        "queue_state": queue.get("state", "not_started"),
        "gpu_index": queue.get("gpu_index"),
        "jobs": results,
        "engineering_pipeline_complete": all(
            results[name]["state"] == "passed"
            and results[name].get("manifest_complete") is True
            for name in ("reference_final", "warmup_final", "q_bc_eval", "bc_only_eval")
        ),
        "paper_reproduction_validated": False,
        "robodojo_closed_loop_validated": False,
        "scope": "single-task development pilot; not evidence of an RL improvement",
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
    parser.add_argument("mode", choices=["run", "wait-assets", "paired-gate", "status"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.mode == "status":
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
        plan = campaign_plan(args.root, repo, args.assets, args.source, args.gpu)
        try:
            run_plan(plan, args.root / "queue")
        finally:
            write_json(args.root / "summary.json", summarize_campaign(args.root))


if __name__ == "__main__":
    main()
