# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Preserve completed RoboDojo episodes before an explicit native continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from toolkits.rlt import robodojo_eval as evaluation

SCHEMA = "robodojo-resume-v1"


def _snapshot(path: Path) -> dict:
    data = evaluation.read_json(path)
    raw = path.read_bytes()
    if json.loads(raw) != data:
        raise ValueError(f"File changed while reading: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "text": raw.decode("utf-8"),
    }


def _saved(snapshot: dict) -> dict:
    if (
        hashlib.sha256(snapshot["text"].encode("utf-8")).hexdigest()
        != snapshot["sha256"]
    ):
        raise ValueError("Receipt snapshot checksum mismatch")
    data = json.loads(snapshot["text"])
    if not isinstance(data, dict):
        raise ValueError("Receipt snapshot must contain a JSON object")
    return data


def _unchanged(snapshot: dict) -> None:
    path = Path(snapshot["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != snapshot["sha256"]:
        raise ValueError(f"Changed since recovery preparation: {path}")


def _paths(plan: dict) -> tuple[Path, Path]:
    config = plan["config"]
    evaluation.validate_config(config)
    root = Path(config["robodojo_root"])
    parent = (
        root
        / "eval_result/RoboDojo"
        / config["task"]
        / config["policy"]["policy_name"]
        / config["policy"]["env_cfg_type"]
        / f"{config['seed']}_{config['group']}"
    )
    result = parent / config["run_id"] / "_result.json"
    if result.resolve() != Path(plan["expected_result"]).resolve():
        raise ValueError("Plan result path does not match its run identity")
    if plan["commands"] != evaluation._commands(config):
        raise ValueError("Plan commands differ from the sealed configuration")
    return result, parent / f"_resume_{config['run_id']}.json"


def _check_manifest(plan: dict, result: dict, manifest: dict) -> list[int]:
    config = plan["config"]
    identity = {
        "run_id": config["run_id"],
        "task_name": config["task"],
        "policy_name": config["policy"]["policy_name"],
        "config_name": config["policy"]["env_cfg_type"],
        "eval_seed": config["seed"],
        "additional_info": config["group"],
    }
    for key, value in identity.items():
        if type(manifest.get(key)) is not type(value) or manifest[key] != value:
            raise ValueError(f"Resume identity mismatch: {key}")
    save_dir = manifest.get("save_dir")
    if not isinstance(save_dir, str) or not save_dir:
        raise ValueError("Resume save_dir is missing")
    directory = Path(save_dir)
    if not directory.is_absolute():
        directory = Path(config["robodojo_root"]) / directory
    if directory.resolve() != Path(plan["expected_result"]).parent.resolve():
        raise ValueError("Resume save_dir differs from planned output")
    details = result["details"]
    if manifest.get("details") != details:
        raise ValueError("Resume details differ from recorded episode outcomes")
    for key, row in details.items():
        if key != str(row["layout_id"]):
            raise ValueError("Native detail keys must equal their layout IDs")
    completed = sorted(row["layout_id"] for row in details.values())
    declared = manifest.get("completed_layout_ids")
    if (
        not isinstance(declared, list)
        or any(type(index) is not int for index in declared)
        or sorted(declared) != completed
    ):
        raise ValueError("Resume completed_layout_ids differ from recorded outcomes")
    if manifest.get("abandoned_layout_ids") != []:
        raise ValueError("Fixed-layout continuation forbids abandoned layouts")
    successes = sum(row["success"] for row in details.values())
    for key, value in (
        ("success_nums", successes),
        ("fail_nums", len(completed) - successes),
        ("unstable_nums", 0),
    ):
        if type(manifest.get(key)) is not int or manifest[key] != value:
            raise ValueError(f"Resume counter mismatch: {key}")
    restarts = manifest.get("restart_count")
    if type(restarts) is not int or restarts < 0:
        raise ValueError("Invalid resume restart_count")
    score = manifest.get("total_score")
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or not math.isclose(
            score,
            sum(row["score"] for row in details.values()),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("Resume total_score differs from recorded outcomes")
    return completed


def prepare_resume(plan_path: Path) -> dict:
    """Snapshot a stopped partial run without changing any native artifact.

    Stop the evaluator first. The returned receipt preserves all completed
    outcomes, including failures. Verify it again under the GPU lease immediately
    before starting the original commands. A receipt is not a runtime checkpoint.
    """
    plan_snapshot = _snapshot(plan_path)
    plan = _saved(plan_snapshot)
    result_path, manifest_path = _paths(plan)
    snapshots = {
        "plan": plan_snapshot,
        "result": _snapshot(result_path),
        "resume": _snapshot(manifest_path),
    }
    report = evaluation.audit(plan, result_path)
    if report["status"] != "incomplete" or not report["completed_episodes"]:
        raise ValueError(f"Require a valid, nonempty partial evaluation: {report}")
    completed = _check_manifest(
        plan, _saved(snapshots["result"]), _saved(snapshots["resume"])
    )
    for snapshot in snapshots.values():
        _unchanged(snapshot)
    return {
        "schema": SCHEMA,
        "status": "resume_prepared",
        "snapshots": snapshots,
        "completed_layout_ids": completed,
        "remaining_layout_ids": [
            index
            for index in range(plan["config"]["episodes"])
            if index not in completed
        ],
        "success_rate": None,
        "gpu_started": False,
        "resume_granularity": "episode_boundary",
    }


def _receipt(receipt_path: Path) -> tuple[dict, dict]:
    receipt = evaluation.read_json(receipt_path)
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "resume_prepared":
        raise ValueError("Require a prepared continuation receipt")
    snapshots = receipt["snapshots"]
    plan = _saved(snapshots["plan"])
    result_path, manifest_path = _paths(plan)
    for key, path in (("result", result_path), ("resume", manifest_path)):
        if Path(snapshots[key]["path"]).resolve() != path.resolve():
            raise ValueError(f"Receipt {key} path differs from planned run identity")
    completed = _check_manifest(
        plan, _saved(snapshots["result"]), _saved(snapshots["resume"])
    )
    remaining = [i for i in range(plan["config"]["episodes"]) if i not in completed]
    if (
        receipt.get("completed_layout_ids") != completed
        or receipt.get("remaining_layout_ids") != remaining
    ):
        raise ValueError("Receipt layout partition differs from saved outcomes")
    return receipt, plan


def verify_resume(receipt_path: Path) -> dict:
    """Recheck original bytes and live source/layout checks before GPU launch."""
    receipt, _ = _receipt(receipt_path)
    for snapshot in receipt["snapshots"].values():
        _unchanged(snapshot)
    current = prepare_resume(Path(receipt["snapshots"]["plan"]["path"]))
    if current != receipt:
        raise ValueError("Recovery receipt differs from live evaluation state")
    return {
        "status": "resume_verified",
        "completed_layout_ids": receipt["completed_layout_ids"],
        "remaining_layout_ids": receipt["remaining_layout_ids"],
        "success_rate": None,
        "gpu_started": False,
    }


def audit_continuation(receipt_path: Path) -> dict:
    """Audit the full planned run and require all previously recorded rows intact."""
    receipt, plan = _receipt(receipt_path)
    _unchanged(receipt["snapshots"]["plan"])
    result_path, manifest_path = _paths(plan)
    report = evaluation.audit(plan, result_path)
    if result_path.is_file():
        result = evaluation.read_json(result_path)
        previous = _saved(receipt["snapshots"]["result"])["details"]
        for key, row in previous.items():
            if result["details"].get(key) != row:
                report["errors"].append(f"Previously completed outcome changed: {key}")
        if manifest_path.exists():
            try:
                _check_manifest(plan, result, evaluation.read_json(manifest_path))
            except ValueError as error:
                report["errors"].append(str(error))
    if report["errors"]:
        report.update(status="failed", success_rate=None, partial_score_percent=None)
    report["preserved_episodes"] = len(receipt["completed_layout_ids"])
    return report


def main() -> int:
    """Prepare a CPU-only recovery receipt, verify it, or audit its continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("verify", "audit"):
        command = commands.add_parser(name)
        command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare_resume(args.plan)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        summary = {key: value for key, value in report.items() if key != "snapshots"}
    else:
        report = (
            verify_resume(args.receipt)
            if args.command == "verify"
            else audit_continuation(args.receipt)
        )
        summary = report
    print(json.dumps(summary, indent=2, allow_nan=False))
    return {
        "resume_prepared": 0,
        "resume_verified": 0,
        "passed": 0,
        "failed": 2,
        "incomplete": 3,
    }[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
