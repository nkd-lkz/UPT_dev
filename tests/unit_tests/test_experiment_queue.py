# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Finite GPU experiment queue ownership and completion contracts."""

import fcntl
import json
import sys

import pytest

from toolkits.rlt.experiment_queue import run_plan, validate_plan


@pytest.fixture(autouse=True)
def isolated_leases(tmp_path, monkeypatch):
    monkeypatch.setattr("toolkits.rlt.experiment_queue.LEASE_DIRECTORY", tmp_path)


def job(tmp_path, name, code, **kwargs):
    return {
        "name": name,
        "argv": [sys.executable, "-c", code],
        "cwd": str(tmp_path),
        "timeout_seconds": 2,
        "requires_gpu": False,
        **kwargs,
    }


def test_queue_requires_explicit_gpu_and_finite_deadline(tmp_path):
    plan = {"gpu": 2, "hours": 1, "jobs": [job(tmp_path, "smoke", "pass")]}
    validate_plan(plan)
    for gpu in (0, 1, 2):
        validate_plan({**plan, "gpu": gpu})
    for changes in (
        {"gpu": -1},
        {"gpu": True},
        {"gpu": None},
        {"hours": 0},
        {"hours": 13},
        {"hours": float("nan")},
    ):
        with pytest.raises(ValueError):
            validate_plan({**plan, **changes})
    plan["jobs"][0]["depends_on"] = ["future"]
    with pytest.raises(ValueError, match="earlier"):
        validate_plan(plan)


def test_queue_propagates_failures_and_rejects_incomplete_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"complete": false}')
    plan = {
        "gpu": 2,
        "hours": 0.01,
        "jobs": [
            job(tmp_path, "fails", "raise SystemExit(7)"),
            job(
                tmp_path,
                "skipped",
                "raise AssertionError('must not start')",
                depends_on=["fails"],
            ),
            job(tmp_path, "incomplete", "pass", manifest=str(manifest)),
            job(tmp_path, "passes", "print('completed')"),
        ],
    }
    output = tmp_path / "queue"
    result = run_plan(plan, output)
    assert result["state"] == "finished"
    assert result["jobs"]["fails"]["exit_code"] == 7
    assert result["jobs"]["skipped"]["state"] == "skipped_dependency"
    assert not (output / "skipped.log").exists()
    assert result["jobs"]["incomplete"]["state"] == "failed"
    assert result["jobs"]["passes"]["state"] == "passed"
    assert json.loads((output / "status.json").read_text()) == result


def test_queue_timeout_does_not_mark_success(tmp_path):
    plan = {
        "gpu": 2,
        "hours": 0.01,
        "jobs": [
            job(tmp_path, "slow", "import time; time.sleep(60)", timeout_seconds=0.1),
        ],
    }
    result = run_plan(plan, tmp_path / "queue")
    assert result["jobs"]["slow"]["state"] == "timeout"


def test_queue_lease_is_per_gpu_and_never_steals_an_existing_lease(tmp_path):
    plan = {"gpu": 2, "hours": 0.01, "jobs": [job(tmp_path, "smoke", "pass")]}
    with (tmp_path / "rlt-research-gpu2.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="GPU 2 lease"):
            run_plan(plan, tmp_path / "blocked")
        result = run_plan({**plan, "gpu": 1}, tmp_path / "other_gpu")
    assert result["jobs"]["smoke"]["state"] == "passed"


def test_busy_gpu_never_starts_work_and_marks_remaining_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "toolkits.rlt.experiment_queue.gpu_idle", lambda gpu: (False, "busy")
    )
    plan = {
        "gpu": 0,
        "hours": 0.00001,
        "jobs": [
            {
                **job(tmp_path, name, "raise AssertionError('must not run')"),
                "requires_gpu": True,
            }
            for name in ("first", "second")
        ],
    }
    result = run_plan(plan, tmp_path / "queue")
    assert result["state"] == "deadline"
    assert all(
        row["state"] == "not_started_deadline" for row in result["jobs"].values()
    )
    assert not list((tmp_path / "queue").glob("*.log"))
