# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Finite GPU experiment queue ownership and completion contracts."""

import json
import sys

import pytest

from toolkits.rlt.experiment_queue import run_plan, validate_plan


def job(tmp_path, name, code, **kwargs):
    return {
        "name": name,
        "argv": [sys.executable, "-c", code],
        "cwd": str(tmp_path),
        "timeout_seconds": 2,
        "requires_gpu": False,
        **kwargs,
    }


def test_queue_requires_gpu2_and_finite_deadline(tmp_path):
    plan = {"gpu": 2, "hours": 1, "jobs": [job(tmp_path, "smoke", "pass")]}
    validate_plan(plan)
    for changes in ({"gpu": 0}, {"gpu": 1}, {"hours": 0}, {"hours": 13}):
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
