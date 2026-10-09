# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fixed multi-suite experiment declaration and complete-grid aggregation."""

import copy
import json
from pathlib import Path

import pytest

from toolkits.rlt.experiment_queue import validate_plan
from toolkits.rlt.pi05_baseline import SUITE_HORIZONS
from toolkits.rlt.pi05_campaign import FINAL_ARMS
from toolkits.rlt.pi05_matrix import (
    TRAINING_SEEDS,
    aggregate_cells,
    prepare_matrix,
    summarize_matrix,
    validate_matrix,
)


def test_matrix_declares_all_tasks_seeds_and_paired_budgets_before_running(tmp_path):
    root = tmp_path / "matrix"
    repo = Path(__file__).resolve().parents[2]
    matrix = prepare_matrix(root, repo, tmp_path, tmp_path, [0, 1], "team")
    assert matrix["planned_eval_episodes"] == 48000
    assert len(matrix["cells"]) == 120
    assert len({(c["suite"], c["task"], c["seed"]) for c in matrix["cells"]}) == 120
    assert {c["seed"] for c in matrix["cells"][:40]} == {42}
    all_names = set()
    for gpu in (0, 1):
        plan = json.loads((root / "plans" / f"gpu{gpu}.json").read_text())
        validate_plan(plan)
        assert plan["wait_for_lease"] is True
        assert len(plan["jobs"]) == 60 * 13
        names = {j["name"] for j in plan["jobs"]}
        assert not all_names & names
        all_names.update(names)
        assert {c["suite"] for c in matrix["cells"] if c["gpu"] == gpu} == set(
            SUITE_HORIZONS
        )
    validate_matrix(root)
    status = summarize_matrix(root)
    assert status["completed_cells"] == 0
    assert status["four_suite_means"] == {}
    with pytest.raises(FileExistsError):
        prepare_matrix(root, repo, tmp_path, tmp_path, [0, 1], "team")
    case_path = Path(matrix["cells"][0]["root"]) / "cases/heldout.json"
    cases = json.loads(case_path.read_text())
    cases[0]["seed"] += 1
    case_path.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="Declared plan changed"):
        validate_matrix(root)


def test_matrix_aggregation_never_drops_failed_or_missing_tasks():
    cells = [
        {
            "suite": suite,
            "task": task,
            "seed": seed,
            "state": "passed",
            "comparison": {
                arm: {"success_rate": 0.5 + (seed - 42) / 10} for arm in FINAL_ARMS
            },
        }
        for suite in SUITE_HORIZONS
        for task in range(10)
        for seed in TRAINING_SEEDS
    ]
    result = aggregate_cells(cells, list(TRAINING_SEEDS))
    assert result["four_suite_means"]["q_bc_eval"] == pytest.approx(0.6)
    assert result["suites"]["libero_10"]["arms"]["q_bc_eval"][
        "seed_std"
    ] == pytest.approx(0.1)
    missing = aggregate_cells(cells[:-1], list(TRAINING_SEEDS))
    assert not missing["four_suite_means"]
    assert missing["suites"]["libero_10"]["arms"] == {}
    failed = copy.deepcopy(cells)
    failed[-1]["state"] = "failed"
    assert not aggregate_cells(failed, list(TRAINING_SEEDS))["four_suite_means"]
    duplicate = cells[:-1] + [cells[-2]]
    assert not aggregate_cells(duplicate, list(TRAINING_SEEDS))["four_suite_means"]
