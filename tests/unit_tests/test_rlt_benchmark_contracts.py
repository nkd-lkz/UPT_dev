# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Portable online-RLT contracts and RoboDojo task-plan audits."""

import numpy as np
import pytest

from toolkits.rlt.benchmark_contracts import CONTRACTS, OnlineRLTBoundary
from toolkits.rlt.robodojo_task_plan import CANDIDATES, audit


class FakeEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.calls = []

    def reset(self, **kwargs):
        return kwargs

    def step(self, action, *, auto_reset):
        self.calls.append((action.copy(), auto_reset))
        return (
            {"state": np.ones((self.num_envs, 3))},
            np.ones(self.num_envs),
            np.zeros(self.num_envs, dtype=bool),
            np.ones(self.num_envs, dtype=bool),
            {"source": "fake"},
        )


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_online_boundary_preserves_single_tick_and_terminal_observation(name):
    contract = CONTRACTS[name]
    env = FakeEnv(2)
    boundary = OnlineRLTBoundary(env, contract, num_envs=2)
    action = np.zeros((2, contract.action_dim))
    result = boundary.step(action)
    assert env.calls[0][1] is False
    assert result.observation["state"].shape == (2, 3)
    assert result.truncated.tolist() == [True, True]
    np.testing.assert_array_equal(result.proposed_action, action)


@pytest.mark.parametrize("bad", [np.zeros((1, 7)), np.full((2, 7), np.nan)])
def test_online_boundary_rejects_invalid_action_before_execution(bad):
    env = FakeEnv(2)
    boundary = OnlineRLTBoundary(env, CONTRACTS["libero_pro"], num_envs=2)
    with pytest.raises(ValueError):
        boundary.step(bad)
    assert env.calls == []


def test_task_plan_reports_missing_layouts_as_not_ready(tmp_path):
    config_root = tmp_path / "task/RoboDojo/config"
    source_root = tmp_path / "task/RoboDojo/tasks"
    config_root.mkdir(parents=True)
    source_root.mkdir(parents=True)
    (config_root / "_task.yml").write_text(
        "common: {robot_config: dual_x5}\ntasks: {}\n"
    )
    for candidate in CANDIDATES:
        (config_root / f"{candidate.name}.yml").write_text(
            "Rigid:\n  - category:\n      - name: fixture\n"
        )
        (source_root / f"{candidate.name}.py").write_text(
            f"class {candidate.name}:\n    pass\n"
        )
    report = audit(tmp_path)
    assert not report["primary_ready"]
    assert all(row["eval_layout_count"] == 0 for row in report["tasks"])


def test_task_plan_accepts_dual_x5_tasks_with_layouts(tmp_path):
    config_root = tmp_path / "task/RoboDojo/config"
    source_root = tmp_path / "task/RoboDojo/tasks"
    layout_root = tmp_path / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
    training_layout_root = tmp_path / "training-layouts"
    config_root.mkdir(parents=True)
    source_root.mkdir(parents=True)
    layout_root.mkdir(parents=True)
    training_layout_root.mkdir(parents=True)
    (config_root / "_task.yml").write_text(
        "common: {robot_config: dual_x5}\ntasks: {}\n"
    )
    for candidate in CANDIDATES:
        (config_root / f"{candidate.name}.yml").write_text(
            "Rigid:\n  - category:\n      - name: fixture\n"
        )
        (source_root / f"{candidate.name}.py").write_text(
            f"class {candidate.name}:\n    pass\n"
        )
        (layout_root / f"{candidate.name}_0.json").write_text("{}\n")
        (training_layout_root / f"{candidate.name}_0.json").write_text("{}\n")
    (tmp_path / "Assets/Object/RoboDojo/Rigid/fixture").mkdir(parents=True)
    report = audit(tmp_path, training_layouts_root=training_layout_root)
    assert report["primary_ready"]
    assert all(row["online_rlt_ready"] for row in report["tasks"])
