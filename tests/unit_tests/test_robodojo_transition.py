# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Native command acknowledgement, episode outcomes, and layout isolation."""

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from toolkits.rlt.robodojo_boundary import FIELDS
from toolkits.rlt.robodojo_collect import native_task_name, prior_outcomes
from toolkits.rlt.robodojo_transition import RoboDojoTransitions
from toolkits.rlt.robodojo_transition_probe import validate_probe_config


def test_native_collection_uses_explicit_configured_task():
    assert native_task_name({"task": "plug_in_charger"}) == "plug_in_charger"
    for config in ({}, {"task": ""}, {"task": "StackBowls"}, {"task": "../bad"}):
        with pytest.raises(ValueError, match="explicit snake_case"):
            native_task_name(config)


def test_compatibility_probe_accepts_only_selected_tasks_and_reference_backend():
    assert (
        validate_probe_config(
            {"task": "insert_key", "policy": {"backend": "openpi_reference"}}
        )
        == "insert_key"
    )
    with pytest.raises(ValueError, match="outside the selected"):
        validate_probe_config(
            {"task": "stack_bowls", "policy": {"backend": "openpi_reference"}}
        )
    with pytest.raises(ValueError, match="frozen OpenPI"):
        validate_probe_config({"task": "insert_key", "policy": {"backend": "rlt"}})


class NativeReward:
    def __init__(self, env):
        self.env = env
        self.calls = 0
        self.check_list = [[]]
        self.final_check_list = [[]]
        self.query_list = [[]]
        self.trigger_check_list = [[]]
        self.trigger_query_list = [[]]

    def get_reward(self, final_check=True):
        self.calls += 1
        if self.env.outcome == "final_failure" and final_check:
            self.env.success[0] = False
        return [1.0 if not self.check_list[0] or self.env.outcome == "success" else 0.0]

    def get_score(self):
        return [25.0 if self.env.outcome == "partial" else 0.0]


class NativeBoundary:
    """External simulator boundary; no model or training internals are mocked."""

    num_envs = 1
    step_lim = 3

    def __init__(self, layout):
        self.seed_manager = SimpleNamespace(
            seed_info={0: {"scene_layout": str(layout)}}
        )
        self.scene_manager = SimpleNamespace(
            layout_manager=SimpleNamespace(layout_valid=[True])
        )
        self.reward_manager = NativeReward(self)
        self.outcome = "ongoing"
        self.commands = []
        self.registrations = 0
        self.score_registrations = 0
        self.reset()

    def reset(self, seed=None):
        self.success = [True]
        self.end_flag = [False]
        self.take_action_cnt = [0]
        self.unstable_envs = set()
        self.reward_manager.check_list = [[]]

    def run_reward(self):
        self.registrations += 1
        self.reward_manager.check_list[0].append("stacked_bowls")

    def get_score(self):
        self.score_registrations += 1

    def get_obs(self):
        return {
            "instruction": "stack bowls",
            "state": {name: np.zeros(dim) for name, dim in FIELDS},
            "vision": {
                name: {"color": np.zeros((2, 2, 3), dtype=np.uint8)}
                for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
            },
        }

    def get_obs_batch(self, **kwargs):
        return [self.get_obs()]

    def take_action_batch(self, commands, env_idx_list):
        self.commands.extend(commands)
        if self.outcome == "exception":
            raise RuntimeError("PhysX failure")
        self.take_action_cnt[0] += 1
        if self.outcome == "failure":
            self.success[0] = False
        self.is_episode_end()

    def is_episode_end(self):
        at_limit = self.take_action_cnt[0] >= self.step_lim
        reward = self.reward_manager.get_reward(
            final_check=at_limit or not self.success[0]
        )[0]
        if reward > 0.999:
            self.end_flag[0], self.success[0] = True, True
        elif at_limit or not self.success[0]:
            self.end_flag[0], self.success[0] = True, False
        return self.end_flag[0]


@pytest.fixture(params=["contract", "pinned_native"])
def native(tmp_path, request):
    layout = tmp_path / "layout.json"
    layout.write_text('{"scene": "test"}')
    env = NativeBoundary(layout)
    if request.param == "pinned_native":
        # Execute the actual dependency method without importing Isaac at collection.
        source = Path(
            "/home/luokz/rlinf_rlt/robodojo_native_20261010/src/eval_client/eval_env.py"
        )
        if not source.exists():
            pytest.skip("Pinned native RoboDojo source is not installed")
        parsed = ast.parse(source.read_text())
        methods = [
            node
            for node in ast.walk(parsed)
            if isinstance(node, ast.FunctionDef) and node.name == "is_episode_end"
        ]
        assert len(methods) == 1
        namespace = {"deepcopy": copy.deepcopy}
        exec(
            compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
            namespace,
        )
        env.is_episode_end = namespace["is_episode_end"].__get__(env)
    digest = hashlib.sha256(layout.read_bytes()).hexdigest()
    return env, layout, digest


def adapter(native, **kwargs):
    env, _, digest = native
    return RoboDojoTransitions(
        env,
        layouts={0: digest},
        protected_evaluation_hashes=frozenset({digest}),
        mode="probe",
        **kwargs,
    )


@pytest.mark.parametrize(
    "outcome,at_limit,terminated,truncated,reason",
    [
        ("ongoing", False, False, False, None),
        ("partial", False, False, False, None),
        ("ongoing", True, False, True, "time_limit"),
        ("success", False, True, False, "success"),
        ("success", True, True, False, "success"),
        ("failure", False, True, False, "failure"),
        ("failure", True, True, False, "failure"),
        ("final_failure", True, True, False, "failure"),
    ],
)
def test_outcome_precedence_matches_native_without_double_reward(
    native, outcome, at_limit, terminated, truncated, reason
):
    env, _, _ = native
    wrapped = adapter(native)
    obs, info = wrapped.reset(0)
    assert obs["state"].shape == (14,) and info["training_eligible"] is False
    assert info["initial_process_score"] == 0.0
    env.step_lim = 1 if at_limit else 3
    env.outcome = outcome
    result = wrapped.step(np.zeros(14), source="reference_teacher")
    assert (result.terminated, result.truncated) == (terminated, truncated)
    assert result.info["end_reason"] == reason
    assert result.info["bootstrap_allowed"] is (not terminated)
    assert result.info["process_score"] == (25.0 if outcome == "partial" else 0.0)
    assert env.reward_manager.calls == 1
    assert "get_reward" not in vars(env.reward_manager)
    if terminated or truncated:
        with pytest.raises(RuntimeError, match="Reset"):
            wrapped.step(np.zeros(14))


def test_records_clipped_executed_grippers_and_preserves_arm_units(native):
    env, _, _ = native
    wrapped = adapter(native)
    wrapped.reset(0)
    action = np.full(14, 2.3)
    action[6], action[13] = -0.5, 1.8
    result = wrapped.step(action, source="planner")
    np.testing.assert_array_equal(
        result.info["executed_action"], np.concatenate(list(env.commands[-1].values()))
    )
    assert result.info["executed_action"][0] == pytest.approx(2.3)
    assert list(result.info["executed_action"][[6, 13]]) == [0.0, 1.0]
    assert result.info["proposed_action"][6] == -0.5


def test_exception_restores_reward_observer_and_invalidates_episode(native):
    env, _, _ = native
    wrapped = adapter(native)
    wrapped.reset(0)
    env.outcome = "exception"
    with pytest.raises(RuntimeError, match="PhysX"):
        wrapped.step(np.zeros(14))
    assert "get_reward" not in vars(env.reward_manager)
    with pytest.raises(RuntimeError, match="Reset"):
        wrapped.step(np.zeros(14))


@pytest.mark.parametrize("action", [np.zeros(7), np.full(14, np.nan)])
def test_rejects_invalid_action_before_native_execution(native, action):
    env, _, _ = native
    wrapped = adapter(native)
    wrapped.reset(0)
    with pytest.raises(ValueError):
        wrapped.step(action)
    assert env.commands == []


def test_training_rejects_protected_layout_and_probe_cannot_reset_changed_file(native):
    env, path, digest = native
    with pytest.raises(ValueError, match="disjoint"):
        RoboDojoTransitions(
            env,
            layouts={0: digest},
            protected_evaluation_hashes=frozenset({digest}),
            mode="training",
        )
    wrapped = adapter(native)
    path.write_text('{"scene": "changed"}')
    with pytest.raises(ValueError, match="changed"):
        wrapped.reset(0)


def test_unstable_layouts_are_not_silently_replaced(native):
    env, _, _ = native
    wrapped = adapter(native)
    wrapped.reset(0)
    env.unstable_envs.add(0)
    with pytest.raises(RuntimeError, match="unstable"):
        wrapped.step(np.zeros(14))


def test_reset_registers_reward_and_score_once_per_episode(native):
    env, _, _ = native
    wrapped = adapter(native)
    for episode in range(2):
        _, info = wrapped.reset(0)
        assert info["reward_condition_groups"] == 1
        assert env.registrations == episode + 1
        assert env.score_registrations == episode + 1
        result = wrapped.step(np.zeros(14))
        assert result.reward == 0 and not result.terminated


def test_empty_task_conditions_cannot_be_accepted_as_success(native):
    env, _, _ = native
    env.run_reward = lambda: None
    wrapped = adapter(native)
    with pytest.raises(RuntimeError, match="reward conditions"):
        wrapped.reset(0)
    with pytest.raises(RuntimeError, match="Reset"):
        wrapped.step(np.zeros(14))
    assert not env.commands


def test_external_control_step_is_detected_before_action(native):
    env, _, _ = native
    wrapped = adapter(native)
    wrapped.reset(0)
    env.take_action_cnt[0] += 1
    with pytest.raises(RuntimeError, match="outside"):
        wrapped.step(np.zeros(14))
    assert not env.commands


def test_invalid_layout_discovered_during_observation_emits_no_transition(native):
    env, _, _ = native
    wrapped = adapter(native)
    wrapped.reset(0)
    original = env.get_obs

    def invalid_observation():
        env.scene_manager.layout_manager.layout_valid[0] = False
        return original()

    env.get_obs = invalid_observation
    with pytest.raises(RuntimeError, match="unstable"):
        wrapped.step(np.zeros(14))
    with pytest.raises(RuntimeError, match="Reset"):
        wrapped.step(np.zeros(14))


def test_collection_resume_keeps_only_explicit_completed_layouts(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"layouts": []}')
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (attempt / "provenance.json").write_text('{"layout_sha256": "' + digest + '"}')
    (attempt / "collection-failure.json").write_text(
        """{
          "outcomes": [
            {"layout_id": 0, "role": "train", "state": "complete",
             "ticks": 10, "end_reason": "success"},
            {"layout_id": 1, "role": "train", "state": "complete",
             "ticks": 800, "end_reason": "time_limit"},
            {"layout_id": 2, "role": "train", "state": "failed",
             "error": "unstable"}
          ]
        }"""
    )
    rows = prior_outcomes(attempt, manifest)
    assert [row["layout_id"] for row in rows] == [0, 1]
    assert all(row["artifact_root"] == str(attempt) for row in rows)
    manifest.write_text('{"layouts": [1]}')
    with pytest.raises(ValueError, match="different layout manifest"):
        prior_outcomes(attempt, manifest)
