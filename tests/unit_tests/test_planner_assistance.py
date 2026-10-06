# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for the planner-assistance component, without GPU dependencies."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from rlinf.envs.sim.maniskill.planner_assistance import (
    PegEvidence,
    RecoveryConfig,
    RecoveryTrigger,
    joint_target_to_delta,
)


def test_report_counts_executed_work_not_just_outer_steps():
    from toolkits.rlt.planner_report import summarize_budget

    history = {
        "env/num_trajectories": {0: 1, 1: 1},
        "env/episode_len": {0: 137, 1: 500},
        "env/success_once": {0: 1, 1: 0},
        "env/planner_steps": {0: 60, 1: 100},
        "train/rlt/critic_updates_run": {0: 128, 1: 20},
    }
    result = summarize_budget(history)
    assert result["training_episodes"] == 2
    assert result["training_control_ticks"] == 637
    assert result["planner_ticks"] == 160
    assert result["critic_updates"] == 148
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_budget({**history, "env/episode_len": {0: 137}})


def evidence(**kwargs):
    return replace(PegEvidence(0, False, False, -0.2, 0.08, True), **kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "approach_timeout",
        "stall_ticks",
        "max_attempts",
        "max_planner_ticks",
        "min_progress",
    ],
)
def test_recovery_rejects_unbounded_config(field):
    with pytest.raises(ValueError):
        RecoveryConfig(**{field: 0})


def test_trigger_timeout_and_attempt_budget():
    trigger = RecoveryTrigger(RecoveryConfig())
    assert trigger.observe(evidence(tick=179), critical_phase=False) is None
    assert (
        trigger.observe(evidence(tick=180), critical_phase=False) == "approach_timeout"
    )
    assert trigger.observe(evidence(tick=400), critical_phase=False) is None


def test_trigger_lost_grasp_not_initially_open():
    trigger = RecoveryTrigger(RecoveryConfig())
    assert trigger.observe(evidence(), critical_phase=False) is None
    assert (
        trigger.observe(evidence(tick=10, grasped=True), critical_phase=False) is None
    )
    assert trigger.observe(evidence(tick=20), critical_phase=False) == "lost_grasp"


def test_trigger_uses_executed_ticks_and_resets_when_progress_improves():
    trigger = RecoveryTrigger(RecoveryConfig())
    e = evidence(grasped=True, hole_x=-0.1, hole_yz=0.02)
    assert trigger.observe(e, critical_phase=True) is None
    assert trigger.observe(replace(e, tick=20), critical_phase=True) is None
    e = replace(e, tick=25, hole_x=-0.09)
    assert trigger.observe(e, critical_phase=True) is None
    assert trigger.observe(replace(e, tick=54), critical_phase=True) is None
    assert (
        trigger.observe(replace(e, tick=55), critical_phase=True) == "insertion_stall"
    )


@pytest.mark.parametrize("kw", [{"success": True}, {"recoverable": False}])
def test_trigger_does_not_take_over_finished_or_out_of_workspace_states(kw):
    trigger = RecoveryTrigger(RecoveryConfig())
    assert trigger.observe(evidence(tick=400, **kw), critical_phase=False) is None
    assert trigger.attempts == 0


def test_delta_converter_is_current_state_based_and_bounded():
    q = np.arange(7) * 0.1
    target = q + np.array([0.02, -0.03, 0.2, -0.2, 0, 0.01, -0.01])
    np.testing.assert_allclose(
        joint_target_to_delta(target, q), [0.2, -0.3, 1, -1, 0, 0.1, -0.1], atol=1e-6
    )
    np.testing.assert_allclose(joint_target_to_delta(q, q), 0)
    with pytest.raises(ValueError):
        joint_target_to_delta(np.full(7, np.nan), q)


def test_pilot_arms_differ_only_in_assistance_and_use_distinct_eval_seeds():
    from toolkits.rlt.planner_experiment import experiment_overrides

    plain = experiment_overrides("none", 20, 20)
    assisted = experiment_overrides("planner", 20, 20)
    assert [(a, b) for a, b in zip(plain, assisted) if a != b] == [
        (
            "env.train.planner_assistance.enable=False",
            "env.train.planner_assistance.enable=True",
        )
    ]
    import ast

    seeds = ast.literal_eval(plain[-1].split("=", 1)[1])
    assert len(set(seeds)) == 20 and min(seeds) >= 12026
    for component in ("actor", "rollout", "env"):
        assert f"cluster.component_placement.{component}=2-2" in plain
    with pytest.raises(ValueError):
        experiment_overrides("none", 0, 20)
    with pytest.raises(ValueError):
        experiment_overrides("planner", 20, 21)


def test_pilot_storage_refuses_output_overwrite(tmp_path):
    from toolkits.rlt.planner_experiment import validate_storage

    with pytest.raises(ValueError, match="Output must be new"):
        validate_storage(tmp_path, Path("/dev/shm"))


def test_pilot_report_does_not_call_reference_only_rollout_actor_success():
    from copy import deepcopy

    from toolkits.rlt.planner_report import compare

    plain = {
        "launch": {"arm": "none", "eval_seeds": [12026]},
        "config": {"budget": 20},
        "scalars": {"train/rlt/ready_for_online": 0, "eval/success_once": 0.4},
    }
    assisted = deepcopy(plain)
    assisted["launch"]["arm"] = "planner"
    assisted["scalars"]["train/rlt/ready_for_online"] = 1
    assert not compare(plain, assisted)["both_learners_finished_warmup"]
    assisted["config"]["budget"] = 30
    with pytest.raises(ValueError, match="configurations differ"):
        compare(plain, assisted)


def test_planner_replay_keeps_executed_action_source_and_original_reference():
    from rlinf.data.schema.embodied_types import (
        EnvPart,
        EnvTransition,
        PolicyOutput,
        PolicyPart,
        TrajectoryStep,
    )

    obs = {
        "z_rl": torch.zeros(1, 3),
        "proprio": torch.zeros(1, 2),
        "ref_chunk": torch.zeros(1, 2, 2),
    }
    policy = PolicyPart(
        sources=[],
        obs={},
        output=PolicyOutput(
            forward_inputs={
                **obs,
                "action": torch.zeros(1, 4),
                "record_transition": torch.zeros(1, 1, dtype=torch.bool),
            },
            intervene_flags=torch.zeros(1, 2, dtype=torch.bool),
        ),
    )
    transition = EnvTransition(
        intervene_actions=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        intervene_flags=torch.tensor([[False, True]]),
        planner_flags=torch.tensor([[False, True]]),
    )
    transition = EnvTransition.merge(transition.split([1]))
    env = EnvPart(sources=[], transition=transition, next_rlt_obs=obs)
    step = TrajectoryStep.from_parts(
        policy,
        env,
        rewards=None,
        collect_prev_infos=False,
        collect_transitions=True,
        enable_rlt=True,
        include_final_value=False,
    )
    assert torch.equal(step.actions, torch.tensor([[0.0, 0.0, 0.3, 0.4]]))
    assert step.intervene_flags.tolist() == [[False, False, True, True]]
    assert torch.equal(step.curr_obs["ref_chunk"], obs["ref_chunk"])
    assert step.forward_inputs["planner_flags"].tolist() == [[False, True]]
    assert set(step.curr_obs) == set(step.next_obs) == set(obs)
    assert step.forward_inputs["record_transition"].item()

    # Terminal replay substitutes current features for next features. Inserting
    # both types must preserve a stable schema while retaining provenance.
    from rlinf.data.storage.replay.buffer import TrajectoryCache

    cache = TrajectoryCache(max_size=2)
    for row, next_obs in enumerate((step.next_obs, step.curr_obs)):
        cache.put(
            row,
            {
                "curr_obs": step.curr_obs,
                "next_obs": next_obs,
                "forward_inputs": step.forward_inputs,
            },
        )


def test_planner_provenance_rejects_unexecuted_or_unmarked_actions():
    from rlinf.data.schema.embodied_types import EnvTransition

    with pytest.raises(ValueError, match="requires executed"):
        EnvTransition(planner_flags=torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="subset"):
        EnvTransition(
            planner_flags=torch.ones(1, 2, dtype=torch.bool),
            intervene_flags=torch.zeros(1, 2, dtype=torch.bool),
            intervene_actions=torch.zeros(1, 16),
        )


def test_bc_rejects_missing_reference_in_normal_mode(tmp_path):
    import hashlib

    from toolkits.rlt.actor_bc import load_episodes

    data = {
        "z_rl": torch.zeros(12, 2048),
        "proprio": torch.zeros(12, 9),
        "actions": torch.zeros(12, 8),
        "frame_index": torch.arange(12),
    }
    torch.save(data, tmp_path / "episode.pt")
    torch.save(
        {
            "complete": True,
            "feature_contract": {
                "control_mode": "pd_joint_delta_pos",
                "control_freq": 10,
                "action_space": "environment_pd_joint_delta_pos",
            },
            "episodes": [
                {
                    "id": "0",
                    "file": "episode.pt",
                    "sha256": hashlib.sha256(
                        (tmp_path / "episode.pt").read_bytes()
                    ).hexdigest(),
                }
            ],
        },
        tmp_path / "manifest.pt",
    )
    with pytest.raises(ValueError, match="lacks VLA"):
        load_episodes(tmp_path, horizon=10, reference_mode="cached")
    episodes, _ = load_episodes(tmp_path, horizon=10, reference_mode="zero-diagnostic")
    assert episodes[0]["target"].shape == (3, 10, 8)


def test_bc_distillation_uses_reference_not_demonstration(tmp_path):
    import hashlib

    from toolkits.rlt.actor_bc import fit

    cache = tmp_path / "cache"
    cache.mkdir()
    records = []
    for episode in range(4):
        path = cache / f"episode_{episode}.pt"
        torch.save(
            {
                "z_rl": torch.zeros(12, 2048),
                "proprio": torch.zeros(12, 9),
                "actions": torch.full((12, 8), -0.25),
                "ref_chunk": torch.full((12, 10, 8), 0.25),
                "frame_index": torch.arange(12),
            },
            path,
        )
        records.append(
            {
                "id": episode,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    torch.save(
        {
            "complete": True,
            "episodes": records,
            "feature_contract": {
                "control_mode": "pd_joint_delta_pos",
                "control_freq": 10,
                "action_space": "environment_pd_joint_delta_pos",
                "reference_source": "frozen_vla_pre_action",
            },
        },
        cache / "manifest.pt",
    )
    report = fit(
        cache,
        tmp_path / "distill",
        steps=1,
        seed=4,
        reference_mode="cached",
        target_source="reference",
    )
    assert report["reference_validation"]["mse"] == 0
    assert set(report["train_episodes"]).isdisjoint(report["validation_episodes"])
    checkpoint = torch.load(tmp_path / "distill/best_actor.pt", weights_only=True)
    assert checkpoint["target_source"] == "reference"
    assert not report["eligible_for_stage2_initialization"]
    assert report["closed_loop_success_rate"] is None
    demo = fit(
        cache,
        tmp_path / "demo",
        steps=1,
        seed=4,
        reference_mode="cached",
        target_source="demonstration",
    )
    assert demo["reference_validation"]["mse"] == 0.25
    with pytest.raises(ValueError, match="actual cached"):
        fit(
            cache,
            tmp_path / "invalid",
            steps=1,
            seed=4,
            reference_mode="zero-diagnostic",
            target_source="reference",
        )


@pytest.mark.skipif(
    os.environ.get("RLT_PLANNER_PHYSICS") != "1",
    reason="Opt-in CPU SAPIEN/MPLib physics integration",
)
@pytest.mark.parametrize("as_numpy", [False, True])
def test_single_env_planner_chunk_execution_and_terminal_freeze(as_numpy):
    from omegaconf import OmegaConf

    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv
    from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
        register_rlinf_peg_insertion_side_variants,
    )

    register_rlinf_peg_insertion_side_variants()
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "examples/embodiment/config/env/maniskill_rlt.yaml")
    cfg.total_num_envs = 1
    cfg.seed = 2026
    cfg.wrap_obs_mode = "simple"
    cfg.video_cfg.video_base_dir = "/dev/shm"
    cfg.init_params.obs_mode = "state"
    cfg.init_params.sim_backend = "cpu"
    cfg.init_params.render_backend = "none"
    cfg.init_params.render_mode = None
    cfg.init_params.max_episode_steps = 500
    cfg.planner_assistance = {"enable": True, "approach_timeout": 20}
    env = ManiskillRLTEnv(cfg, 1, 0, 1, None)
    try:
        env.reset(seed=2026)
        proposed = torch.zeros(1, 10, 8)
        proposed[..., -1] = 1
        count = 0
        done = False
        for _ in range(50):
            _, rewards, terminated, truncated, infos = env.chunk_step(
                proposed.numpy() if as_numpy else proposed
            )
            last = infos[-1]
            mask = last["planner_flags"]
            actual = last["intervene_action"].reshape(1, 10, 8)
            assert actual.abs().max() <= 1
            assert torch.equal(actual[~mask], proposed[~mask])
            count += int(mask.sum())
            done = bool((terminated | truncated).any())
            if done:
                break
        assert done and count > 0
        assert bool(last["success_current"][0])
        outcomes = [
            last["episode"][key].item()
            for key in (
                "success_with_actor_phase",
                "success_before_actor_phase",
                "failure_before_actor_phase",
                "failure_after_actor_phase",
            )
        ]
        assert sum(outcomes) == 1
        assert any(outcomes[:2])
        q = env.env.unwrapped.agent.robot.get_qpos().clone()
        elapsed = env.elapsed_steps.clone()
        _, rewards, _, _, infos = env.chunk_step(proposed)
        assert not infos[-1]["planner_flags"].any()
        assert rewards.sum() == 0
        assert torch.equal(env.elapsed_steps, elapsed)
        assert torch.equal(env.env.unwrapped.agent.robot.get_qpos(), q)
        env.reset(seed=2027)
        _, _, _, _, infos = env.chunk_step(proposed)
        assert not infos[-1]["planner_flags"].any()
    finally:
        env.env.close()


@pytest.mark.skipif(
    os.environ.get("RLT_PLANNER_PHYSICS") != "1",
    reason="Opt-in CPU SAPIEN evaluation seed integration",
)
def test_eval_seed_sequence_is_distinct_and_repeats_the_same_initial_states():
    from omegaconf import OmegaConf

    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv
    from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
        register_rlinf_peg_insertion_side_variants,
    )

    register_rlinf_peg_insertion_side_variants()
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "examples/embodiment/config/env/maniskill_rlt.yaml")
    cfg.total_num_envs = 1
    cfg.policy_mode = "eval"
    cfg.evaluation_reset_seeds = [12026, 12027]
    cfg.use_fixed_reset_state_ids = True
    cfg.wrap_obs_mode = "simple"
    cfg.video_cfg.video_base_dir = "/dev/shm"
    cfg.init_params.obs_mode = "state"
    cfg.init_params.sim_backend = "cpu"
    cfg.init_params.render_backend = "none"
    cfg.init_params.render_mode = None
    cfg.init_params.max_episode_steps = 500
    env = ManiskillRLTEnv(cfg, 1, 0, 1, None)
    try:
        poses = []
        for _ in range(4):
            env.reset()
            poses.append(env.env.unwrapped.peg.pose.p.clone())
        assert not torch.allclose(poses[0], poses[1])
        assert torch.equal(poses[0], poses[2])
        assert torch.equal(poses[1], poses[3])
    finally:
        env.env.close()
