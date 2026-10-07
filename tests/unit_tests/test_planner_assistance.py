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
        experiment_overrides("planner", 20, 501)


def test_pilot_storage_refuses_output_overwrite(tmp_path):
    from toolkits.rlt.planner_experiment import validate_storage

    with pytest.raises(ValueError, match="Output must be new"):
        validate_storage(tmp_path, Path("/dev/shm"))


def test_fixture_population_preserves_setup_failures(tmp_path):
    import json

    from toolkits.rlt.planner_experiment import experiment_overrides, fixture_population

    rows = [
        {
            "seed": seed,
            "case": "insertion_fixture",
            "source": "planner_prefix_then_constant_control",
            "fixture_ready": seed != 12027,
            "success": False,
            "failure": "none" if seed != 12027 else "planning_failed:grasp",
            "prefix_steps": 90,
        }
        for seed in range(12026, 12029)
    ]
    path = tmp_path / "probe.json"
    path.write_text(json.dumps({"results": rows}))
    population = fixture_population(path, [12026, 12027, 12028])
    assert population["coverage"] == 2 / 3
    assert population["selected_seeds"] == [12026, 12028]
    assert population["excluded"] == [
        {"seed": 12027, "reason": "planning_failed:grasp"}
    ]
    overrides = experiment_overrides(
        "none", 1, 2, eval_seeds=population["selected_seeds"]
    )
    assert overrides[-1] == "env.eval.evaluation_reset_seeds=[12026, 12028]"
    with pytest.raises(ValueError, match="exact candidate"):
        fixture_population(path, [12026, 12028])
    rows[0]["success"] = True
    path.write_text(json.dumps({"results": rows}))
    with pytest.raises(ValueError, match="already solves"):
        fixture_population(path, [12026, 12027, 12028])
    with pytest.raises(ValueError, match="distinct"):
        experiment_overrides("none", 1, 2, eval_seeds=[12026, 12026])


def test_checkpoint_report_matches_within_scope_and_retains_fixture_coverage(tmp_path):
    import json

    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    from toolkits.rlt.planner_report import checkpoint_evaluations

    def write_run(name, scope, seeds):
        path = tmp_path / name
        path.mkdir()
        population = (
            {
                "selected_seeds": seeds,
                "candidate_seeds": [12026, 12027],
                "coverage": 0.5,
            }
            if scope == "insertion"
            else None
        )
        (path / "launch.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "exit_code": 0,
                    "checkpoint": name,
                    "eval_seeds": seeds,
                    "eval_scope": scope,
                    "fixture_population": population,
                }
            )
        )
        config = {
            "env": {
                "eval": {
                    "planner_assistance": {"enable": False},
                    "rlt_policy_switch": {"expert_takeover": {"enable": False}},
                    "insertion_fixture": scope == "insertion",
                    "max_episode_steps": 500,
                }
            },
            "runner": {"logger": {}},
            "actor": {"model": {"dim": 8}},
            "rollout": {"rlt_feature_model": {"revision": "same"}},
        }
        (path / "resolved.yaml").write_text(json.dumps(config))
        writer = EventFileWriter(str(path / "tensorboard"))
        values = {"eval/success_once": 0.5, "eval/num_trajectories": len(seeds)}
        if scope == "insertion":
            values["eval/fixture_prefix_ticks"] = 90
        writer.add_event(
            Event(
                wall_time=1,
                step=0,
                summary=Summary(
                    value=[
                        Summary.Value(tag=k, simple_value=v) for k, v in values.items()
                    ]
                ),
            )
        )
        writer.close()
        return path

    full = write_run("full", "full_task", [12026, 12027])
    first = write_run("first", "insertion", [12026])
    second = write_run("second", "insertion", [12026])
    result = checkpoint_evaluations([full, first, second])
    assert result["evaluation_seeds_by_scope"]["insertion"] == [12026]
    assert result["results"][1]["fixture_population"]["coverage"] == 0.5
    config = json.loads((second / "resolved.yaml").read_text())
    config["env"]["eval"]["max_episode_steps"] = 1000
    (second / "resolved.yaml").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Within-scope"):
        checkpoint_evaluations([first, second])


def test_diagnostic_evaluation_rejects_changed_feature_contract(tmp_path):
    import json

    from toolkits.rlt.planner_experiment import validate_diagnostic_checkpoint

    model = {"z_dim": 2048}
    features = {"model_path": "stage1"}
    (tmp_path / "contract.json").write_text(
        json.dumps(
            {"format": "rlt_corrections_v1", "model": model, "feature_model": features}
        )
    )
    checkpoint = tmp_path / "bc_only/model.pt"
    validate_diagnostic_checkpoint(checkpoint, model, features)
    with pytest.raises(ValueError, match="original model/features"):
        validate_diagnostic_checkpoint(
            checkpoint, model, {"model_path": "other-stage1"}
        )


def test_new_protocol_and_scopes_are_explicit():
    from toolkits.rlt.planner_experiment import experiment_overrides

    result = experiment_overrides(
        "planner",
        1000,
        50,
        seed=1235,
        protocol="preinsert_handoff",
        control_budget=30000,
    )
    assert "env.train.seed=1235" in result and "actor.seed=1235" in result
    assert "+env.train.training_control_budget=30000" in result
    evaluation = experiment_overrides(
        "none", 1, 50, eval_scope="insertion", checkpoint=Path("/tmp/test-model.pt")
    )
    assert "+env.eval.insertion_fixture=True" in evaluation
    assert "algorithm.rlt_schedule.enable=False" in evaluation
    for kwargs in (
        {"control_budget": 1, "update_budget": 1},
        {"eval_scope": "mixed"},
        {"checkpoint": Path("x"), "control_budget": 1},
    ):
        with pytest.raises(ValueError):
            experiment_overrides("none", 1, 20, **kwargs)


def test_two_gpu_stage2_composes_warm_start_without_eval_assistance(monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from toolkits.rlt.planner_stage2 import training_overrides, validate_contract

    root = Path(__file__).resolve().parents[2]
    for key, value in {
        "RLT_STAGE1_ACTOR": "/stage1/actor",
        "RLT_DATASET_DIR": "/dataset",
        "RLT_PLANNER_OUTPUT": "/run",
        "RLT_PLANNER_NAME": "planner-stage2",
        "RLT_PLANNER_METRICS": "/local/metrics",
        "RLT_PLANNER_RENDER_DEVICE": "pci:0000:c1:00.0",
        "RLT_PLANNER_INITIAL_WEIGHTS": "/weights.pt",
        "WANDB_ENTITY": "test-entity",
        "EMBODIED_PATH": str(root / "examples/embodiment"),
    }.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="maniskill_rlt_stage2_ac_mlp",
            overrides=training_overrides(400, 50, 50, 100),
        )
    OmegaConf.resolve(cfg)
    validate_contract(cfg)
    assert cfg.runner.ckpt_path == "/weights.pt"
    assert cfg.runner.max_steps == 400 and cfg.runner.save_interval == 100
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.runner.logger.backend_log_path == "/local/metrics"
    assert cfg.env.train.planner_assistance.protocol == "complete"
    assert cfg.env.eval.evaluation_reset_seeds == list(range(12026, 12076))
    assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates == 4096
    assert cfg.runner.resume_dir is None
    for key, value in {
        "runner.resume_dir": "/old/run",
        "env.eval.planner_assistance.enable": True,
        "env.train.total_num_envs": 2,
        "env.eval.evaluation_reset_seeds": [12026] * 50,
        "cluster.component_placement.actor": "2-2",
    }.items():
        altered = OmegaConf.create(OmegaConf.to_container(cfg))
        OmegaConf.update(altered, key, value)
        with pytest.raises(ValueError):
            validate_contract(altered)
    with pytest.raises(ValueError):
        training_overrides(0, 50, 50, 100)


def test_planner_warm_start_validates_strict_finite_weights_on_cpu(tmp_path):
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.mlp_policy import get_model
    from toolkits.rlt.planner_stage2 import validate_weights

    cfg = OmegaConf.create(
        {
            "model_type": "rlt_mlp_policy",
            "z_dim": 16,
            "proprio_dim": 9,
            "action_dim": 8,
            "num_action_chunks": 10,
        }
    )
    state = get_model(cfg).state_dict()
    path = tmp_path / "weights.pt"
    torch.save(state, path)
    assert len(validate_weights(path, cfg)) == 64
    state["backbone.0.weight"][0, 0] = float("nan")
    torch.save(state, path)
    with pytest.raises(ValueError, match="non-finite"):
        validate_weights(path, cfg)
    del state["backbone.0.weight"]
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        validate_weights(path, cfg)


def test_initial_weights_copy_does_not_require_cifs_metadata(tmp_path, monkeypatch):
    import hashlib
    import shutil

    from toolkits.rlt.planner_stage2 import copy_initial_weights

    def reject_attributes(*args, **kwargs):
        raise PermissionError("CIFS metadata changes denied")

    monkeypatch.setattr(shutil, "copystat", reject_attributes)
    source, target = tmp_path / "source.pt", tmp_path / "copy.pt"
    source.write_bytes(b"weight-content")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    copy_initial_weights(source, target, digest)
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="differ"):
        copy_initial_weights(source, target, "not-the-expected-digest")


def test_executed_work_budget_requires_exact_counts():
    from rlinf.algorithms.rlt.experiment_budget import ExperimentBudget

    work = ExperimentBudget(control_limit=17)
    assert not work.observe(
        {"num_trajectories": 1, "episode_len": 10},
        {"rlt/critic_updates_run": 2, "rlt/actor_updates_run": 1},
    )
    assert work.observe(
        {"num_trajectories": 1, "episode_len": 7},
        {"rlt/critic_updates_run": 1, "rlt/actor_updates_run": 0},
    )
    assert (work.control_ticks, work.critic_updates, work.actor_updates) == (17, 3, 1)
    updates = ExperimentBudget(update_limit=3)
    assert updates.observe(
        {"num_trajectories": 1, "episode_len": 100},
        {"rlt/critic_updates_run": 3, "rlt/actor_updates_run": 1},
    )
    with pytest.raises(RuntimeError, match="exceeded"):
        updates.observe(
            {"num_trajectories": 1, "episode_len": 1},
            {"rlt/critic_updates_run": 1, "rlt/actor_updates_run": 0},
        )
    for kwargs in ({}, {"update_limit": 1, "control_limit": 1}, {"control_limit": -1}):
        with pytest.raises(ValueError):
            ExperimentBudget(**kwargs)


def correction_cache(tmp_path):
    from types import SimpleNamespace

    from rlinf.algorithms.rlt.correction_data import (
        export_episode,
        finalize_corrections,
    )

    directory = tmp_path / "corrections"
    contract = {
        "model": {
            "model_type": "rlt_mlp_policy",
            "z_dim": 2048,
            "proprio_dim": 9,
            "action_dim": 8,
            "num_action_chunks": 10,
        },
        "algorithm": {"gamma": 0.99, "reference_dropout_prob": 0.5},
        "reference_source": "frozen_vla_pre_action",
        "action_space": "environment_pd_joint_delta_pos",
        "environment": {
            "init_params": {
                "id": "PegInsertionSideWideClearance-v1",
                "control_mode": "pd_joint_delta_pos",
            }
        },
    }
    for i in range(5):
        obs = {
            "z_rl": torch.zeros(1, 1, 2048),
            "proprio": torch.zeros(1, 1, 9),
            "ref_chunk": torch.full((1, 1, 80), -0.25),
        }
        rewards = torch.zeros(1, 1, 10)
        if i != 1:
            rewards[..., 4] = 1
        transition = SimpleNamespace(
            curr_obs=obs,
            next_obs=obs,
            actions=torch.full((1, 1, 80), 0.25),
            rewards=rewards,
            dones=torch.tensor([[[False] * 4 + [True] * 6]]),
            forward_inputs={"planner_flags": torch.ones(1, 1, 10, dtype=torch.bool)},
        )
        export_episode(directory, str(i), [transition], contract)
    finalize_corrections(directory)
    return directory


def test_corrections_retain_failed_td_but_reject_failed_bc_and_terminal_padding(
    tmp_path,
):
    from rlinf.algorithms.rlt.correction_data import correction_batch, load_corrections

    cache = correction_cache(tmp_path)
    episodes, _, digest = load_corrections(cache)
    batch = correction_batch(episodes)
    assert len(digest) == 64 and len(batch["actions"]) == 5
    assert batch["valid"].sum().item() == 25
    assert batch["accepted_planner"].sum().item() == 20
    assert not batch["bc_mask"][1].any()
    assert batch["rewards"][1].sum() == 0
    assert torch.all(batch["target"][0, :5] == 0.25)
    assert torch.all(batch["curr_obs"]["ref_chunk"] == -0.25)
    (cache / "episode_0.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_corrections(cache)


def test_controller_mapping_is_explicit_and_keeps_submitted_evidence():
    from rlinf.algorithms.rlt.correction_data import controller_actions

    contract = {
        "environment": {
            "init_params": {
                "id": "PegInsertionSideWideClearance-v1",
                "control_mode": "pd_joint_delta_pos",
            }
        }
    }
    submitted = torch.tensor([1.2, -1.015, 0.1])
    actual = controller_actions(submitted, contract)
    torch.testing.assert_close(actual, torch.tensor([1.0, -1.0, 0.1]))
    assert submitted[0] > 1 and submitted[1] < -1
    with pytest.raises(ValueError, match="Unverified"):
        controller_actions(submitted, {})
    with pytest.raises(ValueError, match="nonfinite"):
        controller_actions(torch.tensor([float("nan")]), contract)


def test_corrections_reject_unfinished_and_modified_inventory(tmp_path):
    from rlinf.algorithms.rlt.correction_data import load_corrections

    cache = correction_cache(tmp_path)
    (cache / "complete.json").rename(cache / "held_complete.json")
    with pytest.raises(ValueError, match="unfinished"):
        load_corrections(cache)
    (cache / "held_complete.json").rename(cache / "complete.json")
    with (cache / "index.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="inventory changed"):
        load_corrections(cache)


@pytest.mark.parametrize("q_weight", [0.0, 0.45])
def test_same_data_diagnostic_matches_work_and_null_q_ablation(tmp_path, q_weight):
    from toolkits.rlt.correction_fit import fit_pair

    torch.set_num_threads(2)
    report = fit_pair(
        correction_cache(tmp_path),
        tmp_path / "fit",
        updates=5,
        warmup_updates=0,
        q_weight=q_weight,
    )
    assert set(report["train_episodes"]).isdisjoint(report["validation_episodes"])
    assert report["actor_updates_per_arm"] == 2
    assert report["critic_updates_per_arm"] == 5
    assert report["closed_loop_success"] is None
    first = torch.load(tmp_path / "fit/bc_only/model.pt", weights_only=True)
    second = torch.load(tmp_path / "fit/q_bc/model.pt", weights_only=True)
    assert all(torch.equal(first[key], second[key]) for key in first) == (q_weight == 0)
    assert report["quality"]["train"]["planner_ticks"] == 15
    assert report["quality"]["train"]["accepted_planner_ticks"] == 10


def test_paired_corrections_support_identical_warm_start_and_reject_nan(tmp_path):
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from toolkits.rlt.correction_fit import fit_pair

    torch.set_num_threads(2)
    cache = correction_cache(tmp_path)
    initial = RLTMLPPolicy(2048, 9, 8, 10)
    checkpoint = tmp_path / "initial.pt"
    torch.save(initial.state_dict(), checkpoint)
    result = fit_pair(
        cache, tmp_path / "fit", initial_weights=checkpoint, updates=1, q_weight=0
    )
    assert result["initialization"] == "weights_only_fresh_optimizers"
    assert len(result["initial_weights_sha256"]) == 64
    a = torch.load(tmp_path / "fit/bc_only/model.pt", weights_only=True)
    b = torch.load(tmp_path / "fit/q_bc/model.pt", weights_only=True)
    assert all(torch.equal(a[key], b[key]) for key in a)
    a[next(iter(a))].fill_(float("nan"))
    torch.save(a, checkpoint)
    with pytest.raises(ValueError, match="finite"):
        fit_pair(cache, tmp_path / "invalid", initial_weights=checkpoint, updates=1)
    assert not (tmp_path / "invalid").exists()


def test_pilot_warm_start_is_not_resume_or_evaluation(tmp_path):
    from toolkits.rlt.planner_experiment import experiment_overrides

    initial = tmp_path / "initial.pt"
    kwargs = {"initial_weights": initial, "control_budget": 10000}
    none = experiment_overrides("none", 1000, 50, **kwargs)
    planner = experiment_overrides("planner", 1000, 50, **kwargs)
    assert f"runner.ckpt_path={initial}" in none
    assert "runner.resume_dir=null" in none
    assert "runner.only_eval=True" not in none
    assert len(set(none) ^ set(planner)) == 2
    with pytest.raises(ValueError, match="initialization"):
        experiment_overrides("none", 1, 20, checkpoint=initial, initial_weights=initial)


def test_runtime_snapshot_includes_evaluation_and_untracked_toolkit(tmp_path):
    import subprocess
    import tarfile

    from toolkits.rlt.planner_stage2 import snapshot_source

    root, runtime = tmp_path / "repo", tmp_path / "runtime"
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    names = [
        "examples/embodiment/train_embodied_agent.py",
        "evaluations/eval_embodied_agent.py",
        "toolkits/rlt/new_helper.py",
    ]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# runtime source\n")
    archive = tmp_path / "source.tar.gz"
    snapshot_source(root, runtime, archive)
    with tarfile.open(archive) as stored:
        assert set(stored.getnames()) == set(names)
    assert all(
        (runtime / name).read_bytes() == (root / name).read_bytes() for name in names
    )


def test_export_saved_replay_preserves_real_episode_boundaries(tmp_path):
    import json

    from omegaconf import OmegaConf

    from rlinf.algorithms.rlt.correction_data import load_corrections
    from toolkits.rlt.correction_export import export_checkpoint

    cache = correction_cache(tmp_path)
    episodes, contract, _ = load_corrections(cache)
    run = tmp_path / "run"
    replay = run / "checkpoints/global_step_5/actor/sac_components/replay_buffer/rank_0"
    replay.mkdir(parents=True)
    (run / "launch.json").write_text(json.dumps({"complete": True, "exit_code": 0}))
    cfg = {
        "env": {
            "train": {
                **contract["environment"],
                "total_num_envs": 1,
                "auto_reset": False,
            }
        },
        "runner": {"resume_dir": None},
        "actor": {"model": contract["model"]},
        "rollout": {"rlt_feature_model": {}},
        "algorithm": contract["algorithm"],
    }
    OmegaConf.save(OmegaConf.create(cfg), run / "resolved.yaml")
    index = {}
    for i, e in enumerate(episodes):
        row = {key: e[key].unsqueeze(0) for key in ("actions", "rewards", "dones")}
        row["actions"] = row["actions"].reshape(1, 1, 80)
        row.update(
            model_weights_id="fixed",
            curr_obs={k: v.unsqueeze(0) for k, v in e["curr_obs"].items()},
            next_obs={k: v.unsqueeze(0) for k, v in e["next_obs"].items()},
            forward_inputs={"planner_flags": e["planner"].unsqueeze(0)},
        )
        torch.save(row, replay / f"trajectory_{i}_fixed.pt")
        index[str(i)] = {
            "num_samples": 1,
            "max_episode_length": 1,
            "model_weights_id": "fixed",
        }
    (replay / "metadata.json").write_text(
        json.dumps(
            {
                "trajectory_format": "pt",
                "trajectory_counter": 5,
                "size": 5,
                "total_samples": 5,
            }
        )
    )
    (replay / "trajectory_index.json").write_text(
        json.dumps({"trajectory_index": index, "trajectory_id_list": list(range(5))})
    )
    result = export_checkpoint(run, replay, tmp_path / "export")
    actual, _, _ = load_corrections(tmp_path / "export")
    assert result["recorded_episodes"] == 5 and len(actual) == 5
    assert not actual[1]["success"]
    first = torch.load(replay / "trajectory_0_fixed.pt", weights_only=True)
    second = torch.load(replay / "trajectory_1_fixed.pt", weights_only=True)
    first["dones"].zero_()
    second["curr_obs"]["proprio"].fill_(123)
    torch.save(first, replay / "trajectory_0_fixed.pt")
    torch.save(second, replay / "trajectory_1_fixed.pt")
    with pytest.raises(ValueError, match="discontinuous"):
        export_checkpoint(run, replay, tmp_path / "gap")
    assert not (tmp_path / "gap/complete.json").exists()
    assert (
        json.loads((tmp_path / "gap/failure.json").read_text())["next_transition_id"]
        == 1
    )
    (replay / "metadata.json").write_text(
        json.dumps(
            {
                "trajectory_format": "pt",
                "trajectory_counter": 6,
                "size": 5,
                "total_samples": 5,
            }
        )
    )
    with pytest.raises(ValueError, match="evicted"):
        export_checkpoint(run, replay, tmp_path / "evicted")


def test_report_never_merges_protocols_or_evaluation_scopes():
    from copy import deepcopy

    from toolkits.rlt.planner_report import aggregate, compare

    plain = {
        "launch": {"arm": "none", "seed": 1, "eval_seeds": [12026]},
        "config": {},
        "scalars": {"eval/success_once": 0.4},
    }
    assisted = deepcopy(plain)
    assisted["launch"]["arm"] = "planner"
    for key, value in (("eval_scope", "insertion"), ("protocol", "preinsert_handoff")):
        changed = deepcopy(assisted)
        changed["launch"][key] = value
        with pytest.raises(ValueError, match="differs"):
            compare(plain, changed)
    pair = compare(plain, assisted)
    with pytest.raises(ValueError, match="three distinct"):
        aggregate([pair] * 3)
    pairs = [{**deepcopy(pair), "seed": i} for i in range(3)]
    assert aggregate(pairs)["paired_difference"]["mean"] == 0


def test_campaign_is_three_seed_budgeted_and_selects_last_not_best(tmp_path):
    from argparse import Namespace

    from toolkits.rlt.planner_campaign import campaign_commands, final_weights

    args = Namespace(
        seeds=[1234, 1235, 1236],
        budget=30000,
        max_episodes=1000,
        eval_episodes=50,
        stage1=Path("actor"),
        dataset=Path("data"),
        gpu=2,
        port=6535,
        protocol="complete",
        output=tmp_path,
        budget_basis="control",
    )
    commands = campaign_commands(args)
    assert len(commands) == 6
    assert all(
        "--control-budget" in command and "--export-corrections" in command
        for command in commands
    )
    (tmp_path / "resolved.yaml").write_text(
        "runner:\n  logger:\n    experiment_name: pilot\n"
    )
    for step in (9, 10):
        path = (
            tmp_path
            / f"pilot/checkpoints/global_step_{step}/actor/model_state_dict/full_weights.pt"
        )
        path.parent.mkdir(parents=True)
        path.touch()
    assert "global_step_10" in str(final_weights(tmp_path))


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


def physics_config():
    from omegaconf import OmegaConf

    from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
        register_rlinf_peg_insertion_side_variants,
    )

    register_rlinf_peg_insertion_side_variants()
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "examples/embodiment/config/env/maniskill_rlt.yaml")
    cfg.total_num_envs = 1
    cfg.seed = 2026
    cfg.auto_reset = False
    cfg.ignore_terminations = False
    cfg.wrap_obs_mode = "simple"
    cfg.video_cfg.video_base_dir = "/dev/shm"
    cfg.init_params.obs_mode = "state"
    cfg.init_params.sim_backend = "cpu"
    cfg.init_params.render_backend = "none"
    cfg.init_params.render_mode = None
    cfg.init_params.max_episode_steps = 500
    return cfg


@pytest.mark.skipif(
    os.environ.get("RLT_PLANNER_PHYSICS") != "1", reason="Opt-in CPU PhysX/MPLib"
)
@pytest.mark.parametrize("seed", [12026, 12027, 12028])
def test_insertion_fixture_is_held_unsuccessful_and_costed(seed):
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv
    from rlinf.envs.sim.maniskill.planner_assistance import (
        handoff_ready,
        read_peg_evidence,
    )

    cfg = physics_config()
    cfg.policy_mode = "eval"
    cfg.insertion_fixture = True
    cfg.rlt_policy_switch.task_mode = "critical_phase"
    cfg.rlt_policy_switch.expert_takeover.enable = False
    env = ManiskillRLTEnv(cfg, 1, 0, 1, None)
    try:
        obs, info = env.reset(seed=seed)
        initial = read_peg_evidence(env.env)
        assert handoff_ready(initial, RecoveryConfig())
        assert 0 < initial.tick < 500
        assert bool(info["rlt_switch_flags"][0])
        action = torch.zeros(1, 10, 8)
        action[..., -1] = -1
        _, _, _, _, infos = env.chunk_step(action)
        episode = infos[-1]["episode"]
        assert int(episode["fixture_prefix_ticks"][0]) == initial.tick
        assert int(episode["insertion_policy_ticks"][0]) == 10
        assert "planner_flags" not in infos[-1]
        # A held fixture must not finish just from the prefix's residual motion.
        # This is a constant-command control, not an actor performance test.
        for _ in range(50):
            _, _, terminated, truncated, infos = env.chunk_step(action)
            if (terminated | truncated).any():
                break
        assert not infos[-1]["episode"]["success_once"].any()
    finally:
        env.env.close()


@pytest.mark.skipif(
    os.environ.get("RLT_PLANNER_PHYSICS") != "1", reason="Opt-in CPU PhysX"
)
def test_control_budget_truncates_inside_chunk_without_hidden_steps():
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv

    cfg = physics_config()
    cfg.policy_mode = "train"
    cfg.training_control_budget = 17
    env = ManiskillRLTEnv(cfg, 1, 0, 1, None)
    try:
        env.reset(seed=2026)
        action = torch.zeros(1, 10, 8)
        action[..., -1] = 1
        env.chunk_step(action)
        _, _, _, truncated, infos = env.chunk_step(action)
        assert truncated.tolist() == [[False] * 6 + [True] + [False] * 3]
        assert int(env.elapsed_steps[0]) == 17
        assert int(infos[-1]["episode"]["episode_len"][0]) == 17
        env.chunk_step(action)
        assert int(env.elapsed_steps[0]) == 17
        with pytest.raises(RuntimeError, match="budget exhausted"):
            env.reset()
    finally:
        env.env.close()


@pytest.mark.skipif(
    os.environ.get("RLT_PLANNER_PHYSICS") != "1", reason="Opt-in CPU PhysX/MPLib"
)
def test_bounded_handoff_waits_for_fresh_policy_chunk():
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv

    cfg = physics_config()
    cfg.planner_assistance = {
        "enable": True,
        "approach_timeout": 20,
        "protocol": "preinsert_handoff",
    }
    env = ManiskillRLTEnv(cfg, 1, 0, 1, None)
    try:
        env.reset(seed=2026)
        action = torch.zeros(1, 10, 8)
        action[..., -1] = 1
        for _ in range(40):
            _, _, term, trunc, infos = env.chunk_step(action)
            last = infos[-1]
            assert not (term | trunc).any(), "Fixture unexpectedly ended before handoff"
            if last["episode"]["planner_handoffs"].item():
                assert last["planner_flags"].all(), (
                    "Old policy command escaped inside the handoff chunk"
                )
                actual = last["intervene_action"].reshape(1, 10, 8)
                assert torch.all(actual[0, -1, :7] == 0)
                assert 1 <= last["episode"]["planner_handoff_hold_ticks"].item() <= 10
                break
        else:
            pytest.fail("Planner never reached the handoff fixture")
        fresh = torch.zeros(1, 10, 8)
        fresh[..., -1] = -1
        _, _, _, _, infos = env.chunk_step(fresh)
        assert not infos[-1]["planner_flags"].any()
        assert torch.equal(infos[-1]["intervene_action"].reshape(1, 10, 8), fresh)
    finally:
        env.env.close()
