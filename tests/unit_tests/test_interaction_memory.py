# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for bounded interaction evidence and RLT conditioning."""

import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.interaction_memory import (
    InteractionMemory,
    InteractionMemoryConfig,
    JointMemoryCollector,
    copy_memory_observation,
    validate_interaction_memory_cfg,
)
from rlinf.algorithms.rlt.rollout import predict_rlt_actions
from rlinf.algorithms.rlt.route import SimulatorRLTRoute
from rlinf.algorithms.rlt.transition import extract_rlt_obs_from_forward_inputs
from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.modules.rlt_memory_encoder import RLTMemoryEncoder


def test_response_context_reads_only_completed_owned_evidence():
    from rlinf.algorithms.rlt.response_context import ResponseContext

    memory = ResponseContext("error")
    command = torch.ones(7) * 0.1
    assert torch.equal(memory.predict(command), torch.zeros(7))
    memory.observe(command, command * 0.2)
    before = memory.predict(command)
    snapshot = memory.snapshot()
    snapshot["commands"].fill_(999)
    command.fill_(999)
    assert torch.equal(memory.predict(torch.ones(7) * 0.1), before)
    with pytest.raises(ValueError, match="finite"):
        memory.observe(torch.ones(7), torch.full((7,), float("nan")))
    assert torch.equal(memory.predict(torch.ones(7) * 0.1), before)


def test_response_weights_recover_old_condition_without_deleting_evidence():
    from rlinf.algorithms.rlt.response_context import ResponseContext

    memory = ResponseContext("error")
    command = torch.ones(7) * 0.1
    for _ in range(8):
        memory.observe(command, command)
    for _ in range(4):
        memory.observe(command, command * 0.2)
    low = memory.snapshot()
    assert low["weights"][:8].mean() < 0.1
    for _ in range(4):
        memory.observe(command, command)
    restored = memory.snapshot()
    assert len(restored["commands"]) == 16
    assert restored["weights"][:8].mean() > 0.9
    assert restored["weights"][8:12].mean() < 0.1
    memory.begin_attempt()
    assert len(memory.snapshot()["commands"]) == 16
    cleared = ResponseContext("clear")
    cleared.observe(command, command)
    cleared.begin_attempt()
    assert len(cleared.snapshot()["commands"]) == 0


def test_response_stream_future_changes_cannot_alter_past_predictions():
    from toolkits.rlt.probe_memory_conditions import analyze_stream

    commands = torch.ones(18, 7) * 0.08
    outcomes = commands.clone()
    original = analyze_stream(commands, outcomes, [0, 6, 12])
    outcomes[9:] *= 0.1
    changed = analyze_stream(commands, outcomes, [0, 6, 12])
    for mode in original:
        assert original[mode]["predictions"][:10] == changed[mode]["predictions"][:10]
    assert len(original["clear"]["weights_before_decision"][6]) == 0
    assert len(original["retain"]["weights_before_decision"][6]) == 6


def _paired_response_batch():
    from torch.utils.data import default_collate

    rows = []
    c = InteractionMemoryConfig()
    command = torch.zeros(10, 8)
    command[:, :7] = 0.1
    for condition in (0, 1):
        memory = InteractionMemory(c)
        memory.begin_attempt("pair")
        memory.append_completed(
            torch.zeros(9), command, torch.ones(9) * (0.1 + condition * 0.1)
        )
        rows.append(
            {
                **memory.snapshot(torch.zeros(9)),
                "velocity": torch.zeros(9),
                "command": torch.ones(7) * 0.1,
                "target": torch.ones(7) * condition,
                "pair": torch.tensor(0),
                "query_id": torch.tensor(0),
                "condition": torch.tensor(condition),
            }
        )
    return default_collate(rows)


def test_wrong_history_matches_current_input_and_does_not_mutate_evidence():
    from toolkits.rlt.probe_memory_conditions import wrong_history

    batch = _paired_response_batch()
    original = batch["memory_events"].clone()
    wrong = wrong_history(batch)
    assert torch.equal(wrong["memory_events"][0], original[1])
    assert torch.equal(batch["memory_events"], original)
    assert torch.equal(wrong["target"], batch["target"])
    batch["velocity"][0, 0] = 1
    with pytest.raises(ValueError, match="Current observation"):
        wrong_history(batch)


def test_response_probe_ignores_condition_labels_and_current_outcome():
    from toolkits.rlt.probe_memory_conditions import MatchedResponseProbe

    batch = _paired_response_batch()
    model = MatchedResponseProbe()
    first = model(batch, history=True)
    for field in ("condition", "pair", "query_id", "target"):
        batch[field] = torch.full_like(batch[field], 777)
    assert torch.equal(first, model(batch, history=True))
    assert torch.equal(model(batch, history=False)[0], model(batch, history=False)[1])


def test_matched_probe_keeps_pairs_disjoint_and_shares_training_budget():
    from toolkits.rlt.probe_memory_conditions import analyze_matched

    batch = _paired_response_batch()
    rows = []
    for pair in (0, 32, 40):
        for i in range(2):
            row = {key: value[i].clone() for key, value in batch.items()}
            row["pair"] = torch.tensor(pair)
            rows.append(row)
    result = analyze_matched(rows, updates=3)
    assert set(result["fixed"]["test"]["correct_response"]["pair_mse"]) == {"40"}
    for left, right in zip(result["learned"][::2], result["learned"][1::2]):
        assert left["initial_sha256"] == right["initial_sha256"]
        assert left["samples_sha256"] == right["samples_sha256"]
        assert left["updates"] == right["updates"] == 3


def test_standardized_response_ignores_labels_and_disabled_history():
    from toolkits.rlt.probe_memory_conditions import StandardizedResponseProbe

    batch = _paired_response_batch()
    model = StandardizedResponseProbe(batch)
    before = model(batch, history=True)
    for key in ("target", "condition", "pair", "query_id"):
        batch[key].fill_(999)
    assert torch.equal(model(batch, history=True), before)
    off = model(batch, history=False)
    assert torch.equal(off[0], off[1])
    batch["memory_events"].mul_(2)
    assert torch.equal(model(batch, history=False), off)


def test_response_tracking_uses_past_executed_commands_and_shared_cold_start():
    from toolkits.rlt.probe_response_control import METHODS, ResponseTracker

    error = torch.full((7,), 0.02)
    prior = torch.ones(7)
    trackers = {name: ResponseTracker(name, prior) for name in METHODS}
    prior.fill_(99)
    for tracker in trackers.values():
        assert torch.equal(tracker.command(error)[0], error)
        tracker.observe(torch.full((7,), 0.08), torch.full((7,), 0.04))
    assert torch.equal(trackers["fixed"].command(error)[0], error)
    adapted = trackers["retain"].command(error)[0]
    assert torch.all(adapted > error) and torch.all(adapted < 0.08)
    trackers["clear"].begin_attempt()
    trackers["retain"].begin_attempt()
    assert torch.equal(trackers["clear"].command(error)[0], error)
    assert torch.equal(trackers["retain"].command(error)[0], adapted)
    action, audit = trackers["retain"].command(torch.ones(7))
    assert torch.all(action <= 0.08) and audit["clipped_joints"] == 7
    with pytest.raises(ValueError, match="finite"):
        trackers["retain"].command(torch.full((7,), float("nan")))


def test_tracking_prior_excludes_validation_and_test_outcomes():
    from toolkits.rlt.probe_response_control import fit_prior, target_offsets

    rows = [
        {
            "pair": torch.tensor(pair),
            "command": torch.full((7,), 0.1),
            "target": torch.full((7,), value, dtype=torch.float32),
        }
        for pair, value in ((0, 0.08), (32, 99), (40, 999))
    ]
    a = fit_prior(rows)
    rows[-1]["target"].fill_(float("nan"))
    assert torch.equal(a, fit_prior(rows))
    assert torch.equal(target_offsets(58101), target_offsets(58101))
    assert not torch.equal(target_offsets(58101), target_offsets(58102))
    assert target_offsets(58101).abs().max() <= 0.06


def test_tracking_queue_retains_contact_failures_but_rejects_unmatched_results(
    tmp_path,
):
    import json

    from toolkits.rlt.run_research_queue import validate_job

    job = {"kind": "control", "output": str(tmp_path), "streams": 1}
    row = {
        "seed": 58101,
        "stage": "free_fixed",
        "dynamics": "stationary",
        "method": "fixed",
        "control_ticks": 780,
        "contact_valid": False,
    }
    report = {
        "completed": True,
        "scope": "test",
        "rows": [row],
        "control_ticks": 780,
        "all_initial_states_matched": True,
        "invalid_contact_streams": 1,
    }
    target = tmp_path / "results.json"
    target.write_text(json.dumps(report))
    assert validate_job(job)["scope"] == "test"
    report["all_initial_states_matched"] = False
    target.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="unmatched"):
        validate_job(job)


@pytest.mark.parametrize("nested", [False, True])
def test_research_queue_rejects_invalid_frozen_evidence(tmp_path, nested):
    import json

    from toolkits.rlt.run_research_queue import validate_evaluation

    run = tmp_path / "stage2_test"
    output = run / "stage2_portable" if nested else run
    output.mkdir(parents=True)
    (run / "exit_code.txt").write_text("0\n")
    audit = {
        "weights_unchanged": True,
        "actor_slots": 0,
        "initial_observation_sha256": "a" * 64,
    }
    (output / "route-audit.json").write_text(json.dumps(audit))
    episodes = [
        {"lane": i, "seed": 4101, "success_once": int(i < 4)} for i in range(16)
    ]
    path = output / "episode-records.json"
    path.write_text(json.dumps(episodes))
    assert validate_evaluation(tmp_path, seed=4101, reference=True)["successes"] == 4
    episodes[0]["lane"] = 1
    path.write_text(json.dumps(episodes))
    with pytest.raises(ValueError, match="distinct"):
        validate_evaluation(tmp_path, seed=4101, reference=True)
    episodes[0]["lane"] = 0
    path.write_text(json.dumps(episodes))
    audit["actor_slots"] = 1
    (output / "route-audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="Reference comparator"):
        validate_evaluation(tmp_path, seed=4101, reference=True)


def test_research_queue_executes_owned_child_with_timeout(tmp_path):
    import sys

    from toolkits.rlt.run_research_queue import execute_job

    (tmp_path / "logs").mkdir()
    job = {
        "id": "test",
        "cwd": str(tmp_path),
        "command": [sys.executable, "-c", "import time; time.sleep(30)"],
    }
    states = []
    code, elapsed = execute_job(
        job, tmp_path, timeout=0.15, heartbeat=lambda pid, dt: states.append(pid)
    )
    assert code == 124 and elapsed < 5 and states


def test_contact_factorial_separates_drive_and_stage_changes():
    from toolkits.rlt.probe_memory_conditions import phase_schedule

    schedules = phase_schedule()
    assert len(schedules) == 8
    for stage in {row["stage"] for row in schedules}:
        pair = [row for row in schedules if row["stage"] == stage]
        assert pair[0]["contact"] == pair[1]["contact"]
        assert pair[0]["stiffness"] == [1000.0, 1000.0, 1000.0]
        assert pair[1]["stiffness"] == [1000.0, 250.0, 1000.0]
    assert sum(len(set(row["contact"])) == 1 for row in schedules) == 4


def test_contact_audit_rejects_slip_and_unintended_contact():
    from toolkits.rlt.probe_memory_conditions import validate_contact_trace

    held = {"grasped": True, "finger_peg_force_newtons": [1.0, 1.0]}
    free = {"grasped": False, "finger_peg_force_newtons": [0.0, 0.0]}
    assert validate_contact_trace([held] * 10, grasped=True)["valid"]
    assert validate_contact_trace([free] * 10, grasped=False)["valid"]
    assert not validate_contact_trace([held] * 8 + [free] * 2, grasped=True)["valid"]
    assert not validate_contact_trace([held], grasped=False)["valid"]
    touching = {"grasped": False, "finger_peg_force_newtons": [0.2, 0.0]}
    assert not validate_contact_trace([touching], grasped=False)["valid"]
    with pytest.raises(ValueError, match="Invalid"):
        validate_contact_trace(
            [{"grasped": False, "finger_peg_force_newtons": [float("nan"), 0.0]}],
            grasped=False,
        )


def test_queue_requires_a_valid_contact_factorial(tmp_path):
    import json

    from toolkits.rlt.run_research_queue import validate_job

    job = {"kind": "phase", "output": str(tmp_path), "streams": 4}
    report = {
        "completed": True,
        "scope": "test",
        "rows": [{}] * 4,
        "all_contact_stages_valid": False,
    }
    (tmp_path / "results.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="contact"):
        validate_job(job)
    report["all_contact_stages_valid"] = True
    (tmp_path / "results.json").write_text(json.dumps(report))
    assert validate_job(job)["scope"] == "test"


def test_queue_waits_for_external_evidence_without_executing_or_overwriting_it(
    tmp_path,
):
    import json
    import time

    from toolkits.rlt.run_research_queue import job_states, run_queue

    source, followup = tmp_path / "source", tmp_path / "followup"
    (source / "jobs").mkdir(parents=True)
    followup.mkdir()
    evidence = source / "jobs" / "baseline.json"
    evidence.write_text(json.dumps({"state": "failed", "error": "audit rejected"}))
    original = evidence.read_bytes()
    jobs = [
        {"id": "baseline", "gpu": 0, "evidence_campaign": str(source)},
        {"id": "phase", "gpu": 0, "dependencies": ["baseline"]},
    ]
    (followup / "manifest.json").write_text(
        json.dumps(
            {"jobs": jobs, "wait_deadline": time.time() + 10, "run_seconds_per_gpu": 10}
        )
    )
    assert job_states(followup, jobs)["baseline"]["state"] == "failed"
    run_queue(followup, 0)
    assert evidence.read_bytes() == original
    assert not (followup / "jobs" / "baseline.json").exists()
    assert (
        json.loads((followup / "jobs" / "phase.json").read_text())["state"]
        == "dependency_failed"
    )


def test_diagnostic_rejects_nonfinite_unrecorded_and_intervened_rows(config):
    from toolkits.rlt.diagnose_actor_learning import transition_batch

    obs = {key: value[:1].unsqueeze(0) for key, value in _obs(config).items()}
    record = {
        "max_episode_length": 1,
        "curr_obs": obs,
        "next_obs": copy.deepcopy(obs),
        "actions": torch.zeros(1, 1, 4),
        "rewards": torch.zeros(1, 1, 2),
        "dones": torch.zeros(1, 1, 2, dtype=torch.bool),
        "terminations": torch.zeros(1, 1, 2, dtype=torch.bool),
        "truncations": torch.zeros(1, 1, 2, dtype=torch.bool),
        "intervene_flags": torch.zeros(1, 1, 4, dtype=torch.bool),
        "forward_inputs": {"record_transition": torch.ones(1, 1, 1, dtype=torch.bool)},
    }
    batch = transition_batch(record)
    assert batch["actions"].shape == (1, 4)
    assert torch.equal(batch["curr_obs"]["z_rl"], obs["z_rl"][0])
    for key in ("curr_obs", "next_obs"):
        bad = copy.deepcopy(record)
        bad[key]["z_rl"].fill_(float("nan"))
        with pytest.raises(ValueError, match="Nonfinite"):
            transition_batch(bad)
    bad = copy.deepcopy(record)
    bad["forward_inputs"]["record_transition"].zero_()
    with pytest.raises(ValueError, match="critical-phase"):
        transition_batch(bad)
    bad = copy.deepcopy(record)
    bad["intervene_flags"].fill_(True)
    with pytest.raises(ValueError, match="unassisted"):
        transition_batch(bad)
    bad = copy.deepcopy(record)
    bad["dones"].fill_(True)
    with pytest.raises(ValueError, match="termination"):
        transition_batch(bad)


def test_gpu_wait_requires_free_memory_and_no_compute_process():
    from toolkits.rlt.wait_for_gpu import gpu_available

    assert gpu_available("5\n", "")
    assert not gpu_available("5\n", "1234\n")
    assert not gpu_available("46945\n", "1234\n")
    assert not gpu_available("2048\n", "")
    for invalid in ("N/A", "5\n5", "-1", ""):
        with pytest.raises(ValueError):
            gpu_available(invalid, "")


def test_diagnostic_cache_splits_versions_and_reports_missing_files(config, tmp_path):
    import json

    from toolkits.rlt.diagnose_actor_learning import prepare_cache

    source = tmp_path / "source"
    source.mkdir()
    obs = {key: value[:1].unsqueeze(0) for key, value in _obs(config).items()}
    record = {
        "max_episode_length": 1,
        "curr_obs": obs,
        "next_obs": copy.deepcopy(obs),
        "actions": torch.zeros(1, 1, 4),
        "rewards": torch.zeros(1, 1, 2),
        "dones": torch.zeros(1, 1, 2, dtype=torch.bool),
        "terminations": torch.zeros(1, 1, 2, dtype=torch.bool),
        "truncations": torch.zeros(1, 1, 2, dtype=torch.bool),
        "intervene_flags": torch.zeros(1, 1, 4, dtype=torch.bool),
        "forward_inputs": {"record_transition": torch.ones(1, 1, 1, dtype=torch.bool)},
    }
    index = {}
    for i in range(40):
        group = f"collection-{i // 4}"
        index[str(i)] = {"model_weights_id": group}
        if i < 32:
            record["model_weights_id"] = group
            torch.save(record, source / f"trajectory_{i}_{group}.pt")
    path = source / "trajectory_index.json"
    path.write_text(json.dumps({"trajectory_index": index}))
    original = path.read_bytes()
    manifest = prepare_cache(source, tmp_path / "cache", limit=32, split_seed=601)
    assert manifest["missing_indexed_records"] == 8
    assert manifest["selected_records"] == 32
    train = {
        x["collection_version"] for x in manifest["sources"] if x["split"] == "train"
    }
    val = {
        x["collection_version"]
        for x in manifest["sources"]
        if x["split"] == "validation"
    }
    assert train and val and not train & val
    assert path.read_bytes() == original


def test_matched_diagnostic_shares_samples_and_separates_q_effect(config, tmp_path):
    from toolkits.rlt.diagnose_actor_learning import train_arm

    cfg = OmegaConf.create(
        {
            "actor": {
                "global_batch_size": 2,
                "micro_batch_size": 1,
                "model": {
                    "model_type": "rlt_mlp_policy",
                    "q_head_type": "default",
                    "z_dim": 4,
                    "proprio_dim": 3,
                    "action_dim": 2,
                    "num_action_chunks": 2,
                    "ref_num_action_chunks": 2,
                    "add_q_head": True,
                    "fixed_std": 0.002,
                    "interaction_memory": {
                        "enabled": True,
                        **asdict(config),
                        "reader_type": "zero",
                    },
                },
                "optim": {"lr": 1e-4, "clip_grad": 10.0},
                "critic_optim": {"lr": 1e-4, "clip_grad": 10.0},
            },
            "algorithm": {
                "q_head_type": "default",
                "target_update_type": "all",
                "target_update_freq": 1,
                "tau": 0.005,
                "gamma": 0.99,
                "critic_actor_ratio": 1,
                "reference_dropout_prob": 0.5,
                "actor_weight_schedule": {
                    "enable": True,
                    "warmup_updates": 2,
                    "ramp_updates": 0,
                    "warmup_bc_weight": 7.0,
                    "online_bc_weight": 7.0,
                    "warmup_q_weight": 0.0,
                    "online_q_weight": 0.05,
                },
            },
            "env": {"train": {"env_type": "maniskill_rlt"}},
        }
    )
    obs = _obs(config)
    batch = {
        "curr_obs": obs,
        "next_obs": copy.deepcopy(obs),
        "actions": torch.zeros(2, 4),
        "rewards": torch.ones(2, 2),
        "dones": torch.zeros(2, 2, dtype=torch.bool),
        "terminations": torch.zeros(2, 2, dtype=torch.bool),
        "intervene_flags": torch.zeros(2, 4, dtype=torch.bool),
    }
    cache = {"train": batch, "validation": batch}
    for steps in (2, 4):
        results = [
            train_arm(
                cfg,
                cache,
                tmp_path / f"{arm}_{steps}",
                variant=arm,
                steps=steps,
                seed=11,
            )
            for arm in ("bc_only", "q_bc")
        ]
        left, right = results
        assert left["initial_weights_sha256"] == right["initial_weights_sha256"]
        assert left["sample_sequence_sha256"] == right["sample_sequence_sha256"]
        assert left["actor_updates"] == right["actor_updates"] == steps
        assert (left["final_weights_sha256"] == right["final_weights_sha256"]) == (
            steps == 2
        )
    assert cfg.algorithm.actor_weight_schedule.online_q_weight == 0.05


@pytest.fixture
def config():
    return InteractionMemoryConfig(
        proprio_dim=3,
        action_dim=2,
        chunk_len=2,
        brief_size=2,
        archive_size=5,
        retrieval_size=2,
        hidden_dim=8,
        num_heads=2,
    )


def _append(memory, value, *, ticks=2):
    c = memory.config
    start = torch.full((c.proprio_dim,), float(value))
    memory.append_completed(start, torch.full((ticks, c.action_dim), 0.2), start + 0.1)


def _obs(config, *, empty=False):
    memory = InteractionMemory(config)
    memory.begin_attempt("test-instance")
    if not empty:
        _append(memory, 0)
    obs = {
        k: v.unsqueeze(0).repeat(2, *([1] * v.ndim))
        for k, v in memory.snapshot(torch.zeros(config.proprio_dim)).items()
    }
    obs.update(
        z_rl=torch.ones(2, 4),
        proprio=torch.zeros(2, config.proprio_dim),
        ref_chunk=torch.zeros(2, config.chunk_len, config.action_dim),
    )
    return obs


def _policy(config, *, enabled=True):
    return RLTMLPPolicy(
        4,
        config.proprio_dim,
        config.action_dim,
        config.chunk_len,
        interaction_memory={"enabled": enabled, **asdict(config)},
    )


def test_bounded_retrieval_is_past_only_unique_and_snapshot_owned(config):
    memory = InteractionMemory(config)
    with pytest.raises(RuntimeError):
        memory.snapshot(torch.zeros(3))
    memory.begin_attempt("case-a")
    empty = memory.snapshot(torch.zeros(3))
    for i in range(7):
        _append(memory, i)
    assert not empty["memory_valid"].any()
    snapshot = memory.snapshot(torch.full((3,), 2.0))
    assert snapshot["memory_events"][:, 0].tolist() == [5, 6, 2, 3]
    assert snapshot["memory_valid"].all()
    _append(memory, 99)
    assert snapshot["memory_events"][:, 0].tolist() == [5, 6, 2, 3]
    snapshot["memory_events"].zero_()
    assert memory.snapshot(torch.zeros(3))["memory_events"][-1, 0] != 0


def test_retry_requires_instance_identity_and_does_not_average_conflicts(config):
    memory = InteractionMemory(config)
    memory.begin_attempt("case-a")
    _append(memory, 0)
    memory.append_completed(torch.zeros(3), torch.zeros(2, 2), -torch.ones(3))
    with pytest.raises(ValueError, match="different physical"):
        memory.begin_attempt("case-b", retry=True)
    memory.begin_attempt("case-a", retry=True)
    result = memory.snapshot(torch.zeros(3))
    assert result["memory_valid"].tolist() == [False, False, True, True]
    assert result["memory_events"][2, 7] != result["memory_events"][3, 7]
    memory.begin_attempt("case-b")
    assert not memory.snapshot(torch.zeros(3))["memory_valid"].any()


def test_memory_checkpoint_roundtrip_and_schema_rejection(config, tmp_path):
    memory = InteractionMemory(config)
    memory.begin_attempt("case")
    _append(memory, 1)
    path = tmp_path / "memory.pt"
    torch.save(memory.state_dict(), path)
    restored = InteractionMemory(config)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for key, value in memory.snapshot(torch.zeros(3)).items():
        assert torch.equal(value, restored.snapshot(torch.zeros(3))[key])
    state = memory.state_dict()
    state["config"]["chunk_len"] = 8
    with pytest.raises(ValueError, match="schema"):
        restored.load_state_dict(state)


def test_collector_masks_terminal_tail_and_isolates_partial_resets(config):
    collector = JointMemoryCollector(config, 2)
    start = torch.zeros(2, 3)
    collector.reset([0, 1], ["a", "b"], start)
    before = collector.snapshot(start)
    actions = torch.full((2, 2, 2), 0.2)
    actions[0, 1] = float("nan")  # Never executed; must not enter evidence.
    valid = torch.tensor([[True, False], [True, True]])
    done = torch.tensor([[True, False], [False, False]])
    collector.complete(actions, torch.ones(2, 3), valid, done, torch.zeros_like(done))
    after = collector.snapshot(start)
    assert not before["memory_valid"].any()
    assert torch.isfinite(after["memory_events"]).all()
    row = after["memory_events"][0, config.brief_size - 1]
    assert row[-4:].tolist() == [1, 0, 1, 0]
    collector.reset([0], ["c"], start)
    reset = collector.snapshot(start)
    assert not reset["memory_valid"][0].any()
    assert reset["memory_valid"][1].sum() == 1


def test_collector_rejects_invalid_prefix_without_partial_write(config):
    collector = JointMemoryCollector(config, 2)
    states = torch.zeros(2, 3)
    collector.reset([0, 1], ["a", "b"], states)
    valid = torch.tensor([[True, True], [False, True]])
    with pytest.raises(ValueError, match="contiguous"):
        collector.complete(torch.zeros(2, 2, 2), states, valid, valid, valid)
    assert not collector.snapshot(states)["memory_valid"].any()


def test_empty_reader_is_finite_zero_and_padding_cannot_influence_output(config):
    encoder = RLTMemoryEncoder(config)
    empty = _obs(config, empty=True)
    empty["memory_events"].fill_(float("nan"))
    assert torch.equal(encoder(empty), torch.zeros(2, config.hidden_dim))
    obs = _obs(config)
    expected = encoder(obs)
    obs["memory_events"][~obs["memory_valid"]] = float("nan")
    torch.testing.assert_close(encoder(obs), expected)


def test_memory_gradients_owned_by_critic_and_target_is_detached(config):
    model = _policy(config)
    obs = _obs(config)
    actions, _, _ = model.sac_forward(obs, deterministic=True)
    actions.square().mean().backward()
    assert all(p.grad is None for p in model.memory_encoder.parameters())
    model.zero_grad(set_to_none=True)
    target = copy.deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        next_actions, _, _ = model.sac_forward(obs, deterministic=True)
        y = (
            1
            + 0.99**config.chunk_len
            * target.sac_q_forward(obs, next_actions).min(-1, keepdim=True).values
        )
    q = model.sac_q_forward(obs, actions.detach())
    (q - y).square().mean().backward()
    grads = [p.grad for p in model.memory_encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    assert all(p.grad is None for p in target.parameters())
    # Matches the existing FSDP critic optimizer's public name filter.
    assert all(
        "encoder" in name
        for name, _ in model.named_parameters()
        if name.startswith("memory_")
    )


def test_disabled_feature_preserves_parameter_schema_rng_and_predictions(config):
    torch.manual_seed(12)
    baseline = RLTMLPPolicy(4, config.proprio_dim, config.action_dim, config.chunk_len)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(12)
    disabled = _policy(config, enabled=False)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    assert baseline.state_dict().keys() == disabled.state_dict().keys()
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[key])
    obs = _obs(config)
    torch.testing.assert_close(
        baseline.sac_forward(obs, deterministic=True)[0],
        disabled.sac_forward(obs, deterministic=True)[0],
    )


def test_critic_optimizer_owns_and_updates_memory_reader(config):
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

    policy = _policy(config)
    optim = OmegaConf.create({"lr": 0.001})
    actor, critic = FSDPModelManager.build_optimizers(
        None,
        policy,
        optim,
        {"critic": ["encoders", "encoder", "q_head", "state_proj"]},
        {"critic": optim},
    )
    reader_ids = {id(p) for p in policy.memory_encoder.parameters()}
    actor_ids = {id(p) for g in actor.param_groups for p in g["params"]}
    critic_ids = {id(p) for g in critic.param_groups for p in g["params"]}
    assert not reader_ids & actor_ids
    assert reader_ids <= critic_ids
    before = {k: v.clone() for k, v in policy.memory_encoder.state_dict().items()}
    obs = _obs(config)
    q = policy.sac_q_forward(obs, torch.zeros(2, config.chunk_len * config.action_dim))
    (q - 1).square().mean().backward()
    critic.step()
    assert any(
        not torch.equal(before[k], v)
        for k, v in policy.memory_encoder.state_dict().items()
    )


def test_memory_changes_policy_context_and_roundtrips_weights(config, tmp_path):
    torch.manual_seed(7)
    model = _policy(config)
    obs = _obs(config)
    action = model.sac_forward(obs, deterministic=True)[0]
    empty_action = model.sac_forward(_obs(config, empty=True), deterministic=True)[0]
    assert not torch.equal(action, empty_action)
    path = tmp_path / "actor.pt"
    torch.save(model.state_dict(), path)
    restored = _policy(config)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.sac_forward(obs, deterministic=True)[0], action)


def test_transport_and_replay_keep_terminal_snapshot_not_reset_memory(config):
    class FeatureModel:
        def extract_rlt_obs(self, obs):
            return {key: obs[key] for key in ("z_rl", "proprio", "ref_chunk")}

    terminal, reset = _obs(config), _obs(config, empty=True)
    transport = EnvOutput(obs=reset, final_obs=terminal).to_dict()
    assert transport["final_obs"]["memory_valid"].any()
    assert not transport["obs"]["memory_valid"].any()
    _, output = predict_rlt_actions(
        policy_model=_policy(config),
        feature_model=FeatureModel(),
        rlt_route=SimulatorRLTRoute(use_schedule=False, warmup_updates=0),
        env_obs=reset,
        final_obs=terminal,
        mode="eval",
        rlt_switch_flags=torch.ones(2, dtype=torch.bool),
    )
    current = extract_rlt_obs_from_forward_inputs(output["forward_inputs"])
    successor = extract_rlt_obs_from_forward_inputs(
        output["forward_inputs"], transition=True
    )
    assert not current["memory_valid"].any()
    assert successor["memory_valid"].any()
    terminal["memory_valid"].zero_()
    assert successor["memory_valid"].any()
    bad = {"memory_valid": torch.zeros(2, 4, dtype=torch.bool)}
    with pytest.raises(ValueError, match="Incomplete"):
        copy_memory_observation(bad, {})


@pytest.mark.parametrize(
    "name", ["maniskill_rlt_stage2_ac_mlp", "maniskill_rlt_stage2_smoke_gpu2"]
)
@pytest.mark.parametrize(
    "overlay", ["rlt_memory", "rlt_memory_response", "rlt_memory_zero"]
)
def test_hydra_overlays_compose_without_starting_ray(name, overlay, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    for key in ("RLT_SMOKE_RUN_DIR", "RLT_STAGE1_ACTOR", "RLT_DATASET_DIR"):
        monkeypatch.setenv(key, "/validation-only/not-used")
    monkeypatch.setenv("RLT_SMOKE_RENDER_DEVICE", "cuda:0")
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base=None
    ):
        cfg = compose(config_name=name, overrides=[f"+experiment={overlay}"])
        validate_interaction_memory_cfg(cfg)
        cfg.actor.fsdp_config.use_orig_params = False
        if overlay == "rlt_memory":
            with pytest.raises(ValueError, match="use_orig_params"):
                validate_interaction_memory_cfg(cfg)
        else:
            validate_interaction_memory_cfg(cfg)
        cfg.actor.fsdp_config.use_orig_params = True
        cfg.algorithm.target_update_type = "q_head_only"
        with pytest.raises(ValueError, match="target_update_type"):
            validate_interaction_memory_cfg(cfg)


@pytest.mark.parametrize(
    "field,value", [("chunk_len", 0), ("retrieval_size", -1), ("hidden_dim", 7)]
)
def test_invalid_dimensions_fail_fast(config, field, value):
    values = asdict(config)
    values[field] = value
    with pytest.raises(ValueError):
        InteractionMemoryConfig(**values)


class _JointSimulator:
    """Deterministic simulator boundary: lane 0 terminates after one real tick."""

    num_envs = 2
    device = torch.device("cpu")
    obs_mode = "rgb"
    single_action_space = SimpleNamespace(shape=(8,), low=-np.ones(8), high=np.ones(8))

    def __init__(self):
        self.unwrapped = self
        self.elapsed_steps = torch.zeros(2, dtype=torch.long)
        self.qpos = torch.zeros(2, 9)

    def _obs(self):
        image = torch.zeros(2, 2, 2, 3, dtype=torch.uint8)
        return {
            "agent": {"qpos": self.qpos.clone()},
            "sensor_param": {},
            "sensor_data": {
                "3rd_view_camera": {"rgb": image},
                "wide_hand_camera": {"rgb": image},
            },
        }

    def reset(self, *, seed=None, options=None):
        del seed
        indices = (options or {}).get("env_idx", torch.arange(2))
        self.qpos[indices] = 0
        self.elapsed_steps[indices] = 0
        return self._obs(), {}

    def step(self, actions):
        self.qpos[:, :8] += torch.as_tensor(actions).clamp(-1, 1)
        self.elapsed_steps += 1
        done = torch.tensor([True, False])
        return self._obs(), torch.zeros(2), done, torch.zeros_like(done), {}

    def get_state_dict(self):
        return {"agent": {"qpos": self.qpos.clone()}}


@pytest.mark.parametrize("auto_reset", [False, True])
@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("command,use_numpy", [(0.2, False), (1.5, True)])
def test_real_wrapper_records_terminal_prefix_before_reset(
    monkeypatch, auto_reset, retain, command, use_numpy
):
    from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv

    backend = _JointSimulator()
    monkeypatch.setattr("gymnasium.make", lambda **kwargs: backend)
    config = InteractionMemoryConfig(
        chunk_len=2, brief_size=2, retrieval_size=1, retain_on_identical_reset=retain
    )
    cfg = OmegaConf.create(
        {
            "seed": 1,
            "auto_reset": auto_reset,
            "use_rel_reward": False,
            "ignore_terminations": False,
            "group_size": 1,
            "use_fixed_reset_state_ids": retain,
            "video_cfg": {},
            "init_params": {"id": "MemoryTest", "control_mode": "pd_joint_delta_pos"},
            "wrap_obs_mode": "rlt_openpi_joint",
            "reward_mode": "raw",
            "interaction_memory": {"enabled": True, **asdict(config)},
        }
    )
    env = ManiskillRLTEnv(cfg, 2, 0, 1, None, record_metrics=False)
    initial, _ = env.reset()
    assert not initial["memory_valid"].any()
    with pytest.raises(RuntimeError, match="chunk_step"):
        env.step(torch.zeros(2, 8))
    actions = torch.full((2, 2, 8), command)
    obs, _, done, _, infos = env.chunk_step(actions.numpy() if use_numpy else actions)
    effective = min(1.0, command)
    terminal = infos[-1]["final_observation"] if auto_reset else obs[-1]
    assert done.tolist() == [[True, False], [False, False]]
    row = terminal["memory_events"][0, 1]
    assert row[-4:].tolist() == [1, 0, 1, 0]
    torch.testing.assert_close(row[9:17], torch.full((8,), effective))
    assert not row[17:25].any()
    torch.testing.assert_close(row[25:33], torch.full((8,), effective))
    if auto_reset:
        assert obs[-1]["memory_valid"][0].sum() == int(retain)
        assert obs[-1]["memory_valid"][1].sum() == 1
    else:
        second, *_ = env.chunk_step(torch.full((2, 2, 8), 0.3))
        assert second[-1]["memory_valid"][0].sum() == 1
        torch.testing.assert_close(
            second[-1]["memory_events"][0], terminal["memory_events"][0]
        )


def test_hidden_dynamics_probe_preserves_pair_splits_and_masks():
    from toolkits.rlt.probe_memory_dynamics import build_splits, mask_memory, pair_split

    assert [pair_split(i) for i in (0, 31, 32, 39, 40)] == [
        "train",
        "train",
        "validation",
        "validation",
        "test",
    ]
    trajectories = []
    for pair in (0, 32, 40):
        for stiffness in (250, 1000):
            trajectories.append(
                {
                    "pair": pair,
                    "split": pair_split(pair),
                    "stiffness": stiffness,
                    "states": torch.zeros(61, 9),
                    "actions": torch.zeros(61, 8),
                }
            )
    splits = build_splits(trajectories)
    for batch in splits.values():
        assert "stiffness" not in batch
        assert not mask_memory(batch, "none")["memory_valid"].any()
        assert not mask_memory(batch, "recent")["memory_valid"][:, 4:].any()
        assert not mask_memory(batch, "archive")["memory_valid"][:, :4].any()
        assert batch["memory_valid"].any()
    assert not set(splits["train"]["episode"].tolist()) & set(
        splits["test"]["episode"].tolist()
    )
    trajectories[0]["split"] = "test"
    with pytest.raises(ValueError, match="crosses"):
        build_splits(trajectories)


def test_empirical_response_uses_completed_commands_not_future_labels():
    from toolkits.rlt.probe_memory_dynamics import response_summary

    c = InteractionMemoryConfig()
    memory = InteractionMemory(c)
    memory.begin_attempt("test")
    commands = torch.zeros(10, 8)
    commands[:, :7] = 0.1
    start = torch.zeros(9)
    end = start.clone()
    end[:7] = 0.05
    memory.append_completed(start, commands, end)
    obs = {k: v.unsqueeze(0) for k, v in memory.snapshot(end).items()}
    summary = response_summary(obs)
    torch.testing.assert_close(summary[:, :7], torch.full((1, 7), 0.005 / 0.0101))
    obs["target"] = torch.full((1, 9), float("nan"))
    torch.testing.assert_close(summary, response_summary(obs))
    obs["memory_events"][~obs["memory_valid"]] = float("nan")
    torch.testing.assert_close(summary, response_summary(obs))
    obs["memory_valid"][:] = False
    assert torch.count_nonzero(response_summary(obs)) == 0


def test_offline_probe_uses_past_only_evidence():
    from toolkits.rlt.probe_interaction_memory import episode_examples

    states = torch.arange(31, dtype=torch.float32)[:, None].expand(-1, 9).clone()
    actions = torch.zeros(31, 8)
    rows = episode_examples(states, actions)
    assert len(rows) == 3
    assert not rows[0]["memory_valid"].any()
    assert rows[1]["memory_valid"].sum() == 1
    changed = states.clone()
    changed[20:] += 999
    other = episode_examples(changed, actions)
    torch.testing.assert_close(rows[1]["memory_events"], other[1]["memory_events"])
    assert not torch.equal(rows[1]["target"], other[1]["target"])


def test_response_reader_matches_empirical_descriptor_and_has_no_parameters():
    from dataclasses import replace

    from toolkits.rlt.probe_memory_dynamics import response_summary

    c = replace(InteractionMemoryConfig(), reader_type="response")
    obs = _obs(c)
    reader = RLTMemoryEncoder(c)
    context = reader(obs)
    torch.testing.assert_close(context[:, :14], response_summary(obs))
    assert torch.count_nonzero(context[:, 14:]) == 0
    assert list(reader.parameters()) == []
    assert torch.count_nonzero(reader(_obs(c, empty=True))) == 0
    policy = _policy(c)
    action = policy.sac_forward(obs, deterministic=True)[0]
    q = policy.sac_q_forward(obs, action.detach())
    (action.mean() + q.mean()).backward()
    assert torch.isfinite(action).all() and torch.isfinite(q).all()
    assert any(p.grad is not None for p in policy.q_head.parameters())


def test_response_reader_partial_commands_and_old_checkpoint(config):
    from dataclasses import replace

    c = replace(config, reader_type="response")
    memory = InteractionMemory(c)
    memory.begin_attempt("partial")
    start = torch.zeros(c.proprio_dim)
    end = start.clone()
    end[0] = 0.05
    memory.append_completed(start, torch.tensor([[0.5, 1.0]]), end)
    obs = {k: v.unsqueeze(0) for k, v in memory.snapshot(end).items()}
    expected = RLTMemoryEncoder(c)(obs)
    assert expected[0, 0] == pytest.approx(0.0025 / 0.0026)
    # A padded action is not an executed command, even inside a valid record.
    slot = torch.nonzero(obs["memory_valid"][0])[0, 0]
    obs["memory_events"][0, slot, c.proprio_dim + c.action_dim] = torch.nan
    torch.testing.assert_close(RLTMemoryEncoder(c)(obs), expected)
    legacy = InteractionMemory(config)
    legacy.begin_attempt("legacy")
    state = legacy.state_dict()
    for key in ("reader_type", "joint_delta_scale"):
        state["config"].pop(key)
    restored = InteractionMemory(config)
    restored.load_state_dict(state)
    assert restored.instance_id == "legacy"


def test_zero_context_matches_response_capacity_and_initialization():
    """Attribute this comparison to history, not a wider or different initial head."""
    from dataclasses import replace

    response_config = replace(InteractionMemoryConfig(), reader_type="response")
    zero_config = replace(response_config, reader_type="zero")
    torch.manual_seed(1234)
    response = _policy(response_config)
    torch.manual_seed(1234)
    zero = _policy(zero_config)
    assert response.state_dict().keys() == zero.state_dict().keys()
    for key, tensor in response.state_dict().items():
        torch.testing.assert_close(tensor, zero.state_dict()[key], rtol=0, atol=0)
    empty = _obs(response_config, empty=True)
    torch.testing.assert_close(
        response.sac_forward(empty, deterministic=True)[0],
        zero.sac_forward(empty, deterministic=True)[0],
        rtol=0,
        atol=0,
    )
    history = _obs(response_config)
    assert torch.count_nonzero(response.memory_encoder(history)) > 0
    torch.testing.assert_close(
        zero.sac_forward(history, deterministic=True)[0],
        zero.sac_forward(empty, deterministic=True)[0],
        rtol=0,
        atol=0,
    )
    history["memory_events"].fill_(float("nan"))
    assert torch.count_nonzero(zero.memory_encoder(history)) == 0
    assert not list(zero.memory_encoder.parameters())


@pytest.mark.parametrize("variant", ["native", "zero_context", "reference"])
def test_frozen_evaluation_routes_at_version_zero_without_mutating_training(
    monkeypatch, tmp_path, variant
):
    from rlinf.algorithms.rlt.route import RLTRouteContext, build_rlt_route
    from toolkits.rlt.evaluate_memory_checkpoint import evaluation_config

    root = Path(__file__).resolve().parents[2]
    for key, value in {
        "EMBODIED_PATH": str(root / "examples/embodiment"),
        "RLT_STAGE1_ACTOR": str(tmp_path / "stage1"),
        "RLT_DATASET_DIR": str(tmp_path / "dataset"),
        "RLT_SMOKE_RUN_DIR": str(tmp_path / "output"),
        "RLT_SMOKE_RENDER_DEVICE": "pci:0000:46:00.0",
    }.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="maniskill_rlt_stage2_smoke_gpu2",
            overrides=["+experiment=rlt_memory_response", "+pilot=rlt_memory_matched"],
        )
    original = OmegaConf.to_container(cfg, resolve=True)
    checkpoint = tmp_path / "full_weights.pt"
    checkpoint.touch()
    evaluated = evaluation_config(
        cfg, checkpoint=checkpoint, variant=variant, num_envs=16, seed=4001
    )
    validate_interaction_memory_cfg(evaluated)
    route = build_rlt_route(evaluated)
    student = torch.ones(2, 10, 8)
    result = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs={"ref_chunk": torch.zeros_like(student)},
            student_actions=student,
            result={"forward_inputs": {}},
            mode="eval",
            version=0,
            rlt_switch_flags=torch.tensor([True, False]),
        )
    )
    expected = torch.zeros_like(student)
    if variant != "reference":
        expected[0] = 1
    torch.testing.assert_close(result.actions, expected)
    assert evaluated.runner.only_eval and evaluated.runner.resume_dir is None
    assert evaluated.rollout.expert_model is None
    assert evaluated.env.eval.total_num_envs == 16
    assert OmegaConf.to_container(cfg, resolve=True) == original


def test_frozen_memory_counterfactual_preserves_weights_input_and_rng(config):
    from dataclasses import replace

    from toolkits.rlt.evaluate_memory_checkpoint import tensor_digest

    model = _policy(replace(config, reader_type="response"))
    model.eval().requires_grad_(False)
    obs = _obs(config)
    weights, inputs = tensor_digest(model.state_dict()), tensor_digest(obs)
    rng = torch.get_rng_state().clone()
    native, _ = model.predict_action_batch(obs, mode="eval")
    empty = {**obs, "memory_valid": torch.zeros_like(obs["memory_valid"])}
    cleared, _ = model.predict_action_batch(empty, mode="eval")
    assert not torch.equal(native, cleared)
    assert tensor_digest(model.state_dict()) == weights
    assert tensor_digest(obs) == inputs
    assert torch.equal(torch.get_rng_state(), rng)


def test_frozen_memory_summary_rejects_unpaired_or_incomplete_evidence():
    import copy

    from toolkits.rlt.summarize_memory_evaluation import summarize

    status = {
        "checked_at": "test",
        "campaign": "test",
        "code_revision": "test",
        "jobs": [],
    }
    for reader, variants in (
        ("zero", ("native", "reference")),
        ("response", ("native", "zero_context")),
    ):
        job = {"reader": reader, "exit_code": 0, "runs": []}
        status["jobs"].append(job)
        for seed in range(4001, 4005):
            for variant in variants:
                job["runs"].append(
                    {
                        "label": f"seed{seed}_{variant}",
                        "exit_code": 0,
                        "audit": {
                            "weights_unchanged": True,
                            "initial_observation_sha256": str(seed),
                            "actor_slots": 0,
                            "slots": 16,
                            "action_difference_sum": 0,
                            "action_difference_count": 16,
                        },
                        "episodes": [
                            {
                                "seed": seed,
                                "lane": lane,
                                "success_once": int(lane == 0),
                                "entered_actor_phase_once": 1,
                                "episode_len": 100,
                            }
                            for lane in range(16)
                        ],
                    }
                )
    result = summarize(status)
    assert result["arms"]["response_native"]["successes"] == 4
    assert result["paired_outcomes"]["response_native_vs_zero_native"] == {
        "both_success": 4,
        "left_only_success": 0,
        "right_only_success": 0,
        "both_fail": 60,
    }
    for fault in ("hash", "duplicate_lane", "weights", "queue"):
        corrupt = copy.deepcopy(status)
        run = corrupt["jobs"][1]["runs"][0]
        if fault == "hash":
            run["audit"]["initial_observation_sha256"] = "different"
        elif fault == "duplicate_lane":
            run["episodes"][1]["lane"] = 0
        elif fault == "weights":
            run["audit"]["weights_unchanged"] = False
        else:
            corrupt["jobs"][1]["exit_code"] = None
        with pytest.raises(ValueError):
            summarize(corrupt)
