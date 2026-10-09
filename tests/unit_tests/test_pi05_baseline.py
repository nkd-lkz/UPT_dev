# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint, action and feature contracts for the Pi0.5 RLT adapter."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from toolkits.rlt.experiment_queue import validate_plan
from toolkits.rlt.libero_acceptance_learning import AcceptanceLearner, td_eligible
from toolkits.rlt.pi05_assets import file_hash
from toolkits.rlt.pi05_baseline import (
    PROTOCOL,
    Pi05Reference,
    env_commands,
    runtime_config,
)
from toolkits.rlt.pi05_campaign import campaign_plan, paired_gate, summarize_campaign
from toolkits.rlt.pi05_learning import validate_checkpoint


def test_legacy_config_is_explicit_and_does_not_mutate_release(tmp_path):
    released = {
        "framework": {
            "name": "PaliGemmaPi0",
            "pi05": True,
            "action_model": {"action_dim": 7, "action_horizon": 10},
            "paligemma": {},
        },
        "trainer": {"pretrained_checkpoint": "old"},
    }
    before = copy.deepcopy(released)
    config = runtime_config(released, tmp_path)
    assert released == before
    assert config["framework"]["name"] == "PaliGemmaPi05"
    assert config["framework"]["gripper_remap"] is False
    assert config["framework"]["paligemma"]["max_token_len"] == 48
    released["framework"]["action_model"]["action_horizon"] = 8
    with pytest.raises(ValueError, match="action contract"):
        runtime_config(released, tmp_path)


def test_native_actions_are_not_unnormalized_or_inverted_twice():
    native = np.ones((10, 7), dtype=np.float32) * 0.2
    native[:, 6] = [-1, 1] * 5
    before = native.copy()
    commands = env_commands(native)
    np.testing.assert_array_equal(commands, native)
    np.testing.assert_array_equal(native, before)
    native[0, 0] = 2
    assert env_commands(native)[0, 0] == 1
    native[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        env_commands(native)
    with pytest.raises(ValueError):
        env_commands(np.zeros((8, 7)))


def test_feature_capture_preserves_reference_and_releases_hook():
    class ExternalModel:
        """Small stand-in for the external framework's public inference boundary."""

        def __init__(self):
            self.vlm_interface = SimpleNamespace(
                model=SimpleNamespace(language_model=torch.nn.Identity())
            )

        def predict_action(self, **kwargs):
            result = SimpleNamespace(last_hidden_state=torch.ones(1, 816, 2048))
            self.vlm_interface.model.language_model(result)
            return {"normalized_actions": np.zeros((1, 10, 7))}

    reference = Pi05Reference(ExternalModel())
    obs = {
        "primary_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(8),
    }
    plain = reference.predict(obs, "test")
    captured = reference.predict(obs, "test", capture=True)
    np.testing.assert_array_equal(plain["ref"], captured["ref"])
    assert captured["hidden"].shape == (512, 2048)
    assert not captured["hidden"].requires_grad
    assert "hidden" not in reference.predict(obs, "test")


def test_protocol_and_file_digests_reject_cross_route_artifacts(tmp_path):
    validate_checkpoint({"protocol": PROTOCOL})
    with pytest.raises(ValueError, match="protocol"):
        validate_checkpoint({"protocol": PROTOCOL | {"track": "rlt_a"}})
    path = tmp_path / "model"
    path.write_bytes(b"example")
    assert file_hash(path) == hashlib.sha256(b"example").hexdigest()
    assert (
        file_hash(path, "sha1", git_blob=True)
        == hashlib.sha1(b"blob 7\0example").hexdigest()
    )


def test_shared_heads_accept_new_rlt_width_and_ten_tick_actions(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "third_party/AlphaBrain"
    if not source.exists():
        pytest.skip("Pinned external AlphaBrain source is not installed")
    monkeypatch.syspath_prepend(str(source))
    torch.set_num_threads(2)
    learner = AcceptanceLearner(
        "cpu", bottleneck_dim=2048, chunk_len=10, gripper_threshold=0, ref_dropout=0
    )
    row = {
        "z": torch.zeros(2048),
        "ref": torch.ones(10, 7) * 0.2,
        "prop": torch.zeros(8),
    }
    assert learner.command(row, deterministic=True).shape == (10, 7)
    transition = {
        **row,
        "action": row["ref"],
        **{f"next_{k}": v for k, v in row.items()},
        "ticks": 10,
        "reward": 0,
        "terminal": False,
    }
    metrics = learner.update([transition])
    assert metrics["critic_updates"] == 1
    assert not td_eligible(transition)  # Legacy default remains eight ticks.
    assert td_eligible(transition, 10)
    assert not td_eligible(transition | {"ticks": 8}, 10)
    json.dumps(metrics, allow_nan=False)


def test_campaign_budgets_dependencies_and_paired_gate(tmp_path):
    plan = campaign_plan(tmp_path, tmp_path, tmp_path, tmp_path, 2)
    validate_plan(plan)
    jobs = {row["name"]: row for row in plan["jobs"]}
    assert jobs["q_bc"]["depends_on"] == ["paired_gate"]
    assert jobs["bc_only"]["depends_on"] == ["paired_gate"]
    assert jobs["assets"]["requires_gpu"] is False
    assert jobs["warmup_final"]["depends_on"] == ["paired_gate"]
    final_argv = jobs["warmup_final"]["argv"]
    assert final_argv[final_argv.index("--checkpoint") + 1].endswith("warmup/warmup.pt")
    cases = [
        {
            "id": str(i),
            "task": 0,
            "kind": "published",
            "state": i,
            "seed": i,
            "state_sha256": str(i),
            "success": True,
        }
        for i in range(5)
    ]
    reference = {"complete": True, "exit_code": 0, "episodes": cases}
    assert paired_gate(reference, reference)["gate_passed"]
    actor = copy.deepcopy(reference)
    actor["episodes"][0]["success"] = False
    actor["episodes"][1]["success"] = False
    assert not paired_gate(reference, actor)["gate_passed"]
    actor["episodes"][0]["state_sha256"] = "wrong"
    with pytest.raises(ValueError, match="reset cases"):
        paired_gate(reference, actor)


def test_new_rlt_encoder_preserves_width_and_reconstructs_with_gradients(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "third_party/AlphaBrain"
    if not source.exists():
        pytest.skip("Pinned external AlphaBrain source is not installed")
    monkeypatch.syspath_prepend(str(source))
    from AlphaBrain.training.reinforcement_learning.algos.RLT.encoder_decoder import (
        RLTokenEncoderDecoder,
    )

    torch.set_num_threads(2)
    model = RLTokenEncoderDecoder(
        hidden_dim=16, num_heads=2, encoder_layers=1, decoder_layers=1, max_len=32
    )
    hidden = torch.randn(2, 10, 16, requires_grad=True)
    token, loss = model(hidden)
    assert token.shape == (2, 1, 16) and torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is None  # Stage 1 never updates the frozen VLA.
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )


def test_campaign_summary_keeps_missing_scores_distinct_from_zero(tmp_path):
    report = summarize_campaign(tmp_path)
    assert report["queue_state"] == "not_started"
    assert not report["engineering_pipeline_complete"]
    assert "success_rate" not in report["jobs"]["q_bc_eval"]
    output = tmp_path / "q_bc_eval"
    output.mkdir()
    manifest = {"complete": False, "success_rate": 0.0, "error": "interrupted"}
    (output / "manifest.json").write_text(json.dumps(manifest))
    report = summarize_campaign(tmp_path)
    assert report["jobs"]["q_bc_eval"]["error"] == "interrupted"
    assert "success_rate" not in report["jobs"]["q_bc_eval"]
    manifest.update(complete=True, exit_code=0, episodes=[], successes=0)
    (output / "manifest.json").write_text(json.dumps(manifest))
    report = summarize_campaign(tmp_path)
    assert report["jobs"]["q_bc_eval"]["success_rate"] == 0
    assert not report["engineering_pipeline_complete"]
