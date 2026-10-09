# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint, action and feature contracts for the Pi0.5 RLT adapter."""

import copy
import hashlib
import json
import subprocess
import sys
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
    evaluation_cases,
    runtime_config,
)
from toolkits.rlt.pi05_campaign import (
    acceptance_cases,
    acceptance_plan,
    audit_acceptance,
    campaign_plan,
    episode_outcomes,
    paired_comparison,
    paired_gate,
    summarize_campaign,
)
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
    assert config["framework"]["gripper_remap"] is True
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


def test_upstream_dataset_gripper_remap_reaches_native_commands():
    source = Path(__file__).resolve().parents[3] / "third_party/AlphaBrain"
    if not source.exists():
        pytest.skip("Pinned external AlphaBrain source is not installed")
    code = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(sys.argv[1]).parent / "alphabrain-deps"))
import numpy as np
import torch
from types import SimpleNamespace
from AlphaBrain.model.framework.PaliGemmaPi import PaliGemmaPi
from toolkits.rlt.pi05_baseline import env_commands, runtime_config
released = {"framework": {"name": "PaliGemmaPi0", "pi05": True,
            "action_model": {"action_dim": 7, "action_horizon": 10},
            "paligemma": {}}, "trainer": {"pretrained_checkpoint": "old"}}
config = runtime_config(released, Path("tokenizer"))
model = SimpleNamespace(gripper_remap=config["framework"]["gripper_remap"])
actions = torch.ones(1, 10, 7) * .2
actions[0, :, 6] = torch.tensor([0., 1., .1, .9, .49, .51, 0., 1., .2, .8])
native = PaliGemmaPi._maybe_remap_gripper(model, actions)[0].numpy()
commands = env_commands(native)
np.testing.assert_array_equal(commands[:, 6], [1, -1] * 5)
np.testing.assert_array_equal(commands[:, :6], np.full((10, 6), .2, dtype=np.float32))
"""
    subprocess.run([sys.executable, "-c", code, str(source)], check=True, timeout=60)


def test_feature_capture_preserves_reference_and_releases_hook():
    class ExternalModel:
        """Small stand-in for the external framework's public inference boundary."""

        def __init__(self):
            self.cache_tokens = 816
            self.vlm_interface = SimpleNamespace(
                model=SimpleNamespace(language_model=torch.nn.Identity())
            )

        def predict_action(self, **kwargs):
            result = SimpleNamespace(
                last_hidden_state=torch.ones(1, 816, 2048),
                past_key_values=SimpleNamespace(
                    get_seq_length=lambda: self.cache_tokens
                ),
            )
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
    reference.model.cache_tokens = 0
    with pytest.raises(RuntimeError, match="prefix KV cache"):
        reference.predict(obs, "test")
    reference.model.cache_tokens = 816
    assert "hidden" not in reference.predict(obs, "test")


def test_gemma_prefix_cache_survives_upstream_decoder_and_conditions_suffix():
    source = Path(__file__).resolve().parents[3] / "third_party/AlphaBrain"
    patch = (
        source
        / "AlphaBrain/model/modules/action_model/pi0_flow_matching_head/adarms_patch.py"
    )
    if not patch.exists():
        pytest.skip("Pinned external AlphaBrain source is not installed")
    # The upstream patch changes global Transformers classes; isolate it from
    # other model tests while using real attention, cache and decoder objects.
    code = r"""
import importlib.util
import sys
import torch
from transformers import GemmaConfig, GemmaModel
from transformers.modeling_utils import no_init_weights
from toolkits.rlt.pi05_baseline import adapt_gemma_cache_keyword
torch.set_num_threads(2)
torch.manual_seed(42)
spec = importlib.util.spec_from_file_location("adarms", sys.argv[1])
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)
patch.patch_gemma_for_adarms()
cfg = GemmaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=2,
                  num_key_value_heads=1, head_dim=16)
cfg._attn_implementation = "eager"
model = GemmaModel(cfg).eval()
with no_init_weights():
    restored = GemmaModel(cfg).eval()
restored.load_state_dict(model.state_dict(), strict=True)
adapt_gemma_cache_keyword(model)
adapt_gemma_cache_keyword(restored)
assert adapt_gemma_cache_keyword(model) == 0
prefix = torch.randn(1, 5, 32)
suffix = torch.randn(1, 2, 32)
with torch.no_grad():
    outputs = []
    for context in (prefix, -prefix):
        result = model(inputs_embeds=context, attention_mask=torch.zeros(1, 1, 5, 5), use_cache=True)
        restored_result = restored(inputs_embeds=context, attention_mask=torch.zeros(1, 1, 5, 5), use_cache=True)
        torch.testing.assert_close(result.last_hidden_state, restored_result.last_hidden_state)
        cache = result.past_key_values
        assert cache.get_seq_length() == 5
        cached = model(inputs_embeds=suffix, past_key_values=cache,
                       attention_mask=torch.zeros(1, 1, 2, 7), use_cache=False)
        assert cache.get_seq_length() == 7
        mask = torch.zeros(1, 1, 7, 7)
        mask[:, :, :5, 5:] = torch.finfo(torch.float32).min
        joint = model(inputs_embeds=torch.cat((context, suffix), dim=1),
                      attention_mask=mask, use_cache=False)
        torch.testing.assert_close(cached.last_hidden_state,
                                   joint.last_hidden_state[:, -2:], atol=1e-5, rtol=1e-5)
        outputs.append(cached.last_hidden_state)
    assert not torch.allclose(*outputs)
"""
    subprocess.run([sys.executable, "-c", code, str(patch)], check=True, timeout=60)


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


def test_acceptance_case_sets_are_disjoint_and_do_not_cycle_published_states(tmp_path):
    sets = acceptance_cases(tmp_path, 0)
    assert {k: len(v) for k, v in sets.items()} == {
        "collection": 64,
        "online": 2048,
        "development": 20,
        "heldout": 100,
    }
    flattened = [c for cases in sets.values() for c in cases]
    assert len({c["seed"] for c in flattened}) == len(flattened)
    assert len({c["snapshot"] for c in flattened}) == len(flattened)
    assert all(c["kind"] == "generated" for c in flattened)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(sets["heldout"]))
    assert evaluation_cases(SimpleNamespace(cases=path, task=0)) == sets["heldout"]
    with pytest.raises(ValueError, match="task"):
        evaluation_cases(SimpleNamespace(cases=path, task=1))


@pytest.mark.parametrize("gpu", [0, 1])
def test_acceptance_matches_warmstart_budget_and_heldout_cases(tmp_path, gpu):
    plan = acceptance_plan(
        tmp_path, tmp_path, tmp_path, tmp_path, gpu, 0, 42, "online", "team"
    )
    validate_plan(plan)
    jobs = {j["name"]: j for j in plan["jobs"]}

    def option(name, flag):
        args = jobs[name]["argv"]
        return args[args.index(flag) + 1]

    for name in ("q_bc", "bc_only"):
        assert option(name, "--interaction-ticks") == "20000"
        assert option(name, "--checkpoint").endswith("warmup/warmup.pt")
        assert option(name, "--gpu") == str(gpu)
        assert option(name, "--wandb") == "online"
    assert jobs["reference_final"]["depends_on"] == ["q_bc", "bc_only"]
    for name in ("reference_final", "warmup_final", "q_bc_eval", "bc_only_eval"):
        assert option(name, "--cases") == str(tmp_path / "cases/heldout.json")
        assert option(name, "--evaluation-role") == "heldout"
        assert "--collect" not in jobs[name]["argv"]


def test_paired_final_comparison_reports_uncertainty_and_rejects_changed_resets():
    cases = [
        {
            "id": str(i),
            "task": 0,
            "kind": "generated",
            "seed": i,
            "state_sha256": str(i),
            "success": i < 60,
        }
        for i in range(100)
    ]
    actor = copy.deepcopy(cases)
    actor[0]["success"] = False
    actor[70]["success"] = True
    reports = {"reference_final": {"episodes": cases}, "q_bc_eval": {"episodes": actor}}
    comparison = paired_comparison(reports)
    assert comparison["q_bc_eval"]["gained"] == comparison["q_bc_eval"]["lost"] == 1
    low, high = comparison["q_bc_eval"]["wilson95"]
    assert 0.49 < low < 0.6 < high < 0.70
    actor[0]["state_sha256"] = "changed"
    with pytest.raises(ValueError, match="do not match"):
        paired_comparison(reports)


@pytest.fixture
def completed_acceptance(tmp_path):
    """Write the public artifacts of a complete, fixed-budget campaign."""
    plan = acceptance_plan(
        tmp_path, tmp_path, tmp_path, tmp_path, 1, 1, 42, "online", "team"
    )
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue/plan.json").write_text(json.dumps(plan))
    jobs = {job["name"]: job for job in plan["jobs"]}
    (tmp_path / "queue/status.json").write_text(
        json.dumps(
            {"state": "finished", "jobs": {name: {"state": "passed"} for name in jobs}}
        )
    )
    sets = acceptance_cases(tmp_path, 1)
    (tmp_path / "cases").mkdir()
    for split, cases in sets.items():
        (tmp_path / "cases" / f"{split}.json").write_text(json.dumps(cases))
    stages = {
        "reference_gate": "development",
        "reference_collect": "collection",
        "warmup": None,
        "actor_gate": "development",
        "q_bc": "online",
        "bc_only": "online",
        "reference_final": "heldout",
        "warmup_final": "heldout",
        "q_bc_eval": "heldout",
        "bc_only_eval": "heldout",
    }
    for order, (name, split) in enumerate(stages.items()):
        directory = tmp_path / name
        directory.mkdir()
        args = jobs[name]["argv"]
        # Parse only the public options represented by the stage manifests.
        options = {
            flag[2:].replace("-", "_"): args[args.index(flag) + 1]
            for flag in (
                "--task",
                "--seed",
                "--wandb",
                "--evaluation-role",
                "--checkpoint",
                "--objective",
                "--interaction-ticks",
                "--encoder-steps",
                "--bc-steps",
            )
            if flag in args
        }
        for key in ("task", "seed", "interaction_ticks", "encoder_steps", "bc_steps"):
            if key in options:
                options[key] = int(options[key])
        options.update(
            mode=args[2], updates_per_chunk=5, collect=name == "reference_collect"
        )
        actor_updates = 0 if name.startswith("reference") else 6000
        critic_updates = 0
        if name in ("q_bc", "bc_only", "q_bc_eval", "bc_only_eval"):
            actor_updates, critic_updates = 11000, 10000
        report = {
            "complete": True,
            "exit_code": 0,
            "gate_passed": True,
            "args": options,
            "protocol": PROTOCOL,
            "adapter_sha256": {"adapter.py": "fixed-digest"},
            "versions": {"torch": "fixed-version"},
            "started_at": order * 100,
            "finished_at": order * 100 + 50,
            "actor_updates": actor_updates,
            "critic_updates": critic_updates,
        }
        if split is None:
            ids = [c["id"] for c in sets["collection"]]
            report.update(
                encoder_updates=1000, train_episodes=ids[:52], heldout_episodes=ids[52:]
            )
        else:
            cases = sets[split][:100] if split == "online" else sets[split]
            ticks = 200 if split == "online" else 90
            episodes = [
                {
                    **c,
                    "state_sha256": f"reset-{c['seed']}",
                    "success": i < (99 if split == "online" else 60),
                    "ticks": ticks
                    if split == "online" or i < 60
                    else PROTOCOL["max_control_steps"],
                }
                for i, c in enumerate(cases)
            ]
            successes = sum(e["success"] for e in episodes)
            report.update(
                episodes=episodes,
                successes=successes,
                success_rate=successes / len(episodes),
                interaction_ticks=sum(e["ticks"] for e in episodes),
                settling_ticks=10 * len(episodes),
                case_manifest_sha256=file_hash(tmp_path / "cases" / f"{split}.json"),
            )
        (directory / "manifest.json").write_text(json.dumps(report))
        (directory / "wandb_run.json").write_text(
            json.dumps({"mode": "online", "url": f"https://wandb.example/{name}"})
        )
    return tmp_path


def test_acceptance_audit_checks_complete_artifacts_and_shared_cost(
    completed_acceptance,
):
    result = audit_acceptance(completed_acceptance)
    assert result["state"] == "passed", result["violations"]
    assert not result["pending_stages"]
    cost = result["completed_training_cost_by_arm"]["q_bc_eval"]
    assert cost == {
        "stage_wall_seconds": 150,
        "interaction_ticks": 26680,
        "settling_ticks": 1640,
    }
    assert (
        result["completed_training_cost_by_arm"]["warmup_final"]["interaction_ticks"]
        == 6680
    )
    assert summarize_campaign(completed_acceptance)["engineering_pipeline_complete"]
    outcomes = result["stage_costs"]["q_bc"]["episode_outcomes"]
    assert outcomes["successes"] == 99
    assert outcomes["budget_truncated"] == 1
    assert outcomes["horizon_failures"] == outcomes["unresolved_non_successes"] == 0


@pytest.mark.parametrize(
    ("mode", "budget", "total", "last_ticks", "success", "expected"),
    [
        ("online", 520, 520, 200, False, "budget_truncated"),
        ("online", 520, 520, 200, True, "successes"),
        ("online", 640, 640, 320, False, "horizon_failures"),
        ("online", 600, 520, 200, False, "unresolved_non_successes"),
        ("online", 520, 519, 200, False, "unresolved_non_successes"),
        ("actor-eval", 520, 520, 200, False, "unresolved_non_successes"),
    ],
)
def test_episode_outcomes_separates_budget_truncation_from_failures(
    mode, budget, total, last_ticks, success, expected
):
    report = {
        "protocol": PROTOCOL,
        "complete": True,
        "exit_code": 0,
        "args": {"mode": mode, "interaction_ticks": budget},
        "interaction_ticks": total,
        "episodes": [
            {"id": "first", "success": False, "ticks": 320},
            {"id": "last", "success": success, "ticks": last_ticks},
        ],
        "success_rate": int(success) / 2,
    }
    before = copy.deepcopy(report)
    counts = episode_outcomes(report)
    assert counts[expected] == (2 if expected == "horizon_failures" else 1)
    assert counts["budget_truncated_case_ids"] == (
        ["last"] if expected == "budget_truncated" else []
    )
    assert report == before
    report["complete"] = False
    with pytest.raises(ValueError, match="completed"):
        episode_outcomes(report)


def test_acceptance_audit_rejects_nonfinal_early_unsuccessful_episode(
    completed_acceptance,
):
    path = completed_acceptance / "q_bc/manifest.json"
    report = json.loads(path.read_text())
    report["episodes"][0]["success"] = False
    report["successes"] -= 1
    report["success_rate"] = report["successes"] / len(report["episodes"])
    path.write_text(json.dumps(report))
    result = audit_acceptance(completed_acceptance)
    assert result["state"] == "failed"
    assert any("unexplained early unsuccessful" in e for e in result["violations"])


def test_acceptance_audit_preserves_missing_and_live_work(completed_acceptance):
    root = completed_acceptance
    for name in (
        "bc_only",
        "reference_final",
        "warmup_final",
        "q_bc_eval",
        "bc_only_eval",
    ):
        (root / name / "manifest.json").unlink()
    path = root / "q_bc/manifest.json"
    report = json.loads(path.read_text())
    report.update(complete=False)
    for key in (
        "exit_code",
        "finished_at",
        "episodes",
        "interaction_ticks",
        "actor_updates",
        "critic_updates",
    ):
        report.pop(key)
    path.write_text(json.dumps(report))
    (root / "q_bc/online_status.json").write_text(
        json.dumps(
            {"interaction_ticks": 5000, "actor_updates": 7250, "critic_updates": 2500}
        )
    )
    result = audit_acceptance(root)
    assert result["state"] == "incomplete" and not result["violations"]
    assert "q_bc" in result["pending_stages"]
    assert result["stage_costs"]["q_bc"]["interaction_ticks"] == 5000
    assert "q_bc_eval" not in result["completed_training_cost_by_arm"]
    assert not summarize_campaign(root)["engineering_pipeline_complete"]


@pytest.mark.parametrize(
    ("name", "field", "value", "message"),
    [
        ("q_bc", "interaction_ticks", 19999, "interaction accounting"),
        ("warmup", "encoder_updates", 999, "encoder budget"),
        ("q_bc", "actor_updates", 6000, "learner update"),
        ("q_bc_eval", "critic_updates", 0, "checkpoint update counters"),
        (
            "bc_only",
            "adapter_sha256",
            {"adapter.py": "changed"},
            "adapter_sha256 mismatch",
        ),
        ("reference_final", "started_at", 1, "started before"),
        ("warmup_final", "success_rate", 1.0, "success rate mismatch"),
        ("q_bc", "protocol", {"track": "rlt_a"}, "incompatible protocol"),
        ("q_bc", "finished_at", 0, "invalid stage timestamps"),
        ("q_bc", "finished_at", None, "invalid stage timestamps"),
    ],
)
def test_acceptance_audit_rejects_incomparable_results(
    completed_acceptance, name, field, value, message
):
    path = completed_acceptance / name / "manifest.json"
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    result = audit_acceptance(completed_acceptance)
    assert result["state"] == "failed"
    assert any(message in error for error in result["violations"])
    assert not summarize_campaign(completed_acceptance)["engineering_pipeline_complete"]


def test_acceptance_audit_rejects_reset_leakage_and_changed_cases(completed_acceptance):
    root = completed_acceptance
    path = root / "q_bc_eval/manifest.json"
    report = json.loads(path.read_text())
    report["episodes"][0]["state_sha256"] = (
        "reset-110000"  # A collection initial state.
    )
    report["args"]["checkpoint"] = str(root / "warmup/warmup.pt")
    path.write_text(json.dumps(report))
    declared = root / "cases/heldout.json"
    cases = json.loads(declared.read_text())
    cases[0]["seed"] += 999
    declared.write_text(json.dumps(cases))
    result = audit_acceptance(root)
    assert result["state"] == "failed"
    for text in (
        "different cases share",
        "reset changed",
        "checkpoint path mismatch",
        "declared cases differ",
        "case manifest digest mismatch",
    ):
        assert any(text in error for error in result["violations"])
    summary = summarize_campaign(root)
    assert not summary["engineering_pipeline_complete"]
    assert summary["comparison_error"]
    assert not summary["comparison"]


def test_acceptance_audit_rechecks_warmup_gate_and_logging(completed_acceptance):
    root = completed_acceptance
    (root / "q_bc/wandb_run.json").unlink()
    path = root / "actor_gate/manifest.json"
    report = json.loads(path.read_text())
    for episode in report["episodes"][:3]:
        episode["success"] = False
    report.update(successes=17, success_rate=0.85)
    path.write_text(json.dumps(report))
    result = audit_acceptance(root)
    assert result["state"] == "failed"
    assert any("paired_gate" in error for error in result["violations"])
    assert any("missing online W&B" in error for error in result["violations"])


def test_acceptance_audit_rejects_a_consistently_reported_short_budget(
    completed_acceptance,
):
    path = completed_acceptance / "q_bc/manifest.json"
    report = json.loads(path.read_text())
    report["episodes"][-1]["ticks"] -= 1
    report["interaction_ticks"] -= 1
    path.write_text(json.dumps(report))
    result = audit_acceptance(completed_acceptance)
    assert result["state"] == "failed"
    assert any("executed budget mismatch" in error for error in result["violations"])


def test_acceptance_audit_cli_uses_distinct_exit_codes_and_new_snapshots(
    completed_acceptance,
):
    root = completed_acceptance
    report = root / "audit.json"
    command = [
        sys.executable,
        "-m",
        "toolkits.rlt.pi05_campaign",
        "audit",
        "--root",
        str(root),
        "--report",
        str(report),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(report.read_text())["state"] == "passed"
    original = report.read_bytes()
    again = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert again.returncode == 2 and report.read_bytes() == original
    (root / "bc_only_eval/manifest.json").unlink()
    incomplete = subprocess.run(
        command[:-2], capture_output=True, text=True, timeout=30
    )
    assert incomplete.returncode == 3
    (root / "queue/plan.json").unlink()
    absent = subprocess.run(command[:-2], capture_output=True, text=True, timeout=30)
    assert absent.returncode == 4


def test_seeded_reference_does_not_consume_learning_rng():
    class ExternalModel:
        def __init__(self):
            self.vlm_interface = SimpleNamespace(
                model=SimpleNamespace(language_model=torch.nn.Identity())
            )

        def predict_action(self, **kwargs):
            self.vlm_interface.model.language_model(
                SimpleNamespace(
                    last_hidden_state=torch.ones(1, 816, 2048),
                    past_key_values=SimpleNamespace(get_seq_length=lambda: 816),
                )
            )
            return {"normalized_actions": torch.rand(1, 10, 7).numpy()}

    obs = {
        "primary_image": np.zeros((4, 4, 3)),
        "wrist_image": np.zeros((4, 4, 3)),
        "state": np.zeros(8),
    }
    reference = Pi05Reference(ExternalModel())
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    first = reference.predict(obs, "test", seed=32)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(20)  # Different learner workloads do not change reference noise.
    second = reference.predict(obs, "test", seed=32)
    np.testing.assert_array_equal(first["ref"], second["ref"])
    assert not np.array_equal(
        first["ref"], reference.predict(obs, "test", seed=33)["ref"]
    )


def test_required_wandb_failure_is_not_silently_disabled(tmp_path, monkeypatch):
    from toolkits.rlt.libero_acceptance import Metrics

    def unavailable(**kwargs):
        raise ConnectionError("service unavailable")

    monkeypatch.setattr("wandb.init", unavailable)
    args = SimpleNamespace(
        output=tmp_path, wandb="online", group="test", wandb_required=True
    )
    with pytest.raises(RuntimeError, match="Required W&B"):
        Metrics(args)
