# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance protocol, numerical learner and paired experiment contracts."""

import copy
import json
import os
import random
from pathlib import Path

import pytest
import torch

from toolkits.rlt.libero_acceptance import (
    PROTOCOL,
    RemoteEnvironment,
    compatible,
    env_commands,
    read_cases,
    split_cache,
)
from toolkits.rlt.libero_acceptance_campaign import make_cases, matched, warmup_gate
from toolkits.rlt.libero_acceptance_env import snapshot_digest
from toolkits.rlt.libero_acceptance_learning import (
    AcceptanceLearner,
    td_eligible,
    td_target,
)
from toolkits.rlt.libero_reproduction import ASSETS


@pytest.fixture
def source(monkeypatch):
    """Exercise the real pinned actor/critic, without loading a VLA or CUDA."""
    path = Path(__file__).resolve().parents[3] / "third_party/AlphaBrain"
    if not path.exists():
        pytest.skip("Pinned AlphaBrain dependency is not installed")
    monkeypatch.syspath_prepend(str(path))
    torch.set_num_threads(2)


def rows():
    """Make typed transition evidence with easily separable reference labels."""
    return [
        {
            "z": torch.ones(256) * 0.1,
            "prop": torch.zeros(8),
            "ref": torch.ones(8, 7) * 0.7,
            "action": torch.ones(8, 7) * 0.7,
            "next_z": torch.ones(256) * 0.2,
            "next_prop": torch.zeros(8),
            "next_ref": torch.ones(8, 7) * 0.5,
            "ticks": 8,
            "terminal": False,
            "reward": 0.0,
            "case_id": f"published_{i:02d}",
        }
        for i in range(10)
    ]


def test_td_discount_and_terminal_mask_use_executed_ticks():
    actual = td_target(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([8, 1, 3]),
        torch.tensor([False, False, True]),
        torch.tensor([4.0, 5.0, 99.0]),
        0.9,
    )
    torch.testing.assert_close(actual, torch.tensor([1 + 0.9**8 * 4, 6.5, 3.0]))
    assert td_eligible({"ticks": 8, "terminal": False})
    assert td_eligible({"ticks": 3, "terminal": True})
    assert not td_eligible({"ticks": 3, "terminal": False})
    with pytest.raises(ValueError, match="flat batch shape"):
        td_target(
            torch.ones(2),
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            torch.ones(2, 1),
        )


def test_split_excludes_whole_episodes_and_protocol_rejects_old_runtime():
    data = rows() + rows()
    train, heldout = split_cache(data)
    assert {r["case_id"] for r in train}.isdisjoint({r["case_id"] for r in heldout})
    assert len(train) == 16 and len(heldout) == 4
    with pytest.raises(ValueError, match="five"):
        split_cache(rows()[:1])
    compatible({"protocol": PROTOCOL, "assets": ASSETS})
    with pytest.raises(ValueError, match="protocol"):
        compatible({"protocol": PROTOCOL | {"mujoco": "3.8.1"}, "assets": ASSETS})


def test_cases_separate_training_from_gate_and_generated_validation(tmp_path):
    paths = make_cases(tmp_path)
    groups = {name: read_cases(path) for name, path in paths.items()}
    assert len(groups["validation"]) == 50
    assert all(
        c["kind"] == "generated" and c["seed"] >= 20000 for c in groups["validation"]
    )
    train = {c["id"] for c in groups["training"]}
    assert not train & {c["id"] for c in groups["development"]}
    assert not train & {c["id"] for c in groups["validation"]}
    path = tmp_path / "bad.json"
    case = groups["collection"][0]
    for invalid in ([case, case], [case | {"state": 50}], []):
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            read_cases(path)


def outcomes(successes):
    return [
        {
            "case_id": str(i),
            "success": i < successes,
            "budget_cut": False,
            "initial_observation": {"image_hash": str(i)},
            "state_sha256": str(i),
        }
        for i in range(20)
    ]


def test_matched_comparison_requires_same_physical_and_rendered_inputs():
    a, b = outcomes(16), outcomes(14)
    result = matched(a, b)
    assert result["losses"] == 2 and result["gains"] == 0
    assert result["paired_difference"] == -0.1
    assert result["exact_mcnemar_two_sided"] == 0.5
    for key in ("state_sha256", "initial_observation"):
        corrupted = copy.deepcopy(b)
        corrupted[0][key] = "changed"
        with pytest.raises(ValueError, match="mismatch"):
            matched(a, corrupted)
    with pytest.raises(ValueError):
        matched(a, b + b[:1])


def test_warmup_gate_rejects_good_loss_with_control_collapse():
    fit = {
        "initial_following": {"mse": 0.2},
        "final_following": {"mse": 0.001, "gripper_disagreement": 0.01},
    }
    assert warmup_gate(fit, matched(outcomes(16), outcomes(16)))["passed"]
    assert not warmup_gate(fit, matched(outcomes(16), outcomes(10)))["passed"]
    fit["final_following"]["mse"] = float("nan")
    assert not warmup_gate(fit, matched(outcomes(16), outcomes(16)))["passed"]


def test_warmup_updates_actor_without_critic_and_resume_restores_counters(source):
    torch.manual_seed(5)
    learner = AcceptanceLearner("cpu")
    replay = rows()
    before = learner.following(replay)
    critic = copy.deepcopy(learner.critic.state_dict())
    for _ in range(30):
        learner.update(replay, warmup=True)
    after = learner.following(replay)
    assert after["mse"] < before["mse"]
    assert learner.actor_updates == 30 and learner.critic_updates == 0
    for key, value in critic.items():
        torch.testing.assert_close(
            value, learner.critic.state_dict()[key], rtol=0, atol=0
        )
    learner.publish_warmup()
    resumed = AcceptanceLearner("cpu", objective="bc_only")
    resumed.load_state_dict(learner.state_dict())
    torch.testing.assert_close(
        resumed.command(replay[0], deterministic=True),
        learner.command(replay[0], deterministic=True),
        rtol=0,
        atol=0,
    )
    assert resumed.actor_updates == 30
    resumed.update(replay)
    resumed.update(replay)
    assert resumed.critic_updates == 2 and resumed.actor_updates == 31


def test_bc_actor_ignores_critic_values_while_q_bc_uses_them(source):
    torch.manual_seed(8)
    initial = AcceptanceLearner("cpu").state_dict()
    actors = []
    for objective, change_critic in (
        ("bc_only", False),
        ("bc_only", True),
        ("q_bc", True),
    ):
        learner = AcceptanceLearner("cpu", objective=objective)
        learner.load_state_dict(copy.deepcopy(initial))
        if change_critic:
            with torch.no_grad():
                for parameter in learner.critic.parameters():
                    parameter.add_(0.03)
        random.seed(19)
        torch.manual_seed(19)
        learner.update(rows())
        learner.update(rows())
        actors.append(learner.actor.state_dict())
    for key in actors[0]:
        torch.testing.assert_close(actors[0][key], actors[1][key], rtol=0, atol=0)
    assert any(not torch.equal(actors[0][key], actors[2][key]) for key in actors[0])


def test_candidate_conditioner_is_used_and_checkpoint_contract_is_explicit(source):
    class Conditioner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, token, action, reference):
            self.calls += 1
            return token + (action - reference).mean((-2, -1))[:, None]

    conditioner = Conditioner()
    learner = AcceptanceLearner(
        "cpu",
        critic_conditioner=conditioner,
        critic_conditioning="candidate_test",
    )
    learner.update(rows())
    learner.update(rows())
    assert conditioner.calls == 5
    state = learner.state_dict()
    assert state["critic_conditioning"] == "candidate_test"
    with pytest.raises(ValueError, match="conditioning differs"):
        AcceptanceLearner("cpu").load_state_dict(state)
    with pytest.raises(ValueError, match="explicit module"):
        AcceptanceLearner("cpu", critic_conditioning="missing")


def test_critic_can_receive_a_separate_augmented_token(source):
    class Conditioner(torch.nn.Module):
        def forward(self, token, action, reference):
            return torch.cat((token, torch.zeros_like(token)), dim=-1)

    learner = AcceptanceLearner(
        "cpu",
        critic_conditioner=Conditioner(),
        critic_conditioning="candidate_zero_concat",
        critic_bottleneck_dim=512,
    )
    learner.update(rows())
    state = learner.state_dict()
    assert state["critic_bottleneck_dim"] == 512
    AcceptanceLearner(
        "cpu",
        critic_conditioner=Conditioner(),
        critic_conditioning="candidate_zero_concat",
        critic_bottleneck_dim=512,
    ).load_state_dict(state)
    with pytest.raises(ValueError, match="token width differs"):
        AcceptanceLearner(
            "cpu",
            critic_conditioner=Conditioner(),
            critic_conditioning="candidate_zero_concat",
            critic_bottleneck_dim=256,
        ).load_state_dict(state)
    with pytest.raises(ValueError, match="distinct critic token width"):
        AcceptanceLearner("cpu", critic_bottleneck_dim=512)


def test_candidate_conditioning_cannot_change_bc_warmup(source):
    class Conditioner(torch.nn.Module):
        def forward(self, token, action, reference):
            return token + (action - reference).mean((-2, -1))[:, None]

    actors = []
    for conditioned in (False, True):
        torch.manual_seed(29)
        learner = AcceptanceLearner(
            "cpu",
            critic_conditioner=Conditioner() if conditioned else None,
            critic_conditioning="candidate_test" if conditioned else "latent",
        )
        random.seed(37)
        torch.manual_seed(37)
        learner.update(rows(), warmup=True)
        actors.append(learner.actor.state_dict())
    for name in actors[0]:
        torch.testing.assert_close(actors[0][name], actors[1][name], rtol=0, atol=0)


def test_reference_correction_is_bounded_and_checkpointed(source):
    learner = AcceptanceLearner("cpu", correction_limit=0.1)
    reference = torch.linspace(-0.8, 0.8, 14).reshape(1, 2, 7)
    raw = reference + torch.tensor([-10.0, 10.0])[None, :, None]
    action = learner._policy_action(raw, reference)
    assert (action - reference).abs().max() <= 0.1 + 1e-6
    torch.testing.assert_close(learner._policy_action(reference, reference), reference)
    state = learner.state_dict()
    AcceptanceLearner("cpu", correction_limit=0.1).load_state_dict(state)
    with pytest.raises(ValueError, match="correction limit differs"):
        AcceptanceLearner("cpu").load_state_dict(state)
    for invalid in (0, -0.1, 2.1):
        with pytest.raises(ValueError, match="Correction limit"):
            AcceptanceLearner("cpu", correction_limit=invalid)

    unbounded = AcceptanceLearner("cpu")
    torch.testing.assert_close(unbounded._policy_action(raw, reference), raw)


def test_reference_correction_can_match_supervised_action_support(source):
    learner = AcceptanceLearner("cpu", correction_limit=0.1, correction_axes=(0, 2))
    reference = torch.linspace(-0.8, 0.8, 14).reshape(1, 2, 7)
    raw = (reference + 0.5).requires_grad_()
    action = learner._policy_action(raw, reference)
    assert (action[..., (0, 2)] - reference[..., (0, 2)]).abs().max() <= 0.1 + 1e-6
    torch.testing.assert_close(
        action[..., (1, 3, 4, 5, 6)], reference[..., (1, 3, 4, 5, 6)]
    )
    action.sum().backward()
    assert (raw.grad[..., (0, 2)] > 0).all()
    assert (raw.grad[..., (1, 3, 4, 5, 6)] == 0).all()

    state = learner.state_dict()
    AcceptanceLearner(
        "cpu", correction_limit=0.1, correction_axes=(0, 2)
    ).load_state_dict(state)
    with pytest.raises(ValueError, match="correction axes differ"):
        AcceptanceLearner("cpu", correction_limit=0.1).load_state_dict(state)
    for axes in ((), (0, 0), (-1,), (7,)):
        with pytest.raises(ValueError, match="Correction axes"):
            AcceptanceLearner("cpu", correction_limit=0.1, correction_axes=axes)
    with pytest.raises(ValueError, match="require a correction limit"):
        AcceptanceLearner("cpu", correction_axes=(0,))


def test_zero_initialized_residual_actor_starts_at_reference(source):
    learner = AcceptanceLearner(
        "cpu",
        correction_limit=0.1,
        correction_axes=(0,),
        residual_output=True,
    )
    token = torch.randn(256)
    reference = torch.linspace(-0.8, 0.8, 56).reshape(8, 7)
    row = {"z": token, "ref": reference, "prop": torch.randn(8)}
    torch.testing.assert_close(learner.command(row, deterministic=True), reference)
    torch.testing.assert_close(
        learner._policy_action(torch.zeros_like(reference), reference), reference
    )
    learner.update(rows(), warmup=True)
    torch.testing.assert_close(learner.command(row, deterministic=True), reference)
    state = learner.state_dict()
    AcceptanceLearner(
        "cpu",
        correction_limit=0.1,
        correction_axes=(0,),
        residual_output=True,
    ).load_state_dict(state)
    with pytest.raises(ValueError, match="residual-output mode differs"):
        AcceptanceLearner(
            "cpu", correction_limit=0.1, correction_axes=(0,)
        ).load_state_dict(state)
    with pytest.raises(ValueError, match="requires a correction limit"):
        AcceptanceLearner("cpu", residual_output=True)


def test_command_mapping_matches_official_conversion(source):
    import numpy as np
    from AlphaBrain.training.reinforcement_learning.common.rollout import (
        _postprocess_action,
        _unnormalize,
    )

    action = torch.linspace(-2, 2, 56).reshape(8, 7)
    original = action.clone()
    stats = {"q01": [-1.0] * 7, "q99": [1.0] * 7, "mask": [True] * 6 + [False]}
    actual = env_commands(action, stats)
    expected = np.stack(
        [_postprocess_action(a) for a in _unnormalize(action.numpy(), stats)]
    ).clip(-1, 1)
    np.testing.assert_array_equal(actual, expected)
    torch.testing.assert_close(action, original)
    assert set(actual[:, 6]) <= {-1, 1}


def test_state_digest_is_dtype_stable_but_sensitive_to_state():
    import numpy as np

    a = np.asarray([1, 2, 3], dtype=np.float32)
    assert snapshot_digest(a) == snapshot_digest(a.astype(np.float64))
    assert snapshot_digest(a) != snapshot_digest(a + 1)


@pytest.mark.skipif(
    os.environ.get("RLT_ACCEPTANCE_SIM_SMOKE") != "1",
    reason="Requires LIBERO assets and a working native renderer",
)
def test_simulator_worker_restores_fresh_initial_state_without_policy_leak(
    source, monkeypatch, tmp_path
):
    import numpy as np

    monkeypatch.setenv("RLT_LIBERO_WORKER_LOG_DIR", str(tmp_path / "workers"))
    monkeypatch.setenv("RLT_LIBERO_EGL_DEVICE_ID", "0")
    env = RemoteEnvironment()
    try:
        hashes = []
        for seed in (20000, 20001):
            case = {
                "id": str(seed),
                "kind": "generated",
                "task": 0,
                "seed": seed,
                "snapshot": str(tmp_path / f"{seed}.json"),
            }
            first = env.reset(case, create_snapshot=True)
            original_hash = env.reset_hash
            env.step(np.asarray([0.0] * 6 + [-1.0], dtype=np.float32))
            restored = env.reset(case)
            assert env.reset_hash == original_hash
            np.testing.assert_array_equal(first["state"], restored["state"])
            assert set(first) == {"state", "primary_image", "wrist_image"}
            hashes.append(original_hash)
        assert len(set(hashes)) == 2
    finally:
        env.close()
