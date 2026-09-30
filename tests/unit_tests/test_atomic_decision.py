# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Candidate, loss, routing, replay and checkpoint contracts on CPU."""

import copy
import io
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.atomic_decision import (
    atomic_actor_loss,
    atomic_next_value,
    candidate_bc_errors,
    candidate_q_values,
    validate_atomic_config,
)
from rlinf.algorithms.rlt.route import RLTRouteContext, SimulatorRLTRoute
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy import get_model
from rlinf.models.embodiment.mlp_policy.rlt_atomic_policy import RLTAtomicPolicy
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.modules.rlt_action_candidates import JointActionCandidates
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(**overrides):
    args = {
        "z_dim": 4,
        "proprio_dim": 3,
        "action_dim": 3,
        "num_action_chunks": 2,
        "atomic_decision": {"radius": 0.1, "reference_prior": 0.8},
    }
    args.update(overrides)
    return RLTAtomicPolicy(**args)


def make_obs(batch=4):
    return {
        "z_rl": torch.randn(batch, 4),
        "proprio": torch.randn(batch, 3),
        "ref_chunk": torch.full((batch, 2, 3), 0.4),
    }


def make_worker(model):
    worker = RLTACLossMixin()
    worker.model = model
    worker.target_model = copy.deepcopy(model)
    worker.torch_dtype = torch.float32
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "num_action_chunks": 2,
                    "action_dim": model.step_action_dim,
                    "model_type": "rlt_mlp_policy",
                    "q_head_type": "default",
                    "atomic_decision": {"enabled": True},
                }
            },
            "algorithm": {
                "gamma": 0.9,
                "bc_weight": 1.0,
                "q_weight": 1.0,
                "reference_dropout_prob": 0.0,
                "loss_type": "rlt_ac",
            },
            "env": {
                "train": {
                    "env_type": "maniskill_rlt",
                    "init_params": {"control_mode": "pd_joint_delta_pos"},
                }
            },
            "rollout": {"model": {"atomic_decision": {"enabled": True}}},
        }
    )
    return worker


def test_candidates_bound_gripper_dedup_and_names():
    bank = JointActionCandidates(8, 10, 0.08)
    ref = torch.randn(16, 10, 8) * 1.2
    choices, valid = bank(ref)
    assert choices.shape == (16, 17, 10, 8)
    assert bank.names[3:5] == ("joint_0_plus", "joint_0_minus")
    assert torch.equal(choices[:, 0], ref.clamp(-1, 1))
    assert ((choices - ref.clamp(-1, 1)[:, None]).abs() <= 0.080001).all()
    assert torch.equal(
        choices[..., -1], ref[..., -1].clamp(-1, 1)[:, None].expand(-1, 17, -1)
    )
    assert valid[:, 0].all()
    _, valid = bank(torch.zeros(1, 10, 8))
    assert not valid[0, 1:3].any()
    assert valid.sum() == 15


@pytest.mark.parametrize("radius", [0, -1, 1.1, float("nan"), float("inf")])
def test_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        JointActionCandidates(8, 10, radius)


@pytest.mark.parametrize("key", ["z_rl", "proprio", "ref_chunk"])
def test_nonfinite_obs_rejected(key):
    obs = make_obs()
    obs[key].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        make_model().decision_forward(obs)


def test_selection_is_candidate_not_mean_and_has_proposal_metadata():
    torch.manual_seed(4)
    model, obs = make_model(), make_obs(256)
    decision = model.decision_forward(obs)
    action, result = model.predict_action_batch(obs)
    idx = result["forward_inputs"]["atomic_proposed_id"].squeeze(-1)
    assert torch.equal(action, decision["candidates"][torch.arange(256), idx])
    assert idx.unique().numel() > 1
    assert torch.isfinite(result["prev_logprobs"]).all()
    assert torch.equal(
        model.predict_action_batch(obs, mode="eval")[0], obs["ref_chunk"]
    )
    assert torch.allclose(decision["probabilities"].sum(-1), torch.ones(256))


def test_exact_actor_gradient_reaches_selector_not_critic_or_observation():
    model, obs = make_model(), make_obs()
    for value in obs.values():
        value.requires_grad_(True)
    worker = make_worker(model)
    batch = {"curr_obs": obs, "actions": obs["ref_chunk"].detach()}
    loss, entropy, metrics = atomic_actor_loss(worker, batch)
    loss.backward()
    assert model.selector.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.q_head.parameters())
    assert all(value.grad is None for value in obs.values())
    assert entropy > 0 and 0 < metrics["atomic/reference_probability"] <= 1
    decision = model.decision_forward(obs)
    with torch.no_grad():
        q = candidate_q_values(model, obs, decision["candidates"])
        bc, _ = candidate_bc_errors(
            decision["candidates"], batch["actions"], obs["ref_chunk"], None
        )
        target = torch.softmax(decision["prior_logits"] - (bc - q[..., 0]) / 0.1, -1)
        logs = torch.log_softmax(decision["logits"], -1).masked_fill(
            ~decision["valid"], 0
        )
        expected = -(target * logs).sum(-1).mean()
    assert torch.allclose(loss.detach(), expected)


def test_saturated_selector_keeps_recovery_gradient():
    model, obs = make_model(), make_obs()
    with torch.no_grad():
        model.selector.bias[:] = -100
        model.selector.bias[1] = 100
    worker = make_worker(model)
    # Only BC is active: reference is preferable to the dampened proposal.
    worker.cfg.algorithm.q_weight = 0
    loss, _, _ = atomic_actor_loss(
        worker, {"curr_obs": obs, "actions": obs["ref_chunk"]}
    )
    loss.backward()
    assert model.selector.bias.grad[0] < -0.1
    assert model.selector.bias.grad[1] > 0.1


def test_intervention_bc_uses_actual_actions_outside_vocabulary():
    model, obs = make_model(), make_obs()
    choices = model.decision_forward(obs)["candidates"]
    action = torch.full_like(obs["ref_chunk"], -0.9)
    flags = torch.ones_like(action, dtype=torch.bool)
    errors, human = candidate_bc_errors(choices, action, obs["ref_chunk"], flags)
    assert human.all()
    assert torch.allclose(errors, (choices - action[:, None]).square().mean((-1, -2)))
    assert errors.min() > 0.1  # A restricted vocabulary cannot fake expert coverage.


def test_next_value_exact_categorical_expectation():
    model, obs = make_model(), make_obs()
    target = copy.deepcopy(model)
    decision = model.decision_forward(obs)
    q = candidate_q_values(target, obs, decision["candidates"])
    expected = (decision["probabilities"] * q.min(-1).values).sum(-1, keepdim=True)
    value = atomic_next_value(model, target, obs)
    assert torch.allclose(value, expected)
    assert not value.requires_grad


def test_checkpoint_roundtrip_and_incompatible_radius():
    model, obs = make_model(), make_obs()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    worker = make_worker(model)
    atomic_actor_loss(worker, {"curr_obs": obs, "actions": obs["ref_chunk"]})[
        0
    ].backward()
    optimizer.step()
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optim": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    clone = make_model()
    clone.load_state_dict(state["model"])
    clone_optim = torch.optim.Adam(clone.parameters(), lr=1e-3)
    clone_optim.load_state_dict(state["optim"])
    assert clone_optim.state_dict()["state"]
    assert torch.equal(
        model.decision_forward(obs)["probabilities"],
        clone.decision_forward(obs)["probabilities"],
    )
    with pytest.raises(RuntimeError, match="contract mismatch"):
        make_model(atomic_decision={"radius": 0.2}).load_state_dict(state["model"])


def test_disabled_factory_is_bitwise_baseline():
    cfg = OmegaConf.create(
        {
            "model_type": "rlt_mlp_policy",
            "z_dim": 4,
            "proprio_dim": 3,
            "action_dim": 3,
            "num_action_chunks": 2,
        }
    )
    torch.manual_seed(10)
    original = get_model(cfg)
    cfg.atomic_decision = {"enabled": False}
    torch.manual_seed(10)
    disabled = get_model(cfg)
    assert type(disabled) is RLTMLPPolicy
    assert original.state_dict().keys() == disabled.state_dict().keys()
    for key in original.state_dict():
        assert torch.equal(original.state_dict()[key], disabled.state_dict()[key])


def test_routing_preserves_actual_execution_and_proposed_id():
    model, obs = make_model(), make_obs()
    proposed, result = model.predict_action_batch(obs)
    route = SimulatorRLTRoute(use_schedule=True, warmup_updates=2)
    output = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs=obs,
            student_actions=proposed,
            result=result,
            mode="train",
            version=0,
            rlt_switch_flags=torch.ones(4, 1, dtype=torch.bool),
            intervene_requested=None,
            expert_model=None,
        )
    )
    assert torch.equal(output.actions, obs["ref_chunk"])
    assert torch.equal(
        output.result["forward_inputs"]["action"], obs["ref_chunk"].flatten(1)
    )
    assert "atomic_proposed_id" in output.result["forward_inputs"]
    assert not output.result["forward_inputs"]["actor_switch"].any()


def test_overlay_config(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "examples/embodiment/config"
    for key, value in {
        "EMBODIED_PATH": str(config_dir.parent),
        "RLT_SMOKE_RUN_DIR": "/tmp/not-created",
        "RLT_STAGE1_ACTOR": "/tmp/actor",
        "RLT_DATASET_DIR": "/tmp/data",
        "RLT_SMOKE_RENDER_DEVICE": "pci:0000:e1:00.0",
    }.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name="maniskill_rlt_stage2_atomic_gpu2")
    OmegaConf.resolve(cfg)
    validate_atomic_config(cfg)
    assert set(cfg.cluster.component_placement.values()) == {"2-2"}
    assert cfg.rollout.expert_model is None
    assert cfg.env.train.max_episode_steps == 500
    model = get_model(cfg.actor.model)
    assert isinstance(model, RLTAtomicPolicy)
    cfg.algorithm.loss_type = "rlt_td3"
    with pytest.raises(ValueError, match="rlt_ac"):
        validate_atomic_config(cfg)


@pytest.mark.parametrize("reward_horizon", [1, 2])
def test_worker_td_uses_reward_horizon_and_masks_terminal(reward_horizon):
    model = make_model(action_dim=8)
    worker = make_worker(model)
    obs, next_obs = make_obs(2), make_obs(2)
    obs["ref_chunk"] = torch.full((2, 2, 8), 0.4)
    next_obs["ref_chunk"] = torch.full((2, 2, 8), 0.2)
    # An actual intervention is not restricted to the candidate bank.
    actual = torch.full((2, 2, 8), -0.8)
    batch = {
        "curr_obs": obs,
        "next_obs": next_obs,
        "actions": actual,
        "rewards": torch.ones(2, reward_horizon),
        "dones": torch.tensor([[True], [False]]),
        "terminations": torch.zeros(2, 1, dtype=torch.bool),
    }
    method = RLTACLossMixin.forward_critic
    while hasattr(method, "__wrapped__"):
        method = method.__wrapped__
    loss, _ = method(worker, batch)
    q_data = model(forward_type=ForwardType.SAC_Q, obs=obs, actions=actual)
    target = worker._discounted_chunk_rewards(batch["rewards"])
    target[1] += (
        0.9**reward_horizon * atomic_next_value(model, worker.target_model, next_obs)[1]
    )
    assert torch.allclose(loss, (q_data - target).square().mean())
    loss.backward()
    assert all(p.grad is None for p in model.selector.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.q_head.parameters()
    )


def test_real_optimizer_partition_has_no_orphans_or_overlap():
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

    model = make_model()
    optim_cfg = OmegaConf.create({"lr": 1e-3})
    actor, critic = FSDPModelManager.build_optimizers(
        None,
        model,
        optim_cfg,
        {"critic": ["encoders", "encoder", "q_head", "state_proj"]},
        {"critic": optim_cfg},
    )
    actor_ids = {id(p) for g in actor.param_groups for p in g["params"]}
    critic_ids = {id(p) for g in critic.param_groups for p in g["params"]}
    assert not actor_ids & critic_ids
    assert actor_ids | critic_ids == {id(p) for p in model.parameters()}
    assert {id(p) for p in model.q_head.parameters()} == critic_ids


def test_real_replay_preserves_executed_action_and_proposal_separately():
    from rlinf.data.schema.embodied_types import Trajectory
    from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer

    model, obs = make_model(), make_obs(1)
    proposal, info = model.predict_action_batch(obs, mode="eval")
    # An environment intervention replaces the proposal before storing a row.
    actual = torch.full_like(proposal, -0.8).flatten(1)
    inputs = info["forward_inputs"]
    inputs["action"] = actual
    trajectory = Trajectory(
        max_episode_length=1,
        actions=actual.unsqueeze(0),
        rewards=torch.ones(1, 1, 2),
        dones=torch.ones(1, 1, 1, dtype=torch.bool),
        forward_inputs={key: value.unsqueeze(0) for key, value in inputs.items()},
        curr_obs={key: value.unsqueeze(0) for key, value in obs.items()},
        next_obs={key: value.unsqueeze(0) for key, value in obs.items()},
    )
    replay = TrajectoryReplayBuffer(auto_save=False)
    try:
        replay.add_trajectories([trajectory])
        batch = replay.sample(1)
        assert torch.equal(batch["actions"], actual)
        assert (batch["forward_inputs"]["atomic_proposed_id"] == 0).all()
        assert torch.equal(batch["forward_inputs"]["model_action"], proposal.flatten(1))
    finally:
        replay.close()


def test_worker_update_clips_only_active_optimizer_gradients():
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

    clipped_groups = []

    class CPUModel(RLTAtomicPolicy):
        def clip_grad_norm_(self, max_norm):
            clipped_groups.append(
                {name for name, p in self.named_parameters() if p.grad is not None}
            )
            return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    model = CPUModel(
        z_dim=4, proprio_dim=3, action_dim=8, num_action_chunks=2, atomic_decision={}
    )
    worker = make_worker(model)
    worker.cfg.actor.global_batch_size = 4
    worker.cfg.actor.micro_batch_size = 2
    worker.cfg.actor.optim = {"lr": 1e-3, "clip_grad": 1.0}
    worker.cfg.actor.critic_optim = {"lr": 1e-3, "clip_grad": 1.0}
    worker.optimizer, worker.qf_optimizer = FSDPModelManager.build_optimizers(
        None,
        model,
        worker.cfg.actor.optim,
        {"critic": ["q_head"]},
        {"critic": worker.cfg.actor.critic_optim},
    )
    worker.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizer, lambda _: 1
    )
    worker.qf_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.qf_optimizer, lambda _: 1
    )
    worker._world_size, worker.gradient_accumulation = 1, 2
    worker.device, worker.enable_drq = torch.device("cpu"), False
    worker.worker_timer = lambda _: nullcontext()
    worker.update_step, worker.critic_actor_ratio = 0, 1
    worker.alpha_optimizer = None
    worker.entropy_temp = SimpleNamespace(alpha=0.0)
    worker.target_model_initialized = False
    for name in ("forward_actor", "forward_critic"):
        method = getattr(RLTACLossMixin, name)
        while hasattr(method, "__wrapped__"):
            method = method.__wrapped__
        setattr(worker, name, MethodType(method, worker))
    obs = make_obs(4)
    obs["ref_chunk"] = torch.full((4, 2, 8), 0.3)
    batch = {
        "curr_obs": obs,
        "next_obs": obs,
        "actions": obs["ref_chunk"],
        "rewards": torch.ones(4, 2),
        "dones": torch.zeros(4, 1, dtype=torch.bool),
        "terminations": torch.zeros(4, 1, dtype=torch.bool),
    }
    worker.buffer_dataloader_iter = iter([batch, batch])
    update = EmbodiedSACFSDPPolicy.update_one_epoch
    while hasattr(update, "__wrapped__"):
        update = update.__wrapped__
    for _ in range(2):
        metrics = update(worker)
        assert torch.isfinite(torch.tensor(metrics["sac/actor_loss"]))
    assert len(clipped_groups) == 4
    for critic_group, actor_group in zip(clipped_groups[::2], clipped_groups[1::2]):
        assert critic_group and all("q_head" in name for name in critic_group)
        assert actor_group and all("q_head" not in name for name in actor_group)
