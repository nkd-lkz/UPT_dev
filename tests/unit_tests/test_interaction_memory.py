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
