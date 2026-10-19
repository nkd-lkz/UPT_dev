# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the native evaluator/client boundary without an Isaac runtime."""

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from toolkits.rlt.robodojo_boundary import ACTION_SEMANTICS, CAMERAS, FIELDS
from toolkits.rlt.robodojo_collect import (
    select_collection_layouts,
    select_reserved_layouts,
)
from toolkits.rlt.robodojo_layouts import digest, verify
from toolkits.rlt.robodojo_learning import ArxLearner, ArxNormalizer, load_rollout_rows
from toolkits.rlt.robodojo_online import (
    action_deviation,
    apply_frozen_action_adapter,
    attach_successor,
    checkpoint,
    disallowed_gpu_processes,
    load_reference_replay,
    prediction_arrays,
    process_details,
    renderer_failed,
    restore_online_checkpoint,
    select_layout,
)
from toolkits.rlt.robodojo_policy import (
    Model,
    eval_one_episode,
    eval_one_episode_batch,
    validate_normalization,
)


def test_collection_layout_window_selects_unseen_non_evaluation_ids():
    split = {
        "layouts": [
            *({"id": index, "role": "train"} for index in range(4)),
            *({"id": index, "role": "development"} for index in range(4, 6)),
            {"id": 6, "role": "reserved"},
        ]
    }
    selected = select_collection_layouts(
        split,
        train_episodes=2,
        development_episodes=1,
        train_offset=2,
        development_offset=1,
    )
    assert [row["id"] for row in selected] == [2, 3, 5]
    with pytest.raises(ValueError, match="exceeds"):
        select_collection_layouts(
            split,
            train_episodes=2,
            development_episodes=1,
            train_offset=3,
        )


def test_reserved_reference_evaluation_selects_only_protected_layouts():
    split = {
        "layouts": [
            {"id": 0, "role": "train"},
            {"id": 1, "role": "development"},
            {"id": 2, "role": "reserved"},
            {"id": 3, "role": "reserved"},
        ]
    }
    assert [row["id"] for row in select_reserved_layouts(split, 2)] == [2, 3]
    with pytest.raises(ValueError, match="reserved split"):
        select_reserved_layouts(split, 3)


def test_prediction_arrays_accept_inline_reserved_features_without_a_cache(tmp_path):
    hidden = np.arange(6, dtype=np.float32).reshape(2, 3)
    state = np.arange(14, dtype=np.float32)
    reference = np.arange(140, dtype=np.float32).reshape(10, 14)
    inline = prediction_arrays(
        {
            "cache_file": None,
            "features": {
                "hidden": hidden,
                "state": state,
                "reference": reference,
            },
        }
    )
    for actual, expected in zip(inline, (hidden, state, reference), strict=True):
        np.testing.assert_array_equal(actual, expected)

    cache = tmp_path / "features.npz"
    np.savez(cache, hidden=hidden, state=state, reference=reference)
    cached = prediction_arrays({"cache_file": str(cache)})
    for actual, expected in zip(cached, (hidden, state, reference), strict=True):
        np.testing.assert_array_equal(actual, expected)

    with pytest.raises(ValueError, match="neither a cache file nor inline features"):
        prediction_arrays({"cache_file": None})


@pytest.mark.parametrize("change", ["short", "nan", "negative_std", "quantile_order"])
def test_normalization_rejects_incompatible_statistics(tmp_path, change):
    stats = {
        name: {
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
            "q01": [-1.0] * 14,
            "q99": [1.0] * 14,
        }
        for name in ("state", "actions")
    }
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": stats}))
    validate_normalization(path)
    if change == "short":
        stats["actions"]["mean"] = [0.0] * 7
    elif change == "nan":
        stats["actions"]["mean"][0] = float("nan")
    elif change == "negative_std":
        stats["state"]["std"][0] = -1
    else:
        stats["actions"]["q01"][0] = 2
    path.write_text(json.dumps({"norm_stats": stats}))
    with pytest.raises(ValueError):
        validate_normalization(path)


def normalization(tmp_path):
    stats = {
        name: {
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
            "q01": [-1.0] * 14,
            "q99": [1.0] * 14,
        }
        for name in ("state", "actions")
    }
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": stats}))
    return ArxNormalizer(path)


def test_learning_normalizer_uses_joint_deltas_and_absolute_grippers(tmp_path):
    normalizer = normalization(tmp_path)
    state = np.arange(14, dtype=np.float32) / 10
    physical = np.tile(state, (3, 1))
    physical[:, normalizer.joints] += 0.25
    physical[:, [6, 13]] = [[-0.2, 1.2], [0.4, 0.7], [0.0, 1.0]]
    encoded = normalizer.encode(physical, state)
    decoded = normalizer.decode(encoded, state)
    np.testing.assert_allclose(
        decoded[:, normalizer.joints], physical[:, normalizer.joints]
    )
    np.testing.assert_allclose(decoded[:, [6, 13]], np.clip(physical[:, [6, 13]], 0, 1))


def test_rollout_loader_checks_acknowledged_physical_commands(tmp_path):
    normalizer = normalization(tmp_path)
    root = tmp_path / "collection"
    episode = root / "episode-3"
    episode.mkdir(parents=True)
    state = np.zeros(14, dtype=np.float32)
    physical = np.zeros((10, 14), dtype=np.float32)
    physical[:, [6, 13]] = 1.2
    cache = episode / "chunk-0.npz"
    np.savez(cache, hidden=np.zeros((768, 2048)), state=state, reference=physical)
    record = {
        "cache_file": str(cache),
        "ticks": 2,
        "executed_physical": np.clip(physical[:2], 0, 1).tolist(),
        "reward": 1.0,
        "terminated": True,
        "truncated": False,
    }
    (episode / "episode.json").write_text(
        json.dumps(
            {
                "layout": {"id": 3, "sha256": "digest", "role": "development"},
                "complete": True,
                "records": [record],
            }
        )
    )
    rows = load_rollout_rows([root], normalizer)
    assert len(rows) == 1 and rows[0]["td_eligible"] is True
    np.testing.assert_allclose(rows[0]["executed_action"], rows[0]["ref"][:2])
    record["executed_physical"][0][0] = 0.5
    (episode / "episode.json").write_text(
        json.dumps(
            {
                "layout": {"id": 3, "sha256": "digest", "role": "development"},
                "complete": True,
                "records": [record],
            }
        )
    )
    with pytest.raises(ValueError, match="Acknowledged commands differ"):
        load_rollout_rows([root], normalizer)


def test_rollout_loader_rejects_timeout_as_bc_demonstration(tmp_path):
    normalizer = normalization(tmp_path)
    root = tmp_path / "collection"
    episode = root / "episode-3"
    episode.mkdir(parents=True)
    cache = episode / "chunk-0.npz"
    state = np.zeros(14, dtype=np.float32)
    physical = np.zeros((10, 14), dtype=np.float32)
    np.savez(cache, hidden=np.zeros((768, 2048)), state=state, reference=physical)
    (episode / "episode.json").write_text(
        json.dumps(
            {
                "layout": {"id": 3, "sha256": "digest", "role": "train"},
                "complete": True,
                "records": [
                    {
                        "cache_file": str(cache),
                        "ticks": 10,
                        "executed_physical": physical.tolist(),
                        "reward": 0.0,
                        "terminated": False,
                        "truncated": True,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="No successful complete rollout"):
        load_rollout_rows([root], normalizer)


def test_online_transition_attaches_next_pre_action_state():
    current = {"z": "old-z", "ref": "old-ref", "prop": "old-prop", "reward": 0}
    successor = {"z": "new-z", "ref": "new-ref", "prop": "new-prop"}
    result = attach_successor(current, successor)
    assert result == current | {
        "next_z": "new-z",
        "next_ref": "new-ref",
        "next_prop": "new-prop",
    }
    assert "next_z" not in current


def test_action_deviation_uses_only_acknowledged_prefix():
    reference = np.zeros((10, 14), dtype=np.float32)
    issued = reference.copy()
    issued[:2] = 0.25
    issued[2:] = 10
    assert action_deviation(issued, reference, 2) == {
        "reference_mean_abs": 0.25,
        "reference_max_abs": 0.25,
    }
    with pytest.raises(ValueError, match="10x14"):
        action_deviation(issued[:2], reference, 2)


def test_frozen_reference_gripper_adapter_preserves_all_arm_actions():
    action = np.arange(140, dtype=np.float32).reshape(10, 14)
    reference = -action
    result = apply_frozen_action_adapter(action, reference, "reference_grippers")
    arm = np.ones(14, dtype=bool)
    arm[[6, 13]] = False
    np.testing.assert_array_equal(result[:, arm], action[:, arm])
    np.testing.assert_array_equal(result[:, [6, 13]], reference[:, [6, 13]])
    assert not np.shares_memory(result, action)
    np.testing.assert_array_equal(
        apply_frozen_action_adapter(action, reference, "none"), action
    )
    with pytest.raises(ValueError, match="Unknown"):
        apply_frozen_action_adapter(action, reference, "residual")


def test_reference_anchor_bounds_the_learned_correction():
    learner = object.__new__(ArxLearner)
    reference = torch.tensor([[-1.0, 0.0, 1.0]])
    action = torch.tensor([[3.0, 2.0, -1.0]])
    learner.configure_reference_anchor(0.25)
    torch.testing.assert_close(
        learner._anchored(action, reference),
        torch.tensor([[0.0, 0.5, 0.5]]),
    )
    learner.configure_reference_anchor(0.0)
    torch.testing.assert_close(learner._anchored(action, reference), reference)
    learner.configure_reference_anchor(1.0)
    torch.testing.assert_close(learner._anchored(action, reference), action)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        learner.configure_reference_anchor(1.01)


def test_candidate_layout_manifest_requires_independent_provenance(tmp_path):
    protected = tmp_path / "protected.json"
    protected.write_text('{"protected": true}\n')
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "Rigid": {},
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "robodojo-candidate-layouts-v1",
                "task": "insert_key",
                "protected": [{"path": str(protected), "sha256": digest(protected)}],
                "layouts": [
                    {
                        "id": 0,
                        "role": "train",
                        "path": str(layout),
                        "sha256": digest(layout),
                        "provenance": {
                            "task": "insert_key",
                            "layout_domain": "train",
                            "layout_id": 0,
                        },
                    }
                ],
            }
        )
    )
    assert verify(manifest)["task"] == "insert_key"
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["layouts"][0]["provenance"]["task"] = "stack_bowls"
    manifest.write_text(json.dumps(manifest_payload))
    with pytest.raises(ValueError, match="provenance"):
        verify(manifest)


def test_frozen_evaluation_never_repeats_a_layout_after_reset_failure():
    layouts = [{"id": value} for value in (16, 17, 18, 19)]
    outcomes = [
        {"layout_id": 16, "state": "complete"},
        {"layout_id": 17, "state": "complete"},
        {"layout_id": 18, "state": "reset_failed"},
    ]
    assert select_layout(layouts, {18}, outcomes, 3, frozen_evaluation=True) == {
        "id": 19
    }
    assert select_layout(layouts, {18}, outcomes, 4, frozen_evaluation=False) == {
        "id": 17
    }


class LearnerState:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, value):
        self.value = value["value"]


def test_online_checkpoint_restores_restart_boundary_and_weights(tmp_path):
    encoder = torch.nn.Linear(2, 2)
    expected = {
        name: value.detach().clone() for name, value in encoder.state_dict().items()
    }
    learner = LearnerState(17)
    state = {
        "control_steps": 1000,
        "updates": 4,
        "episode_id": 2,
        "replay": [{"reward": 1.0}],
        "outcomes": [],
        "invalid_layouts": [],
        "python_rng_state": (3, (), None),
        "torch_rng_state": torch.get_rng_state(),
        "restart_boundary": "fresh_episode",
    }
    path = tmp_path / "online.pt"
    checkpoint(
        path,
        encoder,
        learner,
        {
            "layout_digest": "layout",
            "reference_replay_sha256": "reference",
            "seed": 42,
            "exploration_std": 0.01,
            "target_noise_std": 0.02,
            "target_noise_clip": 0.05,
            "reference_anchor_scale": 1.0,
        },
        state,
    )
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
    learner.value = -1
    restored = restore_online_checkpoint(
        path,
        encoder,
        learner,
        layout_digest="layout",
        reference_replay_digest="reference",
        seed=42,
        exploration_std=0.01,
        target_noise_std=0.02,
        target_noise_clip=0.05,
        reference_anchor_scale=1.0,
    )
    assert restored["restart_boundary"] == "fresh_episode"
    assert learner.value == 17
    for name, value in encoder.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    with pytest.raises(ValueError, match="provenance"):
        restore_online_checkpoint(
            path,
            encoder,
            learner,
            layout_digest="other",
            reference_replay_digest="reference",
            seed=42,
            exploration_std=0.01,
            target_noise_std=0.02,
            target_noise_clip=0.05,
            reference_anchor_scale=1.0,
        )
    with pytest.raises(ValueError, match="provenance"):
        restore_online_checkpoint(
            path,
            encoder,
            learner,
            layout_digest="layout",
            reference_replay_digest="reference",
            seed=42,
            exploration_std=0.01,
            target_noise_std=0.02,
            target_noise_clip=0.05,
            reference_anchor_scale=0.25,
        )


def test_reference_replay_seeds_sparse_reward_with_train_rows_only(tmp_path):
    accepted = {0: "a", 1: "b"}
    base = {
        "z": torch.zeros(2048),
        "ref": torch.zeros(10, 14),
        "prop": torch.zeros(14),
        "action": torch.zeros(10, 14),
        "executed_action": np.zeros((10, 14), dtype=np.float32),
        "ticks": 10,
        "terminated": False,
        "truncated": False,
        "next_z": torch.zeros(2048),
        "next_ref": torch.zeros(10, 14),
        "next_prop": torch.zeros(14),
        "role": "train",
        "td_eligible": True,
    }
    rows = []
    for index in range(64):
        positive = index in {0, 1}
        rows.append(
            base
            | {
                "episode": (index % 2, "a" if index % 2 == 0 else "b"),
                "reward": float(positive),
                "terminated": positive,
            }
        )
    rows.append(base | {"role": "development", "reward": 1.0})
    path = tmp_path / "encoded.pt"
    torch.save(rows, path)
    replay = load_reference_replay(path, accepted)
    assert len(replay) == 64
    assert sum(row["reward"] > 0 for row in replay) == 2
    rows[0]["episode"] = (0, "changed")
    torch.save(rows, path)
    with pytest.raises(ValueError, match="differs from warmup"):
        load_reference_replay(path, accepted)


def test_online_watchdog_requires_the_fatal_renderer_error_pair(tmp_path):
    log = tmp_path / "client.log"
    log.write_text("Unable to allocate descriptor sets.\n")
    assert not renderer_failed(log)
    log.write_text(
        "Unable to allocate descriptor sets.\n"
        "Failed to execute RenderGraph on device 0. Error Code: 7\n"
    )
    assert renderer_failed(log)


def test_process_details_preserves_missing_gpu_process_identity():
    assert process_details([999999999]) == [
        {
            "pid": 999999999,
            "user": None,
            "elapsed_seconds": None,
            "command": None,
        }
    ]


def test_gpu_coexistence_rejects_unresolved_process_owners():
    assert disallowed_gpu_processes([999999999], {"liym"}) == [999999999]
    assert disallowed_gpu_processes([999999999], set()) == [999999999]


def observation(index):
    return {
        "env_idx": index,
        "instruction": "stack bowls",
        "state": {name: np.full(dim, index) for name, dim in FIELDS},
        "vision": {
            name: {"color": np.zeros((4, 5, 3), dtype=np.uint8)} for name, _ in CAMERAS
        },
        "privileged_success": True,
    }


class Backend:
    def __init__(self, width=14):
        self.width = width
        self.inputs = []

    def infer(self, obs):
        self.inputs.append(obs)
        result = np.zeros((3, self.width), dtype=np.float32)
        result[:, 0] = obs["state"][0]
        return {"actions": result}


def model(backend=None):
    return Model(
        {
            "action_type": "joint",
            "env_cfg_type": "arx_x5",
            "action_horizon": 3,
            "model_contract": {
                "embodiment": "arx_x5",
                "action_dim": 14,
                "action_semantics": ACTION_SEMANTICS,
                "normalization_id": "test",
            },
        },
        backend=backend or Backend(),
    )


class Client:
    """The external WS dispatcher delivers decoded observations positionally."""

    def __init__(self, policy):
        self.policy = policy

    def call(self, *, func_name, obs=None):
        function = getattr(self.policy, func_name)
        return function() if obs is None else function(obs)


class BatchEnv:
    """Native IDs reorder, and one environment finishes inside a chunk."""

    def __init__(self):
        self.remaining = {8: 1, 2: 4}
        self.ticks = []

    def is_episode_end(self):
        return not any(self.remaining.values())

    def get_running_env_idx_list(self):
        order = [8, 2] if not self.ticks else [2, 8]
        return [i for i in order if self.remaining[i] > 0]

    def get_obs_batch(self, indices):
        return [observation(i) for i in indices]

    def take_action_batch(self, actions, indices):
        for i, action in zip(indices, actions, strict=True):
            assert self.remaining[i] > 0
            assert action["left_arm_joint_state"][0] == i
            self.remaining[i] -= 1
        self.ticks.append(list(indices))


@pytest.mark.parametrize("native_keywords", [False, True])
def test_native_batch_loop_reorders_shrinks_and_refreshes_observations(native_keywords):
    policy, env = model(), BatchEnv()
    if native_keywords:
        eval_one_episode_batch(TASK_ENV=env, model_client=Client(policy))
    else:
        eval_one_episode_batch(env, Client(policy))
    assert env.ticks == [[8, 2], [2], [2], [2]]
    assert [o["state"][0] for o in policy.backend.inputs] == [8, 2, 2]
    assert all(set(o) == {"state", "images", "prompt"} for o in policy.backend.inputs)


@pytest.mark.parametrize("native_keywords", [False, True])
def test_single_episode_stops_mid_chunk_and_reset_invalidates_inputs(native_keywords):
    class SingleEnv:
        count = 0

        def is_episode_end(self):
            return self.count == 1

        def get_obs(self):
            obs = observation(0)
            del obs["env_idx"]
            return obs

        def take_action(self, action):
            self.count += 1

    policy, env = model(), SingleEnv()
    if native_keywords:
        eval_one_episode(TASK_ENV=env, model_client=Client(policy))
    else:
        eval_one_episode(env, Client(policy))
    assert env.count == 1
    with pytest.raises(ValueError, match="update_obs"):
        policy.get_action()
    policy.update_obs(observation(2))
    policy.reset()
    policy.reset()
    with pytest.raises(ValueError, match="Update"):
        policy.get_action_batch([2])


def test_mislabeled_libero_backend_is_rejected_before_environment_execution():
    policy = model(Backend(width=7))
    policy.update_obs(observation(0))
    with pytest.raises(ValueError, match="LIBERO"):
        policy.get_action()
    with pytest.raises(ValueError, match="update_obs"):
        policy.get_action()


@pytest.mark.parametrize("chunks", [[], [[]], [[], []], [[{}], [{}, {}]]])
def test_bad_server_batch_stops_before_any_simulator_step(chunks):
    class BrokenClient:
        def call(self, *, func_name, obs=None):
            if func_name == "get_action_batch":
                return chunks

    env = BatchEnv()
    with pytest.raises(ValueError):
        eval_one_episode_batch(env, BrokenClient())
    assert env.ticks == []


def test_invalid_model_contract_does_not_attempt_backend_loading():
    with pytest.raises(ValueError, match="LIBERO"):
        Model(
            {
                "action_type": "joint",
                "env_cfg_type": "arx_x5",
                "action_horizon": 1,
                "model_contract": {"action_dim": 7},
            }
        )


@pytest.mark.skipif(
    not os.environ.get("RP_XPOLICYLAB_SOURCE"),
    reason="Optional pinned XPolicyLab source and WS dependencies required",
)
@pytest.mark.parametrize("encoded", [False, True])
def test_native_websocket_transport_roundtrip(monkeypatch, encoded):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    source = Path(os.environ["RP_XPOLICYLAB_SOURCE"])
    monkeypatch.syspath_prepend(str(source.parent))
    monkeypatch.syspath_prepend(str(source))
    from client_server.ws.model_client import WsModelClient
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from XPolicyLab.utils.process_data import encode_image_bit

    policy = model()

    class CameraEnv(BatchEnv):
        def get_obs_batch(self, indices):
            observations = super().get_obs_batch(indices)
            for obs in observations:
                for index, camera in enumerate(obs["vision"].values()):
                    camera["color"][:] = [230, 50, 5]
                    if encoded:
                        payload = encode_image_bit(camera["color"])
                        camera["color"] = (
                            payload if index % 2 else np.frombuffer(payload, np.uint8)
                        )
            return observations

    async def run():
        server = PolicyServer(policy, PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()

        def request():
            with WsModelClient(
                url=server.url,
                evaluation_id="cpu-contract",
                trial_id="trial-0",
                request_timeout_s=3,
                connect_timeout_s=3,
                handshake_timeout_s=3,
                max_connect_attempts=1,
                max_connect_seconds=5,
                close_timeout_s=1,
            ) as client:
                env = CameraEnv()
                eval_one_episode_batch(TASK_ENV=env, model_client=client)
                assert env.ticks == [[8, 2], [2], [2], [2]]
                for obs in policy.backend.inputs:
                    for rgb in obs["images"].values():
                        np.testing.assert_allclose(rgb[:, 0, 0], [230, 50, 5], atol=3)

        try:
            await asyncio.wait_for(asyncio.to_thread(request), timeout=20)
        finally:
            await server.stop()

    asyncio.run(run())
