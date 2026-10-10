# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the native evaluator/client boundary without an Isaac runtime."""

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

from toolkits.rlt.robodojo_boundary import ACTION_SEMANTICS, CAMERAS, FIELDS
from toolkits.rlt.robodojo_policy import Model, eval_one_episode, eval_one_episode_batch


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


def test_native_batch_loop_reorders_shrinks_and_refreshes_observations():
    policy, env = model(), BatchEnv()
    eval_one_episode_batch(env, Client(policy))
    assert env.ticks == [[8, 2], [2], [2], [2]]
    assert [o["state"][0] for o in policy.backend.inputs] == [8, 2, 2]
    assert all(set(o) == {"state", "images", "prompt"} for o in policy.backend.inputs)


def test_single_episode_stops_mid_chunk_and_reset_invalidates_inputs():
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
def test_native_websocket_transport_roundtrip(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    source = Path(os.environ["RP_XPOLICYLAB_SOURCE"])
    monkeypatch.syspath_prepend(str(source.parent))
    monkeypatch.syspath_prepend(str(source))
    from client_server.ws.model_client import WsModelClient
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    policy = model()

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
                env = BatchEnv()
                eval_one_episode_batch(env, client)
                assert env.ticks == [[8, 2], [2], [2], [2]]

        try:
            await asyncio.wait_for(asyncio.to_thread(request), timeout=20)
        finally:
            await server.stop()

    asyncio.run(run())
