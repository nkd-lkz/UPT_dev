# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contracts for local takeover, bounded RPC and executed-transition recording."""

import os
import socket
import struct
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from toolkits.rlt_vr.client import TransitionRecorder
from toolkits.rlt_vr.control import (
    ActionGate,
    OperatorControl,
    map_relative_target,
    relative_target,
)
from toolkits.rlt_vr.protocol import (
    CONTRACT,
    MAX_MESSAGE,
    decode_image,
    encode_image,
    receive,
    request,
    send,
    validate_actions,
)
from toolkits.rlt_vr.server import InferenceServer
from toolkits.rlt_vr.vr import SteamVRController


def test_online_gpu_guard_rejects_busy_device_and_pins_uuid(monkeypatch):
    from toolkits.rlt_vr.gpu_guard import isolate_gpu2

    monkeypatch.setattr(
        "subprocess.check_output", lambda *a, **kw: "GPU-test, 00000000:E1:00.0, 1200"
    )
    with pytest.raises(RuntimeError, match="busy"):
        isolate_gpu2()
    monkeypatch.setattr(
        "subprocess.check_output", lambda *a, **kw: "GPU-test, 00000000:E1:00.0, 4"
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "invalid-before-guard")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    assert isolate_gpu2() == {"uuid": "GPU-test", "render_backend": "pci:0000:e1:00.0"}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-test"


@pytest.fixture
def online_config():
    from pathlib import Path

    from omegaconf import OmegaConf

    config = OmegaConf.to_container(
        OmegaConf.load(Path(__file__).parents[2] / "toolkits/rlt_vr/online_smoke.yaml")
    )
    config.update(
        z_dim=8,
        batch_size=4,
        min_replay=2,
        publish_interval=1,
        actor_after_updates=1,
        checkpoint_interval=100,
    )
    return config


def online_features(config, value=0.0):
    import torch

    return {
        "z_rl": torch.full((1, config["z_dim"]), value, requires_grad=True),
        "proprio": torch.full((1, 9), value),
        "ref_chunk": torch.full((1, 10, 8), 0.25),
    }


def online_transition(config, *, human=True, terminated=False, truncated=False):
    return {
        "obs": online_features(config),
        "next_obs": online_features(config, 0.1),
        "action": [-0.25] * 8,
        "reward": 1.0,
        "human": human,
        "terminated": terminated,
        "truncated": truncated,
    }


def test_online_updates_publish_and_resume_optimizer_replay_rng(
    online_config, tmp_path
):
    import torch

    from toolkits.rlt_vr.online_learner import OnlineLearner

    learner = OnlineLearner(online_config)
    sample = online_transition(online_config)
    reference, source = learner.predict(sample["obs"])
    assert source == "reference" and reference.shape == (1, 8)
    before = {k: v.clone() for k, v in learner.model.state_dict().items()}
    for _ in range(4):
        learner.observe(sample)
    assert sample["obs"]["z_rl"].grad is None
    assert learner.status()["human_accepted"] == 4
    assert learner.status()["demo_sample_ratio"] == 0.5
    action, source = learner.predict(sample["obs"])
    assert source == "actor" and torch.isfinite(action).all()
    for prefix in ("backbone.", "q_head."):
        assert any(
            not torch.equal(v, before[k])
            for k, v in learner.model.state_dict().items()
            if k.startswith(prefix)
        )
    assert all(p.grad is None for p in learner.target.parameters())
    learner.save(tmp_path / "learner.pt", {"sequence": 3})
    expected = learner.observe(sample)
    resumed = OnlineLearner(online_config)
    assert resumed.load(tmp_path / "learner.pt") == {"sequence": 3}
    assert resumed.actor_optim.state and resumed.critic_optim.state
    assert resumed.observe(sample) == expected
    for key, value in learner.model.state_dict().items():
        torch.testing.assert_close(
            value, resumed.model.state_dict()[key], rtol=0, atol=0
        )
    resumed.load(tmp_path / "learner.pt")
    assert resumed.status()["replay_size"] == 4  # reload does not append twice


@pytest.mark.parametrize("truncated", [False, True])
def test_online_terminal_target_and_human_bc_use_executed_action(
    online_config, monkeypatch, truncated
):
    import torch
    import torch.nn.functional as functional

    from toolkits.rlt_vr.online_learner import OnlineLearner

    online_config["min_replay"] = 1
    learner = OnlineLearner(online_config)
    losses = []
    original = functional.mse_loss

    def mse(prediction, target, *args, **kwargs):
        losses.append(target.detach().clone())
        return original(prediction, target, *args, **kwargs)

    monkeypatch.setattr(functional, "mse_loss", mse)
    learner.observe(
        online_transition(online_config, terminated=not truncated, truncated=truncated)
    )
    torch.testing.assert_close(losses[0], torch.ones_like(losses[0]))
    torch.testing.assert_close(losses[1], torch.full_like(losses[1], -0.25))


@pytest.fixture
def online_rpc(online_config, tmp_path):
    from toolkits.rlt_vr.online_learner import OnlineLearner
    from toolkits.rlt_vr.online_service import OnlineService
    from toolkits.rlt_vr.online_transport import encode_observation

    learner = OnlineLearner(online_config)
    service = OnlineService(
        learner, lambda obs: online_features(online_config), tmp_path, "fixed-feature"
    )
    obs = {
        "state": np.zeros(9, np.float32),
        "main_image": np.zeros((384, 384, 3), np.uint8),
        "wrist_image": np.zeros((384, 384, 3), np.uint8),
    }
    payload = {
        "op": "observe",
        "session": "test-session",
        "sequence": 0,
        "episode": 0,
        "observation": encode_observation(obs),
        "next_observation": encode_observation(obs),
        "action": [0.0] * 8,
        "reward": 0,
        "human": True,
        "terminated": False,
        "truncated": False,
        "policy_version": -1,
    }
    service({"op": "begin", "session": "test-session"})
    return service, payload, obs


def test_online_protocol_order_duplicate_and_episode_boundaries(online_rpc):
    import copy

    service, payload, _ = online_rpc
    first = service(payload)
    assert first["metrics"]["accepted"] == 1
    assert service({**payload, "request_id": 33})["duplicate"]
    with pytest.raises(ValueError, match="Conflicting"):
        service({**payload, "human": False})
    with pytest.raises(ValueError, match="Out-of-order"):
        service({**payload, "sequence": 2})
    with pytest.raises(ValueError, match="session"):
        service({**payload, "session": "another-session"})
    bad = copy.deepcopy(payload)
    bad["sequence"] = 1
    bad["observation"]["state"][0] = 1
    with pytest.raises(ValueError, match="contiguous"):
        service(bad)
    service({**payload, "sequence": 1, "truncated": True})
    with pytest.raises(ValueError, match="contiguous"):
        service({**payload, "sequence": 2})
    result = service({**payload, "sequence": 2, "episode": 1})
    assert result["metrics"]["accepted"] == 3
    assert result["metrics"]["human_accepted"] == 3


def test_online_training_fault_is_fail_closed(online_rpc, monkeypatch):
    service, payload, _ = online_rpc

    def fail(item):
        raise RuntimeError("nonfinite gradient")

    monkeypatch.setattr(service.learner, "observe", fail)
    with pytest.raises(RuntimeError, match="nonfinite"):
        service(payload)
    assert service({"op": "health"})["faulted"]
    with pytest.raises(RuntimeError, match="faulted"):
        service(payload)
    with pytest.raises(RuntimeError, match="faulted"):
        service.checkpoint()


def test_online_budget_and_release_gate_survive_resume(online_config, tmp_path):
    from toolkits.rlt_vr.online_learner import OnlineLearner

    online_config.update(max_updates=2, actor_max_bc_loss=0.0)
    learner = OnlineLearner(online_config)
    sample = online_transition(online_config)
    for _ in range(5):
        learner.observe(sample)
    assert learner.status()["update_budget_exhausted"]
    assert learner.update_step == 2 and learner.accepted == 5
    assert learner.predict(sample["obs"])[1] == "reference"
    assert learner.status()["published_bc_loss"] > 0
    learner.save(tmp_path / "learner.pt", {})
    resumed = OnlineLearner(online_config)
    resumed.load(tmp_path / "learner.pt")
    assert resumed.status() == learner.status()
    assert resumed.predict(sample["obs"])[1] == "reference"


def test_online_report_counts_segments_without_calling_them_success(
    online_rpc, tmp_path
):
    from toolkits.rlt_vr.summarize_online import summarize

    service, payload, _ = online_rpc
    service(payload)
    service(payload)  # Retry must not become another human step.
    service({**payload, "sequence": 1, "human": False})
    service({**payload, "sequence": 2, "truncated": True})
    report = summarize(tmp_path / "metrics.jsonl")
    assert report["declared_human_steps"] == 2
    assert report["declared_human_segments"] == 2
    assert report["completed_episodes"] == 1
    assert report["success_among_completed"] == 0


def test_online_gpu_lease_is_exclusive_and_recoverable(tmp_path, monkeypatch):
    from toolkits.rlt_vr.gpu_guard import gpu2_lease

    original_open = open
    monkeypatch.setattr(
        "builtins.open", lambda path, mode: original_open(tmp_path / "gpu.lock", mode)
    )
    with gpu2_lease():
        with pytest.raises(RuntimeError, match="lease"):
            with gpu2_lease():
                pass
    with gpu2_lease():
        pass


@pytest.mark.parametrize(
    "key,value",
    [
        ("tau", float("nan")),
        ("demo_ratio", 2),
        ("batch_size", 1.5),
        ("bootstrap_truncation", "false"),
        ("typo", 1),
    ],
)
def test_online_config_rejects_invalid_before_cuda(online_config, key, value):
    from toolkits.rlt_vr.online_settings import validate_config

    with pytest.raises(ValueError):
        validate_config({**online_config, key: value})


def test_online_background_upload_uses_authenticated_real_socket(online_rpc):
    import time

    from toolkits.rlt_vr.online_transport import TransitionUploader

    service, _, obs = online_rpc
    token = "t" * 32
    with InferenceServer(
        ("127.0.0.1", 0), token, lambda obs: None, "test", dispatch=service
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        uploader = TransitionUploader(server.server_address[1], token, "test-session")
        try:
            for i in range(3):
                uploader.submit(
                    i,
                    obs,
                    obs,
                    episode=0,
                    action=[0.0] * 8,
                    reward=0,
                    human=i == 1,
                    terminated=False,
                    truncated=False,
                    policy_version=-1,
                )
            deadline = time.monotonic() + 20
            while (
                uploader.accepted_sequence < 2
                and not uploader.error
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert uploader.error is None and uploader.accepted_sequence == 2
            assert uploader.metrics["human_accepted"] == 1
            assert uploader.metrics["update_step"] == 2
        finally:
            uploader.close()
            server.shutdown()
            thread.join(timeout=10)


def test_intervention_discards_pending_and_remaining_actions():
    gate = ActionGate()
    gate.change("policy")
    old = gate.generation
    assert gate.accept(old, np.ones((10, 8)))
    np.testing.assert_array_equal(gate.next_action(), np.ones(8))
    gate.change("human")
    assert gate.next_action() is None
    assert not gate.accept(old, np.ones((10, 8)))
    gate.change("paused")
    gate.change("policy")
    assert gate.needs_prediction
    assert not gate.accept(old, np.ones((10, 8)))
    assert gate.accept(gate.generation, np.zeros((10, 8)))
    np.testing.assert_array_equal(gate.next_action(), np.zeros(8))


def test_reset_invalidates_even_already_paused_generation():
    gate = ActionGate()
    ticket = gate.generation
    gate.reset()
    gate.change("policy")
    assert not gate.accept(ticket, np.zeros((1, 8)))


@pytest.mark.parametrize("fault", ["tracking", "stall", "pause"])
def test_fault_requires_clutch_release(fault):
    operator = OperatorControl()
    assert operator.update(valid=True, clutch=True) == "human"
    assert (
        operator.update(
            valid=fault != "tracking",
            clutch=True,
            stalled=fault == "stall",
            command="pause" if fault == "pause" else "",
        )
        == "paused"
    )
    assert operator.update(valid=True, clutch=True) == "paused"
    assert operator.update(valid=True, clutch=False) == "paused"
    assert operator.update(valid=True, clutch=True) == "human"
    assert operator.update(valid=True, clutch=False) == "paused"
    assert operator.update(valid=True, clutch=False, command="policy") == "policy"


def test_finished_episode_requires_reset():
    operator = OperatorControl()
    operator.finish()
    for _ in range(3):
        assert operator.update(valid=True, clutch=False, command="policy") == "paused"
        assert operator.update(valid=True, clutch=True) == "paused"
    operator.reset()
    operator.update(valid=True, clutch=False)
    assert operator.update(valid=True, clutch=True) == "human"


@pytest.mark.parametrize(
    "bad", [np.zeros((8,)), np.zeros((11, 8)), [[float("nan")] * 8], [[2] * 8]]
)
def test_invalid_policy_actions_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_actions(bad)


def test_torch_ik_damped_step_is_finite():
    torch = pytest.importorskip("torch")

    from toolkits.rlt_vr.simulation import _TorchPandaIK

    class Transform:
        def get_matrix(self):
            return torch.eye(4).unsqueeze(0)

    class Chain:
        def forward_kinematics(self, qpos):
            assert qpos.shape == (1, 7)
            return Transform()

        def jacobian(self, qpos):
            assert qpos.shape == (1, 7)
            return torch.eye(6, 7).unsqueeze(0)

    solver = object.__new__(_TorchPandaIK)
    solver.torch = torch
    solver.pk = SimpleNamespace(
        matrix_to_axis_angle=lambda matrix: torch.zeros((len(matrix), 3))
    )
    solver.chain = Chain()
    target = np.eye(4, dtype=np.float32)
    target[:3, 3] = [0.01, -0.02, 0.03]

    delta = solver.joint_delta(np.zeros(9, dtype=np.float32), target)

    assert delta is not None and np.isfinite(delta).all()
    np.testing.assert_allclose(delta[:3], target[:3, 3], atol=5e-6)
    np.testing.assert_allclose(delta[3:], 0, atol=1e-7)


def test_image_codec_preserves_baseline_pixels():
    image = np.random.default_rng(0).integers(0, 256, (384, 384, 3), dtype=np.uint8)
    np.testing.assert_array_equal(decode_image(encode_image(image)), image)
    with pytest.raises(ValueError):
        encode_image(image[:100])
    with pytest.raises(ValueError):
        decode_image("invalid-base64")


def test_bounded_frames_and_truncated_connections():
    left, right = socket.socketpair()
    try:
        send(left, {"ok": True})
        assert receive(right) == {"ok": True}
        left.sendall(struct.pack("!I", MAX_MESSAGE + 1))
        with pytest.raises(ValueError):
            receive(right)
        left.sendall(struct.pack("!I", 20) + b"short")
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(ConnectionError):
            receive(right)
    finally:
        left.close()
        right.close()


def test_relative_pose_and_motion_bounds():
    anchor = np.eye(4)
    current = anchor.copy()
    current[0, 3] = 0.1
    result = relative_target(anchor, current, anchor)
    np.testing.assert_allclose(result[:3, 3], [0, -0.05, 0])
    current[0, 3] = 10
    assert np.linalg.norm(
        relative_target(anchor, current, anchor)[:3, 3]
    ) == pytest.approx(0.15)
    np.testing.assert_allclose(relative_target(anchor, anchor, anchor), anchor)
    mapped = map_relative_target(anchor, current, anchor)
    assert mapped.translation_limited
    assert not mapped.rotation_limited
    assert mapped.requested_translation == pytest.approx(5.0)
    assert mapped.applied_translation == pytest.approx(0.15)


@pytest.fixture
def rpc_server():
    token = "test-token-" * 4
    calls = []

    def predict(observation):
        calls.append(observation)
        return np.zeros((10, 8))

    with InferenceServer(("127.0.0.1", 0), token, predict, "test-policy") as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_address[1], token, calls
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_rpc_auth_contract_and_round_trip(rpc_server):
    port, token, calls = rpc_server
    response = request("127.0.0.1", port, token, {"op": "health", "request_id": 1}, 2)
    assert response["contract"] == CONTRACT
    image = encode_image(np.zeros((384, 384, 3), np.uint8))
    payload = {
        "op": "predict",
        "request_id": 2,
        "contract": CONTRACT,
        "state": [0] * 9,
        "main_image": image,
        "wrist_image": image,
    }
    with pytest.raises(RuntimeError):
        request("127.0.0.1", port, "wrong", payload, 2)
    with pytest.raises(RuntimeError):
        request("127.0.0.1", port, token, {**payload, "contract": {}}, 2)
    assert not calls
    response = request("127.0.0.1", port, token, payload, 2)
    assert len(calls) == 1
    assert validate_actions(response["actions"]).shape == (10, 8)


def test_late_rpc_reply_cannot_undo_local_takeover():
    from concurrent.futures import ThreadPoolExecutor

    entered, release = threading.Event(), threading.Event()

    def predict(observation):
        entered.set()
        if not release.wait(timeout=3):
            raise TimeoutError("Test did not release inference")
        return np.ones((10, 8))

    token = "test-token-" * 4
    gate = ActionGate()
    gate.change("policy")
    ticket = gate.generation
    image = encode_image(np.zeros((384, 384, 3), np.uint8))
    with InferenceServer(("127.0.0.1", 0), token, predict, "delayed") as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    request,
                    "127.0.0.1",
                    server.server_address[1],
                    token,
                    {
                        "op": "predict",
                        "request_id": 1,
                        "contract": CONTRACT,
                        "state": [0] * 9,
                        "main_image": image,
                        "wrist_image": image,
                    },
                    5,
                )
                assert entered.wait(timeout=2)
                gate.change("human")
                release.set()
                assert not gate.accept(ticket, result.result(timeout=3)["actions"])
                assert gate.next_action() is None
        finally:
            release.set()
            server.shutdown()
            thread.join(timeout=2)


def test_record_only_executed_actions_without_pickle(tmp_path):
    recorder = TransitionRecorder(tmp_path / "run", {"model_id": "test"})
    obs = {"state": np.zeros(9), "main_image": np.zeros((384, 384, 3), np.uint8)}
    recorder.append(obs, np.ones(8), obs, 0, False, False, "human", 0, "test")
    with np.load(tmp_path / "run/step_00000000.npz", allow_pickle=False) as item:
        assert item["human_intervention"].item()
        assert item["source"].item() == "human"
        np.testing.assert_array_equal(item["action"], np.ones(8))
    with pytest.raises(FileExistsError):
        TransitionRecorder(tmp_path / "run", {})


def test_vendor_tracking_loss_and_button_mapping(monkeypatch):
    import sys

    pose = SimpleNamespace(
        mDeviceToAbsoluteTracking=SimpleNamespace(m=np.eye(4)[:3]),
        bPoseIsValid=True,
        bDeviceIsConnected=True,
        eTrackingResult=200,
    )
    system = SimpleNamespace(
        getTrackedDeviceIndexForControllerRole=lambda role: 1,
        getControllerStateWithPose=lambda origin, index: (
            True,
            SimpleNamespace(ulButtonPressed=(1 << 2) | (1 << 33)),
            pose,
        ),
        isInputAvailable=lambda: True,
    )
    calls = []
    sdk = SimpleNamespace(
        init=lambda kind: system,
        shutdown=lambda: calls.append("closed"),
        VRApplication_Background=3,
        TrackedControllerRole_RightHand=2,
        k_unTrackedDeviceIndexInvalid=0xFFFFFFFF,
        TrackingUniverseStanding=1,
        TrackingResult_Running_OK=200,
    )
    monkeypatch.setitem(sys.modules, "openvr", sdk)
    controller = SteamVRController()
    assert controller.read().valid
    assert controller.read().clutch and controller.read().close_gripper
    pose.bPoseIsValid = False
    assert not controller.read().valid
    controller.close()
    controller.close()
    assert calls == ["closed"]


def test_vendor_analog_trigger_mapping(monkeypatch):
    import sys

    pose = SimpleNamespace(
        mDeviceToAbsoluteTracking=SimpleNamespace(m=np.eye(4)[:3]),
        bPoseIsValid=True,
        bDeviceIsConnected=True,
        eTrackingResult=200,
    )
    axes = [SimpleNamespace(x=0.0) for _ in range(5)]
    axes[1].x = 0.75
    state = SimpleNamespace(ulButtonPressed=0, rAxis=axes)
    system = SimpleNamespace(
        getTrackedDeviceIndexForControllerRole=lambda role: 1,
        getControllerStateWithPose=lambda origin, index: (True, state, pose),
        getInt32TrackedDeviceProperty=lambda index, prop: 7 if prop == 101 else 0,
        isInputAvailable=lambda: True,
    )
    sdk = SimpleNamespace(
        init=lambda kind: system,
        shutdown=lambda: None,
        VRApplication_Background=3,
        TrackedControllerRole_RightHand=2,
        k_unTrackedDeviceIndexInvalid=0xFFFFFFFF,
        TrackingUniverseStanding=1,
        TrackingResult_Running_OK=200,
        k_unControllerStateAxisCount=5,
        Prop_Axis0Type_Int32=100,
        k_eControllerAxis_Trigger=7,
    )
    monkeypatch.setitem(sys.modules, "openvr", sdk)

    reading = SteamVRController(trigger_threshold=0.6).read()

    assert reading.close_gripper
    assert reading.trigger_value == pytest.approx(0.75)


@pytest.mark.skipif(
    os.environ.get("RLT_VR_SIM_TEST") != "1",
    reason="Requires ManiSkill/SAPIEN/Vulkan; enable on an explicitly selected free GPU",
)
def test_cpu_simulation_ik_reset_and_step():
    from toolkits.rlt_vr.simulation import LocalSimulation

    env = LocalSimulation(os.environ.get("RLT_VR_RENDER_BACKEND", "gpu"))
    try:
        assert env.observation()["state"].shape == (9,)
        action = env.human_action(env.tcp_matrix(), 1)
        assert action is not None
        np.testing.assert_allclose(action[:7], 0, atol=1e-3)
        target = env.tcp_matrix()
        target[2, 3] += 0.005
        moved = env.human_action(target, 1)
        assert moved is not None
        assert np.any(np.abs(moved[:7]) > 1e-4)
        assert np.max(np.abs(moved[:7])) <= 0.25
        observation, reward, _, _ = env.step(action)
        assert observation["main_image"].shape == (384, 384, 3)
        assert np.isfinite(reward)
        assert env.reset(1)["wrist_image"].shape == (384, 384, 3)
    finally:
        env.close()


@pytest.mark.skipif(
    not os.environ.get("RLT_VR_STAGE1"),
    reason="Requires a trusted Stage1 export, norms and a free GPU",
)
def test_checkpoint_inference_over_rpc():
    from pathlib import Path

    from toolkits.rlt_vr.server import RLTInference
    from toolkits.rlt_vr.simulation import LocalSimulation

    actor = os.environ.get("RLT_VR_ACTOR")
    model = RLTInference(
        Path("toolkits/rlt_vr/model.yaml"),
        Path(os.environ["RLT_VR_STAGE1"]),
        Path(os.environ["RLT_VR_DATASET"]),
        Path(actor) if actor else None,
    )
    env = LocalSimulation(os.environ.get("RLT_VR_RENDER_BACKEND", "gpu"))
    try:
        with InferenceServer(
            ("127.0.0.1", 0), "test-token-" * 4, model, "checkpoint-test"
        ) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                observation = env.observation()
                payload = {
                    "op": "predict",
                    "request_id": 1,
                    "contract": CONTRACT,
                    "state": observation["state"].tolist(),
                    **{
                        k: encode_image(observation[k])
                        for k in ("main_image", "wrist_image")
                    },
                }
                response = request(
                    "127.0.0.1",
                    server.server_address[1],
                    "test-token-" * 4,
                    payload,
                    30,
                )
                actions = validate_actions(response["actions"])
                assert actions.shape == (10, 8)
                assert response["inference_ms"] > 0
                _, reward, _, _ = env.step(actions[0])
                assert np.isfinite(reward)
            finally:
                server.shutdown()
                thread.join(timeout=2)
    finally:
        env.close()
