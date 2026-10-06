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


def wait_for_processed(transport, count, timeout=10):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = transport({"op": "status"})
        assert not status["faulted"], status
        if status["metrics"]["accepted"] >= count and not status["pending_learning"]:
            return status
        time.sleep(0.01)
    pytest.fail("Learner did not process durable receipts")


def test_pause_reason_survives_tracking_recovery_and_finished_episode():
    control = OperatorControl()
    control.update(valid=True, clutch=True)
    control.pause("local_outbox_full", "wait for receipts")
    assert control.update(valid=True, clutch=True) == "paused"
    assert "local_outbox_full" in control.instruction
    assert "release grip" in control.instruction
    control.update(valid=True, clutch=False)
    assert "local_outbox_full" in control.instruction
    assert control.update(valid=True, clutch=True) == "human"
    control.finish("1000-step time limit")
    control.pause("review_required", "Y/N")
    assert "1000-step time limit" in control.instruction
    assert "R resets" in control.instruction


def test_target_filter_and_joint_limits_bound_commands():
    from scipy.spatial.transform import Rotation

    from toolkits.rlt_vr.control import TargetFilter, bounded_joint_delta

    smoother = TargetFilter(np.eye(4))
    target = np.eye(4)
    target[0, 3] = 0.15
    target[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    previous = np.eye(4)
    for _ in range(20):
        pose = smoother.update(target)
        assert 0 <= pose[0, 3] <= target[0, 3]
        assert np.linalg.norm(pose[:3, 3] - previous[:3, 3]) <= 0.012 + 1e-8
        assert (
            Rotation.from_matrix(pose[:3, :3] @ previous[:3, :3].T).magnitude()
            <= 0.08 + 1e-8
        )
        previous = pose
    q = np.zeros(7)
    limits = np.tile([-1.0, 1.0], (7, 1))
    first, _ = bounded_joint_delta(np.ones(7), q, limits, np.zeros(7))
    assert np.max(first) <= 0.0100001
    second, _ = bounded_joint_delta(np.ones(7), q, limits, first)
    assert np.max(second - first) <= 0.0100001
    q[0] = 0.99
    clipped, limited = bounded_joint_delta(np.ones(7), q, limits, second)
    assert limited and clipped[0] == 0
    inward, _ = bounded_joint_delta(-np.ones(7), q, limits, np.zeros(7))
    assert inward[0] < 0
    with pytest.raises(ValueError):
        TargetFilter(np.eye(4), speed=float("nan"))


def test_raw_journal_needs_explicit_review_before_demo_upload(tmp_path):
    import json

    recorder = TransitionRecorder(tmp_path / "records", {})
    image = np.zeros((384, 384, 3), np.uint8)
    obs = {"state": np.zeros(9, np.float32), "main_image": image, "wrist_image": image}
    path = recorder.append(obs, np.zeros(8), obs, 0, False, False, "human", 0, "v0")
    assert recorder.pending_review == 1 and recorder.upload_item(path) is None
    recorder.review_pending(True)
    assert recorder.upload_item(path)["quality"] == "approved"
    assert (
        json.loads((recorder.directory / "review_000000.json").read_text())["quality"]
        == "approved"
    )
    path2 = recorder.append(obs, np.zeros(8), obs, 0, False, False, "human", 0, "v0")
    recorder.review_pending(None)
    assert recorder.upload_item(path2)["quality"] == "unreviewed"
    with np.load(path2, allow_pickle=False) as raw:
        assert raw["quality"].item() == "unreviewed"
    small = TransitionRecorder(tmp_path / "small", {}, max_bytes=10)
    assert not small.ready


@pytest.mark.parametrize("fault", ["tracking", "outbox"])
def test_client_loop_latches_reason_reviews_and_reanchors(monkeypatch, tmp_path, fault):
    """Exercise the real UI loop with only external device/network edges faked."""
    import sys

    from toolkits.rlt_vr import client

    frame = {"index": -1, "clock": 0.0}
    rendered, executed, anchors = [], [], []
    keys = {4: ord("y"), 8: ord("n"), 9: ord("q")}
    image = np.zeros((384, 384, 3), np.uint8)
    obs = {"state": np.zeros(9, np.float32), "main_image": image, "wrist_image": image}

    class Device:
        def __init__(self, *args):
            pass

        def read(self):
            frame["index"] += 1
            i = frame["index"]
            assert i < 11, "Client did not terminate"
            return SimpleNamespace(
                valid=not (fault == "tracking" and i == 1),
                clutch=i in (0, 1, 2, 6, 7),
                pose=np.eye(4),
                close_gripper=False,
                buttons=4,
                trigger_value=0.0,
            )

        def close(self):
            pass

    class Simulator:
        def __init__(self, *args):
            self.teleop_diagnostics = {}

        def observation(self):
            return obs

        def tcp_matrix(self):
            return np.eye(4)

        def reset_teleop(self):
            anchors.append(frame["index"])

        def human_action(self, target, gripper):
            return np.zeros(8, np.float32)

        def step(self, action):
            executed.append(frame["index"])
            return obs, 0.0, False, False

        def close(self):
            pass

    class Uploader:
        error = None
        outstanding = server_pending = 0
        accepted_sequence = processed_sequence = -1
        storage_full = False
        metrics = {}

        def __init__(self, *args, **kwargs):
            pass

        @property
        def ready(self):
            return not (fault == "outbox" and frame["index"] == 1)

        def submit_path(self, path):
            assert path.is_file()

        def close(self):
            pass

    def clock():
        frame["clock"] += 0.11
        return frame["clock"]

    sdk = SimpleNamespace(
        COLOR_RGB2BGR=0,
        WND_PROP_VISIBLE=0,
        cvtColor=lambda panel, mode: panel,
        putText=lambda panel, text, *a: rendered.append((frame["index"], text)),
        imshow=lambda *a: None,
        waitKey=lambda delay: keys.get(frame["index"], -1),
        getWindowProperty=lambda *a: 1,
        destroyAllWindows=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "cv2", sdk)
    monkeypatch.setattr(
        client,
        "time",
        SimpleNamespace(monotonic=clock, time=lambda: 0.0, sleep=lambda delay: None),
    )
    monkeypatch.setattr(client, "LocalSimulation", Simulator)
    monkeypatch.setattr(client, "SteamVRController", Device)
    monkeypatch.setattr(client, "TransitionUploader", Uploader)
    monkeypatch.setattr(
        client,
        "request",
        lambda *a: {
            "contract": CONTRACT,
            "model_id": "test",
            "online": True,
            "horizon": 1,
            "online_protocol": 2,
        },
    )
    args = SimpleNamespace(
        manual_only=False,
        online=True,
        no_vr=False,
        port=12345,
        record=tmp_path / "record",
        seed=0,
        yaw_degrees=0,
        max_episode_steps=1000,
        render_backend="cpu",
        clutch_button=2,
        trigger_button=33,
        trigger_threshold=0.6,
        translation_scale=0.5,
        max_displacement=0.15,
        max_rotation_degrees=30,
        stall_timeout=10,
        log_interval=10,
        reply_ttl=10,
    )
    client.run(args)
    assert executed == [0, 6, 7]
    assert anchors == [0, 6]
    reason = "tracking_invalid" if fault == "tracking" else "local_outbox_full"
    assert any(i == 2 and reason in text for i, text in rendered)
    assert any(i == 3 and "release grip" in text for i, text in rendered)
    import json

    reviews = [
        json.loads(p.read_text()) for p in sorted(args.record.glob("review_*.json"))
    ]
    assert [r["quality"] for r in reviews] == ["approved", "rejected"]


def test_reviewed_disk_outbox_reaches_async_learner_in_order(online_rpc, tmp_path):
    import time

    from toolkits.rlt_vr.async_service import AsyncOnlineService
    from toolkits.rlt_vr.online_transport import TransitionUploader
    from toolkits.rlt_vr.server import ConcurrentInferenceServer

    core, _, obs = online_rpc
    transport = AsyncOnlineService(core, reserve_bytes=0)
    session, token = "disk-test-session", "test-secret" * 4
    recorder = TransitionRecorder(tmp_path / "windows", {})
    transport({"op": "begin", "session": session, "online_protocol": 2})
    with ConcurrentInferenceServer(
        ("127.0.0.1", 0), token, None, "test", dispatch=transport
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        uploader = TransitionUploader(
            server.server_address[1], token, session, capacity=2, recorder=recorder
        )
        try:
            first = recorder.append(
                obs, np.zeros(8), obs, 0, False, False, "human", 0, "test"
            )
            uploader.submit_path(first)
            second = recorder.append(
                obs,
                np.zeros(8),
                obs,
                0,
                False,
                False,
                "policy",
                0,
                "test",
                policy_source="reference",
            )
            uploader.submit_path(second)
            assert not uploader.ready
            assert (
                core.learner.accepted == 0
            )  # Review is a gate, not an implicit approval.
            recorder.review_pending(False)
            wait_for_processed(transport, 2)
            deadline = time.monotonic() + 2
            while uploader.outstanding and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (
                uploader.ready
                and uploader.accepted_sequence == 1
                and not uploader.error
            )
            assert core.learner.human_accepted == 1 and not core.learner.demos
            assert first.is_file() and second.is_file()
        finally:
            uploader.close()
            server.shutdown()
            thread.join(5)
            transport.close()


@pytest.mark.parametrize("quality", ["unreviewed", "rejected"])
def test_unapproved_human_is_critic_data_not_bc(online_config, quality):
    from toolkits.rlt_vr.online_learner import OnlineLearner

    online_config.update(min_replay=1, q_weight=0.0, actor_max_bc_loss=0.01)
    learner = OnlineLearner(online_config)
    sample = {**online_transition(online_config), "quality": quality}
    status = learner.observe(sample)
    assert status["human_accepted"] == 1 and status["approved_accepted"] == 0
    assert status["replay_size"] == 1 and status["demo_size"] == 0
    assert status["bc_loss"] == 0 and status["bc_eligible_ratio"] == 0
    assert status["published_bc_loss"] is None and not status["actor_ready"]


def test_durable_ack_and_health_do_not_wait_for_feature_extraction(online_rpc):
    from toolkits.rlt_vr.async_service import AsyncOnlineService
    from toolkits.rlt_vr.online_service import ONLINE_PROTOCOL
    from toolkits.rlt_vr.server import ConcurrentInferenceServer

    core, payload, _ = online_rpc
    entered, release = threading.Event(), threading.Event()
    original = core.extract

    def delayed(obs):
        entered.set()
        assert release.wait(10)
        return original(obs)

    core.extract = delayed
    transport = AsyncOnlineService(core, max_pending=1, reserve_bytes=0)
    token = "test-secret" * 4
    with ConcurrentInferenceServer(
        ("127.0.0.1", 0), token, None, "test", dispatch=transport
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def rpc(message):
            return request(
                "127.0.0.1",
                server.server_address[1],
                token,
                {"request_id": 0, **message},
                2,
            )

        try:
            rpc(
                {
                    "op": "begin",
                    "session": "test-session",
                    "online_protocol": ONLINE_PROTOCOL,
                }
            )
            ack = rpc(payload)
            assert ack["received_sequence"] == 0 and ack["metrics"]["accepted"] == 0
            assert entered.wait(2)
            assert rpc({"op": "health"})["pending_learning"] == 1
            assert rpc(payload)["duplicate"]
            assert rpc({**payload, "sequence": 1})["busy"]
            with pytest.raises(RuntimeError):
                rpc({**payload, "quality": "rejected"})
            assert len(list((core.directory / "inbox").glob("*.json"))) == 1
            release.set()
            wait_for_processed(transport, 1)
        finally:
            release.set()
            server.shutdown()
            thread.join(5)
            transport.close()


def test_pending_durable_receipts_survive_worker_fault_and_resume(
    online_rpc, online_config, tmp_path
):
    import time

    from toolkits.rlt_vr.async_service import AsyncOnlineService
    from toolkits.rlt_vr.online_learner import OnlineLearner
    from toolkits.rlt_vr.online_service import ONLINE_PROTOCOL, OnlineService

    core, payload, _ = online_rpc

    def failed_extract(obs):
        raise RuntimeError("simulated feature service failure")

    core.extract = failed_extract
    transport = AsyncOnlineService(core, reserve_bytes=0)
    transport(
        {"op": "begin", "session": "test-session", "online_protocol": ONLINE_PROTOCOL}
    )
    assert transport(payload)["received_sequence"] == 0
    deadline = time.monotonic() + 5
    while not transport({"op": "status"})["faulted"] and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match="worker_"):
        transport.close()
    learner = OnlineLearner(online_config)
    metadata = learner.load(core.directory / "learner.pt")
    assert learner.accepted == 0  # The receipt was durable, but not learned.
    directory = tmp_path / "resumed"
    directory.mkdir()
    resumed_core = OnlineService(
        learner, lambda obs: online_features(online_config), directory, "fixed-feature"
    )
    resumed_core.restore(metadata)
    resumed = AsyncOnlineService(resumed_core, restored=metadata, reserve_bytes=0)
    try:
        status = wait_for_processed(resumed, 1)
        assert status["metrics"]["approved_accepted"] == 1
        assert not resumed(
            {
                "op": "begin",
                "session": "next-session",
                "online_protocol": ONLINE_PROTOCOL,
            }
        )["busy"]
        resumed({**payload, "session": "next-session"})
        wait_for_processed(resumed, 2)
    finally:
        resumed.close()
    assert len(list((core.directory / "inbox").glob("*.json"))) == 1


def test_async_quota_and_protocol_reject_without_claiming_records(online_rpc):
    from toolkits.rlt_vr.async_service import AsyncOnlineService

    core, payload, _ = online_rpc
    transport = AsyncOnlineService(core, max_journal_bytes=1, reserve_bytes=0)
    try:
        with pytest.raises(ValueError, match="protocol"):
            transport({"op": "begin", "session": "test-session"})
        transport({"op": "begin", "session": "test-session", "online_protocol": 2})
        response = transport(payload)
        assert (
            response["busy"]
            and response["storage_full"]
            and response["received_sequence"] == -1
        )
        assert not list((core.directory / "inbox").glob("*.json"))
    finally:
        transport.close()


def test_async_reference_correction_actor_and_return_to_human(online_rpc):
    from toolkits.rlt_vr.async_service import AsyncOnlineService
    from toolkits.rlt_vr.online_transport import encode_observation
    from toolkits.rlt_vr.summarize_online import summarize

    core, payload, obs = online_rpc
    transport = AsyncOnlineService(core, reserve_bytes=0)
    try:
        transport({"op": "begin", "session": "test-session", "online_protocol": 2})
        predict = {
            "op": "predict",
            "session": "test-session",
            **encode_observation(obs),
        }
        assert transport(predict)["policy_source"] == "reference"
        transport(
            {
                **payload,
                "human": False,
                "quality": "policy",
                "policy_source": "reference",
            }
        )
        transport({**payload, "sequence": 1})
        wait_for_processed(transport, 2)
        response = transport(predict)
        assert response["policy_source"] == "actor"
        transport(
            {
                **payload,
                "sequence": 2,
                "human": False,
                "quality": "policy",
                "policy_source": "actor",
                "action": response["actions"][0],
                "policy_version": response["metrics"]["policy_version"],
            }
        )
        transport({**payload, "sequence": 3})
        status = wait_for_processed(transport, 4)
        assert status["metrics"]["human_accepted"] == 2
        assert status["metrics"]["approved_accepted"] == 2
        report = summarize(core.directory / "metrics.jsonl")
        assert report["declared_reference_correction_actor_retakeover"]
        assert report["actor_executed_steps"] == 1
        assert not transport({"op": "end", "session": "test-session"})["busy"]
        new_session = transport(
            {"op": "begin", "session": "another-session", "online_protocol": 2}
        )
        assert not new_session["busy"] and new_session["sequence"] == -1
    finally:
        transport.close()


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
        "quality": "approved" if human else "policy",
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
        "quality": "approved",
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
        service({**payload, "human": False, "quality": "policy"})
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
    service({**payload, "sequence": 1, "human": False, "quality": "policy"})
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


def test_panda_urdf_following_with_smoothed_bounded_commands():
    """Check actual Panda kinematics without claiming physics/render acceptance."""
    import importlib.util
    from pathlib import Path

    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_kinematics")
    spec = importlib.util.find_spec("mani_skill")
    if spec is None:
        pytest.skip("Optional ManiSkill Panda URDF is unavailable")
    from toolkits.rlt_vr.control import TargetFilter, bounded_joint_delta
    from toolkits.rlt_vr.simulation import _TorchPandaIK

    path = Path(spec.origin).parent / "assets/robots/panda/panda_v2.urdf"
    if not path.is_file():
        pytest.skip("Optional Panda URDF is unavailable")
    solver = _TorchPandaIK(str(path), "panda_hand_tcp")
    q = np.array([0, np.pi / 8, 0, -5 * np.pi / 8, 0, 3 * np.pi / 4, np.pi / 4])
    limits = np.array(solver.chain.get_joint_limits()).T

    def fk():
        return (
            solver.chain.forward_kinematics(torch.tensor(q, dtype=torch.float32)[None])
            .get_matrix()[0]
            .numpy()
        )

    initial = fk()
    target = initial.copy()
    target[:3, 3] += [0.01, -0.01, 0.02]
    smoother = TargetFilter(initial)
    previous = np.zeros(7)
    for _ in range(60):
        dq = solver.joint_delta(q, smoother.update(target))
        assert dq is not None
        delta, _ = bounded_joint_delta(dq, q, limits, previous)
        assert np.max(np.abs(delta)) <= 0.0250001
        assert np.max(np.abs(delta - previous)) <= 0.0100001
        q += delta
        previous = delta
    assert np.linalg.norm(fk()[:3, 3] - target[:3, 3]) < 0.0005


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
