# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Published LIBERO reproduction protocol, independent of model installation."""

import pytest

from toolkits.rlt.libero_audit import compare_records, failed_cases, observation_record
from toolkits.rlt.libero_environment_audit import (
    compare_preprocessing,
    dependency_conflicts,
    git_blob_hash,
    recorded_episode,
)
from toolkits.rlt.libero_reproduction import (
    ReferenceActor,
    ResizedVLA,
    episode_plan,
    paired_summary,
    summarize_update_budget,
    training_arguments,
    validate_training_history,
)


def test_preprocessing_report_separates_pilot_and_rejects_unmatched_runs(tmp_path):
    import json

    base, resized = tmp_path / "base", tmp_path / "resized"
    manifest = {
        "complete": True,
        "mode": "evaluate",
        "assistance": False,
        "learner_dir": None,
        "source_revision": "fixed",
        "assets": {},
        "tasks": [5],
        "states": [0, 3],
        "seed": 42,
        "variant": "rlt_a",
        "runtime_versions": {"mujoco": "3.8.1"},
    }
    for path, size in ((base, None), (resized, 224)):
        path.mkdir()
        (path / "manifest.json").write_text(
            json.dumps({**manifest, "input_image_size": size})
        )
        rows = [
            {
                "task": 5,
                "state": state,
                "success": bool(size) if state == 0 else not bool(size),
            }
            for state in (0, 3)
        ]
        for arm in ("reference", "rlt_a"):
            (path / f"{arm}.json").write_text(json.dumps(rows))
    result = compare_preprocessing(base, resized)["arms"]["rlt_a"]
    assert result["all"]["new_successes"] == result["all"]["lost_successes"] == 1
    assert result["inspected_pilot"]["resized_success"] == 1
    assert result["outside_pilot"]["resized_success"] == 0
    (resized / "manifest.json").write_text(
        json.dumps({**manifest, "input_image_size": 224, "seed": 43})
    )
    with pytest.raises(ValueError, match="seed"):
        compare_preprocessing(base, resized)
    (resized / "manifest.json").write_text(
        json.dumps({**manifest, "input_image_size": 224})
    )
    (resized / "rlt_a.json").write_text("[]")
    with pytest.raises(ValueError, match="incomplete"):
        compare_preprocessing(base, resized)


@pytest.mark.parametrize(
    "tasks,states",
    [([], [0]), ([0, 0], [0]), ([10], [0]), ([0], [50]), ([0], [1, 1]), ([0], [-1])],
)
def test_reject_duplicate_or_wrapped_initial_states(tasks, states):
    with pytest.raises(ValueError):
        episode_plan(tasks, states)


def test_plan_has_exact_task_state_pairs():
    assert episode_plan([0, 3], [20, 21]) == [(0, 20), (0, 21), (3, 20), (3, 21)]


def test_paired_outcomes_not_just_aggregate_success():
    a = [{"task": 0, "state": i, "success": i % 2 == 0} for i in range(4)]
    b = [{"task": 0, "state": i, "success": i < 2} for i in range(4)]
    summary = paired_summary(a, b)
    assert summary["reference_success"] == summary["rlt_a_success"] == 0.5
    assert summary["reference_only_successes"] == summary["rlt_a_only_successes"] == 1
    with pytest.raises(ValueError):
        paired_summary(a, b[:-1])
    with pytest.raises(ValueError):
        paired_summary(a + a[:1], b)


def test_reference_arm_preserves_exact_proposal():
    import torch

    proposal = torch.linspace(-1, 1, 56, dtype=torch.bfloat16).reshape(1, 8, 7)
    action, _ = ReferenceActor()(None, proposal, None, deterministic=True)
    assert action.dtype == torch.float32
    assert torch.equal(action.to(proposal.dtype), proposal)
    assert action.numpy().shape == (1, 8, 7)


def test_image_resize_diagnostic_preserves_views_and_model_outputs():
    import cv2
    import numpy as np
    from PIL import Image

    class VLABoundary:
        def eval(self):
            return self

        def get_vla_action(self, *, batch_images, instructions):
            return batch_images, instructions

    pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    image = Image.fromarray(pixels)
    model = ResizedVLA(VLABoundary(), 16).eval()
    images, tasks = model.get_vla_action(
        batch_images=[[image, image]], instructions=["task"]
    )
    assert tasks == ["task"] and len(images[0]) == 2
    expected = cv2.resize(pixels, (16, 16), interpolation=cv2.INTER_AREA)
    for view in images[0]:
        np.testing.assert_array_equal(view, expected)
    np.testing.assert_array_equal(image, pixels)
    with pytest.raises(ValueError, match="positive"):
        ResizedVLA(VLABoundary(), 0)


def test_audit_selects_failed_pairs_without_cross_product_or_duplicates(tmp_path):
    import json

    from toolkits.rlt.libero_reproduction import ASSETS

    manifest = {
        "complete": True,
        "mode": "evaluate",
        "assets": ASSETS,
        "tasks": [5, 6, 9],
        "states": [0, 1],
        "learner_dir": None,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows = [
        {"task": t, "state": s, "success": s == t % 2}
        for t in manifest["tasks"]
        for s in manifest["states"]
    ]
    (tmp_path / "rlt_a.json").write_text(json.dumps(rows))
    assert failed_cases(tmp_path, [5, 6, 9]) == {5: [0], 6: [1], 9: [0]}
    (tmp_path / "rlt_a.json").write_text(json.dumps(rows + rows[:1]))
    with pytest.raises(ValueError, match="duplicate"):
        failed_cases(tmp_path, [5, 6, 9])
    (tmp_path / "rlt_a.json").write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError, match="Missing"):
        failed_cases(tmp_path, [5, 6, 9])
    manifest["learner_dir"] = "/a/different/checkpoint"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="same pinned"):
        failed_cases(tmp_path, [5, 6, 9])


def test_audit_detects_input_divergence_even_when_actions_match():
    import copy

    import numpy as np

    obs = {
        "state": np.zeros(8),
        "primary_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    left = [
        {"kind": "reset", "observation": observation_record(obs)},
        {"kind": "step", "action": [0.0] * 7},
    ]
    right = copy.deepcopy(left)
    assert compare_records(left, right)["exact_match"]
    obs["primary_image"][0, 0, 0] = 1
    right[0]["observation"] = observation_record(obs)
    report = compare_records(left, right)
    assert not report["exact_match"]
    assert report["first_difference"]["record"] == 0
    assert report["max_executed_action_abs_diff_common_prefix"] == 0.0


def test_audit_detects_action_and_trace_length_mismatches():
    left = [{"kind": "step", "action": [0.0] * 7}]
    right = [{"kind": "step", "action": [0.5] * 7}]
    report = compare_records(left, right)
    assert report["max_executed_action_abs_diff_common_prefix"] == 0.5
    assert report["first_difference"]["fields"] == ["action"]
    assert not compare_records(left, left + left)["exact_match"]


def test_environment_audit_checks_active_dependency_constraints(tmp_path, monkeypatch):
    """Read actual wheel metadata from a temporary interpreter search path."""
    for name, version, requirements in [
        ("audit_sim", "0.1", ["audit_engine>=3.0,<3.4", "absent; extra == 'train'"]),
        ("audit_engine", "3.8.1", []),
    ]:
        package = tmp_path / f"{name}-{version}.dist-info"
        package.mkdir()
        lines = [f"Name: {name}", f"Version: {version}"]
        lines.extend(f"Requires-Dist: {r}" for r in requirements)
        (package / "METADATA").write_text("\n".join(lines) + "\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert dependency_conflicts("audit_sim") == [
        {"requirement": "audit_engine>=3.0,<3.4", "installed": "3.8.1"}
    ]


def test_environment_audit_git_hash_matches_git(tmp_path):
    import subprocess

    data = tmp_path / "binary.dat"
    data.write_bytes(b"checkpoint\0metadata\n")
    expected = subprocess.check_output(
        ["git", "hash-object", str(data)], text=True
    ).strip()
    assert git_blob_hash(data) == expected


def test_environment_replay_requires_complete_finite_actions(tmp_path):
    import json

    trace = tmp_path / "trace.jsonl"
    rows = [
        {"kind": "reset", "task": 9, "initial_state": 7, "seed": 42},
        {"kind": "step", "tick": 0, "action": [0.0] * 7},
        {"kind": "close", "ticks": 1},
    ]

    def write():
        trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    write()
    assert recorded_episode(trace) == (rows[0], [rows[1]])
    rows[1]["action"][0] = float("nan")
    write()
    with pytest.raises(ValueError, match="finite"):
        recorded_episode(trace)
    rows[1]["action"][0] = 0.0
    rows[1]["tick"] = 1
    write()
    with pytest.raises(ValueError, match="consecutive"):
        recorded_episode(trace)
    rows.pop()
    write()
    with pytest.raises(ValueError, match="complete"):
        recorded_episode(trace)


@pytest.mark.parametrize("arm", ["reference", "rlt_a"])
def test_audit_observers_preserve_upstream_actions_and_weights(
    tmp_path, monkeypatch, arm
):
    """Use the actual optional upstream actor/evaluator with a fake sim boundary."""
    import json
    import os

    import numpy as np
    import torch
    from PIL import Image

    source = os.environ.get("RLT_ALPHABRAIN_SOURCE")
    if not source:
        pytest.skip("Set RLT_ALPHABRAIN_SOURCE to test the optional upstream adapter")
    monkeypatch.syspath_prepend(source)
    from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import (
        ActionTokenActor,
    )
    from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_encoder_decoder import (
        ActionTokenEncoderDecoder,
    )
    from AlphaBrain.training.reinforcement_learning.envs import libero_env
    from AlphaBrain.training.reinforcement_learning.eval import eval_helpers

    from toolkits.rlt.libero_audit import trace_evaluation
    from toolkits.rlt.libero_reproduction import ReferenceEncoder

    executed = []

    class SimBoundary:
        task_description = "deterministic fixture"

        def __init__(self, **kwargs):
            self.simulation_step = 0

        def observation(self):
            pixels = np.full((8, 8, 3), self.simulation_step, dtype=np.uint8)
            return {
                "primary_image": Image.fromarray(pixels),
                "wrist_image": Image.fromarray(pixels),
                "state": np.full(8, self.simulation_step / 100, dtype=np.float32),
            }

        def reset(self, **kwargs):
            assert kwargs["task_id"] == 5 and kwargs["initial_state_idx"] == 1
            return self.observation()

        def step(self, action):
            executed.append(np.asarray(action).copy())
            self.simulation_step += 1
            done = self.simulation_step == 12
            return self.observation(), float(done), done

        def close(self):
            pass

    class VLABoundary(torch.nn.Module):
        def get_vla_action(self, **kwargs):
            queries = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
            proposals = torch.linspace(-0.8, 0.8, 14).reshape(1, 2, 7)
            return queries, proposals.to(torch.bfloat16)

    encoder = (
        ActionTokenEncoderDecoder(
            input_dim=4,
            bottleneck_dim=4,
            chunk_len=2,
            num_heads=2,
            encoder_layers=1,
            decoder_layers=1,
        )
        .eval()
        .requires_grad_(False)
    )
    actor = (
        ActionTokenActor(
            bottleneck_dim=4,
            action_dim=7,
            chunk_len=2,
            hidden_dim=16,
            prop_dim=8,
        )
        .eval()
        .requires_grad_(False)
    )
    before = [
        p.detach().clone() for model in (encoder, actor) for p in model.parameters()
    ]
    monkeypatch.setattr(libero_env, "LiberoEnv", SimBoundary)
    kwargs = {
        "frozen_vla": VLABoundary(),
        "encoder": encoder if arm == "rlt_a" else ReferenceEncoder(),
        "actor": actor if arm == "rlt_a" else ReferenceActor(),
        "suite_name": "libero_goal",
        "task_id": 5,
        "action_norm_stats": {"q01": [-1.0] * 7, "q99": [1.0] * 7},
        "max_steps": 2,
        "chunk_len": 2,
        "episode_indices": [1],
        "num_steps_wait": 10,
        "seed": 42,
        "device": "cpu",
    }
    expected = eval_helpers._eval_deterministic_local(**kwargs)
    unobserved_actions = np.asarray(executed)
    executed.clear()
    with trace_evaluation(tmp_path, {5: [1]}) as outcomes:
        actual = eval_helpers._eval_deterministic_local(
            **{**kwargs, "episode_indices": [0, 1]}
        )
    assert actual == expected == [(1, 1, True)]
    assert outcomes == [{"arm": arm, "task": 5, "state": 1, "success": True}]
    np.testing.assert_array_equal(unobserved_actions, np.asarray(executed))
    trace = tmp_path / "traces" / arm / "task_5_state_1/trace.jsonl"
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert records[0]["kind"] == "reset" and records[-1]["kind"] == "close"
    recorded = np.asarray([r["action"] for r in records if r["kind"] == "step"])
    np.testing.assert_array_equal(recorded, unobserved_actions)
    decisions = [r for r in records if r["kind"] == "decision"]
    assert len(decisions) == 1 and decisions[0]["tick"] == 10
    assert ("rl_token" in decisions[0]) == (arm == "rlt_a")
    after = [p for model in (encoder, actor) for p in model.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_training_is_one_visible_gpu_without_concurrent_eval(tmp_path):
    command = training_arguments(tmp_path, tmp_path / "output", 0, 20)

    def value(name):
        return command[command.index(name) + 1]

    assert value("--rollout_gpus") == value("--train_gpu") == "0"
    assert value("--eval_interval") == "0"
    assert value("--encoder_mode") == "action_token"
    assert value("--max_iter") == value("--save_interval") == "20"
    assert "--use_wandb" not in command and "--finetune_vla" not in command
    with pytest.raises(ValueError):
        training_arguments(tmp_path, tmp_path / "output", 0, 0)


def test_update_budget_does_not_count_zero_actor_loss_placeholders_as_updates():
    config = {
        "td_batch_size": 128,
        "utd_ratio": 2.0,
        "td_updates_per_iter": 128,
        "actor_update_freq": 2,
    }
    history = [
        {"iter": 1, "n_pushed": 50, "buffer_size": 50},
        {
            "iter": 2,
            "n_pushed": 80,
            "buffer_size": 130,
            "critic_loss": 1.0,
            "actor_loss": 0.0,
        },
        {
            "iter": 3,
            "n_pushed": 160,
            "buffer_size": 290,
            "critic_loss": 0.5,
            "actor_loss": 0.8,
        },
    ]
    report = summarize_update_budget(history, config)
    assert report["critic_updates"] == 3
    assert report["actor_updates"] == 1
    assert report["critic_only_iterations"] == [2]
    measured = [
        {"iter": 1, "critic_updates": 1, "actor_updates": 0, "rollout_sync_step": 1},
        {"iter": 2, "critic_updates": 1, "actor_updates": 1, "rollout_sync_step": 2},
    ]
    report = summarize_update_budget(measured, config)
    assert report["actor_updates"] == 1
    assert report["critic_updates"] == 2
    assert report["rollout_sync_steps"] == [1, 2]
    assert report["provenance"].startswith("Instrumented")


@pytest.mark.parametrize("file_logging", [False, True])
def test_worker_preserves_binary_protocol_and_redirects_python_and_c_logs(
    tmp_path, file_logging
):
    import os
    import subprocess
    import sys
    from pathlib import Path

    worker = (
        tmp_path
        / "AlphaBrain/training/reinforcement_learning/envs/libero_env_worker.py"
    )
    worker.parent.mkdir(parents=True)
    worker.write_text(
        "import os, sys\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
        "assert os.environ['MUJOCO_EGL_DEVICE_ID'] == '0'\n"
        "print('python log')\nos.write(1, b'C log\\n')\n"
        "print('x' * 100000)\n"
        "sys.stdout.buffer.write(b'\\x04\\x00\\x00\\x00data')\n"
    )
    wrapper = Path(__file__).resolve().parents[2] / "toolkits/rlt/libero_worker.py"
    result = subprocess.run(
        [sys.executable, str(wrapper)],
        capture_output=True,
        timeout=10,
        env={
            **os.environ,
            "RLT_ALPHABRAIN_SOURCE": str(tmp_path),
            "RLT_LIBERO_EGL_DEVICE_ID": "0",
            "CUDA_VISIBLE_DEVICES": "2",
            "RLT_LIBERO_WORKER_LOG_DIR": str(tmp_path / "logs") if file_logging else "",
        },
        check=True,
    )
    assert result.stdout == b"\x04\x00\x00\x00data"
    if file_logging:
        logs = list((tmp_path / "logs").glob("worker-*.log"))
        assert len(logs) == 1 and result.stderr == b""
        output = logs[0].read_bytes()
    else:
        output = result.stderr
    assert b"python log" in output and b"C log" in output
    assert b"x" * 100000 in output


def test_training_acceptance_requires_fresh_interaction_every_iteration():
    validate_training_history([{"iter": 1, "iter_env_steps": 100}], 1)
    with pytest.raises(ValueError, match="No new"):
        validate_training_history([{"iter": 1, "iter_env_steps": 0}], 1)
    with pytest.raises(ValueError, match="every requested"):
        validate_training_history([{"iter": 1, "iter_env_steps": 100}], 2)
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_training_history(
            [{"iter": 1, "iter_env_steps": 100, "actor_loss": float("nan")}], 1
        )
    with pytest.raises(ValueError, match="No actor updates"):
        validate_training_history(
            [{"iter": i, "iter_env_steps": 100} for i in range(1, 7)], 6
        )


def test_fast_worker_preserves_inherited_socket_and_uses_mapped_egl(tmp_path):
    import os
    import socket
    import subprocess
    import sys
    from pathlib import Path

    worker = (
        tmp_path
        / "AlphaBrain/training/reinforcement_learning/envs/libero_env_worker_fast.py"
    )
    worker.parent.mkdir(parents=True)
    worker.write_text(
        "import os, socket, sys\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
        "assert os.environ['MUJOCO_EGL_DEVICE_ID'] == '2'\n"
        "sock = socket.socket(fileno=int(sys.argv[1]))\n"
        "assert sock.recv(4) == b'ping'\n"
        "print('fast diagnostics')\n"
        "sock.sendall(b'pong')\n"
    )
    wrapper = Path(__file__).resolve().parents[2] / "toolkits/rlt/libero_worker.py"
    parent, child = socket.socketpair()
    with parent, child:
        parent.settimeout(5)
        process = subprocess.Popen(
            [sys.executable, str(wrapper), str(child.fileno())],
            pass_fds=(child.fileno(),),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **os.environ,
                "RLT_ALPHABRAIN_SOURCE": str(tmp_path),
                "RLT_LIBERO_EGL_DEVICE_ID": "2",
                "MUJOCO_EGL_DEVICE_ID": "0",
                "RLT_LIBERO_WORKER_LOG_DIR": str(tmp_path / "logs"),
            },
        )
        try:
            parent.sendall(b"ping")
            assert parent.recv(4) == b"pong"
            stdout, stderr = process.communicate(timeout=5)
            assert process.returncode == 0 and stdout == stderr == b""
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_egl_device_order_is_not_cuda_order():
    from toolkits.rlt.libero_egl import unique_device_index

    gpu = "4662787b-485a-0e8f-e4b2-dd47352ed69c"
    other = "effa2680-752a-d5ca-2d56-9b74340eafa6"
    assert unique_device_index([gpu, None, other], f"GPU-{gpu}") == 0
    assert unique_device_index([other, None, gpu], f"GPU-{gpu}") == 2
    with pytest.raises(RuntimeError, match="Need one"):
        unique_device_index([other], f"GPU-{gpu}")
    with pytest.raises(RuntimeError, match="Need one"):
        unique_device_index([gpu, gpu], f"GPU-{gpu}")
