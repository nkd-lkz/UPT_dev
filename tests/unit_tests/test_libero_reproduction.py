# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Published LIBERO reproduction protocol, independent of model installation."""

import pytest

from toolkits.rlt.libero_reproduction import (
    ReferenceActor,
    episode_plan,
    paired_summary,
    training_arguments,
    validate_training_history,
)


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
