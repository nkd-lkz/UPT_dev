# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO subprocess with explicit published or independently generated resets.

Only the simulator sees XML and physical state. The policy receives the same
two images and eight proprioceptive values as the upstream LIBERO adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def snapshot_digest(state: np.ndarray) -> str:
    """Identify a simulator state without depending on JSON float formatting."""
    return hashlib.sha256(np.asarray(state, dtype="<f8").tobytes()).hexdigest()


def worker() -> None:
    """Own one environment and close it when its parent closes the protocol."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["RLT_LIBERO_EGL_DEVICE_ID"]
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    directory = Path(os.environ["RLT_LIBERO_WORKER_LOG_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"worker-{os.getpid()}.log").open("x", buffering=1) as log:
        os.dup2(log.fileno(), sys.stderr.fileno())
        os.dup2(log.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        from AlphaBrain.training.reinforcement_learning.common.parent_death import (
            set_die_with_parent,
        )
        from AlphaBrain.training.reinforcement_learning.envs.libero_env_worker import (
            _parse_obs,
            _read_msg,
            _write_msg,
        )
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        set_die_with_parent()
        suite = benchmark.get_benchmark_dict()["libero_goal"]()
        env = None
        try:
            while True:
                try:
                    message = _read_msg(sys.stdin.buffer)
                except EOFError:
                    break
                try:
                    if message["cmd"] == "close":
                        break
                    if message["cmd"] == "reset":
                        if env is not None:
                            env.close()
                        case = message["case"]
                        task = suite.get_task(case["task"])
                        env = OffScreenRenderEnv(
                            bddl_file_name=suite.get_task_bddl_file_path(case["task"]),
                            camera_heights=256,
                            camera_widths=256,
                        )
                        env.seed(case["seed"])
                        obs = env.reset()
                        if case["kind"] == "published":
                            obs = env.set_init_state(
                                suite.get_task_init_states(case["task"])[case["state"]]
                            )
                        elif case["kind"] == "generated":
                            path = Path(case["snapshot"])
                            if path.is_file():
                                saved = json.loads(path.read_text())
                                if (
                                    saved["task"] != case["task"]
                                    or saved["seed"] != case["seed"]
                                ):
                                    raise ValueError("Snapshot identity mismatch")
                                env.reset_from_xml_string(saved["xml"])
                                obs = env.set_init_state(np.asarray(saved["state"]))
                            else:
                                if not message.get("create_snapshot", False):
                                    raise FileNotFoundError(path)
                                state = np.asarray(env.get_sim_state())
                                published = suite.get_task_init_states(case["task"])
                                if any(
                                    np.array_equal(state, np.asarray(s))
                                    for s in published
                                ):
                                    raise ValueError(
                                        "Generated reset duplicates a published state"
                                    )
                                saved = {
                                    "task": case["task"],
                                    "seed": case["seed"],
                                    "state": state.tolist(),
                                    "xml": env.sim.model.get_xml(),
                                    "state_sha256": snapshot_digest(state),
                                }
                                path.parent.mkdir(parents=True, exist_ok=True)
                                with path.open("x") as stream:
                                    json.dump(saved, stream, allow_nan=False)
                                # Generation and subsequent readers use the same
                                # restoration path, not different reset dynamics.
                                env.reset_from_xml_string(saved["xml"])
                                obs = env.set_init_state(np.asarray(saved["state"]))
                            if (
                                snapshot_digest(env.get_sim_state())
                                != saved["state_sha256"]
                            ):
                                raise ValueError(
                                    "Restored physical state differs from snapshot"
                                )
                        else:
                            raise ValueError("Unknown initial-state source")
                        _write_msg(
                            protocol,
                            {
                                "status": "ok",
                                "obs": _parse_obs(obs),
                                "task_description": task.language,
                                "state_sha256": snapshot_digest(env.get_sim_state()),
                            },
                        )
                    elif message["cmd"] == "step":
                        command = np.asarray(message["action"], dtype=np.float32)
                        if command.shape != (7,) or not np.isfinite(command).all():
                            raise ValueError("Invalid command")
                        obs, reward, done, _ = env.step(command.tolist())
                        _write_msg(
                            protocol,
                            {
                                "status": "ok",
                                "obs": _parse_obs(obs),
                                "reward": float(reward),
                                "done": bool(done),
                            },
                        )
                    else:
                        raise ValueError("Unknown command")
                except Exception:
                    _write_msg(
                        protocol, {"status": "error", "message": traceback.format_exc()}
                    )
        finally:
            if env is not None:
                env.close()
            protocol.close()


if __name__ == "__main__":
    worker()
