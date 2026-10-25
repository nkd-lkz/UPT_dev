# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Expose explicit LIBERO-Pro cases through the AlphaBrain RLT env contract."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

PERTURBATIONS = {"lan", "object", "swap", "task"}


def _numpy(value: Any) -> np.ndarray:
    """Copy a tensor or array to host NumPy storage."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _observation(value: dict[str, Any]) -> dict[str, np.ndarray]:
    """Translate RLinf's vector observation to AlphaBrain's single-env schema."""
    primary = _numpy(value["main_images"])[0]
    wrist = _numpy(value["wrist_images"])[0]
    state = _numpy(value["states"])[0].astype(np.float32, copy=False)
    if primary.ndim != 3 or wrist.ndim != 3 or state.shape != (8,):
        raise ValueError(
            "Unexpected LIBERO-Pro observation shapes: "
            f"primary={primary.shape}, wrist={wrist.shape}, state={state.shape}"
        )
    if primary.shape[0] in (1, 3, 4) and primary.shape[-1] not in (1, 3, 4):
        primary = np.moveaxis(primary, 0, -1)
    if wrist.shape[0] in (1, 3, 4) and wrist.shape[-1] not in (1, 3, 4):
        wrist = np.moveaxis(wrist, 0, -1)
    return {
        "primary_image": np.ascontiguousarray(primary),
        "wrist_image": np.ascontiguousarray(wrist),
        "state": np.ascontiguousarray(state),
    }


def observation_digest(observation: dict[str, np.ndarray], identity: str) -> str:
    """Identify the rendered and proprioceptive initial condition together."""
    digest = hashlib.sha256(identity.encode())
    for key in ("primary_image", "wrist_image", "state"):
        value = np.ascontiguousarray(observation[key])
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


class LiberoProEnvironment:
    """Run one named task/state/perturbation without a standard-LIBERO fallback."""

    variant = "libero_pro"
    settling_steps_consumed = 1

    def __init__(self) -> None:
        if os.environ.get("LIBERO_TYPE") != "pro":
            raise RuntimeError("LIBERO-Pro RLT requires LIBERO_TYPE=pro")
        egl_device = os.environ.get("RLT_LIBERO_EGL_DEVICE_ID")
        if egl_device is None or not egl_device.isdigit():
            raise RuntimeError("LIBERO-Pro RLT requires an integer EGL device mapping")
        # The policy process is isolated by physical GPU UUID.  Robosuite 1.4's
        # EGL helper cannot parse UUIDs, so give only the simulator subprocess
        # the already-verified physical EGL index through MuJoCo's dedicated
        # selector.  This preserves CUDA UUID isolation for the policy model.
        os.environ["MUJOCO_EGL_DEVICE_ID"] = egl_device
        self.env = None
        self.instruction = ""
        self.reset_hash = ""
        self.selected_perturbation_folder = ""

    @staticmethod
    def _config(case: dict[str, Any], reset_id: int):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "env_type": "libero",
                "task_suite_name": case.get("suite", "libero_10"),
                "total_num_envs": 1,
                "auto_reset": False,
                "ignore_terminations": False,
                # The RLT driver owns the suite-specific control horizon. Keep
                # RLinf's internal truncation beyond 10 settling + 520 control
                # steps so a time limit can never be mistaken for success.
                "max_steps_per_rollout_epoch": 600,
                "max_episode_steps": 600,
                "use_fixed_reset_state_ids": True,
                "use_ordered_reset_state_ids": False,
                "use_rel_reward": True,
                "use_step_penalty": False,
                "reward_coef": 1.0,
                "reset_gripper_open": True,
                "is_eval": True,
                "seed": case["seed"],
                "group_size": 1,
                "specific_reset_id": reset_id,
                "task_id_filter": None,
                # LiberoEnv needs one step to materialize its first raw observation;
                # AlphaBrain's outer loop executes the remaining nine settling steps.
                "num_steps_wait": 1,
                "skip_intermediate_renders": False,
                "video_cfg": {
                    "save_video": False,
                    "info_on_video": False,
                    "video_base_dir": str(Path(os.devnull).parent),
                },
                "init_params": {"camera_heights": 256, "camera_widths": 256},
            }
        )

    @staticmethod
    def _reset_id(case: dict[str, Any]) -> int:
        from liberopro.liberopro.benchmark import get_benchmark

        suite = get_benchmark(case.get("suite", "libero_10"))()
        task = int(case["task"])
        state = int(case["state"])
        bins = [
            len(suite.get_task_init_states(i)) for i in range(suite.get_num_tasks())
        ]
        if task not in range(len(bins)) or state not in range(bins[task]):
            raise ValueError(f"Invalid LIBERO-Pro task/state: {task}/{state}")
        return sum(bins[:task]) + state

    def reset(self, case: dict[str, Any], *, create_snapshot: bool = False) -> dict:
        """Create exactly the declared Pro perturbation and published state."""
        if create_snapshot or case.get("kind") != "published":
            raise ValueError("LIBERO-Pro RLT currently requires published states")
        perturbation = case.get("perturbation")
        if perturbation not in PERTURBATIONS:
            raise ValueError(f"Invalid LIBERO-Pro perturbation: {perturbation}")
        self.close()
        os.environ["LIBERO_PERTURBATION"] = perturbation
        from rlinf.envs.sim.libero.libero_env import LiberoEnv

        self.env = LiberoEnv(
            self._config(case, self._reset_id(case)),
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        wrapped, _ = self.env.reset()
        observation = _observation(wrapped)
        self.instruction = str(wrapped["task_descriptions"][0])
        self.selected_perturbation_folder = str(self.env._pert_init_folders[0])
        identity = ":".join(
            (
                case.get("suite", "libero_10"),
                str(case["task"]),
                str(case["state"]),
                perturbation,
                str(case["seed"]),
                self.selected_perturbation_folder,
                self.instruction,
            )
        )
        self.reset_hash = observation_digest(observation, identity)
        return observation

    def step(self, action: np.ndarray) -> tuple[dict, float, bool]:
        """Execute one native LIBERO command and return AlphaBrain-compatible data."""
        if self.env is None:
            raise RuntimeError("Reset LIBERO-Pro before stepping")
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError("LIBERO-Pro action must be a finite 7-vector")
        wrapped, reward, terminated, truncated, _ = self.env.step(
            command[None], auto_reset=False
        )
        scalar_reward = float(_numpy(reward).reshape(-1)[0])
        succeeded = bool(_numpy(terminated).reshape(-1)[0])
        timed_out = bool(_numpy(truncated).reshape(-1)[0])
        if timed_out and not succeeded:
            raise RuntimeError("LIBERO-Pro truncated before the RLT control budget")
        return _observation(wrapped), scalar_reward, succeeded

    def close(self) -> None:
        """Idempotently close the current Pro simulator subprocess."""
        if self.env is not None:
            self.env.env.close()
            self.env = None
