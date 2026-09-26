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

"""Single CPU-physics environment with baseline-compatible cameras and actions."""

from __future__ import annotations

import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import numpy as np

from .protocol import ENV_ID

DEFAULT_RENDER_BACKEND = "cpu" if sys.platform == "win32" else "gpu"


class _TorchPandaIK:
    """Map a nearby Cartesian target to Panda joint deltas with PyTorch."""

    def __init__(self, urdf_path: str, end_link_name: str) -> None:
        import pytorch_kinematics as pk
        import torch

        self.pk = pk
        self.torch = torch
        with open(urdf_path, "rb") as urdf:
            description = urdf.read()
        # Panda's URDF has simulator-only dynamics attributes that the generic
        # kinematics parser reports but safely ignores.
        with open(os.devnull, "w") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                self.chain = pk.build_serial_chain_from_urdf(
                    description, end_link_name=end_link_name
                ).to(device="cpu", dtype=torch.float32)

    def joint_delta(
        self, qpos: np.ndarray, target_at_base: np.ndarray
    ) -> np.ndarray | None:
        """Return one damped least-squares step for a nearby target pose."""
        torch = self.torch
        q = torch.as_tensor(qpos[:7], dtype=torch.float32).unsqueeze(0)
        target = torch.as_tensor(target_at_base, dtype=torch.float32).unsqueeze(0)
        current = self.chain.forward_kinematics(q).get_matrix()
        position_error = target[:, :3, 3] - current[:, :3, 3]
        rotation_error = self.pk.matrix_to_axis_angle(
            target[:, :3, :3] @ current[:, :3, :3].transpose(1, 2)
        )
        error = torch.cat((position_error, rotation_error), dim=-1).unsqueeze(-1)
        jacobian = self.chain.jacobian(q)
        transpose = jacobian.transpose(1, 2)
        regularizer = 1e-4 * torch.eye(
            jacobian.shape[-1], dtype=jacobian.dtype
        ).unsqueeze(0)
        try:
            delta = torch.linalg.solve(
                transpose @ jacobian + regularizer, transpose @ error
            )[0, :, 0]
        except RuntimeError:
            return None
        if not torch.isfinite(delta).all():
            return None
        return delta.detach().cpu().numpy()


class LocalSimulation:
    """Own a local Panda environment and its CPU inverse-kinematics model."""

    def __init__(
        self, render_backend: str = DEFAULT_RENDER_BACKEND, seed: int = 0
    ) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
            PANDA_WIDE_WRISTCAM_UID,
            register_rlinf_peg_insertion_side_variants,
        )
        from rlinf.envs.sim.maniskill.utils import allow_pci_render_backend

        register_rlinf_peg_insertion_side_variants()
        allow_pci_render_backend()
        self.env = gym.make(
            ENV_ID,
            num_envs=1,
            robot_uids=PANDA_WIDE_WRISTCAM_UID,
            obs_mode="rgb",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            sim_backend="physx_cpu",
            render_backend=render_backend,
            render_mode="rgb_array",
            sim_config={"sim_freq": 100, "control_freq": 10},
            sensor_configs={"width": 384, "height": 384},
        )
        try:
            self.raw, _ = self.env.reset(seed=seed)
            self.robot = self.env.unwrapped.agent.robot
            self.tcp = self.env.unwrapped.agent.tcp
            self.ik = _TorchPandaIK(
                self.env.unwrapped.agent.urdf_path,
                self.env.unwrapped.agent.ee_link_name,
            )
            arm = self.env.unwrapped.agent.controller.controllers["arm"]
            if arm.config.lower != -0.1 or arm.config.upper != 0.1:
                raise ValueError("Expected baseline Panda joint delta bounds +/-0.1")
        except BaseException:
            self.env.close()
            raise

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        return value.detach().cpu().numpy()

    def observation(self) -> dict[str, np.ndarray]:
        """Return unnormalized qpos and the two baseline RGB camera images."""
        sensors = self.raw["sensor_data"]
        return {
            "state": self._array(self.raw["agent"]["qpos"])[0, :9].copy(),
            "main_image": self._array(sensors["3rd_view_camera"]["rgb"])[0].copy(),
            "wrist_image": self._array(sensors["wide_hand_camera"]["rgb"])[0].copy(),
        }

    def tcp_matrix(self) -> np.ndarray:
        """Return the current TCP pose in the world frame."""
        return self.tcp.pose.sp.to_transformation_matrix().copy()

    def human_action(self, target: np.ndarray, gripper: float) -> np.ndarray | None:
        """Solve a world-frame pose; reject IK failures instead of sending NaNs."""
        import sapien

        qpos = self._array(self.robot.get_qpos())[0]
        base_target = self.robot.pose.sp.inv() * sapien.Pose(target)
        delta_qpos = self.ik.joint_delta(qpos, base_target.to_transformation_matrix())
        if delta_qpos is None:
            return None
        # Bound teleop to 0.025 rad/control step, below the controller's limit.
        delta = np.clip(delta_qpos / 0.1, -0.25, 0.25)
        return np.r_[delta, np.clip(gripper, -1, 1)].astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool]:
        """Execute exactly one 100 ms control step; never auto-reset."""
        self.raw, reward, terminated, truncated, _ = self.env.step(action[None])
        return (
            self.observation(),
            float(reward.item()),
            bool(terminated.item()),
            bool(truncated.item()),
        )

    def reset(self, seed: int) -> dict:
        """Reset the episode; rebuild IK if ManiSkill reconfigures the scene."""
        backend = self.render_backend
        self.env.close()
        self.__init__(backend, seed)
        return self.observation()

    @property
    def render_backend(self) -> str:
        """Current SAPIEN backend, retained for explicit episode resets."""
        return str(self.env.unwrapped.backend.render_backend)

    def close(self) -> None:
        """Release local simulator and renderer resources."""
        self.env.close()
