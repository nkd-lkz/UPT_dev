# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU-testable ARX X5 policy boundary for a future RoboDojo model backend.

This is not an Isaac simulator or a LIBERO-to-ARX action conversion. A backend
must already output physical absolute joints and normalized gripper openings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

ROBO_DOJO_REVISION = "cb2a37d58dbbeb2777b0c2b30e97ecb8ab91095c"
XPOLICYLAB_REVISION = "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2"
FIELDS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)
CAMERAS = (
    ("cam_head", "cam_high"),
    ("cam_left_wrist", "cam_left_wrist"),
    ("cam_right_wrist", "cam_right_wrist"),
)
ACTION_SEMANTICS = "absolute_joint_radians_gripper_01"


def encode_observation(observation: Mapping) -> dict:
    """Copy three decoded RGB views and native state, without simulator metadata."""
    instruction = observation.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Require a nonempty task instruction")
    parts = []
    for name, dim in FIELDS:
        part = np.asarray(observation["state"][name], dtype=np.float32)
        if part.shape != (dim,) or not np.isfinite(part).all():
            raise ValueError(f"Invalid state field: {name}")
        parts.append(part)
    images = {}
    for native, target in CAMERAS:
        image = np.asarray(observation["vision"][native]["color"])
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[-1] != 3
            or 0 in image.shape
        ):
            raise ValueError(f"Require decoded HWC uint8 RGB: {native}")
        images[target] = image.transpose(2, 0, 1).copy()
    return {"state": np.concatenate(parts), "images": images, "prompt": instruction}


def decode_actions(chunk: np.ndarray, *, semantics: str, horizon: int) -> list[dict]:
    """Validate a physical chunk and clamp grippers as the native evaluator does.

    Arm joints remain absolute radians. Finite gripper predictions saturate to
    [0, 1], matching RoboDojo's ``take_action_batch``; no arm values are clipped.
    """
    if semantics != ACTION_SEMANTICS:
        raise ValueError("Require physical absolute joint actions, not LIBERO deltas")
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 14 or not len(chunk):
        raise ValueError(
            "ARX X5 requires [H, 14] actions; a LIBERO head is incompatible"
        )
    if type(horizon) is not int or not 1 <= horizon <= len(chunk):
        raise ValueError("Invalid action execution horizon")
    if not np.isfinite(chunk).all():
        raise ValueError("Require finite arm and gripper actions")
    result = []
    for row in chunk[:horizon]:
        offset, command = 0, {}
        for name, dim in FIELDS:
            command[name] = row[offset : offset + dim].copy()
            offset += dim
        for name in ("left_ee_joint_state", "right_ee_joint_state"):
            command[name] = np.clip(command[name], 0.0, 1.0)
        result.append(command)
    return result


class RoboDojoBoundary:
    """Route active environment IDs to an explicitly compatible inference backend.

    Mirrors the update/get/reset order used by XPolicyLab. It has no network or
    simulator ownership. Each request must be preceded by a fresh observation;
    episode resets clear all stored inputs. The backend owns normalization.
    """

    def __init__(
        self,
        predict: Callable[[dict], np.ndarray],
        *,
        model_contract: dict,
        horizon: int = 10,
    ) -> None:
        expected = {
            "embodiment": "arx_x5",
            "action_dim": 14,
            "action_semantics": ACTION_SEMANTICS,
        }
        if any(model_contract.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "Load a RoboDojo ARX X5 checkpoint, not a LIBERO checkpoint"
            )
        if (
            not model_contract.get("normalization_id")
            or type(horizon) is not int
            or horizon < 1
        ):
            raise ValueError("Require identified normalization and a positive horizon")
        self.predict, self.horizon = predict, horizon
        self.observations: dict[int, dict] = {}

    def reset(self) -> None:
        """Discard all observations at an episode boundary."""
        self.observations.clear()

    def update_obs_batch(self, observations: list[dict]) -> None:
        """Replace observations atomically, keyed by explicit environment IDs."""
        pending = {}
        for observation in observations:
            index = observation.get("env_idx")
            if type(index) is not int or index < 0 or index in pending:
                raise ValueError("Require unique nonnegative env_idx values")
            pending[index] = encode_observation(observation)
        self.observations = pending

    def get_action_batch(self, env_idx_list: list[int]) -> list[list[dict]]:
        """Return chunks in requested order, even after the active batch shrinks."""
        if (
            not env_idx_list
            or any(type(i) is not int or i < 0 for i in env_idx_list)
            or len(set(env_idx_list)) != len(env_idx_list)
        ):
            raise ValueError("Require a nonempty list of unique active environment IDs")
        if any(index not in self.observations for index in env_idx_list):
            raise ValueError("Update each requested environment before inference")
        # Consume before inference so a failed/partial request cannot reuse stale data.
        observations = [self.observations.pop(index) for index in env_idx_list]
        return [
            decode_actions(
                self.predict(obs), semantics=ACTION_SEMANTICS, horizon=self.horizon
            )
            for obs in observations
        ]
