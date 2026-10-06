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

"""Local authority over action chunks, independent of network latency."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .protocol import validate_actions


class ActionGate:
    """Discard policy actions whenever authority changes or tracking fails.

    The caller polls VR on the display loop, not only at action-chunk boundaries.
    A request ticket remains valid only until the next authority change. Paused
    simulation does not age observations, but network replies still have a TTL.
    """

    def __init__(self) -> None:
        self.generation = 0
        self.mode = "paused"
        self._actions: deque[np.ndarray] = deque()

    def change(self, mode: str) -> None:
        """Set authority and invalidate pending chunks on transitions."""
        if mode not in {"paused", "human", "policy"}:
            raise ValueError("Unknown control authority")
        if mode != self.mode:
            self.generation += 1
            self._actions.clear()
            self.mode = mode

    def reset(self) -> None:
        """Invalidate all pre-reset responses, even while already paused."""
        self.generation += 1
        self._actions.clear()
        self.mode = "paused"

    def accept(self, generation: int, actions: np.ndarray) -> bool:
        """Accept a fresh chunk only while the policy owns control."""
        if generation != self.generation or self.mode != "policy":
            return False
        if self._actions:
            raise RuntimeError("Cannot overwrite an executing action chunk")
        self._actions.extend(validate_actions(actions))
        return True

    def next_action(self) -> np.ndarray | None:
        """Consume one action or return None to pause physics, not the UI."""
        if self.mode != "policy" or not self._actions:
            return None
        return self._actions.popleft().copy()

    @property
    def needs_prediction(self) -> bool:
        """Whether a fresh observation should be submitted for inference."""
        return self.mode == "policy" and not self._actions


class OperatorControl:
    """Require grip release after faults and reset after episode termination."""

    def __init__(self) -> None:
        self.gate = ActionGate()
        self.finished = False
        self.require_release = False
        self.pause_reason = "startup"
        self.pause_detail = "P=policy; hold grip=human"

    def pause(self, reason: str = "operator", detail: str = "") -> None:
        """Latch a fault until the operator releases the clutch."""
        if self.finished:
            return
        self.gate.change("paused")
        self.require_release = True
        self.pause_reason, self.pause_detail = reason, detail

    @property
    def instruction(self) -> str:
        """Describe the latched cause and the action needed to recover."""
        if self.finished:
            return f"episode_ended: {self.pause_detail}; R resets"
        if self.require_release:
            return f"{self.pause_reason}: {self.pause_detail}; release grip, then re-grip/P"
        if self.gate.mode == "paused":
            return f"paused ({self.pause_reason}); P=policy; hold grip=human"
        return f"{self.gate.mode}: release grip=paused; Space=pause"

    def finish(self, detail: str = "task terminal or time limit") -> None:
        """Prevent all stepping until an explicit episode reset."""
        self.pause("episode_ended", detail)
        self.finished = True

    def reset(self) -> None:
        """Invalidate pre-reset responses and require a fresh clutch press."""
        self.gate.reset()
        self.finished = False
        self.require_release = True
        self.pause_reason, self.pause_detail = "reset", "fresh episode"

    def update(
        self, *, valid: bool, clutch: bool, command: str = "", stalled: bool = False
    ) -> str:
        """Apply local intent before considering any network result."""
        if not valid:
            self.pause("tracking_invalid", "check SteamVR tracking/input focus")
        elif stalled:
            self.pause("ui_stall", "input/render watchdog")
        elif command == "pause":
            self.pause("operator", "Space pressed")
        elif self.finished:
            self.gate.change("paused")
        elif self.require_release:
            if not clutch:
                self.require_release = False
        elif clutch:
            self.gate.change("human")
        elif self.gate.mode == "human":
            self.gate.change("paused")
            self.pause_reason = "grip_released"
        elif command == "policy":
            self.gate.change("policy")
        return self.gate.mode


class TargetFilter:
    """Smooth and rate-limit Cartesian targets at the unchanged control rate."""

    def __init__(
        self,
        pose: np.ndarray,
        speed: float = 0.12,
        angular_speed: float = 0.8,
        time_constant: float = 0.12,
    ) -> None:
        if not all(
            np.isfinite(x) and x > 0 for x in (speed, angular_speed, time_constant)
        ):
            raise ValueError("Target filter limits must be positive and finite")
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Expected a finite 4x4 anchor pose")
        self.pose = pose.copy()
        self.speed, self.angular_speed, self.time_constant = (
            speed,
            angular_speed,
            time_constant,
        )

    def update(self, target: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Advance one simulation control interval, never a network-wait interval."""
        from scipy.spatial.transform import Rotation

        if (
            target.shape != (4, 4)
            or not np.isfinite(target).all()
            or not np.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError("Invalid target or control interval")
        alpha = 1 - np.exp(-dt / self.time_constant)
        delta = alpha * (target[:3, 3] - self.pose[:3, 3])
        delta *= min(1.0, self.speed * dt / max(np.linalg.norm(delta), 1e-9))
        rotation = (
            alpha
            * Rotation.from_matrix(target[:3, :3] @ self.pose[:3, :3].T).as_rotvec()
        )
        rotation *= min(
            1.0, self.angular_speed * dt / max(np.linalg.norm(rotation), 1e-9)
        )
        self.pose[:3, 3] += delta
        self.pose[:3, :3] = (
            Rotation.from_rotvec(rotation).as_matrix() @ self.pose[:3, :3]
        )
        return self.pose.copy()


def bounded_joint_delta(
    delta: np.ndarray,
    qpos: np.ndarray,
    limits: np.ndarray,
    previous: np.ndarray,
    dt: float = 0.1,
) -> tuple[np.ndarray, bool]:
    """Bound commanded joint speed, acceleration and joint-limit approach.

    Hard joint limits take priority over acceleration smoothing. The result is a
    position increment in radians, not a guarantee on measured joint velocity.
    """
    if (
        delta.shape != (7,)
        or qpos.shape != (7,)
        or previous.shape != (7,)
        or limits.shape != (7, 2)
        or not np.isfinite(dt)
        or dt <= 0
        or not all(np.isfinite(x).all() for x in (delta, qpos, limits, previous))
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("Nonfinite joint input")
    candidate = delta * min(1.0, 0.25 * dt / max(np.max(np.abs(delta)), 1e-9))
    candidate = np.clip(candidate, previous - dt * dt, previous + dt * dt)
    # Outside the soft band, allow motion back inward but never force a jump.
    lower = np.minimum(limits[:, 0] + 0.02 - qpos, 0.0)
    upper = np.maximum(limits[:, 1] - 0.02 - qpos, 0.0)
    result = np.clip(candidate, lower, upper)
    return result, bool(np.any(np.abs(result - candidate) > 1e-8))


@dataclass(frozen=True)
class MappedTarget:
    """A bounded robot target and the limits applied to the controller motion."""

    pose: np.ndarray
    requested_translation: float
    applied_translation: float
    requested_rotation: float
    applied_rotation: float

    @property
    def translation_limited(self) -> bool:
        """Whether controller translation reached the configured bound."""
        return self.requested_translation > self.applied_translation + 1e-6

    @property
    def rotation_limited(self) -> bool:
        """Whether controller rotation reached the configured bound."""
        return self.requested_rotation > self.applied_rotation + 1e-6

    @property
    def limited(self) -> bool:
        """Whether either motion component reached its configured bound."""
        return self.translation_limited or self.rotation_limited


def map_relative_target(
    anchor_vr: np.ndarray,
    current_vr: np.ndarray,
    anchor_tcp: np.ndarray,
    *,
    scale: float = 0.5,
    yaw_degrees: float = 0,
    max_displacement: float = 0.15,
    max_rotation: float = 0.5,
) -> MappedTarget:
    """Map OpenVR motion to a bounded target and report active limits.

    Translation and rotation are bounded relative to the clutch anchor. The
    rotation bound is in radians; positions are meters. This is not a collision
    avoidance controller. Calibration must be checked with a free-space task.
    """
    from scipy.spatial.transform import Rotation

    for pose in (anchor_vr, current_vr, anchor_tcp):
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Expected finite homogeneous poses")
    if not 0 < scale <= 1 or max_displacement <= 0 or max_rotation <= 0:
        raise ValueError("Invalid motion limits")
    basis = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    basis = Rotation.from_euler("z", yaw_degrees, degrees=True).as_matrix() @ basis
    delta = scale * basis @ (current_vr[:3, 3] - anchor_vr[:3, 3])
    requested_translation = float(np.linalg.norm(delta))
    applied_translation = min(requested_translation, max_displacement)
    delta *= min(1, max_displacement / max(requested_translation, 1e-9))
    rotation = basis @ current_vr[:3, :3] @ anchor_vr[:3, :3].T @ basis.T
    vector = Rotation.from_matrix(rotation).as_rotvec()
    requested_rotation = float(np.linalg.norm(vector))
    applied_rotation = min(requested_rotation, max_rotation)
    vector *= min(1, max_rotation / max(requested_rotation, 1e-9))
    result = anchor_tcp.copy()
    result[:3, 3] += delta
    result[:3, :3] = Rotation.from_rotvec(vector).as_matrix() @ anchor_tcp[:3, :3]
    return MappedTarget(
        pose=result,
        requested_translation=requested_translation,
        applied_translation=applied_translation,
        requested_rotation=requested_rotation,
        applied_rotation=applied_rotation,
    )


def relative_target(
    anchor_vr: np.ndarray,
    current_vr: np.ndarray,
    anchor_tcp: np.ndarray,
    *,
    scale: float = 0.5,
    yaw_degrees: float = 0,
    max_displacement: float = 0.15,
    max_rotation: float = 0.5,
) -> np.ndarray:
    """Return the bounded pose while preserving the original public contract."""
    return map_relative_target(
        anchor_vr,
        current_vr,
        anchor_tcp,
        scale=scale,
        yaw_degrees=yaw_degrees,
        max_displacement=max_displacement,
        max_rotation=max_rotation,
    ).pose
