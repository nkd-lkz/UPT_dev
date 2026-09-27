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

"""SteamVR input adapter for PICO Business Streaming on the local PC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VRReading:
    """A tracked pose and explicit operator intent from the right controller."""

    pose: np.ndarray
    valid: bool
    clutch: bool
    close_gripper: bool
    buttons: int
    trigger_value: float


class SteamVRController:
    """Read legacy SteamVR grip/trigger bindings; verify mappings with --probe.

    The adapter is experimental: PICO SteamVR profiles must expose Grip and
    SteamVR_Trigger through the legacy controller API. Invalid tracking prevents
    stepping, including during autonomous execution.
    """

    def __init__(
        self,
        clutch_button: int = 2,
        trigger_button: int = 33,
        trigger_threshold: float = 0.6,
    ) -> None:
        import openvr

        if not 0 <= clutch_button < 64 or not 0 <= trigger_button < 64:
            raise ValueError("Button ids must be in [0, 63]")
        if not 0 < trigger_threshold <= 1:
            raise ValueError("Trigger threshold must be in (0, 1]")
        self.vr = openvr
        self.system = openvr.init(openvr.VRApplication_Background)
        self.clutch_button = clutch_button
        self.trigger_button = trigger_button
        self.trigger_threshold = trigger_threshold
        self.closed = False

    def _trigger_axis_value(self, index: int, state: object) -> float:
        """Read a vendor-advertised analog trigger when no button bit is set."""
        axis_count = int(getattr(self.vr, "k_unControllerStateAxisCount", 5))
        property_base = getattr(self.vr, "Prop_Axis0Type_Int32", None)
        trigger_type = getattr(self.vr, "k_eControllerAxis_Trigger", None)
        axes = getattr(state, "rAxis", ())
        if property_base is None or trigger_type is None:
            return 0.0
        for axis_index in range(min(axis_count, len(axes))):
            try:
                axis_type = self.system.getInt32TrackedDeviceProperty(
                    index, property_base + axis_index
                )
            except Exception:  # Vendor runtimes differ in property support.
                continue
            if axis_type == trigger_type:
                return float(np.clip(axes[axis_index].x, 0, 1))
        return 0.0

    def read(self) -> VRReading:
        """Fetch current tracking and button state without a network round trip."""
        index = self.system.getTrackedDeviceIndexForControllerRole(
            self.vr.TrackedControllerRole_RightHand
        )
        if index == self.vr.k_unTrackedDeviceIndexInvalid:
            return VRReading(np.eye(4), False, False, False, 0, 0.0)
        ok, state, pose = self.system.getControllerStateWithPose(
            self.vr.TrackingUniverseStanding, index
        )
        matrix = np.eye(4)
        matrix[:3] = np.asarray(pose.mDeviceToAbsoluteTracking.m)
        valid = bool(
            ok
            and pose.bPoseIsValid
            and pose.bDeviceIsConnected
            and pose.eTrackingResult == self.vr.TrackingResult_Running_OK
            and self.system.isInputAvailable()
            and np.isfinite(matrix).all()
        )
        buttons = int(state.ulButtonPressed)
        trigger_value = self._trigger_axis_value(index, state)
        return VRReading(
            matrix,
            valid,
            bool(buttons & (1 << self.clutch_button)),
            bool(
                buttons & (1 << self.trigger_button)
                or trigger_value >= self.trigger_threshold
            ),
            buttons,
            trigger_value,
        )

    def close(self) -> None:
        """Release SteamVR exactly once."""
        if not self.closed:
            self.vr.shutdown()
            self.closed = True
