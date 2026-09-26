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


class SteamVRController:
    """Read legacy SteamVR grip/trigger bindings; verify mappings with --probe.

    The adapter is experimental: PICO SteamVR profiles must expose Grip and
    SteamVR_Trigger through the legacy controller API. Invalid tracking prevents
    stepping, including during autonomous execution.
    """

    def __init__(self, clutch_button: int = 2, trigger_button: int = 33) -> None:
        import openvr

        if not 0 <= clutch_button < 64 or not 0 <= trigger_button < 64:
            raise ValueError("Button ids must be in [0, 63]")
        self.vr = openvr
        self.system = openvr.init(openvr.VRApplication_Background)
        self.clutch_button = clutch_button
        self.trigger_button = trigger_button
        self.closed = False

    def read(self) -> VRReading:
        """Fetch current tracking and button state without a network round trip."""
        index = self.system.getTrackedDeviceIndexForControllerRole(
            self.vr.TrackedControllerRole_RightHand
        )
        if index == self.vr.k_unTrackedDeviceIndexInvalid:
            return VRReading(np.eye(4), False, False, False, 0)
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
        return VRReading(
            matrix,
            valid,
            bool(buttons & (1 << self.clutch_button)),
            bool(buttons & (1 << self.trigger_button)),
            buttons,
        )

    def close(self) -> None:
        """Release SteamVR exactly once."""
        if not self.closed:
            self.vr.shutdown()
            self.closed = True
