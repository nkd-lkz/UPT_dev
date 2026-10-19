# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Execution-side perturbations for A-B-A physical-condition experiments.

A perturbation changes how issued commands execute, after the learner commits
its chunk and before the native environment receives it. The learner's ledger
keeps the issued chunk while the acknowledged executed commands reflect the
perturbation, so the commanded-versus-executed deviation that grounding
consumes is produced by the same path that records it.

A schedule assigns one perturbation phase per episode. ``"none:20,delay=2:20,
none:20"`` is the canonical A-B-A protocol: twenty unperturbed episodes,
twenty with a two-tick control delay, then a return to the original condition.
The final phase persists beyond its count. Delay is measured in control ticks;
gain scales commanded joint increments and leaves gripper openings unchanged.
"""

from __future__ import annotations

import re
from collections import deque

import numpy as np

from toolkits.rlt.robodojo_experience import ACTION_DIM, JOINT_MASK

_PHASE = re.compile(
    r"(?:none|delay=(?P<delay>\d+)|gain=(?P<gain>\d+(?:\.\d+)?))"
    r"(?::(?P<episodes>\d+))?"
)


def _command(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
        raise ValueError("Perturbers require finite fourteen-dimensional commands")
    return array.copy()


def parse_perturb_schedule(spec: str) -> list[dict]:
    """Parse comma-separated phases of ``kind[:episodes]`` into a schedule.

    Every phase except the last needs an explicit episode count; the last
    phase, counted or not, covers all remaining episodes. Delays outside
    [1, 20] ticks and gains outside [0.25, 2.0] are rejected because they fall
    outside what the ARX X5 control path can execute meaningfully.
    """
    phases = []
    for part in spec.split(","):
        match = _PHASE.fullmatch(part.strip())
        if match is None:
            raise ValueError(f"Invalid perturbation phase: {part.strip()!r}")
        episodes = match.group("episodes")
        phase: dict = {"episodes": int(episodes) if episodes is not None else None}
        if phase["episodes"] is not None and phase["episodes"] < 1:
            raise ValueError("Phase episode counts must be positive")
        if match.group("delay") is not None:
            delay = int(match.group("delay"))
            if not 1 <= delay <= 20:
                raise ValueError("Delay must be in [1, 20] control ticks")
            phase.update(kind="delay", value=delay)
        elif match.group("gain") is not None:
            gain = float(match.group("gain"))
            if not 0.25 <= gain <= 2.0:
                raise ValueError("Gain must be in [0.25, 2.0]")
            phase.update(kind="gain", value=gain)
        else:
            phase.update(kind="none", value=None)
        phases.append(phase)
    if any(phase["episodes"] is None for phase in phases[:-1]):
        raise ValueError("Only the final phase may omit an episode count")
    return phases


def phase_for_episode(schedule: list[dict], episode_id: int) -> dict:
    """Return the phase covering one episode; the final phase persists."""
    if type(episode_id) is not int or episode_id < 0:
        raise ValueError("Episode index must be a nonnegative integer")
    start = 0
    for phase in schedule:
        if phase["episodes"] is None or episode_id < start + phase["episodes"]:
            return phase
        start += phase["episodes"]
    return schedule[-1]


class DelayLine:
    """Execute each command a fixed number of control ticks late.

    The line is seeded with hold commands at the episode's initial state, so
    the first ``delay`` ticks hold still while the commanded stream catches
    up. It persists across chunks within an episode because the control
    stream is continuous even though decisions arrive chunk by chunk.
    """

    def __init__(self, delay: int, hold: np.ndarray) -> None:
        hold = _command(hold)
        self.delay = delay
        self._queue: deque[np.ndarray] = deque(hold.copy() for _ in range(delay))

    def step(self, command: np.ndarray) -> np.ndarray:
        self._queue.append(_command(command))
        return self._queue.popleft()

    def describe(self) -> dict:
        return {"kind": "delay", "value": self.delay}


class GainShaper:
    """Scale commanded joint increments, emulating controller gain error.

    Each applied joint command advances from the previously applied command by
    ``gain`` times the commanded increment, so error accumulates the way a
    miscalibrated tracking controller accumulates it. Grippers pass through
    unchanged because their openings are commands, not measured responses.
    Gain one is exactly the identity.
    """

    def __init__(self, gain: float, hold: np.ndarray) -> None:
        hold = _command(hold)
        self.gain = gain
        self._previous_command = hold.copy()
        self._previous_applied = hold.copy()

    def step(self, command: np.ndarray) -> np.ndarray:
        command = _command(command)
        applied = command.copy()
        applied[JOINT_MASK] = self._previous_applied[JOINT_MASK] + self.gain * (
            command[JOINT_MASK] - self._previous_command[JOINT_MASK]
        )
        self._previous_command = command
        self._previous_applied = applied.copy()
        return applied

    def describe(self) -> dict:
        return {"kind": "gain", "value": self.gain}


def make_perturber(phase: dict, hold: np.ndarray) -> DelayLine | GainShaper | None:
    """Build one episode's perturber from its phase, or None for an A phase."""
    if phase["kind"] == "none":
        return None
    if phase["kind"] == "delay":
        return DelayLine(phase["value"], hold)
    if phase["kind"] == "gain":
        return GainShaper(phase["value"], hold)
    raise ValueError(f"Unknown perturbation kind: {phase['kind']}")
