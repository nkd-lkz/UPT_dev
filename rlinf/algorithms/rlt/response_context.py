# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Past-only response weighting for diagnostics, independent of policy training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResponseContextConfig:
    """Bound a seven-joint evidence store and its fixed weighting rules.

    Commands are cumulative controller-scaled joint targets; outcomes are raw
    joint displacements. No hidden condition identifier is accepted. Error
    weighting measures agreement with recent evidence, not physical truth.
    """

    capacity: int = 32
    recent: int = 4
    ridge: float = 1e-4
    half_life: float = 4.0
    error_scale: float = 0.015
    minimum_weight: float = 0.01

    def __post_init__(self) -> None:
        if not 1 <= self.recent <= self.capacity:
            raise ValueError("Require 1 <= recent <= capacity")
        if any(
            not math.isfinite(x) or x <= 0
            for x in (self.ridge, self.half_life, self.error_scale)
        ):
            raise ValueError("Response scales must be positive and finite")
        if not 0 < self.minimum_weight <= 1:
            raise ValueError("minimum_weight must be in (0, 1]")


class ResponseContext:
    """Observe completed commands, then read a response for the next decision.

    ``begin_attempt`` optionally clears evidence at a declared retry boundary.
    ``predict`` and ``snapshot`` never update the store. Error weights can rise
    again when older evidence agrees with newly completed interactions.
    """

    def __init__(self, mode: str, config: ResponseContextConfig | None = None):
        if mode not in ("clear", "retain", "decay", "error"):
            raise ValueError("Unknown response retention mode")
        self.mode = mode
        self.config = config or ResponseContextConfig()
        self.commands = torch.empty(0, 7)
        self.outcomes = torch.empty(0, 7)

    def begin_attempt(self) -> None:
        """Clear only the clear comparator, on every declared attempt boundary."""
        if self.mode == "clear":
            self.commands = torch.empty(0, 7)
            self.outcomes = torch.empty(0, 7)

    @staticmethod
    def _vector(value: torch.Tensor) -> torch.Tensor:
        if value.shape != (7,) or not torch.isfinite(value).all():
            raise ValueError("Expected a finite seven-joint vector")
        return value.detach().float().cpu().clone()

    def observe(self, command: torch.Tensor, outcome: torch.Tensor) -> None:
        """Append a completed transition; never provide a live target to predict."""
        command, outcome = self._vector(command), self._vector(outcome)
        self.commands = torch.cat((self.commands, command[None]))[
            -self.config.capacity :
        ]
        self.outcomes = torch.cat((self.outcomes, outcome[None]))[
            -self.config.capacity :
        ]

    def _fit(self, commands, outcomes, weights):
        energy = (weights[:, None] * commands.square()).sum(0)
        cross = (weights[:, None] * commands * outcomes).sum(0)
        gain = (cross / (energy + self.config.ridge)).clamp(-2, 2)
        return gain, energy / (energy + self.config.ridge)

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Return owned evidence, past-only weights, gain and excitation support."""
        n = len(self.commands)
        weights = torch.ones(n)
        if self.mode == "decay":
            ages = torch.arange(n - 1, -1, -1)
            weights = 2.0 ** (-ages / self.config.half_life)
        elif self.mode == "error" and n >= self.config.recent:
            recent_gain, support = self._fit(
                self.commands[-self.config.recent :],
                self.outcomes[-self.config.recent :],
                torch.ones(self.config.recent),
            )
            # Weakly excited recent joints provide less evidence of mismatch.
            residual = (self.commands * recent_gain - self.outcomes).square()
            error = (residual * support).mean(-1) / self.config.error_scale**2
            weights = torch.exp(-0.5 * error).clamp_min(self.config.minimum_weight)
        gain, support = self._fit(self.commands, self.outcomes, weights)
        return {
            "commands": self.commands.clone(),
            "outcomes": self.outcomes.clone(),
            "weights": weights,
            "gain": gain,
            "support": support,
        }

    def predict(self, command: torch.Tensor) -> torch.Tensor:
        """Predict joint displacement using only previously observed responses."""
        return self.snapshot()["gain"] * self._vector(command)
