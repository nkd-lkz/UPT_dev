# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Executed-work limits for fresh single-environment RLT experiments."""

import math
from dataclasses import dataclass


@dataclass
class ExperimentBudget:
    """Stop at a control-tick limit or an exact critic-update limit.

    The environment enforces the tick boundary, and the learner enforces the
    update boundary. This counter verifies both before final evaluation.
    """

    control_limit: int = 0
    update_limit: int = 0
    control_ticks: int = 0
    critic_updates: int = 0
    actor_updates: int = 0
    episodes: int = 0

    def __post_init__(self) -> None:
        if min(self.control_limit, self.update_limit) < 0 or not (
            bool(self.control_limit) ^ bool(self.update_limit)
        ):
            raise ValueError("Choose exactly one positive executed-work budget")

    def observe(self, env: dict, train: dict) -> bool:
        """Account for one completed rollout, including a budget-truncated episode."""
        if env.get("num_trajectories") != 1:
            raise ValueError(
                "Budget accounting requires exactly one episode per rollout"
            )
        values = [
            env["episode_len"],
            train["rlt/critic_updates_run"],
            train["rlt/actor_updates_run"],
        ]
        if any(not math.isfinite(v) or v < 0 or int(v) != v for v in values):
            raise ValueError(
                "Executed-work counters must be finite nonnegative integers"
            )
        self.control_ticks += int(values[0])
        self.critic_updates += int(values[1])
        self.actor_updates += int(values[2])
        self.episodes += 1
        if (self.control_limit and self.control_ticks > self.control_limit) or (
            self.update_limit and self.critic_updates > self.update_limit
        ):
            raise RuntimeError("Executed work exceeded the declared budget")
        return bool(
            (self.control_limit and self.control_ticks == self.control_limit)
            or (self.update_limit and self.critic_updates == self.update_limit)
        )
