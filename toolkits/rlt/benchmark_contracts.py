# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Shared transition contracts for porting RLT to manipulation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BenchmarkContract:
    """The action and lifecycle boundary required before online RLT is enabled."""

    name: str
    env_type: str
    action_dim: int
    action_semantics: str
    model: str
    online_rl_status: str
    required_gate: str


CONTRACTS = {
    "robotwin2": BenchmarkContract(
        name="robotwin2",
        env_type="robotwin",
        action_dim=14,
        action_semantics="dual-arm joint targets with two grippers",
        model="openpi-pi05",
        online_rl_status="environment-supported-adapter-pending",
        required_gate="verify post-action observations and task-specific normalization",
    ),
    "robocasa": BenchmarkContract(
        name="robocasa",
        env_type="robocasa",
        action_dim=12,
        action_semantics="relative dual-arm end-effector, gripper, torso, and base control",
        model="openpi",
        online_rl_status="environment-supported-adapter-pending",
        required_gate="freeze controller mode and verify executed-action acknowledgement",
    ),
    "libero_pro": BenchmarkContract(
        name="libero_pro",
        env_type="libero",
        action_dim=7,
        action_semantics="end-effector delta pose and gripper command",
        model="openpi-pi05",
        online_rl_status="environment-supported-adapter-pending",
        required_gate="set LIBERO_TYPE=pro and validate reset-state separation",
    ),
}


@dataclass(frozen=True)
class Transition:
    """One simulator tick retained by replay and intervention accounting."""

    observation: Any
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: Any
    proposed_action: np.ndarray


class OnlineRLTBoundary:
    """Enforce one-tick, no-auto-reset semantics around an RLinf environment."""

    def __init__(self, env: Any, contract: BenchmarkContract, *, num_envs: int):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.env = env
        self.contract = contract
        self.num_envs = num_envs

    def reset(self, **kwargs: Any) -> Any:
        """Reset explicitly so terminal observations are never overwritten."""
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> Transition:
        """Execute exactly one action per environment and preserve done semantics."""
        action = np.asarray(action, dtype=np.float32)
        expected = (self.num_envs, self.contract.action_dim)
        if action.shape != expected:
            raise ValueError(f"Expected action shape {expected}, got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("Actions must be finite")
        observation, reward, terminated, truncated, info = self.env.step(
            action, auto_reset=False
        )
        reward = np.asarray(reward)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
        expected_batch = (self.num_envs,)
        for name, value in (
            ("reward", reward),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            if value.shape != expected_batch:
                raise ValueError(
                    f"Expected {name} shape {expected_batch}, got {value.shape}"
                )
        return Transition(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            proposed_action=action.copy(),
        )
