# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Single-environment transitions around the pinned native RoboDojo evaluator."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from toolkits.rlt.robodojo_boundary import (
    ACTION_SEMANTICS,
    FIELDS,
    decode_actions,
    encode_observation,
)


@dataclass(frozen=True)
class Transition:
    """An acknowledged control action, not a physics tick or a predicted action."""

    observation: dict
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class RoboDojoTransitions:
    """Borrow one native EvalEnv and expose reset/action/termination semantics.

    The caller owns and closes the simulator and its policy client. This adapter
    is synchronous and single-threaded; do not step the borrowed environment
    elsewhere. A failed reset/step invalidates the episode until explicit reset.
    Probe episodes are never eligible for training. Training requires declared
    layout hashes disjoint from the protected evaluation layouts.
    """

    def __init__(
        self,
        native: Any,
        *,
        layouts: Mapping[int, str],
        protected_evaluation_hashes: frozenset[str],
        mode: Literal["probe", "training"],
    ) -> None:
        if native.num_envs != 1 or mode not in {"probe", "training"}:
            raise ValueError(
                "Require one native environment and explicit probe/training mode"
            )
        if not layouts or any(type(i) is not int or i < 0 for i in layouts):
            raise ValueError("Require explicit nonnegative native layout ids")
        hashes = list(layouts.values()) + list(protected_evaluation_hashes)
        if any(
            not isinstance(h, str) or not re.fullmatch(r"[a-f0-9]{64}", h)
            for h in hashes
        ):
            raise ValueError("Require SHA256 layout identities")
        if len(set(layouts.values())) != len(layouts):
            raise ValueError("Duplicate layout file hashes")
        if getattr(native, "interact", False):
            raise ValueError("Interactive native tasks need a separate setup contract")
        if mode == "training" and (
            not protected_evaluation_hashes
            or set(layouts.values()) & protected_evaluation_hashes
        ):
            raise ValueError(
                "Training layouts must be disjoint from declared protected evaluation"
            )
        self.native = native
        self.layouts = dict(layouts)
        self.mode = mode
        self._active = False
        self._layout: int | None = None
        self._ticks = 0

    def _observation(self) -> dict:
        raw = self.native.get_obs()
        if (
            self.native.unstable_envs
            or not self.native.scene_manager.layout_manager.layout_valid[0]
        ):
            raise RuntimeError(
                "Native unstable layout is not a valid training transition"
            )
        observation = encode_observation(raw)
        # The native state includes commanded gripper openings, not independent
        # measured gripper response. Retain that distinction in every transition.
        return observation

    def reset(self, layout_id: int) -> tuple[dict, dict]:
        """Restore exactly the declared layout; never replace a failed layout."""
        self._active = False
        if type(layout_id) is not int or layout_id not in self.layouts:
            raise ValueError("Undeclared layout id")
        path = Path(self.native.seed_manager.seed_info[layout_id]["scene_layout"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.layouts[layout_id]:
            raise ValueError("Native layout changed since declaration")
        self.native.reset(seed=[layout_id])
        # Native reset clears reward conditions; run_eval normally registers them.
        # Skipping that setup makes an empty condition list report success at once.
        self.native.run_reward()
        process_score = 0.0
        if hasattr(self.native, "get_score"):
            self.native.get_score()
            process_score = float(self.native.reward_manager.get_score()[0])
        if not math.isfinite(process_score) or not 0 <= process_score <= 100:
            raise RuntimeError("Expected a finite native process score in [0, 100]")
        condition_count = sum(
            len(getattr(self.native.reward_manager, name)[0])
            for name in (
                "check_list",
                "final_check_list",
                "query_list",
                "trigger_check_list",
                "trigger_query_list",
            )
        )
        if not condition_count:
            raise RuntimeError("Native task did not register any reward conditions")
        observation = self._observation()
        if (
            self.native.end_flag[0]
            or not self.native.success[0]
            or self.native.take_action_cnt[0] != 0
        ):
            raise RuntimeError(
                "Native reset did not produce a live, zero-action episode"
            )
        self._layout = layout_id
        self._ticks = 0
        self._active = True
        return observation, {
            "layout_id": layout_id,
            "layout_sha256": digest,
            "training_eligible": self.mode == "training",
            "reward_condition_groups": condition_count,
            "initial_process_score": process_score,
            "state_semantics": "measured_arm_joints_and_commanded_gripper_openings",
        }

    def step(self, action: np.ndarray, *, source: str = "policy") -> Transition:
        """Execute one physical 14-D command and preserve the native end reason.

        Native ``take_action_batch`` performs interpolation and reward updates.
        Its internal end check overwrites ``success`` on time limits, making a
        later read ambiguous. Observe the existing reward call during this one
        synchronous action, before that overwrite, without evaluating rewards
        twice. The temporary observer is restored even when execution fails.
        """
        if not self._active:
            raise RuntimeError(
                "Reset before stepping, including after an ended/failed episode"
            )
        self._active = False
        if source not in {
            "policy",
            "human",
            "planner",
            "reference_teacher",
            "scripted_test",
        }:
            raise ValueError("Unknown executed-action source")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,):
            raise ValueError("Require one 14-D absolute joint/gripper command")
        command = decode_actions(action[None], semantics=ACTION_SEMANTICS, horizon=1)[0]
        actual = np.concatenate([command[name] for name, _ in FIELDS])
        env = self.native
        if (
            env.end_flag[0]
            or not env.success[0]
            or env.take_action_cnt[0] >= env.step_lim
        ):
            raise RuntimeError(
                "Native environment was stepped or ended outside this adapter"
            )
        before = env.take_action_cnt[0]
        if env.unstable_envs or not env.scene_manager.layout_manager.layout_valid[0]:
            raise RuntimeError(
                "Native unstable environment cannot execute another action"
            )
        if before != self._ticks:
            raise RuntimeError("Native environment changed outside this adapter")
        manager = env.reward_manager
        original = manager.get_reward
        observations = []

        def observe_reward(*args: Any, **kwargs: Any) -> Any:
            rewards = original(*args, **kwargs)
            observations.append((float(rewards[0]), bool(env.success[0])))
            return rewards

        # Restore the object's original attribute shape, not a bound-method shadow.
        had_override = "get_reward" in vars(manager)
        previous_override = vars(manager).get("get_reward")
        manager.get_reward = observe_reward
        try:
            env.take_action_batch([command], env_idx_list=[0])
        finally:
            if had_override:
                manager.get_reward = previous_override
            else:
                del manager.get_reward
        if env.take_action_cnt[0] != before + 1 or len(observations) != 1:
            raise RuntimeError("Native execution/reward call contract changed")
        reward, viable = observations[0]
        if not math.isfinite(reward) or reward not in {0.0, 1.0}:
            raise RuntimeError("Expected native binary task reward, not process score")
        process_score = float(manager.get_score()[0])
        if not math.isfinite(process_score) or not 0 <= process_score <= 100:
            raise RuntimeError("Expected a finite native process score in [0, 100]")
        succeeded = viable and reward > 1 - 1e-3
        failed = not viable
        timeout = env.take_action_cnt[0] >= env.step_lim
        terminated = succeeded or failed
        truncated = timeout and not terminated
        if bool(env.end_flag[0]) != (terminated or truncated):
            raise RuntimeError("Native end flag does not match captured outcome")
        observation = self._observation()
        if bool(env.success[0]) != (not failed and not truncated):
            raise RuntimeError(
                "Native outcome changed while collecting final observation"
            )
        self._active = not (terminated or truncated)
        self._ticks = env.take_action_cnt[0]
        return Transition(
            observation,
            reward,
            terminated,
            truncated,
            {
                "layout_id": self._layout,
                "control_step": env.take_action_cnt[0],
                "action_semantics": ACTION_SEMANTICS,
                "source": source,
                "proposed_action": action.copy(),
                "executed_action": actual.copy(),
                "success": succeeded,
                "process_score": process_score,
                "end_reason": "success"
                if succeeded
                else "failure"
                if failed
                else "time_limit"
                if truncated
                else None,
                "bootstrap_allowed": not terminated,
                "training_eligible": self.mode == "training",
                "state_semantics": "measured_arm_joints_and_commanded_gripper_openings",
            },
        )
