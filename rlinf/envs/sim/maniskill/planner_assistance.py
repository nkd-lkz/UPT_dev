# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, privileged-state peg recovery for single-environment experiments.

The planner proposes one command at a time; the caller owns every env.step.
No reset, rewind, reward replacement, or hidden simulation is performed here.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from rlinf.utils.logging import get_logger

logger = get_logger()


@dataclass(frozen=True)
class RecoveryConfig:
    """Training-only assistance budget, expressed in executed control ticks."""

    approach_timeout: int = 180
    stall_ticks: int = 30
    min_progress: float = 0.003
    max_attempts: int = 1
    max_planner_ticks: int = 300
    protocol: str = "complete"
    handoff_min_x: float = -0.16
    handoff_max_yz: float = 0.045

    def __post_init__(self) -> None:
        if (
            min(
                self.approach_timeout,
                self.stall_ticks,
                self.max_attempts,
                self.max_planner_ticks,
            )
            <= 0
            or not np.isfinite(self.min_progress)
            or self.min_progress <= 0
        ):
            raise ValueError("Recovery budgets and progress threshold must be positive")
        if self.protocol not in {"complete", "preinsert_handoff"}:
            raise ValueError("Unknown planner protocol")
        if (
            not np.isfinite([self.handoff_min_x, self.handoff_max_yz]).all()
            or self.handoff_min_x >= 0
            or self.handoff_max_yz <= 0
        ):
            raise ValueError("Handoff requires finite pre-insertion bounds")


@dataclass(frozen=True)
class PegEvidence:
    """Current oracle evidence, never a learner observation or reward."""

    tick: int
    grasped: bool
    success: bool
    hole_x: float
    hole_yz: float
    recoverable: bool


class RecoveryTrigger:
    """Request a bounded attempt after lost grasp, stalled insertion or timeout."""

    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.attempts = 0
        self.ever_grasped = False
        self.best_score = -np.inf
        self.last_progress_tick: int | None = None

    def observe(self, evidence: PegEvidence, *, critical_phase: bool) -> str | None:
        """Return a reason at a chunk boundary; consume no future evidence."""
        e = evidence
        if e.success or not e.recoverable or self.attempts >= self.config.max_attempts:
            return None
        lost = self.ever_grasped and not e.grasped
        self.ever_grasped |= e.grasped
        reason = None
        if lost:
            reason = "lost_grasp"
        elif not critical_phase and e.tick >= self.config.approach_timeout:
            reason = "approach_timeout"
        elif critical_phase and e.grasped:
            score = e.hole_x - e.hole_yz
            if (
                self.last_progress_tick is None
                or score > self.best_score + self.config.min_progress
            ):
                self.best_score = score
                self.last_progress_tick = e.tick
            elif e.tick - self.last_progress_tick >= self.config.stall_ticks:
                reason = "insertion_stall"
        else:
            self.best_score = -np.inf
            self.last_progress_tick = None
        if reason is not None:
            self.attempts += 1
        return reason


class RecoveryFailure(RuntimeError):
    """A recovery recipe cannot continue within its declared contract."""


def handoff_ready(evidence: PegEvidence, config: RecoveryConfig) -> bool:
    """Check the declared held-near-hole fixture, not a collision certificate."""
    return bool(
        evidence.grasped
        and not evidence.success
        and evidence.recoverable
        and evidence.hole_x >= config.handoff_min_x
        and evidence.hole_yz <= config.handoff_max_yz
    )


def prepare_insertion_fixture(env, config: RecoveryConfig | None = None) -> dict:
    """Execute a planner prefix after reset and fail closed on invalid fixtures.

    The caller owns reset. Prefix ticks consume the ordinary episode horizon;
    no simulated state or clock is rewound. Never use this as full-task eval.
    """
    config = config or RecoveryConfig(protocol="preinsert_handoff")
    if config.protocol != "preinsert_handoff":
        raise ValueError("Insertion fixtures require preinsert_handoff")
    recovery = PegRecovery(env, config)
    ticks = 0
    try:
        for action in recovery.actions():
            _, _, terminated, truncated, _ = env.step(action)
            ticks += 1
            if bool(terminated[0]) or bool(truncated[0]):
                raise RecoveryFailure("fixture_ended_before_handoff")
        e = read_peg_evidence(env)
        if not handoff_ready(e, config):
            raise RecoveryFailure("fixture_not_held_near_hole")
        return {"prefix_ticks": ticks, "hole_x": e.hole_x, "hole_yz": e.hole_yz}
    finally:
        recovery.close()


def joint_target_to_delta(
    target: np.ndarray, current: np.ndarray, *, scale: float = 0.1
) -> np.ndarray:
    """Convert absolute arm targets to normalized current-state delta commands."""
    target, current = np.asarray(target), np.asarray(current)
    if target.shape != (7,) or current.shape != (7,):
        raise ValueError("Panda recovery requires seven arm joints")
    if not np.isfinite(target).all() or not np.isfinite(current).all() or scale <= 0:
        raise ValueError("Nonfinite joint state or invalid controller scale")
    return np.clip((target - current) / scale, -1, 1).astype(np.float32)


def read_peg_evidence(env) -> PegEvidence:
    """Read current simulator truth for the assistance supervisor only."""
    base = env.unwrapped
    head = (base.box_hole_pose.inv() * base.peg_head_pose).p[0].cpu().numpy()
    peg = base.peg.pose.p[0].cpu().numpy()
    q = base.agent.robot.get_qpos()[0].cpu().numpy()
    # Conservative table workspace filter, not a collision/safety certificate.
    reachable = bool(
        np.isfinite(np.concatenate([head, peg, q])).all()
        and -0.45 <= peg[0] <= 0.45
        and abs(peg[1]) <= 0.45
        and 0.0 <= peg[2] <= 0.5
    )
    return PegEvidence(
        tick=int(base.elapsed_steps[0]),
        grasped=bool(base.agent.is_grasping(base.peg)[0]),
        success=bool(base.evaluate()["success"][0]),
        hole_x=float(head[0]),
        hole_yz=float(np.linalg.norm(head[1:])),
        recoverable=reachable,
    )


class PegRecovery:
    """Generate executed-state feedback commands for regrasp and insertion.

    Supports only one Panda with normalized pd_joint_delta_pos. The recipe
    uses object poses and dimensions from simulation. MPLib does not model
    arbitrary environmental contacts here; this is not a real-robot controller.
    """

    def __init__(self, env, config: RecoveryConfig):
        from mani_skill.examples.motionplanning.panda.motionplanner import (
            PandaArmMotionPlanningSolver,
        )

        self.env = env.unwrapped
        self.config = config
        if self.env.num_envs != 1 or self.env.control_mode != "pd_joint_delta_pos":
            raise ValueError("Planner assistance requires one pd_joint_delta_pos env")
        arm = self.env.agent.controller.controllers["arm"].config
        if (
            not arm.use_delta
            or arm.use_target
            or not arm.normalize_action
            or not np.allclose(arm.lower, -0.1)
            or not np.allclose(arm.upper, 0.1)
        ):
            raise ValueError(
                "Unsupported controller: require normalized current-q delta ±0.1"
            )
        self.solver = PandaArmMotionPlanningSolver(
            env,
            vis=False,
            print_env_info=False,
            visualize_target_grasp_pose=False,
            base_pose=self.env.agent.robot.pose.sp,
            joint_vel_limits=0.5,
            joint_acc_limits=0.5,
        )
        self.ticks = 0
        self.stage = "ready"
        self.gripper = -1.0 if read_peg_evidence(env).grasped else 1.0

    def close(self) -> None:
        """Release the planner without touching the caller's environment."""
        self.solver.close()

    def _command(self, target: np.ndarray) -> np.ndarray:
        if self.ticks >= self.config.max_planner_ticks:
            raise RecoveryFailure("planner_tick_budget")
        if not read_peg_evidence(self.env).recoverable:
            raise RecoveryFailure("outside_recovery_workspace")
        q = self.env.agent.robot.get_qpos()[0, :7].cpu().numpy()
        self.ticks += 1
        return np.r_[joint_target_to_delta(target, q), self.gripper].astype(np.float32)

    def _hold(self, gripper: float, ticks: int) -> Iterator[np.ndarray]:
        self.gripper = gripper
        q = self.env.agent.robot.get_qpos()[0, :7].cpu().numpy().copy()
        for _ in range(ticks):
            yield self._command(q)

    def _move(self, pose, *, require_grasp: bool = False) -> Iterator[np.ndarray]:
        try:
            result = self.solver.move_to_pose_with_screw(pose, dry_run=True)
        except (RuntimeError, ValueError) as exc:
            raise RecoveryFailure(
                f"planning_error:{self.stage}:{type(exc).__name__}"
            ) from exc
        if not isinstance(result, dict) or result.get("status") != "Success":
            raise RecoveryFailure(f"planning_failed:{self.stage}")
        path = result["position"]
        if len(path) == 0:
            raise RecoveryFailure("empty_plan")
        for target in np.concatenate([path, np.repeat(path[-1:], 5, axis=0)]):
            if require_grasp and not read_peg_evidence(self.env).grasped:
                raise RecoveryFailure(f"lost_grasp:{self.stage}")
            yield self._command(target[:7])

    def actions(self) -> Iterator[np.ndarray]:
        """Yield one command, then read the state after the caller executes it."""
        import sapien
        from mani_skill.examples.motionplanning.base_motionplanner.utils import (
            compute_grasp_info_by_obb,
            get_actor_obb,
        )

        env = self.env
        evidence = read_peg_evidence(env)
        if evidence.success:
            return
        if not evidence.recoverable:
            raise RecoveryFailure("outside_recovery_workspace")
        if not evidence.grasped:
            self.stage = "open"
            yield from self._hold(1, 6)
            closing = (
                env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
            )
            grasp = compute_grasp_info_by_obb(
                get_actor_obb(env.peg),
                approaching=np.array([0, 0, -1]),
                target_closing=closing,
                depth=0.025,
            )
            pose = env.agent.build_grasp_pose(
                np.array([0, 0, -1]), grasp["closing"], grasp["center"]
            )
            length = float(env.peg_half_sizes[0, 0])
            pose = pose * sapien.Pose([-max(0.05, length / 2 + 0.01), 0, 0])
            self.stage = "reach"
            yield from self._move(pose * sapien.Pose([0, 0, -0.06]))
            self.stage = "grasp"
            yield from self._move(pose)
            yield from self._hold(-1, 8)
            if not read_peg_evidence(env).grasped:
                raise RecoveryFailure("grasp_not_established")
        self.gripper = -1
        # Recompute the peg-to-TCP transform after each real execution segment.
        # Pull back before realignment, rather than continuing to push a jam.
        length = float(env.peg_half_sizes[0, 0])
        for _ in range(3):
            self.stage = "preinsert"
            pose = (
                env.goal_pose.sp
                * sapien.Pose([-length - 0.02, 0, 0])
                * env.peg.pose.sp.inv()
                * env.agent.tcp.pose.sp
            )
            yield from self._move(pose, require_grasp=True)
        if self.config.protocol == "preinsert_handoff":
            if not handoff_ready(read_peg_evidence(env), self.config):
                raise RecoveryFailure("handoff_not_held_near_hole")
            self.stage = "handoff"
            return
        self.stage = "insert"
        pose = (
            env.goal_pose.sp
            * sapien.Pose([0.03, 0, 0])
            * env.peg.pose.sp.inv()
            * env.agent.tcp.pose.sp
        )
        yield from self._move(pose, require_grasp=True)
        self.stage = "verify"
        if not read_peg_evidence(env).success:
            raise RecoveryFailure("insertion_not_successful")


class PlannerAssistance:
    """Own one training episode's trigger, recipe, and accounting."""

    def __init__(self, env, config: RecoveryConfig):
        self.env = env
        self.config = config
        self.trigger = RecoveryTrigger(config)
        self.recovery: PegRecovery | None = None
        self.iterator: Iterator[np.ndarray] | None = None
        self.executed_ticks = 0
        self.reason = "none"
        self.failure = "none"
        self.handoffs = 0
        self.handoff_holds = 0
        self._hold_until_boundary = False

    def begin_chunk(self, *, critical_phase: bool) -> None:
        """Check past progress only when no recovery is already in progress."""
        if self._hold_until_boundary:
            self._hold_until_boundary = False
            # A fresh observation must reach the policy before another attempt.
            return
        if self.iterator is not None:
            return
        reason = self.trigger.observe(
            read_peg_evidence(self.env), critical_phase=critical_phase
        )
        if reason is not None:
            self.reason = reason
            logger.info(
                "Planner takeover: reason=%s tick=%s attempt=%s",
                reason,
                read_peg_evidence(self.env).tick,
                self.trigger.attempts,
            )
            try:
                self.recovery = PegRecovery(self.env, self.config)
            except RuntimeError as exc:
                self.failure = f"planner_initialization:{type(exc).__name__}"
                logger.warning("Planner recovery failed: %s", self.failure)
                return
            self.iterator = self.recovery.actions()

    def action(self, policy_action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Choose the next command; failed recovery hands control back explicitly."""
        if self._hold_until_boundary:
            return self._handoff_hold()
        if self.iterator is None:
            return policy_action, False
        try:
            action = next(self.iterator)
            self.executed_ticks += 1
            return action, True
        except StopIteration:
            if self.config.protocol == "preinsert_handoff":
                self.handoffs += 1
                self._hold_until_boundary = True
            self.close()
        except RecoveryFailure as exc:
            self.failure = str(exc)
            logger.warning(
                "Planner recovery failed: %s; executed_ticks=%s",
                self.failure,
                self.executed_ticks,
            )
            self._hold_until_boundary = self.config.protocol == "preinsert_handoff"
            self.close()
        if self._hold_until_boundary:
            return self._handoff_hold()
        return policy_action, False

    def _handoff_hold(self) -> tuple[np.ndarray, bool]:
        self.executed_ticks += 1
        self.handoff_holds += 1
        grip = -1.0 if read_peg_evidence(self.env).grasped else 1.0
        return np.r_[np.zeros(7), grip].astype(np.float32), True

    def close(self) -> None:
        """Idempotently release the current recipe, retaining episode counters."""
        if self.iterator is not None:
            self.iterator.close()
            self.iterator = None
        if self.recovery is not None:
            self.recovery.close()
            self.recovery = None
