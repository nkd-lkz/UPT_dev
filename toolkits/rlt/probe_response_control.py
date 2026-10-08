# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Test past response gains in closed-loop tracking, not insertion or RLT training."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from rlinf.algorithms.rlt.response_context import ResponseContext
from toolkits.rlt.probe_memory_conditions import (
    contact_observation,
    create_env,
    phase_schedule,
    prepare_contact,
    validate_contact_trace,
    write_report,
)

METHODS = ("fixed", "clear", "retain", "decay", "error")


class ResponseTracker:
    """Choose a bounded command from current goal error and completed evidence.

    The fixed prior is fitted on earlier training pairs, shared by all methods.
    New commands are observed only after execution. Stage and stiffness labels
    are deliberately absent from this interface.
    """

    def __init__(self, method: str, prior: torch.Tensor):
        if method not in METHODS:
            raise ValueError("Unknown tracking method")
        if prior.shape != (7,) or not torch.isfinite(prior).all() or (prior <= 0).any():
            raise ValueError("Require seven positive finite prior gains")
        self.prior = prior.detach().float().cpu().clone()
        self.memory = None if method == "fixed" else ResponseContext(method)

    def begin_attempt(self) -> None:
        """Use the same declared retry boundary for every comparator."""
        if self.memory is not None:
            self.memory.begin_attempt()

    def command(self, error: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute an action before the simulator produces the next outcome."""
        if error.shape != (7,) or not torch.isfinite(error).all():
            raise ValueError("Require a finite seven-joint goal error")
        gain = self.prior.clone()
        if self.memory is not None:
            evidence = self.memory.snapshot()
            # Ridge toward the shared prior rather than toward zero. Empty or
            # weakly excited histories cannot create a large inverse gain.
            gain = evidence["gain"] + (1 - evidence["support"]) * self.prior
        gain = gain.clamp(0.25, 2.0)
        proposed = error.detach().float().cpu() / gain
        action = proposed.clamp(-0.08, 0.08)
        return action, {
            "gain": gain.tolist(),
            "clipped_joints": int((proposed.abs() > 0.08).sum()),
        }

    def observe(self, command: torch.Tensor, displacement: torch.Tensor) -> None:
        """Store the command actually executed, not the requested goal change."""
        if self.memory is not None:
            self.memory.observe(command, displacement)


def fit_prior(rows: list[dict]) -> torch.Tensor:
    """Use only original training pair IDs; no new control outcomes are read."""
    training = [r for r in rows if 0 <= int(r["pair"]) < 32]
    if not training:
        raise ValueError("Missing training pairs for the fixed comparator")
    commands = torch.stack([r["command"] for r in training])
    targets = torch.stack([r["target"] for r in training])
    if commands.shape != targets.shape or commands.shape[1:] != (7,):
        raise ValueError("Require seven-joint commands and targets")
    if not torch.isfinite(commands).all() or not torch.isfinite(targets).all():
        raise ValueError("Nonfinite prior data")
    return ((commands * targets).sum(0) / (commands.square().sum(0) + 1e-4)).clamp(
        0.25, 2.0
    )


def target_offsets(seed: int) -> torch.Tensor:
    """Generate the same bounded joint-space path for all methods and attempts."""
    rng = torch.Generator().manual_seed(seed)
    phases = torch.rand(7, generator=rng) * (2 * torch.pi)
    steps = torch.arange(1, 25)[:, None] * (2 * torch.pi / 24)
    return 0.03 * (torch.sin(steps + phases) - torch.sin(phases))


def collect_control(
    env, output: Path, prior: torch.Tensor, *, stages: str, seeds: int, smoke: bool
) -> dict:
    """Run paired trajectories, preserving contact failures in aggregate results."""
    rows = []
    first_seed = 58001 if smoke else 58101
    for seed in range(first_seed, first_seed + seeds):
        offsets = target_offsets(seed)
        for schedule in phase_schedule():
            fixed = len(set(schedule["contact"])) == 1
            if (stages == "fixed") != fixed:
                continue
            fingerprints, initial_positions = {}, {}
            for method in METHODS:
                tracker = ResponseTracker(method, prior)
                attempts = []
                for attempt, (grasped, stiffness) in enumerate(
                    zip(schedule["contact"], schedule["stiffness"])
                ):
                    preparation = prepare_contact(env, seed, stiffness, grasped=grasped)
                    initial = (
                        env.unwrapped.agent.robot.get_qpos()[0, :7]
                        .detach()
                        .cpu()
                        .clone()
                    )
                    if attempt not in fingerprints:
                        fingerprints[attempt] = preparation["state_sha256"]
                        initial_positions[attempt] = initial
                    if preparation["state_sha256"] != fingerprints[
                        attempt
                    ] or not torch.equal(initial, initial_positions[attempt]):
                        raise ValueError(
                            "Initial state differs across tracking comparators"
                        )
                    controller = env.unwrapped.agent.controller.controllers[
                        "arm"
                    ].config
                    if (
                        not controller.normalize_action
                        or not torch.allclose(
                            torch.as_tensor(controller.lower), torch.tensor(-0.1)
                        )
                        or not torch.allclose(
                            torch.as_tensor(controller.upper), torch.tensor(0.1)
                        )
                    ):
                        raise ValueError(
                            "Tracking requires normalized +/-0.1 joint delta scaling"
                        )
                    tracker.begin_attempt()
                    trace, records = [], []
                    for offset in offsets:
                        start = (
                            env.unwrapped.agent.robot.get_qpos()[0, :7]
                            .detach()
                            .cpu()
                            .clone()
                        )
                        goal = initial + offset
                        command, decision = tracker.command(goal - start)
                        action = torch.cat((command, torch.tensor([-1.0])))
                        for _ in range(10):
                            _, _, terminated, truncated, _ = env.step(action.numpy())
                            if bool(terminated.any()) or bool(truncated.any()):
                                raise ValueError("Tracking prefix ended early")
                            trace.append(contact_observation(env))
                        end = (
                            env.unwrapped.agent.robot.get_qpos()[0, :7]
                            .detach()
                            .cpu()
                            .clone()
                        )
                        if not torch.isfinite(end).all():
                            raise ValueError("Nonfinite tracking state")
                        tracker.observe(command, end - start)
                        records.append(
                            {
                                "start": start.tolist(),
                                "goal": goal.tolist(),
                                "end": end.tolist(),
                                "command": command.tolist(),
                                **decision,
                                "mse": float((goal - end).square().mean()),
                                "within_5mrad": bool((goal - end).abs().max() <= 0.005),
                            }
                        )
                    audit = validate_contact_trace(trace, grasped=grasped)
                    attempts.append(
                        {
                            "attempt": attempt,
                            "preparation": preparation,
                            "contact_audit": audit,
                            "contact_trace": trace,
                            "records": records,
                            "mse": sum(r["mse"] for r in records) / len(records),
                            "first_four_mse": sum(r["mse"] for r in records[:4]) / 4,
                            "within_5mrad_fraction": sum(
                                r["within_5mrad"] for r in records
                            )
                            / len(records),
                            "clipped_joint_fraction": sum(
                                r["clipped_joints"] for r in records
                            )
                            / (7 * len(records)),
                            "command_energy": sum(
                                sum(x * x for x in r["command"]) for r in records
                            ),
                            "control_ticks": 260,
                        }
                    )
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        **schedule,
                        "attempts": attempts,
                        "contact_valid": all(
                            a["contact_audit"]["valid"] for a in attempts
                        ),
                        "mse": sum(a["mse"] for a in attempts) / 3,
                        "control_ticks": 780,
                    }
                )
                write_report(
                    output,
                    {
                        "completed": False,
                        "scope": "tracking diagnostic in progress",
                        "rows": rows,
                    },
                )
    return {
        "completed": True,
        "scope": "paired closed-loop joint tracking; privileged initial scenes; no insertion, RLT learning or intervention result",
        "prior_gain": prior.tolist(),
        "methods": list(METHODS),
        "stages": stages,
        "smoke": smoke,
        "seeds": seeds,
        "all_initial_states_matched": True,
        "invalid_contact_streams": sum(not r["contact_valid"] for r in rows),
        "control_ticks": sum(r["control_ticks"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--stages", choices=("fixed", "changing"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seeds <= 12:
        raise ValueError("Use 1..12 seeds")
    torch.set_num_threads(2)
    data = torch.load(args.dataset, weights_only=True, map_location="cpu")
    prior = fit_prior(data["rows"])
    args.output.mkdir(parents=True, exist_ok=False)
    env, lease = create_env(args.gpu, args.output)
    try:
        result = collect_control(
            env,
            args.output,
            prior,
            stages=args.stages,
            seeds=1 if args.smoke else args.seeds,
            smoke=args.smoke,
        )
        result["prior_data_sha256"] = hashlib.sha256(
            args.dataset.read_bytes()
        ).hexdigest()
        write_report(args.output, result)
    finally:
        env.close()
        lease.close()


if __name__ == "__main__":
    main()
