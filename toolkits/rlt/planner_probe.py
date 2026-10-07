# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Verify planner recipes on real CPU physics, not a policy success benchmark."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from rlinf.envs.sim.maniskill.planner_assistance import (
    PegRecovery,
    RecoveryConfig,
    RecoveryFailure,
    handoff_ready,
    read_peg_evidence,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


def run(
    output: Path,
    seeds: list[int],
    case: str,
    video: bool,
    render_backend: str = "cuda:0",
) -> dict:
    """Record every real action and endpoint; never reset within recovery."""
    import gymnasium as gym

    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Require distinct nonnegative reset seeds")
    if case not in {
        "from_reset",
        "held_recovery",
        "dropped_recovery",
        "insertion_fixture",
    }:
        raise ValueError("Unknown planner probe case")

    from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
        register_rlinf_peg_insertion_side_variants,
    )
    from rlinf.envs.sim.maniskill.utils import allow_pci_render_backend

    output.mkdir(parents=True, exist_ok=False)
    register_rlinf_peg_insertion_side_variants()
    allow_pci_render_backend()
    results = []
    for seed in seeds:
        env = gym.make(
            "PegInsertionSideWideClearance-v1",
            num_envs=1,
            obs_mode="state",
            sim_backend="cpu",
            render_backend=render_backend if video else "none",
            control_mode="pd_joint_delta_pos",
            max_episode_steps=500,
            sim_config={"sim_freq": 100, "control_freq": 10},
            render_mode="rgb_array" if video else None,
        )
        recovery = None
        writer = None
        rows = []
        reason = "none"
        try:
            env.reset(seed=seed)
            if video:
                import imageio.v2 as imageio

                writer = imageio.get_writer(
                    output / f"seed_{seed}.mp4", fps=10, ffmpeg_params=["-threads", "2"]
                )
            config = RecoveryConfig(
                protocol="preinsert_handoff"
                if case == "insertion_fixture"
                else "complete"
            )
            recovery = PegRecovery(env, config)
            actions = recovery.actions()
            restarted = False
            while True:
                try:
                    action = next(actions)
                except StopIteration:
                    break
                except RecoveryFailure as exc:
                    reason = str(exc)
                    break
                # Transfer from the initial demonstration to a NEW current-state
                # planner at the insertion boundary, without teleportation.
                if (
                    case in {"held_recovery", "dropped_recovery"}
                    and not restarted
                    and recovery.stage == "insert"
                ):
                    actions.close()
                    recovery.close()
                    if case == "dropped_recovery":
                        for _ in range(15):
                            before = (
                                env.unwrapped.agent.robot.get_qpos()[0]
                                .cpu()
                                .numpy()
                                .copy()
                            )
                            release = np.r_[np.zeros(7), 1].astype(np.float32)
                            _, reward, terminated, truncated, _ = env.step(release)
                            after = (
                                env.unwrapped.agent.robot.get_qpos()[0]
                                .cpu()
                                .numpy()
                                .copy()
                            )
                            rows.append(
                                {
                                    "q_before": before,
                                    "q_after": after,
                                    "action": release,
                                    "reward": float(reward[0]),
                                    "terminated": bool(terminated[0]),
                                    "truncated": bool(truncated[0]),
                                    "stage": "drop_injection",
                                    "source": "perturbation",
                                }
                            )
                            if writer is not None:
                                writer.append_data(env.render()[0].cpu().numpy())
                            if bool(terminated[0]) or bool(truncated[0]):
                                break
                        if bool(terminated[0]) or bool(truncated[0]):
                            break
                    recovery = PegRecovery(env, RecoveryConfig())
                    actions = recovery.actions()
                    restarted = True
                    continue
                before = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy().copy()
                _, reward, terminated, truncated, info = env.step(action)
                after = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy().copy()
                rows.append(
                    {
                        "q_before": before,
                        "q_after": after,
                        "action": action,
                        "reward": float(reward[0]),
                        "terminated": bool(terminated[0]),
                        "truncated": bool(truncated[0]),
                        "stage": recovery.stage,
                        "source": "planner",
                    }
                )
                if writer is not None:
                    writer.append_data(env.render()[0].cpu().numpy())
                if bool(terminated[0]) or bool(truncated[0]):
                    break
            prefix_steps = len(rows)
            fixture_ready = (
                handoff_ready(read_peg_evidence(env), config)
                if case == "insertion_fixture"
                else None
            )
            if fixture_ready:
                # Constant-command negative control: no further planning or policy.
                while int(env.unwrapped.elapsed_steps[0]) < 500:
                    before = (
                        env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy().copy()
                    )
                    hold = np.r_[np.zeros(7), -1].astype(np.float32)
                    _, reward, terminated, truncated, _ = env.step(hold)
                    rows.append(
                        {
                            "q_before": before,
                            "q_after": env.unwrapped.agent.robot.get_qpos()[0]
                            .cpu()
                            .numpy()
                            .copy(),
                            "action": hold,
                            "reward": float(reward[0]),
                            "terminated": bool(terminated[0]),
                            "truncated": bool(truncated[0]),
                            "stage": "hold_control",
                            "source": "constant_control",
                        }
                    )
                    if writer is not None:
                        writer.append_data(env.render()[0].cpu().numpy())
                    if bool(terminated[0]) or bool(truncated[0]):
                        break
            success = read_peg_evidence(env).success
            result = {
                "seed": seed,
                "case": case,
                "success": success,
                "steps": len(rows),
                "source": "planner",
                "failure": reason,
                "restarted_from_held": restarted,
            }
            if case == "insertion_fixture":
                result.update(
                    fixture_ready=fixture_ready,
                    prefix_steps=prefix_steps,
                    hold_steps=len(rows) - prefix_steps,
                    source="planner_prefix_then_constant_control",
                )
            np.savez_compressed(
                output / f"seed_{seed}.npz",
                **{key: np.asarray([row[key] for row in rows]) for key in rows[0]}
                if rows
                else {},
            )
            results.append(result)
            logger.info("Planner probe: %s", result)
        finally:
            if recovery is not None:
                recovery.close()
            if writer is not None:
                writer.close()
            env.close()
    report = {
        "protocol": "planner feasibility; not learner evaluation",
        "results": results,
        "success_rate": sum(r["success"] for r in results) / len(results),
    }
    if case == "insertion_fixture":
        report.update(
            protocol="Insertion fixture feasibility and constant-command control; no learner evaluated",
            fixture_ready_rate=sum(r["fixture_ready"] for r in results) / len(results),
            hold_success_rate=report.pop("success_rate"),
        )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run an explicitly bounded set of training-side initial states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument(
        "--case",
        choices=[
            "from_reset",
            "held_recovery",
            "dropped_recovery",
            "insertion_fixture",
        ],
        default="from_reset",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Requires an idle GPU and working Vulkan driver",
    )
    parser.add_argument("--gpu", type=int, default=2, help="Used only for --video")
    args = parser.parse_args()
    render_backend = "none"
    if args.video:
        used = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(args.gpu),
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        if used > 512:
            parser.error(f"Video GPU {args.gpu} is busy ({used} MiB)")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        bus = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(args.gpu),
                    "--query-gpu=pci.bus_id",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .lower()
        )
        domain, bus_id, slot = bus.split(":")
        render_backend = f"pci:{int(domain, 16):04x}:{bus_id}:{slot}"
    run(args.output, args.seeds, args.case, args.video, render_backend)


if __name__ == "__main__":
    main()
