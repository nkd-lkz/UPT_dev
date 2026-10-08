# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Matched-state response and A-B-A diagnostics; these are not policy evaluations."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import default_collate

from rlinf.algorithms.rlt.interaction_memory import (
    InteractionMemory,
    InteractionMemoryConfig,
)
from rlinf.algorithms.rlt.response_context import ResponseContext
from rlinf.models.embodiment.modules.rlt_memory_encoder import response_features
from toolkits.rlt.wait_for_gpu import gpu_available


def pair_split(pair: int) -> str:
    """Keep all states, histories and conditions of a seed in one partition."""
    return "train" if pair < 32 else "validation" if pair < 40 else "test"


def tensor_digest(value) -> str:
    """Fingerprint a nested simulator state without pickle metadata."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(key.encode())
                visit(item[key])
        else:
            tensor = torch.as_tensor(item).detach().cpu().contiguous()
            digest.update(str((tensor.shape, tensor.dtype)).encode())
            digest.update(tensor.numpy().tobytes())

    visit(value)
    return digest.hexdigest()


def wrong_history(batch: dict) -> dict:
    """Swap paired condition histories, preserving the exact current input."""
    result = {key: value.clone() for key, value in batch.items()}
    lookup = {}
    for i, (pair, query, condition) in enumerate(
        zip(
            batch["pair"].tolist(),
            batch["query_id"].tolist(),
            batch["condition"].tolist(),
        )
    ):
        key = (pair, query, condition)
        if key in lookup:
            raise ValueError("Duplicate pair/query/condition")
        lookup[key] = i
    for (pair, query, condition), i in lookup.items():
        j = lookup.get((pair, query, 1 - condition))
        if j is None:
            raise ValueError("Missing paired condition")
        for key in ("memory_query", "velocity", "command"):
            if not torch.equal(batch[key][i], batch[key][j]):
                raise ValueError(
                    "Current observation or command differs across conditions"
                )
        # History controls and padding must match. Only the observed response
        # and history states may differ between real execution conditions.
        p, h, a = 9, 10, 8
        if not torch.equal(batch["memory_valid"][i], batch["memory_valid"][j]):
            raise ValueError("History validity differs")
        controls = [
            sorted(
                tuple(row)
                for row in batch["memory_events"][idx, :, p : p + h * a].tolist()
            )
            for idx in (i, j)
        ]
        if controls[0] != controls[1]:
            raise ValueError("History commands differ")
        for key in ("memory_events", "memory_valid"):
            result[key][i] = batch[key][j]
    return result


class MatchedResponseProbe(nn.Module):
    """Predict displacement from q, velocity, command and a response descriptor."""

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(9 + 9 + 7 + 14, 64), nn.SiLU(), nn.Linear(64, 7)
        )

    def forward(self, batch: dict, *, history: bool) -> torch.Tensor:
        context = response_features(batch, InteractionMemoryConfig())
        if not history:
            context = torch.zeros_like(context)
        # Explicit allowlist: condition, pair, query_id and target are labels only.
        return self.head(
            torch.cat(
                (batch["memory_query"], batch["velocity"], batch["command"], context),
                -1,
            )
        )


class StandardizedResponseProbe(MatchedResponseProbe):
    """Use training-only input statistics while preserving the original head."""

    def __init__(self, training_batch: dict):
        super().__init__()
        values = self.features(training_batch)
        self.register_buffer("input_mean", values.mean(0))
        self.register_buffer("input_scale", values.std(0).clamp_min(1e-3))

    @staticmethod
    def features(batch: dict) -> torch.Tensor:
        """Read the same input allowlist as the unscaled diagnostic."""
        return torch.cat(
            (
                batch["memory_query"],
                batch["velocity"],
                batch["command"],
                response_features(batch, InteractionMemoryConfig()),
            ),
            -1,
        )

    def forward(self, batch: dict, *, history: bool) -> torch.Tensor:
        values = (self.features(batch) - self.input_mean) / self.input_scale
        if not history:
            values = torch.cat((values[:, :25], torch.zeros_like(values[:, 25:])), -1)
        return self.head(values)


def metrics(prediction: torch.Tensor, batch: dict) -> dict:
    """Report raw-radian squared error with pair-level measurements."""
    error = (prediction - batch["target"]).square().mean(-1)
    values = {
        str(p): float(error[batch["pair"] == p].mean())
        for p in batch["pair"].unique().tolist()
    }
    return {"mse": float(error.mean()), "pair_mse": values}


def analyze_matched(
    rows: list[dict], *, updates: int = 512, standardize: bool = False
) -> dict:
    """Train same-capacity heads; final test uses the predeclared final update."""
    batches = {
        split: default_collate([r for r in rows if pair_split(int(r["pair"])) == split])
        for split in ("train", "validation", "test")
    }
    for batch in batches.values():
        wrong_history(batch)
    train = batches["train"]
    u, d = train["command"], train["target"]
    global_gain = ((u * d).sum(0) / (u.square().sum(0) + 1e-4)).clamp(-2, 2)
    fixed = {}
    for split, batch in batches.items():
        wrong = wrong_history(batch)
        fixed[split] = {
            "global_gain_from_training": metrics(global_gain * batch["command"], batch),
            "correct_response": metrics(
                response_features(batch, InteractionMemoryConfig())[:, :7]
                * batch["command"],
                batch,
            ),
            "wrong_response": metrics(
                response_features(wrong, InteractionMemoryConfig())[:, :7]
                * batch["command"],
                batch,
            ),
        }
    learned = []
    scale = train["target"].std(0).clamp_min(1e-3)
    for seed in (5201, 5202, 5203):
        for enabled in (False, True):
            torch.manual_seed(seed)
            model = (
                StandardizedResponseProbe(train)
                if standardize
                else MatchedResponseProbe()
            )
            initial_hash = tensor_digest(model.state_dict())
            optim = torch.optim.Adam(model.parameters(), lr=3e-4)
            sampler = torch.Generator().manual_seed(seed + 100)
            sampling_hash = hashlib.sha256()
            for _ in range(updates):
                ix = torch.randint(len(u), (32,), generator=sampler)
                sampling_hash.update(ix.numpy().tobytes())
                batch = {key: value[ix] for key, value in train.items()}
                loss = (
                    (model(batch, history=enabled) - batch["target"] / scale)
                    .square()
                    .mean()
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite response loss")
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), 10, error_if_nonfinite=True
                )
                optim.step()
            with torch.no_grad():
                result = {
                    split: metrics(model(batch, history=enabled) * scale, batch)
                    for split, batch in batches.items()
                }
                result["test_wrong_history"] = metrics(
                    model(wrong_history(batches["test"]), history=enabled) * scale,
                    batches["test"],
                )
                result["test_history_off"] = metrics(
                    model(batches["test"], history=False) * scale, batches["test"]
                )
            learned.append(
                {
                    "seed": seed,
                    "history": enabled,
                    "updates": updates,
                    "standardized_inputs": standardize,
                    "initial_sha256": initial_hash,
                    "samples_sha256": sampling_hash.hexdigest(),
                    "metrics": result,
                }
            )
    return {
        "fixed": fixed,
        "learned": learned,
        "units": "mean squared raw arm-joint displacement; not control success",
        "selection": "final step only; no test selection",
    }


def analyze_stream(
    commands: torch.Tensor, outcomes: torch.Tensor, boundaries: list[int]
) -> dict:
    """Replay a recorded stream using identical attempt boundaries in every arm."""
    if commands.shape != outcomes.shape or commands.ndim != 2 or commands.shape[1] != 7:
        raise ValueError("Expected paired [T, 7] evidence")
    if not torch.isfinite(commands).all() or not torch.isfinite(outcomes).all():
        raise ValueError("Nonfinite stream")
    if (
        boundaries != sorted(set(boundaries))
        or not boundaries
        or boundaries[0] != 0
        or boundaries[-1] >= len(commands)
    ):
        raise ValueError("Invalid attempt boundaries")
    result = {}
    for mode in ("clear", "retain", "decay", "error"):
        memory = ResponseContext(mode)
        predictions, weights = [], []
        for t, (u, d) in enumerate(zip(commands, outcomes)):
            if t in boundaries:
                memory.begin_attempt()
            predictions.append(memory.predict(u))
            weights.append(memory.snapshot()["weights"].tolist())
            memory.observe(u, d)
        error = (torch.stack(predictions) - outcomes).square().mean(-1)
        edges = boundaries + [len(commands)]
        phases = [error[a:b] for a, b in zip(edges, edges[1:])]
        result[mode] = {
            "mse": float(error.mean()),
            "error_by_chunk": error.tolist(),
            "weights_before_decision": weights,
            "phase_mse": [float(e.mean()) for e in phases],
            "first_four_mse": [float(e[:4].mean()) for e in phases],
            "predictions": torch.stack(predictions).tolist(),
        }
    return result


def synthetic(output: Path) -> None:
    """Exercise A-B-A and stationary/noisy controls; never call this simulation."""
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in range(5301, 5307):
        rng = torch.Generator().manual_seed(seed)
        commands = torch.randn(36, 7, generator=rng) * 0.05
        noise = torch.randn(36, 7, generator=rng)
        for name, sigma in (
            ("stationary", 0.002),
            ("A_B_A", 0.002),
            ("stationary_noisy", 0.012),
        ):
            gain = torch.ones(36, 1)
            if name == "A_B_A":
                gain[12:24] = 0.2
            outcomes = commands * gain + noise * sigma
            rows.append(
                {
                    "seed": seed,
                    "condition": name,
                    "results": analyze_stream(commands, outcomes, [0, 12, 24]),
                }
            )
    write_report(
        output,
        {
            "scope": "synthetic linear plant; not robot or task control",
            "rows": rows,
            "completed": True,
        },
    )


def write_report(output: Path, report: dict) -> None:
    """Publish results only after the complete diagnostic succeeds."""
    target = output / "results.json"
    temp = output / "results.tmp"
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(target)


def create_env(gpu: int, output: Path):
    """Acquire the shared project lease and require an idle physical GPU."""
    lease = open(f"/tmp/rlt-atomic-gpu{gpu}.lock", "a")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def query(option):
            return subprocess.check_output(
                ["nvidia-smi", "-i", str(gpu), option, "--format=csv,noheader,nounits"],
                text=True,
                timeout=15,
            )

        if not gpu_available(
            query("--query-gpu=memory.used"), query("--query-compute-apps=pid")
        ):
            raise RuntimeError("GPU is busy; refusing to allocate")
        uuid, bus = (s.strip() for s in query("--query-gpu=uuid,pci.bus_id").split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        os.environ["SAPIEN_VULKAN_LIBRARY_PATH"] = (
            "/home/luokz/.local/rlinf-vulkan/lib/libvulkan.so.1.4.357"
        )
        icd = output / "nvidia_icd.json"
        icd.write_text(
            json.dumps(
                {
                    "file_format_version": "1.0.0",
                    "ICD": {
                        "library_path": "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0",
                        "api_version": "1.2.0",
                    },
                }
            )
        )
        os.environ["VK_DRIVER_FILES"] = os.environ["VK_ICD_FILENAMES"] = str(icd)
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
            PANDA_WIDE_WRISTCAM_UID,
            register_rlinf_peg_insertion_side_variants,
        )
        from rlinf.envs.sim.maniskill.utils import allow_pci_render_backend

        if torch.cuda.device_count() != 1 or str(
            torch.cuda.get_device_properties(0).uuid
        ).removeprefix("GPU-") != uuid.removeprefix("GPU-"):
            raise RuntimeError("CUDA device isolation mismatch")
        domain, bus_id, slot = bus.lower().split(":")
        register_rlinf_peg_insertion_side_variants()
        allow_pci_render_backend()
        env = gym.make(
            "PegInsertionSideWideClearanceObserverWideWrist-v1",
            robot_uids=PANDA_WIDE_WRISTCAM_UID,
            num_envs=1,
            obs_mode="state_dict",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            render_mode=None,
            sim_backend="physx_cpu",
            render_backend=f"pci:{int(domain, 16):04x}:{bus_id}:{slot}",
            max_episode_steps=500,
            sim_config={"sim_freq": 100, "control_freq": 10},
        )
        return env, lease
    except BaseException:
        lease.close()
        raise


def reset_condition(
    env, seed: int, stiffness: float
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Restore one seeded query scene and set actual arm drive properties."""
    env.reset(seed=seed)
    arm = env.unwrapped.agent.controller.controllers["arm"]
    if arm.config.use_target or not arm.config.use_delta:
        raise ValueError("Diagnostic requires current-qpos delta targets")
    arm.config.stiffness = stiffness
    arm.set_drive_property()
    robot = env.unwrapped.agent.robot
    return (
        robot.get_qpos()[0, :9].detach().cpu().clone(),
        robot.get_qvel()[0, :9].detach().cpu().clone(),
        tensor_digest(env.unwrapped.get_state_dict()),
    )


def execute(env, command: torch.Tensor) -> torch.Tensor:
    """Execute ten identical normalized commands; reject incomplete prefixes."""
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(command.numpy())
        if bool(terminated.any()) or bool(truncated.any()):
            raise RuntimeError(
                "Diagnostic action prefix terminated before its endpoint"
            )
    result = env.unwrapped.agent.robot.get_qpos()[0, :9].detach().cpu().clone()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite simulator state")
    return result


def collect_matched(env, output: Path, *, pairs: int) -> None:
    """Collect real histories, restore the common query scene, then query outcomes.

    Restoring a scene while retaining calibration evidence is a privileged
    diagnostic protocol, not a normal task rollout or a deployment capability.
    """
    rows, audits = [], []
    for pair in range(pairs):
        generator = torch.Generator().manual_seed(54000 + pair)
        commands = (torch.rand(12, 8, generator=generator) - 0.5) * 0.16
        commands[:, 7] = 1
        hashes = []
        for condition, stiffness in enumerate((250.0, 1000.0)):
            start, _, initial_hash = reset_condition(env, 55000 + pair, stiffness)
            memory = InteractionMemory(InteractionMemoryConfig())
            memory.begin_attempt(f"calibration-{pair}-{condition}")
            for command in commands[:8]:
                end = execute(env, command)
                memory.append_completed(start, command.repeat(10, 1), end)
                start = end
            for query_id, command in enumerate(commands[8:]):
                query, velocity, query_hash = reset_condition(
                    env, 55000 + pair, stiffness
                )
                if query_hash != initial_hash:
                    raise ValueError("Seeded query restoration changed simulator state")
                row = {
                    **memory.snapshot(query),
                    "velocity": velocity,
                    "command": command[:7].clone(),
                    "pair": torch.tensor(pair),
                    "query_id": torch.tensor(query_id),
                    "condition": torch.tensor(condition),
                }
                # 10 commands x 0.1 rad scaling -> cumulative target == command.
                row["target"] = (execute(env, command) - query)[:7]
                rows.append(row)
                hashes.append(query_hash)
        if len(set(hashes)) != 1:
            raise ValueError("Current simulator state differs between conditions")
        audits.append({"pair": pair, "query_state_sha256": hashes[0], "queries": 4})
        (output / "progress.json").write_text(
            json.dumps({"completed_pairs": pair + 1, "planned_pairs": pairs})
        )
    wrong_history(default_collate(rows))
    dataset = output / "matched.pt"
    torch.save({"rows": rows, "audits": audits}, dataset)
    differences = []
    for pair in range(pairs):
        subset = [r for r in rows if int(r["pair"]) == pair]
        differences.extend(
            float((subset[q]["target"] - subset[q + 4]["target"]).square().mean())
            for q in range(4)
        )
    intervention_mse = sum(differences) / len(differences)
    result = analyze_matched(rows) if pairs >= 41 else {"engineering_smoke_only": True}
    write_report(
        output,
        {
            "scope": "real simulator response identification; no actor, reward or task success",
            "completed": True,
            "pairs": pairs,
            "queries_per_condition": 4,
            "data_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "hidden_stiffness_labels": [250, 1000],
            "intervention_effect_mse": intervention_mse,
            "intervention_detected": intervention_mse > 1e-10,
            "query_state_audits": audits,
            "results": result,
        },
    )


def collect_shift(env, output: Path, *, seeds: int) -> None:
    """Record A-B-A and stationary streams using common commands and boundaries."""
    rows, raw = [], []
    for seed in range(56001, 56001 + seeds):
        generator = torch.Generator().manual_seed(seed)
        commands = (torch.rand(36, 8, generator=generator) - 0.5) * 0.08
        commands[:, 7] = 1
        for condition, schedule in (
            ("stationary", [1000.0, 1000.0, 1000.0]),
            ("A_B_A", [1000.0, 250.0, 1000.0]),
        ):
            outcomes = []
            for phase, stiffness in enumerate(schedule):
                start, _, _ = reset_condition(env, seed, stiffness)
                for command in commands[phase * 12 : (phase + 1) * 12]:
                    end = execute(env, command)
                    outcomes.append((end - start)[:7])
                    start = end
            data = torch.stack(outcomes)
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "results": analyze_stream(commands[:, :7], data, [0, 12, 24]),
                }
            )
            raw.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "commands": commands,
                    "outcomes": data,
                }
            )
            (output / "progress.json").write_text(
                json.dumps(
                    {"completed_streams": len(rows), "planned_streams": 2 * seeds}
                )
            )
    path = output / "streams.pt"
    torch.save(raw, path)
    write_report(
        output,
        {
            "scope": "real simulator response prediction; no policy learning or task success",
            "completed": True,
            "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "boundaries": [0, 12, 24],
            "clear_rule": "clear at every declared attempt start, including stationary controls",
            "rows": rows,
        },
    )


def contact_observation(env) -> dict:
    """Read physical contact evidence for audit, never as a predictor input."""
    base = env.unwrapped
    forces = [
        float(
            torch.linalg.vector_norm(
                base.scene.get_pairwise_contact_forces(link, base.peg)
            )
        )
        for link in (base.agent.finger1_link, base.agent.finger2_link)
    ]
    return {
        "finger_peg_force_newtons": forces,
        "grasped": bool(base.agent.is_grasping(base.peg).item()),
    }


def phase_schedule() -> list[dict]:
    """Cross physical drive changes with independently specified contact stages."""
    return [
        {
            "stage": stage,
            "contact": contacts,
            "dynamics": dynamics,
            "stiffness": stiffness,
        }
        for stage, contacts in (
            ("free_fixed", [False, False, False]),
            ("grasp_fixed", [True, True, True]),
            ("free_grasp_free", [False, True, False]),
            ("grasp_free_grasp", [True, False, True]),
        )
        for dynamics, stiffness in (
            ("stationary", [1000.0, 1000.0, 1000.0]),
            ("A_B_A", [1000.0, 250.0, 1000.0]),
        )
    ]


def validate_contact_trace(trace: list[dict], *, grasped: bool) -> dict:
    """Require the declared stage throughout a probe; reject failed preparation."""
    if not trace:
        raise ValueError("Missing contact trace")
    force = torch.tensor([x["finger_peg_force_newtons"] for x in trace])
    if (
        force.shape != (len(trace), 2)
        or not torch.isfinite(force).all()
        or (force < 0).any()
    ):
        raise ValueError("Invalid contact force trace")
    fraction = sum(x["grasped"] for x in trace) / len(trace)
    valid = fraction >= 0.9 if grasped else fraction == 0 and float(force.max()) < 0.1
    return {
        "valid": valid,
        "grasp_fraction": fraction,
        "max_force_newtons": float(force.max()),
    }


def prepare_contact(env, seed: int, stiffness: float, *, grasped: bool) -> dict:
    """Prepare a privileged calibration scene, then physically settle the grasp.

    Placing the peg at the fingers is diagnostic state construction. This is
    neither a learned recovery skill nor an assisted task-success measurement.
    """
    import sapien

    reset_condition(env, seed, stiffness)
    base = env.unwrapped
    if grasped:
        # Start the fingers beside the peg so gravity cannot drop it during
        # a long open-to-closed approach. Contact must still pass the audit.
        qpos = base.agent.robot.get_qpos().clone()
        qpos[:, -2:] = base.peg_half_sizes[:, 1:2] + 0.0005
        base.agent.robot.set_qpos(qpos)
        base.peg.set_pose(base.agent.tcp.pose * sapien.Pose([0.04, 0, 0]))
        base.peg.set_linear_velocity(torch.zeros(1, 3))
        base.peg.set_angular_velocity(torch.zeros(1, 3))
    hold = torch.zeros(8)
    hold[7] = -1
    trace = []
    for _ in range(20):
        _, _, terminated, truncated, _ = env.step(hold.numpy())
        if bool(terminated.any()) or bool(truncated.any()):
            raise ValueError("Contact preparation terminated")
        trace.append(contact_observation(env))
    audit = validate_contact_trace(trace[-10:], grasped=grasped)
    if not audit["valid"]:
        raise ValueError(
            f"Contact preparation failed: requested_grasp={grasped}, audit={audit}"
        )
    return {"state_sha256": tensor_digest(base.get_state_dict()), "contact": audit}


def collect_phase_shift(env, output: Path, *, seeds: int, stages: str = "all") -> None:
    """Cross contact-stage and drive shifts using paired commands and seeds."""
    rows, raw = [], []
    for seed in range(57001, 57001 + seeds):
        generator = torch.Generator().manual_seed(seed)
        # Repeat the same excitation in all three attempts to isolate changes.
        block = (torch.rand(12, 8, generator=generator) - 0.5) * 0.04
        block[:, 7] = -1
        commands = block.repeat(3, 1)
        for schedule in phase_schedule():
            fixed = len(set(schedule["contact"])) == 1
            if (stages == "fixed" and not fixed) or (stages == "changing" and fixed):
                continue
            outcomes, attempts = [], []
            for attempt, (grasped, stiffness) in enumerate(
                zip(schedule["contact"], schedule["stiffness"])
            ):
                preparation = prepare_contact(env, seed, stiffness, grasped=grasped)
                trace = []
                start = (
                    env.unwrapped.agent.robot.get_qpos()[0, :9].detach().cpu().clone()
                )
                for command in block:
                    for _ in range(10):
                        _, _, terminated, truncated, _ = env.step(command.numpy())
                        if bool(terminated.any()) or bool(truncated.any()):
                            raise ValueError(
                                "Phase probe terminated before completing a chunk"
                            )
                        trace.append(contact_observation(env))
                    end = (
                        env.unwrapped.agent.robot.get_qpos()[0, :9]
                        .detach()
                        .cpu()
                        .clone()
                    )
                    outcomes.append((end - start)[:7])
                    start = end
                audit = validate_contact_trace(trace, grasped=grasped)
                attempts.append(
                    {
                        "attempt": attempt,
                        "requested_grasp": grasped,
                        "preparation": preparation,
                        "contact_audit": audit,
                        "trace": trace,
                    }
                )
            data = torch.stack(outcomes)
            if not torch.isfinite(data).all():
                raise ValueError("Nonfinite phase response")
            valid = all(x["contact_audit"]["valid"] for x in attempts)
            rows.append(
                {
                    "seed": seed,
                    **schedule,
                    "contact_valid": valid,
                    "attempts": attempts,
                    "results": analyze_stream(commands[:, :7], data, [0, 12, 24]),
                }
            )
            raw.append(
                {"seed": seed, **schedule, "commands": commands, "outcomes": data}
            )
            write_report(
                output,
                {
                    "completed": False,
                    "scope": "contact-stage engineering probe",
                    "rows": rows,
                },
            )
    path = output / "phase-streams.pt"
    torch.save(raw, path)
    write_report(
        output,
        {
            "completed": True,
            "scope": "privileged contact-stage x drive response diagnostic; no task success or policy learning",
            "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "all_contact_stages_valid": all(row["contact_valid"] for row in rows),
            "rows": rows,
        },
    )


def main() -> None:
    """Run bounded diagnostics with fixed methods and no test-driven tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("synthetic", "matched", "shift", "phase", "fit")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--pairs", type=int, default=56)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stages", choices=("all", "fixed", "changing"), default="all")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--standardize", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "fit":
        if args.dataset is None or not args.dataset.is_file():
            raise ValueError("Fitting requires an existing matched dataset")
        args.output.mkdir(parents=True, exist_ok=False)
        records = torch.load(args.dataset, weights_only=True, map_location="cpu")
        write_report(
            args.output,
            {
                "completed": True,
                "scope": "post-hoc input-scaling diagnostic on inspected data; no new held-out claim",
                "data_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "results": analyze_matched(
                    records["rows"], standardize=args.standardize
                ),
            },
        )
        return
    if args.mode == "synthetic":
        synthetic(args.output)
        return
    if args.smoke:
        args.pairs, args.seeds = 2, 1
    if (
        args.gpu not in (0, 1)
        or not (args.smoke or 41 <= args.pairs <= 64)
        or not 1 <= args.seeds <= 12
    ):
        raise ValueError("Use GPU 0/1, 41..64 pairs and 1..12 stream seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    env, lease = create_env(args.gpu, args.output)
    try:
        if args.mode == "matched":
            collect_matched(env, args.output, pairs=args.pairs)
        elif args.mode == "phase":
            collect_phase_shift(env, args.output, seeds=args.seeds, stages=args.stages)
        else:
            collect_shift(env, args.output, seeds=args.seeds)
    finally:
        env.close()
        lease.close()


if __name__ == "__main__":
    main()
