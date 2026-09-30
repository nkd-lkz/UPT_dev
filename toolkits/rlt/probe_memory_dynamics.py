# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Test past-only memory under hidden motor stiffness in real simulation.

This is supervised system-response diagnosis, not task learning or a replacement
for the production environment. Simulator condition IDs never enter the model.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from rlinf.algorithms.rlt.interaction_memory import InteractionMemoryConfig
from rlinf.models.embodiment.modules.rlt_memory_encoder import response_features

from .probe_interaction_memory import ConsequenceProbe, episode_examples


def mask_memory(batch: dict, mode: str) -> dict:
    """Keep recent, retrieved, all or no records without mutating replay."""
    valid = batch["memory_valid"].clone()
    if mode == "none":
        valid[:] = False
    elif mode == "recent":
        valid[:, 4:] = False
    elif mode == "archive":
        valid[:, :4] = False
    elif mode not in ("full", "response", "response_residual"):
        raise ValueError("Unknown memory ablation")
    return {**batch, "memory_valid": valid}


def response_summary(batch: dict) -> torch.Tensor:
    """Estimate seven empirical joint responses from completed commands only.

    Panda arm commands correspond to +/-0.1 rad delta targets. This ridge slope
    is an empirical controller-response descriptor, not a physical parameter or
    contact label. Padding and future targets are excluded before arithmetic.
    """
    if batch["memory_events"].shape[1:] != (8, 110):
        raise ValueError("Response diagnostic expects default Panda memory schema")
    return response_features(batch, InteractionMemoryConfig())


class ResponseProbe(ConsequenceProbe):
    """Compare an interpretable past-response context using the same MLP head."""

    def forward(self, batch: dict, *, memory: bool) -> torch.Tensor:
        context = torch.nn.functional.pad(response_summary(batch), (0, 50))
        if not memory:
            context = context * 0
        return self.head(
            torch.cat((batch["memory_query"], batch["commands"], context), -1)
        )


class ResidualResponseProbe(ResponseProbe):
    """Start at the empirical response and learn its error from training data.

    The analytic term reads only completed transitions and known controller
    scaling. Validation selection can retain step zero if fitting hurts.
    """

    def __init__(self) -> None:
        super().__init__()
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch: dict, *, memory: bool) -> torch.Tensor:
        prior = (
            empirical_prediction(batch)
            if memory
            else torch.zeros_like(batch["memory_query"])
        )
        return prior + super().forward(batch, memory=memory)


def pair_split(pair: int) -> str:
    """Keep both motor conditions of one command/scene seed in the same split."""
    return "train" if pair < 32 else "validation" if pair < 40 else "test"


def collect(output: Path, *, pairs: int = 56, steps: int = 120) -> None:
    """Record exact commands and true qpos under two undisclosed drive settings."""
    if not 44 <= pairs <= 64 or not 60 <= steps <= 160 or steps % 10:
        raise ValueError("Use 44..64 pairs and 60..160 ticks divisible by ten")
    uuid, bus, used = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "2",
                "--query-gpu=uuid,pci.bus_id,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(",")
    )
    uuid, bus = uuid.strip(), bus.strip().lower()
    if int(used) > 1024 or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("Select idle physical GPU 2 by CUDA_VISIBLE_DEVICES UUID")
    if torch.cuda.device_count() != 1 or str(
        torch.cuda.get_device_properties(0).uuid
    ).removeprefix("GPU-") != uuid.removeprefix("GPU-"):
        raise RuntimeError("CUDA isolation mismatch")
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
        PANDA_WIDE_WRISTCAM_UID,
        register_rlinf_peg_insertion_side_variants,
    )
    from rlinf.envs.sim.maniskill.utils import allow_pci_render_backend

    output.mkdir(parents=True, exist_ok=False)
    register_rlinf_peg_insertion_side_variants()
    allow_pci_render_backend()
    domain, bus_id, slot = bus.split(":")
    renderer = f"pci:{int(domain, 16):04x}:{bus_id}:{slot}"
    env = gym.make(
        "PegInsertionSideWideClearanceObserverWideWrist-v1",
        robot_uids=PANDA_WIDE_WRISTCAM_UID,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode=None,
        sim_backend="physx_cpu",
        render_backend=renderer,
        max_episode_steps=steps,
        sim_config={"sim_freq": 100, "control_freq": 10},
    )
    trajectories = []
    try:
        for pair in range(pairs):
            rng = np.random.default_rng(91000 + pair)
            chunks = rng.uniform(-0.12, 0.12, (steps // 10, 8)).astype(np.float32)
            chunks[:, 7] = 1.0
            commands = np.repeat(chunks, 10, axis=0)
            for stiffness in (250.0, 1000.0):
                obs, _ = env.reset(seed=81000 + pair)
                arm = env.unwrapped.agent.controller.controllers["arm"]
                # Use the controller's public drive-property API. Subsequent
                # set_action changes targets, not these physical parameters.
                arm.config.stiffness = stiffness
                arm.set_drive_property()
                states = [obs["agent"]["qpos"][0, :9].detach().cpu().clone()]
                executed = []
                for action in commands:
                    obs, _, terminated, truncated, _ = env.step(action)
                    executed.append(torch.from_numpy(action.copy()))
                    states.append(obs["agent"]["qpos"][0, :9].detach().cpu().clone())
                    if bool(terminated.any()) or bool(truncated.any()):
                        break
                # episode_examples excludes the final action without a successor;
                # append a never-used padding command to align T states with T rows.
                actions = torch.stack(executed + [torch.zeros(8)])
                trajectories.append(
                    {
                        "pair": pair,
                        "split": pair_split(pair),
                        "stiffness": stiffness,
                        "states": torch.stack(states),
                        "actions": actions,
                    }
                )
            print(
                json.dumps({"collected_pairs": pair + 1, "gpu_uuid": uuid}), flush=True
            )
    finally:
        env.close()
    path = output / "trajectories.pt"
    torch.save({"schema": 1, "trajectories": trajectories}, path)
    manifest = {
        "kind": "real_cpu_physics_hidden_stiffness_not_RL",
        "pairs": pairs,
        "steps": steps,
        "gpu_uuid": uuid,
        "renderer": renderer,
        "control_hz": 10,
        "stiffness": [250, 1000],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "hidden_parameters_are_labels_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def build_splits(trajectories: list[dict]) -> dict:
    """Construct causal histories; simulator metadata stays outside inputs."""
    splits = {s: [] for s in ("train", "validation", "test")}
    for trajectory in trajectories:
        split = pair_split(trajectory["pair"])
        if trajectory["split"] != split:
            raise ValueError("Command-seed pair crosses the registered split")
        rows = episode_examples(trajectory["states"], trajectory["actions"])
        for age, row in enumerate(rows):
            splits[split].append(
                {
                    **row,
                    "age": torch.tensor(age),
                    "episode": torch.tensor(
                        2 * trajectory["pair"] + int(trajectory["stiffness"] == 1000)
                    ),
                }
            )
    if not all(splits.values()):
        raise ValueError("All three disjoint splits must be nonempty")
    return {key: default_collate(rows) for key, rows in splits.items()}


@torch.no_grad()
def evaluate(model: ConsequenceProbe, batch: dict, mode: str) -> dict:
    """Report episode-balanced joint-response MSE and late-history subset."""
    return prediction_metrics(model(mask_memory(batch, mode), memory=True), batch)


def prediction_metrics(prediction: torch.Tensor, batch: dict) -> dict:
    """Compare predictions with true outcomes, grouped by trajectory and history."""
    errors = (prediction - batch["target"]).square().mean(-1)
    episodes = {
        str(i): float(errors[batch["episode"] == i].mean())
        for i in batch["episode"].unique().tolist()
    }
    late = batch["age"] >= 4
    return {
        "mse": float(errors.mean()),
        "episode_mse": episodes,
        "episode_balanced_mse": sum(episodes.values()) / len(episodes),
        "late_mse": float(errors[late].mean()) if late.any() else None,
    }


def empirical_prediction(batch: dict) -> torch.Tensor:
    """Apply the past-only slope to proposed commands; gripper change is zero.

    This diagnostic uses the known joint-delta controller scale, not the hidden
    stiffness. It is not an actor, a safety constraint, or a learned world model.
    """
    gain = response_summary(batch)[:, :7]
    commands = batch["commands"].reshape(-1, 10, 8)[:, :, :7].sum(1) * 0.1
    return torch.nn.functional.pad(gain * commands, (0, 2))


def audit_empirical(data_dir: Path, output: Path) -> dict:
    """Test a fixed response formula without optimizing on any held-out labels."""
    path = data_dir / "trajectories.pt"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Trajectory checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    batches = build_splits(payload["trajectories"])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "empirical_past_response_not_policy_or_RL",
        "data": manifest,
        "formula": "sum(u*delta)/(sum(u^2)+1e-4), u=0.1*sum(executed_arm_commands)",
        "results": {
            split: prediction_metrics(empirical_prediction(batch), batch)
            for split, batch in batches.items()
        },
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def fit(
    data_dir: Path,
    output: Path,
    *,
    updates: int = 600,
    seeds: tuple[int, ...] = (2026, 2027, 2028),
) -> dict:
    """Compare learned and empirical memory; select on validation only."""
    if not 1 <= updates <= 1000:
        raise ValueError("Use 1..1000 updates")
    if not 1 <= len(seeds) <= 10:
        raise ValueError("Use 1..10 seeds")
    path = data_dir / "trajectories.pt"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Trajectory checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != 1:
        raise ValueError("Unknown trajectory schema")
    batches = build_splits(payload["trajectories"])
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in seeds:
        for mode in (
            "none",
            "recent",
            "archive",
            "full",
            "response",
            "response_residual",
        ):
            torch.manual_seed(seed)
            model = (
                ResidualResponseProbe()
                if mode == "response_residual"
                else ResponseProbe()
                if mode == "response"
                else ConsequenceProbe()
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            generator = torch.Generator().manual_seed(seed + 1)
            best, best_step = float("inf"), 0
            checkpoint = output / f"seed{seed}_{mode}.pt"
            # All modes may select their untrained initialization; the residual
            # path must never lose the useful fixed prior merely by fitting.
            best = evaluate(model, batches["validation"], mode)["mse"]
            torch.save(model.state_dict(), checkpoint)
            for step in range(1, updates + 1):
                ids = torch.randint(
                    len(batches["train"]["target"]), (32,), generator=generator
                )
                batch = {k: v[ids] for k, v in batches["train"].items()}
                prediction = model(mask_memory(batch, mode), memory=True)
                loss = (prediction - batch["target"]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite diagnostic loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1, error_if_nonfinite=True
                )
                optimizer.step()
                if step % 50 == 0 or step == updates:
                    score = evaluate(model, batches["validation"], mode)["mse"]
                    if score < best:
                        best, best_step = score, step
                        torch.save(model.state_dict(), checkpoint)
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            test = evaluate(model, batches["test"], mode)
            rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "best_step": best_step,
                    "validation_mse": best,
                    "test": test,
                }
            )
            report = {
                "kind": "supervised_response_not_RL_or_task_transfer",
                "data": manifest,
                "updates_per_model": updates,
                "seeds": list(seeds),
                "samples": {k: len(v["target"]) for k, v in batches.items()},
                "results": rows,
            }
            (output / "results.json").write_text(
                json.dumps(report, indent=2, allow_nan=False)
            )
            print(
                json.dumps({"seed": seed, "mode": mode, "test_mse": test["mse"]}),
                flush=True,
            )
    return report


def main() -> None:
    """Collect with isolated GPU 2 rendering or fit CPU-only diagnostic models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "fit", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--updates", type=int, default=600)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "collect":
        collect(args.output)
    else:
        if args.data is None:
            parser.error("fit/audit requires --data")
        if args.mode == "audit":
            print(json.dumps(audit_empirical(args.data, args.output), indent=2))
        else:
            fit(args.data, args.output, updates=args.updates)


if __name__ == "__main__":
    main()
