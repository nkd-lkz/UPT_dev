# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Small matched BC/Q+BC closed-loop pilots with frozen physical readouts."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np
import torch

from toolkits.rlt.libero_acceptance_learning import AcceptanceLearner
from toolkits.rlt.libero_pro_environment import LiberoProEnvironment
from toolkits.rlt.physrlt.data import PhysicalReference, physical_reset_hash, projection
from toolkits.rlt.physrlt.model import PhysicalReadout
from toolkits.rlt.physrlt.probe import inputs
from toolkits.rlt.pi05_baseline import (
    configure_gpu,
    configure_source,
    env_commands,
    load_model,
    write_json,
)


class FrozenReadout:
    """Own a fixed representation version throughout a control experiment."""

    def __init__(self, path: Path, mode: str = "latent", anchor: Path | None = None):
        if mode not in {"latent", "outcomes", "mixed"}:
            raise ValueError("Unknown physical token deployment mode")
        self.mode = mode
        self.payload = torch.load(path, map_location="cpu", weights_only=True)
        self.anchor = FrozenReadout(anchor) if anchor is not None else None
        if (
            self.anchor is not None
            and self.anchor.payload["dataset"] != self.payload["dataset"]
        ):
            raise ValueError(
                "Anchor and physical readout must use the same training dataset"
            )
        self.model = PhysicalReadout(
            self.payload["response_dim"],
            self.payload["geometry_dim"],
            factorized=self.payload.get("factorized", False),
        )
        self.model.load_state_dict(self.payload["model"], strict=True)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, prediction: dict, prop) -> dict:
        data = {
            "action_feature": prediction["action_feature"][None].float(),
            "prefix_feature": prediction["prefix_feature"][None].float(),
            "ref": torch.as_tensor(prediction["ref"])[None].float(),
            "prop": torch.as_tensor(prop)[None].float(),
        }
        z = self.model.encode(
            *inputs(data, self.payload["stats"], self.payload["method"])
        )
        if self.mode != "latent":
            commands = torch.from_numpy(env_commands(data["ref"][0].numpy()))[None]
            predictions = self.model.consequences(z, commands)
            depth = predictions["geometry"][..., :64].reshape(2, 1, 8, 8)
            depth = torch.nn.functional.avg_pool2d(depth, 2).flatten()
            quantities = torch.cat(
                (
                    predictions["response"].flatten(),
                    depth,
                    predictions["geometry"][..., 64:].flatten(),
                )
            )
            if quantities.numel() > 128:
                raise ValueError("Explicit physical quantities exceed token budget")
            physical = torch.nn.functional.pad(
                quantities, (0, 128 - quantities.numel())
            )[None]
            if self.mode == "outcomes":
                z = physical
            else:
                # Keep half the original readout and a fixed-budget view of
                # normalized predicted physical consequences, without future GT.
                compact = torch.nn.functional.adaptive_avg_pool1d(
                    physical[:, None], 64
                )[:, 0]
                z = torch.cat((z[:, :64], compact), -1)
        if self.anchor is not None:
            prior = self.anchor.encode(prediction, prop)["z"][None]
            z = torch.cat((prior, z), dim=-1) @ projection(256)
        return {"z": z[0], "ref": data["ref"][0].clamp(-1, 1), "prop": data["prop"][0]}


def warmup(readout: FrozenReadout, dataset: dict, seed: int, steps: int):
    """Fit the same upstream AlphaBrain actor on each frozen representation."""
    random.seed(seed)
    torch.manual_seed(seed)
    learner = AcceptanceLearner(
        "cuda",
        "q_bc",
        ref_dropout=0,
        bottleneck_dim=128,
        chunk_len=10,
        gripper_threshold=0,
    )
    train, validation = [], []
    for row in dataset["rows"]:
        if row["split"] not in {"train", "validation"}:
            continue
        features = readout.encode(row, row["prop"])
        replay_row = features | {
            "action": features["ref"].clone(),
            "next_z": features["z"].clone(),
            "next_ref": features["ref"].clone(),
            "next_prop": features["prop"].clone(),
        }
        (train if row["split"] == "train" else validation).append(replay_row)
    initial = learner.following(validation)
    trace = []
    for step in range(steps):
        result = learner.update(train, warmup=True)
        if step % 100 == 0 or step == steps - 1:
            trace.append({"step": step + 1, **result})
    learner.publish_warmup()
    final = learner.following(validation)
    return learner, {"before": initial, "after": final, "trace": trace}


def rollout(
    env,
    reference,
    readout,
    learner,
    case,
    limit,
    replay=None,
    *,
    execution_gain: float = 1.0,
    delay_ticks: int = 0,
) -> dict:
    """Execute an explicit control budget, optionally updating only small heads."""
    if not 0 < execution_gain <= 1.5 or not 0 <= delay_ticks <= 5:
        raise ValueError("Actuation stress is outside the declared development range")
    if delay_ticks and replay is not None:
        raise ValueError(
            "Delayed online RL needs pending-command state augmentation first"
        )
    pending = deque(
        [np.array([0, 0, 0, 0, 0, 0, -1], np.float32) for _ in range(delay_ticks)]
    )
    obs = env.reset(case)
    for _ in range(9):
        obs, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
    digest = physical_reset_hash(env)
    expected = env.pilot_reset_hashes.get(case["id"])
    if expected is not None and digest != expected:
        raise RuntimeError(
            "Control reset differs from the recorded collection initial state"
        )
    if expected is None and digest in env.pilot_reset_hashes.values():
        raise RuntimeError(
            "New development case duplicates a previously used physical layout"
        )
    env.pilot_reset_hashes[case["id"]] = digest
    ticks, success, rewards, inference_seconds = 0, False, 0.0, 0.0
    start = time.monotonic()
    prediction = reference.predict(obs, env.instruction, case["seed"] * 1000)
    features = readout.encode(prediction, obs["state"]) if readout else None
    updates = []
    while ticks < limit:
        begin = time.monotonic()
        action = (
            learner.command(features, deterministic=replay is None)
            if learner
            else torch.from_numpy(prediction["ref"])
        )
        commands = env_commands(action.numpy())
        commands[:, :3] = (commands[:, :3] * execution_gain).clip(-1, 1)
        accepted = np.zeros_like(commands)
        inference_seconds += time.monotonic() - begin
        chunk_ticks, chunk_reward = 0, 0.0
        for command in commands:
            if delay_ticks:
                pending.append(command.copy())
                command = pending.popleft()
            obs, reward, done = env.step(command)
            accepted[chunk_ticks] = command
            chunk_ticks += 1
            ticks += 1
            chunk_reward += reward
            success = success or done
            if done or ticks >= limit:
                break
        # Terminal transitions do not bootstrap, but keep a correctly shaped
        # successor. A truncated partial chunk is ineligible for TD updates.
        if not success:
            begin = time.monotonic()
            following = reference.predict(
                obs, env.instruction, case["seed"] * 1000 + ticks
            )
            next_features = readout.encode(following, obs["state"]) if readout else None
            inference_seconds += time.monotonic() - begin
        else:
            following, next_features = prediction, features
        if replay is not None:
            accepted_commands = torch.from_numpy(accepted)
            replay.append(
                features
                | {
                    # Q uses the policy proposal domain, as do its actor/target
                    # calls. The deterministic environment mapping belongs to
                    # the transition. Keep the accepted commands separately for
                    # physical supervision, not as mismatched Q input labels.
                    "action": action,
                    "accepted_commands": accepted_commands,
                    "next_z": next_features["z"],
                    "next_ref": next_features["ref"],
                    "next_prop": torch.as_tensor(obs["state"], dtype=torch.float32),
                    "ticks": chunk_ticks,
                    "terminal": success,
                    "reward": chunk_reward * 5,
                }
            )
            if chunk_ticks == 10 or success:
                updates.append(learner.update(replay))
        rewards += chunk_reward
        prediction, features = following, next_features
        if success:
            break
    return {
        "case_id": case["id"],
        "split": case["split"],
        "state_id": case["state"],
        "reset_hash": digest,
        "success": success,
        "ticks": ticks,
        "reward": rewards,
        "seconds": time.monotonic() - start,
        "inference_seconds_partial": inference_seconds,
        "updates": updates,
        "assistance": False,
        "memory": False,
        "execution_gain": execution_gain,
        "delay_ticks": delay_ticks,
    }


def development_cases(path: Path, original: list[dict]) -> list[dict]:
    """Load new development layouts without reusing training or pilot identities."""
    cases = json.loads(path.read_text())
    if not cases:
        raise ValueError("New development cases must not be empty")
    identities = {(c["suite"], c["task"], c["state"]) for c in original}
    tasks = {(c["suite"], c["task"], c["perturbation"]) for c in original}
    names = {c["id"] for c in original}
    for case in cases:
        identity = (case["suite"], case["task"], case["state"])
        if (
            case["split"] != "development"
            or identity in identities
            or case["id"] in names
        ):
            raise ValueError(
                "New development layouts must be unique and disjoint from the pilot"
            )
        if (case["suite"], case["task"], case["perturbation"]) not in tasks:
            raise ValueError("Development task and perturbation must match training")
        identities.add(identity)
        names.add(case["id"])
    return cases


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Completed control pilots are immutable")
    write_json(
        args.output / "run_manifest.json",
        {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "critic_action_space": "policy_proposal_with_fixed_environment_mapping",
            "physical_target_action_space": "accepted_native_commands",
            "control_protocol_version": 2,
        },
    )
    configure_gpu(args.gpu, args.output)
    reservation = torch.empty(1, device="cuda")
    configure_source(args.source)
    torch.set_num_threads(4)
    model, _ = load_model(args.assets, "cuda")
    reference = PhysicalReference(model)
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
    cases = dataset["meta"]["cases"]
    train_cases = [c for c in cases if c["split"] == "train"]
    validation_cases = [c for c in cases if c["split"] == "validation"]
    dev_cases = [c for c in cases if c["split"] == "development"]
    if getattr(args, "development_cases", None):
        dev_cases = development_cases(args.development_cases, cases)
    env = LiberoProEnvironment()
    env.pilot_reset_hashes = {
        episode["case"]["id"]: episode["reset_hash"] for episode in dataset["episodes"]
    }
    records = []
    rollout_case = partial(rollout, execution_gain=getattr(args, "execution_gain", 1.0))
    try:
        for case in validation_cases + dev_cases:
            row = rollout_case(env, reference, None, None, case, 520)
            records.append(row | {"method": "reference", "stage": "frozen"})
            write_json(args.output / "episodes.json", records)
            print(
                f"reference {case['id']}: success={row['success']} ticks={row['ticks']}",
                flush=True,
            )
        for method in args.methods:
            base_method, mode = method, "latent"
            anchor = None
            if method.endswith("_anchor"):
                base_method = method.removesuffix("_anchor")
                # Every arm has two 400-step encoders. The fixed prior uses a
                # different initialization, including in the dual-reconstruction
                # control; this avoids a rank-deficient duplicate baseline.
                anchor = args.probes / "action_recon-seed59" / "readout.pt"
            for suffix in ("_outcomes", "_mixed"):
                if method.endswith(suffix):
                    base_method, mode = method.removesuffix(suffix), suffix[1:]
            path = args.probes / f"{base_method}-seed{args.seed}" / "readout.pt"
            readout = FrozenReadout(path, mode=mode, anchor=anchor)
            learner, fidelity = warmup(readout, dataset, args.seed, args.warmup_steps)
            write_json(args.output / f"{method}-warmup.json", fidelity)
            torch.save(learner.state_dict(), args.output / f"{method}-warmup.pt")
            validation_results = []
            for case in validation_cases:
                row = rollout_case(env, reference, readout, learner, case, 520)
                validation_results.append(row)
                records.append(row | {"method": method, "stage": "warmup"})
                write_json(args.output / "episodes.json", records)
            gate = (
                fidelity["after"]["arm_mse"] < 0.02
                and fidelity["after"]["gripper_disagreement"] < 0.05
            )
            gate = gate and any(r["success"] for r in validation_results)
            write_json(
                args.output / f"{method}-gate.json",
                {
                    "passed": gate,
                    "selection_split": "validation",
                    "rule": "arm MSE < .02, gripper mismatch < .05, at least one validation success",
                    "meaning": "development safety gate; unequal passing arms are not a fair RL benchmark",
                },
            )
            for case in dev_cases:
                records.append(
                    rollout_case(env, reference, readout, learner, case, 520)
                    | {"method": method, "stage": "warmup"}
                )
                write_json(args.output / "episodes.json", records)
            if gate and args.online_ticks:
                replay, count = [], 0
                for case in train_cases:
                    budget = min(520, args.online_ticks - count)
                    if budget <= 0:
                        break
                    row = rollout_case(
                        env, reference, readout, learner, case, budget, replay
                    )
                    count += row["ticks"]
                    records.append(row | {"method": method, "stage": "online_train"})
                    write_json(args.output / "episodes.json", records)
                    torch.save(
                        learner.state_dict(), args.output / f"{method}-online.pt"
                    )
                    torch.save(replay, args.output / f"{method}-replay.pt")
                for case in validation_cases + dev_cases:
                    records.append(
                        rollout_case(env, reference, readout, learner, case, 520)
                        | {"method": method, "stage": "online_frozen_eval"}
                    )
                    write_json(args.output / "episodes.json", records)
            del learner, readout
            torch.cuda.empty_cache()
            print(f"completed method={method} seed={args.seed} gate={gate}", flush=True)
    finally:
        env.close()
        del reservation
    result = {
        "state": "complete",
        "episodes": len(records),
        "seed": args.seed,
        "role": "small development pilots, not final-test success claims",
        "warmup_steps": args.warmup_steps,
        "online_tick_cap": args.online_ticks,
        "control_protocol_version": 2,
    }
    write_json(args.output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "assets", "dataset", "probes", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--online-ticks", type=int, default=1040)
    parser.add_argument("--execution-gain", type=float, default=1.0)
    parser.add_argument("--development-cases", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["action_recon", "geometry", "response", "joint"],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
