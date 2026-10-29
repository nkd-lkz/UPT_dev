# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Small matched BC/Q+BC closed-loop pilots with frozen physical readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import subprocess
import time
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np
import torch

from toolkits.rlt.libero_acceptance_learning import AcceptanceLearner, td_eligible
from toolkits.rlt.libero_pro_environment import LiberoProEnvironment
from toolkits.rlt.physrlt.conditioning import (
    CandidateConsequenceConditioner,
    SignedCandidateConsequence,
    load_sparse_motion_conditioner,
    physical_quantities,
)
from toolkits.rlt.physrlt.data import (
    PhysicalReference,
    physical_reset_hash,
    physical_reset_state,
    projection,
)
from toolkits.rlt.physrlt.model import PhysicalReadout
from toolkits.rlt.physrlt.probe import dataset_sha256, inputs
from toolkits.rlt.pi05_baseline import (
    configure_gpu,
    configure_source,
    env_commands,
    load_model,
    write_json,
)


class FrozenReadout:
    """Own a fixed representation version throughout a control experiment."""

    def __init__(
        self,
        path: Path,
        mode: str = "latent",
        anchor: Path | None = None,
        consequence: Path | None = None,
        actor_conditioning: str = "base",
        *,
        expected_dataset: Path | None = None,
        expected_method: str | None = None,
    ):
        if mode not in {"latent", "outcomes", "mixed"}:
            raise ValueError("Unknown physical token deployment mode")
        if actor_conditioning not in {
            "base",
            "zero_concat",
            "physical_concat",
            "physical_outcomes_concat",
        }:
            raise ValueError("Unknown actor conditioning protocol")
        self.mode = mode
        self.actor_conditioning = actor_conditioning
        self.payload = torch.load(path, map_location="cpu", weights_only=True)
        if self.payload.get("schema") != 1:
            raise ValueError("Unsupported physical readout schema")
        if (
            expected_method is not None
            and self.payload.get("method") != expected_method
        ):
            raise ValueError("Physical readout method differs from the requested arm")
        if expected_dataset is not None and Path(self.payload["dataset"]).resolve() != (
            expected_dataset.resolve()
        ):
            raise ValueError("Physical readout was fitted on a different dataset")
        expected_digest = self.payload.get("dataset_sha256")
        if (
            expected_dataset is not None
            and expected_digest is not None
            and dataset_sha256(expected_dataset) != expected_digest
        ):
            raise ValueError("Physical readout dataset contents have changed")
        self.anchor = (
            FrozenReadout(
                anchor,
                expected_dataset=expected_dataset,
                expected_method="action_recon",
            )
            if anchor is not None
            else None
        )
        self.consequence = (
            FrozenReadout(consequence, expected_dataset=expected_dataset)
            if consequence is not None
            else None
        )
        if (
            self.anchor is not None
            and Path(self.anchor.payload["dataset"]).resolve()
            != Path(self.payload["dataset"]).resolve()
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

    def candidate_conditioner(
        self, mode: str = "mixed", component: str = "joint"
    ) -> CandidateConsequenceConditioner:
        """Build a frozen, action-conditioned consequence view for a critic."""
        return CandidateConsequenceConditioner(
            (self.consequence or self).model,
            (self.consequence or self).payload["method"],
            mode=mode,
            component=component,
        )

    @torch.no_grad()
    def encode(self, prediction: dict, prop) -> dict:
        data = {
            "action_feature": prediction["action_feature"][None].float(),
            "prefix_feature": prediction["prefix_feature"][None].float(),
            "ref": torch.as_tensor(prediction["ref"])[None].float(),
            "prop": torch.as_tensor(prop)[None].float(),
        }
        if "flow_feature" in prediction:
            data["flow_feature"] = prediction["flow_feature"][None].float()
        z = self.model.encode(
            *inputs(data, self.payload["stats"], self.payload["method"])
        )
        if self.mode != "latent":
            physical = physical_quantities(
                self.model, z, data["ref"].clamp(-1, 1), self.payload["method"]
            )
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
        result = {
            "z": z[0],
            "ref": data["ref"][0].clamp(-1, 1),
            "prop": data["prop"][0],
        }
        if self.consequence is not None:
            result["critic_z"] = self.consequence.encode(prediction, prop)["z"]
        if self.actor_conditioning == "zero_concat":
            result["actor_z"] = torch.cat((result["z"], torch.zeros_like(result["z"])))
        elif self.actor_conditioning == "physical_concat":
            if "critic_z" not in result:
                raise ValueError(
                    "Physical actor conditioning requires a consequence token"
                )
            result["actor_z"] = torch.cat((result["z"], result["critic_z"]))
        elif self.actor_conditioning == "physical_outcomes_concat":
            if self.consequence is None:
                raise ValueError(
                    "Physical actor outcomes require a consequence readout"
                )
            physical = physical_quantities(
                self.consequence.model,
                result["critic_z"][None],
                result["ref"][None],
                self.consequence.payload["method"],
            )[0]
            result["actor_z"] = torch.cat((result["z"], physical))
        return result


def warmup(
    readout: FrozenReadout,
    dataset: dict,
    seed: int,
    steps: int,
    critic_conditioning: str = "latent",
    correction_limit: float | None = None,
    correction_axes: tuple[int, ...] | None = None,
    residual_output: bool = False,
    critic_bottleneck_dim: int | None = None,
    actor_bottleneck_dim: int | None = None,
    actor_physical_gate: bool = False,
    candidate_selection_count: int = 0,
    candidate_selection_margin: float = 0.0,
    candidate_uncertainty_penalty: float = 0.0,
    critic_conditioner_override: torch.nn.Module | None = None,
    critic_response_aux_weight: float = 0.0,
    critic_response_aux_shuffled: bool = False,
    critic_consequence_dueling: bool = False,
):
    """Fit the same upstream AlphaBrain actor on each frozen representation."""
    random.seed(seed)
    torch.manual_seed(seed)
    candidate_mode = (
        critic_conditioning.removeprefix("candidate_")
        if critic_conditioning.startswith("candidate_")
        else None
    )
    if critic_conditioner_override is not None:
        if critic_conditioning not in {
            "candidate_relative_concat",
            "sparse_relative_concat",
        }:
            raise ValueError("External critic conditioner has an incompatible mode")
    elif critic_conditioning != "latent" and candidate_mode not in {
        "outcomes",
        "mixed",
        "relative",
        "zero_concat",
        "relative_concat",
        "relative_unit_concat",
    }:
        raise ValueError("Unknown critic conditioning protocol")
    learner = AcceptanceLearner(
        "cuda",
        "q_bc",
        ref_dropout=0,
        bottleneck_dim=128,
        chunk_len=10,
        gripper_threshold=0,
        critic_conditioner=(
            critic_conditioner_override
            or (
                readout.candidate_conditioner(candidate_mode)
                if candidate_mode
                else None
            )
        ),
        critic_conditioning=critic_conditioning,
        correction_limit=correction_limit,
        correction_axes=correction_axes,
        residual_output=residual_output,
        critic_bottleneck_dim=critic_bottleneck_dim,
        actor_bottleneck_dim=actor_bottleneck_dim,
        actor_physical_gate=actor_physical_gate,
        candidate_selection_count=candidate_selection_count,
        candidate_selection_margin=candidate_selection_margin,
        candidate_uncertainty_penalty=candidate_uncertainty_penalty,
        critic_response_aux_weight=critic_response_aux_weight,
        critic_response_aux_shuffled=critic_response_aux_shuffled,
        critic_consequence_dueling=critic_consequence_dueling,
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
        if "critic_z" in features:
            replay_row["next_critic_z"] = features["critic_z"].clone()
        if "actor_z" in features:
            replay_row["next_actor_z"] = features["actor_z"].clone()
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


def parse_external_readouts(specs: list[str]) -> dict[str, Path]:
    """Parse explicit labels without inferring scientific meaning from filenames."""
    result = {}
    for spec in specs:
        label, separator, path = spec.partition("=")
        if not separator or not label or not path or label in result:
            raise ValueError("External readouts require unique LABEL=PATH entries")
        result[label] = Path(path).resolve()
    return result


def parse_online_checkpoint_arms(specs: list[str]) -> dict[str, str]:
    """Parse explicit control labels and their causal consequence arms."""
    result = {}
    for spec in specs:
        label, separator, arm = spec.partition("=")
        if (
            not separator
            or not label
            or arm not in {"correct", "reversed", "zero"}
            or label in result
        ):
            raise ValueError(
                "Checkpoint arms require unique LABEL=correct|reversed|zero entries"
            )
        result[label] = arm
    return result


def parse_online_checkpoint_components(specs: list[str]) -> dict[str, str]:
    """Parse control labels and the physical consequence channel they retain."""
    result = {}
    for spec in specs:
        label, separator, component = spec.partition("=")
        if (
            not separator
            or not label
            or component not in {"joint", "response", "geometry"}
            or label in result
        ):
            raise ValueError(
                "Checkpoint components require unique "
                "LABEL=joint|response|geometry entries"
            )
        result[label] = component
    return result


def warmup_gate(
    fidelity: dict, validation_results: list[dict], residual_output: bool
) -> tuple[bool, str]:
    """Require reference fidelity and an appropriate closed-loop safety check."""
    passed = (
        fidelity["after"]["arm_mse"] < 0.02
        and fidelity["after"]["gripper_disagreement"] < 0.05
    )
    if residual_output:
        return passed, "arm MSE < .02 and gripper mismatch < .05"
    return (
        passed and any(row["success"] for row in validation_results),
        "arm MSE < .02, gripper mismatch < .05, at least one validation success",
    )


def file_sha256(path: Path) -> str:
    """Hash a readout used by a control run."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gated_online_checkpoint(
    path: Path,
    *,
    expected_arm: str,
    expected_component: str,
    optimizer_seed: int,
    base_readout: Path,
    consequence_readout: Path,
) -> dict:
    """Load one exact critic arm emitted by a passed consequence gate."""
    if expected_arm not in {"correct", "reversed", "zero"}:
        raise ValueError("Expected checkpoint arm must be correct, reversed or zero")
    if expected_component not in {"joint", "response", "geometry"}:
        raise ValueError("Expected component must be joint, response or geometry")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "state_dict",
        "gate_passed",
        "arm",
        "optimizer_seed",
        "train_trace_sha256",
        "base_readout_sha256",
        "consequence_readout_sha256",
        "consequence_component",
    }
    if required - set(payload):
        raise ValueError("Gated online checkpoint lacks provenance fields")
    if not payload["gate_passed"]:
        raise ValueError("Gated online checkpoint was not emitted by a passed gate")
    if payload["arm"] != expected_arm:
        raise ValueError("Gated online checkpoint arm differs")
    if payload["consequence_component"] != expected_component:
        raise ValueError("Gated online checkpoint physical component differs")
    if int(payload["optimizer_seed"]) != optimizer_seed:
        raise ValueError("Gated online checkpoint optimizer seed differs")
    if payload["base_readout_sha256"] != file_sha256(base_readout):
        raise ValueError("Gated online checkpoint base readout differs")
    if payload["consequence_readout_sha256"] != file_sha256(consequence_readout):
        raise ValueError("Gated online checkpoint consequence readout differs")
    return payload


def replay_trace_sha256(replay: list[dict]) -> str:
    """Hash executed evidence while excluding representation-specific fields."""
    digest = hashlib.sha256()
    for row in replay:
        for key in ("action", "accepted_commands"):
            value = row[key].detach().cpu().contiguous().float().numpy()
            digest.update(value.shape.__repr__().encode())
            digest.update(value.tobytes())
        digest.update(
            struct.pack(
                "<q?d",
                int(row["ticks"]),
                bool(row["terminal"]),
                float(row["reward"]),
            )
        )
    return digest.hexdigest()


def deferred_update_summary(metrics: list[dict], replay: list[dict]) -> dict:
    """Summarize delayed learning without discarding drift diagnostics."""
    names = (
        "critic_loss",
        "critic_response_aux_loss",
        "q_mean",
        "target_mean",
        "actor_loss",
        "bc_sum",
        "q_actor",
        "actor_grad",
        "correction_rms",
    )
    summary = {}
    for name in names:
        values = [float(row[name]) for row in metrics if name in row]
        if values:
            summary[name] = {
                "first": values[0],
                "last": values[-1],
                "mean": sum(values) / len(values),
                "max": max(values),
            }
    return {
        "updates": len(metrics),
        "transitions": len(replay),
        "positive_transitions": sum(float(row["reward"]) > 0 for row in replay),
        "reward_sum": sum(float(row["reward"]) for row in replay),
        "metrics": summary,
    }


def skip_deferred_updates(replay: list[dict], require_positive: bool) -> bool:
    """Prevent unsupported actor drift when sparse replay has no success."""
    return require_positive and not any(float(row["reward"]) > 0 for row in replay)


def validate_fixed_replay(replay: list[dict]) -> dict:
    """Validate an immutable replay before using it across optimization seeds."""
    if not replay:
        raise ValueError("Fixed replay must contain transitions")
    tensor_fields = (
        "z",
        "ref",
        "prop",
        "action",
        "accepted_commands",
        "next_z",
        "next_ref",
        "next_prop",
    )
    for index, row in enumerate(replay):
        required = set(tensor_fields) | {"ticks", "terminal", "reward"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"Fixed replay row {index} lacks {sorted(missing)}")
        if not all(
            isinstance(row[name], torch.Tensor) and torch.isfinite(row[name]).all()
            for name in tensor_fields
        ):
            raise ValueError(f"Fixed replay row {index} has invalid tensors")
        if not td_eligible(row, chunk_len=10):
            raise ValueError(f"Fixed replay row {index} is not TD eligible")
    return {
        "transitions": len(replay),
        "positive_transitions": sum(float(row["reward"]) > 0 for row in replay),
        "reward_sum": sum(float(row["reward"]) for row in replay),
        "trace_sha256": replay_trace_sha256(replay),
    }


def merge_replay_views(base: list[dict], consequence: list[dict]) -> list[dict]:
    """Attach a physical-token view to an identical executed replay."""
    base_summary = validate_fixed_replay(base)
    consequence_summary = validate_fixed_replay(consequence)
    if (
        len(base) != len(consequence)
        or base_summary["trace_sha256"] != consequence_summary["trace_sha256"]
    ):
        raise ValueError("Base and physical replay traces differ")
    merged = []
    for base_row, consequence_row in zip(base, consequence):
        critic_z = consequence_row.get("critic_z", consequence_row["z"])
        next_critic_z = consequence_row.get("next_critic_z", consequence_row["next_z"])
        if (
            base_row["z"].shape != critic_z.shape
            or base_row["next_z"].shape != next_critic_z.shape
        ):
            raise ValueError("Base and physical replay token shapes differ")
        merged.append(
            base_row
            | {
                "critic_z": critic_z.clone(),
                "next_critic_z": next_critic_z.clone(),
            }
        )
    return merged


def attach_actor_view(
    replay: list[dict], mode: str, readout: FrozenReadout | None = None
) -> list[dict]:
    """Attach a capacity-matched actor token without changing executed replay."""
    if mode == "base":
        return replay
    if mode not in {
        "zero_concat",
        "physical_concat",
        "physical_outcomes_concat",
    }:
        raise ValueError("Unknown actor conditioning protocol")
    attached = []
    for index, row in enumerate(replay):
        physical = row.get("critic_z")
        next_physical = row.get("next_critic_z")
        if mode in {"physical_concat", "physical_outcomes_concat"} and (
            physical is None or next_physical is None
        ):
            raise ValueError(
                f"Physical actor replay row {index} lacks consequence tokens"
            )
        if mode == "zero_concat":
            physical = torch.zeros_like(row["z"])
            next_physical = torch.zeros_like(row["next_z"])
        elif mode == "physical_outcomes_concat":
            if readout is None or readout.consequence is None:
                raise ValueError(
                    "Physical actor outcomes require a consequence readout"
                )
            consequence = readout.consequence
            physical = physical_quantities(
                consequence.model,
                physical[None],
                row["ref"][None],
                consequence.payload["method"],
            )[0]
            next_physical = physical_quantities(
                consequence.model,
                next_physical[None],
                row["next_ref"][None],
                consequence.payload["method"],
            )[0]
        if (
            row["z"].shape != physical.shape
            or row["next_z"].shape != next_physical.shape
        ):
            raise ValueError("Base and physical actor token shapes differ")
        attached.append(
            row
            | {
                "actor_z": torch.cat((row["z"], physical)),
                "next_actor_z": torch.cat((row["next_z"], next_physical)),
            }
        )
    return attached


def paired_reset_delta(environment, case_id: str) -> float:
    """Require every control arm to start from the same current-run physics state.

    Collection and control may use different simulator builds or assets, so an
    old byte hash is provenance rather than a valid cross-run equality test.
    The first reset in this process becomes the control reference; subsequent
    methods must reproduce it within numerical tolerance.
    """
    state = physical_reset_state(environment)
    expected = environment.control_reset_states.get(case_id)
    if expected is None:
        environment.control_reset_states[case_id] = state
        return 0.0
    if state.shape != expected.shape:
        raise RuntimeError("Paired control reset state shape changed")
    delta = float(np.max(np.abs(state - expected)))
    if not np.allclose(state, expected, rtol=0, atol=1e-9):
        raise RuntimeError(
            f"Paired control reset differs for {case_id}: max_abs_delta={delta:.3e}"
        )
    return delta


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
    update_online: bool = True,
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
    reset_delta = paired_reset_delta(env, case["id"])
    ticks, success, rewards, inference_seconds = 0, False, 0.0, 0.0
    start = time.monotonic()
    prediction = reference.predict(obs, env.instruction, case["seed"] * 1000)
    features = readout.encode(prediction, obs["state"]) if readout else None
    updates, selection_diagnostics = [], []
    while ticks < limit:
        begin = time.monotonic()
        action = (
            learner.command(features, deterministic=replay is None)
            if learner
            else torch.from_numpy(prediction["ref"])
        )
        diagnostic = getattr(learner, "last_command_diagnostics", None)
        if diagnostic and diagnostic.get("enabled"):
            selection_diagnostics.append(diagnostic)
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
        if replay is not None and (chunk_ticks == 10 or success):
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
            if "critic_z" in features:
                replay[-1]["next_critic_z"] = next_features["critic_z"]
            if "actor_z" in features:
                replay[-1]["next_actor_z"] = next_features["actor_z"]
            if update_online:
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
        "collection_reset_hash": env.collection_reset_hashes.get(case["id"]),
        "paired_reset_max_abs_delta": reset_delta,
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
        "candidate_selection_calls": len(selection_diagnostics),
        "candidate_nonreference_fraction": (
            sum(row["selected_scale"] > 0 for row in selection_diagnostics)
            / len(selection_diagnostics)
            if selection_diagnostics
            else None
        ),
        "candidate_scale_mean": (
            sum(row["selected_scale"] for row in selection_diagnostics)
            / len(selection_diagnostics)
            if selection_diagnostics
            else None
        ),
        "candidate_score_improvement_mean": (
            sum(row["score_improvement"] for row in selection_diagnostics)
            / len(selection_diagnostics)
            if selection_diagnostics
            else None
        ),
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
            "reset_pairing_protocol": "current_run_physics_state_atol_1e-9",
            "collection_reset_hash_role": "provenance_only",
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
    env.collection_reset_hashes = {
        episode["case"]["id"]: episode["reset_hash"] for episode in dataset["episodes"]
    }
    env.control_reset_states = {}
    records = []
    external_readouts = parse_external_readouts(getattr(args, "external_readouts", []))
    fixed_replays = parse_external_readouts(getattr(args, "fixed_replays", []))
    critic_readouts = parse_external_readouts(getattr(args, "critic_readouts", []))
    fixed_critic_replays = parse_external_readouts(
        getattr(args, "fixed_critic_replays", [])
    )
    online_checkpoints = parse_external_readouts(
        getattr(args, "online_checkpoints", [])
    )
    online_checkpoint_arms = parse_online_checkpoint_arms(
        getattr(args, "online_checkpoint_arms", [])
    )
    if not set(online_checkpoint_arms).issubset(online_checkpoints):
        raise ValueError("Checkpoint arm labels require matching online checkpoints")
    online_checkpoint_arms = {
        label: online_checkpoint_arms.get(label, "correct")
        for label in online_checkpoints
    }
    online_checkpoint_components = parse_online_checkpoint_components(
        getattr(args, "online_checkpoint_components", [])
    )
    if not set(online_checkpoint_components).issubset(online_checkpoints):
        raise ValueError(
            "Checkpoint component labels require matching online checkpoints"
        )
    online_checkpoint_components = {
        label: online_checkpoint_components.get(label, "joint")
        for label in online_checkpoints
    }
    sparse_motion_heads = parse_external_readouts(
        getattr(args, "sparse_motion_heads", [])
    )
    sparse_zero_controls = set(getattr(args, "sparse_zero_controls", []))
    if fixed_replays and (
        set(fixed_replays) != set(args.methods)
        or not getattr(args, "deferred_online_updates", False)
    ):
        raise ValueError(
            "Fixed replay requires deferred updates and one explicit replay per method"
        )
    if not set(critic_readouts).issubset(args.methods):
        raise ValueError("Critic readout labels must also be requested methods")
    if not set(fixed_critic_replays).issubset(fixed_replays):
        raise ValueError("Fixed critic replay requires the same fixed base method")
    candidate_critics = set(getattr(args, "candidate_critics", []))
    relative_critics = set(getattr(args, "relative_critics", []))
    concat_control_critics = set(getattr(args, "concat_control_critics", []))
    concat_relative_critics = set(getattr(args, "concat_relative_critics", []))
    unit_concat_relative_critics = set(
        getattr(args, "unit_concat_relative_critics", [])
    )
    conservative_selectors = set(getattr(args, "conservative_selectors", []))
    actor_zero_concat = set(getattr(args, "actor_zero_concat", []))
    actor_physical_concat = set(getattr(args, "actor_physical_concat", []))
    actor_physical_outcomes_concat = set(
        getattr(args, "actor_physical_outcomes_concat", [])
    )
    actor_physical_gates = set(getattr(args, "actor_physical_gates", []))
    response_aux_critics = set(getattr(args, "critic_response_aux", []))
    shuffled_response_aux_critics = set(
        getattr(args, "critic_response_aux_shuffled", [])
    )
    consequence_dueling_critics = set(getattr(args, "critic_consequence_dueling", []))
    if response_aux_critics & shuffled_response_aux_critics:
        raise ValueError("Correct and shuffled response supervision must be disjoint")
    if not (response_aux_critics | shuffled_response_aux_critics).issubset(
        args.methods
    ):
        raise ValueError("Critic response supervision requires a requested method")
    if not consequence_dueling_critics.issubset(concat_relative_critics):
        raise ValueError(
            "Consequence-dueling critics require relative-consequence conditioning"
        )
    critic_groups = (
        candidate_critics,
        relative_critics,
        concat_control_critics,
        concat_relative_critics,
        unit_concat_relative_critics,
        set(sparse_motion_heads),
    )
    if sum(map(len, critic_groups)) != len(set().union(*critic_groups)):
        raise ValueError("Critic conditioning labels must be disjoint")
    unknown_candidates = set().union(*critic_groups) - set(args.methods)
    if unknown_candidates:
        raise ValueError("Candidate critic labels must also be requested methods")
    if not conservative_selectors.issubset(args.methods):
        raise ValueError("Conservative selector labels must also be requested methods")
    if not set(online_checkpoints).issubset(
        conservative_selectors & concat_relative_critics & consequence_dueling_critics
    ):
        raise ValueError(
            "Online checkpoints require conservative consequence-dueling selectors"
        )
    if not set(online_checkpoints).issubset(critic_readouts):
        raise ValueError("Online checkpoints require a matched consequence readout")
    if not sparse_zero_controls.issubset(sparse_motion_heads):
        raise ValueError("Sparse zero controls require a sparse-motion head")
    actor_groups = (
        actor_zero_concat,
        actor_physical_concat,
        actor_physical_outcomes_concat,
    )
    if sum(map(len, actor_groups)) != len(set().union(*actor_groups)):
        raise ValueError("Actor conditioning labels must be disjoint")
    actor_conditioned = set().union(*actor_groups)
    if not actor_conditioned.issubset(args.methods):
        raise ValueError("Actor conditioning labels must also be requested methods")
    if not actor_physical_gates.issubset(actor_conditioned):
        raise ValueError("Actor physical gates require an appended actor channel")
    actor_physical = actor_physical_concat | actor_physical_outcomes_concat
    if not actor_physical.issubset(critic_readouts):
        raise ValueError(
            "Physical actor conditioning requires a separate consequence readout"
        )
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
            actor_conditioning = "base"
            if method in actor_zero_concat:
                actor_conditioning = "zero_concat"
            elif method in actor_physical_concat:
                actor_conditioning = "physical_concat"
            elif method in actor_physical_outcomes_concat:
                actor_conditioning = "physical_outcomes_concat"
            if method in external_readouts:
                path = external_readouts[method]
                expected_method = None
            elif method.endswith("_anchor"):
                base_method = method.removesuffix("_anchor")
                # Every arm has two 400-step encoders. The fixed prior uses a
                # different initialization, including in the dual-reconstruction
                # control; this avoids a rank-deficient duplicate baseline.
                anchor = args.probes / "action_recon-seed59" / "readout.pt"
                path = args.probes / f"{base_method}-seed{args.seed}" / "readout.pt"
                expected_method = base_method
            for suffix in ("_outcomes", "_mixed"):
                if method not in external_readouts and method.endswith(suffix):
                    base_method, mode = method.removesuffix(suffix), suffix[1:]
                    path = args.probes / f"{base_method}-seed{args.seed}" / "readout.pt"
                    expected_method = base_method
            if method not in external_readouts and not method.endswith("_anchor"):
                path = args.probes / f"{base_method}-seed{args.seed}" / "readout.pt"
                expected_method = base_method
            readout = FrozenReadout(
                path,
                mode=mode,
                anchor=anchor,
                consequence=critic_readouts.get(method),
                actor_conditioning=actor_conditioning,
                expected_dataset=args.dataset,
                expected_method=expected_method,
            )
            sparse_conditioner = None
            sparse_payload = None
            critic_conditioner_override = None
            critic_conditioning = "latent"
            if method in candidate_critics:
                critic_conditioning = "candidate_mixed"
            elif method in relative_critics:
                critic_conditioning = "candidate_relative"
            elif method in concat_control_critics:
                critic_conditioning = "candidate_zero_concat"
            elif method in concat_relative_critics:
                critic_conditioning = "candidate_relative_concat"
            elif method in unit_concat_relative_critics:
                critic_conditioning = "candidate_relative_unit_concat"
            elif method in sparse_motion_heads:
                critic_conditioning = "sparse_relative_concat"
                sparse_conditioner, sparse_payload = load_sparse_motion_conditioner(
                    sparse_motion_heads[method],
                    expected_readout_sha256=file_sha256(path),
                    zero_channel=method in sparse_zero_controls,
                )
                critic_conditioner_override = sparse_conditioner
            if method in online_checkpoints:
                critic_conditioner_override = SignedCandidateConsequence(
                    readout.candidate_conditioner(
                        "relative_concat",
                        component=online_checkpoint_components[method],
                    ),
                    {"correct": 1, "reversed": -1, "zero": 0}[
                        online_checkpoint_arms[method]
                    ],
                )
            critic_bottleneck_dim = (
                256 if critic_conditioning.endswith("_concat") else None
            )
            actor_bottleneck_dim = 256 if actor_conditioning != "base" else None
            write_json(
                args.output / f"{method}-readout.json",
                {
                    "checkpoint": str(path),
                    "checkpoint_sha256": file_sha256(path),
                    "method": readout.payload["method"],
                    "effect_supervision": readout.payload.get("effect_supervision"),
                    "effect_measurements_sha256": readout.payload.get(
                        "effect_measurements_sha256"
                    ),
                    "critic_conditioning": critic_conditioning,
                    "actor_conditioning": actor_conditioning,
                    "actor_physical_gate": method in actor_physical_gates,
                    "correction_limit": getattr(args, "correction_limit", None),
                    "correction_axes": getattr(args, "correction_axes", None),
                    "residual_output": getattr(args, "residual_output", False),
                    "conservative_selector": method in conservative_selectors,
                    "online_checkpoint_arm": online_checkpoint_arms.get(method),
                    "online_checkpoint_component": online_checkpoint_components.get(
                        method
                    ),
                    "candidate_selection_count": getattr(
                        args, "candidate_selection_count", 0
                    ),
                    "candidate_selection_margin": getattr(
                        args, "candidate_selection_margin", 0.0
                    ),
                    "candidate_uncertainty_penalty": getattr(
                        args, "candidate_uncertainty_penalty", 0.0
                    ),
                    "critic_response_aux_weight": (
                        getattr(args, "critic_response_aux_weight", 0.0)
                        if method
                        in response_aux_critics | shuffled_response_aux_critics
                        else 0.0
                    ),
                    "critic_response_aux_shuffled": method
                    in shuffled_response_aux_critics,
                    "critic_consequence_dueling": method in consequence_dueling_critics,
                    "consequence_checkpoint": (
                        str(critic_readouts[method])
                        if method in critic_readouts
                        else None
                    ),
                    "sparse_motion_checkpoint": (
                        str(sparse_motion_heads[method])
                        if method in sparse_motion_heads
                        else None
                    ),
                    "sparse_motion_arm": (
                        sparse_payload.get("arm")
                        if sparse_payload is not None
                        else None
                    ),
                    "sparse_zero_control": method in sparse_zero_controls,
                },
            )
            learner, fidelity = warmup(
                readout,
                dataset,
                args.seed,
                args.warmup_steps,
                critic_conditioning,
                getattr(args, "correction_limit", None),
                (
                    tuple(args.correction_axes)
                    if getattr(args, "correction_axes", None) is not None
                    else None
                ),
                getattr(args, "residual_output", False),
                critic_bottleneck_dim,
                actor_bottleneck_dim,
                method in actor_physical_gates,
                (
                    getattr(args, "candidate_selection_count", 0)
                    if method in conservative_selectors
                    else 0
                ),
                getattr(args, "candidate_selection_margin", 0.0),
                getattr(args, "candidate_uncertainty_penalty", 0.0),
                critic_conditioner_override=critic_conditioner_override,
                critic_response_aux_weight=(
                    getattr(args, "critic_response_aux_weight", 0.0)
                    if method in response_aux_critics | shuffled_response_aux_critics
                    else 0.0
                ),
                critic_response_aux_shuffled=method in shuffled_response_aux_critics,
                critic_consequence_dueling=method in consequence_dueling_critics,
            )
            write_json(args.output / f"{method}-warmup.json", fidelity)
            torch.save(learner.state_dict(), args.output / f"{method}-warmup.pt")
            validation_results = []
            for case in validation_cases:
                row = rollout_case(env, reference, readout, learner, case, 520)
                validation_results.append(row)
                records.append(row | {"method": method, "stage": "warmup"})
                write_json(args.output / "episodes.json", records)
            gate, gate_rule = warmup_gate(
                fidelity,
                validation_results,
                getattr(args, "residual_output", False),
            )
            write_json(
                args.output / f"{method}-gate.json",
                {
                    "passed": gate,
                    "selection_split": "validation",
                    "rule": gate_rule,
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
                if method in online_checkpoints:
                    checkpoint_payload = load_gated_online_checkpoint(
                        online_checkpoints[method],
                        expected_arm=online_checkpoint_arms[method],
                        expected_component=online_checkpoint_components[method],
                        optimizer_seed=args.seed,
                        base_readout=path,
                        consequence_readout=critic_readouts[method],
                    )
                    learner.load_state_dict(checkpoint_payload["state_dict"])
                    write_json(
                        args.output / f"{method}-online-checkpoint.json",
                        {
                            "source": str(online_checkpoints[method]),
                            "source_sha256": file_sha256(online_checkpoints[method]),
                            "optimizer_seed": args.seed,
                            "arm": checkpoint_payload["arm"],
                            "consequence_component": checkpoint_payload[
                                "consequence_component"
                            ],
                            "train_trace_sha256": checkpoint_payload[
                                "train_trace_sha256"
                            ],
                            "safety_rule": (
                                "checkpoint contract preserves a 0.1 bounded residual "
                                "on action axis 0; all other axes remain reference"
                            ),
                        },
                    )
                    torch.save(
                        learner.state_dict(), args.output / f"{method}-online.pt"
                    )
                elif method in fixed_replays:
                    source = fixed_replays[method]
                    replay = torch.load(source, map_location="cpu", weights_only=True)
                    critic_source = fixed_critic_replays.get(method)
                    if critic_source is not None:
                        critic_replay = torch.load(
                            critic_source, map_location="cpu", weights_only=True
                        )
                        replay = merge_replay_views(replay, critic_replay)
                    replay = attach_actor_view(replay, actor_conditioning, readout)
                    fixed_summary = validate_fixed_replay(replay)
                    write_json(
                        args.output / f"{method}-fixed-replay.json",
                        {
                            "source": str(source),
                            "source_sha256": file_sha256(source),
                            "critic_source": (
                                str(critic_source)
                                if critic_source is not None
                                else None
                            ),
                            "critic_source_sha256": (
                                file_sha256(critic_source)
                                if critic_source is not None
                                else None
                            ),
                            **fixed_summary,
                        },
                    )
                    torch.save(replay, args.output / f"{method}-replay.pt")
                else:
                    for case in train_cases:
                        budget = min(520, args.online_ticks - count)
                        if budget <= 0:
                            break
                        row = rollout_case(
                            env,
                            reference,
                            readout,
                            learner,
                            case,
                            budget,
                            replay,
                            update_online=not getattr(
                                args, "deferred_online_updates", False
                            ),
                        )
                        count += row["ticks"]
                        records.append(
                            row | {"method": method, "stage": "online_train"}
                        )
                        write_json(args.output / "episodes.json", records)
                        torch.save(
                            learner.state_dict(), args.output / f"{method}-online.pt"
                        )
                        torch.save(replay, args.output / f"{method}-replay.pt")
                if getattr(args, "deferred_online_updates", False) and method not in (
                    online_checkpoints
                ):
                    skip_updates = skip_deferred_updates(
                        replay,
                        getattr(args, "require_positive_replay", False),
                    )
                    deferred_metrics = (
                        [] if skip_updates else [learner.update(replay) for _ in replay]
                    )
                    update_summary = deferred_update_summary(deferred_metrics, replay)
                    write_json(
                        args.output / f"{method}-deferred.json",
                        {
                            **update_summary,
                            "updates_skipped": skip_updates,
                            "skip_reason": (
                                "no_positive_transition" if skip_updates else None
                            ),
                            "post_update_fidelity": learner.following(replay),
                            "trace_sha256": replay_trace_sha256(replay),
                            "last_update": deferred_metrics[-1]
                            if deferred_metrics
                            else None,
                        },
                    )
                    torch.save(
                        learner.state_dict(), args.output / f"{method}-online.pt"
                    )
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
        "complete": True,
        "state": "complete",
        "episodes": len(records),
        "seed": args.seed,
        "role": "small development pilots, not final-test success claims",
        "warmup_steps": args.warmup_steps,
        "online_tick_cap": args.online_ticks,
        "correction_limit": getattr(args, "correction_limit", None),
        "correction_axes": getattr(args, "correction_axes", None),
        "residual_output": getattr(args, "residual_output", False),
        "actor_physical_gates": getattr(args, "actor_physical_gates", []),
        "critic_response_aux": getattr(args, "critic_response_aux", []),
        "critic_response_aux_shuffled": getattr(
            args, "critic_response_aux_shuffled", []
        ),
        "critic_response_aux_weight": getattr(args, "critic_response_aux_weight", 0.0),
        "critic_consequence_dueling": getattr(args, "critic_consequence_dueling", []),
        "deferred_online_updates": getattr(args, "deferred_online_updates", False),
        "require_positive_replay": getattr(args, "require_positive_replay", False),
        "conservative_selectors": getattr(args, "conservative_selectors", []),
        "candidate_selection_count": getattr(args, "candidate_selection_count", 0),
        "candidate_selection_margin": getattr(args, "candidate_selection_margin", 0.0),
        "candidate_uncertainty_penalty": getattr(
            args, "candidate_uncertainty_penalty", 0.0
        ),
        "fixed_replays": {
            key: str(value)
            for key, value in parse_external_readouts(
                getattr(args, "fixed_replays", [])
            ).items()
        },
        "online_checkpoints": {
            key: str(value) for key, value in online_checkpoints.items()
        },
        "online_checkpoint_arms": online_checkpoint_arms,
        "online_checkpoint_components": online_checkpoint_components,
        "control_protocol_version": 2,
        "reset_pairing_protocol": "current_run_physics_state_atol_1e-9",
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
    parser.add_argument("--correction-limit", type=float)
    parser.add_argument(
        "--correction-axes",
        type=int,
        choices=range(7),
        nargs="+",
        help="Restrict bounded corrections to listed action dimensions",
    )
    parser.add_argument(
        "--residual-output",
        action="store_true",
        help="Zero-initialize the actor as a bounded residual around reference",
    )
    parser.add_argument(
        "--deferred-online-updates",
        action="store_true",
        help="Collect the bounded rollout before applying matched online updates",
    )
    parser.add_argument(
        "--require-positive-replay",
        action="store_true",
        help="Keep the residual at reference when sparse replay has no success",
    )
    parser.add_argument("--development-cases", type=Path)
    parser.add_argument(
        "--external-readouts",
        nargs="*",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--fixed-replays",
        nargs="*",
        default=[],
        metavar="LABEL=PATH",
        help="Use one immutable positive replay per method across optimizer seeds",
    )
    parser.add_argument(
        "--critic-readouts",
        nargs="*",
        default=[],
        metavar="LABEL=PATH",
        help="Use a separate physical-token encoder for candidate consequences",
    )
    parser.add_argument(
        "--fixed-critic-replays",
        nargs="*",
        default=[],
        metavar="LABEL=PATH",
        help="Attach matched physical-token views to immutable base replay",
    )
    parser.add_argument(
        "--online-checkpoints",
        nargs="*",
        default=[],
        metavar="LABEL=PATH",
        help="Evaluate exact checkpoints emitted by a passed critic gate",
    )
    parser.add_argument(
        "--online-checkpoint-arms",
        nargs="*",
        default=[],
        metavar="LABEL=ARM",
        help="Preserve the correct, reversed or zero consequence arm at inference",
    )
    parser.add_argument(
        "--online-checkpoint-components",
        nargs="*",
        default=[],
        metavar="LABEL=COMPONENT",
        help="Retain joint, response-only or geometry-only physical consequences",
    )
    parser.add_argument(
        "--sparse-motion-heads",
        nargs="*",
        default=[],
        metavar="LABEL=PATH",
        help="Condition a critic on a frozen sparse candidate-motion head",
    )
    parser.add_argument(
        "--sparse-zero-controls",
        nargs="*",
        default=[],
        help="Replace the matched sparse consequence channel with zeros",
    )
    parser.add_argument("--candidate-critics", nargs="*", default=[])
    parser.add_argument("--relative-critics", nargs="*", default=[])
    parser.add_argument("--concat-control-critics", nargs="*", default=[])
    parser.add_argument("--concat-relative-critics", nargs="*", default=[])
    parser.add_argument("--unit-concat-relative-critics", nargs="*", default=[])
    parser.add_argument(
        "--actor-zero-concat",
        nargs="*",
        default=[],
        help="Give the actor a 256-D base-plus-zero token capacity control",
    )
    parser.add_argument(
        "--actor-physical-concat",
        nargs="*",
        default=[],
        help="Give the actor a 256-D semantic-plus-physical token",
    )
    parser.add_argument(
        "--actor-physical-outcomes-concat",
        nargs="*",
        default=[],
        help="Append predicted reference-action physical outcomes to the actor token",
    )
    parser.add_argument(
        "--actor-physical-gates",
        nargs="*",
        default=[],
        help="Learn a zero-initialized scalar gate for an appended actor channel",
    )
    parser.add_argument(
        "--critic-response-aux",
        nargs="*",
        default=[],
        help="Train selected critics to predict measured state response",
    )
    parser.add_argument(
        "--critic-response-aux-shuffled",
        nargs="*",
        default=[],
        help="Use rolled response labels as a matched negative control",
    )
    parser.add_argument("--critic-response-aux-weight", type=float, default=0.1)
    parser.add_argument(
        "--critic-consequence-dueling",
        nargs="*",
        default=[],
        help=(
            "Force selected relative-consequence critics to express every "
            "action advantage through the predicted physical outcome"
        ),
    )
    parser.add_argument("--conservative-selectors", nargs="*", default=[])
    parser.add_argument("--candidate-selection-count", type=int, default=5)
    parser.add_argument("--candidate-selection-margin", type=float, default=0.0)
    parser.add_argument("--candidate-uncertainty-penalty", type=float, default=1.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["action_recon", "geometry", "response", "joint"],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
