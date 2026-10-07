# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compare BC-only and Q+BC on one immutable cache of simulator transitions.

This diagnostic uses the production RLT model and losses without Ray, FSDP,
VLA inference or simulator execution. Its metrics are not control success.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from toolkits.rlt.evaluate_memory_checkpoint import tensor_digest


def write_json(path: Path, value: dict) -> None:
    """Atomically publish a finite JSON result."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def map_tensors(value, function):
    """Apply a tensor operation to a nested batch."""
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}
    return function(value)


def transition_batch(record: dict) -> dict:
    """Validate a single simulator replay row and remove its time axis.

    General trajectory files and absent intervention metadata are rejected.
    This first diagnostic only supports zero-context, unassisted data.
    """
    fields = (
        "actions",
        "rewards",
        "dones",
        "terminations",
        "truncations",
        "intervene_flags",
    )
    result = {name: record[name] for name in fields}
    result.update({name: record[name] for name in ("curr_obs", "next_obs")})
    if record["max_episode_length"] != 1:
        raise ValueError("Expected one simulator transition per file")

    def flatten(tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.shape[:2] != (1, 1):
            raise ValueError("Expected a [1, 1, ...] simulator transition tensor")
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite replay tensor")
        return tensor.reshape(1, *tensor.shape[2:]).contiguous()

    result = map_tensors(result, flatten)
    if result["intervene_flags"].any():
        raise ValueError("This diagnostic requires unassisted replay")
    if not record["forward_inputs"]["record_transition"].all():
        raise ValueError("Expected a recorded critical-phase transition")
    if not torch.equal(result["dones"], result["terminations"] | result["truncations"]):
        raise ValueError("Replay termination flags disagree")
    return result


def prepare_cache(replay: Path, output: Path, *, limit: int, split_seed: int) -> dict:
    """Copy available records without mutating an incomplete source checkpoint.

    Split by collection model version, not by individual chunk. A version split
    is a development check; it does not establish unseen physical instances.
    """
    if not 32 <= limit <= 8192:
        raise ValueError("Expected 32..8192 source files")
    output.mkdir(parents=True, exist_ok=False)
    index_bytes = (replay / "trajectory_index.json").read_bytes()
    index = json.loads(index_bytes)["trajectory_index"]
    expected = {
        f"trajectory_{key}_{entry['model_weights_id']}.pt": entry
        for key, entry in index.items()
    }
    names = {name for name in os.listdir(replay) if name.endswith(".pt")}
    if names - expected.keys():
        raise ValueError("Saved files are missing from the source index")
    selected = sorted(
        names, key=lambda name: hashlib.sha256(f"{split_seed}:{name}".encode()).digest()
    )[:limit]
    if len(selected) < 32:
        raise ValueError("Not enough available transitions")
    groups = sorted({expected[name]["model_weights_id"] for name in selected})
    groups.sort(
        key=lambda group: hashlib.sha256(f"{split_seed}:{group}".encode()).digest()
    )
    if len(groups) < 5:
        raise ValueError("Need at least five collection versions for a group split")
    validation_groups = set(groups[: max(1, len(groups) // 5)])
    batches = {"train": [], "validation": []}
    sources = []

    def load(name):
        data = (replay / name).read_bytes()
        record = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
        group = expected[name]["model_weights_id"]
        if record["model_weights_id"] != group:
            raise ValueError("Saved record has the wrong collection version")
        split = "validation" if group in validation_groups else "train"
        return transition_batch(record), {
            "file": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "collection_version": group,
            "split": split,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, (batch, source) in enumerate(pool.map(load, selected), 1):
            batches[source["split"]].append(batch)
            sources.append(source)
            if number % 128 == 0:
                print(
                    f"Validated {number}/{len(selected)} cached transitions", flush=True
                )

    def concatenate(items):
        if isinstance(items[0], dict):
            return {key: concatenate([item[key] for item in items]) for key in items[0]}
        return torch.cat(items)

    cache = {}
    statistics = {}
    for split, rows in batches.items():
        if not rows:
            raise ValueError("Empty cache split")
        cache[split] = concatenate(rows)
        statistics[split] = {
            "transitions": len(rows),
            "collection_versions": len(
                {s["collection_version"] for s in sources if s["split"] == split}
            ),
            "positive_reward_rows": int((cache[split]["rewards"] > 0).any(-1).sum()),
            "done_rows": int(cache[split]["dones"].any(-1).sum()),
        }
    torch.save(cache, output / "transitions.pt")
    manifest = {
        "source_replay": str(replay.resolve()),
        "source_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "indexed_records": len(index),
        "available_records": len(names),
        "missing_indexed_records": len(expected.keys() - names),
        "selected_records": len(sources),
        "split_seed": split_seed,
        "split_unit": "collection_model_version",
        "cache_sha256": hashlib.sha256(
            (output / "transitions.pt").read_bytes()
        ).hexdigest(),
        "statistics": statistics,
        "sources": sources,
        "limitations": [
            "Only saved records are used; this is not a full replay resume.",
            "A collection-version split does not establish unseen physical conditions.",
            "Source heads have already trained on these records; only fresh heads have a held-out split.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def build_head(cfg) -> RLTMLPPolicy:
    """Construct the same zero-context head as the matched online campaign."""
    model = cfg.actor.model
    if model.model_type != "rlt_mlp_policy" or model.q_head_type != "default":
        raise ValueError("Expected the RLT MLP with default twin Q heads")
    if model.interaction_memory.reader_type != "zero":
        raise ValueError("Diagnose the zero-context baseline before memory variants")
    return RLTMLPPolicy(
        **{
            name: model[name]
            for name in (
                "z_dim",
                "proprio_dim",
                "action_dim",
                "num_action_chunks",
                "ref_num_action_chunks",
                "add_q_head",
                "q_head_type",
                "fixed_std",
            )
        },
        interaction_memory=OmegaConf.to_container(
            model.interaction_memory, resolve=True
        ),
    )


def split_parameters(model) -> tuple[list, list]:
    """Match the production SAC worker's actor/critic optimizer ownership."""
    actor, critic = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            target = (
                critic
                if any(
                    s in name for s in ("encoders", "encoder", "q_head", "state_proj")
                )
                else actor
            )
            target.append(parameter)
    if not actor or not critic:
        raise ValueError("Both optimizer groups must be nonempty")
    return actor, critic


def optimizer(parameters, config):
    """Use production Adam defaults with no weight decay."""
    return torch.optim.Adam(
        parameters,
        lr=config.lr,
        betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
        eps=config.get("adam_eps", 1e-8),
    )


@torch.no_grad()
def evaluate_head(model, batch: dict, *, batch_size: int = 128) -> dict:
    """Measure deterministic imitation and learned Q preference on cached states."""
    from rlinf.models.embodiment.base_policy import ForwardType

    totals = Counter()
    model.eval()
    count = len(batch["actions"])
    for start in range(0, count, batch_size):
        part = map_tensors(batch, lambda value: value[start : start + batch_size])
        obs = part["curr_obs"]
        pi, _, _ = model(forward_type=ForwardType.SAC, obs=obs, deterministic=True)
        ref = obs["ref_chunk"].reshape(len(pi), -1)[:, : pi.shape[-1]]
        actor_q = model(forward_type=ForwardType.SAC_Q, obs=obs, actions=pi)
        ref_q = model(forward_type=ForwardType.SAC_Q, obs=obs, actions=ref)
        delta = pi - ref
        totals["reference_mse"] += delta.square().mean(-1).sum().item()
        totals["reference_mae"] += delta.abs().mean(-1).sum().item()
        totals["q1_actor_minus_reference"] += (
            (actor_q[..., 0] - ref_q[..., 0]).sum().item()
        )
        totals["q1_prefers_actor_fraction"] += (
            (actor_q[..., 0] > ref_q[..., 0]).sum().item()
        )
        totals["mean_abs_q_disagreement"] += (
            (actor_q[..., 0] - actor_q[..., 1]).abs().sum().item()
        )
        totals["saturated_action_fraction"] += (
            (pi.abs() > 0.99).float().mean(-1).sum().item()
        )
    model.train()
    return {key: value / count for key, value in totals.items()}


def train_arm(
    cfg, cache: dict, output: Path, *, variant: str, steps: int, seed: int
) -> dict:
    """Train a fresh head using the worker losses and its microbatch order.

    Both arms train a critic. BC-only zeroes the actor's Q coefficient while
    preserving its BC schedule. Worker timing decorators alone are bypassed;
    their underlying loss functions are used without copying the equations.
    """
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin

    if variant not in {"bc_only", "q_bc"} or not 1 <= steps <= 20000:
        raise ValueError("Expected bc_only/q_bc and 1..20000 critic updates")
    cfg = copy.deepcopy(cfg)
    if (
        cfg.algorithm.q_head_type != "default"
        or cfg.algorithm.target_update_type != "all"
    ):
        raise ValueError("Diagnostic supports default Q heads and all-parameter EMA")
    if variant == "bc_only":
        cfg.algorithm.q_weight = 0.0
        for key in ("warmup_q_weight", "online_q_weight"):
            cfg.algorithm.actor_weight_schedule[key] = 0.0
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    torch.manual_seed(seed)
    model = build_head(cfg)
    initial_digest = tensor_digest(model.state_dict())
    torch.save(model.state_dict(), output / "initial.pt")
    target = copy.deepcopy(model).requires_grad_(False)
    losses = RLTACLossMixin()
    losses.cfg, losses.torch_dtype = cfg, torch.float32
    losses.model, losses.target_model = model, target
    actor_parameters, critic_parameters = split_parameters(model)
    actor_optim = optimizer(actor_parameters, cfg.actor.optim)
    critic_optim = optimizer(critic_parameters, cfg.actor.critic_optim)
    sampler = torch.Generator().manual_seed(seed + 10000)
    batch_size = int(cfg.actor.global_batch_size)
    micro = int(cfg.actor.micro_batch_size)
    if batch_size % micro:
        raise ValueError("Global batch must be divisible by micro batch")
    accumulation = batch_size // micro
    ratio = int(cfg.algorithm.critic_actor_ratio)
    sample_digest = hashlib.sha256()
    rows = [
        {
            "critic_updates": 0,
            "actor_updates": 0,
            "validation": evaluate_head(model, cache["validation"]),
        }
    ]
    start_time = time.monotonic()
    actor_updates = 0
    for step in range(steps):
        losses.update_step = step
        ix = torch.randint(
            len(cache["train"]["actions"]), (batch_size,), generator=sampler
        )
        sample_digest.update(ix.numpy().tobytes())
        batch = map_tensors(cache["train"], lambda value: value[ix])
        parts = [
            map_tensors(batch, lambda value, i=i: value[i : i + micro])
            for i in range(0, batch_size, micro)
        ]
        critic_optim.zero_grad()
        for part in parts:
            loss, _ = RLTACLossMixin.forward_critic.__wrapped__(losses, part)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite critic loss")
            (loss / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.actor.critic_optim.clip_grad,
            error_if_nonfinite=True,
        )
        critic_optim.step()
        if step % ratio == 0:
            actor_optim.zero_grad()
            for part in parts:
                loss, _, metrics = RLTACLossMixin.forward_actor.__wrapped__(
                    losses, part
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite actor loss")
                (loss / accumulation).backward()
            # Match the online worker's whole-model gradient clipping, including
            # critic gradients still present at this point in its update cycle.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.actor.optim.clip_grad, error_if_nonfinite=True
            )
            actor_optim.step()
            actor_updates += 1
        if step % int(cfg.algorithm.target_update_freq) == 0:
            with torch.no_grad():
                for online, frozen in zip(
                    model.parameters(), target.parameters(), strict=True
                ):
                    frozen.lerp_(online, float(cfg.algorithm.tau))
        completed = step + 1
        if completed % 256 == 0 or completed == steps:
            result = {
                "critic_updates": completed,
                "actor_updates": actor_updates,
                "wall_seconds": time.monotonic() - start_time,
                "train": evaluate_head(model, cache["train"]),
                "validation": evaluate_head(model, cache["validation"]),
                "last_actor": metrics,
                "weights_sha256": tensor_digest(model.state_dict()),
            }
            rows.append(result)
            torch.save(model.state_dict(), output / f"step_{completed}.pt")
            write_json(
                output / "progress.json",
                {"variant": variant, "seed": seed, "rows": rows},
            )
            print(
                json.dumps({"variant": variant, **result}, allow_nan=False), flush=True
            )
    summary = {
        "variant": variant,
        "seed": seed,
        "initial_weights_sha256": initial_digest,
        "sample_sequence_sha256": sample_digest.hexdigest(),
        "rows": rows,
        "critic_updates": steps,
        "actor_updates": actor_updates,
        "final_weights_sha256": tensor_digest(model.state_dict()),
        "status": "completed",
        "device": "cpu",
        "fresh_initialization": True,
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    """Prepare a read-only source cache or run both matched CPU diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--replay", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=2048)
    prepare.add_argument("--split-seed", type=int, default=601)
    train = sub.add_parser("train")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=2048)
    train.add_argument("--seed", type=int, default=1234)
    train.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_cache(
            args.replay, args.output, limit=args.limit, split_seed=args.split_seed
        )
        return
    if not 1 <= args.threads <= 8:
        raise ValueError("CPU diagnostic supports 1..8 threads")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    raw = (args.cache / "transitions.pt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["cache_sha256"]:
        raise ValueError("Frozen cache changed after preparation")
    cache = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    cfg = OmegaConf.load(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = [
        train_arm(
            cfg, cache, args.output / arm, variant=arm, steps=args.steps, seed=args.seed
        )
        for arm in ("bc_only", "q_bc")
    ]
    for field in (
        "initial_weights_sha256",
        "sample_sequence_sha256",
        "critic_updates",
        "actor_updates",
    ):
        if summaries[0][field] != summaries[1][field]:
            raise RuntimeError(f"Matched diagnostic differs in {field}")
    write_json(
        args.output / "comparison.json",
        {
            "cache_sha256": manifest["cache_sha256"],
            "matched": True,
            "arms": summaries,
            "limitations": [
                "Offline diagnostic, not online RL or control success.",
                "Fresh heads, fresh optimizer state, one seed per invocation.",
                "Q preference is a learned estimate, not a counterfactual return measurement.",
            ],
        },
    )


if __name__ == "__main__":
    main()
