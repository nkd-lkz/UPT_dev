# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Train the new RLT encoder and accept a Pi0.5-conditioned small actor."""

from __future__ import annotations

import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from toolkits.rlt.libero_acceptance import Metrics, split_cache
from toolkits.rlt.libero_acceptance_learning import AcceptanceLearner, model_state
from toolkits.rlt.pi05_baseline import suite_protocol, write_json


def make_encoder(device: str = "cuda") -> torch.nn.Module:
    """Construct the pinned AlphaBrain RLT cross-attention reconstruction model."""
    from AlphaBrain.training.reinforcement_learning.algos.RLT.encoder_decoder import (
        RLTokenEncoderDecoder,
    )

    return RLTokenEncoderDecoder(
        hidden_dim=2048, num_heads=8, encoder_layers=2, decoder_layers=2
    ).to(device)


def make_learner(device: str = "cuda", objective: str = "q_bc") -> AcceptanceLearner:
    """Use the heads shared by upstream RLT/RLT_a with new token/action dimensions."""
    return AcceptanceLearner(
        device,
        objective=objective,
        ref_dropout=0.0,
        bottleneck_dim=2048,
        chunk_len=10,
        gripper_threshold=0.0,
    )


def validate_checkpoint(payload: dict, suite: str = "libero_goal") -> None:
    """Reject RLT_a and any release with a different observation/action contract."""
    if payload.get("protocol") != suite_protocol(suite):
        raise ValueError("Checkpoint is not the pinned RLT/Pi0.5 protocol")


def load_heads(
    path: Path, objective: str = "q_bc", suite: str = "libero_goal"
) -> tuple[torch.nn.Module, AcceptanceLearner]:
    """Restore frozen encoder and trainable heads from a complete warmup artifact."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint(payload, suite)
    encoder = make_encoder().eval().requires_grad_(False)
    encoder.load_state_dict(payload["encoder"], strict=True)
    learner = make_learner(objective=objective)
    learner.load_state_dict(payload["learner"])
    return encoder, learner


def encode_row(
    encoder: torch.nn.Module, prediction: dict, prop: np.ndarray | torch.Tensor
) -> dict:
    """Encode pre-action observations; reference labels remain native signed actions."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = (
            encoder.encode(prediction["hidden"][None].cuda().float())[0, 0]
            .float()
            .cpu()
        )
    return {
        "z": z,
        "ref": torch.as_tensor(prediction["ref"], dtype=torch.float32).clamp(-1, 1),
        "prop": torch.as_tensor(prop, dtype=torch.float32).clone(),
    }


def save_heads(
    path: Path,
    encoder: torch.nn.Module,
    learner: AcceptanceLearner,
    suite: str = "libero_goal",
) -> None:
    """Persist the exact protocol and all learner optimizer/target state."""
    temp = path.with_suffix(".tmp")
    torch.save(
        {
            "protocol": suite_protocol(suite),
            "encoder": model_state(encoder),
            "learner": learner.state_dict(),
        },
        temp,
    )
    temp.replace(path)


def warmup(args: Namespace, metrics: Metrics) -> dict:
    """Fit reconstruction and BC on development episodes, keeping one fifth out."""
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    validate_checkpoint(cache, args.suite)
    train, heldout = split_cache(cache["rows"])
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    encoder = make_encoder()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=0.01)

    def reconstruction(rows):
        encoder.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                _, loss = encoder(row["hidden"][None].cuda().float())
                losses.append(loss.item())
        return sum(losses) / len(losses)

    before = reconstruction(heldout)
    history = []
    for step in range(args.encoder_steps):
        encoder.train()
        batch = (
            torch.stack([r["hidden"] for r in random.choices(train, k=2)])
            .cuda()
            .float()
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = encoder(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite encoder reconstruction")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0:
            history.append({"step": step + 1, "reconstruction": loss.item()})
            write_json(args.output / "encoder_history.json", {"updates": history})
            metrics.append(
                {"encoder/update": step + 1, "encoder/reconstruction": loss.item()}
            )
    after = reconstruction(heldout)
    encoder.eval().requires_grad_(False)
    del optimizer
    torch.cuda.empty_cache()

    def encoded(rows):
        result = []
        for original in rows:
            row = encode_row(encoder, original, original["prop"])
            result.append(
                {
                    **row,
                    "action": row["ref"],
                    "next_z": row["z"],
                    "next_ref": row["ref"],
                    "next_prop": row["prop"],
                }
            )
        return result

    train_rows, heldout_rows = encoded(train), encoded(heldout)
    learner = make_learner()
    initial = learner.following(heldout_rows)
    for step in range(args.bc_steps):
        update_metrics = learner.update(train_rows, warmup=True)
        if step % 100 == 0:
            write_json(args.output / "bc_status.json", update_metrics)
            metrics.append({f"warmup/{k}": v for k, v in update_metrics.items()})
    learner.publish_warmup()
    final = learner.following(heldout_rows)
    metrics.append(
        {
            "encoder/heldout_before": before,
            "encoder/heldout_after": after,
            **{f"warmup/heldout_{k}": v for k, v in final.items()},
        }
    )
    save_heads(args.output / "warmup.pt", encoder, learner, args.suite)
    passed = (
        after < before
        and final["mse"] < initial["mse"]
        and final["gripper_disagreement"] <= 0.1
    )
    return {
        "gate_passed": passed,
        "reconstruction_initial": before,
        "reconstruction_final": after,
        "initial_following": initial,
        "final_following": final,
        "train_episodes": sorted({r["case_id"] for r in train}),
        "heldout_episodes": sorted({r["case_id"] for r in heldout}),
        "encoder_updates": args.encoder_steps,
        "actor_updates": learner.actor_updates,
        "critic_updates": learner.critic_updates,
        "training_scope": "engineering pilot: reference rollout observations, not official demo recipe",
    }
