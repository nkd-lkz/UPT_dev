# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Pretrain AlphaBrain's new-RLT encoder on frozen RoboTwin Pi0.5 features."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from toolkits.rlt.robotwin2_features import FrozenRoboTwinPi05
from toolkits.rlt.robotwin2_learning import PROTOCOL


def data_loader(checkpoint: Path, dataset: Path, *, batch_size: int, steps: int):
    """Build the pinned local LeRobot loader without contacting the Hub."""
    from openpi.training.data_loader import create_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    stats = checkpoint / "physical-intelligence/robotwin/norm_stats.json"
    config = get_openpi_config(
        "pi05_aloha_robotwin",
        model_path=str(checkpoint),
        batch_size=batch_size,
        repo_id=str(dataset),
        data_kwargs={"norm_stats_path": str(stats)},
    )
    config = dataclasses.replace(config, num_workers=0)
    return create_data_loader(
        config,
        shuffle=True,
        num_batches=steps,
        framework="pytorch",
    )


def train(args: argparse.Namespace) -> dict:
    """Run a bounded reconstruction job and publish one auditable checkpoint."""
    from AlphaBrain.training.reinforcement_learning.algos.RLT.encoder_decoder import (
        RLTokenEncoderDecoder,
    )

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    status_path = args.output / "status.json"
    status = {
        "state": "loading",
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "steps_requested": args.steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "started_at": time.time(),
    }

    def save_status() -> None:
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    save_status()
    try:
        reference = FrozenRoboTwinPi05(args.checkpoint, device=args.device)
        encoder = RLTokenEncoderDecoder(
            hidden_dim=2048,
            num_heads=8,
            encoder_layers=2,
            decoder_layers=2,
        ).to(args.device)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr)
        losses = []
        status["state"] = "training"
        save_status()
        for step, (observation, _) in enumerate(
            data_loader(
                args.checkpoint,
                args.dataset,
                batch_size=args.batch_size,
                steps=args.steps,
            ),
            start=1,
        ):
            hidden, valid = reference.prefix_features(observation)
            optimizer.zero_grad(set_to_none=True)
            _, loss = encoder(hidden, key_padding_mask=~valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite reconstruction loss at step {step}"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                encoder.parameters(), 10.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            status.update(
                step=step,
                loss=losses[-1],
                grad_norm=float(grad_norm),
                peak_memory_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if torch.cuda.is_available()
                else 0,
            )
            if step == 1 or step % args.log_interval == 0:
                save_status()
        if len(losses) != args.steps:
            raise RuntimeError(
                f"Data loader stopped after {len(losses)} of {args.steps} steps"
            )
        payload = {
            "protocol": PROTOCOL,
            "encoder": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": args.steps,
            "loss": losses[-1],
        }
        torch.save(payload, args.output / "encoder.pt")
        status.update(
            state="passed",
            finished_at=time.time(),
            initial_loss=losses[0],
            final_loss=losses[-1],
        )
        save_status()
        return status
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        save_status()
        raise


def main() -> None:
    """Parse the bounded Stage 1 job configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.log_interval < 1:
        parser.error("steps, batch-size, lr and log-interval must be positive")
    train(args)


if __name__ == "__main__":
    main()
