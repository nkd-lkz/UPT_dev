# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Train new-RLT reconstruction and ARX X5 BC heads on isolated rollouts."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import subprocess
import time
from pathlib import Path

import torch
import yaml

from toolkits.rlt.experiment_queue import gpu_idle
from toolkits.rlt.pi05_baseline import configure_source
from toolkits.rlt.robodojo_grounding import ground_replay_rows, load_grounding
from toolkits.rlt.robodojo_learning import (
    ArxLearner,
    ArxNormalizer,
    learner_protocol,
    load_hidden,
    load_rollout_rows,
    make_encoder,
)


def save_json(path: Path, payload: dict) -> None:
    """Atomically publish one finite training status."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def normalization_path(config: Path) -> Path:
    """Resolve the pinned native ARX X5 statistics from a launch config."""
    policy = yaml.safe_load(config.read_text())["policy"]
    return (
        Path(policy["checkpoint_path"])
        / "assets"
        / policy["model_contract"]["normalization_id"]
        / "norm_stats.json"
    )


def encode_rows(encoder: torch.nn.Module, rows: list[dict]) -> list[dict]:
    """Freeze one RL token per pre-action observation and attach TD successors."""
    encoded = []
    encoder.eval()
    with torch.no_grad():
        for row in rows:
            hidden = torch.from_numpy(load_hidden(row))[None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                token = encoder.encode(hidden)[0, 0].float().cpu()
            encoded.append(
                {
                    **{
                        key: value for key, value in row.items() if key != "hidden_path"
                    },
                    "z": token,
                    "ref": torch.as_tensor(row["ref"], dtype=torch.float32),
                    "prop": torch.as_tensor(row["prop"], dtype=torch.float32),
                    "action": torch.as_tensor(row["action"], dtype=torch.float32),
                }
            )
    for index, row in enumerate(encoded):
        successor = (
            encoded[index + 1]
            if index + 1 < len(encoded)
            and encoded[index + 1]["episode"] == row["episode"]
            else row
        )
        row.update(
            next_z=successor["z"],
            next_ref=successor["ref"],
            next_prop=successor["prop"],
        )
    return encoded


def run(args: argparse.Namespace) -> None:
    """Fit the encoder and actor, preserving development rows as a fixed gate."""
    configure_source(args.source)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = grounding_digest = None
    if args.grounding is not None:
        tokenizer, grounding_digest = load_grounding(args.grounding, device="cuda")
    protocol = learner_protocol(tokenizer.token_dim if tokenizer else 0)
    normalizer = ArxNormalizer(normalization_path(args.config))
    rows = load_rollout_rows(args.collection, normalizer)
    train = [row for row in rows if row["role"] == "train"]
    development = [row for row in rows if row["role"] == "development"]
    if len({row["episode"] for row in train}) < 2 or not development:
        raise ValueError(
            "Require at least two train layouts and one development layout"
        )
    status = {
        "state": "encoder",
        "protocol": protocol,
        "encoder_steps": args.encoder_steps,
        "bc_steps": args.bc_steps,
        "train_chunks": len(train),
        "development_chunks": len(development),
        "grounding_sha256": grounding_digest,
        "online_rl": False,
    }
    save_json(args.output / "status.json", status)
    encoder = make_encoder()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=0.01)
    history = []
    for step in range(args.encoder_steps):
        sample = random.choice(train)
        hidden = torch.from_numpy(load_hidden(sample))[None].cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = encoder(hidden)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite encoder reconstruction")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step + 1 == args.encoder_steps:
            history.append({"step": step + 1, "loss": float(loss.item())})
            save_json(args.output / "encoder-history.json", {"updates": history})
    encoder.eval().requires_grad_(False)
    del optimizer
    torch.cuda.empty_cache()
    encoded = encode_rows(encoder, rows)
    if tokenizer is not None:
        # Reconstructed per-episode histories append grounded tokens exactly
        # as the online client would have produced them at each decision.
        encoded = ground_replay_rows(encoded, tokenizer)
    torch.save(encoded, args.output / "encoded-rollouts.pt")
    train = [row for row in encoded if row["role"] == "train"]
    development = [row for row in encoded if row["role"] == "development"]
    learner = ArxLearner(normalizer, token_dim=tokenizer.token_dim if tokenizer else 0)
    before = learner.following(development)
    status["state"] = "actor_bc"
    save_json(args.output / "status.json", status)
    for step in range(args.bc_steps):
        batch = random.choices(train, k=min(args.batch_size, len(train)))
        metrics = learner.update(batch, warmup=True)
        if step % 100 == 0 or step + 1 == args.bc_steps:
            save_json(args.output / "bc-status.json", {"step": step + 1, **metrics})
    learner.publish_warmup()
    after = learner.following(development)
    checkpoint = args.output / "warmup.pt"
    torch.save(
        {
            "protocol": protocol,
            "encoder": encoder.state_dict(),
            "learner": learner.state_dict(),
            "collections": [str(path) for path in args.collection],
            "grounding_sha256": grounding_digest,
            "train_layouts": [
                {"id": layout_id, "sha256": layout_hash}
                for layout_id, layout_hash in sorted({row["episode"] for row in train})
            ],
            "development_layouts": [
                {"id": layout_id, "sha256": layout_hash}
                for layout_id, layout_hash in sorted(
                    {row["episode"] for row in development}
                )
            ],
        },
        checkpoint,
    )
    status.update(
        state="complete" if after["mse"] < before["mse"] else "failed",
        initial_following=before,
        final_following=after,
        checkpoint=str(checkpoint),
        gate_passed=after["mse"] < before["mse"],
    )
    save_json(args.output / "status.json", status)
    if not status["gate_passed"]:
        raise RuntimeError("BC warmup did not improve held-out reference following")


def main() -> None:
    """Parse one bounded warmup job; online interaction is a separate command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--collection", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grounding", type=Path)
    parser.add_argument("--encoder-steps", type=int, default=1000)
    parser.add_argument("--bc-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--wait-idle-hours", type=float, default=0)
    parser.add_argument("--wait-for", type=Path, action="append", default=[])
    parser.add_argument("--wait-collections-hours", type=float, default=0)
    args = parser.parse_args()
    if (
        not 1 <= args.encoder_steps <= 10000
        or not 1 <= args.bc_steps <= 100000
        or not 1 <= args.batch_size <= 512
    ):
        parser.error("Invalid bounded training budget")
    if (
        args.gpu < 0
        or not 0 <= args.wait_idle_hours <= 72
        or not 0 <= args.wait_collections_hours <= 72
    ):
        parser.error("Require nonnegative GPU and wait budgets in [0, 72]")
    if args.grounding is not None and not args.grounding.is_file():
        parser.error("Grounding artifact does not exist")
    args.output.mkdir(parents=True, exist_ok=False)
    queue_status = {"state": "waiting_collections", "online_rl": False}
    save_json(args.output / "status.json", queue_status)
    collection_deadline = time.monotonic() + args.wait_collections_hours * 3600
    while args.wait_for:
        states = []
        for path in args.wait_for:
            try:
                states.append(json.loads(path.read_text()).get("state"))
            except (OSError, ValueError):
                states.append(None)
        if all(state == "complete" for state in states):
            break
        if any(state == "failed" for state in states):
            queue_status.update(state="failed", error=f"Required collection: {states}")
            save_json(args.output / "status.json", queue_status)
            raise RuntimeError(f"Required collection failed: {states}")
        if not args.wait_collections_hours or time.monotonic() >= collection_deadline:
            queue_status.update(state="failed", error=f"Collection timeout: {states}")
            save_json(args.output / "status.json", queue_status)
            raise TimeoutError(f"Collections not complete: {states}")
        queue_status["collections"] = states
        save_json(args.output / "status.json", queue_status)
        time.sleep(min(30, max(0, collection_deadline - time.monotonic())))
    lease = Path(f"/dev/shm/rlt-research-gpu{args.gpu}.lock").open("a")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + args.wait_idle_hours * 3600
        checks = 0
        while checks < 2:
            idle, detail = gpu_idle(args.gpu)
            checks = checks + 1 if idle else 0
            queue_status.update(state="waiting_gpu", gpu=detail, idle_checks=checks)
            save_json(args.output / "status.json", queue_status)
            if checks >= 2:
                break
            if not args.wait_idle_hours or time.monotonic() >= deadline:
                queue_status.update(state="failed", error=f"GPU occupied: {detail}")
                save_json(args.output / "status.json", queue_status)
                raise RuntimeError(f"GPU {args.gpu} occupied: {detail}")
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        uuid = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        try:
            run(args)
        except BaseException as exc:
            if args.output.exists():
                try:
                    status = json.loads((args.output / "status.json").read_text())
                except (OSError, ValueError):
                    status = {}
                status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
                save_json(args.output / "status.json", status)
            raise
    finally:
        lease.close()


if __name__ == "__main__":
    main()
