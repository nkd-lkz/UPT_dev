# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise AlphaBrain new-RLT reconstruction, BC, and Q+BC on real caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from toolkits.rlt.robodojo_learning import make_encoder
from toolkits.rlt.robotwin2_learning import AlohaLearner, AlohaNormalizer


def load_episode(path: Path) -> tuple[dict, list[dict]]:
    """Load a complete H=10 rollout and its immutable decision caches."""
    episode = json.loads(path.read_text())
    if episode.get("execution_horizon") != 10:
        raise ValueError("AlphaBrain learner smoke requires an H=10 rollout")
    rows = []
    for record in episode.get("records", []):
        with np.load(record["cache_file"]) as payload:
            hidden = np.asarray(payload["hidden"], dtype=np.float32)
            mask = np.asarray(payload["mask"], dtype=np.bool_)
            state = np.asarray(payload["state"], dtype=np.float32)
            reference = np.asarray(payload["reference"], dtype=np.float32)
        if hidden.shape != (768, 2048) or mask.shape != (768,) or not mask.all():
            raise ValueError(f"Invalid frozen feature cache: {record['cache_file']}")
        if state.shape != (14,) or reference.shape != (50, 14):
            raise ValueError(f"Invalid state/reference cache: {record['cache_file']}")
        rows.append(
            {
                "hidden": hidden,
                "mask": mask,
                "state": state,
                "reference": reference[:10],
                "reward": float(record["reward"]),
                "ticks": int(record["ticks"]),
                "terminated": bool(record["terminated"]),
                "truncated": bool(record["truncated"]),
            }
        )
    if len(rows) < 2:
        raise ValueError("Learner smoke requires at least two decisions")
    return episode, rows


def run(args: argparse.Namespace) -> dict:
    """Run requested reconstruction steps and one BC plus twin-Q update."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    episode, rows = load_episode(args.episode.resolve())
    encoder = make_encoder(str(device))
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    reconstruction = []
    encoder.train()
    for step in range(args.reconstruction_steps):
        row = rows[step % len(rows)]
        hidden = torch.from_numpy(row["hidden"]).unsqueeze(0).to(device)
        padding = ~torch.from_numpy(row["mask"]).unsqueeze(0).to(device)
        _, loss = encoder(hidden, key_padding_mask=padding)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            encoder.parameters(), 10, error_if_nonfinite=True
        )
        optimizer.step()
        reconstruction.append(
            {"loss": float(loss.item()), "grad_norm": float(grad_norm.item())}
        )

    encoder.eval()
    normalizer = AlohaNormalizer(args.norm_stats.resolve())
    learned_rows = []
    with torch.no_grad():
        for row in rows[:2]:
            hidden = torch.from_numpy(row["hidden"]).unsqueeze(0).to(device)
            padding = ~torch.from_numpy(row["mask"]).unsqueeze(0).to(device)
            z = encoder.encode(hidden, key_padding_mask=padding)[0, 0].cpu().numpy()
            reference = normalizer.encode(row["reference"], row["state"])
            learned_rows.append(
                row
                | {
                    "z": z,
                    "ref": reference,
                    "prop": normalizer.state(row["state"]),
                    "action": reference.copy(),
                    "executed_action": reference.copy(),
                }
            )

    learner = AlohaLearner(normalizer, device=str(device), objective="q_bc")
    warmup_metrics = learner.update([learned_rows[0]], warmup=True)
    learner.publish_warmup()
    current, following = learned_rows
    online = current | {
        "next_z": following["z"],
        "next_ref": following["ref"],
        "next_prop": following["prop"],
    }
    q_bc_metrics = learner.update([online], warmup=False)
    metrics = {
        "schema_version": 1,
        "benchmark": episode["benchmark"],
        "task": episode["task"],
        "execution_horizon": episode["execution_horizon"],
        "reconstruction": reconstruction,
        "warmup": warmup_metrics,
        "q_bc": q_bc_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reconstruction-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
