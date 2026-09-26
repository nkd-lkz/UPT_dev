# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Probe whether past joint-control evidence helps predict held-out joint changes.

This supervised diagnostic trains a fresh memory reader, not the online actor.
It measures predictive evidence, not reward, causal identification or RL speed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rlinf.algorithms.rlt.interaction_memory import (
    InteractionMemory,
    InteractionMemoryConfig,
)
from rlinf.models.embodiment.modules.rlt_memory_encoder import RLTMemoryEncoder


def episode_examples(states: torch.Tensor, actions: torch.Tensor) -> list[dict]:
    """Read history before execution; label with the true ten-tick successor."""
    memory = InteractionMemory(InteractionMemoryConfig())
    memory.begin_attempt("offline-episode")
    rows = []
    for t in range(0, len(states) - 10, 10):
        commands = actions[t : t + 10].clamp(-1, 1)
        rows.append(
            {
                **memory.snapshot(states[t]),
                "commands": commands.flatten(),
                "target": states[t + 10] - states[t],
            }
        )
        memory.append_completed(states[t], commands, states[t + 10])
    return rows


class ConsequenceProbe(nn.Module):
    """Use the production memory reader with a small diagnostic prediction head."""

    def __init__(self) -> None:
        super().__init__()
        self.reader = RLTMemoryEncoder(InteractionMemoryConfig())
        self.head = nn.Sequential(
            nn.Linear(9 + 80 + 64, 64), nn.SiLU(), nn.Linear(64, 9)
        )

    def forward(self, batch: dict, *, memory: bool) -> torch.Tensor:
        """Predict joint change with either real context or a zero context."""
        context = self.reader(batch)
        if not memory:
            context = context * 0
        return self.head(
            torch.cat((batch["memory_query"], batch["commands"], context), -1)
        )


def run_probe(dataset: Path, output: Path, *, episodes: int, steps: int) -> dict:
    """Run three fixed seeds on episode-disjoint data with equal optimization budgets."""
    import pyarrow.parquet as pq
    from torch.utils.data import default_collate

    if not 8 <= episodes <= 64 or not 1 <= steps <= 1000:
        raise ValueError("Diagnostic budget: 8..64 episodes, 1..1000 optimizer steps")
    output.mkdir(parents=True, exist_ok=False)
    info = json.loads((dataset / "meta/info.json").read_text())
    if info["fps"] != 10 or info["codebase_version"] != "v2.1":
        raise ValueError("Expected 10 Hz LeRobot v2.1")
    entries = [
        json.loads(line)
        for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()
    ][:episodes]
    if len(entries) != episodes:
        raise ValueError("Not enough episodes")
    splits, ids, sources = (
        {"train": [], "validation": []},
        {"train": [], "validation": []},
        [],
    )
    for entry in entries:
        idx = int(entry["episode_index"])
        path = dataset / info["data_path"].format(
            episode_chunk=idx // info["chunks_size"], episode_index=idx
        )
        data = pq.read_table(
            path, columns=["state", "actions", "frame_index", "episode_index"]
        ).to_pydict()
        frames = np.asarray(data["frame_index"]).reshape(-1)
        if not np.all(np.diff(frames) == 1) or not np.all(
            np.asarray(data["episode_index"]) == idx
        ):
            raise ValueError("Discontinuous or mixed episode")
        state, action = (
            torch.tensor(np.asarray(data[key]), dtype=torch.float32)
            for key in ("state", "actions")
        )
        split = "validation" if idx % 4 == 0 else "train"
        splits[split].extend(episode_examples(state, action))
        ids[split].append(idx)
        sources.append(
            {"episode": idx, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    batches = {key: default_collate(rows) for key, rows in splits.items()}
    results = []
    for seed in (2026, 2027, 2028):
        for enabled in (False, True):
            torch.manual_seed(seed)
            model = ConsequenceProbe()
            optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
            sampler = torch.Generator().manual_seed(seed + 1)
            for _ in range(steps):
                ix = torch.randint(len(splits["train"]), (32,), generator=sampler)
                batch = {key: value[ix] for key, value in batches["train"].items()}
                loss = (model(batch, memory=enabled) - batch["target"]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite probe loss")
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optim.step()
            with torch.no_grad():
                val = batches["validation"]
                mse = (
                    (model(val, memory=enabled) - val["target"]).square().mean().item()
                )
                # Keep current state/actions fixed, swap only historical evidence.
                permutation = torch.randperm(
                    len(splits["validation"]), generator=sampler
                )
                shuffled = {
                    **val,
                    **{
                        key: val[key][permutation]
                        for key in ("memory_events", "memory_valid")
                    },
                }
                shuffled_mse = (
                    (model(shuffled, memory=enabled) - val["target"])
                    .square()
                    .mean()
                    .item()
                )
            results.append(
                {
                    "seed": seed,
                    "memory": enabled,
                    "validation_mse": mse,
                    "shuffled_memory_mse": shuffled_mse,
                }
            )
    report = {
        "kind": "supervised_joint_change_probe_not_online_RL",
        "steps_per_model": steps,
        "episodes": ids,
        "samples": {key: len(rows) for key, rows in splits.items()},
        "sources": sources,
        "persistence_mse": batches["validation"]["target"].square().mean().item(),
        "results": results,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    """Run a CPU-only bounded probe and save source fingerprints with results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", default=32, type=int)
    parser.add_argument("--steps", default=300, type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(
        json.dumps(
            run_probe(
                args.dataset, args.output, episodes=args.episodes, steps=args.steps
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
