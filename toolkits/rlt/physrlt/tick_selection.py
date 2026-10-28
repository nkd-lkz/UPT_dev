# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Predeclare paired-intervention ticks from training geometry only."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from toolkits.rlt.pi05_baseline import write_json


def file_sha256(path: Path) -> str:
    """Hash the immutable transition dataset used for tick selection."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_motion(row: dict) -> float:
    """Return the largest 10-tick object displacement in one training row."""
    mask = row["horizon_mask"]
    geometry = row["geometry"]
    if len(mask) < 1 or not bool(mask[-1]):
        raise ValueError("Tick selection requires a valid longest horizon")
    positions = geometry[-1, 64:]
    if positions.numel() == 0 or positions.numel() % 3:
        raise ValueError("Tick selection requires 3-D object geometry")
    return float(positions.reshape(-1, 3).norm(dim=-1).max())


def select_ticks(
    dataset: dict,
    *,
    count: int = 4,
    minimum_cases: int = 2,
    minimum_spacing: int = 10,
) -> dict:
    """Select high-motion timestamps without reading non-training layouts."""
    if count < 1 or minimum_cases < 1 or minimum_spacing < 0:
        raise ValueError("Tick-selection limits must be positive")
    grouped: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for row in dataset["rows"]:
        if row["split"] != "train" or not bool(row["horizon_mask"][-1]):
            continue
        grouped[int(row["start_tick"])].append(
            (str(row["case_id"]), object_motion(row))
        )
    candidates = []
    for tick, values in grouped.items():
        cases = {case for case, _ in values}
        if len(cases) < minimum_cases:
            continue
        effects = [effect for _, effect in values]
        candidates.append(
            {
                "tick": tick,
                "cases": len(cases),
                "mean_max_object_motion_m": statistics.mean(effects),
                "median_max_object_motion_m": statistics.median(effects),
            }
        )
    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["median_max_object_motion_m"],
            -row["mean_max_object_motion_m"],
            -row["cases"],
            row["tick"],
        ),
    )
    selected = []
    for candidate in ranked:
        if all(
            abs(candidate["tick"] - previous["tick"]) >= minimum_spacing
            for previous in selected
        ):
            selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError("Insufficient covered, separated training ticks")
    return {
        "state": "complete",
        "selected_ticks": sorted(row["tick"] for row in selected),
        "selected": selected,
        "training_cases": len(
            {row["case_id"] for row in dataset["rows"] if row["split"] == "train"}
        ),
        "minimum_cases": minimum_cases,
        "minimum_spacing": minimum_spacing,
        "role": "train-only intervention scheduling; no evaluation outcomes read",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--minimum-cases", type=int, default=2)
    parser.add_argument("--minimum-spacing", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Tick-selection output must be a new file")
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
    result = select_ticks(
        dataset,
        count=args.count,
        minimum_cases=args.minimum_cases,
        minimum_spacing=args.minimum_spacing,
    )
    result["dataset"] = str(args.dataset.resolve())
    result["dataset_sha256"] = file_sha256(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
