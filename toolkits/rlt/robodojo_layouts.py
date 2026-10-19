# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Declare reproducible, geometrically separated stack-bowls development layouts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    """Hash a file without interpreting its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positions(layout: dict) -> np.ndarray:
    """Read three bowl positions; object labels are irrelevant to stacking."""
    values = np.asarray(
        [obj["default_pos"] for obj in layout["Rigid"]["bowl"]], dtype=np.float64
    )
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise ValueError("Require exactly three finite bowl positions")
    return values


def separation(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum maximum bowl displacement over label permutations, in metres."""
    return min(
        float(np.linalg.norm(a - b[list(order)], axis=1).max())
        for order in itertools.permutations(range(3))
    )


def prepare(protected: Path, output: Path, seed: int = 130013) -> dict:
    """Sample 16 train, four development and eight reserved layouts, without scores.

    Static fixtures and bowl asset 10 come from the published task-0 schema.
    All bowl XY positions are resampled in a declared conservative workspace.
    This is a same-asset engineering distribution, not the native benchmark.
    """
    files = sorted(protected.glob("stack_bowls_*.json"))
    if len(files) != 25:
        raise ValueError("Require the complete protected 25-layout inventory")
    template = json.loads((protected / "stack_bowls_0.json").read_text())
    if {obj["category_idx"] for obj in template["Rigid"]["bowl"]} != {10}:
        raise ValueError("Require the declared bowl-10 asset template")
    accepted = [positions(json.loads(path.read_text())) for path in files]
    rng = np.random.default_rng(seed)
    manifest = {
        "schema": "robodojo-stack-bowls-layouts-v1",
        "seed": seed,
        "distribution": "same_bowl10_static_fixtures_resampled_xy_not_native_benchmark",
        "xy_bounds": [[-0.35, 0.35], [-0.20, 0.02]],
        "minimum_bowl_separation_m": 0.16,
        "minimum_layout_separation_m": 0.04,
        "protected": [{"path": str(p.resolve()), "sha256": digest(p)} for p in files],
        "layouts": [],
        "physics_validated": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    for index, role in enumerate(
        ["train"] * 16 + ["development"] * 4 + ["reserved"] * 8
    ):
        for _ in range(10000):
            xyz = np.column_stack(
                (
                    rng.uniform(-0.35, 0.35, 3),
                    rng.uniform(-0.20, 0.02, 3),
                    np.full(3, 0.79),
                )
            )
            if (
                min(
                    np.linalg.norm(xyz[i] - xyz[j])
                    for i, j in itertools.combinations(range(3), 2)
                )
                < 0.16
            ):
                continue
            if min(separation(xyz, old) for old in accepted) < 0.04:
                continue
            break
        else:
            raise RuntimeError(
                "Layout rejection budget exhausted; preserve partial output"
            )
        layout = copy.deepcopy(template)
        for obj, pos in zip(layout["Rigid"]["bowl"], xyz, strict=True):
            obj["default_pos"] = pos.tolist()
        path = output / f"stack_bowls_{index}.json"
        path.write_text(json.dumps(layout, indent=2, allow_nan=False) + "\n")
        manifest["layouts"].append(
            {
                "id": index,
                "role": role,
                "path": str(path.resolve()),
                "sha256": digest(path),
            }
        )
        accepted.append(xyz)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(path: Path) -> dict:
    """Recheck bytes and geometric separation before simulator collection."""
    data = json.loads(path.read_text())
    if data["schema"] == "robodojo-candidate-layouts-v1":
        roles = {"train", "development", "reserved"}
        seen_ids = set()
        protected_hashes = set()
        for entry in data.get("protected", []):
            file = Path(entry["path"])
            if digest(file) != entry["sha256"]:
                raise ValueError("Declared protected layout bytes changed")
            protected_hashes.add(entry["sha256"])
        if not protected_hashes:
            raise ValueError("Require protected evaluation identities")
        for entry in data.get("layouts", []):
            file = Path(entry["path"])
            if (
                type(entry.get("id")) is not int
                or entry["id"] in seen_ids
                or entry.get("role") not in roles
                or digest(file) != entry.get("sha256")
                or entry.get("sha256") in protected_hashes
            ):
                raise ValueError("Invalid or overlapping candidate layout entry")
            layout = json.loads(file.read_text())
            provenance = entry.get("provenance")
            if (
                not isinstance(provenance, dict)
                or provenance.get("task") != data.get("task")
                or provenance.get("layout_domain") != "train"
                or provenance.get("layout_id") != entry["id"]
                or "_train_layout" in layout
            ):
                raise ValueError("Candidate layout provenance differs from manifest")
            seen_ids.add(entry["id"])
        if not seen_ids or seen_ids != set(range(len(seen_ids))):
            raise ValueError("Candidate layout IDs must be contiguous from zero")
        return data
    if data["schema"] != "robodojo-stack-bowls-layouts-v1":
        raise ValueError("Unknown layout protocol")
    prior = []
    for entry in data["protected"] + data["layouts"]:
        file = Path(entry["path"])
        if digest(file) != entry["sha256"]:
            raise ValueError("Declared layout bytes changed")
        xyz = positions(json.loads(file.read_text()))
        if "role" in entry and any(separation(xyz, old) < 0.04 for old in prior):
            raise ValueError(
                "Training/development/reserved or protected layout overlap"
            )
        prior.append(xyz)
    return data


def main() -> None:
    """Create a fresh split or verify an existing split without starting Isaac."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    elif args.protected and args.output:
        prepare(args.protected, args.output)
    else:
        parser.error("Require --verify or --protected and --output")


if __name__ == "__main__":
    main()
