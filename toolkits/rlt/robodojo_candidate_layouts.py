# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Generate independent config-derived layouts for selected RoboDojo tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from toolkits.rlt.robodojo_layouts import verify
from toolkits.rlt.robodojo_task_plan import CANDIDATES

TASKS = frozenset(candidate.name for candidate in CANDIDATES)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(
    root: Path,
    output: Path,
    task: str,
    *,
    seed: int,
    train: int,
    development: int,
    reserved: int,
) -> dict:
    """Generate layouts before inventorying protected evaluation identities."""
    if task not in TASKS:
        raise ValueError(f"Task is outside the selected set: {task}")
    if min(seed, train, development, reserved) < 0 or train < 2 or development < 1:
        raise ValueError("Require a nonnegative seed, two train and one development")
    runtime_root = root / "XPolicyLab/policy/physicalRSI/runtime"
    sys.path.insert(0, str(runtime_root))
    try:
        from PhysicalRSI_baselines.robodojo.layout_sampling.catalog import (
            generate_train_layout,
        )
    finally:
        sys.path.pop(0)

    output.mkdir(parents=True, exist_ok=False)
    roles = ["train"] * train + ["development"] * development + ["reserved"] * reserved
    entries = []
    for index, role in enumerate(roles):
        layout = generate_train_layout(
            source_root=root,
            assets_root=root / "Assets",
            task=task,
            generation_seed=seed,
            layout_id=index,
        )
        provenance = layout.pop("_train_layout")
        path = output / f"{task}_{index}.json"
        path.write_text(json.dumps(layout, indent=2, allow_nan=False) + "\n")
        entries.append(
            {
                "id": index,
                "role": role,
                "path": str(path.resolve()),
                "sha256": _digest(path),
                "provenance": provenance,
            }
        )

    protected_root = root / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
    protected = [
        {"path": str(path.resolve()), "sha256": _digest(path)}
        for path in sorted(protected_root.glob(f"{task}_*.json"))
    ]
    if len(protected) != 50:
        raise ValueError(f"Require 50 protected layouts for {task}")
    manifest = {
        "schema": "robodojo-candidate-layouts-v1",
        "task": task,
        "seed": seed,
        "distribution": "public_config_scene_and_object_catalog_no_eval_ancestor",
        "protected": protected,
        "layouts": entries,
        "physics_validated": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return verify(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--seed", type=int, default=130013)
    parser.add_argument("--train", type=int, default=4)
    parser.add_argument("--development", type=int, default=2)
    parser.add_argument("--reserved", type=int, default=2)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        report = verify(args.verify)
    elif args.root and args.output and args.task:
        report = generate(
            args.root,
            args.output,
            args.task,
            seed=args.seed,
            train=args.train,
            development=args.development,
            reserved=args.reserved,
        )
    else:
        parser.error("Require --verify or --root, --output and --task")
    print(json.dumps({"task": report.get("task"), "layouts": len(report["layouts"])}))


if __name__ == "__main__":
    main()
