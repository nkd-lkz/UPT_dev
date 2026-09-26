# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check actual research-module updates in saved full-model checkpoints."""

import argparse
import json
from pathlib import Path

import torch


def compare(before: Path, after: Path, prefix: str) -> dict:
    """Reject missing, incompatible or nonfinite weights and report real changes."""
    old = torch.load(before, map_location="cpu", weights_only=True)
    new = torch.load(after, map_location="cpu", weights_only=True)
    keys = [key for key in old if key.startswith(prefix)]
    if not keys or set(keys) != {key for key in new if key.startswith(prefix)}:
        raise ValueError("No matching adapter or different checkpoint schemas")
    changes = {}
    for key in keys:
        if old[key].shape != new[key].shape:
            raise ValueError(f"Incompatible shape: {key}")
        if not torch.isfinite(old[key]).all() or not torch.isfinite(new[key]).all():
            raise ValueError(f"Nonfinite weights: {key}")
        changes[key] = (old[key].float() - new[key].float()).abs().max().item()
    return {
        "prefix": prefix,
        "tensors": len(keys),
        "changed_tensors": sum(value > 0 for value in changes.values()),
        "max_abs_change": max(changes.values()),
        "per_tensor": changes,
    }


def main() -> None:
    """Audit trusted checkpoints without CUDA or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--require-change", action="store_true")
    args = parser.parse_args()
    result = compare(args.before, args.after, args.prefix)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.require_change and not result["changed_tensors"]:
        raise RuntimeError("Adapter did not update despite a successful smoke exit")


if __name__ == "__main__":
    main()
