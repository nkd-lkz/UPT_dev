# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Merge an indexed Hugging Face safetensors checkpoint for RLinf OpenPI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_shards(index_path: Path, output: Path) -> dict[str, int]:
    """Merge every shard named by an index into one atomically published file."""
    from safetensors.torch import load_file, save_file

    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Safetensors index has no nonempty weight_map")
    shards = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or Path(name).name != name for name in shards):
        raise ValueError("Shard names must be local basenames")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    tensors = {}
    for name in shards:
        shard_path = index_path.parent / name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
        for key, value in load_file(str(shard_path), device="cpu").items():
            if key in tensors:
                raise ValueError(f"Duplicate tensor across shards: {key}")
            tensors[key] = value
    if set(tensors) != set(weight_map):
        missing = sorted(set(weight_map) - set(tensors))
        unexpected = sorted(set(tensors) - set(weight_map))
        raise ValueError(
            f"Shard/index tensor mismatch; missing={missing[:4]} "
            f"unexpected={unexpected[:4]}"
        )

    temporary = output.with_name(f".{output.name}.incomplete")
    if temporary.exists():
        temporary.unlink()
    save_file(tensors, str(temporary), metadata={"format": "pt"})
    temporary.replace(output)
    return {
        "shards": len(shards),
        "tensors": len(tensors),
        "bytes": output.stat().st_size,
    }


def main() -> None:
    """Merge a downloaded checkpoint selected on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    result = merge_shards(
        args.checkpoint / "model.safetensors.index.json",
        args.checkpoint / "model.safetensors",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
