# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Cache frozen Stage 1 features from the ManiSkill LeRobot v2.1 dataset.

This explicit command runs GPU inference, not training. No cache is created
merely by importing it. Use a completed, immutable Stage 1 checkpoint.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from rlinf.data.datasets.rlt_latent import (
    RLTLatentDataset,
    atomic_torch_save,
    feature_contract,
    file_sha256,
)


def image_array(value: dict, root: Path) -> np.ndarray:
    """Decode a v2.1 parquet image cell into the environment's uint8 HWC format."""
    source = (
        io.BytesIO(value["bytes"])
        if value.get("bytes") is not None
        else root / value["path"]
    )
    with Image.open(source) as image:
        return np.asarray(image.convert("RGB")).copy()


def cache(config_path: str, *, device: str, batch_size: int) -> None:
    """Publish immutable episode features, resuming only identical source data."""
    import pyarrow.parquet as pq

    from rlinf.models.embodiment.openpi import get_model

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cfg = OmegaConf.load(config_path)
    root, output = Path(cfg.dataset_root), Path(cfg.cache_dir)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError("This exporter supports ManiSkill LeRobot v2.1 only")
    if info["fps"] != cfg.control_freq or cfg.control_mode != "pd_joint_delta_pos":
        raise ValueError(
            "Cache control rate/units must match the demonstration dataset"
        )
    tasks_path = root / "meta/tasks.jsonl"
    tasks = {
        row["task_index"]: row["task"]
        for row in map(json.loads, tasks_path.read_text().splitlines())
    }
    episodes_path = root / "meta/episodes.jsonl"
    entries = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    contract = feature_contract(
        cfg.feature_model, control_mode=cfg.control_mode, control_freq=cfg.control_freq
    )
    source = {
        "info": file_sha256(info_path),
        "episodes": file_sha256(episodes_path),
        "tasks": file_sha256(tasks_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.pt"
    manifest = {
        "format_version": 1,
        "complete": False,
        "feature_contract": contract,
        "source": source,
        "episodes": [],
        "batch_size": batch_size,
    }
    if manifest_path.exists():
        previous = torch.load(manifest_path, map_location="cpu", weights_only=True)
        for key in ("format_version", "feature_contract", "source", "batch_size"):
            if previous.get(key) != manifest[key]:
                raise ValueError(
                    f"Existing cache differs in {key}; use a new cache_dir"
                )
        manifest = previous
    known = {row["id"]: row for row in manifest["episodes"]}
    torch.manual_seed(int(cfg.seed))
    model = None  # A completed cache can be verified without allocating the VLA.
    for entry in entries:
        episode_id = int(entry["episode_index"])
        path = root / info["data_path"].format(
            episode_chunk=episode_id // info["chunks_size"],
            episode_index=episode_id,
        )
        source_hash = file_sha256(path)
        key = str(episode_id)
        if key in known:
            prior = known[key]
            if (
                prior["source_sha256"] != source_hash
                or file_sha256(output / prior["file"]) != prior["sha256"]
            ):
                raise ValueError(
                    f"Episode {key} changed or its cache is corrupt; use a new cache_dir"
                )
            continue
        if model is None:
            model = get_model(cfg.feature_model).to(device).eval().requires_grad_(False)
        columns = pq.read_table(path).to_pydict()
        frames = torch.as_tensor(columns["frame_index"]).reshape(-1)
        if len(frames) != int(entry["length"]):
            raise ValueError(f"Episode length differs from metadata in {path}")
        episode_indices = torch.as_tensor(columns["episode_index"]).reshape(-1)
        task_indices = torch.as_tensor(columns["task_index"]).reshape(-1)
        timestamps = torch.as_tensor(columns["timestamp"]).reshape(-1)
        if (
            not torch.all(episode_indices == episode_id)
            or task_indices.unique().numel() != 1
        ):
            raise ValueError(f"Mixed episode or task IDs in {path}")
        if not torch.allclose(
            timestamps[1:] - timestamps[:-1],
            torch.full_like(timestamps[1:], 1 / cfg.control_freq),
            atol=1e-4,
        ):
            raise ValueError(f"Nonconsecutive control timestamps in {path}")
        prompt = tasks[int(task_indices[0])]
        z_features, proprio_features = [], []
        for start in range(0, len(frames), batch_size):
            end = min(start + batch_size, len(frames))
            env_obs = {
                "main_images": np.stack(
                    [image_array(x, root) for x in columns["image"][start:end]]
                ),
                "wrist_images": np.stack(
                    [image_array(x, root) for x in columns["wrist_image"][start:end]]
                ),
                "states": torch.as_tensor(np.asarray(columns["state"][start:end])),
                "task_descriptions": [prompt] * (end - start),
            }
            with torch.inference_mode():
                obs = model.extract_rlt_obs(env_obs, include_reference=False)
            z_features.append(obs["z_rl"].cpu())
            proprio_features.append(obs["proprio"].cpu())
        episode = {
            "z_rl": torch.cat(z_features),
            "proprio": torch.cat(proprio_features),
            "actions": torch.as_tensor(
                np.asarray(columns["actions"]), dtype=torch.float32
            ),
            "frame_index": frames,
            "task_index": int(task_indices[0]),
        }
        RLTLatentDataset.validate_episode(episode)
        name = f"episode_{episode_id:06d}.pt"
        atomic_torch_save(episode, output / name)
        manifest["episodes"].append(
            {
                "id": key,
                "file": name,
                "sha256": file_sha256(output / name),
                "source_sha256": source_hash,
            }
        )
        manifest["complete"] = False
        atomic_torch_save(manifest, manifest_path)
        print(f"Cached episode {episode_id}: {len(frames)} frames", flush=True)
    manifest["complete"] = True
    atomic_torch_save(manifest, manifest_path)
    print(f"Complete cache: {output}; {len(manifest['episodes'])} episodes", flush=True)


def main() -> None:
    """Parse an explicit export request; configuration paths are never guessed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    cache(args.config, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
