# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Export genuine pre-action VLA references and RL features for actor BC."""

import argparse
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    """Export a bounded demonstration subset only on an idle, explicit GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    used = int(
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    )
    if used > 512:
        parser.error(f"GPU {args.gpu} is busy ({used} MiB); no model was loaded")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    from rlinf.models.embodiment.openpi import get_model

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(
        root / "examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml"
    )
    cfg.rollout.rlt_feature_model.model_path = str(args.stage1)
    cfg.rollout.rlt_feature_model.openpi_data.repo_id = str(args.dataset)
    cfg.rollout.rlt_feature_model.openpi_data.norm_stats_path = str(
        args.dataset / "norm_stats.json"
    )
    cfg.rollout.rlt_feature_model.openpi.torch_compile = False
    model_cfg = OmegaConf.to_container(cfg.rollout.rlt_feature_model, resolve=True)
    info = json.loads((args.dataset / "meta/info.json").read_text())
    if info["fps"] != 10 or info["codebase_version"] != "v2.1":
        parser.error("Require the reviewed 10 Hz LeRobot v2.1 joint-delta dataset")
    entries = [
        json.loads(line)
        for line in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()
    ]
    tasks = {
        r["task_index"]: r["task"]
        for r in map(
            json.loads, (args.dataset / "meta/tasks.jsonl").read_text().splitlines()
        )
    }
    if len(set(args.episodes)) != len(args.episodes) or not set(args.episodes) <= {
        r["episode_index"] for r in entries
    }:
        parser.error("Unknown or duplicate episode IDs")
    args.output.mkdir(parents=True, exist_ok=False)

    def sha(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        return digest.hexdigest()

    manifest = {
        "complete": False,
        "format_version": 1,
        "episodes": [],
        "feature_contract": {
            "weights_sha256": sha(args.stage1 / "model_state_dict/full_weights.pt"),
            "norm_stats_sha256": sha(args.dataset / "norm_stats.json"),
            "model_config": model_cfg,
            "control_mode": "pd_joint_delta_pos",
            "control_freq": 10,
            "action_space": "environment_pd_joint_delta_pos",
            "proprio_space": "openpi_normalized",
            "reference_source": "frozen_vla_pre_action",
            "inference_seed": 1234,
        },
    }
    torch.save(manifest, args.output / "manifest.pt")
    torch.manual_seed(1234)
    model = (
        get_model(OmegaConf.create(model_cfg)).to("cuda:0").eval().requires_grad_(False)
    )

    def decode(value):
        source = (
            io.BytesIO(value["bytes"])
            if value.get("bytes") is not None
            else args.dataset / value["path"]
        )
        with Image.open(source) as img:
            return np.asarray(img.convert("RGB")).copy()

    for entry in entries:
        idx = entry["episode_index"]
        if idx not in args.episodes:
            continue
        source = args.dataset / info["data_path"].format(
            episode_chunk=idx // info["chunks_size"], episode_index=idx
        )
        columns = pq.read_table(source).to_pydict()
        tensors = {k: [] for k in ("z_rl", "proprio", "ref_chunk")}
        for start in range(0, len(columns["frame_index"]), 2):
            end = start + 2
            obs = {
                "main_images": np.stack(
                    [decode(v) for v in columns["image"][start:end]]
                ),
                "wrist_images": np.stack(
                    [decode(v) for v in columns["wrist_image"][start:end]]
                ),
                "states": torch.tensor(columns["state"][start:end]),
                "task_descriptions": [
                    tasks[t] for t in columns["task_index"][start:end]
                ],
            }
            with torch.inference_mode():
                features = model.extract_rlt_obs(obs)
            for key in tensors:
                tensors[key].append(features[key].cpu())
        data = {key: torch.cat(value) for key, value in tensors.items()}
        data.update(
            actions=torch.tensor(columns["actions"], dtype=torch.float32),
            frame_index=torch.tensor(columns["frame_index"]),
        )
        name = f"episode_{idx:06d}.pt"
        torch.save(data, args.output / name)
        manifest["episodes"].append(
            {
                "id": str(idx),
                "file": name,
                "sha256": sha(args.output / name),
                "source_sha256": sha(source),
            }
        )
        torch.save(manifest, args.output / "manifest.pt")
    manifest["complete"] = True
    torch.save(manifest, args.output / "manifest.pt")


if __name__ == "__main__":
    main()
