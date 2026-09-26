# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Bounded real-data Stage 1B experiment using an immutable Stage 1 checkpoint."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

from toolkits.rlt.cache_latents import cache
from toolkits.rlt.evaluate_latent_world import evaluate_checkpoint
from toolkits.rlt.train_latent_world import train


def main() -> None:
    """Export 12 full episodes, train 300 steps, and compare held-out predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    expected = (
        subprocess.check_output(
            ["nvidia-smi", "-i", "2", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
        )
        .strip()
        .lower()
        .removeprefix("gpu-")
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("2", "GPU-" + expected):
        raise RuntimeError("Pilot requires explicit physical GPU 2 isolation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Pilot requires exactly one visible GPU")
    if (
        str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
        != expected
    ):
        raise RuntimeError("CUDA device does not match physical GPU 2")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    cache_cfg = OmegaConf.load(
        root / "experiments/maniskill_rlt/config/cache_latents.yaml"
    )
    cache_cfg.cache_dir = str(args.output / "cache")
    cache_cfg.episode_ids = list(range(12))
    cache_path = args.output / "cache_config.yaml"
    OmegaConf.save(cache_cfg, cache_path, resolve=True)
    cache(str(cache_path), device="cuda:0", batch_size=2)
    torch.cuda.empty_cache()
    if args.cache_only:
        return
    cfg = OmegaConf.load(root / "experiments/maniskill_rlt/config/stage1b.yaml")
    cfg.cache_dir = cache_cfg.cache_dir
    cfg.output_dir = str(args.output / "stage1b")
    cfg.validation_fraction = 0.25
    cfg.batch_size = 32
    cfg.micro_batch_size = 16
    cfg.max_steps = 300
    cfg.warmup_steps = 20
    cfg.validate_every = 50
    cfg.log_every = 25
    cfg.wandb.mode = "disabled"
    config_path = args.output / "train_config.yaml"
    OmegaConf.save(cfg, config_path, resolve=True)
    checkpoint = train(str(config_path), device="cuda:0")
    result = evaluate_checkpoint(str(checkpoint), cache_cfg.cache_dir, device="cuda:0")
    (args.output / "diagnostics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
