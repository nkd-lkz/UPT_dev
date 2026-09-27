# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compare direct/residual future prediction with fixed data splits and seeds."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .evaluate_latent_world import evaluate_checkpoint
from .train_latent_world import train


def run(
    config_path: Path,
    test_cache: Path,
    output: Path,
    *,
    device: str,
    steps: int = 300,
    seeds: tuple[int, ...] = (2026, 2027, 2028),
) -> dict:
    """Keep one split and one budget; never select models using test errors."""
    if (
        not 1 <= steps <= 1000
        or not 1 <= len(seeds) <= 5
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Use 1..1000 steps and 1..5 distinct seeds")
    config = OmegaConf.load(config_path)
    split_seed = int(config.get("split_seed", config.seed))
    if config.batch_size > 64:
        raise ValueError("Bounded repeat pilot requires batch_size <= 64")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in seeds:
        for residual in (False, True):
            cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            cfg.seed, cfg.split_seed = seed, split_seed
            cfg.max_steps = steps
            cfg.world_model.predict_residual = residual
            cfg.wandb.mode = "disabled"
            cfg.output_dir = str(output / f"seed{seed}_residual{int(residual)}")
            path = output / f"seed{seed}_residual{int(residual)}.yaml"
            OmegaConf.save(cfg, path)
            best = train(str(path), device=device)
            report = evaluate_checkpoint(
                str(best), str(test_cache), device=device, independent_test=True
            )
            row = {
                "seed": seed,
                "residual": residual,
                "checkpoint": str(best),
                **report,
            }
            rows.append(row)
            (output / "results.json").write_text(
                json.dumps(
                    {
                        "split_seed": split_seed,
                        "steps_per_model": steps,
                        "results": rows,
                    },
                    indent=2,
                    allow_nan=False,
                )
            )
            print(
                json.dumps({"completed_seed": seed, "residual": residual}), flush=True
            )
    return {"split_seed": split_seed, "steps_per_model": steps, "results": rows}


def main() -> None:
    """Run the bounded comparison on CPU or explicitly isolated physical GPU 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.device == "cuda:0":
        uuid, memory = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    "2",
                    "--query-gpu=uuid,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .split(",")
        )
        uuid = uuid.strip()
        if int(memory) > 1024 or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
            raise RuntimeError(
                "Select idle physical GPU 2 by exact CUDA_VISIBLE_DEVICES UUID"
            )
        if torch.cuda.device_count() != 1 or str(
            torch.cuda.get_device_properties(0).uuid
        ).removeprefix("GPU-") != uuid.removeprefix("GPU-"):
            raise RuntimeError("CUDA UUID isolation failed")
    torch.set_num_threads(2)
    run(args.config, args.test_cache, args.output, device=args.device, steps=args.steps)


if __name__ == "__main__":
    main()
