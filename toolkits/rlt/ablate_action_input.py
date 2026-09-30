# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Matched offline prediction tests using an existing frozen-feature cache."""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from toolkits.rlt.evaluate_latent_world import evaluate_checkpoint
from toolkits.rlt.train_latent_world import train


def run(
    config_path: Path,
    test_cache: Path,
    output: Path,
    *,
    device: str,
    steps: int = 1000,
    seeds: tuple[int, ...] = (2026, 2027, 2028),
) -> dict:
    """Compare direct, residual and action-free models on one unchanged split."""
    if not 1 <= steps <= 1000:
        raise ValueError("A diagnostic model is limited to 1..1000 updates")
    if not 1 <= len(seeds) <= 10:
        raise ValueError("Use 1..10 seeds")
    base = OmegaConf.load(config_path)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in seeds:
        for name, residual, actions in (
            ("direct", False, True),
            ("residual", True, True),
            ("action_free", True, False),
        ):
            cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
            cfg.seed, cfg.split_seed = seed, 2026
            cfg.max_steps = steps
            cfg.early_stopping_patience = 0
            cfg.batch_size, cfg.micro_batch_size = 32, 16
            cfg.world_model.predict_residual = residual
            cfg.world_model.condition_on_actions = actions
            cfg.wandb.mode = "disabled"
            cfg.output_dir = str(output / f"{name}_{seed}")
            path = output / f"{name}_{seed}.yaml"
            OmegaConf.save(cfg, path)
            best = train(str(path), device=device)
            metrics = evaluate_checkpoint(
                str(best), str(test_cache), device=device, independent_test=True
            )
            rows.append(
                {"mode": name, "seed": seed, "checkpoint": str(best), **metrics}
            )
            report = {
                "scope": "Repeated development-test prediction; not sealed test or control success",
                "steps_per_model": steps,
                "seeds": list(seeds),
                "results": rows,
            }
            (output / "results.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps({"completed": name, "seed": seed}), flush=True)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.config, args.test_cache, args.output, device="cpu", steps=args.steps)


if __name__ == "__main__":
    main()
