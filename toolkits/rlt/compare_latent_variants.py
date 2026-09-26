# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded residual-prediction comparison on an existing frozen cache."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from toolkits.rlt.evaluate_latent_world import evaluate_checkpoint
from toolkits.rlt.train_latent_world import train


def main() -> None:
    """Keep data, seed and budget fixed; change only latent residual prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.original_config)
    if cfg.max_steps > 1000:
        raise ValueError("Comparison is limited to 1000 optimizer steps")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg.world_model.predict_residual = True
    cfg.output_dir = str(args.output / "stage1b")
    path = args.output / "train_config.yaml"
    OmegaConf.save(cfg, path, resolve=True)
    checkpoint = train(str(path), device=args.device)
    results = evaluate_checkpoint(str(checkpoint), cfg.cache_dir, device=args.device)
    (args.output / "diagnostics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False)
    )
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
