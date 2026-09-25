# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate RLT research configuration/artifacts without Ray, simulation or training."""

import argparse
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.rlt.latent_world import validate_latent_world_rollout


def compose_stage2(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    """Resolve an embodied config without constructing a cluster or workers."""
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/embodiment/config")
    ):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    OmegaConf.resolve(cfg)
    return cfg


def main() -> None:
    """Check shape/configuration only, or additionally hash and verify real artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="maniskill_rlt_stage2_latent_world")
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Do not open weights; this is not a runtime readiness check",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = compose_stage2(args.config_name, args.overrides)
    if not args.config_only:
        validate_latent_world_rollout(cfg)
        if not cfg.actor.model.latent_world.enabled:
            from rlinf.data.datasets.rlt_latent import feature_contract

            feature_contract(
                cfg.rollout.rlt_feature_model,
                control_mode=cfg.env.train.init_params.control_mode,
                control_freq=cfg.env.train.init_params.sim_config.control_freq,
            )
    print(
        json.dumps(
            {
                "config": args.config_name,
                "check": "config-only" if args.config_only else "feature-contract",
                "placement": OmegaConf.to_container(cfg.cluster.component_placement),
                "actor_batch": cfg.actor.global_batch_size,
                "world_enabled": cfg.actor.model.latent_world.enabled,
                "world_weight": cfg.algorithm.latent_world_weight,
                "feature_checkpoint": cfg.rollout.rlt_feature_model.model_path,
                "world_checkpoint": cfg.actor.model.latent_world.checkpoint,
                "new_training_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
