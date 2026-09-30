# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Read-only preflight. No Ray connection, model loading, or CUDA allocation."""

import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.atomic_decision import validate_atomic_config
from rlinf.utils.logging import get_logger


def main() -> None:
    """Validate config, fixed GPU placement, assets and isolated Ray port."""
    with initialize_config_dir(
        config_dir=os.environ["EMBODIED_PATH"] + "/config", version_base="1.1"
    ):
        cfg = compose(config_name="maniskill_rlt_stage2_atomic_gpu2")
    OmegaConf.resolve(cfg)
    validate_atomic_config(cfg)
    if dict(cfg.cluster.component_placement) != {
        "actor": "2-2",
        "env": "2-2",
        "rollout": "2-2",
    }:
        raise ValueError("Every component must be isolated on physical GPU 2.")
    weights = (
        Path(cfg.rollout.rlt_feature_model.model_path)
        / "model_state_dict/full_weights.pt"
    )
    if not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(f"Missing Stage 1 weights: {weights}")
    stats_path = Path(cfg.rollout.rlt_feature_model.openpi_data.norm_stats_path)
    if (
        not {"state", "actions"}
        <= json.loads(stats_path.read_text())["norm_stats"].keys()
    ):
        raise ValueError(f"Invalid normalization stats: {stats_path}")
    for name in (
        "SAPIEN_VULKAN_LIBRARY_PATH",
        "VK_DRIVER_FILES",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
    ):
        if not Path(os.environ[name]).is_file():
            raise FileNotFoundError(f"Missing {name}")
    if cfg.rollout.expert_model is not None:
        raise ValueError("The initial atomic smoke must not load a second VLA expert.")
    if cfg.actor.global_batch_size % cfg.actor.micro_batch_size:
        raise ValueError("Global batch must be divisible by micro batch.")
    steps = int(os.environ.get("RLT_ATOMIC_STEPS", "2"))
    if not 1 <= steps <= 20:
        raise ValueError("This launcher is bounded to 1..20 outer steps.")
    port = int(os.environ["RLT_SMOKE_RAY_PORT"])
    if not 6500 <= port <= 6590:
        raise ValueError("Use a separate Ray port in 6500..6590 for this experiment.")
    get_logger().info(
        "Atomic config/path preflight OK: GPU 2 only, 2 train / 1 eval env, "
        "500 control steps per episode, %s outer steps, Stage 1=%s. "
        "No GPU or distributed execution was tested.",
        steps,
        weights,
    )


if __name__ == "__main__":
    main()
