# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Perform read-only checks for the formal two-L40 RLT Stage2 job."""

from __future__ import annotations

import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def compose_config():
    """Compose and resolve the formal Stage2 configuration."""
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="maniskill_rlt_stage2_ac_mlp",
            overrides=["+experiment=rlt_baseline_2xl40"],
        )
    OmegaConf.resolve(cfg)
    return cfg


def check_config(cfg) -> None:
    """Reject accidental expert, GPU, path, or training-budget changes."""
    expected_placement = {"actor": "0-0", "env": "0-1", "rollout": "0-1"}
    if dict(cfg.cluster.component_placement) != expected_placement:
        raise RuntimeError(
            f"Unexpected Stage2 placement: {cfg.cluster.component_placement}"
        )
    if cfg.runner.max_steps != 5000 or cfg.runner.max_epochs != 5000:
        raise RuntimeError("Formal Stage2 must use the reviewed 5000-step budget")
    if cfg.runner.logger.wandb_entity != "c6522513-sustech":
        raise RuntimeError("Formal Stage2 is not assigned to the expected W&B entity")
    if cfg.algorithm.loss_type != "rlt_ac" or not cfg.algorithm.rlt_schedule.enable:
        raise RuntimeError("Formal Stage2 must use the RLT AC learner and schedule")
    if cfg.actor.model.model_type != "rlt_mlp_policy" or cfg.actor.model.model_path:
        raise RuntimeError("Stage2 MLP must start from scratch")
    if cfg.rollout.expert_model is not None:
        raise RuntimeError("Formal baseline must not load an automatic expert model")
    for mode in (cfg.env.train, cfg.env.eval):
        if mode.rlt_policy_switch.expert_takeover.enable:
            raise RuntimeError("Automatic expert takeover must remain disabled")
        if mode.init_params.sim_backend != "physx_cuda:0":
            raise RuntimeError(
                "Each environment worker must use local CUDA ordinal zero"
            )
        if mode.init_params.render_backend != "cuda:0":
            raise RuntimeError("Each renderer must use worker-local CUDA ordinal zero")
    if cfg.env.train.total_num_envs != 64 or cfg.env.eval.total_num_envs != 256:
        raise RuntimeError("Unexpected two-L40 environment batch")


def check_inputs(cfg) -> tuple[Path, Path]:
    """Validate the selected Stage1 export and normalization statistics."""
    actor_path = Path(cfg.rollout.rlt_feature_model.model_path)
    weights = actor_path / "model_state_dict/full_weights.pt"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(f"Missing Stage1 weights: {weights}")
    stats_path = Path(cfg.rollout.rlt_feature_model.openpi_data.norm_stats_path)
    try:
        stats = json.loads(stats_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid norm stats: {stats_path}: {exc}") from exc
    if not {"state", "actions"} <= stats.get("norm_stats", {}).keys():
        raise RuntimeError(f"Invalid norm stats: {stats_path}")
    for name in (
        "SAPIEN_VULKAN_LIBRARY_PATH",
        "VK_DRIVER_FILES",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
    ):
        path = Path(os.environ[name])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    return weights, stats_path


def main() -> None:
    """Run checks without initializing Ray, CUDA, Vulkan, or model weights."""
    cfg = compose_config()
    check_config(cfg)
    weights, stats_path = check_inputs(cfg)
    print("Placement: actor=GPU0; env/rollout=GPU0+GPU1")
    print(f"Stage1 weights: {weights} ({weights.stat().st_size:,} bytes)")
    print(f"Norm stats: {stats_path}")
    print("Training: 5000 RLT AC steps; 64 train envs; 256 fixed eval envs")
    print("Intervention: automatic OpenPI expert disabled; VR is not claimed here")
    print(f"Planned output: {os.environ['RLT_STAGE2_RUN_DIR']}")
    print("Preflight OK (read-only; Ray and training were not started).")


if __name__ == "__main__":
    main()
