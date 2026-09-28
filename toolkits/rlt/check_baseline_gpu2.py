# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Read-only release checks for the Stage1 eval and Stage2 baseline jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("stage1-eval", "stage2-train"))
    return parser.parse_args()


def _compose(job: str):
    if job == "stage1-eval":
        config_dir = ROOT / "evaluations/maniskill"
        config_name = "maniskill_rlt_stage1_eval20"
        overrides = []
    else:
        config_dir = ROOT / "examples/embodiment/config"
        config_name = "maniskill_rlt_stage2_ac_mlp"
        overrides = ["+experiment=rlt_baseline_gpu2"]
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _check_inputs(cfg) -> tuple[Path, Path]:
    actor_path = (
        cfg.rollout.model.model_path
        if cfg.runner.only_eval
        else cfg.rollout.rlt_feature_model.model_path
    )
    weights = Path(actor_path) / "model_state_dict/full_weights.pt"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(f"Missing weights: {weights}")

    model_cfg = (
        cfg.rollout.model if cfg.runner.only_eval else cfg.rollout.rlt_feature_model
    )
    stats_path = Path(model_cfg.openpi_data.norm_stats_path)
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


def _check_common(cfg) -> None:
    expected = {"env": "2-2", "rollout": "2-2"}
    placement = dict(cfg.cluster.component_placement)
    for component, hardware in expected.items():
        if placement.get(component) != hardware:
            raise RuntimeError(f"Unsafe {component} placement: {placement}")
    if cfg.env.eval.init_params.sim_backend != "physx_cuda:0":
        raise RuntimeError("Evaluation must use worker-local CUDA ordinal 0")
    if cfg.env.eval.init_params.render_backend != os.environ["RLT_GPU2_RENDER_DEVICE"]:
        raise RuntimeError("Evaluation render device does not match GPU 2")

    port = int(os.environ["RLT_RAY_PORT"])
    occupied_by_stage1 = {6379, 6385, 6386, 6387}
    if not 1024 <= port <= 65533 or set(range(port, port + 3)) & occupied_by_stage1:
        raise RuntimeError(f"Unsafe isolated Ray port range starting at {port}")


def _check_stage1_eval(cfg) -> None:
    if dict(cfg.cluster.component_placement) != {"env": "2-2", "rollout": "2-2"}:
        raise RuntimeError("Stage1 eval may place only env and rollout on GPU 2")
    if not cfg.runner.only_eval:
        raise RuntimeError("Stage1 release config must be eval-only")
    if cfg.env.eval.total_num_envs != 20 or cfg.env.eval.rollout_epoch != 1:
        raise RuntimeError("Stage1 release evaluation must run exactly 20 episodes")
    if cfg.env.eval.auto_reset or not cfg.env.eval.use_fixed_reset_state_ids:
        raise RuntimeError("Stage1 eval requires fixed, non-auto-reset episodes")
    if cfg.env.eval.rlt_policy_switch.enable:
        raise RuntimeError("Stage1 eval must use the OpenPI policy directly")
    if cfg.rollout.model.model_type != "openpi" or not cfg.rollout.model.openpi.use_rlt:
        raise RuntimeError("Stage1 eval model shape does not match the RLT checkpoint")
    if not cfg.env.eval.video_cfg.save_video or cfg.env.eval.video_cfg.record_every != 1:
        raise RuntimeError("Stage1 release evaluation must record the full rollout")


def _check_stage2_train(cfg) -> None:
    if dict(cfg.cluster.component_placement) != {
        "actor": "2-2",
        "env": "2-2",
        "rollout": "2-2",
    }:
        raise RuntimeError("Stage2 baseline must be isolated on GPU 2")
    if (
        cfg.runner.only_eval
        or cfg.runner.max_steps <= 0
        or cfg.runner.max_epochs < cfg.runner.max_steps
    ):
        raise RuntimeError("Stage2 baseline must have a bounded training budget")
    if cfg.actor.model.model_type != "rlt_mlp_policy" or cfg.actor.model.model_path:
        raise RuntimeError("Stage2 baseline MLP must start from scratch")
    if cfg.algorithm.loss_type != "rlt_ac" or not cfg.algorithm.rlt_schedule.enable:
        raise RuntimeError("Stage2 baseline must use the RLT AC schedule")
    if cfg.rollout.expert_model is not None:
        raise RuntimeError("Stage2 baseline must not load an expert takeover model")
    for mode in (cfg.env.train, cfg.env.eval):
        if mode.rlt_policy_switch.expert_takeover.enable:
            raise RuntimeError("Stage2 baseline expert takeover must remain disabled")
    if cfg.env.train.total_num_envs != 16 or cfg.env.eval.total_num_envs != 20:
        raise RuntimeError("Unexpected single-GPU Stage2 environment batch")
    if OmegaConf.select(cfg, "actor.model.interaction_memory.enabled", default=False):
        raise RuntimeError("Exploration interaction memory is not part of the baseline")


def main() -> None:
    args = _parse_args()
    cfg = _compose(args.job)
    _check_common(cfg)
    if args.job == "stage1-eval":
        _check_stage1_eval(cfg)
    else:
        _check_stage2_train(cfg)
    weights, stats_path = _check_inputs(cfg)
    print(f"Job: {args.job}; physical GPU: 2; worker CUDA ordinal: 0")
    print(f"Stage1 weights: {weights} ({weights.stat().st_size:,} bytes)")
    print(f"Norm stats: {stats_path}")
    if args.job == "stage1-eval":
        print("Evaluation: exactly 20 fixed-reset episodes; full tiled MP4 enabled")
    else:
        print("Training: RLT AC baseline; 16 train envs; 20 eval envs; no expert")
    print(f"Planned output: {os.environ['RLT_JOB_RUN_DIR']}")
    print("Preflight OK (read-only; Ray and training were not started).")


if __name__ == "__main__":
    main()
