# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Explicit control-duration and provenance checks for latent-world RLT replay."""

from omegaconf import DictConfig


def replay_world_batch(batch: dict, model_cfg: DictConfig) -> dict:
    """Pair executed chunks with their successors, excluding any terminal row.

    ManiSkill returns one observation after executing an entire chunk. Reward
    slots specify its duration in control ticks. Terminal rows in RLT replay
    substitute curr_obs for next_obs, so they must never teach dynamics.
    """
    if not model_cfg.get("latent_world", {}).get("enabled", False):
        raise ValueError(
            "latent_world_weight requires actor.model.latent_world.enabled"
        )
    n = batch["actions"].shape[0]
    horizon = int(model_cfg.num_action_chunks)
    action_dim = int(model_cfg.action_dim)
    if batch["rewards"].reshape(n, -1).shape[1] != horizon:
        raise ValueError(
            "Replay duration must match the full action chunk; per-tick replay is unsupported"
        )
    actions = batch["actions"].reshape(n, -1, action_dim)
    if actions.shape[1] != horizon:
        raise ValueError("Replay actions must be the full executed chunk")
    valid = ~batch["dones"].reshape(n, -1).bool().any(-1, keepdim=True)
    return {
        "z_rl": batch["curr_obs"]["z_rl"].reshape(n, -1),
        "proprio": batch["curr_obs"]["proprio"].reshape(n, -1),
        "actions": actions,
        "horizons": (horizon,),
        "valid": valid,
        "future_z": batch["next_obs"]["z_rl"].reshape(n, 1, -1),
        "future_proprio": batch["next_obs"]["proprio"].reshape(n, 1, -1),
    }


def validate_latent_world_rollout(cfg: DictConfig) -> None:
    """Fail before rollout if frozen feature coordinates or algorithm differ."""
    from omegaconf import OmegaConf

    from rlinf.data.datasets.rlt_latent import feature_contract
    from rlinf.models.embodiment.modules.rlt_latent_world import RLTLatentWorld

    world_cfg = OmegaConf.select(cfg, "actor.model.latent_world", default={})
    if not world_cfg.get("enabled", False):
        if cfg.algorithm.get("latent_world_weight", 0) != 0:
            raise ValueError(
                "Disable latent_world_weight when disabling the world model"
            )
        return
    if cfg.algorithm.loss_type != "rlt_ac" or cfg.env.train.env_type != "maniskill_rlt":
        raise ValueError("Latent-world replay currently supports ManiSkill RLT AC only")
    if cfg.actor.model.model_type != "rlt_mlp_policy":
        raise ValueError(
            "The latent-world adapter is implemented for rlt_mlp_policy only"
        )
    if cfg.algorithm.get("latent_world_weight", 0) < 0:
        raise ValueError("latent_world_weight must be nonnegative")
    if cfg.actor.model.precision != "fp32" or not cfg.actor.fsdp_config.use_orig_params:
        raise ValueError(
            "Latent-world RLT requires fp32 heads and FSDP use_orig_params=True"
        )
    if cfg.algorithm.get("target_update_type", "all") != "all":
        raise ValueError(
            "Target updates must include the learned world encoder, not just q_head"
        )
    entropy = cfg.algorithm.entropy_tuning
    if entropy.alpha_type != "fixed_alpha" or float(entropy.initial_alpha) != 0:
        raise ValueError("Latent-world RLT requires fixed zero entropy")
    actor_cfg = OmegaConf.to_container(world_cfg, resolve=True)
    rollout_cfg = OmegaConf.to_container(
        cfg.rollout.model.get("latent_world", {}), resolve=True
    )
    if actor_cfg != rollout_cfg:
        raise ValueError(
            "Actor and rollout latent_world configurations must be identical"
        )
    env = cfg.env.train
    if not cfg.rollout.collect_transitions:
        raise ValueError("Latent-world replay requires collect_transitions")
    contract = feature_contract(
        cfg.rollout.rlt_feature_model,
        control_mode=env.init_params.control_mode,
        control_freq=env.init_params.sim_config.control_freq,
    )
    world = RLTLatentWorld.from_checkpoint(
        world_cfg.checkpoint, expected_contract=contract
    )
    if world.config.chunk_len != cfg.actor.model.num_action_chunks:
        raise ValueError("Cache/sidecar and Stage 2 chunk lengths differ")
    for mode in ("train", "eval"):
        settings = cfg.env[mode].init_params
        if (
            settings.control_mode != contract["control_mode"]
            or settings.sim_config.control_freq != contract["control_freq"]
        ):
            raise ValueError(
                f"{mode} controller/rate differs from the latent feature contract"
            )
