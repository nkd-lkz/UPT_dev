# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate one-environment learner settings before allocating model weights."""

import math


def validate_config(config: dict) -> dict:
    """Return an owned configuration, rejecting typos and nonfinite settings."""
    counts = (
        "horizon",
        "z_dim",
        "proprio_dim",
        "action_dim",
        "reference_horizon",
        "batch_size",
        "min_replay",
        "capacity",
        "critic_actor_ratio",
        "publish_interval",
        "max_updates",
        "checkpoint_interval",
    )
    scalars = (
        "gamma",
        "tau",
        "lr",
        "demo_ratio",
        "reference_dropout",
        "bc_weight",
        "q_weight",
    )
    required = {
        *counts,
        *scalars,
        "seed",
        "actor_after_updates",
        "bootstrap_truncation",
    }
    if set(config) - required - {"actor_max_bc_loss"} or required - set(config):
        raise ValueError("Unknown or missing online learner setting")
    for key in (*counts, "seed", "actor_after_updates"):
        minimum = 0 if key in ("seed", "actor_after_updates") else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"Invalid integer {key}")
    for key in scalars:
        if (
            type(config[key]) not in (int, float)
            or not math.isfinite(config[key])
            or config[key] < 0
        ):
            raise ValueError(f"Invalid finite scalar {key}")
    if not 0 < config["gamma"] <= 1 or not 0 < config["tau"] <= 1 or config["lr"] <= 0:
        raise ValueError("Invalid discount, target update or learning rate")
    if config["demo_ratio"] > 1 or config["reference_dropout"] > 1:
        raise ValueError("Invalid replay/dropout probability")
    if config["horizon"] != 1 or config["min_replay"] > config["capacity"]:
        raise ValueError("Online VR requires horizon=1 and bounded replay")
    if config["actor_after_updates"] > config["max_updates"]:
        raise ValueError("Actor warmup exceeds update budget")
    if type(config["bootstrap_truncation"]) is not bool:
        raise ValueError("bootstrap_truncation must be boolean")
    threshold = config.get("actor_max_bc_loss")
    if threshold is not None and (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("Invalid actor_max_bc_loss")
    return dict(config)
