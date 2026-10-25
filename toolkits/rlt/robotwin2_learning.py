# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""AlphaBrain new-RLT heads for RoboTwin 2.0 ALOHA joint control."""

from __future__ import annotations

import numpy as np

from toolkits.rlt.robodojo_learning import ArxLearner, ArxNormalizer

PROTOCOL = {
    "name": "robotwin2-aloha-pi05-alphabrain-rlt-v1",
    "track": "alphabrain-rlt",
    "backbone": "pi0.5-flow-matching",
    "feature_shape": [768, 2048],
    "action_dim": 14,
    "prop_dim": 14,
    "reference_horizon": 50,
    "chunk_len": 10,
    "actor_space": "quantile_normalized_joint_delta_absolute_gripper",
    "execution_space": "absolute_joint_radians_gripper_01",
    "reference_dropout": 0.0,
    "memory": False,
    "intervention": False,
}


class AlohaNormalizer(ArxNormalizer):
    """Normalize RoboTwin ALOHA actions using the pinned Pi0.5 statistics.

    RoboTwin and RoboDojo use the same fourteen-axis mathematical contract:
    twelve arm joints are represented as deltas while the two grippers remain
    absolute. The class has its own embodiment name so checkpoints cannot be
    mistaken for ARX X5 artifacts.
    """


class AlohaLearner(ArxLearner):
    """Use AlphaBrain's actor/twin-Q heads with a RoboTwin checkpoint contract."""

    def __init__(self, normalizer: AlohaNormalizer, **kwargs) -> None:
        super().__init__(normalizer, **kwargs)
        token_dim = int(getattr(self, "token_dim", 0))
        self.protocol = dict(PROTOCOL)
        if token_dim:
            self.protocol["memory"] = {
                "kind": "grounded_v1",
                "token_dim": token_dim,
            }


def validate_feature_batch(hidden: np.ndarray, mask: np.ndarray) -> None:
    """Validate the three-view image-token slice consumed by new RLT."""
    hidden = np.asarray(hidden)
    mask = np.asarray(mask)
    if hidden.ndim != 3 or hidden.shape[1:] != (768, 2048):
        raise ValueError("RoboTwin Pi0.5 features must have shape [B, 768, 2048]")
    if mask.shape != hidden.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("Feature validity mask must be boolean with shape [B, 768]")
    if not np.isfinite(hidden).all() or not mask.all():
        raise ValueError("Require finite features from all three RoboTwin views")
