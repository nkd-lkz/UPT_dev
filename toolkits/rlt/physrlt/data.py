# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Causal action-side capture and measured transition supervision."""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from toolkits.rlt.pi05_baseline import Pi05Reference

HORIZONS = (3, 10)
FEATURE_DIM = 512


@lru_cache(maxsize=4)
def projection(width: int) -> torch.Tensor:
    """Return a fixed projection, shared across all supervision arms."""
    rng = torch.Generator().manual_seed(25671 + width)
    return torch.randn(width, 128, generator=rng) / width**0.5


def compact_features(hidden: torch.Tensor) -> torch.Tensor:
    """Keep four temporal/spatial bins with a fixed 512-dimensional budget.

    This is a development readout, not the original learned RLT encoder.
    Prefix and action-side probes receive the same output width and learner.
    """
    if hidden.ndim != 2 or not torch.isfinite(hidden).all():
        raise ValueError("Expected finite token-by-channel features")
    bins = F.adaptive_avg_pool1d(hidden.float().cpu().T[None], 4)[0].T
    return (bins @ projection(hidden.shape[-1])).flatten()


def flow_path_features(hidden_states: list[torch.Tensor]) -> torch.Tensor:
    """Summarize four fixed points along one flow-matching generation path."""
    if len(hidden_states) < 4:
        raise ValueError("Flow-path features require at least four denoising forwards")
    shape = hidden_states[0].shape
    if (
        len(shape) != 2
        or shape[0] != 10
        or any(value.shape != shape for value in hidden_states)
        or not all(torch.isfinite(value).all() for value in hidden_states)
    ):
        raise ValueError("Expected finite, shape-matched 10-token flow states")
    final = len(hidden_states) - 1
    indices = (0, round(final / 3), round(2 * final / 3), final)
    projected = [
        hidden_states[index].float().cpu() @ projection(shape[-1])
        for index in indices
    ]
    return torch.cat([value.mean(0) for value in projected])


class PhysicalReference:
    """Read the last action-expert forward without changing the VLA output."""

    def __init__(self, model: nn.Module) -> None:
        self.reference = Pi05Reference(model)
        self.head = model.flow_matching_head

    def predict(self, observation: dict, instruction: str, seed: int) -> dict:
        """Return pre-execution features, reference and native commands."""
        captured = []

        def record(module, args):
            value = args[0]
            if value.ndim != 3 or value.shape[:2] != (1, 10):
                raise ValueError("Action expert must return a 10-action chunk")
            captured.append(value.detach().clone())

        handle = self.head.action_out_proj.register_forward_pre_hook(record)
        try:
            row = self.reference.predict(
                observation, instruction, capture=True, seed=seed
            )
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("No action-generation features were captured")
        row["action_feature"] = compact_features(captured[-1][0])
        row["flow_feature"] = flow_path_features([value[0] for value in captured])
        row["prefix_feature"] = compact_features(row.pop("hidden"))
        row["denoise_forwards"] = len(captured)
        return row


def metric_depth(buffer: np.ndarray, near: float, far: float) -> np.ndarray:
    """Convert MuJoCo's OpenGL depth buffer to camera-axis depth in metres."""
    value = np.asarray(buffer, dtype=np.float64)
    if not 0 < near < far or not np.isfinite(value).all():
        raise ValueError("Invalid camera depth calibration")
    if np.any((value < 0) | (value > 1)):
        raise ValueError("Expected a normalized OpenGL depth buffer")
    return (near / (1 - value * (1 - near / far))).astype(np.float32)


def measured_snapshot(environment, object_keys: list[str] | None = None) -> dict:
    """Read actual observations; privileged geometry is a training target only."""
    env = environment.env
    raw = env.current_raw_obs[0]
    if object_keys is None:
        object_keys = sorted(
            key
            for key in raw
            if key.endswith("_pos")
            and not key.startswith("robot")
            and "_to_" not in key
            and np.asarray(raw[key]).shape == (3,)
        )
    q = np.asarray(raw["robot0_joint_pos"], dtype=np.float32)
    ee = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
    grip = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
    response = np.concatenate((q, ee, grip))
    if response.shape != (12,) or not object_keys:
        raise ValueError(
            "Require measured 7-joint/3-EE/2-gripper state and world object positions"
        )
    meta = env.get_camera_meta(height=32, width=32)
    _, depth_buffer = env.render_camera(height=32, width=32, depth=True)
    depth = metric_depth(depth_buffer, meta["depth_near"], meta["depth_far"])
    depth = F.avg_pool2d(torch.from_numpy(depth)[None, None], 4).flatten().numpy()
    objects = np.concatenate([np.asarray(raw[key]) for key in object_keys])
    geometry = np.concatenate((depth, objects)).astype(np.float32)
    if not np.isfinite(response).all() or not np.isfinite(geometry).all():
        raise FloatingPointError("Nonfinite measured transition")
    return {
        "response": torch.from_numpy(response),
        "geometry": torch.from_numpy(geometry),
        "object_keys": object_keys,
        "response_fields": ["joint_pos_7", "eef_position_3", "gripper_qpos_2"],
        "camera": meta,
        "simulation_time_seconds": float(env.env.get_sim_state()[0][0]),
    }


def measured_response_snapshot(environment) -> torch.Tensor:
    """Read deployable joint, end-effector and gripper positions only."""
    raw = environment.env.current_raw_obs[0]
    response = np.concatenate(
        (
            np.asarray(raw["robot0_joint_pos"], dtype=np.float32),
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
            np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    if response.shape != (12,) or not np.isfinite(response).all():
        raise FloatingPointError("Expected finite 12-dimensional robot response")
    return torch.from_numpy(response)


def physical_reset_state(environment) -> np.ndarray:
    """Copy the numeric simulator state without process or case metadata."""
    state = environment.env.env.get_sim_state()[0]
    return np.asarray(state, dtype="<f8").copy()


def physical_reset_hash(environment) -> str:
    """Hash the numeric physics state for audit logs."""
    return hashlib.sha256(physical_reset_state(environment).tobytes()).hexdigest()


def transition_targets(before: dict, following: dict[int, dict]) -> dict:
    """Mask horizons that were not executed; never pad a terminal as motion."""
    mask = torch.tensor([h in following for h in HORIZONS])
    result = {"horizon_mask": mask}
    for field in ("response", "geometry"):
        result[field] = torch.stack(
            [
                following[h][field] - before[field]
                if h in following
                else torch.zeros_like(before[field])
                for h in HORIZONS
            ]
        )
    return result


def check_split(rows: list[dict]) -> None:
    """Prevent physical-layout leakage, even across seeds and perturbations."""
    owners = {}
    hashes = {}
    for row in rows:
        split = row["split"]
        if split not in {"train", "validation", "development"}:
            raise ValueError("Final-test data are not allowed in development fitting")
        # Same original layout across different perturbations stays in one split.
        identity = (row["suite"], row["task"], row["state_id"])
        if identity in owners and owners[identity] != split:
            raise ValueError("Physical layout appears in multiple splits")
        owners[identity] = split
        digest = row["reset_hash"]
        if digest in hashes and hashes[digest] != split:
            raise ValueError("Identical simulator reset appears in multiple splits")
        hashes[digest] = split


def masked_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Average valid horizon errors, not padded terminal values."""
    if not mask.any():
        return prediction.sum() * 0
    return (prediction - target).square().mean(-1)[mask].mean()
