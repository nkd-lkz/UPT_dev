# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Candidate-action physical consequences for value conditioning."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from toolkits.rlt.physrlt.data import HORIZONS
from toolkits.rlt.physrlt.model import SparseObjectMotionHead
from toolkits.rlt.physrlt.probe import response_indices


def differentiable_env_commands(actions: torch.Tensor) -> torch.Tensor:
    """Match LIBERO command mapping while retaining a gripper surrogate gradient."""
    if actions.ndim != 3 or actions.shape[-2:] != (10, 7):
        raise ValueError("Require a batch of 10 x 7 policy action chunks")
    if not torch.isfinite(actions).all():
        raise ValueError("Policy action chunks must be finite")
    commands = actions.clamp(-1, 1)
    soft_gripper = torch.tanh(5 * actions[..., 6])
    hard_gripper = torch.where(
        actions[..., 6] >= 0,
        torch.ones_like(soft_gripper),
        -torch.ones_like(soft_gripper),
    )
    # The forward pass exactly matches the environment's binary command. The
    # backward pass is only a bounded surrogate because the real threshold has
    # zero derivative almost everywhere.
    gripper = soft_gripper + (hard_gripper - soft_gripper).detach()
    return torch.cat((commands[..., :6], gripper[..., None]), dim=-1)


def physical_quantities(
    model: nn.Module,
    token: torch.Tensor,
    actions: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """Predict a fixed-width normalized consequence vector for each action."""
    commands = differentiable_env_commands(actions)
    predictions = model.consequences(token, commands)
    response = predictions["response"].clone()
    supervised = response_indices(method)
    omitted = sorted(set(range(response.shape[-1])) - set(supervised))
    if omitted:
        response[..., omitted] = 0
    geometry = predictions["geometry"]
    if geometry.shape[-1] < 64:
        raise ValueError("Geometry prediction lacks the 8 x 8 depth target")
    depth = geometry[..., :64].reshape(-1, 2, 1, 8, 8)
    depth = torch.nn.functional.avg_pool2d(depth.flatten(0, 1), 2)
    depth = depth.reshape(len(token), -1)
    quantities = torch.cat(
        (response.flatten(1), depth, geometry[..., 64:].flatten(1)), dim=-1
    )
    if quantities.shape[-1] > 128:
        raise ValueError("Explicit physical quantities exceed token budget")
    return torch.nn.functional.pad(quantities, (0, 128 - quantities.shape[-1]))


class CandidateConsequenceConditioner(nn.Module):
    """Condition a critic on predicted outcomes of its candidate action.

    The readout is copied and frozen. Gradients can pass from the critic through
    its action input, but never update the physical readout during online RL.
    """

    def __init__(self, model: nn.Module, method: str, mode: str = "mixed") -> None:
        super().__init__()
        if mode not in {
            "outcomes",
            "mixed",
            "relative",
            "zero_concat",
            "relative_concat",
            "relative_unit_concat",
        }:
            raise ValueError("Unknown candidate consequence conditioning mode")
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.method = method
        self.mode = mode

    def train(self, mode: bool = True):
        """Keep the frozen predictor in evaluation mode."""
        super().train(False)
        self.model.eval()
        return self

    def forward(
        self,
        token: torch.Tensor,
        actions: torch.Tensor,
        reference: torch.Tensor,
        physical_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a 128-D candidate-aware critic token."""
        if token.ndim != 2 or token.shape[-1] != 128:
            raise ValueError("Candidate conditioning requires a 128-D base token")
        consequence_token = physical_token if physical_token is not None else token
        if consequence_token.shape != token.shape:
            raise ValueError("Physical and base tokens must share shape")
        if reference.shape != actions.shape:
            raise ValueError("Reference and candidate action chunks must match")
        physical = physical_quantities(
            self.model, consequence_token, actions, self.method
        )
        if self.mode == "outcomes":
            result = physical
        elif self.mode == "zero_concat":
            result = torch.cat((token, torch.zeros_like(physical)), dim=-1)
        elif self.mode == "relative_concat":
            reference_physical = physical_quantities(
                self.model, consequence_token, reference, self.method
            )
            result = torch.cat((token, physical - reference_physical), dim=-1)
        elif self.mode == "relative_unit_concat":
            reference_physical = physical_quantities(
                self.model, consequence_token, reference, self.method
            )
            # The critic already receives the candidate action and therefore
            # its magnitude. Normalize the active consequence coordinates so
            # their direction is not hidden by the unit-scale base token.
            active = (physical - reference_physical)[:, :68]
            scale = active.square().mean(-1, keepdim=True).sqrt()
            normalized = torch.where(
                scale > 1e-6, active / scale.clamp_min(1e-6), torch.zeros_like(active)
            )
            result = torch.cat(
                (
                    token,
                    torch.nn.functional.pad(normalized, (0, 128 - active.shape[-1])),
                ),
                dim=-1,
            )
        else:
            if self.mode == "relative":
                reference_physical = physical_quantities(
                    self.model, consequence_token, reference, self.method
                )
                physical = physical - reference_physical
            compact = torch.nn.functional.adaptive_avg_pool1d(physical[:, None], 64)[
                :, 0
            ]
            result = torch.cat((token[:, :64], compact), dim=-1)
        expected_width = 256 if self.mode.endswith("_concat") else 128
        if (
            result.ndim != 2
            or result.shape != (len(token), expected_width)
            or not torch.isfinite(result).all()
        ):
            raise FloatingPointError("Invalid candidate consequence token")
        return result


class SparseMotionConditioner(nn.Module):
    """Expose candidate-relative sparse object motion to a matched critic."""

    def __init__(
        self,
        head: SparseObjectMotionHead,
        *,
        zero_channel: bool = False,
    ) -> None:
        super().__init__()
        self.head = copy.deepcopy(head).eval().requires_grad_(False)
        self.zero_channel = zero_channel

    def train(self, mode: bool = True):
        """Keep the frozen sparse predictor in evaluation mode."""
        super().train(False)
        self.head.eval()
        return self

    @staticmethod
    def _quantities(
        head: SparseObjectMotionHead,
        token: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        commands = differentiable_env_commands(actions)
        event_logit, conditional, expected = head(token, commands, HORIZONS[-1])
        probability = (
            event_logit.sigmoid()[:, None]
            if event_logit is not None
            else token.new_ones((len(token), 1))
        )
        return torch.cat((probability, conditional, expected), dim=-1)

    def forward(
        self,
        token: torch.Tensor,
        actions: torch.Tensor,
        reference: torch.Tensor,
        physical_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append a fixed-width candidate-minus-reference consequence channel."""
        if token.ndim != 2 or token.shape[-1] != 128:
            raise ValueError("Sparse conditioning requires a 128-D base token")
        consequence_token = physical_token if physical_token is not None else token
        if consequence_token.shape != token.shape or actions.shape != reference.shape:
            raise ValueError("Sparse conditioning inputs have incompatible shapes")
        candidate = self._quantities(self.head, consequence_token, actions)
        baseline = self._quantities(self.head, consequence_token, reference)
        relative = candidate - baseline
        if self.zero_channel:
            relative = torch.zeros_like(relative)
        result = torch.cat(
            (token, torch.nn.functional.pad(relative, (0, 128 - relative.shape[-1]))),
            dim=-1,
        )
        if result.shape != (len(token), 256) or not torch.isfinite(result).all():
            raise FloatingPointError("Invalid sparse candidate consequence token")
        return result


def load_sparse_motion_conditioner(
    path: Path,
    *,
    expected_readout_sha256: str | None = None,
    zero_channel: bool = False,
) -> tuple[SparseMotionConditioner, dict]:
    """Load an immutable sparse head and validate its representation version."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != 1 or "model" not in payload:
        raise ValueError("Unsupported sparse-motion checkpoint schema")
    if (
        expected_readout_sha256 is not None
        and payload.get("readout_sha256") != expected_readout_sha256
    ):
        raise ValueError("Sparse head and physical readout versions differ")
    head = SparseObjectMotionHead(bool(payload.get("gated")))
    head.load_state_dict(payload["model"], strict=True)
    return SparseMotionConditioner(head, zero_channel=zero_channel), payload
