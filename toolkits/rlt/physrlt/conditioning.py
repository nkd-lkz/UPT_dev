# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Candidate-action physical consequences for value conditioning."""

from __future__ import annotations

import copy

import torch
from torch import nn

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
        if mode not in {"outcomes", "mixed"}:
            raise ValueError("Unknown candidate consequence conditioning mode")
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.method = method
        self.mode = mode

    def train(self, mode: bool = True):
        """Keep the frozen predictor in evaluation mode."""
        super().train(False)
        self.model.eval()
        return self

    def forward(self, token: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return a 128-D candidate-aware critic token."""
        if token.ndim != 2 or token.shape[-1] != 128:
            raise ValueError("Candidate conditioning requires a 128-D base token")
        physical = physical_quantities(self.model, token, actions, self.method)
        if self.mode == "outcomes":
            result = physical
        else:
            compact = torch.nn.functional.adaptive_avg_pool1d(physical[:, None], 64)[
                :, 0
            ]
            result = torch.cat((token[:, :64], compact), dim=-1)
        if result.shape != token.shape or not torch.isfinite(result).all():
            raise FloatingPointError("Invalid candidate consequence token")
        return result
