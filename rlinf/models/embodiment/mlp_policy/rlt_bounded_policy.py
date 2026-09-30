# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Continuous residual comparator for the finite joint-candidate experiment."""

import math

import torch

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


class RLTBoundedResidualPolicy(RLTMLPPolicy):
    """Keep the reference gripper and bound every arm correction by one radius.

    This matches the candidate bank's per-component range, not its discrete
    hypothesis class: multiple joints and timesteps can be corrected together.
    The deterministic initial policy equals the clipped reference. Pre-transform
    log probabilities are diagnostic only; training requires zero entropy weight.
    """

    def __init__(self, *, bounded_residual: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        if set(bounded_residual) - {"enabled", "radius"}:
            raise ValueError("Unknown bounded_residual field")
        self.radius = float(bounded_residual.get("radius", 0.08))
        if not math.isfinite(self.radius) or not 0 < self.radius <= 1:
            raise ValueError("Residual radius must be finite and in (0, 1]")
        if self.step_action_dim < 2:
            raise ValueError("Residual comparator requires arm and gripper commands")
        torch.nn.init.zeros_(self.actor_mean.weight)
        torch.nn.init.zeros_(self.actor_mean.bias)
        self.register_buffer("residual_contract", torch.tensor([1.0, self.radius]))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        incoming = state_dict.get(prefix + "residual_contract")
        if incoming is not None and not torch.equal(
            incoming.detach().cpu(), self.residual_contract.detach().cpu()
        ):
            raise ValueError("Residual checkpoint radius contract mismatch")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def sac_forward(self, obs: dict, deterministic: bool = False, **kwargs):
        """Generate reference-relative arm controls while preserving the gripper."""
        residual, logprobs, value = super().sac_forward(
            obs, deterministic=deterministic, **kwargs
        )
        correction = residual.reshape(-1, self.chunk_len, self.step_action_dim)
        arm_mask = torch.ones_like(correction)
        arm_mask[..., -1] = 0
        reference = self._get_ref_chunk(obs).clamp(-1, 1)
        action = (reference + self.radius * (correction * arm_mask).flatten(1)).clamp(
            -1, 1
        )
        return action, logprobs, value

    def sft_forward(self, data: dict, **kwargs):
        raise NotImplementedError("Residual comparator uses the Stage 2 Q+BC objective")
