# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Bounded joint-command candidates, not Cartesian skills or safety proofs."""

import math

import torch
from torch import nn


@torch.no_grad()
def select_supported_candidate(
    q_values: torch.Tensor,
    valid: torch.Tensor,
    *,
    minimum_advantage: float = 0.0,
    disagreement_penalty: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Rank finite candidates against reference ID 0 using paired critic gains.

    Inputs are [batch, candidate, critic] scores and a boolean candidate mask.
    Every critic must predict an improvement above the margin after penalizing
    disagreement of the gains. Ties and invalid reference scores retain ID 0.
    This is an inference diagnostic, not a calibrated confidence or safety test;
    mutually biased critics can still accept a harmful correction.
    """
    if (
        q_values.ndim != 3
        or min(q_values.shape) < 1
        or q_values.shape[-1] < 2
        or not q_values.is_floating_point()
        or valid.shape != q_values.shape[:2]
        or valid.dtype != torch.bool
        or valid.device != q_values.device
        or not valid[:, 0].all()
    ):
        raise ValueError("Expected [B,K,E>=2] Q and boolean [B,K] valid reference")
    if any(
        not math.isfinite(v) or v < 0 for v in (minimum_advantage, disagreement_penalty)
    ):
        raise ValueError("Admission margin and penalty must be finite and nonnegative")
    finite = torch.isfinite(q_values).all(-1)
    reference_finite = finite[:, 0]
    safe_q = torch.where(torch.isfinite(q_values), q_values, 0)
    gains = safe_q - safe_q[:, :1]
    score = gains.min(-1).values - disagreement_penalty * gains.std(-1, unbiased=False)
    admissible = valid & finite & reference_finite[:, None]
    admissible[:, 0] = False
    admissible &= torch.isfinite(score) & (score > minimum_advantage)
    ranked = score.masked_fill(~admissible, -torch.inf)
    choice = ranked.argmax(-1)
    choice = torch.where(admissible.any(-1), choice, 0)
    return {
        "choice": choice,
        "admissible": admissible,
        "paired_advantage": score,
        "invalid_reference_q": ~reference_finite,
        "invalid_candidate_q": ~finite,
    }


class JointActionCandidates(nn.Module):
    """Build reference-relative chunks in normalized joint-delta coordinates.

    The final action dimension is a gripper command and is never modified.
    Every arm correction is bounded by ``radius`` per control step. Damping
    and braking move toward zero within that bound; braking is not an e-stop.
    Candidate IDs are stable across observations. Exact duplicate chunks are
    masked, keeping the first representative (the reference always survives).
    """

    def __init__(self, action_dim: int, chunk_len: int, radius: float = 0.08) -> None:
        super().__init__()
        if action_dim < 2 or chunk_len < 1:
            raise ValueError("Need at least one arm joint, one gripper, and one step.")
        if not math.isfinite(radius) or not 0 < radius <= 1:
            raise ValueError("Candidate radius must be finite and in (0, 1].")
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.radius = radius
        self.names = ("reference", "dampen_arm", "brake_arm") + tuple(
            f"joint_{joint}_{direction}"
            for joint in range(action_dim - 1)
            for direction in ("plus", "minus")
        )
        offsets = torch.zeros(len(self.names), chunk_len, action_dim)
        for joint in range(action_dim - 1):
            offsets[3 + 2 * joint, :, joint] = radius
            offsets[4 + 2 * joint, :, joint] = -radius
        self.register_buffer("offsets", offsets)
        # Reject semantically incompatible checkpoints, even with identical shapes.
        self.register_buffer(
            "contract", torch.tensor([1, action_dim, chunk_len, radius])
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        incoming = state_dict.get(prefix + "contract")
        if incoming is not None and not torch.equal(
            incoming.detach().cpu(), self.contract.detach().cpu()
        ):
            error_msgs.append("Atomic candidate checkpoint contract mismatch.")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return candidates [B,K,H,D] and unique-candidate mask [B,K]."""
        if reference.ndim != 3 or reference.shape[1:] != (
            self.chunk_len,
            self.action_dim,
        ):
            raise ValueError(
                "Reference must have shape [batch, chunk_len, action_dim]."
            )
        if not torch.isfinite(reference).all():
            raise ValueError("Non-finite reference actions cannot form candidates.")
        # The baseline environment clips normalized controls to [-1, 1].
        ref = reference.detach().clamp(-1, 1)
        corrections = self.offsets[None].expand(ref.shape[0], -1, -1, -1).clone()
        corrections[:, 1, :, :-1] = (-0.5 * ref[..., :-1]).clamp(
            -self.radius, self.radius
        )
        corrections[:, 2, :, :-1] = (-ref[..., :-1]).clamp(-self.radius, self.radius)
        candidates = (ref[:, None] + corrections).clamp(-1, 1)
        # Pairwise comparison is small (17 candidates for Panda), needs no CPU sync.
        flat = candidates.flatten(2)
        same = (flat[:, :, None] == flat[:, None, :]).all(-1)
        duplicate = torch.tril(same, diagonal=-1).any(-1)
        return candidates, ~duplicate
