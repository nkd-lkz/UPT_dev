# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Learn a context from completed interaction records, never from live replay."""

import torch
from torch import nn

from rlinf.algorithms.rlt.interaction_memory import InteractionMemoryConfig


def response_features(
    obs: dict[str, torch.Tensor],
    c: InteractionMemoryConfig,
    *,
    record_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate empirical arm response and excitation from completed records.

    The ratio relates summed delta targets to net joint motion. It is a local
    controller-response descriptor, not a stiffness estimate or a contact label.
    The last action component is the gripper and is excluded. Inputs must use
    the controller scale declared by ``joint_delta_scale``. Optional [B, slots]
    nonnegative weights are supplied by the caller using past-record metadata;
    the default retains the existing unweighted estimator exactly.
    """
    valid = obs["memory_valid"]
    events = torch.where(valid[..., None], obs["memory_events"], 0.0)
    p, a, h = c.proprio_dim, c.action_dim, c.chunk_len
    commands = events[..., p : p + h * a].reshape(*events.shape[:2], h, a)
    ticks = events[..., 2 * p + h * a : 2 * p + h * a + h].bool()
    commands = torch.where(ticks[..., None], commands, 0.0)
    commanded = commands[..., : a - 1].sum(-2) * c.joint_delta_scale
    delta = events[..., p + h * a : p + h * a + a - 1]
    if record_weights is None:
        energy = commanded.square().sum(1)
        cross = (commanded * delta).sum(1)
    else:
        if record_weights.shape != valid.shape:
            raise ValueError("Record weights must match the memory slot shape")
        weights = record_weights.to(commanded)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Record weights must be finite and nonnegative")
        weights = torch.where(valid, weights, 0)[..., None]
        energy = (weights * commanded.square()).sum(1)
        cross = (weights * commanded * delta).sum(1)
    slope = (cross / (energy + 1e-4)).clamp(-2, 2)
    support = energy / (energy + 1e-4)
    return torch.cat((slope, support), -1)


class RLTMemoryEncoder(nn.Module):
    """Read decision-time raw evidence with a current-proprio attention query."""

    def __init__(self, config: InteractionMemoryConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        if c.reader_type in ("response", "zero"):
            # Parameter-free comparator with the same downstream context width.
            return
        self.event = nn.Sequential(
            nn.Linear(c.event_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )
        self.query = nn.Linear(c.proprio_dim, c.hidden_dim)
        self.position = nn.Parameter(torch.randn(c.slots, c.hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            c.hidden_dim, c.num_heads, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(c.hidden_dim), nn.Linear(c.hidden_dim, c.hidden_dim), nn.SiLU()
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return [B, hidden_dim]; an empty memory returns exactly zero."""
        c = self.config
        events, valid, query = (
            obs[key] for key in ("memory_events", "memory_valid", "memory_query")
        )
        batch = events.shape[0]
        if (
            events.shape != (batch, c.slots, c.event_dim)
            or valid.shape != (batch, c.slots)
            or query.shape != (batch, c.proprio_dim)
        ):
            raise ValueError("Memory observation does not match configured schema")
        if valid.dtype != torch.bool:
            raise ValueError("memory_valid must be boolean")
        if c.reader_type == "zero":
            # Preserve the response comparator's head width and initialization.
            return events.new_zeros((batch, c.hidden_dim))
        if c.reader_type == "response":
            features = response_features(obs, c)
            return torch.nn.functional.pad(
                features, (0, c.hidden_dim - features.shape[-1])
            )
        dtype, device = self.position.dtype, self.position.device
        valid = valid.to(device)
        events = events.to(device=device, dtype=dtype)
        query = query.to(device=device, dtype=dtype)
        # Mask before the MLP, so an invalid padded NaN cannot contaminate attention.
        events = torch.where(valid[..., None], events, torch.zeros_like(events))
        tokens = self.event(events) + self.position[None]
        empty = ~valid.any(dim=-1)
        padding = ~valid.clone()
        padding[empty, 0] = False
        result, _ = self.attention(
            self.query(query)[:, None],
            tokens,
            tokens,
            key_padding_mask=padding,
            need_weights=False,
        )
        context = self.output(result[:, 0])
        return torch.where(empty[:, None], torch.zeros_like(context), context)
