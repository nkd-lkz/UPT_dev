# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Learn a context from completed interaction records, never from live replay."""

import torch
from torch import nn

from rlinf.algorithms.rlt.interaction_memory import InteractionMemoryConfig


class RLTMemoryEncoder(nn.Module):
    """Read decision-time raw evidence with a current-proprio attention query."""

    def __init__(self, config: InteractionMemoryConfig) -> None:
        super().__init__()
        self.config = config
        c = config
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
