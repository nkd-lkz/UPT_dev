# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""A capacity-matched physical readout with training-only consequence decoders."""

from __future__ import annotations

import torch
from torch import nn

from toolkits.rlt.physrlt.data import FEATURE_DIM, HORIZONS


class PhysicalReadout(nn.Module):
    """Encode current action intent; predict outcomes for an explicit command.

    The encoder never consumes an executed future or privileged object state.
    Dynamics decoders accept the command actually executed during training.
    Actor/critic deployment may discard both decoders and retain ``encode``.
    """

    def __init__(self, response_dim: int, geometry_dim: int, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM + 8 + 70, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.reconstruct = nn.Linear(latent_dim, FEATURE_DIM)
        self.response = self._decoder(latent_dim + 70 + 1, response_dim)
        self.geometry = self._decoder(latent_dim + 70 + 1, geometry_dim)

    @staticmethod
    def _decoder(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, output_dim)
        )

    def encode(
        self, feature: torch.Tensor, prop: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Return a current-state token; all inputs exist before execution."""
        return self.encoder(torch.cat((feature, prop, reference.flatten(1)), dim=-1))

    def consequences(self, token: torch.Tensor, commands: torch.Tensor) -> dict:
        """Predict each horizon using only its accepted command prefix."""
        response, geometry = [], []
        for horizon in HORIZONS:
            prefix = commands.clone()
            prefix[:, horizon:] = 0
            time = token.new_full((len(token), 1), horizon / 10)
            value = torch.cat((token, prefix.flatten(1), time), -1)
            response.append(self.response(value))
            geometry.append(self.geometry(value))
        return {
            "response": torch.stack(response, dim=1),
            "geometry": torch.stack(geometry, dim=1),
        }
