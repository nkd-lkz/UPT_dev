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

    def __init__(
        self,
        response_dim: int,
        geometry_dim: int,
        latent_dim: int = 128,
        *,
        factorized: bool = False,
    ):
        super().__init__()
        self.factorized = factorized
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
            response_token, geometry_token = token, token
            if self.factorized:
                # Equal total capacity, with disjoint gradient supervision for
                # geometry and execution response; the policy still gets both.
                middle = token.shape[-1] // 2
                response_token = torch.cat(
                    (torch.zeros_like(token[:, :middle]), token[:, middle:]), -1
                )
                geometry_token = torch.cat(
                    (token[:, :middle], torch.zeros_like(token[:, middle:])), -1
                )
            response.append(
                self.response(torch.cat((response_token, prefix.flatten(1), time), -1))
            )
            geometry.append(
                self.geometry(torch.cat((geometry_token, prefix.flatten(1), time), -1))
            )
        return {
            "response": torch.stack(response, dim=1),
            "geometry": torch.stack(geometry, dim=1),
        }


class SparseObjectMotionHead(nn.Module):
    """Predict a motion event and conditional 3-D object displacement."""

    def __init__(self, gated: bool) -> None:
        super().__init__()
        self.gated = gated
        self.network = nn.Sequential(
            nn.Linear(128 + 70 + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 4 if gated else 3),
        )

    def forward(
        self, token: torch.Tensor, commands: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Return event logit, conditional motion, and expected displacement."""
        if commands.ndim != 3 or commands.shape[-2:] != (10, 7):
            raise ValueError("Sparse motion requires 10 x 7 command chunks")
        if horizon not in HORIZONS:
            raise ValueError("Sparse motion requires a supervised horizon")
        prefix = commands.clone()
        prefix[:, horizon:] = 0
        time = token.new_full((len(token), 1), horizon / 10)
        output = self.network(torch.cat((token, prefix.flatten(1), time), -1))
        if not self.gated:
            return None, output, output
        event_logit = output[:, 0]
        conditional = output[:, 1:]
        expected = event_logit.sigmoid()[:, None] * conditional
        return event_logit, conditional, expected
