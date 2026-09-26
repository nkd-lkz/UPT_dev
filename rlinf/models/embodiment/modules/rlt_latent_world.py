# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Action-conditioned future features in a frozen RLT coordinate system."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LatentWorldConfig:
    """Shapes and losses shared by offline adaptation and online replay."""

    z_dim: int = 2048
    proprio_dim: int = 9
    action_dim: int = 8
    chunk_len: int = 10
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ensemble_size: int = 3
    horizons: tuple[int, ...] = (1, 5, 10)
    future_weight: float = 1.0
    latent_mse_weight: float = 0.1
    proprio_weight: float = 0.5
    anchor_weight: float = 0.1
    bc_weight: float = 0.1
    bootstrap_probability: float = 0.8
    predict_residual: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", tuple(self.horizons))
        dimensions = (
            self.z_dim,
            self.proprio_dim,
            self.action_dim,
            self.chunk_len,
            self.hidden_dim,
            self.num_heads,
            self.num_layers,
            self.ensemble_size,
        )
        if min(dimensions) <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError(
                "Positive dimensions and hidden_dim % num_heads == 0 required"
            )
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be nonempty, unique and increasing")
        if self.horizons[0] < 1 or self.horizons[-1] != self.chunk_len:
            raise ValueError(
                "horizons must include chunk_len and stay in [1, chunk_len]"
            )
        if not 0 < self.bootstrap_probability <= 1:
            raise ValueError("bootstrap_probability must be in (0, 1]")
        if (
            min(
                self.future_weight,
                self.latent_mse_weight,
                self.proprio_weight,
                self.anchor_weight,
                self.bc_weight,
            )
            < 0
        ):
            raise ValueError("Loss weights must be nonnegative")


class RLTLatentWorld(nn.Module):
    """Predict frozen future RL tokens and normalized proprioceptive changes.

    Future queries attend only to the action prefix for their own horizon.
    The target encoder is external and frozen; no future observations enter
    ``encode`` or policy inference. Ensemble disagreement is a heuristic, not
    calibrated probability or a contact/safety detector.
    """

    def __init__(self, config: LatentWorldConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.encoder = nn.Sequential(
            nn.Linear(c.z_dim + c.proprio_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )
        self.action_embedding = nn.Linear(c.action_dim, c.hidden_dim)
        self.positions = nn.Parameter(torch.randn(c.chunk_len, c.hidden_dim) * 0.02)
        self.future_queries = nn.Parameter(
            torch.randn(c.chunk_len, c.hidden_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            c.hidden_dim,
            c.num_heads,
            dim_feedforward=2 * c.hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            c.num_layers,
            norm=nn.LayerNorm(c.hidden_dim),
            enable_nested_tensor=False,
        )
        # Independent initialization avoids identical ensemble members.
        self.predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(c.hidden_dim, c.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(c.hidden_dim, c.z_dim + c.proprio_dim),
                )
                for _ in range(c.ensemble_size)
            ]
        )
        if c.predict_residual:
            # Start at the strong short-horizon persistence comparator.
            with torch.no_grad():
                for head in self.predictors:
                    head[-1].weight[: c.z_dim].zero_()
                    head[-1].bias[: c.z_dim].zero_()
        self.anchor = nn.Linear(c.hidden_dim, c.z_dim)
        self.behavior = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, c.chunk_len * c.action_dim),
        )

    def encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode only current observation features, in float32."""
        z = obs["z_rl"].reshape(-1, self.config.z_dim).float()
        p = obs["proprio"].reshape(z.shape[0], self.config.proprio_dim).float()
        return self.encoder(torch.cat((F.layer_norm(z, (z.shape[-1],)), p), dim=-1))

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predictions [ensemble, batch, feature] after ``horizon`` actions."""
        c = self.config
        if horizon not in c.horizons:
            raise ValueError(
                f"Untrained horizon {horizon}; expected one of {c.horizons}"
            )
        state = self.encode(obs)
        actions = actions.reshape(state.shape[0], -1, c.action_dim).float()
        if actions.shape[1] < horizon:
            raise ValueError("An action for every predicted control step is required")
        action_tokens = (
            self.action_embedding(actions[:, :horizon]) + self.positions[:horizon]
        )
        query = self.future_queries[horizon - 1].expand(state.shape[0], 1, -1)
        tokens = torch.cat((state[:, None], action_tokens, query), dim=1)
        future = self.transformer(tokens)[:, -1]
        predictions = torch.stack([head(future) for head in self.predictors])
        future_z = predictions[..., : c.z_dim]
        if c.predict_residual:
            current = obs["z_rl"].detach().reshape(-1, c.z_dim).float()
            future_z = future_z + F.layer_norm(current, (c.z_dim,))[None]
        return future_z, predictions[..., c.z_dim :]

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # where avoids invalid targets contributing NaN * 0.
        return torch.where(mask, values, 0.0).sum() / mask.sum().clamp_min(1)

    def loss(
        self, batch: dict[str, torch.Tensor], *, offline: bool
    ) -> tuple[torch.Tensor, dict]:
        """Supervise valid episode-local futures; terminal replay rows are masked.

        Batch keys: z_rl, proprio, actions [B,H,A], future_z [B,K,Z],
        future_proprio [B,K,P], valid [B,K], horizons [K]. Offline batches also
        carry action_valid [B,H]. Online uses one K equal to the executed chunk
        duration, not one simulator tick, and omits behavior cloning.
        """
        c = self.config
        obs = {key: batch[key] for key in ("z_rl", "proprio")}
        losses, cosine_values, delta_values = [], [], []
        for idx, horizon in enumerate(batch["horizons"]):
            pred_z, pred_delta = self(obs, batch["actions"], int(horizon))
            target_z = F.layer_norm(
                batch["future_z"][:, idx].detach().float(), (c.z_dim,)
            )
            target_delta = (
                batch["future_proprio"][:, idx].detach().float()
                - batch["proprio"].detach().float()
            )
            cosine = 1 - F.cosine_similarity(pred_z, target_z[None], dim=-1)
            mse = (pred_z - target_z[None]).square().mean(-1)
            delta = F.smooth_l1_loss(
                pred_delta, target_delta[None].expand_as(pred_delta), reduction="none"
            ).mean(-1)
            valid = batch["valid"][:, idx].bool()[None].expand(c.ensemble_size, -1)
            if self.training:
                bootstrap = torch.rand_like(cosine) < c.bootstrap_probability
                # The first head sees all valid transitions, even in tiny batches.
                bootstrap[0] = True
                valid = valid & bootstrap
            losses.append(
                self._masked_mean(
                    cosine + c.latent_mse_weight * mse + c.proprio_weight * delta, valid
                )
            )
            cosine_values.append(self._masked_mean(cosine, valid).detach())
            delta_values.append(self._masked_mean(delta, valid).detach())
        state = self.encode(obs)
        row_valid = batch["valid"].bool().any(-1)
        anchor_target = F.layer_norm(batch["z_rl"].detach().float(), (c.z_dim,))
        anchor = self._masked_mean(
            (self.anchor(state) - anchor_target).square().mean(-1), row_valid
        )
        loss = c.future_weight * torch.stack(losses).mean() + c.anchor_weight * anchor
        # Touch all parameters for FSDP on online and all-terminal minibatches.
        bc_pred = self.behavior(state).reshape(-1, c.chunk_len, c.action_dim).tanh()
        bc = bc_pred.sum() * 0.0
        if offline:
            error = (bc_pred - batch["actions"].detach()).square().mean(-1)
            bc = self._masked_mean(error, batch["action_valid"].bool())
        loss = loss + c.bc_weight * bc
        return loss, {
            "world/loss": loss.detach(),
            "world/cosine_error": torch.stack(cosine_values).mean(),
            "world/proprio_error": torch.stack(delta_values).mean(),
            "world/anchor": anchor.detach(),
            "world/bc": bc.detach(),
            "world/valid_fraction": batch["valid"].float().mean(),
        }

    @torch.no_grad()
    def disagreement(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> torch.Tensor:
        """Return per-row normalized-latent ensemble variance, without gradients."""
        prediction, _ = self(obs, actions, self.config.chunk_len)
        return prediction.var(dim=0, unbiased=False).mean(-1, keepdim=True)

    def checkpoint(self, feature_contract: dict) -> dict:
        """Return a weights-only-loadable sidecar with feature provenance."""
        return {
            "format_version": 1,
            "config": asdict(self.config),
            "feature_contract": feature_contract,
            "model": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, expected_contract: dict | None = None
    ) -> "RLTLatentWorld":
        """Load a sidecar; reject incompatible provenance instead of silently adapting."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported latent world checkpoint version")
        if (
            expected_contract is not None
            and payload["feature_contract"] != expected_contract
        ):
            raise ValueError(
                "Frozen RLT feature contract mismatch; rebuild the cache/sidecar"
            )
        model = cls(LatentWorldConfig(**payload["config"]))
        model.load_state_dict(payload["model"], strict=True)
        model.feature_contract = payload["feature_contract"]
        return model
