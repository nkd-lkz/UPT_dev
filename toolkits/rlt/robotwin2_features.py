# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen RoboTwin Pi0.5 prefix features and unchanged flow-matching actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from toolkits.rlt.robotwin2_learning import validate_feature_batch

MODEL_REVISION = "fa8df6ed103db0f5549c122f3a17c00ba6426c98"
ROBOTWIN_REVISION = "0008ae6800df9f75fc8de7098bacb01735fd8fd2"
IMAGE_TOKENS = 3 * 256
REFERENCE_HORIZON = 50


def model_config(checkpoint: Path) -> Any:
    """Build the exact RLinf Pi0.5 reference configuration for RoboTwin."""
    norm_stats = checkpoint / "physical-intelligence/robotwin/norm_stats.json"
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError("RoboTwin reference requires merged model.safetensors")
    if not norm_stats.is_file():
        raise FileNotFoundError(f"Missing RoboTwin normalization: {norm_stats}")
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": str(checkpoint),
            "precision": "bf16",
            "pi05": True,
            # Keep the network and output transform at the horizon used by the
            # published checkpoint. Online RLT slices this reference to its
            # shorter learning/execution window; it must not change the VLA's
            # flow-matching distribution.
            "num_action_chunks": REFERENCE_HORIZON,
            "action_dim": 14,
            "num_steps": 5,
            # The published checkpoint contains a value head.  Keep it in the
            # module graph so strict loading matches the official evaluation
            # config; reference inference below never reads its output.
            "add_value_head": True,
            "openpi_data": {"norm_stats_path": str(norm_stats)},
            "openpi": {
                # The released RLinf checkpoint was exported from the RL task
                # wrapper and includes its value head.  Use the same wrapper
                # for strict compatibility, then freeze every parameter below.
                "task": "rl",
                "config_name": "pi05_aloha_robotwin",
                "action_horizon": REFERENCE_HORIZON,
                "action_chunk": REFERENCE_HORIZON,
                "action_env_dim": 14,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "noise_level": 0.3,
                "detach_critic_input": True,
                "torch_compile": False,
            },
        }
    )


def validate_env_observation(env_obs: dict[str, Any]) -> None:
    """Require the batched three-camera and 14-D state contract."""

    def as_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    main = as_numpy(env_obs.get("main_images"))
    wrist = as_numpy(env_obs.get("wrist_images"))
    states = as_numpy(env_obs.get("states"))
    prompts = env_obs.get("task_descriptions")
    if main.ndim != 4 or main.shape[-1] != 3 or main.dtype != np.uint8:
        raise ValueError("main_images must be batched HWC uint8 RGB")
    if (
        wrist.ndim != 5
        or wrist.shape[:2] != (main.shape[0], 2)
        or wrist.shape[-1] != 3
        or wrist.dtype != np.uint8
    ):
        raise ValueError("wrist_images must contain two batched HWC uint8 views")
    if states.shape != (main.shape[0], 14) or not np.isfinite(states).all():
        raise ValueError("RoboTwin ALOHA state must have shape [B, 14]")
    if (
        not isinstance(prompts, (list, tuple))
        or len(prompts) != main.shape[0]
        or any(not isinstance(item, str) or not item.strip() for item in prompts)
    ):
        raise ValueError("Require one nonempty instruction per environment")


class FrozenRoboTwinPi05:
    """Read frozen image-prefix states without changing Pi0.5 action sampling."""

    def __init__(self, checkpoint: Path, *, device: str = "cuda") -> None:
        from rlinf.models.embodiment.openpi import get_model

        self.model = get_model(model_config(checkpoint)).to(device).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def prefix_features(self, observation: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the valid final-layer tokens for the three camera views."""
        from rlinf.models.embodiment.openpi.modules.model import (
            Observation,
            preprocess_observation,
        )

        observation = self.model._observation_to_device(
            Observation.from_observation_like(observation)
        )
        prepared = preprocess_observation(observation, train=False)
        prefix, prefix_mask, _ = self.model.build_prefix_cache(prepared)
        hidden = prefix[:, :IMAGE_TOKENS].to(dtype=torch.float32)
        valid = prefix_mask[:, :IMAGE_TOKENS].to(dtype=torch.bool)
        validate_feature_batch(hidden.cpu().numpy(), valid.cpu().numpy())
        return hidden, valid

    @torch.no_grad()
    def infer(
        self,
        env_obs: dict[str, Any],
        *,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return final image-token states and the decoded 50x14 reference."""
        validate_env_observation(env_obs)
        observation = self.model.env_obs_to_observation(env_obs)
        from rlinf.models.embodiment.openpi.modules.model import (
            preprocess_observation,
        )

        prepared = preprocess_observation(observation, train=False)
        prefix, prefix_mask, kv_cache = self.model.build_prefix_cache(prepared)
        hidden = prefix[:, :IMAGE_TOKENS].to(dtype=torch.float32)
        valid = prefix_mask[:, :IMAGE_TOKENS].to(dtype=torch.bool)
        validate_feature_batch(hidden.cpu().numpy(), valid.cpu().numpy())

        batch_size = prepared.state.shape[0]
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.model.action_horizon,
                self.model.action_dim,
                device=prepared.state.device,
            )
        expected_noise = (
            batch_size,
            self.model.action_horizon,
            self.model.action_dim,
        )
        if tuple(noise.shape) != expected_noise:
            raise ValueError(f"Pi0.5 noise must have shape {expected_noise}")

        # Reuse the exact prefix pass whose image tokens are cached for RLT.
        # Pi0RL's public eval path runs the same deterministic Euler loop but
        # would rebuild the prefix and lose this one-to-one provenance.
        model_actions = noise.to(device=prepared.state.device, dtype=torch.float32)
        dt = -1.0 / self.model.num_steps
        time_value = 1.0
        while time_value >= -dt / 2:
            timestep = torch.full(
                (batch_size,),
                time_value,
                device=prepared.state.device,
                dtype=torch.float32,
            )
            suffix = self.model.run_suffix(
                prepared, model_actions, timestep, kv_cache, prefix_mask
            )
            velocity = self.model.velocity_from_suffix(suffix)
            model_actions = model_actions + dt * velocity
            time_value += dt
        reference = self.model.decode_actions(model_actions, observation.state)
        if (
            reference.shape[1:] != (REFERENCE_HORIZON, 14)
            or not torch.isfinite(reference).all()
        ):
            raise ValueError("Pi0.5 reference must be finite with shape [B, 50, 14]")
        return {"hidden": hidden, "mask": valid, "reference": reference}
