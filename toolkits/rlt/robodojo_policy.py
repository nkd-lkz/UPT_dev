# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""XPolicyLab policy API and native evaluation loops, without simulator imports."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from toolkits.rlt.robodojo_boundary import RoboDojoBoundary


class InferenceBackend(Protocol):
    """Return denormalized ARX X5 actions from a single encoded observation."""

    def infer(self, observation: dict) -> dict[str, np.ndarray]:
        """Return an ``actions`` array of shape [H, 14]."""
        ...


def validate_normalization(path: Path) -> None:
    """Require finite fourteen-dimensional native ARX X5 normalization stats."""
    data = json.loads(path.read_text())
    stats = data.get("norm_stats", {})
    for field in ("state", "actions"):
        arrays = {}
        for key in ("mean", "std", "q01", "q99"):
            values = np.asarray(stats.get(field, {}).get(key), dtype=np.float64)
            if values.shape != (14,) or not np.isfinite(values).all():
                raise ValueError(
                    f"ARX X5 requires finite 14-D {field}.{key} statistics"
                )
            arrays[key] = values
        if (arrays["std"] < 0).any() or (arrays["q01"] > arrays["q99"]).any():
            raise ValueError(f"Invalid normalization ranges: {field}")


def inspect_reference(config: dict) -> dict:
    """Check native reference imports and config without loading model parameters.

    Run this in the isolated policy interpreter with CUDA hidden and JAX set to
    CPU. The internal padded action width is distinct from the 14-D robot output.
    """
    if config["backend"] != "openpi_reference":
        raise ValueError("Reference inspection only supports openpi_reference")
    from openpi.policies import policy_config
    from openpi.training import config as training_config

    native = training_config.get_config(config["train_config_name"])
    if (
        not isinstance(native.data, training_config.LeRobotAlohaDataConfig)
        or native.data.assets.asset_id != "arx_x5_sim"
        or config["model_contract"]["normalization_id"] != "arx_x5_sim"
        or native.data.adapt_to_pi
        or not native.model.pi05
    ):
        raise ValueError("Require the native ARX X5 Pi0.5 reference configuration")
    if not 1 <= config["action_horizon"] <= native.model.action_horizon:
        raise ValueError("Execution horizon exceeds the reference action chunk")
    if native.model.action_dim < 14:
        raise ValueError("Reference model cannot represent fourteen robot actions")
    normalization = (
        Path(config["checkpoint_path"]) / "assets/arx_x5_sim/norm_stats.json"
    )
    validate_normalization(normalization)
    return {
        "train_config_name": native.name,
        "normalization_id": native.data.assets.asset_id,
        "padded_action_dim": native.model.action_dim,
        "robot_action_dim": 14,
        "model_action_horizon": native.model.action_horizon,
        "execution_horizon": config["action_horizon"],
        "joint_delta_transform": native.data.use_delta_joint_actions,
        "loader_module": policy_config.__name__,
        "parameters_loaded": False,
    }


def load_backend(config: dict) -> InferenceBackend:
    """Load an explicit reference checkpoint or a trusted local RLT factory.

    A factory is ``module:function`` accepting this configuration and returning
    an InferenceBackend. It must already implement ARX X5 normalization and
    inference; this adapter never converts a LIBERO action head.
    """
    checkpoint = Path(config["checkpoint_path"])
    norm_file = (
        checkpoint
        / "assets"
        / config["model_contract"]["normalization_id"]
        / "norm_stats.json"
    )
    if not checkpoint.is_dir() or not norm_file.is_file():
        raise ValueError("Require an exact checkpoint directory and norm_stats.json")
    validate_normalization(norm_file)
    backend = config["backend"]
    if backend == "factory":
        module, separator, name = config["backend_factory"].partition(":")
        if not separator or not module or not name:
            raise ValueError("backend_factory must be module:function")
        return getattr(importlib.import_module(module), name)(dict(config))
    if backend != "openpi_reference":
        raise ValueError("Select openpi_reference or an embodiment-compatible factory")
    inspect_reference(config)
    # Keep OpenPI and accelerator imports out of CPU contract checks.
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config as training_config

    return policy_config.create_trained_policy(
        training_config.get_config(config["train_config_name"]),
        str(checkpoint),
        norm_stats=normalize.load(norm_file.parent),
    )


class Model:
    """Implement XPolicyLab's reset/update/get lifecycle for a stateless backend.

    The WS server owns image decoding. Inputs reaching this adapter must contain
    HWC uint8 RGB arrays. Batch IDs may shrink or reorder between requests.
    Stateful memory backends need a separate per-environment state contract.
    """

    def __init__(self, model_cfg: dict, *, backend: InferenceBackend | None = None):
        if (
            model_cfg.get("action_type") != "joint"
            or model_cfg.get("env_cfg_type") != "arx_x5"
        ):
            raise ValueError("Require the native arx_x5 joint-control interface")
        self.boundary = RoboDojoBoundary(
            self._predict,
            model_contract=model_cfg["model_contract"],
            horizon=model_cfg["action_horizon"],
        )
        # Validate the embodiment before importing/loading a GPU backend.
        self.backend = backend if backend is not None else load_backend(model_cfg)
        self._single_id: int | None = None

    def _predict(self, observation: dict) -> np.ndarray:
        return self.backend.infer(observation)["actions"]

    def reset(self) -> None:
        """Discard observations; safe to call repeatedly at episode boundaries."""
        self.boundary.reset()
        self._single_id = None

    def update_obs(self, observation: dict) -> None:
        """Store one observation; only single-environment mode defaults ID to zero."""
        observation = dict(observation)
        observation.setdefault("env_idx", 0)
        self.update_obs_batch([observation])
        self._single_id = observation["env_idx"]

    def update_obs_batch(self, observations: list[dict]) -> None:
        """Replace the current active observations using explicit environment IDs."""
        self.boundary.update_obs_batch(observations)
        self._single_id = None

    def get_action(self) -> list[dict]:
        """Consume the single observation and return one native action chunk."""
        if self._single_id is None:
            raise ValueError("Call update_obs before get_action")
        index, self._single_id = self._single_id, None
        return self.get_action_batch([index])[0]

    def get_action_batch(self, env_idx_list: list[int]) -> list[list[dict]]:
        """Consume fresh observations in the caller's requested order."""
        return self.boundary.get_action_batch(env_idx_list)


def eval_one_episode(task_env: Any, model_client: Any) -> None:
    """Execute chunks in the native single-environment evaluator until termination."""
    model_client.call(func_name="reset")
    while not task_env.is_episode_end():
        model_client.call(func_name="update_obs", obs=task_env.get_obs())
        actions = model_client.call(func_name="get_action")
        if not actions:
            raise ValueError("Policy returned an empty action chunk")
        for action in actions:
            task_env.take_action(action)
            if task_env.is_episode_end():
                break


def eval_one_episode_batch(task_env: Any, model_client: Any) -> None:
    """Keep chunks attached to their IDs as native environments finish mid-chunk."""
    model_client.call(func_name="reset")
    while not task_env.is_episode_end():
        indices = task_env.get_running_env_idx_list()
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Native evaluator returned invalid active IDs")
        model_client.call(
            func_name="update_obs_batch", obs=task_env.get_obs_batch(indices)
        )
        chunks = model_client.call(func_name="get_action_batch", obs=indices)
        if len(chunks) != len(indices) or not chunks or not chunks[0]:
            raise ValueError("Policy batch does not match active environments")
        horizon = len(chunks[0])
        if any(len(chunk) != horizon for chunk in chunks):
            raise ValueError("Policy chunks must have equal execution horizons")
        for step in range(horizon):
            running = set(task_env.get_running_env_idx_list())
            active = [
                (index, chunk)
                for index, chunk in zip(indices, chunks, strict=True)
                if index in running
            ]
            if not active:
                break
            task_env.take_action_batch(
                [chunk[step] for _, chunk in active], [index for index, _ in active]
            )
            if task_env.is_episode_end():
                break
