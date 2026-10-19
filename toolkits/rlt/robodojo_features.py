# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen native ARX X5 Pi0.5 prefix features alongside unchanged flow inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np

from toolkits.rlt.robodojo_policy import load_backend


def _prefix(model, observation):
    # Loaded lazily inside the isolated native JAX interpreter.
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask

    observation = model_api.preprocess_observation(None, observation, train=False)
    tokens, mask, autoregression = model.embed_prefix(observation)
    (hidden, _), _ = model.PaliGemma.llm(
        [tokens, None],
        mask=make_attn_mask(mask, autoregression),
        positions=jnp.cumsum(mask, axis=1) - 1,
    )
    # Three 224x224 views, 256 tokens each. Their final hidden states have
    # already attended to the valid language/state-tokenized prefix.
    return hidden[:, :768], mask[:, :768]


class FrozenArxFeatures:
    """Extract image-position VLM features without replacing native action sampling.

    Uses the pinned native Policy's preprocessing and frozen NNX model. A
    separate prefix pass costs extra compute but never modifies its KV cache or
    sampler. Callers supply explicit noise seeds for reproducible reference labels.
    """

    def __init__(self, config: dict) -> None:
        from openpi.shared.nnx_utils import module_jit

        self.policy = load_backend(config)
        if self.policy._is_pytorch_model or not self.policy._model.pi05:
            raise ValueError("Require the pinned native JAX Pi0.5 checkpoint")
        self._extract = module_jit(MethodType(_prefix, self.policy._model))

    def infer(
        self, observation: dict, *, seed: int, check_parity: bool = False
    ) -> dict:
        """Return 768x2048 frozen features and the full physical 50x14 reference."""
        import jax
        import jax.numpy as jnp
        from openpi.models.model import Observation

        if set(observation) != {"state", "images", "prompt"}:
            raise ValueError("Only filtered state, RGB images and prompt may enter VLA")
        model = self.policy._model
        noise = np.random.default_rng(seed).standard_normal(
            (model.action_horizon, model.action_dim), dtype=np.float32
        )
        before = (
            self.policy.infer(observation, noise=noise)["actions"]
            if check_parity
            else None
        )
        inputs = self.policy._input_transform(jax.tree.map(lambda x: x, observation))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        hidden, mask = self._extract(Observation.from_dict(batch))
        hidden, mask = np.asarray(hidden[0], dtype=np.float32), np.asarray(mask[0])
        if (
            hidden.shape != (768, 2048)
            or not mask.all()
            or not np.isfinite(hidden).all()
        ):
            raise ValueError(
                "Expected finite final VLM features for all three real views"
            )
        reference = np.asarray(self.policy.infer(observation, noise=noise)["actions"])
        if (
            reference.shape != (model.action_horizon, 14)
            or not np.isfinite(reference).all()
        ):
            raise ValueError("Invalid physical ARX X5 reference output")
        parity = None if before is None else float(np.max(np.abs(before - reference)))
        if parity is not None and parity != 0:
            raise RuntimeError(
                f"Feature extraction changed fixed-noise actions: {parity}"
            )
        return {"hidden": hidden, "reference": reference, "parity_max_abs": parity}


def main() -> None:
    """Check real checkpoint feature/action parity on declared synthetic inputs."""
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"state": "loading", "training": False, "input_kind": "synthetic"}

    def save():
        (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    save()
    try:
        config = yaml.safe_load(args.config.read_text())["policy"]
        backend = FrozenArxFeatures(config)
        rng = np.random.default_rng(130013)
        observation = {
            "state": np.zeros(14, dtype=np.float32),
            "images": {
                key: rng.integers(0, 256, (3, 224, 224), dtype=np.uint8)
                for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            },
            "prompt": "Stack the three bowls together.",
        }
        result = backend.infer(observation, seed=13, check_parity=True)
        changed = {**observation, "prompt": "Move the left bowl to the right."}
        second = backend.infer(changed, seed=13, check_parity=True)
        delta = float(np.max(np.abs(second["hidden"] - result["hidden"])))
        if delta <= 0:
            raise RuntimeError(
                "Image-position hidden states did not respond to language"
            )
        np.savez(
            args.output / "synthetic.npz",
            hidden=result["hidden"],
            reference=result["reference"],
        )
        status.update(
            state="passed",
            feature_shape=list(result["hidden"].shape),
            action_shape=list(result["reference"].shape),
            parity_max_abs=result["parity_max_abs"],
            changed_prompt_parity_max_abs=second["parity_max_abs"],
            language_feature_max_abs_delta=delta,
            training_eligible=False,
        )
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
