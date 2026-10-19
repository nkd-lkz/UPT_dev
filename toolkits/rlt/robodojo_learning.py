# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""New-RLT reconstruction and 14-D ARX X5 heads in native normalized delta space."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from toolkits.rlt.robodojo_policy import validate_normalization

PROTOCOL = {
    "name": "robodojo-arx-pi05-rlt-v1",
    "feature_shape": [768, 2048],
    "action_dim": 14,
    "prop_dim": 14,
    "chunk_len": 10,
    "actor_space": "native_quantile_normalized_joint_delta_absolute_gripper",
    "execution_space": "absolute_joint_radians_gripper_01",
    "reference_dropout": 0.0,
    "memory": False,
    "intervention": False,
}


def learner_protocol(token_dim: int = 0) -> dict:
    """Return the learner protocol, extended when a grounded token is appended.

    A zero token width reproduces the baseline protocol exactly, so existing
    warmup and online checkpoints keep loading unchanged. A positive width
    declares the grounded-history extension: the token is concatenated onto
    the frozen RL token, and every checkpoint records the widened contract.
    """
    if type(token_dim) is not int or not 0 <= token_dim <= 512:
        raise ValueError("Token width must be an integer in [0, 512]")
    if token_dim == 0:
        return dict(PROTOCOL)
    return PROTOCOL | {"memory": {"kind": "grounded_v1", "token_dim": token_dim}}


class ArxNormalizer:
    """Match native Pi0.5 quantile normalization and twelve joint-delta axes."""

    def __init__(self, path: Path) -> None:
        validate_normalization(path)
        self.stats = json.loads(path.read_text())["norm_stats"]
        self.joints = np.ones(14, dtype=bool)
        self.joints[[6, 13]] = False

    def _range(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        stats = self.stats[key]
        low = np.asarray(stats["q01"], dtype=np.float32)
        span = np.asarray(stats["q99"], dtype=np.float32) - low + 1e-6
        return low, span

    @staticmethod
    def _array(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.ndim < 1 or value.shape[-1] != 14 or not np.isfinite(value).all():
            raise ValueError("Require finite fourteen-dimensional arrays")
        return value.copy()

    def state(self, state: np.ndarray) -> np.ndarray:
        """Normalize physical arm positions and commanded gripper openings."""
        low, span = self._range("state")
        return (self._array(state) - low) / span * 2 - 1

    def encode(self, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Normalize physical commands relative to the pre-chunk state."""
        values, state = self._array(actions), self._array(state)
        values[..., self.joints] -= state[..., self.joints]
        low, span = self._range("actions")
        return (values - low) / span * 2 - 1

    def decode(self, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Invert normalization, add pre-chunk joints, and clip physical grippers."""
        values, state = self._array(actions), self._array(state)
        low, span = self._range("actions")
        values = (values + 1) * 0.5 * span + low
        values[..., self.joints] += state[..., self.joints]
        values[..., [6, 13]] = np.clip(values[..., [6, 13]], 0, 1)
        return values


def load_rollout_rows(
    roots: list[Path], normalizer: ArxNormalizer
) -> list[dict[str, Any]]:
    """Load acknowledged chunks and convert physical commands once at the boundary."""
    rows = []
    seen = set()
    for root in roots:
        for episode_path in sorted(root.glob("episode-*/episode.json")):
            episode = json.loads(episode_path.read_text())
            layout = episode.get("layout", {})
            identity = (layout.get("id"), layout.get("sha256"))
            if identity in seen or layout.get("role") not in {"train", "development"}:
                continue
            records = episode.get("records")
            if not episode.get("complete") or not isinstance(records, list):
                raise ValueError(f"Incomplete collection episode: {episode_path}")
            if not records:
                raise ValueError(f"Empty collection episode: {episode_path}")
            final = records[-1]
            if not final.get("terminated") or float(final.get("reward", 0)) <= 0:
                continue
            seen.add(identity)
            for index, record in enumerate(records):
                cache = Path(record["cache_file"])
                with np.load(cache) as payload:
                    hidden = payload["hidden"]
                    state = payload["state"]
                    reference_physical = payload["reference"].astype(np.float32)
                if hidden.shape != (768, 2048) or not np.isfinite(hidden).all():
                    raise ValueError(f"Invalid frozen features: {cache}")
                reference_physical[:, [6, 13]] = np.clip(
                    reference_physical[:, [6, 13]], 0, 1
                )
                reference = normalizer.encode(reference_physical, state)
                ticks = record["ticks"]
                executed = normalizer.encode(record["executed_physical"], state)
                if (
                    type(ticks) is not int
                    or executed.shape != (ticks, 14)
                    or not np.allclose(executed, reference[:ticks], atol=1e-6, rtol=0)
                ):
                    raise ValueError(
                        f"Acknowledged commands differ from issued chunk: {cache}"
                    )
                rows.append(
                    {
                        "episode": identity,
                        "role": layout["role"],
                        "index": index,
                        "hidden_path": cache,
                        "ref": reference,
                        "prop": normalizer.state(state),
                        "action": reference.copy(),
                        "executed_action": executed,
                        "reward": float(record["reward"]),
                        "ticks": ticks,
                        "terminated": bool(record["terminated"]),
                        "truncated": bool(record["truncated"]),
                    }
                )
    if not rows:
        raise ValueError("No successful complete rollout chunks found")
    for index, row in enumerate(rows):
        same_episode_next = (
            index + 1 < len(rows) and rows[index + 1]["episode"] == row["episode"]
        )
        row["td_eligible"] = row["terminated"] or same_episode_next
    return rows


def load_hidden(row: dict[str, Any]) -> np.ndarray:
    """Read one immutable hidden-state cache after metadata validation."""
    with np.load(row["hidden_path"]) as payload:
        hidden = np.asarray(payload["hidden"], dtype=np.float32)
    if hidden.shape != (768, 2048) or not np.isfinite(hidden).all():
        raise ValueError("Invalid cached VLA features")
    return hidden


def make_encoder(device: str = "cuda") -> torch.nn.Module:
    """Build the pinned new-RLT cross-attention encoder/decoder, preserving 2048 D."""
    from AlphaBrain.training.reinforcement_learning.algos.RLT.encoder_decoder import (
        RLTokenEncoderDecoder,
    )

    return RLTokenEncoderDecoder(
        hidden_dim=2048, num_heads=8, encoder_layers=2, decoder_layers=2
    ).to(device)


class ArxLearner:
    """Adapt AlphaBrain's shared heads without reusing the seven-dimensional head.

    Actor outputs a full normalized chunk, not a structural residual. Online
    TD uses the issued chunk and checks its executed prefix against the ledger.
    Terminal tails bootstrap zero; nonterminal partial time-budget cuts are
    excluded. BC-only and Q+BC differ only in the actor Q term.
    """

    def __init__(
        self,
        normalizer: ArxNormalizer,
        *,
        device: str = "cuda",
        objective: str = "q_bc",
        exploration_std: float = 0.01,
        target_noise_std: float = 0.02,
        target_noise_clip: float = 0.05,
        reference_anchor_scale: float = 1.0,
        token_dim: int = 0,
    ) -> None:
        from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import (
            ActionTokenActor,
            ActionTokenQCritic,
        )

        if objective not in {"bc_only", "q_bc"}:
            raise ValueError("Unknown objective")
        if not 0 <= exploration_std <= 0.1:
            raise ValueError("Exploration std must be in [0, 0.1]")
        if not 0 <= target_noise_std <= target_noise_clip <= 0.5:
            raise ValueError("Invalid target policy smoothing")
        if not 0 <= reference_anchor_scale <= 1:
            raise ValueError("Reference anchor scale must be in [0, 1]")
        self.device, self.objective, self.normalizer = device, objective, normalizer
        self.exploration_std = float(exploration_std)
        self.target_noise_std = float(target_noise_std)
        self.target_noise_clip = float(target_noise_clip)
        self.reference_anchor_scale = float(reference_anchor_scale)
        self.protocol = learner_protocol(token_dim)
        # A grounded history token rides appended to the frozen 2048-D RL
        # token, so widening the bottleneck is the only architectural change.
        self.token_dim = token_dim
        common = {
            "bottleneck_dim": 2048 + token_dim,
            "action_dim": 14,
            "chunk_len": 10,
            "hidden_dim": 512,
            "prop_dim": 14,
        }
        self.actor = ActionTokenActor(
            **common, ref_dropout=0.0, fixed_std=exploration_std
        ).to(device)
        self.critic = ActionTokenQCritic(**common).to(device)
        self.target_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).eval().requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.actor_updates = self.critic_updates = 0

    def _executed(self, action: torch.Tensor) -> torch.Tensor:
        # Differentiable native gripper clipping in normalized coordinates.
        low, span = self.normalizer._range("actions")
        lower = torch.as_tensor(-low / span * 2 - 1, device=action.device)
        upper = torch.as_tensor((1 - low) / span * 2 - 1, device=action.device)
        result = action.clone()
        result[..., [6, 13]] = torch.maximum(
            torch.minimum(result[..., [6, 13]], upper[[6, 13]]), lower[[6, 13]]
        )
        return result

    def _anchored(self, action: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Bound learned corrections around the frozen Pi0.5 reference chunk."""
        return reference + self.reference_anchor_scale * (action - reference)

    def _batch(self, rows: list[dict], names: tuple[str, ...]) -> dict:
        if not rows:
            raise ValueError("Empty learning batch")
        batch = {
            name: torch.stack(
                [torch.as_tensor(row[name], dtype=torch.float32) for row in rows]
            ).to(self.device)
            for name in names
        }
        for name, value in batch.items():
            shape = (
                (len(rows), 10, 14)
                if "ref" in name or name == "action"
                else (len(rows), 2048 + self.token_dim)
                if "z" in name
                else (len(rows), 14)
            )
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid batch field {name}")
        return batch

    def update(self, rows: list[dict], *, warmup: bool = False) -> dict:
        """Take a BC step, or twin-Q TD step plus delayed Q+BC/BC actor update."""
        names = (
            ("z", "ref", "prop")
            if warmup
            else ("z", "ref", "prop", "action", "next_z", "next_ref", "next_prop")
        )
        batch = self._batch(rows, names)
        metrics = {}
        if not warmup:
            for row in rows:
                ticks = row["ticks"]
                if (
                    type(ticks) is not int
                    or not 1 <= ticks <= 10
                    or (ticks < 10 and not row["terminated"])
                ):
                    raise ValueError(
                        "TD requires a complete chunk or a terminal prefix"
                    )
                executed = np.asarray(row["executed_action"])
                if executed.shape != (ticks, 14):
                    raise ValueError(
                        "Critic action does not match the acknowledged prefix"
                    )
                # Execution-side perturbations are environment dynamics: the
                # critic conditions on the issued decision while the client
                # verified the acknowledged stream against the applied one.
                if "perturbation" not in row and not np.allclose(
                    executed, np.asarray(row["action"])[:ticks], atol=1e-6, rtol=0
                ):
                    raise ValueError(
                        "Critic action does not match the acknowledged prefix"
                    )
            with torch.no_grad():
                action, _ = self.target_actor(
                    batch["next_z"],
                    batch["next_ref"],
                    batch["next_prop"],
                    deterministic=True,
                )
                action = self._anchored(action, batch["next_ref"])
                action = self._executed(
                    action
                    + (torch.randn_like(action) * self.target_noise_std).clamp(
                        -self.target_noise_clip, self.target_noise_clip
                    )
                )
                q1, q2 = self.target_critic(batch["next_z"], action, batch["next_prop"])
                rewards = torch.tensor([r["reward"] for r in rows], device=self.device)
                terminal = torch.tensor(
                    [r["terminated"] for r in rows],
                    device=self.device,
                    dtype=torch.bool,
                )
                ticks = torch.tensor([r["ticks"] for r in rows], device=self.device)
                target = rewards + 0.99**ticks * (~terminal).float() * torch.minimum(
                    q1, q2
                )
            q1, q2 = self.critic(batch["z"], batch["action"], batch["prop"])
            loss = ((q1 - target).square() + (q2 - target).square()).mean()
            self._optimize(loss, self.critic_opt, self.critic)
            self.critic_updates += 1
            metrics["critic_loss"] = loss.item()
        if warmup or self.critic_updates % 2 == 0:
            self.critic.requires_grad_(False)
            try:
                # Deterministic BC avoids adding irreducible sampling error in warmup.
                raw_action, _ = self.actor(
                    batch["z"], batch["ref"], batch["prop"], deterministic=warmup
                )
                action = self._anchored(raw_action, batch["ref"])
                # Keep BC in the actor's native output space. Computing it after
                # anchoring would shrink its gradient by scale squared while the
                # Q gradient shrinks only linearly, weakening the safety prior.
                bc = (raw_action - batch["ref"]).square().sum((-2, -1)).mean()
                loss = bc
                if not warmup and self.objective == "q_bc":
                    loss = (
                        loss
                        - self.critic.q1_forward(
                            batch["z"], self._executed(action), batch["prop"]
                        ).mean()
                    )
                self._optimize(loss, self.actor_opt, self.actor)
            finally:
                self.critic.requires_grad_(True)
            self.actor_updates += 1
            metrics.update(bc_sum=bc.item(), actor_loss=loss.item())
            with torch.no_grad():
                for model, target_model in (
                    (self.actor, self.target_actor),
                    (self.critic, self.target_critic),
                ):
                    for param, target_param in zip(
                        model.parameters(), target_model.parameters(), strict=True
                    ):
                        target_param.lerp_(param, 0.005)
        return metrics | {
            "actor_updates": self.actor_updates,
            "critic_updates": self.critic_updates,
        }

    @staticmethod
    def _optimize(
        loss: torch.Tensor, optimizer: torch.optim.Optimizer, model: torch.nn.Module
    ) -> None:
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()

    @torch.no_grad()
    def predict(self, row: dict, *, deterministic: bool = True) -> np.ndarray:
        """Return a normalized chunk; physical decoding clips its grippers."""
        batch = self._batch([row], ("z", "ref", "prop"))
        self.actor.train(not deterministic)
        action, _ = self.actor(
            batch["z"],
            batch["ref"],
            batch["prop"],
            deterministic=deterministic,
        )
        action = self._anchored(action, batch["ref"])
        return action[0].cpu().numpy()

    def publish_warmup(self) -> None:
        """Copy the completed BC initialization into both target networks."""
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    @torch.no_grad()
    def following(self, rows: list[dict]) -> dict[str, float]:
        """Measure deterministic normalized-action imitation on held-out rows."""
        if not rows:
            raise ValueError("Require held-out rows")
        errors = []
        for row in rows:
            prediction = self.predict(row)
            errors.append(np.square(prediction - np.asarray(row["ref"])))
        stacked = np.stack(errors)
        return {
            "mse": float(stacked.mean()),
            "max_abs": float(np.sqrt(stacked).max()),
        }

    def state_dict(self) -> dict:
        """Persist heads, targets, optimizers and update counters for exact resume."""
        return {
            "protocol": dict(self.protocol),
            **{
                name: getattr(self, name).state_dict()
                for name in (
                    "actor",
                    "critic",
                    "target_actor",
                    "target_critic",
                    "actor_opt",
                    "critic_opt",
                )
            },
            "actor_updates": self.actor_updates,
            "critic_updates": self.critic_updates,
            "hyperparameters": {
                "objective": self.objective,
                "exploration_std": self.exploration_std,
                "target_noise_std": self.target_noise_std,
                "target_noise_clip": self.target_noise_clip,
                "reference_anchor_scale": self.reference_anchor_scale,
            },
        }

    def load_state_dict(self, payload: dict) -> None:
        """Restore an exact compatible learner state including optimizer slots."""
        if payload.get("protocol") != self.protocol:
            raise ValueError("Incompatible RoboDojo learner protocol")
        for name in (
            "actor",
            "critic",
            "target_actor",
            "target_critic",
            "actor_opt",
            "critic_opt",
        ):
            getattr(self, name).load_state_dict(payload[name])
        self.actor_updates = int(payload["actor_updates"])
        self.critic_updates = int(payload["critic_updates"])

    def configure_online_noise(
        self,
        *,
        exploration_std: float,
        target_noise_std: float,
        target_noise_clip: float,
    ) -> None:
        """Apply an explicitly recorded action-space noise scale after warmup."""
        if not 0 <= exploration_std <= 0.1:
            raise ValueError("Exploration std must be in [0, 0.1]")
        if not 0 <= target_noise_std <= target_noise_clip <= 0.5:
            raise ValueError("Invalid target policy smoothing")
        self.exploration_std = float(exploration_std)
        self.target_noise_std = float(target_noise_std)
        self.target_noise_clip = float(target_noise_clip)
        self.actor.fixed_std.fill_(exploration_std)

    def configure_reference_anchor(self, scale: float) -> None:
        """Set the fraction of the learned correction executed around reference."""
        if not 0 <= scale <= 1:
            raise ValueError("Reference anchor scale must be in [0, 1]")
        self.reference_anchor_scale = float(scale)
