# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Single-step RLT AC learner for one remotely operated simulator."""

from __future__ import annotations

import copy
import math
import os
import random
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy

from .online_settings import validate_config


class OnlineLearner:
    """Own bounded replay, RLT actor/twin-Q updates, and published weights.

    This smoke uses h=1 with the frozen Stage1 ten-step reference. No unexecuted
    chunk tails enter TD targets. It is not weight-compatible with an h=10 head.
    All methods must be called by the server's single model-owner thread.
    """

    def __init__(self, config: dict, device: str = "cpu") -> None:
        self.cfg = validate_config(config)
        c = self.cfg
        self.device = torch.device(device)
        self.rng = random.Random(c["seed"])
        torch.manual_seed(c["seed"])
        self.model = RLTMLPPolicy(
            z_dim=c["z_dim"],
            proprio_dim=c["proprio_dim"],
            action_dim=c["action_dim"],
            num_action_chunks=1,
            ref_num_action_chunks=c["reference_horizon"],
        ).to(self.device)
        self.target = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.published = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.actor_params = list(self.model.backbone.parameters()) + list(
            self.model.actor_mean.parameters()
        )
        self.critic_params = list(self.model.q_head.parameters())
        self.actor_optim = torch.optim.Adam(self.actor_params, lr=c["lr"])
        self.critic_optim = torch.optim.Adam(self.critic_params, lr=c["lr"])
        self.replay: deque[dict] = deque(maxlen=c["capacity"])
        self.demos: deque[dict] = deque(maxlen=c["capacity"])
        self.update_step = self.actor_updates = self.version = 0
        self.accepted = self.human_accepted = 0
        self.approved_accepted = 0
        self.last_metrics: dict[str, float] = {}
        self.published_bc_loss: float | None = None

    def features(self, values: dict) -> dict[str, torch.Tensor]:
        """Validate frozen features and store owned, float32 CPU tensors."""
        shapes = {
            "z_rl": (1, self.cfg["z_dim"]),
            "proprio": (1, self.cfg["proprio_dim"]),
            "ref_chunk": (1, self.cfg["reference_horizon"], self.cfg["action_dim"]),
        }
        result = {}
        for key, shape in shapes.items():
            value = torch.as_tensor(values[key]).detach().to("cpu", torch.float32)
            if value.numel() != math.prod(shape) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {key} feature")
            result[key] = value.reshape(shape).clone()
        return result

    def observe(self, item: dict[str, Any]) -> dict[str, float]:
        """Insert one executed transition, then run at most one critic update."""
        action = torch.as_tensor(item["action"], dtype=torch.float32).reshape(-1)
        if (
            action.shape != (self.cfg["action_dim"],)
            or not torch.isfinite(action).all()
        ):
            raise ValueError("Invalid executed action")
        if (action.abs() > 1.00001).any():
            raise ValueError("Executed action outside normalized range")
        record = {
            "obs": self.features(item["obs"]),
            "next_obs": self.features(item["next_obs"]),
            "action": action.clone(),
            "reward": float(item["reward"]),
            "terminated": bool(item["terminated"]),
            "truncated": bool(item["truncated"]),
            "human": bool(item["human"]),
            "quality": item.get("quality", "unreviewed"),
        }
        if record["quality"] not in {"policy", "approved", "rejected", "unreviewed"}:
            raise ValueError("Unknown correction quality")
        if record["quality"] == "approved" and not record["human"]:
            raise ValueError("Only human corrections can be approved")
        if not math.isfinite(record["reward"]):
            raise ValueError("Invalid reward")
        self.replay.append(record)
        self.accepted += 1
        if record["human"]:
            self.human_accepted += 1
            if record["quality"] == "approved":
                self.demos.append(record)
                self.approved_accepted += 1
        if (
            len(self.replay) >= self.cfg["min_replay"]
            and self.update_step < self.cfg["max_updates"]
        ):
            self.last_metrics = self._update()
        return self.status()

    def _update(self) -> dict[str, float]:
        c = self.cfg
        demo_n = round(c["batch_size"] * c["demo_ratio"]) if self.demos else 0
        records = self.rng.choices(list(self.replay), k=c["batch_size"] - demo_n)
        records += self.rng.choices(list(self.demos), k=demo_n) if demo_n else []
        obs = {
            k: torch.cat([r["obs"][k] for r in records]).to(self.device)
            for k in records[0]["obs"]
        }
        next_obs = {
            k: torch.cat([r["next_obs"][k] for r in records]).to(self.device)
            for k in obs
        }
        actions = torch.stack([r["action"] for r in records]).to(self.device)
        rewards = torch.tensor([[r["reward"]] for r in records], device=self.device)
        done = torch.tensor(
            [
                [r["terminated"] or (r["truncated"] and not c["bootstrap_truncation"])]
                for r in records
            ],
            device=self.device,
        )
        human = torch.tensor([[r["human"]] for r in records], device=self.device)
        approved = torch.tensor(
            [[r["human"] and r["quality"] == "approved"] for r in records],
            device=self.device,
        )
        bc_mask = approved | ~human
        with torch.no_grad():
            next_actions, _, _ = self.model.sac_forward(next_obs)
            next_q = (
                self.target.sac_q_forward(next_obs, next_actions)
                .min(-1, keepdim=True)
                .values
            )
            target = rewards + c["gamma"] * (~done) * next_q
        q = self.model.sac_q_forward(obs, actions)
        critic_loss = F.mse_loss(q, target.expand_as(q))
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_params, 10.0, error_if_nonfinite=True
        )
        self.critic_optim.step()
        metrics = {
            "critic_loss": float(critic_loss.detach()),
            "demo_sample_ratio": demo_n / len(records),
        }
        if self.update_step % c["critic_actor_ratio"] == 0:
            self.model.q_head.requires_grad_(False)
            try:
                pi, _, _ = self.model.sac_forward(
                    obs,
                    apply_reference_dropout=True,
                    reference_dropout_prob=c["reference_dropout"],
                )
                bc_target = torch.where(approved, actions, obs["ref_chunk"][:, 0])
                bc_loss = (
                    F.mse_loss(pi, bc_target, reduction="none") * bc_mask
                ).sum() / (bc_mask.sum().clamp_min(1) * c["action_dim"])
                actor_loss = (
                    c["bc_weight"] * bc_loss
                    - c["q_weight"] * self.model.sac_q_forward(obs, pi)[:, 0].mean()
                )
                self.actor_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_params, 10.0, error_if_nonfinite=True
                )
                self.actor_optim.step()
                self.actor_updates += 1
                metrics.update(
                    actor_loss=float(actor_loss.detach()),
                    bc_loss=float(bc_loss.detach()),
                    human_batch_ratio=float(human.float().mean()),
                    approved_batch_ratio=float(approved.float().mean()),
                    bc_eligible_ratio=float(bc_mask.float().mean()),
                )
            finally:
                self.model.q_head.requires_grad_(True)
        with torch.no_grad():
            for p, target_p in zip(
                self.model.parameters(), self.target.parameters(), strict=True
            ):
                target_p.lerp_(p, c["tau"])
        self.update_step += 1
        if self.update_step % c["publish_interval"] == 0:
            self.published.load_state_dict(self.model.state_dict())
            self.version = self.update_step
            with torch.no_grad():
                prediction, _, _ = self.published.sac_forward(obs, deterministic=True)
                target = torch.where(approved, actions, obs["ref_chunk"][:, 0])
                self.published_bc_loss = (
                    float(
                        ((prediction - target).square() * bc_mask).sum()
                        / (bc_mask.sum() * c["action_dim"])
                    )
                    if bc_mask.any()
                    else None
                )
        return metrics

    def actor_ready(self) -> bool:
        """Gate deployment on warmup and optional published-batch imitation error.

        This training-batch diagnostic is not a held-out success or safety test.
        """
        limit = self.cfg.get("actor_max_bc_loss")
        return self.version >= self.cfg["actor_after_updates"] and (
            limit is None
            or (self.published_bc_loss is not None and self.published_bc_loss <= limit)
        )

    def predict(self, features: dict) -> tuple[torch.Tensor, str]:
        """Read a completed actor snapshot, or reference actions during warmup."""
        obs = {k: v.to(self.device) for k, v in self.features(features).items()}
        with torch.inference_mode():
            if not self.actor_ready():
                return obs["ref_chunk"][0, :1].cpu(), "reference"
            action, _, _ = self.published.sac_forward(obs, deterministic=True)
            return action.reshape(1, self.cfg["action_dim"]).cpu(), "actor"

    def status(self) -> dict[str, float]:
        """Return observable collection, update, and publication counters."""
        return {
            **self.last_metrics,
            "accepted": self.accepted,
            "human_accepted": self.human_accepted,
            "approved_accepted": self.approved_accepted,
            "replay_size": len(self.replay),
            "demo_size": len(self.demos),
            "update_step": self.update_step,
            "actor_updates": self.actor_updates,
            "policy_version": self.version,
            "actor_ready": self.actor_ready(),
            "update_budget_exhausted": self.update_step >= self.cfg["max_updates"],
            "published_bc_loss": self.published_bc_loss,
        }

    def save(self, path: Path, metadata: dict) -> None:
        """Atomically save networks, optimizers, replay, RNG and RPC progress."""
        state = {
            "schema": 2,
            "config": self.cfg,
            "metadata": metadata,
            "model": self.model.state_dict(),
            "target": self.target.state_dict(),
            "published": self.published.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "replay": list(self.replay),
            "demos": list(self.demos),
            "rng": self.rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
            "counters": {
                k: getattr(self, k)
                for k in (
                    "update_step",
                    "actor_updates",
                    "version",
                    "accepted",
                    "human_accepted",
                    "approved_accepted",
                )
            },
            "published_bc_loss": self.published_bc_loss,
            "last_metrics": self.last_metrics,
        }
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        if os.name == "posix":
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def load(self, path: Path) -> dict:
        """Resume a trusted compatible checkpoint, including optimizer moments."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["schema"] != 2 or state["config"] != self.cfg:
            raise ValueError(
                "Requires schema-2 quality-labelled checkpoint with matching config; start a fresh run for legacy data"
            )
        self.replay.clear()
        self.demos.clear()
        for name in ("model", "target", "published", "actor_optim", "critic_optim"):
            getattr(self, name).load_state_dict(state[name])
        self.replay.extend(state["replay"])
        self.demos.extend(state["demos"])
        for key, value in state["counters"].items():
            setattr(self, key, value)
        self.published_bc_loss = state.get("published_bc_loss")
        self.last_metrics = state.get("last_metrics", {})
        self.rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        return state["metadata"]
