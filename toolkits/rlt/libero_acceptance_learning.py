# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Small-head RLT_a acceptance learner with explicit TD and update accounting."""

from __future__ import annotations

import copy
import random

import torch
from torch import nn


def td_target(
    rewards: torch.Tensor,
    ticks: torch.Tensor,
    terminal: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Bootstrap after a full nonterminal chunk, not after a successful terminal."""
    return rewards + gamma**ticks * (~terminal).float() * next_q


def td_eligible(row: dict) -> bool:
    """Exclude time-budget cuts that did not execute the whole critic command."""
    return row["ticks"] == 8 or row["terminal"]


def model_state(model: nn.Module) -> dict:
    """Copy a model state onto CPU without aliasing live parameters."""
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


class AcceptanceLearner:
    """Train official RLT_a heads; change only the actor Q term between arms.

    Both arms train a diagnostic critic with the same update schedule. In the
    BC-only arm no critic output contributes to actor gradients. A single
    process owns inference and learning, so each chunk records the exact actor
    update count used to generate it. This adapter discounts by actual executed
    ticks and is an explicit training variant, not the unmodified public trainer.
    """

    def __init__(
        self,
        device: str,
        objective: str = "q_bc",
        beta: float = 1.0,
        ref_dropout: float = 0.5,
    ) -> None:
        from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import (
            ActionTokenActor,
            ActionTokenQCritic,
        )

        if objective not in ("bc_only", "q_bc") or beta <= 0:
            raise ValueError("Require bc_only/q_bc and positive BC weight")
        self.device, self.objective, self.beta = device, objective, beta
        self.actor = ActionTokenActor(
            bottleneck_dim=256,
            action_dim=7,
            chunk_len=8,
            hidden_dim=512,
            ref_dropout=ref_dropout,
            fixed_std=0.1,
            prop_dim=8,
        ).to(device)
        self.critic = ActionTokenQCritic(
            bottleneck_dim=256,
            action_dim=7,
            chunk_len=8,
            hidden_dim=512,
            prop_dim=8,
        ).to(device)
        self.target_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).eval().requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.actor_updates = self.critic_updates = 0

    @torch.no_grad()
    def command(self, row: dict, *, deterministic: bool) -> torch.Tensor:
        """Generate a command with dropout disabled for physical execution."""
        self.actor.eval()
        action, _ = self.actor(
            row["z"][None].to(self.device),
            row["ref"][None].to(self.device),
            row["prop"][None].to(self.device),
            deterministic=deterministic,
        )
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite actor command")
        return action[0].float().cpu().clamp(-1, 1)

    def update(self, replay: list[dict], *, warmup: bool = False) -> dict:
        """Update from pre-action reference labels; terminal tails remain in TD."""
        eligible = replay if warmup else [r for r in replay if td_eligible(r)]
        if not eligible:
            raise ValueError("No complete or terminal transitions for updating")
        rows = random.choices(eligible, k=128)
        batch = {
            key: torch.stack([r[key] for r in rows]).to(self.device)
            for key in ("z", "ref", "prop", "action", "next_z", "next_ref", "next_prop")
        }
        metrics = {}
        if not warmup:
            self.critic.train()
            with torch.no_grad():
                next_action, _ = self.target_actor(
                    batch["next_z"],
                    batch["next_ref"],
                    batch["next_prop"],
                    deterministic=True,
                )
                noise = (torch.randn_like(next_action) * 0.2).clamp(-0.5, 0.5)
                next_action = (next_action + noise).clamp(-1, 1)
                q1, q2 = self.target_critic(
                    batch["next_z"], next_action, batch["next_prop"]
                )
                target = td_target(
                    torch.tensor([r["reward"] for r in rows], device=self.device),
                    torch.tensor([r["ticks"] for r in rows], device=self.device),
                    torch.tensor([r["terminal"] for r in rows], device=self.device),
                    torch.minimum(q1, q2),
                )
            q1, q2 = self.critic(batch["z"], batch["action"], batch["prop"])
            critic_loss = ((q1 - target).square() + (q2 - target).square()).mean()
            if not torch.isfinite(critic_loss):
                raise FloatingPointError("Nonfinite critic loss")
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
            self.critic_opt.step()
            self.critic_updates += 1
            metrics.update(
                critic_loss=critic_loss.item(),
                q_mean=q1.mean().item(),
                target_mean=target.mean().item(),
                critic_grad=float(critic_grad),
            )
        if warmup or self.critic_updates % 2 == 0:
            self.actor.train()
            self.critic.requires_grad_(False)
            action, _ = self.actor(
                batch["z"], batch["ref"], batch["prop"], deterministic=False
            )
            # The target is a full proposal issued before execution, not a
            # retroactively constructed expert suffix after observing outcomes.
            bc = (action - batch["ref"]).square().sum((-2, -1)).mean()
            q = self.critic.q1_forward(
                batch["z"], action.clamp(-1, 1), batch["prop"]
            ).mean()
            loss = self.beta * bc
            if not warmup and self.objective == "q_bc":
                loss = loss - q
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite actor loss")
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_opt.step()
            self.critic.requires_grad_(True)
            self.actor_updates += 1
            metrics.update(
                actor_loss=loss.item(),
                bc_sum=bc.item(),
                q_actor=q.item(),
                actor_grad=float(grad),
            )
            with torch.no_grad():
                for model, target_model in (
                    (self.actor, self.target_actor),
                    (self.critic, self.target_critic),
                ):
                    for param, target_param in zip(
                        model.parameters(), target_model.parameters()
                    ):
                        target_param.lerp_(param, 0.005)
        return metrics | {
            "actor_updates": self.actor_updates,
            "critic_updates": self.critic_updates,
        }

    @torch.no_grad()
    def following(self, rows: list[dict]) -> dict:
        """Measure held-out action fidelity with inference-mode reference input."""
        if not rows:
            raise ValueError("Following validation requires whole held-out episodes")
        self.actor.eval()
        errors, gripper_errors, saturated = [], [], []
        for start in range(0, len(rows), 128):
            batch = {
                key: torch.stack([r[key] for r in rows[start : start + 128]]).to(
                    self.device
                )
                for key in ("z", "ref", "prop")
            }
            mean, _ = self.actor(
                batch["z"], batch["ref"], batch["prop"], deterministic=True
            )
            errors.append((mean - batch["ref"]).square().cpu())
            gripper_errors.append(
                ((mean[..., 6] >= 0.5) != (batch["ref"][..., 6] >= 0.5)).cpu()
            )
            saturated.append((mean.abs() > 1).cpu())
        error = torch.cat(errors)
        return {
            "mse": error.mean().item(),
            "arm_mse": error[..., :6].mean().item(),
            "gripper_mse": error[..., 6].mean().item(),
            "gripper_disagreement": torch.cat(gripper_errors).float().mean().item(),
            "saturated_fraction": torch.cat(saturated).float().mean().item(),
            "chunks": len(rows),
        }

    def publish_warmup(self) -> None:
        """Synchronize target networks once after supervised warmup."""
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def state_dict(self) -> dict:
        """Capture optimizer moments, targets and schedules alongside weights."""
        return {
            name: copy.deepcopy(getattr(self, name).state_dict())
            for name in (
                "actor",
                "critic",
                "target_actor",
                "target_critic",
                "actor_opt",
                "critic_opt",
            )
        } | {"actor_updates": self.actor_updates, "critic_updates": self.critic_updates}

    def load_state_dict(self, state: dict) -> None:
        """Restore the same warm start for either explicitly selected objective."""
        for name in (
            "actor",
            "critic",
            "target_actor",
            "target_critic",
            "actor_opt",
            "critic_opt",
        ):
            getattr(self, name).load_state_dict(state[name])
        self.actor_updates, self.critic_updates = (
            state["actor_updates"],
            state["critic_updates"],
        )
