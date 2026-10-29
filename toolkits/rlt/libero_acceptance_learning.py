# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Shared AlphaBrain small-head learner with explicit TD and update accounting."""

from __future__ import annotations

import copy
import random

import torch
from torch import nn


class _GatedActor(nn.Module):
    """Admit an appended physical channel from an exact zero-channel start."""

    def __init__(self, actor: nn.Module, base_dim: int) -> None:
        super().__init__()
        self.actor = actor
        self.base_dim = base_dim
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, token: torch.Tensor, *args, **kwargs):
        if token.shape[-1] != 2 * self.base_dim:
            raise ValueError("Gated actor requires equal base and physical channels")
        base, physical = token.split(self.base_dim, dim=-1)
        token = torch.cat((base, self.physical_gate() * physical), dim=-1)
        return self.actor(token, *args, **kwargs)

    def physical_gate(self) -> torch.Tensor:
        """Return the bounded scalar applied to the physical channel."""
        return self.gate_logit.tanh()


class _ResponseAuxiliaryCritic(nn.Module):
    """Share each Q trunk with a short-horizon execution-response decoder."""

    def __init__(self, critic: nn.Module, response_dim: int) -> None:
        super().__init__()
        if not hasattr(critic, "q1") or not hasattr(critic, "q2"):
            raise ValueError("Response auxiliary loss requires a twin MLP critic")
        self.critic = critic
        hidden_dim = critic.q1[-1].in_features
        self.response1 = nn.Linear(hidden_dim, response_dim)
        self.response2 = nn.Linear(hidden_dim, response_dim)

    def _input(
        self,
        token: torch.Tensor,
        action: torch.Tensor,
        prop: torch.Tensor,
    ) -> torch.Tensor:
        if token.dim() == 3:
            token = token.squeeze(1)
        return torch.cat((token, prop, action.flatten(1)), dim=-1)

    def forward(self, token, action, prop):
        return self.critic(token, action, prop)

    def q1_forward(self, token, action, prop):
        return self.critic.q1_forward(token, action, prop)

    def response(self, token, action, prop) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict normalized measured state change from both Q trunks."""
        value = self._input(token, action, prop)
        return self.response1(self.critic.q1[:-1](value)), self.response2(
            self.critic.q2[:-1](value)
        )


class _ConsequenceDuelingCritic(nn.Module):
    """Route every action-dependent Q difference through physical outcomes.

    The first half of the conditioned token is the frozen semantic state and
    the second half is the predicted candidate-minus-reference consequence.
    A state-value branch sees only the former.  The advantage branch sees both,
    but is centered at a zero consequence so the reference action has exactly
    zero learned advantage.
    """

    def __init__(self, token_dim: int, prop_dim: int = 8, hidden_dim: int = 512):
        super().__init__()
        if token_dim % 2:
            raise ValueError("Consequence-dueling critic needs two equal channels")
        self.base_dim = token_dim // 2
        value_input = self.base_dim + prop_dim
        advantage_input = token_dim + prop_dim

        def mlp(width: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(width, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.v1, self.v2 = mlp(value_input), mlp(value_input)
        self.a1, self.a2 = mlp(advantage_input), mlp(advantage_input)

    def _inputs(
        self, token: torch.Tensor, prop: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token.dim() == 3:
            token = token.squeeze(1)
        if token.ndim != 2 or token.shape[-1] != 2 * self.base_dim:
            raise ValueError("Invalid consequence-dueling token")
        base, consequence = token.split(self.base_dim, dim=-1)
        value_input = torch.cat((base, prop), dim=-1)
        advantage_input = torch.cat((base, consequence, prop), dim=-1)
        zero_input = torch.cat((base, torch.zeros_like(consequence), prop), dim=-1)
        return value_input, advantage_input, zero_input

    def _q(self, token, prop, value, advantage):
        value_input, advantage_input, zero_input = self._inputs(token, prop)
        return (
            value(value_input) + advantage(advantage_input) - advantage(zero_input)
        ).squeeze(-1)

    def forward(self, token, action, prop):
        del action
        return self._q(token, prop, self.v1, self.a1), self._q(
            token, prop, self.v2, self.a2
        )

    def q1_forward(self, token, action, prop):
        del action
        return self._q(token, prop, self.v1, self.a1)


def td_target(
    rewards: torch.Tensor,
    ticks: torch.Tensor,
    terminal: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Bootstrap after a full nonterminal chunk, not after a successful terminal."""
    shapes = {tuple(value.shape) for value in (rewards, ticks, terminal, next_q)}
    if len(shapes) != 1 or rewards.ndim != 1:
        raise ValueError("TD inputs must share one flat batch shape")
    return rewards + gamma**ticks * (~terminal).float() * next_q


def td_eligible(row: dict, chunk_len: int = 8) -> bool:
    """Exclude time-budget cuts that did not execute the whole critic command."""
    return row["ticks"] == chunk_len or row["terminal"]


def model_state(model: nn.Module) -> dict:
    """Copy a model state onto CPU without aliasing live parameters."""
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def cross_episode_response_targets(rows: list[dict]) -> dict[int, torch.Tensor]:
    """Assign each replay row a measured response from another episode."""
    episodes, current = [], []
    for row in rows:
        current.append(row)
        if row["terminal"]:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    if len(episodes) < 2:
        raise ValueError("Shuffled response supervision needs two episodes")
    targets = {}
    for episode_index, episode in enumerate(episodes):
        other = episodes[(episode_index + 1) % len(episodes)]
        for row_index, row in enumerate(episode):
            source = other[row_index % len(other)]
            targets[id(row)] = source["next_prop"] - source["prop"]
    return targets


class AcceptanceLearner:
    """Train AlphaBrain's shared heads; change only the actor Q term between arms.

    Both arms train a diagnostic critic with the same update schedule. In the
    BC-only arm no critic output contributes to actor gradients. A single
    process owns inference and learning, so each chunk records the exact actor
    update count used to generate it. This adapter discounts by actual executed
    ticks and is an explicit training variant, not the unmodified public trainer.
    Defaults retain the Qwen/RLT_a protocol; Pi0.5 callers declare token width,
    chunk length and gripper threshold explicitly.
    """

    def __init__(
        self,
        device: str,
        objective: str = "q_bc",
        beta: float = 1.0,
        ref_dropout: float = 0.5,
        *,
        bottleneck_dim: int = 256,
        chunk_len: int = 8,
        gripper_threshold: float = 0.5,
        critic_conditioner: nn.Module | None = None,
        critic_conditioning: str = "latent",
        correction_limit: float | None = None,
        correction_axes: tuple[int, ...] | None = None,
        residual_output: bool = False,
        critic_bottleneck_dim: int | None = None,
        actor_bottleneck_dim: int | None = None,
        actor_physical_gate: bool = False,
        candidate_selection_count: int = 0,
        candidate_selection_margin: float = 0.0,
        candidate_uncertainty_penalty: float = 0.0,
        critic_response_aux_weight: float = 0.0,
        critic_response_aux_shuffled: bool = False,
        critic_consequence_dueling: bool = False,
    ) -> None:
        from AlphaBrain.training.reinforcement_learning.algos.RLT_a.action_token_actor_critic import (
            ActionTokenActor,
            ActionTokenQCritic,
        )

        if objective not in ("bc_only", "q_bc") or beta <= 0:
            raise ValueError("Require bc_only/q_bc and positive BC weight")
        self.device, self.objective, self.beta = device, objective, beta
        if bottleneck_dim < 1 or chunk_len < 1:
            raise ValueError("Require positive token width and chunk length")
        critic_bottleneck_dim = critic_bottleneck_dim or bottleneck_dim
        actor_bottleneck_dim = actor_bottleneck_dim or bottleneck_dim
        if critic_bottleneck_dim < 1:
            raise ValueError("Require a positive critic token width")
        if actor_bottleneck_dim < 1:
            raise ValueError("Require a positive actor token width")
        if actor_physical_gate and actor_bottleneck_dim != 2 * bottleneck_dim:
            raise ValueError(
                "Gated actor requires one physical channel matching the base width"
            )
        if critic_conditioner is None and critic_bottleneck_dim != bottleneck_dim:
            raise ValueError("A distinct critic token width requires a conditioner")
        if (critic_conditioner is None) != (critic_conditioning == "latent"):
            raise ValueError(
                "Non-latent critic conditioning requires its explicit module"
            )
        if correction_limit is not None and not 0 < correction_limit <= 2:
            raise ValueError("Correction limit must be in (0, 2]")
        if residual_output and correction_limit is None:
            raise ValueError("Residual output requires a correction limit")
        if correction_axes is not None:
            correction_axes = tuple(correction_axes)
            if correction_limit is None:
                raise ValueError("Correction axes require a correction limit")
            if (
                not correction_axes
                or len(set(correction_axes)) != len(correction_axes)
                or any(axis not in range(7) for axis in correction_axes)
            ):
                raise ValueError("Correction axes must be unique action dimensions")
        if candidate_selection_count != 0 and candidate_selection_count < 2:
            raise ValueError(
                "Candidate selection requires zero or at least two choices"
            )
        if candidate_selection_margin < 0 or candidate_uncertainty_penalty < 0:
            raise ValueError("Candidate selection penalties must be nonnegative")
        if critic_response_aux_weight < 0:
            raise ValueError("Critic response auxiliary weight must be nonnegative")
        if critic_response_aux_shuffled and critic_response_aux_weight == 0:
            raise ValueError("Shuffled response supervision requires a positive weight")
        if critic_consequence_dueling and (
            critic_conditioner is None or critic_bottleneck_dim != 2 * bottleneck_dim
        ):
            raise ValueError(
                "Consequence-dueling critic requires an equal-width consequence channel"
            )
        if critic_consequence_dueling and critic_response_aux_weight > 0:
            raise ValueError("Consequence dueling and response auxiliary are disjoint")
        self.chunk_len = chunk_len
        self.gripper_threshold = gripper_threshold
        self.critic_conditioning = critic_conditioning
        self.critic_bottleneck_dim = critic_bottleneck_dim
        self.actor_bottleneck_dim = actor_bottleneck_dim
        self.actor_physical_gate = actor_physical_gate
        self.correction_limit = correction_limit
        self.correction_axes = correction_axes
        self.residual_output = residual_output
        self.candidate_selection_count = candidate_selection_count
        self.candidate_selection_margin = candidate_selection_margin
        self.candidate_uncertainty_penalty = candidate_uncertainty_penalty
        self.critic_response_aux_weight = critic_response_aux_weight
        self.critic_response_aux_shuffled = critic_response_aux_shuffled
        self.critic_consequence_dueling = critic_consequence_dueling
        self.last_command_diagnostics: dict | None = None
        self.critic_conditioner = (
            critic_conditioner.to(device).eval().requires_grad_(False)
            if critic_conditioner is not None
            else None
        )
        actor = ActionTokenActor(
            bottleneck_dim=actor_bottleneck_dim,
            action_dim=7,
            chunk_len=chunk_len,
            hidden_dim=512,
            ref_dropout=ref_dropout,
            fixed_std=0.1,
            prop_dim=8,
        ).to(device)
        if self.residual_output:
            nn.init.zeros_(actor.net[-1].weight)
            nn.init.zeros_(actor.net[-1].bias)
        self.actor = (
            _GatedActor(actor, bottleneck_dim).to(device)
            if actor_physical_gate
            else actor
        )
        critic = (
            _ConsequenceDuelingCritic(critic_bottleneck_dim).to(device)
            if critic_consequence_dueling
            else ActionTokenQCritic(
                bottleneck_dim=critic_bottleneck_dim,
                action_dim=7,
                chunk_len=chunk_len,
                hidden_dim=512,
                prop_dim=8,
            ).to(device)
        )
        self.critic = (
            _ResponseAuxiliaryCritic(critic, response_dim=8).to(device)
            if critic_response_aux_weight > 0
            else critic
        )
        self.target_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).eval().requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.actor_updates = self.critic_updates = 0

    def _critic_token(
        self,
        token: torch.Tensor,
        action: torch.Tensor,
        reference: torch.Tensor,
        physical_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the critic state while preserving action gradients."""
        if self.critic_conditioner is None:
            if (
                physical_token is not None
                and self.actor_bottleneck_dim == token.shape[-1]
            ):
                raise ValueError("Physical token requires an explicit conditioner")
            # Actor-only dual-token ablations retain the matched physical view
            # in replay, but the latent critic intentionally ignores it.
            return token
        conditioned = (
            self.critic_conditioner(token, action, reference, physical_token)
            if physical_token is not None
            else self.critic_conditioner(token, action, reference)
        )
        if (
            conditioned.ndim != 2
            or conditioned.shape[0] != token.shape[0]
            or conditioned.shape[1] != self.critic_bottleneck_dim
            or not torch.isfinite(conditioned).all()
        ):
            raise FloatingPointError("Invalid critic conditioning token")
        return conditioned

    def _policy_action(
        self, raw_action: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Optionally bound every learned correction around the reference."""
        if self.correction_limit is None:
            return raw_action
        difference = (
            raw_action / self.correction_limit
            if self.residual_output
            else (raw_action - reference) / self.correction_limit
        )
        correction = self.correction_limit * torch.tanh(difference)
        if self.correction_axes is not None:
            mask = torch.zeros(7, dtype=correction.dtype, device=correction.device)
            mask[list(self.correction_axes)] = 1
            correction = correction * mask
        return reference + correction

    @torch.no_grad()
    def command(self, row: dict, *, deterministic: bool) -> torch.Tensor:
        """Generate a command with dropout disabled for physical execution."""
        self.actor.eval()
        raw_action, _ = self.actor(
            row.get("actor_z", row["z"])[None].to(self.device),
            row["ref"][None].to(self.device),
            row["prop"][None].to(self.device),
            deterministic=deterministic,
        )
        reference = row["ref"][None].to(self.device)
        action = self._policy_action(raw_action, reference)
        self.last_command_diagnostics = {
            "enabled": False,
            "reason": "not_configured",
        }
        if self.candidate_selection_count:
            self.last_command_diagnostics["reason"] = (
                "stochastic_command" if not deterministic else "critic_not_updated"
            )
        if self.candidate_selection_count and deterministic and self.critic_updates > 0:
            action = self._select_candidate(row, reference, action)
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite actor command")
        return action[0].float().cpu().clamp(-1, 1)

    @torch.no_grad()
    def _select_candidate(
        self,
        row: dict,
        reference: torch.Tensor,
        actor_action: torch.Tensor,
    ) -> torch.Tensor:
        """Back off an actor correction unless its conservative Q gain is credible."""
        scales = torch.linspace(
            0,
            1,
            self.candidate_selection_count,
            device=self.device,
            dtype=actor_action.dtype,
        )
        candidates = reference + scales[:, None, None] * (actor_action - reference)
        token = row["z"][None].to(self.device).expand(len(scales), -1)
        references = reference.expand(len(scales), -1, -1)
        prop = row["prop"][None].to(self.device).expand(len(scales), -1)
        physical = row.get("critic_z")
        if physical is not None:
            physical = physical[None].to(self.device).expand(len(scales), -1)
        self.critic.eval()
        q1, q2 = self.critic(
            self._critic_token(token, candidates, references, physical),
            candidates,
            prop,
        )
        disagreement = (q1 - q2).abs()
        scores = (
            torch.minimum(q1, q2) - self.candidate_uncertainty_penalty * disagreement
        )
        best = int(scores.argmax())
        improvement = float(scores[best] - scores[0])
        selected = best if improvement > self.candidate_selection_margin else 0
        self.last_command_diagnostics = {
            "enabled": True,
            "reason": "selected" if selected else "reference_fallback",
            "selected_index": selected,
            "selected_scale": float(scales[selected]),
            "score_improvement": improvement,
            "reference_score": float(scores[0]),
            "selected_disagreement": float(disagreement[selected]),
        }
        return candidates[selected : selected + 1]

    def update(self, replay: list[dict], *, warmup: bool = False) -> dict:
        """Update from pre-action reference labels; terminal tails remain in TD."""
        eligible = (
            replay if warmup else [r for r in replay if td_eligible(r, self.chunk_len)]
        )
        if not eligible:
            raise ValueError("No complete or terminal transitions for updating")
        optional = ("critic_z", "next_critic_z", "actor_z", "next_actor_z")
        for key in optional:
            present = [key in row for row in eligible]
            if any(present) and not all(present):
                raise ValueError(f"Replay inconsistently provides {key}")
        rows = random.choices(eligible, k=128)
        batch = {
            key: torch.stack([r[key] for r in rows]).to(self.device)
            for key in ("z", "ref", "prop", "action", "next_z", "next_ref", "next_prop")
        }
        for key in optional:
            present = [key in row for row in rows]
            if all(present):
                batch[key] = torch.stack([row[key] for row in rows]).to(self.device)
        metrics = {}
        if not warmup:
            self.critic.train()
            with torch.no_grad():
                next_raw_action, _ = self.target_actor(
                    batch.get("next_actor_z", batch["next_z"]),
                    batch["next_ref"],
                    batch["next_prop"],
                    deterministic=True,
                )
                noise = (torch.randn_like(next_raw_action) * 0.2).clamp(-0.5, 0.5)
                next_action = self._policy_action(
                    next_raw_action + noise, batch["next_ref"]
                ).clamp(-1, 1)
                q1, q2 = self.target_critic(
                    self._critic_token(
                        batch["next_z"],
                        next_action,
                        batch["next_ref"],
                        batch.get("next_critic_z"),
                    ),
                    next_action,
                    batch["next_prop"],
                )
                target = td_target(
                    torch.tensor([r["reward"] for r in rows], device=self.device),
                    torch.tensor([r["ticks"] for r in rows], device=self.device),
                    torch.tensor([r["terminal"] for r in rows], device=self.device),
                    torch.minimum(q1, q2),
                )
            q1, q2 = self.critic(
                self._critic_token(
                    batch["z"],
                    batch["action"],
                    batch["ref"],
                    batch.get("critic_z"),
                ),
                batch["action"],
                batch["prop"],
            )
            critic_loss = ((q1 - target).square() + (q2 - target).square()).mean()
            response_aux_loss = None
            if self.critic_response_aux_weight > 0:
                response_target = batch["next_prop"] - batch["prop"]
                replay_responses = torch.stack(
                    [row["next_prop"] - row["prop"] for row in eligible]
                ).to(self.device)
                response_scale = replay_responses.std(0, unbiased=False).clamp_min(1e-3)
                if self.critic_response_aux_shuffled:
                    shuffled_targets = cross_episode_response_targets(eligible)
                    response_target = torch.stack(
                        [shuffled_targets[id(row)] for row in rows]
                    ).to(self.device)
                response_target = response_target / response_scale
                response1, response2 = self.critic.response(
                    self._critic_token(
                        batch["z"],
                        batch["action"],
                        batch["ref"],
                        batch.get("critic_z"),
                    ),
                    batch["action"],
                    batch["prop"],
                )
                response_aux_loss = (response1 - response_target).square().mean() + (
                    response2 - response_target
                ).square().mean()
                critic_loss = (
                    critic_loss + self.critic_response_aux_weight * response_aux_loss
                )
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
            if response_aux_loss is not None:
                metrics["critic_response_aux_loss"] = response_aux_loss.item()
        if warmup or self.critic_updates % 2 == 0:
            self.actor.train()
            self.critic.requires_grad_(False)
            raw_action, _ = self.actor(
                batch.get("actor_z", batch["z"]),
                batch["ref"],
                batch["prop"],
                deterministic=warmup and self.residual_output,
            )
            action = self._policy_action(raw_action, batch["ref"])
            # The target is a full proposal issued before execution, not a
            # retroactively constructed expert suffix after observing outcomes.
            # Keep BC in the actor's native output space. Applying it after the
            # bounded transform would weaken the safety prior near the limit.
            bc_target = (
                torch.zeros_like(raw_action) if self.residual_output else batch["ref"]
            )
            bc = (raw_action - bc_target).square().sum((-2, -1)).mean()
            q = self.critic.q1_forward(
                self._critic_token(
                    batch["z"],
                    action.clamp(-1, 1),
                    batch["ref"],
                    batch.get("critic_z"),
                ),
                action.clamp(-1, 1),
                batch["prop"],
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
                correction_rms=(action.detach() - batch["ref"])
                .square()
                .mean()
                .sqrt()
                .item(),
            )
            if self.actor_physical_gate:
                metrics["actor_physical_gate"] = self.actor.physical_gate().item()
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
            if "actor_z" in rows[start]:
                batch["actor_z"] = torch.stack(
                    [r["actor_z"] for r in rows[start : start + 128]]
                ).to(self.device)
            raw_mean, _ = self.actor(
                batch.get("actor_z", batch["z"]),
                batch["ref"],
                batch["prop"],
                deterministic=True,
            )
            mean = self._policy_action(raw_mean, batch["ref"])
            errors.append((mean - batch["ref"]).square().cpu())
            gripper_errors.append(
                (
                    (mean[..., 6] >= self.gripper_threshold)
                    != (batch["ref"][..., 6] >= self.gripper_threshold)
                ).cpu()
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
        } | {
            "actor_updates": self.actor_updates,
            "critic_updates": self.critic_updates,
            "critic_conditioning": self.critic_conditioning,
            "critic_bottleneck_dim": self.critic_bottleneck_dim,
            "actor_bottleneck_dim": self.actor_bottleneck_dim,
            "actor_physical_gate": self.actor_physical_gate,
            "correction_limit": self.correction_limit,
            "correction_axes": self.correction_axes,
            "residual_output": self.residual_output,
            "critic_response_aux_weight": self.critic_response_aux_weight,
            "critic_response_aux_shuffled": self.critic_response_aux_shuffled,
            "critic_consequence_dueling": self.critic_consequence_dueling,
            "candidate_selection_count": self.candidate_selection_count,
            "candidate_selection_margin": self.candidate_selection_margin,
            "candidate_uncertainty_penalty": self.candidate_uncertainty_penalty,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the same warm start for either explicitly selected objective."""
        saved_conditioning = state.get("critic_conditioning", "latent")
        if saved_conditioning != self.critic_conditioning:
            raise ValueError("Checkpoint critic conditioning differs from the learner")
        if state.get("critic_bottleneck_dim", self.critic_bottleneck_dim) != (
            self.critic_bottleneck_dim
        ):
            raise ValueError("Checkpoint critic token width differs from the learner")
        if state.get("actor_bottleneck_dim", self.actor_bottleneck_dim) != (
            self.actor_bottleneck_dim
        ):
            raise ValueError("Checkpoint actor token width differs from the learner")
        if state.get("actor_physical_gate", False) != self.actor_physical_gate:
            raise ValueError("Checkpoint actor physical-gate contract differs")
        if state.get("correction_limit") != self.correction_limit:
            raise ValueError("Checkpoint correction limit differs from the learner")
        if state.get("correction_axes") != self.correction_axes:
            raise ValueError("Checkpoint correction axes differ from the learner")
        if state.get("residual_output", False) != self.residual_output:
            raise ValueError("Checkpoint residual-output mode differs from the learner")
        if state.get("critic_response_aux_weight", 0.0) != (
            self.critic_response_aux_weight
        ) or state.get("critic_response_aux_shuffled", False) != (
            self.critic_response_aux_shuffled
        ):
            raise ValueError("Checkpoint critic-response contract differs")
        if state.get("critic_consequence_dueling", False) != (
            self.critic_consequence_dueling
        ):
            raise ValueError("Checkpoint consequence-dueling contract differs")
        selection_contract = (
            "candidate_selection_count",
            "candidate_selection_margin",
            "candidate_uncertainty_penalty",
        )
        for name in selection_contract:
            if state.get(name, 0) != getattr(self, name):
                raise ValueError("Checkpoint candidate-selection contract differs")
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
