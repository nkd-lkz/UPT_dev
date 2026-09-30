# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exact finite-action expectations for an opt-in RLT actor-critic."""

import math
from typing import Any

import torch

from rlinf.models.embodiment.base_policy import ForwardType


def atomic_decision_enabled(cfg: Any) -> bool:
    """Return whether the RLT model uses the finite candidate contract."""
    return bool(cfg.actor.model.get("atomic_decision", {}).get("enabled", False))


def validate_atomic_config(cfg: Any) -> None:
    """Reject unsupported objective/controller combinations before starting Ray."""
    residual = cfg.actor.model.get("bounded_residual", {})
    if residual.get("enabled", False):
        radius = float(residual.get("radius", 0.08))
        if not math.isfinite(radius) or not 0 < radius <= 1:
            raise ValueError("Residual radius must be finite and in (0, 1].")
        if atomic_decision_enabled(cfg):
            raise ValueError(
                "Atomic and continuous residual actors are mutually exclusive"
            )
        if cfg.rollout.model.get("bounded_residual", {}) != residual:
            raise ValueError("Actor and rollout must share bounded_residual settings")
        entropy = cfg.algorithm.get("entropy_tuning", {})
        if (
            entropy.get("alpha_type") != "fixed_alpha"
            or entropy.get("initial_alpha") != 0
        ):
            raise ValueError("Clipped residual comparator requires fixed zero entropy")
        if (
            cfg.algorithm.loss_type != "rlt_ac"
            or cfg.actor.model.model_type != "rlt_mlp_policy"
        ):
            raise ValueError("Residual comparator requires rlt_ac")
        if (
            cfg.actor.model.get("q_head_type", "default") != "default"
            or cfg.algorithm.get("q_head_type", "default") != "default"
        ):
            raise ValueError("Residual comparator does not support CrossQ")
        if cfg.actor.get("fsdp_config", {}) and not cfg.actor.fsdp_config.get(
            "use_orig_params", False
        ):
            raise ValueError("Residual comparator requires use_orig_params=True")
        if (
            cfg.rollout.get("enable_cuda_graph", False)
            or cfg.rollout.get("enable_torch_compile", False)
            or cfg.actor.get("compile_model", False)
        ):
            raise ValueError("Residual comparator compilation is not verified")
        for name in ("train", "eval"):
            env = cfg.env.get(name)
            if env and (
                env.env_type != "maniskill_rlt"
                or env.init_params.control_mode != "pd_joint_delta_pos"
            ):
                raise ValueError(
                    "Residual comparator requires ManiSkill joint-delta controls"
                )
    if not atomic_decision_enabled(cfg):
        return
    if cfg.actor.model.model_type != "rlt_mlp_policy":
        raise ValueError("Atomic decisions require model_type=rlt_mlp_policy.")
    if cfg.algorithm.loss_type != "rlt_ac":
        raise ValueError("Atomic decisions require loss_type=rlt_ac.")
    if (
        cfg.actor.model.q_head_type != "default"
        or cfg.algorithm.get("q_head_type", "default") != "default"
    ):
        raise ValueError("Atomic decisions do not support CrossQ.")
    fsdp = cfg.actor.get("fsdp_config", {})
    if fsdp and not fsdp.get("use_orig_params", False):
        raise ValueError("Atomic actor/critic optimizers require use_orig_params=True.")
    if fsdp and not fsdp.get("disable", False):
        raise ValueError(
            "Atomic actor/critic synchronization requires nested FSDP auto-wrap "
            "to be disabled with fsdp_config.disable=True."
        )
    atomic = cfg.actor.model.atomic_decision
    radius = float(atomic.get("radius", 0.08))
    prior = float(atomic.get("reference_prior", 0.9))
    if not math.isfinite(radius) or not 0 < radius <= 1:
        raise ValueError("Candidate radius must be finite and in (0, 1].")
    if not math.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("reference_prior must be finite and in (0, 1).")
    if cfg.algorithm.get("bootstrap_type", "standard") != "standard":
        raise ValueError("Atomic decisions require terminal-masked standard bootstrap.")
    if (
        cfg.rollout.get("enable_cuda_graph", False)
        or cfg.rollout.get("enable_torch_compile", False)
        or cfg.actor.get("compile_model", False)
    ):
        raise ValueError("Atomic decision compilation/CUDA graphs are not verified.")
    if cfg.rollout.model.get("atomic_decision", {}) != cfg.actor.model.atomic_decision:
        raise ValueError("Actor and rollout must share the atomic_decision contract.")
    for name in ("train", "eval"):
        env = cfg.env.get(name)
        if env is None:
            continue
        if (
            env.env_type != "maniskill_rlt"
            or env.init_params.control_mode != "pd_joint_delta_pos"
        ):
            raise ValueError(
                "Atomic v1 supports ManiSkill RLT pd_joint_delta_pos only."
            )
        if cfg.actor.model.action_dim != 8:
            raise ValueError("The shipped Panda contract requires 7 joints + gripper.")
    penalty = float(cfg.algorithm.get("atomic_disagreement_weight", 0.0))
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("atomic_disagreement_weight must be finite and nonnegative.")
    temperature = float(cfg.algorithm.get("atomic_temperature", 0.1))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("atomic_temperature must be finite and positive.")


def candidate_q_values(
    model: torch.nn.Module, obs: dict, candidates: torch.Tensor
) -> torch.Tensor:
    """Score [B,K,H,D] candidates through the ordinary continuous twin critic."""
    batch, count = candidates.shape[:2]
    repeated = {
        key: value.detach().repeat_interleave(count, dim=0)
        for key, value in obs.items()
        if torch.is_tensor(value)
    }
    q = model(
        forward_type=ForwardType.SAC_Q,
        obs=repeated,
        actions=candidates.flatten(0, 1).flatten(1),
        detach_encoder=True,
    )
    return q.reshape(batch, count, -1)


def candidate_bc_errors(
    candidates: torch.Tensor,
    actions: torch.Tensor,
    reference: torch.Tensor,
    intervene_flags: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-candidate BC errors and per-step human mask, never nearest IDs.

    An intervention may lie outside the finite vocabulary. Its distance is
    retained, not relabeled as if a candidate had been executed successfully.
    """
    batch, _, horizon, dim = candidates.shape
    executed = actions.reshape(batch, horizon, dim).detach()
    ref = reference.reshape(batch, -1, dim)[:, :horizon].detach()
    if intervene_flags is None:
        human = torch.zeros((batch, horizon), dtype=torch.bool, device=actions.device)
    else:
        human = intervene_flags.reshape(batch, horizon, dim).bool().any(-1)
    target = torch.where(human[..., None], executed, ref)
    errors = (candidates - target[:, None]).square().mean(dim=(-1, -2))
    return errors, human


def atomic_actor_loss(
    worker: Any, batch: dict
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Differentiate choice probabilities, not a sampled index or averaged action."""
    obs = batch["curr_obs"]
    decision = worker.model(
        forward_type=ForwardType.RLT_CANDIDATES,
        obs=obs,
        apply_reference_dropout=True,
        reference_dropout_prob=float(
            worker.cfg.algorithm.get("reference_dropout_prob", 0)
        ),
    )
    probs = decision["probabilities"]
    with torch.no_grad():
        q = candidate_q_values(worker.model, obs, decision["candidates"])
        worker._require_twin_q(q)
        bc, human = candidate_bc_errors(
            decision["candidates"],
            batch["actions"],
            worker._ref_chunk(obs),
            batch.get("intervene_flags"),
        )
        disagreement = (q[..., 0] - q[..., 1]).abs()
    bc_weight, q_weight, metrics = worker._actor_objective_weights()
    penalty = float(worker.cfg.algorithm.get("atomic_disagreement_weight", 0))
    cost = -q_weight * q[..., 0] + bc_weight * bc + penalty * disagreement
    # A detached, all-candidate improvement target avoids a saturated selector
    # suppressing the gradient of a currently unlikely but useful candidate.
    # This minimizes KL(target || selector) for a reference-prior improvement,
    # intentionally distinct from the baseline continuous actor's pathwise loss.
    temperature = float(worker.cfg.algorithm.get("atomic_temperature", 0.1))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("atomic_temperature must be finite and positive.")
    if not torch.isfinite(cost).all():
        raise ValueError(
            "Non-finite candidate cost; inspect replay actions and critic."
        )
    target_logits = decision["prior_logits"] - cost / temperature
    target_probs = torch.softmax(target_logits.detach(), dim=-1)
    log_probs = torch.log_softmax(decision["logits"], dim=-1)
    safe_log_probs = log_probs.masked_fill(~decision["valid"], 0)
    loss = -(target_probs * safe_log_probs).sum(-1).mean()
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()
    metrics.update(
        {
            "q_pi": (probs * q[..., 0]).sum(-1).mean().detach().item(),
            "bc_loss": (probs * bc).sum(-1).mean().detach().item(),
            "human_mask_ratio": human.float().mean().item(),
            "atomic/reference_probability": probs[:, 0].mean().detach().item(),
            "atomic/choice_entropy": entropy.detach().item(),
            "atomic/effective_candidates": decision["valid"]
            .float()
            .sum(-1)
            .mean()
            .item(),
            "atomic/bc_error_floor": bc.masked_fill(~decision["valid"], torch.inf)
            .min(-1)
            .values.mean()
            .item(),
            "atomic/q_disagreement": (probs * disagreement)
            .sum(-1)
            .mean()
            .detach()
            .item(),
            "atomic/disagreement_weight": penalty,
            "atomic/temperature": temperature,
            "atomic/target_reference_probability": target_probs[:, 0].mean().item(),
            "atomic/greedy_nonreference_fraction": (probs.argmax(-1) != 0)
            .float()
            .mean()
            .item(),
            "atomic/target_nonreference_fraction": (target_probs.argmax(-1) != 0)
            .float()
            .mean()
            .item(),
            "atomic/best_q_advantage": (
                q[..., 0].masked_fill(~decision["valid"], -torch.inf).max(-1).values
                - q[:, 0, 0]
            )
            .mean()
            .item(),
            "atomic/target_kl": (
                target_probs * (target_probs.clamp_min(1e-12).log() - safe_log_probs)
            )
            .sum(-1)
            .mean()
            .detach()
            .item(),
            "atomic/expected_cost": (probs * cost).sum(-1).mean().detach().item(),
            "atomic/reference_clip_fraction": (worker._ref_chunk(obs).abs() > 1)
            .float()
            .mean()
            .item(),
        }
    )
    return loss, entropy, metrics


@torch.no_grad()
def atomic_next_value(
    model: torch.nn.Module, target_model: torch.nn.Module, next_obs: dict
) -> torch.Tensor:
    """Expected target min-Q under the online categorical policy, shape [B,1]."""
    decision = model(forward_type=ForwardType.RLT_CANDIDATES, obs=next_obs)
    q = candidate_q_values(target_model, next_obs, decision["candidates"])
    if q.shape[-1] < 2:
        raise ValueError("Atomic backup needs two Q heads.")
    return (decision["probabilities"] * torch.minimum(q[..., 0], q[..., 1])).sum(
        -1, keepdim=True
    )
