# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Jev-inspired finite decisions learned locally from RLT replay."""

import math
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.modules.rlt_action_candidates import JointActionCandidates


class RLTAtomicPolicy(RLTMLPPolicy):
    """Select a complete bounded candidate; never execute its probability average.

    ``SAC`` retains the ordinary sampled-action API. ``RLT_CANDIDATES`` returns
    differentiable probabilities plus detached candidates for exact finite-action
    actor updates. The inherited continuous critic scores actual executed chunks,
    including interventions that need not belong to the candidate vocabulary.
    """

    def __init__(self, *, atomic_decision: dict, **kwargs) -> None:
        if kwargs.get("q_head_type", "default") != "default":
            raise ValueError(
                "Atomic decisions currently require default twin Q, not CrossQ."
            )
        super().__init__(**kwargs)
        allowed = {"enabled", "radius", "reference_prior"}
        unknown = set(atomic_decision) - allowed
        if unknown:
            raise ValueError(f"Unknown atomic_decision fields: {sorted(unknown)}")
        self.candidates = JointActionCandidates(
            self.step_action_dim,
            self.chunk_len,
            radius=float(atomic_decision.get("radius", 0.08)),
        )
        prior = float(atomic_decision.get("reference_prior", 0.9))
        if not math.isfinite(prior) or not 0 < prior < 1:
            raise ValueError("reference_prior must be finite and in (0, 1).")
        # Remove unused continuous heads rather than leaving optimizer dead weights.
        del self.actor_mean
        del self.actor_logstd
        self.selector = nn.Linear(256, len(self.candidates.names))
        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)
        with torch.no_grad():
            self.selector.bias[0] = math.log(
                prior * (len(self.candidates.names) - 1) / (1 - prior)
            )
        self.register_buffer(
            "decision_prior_logits", self.selector.bias.detach().clone()
        )

    def forward(
        self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs: Any
    ) -> Any:
        """Dispatch regular RLT forwards or the categorical training interface."""
        if forward_type == ForwardType.RLT_CANDIDATES:
            kwargs["obs"] = self.preprocess_env_obs(kwargs["obs"])
            return self.decision_forward(**kwargs)
        return super().forward(forward_type=forward_type, **kwargs)

    def decision_forward(
        self,
        obs: dict,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Return logits, normalized choice probabilities, candidates and validity."""
        if not 0 <= reference_dropout_prob <= 1:
            raise ValueError("reference_dropout_prob must be in [0, 1].")
        state = self._actor_state(
            obs,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
        ).detach()
        if not torch.isfinite(state).all():
            raise ValueError("Non-finite RLT observations cannot form a decision.")
        reference = self._get_ref_chunk(obs).reshape(
            -1, self.chunk_len, self.step_action_dim
        )
        candidates, valid = self.candidates(reference)
        logits = self.selector(self.backbone(state))
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite selector logits; stop and inspect training.")
        logits = logits.masked_fill(~valid, -torch.inf)
        return {
            "candidates": candidates,
            "valid": valid,
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "prior_logits": self.decision_prior_logits[None]
            .expand_as(logits)
            .masked_fill(~valid, -torch.inf),
        }

    def sac_forward(
        self, obs: dict, deterministic: bool = False, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Return one proposed action chunk and its categorical log probability."""
        decision = self.decision_forward(obs, **kwargs)
        action, logprob, _ = self.select_candidate(decision, deterministic)
        return action, logprob, None

    @staticmethod
    def select_candidate(
        decision: dict, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select one intact candidate and return action, log probability and ID."""
        distribution = Categorical(logits=decision["logits"])
        choice = (
            decision["logits"].argmax(-1) if deterministic else distribution.sample()
        )
        batch = torch.arange(choice.shape[0], device=choice.device)
        action = decision["candidates"][batch, choice].flatten(1)
        return action, distribution.log_prob(choice).unsqueeze(-1), choice

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs: dict,
        calculate_logprobs: bool = True,
        calculate_values: bool = True,
        return_obs: bool = True,
        mode: str = "train",
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict]:
        """Propose a candidate; downstream routing can still replace this proposal."""
        obs = self.preprocess_env_obs(env_obs)
        decision = self.decision_forward(obs)
        action, logprob, choice = self.select_candidate(decision, mode == "eval")
        forward_inputs = {
            "action": action,
            "model_action": action,
            # These describe the student proposal, not an expert/base replacement.
            "atomic_proposed_id": choice[:, None],
            "atomic_proposed_probabilities": decision["probabilities"],
        }
        if return_obs:
            forward_inputs.update(obs)
        return self._format_chunk_actions(action), {
            "prev_logprobs": logprob,
            "prev_values": torch.zeros_like(logprob),
            "forward_inputs": forward_inputs,
        }

    def sft_forward(self, data: dict, **kwargs: Any) -> torch.Tensor:
        """Reject SFT: this research head is trained by the Stage 2 worker."""
        raise NotImplementedError(
            "Atomic decisions use Stage 2 categorical Q+BC, not SFT."
        )
