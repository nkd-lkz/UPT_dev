# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Ground command-response history into an action-conditioned decision token.

The grounding module answers, before a chunk executes: given the current
state, the recent command-response history of this episode, and one candidate
chunk, what measured joint response should be expected and how certain is it.
Training uses only completed transitions already present in warmup rollouts
and online replay, so it runs offline on CPU without touching the frozen
Pi0.5 or the online learner.

The grounded token concatenates the pooled history embedding with the
predicted response mean and standard deviation for the reference chunk. A
learner consumes it appended to the frozen RL token, so a zero token of the
same width is the capacity-matched no-history control.

The predictor must depend on the candidate action: a model that only learns
"what usually happens in this state" cannot rank candidate corrections. The
trainer therefore audits action sensitivity by re-scoring validation samples
with permuted actions and gates on the resulting margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from toolkits.rlt.robodojo_experience import (
    ACTION_DIM,
    CHUNK_LEN,
    EXPERIENCE_DIM,
    JOINT_MASK,
    ExperienceHistory,
    history_view,
    response_valid,
    segment_replay,
)

GROUNDING_KIND = "robodojo-grounding-v1"
HISTORY_EMBED_DIM = 64
TOKEN_DIM = HISTORY_EMBED_DIM + 2 * ACTION_DIM
_LOG_SIGMA_RANGE = (-5.0, 2.0)


class HistoryEncoder(nn.Module):
    """Pool completed-chunk experiences into one episode-context embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(EXPERIENCE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.project = nn.Linear(128, HISTORY_EMBED_DIM)

    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, K, EXPERIENCE_DIM)`` with validity mask to ``(B, 64)``.

        An empty history pools to the zero vector before projection, so the
        encoder degrades to a learned constant rather than an arbitrary value.
        """
        if history.ndim != 3 or history.shape[-1] != EXPERIENCE_DIM:
            raise ValueError("History must be (batch, window, EXPERIENCE_DIM)")
        weights = mask.unsqueeze(-1)
        pooled = (self.embed(history) * weights).sum(1)
        pooled = pooled / weights.sum(1).clamp(min=1.0)
        return self.project(pooled)


class GroundingPredictor(nn.Module):
    """Predict the measured response of one candidate chunk.

    Heads: response mean and log standard deviation over the twelve measured
    joint dimensions (gripper entries exist for shape stability but carry no
    loss), termination logit, and executed-tick fraction. Execution deviation
    is derived downstream from the predicted response and the known command,
    so it has no separate head.
    """

    def __init__(self) -> None:
        super().__init__()
        input_dim = ACTION_DIM + HISTORY_EMBED_DIM + ACTION_DIM * CHUNK_LEN
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.response_mean = nn.Linear(256, ACTION_DIM)
        self.response_log_sigma = nn.Linear(256, ACTION_DIM)
        self.termination = nn.Linear(256, 1)
        self.tick_fraction = nn.Linear(256, 1)

    def forward(
        self, prop: torch.Tensor, context: torch.Tensor, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if action.ndim != 3 or action.shape[1:] != (CHUNK_LEN, ACTION_DIM):
            raise ValueError("Candidate actions must be (batch, 10, 14) chunks")
        features = self.trunk(torch.cat([prop, context, action.flatten(1)], dim=-1))
        return {
            "mean": self.response_mean(features),
            "log_sigma": self.response_log_sigma(features).clamp(*_LOG_SIGMA_RANGE),
            "termination": self.termination(features).squeeze(-1),
            "tick_fraction": self.tick_fraction(features).squeeze(-1),
        }


class GroundedTokenizer(nn.Module):
    """Produce the grounded decision token for one pre-chunk observation."""

    def __init__(self, history_length: int) -> None:
        super().__init__()
        self.history_length = history_length
        self.token_dim = TOKEN_DIM
        self.encoder = HistoryEncoder()
        self.predictor = GroundingPredictor()

    def forward(
        self,
        prop: torch.Tensor,
        history: torch.Tensor,
        mask: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(B, TOKEN_DIM)`` tokens: [context, mu(ref), sigma(ref)]."""
        context = self.encoder(history, mask)
        prediction = self.predictor(prop, context, reference)
        return torch.cat(
            [context, prediction["mean"], prediction["log_sigma"].exp()], dim=-1
        )

    @torch.no_grad()
    def token(
        self,
        prop: torch.Tensor,
        history: np.ndarray,
        mask: np.ndarray,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one decision point from numpy history buffers to a (TOKEN_DIM,) CPU tensor."""
        device = next(self.parameters()).device
        return self(
            prop.to(device, torch.float32)[None],
            torch.from_numpy(np.ascontiguousarray(history))[None].to(device),
            torch.from_numpy(np.ascontiguousarray(mask))[None].to(device),
            reference.to(device, torch.float32)[None],
        )[0].cpu()


_JOINTS = torch.from_numpy(JOINT_MASK.copy())


def grounding_loss(
    prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Gaussian response NLL on measured joints plus termination and duration.

    Terminal rows have no measured post-chunk state, so their response terms
    are masked out; they still supervise the termination and duration heads.
    """
    joints = _JOINTS.to(prediction["mean"].device)
    sigma_valid = batch["response_valid"].unsqueeze(-1) * joints
    nll = (
        prediction["log_sigma"]
        + 0.5
        * (
            (batch["response"] - prediction["mean"]) / prediction["log_sigma"].exp()
        ).square()
    )
    nll = (nll * sigma_valid).sum() / sigma_valid.sum().clamp(min=1.0)
    termination = nn.functional.binary_cross_entropy_with_logits(
        prediction["termination"], batch["terminated"]
    )
    duration = (prediction["tick_fraction"] - batch["tick_fraction"]).square().mean()
    loss = nll + termination + duration
    return loss, {
        "nll": float(nll.item()),
        "termination": float(termination.item()),
        "duration": float(duration.item()),
    }


def replay_rows(path: Path) -> list[dict[str, Any]]:
    """Load completed replay rows from warmup rollouts or an online checkpoint.

    ``encoded-rollouts.pt`` stores the row list directly; online checkpoints
    carry their full replay under ``online_state``. Both share the row schema
    the client builds, which is all the grounding needs.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("online_state"), dict):
        rows = payload["online_state"]["replay"]
    else:
        raise ValueError(f"Unrecognized replay artifact: {path}")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Replay artifact holds no rows: {path}")
    return rows


def build_samples(
    rows: list[dict[str, Any]], history_length: int
) -> list[dict[str, Any]]:
    """Pair each row with the completed history preceding it in its episode."""
    samples = []
    for segment_index, segment in enumerate(segment_replay(rows)):
        history = ExperienceHistory(history_length)
        for row in segment:
            if "next_prop" not in row:
                continue
            features, mask = history.features()
            response = np.zeros(ACTION_DIM, dtype=np.float32)
            valid = response_valid(row)
            if valid:
                response[JOINT_MASK] = (
                    np.asarray(row["next_prop"], dtype=np.float32)
                    - np.asarray(row["prop"], dtype=np.float32)
                )[JOINT_MASK]
            samples.append(
                {
                    "prop": np.asarray(row["prop"], dtype=np.float32),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "history": features,
                    "mask": mask,
                    "response": response,
                    "response_valid": float(valid),
                    "terminated": float(bool(row["terminated"])),
                    "tick_fraction": row["ticks"] / CHUNK_LEN,
                    "segment": segment_index,
                }
            )
            history.append(row)
    if not samples:
        raise ValueError("No grounding samples could be built from the replay")
    return samples


def _collate(samples: list[dict[str, Any]], device: str) -> dict[str, torch.Tensor]:
    batch = {}
    for key in (
        "prop",
        "action",
        "history",
        "mask",
        "response",
        "response_valid",
        "terminated",
        "tick_fraction",
    ):
        batch[key] = torch.as_tensor(
            np.stack([sample[key] for sample in samples])
            if isinstance(samples[0][key], np.ndarray)
            else np.asarray([sample[key] for sample in samples], dtype=np.float32)
        ).to(device, torch.float32)
    return batch


@torch.no_grad()
def action_permutation_margin(
    tokenizer: GroundedTokenizer,
    samples: list[dict[str, Any]],
    device: str,
    rng: random.Random,
) -> dict[str, float]:
    """Measure whether predictions depend on the candidate action.

    Re-scores the validation set with actions permuted across samples. A
    predictor that ignores its action input shows no margin; grounding then
    cannot rank candidate corrections and the artifact is rejected.
    """
    batch = _collate(samples, device)
    context = tokenizer.encoder(batch["history"], batch["mask"])
    matched, _ = grounding_loss(
        tokenizer.predictor(batch["prop"], context, batch["action"]), batch
    )
    order = list(range(len(samples)))
    rng.shuffle(order)
    permuted, _ = grounding_loss(
        tokenizer.predictor(batch["prop"], context, batch["action"][order]), batch
    )
    return {
        "matched_loss": float(matched.item()),
        "permuted_loss": float(permuted.item()),
        "margin": float((permuted - matched).item()),
    }


@torch.no_grad()
def applicability_weights(
    tokenizer: GroundedTokenizer,
    experiences: np.ndarray,
    recent: tuple[np.ndarray, np.ndarray],
    *,
    scale: float = 1.0,
    device: str = "cpu",
) -> np.ndarray:
    """Weight stored experiences by how well the current regime explains them.

    Each stored experience is re-predicted under the context of the recent
    window; its weight is ``exp(-error / scale)`` over measured joints. An
    experience from a changed regime scores poorly and is down-weighted, and
    the same computation restores its weight once the old regime returns, so
    retention never requires deletion.
    """
    if experiences.ndim != 2 or experiences.shape[1] != EXPERIENCE_DIM:
        raise ValueError("Experiences must be (count, EXPERIENCE_DIM)")
    if scale <= 0:
        raise ValueError("Applicability scale must be positive")
    if not len(experiences):
        return np.zeros(0, dtype=np.float32)
    window, mask = recent
    stored = torch.as_tensor(experiences, dtype=torch.float32, device=device)
    prop = stored[:, :ACTION_DIM]
    action = stored[:, ACTION_DIM : ACTION_DIM + ACTION_DIM * CHUNK_LEN].reshape(
        -1, CHUNK_LEN, ACTION_DIM
    )
    response = stored[
        :, ACTION_DIM + ACTION_DIM * CHUNK_LEN : ACTION_DIM * (CHUNK_LEN + 2)
    ]
    context = tokenizer.encoder(
        torch.as_tensor(window, dtype=torch.float32, device=device)[None].expand(
            len(stored), -1, -1
        ),
        torch.as_tensor(mask, dtype=torch.float32, device=device)[None].expand(
            len(stored), -1
        ),
    )
    joints = _JOINTS.to(device)
    prediction = tokenizer.predictor(prop, context, action)
    error = ((prediction["mean"] - response).square() * joints).sum(-1) / joints.sum()
    return torch.exp(-error / scale).cpu().numpy().astype(np.float32)


def ground_replay_rows(
    rows: list[dict[str, Any]],
    tokenizer: GroundedTokenizer,
    *,
    mode: str = "correct",
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Append grounded tokens to recorded replay rows, reconstructing history.

    Seed replay rows were recorded without tokens, but their episode structure
    is recoverable, so each row receives the token its decision point would
    have produced online: conditioned on the completed chunks preceding it in
    its own episode. Successor tokens are attached afterwards from the next
    row of the same episode, matching the online ``attach_successor`` contract
    (terminal rows keep themselves as successor).
    """
    grounded: list[list[dict[str, Any]]] = []
    for segment in segment_replay(rows):
        history = ExperienceHistory(tokenizer.history_length)
        updated_segment = []
        for index, row in enumerate(segment):
            features, mask = history.features()
            if mode != "correct":
                features, mask = history_view(features, mask, mode, rng)
            token = tokenizer.token(
                torch.as_tensor(row["prop"], dtype=torch.float32),
                features,
                mask,
                torch.as_tensor(row["ref"], dtype=torch.float32),
            )
            updated_segment.append(
                row
                | {
                    "z": torch.cat(
                        [torch.as_tensor(row["z"], dtype=torch.float32), token]
                    )
                }
            )
            history.append(row)
        for index, row in enumerate(updated_segment):
            successor = (
                updated_segment[index + 1] if index + 1 < len(updated_segment) else row
            )
            row["next_z"] = successor["z"]
        grounded.append(updated_segment)
    return [row for segment in grounded for row in segment]


def save_grounding(
    path: Path, tokenizer: GroundedTokenizer, metrics: dict[str, Any]
) -> str:
    """Persist the grounding artifact and return its content digest."""
    payload = {
        "kind": GROUNDING_KIND,
        "state_dict": {
            key: value.cpu() for key, value in tokenizer.state_dict().items()
        },
        "config": {
            "history_length": tokenizer.history_length,
            "experience_dim": EXPERIENCE_DIM,
            "token_dim": TOKEN_DIM,
        },
        "metrics": metrics,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_grounding(path: Path, *, device: str = "cpu") -> tuple[GroundedTokenizer, str]:
    """Load one frozen grounding artifact and its content digest."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != GROUNDING_KIND:
        raise ValueError("Artifact is not a RoboDojo grounding module")
    config = payload["config"]
    if (
        config.get("experience_dim") != EXPERIENCE_DIM
        or config.get("token_dim") != TOKEN_DIM
    ):
        raise ValueError("Grounding artifact uses an incompatible experience layout")
    tokenizer = GroundedTokenizer(config["history_length"])
    tokenizer.load_state_dict(payload["state_dict"], strict=True)
    return tokenizer.to(device).eval().requires_grad_(False), digest


def train_grounding(args: argparse.Namespace) -> dict[str, Any]:
    """Fit the grounding offline and gate on action sensitivity."""
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    rows = []
    for path in args.replay:
        rows.extend(replay_rows(path))
    samples = build_samples(rows, args.history_length)
    # Split by episode so chunks of one episode never land in both sets.
    segments = sorted({sample["segment"] for sample in samples})
    rng.shuffle(segments)
    validation_count = max(1, round(len(segments) * args.validation_fraction))
    held_out = set(segments[:validation_count]) if len(segments) > 1 else set()
    train = [sample for sample in samples if sample["segment"] not in held_out]
    validation = [
        sample for sample in samples if sample["segment"] in held_out
    ] or train[-1:]
    tokenizer = GroundedTokenizer(args.history_length).to(args.device)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
    history = []
    for step in range(args.steps):
        batch = _collate(
            [train[rng.randrange(len(train))] for _ in range(args.batch_size)],
            args.device,
        )
        context = tokenizer.encoder(batch["history"], batch["mask"])
        prediction = tokenizer.predictor(batch["prop"], context, batch["action"])
        loss, metrics = grounding_loss(prediction, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite grounding loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 10.0)
        optimizer.step()
        if step % 100 == 0 or step + 1 == args.steps:
            history.append({"step": step + 1, "loss": float(loss.item()), **metrics})
    tokenizer.eval().requires_grad_(False)
    audit = action_permutation_margin(tokenizer, validation, args.device, rng)
    gate_passed = audit["margin"] > 0
    metrics = {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "replay_rows": len(rows),
        "history": history[-5:],
        "action_permutation": audit,
        "gate_passed": gate_passed,
        "seed": args.seed,
        "steps": args.steps,
    }
    digest = save_grounding(args.output / "grounding.pt", tokenizer, metrics)
    (args.output / "status.json").write_text(
        json.dumps(
            {
                "state": "complete" if gate_passed else "failed",
                "grounding_sha256": digest,
                **metrics,
            },
            indent=2,
        )
        + "\n"
    )
    if not gate_passed:
        raise RuntimeError(
            "Grounding predictions do not depend on the candidate action"
        )
    return metrics


def main() -> None:
    """Train one grounding artifact offline from recorded replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (
        not 1 <= args.steps <= 100000
        or not 1 <= args.batch_size <= 2048
        or not 1 <= args.history_length <= 64
        or not 0 < args.validation_fraction < 0.5
    ):
        parser.error("Invalid bounded grounding training configuration")
    for path in args.replay:
        if not path.is_file():
            parser.error(f"Replay artifact does not exist: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    train_grounding(args)


if __name__ == "__main__":
    main()
