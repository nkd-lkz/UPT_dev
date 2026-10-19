# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Command-response experience records and short histories for grounding.

An experience is one completed control chunk: the pre-chunk state, the issued
chunk, and the measured response read back from the next pre-chunk state. A
decision at chunk t may only see chunks that finished before t, so histories
are built from completed replay rows and reset at episode boundaries.

Gripper dimensions (6 and 13) report commanded openings, not measured ones, so
they are excluded from every measured-response field.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

ACTION_DIM = 14
CHUNK_LEN = 10
GRIPPER_DIMS = (6, 13)
JOINT_MASK = np.ones(ACTION_DIM, dtype=bool)
JOINT_MASK[list(GRIPPER_DIMS)] = False

# prop + flattened issued chunk + joint response + execution deviation + meta.
META_DIM = 3
EXPERIENCE_DIM = (
    ACTION_DIM + ACTION_DIM * CHUNK_LEN + ACTION_DIM + ACTION_DIM + META_DIM
)

_RESPONSE_START = ACTION_DIM + ACTION_DIM * CHUNK_LEN
_RESPONSE_END = _RESPONSE_START + 2 * ACTION_DIM

HISTORY_MODES = ("none", "correct", "shuffled", "zero")


def _as_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Experience field {name} must be finite with shape {shape}")
    return array


def response_valid(row: dict[str, Any]) -> bool:
    """A measured joint response exists only when a later chunk observed it.

    Terminal and truncated rows carry a placeholder successor (the same or a
    misaligned observation), so their ``next_prop`` is not a post-chunk state.
    """
    return (
        not bool(row.get("terminated"))
        and not bool(row.get("truncated"))
        and "next_prop" in row
    )


def experience_features(row: dict[str, Any]) -> np.ndarray:
    """Encode one completed replay row as a fixed-size experience vector.

    Layout: ``[prop 14, issued chunk 140, joint response 14, execution
    deviation 14, ticks/10, reward, terminated]``. The joint response is the
    measured normalized state change over the chunk; it is zero on gripper
    dimensions and on rows without a valid successor (flagged through the
    terminated entry). The execution deviation is the last acknowledged
    command minus the issued command at that tick, which is zero unless an
    execution-side perturbation or takeover altered the command.
    """
    if "next_prop" not in row:
        raise ValueError("Experience rows require an attached successor state")
    ticks = row["ticks"]
    if type(ticks) is not int or not 1 <= ticks <= CHUNK_LEN:
        raise ValueError("Experience rows require 1..10 executed ticks")
    prop = _as_array(row["prop"], (ACTION_DIM,), "prop")
    action = _as_array(row["action"], (CHUNK_LEN, ACTION_DIM), "action")
    executed = _as_array(row["executed_action"], (ticks, ACTION_DIM), "executed_action")
    response = np.zeros(ACTION_DIM, dtype=np.float32)
    if response_valid(row):
        next_prop = _as_array(row["next_prop"], (ACTION_DIM,), "next_prop")
        response[JOINT_MASK] = (next_prop - prop)[JOINT_MASK]
    deviation = executed[-1] - action[ticks - 1]
    meta = np.asarray(
        [ticks / CHUNK_LEN, float(row["reward"]), float(bool(row["terminated"]))],
        dtype=np.float32,
    )
    features = np.concatenate([prop, action.reshape(-1), response, deviation, meta])
    if features.shape != (EXPERIENCE_DIM,):
        raise AssertionError("Experience layout drifted from EXPERIENCE_DIM")
    return features


class ExperienceHistory:
    """Fixed-capacity record of the most recent completed chunks in an episode.

    ``append`` accepts completed replay rows (rows that already carry their
    successor state), ``features`` returns the padded feature matrix with its
    validity mask, and ``reset`` clears the record at an episode boundary.
    """

    def __init__(self, maxlen: int = 16) -> None:
        if not 1 <= maxlen <= 64:
            raise ValueError("History length must be in [1, 64]")
        self.maxlen = maxlen
        self._features: deque[np.ndarray] = deque(maxlen=maxlen)

    def reset(self) -> None:
        self._features.clear()

    def append(self, row: dict[str, Any]) -> None:
        self._features.append(experience_features(row))

    def __len__(self) -> int:
        return len(self._features)

    def features(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(maxlen, EXPERIENCE_DIM)`` features and the validity mask."""
        features = np.zeros((self.maxlen, EXPERIENCE_DIM), dtype=np.float32)
        mask = np.zeros(self.maxlen, dtype=np.float32)
        for index, row in enumerate(self._features):
            features[index] = row
            mask[index] = 1.0
        return features, mask


def history_view(
    features: np.ndarray,
    mask: np.ndarray,
    mode: str,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the history under one evaluation mode.

    ``correct`` keeps the record as observed. ``shuffled`` permutes the
    measured-response block across valid entries, breaking the pairing between
    each command and its observed outcome while preserving the marginal
    response distribution (a permutation-invariant pooling would otherwise see
    a reordered history as identical). ``zero`` reports an empty history of
    the same capacity.
    """
    if mode not in HISTORY_MODES or mode == "none":
        raise ValueError(f"Unknown history view mode: {mode}")
    if mode == "correct":
        return features, mask
    if mode == "zero":
        return np.zeros_like(features), np.zeros_like(mask)
    if rng is None:
        raise ValueError("Shuffled history views require an explicit generator")
    valid = np.flatnonzero(mask > 0)
    shuffled = features.copy()
    if len(valid) > 1:
        permutation = valid[rng.permutation(len(valid))]
        shuffled[valid, _RESPONSE_START:_RESPONSE_END] = features[
            permutation, _RESPONSE_START:_RESPONSE_END
        ]
    return shuffled, mask


def segment_replay(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split chronological replay rows back into per-episode segments.

    Online replay rows carry no episode identity, but within an episode each
    row's successor state equals the next row's pre-chunk state. A segment
    therefore ends at a terminal or truncated row, or when that chain breaks.
    """
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        current.append(row)
        chain_broken = True
        if not row.get("terminated") and not row.get("truncated"):
            if index + 1 < len(rows):
                next_prop = np.asarray(row["next_prop"], dtype=np.float32)
                following = np.asarray(rows[index + 1]["prop"], dtype=np.float32)
                chain_broken = not np.allclose(next_prop, following, atol=1e-5, rtol=0)
        if chain_broken:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments
