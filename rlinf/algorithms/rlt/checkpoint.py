# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate the counters needed to resume synchronous RLT scheduling."""

import json
import os
from pathlib import Path

COUNTER_NAMES = (
    "update_step",
    "transitions_since_train",
    "episodes_since_train",
    "total_transitions_added",
    "total_episodes_added",
    "pending_update_budget",
    "_warmup_ready_total_transitions",
    "_warmup_ready_total_episodes",
)


def validate_counters(counters: dict) -> dict:
    """Return a checked copy; missing counters must never default to zero."""
    if not isinstance(counters, dict) or set(counters) != set(COUNTER_NAMES):
        raise ValueError("Incomplete RLT training counters")
    for key, value in counters.items():
        if key.startswith("_warmup_ready") and value is None:
            continue
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid nonnegative integer counter: {key}")
    anchors = [counters[k] for k in COUNTER_NAMES[-2:]]
    if (anchors[0] is None) != (anchors[1] is None):
        raise ValueError("RLT warmup anchors must both be present or both unset")
    for kind in ("transitions", "episodes"):
        if counters[f"{kind}_since_train"] > counters[f"total_{kind}_added"]:
            raise ValueError(f"RLT {kind} interval exceeds total")
    return dict(counters)


def save_training_state(
    directory: str,
    *,
    rank: int,
    world_size: int,
    step: int,
    counters: dict,
    schedule: dict,
) -> None:
    """Atomically publish per-rank metadata after model/replay saving succeeds."""
    counters = validate_counters(counters)
    path = Path(directory) / f"rlt_training_state_rank_{rank}.json"
    payload = {
        "schema_version": 1,
        "rank": rank,
        "world_size": world_size,
        "checkpoint_step": step,
        "counters": counters,
        "schedule": schedule,
    }
    temporary = path.with_suffix(f".json.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_state(
    directory: str,
    *,
    rank: int,
    world_size: int,
    schedule: dict,
) -> dict:
    """Reject legacy, partial, or incompatible resumes before loading weights.

    This restores scheduling, not simulator or random-generator state. Older
    checkpoints require independently verified counter reconstruction; an outer
    step number alone cannot recover replay-derived warmup anchors.
    """
    path = Path(directory) / f"rlt_training_state_rank_{rank}.json"
    if not path.is_file():
        raise ValueError(
            f"Missing RLT training state: {path}. Refusing to restart warmup "
            "silently; verify legacy counters before resuming."
        )
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported RLT training state schema")
    if payload.get("rank") != rank or payload.get("world_size") != world_size:
        raise ValueError("RLT resume requires the original learner rank layout")
    if payload.get("schedule") != schedule:
        raise ValueError("RLT resume schedule differs from the saved schedule")
    parent = Path(directory).parent.name
    if parent.startswith("global_step_"):
        if payload.get("checkpoint_step") != int(parent.removeprefix("global_step_")):
            raise ValueError("RLT counters belong to a different checkpoint step")
    return validate_counters(payload.get("counters"))
