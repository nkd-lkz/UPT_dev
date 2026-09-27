# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Ordered, authenticated RPC operations for the online RLT learner."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from .protocol import CONTRACT, decode_observation, validate_actions


def frozen_feature_identity(stage1: Path, dataset: Path) -> str:
    """Identify immutable Stage1 exports and exact normalization contents."""
    weights = stage1 / "model_state_dict/full_weights.pt"
    return json.dumps(
        {
            "weights": str(weights.resolve()),
            "size": weights.stat().st_size,
            "mtime_ns": weights.stat().st_mtime_ns,
            "stats_sha256": hashlib.sha256(
                (dataset / "norm_stats.json").read_bytes()
            ).hexdigest(),
        },
        sort_keys=True,
    )


def observation_id(observation: dict) -> str:
    """Hash exactly the pixels and float32 qpos used for feature extraction."""
    digest = hashlib.sha256()
    for key in ("state", "main_image", "wrist_image"):
        digest.update(observation[key].tobytes())
    return digest.hexdigest()


class OnlineService:
    """Serialize ingestion, updates and predictions for one client session.

    Retries of the last acknowledged sequence are idempotent. Changing the
    payload for that sequence or skipping a sequence is rejected. A fresh
    client session may begin only through an explicit begin operation.
    """

    def __init__(
        self, learner, extract: Callable, directory: Path, feature_id: str
    ) -> None:
        self.learner, self.extract = learner, extract
        self.directory, self.feature_id = directory, feature_id
        self.session: str | None = None
        self.sequence = -1
        self.digest = self.last_next = ""
        self.episode = -1
        self.done = False
        self.faulted = False
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def metadata(self) -> dict:
        """Return progress and the frozen-feature identity for resume."""
        return {
            "session": self.session,
            "sequence": self.sequence,
            "digest": self.digest,
            "last_next": self.last_next,
            "episode": self.episode,
            "done": self.done,
            "feature_id": self.feature_id,
        }

    def restore(self, metadata: dict) -> None:
        """Reject checkpoints built from a different Stage1 export or stats."""
        if metadata["feature_id"] != self.feature_id:
            raise ValueError("Frozen feature identity differs from checkpoint")
        for key in ("session", "sequence", "digest", "last_next", "episode", "done"):
            setattr(self, key, metadata[key])

    def checkpoint(self) -> None:
        """Save the complete learner and accepted-sequence state."""
        if self.faulted:
            raise RuntimeError("Learner faulted; retain the last good checkpoint")
        self.learner.save(self.directory / "learner.pt", self.metadata())

    def _features(self, obs: dict) -> dict:
        key = observation_id(obs)
        if key not in self.cache:
            self.cache[key] = self.learner.features(self.extract(obs))
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def __call__(self, message: dict) -> dict:
        """Handle a validated RPC without loading any remote Python objects."""
        op = message.get("op")

        def status():
            return {
                "contract": CONTRACT,
                "online": True,
                "horizon": 1,
                "model_id": f"online-v{self.learner.version}",
                "metrics": self.learner.status(),
                "sequence": self.sequence,
                "faulted": self.faulted,
            }

        if op in {"health", "status"}:
            return status()
        if self.faulted:
            raise RuntimeError("Learner faulted; restart from last good checkpoint")
        if op == "begin":
            session = message.get("session")
            if not isinstance(session, str) or not 8 <= len(session) <= 64:
                raise ValueError("Expected bounded client session id")
            if self.session is not None and self.session != session:
                raise ValueError(
                    "Another client session owns this run; start a fresh server run"
                )
            self.session = session
            return status()
        if message.get("session") != self.session or self.session is None:
            raise ValueError("Begin a client session first")
        if op == "predict":
            start = time.monotonic()
            obs = decode_observation(message)
            actions, source = self.learner.predict(self._features(obs))
            return {
                **status(),
                "actions": validate_actions(actions.numpy()).tolist(),
                "policy_source": source,
                "inference_ms": (time.monotonic() - start) * 1000,
            }
        if op != "observe":
            raise ValueError("Unknown online operation")
        seq, episode = message.get("sequence"), message.get("episode")
        if type(seq) is not int or seq < 0 or type(episode) is not int or episode < 0:
            raise ValueError("Invalid transition index")
        payload = {k: v for k, v in message.items() if k not in {"token", "request_id"}}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if seq == self.sequence:
            if digest != self.digest:
                raise ValueError("Conflicting duplicate transition")
            return {**status(), "duplicate": True}
        if seq != self.sequence + 1 or episode < self.episode:
            raise ValueError("Out-of-order transition")
        obs = decode_observation(message["observation"])
        next_obs = decode_observation(message["next_observation"])
        if episode == self.episode and (
            self.done or observation_id(obs) != self.last_next
        ):
            raise ValueError("Transition is not contiguous within the episode")
        for key in ("human", "terminated", "truncated"):
            if type(message.get(key)) is not bool:
                raise ValueError(f"Expected bool {key}")
        reward = message.get("reward")
        if (
            type(reward) not in (int, float)
            or not math.isfinite(reward)
            or reward not in (0, 1)
        ):
            raise ValueError("Expected raw sparse task reward 0 or 1")
        policy_version = message.get("policy_version")
        if (
            type(policy_version) is not int
            or not -1 <= policy_version <= self.learner.version
        ):
            raise ValueError("Invalid behavior policy version")
        action = validate_actions([message["action"]])[0]
        record = {
            "obs": self._features(obs),
            "next_obs": self._features(next_obs),
            "action": action,
            "reward": reward,
            "terminated": message["terminated"],
            "truncated": message["truncated"],
            "human": message["human"],
        }
        try:
            metrics = self.learner.observe(record)
        except BaseException:
            # An optimizer may have mutated before failing. Do not retry an
            # ambiguous insertion or overwrite a previously valid checkpoint.
            self.faulted = True
            raise
        self.sequence, self.episode, self.digest = seq, episode, digest
        self.last_next = observation_id(next_obs)
        self.done = message["terminated"] or message["truncated"]
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "sequence": seq,
                        "episode": episode,
                        "human": message["human"],
                        "behavior_version": policy_version,
                        "time": time.time(),
                        **metrics,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        if self.learner.accepted % self.learner.cfg["checkpoint_interval"] == 0:
            self.checkpoint()
        return {**status(), "duplicate": False}
