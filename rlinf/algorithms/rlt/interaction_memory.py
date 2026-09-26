# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, environment-owned records of completed joint-control interactions.

Records are raw evidence, not learned embeddings. Replay can therefore retain
the exact decision-time context even while the neural reader is being updated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from omegaconf import DictConfig, OmegaConf

MEMORY_OBS_KEYS = ("memory_events", "memory_valid", "memory_query")


@dataclass(frozen=True)
class InteractionMemoryConfig:
    """Joint-space evidence layout shared by environment and policy."""

    proprio_dim: int = 9
    action_dim: int = 8
    chunk_len: int = 10
    brief_size: int = 4
    archive_size: int = 32
    retrieval_size: int = 4
    hidden_dim: int = 64
    num_heads: int = 4
    retain_on_identical_reset: bool = False

    def __post_init__(self) -> None:
        if (
            min(
                self.proprio_dim,
                self.action_dim,
                self.chunk_len,
                self.brief_size,
                self.archive_size,
                self.hidden_dim,
                self.num_heads,
            )
            < 1
        ):
            raise ValueError("Memory dimensions and capacities must be positive")
        if not 0 <= self.retrieval_size <= self.archive_size:
            raise ValueError("retrieval_size must be between zero and archive_size")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

    @property
    def event_dim(self) -> int:
        """Start qpos, executed actions, qpos delta, valid ticks, terminal flags."""
        return (
            2 * self.proprio_dim + self.chunk_len * self.action_dim + self.chunk_len + 2
        )

    @property
    def slots(self) -> int:
        """Number of fixed-shape records carried through replay."""
        return self.brief_size + self.retrieval_size


class InteractionMemory:
    """Own one simulator lane's evidence and explicit attempt boundaries.

    Call ``begin_attempt`` before recording. An explicit retry requires the same
    physical-instance fingerprint. ``snapshot`` returns owned CPU tensors; changing
    the live store never changes an earlier snapshot. Similar records are not
    averaged: conflicting outcomes are evidence, not redundant noise.
    """

    def __init__(self, config: InteractionMemoryConfig) -> None:
        self.config = config
        self.instance_id: str | None = None
        self.attempt = -1
        self._serial = 0
        self._brief: list[tuple[int, torch.Tensor]] = []
        self._archive: list[tuple[int, torch.Tensor]] = []

    def begin_attempt(self, instance_id: str, *, retry: bool = False) -> None:
        """Clear recent context; retain archive only for a verified retry."""
        if not instance_id:
            raise ValueError("A nonempty physical-instance fingerprint is required")
        if retry and instance_id != self.instance_id:
            raise ValueError("Cannot retain memory across different physical instances")
        if not retry:
            self._archive.clear()
            self._serial = 0
            self.attempt = -1
        self.instance_id = instance_id
        self.attempt += 1
        self._brief.clear()

    def append_completed(
        self,
        start: torch.Tensor,
        actions: torch.Tensor,
        end: torch.Tensor,
        *,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        """Record a real executed prefix, including its last terminal tick.

        ``start`` and ``end`` are raw qpos, not OpenPI-normalized states.
        ``actions`` contains only actually executed environment commands.
        """
        c = self.config
        if self.instance_id is None:
            raise RuntimeError("begin_attempt must precede append_completed")
        start, end, actions = (x.detach().float().cpu() for x in (start, end, actions))
        if start.shape != (c.proprio_dim,) or end.shape != start.shape:
            raise ValueError("Expected raw proprio vectors matching proprio_dim")
        if (
            actions.ndim != 2
            or actions.shape[1] != c.action_dim
            or not 1 <= len(actions) <= c.chunk_len
        ):
            raise ValueError(
                "Expected a nonempty executed action prefix [ticks, action_dim]"
            )
        if not all(torch.isfinite(x).all() for x in (start, end, actions)):
            raise ValueError("Interaction evidence must be finite")
        if actions.abs().max() > 1.0001:
            raise ValueError("Expected raw joint-delta commands in [-1, 1]")
        padded = torch.zeros(c.chunk_len, c.action_dim)
        padded[: len(actions)] = actions
        valid_ticks = torch.arange(c.chunk_len) < len(actions)
        event = torch.cat(
            (
                start,
                padded.flatten(),
                end - start,
                valid_ticks.float(),
                torch.tensor([terminated, truncated]).float(),
            )
        )
        row = (self._serial, event)
        self._serial += 1
        self._brief = (self._brief + [row])[-c.brief_size :]
        self._archive = (self._archive + [row])[-c.archive_size :]

    def snapshot(self, query: torch.Tensor) -> dict[str, torch.Tensor]:
        """Read recent and nearest older records without modifying memory.

        Retrieval uses mean squared raw joint-position distance, not task phase
        or a claim of visual/contact similarity. Recent records are not repeated
        in retrieved slots. Missing slots are exactly zero and masked.
        """
        c = self.config
        query = query.detach().float().cpu()
        if self.instance_id is None:
            raise RuntimeError("begin_attempt must precede snapshot")
        if query.shape != (c.proprio_dim,) or not torch.isfinite(query).all():
            raise ValueError("Expected a finite query matching proprio_dim")
        events = torch.zeros(c.slots, c.event_dim)
        valid = torch.zeros(c.slots, dtype=torch.bool)
        for index, (_, event) in enumerate(
            self._brief, start=c.brief_size - len(self._brief)
        ):
            events[index] = event
            valid[index] = True
        recent_ids = {serial for serial, _ in self._brief}
        candidates = [
            (serial, row) for serial, row in self._archive if serial not in recent_ids
        ]
        candidates.sort(
            key=lambda item: (
                float((item[1][: c.proprio_dim] - query).square().mean()),
                -item[0],
            )
        )
        for index, (_, event) in enumerate(
            candidates[: c.retrieval_size], start=c.brief_size
        ):
            events[index] = event
            valid[index] = True
        return {
            "memory_events": events,
            "memory_valid": valid,
            "memory_query": query.clone(),
        }

    def state_dict(self) -> dict:
        """Return a weights-only-loadable snapshot of the complete runtime state."""
        return {
            "format_version": 1,
            "config": asdict(self.config),
            "instance_id": self.instance_id,
            "attempt": self.attempt,
            "serial": self._serial,
            "brief": [(i, x.clone()) for i, x in self._brief],
            "archive": [(i, x.clone()) for i, x in self._archive],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore only the same schema; do not change simulator state here."""
        if state.get("format_version") != 1 or state.get("config") != asdict(
            self.config
        ):
            raise ValueError("Interaction-memory schema mismatch")
        brief, archive = state["brief"], state["archive"]
        if (
            len(brief) > self.config.brief_size
            or len(archive) > self.config.archive_size
        ):
            raise ValueError("Checkpoint exceeds configured memory capacity")
        for _, row in brief + archive:
            if row.shape != (self.config.event_dim,) or not torch.isfinite(row).all():
                raise ValueError("Invalid evidence in memory checkpoint")
        self.instance_id = state["instance_id"]
        self.attempt, self._serial = int(state["attempt"]), int(state["serial"])
        self._brief = [(int(i), x.detach().float().cpu().clone()) for i, x in brief]
        self._archive = [(int(i), x.detach().float().cpu().clone()) for i, x in archive]


def memory_config(config: dict | None) -> InteractionMemoryConfig | None:
    """Parse opt-in configuration; unknown fields fail rather than being ignored."""
    if not config or not config.get("enabled", False):
        return None
    return InteractionMemoryConfig(
        **{key: value for key, value in config.items() if key != "enabled"}
    )


def copy_memory_observation(source: dict, target: dict) -> None:
    """Copy the complete optional memory schema, rejecting partial messages."""
    present = [key in source for key in MEMORY_OBS_KEYS]
    if any(present) and not all(present):
        raise ValueError("Incomplete interaction-memory observation")
    if all(present):
        for key in MEMORY_OBS_KEYS:
            target[key] = source[key].detach().clone()


class JointMemoryCollector:
    """Own isolated memory lanes at the simulator's executed-action boundary."""

    def __init__(self, config: InteractionMemoryConfig, num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.config = config
        self.lanes = [InteractionMemory(config) for _ in range(num_envs)]
        self.last_state = torch.zeros(num_envs, config.proprio_dim)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool)

    def reset(
        self, indices: list[int], instance_ids: list[str], states: torch.Tensor
    ) -> None:
        """Begin lanes at actual reset states; retry only matching fingerprints."""
        if len(indices) != len(instance_ids) or states.shape != self.last_state.shape:
            raise ValueError(
                "Reset indices, instance fingerprints, and full state batch must align"
            )
        if len(set(indices)) != len(indices) or any(
            i < 0 or i >= len(self.lanes) for i in indices
        ):
            raise ValueError("Reset indices must be unique valid lane indices")
        if not torch.isfinite(states).all() or any(not value for value in instance_ids):
            raise ValueError("Reset states must be finite")
        for index, instance in zip(indices, instance_ids, strict=True):
            lane = self.lanes[index]
            retry = (
                self.config.retain_on_identical_reset and lane.instance_id == instance
            )
            lane.begin_attempt(instance, retry=retry)
            self.last_state[index] = states[index].detach().float().cpu()
            self.initialized[index] = True

    def complete(
        self,
        actions: torch.Tensor,
        end: torch.Tensor,
        valid_ticks: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Append only executed prefixes, before any automatic environment reset."""
        n, c = len(self.lanes), self.config
        if (
            actions.shape != (n, c.chunk_len, c.action_dim)
            or valid_ticks.shape != (n, c.chunk_len)
            or end.shape != self.last_state.shape
        ):
            raise ValueError("Completed chunk has an incompatible shape")
        if (
            valid_ticks.dtype != torch.bool
            or terminated.shape != valid_ticks.shape
            or truncated.shape != valid_ticks.shape
        ):
            raise ValueError("Expected per-tick validity and termination masks")
        if not self.initialized.all():
            raise RuntimeError("All memory lanes must be reset before stepping")
        counts = valid_ticks.cpu().sum(dim=1)
        expected = torch.arange(c.chunk_len)[None] < counts[:, None]
        if not torch.equal(valid_ticks.cpu(), expected):
            raise ValueError("Executed controls must form a contiguous prefix")
        executed = actions[valid_ticks.to(actions.device)]
        if not torch.isfinite(end).all() or not torch.isfinite(executed).all():
            raise ValueError("Interaction evidence must be finite")
        if executed.numel() and executed.abs().max() > 1.0001:
            raise ValueError("Expected raw joint-delta commands in [-1, 1]")
        for index, lane in enumerate(self.lanes):
            count = int(counts[index])
            if count:
                lane.append_completed(
                    self.last_state[index],
                    actions[index, :count],
                    end[index],
                    terminated=bool(terminated[index, :count].any()),
                    truncated=bool(truncated[index, :count].any()),
                )
                self.last_state[index] = end[index].detach().float().cpu()

    def snapshot(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return batch snapshots in lane order on the observation's device."""
        if states.shape != self.last_state.shape or not self.initialized.all():
            raise ValueError("Memory snapshot requires initialized matching lanes")
        snapshots = [
            lane.snapshot(state) for lane, state in zip(self.lanes, states, strict=True)
        ]
        return {
            key: torch.stack([item[key] for item in snapshots]).to(states.device)
            for key in MEMORY_OBS_KEYS
        }


def validate_interaction_memory_cfg(cfg: DictConfig) -> None:
    """Reject unsupported algorithms and divergent env/actor/rollout schemas."""
    actor = memory_config(OmegaConf.select(cfg, "actor.model.interaction_memory"))
    rollout = memory_config(OmegaConf.select(cfg, "rollout.model.interaction_memory"))
    envs = [cfg.env[key] for key in ("train", "eval") if key in cfg.env]
    env_configs = [memory_config(env.get("interaction_memory")) for env in envs]
    if actor is None and rollout is None and not any(env_configs):
        return
    if actor is None or rollout != actor or any(env != actor for env in env_configs):
        raise ValueError(
            "Interaction memory must have identical actor, rollout and environment configurations"
        )
    if (
        cfg.algorithm.loss_type != "rlt_ac"
        or cfg.actor.model.model_type != "rlt_mlp_policy"
    ):
        raise ValueError(
            "Interaction memory currently supports rlt_ac with rlt_mlp_policy only"
        )
    if cfg.algorithm.get("q_head_type", "default") != "default":
        raise ValueError(
            "Interaction memory currently requires the baseline twin-Q head"
        )
    if cfg.algorithm.get("target_update_type", "all") != "all":
        raise ValueError("Memory reader requires target_update_type=all")
    if cfg.rollout.get("enable_cuda_graph", False):
        raise ValueError("Interaction-memory CUDA graphs are not validated")
    for env in envs:
        if env.env_type != "maniskill_rlt" or env.wrap_obs_mode != "rlt_openpi_joint":
            raise ValueError(
                "Interaction memory requires ManiSkill RLT joint observations"
            )
        if env.init_params.control_mode != "pd_joint_delta_pos":
            raise ValueError("Interaction memory requires pd_joint_delta_pos controls")
        if actor.retain_on_identical_reset and not env.use_fixed_reset_state_ids:
            raise ValueError("Persistent retries require fixed reset-state IDs")
    model = cfg.actor.model
    if (actor.proprio_dim, actor.action_dim, actor.chunk_len) != (
        model.proprio_dim,
        model.action_dim,
        model.num_action_chunks,
    ):
        raise ValueError("Memory dimensions must match the RLT actor")
