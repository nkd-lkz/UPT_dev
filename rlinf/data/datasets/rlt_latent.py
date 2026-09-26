# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Episode-local latent windows and frozen-encoder provenance for RLT."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


def file_sha256(path: str | Path) -> str:
    """Hash file contents, never paths or modification times."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_contract(
    model_cfg: DictConfig, *, control_mode: str, control_freq: int
) -> dict:
    """Fingerprint the exact frozen checkpoint, preprocessing and control units."""
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights

    cfg = OmegaConf.to_container(model_cfg, resolve=True)
    weights = resolve_full_weights(Path(cfg["model_path"]))
    if weights is None:
        raise ValueError(
            "A completed RLT Stage 1 full_weights.pt is required, not pi05_base"
        )
    data = dict(cfg.get("openpi_data", {}))
    stats = Path(data.pop("norm_stats_path", ""))
    if not stats.is_file():
        raise ValueError(
            "Set openpi_data.norm_stats_path to an explicit norm_stats.json"
        )
    data.pop("repo_id", None)  # Storage location does not define preprocessing.
    cfg.pop("model_path")
    cfg["openpi_data"] = data
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return {
        "version": 1,
        "weights_sha256": file_sha256(weights),
        "norm_stats_sha256": file_sha256(stats),
        "model_config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "control_mode": control_mode,
        "control_freq": int(control_freq),
        "action_space": "environment_pd_joint_delta_pos",
        "proprio_space": "openpi_normalized",
    }


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    """Publish a complete artifact by same-directory rename (also on NAS)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def episode_split(episode_id: str, *, seed: int, validation_fraction: float) -> str:
    """Assign whole episodes reproducibly, independent of filesystem ordering."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    value = int(hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()[:8], 16)
    return "validation" if value / 2**32 < validation_fraction else "train"


class RLTLatentDataset(Dataset):
    """Small in-memory feature cache; raw images remain outside learner memory.

    Each episode contains z_rl [T,Z], proprio [T,P], actions [T,A],
    frame_index [T] and a single task_index. No window crosses an episode or
    frame discontinuity. The final action has no observed successor and is
    excluded. Validation is split by episode, never by overlapping windows.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        horizons: tuple[int, ...],
        split: str,
        seed: int = 2026,
        validation_fraction: float = 0.1,
    ) -> None:
        cache_dir = Path(cache_dir)
        self.manifest = torch.load(
            cache_dir / "manifest.pt", map_location="cpu", weights_only=True
        )
        if self.manifest.get("format_version") != 1 or not self.manifest.get(
            "complete"
        ):
            raise ValueError("Cache is incomplete or uses an unsupported format")
        if split not in ("train", "validation", "all"):
            raise ValueError("split must be train, validation or all")
        self.horizons = tuple(horizons)
        if (
            not horizons
            or tuple(sorted(set(horizons))) != tuple(horizons)
            or min(horizons) < 1
        ):
            raise ValueError("horizons must be positive, unique and increasing")
        self.chunk_len = max(horizons)
        self.episodes, self.index = [], []
        self.episode_ids = []
        for entry in self.manifest["episodes"]:
            if (
                split != "all"
                and episode_split(
                    entry["id"], seed=seed, validation_fraction=validation_fraction
                )
                != split
            ):
                continue
            path = cache_dir / entry["file"]
            if path.parent != cache_dir or not path.name.endswith(".pt"):
                raise ValueError(
                    "Episode files must be direct children of the cache directory"
                )
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Corrupted cache file: {path}")
            episode = torch.load(path, map_location="cpu", weights_only=True)
            self.validate_episode(episode)
            position = len(self.episodes)
            self.index.extend(
                (position, frame) for frame in range(len(episode["z_rl"]) - 1)
            )
            self.episodes.append(episode)
            self.episode_ids.append(entry["id"])
        if not self.index:
            raise ValueError(
                f"Empty {split} split; cache more episodes or adjust split seed"
            )

    @staticmethod
    def validate_episode(episode: dict) -> None:
        """Reject missing frames, nonfinite features and actions in the wrong units."""
        n = len(episode["z_rl"])
        if n < 2:
            raise ValueError("Episodes need at least two observations")
        for key in ("z_rl", "proprio", "actions"):
            value = episode[key]
            if value.ndim != 2 or len(value) != n or not torch.isfinite(value).all():
                raise ValueError(f"Invalid episode field {key}")
        frames = episode["frame_index"].reshape(-1)
        if len(frames) != n or not torch.all(frames[1:] - frames[:-1] == 1):
            raise ValueError("Episode frame indices must be consecutive")
        if episode["actions"].abs().max() > 1.001:
            raise ValueError(
                "Expected raw environment actions in [-1, 1], not normalized SFT actions"
            )
        task = torch.as_tensor(episode["task_index"]).reshape(-1)
        if task.numel() != 1:
            raise ValueError("One task_index per episode is required")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_idx, t = self.index[index]
        episode = self.episodes[episode_idx]
        available = len(episode["z_rl"]) - 1 - t
        count = min(available, self.chunk_len)
        actions = torch.zeros(self.chunk_len, episode["actions"].shape[-1])
        actions[:count] = episode["actions"][t : t + count]
        # Invalid targets are finite placeholders, never a padded training target.
        target_indices = [t + min(h, available) for h in self.horizons]
        return {
            "z_rl": episode["z_rl"][t].float(),
            "proprio": episode["proprio"][t].float(),
            "actions": actions,
            "future_z": episode["z_rl"][target_indices].float(),
            "future_proprio": episode["proprio"][target_indices].float(),
            "valid": torch.tensor([h <= available for h in self.horizons]),
            "action_valid": torch.arange(self.chunk_len) < count,
        }
