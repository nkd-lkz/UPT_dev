# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Episode-grouped command caches for correction-learning diagnostics."""

import hashlib
import json
from pathlib import Path

import torch

FEATURES = ("z_rl", "proprio", "ref_chunk")


def controller_actions(submitted: torch.Tensor, contract: dict) -> torch.Tensor:
    """Map submissions to the verified Panda controller's normalized input.

    ManiSkill clips both the arm and gripper before scaling to joint targets.
    Keep original submissions as evidence; never alter the sealed cache.
    """
    config = contract.get("environment", {}).get("init_params", {})
    if (
        config.get("id")
        not in {"PegInsertionSideWideClearance-v1", "PegInsertionSide-v1"}
        or config.get("control_mode") != "pd_joint_delta_pos"
        or config.get("robot_uids", "panda") != "panda"
        or not torch.isfinite(submitted).all()
    ):
        raise ValueError(
            "Unverified controller mapping or nonfinite submitted commands"
        )
    return submitted.clamp(-1, 1)


def valid_prefix(dones: torch.Tensor) -> torch.Tensor:
    """Include the terminal command, excluding frozen padding after it."""
    return dones.long().cumsum(-1) - dones.long() == 0


def export_episode(
    directory: Path, episode_id: str, transitions: list, contract: dict
) -> None:
    """Persist replay transitions and metadata without modifying policy inputs.

    Each invocation must contain one complete single-env rollout. Empty replay
    episodes remain in the inventory. Terminal transitions are not bootstrapped.
    Commands are those submitted to the environment; the controller clips them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if not episode_id.isdecimal() or (directory / "complete.json").exists():
        raise ValueError("Require a numeric episode ID and an unfinished cache")
    metadata = directory / "contract.json"
    payload = {"format": "rlt_corrections_v1", **contract}
    if metadata.exists():
        if json.loads(metadata.read_text()) != payload:
            raise ValueError("Correction feature contract changed during collection")
    else:
        with metadata.open("x") as stream:
            json.dump(payload, stream, indent=2)
    rows = []
    for transition in transitions:
        row = {
            side: {
                key: getattr(transition, side)[key]
                .detach()
                .cpu()
                .float()
                .reshape(1, -1)
                for key in FEATURES
            }
            for side in ("curr_obs", "next_obs")
        }
        row["actions"] = transition.actions.detach().cpu().float().reshape(1, 10, 8)
        row["rewards"] = transition.rewards.detach().cpu().float().reshape(1, 10)
        row["dones"] = transition.dones.detach().cpu().bool().reshape(1, 10)
        planner = transition.forward_inputs.get(
            "planner_flags", torch.zeros(10, dtype=torch.bool)
        )
        row["planner"] = planner.detach().cpu().bool().reshape(1, 10)
        rows.append(row)

    def concatenate(key: str):
        if key in ("curr_obs", "next_obs"):
            return {name: torch.cat([r[key][name] for r in rows]) for name in FEATURES}
        return torch.cat([r[key] for r in rows])

    data = {key: concatenate(key) for key in rows[0]} if rows else {}
    data.update(episode_id=episode_id, transitions=len(rows))
    if rows:
        data["valid"] = valid_prefix(data["dones"])
        data["success"] = bool((data["rewards"] * data["valid"]).sum() > 0)
    path = directory / f"episode_{episode_id}.pt"
    with path.open("xb") as stream:
        torch.save(data, stream)
    # A crash before this append leaves an unindexed file, never a usable episode.
    with (directory / "index.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "id": episode_id,
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            + "\n"
        )


def finalize_corrections(directory: Path) -> None:
    """Seal an episode inventory only after the collection run succeeds."""
    digest = hashlib.sha256(
        (directory / "contract.json").read_bytes()
        + (directory / "index.jsonl").read_bytes()
    ).hexdigest()
    with (directory / "complete.json").open("x") as stream:
        json.dump({"sha256": digest}, stream)


def load_corrections(directory: Path) -> tuple[list[dict], dict, str]:
    """Verify checksums and keep train/validation splits at episode boundaries."""
    contract_bytes = (directory / "contract.json").read_bytes()
    contract = json.loads(contract_bytes)
    if contract.get("format") != "rlt_corrections_v1":
        raise ValueError("Unsupported correction cache format")
    index_bytes = (directory / "index.jsonl").read_bytes()
    digest = hashlib.sha256(contract_bytes + index_bytes).hexdigest()
    complete = directory / "complete.json"
    if (
        not complete.is_file()
        or json.loads(complete.read_text()).get("sha256") != digest
    ):
        raise ValueError("Correction cache is unfinished or its inventory changed")
    entries = [json.loads(line) for line in index_bytes.splitlines() if line]
    if len({row["id"] for row in entries}) != len(entries):
        raise ValueError("Duplicate correction episode IDs")
    episodes = []
    for entry in entries:
        path = directory / entry["file"]
        if path.resolve().parent != directory.resolve():
            raise ValueError("Correction file must belong to the cache")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Correction cache checksum mismatch")
        data = torch.load(path, weights_only=True, map_location="cpu")
        n = data["transitions"]
        if data["episode_id"] != entry["id"] or not isinstance(n, int) or n < 0:
            raise ValueError("Invalid correction episode identity/count")
        if not n:
            continue
        for side in ("curr_obs", "next_obs"):
            for key, size in (("z_rl", 2048), ("proprio", 9), ("ref_chunk", 80)):
                value = data[side][key]
                if value.shape != (n, size) or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid correction feature {side}/{key}")
        for key in ("dones", "planner", "valid", "rewards"):
            if data[key].shape != (n, 10):
                raise ValueError(f"Invalid correction shape: {key}")
        if any(data[key].dtype != torch.bool for key in ("dones", "planner", "valid")):
            raise ValueError("Correction masks must be boolean")
        if (
            data["actions"].shape != (n, 10, 8)
            or not torch.isfinite(data["actions"]).all()
            or not torch.isfinite(data["rewards"]).all()
        ):
            raise ValueError(
                "Correction commands and rewards must be finite with declared shapes"
            )
        data["submitted_actions"] = data["actions"]
        data["actions"] = controller_actions(data["submitted_actions"], contract)
        if not torch.equal(data["valid"], valid_prefix(data["dones"])):
            raise ValueError("Invalid terminal padding mask")
        if data["dones"][:-1].any():
            raise ValueError(
                "Correction episode contains transitions after termination"
            )
        if data["success"] != bool((data["rewards"] * data["valid"]).sum() > 0):
            raise ValueError("Correction outcome and executed rewards disagree")
        episodes.append(data)
    return episodes, contract, digest


def correction_batch(episodes: list[dict]) -> dict:
    """Keep failed planner transitions for TD, but exclude them from BC labels."""
    if not episodes:
        raise ValueError("Empty correction split")
    batch = {
        side: {key: torch.cat([e[side][key] for e in episodes]) for key in FEATURES}
        for side in ("curr_obs", "next_obs")
    }
    for key in ("actions", "submitted_actions", "rewards", "dones", "valid", "planner"):
        batch[key] = torch.cat([e[key] for e in episodes])
    accepted = (
        torch.cat([e["planner"] & bool(e["success"]) for e in episodes])
        & batch["valid"]
    )
    batch["accepted_planner"] = accepted
    batch["bc_mask"] = batch["valid"] & (~batch["planner"] | accepted)
    batch["target"] = torch.where(
        accepted[..., None],
        batch["actions"],
        batch["curr_obs"]["ref_chunk"].reshape(-1, 10, 8),
    )
    return batch
