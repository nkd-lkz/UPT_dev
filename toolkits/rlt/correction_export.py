# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Recover episode-grouped correction evidence from a completed pilot replay."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.correction_data import export_episode, finalize_corrections
from rlinf.utils.logging import get_logger

logger = get_logger()


def export_checkpoint(run: Path, replay: Path, output: Path) -> dict:
    """Export only a complete, un-evicted, synchronous single-environment replay.

    Recorded suffixes end at actual done flags; no episode boundary is inferred
    from a policy version. Episodes with no recorded transitions are absent from
    a replay checkpoint and are explicitly excluded from this cache's inventory.
    Source checkpoints are read-only. An interrupted export remains unsealed.
    """
    launch = json.loads((run / "launch.json").read_text())
    cfg = OmegaConf.to_container(OmegaConf.load(run / "resolved.yaml"), resolve=True)
    train = cfg["env"]["train"]
    if (
        launch.get("complete") is not True
        or launch.get("exit_code") != 0
        or train["total_num_envs"] != 1
        or train.get("auto_reset", True)
        or train.get("rollout_epoch", 1) != 1
        or cfg["runner"].get("resume_dir")
        or cfg["runner"].get("only_eval", False)
    ):
        raise ValueError("Require completed fresh single-env synchronous training")
    if replay.resolve().name != "rank_0" or not replay.resolve().is_relative_to(
        run.resolve()
    ):
        raise ValueError("Require rank_0 replay inside the declared training run")
    metadata = json.loads((replay / "metadata.json").read_text())
    index = json.loads((replay / "trajectory_index.json").read_text())
    ids = index["trajectory_id_list"]
    if (
        metadata["trajectory_format"] != "pt"
        or not ids
        or ids != list(range(metadata["trajectory_counter"]))
        or metadata["size"] != len(ids)
        or metadata["total_samples"] != len(ids)
        or set(index["trajectory_index"]) != {str(i) for i in ids}
    ):
        raise ValueError(
            "Replay was evicted, incomplete, or not one sample per transition"
        )
    contract = {
        "model": cfg["actor"]["model"],
        "feature_model": cfg["rollout"]["rlt_feature_model"],
        "environment": train,
        "algorithm": {
            key: cfg["algorithm"][key] for key in ("gamma", "reference_dropout_prob")
        },
        "reference_source": "frozen_vla_pre_action",
        "action_space": "environment_pd_joint_delta_pos",
        "terminal_bootstrap": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    records, pending, episode_count = [], [], 0
    for i in ids:
        info = index["trajectory_index"][str(i)]
        if info["num_samples"] != 1 or info["max_episode_length"] != 1:
            raise ValueError("Only transition replay can be exported")
        name = f"trajectory_{i}_{info['model_weights_id']}.pt"
        path = replay / name
        if path.resolve().parent != replay.resolve():
            raise ValueError("Invalid replay file name")
        raw = path.read_bytes()
        # weights_only limits deserialization to tensors and ordinary containers.
        import io

        data = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if data.get("model_weights_id") != info["model_weights_id"]:
            raise ValueError("Replay index and serialized identity disagree")
        if data["actions"].shape != (1, 1, 80) or data["dones"].shape != (1, 1, 10):
            raise ValueError("Expected a single Panda action chunk")
        if pending and any(
            not torch.equal(pending[-1].next_obs[key], data["curr_obs"][key])
            for key in ("z_rl", "proprio", "ref_chunk")
        ):
            (output / "failure.json").write_text(
                json.dumps(
                    {
                        "complete": False,
                        "reason": "nonterminal_recording_gap",
                        "next_transition_id": i,
                        "interpretation": "A recording gap is not evidence of a TD endpoint error. This checkpoint cannot establish complete episode grouping; use collection-time correction export.",
                    },
                    indent=2,
                )
                + "\n"
            )
            raise ValueError(
                "Recorded suffix is discontinuous before its terminal chunk"
            )
        pending.append(SimpleNamespace(**data))
        records.append({"file": name, "sha256": hashlib.sha256(raw).hexdigest()})
        if data["dones"].any():
            export_episode(output, f"{episode_count:06d}", pending, contract)
            pending = []
            episode_count += 1
        if (i + 1) % 500 == 0:
            logger.info("Exported %d/%d recorded transitions", i + 1, len(ids))
    if pending:
        raise ValueError("Replay ends in an unfinished recorded episode")
    report = {
        "complete": True,
        "source_run": str(run.resolve()),
        "source_replay": str(replay.resolve()),
        "recorded_episodes": episode_count,
        "transitions": len(ids),
        "files": records,
        "limitation": "Recorded suffixes only. Empty replay episodes and reference-only prefixes cannot be reconstructed; do not infer total interaction cost from this cache.",
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    finalize_corrections(output)
    return report


def main() -> None:
    """Export completed local replay evidence on CPU without altering weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = export_checkpoint(args.run, args.replay, args.output)
    logger.info(
        "Sealed %d recorded episodes, %d transitions",
        result["recorded_episodes"],
        result["transitions"],
    )


if __name__ == "__main__":
    main()
