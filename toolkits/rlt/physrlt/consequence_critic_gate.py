# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Test whether a critic uses candidate physical consequences across replays."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from toolkits.rlt.libero_acceptance_learning import AcceptanceLearner
from toolkits.rlt.physrlt.control import (
    FrozenReadout,
    replay_trace_sha256,
    validate_fixed_replay,
)
from toolkits.rlt.physrlt.critic_audit import audit_replay
from toolkits.rlt.pi05_baseline import configure_source, write_json


class SignedConsequence(nn.Module):
    """Apply a predeclared sign to the candidate-relative physical channel."""

    def __init__(self, conditioner: nn.Module, sign: int) -> None:
        super().__init__()
        if sign not in (-1, 0, 1):
            raise ValueError("Consequence sign must be -1, 0 or 1")
        self.conditioner = conditioner
        self.sign = sign

    def train(self, mode: bool = True):
        """Keep the wrapped frozen physical predictor in evaluation mode."""
        super().train(False)
        self.conditioner.eval()
        return self

    def forward(self, token, action, reference, physical_token=None):
        conditioned = self.conditioner(token, action, reference, physical_token)
        base, consequence = conditioned.chunk(2, dim=-1)
        return torch.cat((base, self.sign * consequence), dim=-1)


def parse_evaluations(specs: list[str]) -> dict[str, tuple[Path, Path]]:
    """Parse unique ``LABEL=BASE_REPLAY,PHYSICAL_REPLAY`` specifications."""
    result = {}
    for spec in specs:
        label, separator, raw_paths = spec.partition("=")
        base, path_separator, physical = raw_paths.partition(",")
        if (
            not separator
            or not path_separator
            or not label
            or not base
            or not physical
            or label in result
        ):
            raise ValueError(
                "Evaluations require unique LABEL=BASE_REPLAY,PHYSICAL_REPLAY entries"
            )
        result[label] = (Path(base), Path(physical))
    return result


def load_replay(base_path: Path, physical_path: Path) -> list[dict]:
    """Merge semantic and physical views of one immutable executed trace."""
    replay = torch.load(base_path, map_location="cpu", weights_only=True)
    physical = torch.load(physical_path, map_location="cpu", weights_only=True)
    base_summary = validate_fixed_replay(replay)
    physical_summary = validate_fixed_replay(physical)
    if (
        len(replay) != len(physical)
        or base_summary["trace_sha256"] != physical_summary["trace_sha256"]
    ):
        raise ValueError("Semantic and physical replay traces differ")
    merged = []
    for row, physical_row in zip(replay, physical, strict=True):
        critic_z = physical_row.get("critic_z", physical_row["z"])
        next_critic_z = physical_row.get("next_critic_z", physical_row["next_z"])
        if (
            critic_z.shape != row["z"].shape
            or next_critic_z.shape != row["next_z"].shape
        ):
            raise ValueError("Semantic and physical replay token shapes differ")
        merged.append(row | {"critic_z": critic_z, "next_critic_z": next_critic_z})
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"actor_z", "next_actor_z"}
        }
        for row in merged
    ]


def run(args: argparse.Namespace) -> dict:
    """Fit matched critic arms and audit unseen executed replay traces."""
    if args.output.exists() or args.steps < 1 or not args.seeds:
        raise ValueError("Require a new output, positive steps and optimizer seeds")
    configure_source(args.source)
    train = load_replay(args.train_base_replay, args.train_physical_replay)
    evaluations = {
        label: load_replay(*paths)
        for label, paths in parse_evaluations(args.evaluations).items()
    }
    trace_hashes = {"train": replay_trace_sha256(train)} | {
        label: replay_trace_sha256(replay) for label, replay in evaluations.items()
    }
    if len(set(trace_hashes.values())) != len(trace_hashes):
        raise ValueError("Train and evaluation replays must have distinct traces")
    replay_summaries = {"train": validate_fixed_replay(train)} | {
        label: validate_fixed_replay(replay) for label, replay in evaluations.items()
    }
    if any(
        summary["positive_transitions"] == 0 for summary in replay_summaries.values()
    ):
        raise ValueError(
            "Every train and evaluation replay needs a positive transition"
        )
    readout = FrozenReadout(args.base_readout, consequence=args.consequence_readout)
    rows = []
    for seed in args.seeds:
        for arm, sign in (("correct", 1), ("reversed", -1), ("zero", 0)):
            random.seed(seed)
            torch.manual_seed(seed)
            conditioner = SignedConsequence(
                readout.candidate_conditioner("relative_concat"), sign
            )
            learner = AcceptanceLearner(
                "cpu",
                "q_bc",
                ref_dropout=0,
                bottleneck_dim=128,
                chunk_len=10,
                gripper_threshold=0,
                critic_conditioner=conditioner,
                critic_conditioning="candidate_relative_concat",
                critic_bottleneck_dim=256,
                correction_limit=0.1,
                correction_axes=(0,),
                residual_output=True,
                critic_consequence_dueling=True,
            )
            trace = []
            random.seed(seed)
            torch.manual_seed(seed)
            for step in range(args.steps):
                metrics = learner.update(train)
                if step in {0, args.steps - 1}:
                    trace.append({"step": step + 1, **metrics})
            for label, replay in evaluations.items():
                rows.append(
                    {
                        "optimizer_seed": seed,
                        "arm": arm,
                        "evaluation": label,
                        "train_steps": args.steps,
                        "trace": trace,
                        **audit_replay(learner, replay),
                    }
                )
    result = {
        "state": "complete",
        "protocol": (
            "same train replay, architecture, optimizer batches and held-out "
            "replays; only the physical consequence sign changes"
        ),
        "train_base_replay": str(args.train_base_replay),
        "train_physical_replay": str(args.train_physical_replay),
        "trace_sha256": trace_hashes,
        "replay_summaries": replay_summaries,
        "base_readout": str(args.base_readout),
        "consequence_readout": str(args.consequence_readout),
        "seeds": args.seeds,
        "rows": rows,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-base-replay", type=Path, required=True)
    parser.add_argument("--train-physical-replay", type=Path, required=True)
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--base-readout", type=Path, required=True)
    parser.add_argument("--consequence-readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=105)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 47, 59])
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
