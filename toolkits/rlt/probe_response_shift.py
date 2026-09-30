# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Test past-only response forgetting in a synthetic plant, not a simulator."""

import argparse
import json
import math
from pathlib import Path

import torch

from rlinf.algorithms.rlt.interaction_memory import InteractionMemoryConfig
from rlinf.models.embodiment.modules.rlt_memory_encoder import response_features


class ResponseHistory:
    """Forget stale records after repeated errors in completed interactions.

    This CPU diagnostic has a fixed threshold in joint-change units. It is not
    a calibrated contact detector. Two high errors reset older history only
    after their outcomes arrive; no prediction can inspect its own outcome.
    """

    def __init__(self, *, adaptive: bool = False, threshold: float = 0.015) -> None:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be positive and finite")
        self.adaptive = adaptive
        self.threshold = threshold
        self.commands = torch.empty(0, 7)
        self.outcomes = torch.empty(0, 7)
        self.high_errors = 0

    def predict(
        self, command: torch.Tensor, *, half_life: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a response prediction without mutating the evidence history."""
        return predict_from_history(
            self.commands, self.outcomes, command, half_life=half_life
        )

    def observe(self, command: torch.Tensor, outcome: torch.Tensor) -> bool:
        """Add an observed transition; return whether older evidence was dropped."""
        if command.shape != (7,) or outcome.shape != (7,):
            raise ValueError("Expected seven-joint completed transition")
        if not torch.isfinite(command).all() or not torch.isfinite(outcome).all():
            raise ValueError("Cannot store nonfinite completed evidence")
        command, outcome = command.detach().cpu(), outcome.detach().cpu()
        prediction, _ = self.predict(command)
        unexpected = (
            len(self.commands) >= 4
            and (prediction - outcome).square().mean().sqrt() > self.threshold
        )
        self.high_errors = self.high_errors + 1 if unexpected else 0
        self.commands = torch.cat((self.commands, command[None]))[-8:]
        self.outcomes = torch.cat((self.outcomes, outcome[None]))[-8:]
        reset = self.adaptive and self.high_errors >= 2
        if reset:
            self.commands = self.commands[-2:].clone()
            self.outcomes = self.outcomes[-2:].clone()
            self.high_errors = 0
        return reset


def predict_from_history(
    commands: torch.Tensor,
    outcomes: torch.Tensor,
    proposal: torch.Tensor,
    *,
    half_life: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read only completed records; chronological inputs end at the latest record.

    Commands are summed, scaled seven-joint deltas; outcomes are observed joint
    changes. Weight decay is diagnostic-only and does not change online memory.
    """
    if commands.shape != outcomes.shape or commands.ndim != 2 or commands.shape[1] != 7:
        raise ValueError("History must be two matching [time, 7] tensors")
    if proposal.shape != (7,):
        raise ValueError("Expected one proposed seven-joint delta")
    if half_life is not None and not 0 < half_life <= 100:
        raise ValueError("half_life must be finite and in (0, 100]")
    config = InteractionMemoryConfig()
    commands, outcomes = commands[-config.slots :], outcomes[-config.slots :]
    size = len(commands)
    events = commands.new_zeros((1, config.slots, config.event_dim))
    valid = torch.zeros((1, config.slots), dtype=torch.bool, device=commands.device)
    p, a, h = config.proprio_dim, config.action_dim, config.chunk_len
    events[0, :size, p : p + 7] = commands / config.joint_delta_scale
    events[0, :size, p + h * a : p + h * a + 7] = outcomes
    events[0, :size, 2 * p + h * a] = 1
    valid[0, :size] = True
    weights = None
    if half_life is not None:
        weights = commands.new_zeros((1, config.slots))
        ages = torch.arange(size - 1, -1, -1, device=commands.device)
        weights[0, :size] = 2.0 ** (-ages / half_life)
    response = response_features(
        {"memory_events": events, "memory_valid": valid},
        config,
        record_weights=weights,
    )[0]
    return response[:7] * proposal, response[7:]


def run(output: Path) -> dict:
    """Compare declared retention choices on identical noisy action streams."""
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in range(3030, 3036):
        generator = torch.Generator().manual_seed(seed)
        commands = torch.randn(120, 7, generator=generator) * 0.05
        noise = torch.randn(120, 7, generator=generator) * 0.002
        for condition in ("stationary", "gain_drop", "gain_rise"):
            gain = torch.ones(120, 1)
            if condition == "gain_drop":
                gain[60:] = 0.2
            elif condition == "gain_rise":
                gain[:60] = 0.2
            outcomes = commands * gain + noise
            for name, half_life, adaptive in (
                ("uniform", None, False),
                ("decay_1", 1.0, False),
                ("decay_2", 2.0, False),
                ("decay_4", 4.0, False),
                ("adaptive_reset", None, True),
            ):
                errors, supports, resets = [], [], []
                history = ResponseHistory(adaptive=adaptive)
                for tick in range(120):
                    # Predict before making this tick's observed outcome available.
                    pred, support = history.predict(commands[tick], half_life=half_life)
                    errors.append(float((pred - outcomes[tick]).square().mean()))
                    supports.append(float(support.mean()))
                    if history.observe(commands[tick], outcomes[tick]):
                        resets.append(tick)
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "half_life": half_life,
                        "method": name,
                        "reset_ticks": resets,
                        "pre_change_mse": sum(errors[8:60]) / 52,
                        "first_eight_after_change_mse": sum(errors[60:68]) / 8,
                        "late_mse": sum(errors[68:]) / 52,
                        "support_at_change": supports[60],
                        "error_by_tick": errors,
                    }
                )
    report = {
        "scope": "Synthetic gain shift; no robot, contact, task success or learned weights",
        "contract": "Predict at t from completed records <t; condition/gain never enters predictor",
        "seeds": list(range(3030, 3036)),
        "change_tick": 60,
        "reset_rule": "After observing two consecutive RMSE > 0.015 with >=4 past records, retain last two; threshold fixed before adaptive follow-up",
        "results": rows,
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output)


if __name__ == "__main__":
    main()
