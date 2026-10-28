# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Screen causal response calibration before conditioning the RLT policy."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from toolkits.rlt.libero_pro_environment import LiberoProEnvironment
from toolkits.rlt.physrlt.data import (
    measured_response_snapshot,
    physical_reset_hash,
)
from toolkits.rlt.physrlt.gain_probe import realized_translation_score
from toolkits.rlt.pi05_baseline import (
    Pi05Reference,
    configure_gpu,
    configure_source,
    load_model,
    write_json,
)


def response_context(commands: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    """Describe issued translation and current robot state without future data."""
    commands = commands.float()
    proprio = proprio.float().flatten()
    if commands.shape != (10, 7) or proprio.shape != (8,):
        raise ValueError("Expected a 10x7 command chunk and 8-D proprioception")
    translation = commands[:, :3]
    base = torch.cat(
        (
            translation.sum(0),
            translation.abs().sum(0),
            translation.square().sum(0),
            translation.mean(0),
            translation.std(0, unbiased=False),
            translation[0],
            translation[-1],
            proprio,
        )
    )
    return torch.cat((base, base.square()))


class NominalResponseRegressor:
    """Predict nominal 3-D end-effector displacement from training layouts."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor, weight: torch.Tensor):
        self.mean = mean
        self.scale = scale
        self.weight = weight

    @classmethod
    def fit(cls, dataset_path: Path) -> tuple[NominalResponseRegressor, dict]:
        """Select ridge strength on validation layouts without reading development."""
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=True)
        rows = []
        for row in dataset["rows"]:
            if not row["horizon_mask"][1] or row["split"] == "development":
                continue
            displacement = row["response"][1][7:10].float()
            if torch.isfinite(displacement).all():
                rows.append(
                    (
                        row["split"],
                        response_context(row["commands"][:10], row["prop"]),
                        displacement,
                    )
                )
        train = [row for row in rows if row[0] == "train"]
        validation = [row for row in rows if row[0] == "validation"]
        if len(train) < 20 or len(validation) < 10:
            raise ValueError(
                "Nominal response fitting needs separate train and validation rows"
            )
        features = torch.stack([row[1] for row in train]).double()
        targets = torch.stack([row[2] for row in train]).double()
        mean = features.mean(0)
        scale = features.std(0).clamp_min(1e-6)
        design = torch.cat(
            ((features - mean) / scale, torch.ones(len(features), 1)), dim=1
        )
        validation_features = torch.stack([row[1] for row in validation]).double()
        validation_targets = torch.stack([row[2] for row in validation]).double()
        validation_design = torch.cat(
            (
                (validation_features - mean) / scale,
                torch.ones(len(validation_features), 1),
            ),
            dim=1,
        )
        identity = torch.eye(design.shape[1], dtype=torch.double)
        identity[-1, -1] = 0
        candidates = []
        for ridge in (0.1, 1.0, 10.0, 100.0):
            weight = torch.linalg.solve(
                design.T @ design + ridge * identity, design.T @ targets
            )
            rmse = float(
                ((validation_design @ weight - validation_targets).square().mean())
                .sqrt()
                .item()
            )
            candidates.append((rmse, ridge, weight))
        validation_rmse, ridge, weight = min(candidates, key=lambda item: item[0])
        zero_rmse = float(validation_targets.square().mean().sqrt().item())
        predicted = validation_design @ weight
        nonzero = (predicted.norm(dim=1) > 1e-4) & (
            validation_targets.norm(dim=1) > 1e-4
        )
        cosine = (
            float(
                torch.nn.functional.cosine_similarity(
                    predicted[nonzero], validation_targets[nonzero], dim=1
                )
                .mean()
                .item()
            )
            if nonzero.any()
            else float("nan")
        )
        if validation_rmse >= zero_rmse or not np.isfinite(cosine) or cosine <= 0:
            raise RuntimeError(
                "Conditional nominal displacement did not beat its zero-motion control"
            )
        model = cls(mean.float(), scale.float(), weight.float())
        return model, {
            "dataset": str(dataset_path),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "ridge": ridge,
            "validation_rmse": validation_rmse,
            "zero_motion_validation_rmse": zero_rmse,
            "validation_cosine": cosine,
            "development_rows_read": 0,
        }

    def predict(self, commands: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Predict nominal end-effector displacement for one reference chunk."""
        features = response_context(commands, proprio)
        design = torch.cat(((features - self.mean) / self.scale, torch.ones(1)))
        return design @ self.weight


class TranslationCalibrator:
    """Estimate an actuation correction from preceding completed chunks."""

    def __init__(
        self,
        nominal_score: float,
        *,
        alpha: float = 0.5,
        minimum_observations: int = 2,
        correction_bounds: tuple[float, float] = (0.8, 1.3),
        reversed_control: bool = False,
    ) -> None:
        if (
            nominal_score <= 0
            or not 0 < alpha <= 1
            or minimum_observations < 1
            or not 0 < correction_bounds[0] <= 1 <= correction_bounds[1]
        ):
            raise ValueError("Invalid response-calibration contract")
        self.nominal_score = nominal_score
        self.alpha = alpha
        self.minimum_observations = minimum_observations
        self.bounds = correction_bounds
        self.reversed_control = reversed_control
        self.ema: float | None = None
        self.observations = 0

    def correction(self) -> float:
        """Return a bounded factor using only earlier observations."""
        if self.ema is None or self.observations < self.minimum_observations:
            return 1.0
        ratio = 1 / max(self.ema, 0.1)
        if self.reversed_control:
            ratio = 1 / ratio
        return float(np.clip(ratio, *self.bounds))

    def observe(
        self,
        commands: torch.Tensor,
        response: torch.Tensor,
        expected_score: float | None = None,
    ) -> float:
        """Update calibration after the command-response transition completes."""
        score = realized_translation_score(commands, response)
        if not np.isfinite(score):
            return score
        expected_score = (
            self.nominal_score if expected_score is None else expected_score
        )
        observed_gain = score / max(expected_score, 1e-4)
        self.ema = (
            observed_gain
            if self.ema is None
            else self.alpha * observed_gain + (1 - self.alpha) * self.ema
        )
        self.observations += 1
        return score

    def observe_vector(
        self,
        nominal_displacement: torch.Tensor,
        measured_displacement: torch.Tensor,
        issued_scale: float,
    ) -> float:
        """Update gain from a measured displacement after one completed chunk."""
        expected = nominal_displacement.float() * issued_scale
        measured = measured_displacement.float()
        denominator = float(expected.square().sum())
        if denominator <= 1e-8 or not torch.isfinite(measured).all():
            return float("nan")
        observed_gain = float(torch.dot(measured, expected) / denominator)
        self.ema = (
            observed_gain
            if self.ema is None
            else self.alpha * observed_gain + (1 - self.alpha) * self.ema
        )
        self.observations += 1
        return observed_gain


def control_correction(
    method: str, calibrator: TranslationCalibrator, execution_gain: float
) -> float:
    """Return the declared response-calibration control factor."""
    if method == "oracle":
        return float(np.clip(1 / execution_gain, 0.8, 1.3))
    if method == "fixed_high":
        return 1.3
    if method == "delayed_high":
        return 1.0 if calibrator.observations < calibrator.minimum_observations else 1.3
    if method in {"adaptive", "reversed"}:
        return calibrator.correction()
    if method == "reference":
        return 1.0
    raise ValueError("Unknown response-adaptation method")


def rollout(
    env,
    reference,
    case: dict,
    method: str,
    execution_gain: float,
    nominal_score: float,
    limit: int,
    nominal_model: NominalResponseRegressor | None = None,
) -> dict:
    """Run a frozen reference with causal, oracle or deliberately wrong scaling."""
    if method not in {
        "reference",
        "adaptive",
        "reversed",
        "fixed_high",
        "delayed_high",
        "oracle",
    }:
        raise ValueError("Unknown response-adaptation method")
    calibrator = TranslationCalibrator(
        nominal_score, reversed_control=method == "reversed"
    )
    obs = env.reset(case)
    for _ in range(9):
        obs, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
    reset_hash = physical_reset_hash(env)
    ticks, success, reward_sum = 0, False, 0.0
    diagnostics = []
    started = time.monotonic()
    while ticks < limit:
        prediction = reference.predict(
            obs, env.instruction, seed=case["seed"] * 1000 + ticks
        )
        commands = prediction["commands"].copy()
        correction = control_correction(method, calibrator, execution_gain)
        issued = commands.copy()
        issued_translation = issued[:, :3] * correction
        issued_clip_fraction = float(np.mean(np.abs(issued_translation) > 1))
        issued[:, :3] = issued_translation.clip(-1, 1)
        nominal_displacement = (
            nominal_model.predict(
                torch.from_numpy(commands), torch.as_tensor(obs["state"])
            )
            if nominal_model is not None
            else None
        )
        executed = issued.copy()
        executed_translation = executed[:, :3] * execution_gain
        execution_clip_fraction = float(np.mean(np.abs(executed_translation) > 1))
        executed[:, :3] = executed_translation.clip(-1, 1)
        before = measured_response_snapshot(env)
        chunk_ticks = 0
        for command in executed:
            obs, reward, done = env.step(command)
            ticks += 1
            chunk_ticks += 1
            reward_sum += reward
            success = success or done
            if done or ticks >= limit:
                break
        if chunk_ticks == 10:
            response = measured_response_snapshot(env) - before
            if nominal_displacement is None:
                score = calibrator.observe(torch.from_numpy(issued), response)
            else:
                score = realized_translation_score(torch.from_numpy(issued), response)
                calibrator.observe_vector(
                    nominal_displacement, response[7:10], correction
                )
            diagnostics.append(
                {
                    "end_tick": ticks,
                    "correction": correction,
                    "response_score": score,
                    "nominal_displacement": (
                        nominal_displacement.tolist()
                        if nominal_displacement is not None
                        else None
                    ),
                    "measured_displacement": response[7:10].tolist(),
                    "estimated_gain": calibrator.ema,
                    "observations": calibrator.observations,
                    "issued_clip_fraction": issued_clip_fraction,
                    "execution_clip_fraction": execution_clip_fraction,
                }
            )
        if success:
            break
    return {
        "case_id": case["id"],
        "split": case["split"],
        "method": method,
        "execution_gain": execution_gain,
        "success": success,
        "ticks": ticks,
        "reward": reward_sum,
        "reset_hash": reset_hash,
        "seconds": time.monotonic() - started,
        "diagnostics": diagnostics,
        "assistance": False,
        "online_updates": 0,
    }


def run(args: argparse.Namespace) -> None:
    """Evaluate matched calibration variants on declared development layouts."""
    if os.environ.get("LIBERO_TYPE") != "pro":
        raise RuntimeError("Response adaptation requires LIBERO_TYPE=pro")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Completed response-adaptation screens are immutable")
    configure_gpu(args.gpu, args.output)
    reservation = torch.empty(1, device="cuda")
    configure_source(args.source)
    torch.set_num_threads(4)
    model, config = load_model(args.assets, "cuda")
    model.eval().requires_grad_(False)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters:
        raise RuntimeError("Response adaptation requires a frozen reference model")
    reference = Pi05Reference(model)
    nominal_model, nominal_model_metadata = (None, None)
    if args.nominal_dataset is not None:
        nominal_model, nominal_model_metadata = NominalResponseRegressor.fit(
            args.nominal_dataset
        )
    declared_cases = json.loads(args.cases.read_text())
    cases = declared_cases[args.case_offset : args.case_offset + args.case_limit]
    if not cases or any(case["split"] != "development" for case in cases):
        raise ValueError("Response adaptation requires development cases only")
    methods = args.methods
    allowed_methods = {
        "reference",
        "adaptive",
        "reversed",
        "fixed_high",
        "delayed_high",
        "oracle",
    }
    if (
        not methods
        or len(methods) != len(set(methods))
        or not set(methods) <= allowed_methods
    ):
        raise ValueError("Response-adaptation methods must be unique known controls")
    env = LiberoProEnvironment()
    records = []
    try:
        for case in cases:
            expected_hash = None
            for method in methods:
                row = rollout(
                    env,
                    reference,
                    case,
                    method,
                    args.execution_gain,
                    args.nominal_score,
                    args.ticks,
                    nominal_model,
                )
                if expected_hash is None:
                    expected_hash = row["reset_hash"]
                elif row["reset_hash"] != expected_hash:
                    raise RuntimeError("Matched response-adaptation reset changed")
                records.append(row)
                write_json(args.output / "episodes.json", records)
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "method": method,
                            "success": row["success"],
                            "ticks": row["ticks"],
                        }
                    ),
                    flush=True,
                )
    finally:
        env.close()
        del reservation
    summary = {}
    for method in methods:
        selected = [row for row in records if row["method"] == method]
        summary[method] = {
            "successes": sum(row["success"] for row in selected),
            "episodes": len(selected),
            "mean_ticks": sum(row["ticks"] for row in selected) / len(selected),
            "mean_success_ticks": (
                sum(row["ticks"] for row in selected if row["success"])
                / sum(row["success"] for row in selected)
                if any(row["success"] for row in selected)
                else None
            ),
        }
    write_json(
        args.output / "result.json",
        {
            "state": "complete",
            "summary": summary,
            "execution_gain": args.execution_gain,
            "nominal_score": args.nominal_score,
            "nominal_model": nominal_model_metadata,
            "nominal_context_action": (
                "reference_before_correction_with_known_issued_scale"
                if nominal_model_metadata is not None
                else "global_scalar"
            ),
            "cases": cases,
            "model_config": config,
            "frozen_vla_trainable_parameters": trainable_parameters,
            "online_updates": 0,
            "interpretation": (
                "Causal response-calibration screen; oracle denotes a bounded fixed "
                "gain-compensation control, not an optimal policy; not RLT training or "
                "an official LIBERO-Pro score"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "assets", "cases", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--execution-gain", type=float, default=0.7)
    parser.add_argument("--nominal-score", type=float, required=True)
    parser.add_argument("--nominal-dataset", type=Path)
    parser.add_argument("--ticks", type=int, default=520)
    parser.add_argument("--case-limit", type=int, default=2)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["reference", "adaptive", "reversed", "oracle"],
    )
    args = parser.parse_args()
    if not 0.2 <= args.execution_gain <= 1.5 or not 10 <= args.ticks <= 520:
        parser.error("Response-adaptation stress is outside the pilot contract")
    if (
        not 1 <= args.case_limit <= 6
        or args.case_offset < 0
        or args.case_offset + args.case_limit > 6
    ):
        parser.error("Response-adaptation screen is limited to six layouts")
    run(args)


if __name__ == "__main__":
    main()
