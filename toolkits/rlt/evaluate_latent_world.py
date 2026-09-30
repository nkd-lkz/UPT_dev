# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Measure held-out action relevance and uncertainty; never start simulation."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rlinf.data.datasets.rlt_latent import RLTLatentDataset
from rlinf.models.embodiment.modules.rlt_latent_world import RLTLatentWorld
from toolkits.rlt.train_latent_world import device_batch


def selective_prediction_report(
    disagreement: torch.Tensor,
    error: torch.Tensor,
    persistence_error: torch.Tensor,
    episode_ids: torch.Tensor,
    thresholds: dict[str, float],
) -> dict:
    """Measure retained prediction error using thresholds fixed on calibration.

    Frames within episodes are correlated. Report both frame and episode means,
    actual retained fraction (ties can increase it), and empty selections. This
    does not calibrate a safety probability or measure robot task success.
    """
    if (
        disagreement.ndim != 1
        or any(
            x.shape != disagreement.shape
            for x in (error, persistence_error, episode_ids)
        )
        or any(
            not torch.isfinite(x).all()
            for x in (disagreement, error, persistence_error)
        )
        or (disagreement < 0).any()
        or any(not math.isfinite(v) or v < 0 for v in thresholds.values())
    ):
        raise ValueError("Expected matching finite vectors and nonnegative thresholds")
    report = {}
    for label, threshold in thresholds.items():
        selected = disagreement <= threshold
        episode_errors = [
            error[selected & (episode_ids == i)].mean().item()
            for i in episode_ids.unique()
            if (selected & (episode_ids == i)).any()
        ]
        count = int(selected.sum())
        report[label] = {
            "threshold": threshold,
            "accepted_count": count,
            "total_count": len(error),
            "retained_fraction": count / len(error) if len(error) else None,
            "accepted_cosine_error": error[selected].mean().item() if count else None,
            "accepted_persistence_error": persistence_error[selected].mean().item()
            if count
            else None,
            "accepted_episode_mean_error": sum(episode_errors) / len(episode_errors)
            if episode_errors
            else None,
            "episodes_with_accepted_frames": len(episode_errors),
        }
    return report


def action_controls(
    actions: torch.Tensor, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove arm commands or reverse only the executed prefix, preserving grip.

    A full-chunk reversal would leak commands beyond the prediction horizon.
    These perturbations diagnose sensitivity; they are not causal rollouts.
    """
    if actions.ndim != 3 or not 1 <= horizon <= actions.shape[1]:
        raise ValueError("Expected a valid prefix of [batch, time, action]")
    stationary = actions.clone()
    stationary[:, :horizon, :-1] = 0
    reversed_prefix = actions.clone()
    reversed_prefix[:, :horizon] = actions[:, :horizon].flip(1)
    return stationary, reversed_prefix


def mean_arm_prefix(actions: torch.Tensor, horizon: int) -> torch.Tensor:
    """Keep total arm command but remove its within-prefix temporal variation.

    Gripper commands and the unavailable suffix remain unchanged. This is an
    input perturbation, not a physically executed alternative trajectory.
    """
    if (
        actions.ndim != 3
        or actions.shape[-1] < 2
        or not 1 <= horizon <= actions.shape[1]
    ):
        raise ValueError("Expected a valid prefix with arm and gripper commands")
    result = actions.clone()
    result[:, :horizon, :-1] = actions[:, :horizon, :-1].mean(1, keepdim=True)
    return result


def episode_error_summary(episode_ids: torch.Tensor, fields: dict) -> dict:
    """Summarize independent trajectories rather than treating frames as trials.

    The paired bootstrap is descriptive uncertainty over the sampled episodes,
    not a test of control improvement or a correction for repeated model tuning.
    """
    rows = {}
    for episode in episode_ids.unique(sorted=True).tolist():
        mask = episode_ids == episode
        rows[str(episode)] = {
            key: float(values[mask].mean())
            for key, values in fields.items()
            if key.endswith("cosine_error")
        }
    if not rows:
        return {
            "episodes": {},
            "paired_persistence_gain_mean": None,
            "paired_persistence_gain_ci95": None,
        }
    gains = torch.tensor(
        [r["persistence_cosine_error"] - r["cosine_error"] for r in rows.values()]
    )
    interval = None
    if len(gains) >= 2:
        generator = torch.Generator().manual_seed(1729)
        samples = torch.randint(len(gains), (2000, len(gains)), generator=generator)
        interval = torch.quantile(
            gains[samples].mean(1), torch.tensor([0.025, 0.975])
        ).tolist()
    return {
        "episodes": rows,
        "paired_persistence_gain_mean": float(gains.mean()),
        "paired_persistence_gain_ci95": interval,
    }


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str,
    cache_dir: str,
    *,
    device: str = "cpu",
    batch_size: int = 64,
    independent_test: bool = False,
    calibration_cache_dir: str | None = None,
) -> dict:
    """Compare true actions to shuffled full chunks and a persistence predictor.

    Reports latent-space errors, not task success or certified safety. The
    validation split was used for model selection; final claims need a separate
    test set. Uncertainty/error correlation may be absent or negative.
    """
    if batch_size < 2:
        raise ValueError("batch_size >= 2 is required for action shuffling")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    settings = payload["training_config"]
    calibration = None
    if calibration_cache_dir is not None:
        if not independent_test:
            raise ValueError(
                "Calibration thresholds may only be applied to independent test episodes"
            )
        manifest = torch.load(
            Path(calibration_cache_dir) / "manifest.pt",
            map_location="cpu",
            weights_only=True,
        )
        if manifest != payload["cache_manifest"]:
            raise ValueError(
                "Calibration must use the original training/validation cache manifest"
            )
        calibration = evaluate_checkpoint(
            checkpoint, calibration_cache_dir, device=device, batch_size=batch_size
        )
    model = RLTLatentWorld.from_checkpoint(checkpoint).to(device).eval()
    dataset = RLTLatentDataset(
        cache_dir,
        horizons=model.config.horizons,
        split="all" if independent_test else "validation",
        seed=settings.get("split_seed", settings["seed"]),
        validation_fraction=settings["validation_fraction"],
    )
    if dataset.manifest["feature_contract"] != model.feature_contract:
        raise ValueError(
            "Evaluation cache uses a different frozen encoder/preprocessing"
        )
    if independent_test:
        original_ids = {row["id"] for row in payload["cache_manifest"]["episodes"]}
        if original_ids.intersection(dataset.episode_ids):
            raise ValueError(
                "Independent test episodes overlap training/validation cache"
            )
        original_source = payload["cache_manifest"].get("source", {})
        if any(
            not original_source.get(key)
            or dataset.manifest.get("source", {}).get(key) != original_source[key]
            for key in ("info", "episodes", "tasks")
        ):
            raise ValueError(
                "Independent test must use the same source dataset identity"
            )
    donors = [
        i
        for i, (episode, t) in enumerate(dataset.index)
        if len(dataset.episodes[episode]["actions"]) - 1 - t >= model.config.chunk_len
    ]
    if not donors:
        raise ValueError(
            "Action sensitivity needs at least one complete validation chunk"
        )
    generator = torch.Generator().manual_seed(int(settings["seed"]) + 73)
    donors = [
        donors[i] for i in torch.randperm(len(donors), generator=generator).tolist()
    ]
    offset = 0
    records = {
        h: {
            key: []
            for key in (
                "cosine_error",
                "persistence_cosine_error",
                "shuffled_cosine_error",
                "zero_arm_cosine_error",
                "reversed_prefix_cosine_error",
                "mean_prefix_cosine_error",
                "reverse_input_rms",
                "mean_prefix_input_rms",
                "reverse_prediction_rms",
                "mean_prefix_prediction_rms",
                "proprio_mae",
                "latent_mse",
                "disagreement",
                "episode_index",
            )
        }
        for h in model.config.horizons
    }
    for batch in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    ):
        batch = device_batch(batch, torch.device(device), model.config.horizons)
        shuffled_actions = torch.stack(
            [
                dataset[donors[(offset + i) % len(donors)]]["actions"]
                for i in range(len(batch["actions"]))
            ]
        ).to(device)
        episode_indices = torch.tensor(
            [
                episode
                for episode, _ in dataset.index[offset : offset + len(batch["actions"])]
            ],
            device=device,
        )
        offset += len(batch["actions"])
        for i, horizon in enumerate(model.config.horizons):
            predictions, delta = model(batch, batch["actions"], horizon)
            shuffled, _ = model(batch, shuffled_actions, horizon)
            zero_arm, reversed_prefix = action_controls(batch["actions"], horizon)
            stationary, _ = model(batch, zero_arm, horizon)
            reversed_predictions, _ = model(batch, reversed_prefix, horizon)
            averaged_actions = mean_arm_prefix(batch["actions"], horizon)
            averaged_predictions, _ = model(batch, averaged_actions, horizon)
            target = F.layer_norm(batch["future_z"][:, i], (model.config.z_dim,))
            current = F.layer_norm(batch["z_rl"], (model.config.z_dim,))
            valid = batch["valid"][:, i]
            values = {
                "cosine_error": 1
                - F.cosine_similarity(predictions.mean(0), target, dim=-1),
                "persistence_cosine_error": 1
                - F.cosine_similarity(current, target, dim=-1),
                "shuffled_cosine_error": 1
                - F.cosine_similarity(shuffled.mean(0), target, dim=-1),
                "zero_arm_cosine_error": 1
                - F.cosine_similarity(stationary.mean(0), target, dim=-1),
                "reversed_prefix_cosine_error": 1
                - F.cosine_similarity(reversed_predictions.mean(0), target, dim=-1),
                "mean_prefix_cosine_error": 1
                - F.cosine_similarity(averaged_predictions.mean(0), target, dim=-1),
                "reverse_input_rms": (
                    reversed_prefix[:, :horizon] - batch["actions"][:, :horizon]
                )
                .square()
                .mean((1, 2))
                .sqrt(),
                "mean_prefix_input_rms": (
                    averaged_actions[:, :horizon] - batch["actions"][:, :horizon]
                )
                .square()
                .mean((1, 2))
                .sqrt(),
                "reverse_prediction_rms": (
                    reversed_predictions.mean(0) - predictions.mean(0)
                )
                .square()
                .mean(-1)
                .sqrt(),
                "mean_prefix_prediction_rms": (
                    averaged_predictions.mean(0) - predictions.mean(0)
                )
                .square()
                .mean(-1)
                .sqrt(),
                "proprio_mae": (
                    delta.mean(0) - (batch["future_proprio"][:, i] - batch["proprio"])
                )
                .abs()
                .mean(-1),
                "latent_mse": (predictions.mean(0) - target).square().mean(-1),
                "disagreement": predictions.var(0, unbiased=False).mean(-1),
                "episode_index": episode_indices,
            }
            for key, value in values.items():
                records[horizon][key].append(value[valid].cpu())
    results = {}
    for horizon, fields in records.items():
        fields = {key: torch.cat(values) for key, values in fields.items()}
        episode_indices = fields.pop("episode_index")
        episode_summary = episode_error_summary(episode_indices, fields)
        # The public report names dataset episode IDs, not internal positions.
        episode_summary["episodes"] = {
            str(dataset.episode_ids[int(k)]): v
            for k, v in episode_summary["episodes"].items()
        }
        summary = {
            key: float(value.mean()) if len(value) else None
            for key, value in fields.items()
        }
        uncertainty, error = fields["disagreement"], fields["latent_mse"]
        if len(error) > 1 and uncertainty.std() > 0 and error.std() > 0:
            summary["uncertainty_error_pearson"] = float(
                torch.corrcoef(torch.stack((uncertainty, error)))[0, 1]
            )
        else:
            summary["uncertainty_error_pearson"] = None
        summary["disagreement_p95"] = (
            float(torch.quantile(uncertainty, 0.95)) if len(uncertainty) else None
        )
        summary["valid_targets"] = len(error)
        # Quantiles depend only on calibration disagreement, not test outcomes.
        summary["disagreement_quantiles"] = (
            {
                str(fraction): torch.quantile(uncertainty, fraction).item()
                for fraction in (0.25, 0.5, 0.75, 1.0)
            }
            if len(uncertainty)
            else {}
        )
        if calibration is not None:
            thresholds = calibration["horizons"][str(horizon)]["disagreement_quantiles"]
            summary["selective_prediction"] = selective_prediction_report(
                uncertainty,
                fields["cosine_error"],
                fields["persistence_cosine_error"],
                episode_indices,
                thresholds,
            )
        summary["episode_summary"] = episode_summary
        results[str(horizon)] = summary
    return {
        "split": "independent_test"
        if independent_test
        else "validation_not_independent_test",
        "episodes": dataset.episode_ids,
        "horizons": results,
        "calibration_episodes": calibration["episodes"] if calibration else None,
        "calibration_scope": "Validation-selected checkpoint; development diagnostic, not sealed-test or conformal coverage guarantee"
        if calibration
        else None,
    }


def main() -> None:
    """Print diagnostics; choosing an exploration scale remains an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--independent-test", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--calibration-cache-dir",
        help="Original cache; fit disagreement quantiles on its validation episodes only",
    )
    args = parser.parse_args()
    report = evaluate_checkpoint(
        args.checkpoint,
        args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
        independent_test=args.independent_test,
        calibration_cache_dir=args.calibration_cache_dir,
    )
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(serialized + "\n")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
