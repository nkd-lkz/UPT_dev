# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Measure held-out action relevance and uncertainty; never start simulation."""

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rlinf.data.datasets.rlt_latent import RLTLatentDataset
from rlinf.models.embodiment.modules.rlt_latent_world import RLTLatentWorld
from toolkits.rlt.train_latent_world import device_batch


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str, cache_dir: str, *, device: str = "cpu", batch_size: int = 64
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
    model = RLTLatentWorld.from_checkpoint(checkpoint).to(device).eval()
    dataset = RLTLatentDataset(
        cache_dir,
        horizons=model.config.horizons,
        split="validation",
        seed=settings["seed"],
        validation_fraction=settings["validation_fraction"],
    )
    if dataset.manifest["feature_contract"] != model.feature_contract:
        raise ValueError(
            "Evaluation cache uses a different frozen encoder/preprocessing"
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
                "proprio_mae",
                "latent_mse",
                "disagreement",
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
        offset += len(batch["actions"])
        for i, horizon in enumerate(model.config.horizons):
            predictions, delta = model(batch, batch["actions"], horizon)
            shuffled, _ = model(batch, shuffled_actions, horizon)
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
                "proprio_mae": (
                    delta.mean(0) - (batch["future_proprio"][:, i] - batch["proprio"])
                )
                .abs()
                .mean(-1),
                "latent_mse": (predictions.mean(0) - target).square().mean(-1),
                "disagreement": predictions.var(0, unbiased=False).mean(-1),
            }
            for key, value in values.items():
                records[horizon][key].append(value[valid].cpu())
    results = {}
    for horizon, fields in records.items():
        fields = {key: torch.cat(values) for key, values in fields.items()}
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
        results[str(horizon)] = summary
    return {
        "split": "validation_not_independent_test",
        "episodes": dataset.episode_ids,
        "horizons": results,
    }


def main() -> None:
    """Print diagnostics; choosing an exploration scale remains an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_checkpoint(
                args.checkpoint,
                args.cache_dir,
                device=args.device,
                batch_size=args.batch_size,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
