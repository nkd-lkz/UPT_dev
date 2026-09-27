# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Train the opt-in RLT Stage 1B future-latent sidecar (single process)."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, default_collate

from rlinf.data.datasets.rlt_latent import RLTLatentDataset, atomic_torch_save
from rlinf.models.embodiment.modules.rlt_latent_world import (
    LatentWorldConfig,
    RLTLatentWorld,
)


def device_batch(batch: dict, device: torch.device, horizons: tuple) -> dict:
    """Move tensors without turning discrete horizon metadata into CUDA scalars."""
    return {
        **{key: value.to(device) for key, value in batch.items()},
        "horizons": horizons,
    }


@torch.no_grad()
def evaluate(
    model: RLTLatentWorld,
    dataset: RLTLatentDataset,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Evaluate all held-out windows with deterministic, unbootstrapped losses."""
    model.eval()
    totals, count = {}, 0
    for batch in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    ):
        batch = device_batch(batch, device, model.config.horizons)
        _, metrics = model.loss(batch, offline=True)
        n = len(batch["z_rl"])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * n
        # Compare to copying the present latent, a strong slow-motion baseline.
        current = torch.nn.functional.layer_norm(batch["z_rl"], (model.config.z_dim,))
        future = torch.nn.functional.layer_norm(
            batch["future_z"], (model.config.z_dim,)
        )
        error = 1 - torch.nn.functional.cosine_similarity(
            current[:, None], future, dim=-1
        )
        valid = batch["valid"]
        totals["persistence_cosine_error_sum"] = totals.get(
            "persistence_cosine_error_sum", 0.0
        ) + float(error[valid].sum())
        totals["valid_targets"] = totals.get("valid_targets", 0) + int(valid.sum())
        count += n
    result = {
        key: value / count
        for key, value in totals.items()
        if key not in ("persistence_cosine_error_sum", "valid_targets")
    }
    result["persistence_cosine_error"] = totals["persistence_cosine_error_sum"] / max(
        totals["valid_targets"], 1
    )
    model.train()
    return result


def train(config_path: str, *, device: str, resume: str | None = None) -> Path:
    """Train with episode-level validation, atomic checkpoints and exact RNG resume."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "Stage 1B is a single-process lightweight learner; do not use torchrun"
        )
    cfg = OmegaConf.load(config_path)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device(device)
    torch.manual_seed(int(cfg.seed))
    c = LatentWorldConfig(**resolved["world_model"])
    if not (
        cfg.max_steps > 0
        and 0 < cfg.micro_batch_size <= cfg.batch_size
        and cfg.batch_size % cfg.micro_batch_size == 0
    ):
        raise ValueError(
            "Positive steps and an integral gradient-accumulation factor are required"
        )
    if cfg.validate_every < 1 or cfg.log_every < 1 or cfg.warmup_steps < 0:
        raise ValueError("Invalid logging/validation/warmup interval")
    output = Path(cfg.output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(
            "Nonempty output_dir; choose a new run directory or pass --resume last.pt"
        )
    output.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: RLTLatentDataset(
            cfg.cache_dir,
            horizons=c.horizons,
            split=split,
            seed=int(cfg.get("split_seed", cfg.seed)),
            validation_fraction=float(cfg.validation_fraction),
        )
        for split in ("train", "validation")
    }
    example = datasets["train"][0]
    if (
        example["z_rl"].numel(),
        example["proprio"].numel(),
        example["actions"].shape[-1],
    ) != (c.z_dim, c.proprio_dim, c.action_dim):
        raise ValueError("Cache dimensions do not match world_model configuration")
    contract = datasets["train"].manifest["feature_contract"]
    model = RLTLatentWorld(c).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    def lr_factor(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        progress = min(
            1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        )
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    sampler = torch.Generator().manual_seed(int(cfg.seed) + 1)
    start, best, stale, wandb_id = 0, float("inf"), 0, None
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            payload["feature_contract"] != contract
            or payload["training_config"] != resolved
            or payload["cache_manifest"] != datasets["train"].manifest
        ):
            raise ValueError(
                "Resume requires identical configuration, cache manifest and feature contract"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        sampler.set_state(payload["sampler_rng"])
        torch.set_rng_state(payload["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        start, best, stale = payload["step"], payload["best"], payload["stale"]
        wandb_id = payload.get("wandb_id")
    logger = None
    if cfg.wandb.mode != "disabled":
        import wandb

        logger = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=output.name,
            dir=str(output),
            mode=cfg.wandb.mode,
            id=wandb_id,
            resume="must" if wandb_id and cfg.wandb.mode == "online" else None,
            config=resolved,
        )
        wandb_id = logger.id
    accumulation = int(cfg.batch_size // cfg.micro_batch_size)
    try:
        for step in range(start + 1, int(cfg.max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            totals = {}
            for _ in range(accumulation):
                indices = torch.randint(
                    len(datasets["train"]),
                    (int(cfg.micro_batch_size),),
                    generator=sampler,
                )
                batch = default_collate([datasets["train"][int(i)] for i in indices])
                batch = device_batch(batch, device, c.horizons)
                loss, metrics = model.loss(batch, offline=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite world loss at step {step}")
                (loss / accumulation).backward()
                for key, value in metrics.items():
                    totals[f"train/{key}"] = (
                        totals.get(f"train/{key}", 0.0) + float(value) / accumulation
                    )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.clip_grad), error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            totals.update(
                step=step,
                grad_norm=float(grad_norm),
                learning_rate=optimizer.param_groups[0]["lr"],
            )
            validated = step % cfg.validate_every == 0 or step == cfg.max_steps
            improved = False
            if validated:
                validation = evaluate(
                    model, datasets["validation"], int(cfg.micro_batch_size), device
                )
                totals.update(
                    {f"validation/{key}": value for key, value in validation.items()}
                )
                score = validation["world/loss"]
                if not math.isfinite(score):
                    raise FloatingPointError("Nonfinite validation loss")
                improved = score < best
                best, stale = (score, 0) if improved else (best, stale + 1)
                payload = model.checkpoint(contract)
                payload.update(
                    training_config=resolved,
                    cache_manifest=datasets["train"].manifest,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    step=step,
                    best=best,
                    stale=stale,
                    sampler_rng=sampler.get_state(),
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else [],
                    wandb_id=wandb_id,
                )
                atomic_torch_save(payload, output / "last.pt")
                if improved:
                    atomic_torch_save(payload, output / "best.pt")
            if step % cfg.log_every == 0 or validated:
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(totals) + "\n")
                print(json.dumps(totals), flush=True)
                if logger is not None:
                    logger.log(totals, step=step)
            if (
                validated
                and cfg.early_stopping_patience > 0
                and stale >= cfg.early_stopping_patience
            ):
                break
    finally:
        if logger is not None:
            logger.finish()
    return output / "best.pt"


def main() -> None:
    """Run only on an explicit CLI request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(train(args.config, device=args.device, resume=args.resume))


if __name__ == "__main__":
    main()
