# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compare BC-only and Q+BC actors using identical cached executed transitions."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rlinf.algorithms.rlt.correction_data import correction_batch, load_corrections
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.utils.logging import get_logger

logger = get_logger()


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Average action error over executed, explicitly eligible control ticks."""
    error = (prediction - target).square().mean(-1)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def select_batch(batch: dict, indices: torch.Tensor) -> dict:
    """Select the same transitions, features and quality labels for both arms."""
    return {
        k: select_batch(v, indices) if isinstance(v, dict) else v[indices]
        for k, v in batch.items()
    }


def quality_metrics(batch: dict) -> dict:
    """Expose masking and saturated commands separately from imitation error."""
    valid = batch["valid"]
    arm = batch["actions"][..., :7]
    return {
        "transitions": len(arm),
        "executed_ticks": int(valid.sum()),
        "padding_ticks": int((~valid).sum()),
        "controller_clipped_ticks": int(
            ((batch["submitted_actions"] != batch["actions"]).any(-1) & valid).sum()
        ),
        "max_submitted_magnitude": float(batch["submitted_actions"].abs().max()),
        "planner_ticks": int((valid & batch["planner"]).sum()),
        "accepted_planner_ticks": int(batch["accepted_planner"].sum()),
        "saturated_arm_component_fraction": float(
            ((arm.abs() >= 0.999) & valid[..., None]).sum()
            / (7 * valid.sum()).clamp_min(1)
        ),
    }


@torch.no_grad()
def following_metrics(model: RLTMLPPolicy, batch: dict) -> dict:
    """Report prediction errors, not inferred closed-loop success."""
    predictions = []
    for indices in torch.arange(len(batch["actions"])).split(128):
        obs = select_batch(batch["curr_obs"], indices)
        predictions.append(
            model.sac_forward(obs, deterministic=True)[0].reshape(-1, 10, 8)
        )
    prediction = torch.cat(predictions)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite diagnostic actor predictions")

    def score(target, mask, dims=slice(None)):
        if not mask.any():
            return None
        return float(masked_mse(prediction[..., dims], target[..., dims], mask))

    return {
        "bc_mse": score(batch["target"], batch["bc_mask"]),
        "correction_mse": score(batch["actions"], batch["accepted_planner"]),
        "correction_arm_mse": score(
            batch["actions"], batch["accepted_planner"], slice(0, 7)
        ),
        "correction_gripper_mse": score(
            batch["actions"], batch["accepted_planner"], slice(7, 8)
        ),
        "reference_mse": score(
            batch["curr_obs"]["ref_chunk"].reshape(-1, 10, 8), batch["valid"]
        ),
        "accepted_planner_ticks": int(batch["accepted_planner"].sum()),
        "rejected_planner_ticks": int(
            (batch["planner"] & batch["valid"] & ~batch["accepted_planner"]).sum()
        ),
    }


def fit_pair(
    cache: Path,
    output: Path,
    *,
    updates: int = 2000,
    seed: int = 1234,
    warmup_updates: int = 128,
    q_weight: float = 0.45,
    initial_weights: Path | None = None,
) -> dict:
    """Fit paired diagnostic heads; export final weights for separate evaluation.

    Both arms train critics for matched optimizer work. Only Q+BC backpropagates
    Q into its actor. No online collection occurs and neither arm is a resumable
    distributed learner. Successful-episode filtering is a coarse BC label rule,
    not a certification that every correction was optimal.
    """
    if updates < 1 or seed < 0 or warmup_updates < 0 or not 0 <= q_weight <= 10:
        raise ValueError("Invalid paired diagnostic budget")
    episodes, contract, digest = load_corrections(cache)
    if len(episodes) < 4:
        raise ValueError("Need at least four nonempty episodes for disjoint validation")
    model_cfg = contract["model"]
    if (
        model_cfg["model_type"] != "rlt_mlp_policy"
        or model_cfg.get("q_head_type", "default") != "default"
        or tuple(
            model_cfg[k]
            for k in ("z_dim", "proprio_dim", "action_dim", "num_action_chunks")
        )
        != (2048, 9, 8, 10)
        or contract.get("reference_source") != "frozen_vla_pre_action"
        or contract.get("action_space") != "environment_pd_joint_delta_pos"
    ):
        raise ValueError("Require the frozen-feature RLT MLP correction contract")
    split = len(episodes) * 3 // 4
    train, validation = (
        correction_batch(episodes[:split]),
        correction_batch(episodes[split:]),
    )
    if not train["accepted_planner"].any() or not validation["accepted_planner"].any():
        raise ValueError("Both episode splits need successful planner corrections")
    gamma = float(contract["algorithm"]["gamma"])
    dropout = float(contract["algorithm"]["reference_dropout_prob"])
    if not 0 < gamma <= 1 or not 0 <= dropout <= 1:
        raise ValueError("Invalid discount or reference dropout")
    torch.manual_seed(seed)
    initial = RLTMLPPolicy(
        2048, 9, 8, 10, fixed_std=float(model_cfg.get("fixed_std", 0.002))
    )
    initial_sha = None
    if initial_weights is not None:
        weights = torch.load(initial_weights, map_location="cpu", weights_only=True)
        initial.load_state_dict(weights, strict=True)
        if any(not torch.isfinite(value).all() for value in weights.values()):
            raise ValueError("Initial weights must be finite")
        with initial_weights.open("rb") as stream:
            initial_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    arms = {}
    for name in ("bc_only", "q_bc"):
        model = copy.deepcopy(initial)
        actor_params = list(model.backbone.parameters()) + list(
            model.actor_mean.parameters()
        )
        arms[name] = {
            "model": model,
            "target": copy.deepcopy(model).requires_grad_(False),
            "actor_params": actor_params,
            "actor_opt": torch.optim.Adam(actor_params, lr=1e-4),
            "critic_opt": torch.optim.Adam(model.q_head.parameters(), lr=1e-4),
            "history": [],
        }
    initial_metrics = following_metrics(initial, validation)
    generator = torch.Generator().manual_seed(seed + 1)
    batch_digest = hashlib.sha256()
    discounts = gamma ** torch.arange(10).float()
    actor_updates = 0
    for step in range(updates):
        indices = torch.randint(len(train["actions"]), (64,), generator=generator)
        batch_digest.update(indices.numpy().tobytes())
        batch = select_batch(train, indices)
        actor_step = step % 4 == 0
        actor_updates += int(actor_step)
        for name, arm in arms.items():
            # Common random numbers for exploration samples and reference dropout.
            torch.manual_seed(seed + 1000 + step)
            model, target = arm["model"], arm["target"]
            with torch.no_grad():
                next_actions = model.sac_forward(batch["next_obs"])[0]
                next_q = (
                    target.sac_q_forward(batch["next_obs"], next_actions)
                    .min(-1, keepdim=True)
                    .values
                )
                reward = (batch["rewards"] * batch["valid"] * discounts).sum(
                    -1, keepdim=True
                )
                bootstrap = ~batch["dones"].any(-1, keepdim=True)
                td_target = reward + bootstrap * gamma**10 * next_q
            values = model.sac_q_forward(batch["curr_obs"], batch["actions"])
            critic_loss = F.mse_loss(values, td_target.expand_as(values))
            arm["critic_opt"].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.q_head.parameters(), 10, error_if_nonfinite=True
            )
            arm["critic_opt"].step()
            if actor_step:
                model.q_head.requires_grad_(False)
                prediction = model.sac_forward(
                    batch["curr_obs"],
                    apply_reference_dropout=True,
                    reference_dropout_prob=dropout,
                )[0]
                bc_loss = masked_mse(
                    prediction.reshape(-1, 10, 8), batch["target"], batch["bc_mask"]
                )
                actor_loss = 2.5 * bc_loss
                if name == "q_bc" and step >= warmup_updates:
                    actor_loss = (
                        actor_loss
                        - q_weight
                        * model.sac_q_forward(batch["curr_obs"], prediction)[
                            :, 0
                        ].mean()
                    )
                arm["actor_opt"].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    arm["actor_params"], 10, error_if_nonfinite=True
                )
                arm["actor_opt"].step()
                model.q_head.requires_grad_(True)
            with torch.no_grad():
                for online, slow in zip(model.parameters(), target.parameters()):
                    slow.lerp_(online, 0.005)
            if (step + 1) % 100 == 0 or step + 1 == updates:
                scores = following_metrics(model, validation)
                arm["history"].append(
                    {"critic_updates": step + 1, "validation": scores}
                )
    report = {
        "protocol": "same_cache_offline_actor_objective_diagnostic_v1",
        "cache_sha256": digest,
        "minibatches_sha256": batch_digest.hexdigest(),
        "seed": seed,
        "initial_weights": str(initial_weights.resolve()) if initial_weights else None,
        "initial_weights_sha256": initial_sha,
        "initialization": "weights_only_fresh_optimizers"
        if initial_weights
        else "seeded_random",
        "critic_updates_per_arm": updates,
        "actor_updates_per_arm": actor_updates,
        "q_warmup_updates": warmup_updates,
        "q_weight": q_weight,
        "bc_weight": 2.5,
        "train_episodes": [e["episode_id"] for e in episodes[:split]],
        "validation_episodes": [e["episode_id"] for e in episodes[split:]],
        "initial_validation": initial_metrics,
        "quality": {
            "train": quality_metrics(train),
            "validation": quality_metrics(validation),
        },
        "results": {},
        "bc_quality_rule": "Planner commands in successful episodes only; failed attempts retained for TD.",
        "controller_mapping": "Verified Panda pd_joint_delta_pos: clamp normalized commands to [-1,1] before critic use; original submissions and reference inputs retained. Online replay is unchanged.",
        "closed_loop_success": None,
        "limitation": "Fixed-data diagnostic, not an online RL comparison. Final weights selected by budget, not test success.",
    }
    for name, arm in arms.items():
        destination = output / name
        destination.mkdir()
        torch.save(arm["model"].state_dict(), destination / "model.pt")
        report["results"][name] = {
            "train": following_metrics(arm["model"], train),
            "validation": following_metrics(arm["model"], validation),
            "history": arm["history"],
        }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run the same-data comparison on CPU; evaluation is a separate command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", default=2000, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--warmup-updates", default=128, type=int)
    parser.add_argument(
        "--initial-weights",
        type=Path,
        help="Same finite small actor/critic weights for both arms; fresh optimizers",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = fit_pair(
        args.cache,
        args.output,
        updates=args.updates,
        seed=args.seed,
        warmup_updates=args.warmup_updates,
        initial_weights=args.initial_weights,
    )
    logger.info(
        "Paired diagnostic complete: %s; closed-loop success remains unmeasured",
        result["cache_sha256"],
    )


if __name__ == "__main__":
    main()
