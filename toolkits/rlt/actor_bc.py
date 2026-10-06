# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU actor fitting on frozen features, with episode-disjoint validation.

Cached reference actions must be VLA predictions from the same pre-action
observation, never the demonstration target. Old FLARE caches lack them and
are accepted only by the explicitly diagnostic zero-reference mode.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def load_episodes(
    cache: Path, *, horizon: int, reference_mode: str
) -> tuple[list[dict], dict]:
    """Validate units and construct chunks without crossing episode boundaries."""
    manifest = torch.load(cache / "manifest.pt", map_location="cpu", weights_only=True)
    contract = manifest["feature_contract"]
    if (
        not manifest["complete"]
        or contract["control_mode"] != "pd_joint_delta_pos"
        or contract["control_freq"] != 10
        or contract["action_space"] != "environment_pd_joint_delta_pos"
    ):
        raise ValueError("Incomplete cache or incompatible action contract")
    episodes = []
    for row in manifest["episodes"]:
        path = cache / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Cache checksum mismatch: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        n = len(data["actions"]) - horizon + 1
        if n <= 0:
            continue
        if data["actions"].shape[1:] != (8,) or data["actions"].abs().max() > 1.0001:
            raise ValueError(
                "Targets must be bounded environment actions with dimension 8"
            )
        if not torch.all(data["frame_index"][1:] - data["frame_index"][:-1] == 1):
            raise ValueError("Nonconsecutive episode frames")
        target = torch.stack([data["actions"][i : i + horizon] for i in range(n)])
        if reference_mode == "cached":
            if "ref_chunk" not in data:
                raise ValueError(
                    "Cache lacks VLA ref_chunk; export real references first. Do not substitute target actions."
                )
            reference = data["ref_chunk"][:n, :horizon]
        else:
            reference = torch.zeros_like(target)
        sample = {
            "z_rl": data["z_rl"][:n].float(),
            "proprio": data["proprio"][:n].float(),
            "ref_chunk": reference.float(),
            "target": target.float(),
            "id": row["id"],
        }
        for key in ("z_rl", "proprio", "ref_chunk", "target"):
            if not torch.isfinite(sample[key]).all():
                raise ValueError(f"Nonfinite {key}")
        if (
            sample["z_rl"].shape != (n, 2048)
            or sample["proprio"].shape != (n, 9)
            or reference.shape != target.shape
        ):
            raise ValueError("Feature/reference shape mismatch")
        episodes.append(sample)
    return episodes, contract


def fit(
    cache: Path, output: Path, *, steps: int, seed: int, reference_mode: str
) -> dict:
    """Fit only actor parameters; this is an offline diagnostic, not an RL run."""
    if steps < 1:
        raise ValueError("steps must be positive")
    episodes, contract = load_episodes(cache, horizon=10, reference_mode=reference_mode)
    if len(episodes) < 4:
        raise ValueError("Need at least four complete episodes for a disjoint split")
    torch.manual_seed(seed)
    split = max(1, len(episodes) * 3 // 4)

    def batch(rows):
        return {
            k: torch.cat([r[k] for r in rows])
            for k in ("z_rl", "proprio", "ref_chunk", "target")
        }

    train, val = batch(episodes[:split]), batch(episodes[split:])
    model = RLTMLPPolicy(2048, 9, 8, 10)
    params = list(model.backbone.parameters()) + list(model.actor_mean.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    output.mkdir(parents=True, exist_ok=False)

    def evaluate(data):
        with torch.no_grad():
            pred = model.sac_forward(data, deterministic=True)[0].reshape(-1, 10, 8)
            error = (pred - data["target"]).square()
            return {
                "mse": error.mean().item(),
                "arm_mse": error[..., :7].mean().item(),
                "gripper_mse": error[..., 7].mean().item(),
            }

    initial = evaluate(val)
    history = []
    best = float("inf")
    for step in range(1, steps + 1):
        idx = torch.randint(len(train["target"]), (64,))
        sample = {k: v[idx] for k, v in train.items()}
        pred = model.sac_forward(sample, deterministic=True)[0].reshape(-1, 10, 8)
        loss = (pred - sample["target"]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10)
        optimizer.step()
        if step % 50 == 0 or step == steps:
            scores = evaluate(val)
            history.append(
                {"step": step, "train": evaluate(train), "validation": scores}
            )
            if scores["mse"] < best:
                best = scores["mse"]
                # No random critic, optimizer or fake learner schedule is exported.
                torch.save(
                    {
                        "actor_state": {
                            k: v
                            for k, v in model.state_dict().items()
                            if k.startswith(("backbone.", "actor_mean."))
                        },
                        "reference_mode": reference_mode,
                        "feature_contract": contract,
                        "step": step,
                    },
                    output / "best_actor.pt",
                )
    report = {
        "reference_mode": reference_mode,
        "seed": seed,
        "steps": steps,
        "train_episodes": [r["id"] for r in episodes[:split]],
        "validation_episodes": [r["id"] for r in episodes[split:]],
        "initial_validation": initial,
        "best_validation_mse": best,
        "history": history,
        "feature_contract": contract,
        "closed_loop_success_rate": None,
        "eligible_for_stage2_initialization": False,
        "limitation": "Offline prediction only; zero-reference mode is not full actor acceptance. Evaluate closed-loop before deployment.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run a bounded CPU diagnostic without allocating the frozen VLA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--reference-mode", choices=["cached", "zero-diagnostic"], default="cached"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = fit(
        args.cache,
        args.output,
        steps=args.steps,
        seed=args.seed,
        reference_mode=args.reference_mode,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "reference_mode",
                    "best_validation_mse",
                    "closed_loop_success_rate",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
