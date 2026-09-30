# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compare actual RLT loss paths on shared synthetic transition data."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy.rlt_atomic_policy import RLTAtomicPolicy
from rlinf.models.embodiment.mlp_policy.rlt_bounded_policy import (
    RLTBoundedResidualPolicy,
)
from rlinf.models.embodiment.modules.rlt_action_candidates import JointActionCandidates
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin
from toolkits.rlt.atomic_cpu_smoke import observations, plain_method, reward


def run(
    output: Path,
    *,
    device: str = "cpu",
    steps: int = 1000,
    data_mode: str = "candidate",
    actor_start: int = 0,
    actor_lr: float = 1e-3,
) -> dict:
    """Use sampled-action rewards only; neither model sees oracle candidate labels."""
    if not 1 <= steps <= 2000:
        raise ValueError("Use 1..2000 synthetic updates")
    if data_mode not in ("candidate", "mixed"):
        raise ValueError("data_mode must be candidate or mixed")
    if not 0 <= actor_start < steps or not 0 < actor_lr <= 1e-3:
        raise ValueError("Invalid actor warmup or learning rate")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in (2026, 2027, 2028):
        generator = torch.Generator().manual_seed(seed + 1000)
        train_obs = observations(2048, generator)
        test_obs = observations(256, generator)
        bank = JointActionCandidates(8, 2, radius=0.1)
        candidates, _ = bank(train_obs["ref_chunk"])
        ids = torch.randint(candidates.shape[1], (2048,), generator=generator)
        actions = candidates[torch.arange(2048), ids]
        if data_mode == "mixed":
            # Keep half the finite-action data; add full-box arm perturbations.
            # Both learners receive exactly the same sampled transitions.
            residual = (torch.rand((1024, 2, 8), generator=generator) * 2 - 1) * 0.1
            residual[..., -1] = 0
            actions[1024:] = (train_obs["ref_chunk"][1024:] + residual).clamp(-1, 1)
        rewards = reward(train_obs, actions)[:, None]
        train_obs, test_obs = (
            {k: v.to(device) for k, v in obs.items()} for obs in (train_obs, test_obs)
        )
        actions, rewards = actions.to(device), rewards.to(device)
        for mode in ("atomic", "continuous"):
            torch.manual_seed(seed)
            kwargs = {
                "z_dim": 4,
                "proprio_dim": 3,
                "action_dim": 8,
                "num_action_chunks": 2,
            }
            model = (
                RLTAtomicPolicy(
                    **kwargs, atomic_decision={"radius": 0.1, "reference_prior": 0.8}
                )
                if mode == "atomic"
                else RLTBoundedResidualPolicy(
                    **kwargs, bounded_residual={"radius": 0.1}
                )
            ).to(device)
            worker = RLTACLossMixin()
            worker.model, worker.target_model = model, copy.deepcopy(model)
            worker.torch_dtype = torch.float32
            worker.cfg = OmegaConf.create(
                {
                    "actor": {
                        "model": {
                            "num_action_chunks": 2,
                            "action_dim": 8,
                            "model_type": "rlt_mlp_policy",
                            "q_head_type": "default",
                            "atomic_decision": {"enabled": mode == "atomic"},
                        }
                    },
                    "algorithm": {
                        "gamma": 0.9,
                        "bc_weight": 0.1,
                        "q_weight": 1.0,
                        "reference_dropout_prob": 0.0,
                        "loss_type": "rlt_ac",
                    },
                    "env": {
                        "train": {
                            "env_type": "maniskill_rlt",
                            "init_params": {"control_mode": "pd_joint_delta_pos"},
                        }
                    },
                    "rollout": {
                        "model": {"atomic_decision": {"enabled": mode == "atomic"}}
                    },
                }
            )
            actor_params = [p for n, p in model.named_parameters() if "q_head" not in n]
            actor_opt = torch.optim.Adam(actor_params, lr=actor_lr)
            critic_opt = torch.optim.Adam(model.q_head.parameters(), lr=1e-3)
            sampler = torch.Generator().manual_seed(seed + 2000)
            start = time.perf_counter()
            curve = []
            for step in range(steps + 1):
                if step % 100 == 0 or step == steps:
                    with torch.no_grad():
                        chosen = model.predict_action_batch(test_obs, mode="eval")[0]
                        score = reward(test_obs, chosen)
                        curve.append(
                            {
                                "step": step,
                                "mean_reward": score.mean().item(),
                                "near_optimum_fraction": (score > -0.05)
                                .float()
                                .mean()
                                .item(),
                            }
                        )
                if step == steps:
                    break
                idx = torch.randint(2048, (32,), generator=sampler).to(device)
                obs = {k: v[idx] for k, v in train_obs.items()}
                batch = {
                    "curr_obs": obs,
                    "next_obs": obs,
                    "actions": actions[idx],
                    "rewards": rewards[idx],
                    "dones": torch.ones(32, 1, dtype=torch.bool, device=device),
                    "terminations": torch.ones(32, 1, dtype=torch.bool, device=device),
                }
                # Each phase clears all gradients, preserving optimizer ownership.
                model.zero_grad(set_to_none=True)
                loss, _ = plain_method(RLTACLossMixin.forward_critic)(worker, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.q_head.parameters(), 10, error_if_nonfinite=True
                )
                critic_opt.step()
                if step >= actor_start:
                    model.zero_grad(set_to_none=True)
                    actor_loss, _, _ = plain_method(RLTACLossMixin.forward_actor)(
                        worker, batch
                    )
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        actor_params, 10, error_if_nonfinite=True
                    )
                    actor_opt.step()
            rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "seconds": time.perf_counter() - start,
                    "curve": curve,
                    "critic_loss": float(loss),
                    "actor_loss": float(actor_loss),
                }
            )
            report = {
                "scope": "Synthetic one-step reward; not robot success. Continuous actions may extrapolate beyond sampled dataset support.",
                "data_mode": data_mode,
                "actor_start": actor_start,
                "actor_lr": actor_lr,
                "device": device,
                "steps": steps,
                "results": rows,
            }
            (output / "results.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps(rows[-1]), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--data-mode", choices=("candidate", "mixed"), default="candidate"
    )
    parser.add_argument("--actor-start", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(
        args.output,
        steps=args.steps,
        data_mode=args.data_mode,
        actor_start=args.actor_start,
        actor_lr=args.actor_lr,
    )


if __name__ == "__main__":
    main()
