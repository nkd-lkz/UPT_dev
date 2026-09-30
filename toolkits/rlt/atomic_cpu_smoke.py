# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only synthetic learning check, not a robot performance benchmark.

A one-step quadratic reward asks for a signed joint correction encoded in z.
Only the reward of each sampled action enters replay; all-action oracle rewards
are used for evaluation, never for a training target. This checks whether the
critic and categorical actor can learn through their actual loss interfaces.
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.atomic_decision import candidate_bc_errors, candidate_q_values
from rlinf.models.embodiment.mlp_policy.rlt_atomic_policy import RLTAtomicPolicy
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin


def observations(size: int, generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Sample independent contexts; z[0] gives the desired correction sign."""
    z = torch.randn(size, 4, generator=generator)
    z[:, 0] = torch.where(z[:, 0] >= 0, 1.0, -1.0)
    return {
        "z_rl": z,
        "proprio": torch.zeros(size, 3),
        "ref_chunk": torch.full((size, 2, 8), 0.3),
    }


def reward(obs: dict, action: torch.Tensor) -> torch.Tensor:
    """Evaluate an executed action; there are no physical contact dynamics here."""
    target = obs["ref_chunk"].clone()
    target[:, :, 0] += 0.1 * obs["z_rl"][:, 0:1]
    return -(action - target).square().sum((-1, -2)) / (2 * 0.1**2)


def plain_method(method):
    """Remove worker timing wrappers, which need a scheduler-owned worker."""
    while hasattr(method, "__wrapped__"):
        method = method.__wrapped__
    return method


def run_seed(seed: int, steps: int, actor_update: str = "distill") -> dict:
    """Train on sampled transitions and evaluate on fresh, fixed CPU contexts."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    model = RLTAtomicPolicy(
        z_dim=4,
        proprio_dim=3,
        action_dim=8,
        num_action_chunks=2,
        atomic_decision={"radius": 0.1, "reference_prior": 0.8},
    )
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
                    "atomic_decision": {"enabled": True},
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
            "rollout": {"model": {"atomic_decision": {"enabled": True}}},
        }
    )
    actor_params = [p for name, p in model.named_parameters() if "q_head" not in name]
    actor_optim = torch.optim.Adam(actor_params, lr=1e-3)
    critic_optim = torch.optim.Adam(model.q_head.parameters(), lr=1e-3)
    train_obs = observations(1024, generator)
    # Uniform collection ensures coverage; it is not a claim about sparse-reward RL.
    with torch.no_grad():
        candidates = model.decision_forward(train_obs)["candidates"]
        ids = torch.randint(candidates.shape[1], (1024,), generator=generator)
        actions = candidates[torch.arange(1024), ids]
        rewards = reward(train_obs, actions)[:, None]
    test_obs = observations(256, generator)

    def evaluate() -> dict:
        with torch.no_grad():
            action, info = model.predict_action_batch(test_obs, mode="eval")
            optimal_id = torch.where(test_obs["z_rl"][:, 0] > 0, 3, 4)
            chosen = info["forward_inputs"]["atomic_proposed_id"].flatten()
            decision = model.decision_forward(test_obs)
            q = candidate_q_values(model, test_obs, decision["candidates"])[..., 0]
            greedy_q = q.masked_fill(~decision["valid"], -torch.inf).argmax(-1)
            return {
                "mean_reward": reward(test_obs, action).mean().item(),
                "greedy_critic_optimal_fraction": (greedy_q == optimal_id)
                .float()
                .mean()
                .item(),
                "optimal_choice_fraction": (chosen == optimal_id).float().mean().item(),
            }

    before = evaluate()
    critic_fn = plain_method(RLTACLossMixin.forward_critic)
    actor_fn = plain_method(RLTACLossMixin.forward_actor)
    start = time.perf_counter()
    for _ in range(steps):
        idx = torch.randint(1024, (32,), generator=generator)
        obs = {key: value[idx] for key, value in train_obs.items()}
        batch = {
            "curr_obs": obs,
            "next_obs": obs,
            "actions": actions[idx],
            "rewards": rewards[idx],
            "dones": torch.ones(32, 1, dtype=torch.bool),
            "terminations": torch.ones(32, 1, dtype=torch.bool),
        }
        critic_optim.zero_grad(set_to_none=True)
        critic_loss, _ = critic_fn(worker, batch)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.q_head.parameters(), 10)
        critic_optim.step()
        actor_optim.zero_grad(set_to_none=True)
        if actor_update == "distill":
            actor_loss, _, _ = actor_fn(worker, batch)
        else:
            # Preserve the failed first-prototype update for a reproducible
            # diagnostic ablation. It is not a production configuration option.
            decision = model.decision_forward(obs)
            with torch.no_grad():
                q = candidate_q_values(model, obs, decision["candidates"])[..., 0]
                bc, _ = candidate_bc_errors(
                    decision["candidates"], batch["actions"], obs["ref_chunk"], None
                )
            actor_loss = (decision["probabilities"] * (0.1 * bc - q)).sum(-1).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_params, 10)
        actor_optim.step()
    elapsed = time.perf_counter() - start
    return {
        "seed": seed,
        "actor_update": actor_update,
        "updates": steps,
        "seconds": elapsed,
        "before": before,
        "after": evaluate(),
        "critic_loss": critic_loss.item(),
        "actor_loss": actor_loss.item(),
    }


def main() -> None:
    """Write a bounded CPU verification report, refusing to overwrite a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--actor-update", choices=("distill", "expected-cost"), default="distill"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000 or not 1 <= len(args.seeds) <= 5:
        parser.error("CPU smoke is limited to 1..1000 updates and 1..5 seeds.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    report = {
        "scope": "synthetic CPU loss/learning verification; not robot evidence",
        "runs": [run_seed(seed, args.steps, args.actor_update) for seed in args.seeds],
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
