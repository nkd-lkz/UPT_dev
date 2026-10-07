# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a frozen RLT head with paired seeds and explicit action routing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict


def evaluation_config(
    cfg: DictConfig,
    *,
    checkpoint: Path,
    variant: str,
    num_envs: int,
    seed: int,
) -> DictConfig:
    """Copy a pilot config for one frozen evaluation; never restore replay."""
    if variant not in {"native", "zero_context", "reference"}:
        raise ValueError("variant must be native, zero_context or reference")
    if not 1 <= num_envs <= 32 or seed < 0:
        raise ValueError("Expected 1..32 evaluation lanes and a nonnegative seed")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = copy.deepcopy(cfg)
    reader = cfg.actor.model.interaction_memory.reader_type
    if reader not in {"zero", "response"}:
        raise ValueError("Frozen comparison currently supports zero/response heads")
    if cfg.rollout.pipeline_stage_num != 1 or cfg.env.eval.auto_reset:
        raise ValueError("Episode records require one stage and auto_reset=False")
    with open_dict(cfg):
        cfg.runner.only_eval = True
        cfg.runner.task_type = "embodied_eval"
        cfg.runner.resume_dir = None
        cfg.runner.ckpt_path = str(checkpoint.resolve())
        cfg.runner.save_interval = -1
        cfg.runner.val_check_interval = 1
        cfg.runner.eval_protocol = {"variant": variant, "seed": seed}
        # Eval-only has no weight sync: its version starts at zero. Disable the
        # training warmup for a learned policy; explicitly retain it for reference.
        cfg.algorithm.rlt_schedule.enable = variant == "reference"
        cfg.algorithm.rlt_schedule.warmup_post_collect_updates = 1
        cfg.rollout.expert_model = None
        cfg.rollout.enable_cuda_graph = False
        cfg.rollout.collect_transitions = False
        cfg.rollout.model = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
        if variant == "zero_context":
            for model in (cfg.actor.model, cfg.rollout.model):
                model.interaction_memory.reader_type = "zero"
            for split in ("train", "eval"):
                cfg.env[split].interaction_memory.reader_type = "zero"
        cfg.env.eval.seed = seed
        cfg.env.eval.total_num_envs = num_envs
        cfg.env.eval.rollout_epoch = 1
        cfg.env.eval.use_fixed_reset_state_ids = True
        cfg.env.eval.video_cfg.save_video = False
        for split in ("train", "eval"):
            cfg.env[split].rlt_policy_switch.expert_takeover.enable = False
            cfg.env[split].interaction_memory.retain_on_identical_reset = False
        for model in (cfg.actor.model, cfg.rollout.model):
            model.interaction_memory.retain_on_identical_reset = False
    return cfg


def tensor_digest(value) -> str:
    """Fingerprint nested tensors and arrays without saving observations."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(key.encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                visit(entry)
        elif isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def main() -> None:
    """Launch only env and rollout workers; save per-episode and route evidence."""
    from rlinf.algorithms.rlt.transition import extract_rlt_obs_from_forward_inputs
    from rlinf.config import validate_cfg
    from rlinf.envs.utils import get_env_attr
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("native", "zero_context", "reference"), default="native"
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4001)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cfg = evaluation_config(
        OmegaConf.load(args.config),
        checkpoint=args.checkpoint,
        variant=args.variant,
        num_envs=args.num_envs,
        seed=args.seed,
    )
    cfg = validate_cfg(cfg)
    if args.check:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    output = Path(cfg.runner.logger.log_path)
    OmegaConf.save(cfg, output / "evaluation-config.yaml", resolve=True)

    class AuditedRollout(MultiStepRolloutWorker):
        def init_worker(self):
            super().init_worker()
            self.hf_model.requires_grad_(False)
            self.initial_weights = tensor_digest(self.hf_model.state_dict())
            seed = int(self.cfg.runner.eval_protocol.seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.audit = {
                "slots": 0,
                "actor_slots": 0,
                "action_difference_sum": 0.0,
                "action_difference_count": 0,
                "nonempty_slots": 0,
            }

        def _predict_rollout_actions(self, env_obs, *positional, **kwargs):
            if "initial_observation_sha256" not in self.audit:
                self.audit["initial_observation_sha256"] = tensor_digest(env_obs)
            actions, result = super()._predict_rollout_actions(
                env_obs, *positional, **kwargs
            )
            obs = extract_rlt_obs_from_forward_inputs(result["forward_inputs"])
            actor = result["forward_inputs"]["actor_switch"].bool().reshape(-1)
            self.audit["slots"] += actor.numel()
            self.audit["actor_slots"] += int(actor.sum())
            valid = obs["memory_valid"].any(-1)
            self.audit["nonempty_slots"] += int(valid.sum())
            # The same frozen head sees the same state/reference twice. Only
            # historical validity changes. Both predictions are deterministic.
            empty = {**obs, "memory_valid": torch.zeros_like(obs["memory_valid"])}
            native, _ = self.hf_model.predict_action_batch(obs, mode="eval")
            cleared, _ = self.hf_model.predict_action_batch(empty, mode="eval")
            difference = (native - cleared).abs()
            if (
                not torch.isfinite(actions).all()
                or not torch.isfinite(difference).all()
            ):
                raise FloatingPointError("Nonfinite evaluated action")
            self.audit["action_difference_sum"] += float(difference.sum())
            self.audit["action_difference_count"] += difference.numel()
            return actions, result

        async def evaluate(self, *args, **kwargs):
            await super().evaluate(*args, **kwargs)
            self.audit["weights_unchanged"] = self.initial_weights == tensor_digest(
                self.hf_model.state_dict()
            )
            if not self.audit["weights_unchanged"]:
                raise RuntimeError("Frozen evaluation changed model weights")
            if (
                self.cfg.runner.eval_protocol.variant == "reference"
                and self.audit["actor_slots"]
            ):
                raise RuntimeError("Reference-only evaluation routed learned actions")
            (Path(self.cfg.runner.logger.log_path) / "route-audit.json").write_text(
                json.dumps(self.audit, indent=2, allow_nan=False)
            )
            return {
                "rlt/actor_routed_slot_fraction": torch.tensor(
                    [self.audit["actor_slots"] / max(1, self.audit["slots"])]
                )
            }

    class AuditedEnvironment(EnvWorker):
        def env_evaluate_step(self, raw_actions, stage_id):
            before = self.eval_prev_done[stage_id].clone()
            env_output, info = super().env_evaluate_step(raw_actions, stage_id)
            lanes = (
                (self.eval_prev_done[stage_id] & ~before).nonzero().flatten().tolist()
            )
            reset_ids = get_env_attr(self.eval_env_list[stage_id], "reset_state_ids")
            for offset, lane in enumerate(lanes):
                row = {
                    "lane": lane,
                    "reset_id": int(reset_ids[lane]),
                    "seed": int(self.cfg.env.eval.seed),
                }
                row.update({key: float(value[offset]) for key, value in info.items()})
                self.episodes.append(row)
            return env_output, info

        def evaluate(self, *args, **kwargs):
            self.episodes = []
            result = super().evaluate(*args, **kwargs)
            if len(self.episodes) != self.cfg.env.eval.total_num_envs:
                raise RuntimeError(
                    "Not every evaluation lane completed exactly one episode"
                )
            (Path(self.cfg.runner.logger.log_path) / "episode-records.json").write_text(
                json.dumps(
                    sorted(self.episodes, key=lambda row: row["lane"]),
                    indent=2,
                    allow_nan=False,
                )
            )
            return result

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = AuditedRollout.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = AuditedEnvironment.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    runner = EmbodiedEvalRunner(cfg, rollout, env)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
