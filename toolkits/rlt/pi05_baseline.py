# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Accept the public Pi0.5 reference before training AlphaBrain's new RLT."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from toolkits.rlt.pi05_assets import REVISION, SOURCE_COMMIT, file_hash

if TYPE_CHECKING:
    from torch import nn

PROTOCOL = {
    "id": "alphabrain-rlt-pi05-libero-v1",
    "track": "rlt",
    "backbone": "pi05",
    "suite": "libero_goal",
    "checkpoint_revision": REVISION,
    "source_revision": SOURCE_COMMIT,
    "checkpoint_scope": "four-suite SFT, not few-demo reproduction",
    "chunk_len": 10,
    "action_dim": 7,
    "prop_dim": 8,
    "render_size": 256,
    "settling_steps": 10,
    "max_control_steps": 320,
    "prefix": "checkpoint predict_action; 3 image slots, 48 language positions",
    "encoder_input": "512 real-view final VLM hidden states from that same forward",
    "token_width": 2048,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "encoder_heads": 8,
    "action_mapping": "MEAN_STD once in VLA; native signed LIBERO gripper",
    "assistance": False,
    "memory": False,
    "training_variant": "bounded engineering pilot; not the unmodified upstream recipe",
    "stage1_data": "reference rollout observations, not demonstrations",
    "head_learning_rate": 1e-4,
    "bc_weight": 1.0,
    "ref_dropout": 0.0,
    "discount_per_tick": 0.99,
    "reward_scale": 5.0,
    "batch_size": 128,
    "actor_update_every": 2,
    "target_tau": 0.005,
    "exploration_std": 0.1,
}


def write_json(path: Path, value: dict) -> None:
    """Publish complete JSON atomically in the artifact directory."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def runtime_config(released: dict, tokenizer: Path) -> dict:
    """Translate the legacy class alias without changing its action convention."""
    config = copy.deepcopy(released)
    framework = config["framework"]
    if framework["name"] != "PaliGemmaPi0" or framework.get("pi05") is not True:
        raise ValueError("Expected the pinned legacy Pi0.5 release")
    action = framework["action_model"]
    if (action["action_dim"], action["action_horizon"]) != (7, 10):
        raise ValueError("Release action contract changed")
    if framework["paligemma"].get("discrete_state_input", False):
        raise ValueError("This release uses the continuous-state-free Pi0.5 prefix")
    framework["name"] = "PaliGemmaPi05"
    framework["gripper_remap"] = False
    framework["paligemma"]["base_vlm"] = str(tokenizer)
    framework["paligemma"]["max_token_len"] = 48
    config["trainer"]["pretrained_checkpoint"] = None
    return config


def env_commands(actions: np.ndarray) -> np.ndarray:
    """Map native signed actions to LIBERO without a second normalization."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise ValueError("Require a finite 10 x 7 native action chunk")
    commands = actions.clip(-1, 1).copy()
    commands[:, 6] = np.where(actions[:, 6] >= 0, 1.0, -1.0)
    return commands


def configure_source(source: Path) -> None:
    """Reject modified upstream code and configure child environment imports."""
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    if revision != SOURCE_COMMIT or dirty:
        raise ValueError("Require an unmodified pinned AlphaBrain checkout")
    sys.path.insert(0, str(source.resolve()))
    os.environ["PYTHONPATH"] = (
        str(source.resolve()) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )


def load_model(assets: Path, device: str) -> tuple[nn.Module, dict]:
    """Load all weights strictly, retaining the exact released normalization."""
    import torch
    import yaml
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    verified = json.loads((assets / "verified.json").read_text())
    if verified.get("complete") is not True or verified["revision"] != REVISION:
        raise ValueError("Run pi05_assets before loading this model")
    for name in verified["files"]:
        if file_hash(assets / name) != verified["files"][name]["sha256"]:
            raise ValueError(f"Modified release file: {name}")
    tokenizer = assets / "paligemma_pretrained"
    os.environ["PALIGEMMA_TOKENIZER_PATH"] = str(tokenizer)
    from AlphaBrain.model.framework.PaliGemmaPi import PaliGemmaPi

    cfg = runtime_config(
        yaml.safe_load((assets / "framework_config.yaml").read_text()), tokenizer
    )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = PaliGemmaPi(OmegaConf.create(cfg))
    finally:
        torch.set_default_dtype(previous)
    model.load_state_dict(load_file(str(assets / "model.safetensors")), strict=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    if model.chunk_len != 10 or model.gripper_remap:
        raise ValueError("Unexpected runtime action contract")
    return model, cfg


class Pi05Reference:
    """Capture RLT inputs from the reference's own prefix, never a second pass."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def predict(
        self, observation: dict, instruction: str, *, capture: bool = False
    ) -> dict:
        """Return native actions and optionally detached real-image features."""
        import torch

        captured = []

        def record(module, inputs, result):
            captured.append(result.last_hidden_state[:, :512].detach().float().cpu())

        handle = (
            self.model.vlm_interface.model.language_model.register_forward_hook(record)
            if capture
            else None
        )
        try:
            with torch.no_grad():
                result = self.model.predict_action(
                    batch_images=[
                        [
                            np.asarray(observation["primary_image"]),
                            np.asarray(observation["wrist_image"]),
                        ]
                    ],
                    instructions=[instruction],
                    states=[observation["state"]],
                )
        finally:
            if handle is not None:
                handle.remove()
        native = np.asarray(result["normalized_actions"], dtype=np.float32)
        if native.shape != (1, 10, 7):
            raise ValueError("Unexpected Pi0.5 output shape")
        result = {"ref": native[0], "commands": env_commands(native[0])}
        if capture:
            if len(captured) != 1 or tuple(captured[0].shape) != (1, 512, 2048):
                raise ValueError("Expected one shared prefix with two real image views")
            if not torch.isfinite(captured[0]).all():
                raise FloatingPointError("Nonfinite prefix features")
            result["hidden"] = captured[0][0]
        return result


def configure_gpu(gpu: int, output: Path) -> dict:
    """Require an idle physical GPU and map EGL by UUID before importing torch."""
    from toolkits.rlt.experiment_queue import gpu_idle
    from toolkits.rlt.libero_egl import egl_device_index

    idle, detail = gpu_idle(gpu)
    if not idle:
        raise RuntimeError(f"GPU {gpu} is occupied: {detail}")
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    egl = egl_device_index(uuid)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=uuid,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        MUJOCO_GL="egl",
        RLT_LIBERO_EGL_DEVICE_ID=str(egl),
        RLT_LIBERO_WORKER_LOG_DIR=str(output / "worker_logs"),
        LIBERO_PYTHON=sys.executable,
    )
    return {"gpu": gpu, "uuid": uuid, "egl": egl}


def cpu_audit(args: argparse.Namespace) -> dict:
    """Exercise strict loading and real model inference without reserving a GPU."""
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    model, config = load_model(args.assets, "cpu")
    observation = {
        "primary_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = Pi05Reference(model).predict(
            observation, "open the drawer", capture=True
        )
    return {
        "gate_passed": True,
        "strict_loading": True,
        "runtime_config": config,
        "action_shape": list(result["ref"].shape),
        "feature_shape": list(result["hidden"].shape),
        "action_range": [float(result["ref"].min()), float(result["ref"].max())],
        "evaluation_role": "synthetic CPU forward only; not a simulator success rate",
    }


def evaluate(args: argparse.Namespace) -> dict:
    """Run explicit development cases and retain a small RLT training cache."""
    import torch

    from toolkits.rlt.libero_acceptance import RemoteEnvironment

    device = "cpu" if args.mode == "cpu-reference" else "cuda"
    model, config = load_model(args.assets, device)
    reference = Pi05Reference(model)
    encoder = learner = None
    if args.mode in {"actor-eval", "online"}:
        from toolkits.rlt.pi05_learning import encode_row, load_heads, save_heads

        encoder, learner = load_heads(args.checkpoint, args.objective)
    env = RemoteEnvironment()
    results, cache, replay = [], [], []
    parity = None
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            state = args.state_start + episode
            case = {
                "id": f"dev-task{args.task}-state{state}",
                "task": args.task,
                "kind": "published",
                "state": state,
                "seed": seed,
            }
            obs = env.reset(case)
            for _ in range(PROTOCOL["settling_steps"]):
                obs, _, _ = env.step(
                    np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                )
            if parity is None:
                torch.manual_seed(seed)
                plain = reference.predict(obs, env.instruction)
                torch.manual_seed(seed)
                captured = reference.predict(obs, env.instruction, capture=True)
                error = float(np.max(np.abs(plain["ref"] - captured["ref"])))
                if error != 0:
                    raise ValueError(
                        f"Capturing prefix changed reference actions: {error}"
                    )
                parity = {"max_absolute_error": error, "passed": True}
            success, ticks = False, 0
            started = time.monotonic()
            row = None
            while ticks < PROTOCOL["max_control_steps"]:
                if row is None:
                    row = reference.predict(
                        obs,
                        env.instruction,
                        capture=args.collect or learner is not None,
                    )
                if args.collect:
                    cache.append(
                        {
                            "case_id": case["id"],
                            "hidden": row["hidden"].bfloat16(),
                            "ref": torch.from_numpy(row["ref"].copy()),
                            "prop": torch.as_tensor(
                                obs["state"], dtype=torch.float32
                            ).clone(),
                        }
                    )
                if learner is not None:
                    features = encode_row(encoder, row, obs["state"])
                    action_chunk = learner.command(
                        features, deterministic=args.mode != "online"
                    )
                    commands = env_commands(action_chunk.numpy())
                else:
                    commands = row["commands"]
                chunk_ticks, chunk_reward = 0, 0.0
                for action in commands:
                    obs, reward, done = env.step(action)
                    chunk_reward += 5.0 * 0.99**chunk_ticks * float(reward)
                    chunk_ticks += 1
                    ticks += 1
                    if done or reward > 0:
                        success = True
                        break
                    if ticks >= PROTOCOL["max_control_steps"]:
                        break
                row = None
                if args.mode == "online":
                    row = reference.predict(obs, env.instruction, capture=True)
                    following = encode_row(encoder, row, obs["state"])
                    replay.append(
                        {
                            **features,
                            "action": action_chunk,
                            **{f"next_{k}": v for k, v in following.items()},
                            "ticks": chunk_ticks,
                            "terminal": success,
                            "reward": chunk_reward,
                            "case_id": case["id"],
                        }
                    )
                    for _ in range(args.updates_per_chunk):
                        metrics = learner.update(replay)
                    write_json(
                        args.output / "online_status.json",
                        metrics | {"replay_chunks": len(replay)},
                    )
                if success:
                    break
            results.append(
                {
                    **case,
                    "success": success,
                    "ticks": ticks,
                    "seconds": time.monotonic() - started,
                    "state_sha256": env.reset_hash,
                }
            )
            write_json(args.output / "episodes.json", {"episodes": results})
            print(f"Episode {episode}: success={success} ticks={ticks}", flush=True)
            if args.mode == "online":
                save_heads(args.output / "online.pt", encoder, learner)
    finally:
        env.close()
    if args.collect:
        torch.save({"protocol": PROTOCOL, "rows": cache}, args.output / "features.pt")
    successes = sum(r["success"] for r in results)
    return {
        "episodes": results,
        "successes": successes,
        "success_rate": successes / len(results),
        "capture_parity": parity,
        "runtime_config": config,
        "feature_rows": len(cache),
        "gate_passed": successes >= args.min_successes,
        "actor_updates": learner.actor_updates if learner is not None else 0,
        "critic_updates": learner.critic_updates if learner is not None else 0,
        "objective": args.objective if learner is not None else "reference",
        "evaluation_role": "development smoke; never final held-out evaluation",
    }


def main() -> None:
    """Run a bounded reference gate with explicit artifacts and source pin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "reference",
            "warmup",
            "actor-eval",
            "online",
            "cpu-audit",
            "cpu-reference",
        ],
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--min-successes", type=int, default=1)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--state-start", type=int, default=0)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--encoder-steps", type=int, default=200)
    parser.add_argument("--bc-steps", type=int, default=2000)
    parser.add_argument("--updates-per-chunk", type=int, default=5)
    parser.add_argument("--objective", choices=["q_bc", "bc_only"], default="q_bc")
    args = parser.parse_args()
    if (
        not 0 <= args.task < 10
        or not 1 <= args.episodes <= 50
        or not 0 <= args.min_successes <= args.episodes
    ):
        parser.error("Invalid development case/gate budget")
    if not 0 <= args.state_start <= 50 - args.episodes:
        parser.error("Development states must not wrap modulo 50")
    if args.mode == "warmup" and (
        args.cache is None
        or not 1 <= args.encoder_steps <= 2000
        or not 1 <= args.bc_steps <= 10000
    ):
        parser.error("Warmup requires a feature cache and finite update budgets")
    if args.mode in {"actor-eval", "online"} and args.checkpoint is None:
        parser.error("Actor modes require an explicit checkpoint")
    if not 1 <= args.updates_per_chunk <= 10:
        parser.error("Require 1 to 10 updates per action chunk")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "protocol": PROTOCOL,
        "started_at": time.time(),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "versions": {
            name: metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "mujoco",
                "robosuite",
                "numpy",
            )
        },
        "adapter_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "pi05_baseline.py",
                "pi05_learning.py",
                "libero_acceptance_learning.py",
                "libero_acceptance_env.py",
            )
        },
    }
    try:
        configure_source(args.source)
        if args.mode in {"cpu-audit", "cpu-reference"}:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            report["device"] = {"device": "cpu"}
            if args.mode == "cpu-reference":
                os.environ.update(
                    MUJOCO_GL="osmesa",
                    PYOPENGL_PLATFORM="osmesa",
                    RLT_LIBERO_EGL_DEVICE_ID="0",  # Ignored by the OSMesa backend.
                    RLT_LIBERO_WORKER_LOG_DIR=str(args.output / "worker_logs"),
                    LIBERO_PYTHON=sys.executable,
                )
                report["device"]["renderer"] = "osmesa"
        else:
            report["device"] = configure_gpu(args.gpu, args.output)
        write_json(args.output / "manifest.json", report)
        if args.mode == "cpu-audit":
            report.update(cpu_audit(args))
        elif args.mode == "warmup":
            from toolkits.rlt.pi05_learning import warmup

            report.update(warmup(args))
        else:
            import torch

            context = (
                torch.autocast("cpu", dtype=torch.bfloat16)
                if args.mode == "cpu-reference"
                else nullcontext()
            )
            with context:
                report.update(evaluate(args))
        report["complete"] = True
        report["exit_code"] = 0 if report["gate_passed"] else 2
    except Exception as error:
        report.update(error=repr(error), exit_code=1)
        raise
    finally:
        report["finished_at"] = time.time()
        write_json(args.output / "manifest.json", report)
    raise SystemExit(report["exit_code"])


if __name__ == "__main__":
    main()
