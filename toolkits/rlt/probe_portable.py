# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Probe RLinf placement and RGB simulation on one physical GPU."""

import argparse
import os
import subprocess

import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.scheduler import Cluster, Worker


def validate_placement(cfg, gpu: int) -> None:
    """Reject a config that places any component outside the requested GPU."""
    expected = dict.fromkeys(("actor", "env", "rollout"), f"{gpu}-{gpu}")
    if dict(cfg.cluster.component_placement) != expected:
        raise ValueError(f"Expected physical placement {expected}")


class PortableGPUProbe(Worker):
    """Check CUDA identity before allocating and rendering a single environment."""

    def inspect(self, gpu: int, expected_uuid: str, env_args: dict) -> dict:
        """Verify the selected device, then reset, render and step the task."""
        import torch

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != str(gpu):
            raise RuntimeError(f"Unsafe CUDA mask {visible!r}; expected {gpu}")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Probe worker must see exactly one CUDA device")
        props = torch.cuda.get_device_properties(0)
        actual_uuid = str(props.uuid)
        if actual_uuid.lower().removeprefix(
            "gpu-"
        ) != expected_uuid.lower().removeprefix("gpu-"):
            raise RuntimeError(f"Wrong GPU UUID: {actual_uuid} != {expected_uuid}")

        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
            patch_rlt_openpi_joint_env_args,
        )
        from rlinf.envs.sim.maniskill.utils import allow_pci_render_backend

        allow_pci_render_backend()
        env_args = patch_rlt_openpi_joint_env_args(
            dict(env_args, num_envs=1), wrap_obs_mode="rlt_openpi_joint"
        )
        env = gym.make(**env_args)
        try:
            obs, _ = env.reset(seed=2026)
            images = {
                name: list(data["rgb"].shape)
                for name, data in obs["sensor_data"].items()
            }
            if not images:
                raise RuntimeError("No RGB images returned")
            action = torch.zeros((1, 8), device="cuda:0")
            _, reward, _, _, _ = env.step(action)
            if not torch.isfinite(reward).all():
                raise RuntimeError("Nonfinite simulation reward")
            torch.cuda.synchronize()
            return {
                "pid": os.getpid(),
                "gpu": gpu,
                "uuid": actual_uuid,
                "device": props.name,
                "images": images,
            }
        finally:
            env.close()


def main() -> None:
    """Check all placements and run the probe through the actual scheduler."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, type=int)
    args, overrides = parser.parse_known_args()
    if args.gpu < 0 or not os.environ.get("RAY_ADDRESS"):
        raise ValueError(
            "A nonnegative GPU index and isolated RAY_ADDRESS are required"
        )
    expected_uuid = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    with initialize_config_dir(
        config_dir=os.environ["EMBODIED_PATH"] + "/config", version_base="1.1"
    ):
        cfg = compose(
            config_name="maniskill_rlt_stage2_smoke_gpu2", overrides=overrides
        )
    validate_placement(cfg, args.gpu)

    from rlinf.utils.placement import HybridComponentPlacement

    cluster = Cluster(cluster_cfg=cfg.cluster)
    group = None
    try:
        placement = HybridComponentPlacement(cfg, cluster)
        for component in ("actor", "env", "rollout"):
            entries = placement.get_strategy(component).get_placement(cluster)
            if len(entries) != 1 or entries[0].visible_accelerators != [str(args.gpu)]:
                raise RuntimeError(f"Unsafe {component} placement: {entries}")
        group = PortableGPUProbe.create_group().launch(
            cluster,
            name="PortableIsolationProbe",
            placement_strategy=placement.get_strategy("env"),
        )
        result = group.inspect(
            args.gpu,
            expected_uuid,
            OmegaConf.to_container(cfg.env.train.init_params, resolve=True),
        ).wait()
        from rlinf.utils.logging import get_logger

        get_logger().info("GPU_AND_RENDER_PROBE_OK: %s", result)
    finally:
        if group is not None:
            for info in group.worker_info_list:
                ray.kill(info.worker, no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
