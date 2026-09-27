# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate physical GPU assignment before loading an RLT model."""

import argparse
import os
import subprocess

import ray
from hydra import compose, initialize_config_dir

from rlinf.scheduler import Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


class GPUProbe(Worker):
    """Allocate one scalar only after checking the worker device mask."""

    def inspect(self, expected_uuid: str) -> dict:
        """Check device visibility and the CUDA device's physical UUID."""
        import torch

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != "2":
            raise RuntimeError(f"Unsafe GPU assignment: {visible!r}; expected '2'")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Smoke worker must see exactly one CUDA device")
        props = torch.cuda.get_device_properties(0)
        actual_uuid = str(props.uuid)
        if actual_uuid.lower().removeprefix(
            "gpu-"
        ) != expected_uuid.lower().removeprefix("gpu-"):
            raise RuntimeError(f"Wrong physical GPU: {actual_uuid} != {expected_uuid}")
        value = torch.ones(1, device="cuda:0")
        torch.cuda.synchronize()
        return {
            "pid": os.getpid(),
            "visible": visible,
            "uuid": actual_uuid,
            "device": props.name,
            "scalar": value.item(),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        default=os.environ["EMBODIED_PATH"] + "/config",
    )
    parser.add_argument(
        "--config-name",
        default="maniskill_rlt_stage2_smoke_gpu2",
    )
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    """Resolve all component placements, then probe one real RLinf worker."""
    args = _parse_args()
    if not os.environ.get("RAY_ADDRESS"):
        raise RuntimeError("An explicit, isolated RAY_ADDRESS is required")
    expected_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "2", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    with initialize_config_dir(config_dir=args.config_dir, version_base="1.1"):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    components = {
        component.strip()
        for key in cfg.cluster.component_placement
        for component in str(key).split(",")
    }
    for component in sorted(components):
        entries = placement.get_strategy(component).get_placement(cluster)
        if len(entries) != 1 or entries[0].visible_accelerators != ["2"]:
            raise RuntimeError(f"Unsafe {component} placement: {entries}")
        print(f"Verified {component} placement: {entries[0]}", flush=True)
    group = GPUProbe.create_group().launch(
        cluster,
        name="GPU2IsolationProbe",
        placement_strategy=placement.get_strategy("env"),
    )
    try:
        print(f"GPU_PROBE_OK: {group.inspect(expected_uuid).wait()}", flush=True)
    finally:
        for info in group.worker_info_list:
            ray.kill(info.worker, no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
