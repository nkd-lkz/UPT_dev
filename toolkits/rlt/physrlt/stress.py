# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Screen reference headroom under explicit software actuation stress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toolkits.rlt.libero_pro_environment import LiberoProEnvironment
from toolkits.rlt.physrlt.control import rollout
from toolkits.rlt.physrlt.data import PhysicalReference
from toolkits.rlt.pi05_baseline import (
    configure_gpu,
    configure_source,
    load_model,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "assets", "base-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Completed stress screens are immutable")
    configure_gpu(args.gpu, args.output)
    reservation = torch.empty(1, device="cuda")
    configure_source(args.source)
    torch.set_num_threads(4)
    model, _ = load_model(args.assets, "cuda")
    reference = PhysicalReference(model)
    conditions = [(1.0, 0), (0.7, 0), (0.4, 0), (1.0, 2), (1.0, 5)]
    env = LiberoProEnvironment()
    env.pilot_reset_hashes = {}
    records = []
    try:
        for path in sorted(args.base_root.glob("task*/cases.json")):
            # Development selection uses training layouts only, not test outcomes.
            cases = [c for c in json.loads(path.read_text()) if c["split"] == "train"][
                :2
            ]
            for gain, delay in conditions:
                for case in cases:
                    result = rollout(
                        env,
                        reference,
                        None,
                        None,
                        case,
                        520,
                        execution_gain=gain,
                        delay_ticks=delay,
                    )
                    records.append(
                        result
                        | {
                            "task": case["task"],
                            "method": "reference",
                            "role": "custom_actuation_stress_screen",
                        }
                    )
                    write_json(args.output / "episodes.json", records)
    finally:
        env.close()
        del reservation
    write_json(
        args.output / "result.json",
        {
            "state": "complete",
            "episodes": len(records),
            "conditions": conditions,
            "interpretation": "Software translation-command gain and command delay; not official LIBERO-Pro scores or changed friction/contact stiffness",
            "online_rl": False,
            "delay_warning": "Delayed online RL requires pending-command state augmentation and is intentionally disabled",
        },
    )


if __name__ == "__main__":
    main()
