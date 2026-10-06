# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Serve one Windows VR environment with a bounded GPU-2 RLT learner."""

import argparse
import json
import logging
import os
from pathlib import Path

from .gpu_guard import gpu2_lease, isolate_gpu2, verify_cuda
from .online_settings import validate_config


def main() -> None:
    """Start an isolated learner; Ctrl-C saves a resumable checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", default=8775, type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-pending", type=int, default=128)
    parser.add_argument("--journal-limit-gib", type=float, default=4)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check paths/config without CUDA, token or server startup",
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("online_smoke.yaml")
    )
    args = parser.parse_args()
    if not 1 <= args.max_pending <= 512 or not 0.1 <= args.journal_limit_gib <= 32:
        parser.error("Invalid durable inbox quota")
    logging.basicConfig(level=logging.INFO)
    weights = args.stage1 / "model_state_dict/full_weights.pt"
    stats = args.dataset / "norm_stats.json"
    if not weights.is_file() or not stats.is_file():
        parser.error("Stage1 full_weights.pt and dataset norm_stats.json are required")
    from omegaconf import OmegaConf

    config = validate_config(
        OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    )
    if (
        config["z_dim"],
        config["proprio_dim"],
        config["action_dim"],
        config["reference_horizon"],
    ) != (2048, 9, 8, 10):
        parser.error("Server requires the existing Panda/Stage1 feature dimensions")
    if args.resume and not args.resume.is_file():
        parser.error("Resume checkpoint does not exist")
    if args.check:
        logging.info(
            "Path/config check passed: %s; Stage1=%s; no GPU/RPC tested",
            config,
            weights,
        )
        return
    token = os.environ.get("RLT_VR_TOKEN", "")
    if len(token) < 32:
        parser.error("Set RLT_VR_TOKEN to a secret of at least 32 characters")
    with gpu2_lease():
        serve(args, config, token)


def serve(args: argparse.Namespace, config: dict, token: str) -> None:
    """Own GPU isolation, model lifetime and resumable authenticated service."""
    gpu = isolate_gpu2()
    verify_cuda(gpu["uuid"])

    from .async_service import AsyncOnlineService
    from .online_learner import OnlineLearner
    from .online_service import ONLINE_PROTOCOL, OnlineService, frozen_feature_identity
    from .server import ConcurrentInferenceServer, RLTInference

    # File identity and content hash of stats prevent accidentally mixing feature
    # spaces. Checkpoint files are immutable exports; never overwrite them.
    feature_id = frozen_feature_identity(args.stage1, args.dataset)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "learner": config,
                "gpu": gpu,
                "feature_id": feature_id,
                "online_protocol": ONLINE_PROTOCOL,
                "max_pending_learning": args.max_pending,
                "journal_limit_gib": args.journal_limit_gib,
            },
            indent=2,
        )
    )
    model = RLTInference(
        Path(__file__).with_name("model.yaml"), args.stage1, args.dataset, None
    )
    model.feature.requires_grad_(False)
    learner = OnlineLearner(config, device="cuda:0")
    service = OnlineService(learner, model.extract, args.output, feature_id)
    metadata = None
    if args.resume:
        metadata = learner.load(args.resume)
        service.restore(metadata)
    transport = AsyncOnlineService(
        service,
        restored=metadata,
        max_pending=args.max_pending,
        max_journal_bytes=int(args.journal_limit_gib * 1024**3),
    )
    try:
        with ConcurrentInferenceServer(
            ("127.0.0.1", args.port), token, model, "online", dispatch=transport
        ) as server:
            logging.info(
                "Online learner ready: GPU2=%s port=%d horizon=1",
                gpu["uuid"],
                args.port,
            )
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        transport.close()


if __name__ == "__main__":
    main()
