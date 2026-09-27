# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Serve one Windows VR environment with a bounded GPU-2 RLT learner."""

import argparse
import json
import logging
import os
from pathlib import Path

from .gpu_guard import isolate_gpu2, verify_cuda


def main() -> None:
    """Start an isolated learner; Ctrl-C saves a resumable checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", default=8775, type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("online_smoke.yaml")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    token = os.environ.get("RLT_VR_TOKEN", "")
    if len(token) < 32:
        parser.error("Set RLT_VR_TOKEN to a secret of at least 32 characters")
    weights = args.stage1 / "model_state_dict/full_weights.pt"
    stats = args.dataset / "norm_stats.json"
    if not weights.is_file() or not stats.is_file():
        parser.error("Stage1 full_weights.pt and dataset norm_stats.json are required")
    gpu = isolate_gpu2()
    verify_cuda(gpu["uuid"])
    from omegaconf import OmegaConf

    from .online_learner import OnlineLearner
    from .online_service import OnlineService, frozen_feature_identity
    from .server import InferenceServer, RLTInference

    # File identity and content hash of stats prevent accidentally mixing feature
    # spaces. Checkpoint files are immutable exports; never overwrite them.
    feature_id = frozen_feature_identity(args.stage1, args.dataset)
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(
        json.dumps({"learner": config, "gpu": gpu, "feature_id": feature_id}, indent=2)
    )
    model = RLTInference(
        Path(__file__).with_name("model.yaml"), args.stage1, args.dataset, None
    )
    model.feature.requires_grad_(False)
    learner = OnlineLearner(config, device="cuda:0")
    service = OnlineService(learner, model.extract, args.output, feature_id)
    if args.resume:
        service.restore(learner.load(args.resume))
        # A resumed server starts a fresh client session/episode. Replay and
        # optimizer state persist; old unacknowledged packets must not be reused.
        service.session, service.sequence, service.episode = None, -1, -1
    with InferenceServer(
        ("127.0.0.1", args.port), token, model, "online", dispatch=service
    ) as server:
        logging.info(
            "Online learner ready: GPU2=%s port=%d horizon=1", gpu["uuid"], args.port
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if not service.faulted:
                service.checkpoint()
            else:
                logging.error("Learner faulted; last good checkpoint retained")


if __name__ == "__main__":
    main()
