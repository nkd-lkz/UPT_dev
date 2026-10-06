# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real simulation, frozen features and online updates on GPU 2."""

import argparse
import json
import logging
import secrets
import threading
import time
from pathlib import Path

from .gpu_guard import isolate_gpu2, verify_cuda


def main() -> None:
    """Run bounded scripted intervention; no physical headset is impersonated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    if not 16 <= args.steps <= 100:
        parser.error("--steps must be between 16 and 100")
    logging.basicConfig(level=logging.INFO)
    gpu = isolate_gpu2()
    verify_cuda(gpu["uuid"])
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from .async_service import AsyncOnlineService
    from .online_learner import OnlineLearner
    from .online_service import ONLINE_PROTOCOL, OnlineService, frozen_feature_identity
    from .online_transport import encode_observation
    from .protocol import request
    from .server import ConcurrentInferenceServer, RLTInference
    from .simulation import LocalSimulation

    args.output.mkdir(parents=True, exist_ok=False)
    config = OmegaConf.to_container(
        OmegaConf.load(Path(__file__).with_name("online_smoke.yaml"))
    )
    model = RLTInference(
        Path(__file__).with_name("model.yaml"), args.stage1, args.dataset, None
    )
    model.feature.requires_grad_(False)
    learner = OnlineLearner(config, "cuda:0")
    before_actor = torch.cat([p.detach().flatten().cpu() for p in learner.actor_params])
    before_critic = torch.cat(
        [p.detach().flatten().cpu() for p in learner.critic_params]
    )
    service = OnlineService(
        learner,
        model.extract,
        args.output,
        frozen_feature_identity(args.stage1, args.dataset),
    )
    token, session = secrets.token_hex(32), secrets.token_hex(16)
    env = LocalSimulation(gpu["render_backend"], max_episode_steps=20)
    modes = set()
    episode = 0
    duplicate_verified = False
    transport = None
    try:
        transport = AsyncOnlineService(service)
        with ConcurrentInferenceServer(
            ("127.0.0.1", 0), token, model, "smoke", dispatch=transport
        ) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]

            def rpc(payload):
                return request(
                    "127.0.0.1",
                    port,
                    token,
                    {"session": session, "request_id": 0, **payload},
                    60,
                )

            try:
                rpc({"op": "begin", "online_protocol": ONLINE_PROTOCOL})
                obs = env.observation()
                for step in range(args.steps):
                    response = rpc({"op": "predict", **encode_observation(obs)})
                    modes.add(response["policy_source"])
                    human = 4 <= step % 20 < 10
                    if human:
                        target = env.tcp_matrix()
                        target[2, 3] += 0.002
                        action = env.human_action(target, 1.0)
                        if action is None:
                            raise RuntimeError("Scripted intervention IK failed")
                    else:
                        action = np.asarray(response["actions"][0], dtype=np.float32)
                    nxt, reward, terminated, truncated = env.step(action)
                    payload = {
                        "op": "observe",
                        "sequence": step,
                        "episode": episode,
                        "observation": encode_observation(obs),
                        "next_observation": encode_observation(nxt),
                        "action": action.tolist(),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "human": human,
                        # Explicit scripted labels for this engineering test only.
                        "quality": "approved" if human else "policy",
                        "policy_source": "human"
                        if human
                        else response["policy_source"],
                        "policy_version": -1
                        if human
                        else response["metrics"]["policy_version"],
                    }
                    ack = rpc(payload)
                    if step == 10:
                        duplicate = rpc(payload)
                        assert (
                            duplicate["duplicate"]
                            and duplicate["received_sequence"] == step
                        )
                        duplicate_verified = True
                    deadline = time.monotonic() + 60
                    while (
                        ack["metrics"]["accepted"] < step + 1 or ack["pending_learning"]
                    ):
                        if ack["faulted"] or time.monotonic() >= deadline:
                            raise RuntimeError(
                                "Durable receipt was not processed within smoke budget"
                            )
                        time.sleep(0.05)
                        ack = rpc({"op": "status"})
                    logging.info(
                        "step=%d human=%s status=%s", step, human, ack["metrics"]
                    )
                    obs = nxt
                    if terminated or truncated:
                        episode += 1
                        obs = env.reset(episode)
            finally:
                server.shutdown()
                thread.join(timeout=10)
        transport.close()
        restored = OnlineLearner(config, "cuda:0")
        metadata = restored.load(args.output / "learner.pt")
        for key, value in learner.model.state_dict().items():
            torch.testing.assert_close(
                value, restored.model.state_dict()[key], rtol=0, atol=0
            )
        assert metadata["sequence"] == args.steps - 1
        assert restored.actor_optim.state and restored.critic_optim.state
        resumed_update = restored.observe(restored.replay[-1])["update_step"]
        assert resumed_update == learner.update_step + 1
        actor_delta = float(
            (
                torch.cat([p.detach().flatten().cpu() for p in learner.actor_params])
                - before_actor
            )
            .abs()
            .max()
        )
        critic_delta = float(
            (
                torch.cat([p.detach().flatten().cpu() for p in learner.critic_params])
                - before_critic
            )
            .abs()
            .max()
        )
        assert actor_delta > 0 and critic_delta > 0
        assert (
            learner.human_accepted > 0
            and learner.actor_updates > 0
            and learner.version > 0
        )
        assert modes == {"reference", "actor"}
        result = {
            "gpu": gpu,
            "steps": args.steps,
            "episodes_ended": episode,
            "scripted_intervention_only": True,
            "physical_pico_tested": False,
            "online_protocol": ONLINE_PROTOCOL,
            "duplicate_verified": duplicate_verified,
            "resume_verified": True,
            "resumed_optimizer_update": resumed_update,
            "actor_max_change": actor_delta,
            "critic_max_change": critic_delta,
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2,
            **learner.status(),
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        logging.info("PASS: %s", result)
    finally:
        env.close()
        if transport is not None:
            transport.close()


if __name__ == "__main__":
    main()
