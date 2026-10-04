# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint round trips preserve optimizer updates and bound restore memory."""

import pytest
import torch
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions

from rlinf.algorithms.rlt.checkpoint import (
    COUNTER_NAMES,
    load_training_state,
    save_training_state,
)
from rlinf.hybrid_engines.fsdp.strategy.checkpoint import Checkpoint
from rlinf.hybrid_engines.fsdp.utils import FSDPVersion
from rlinf.scheduler import Worker


@pytest.mark.parametrize("ready", [False, True])
def test_rlt_training_counters_round_trip(tmp_path, ready):
    directory = tmp_path / "global_step_800" / "actor"
    directory.mkdir(parents=True)
    counters = dict.fromkeys(COUNTER_NAMES, 0)
    counters.update(
        update_step=123456,
        total_transitions_added=200000,
        total_episodes_added=1000,
        pending_update_budget=137,
        _warmup_ready_total_transitions=10000 if ready else None,
        _warmup_ready_total_episodes=50 if ready else None,
    )
    schedule = {"warmup_post_collect_updates": 30000}
    save_training_state(
        str(directory),
        rank=0,
        world_size=1,
        step=800,
        counters=counters,
        schedule=schedule,
    )
    assert (
        load_training_state(str(directory), rank=0, world_size=1, schedule=schedule)
        == counters
    )
    with pytest.raises(ValueError, match="schedule differs"):
        load_training_state(str(directory), rank=0, world_size=1, schedule={})
    with pytest.raises(ValueError, match="rank layout"):
        load_training_state(str(directory), rank=0, world_size=2, schedule=schedule)


def test_rlt_missing_or_wrong_checkpoint_fails_closed(tmp_path):
    directory = tmp_path / "global_step_800" / "actor"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="Missing RLT training state"):
        load_training_state(str(directory), rank=0, world_size=1, schedule={})
    save_training_state(
        str(directory),
        rank=0,
        world_size=1,
        step=400,
        counters=dict.fromkeys(COUNTER_NAMES, 0),
        schedule={},
    )
    with pytest.raises(ValueError, match="different checkpoint"):
        load_training_state(str(directory), rank=0, world_size=1, schedule={})


@pytest.mark.parametrize(
    "key,value",
    [
        ("update_step", -1),
        ("update_step", 1.5),
        ("pending_update_budget", True),
        ("transitions_since_train", 1),
    ],
)
def test_rlt_rejects_invalid_counters(tmp_path, key, value):
    counters = dict.fromkeys(COUNTER_NAMES, 0)
    counters[key] = value
    with pytest.raises(ValueError):
        save_training_state(
            str(tmp_path), rank=0, world_size=1, step=1, counters=counters, schedule={}
        )
    assert not list(tmp_path.iterdir())


def test_rlt_worker_resume_preserves_next_budget_and_bc_weights(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

    # Substitute only the distributed model/replay I/O boundary. Scheduling and
    # worker save/load methods below are the production implementation.
    monkeypatch.setattr(EmbodiedSACFSDPPolicy, "save_checkpoint", lambda *args: None)
    monkeypatch.setattr(EmbodiedSACFSDPPolicy, "load_checkpoint", lambda *args: None)
    monkeypatch.setattr(RLTACFSDPPolicy, "log_info", lambda *args: None)
    monkeypatch.setattr(
        RLTACFSDPPolicy,
        "_global_rlt_counters",
        lambda self: {
            "min_replay_size": 20000,
            "min_demo_size": 0,
            "total_transitions_added": self.total_transitions_added,
            "total_episodes_added": self.total_episodes_added,
            "transitions_since_train": self.transitions_since_train,
        },
    )

    def worker():
        result = object.__new__(RLTACFSDPPolicy)
        result.cfg = OmegaConf.create(
            {
                "algorithm": {
                    "replay_buffer": {"min_buffer_size": 10000},
                    "update_epoch": 5,
                    "actor_weight_schedule": {
                        "enable": True,
                        "warmup_updates": 20000,
                        "ramp_updates": 50000,
                        "online_bc_weight": 2.5,
                        "online_q_weight": 0.45,
                    },
                }
            }
        )
        result.rlt_schedule_cfg = OmegaConf.create(
            {
                "enable": True,
                "warmup_post_collect_updates": 30000,
                "train_every_transitions": 5,
                "max_updates_per_train_step": 400,
            }
        )
        result.use_rlt_schedule = True
        result.critic_actor_ratio = 1
        result._rank, result._world_size = 0, 1
        for name in COUNTER_NAMES:
            setattr(result, name, 0)
        return result

    source = worker()
    source.update_step = 100000
    source.total_transitions_added = 80500
    source.total_episodes_added = 800
    source._warmup_ready_total_transitions = 10000
    source._warmup_ready_total_episodes = 100
    expected_budget = source._rlt_updates_to_run()
    expected_weights = source._actor_objective_weights()
    source.save_checkpoint(str(tmp_path), 800)
    target = worker()
    target.load_checkpoint(str(tmp_path))
    assert target.get_rollout_sync_version() == 100000
    assert target._rlt_updates_to_run() == expected_budget
    assert target._actor_objective_weights() == expected_weights
    assert expected_budget[0] == 400


@pytest.fixture(autouse=True)
def _torch_platform(monkeypatch):
    # These tests run outside a scheduler worker, including on CPU-only hosts.
    monkeypatch.setattr(Worker, "torch_platform", torch.cuda)
    monkeypatch.setattr(Worker, "torch_device_type", "cuda")


def _training_state(device, width=8, multiple=False):
    model = torch.nn.Linear(width, width, device=device)
    parameters = ([model.weight], [model.bias]) if multiple else (model.parameters(),)
    optimizers = [torch.optim.AdamW(params, lr=0.01) for params in parameters]
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, 1, 0.9) for opt in optimizers]
    checkpoint = Checkpoint(
        model,
        optimizers if multiple else optimizers[0],
        schedulers,
        StateDictOptions(full_state_dict=False, cpu_offload=True),
        FSDPVersion.FSDP,
    )
    return model, optimizers, schedulers, checkpoint


def _update(model, optimizers, schedulers):
    model(
        torch.ones(2, model.in_features, device=model.weight.device)
    ).square().mean().backward()
    for optimizer, scheduler in zip(optimizers, schedulers):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()


@pytest.mark.parametrize("multiple", [False, True])
def test_checkpoint_restores_optimizer_scheduler_and_next_update(tmp_path, multiple):
    source = _training_state("cpu", multiple=multiple)
    for _ in range(3):
        _update(*source[:3])
    dcp.save({"training": source[3]}, checkpoint_id=tmp_path / "checkpoint")
    target = _training_state("cpu", multiple=multiple)
    dcp.load({"training": target[3]}, checkpoint_id=tmp_path / "checkpoint")
    for source_opt, target_opt in zip(source[1], target[1]):
        assert source_opt.param_groups[0]["lr"] == target_opt.param_groups[0]["lr"]
        for source_state, target_state in zip(
            source_opt.state.values(), target_opt.state.values()
        ):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    source_state[key], target_state[key], rtol=0, atol=0
                )
    for source_scheduler, target_scheduler in zip(source[2], target[2]):
        assert source_scheduler.state_dict() == target_scheduler.state_dict()
    _update(*source[:3])
    _update(*target[:3])
    for source_param, target_param in zip(
        source[0].parameters(), target[0].parameters()
    ):
        torch.testing.assert_close(source_param, target_param, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires an isolated CUDA GPU"
)
def test_checkpoint_restore_avoids_duplicate_gpu_optimizer_states():
    model, optimizers, schedulers, checkpoint = _training_state("cuda", width=2048)
    _update(model, optimizers, schedulers)
    state = checkpoint.state_dict()
    optimizer_bytes = sum(
        value.numel() * value.element_size()
        for values in optimizers[0].state.values()
        for value in values.values()
        if isinstance(value, torch.Tensor) and value.is_cuda
    )
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    checkpoint.load_state_dict(state)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - before < optimizer_bytes // 2
    assert all(values["step"].item() == 1 for values in optimizers[0].state.values())
    _update(model, optimizers, schedulers)
    assert all(values["step"].item() == 2 for values in optimizers[0].state.values())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires an isolated CUDA GPU"
)
def test_checkpoint_fsdp_restores_optimizer_and_can_update(tmp_path):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1
    )
    try:
        model = FSDP(
            torch.nn.Linear(2048, 2048, device="cuda"),
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.NO_SHARD,
            use_orig_params=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
        checkpoint = Checkpoint(
            model,
            optimizer,
            scheduler,
            StateDictOptions(full_state_dict=False, cpu_offload=True),
            FSDPVersion.FSDP,
        )
        _update(model, [optimizer], [scheduler])
        dcp.save({"training": checkpoint}, checkpoint_id=tmp_path / "checkpoint")
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        dcp.load({"training": checkpoint}, checkpoint_id=tmp_path / "checkpoint")
        torch.cuda.synchronize()
        # DCP stages one 16 MiB weight tensor; two duplicate Adam moments
        # would require another 32 MiB above the pre-load allocation.
        assert torch.cuda.max_memory_allocated() - before < 24 * 1024**2
        assert all(state["step"].item() == 1 for state in optimizer.state.values())
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.009)
        _update(model, [optimizer], [scheduler])
        assert all(state["step"].item() == 2 for state in optimizer.state.values())
    finally:
        torch.distributed.destroy_process_group()
