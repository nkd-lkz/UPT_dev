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

"""Shared helpers: metrics, entropy aggregation, checkpoint paths, and resume."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.utils import compute_entropy_loss
from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.utils.metric_utils import compute_evaluate_metrics, compute_rollout_metrics


@pytest.mark.parametrize("gpu", [0, 1, 2])
def test_portable_pilot_preserves_budget_and_physical_placement(
    monkeypatch, tmp_path, gpu
):
    from hydra import compose, initialize_config_dir

    from toolkits.rlt.probe_portable import validate_placement

    root = Path(__file__).resolve().parents[2]
    for key, value in {
        "EMBODIED_PATH": str(root / "examples/embodiment"),
        "RLT_STAGE1_ACTOR": str(tmp_path / "actor"),
        "RLT_DATASET_DIR": str(tmp_path / "data"),
        "RLT_SMOKE_RUN_DIR": str(tmp_path / "run"),
        "RLT_SMOKE_RENDER_DEVICE": "pci:0000:46:00.0",
        "RLT_WORLD_CHECKPOINT": str(tmp_path / "world.pt"),
    }.items():
        monkeypatch.setenv(key, value)
    experiment = (
        "rlt_latent_world"
        if (
            root / "examples/embodiment/config/experiment/rlt_latent_world.yaml"
        ).exists()
        else "rlt_memory"
    )
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="maniskill_rlt_stage2_smoke_gpu2",
            overrides=[
                f"+experiment={experiment}",
                "+pilot=rlt_overnight",
                *[
                    f"cluster.component_placement.{name}={gpu}-{gpu}"
                    for name in ("actor", "env", "rollout")
                ],
            ],
        )
    OmegaConf.resolve(cfg)
    validate_placement(cfg, gpu)
    with pytest.raises(ValueError, match="physical placement"):
        validate_placement(cfg, gpu + 1)
    assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates == 512
    assert cfg.algorithm.actor_weight_schedule.warmup_q_weight == 0.0
    assert cfg.actor.global_batch_size == 32
    assert cfg.actor.micro_batch_size == 8
    assert cfg.env.eval.total_num_envs == 4
    assert cfg.env.eval.video_cfg.save_video
    assert cfg.rollout.expert_model is None
    assert not cfg.env.train.rlt_policy_switch.expert_takeover.enable
    if experiment == "rlt_memory":
        from rlinf.algorithms.rlt.interaction_memory import (
            validate_interaction_memory_cfg,
        )

        validate_interaction_memory_cfg(cfg)


def test_portable_launcher_rejects_unsafe_budgets_before_loading_models():
    root = Path(__file__).resolve().parents[2]
    for overrides in [
        {"RLT_SMOKE_STEPS": "0"},
        {"RLT_SMOKE_STEPS": "5001", "RLT_LONG_RUN": "1"},
        {"RLT_EPISODE_STEPS": "501"},
        {"RLT_SMOKE_STEPS": "20", "RLT_SAVE_INTERVAL": "10", "RLT_VAL_INTERVAL": "3"},
    ]:
        result = subprocess.run(
            ["bash", str(root / "run_rlt_portable.sh"), "--check"],
            env={**os.environ, "RLT_PHYSICAL_GPU": "0", **overrides},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 2, result.stderr


def test_portable_busy_gpu_stops_before_ray_or_output_creation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    venv = Path(sys.prefix)
    if not (venv / "bin/activate").is_file():
        pytest.skip("Launcher requires an activated virtualenv")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_smi = bin_dir / "nvidia-smi"
    fake_smi.write_text(
        '#!/bin/sh\ncase "$*" in\n'
        "*memory.used*) echo 18077;;\n"
        "*) echo 00000000:46:00.0;;\nesac\n"
    )
    fake_smi.chmod(0o755)
    actor = tmp_path / "actor"
    (actor / "model_state_dict").mkdir(parents=True)
    (actor / "model_state_dict/full_weights.pt").write_bytes(b"preflight-only")
    data = tmp_path / "data"
    data.mkdir()
    (data / "norm_stats.json").write_text('{"norm_stats":{"state":{}, "actions":{}}}')
    lib = tmp_path / "placeholder"
    lib.touch()
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "RLINF_VENV": str(venv),
        "RLT_PHYSICAL_GPU": "0",
        "RLT_STAGE1_ACTOR": str(actor),
        "RLT_DATASET_DIR": str(data),
        "RLT_OUTPUT_ROOT": str(tmp_path / "runs"),
        "RLT_SMOKE_STEPS": "2",
        "RLT_SMOKE_PROFILE": "smoke",
        "RLT_SAVE_INTERVAL": "1",
        "RLT_VAL_INTERVAL": "1",
        "SAPIEN_VULKAN_LIBRARY_PATH": str(lib),
        "RLT_NVIDIA_EGL_LIBRARY": str(lib),
        "__EGL_VENDOR_LIBRARY_FILENAMES": str(lib),
    }
    result = subprocess.run(
        ["bash", str(root / "run_rlt_portable.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "GPU 0 is busy" in result.stderr, result.stdout + result.stderr
    assert not (tmp_path / "runs").exists()


def test_portable_launcher_requires_explicit_device():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.pop("RLT_PHYSICAL_GPU", None)
    result = subprocess.run(
        ["bash", str(root / "run_rlt_portable.sh"), "--check"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Set RLT_PHYSICAL_GPU" in result.stderr


@pytest.fixture
def stage2_smoke_config(monkeypatch, tmp_path):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    monkeypatch.setenv("RLT_STAGE1_ACTOR", str(tmp_path / "stage1/actor"))
    monkeypatch.setenv("RLT_DATASET_DIR", str(tmp_path / "dataset"))
    monkeypatch.setenv("RLT_SMOKE_RUN_DIR", str(tmp_path / "smoke"))
    monkeypatch.setenv("RLT_SMOKE_RENDER_DEVICE", "pci:0000:e1:00.0")
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="maniskill_rlt_stage2_smoke_gpu2")
    OmegaConf.resolve(cfg)
    return cfg


def test_stage2_smoke_config_is_single_gpu_and_bounded(stage2_smoke_config):
    cfg = stage2_smoke_config
    assert dict(cfg.cluster.component_placement) == {
        "actor": "2-2",
        "env": "2-2",
        "rollout": "2-2",
    }
    assert cfg.runner.max_steps == cfg.runner.max_epochs == 2
    assert cfg.runner.val_check_interval == cfg.runner.save_interval == 1
    assert cfg.actor.global_batch_size == 4
    assert cfg.actor.micro_batch_size == 2
    assert cfg.env.train.total_num_envs == 2
    assert cfg.env.eval.total_num_envs == 1
    assert cfg.rollout.expert_model is None
    assert cfg.rollout.rlt_feature_model.openpi.use_rlt
    assert not cfg.rollout.rlt_feature_model.openpi.torch_compile
    assert not cfg.actor.model.model_path
    assert cfg.rollout.rlt_feature_model.openpi_data.norm_stats_path.endswith(
        "/dataset/norm_stats.json"
    )


def test_stage2_smoke_config_can_reach_replay_updates(stage2_smoke_config):
    cfg = stage2_smoke_config
    for mode in (cfg.env.train, cfg.env.eval):
        assert mode.init_params.sim_backend == "physx_cuda:0"
        assert mode.init_params.render_backend == "pci:0000:e1:00.0"
        assert mode.rlt_policy_switch.trigger_mode == "always_on"
        assert not mode.rlt_policy_switch.expert_takeover.enable
        assert mode.max_episode_steps == mode.max_steps_per_rollout_epoch == 40
        assert mode.num_action_chunks == 10
    chunk_transitions = (
        cfg.env.train.total_num_envs
        * cfg.env.train.max_steps_per_rollout_epoch
        // cfg.env.train.num_action_chunks
    )
    assert chunk_transitions >= cfg.algorithm.rlt_schedule.warmup_min_size
    assert cfg.algorithm.replay_buffer.min_buffer_size >= cfg.actor.global_batch_size
    # A zero warmup budget would skip learner updates on the first collection.
    assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates == 2
    assert cfg.algorithm.rlt_schedule.max_updates_per_train_step == 2
    assert cfg.algorithm.critic_actor_ratio == 1


def test_stage2_smoke_launcher_rejects_missing_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[2]
    if not (root / ".venv/bin/activate").is_file():
        pytest.skip("Host-specific launcher needs the configured baseline venv")
    result = subprocess.run(
        ["bash", str(root / "run_rlt_stage2_smoke_gpu2.sh"), "--check"],
        env={**os.environ, "RLT_STAGE1_ACTOR": str(tmp_path / "missing")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Missing weights:" in result.stderr


def test_stage2_smoke_launcher_rejects_incompatible_intervals():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "run_rlt_stage2_smoke_gpu2.sh"), "--check"],
        env={
            **os.environ,
            "RLT_LONG_RUN": "1",
            "RLT_SMOKE_STEPS": "100",
            "RLT_SAVE_INTERVAL": "25",
            "RLT_VAL_INTERVAL": "10",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "RLT_SAVE_INTERVAL must be divisible" in result.stderr


def test_compute_evaluate_metrics_reports_interact_delay_wait_time_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "success": torch.tensor([1.0, 0.0]),
                "interact_delay": torch.tensor([0.10, 0.30]),
            },
            {
                "success": torch.tensor([0.0, 1.0]),
                "interact_delay": torch.tensor([0.20, 0.40]),
            },
        ]
    )

    assert math.isclose(float(metrics["success"]), 0.5)
    assert float(metrics["average_delay"]) == pytest.approx(0.25)
    assert float(metrics["median_delay"]) == pytest.approx(0.25)
    assert float(metrics["max_delay"]) == pytest.approx(0.40)
    assert float(metrics["min_delay"]) == pytest.approx(0.10)
    assert metrics["num_trajectories"] == 4


def test_compute_evaluate_metrics_ignores_delay_samples_for_trajectory_count():
    metrics = compute_evaluate_metrics(
        [{"interact_delay": torch.tensor([0.05, 0.15, 0.25])}]
    )

    assert float(metrics["average_delay"]) == pytest.approx(0.15)
    assert metrics["num_trajectories"] == 0


def test_compute_evaluate_metrics_reports_prefixed_interact_delay_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "env/success": torch.tensor([1.0]),
                "env/interact_delay": torch.tensor([0.12, 0.24]),
            }
        ]
    )

    assert float(metrics["env/average_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/median_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/max_delay"]) == pytest.approx(0.24)
    assert float(metrics["env/min_delay"]) == pytest.approx(0.12)


@pytest.fixture
def single_rank_reduction(monkeypatch):
    from rlinf.scheduler.worker.worker import Worker

    monkeypatch.setattr(
        Worker, "torch_platform", SimpleNamespace(current_device=lambda: "cpu")
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)


def test_compute_rollout_metrics_reports_loss_mask_fraction(single_rank_reduction):
    metrics = compute_rollout_metrics(
        {
            "loss_mask": torch.tensor([[[True], [False]], [[True], [True]]]),
            "rewards": torch.tensor([[[1.0], [8.0]], [[2.0], [3.0]]]),
        }
    )

    assert metrics["loss_mask_fraction"] == pytest.approx(0.75)
    assert metrics["rewards"] == pytest.approx(2.0)


def test_compute_rollout_metrics_omits_loss_mask_fraction_without_mask(
    single_rank_reduction,
):
    metrics = compute_rollout_metrics({"rewards": torch.tensor([[[1.0], [3.0]]])})

    assert "loss_mask_fraction" not in metrics
    assert metrics["rewards"] == pytest.approx(2.0)


# The embodied actor's entropy bonus aggregation. The shapes below are the ones
# the shipped models actually emit: openpi, lingbotvla,
# dexbotic_pi, dexbotic_dm0 and flow_policy all reduce entropy to [bsz, 1],
# cnn_policy returns [bsz, action_dim], openvla_oft returns [bsz, seq_len], and
# the StarVLA action heads return [bsz, num_action_chunks, action_dim].
# loss_mask is [bsz, 1] under reward_type: chunk_level and
# [bsz, num_action_chunks] otherwise.


def _entropy(*shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=generator) + 0.5


@pytest.mark.parametrize("batch_size", [4, 16, 64, 500])
def test_chunk_level_entropy_does_not_scale_with_batch_size(batch_size):
    """The bug signature: entropy_loss came out multiplied by the micro-batch size."""
    entropy = _entropy(batch_size, 1)
    loss_mask = torch.ones(batch_size, 1, dtype=torch.bool)

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy.mean()), rel=1e-6)


def test_chunk_level_entropy_averages_only_the_valid_rows():
    entropy = _entropy(16, 1)
    loss_mask = torch.zeros(16, 1, dtype=torch.bool)
    loss_mask[:6] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:6].mean()), rel=1e-6)


def test_chunk_level_sums_a_wide_entropy_before_averaging():
    # cnn_policy shape: one entropy per action dimension.
    entropy = _entropy(12, 4)
    loss_mask = torch.zeros(12, 1, dtype=torch.bool)
    loss_mask[:5] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:5].sum(dim=-1).mean()), rel=1e-6)


def test_token_level_averages_over_every_valid_element():
    # openvla_oft shape: entropy per token, mask per chunk step.
    entropy = _entropy(10, 7)
    loss_mask = torch.zeros(10, 1, dtype=torch.bool)
    loss_mask[:4] = True

    got = compute_entropy_loss(entropy, "token_level", loss_mask)

    assert float(got) == pytest.approx(float(entropy[:4].mean()), rel=1e-6)


def test_action_level_sums_action_dim_then_averages():
    entropy = _entropy(6, 3 * 7)
    loss_mask = torch.zeros(6, 1, dtype=torch.bool)
    loss_mask[:2] = True

    got = compute_entropy_loss(
        entropy, "action_level", loss_mask, action_dim=7, batch_size=6
    )

    per_chunk = entropy.reshape(6, 3, 7).sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:2].mean()), rel=1e-6)


def test_a_wider_mask_than_entropy_still_weights_by_valid_steps():
    # lingbotvla: entropy is [bsz, 1] while reward_type != chunk_level keeps the
    # mask at [bsz, num_action_chunks]. Each sample is weighted by its valid steps.
    entropy = _entropy(5, 1)
    loss_mask = torch.zeros(5, 4, dtype=torch.bool)
    loss_mask[0, :4] = True
    loss_mask[1, :1] = True

    got = compute_entropy_loss(entropy, "token_level", loss_mask)

    expected = (entropy[0, 0] * 4 + entropy[1, 0] * 1) / 5
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_three_dim_entropy_reduces_to_the_mask_rank():
    # StarVLA action heads return [bsz, num_action_chunks, action_dim]; the
    # chunk_level sum already lands on the mask's rank, so nothing is unsqueezed.
    entropy = _entropy(4, 8, 7)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    loss_mask[:, :3] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    per_chunk = entropy.sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:, :3].mean()), rel=1e-6)


def test_three_dim_entropy_is_right_when_batch_equals_num_chunks():
    # Same shape family with bsz == num_action_chunks, where a rank mismatch
    # broadcasts successfully instead of raising and would go unnoticed.
    entropy = _entropy(8, 8, 7)
    loss_mask = torch.zeros(8, 8, dtype=torch.bool)
    loss_mask[:3] = True

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    per_chunk = entropy.sum(dim=-1)
    assert float(got) == pytest.approx(float(per_chunk[:3].mean()), rel=1e-6)


def test_no_mask_averages_everything():
    entropy = _entropy(9, 1)

    got = compute_entropy_loss(entropy, "chunk_level", None)

    assert float(got) == pytest.approx(float(entropy.mean()), rel=1e-6)


def test_a_fully_masked_batch_contributes_zero():
    entropy = _entropy(8, 1)
    loss_mask = torch.zeros(8, 1, dtype=torch.bool)

    got = compute_entropy_loss(entropy, "chunk_level", loss_mask)

    assert float(got) == pytest.approx(0.0)


def test_entropy_loss_keeps_the_gradient_path():
    entropy = _entropy(8, 1).requires_grad_(True)
    loss_mask = torch.ones(8, 1, dtype=torch.bool)

    compute_entropy_loss(entropy, "chunk_level", loss_mask).backward()

    # A correct mean spreads 1/8 of the gradient onto each row; the outer-product
    # bug put 1.0 on each instead.
    assert torch.allclose(entropy.grad, torch.full((8, 1), 1 / 8))


def test_entropy_loss_rejects_a_model_that_computes_no_entropy():
    # gr00t, abot_m0 and evo1 return entropy=None; pairing one with a non-zero
    # entropy_bonus is a config error, not a zero bonus.
    with pytest.raises(ValueError, match="algorithm.entropy_bonus"):
        compute_entropy_loss(None, "chunk_level", torch.ones(4, 1, dtype=torch.bool))


def _load_checkpoint_utils():
    module_path = (
        Path(__file__).resolve().parents[2] / "rlinf" / "utils" / "checkpoint.py"
    )
    assert module_path.exists(), "checkpoint path utilities are not implemented"
    spec = importlib.util.spec_from_file_location(
        "_rlinf_utils_checkpoint_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "/tmp/checkpoints/global_step_30",
        "/tmp/checkpoints/global_step_30/",
        "/tmp/checkpoints/global_step_30///",
    ],
)
def test_parse_global_step_accepts_trailing_slashes(checkpoint_path):
    checkpoint_utils = _load_checkpoint_utils()

    assert (
        checkpoint_utils.parse_global_step_from_checkpoint_path(checkpoint_path) == 30
    )


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "/tmp/checkpoints/step_30",
        "/tmp/checkpoints/global_step_latest/",
        "/tmp/checkpoints/global_step_30/actor",
    ],
)
def test_parse_global_step_rejects_invalid_checkpoint_directories(checkpoint_path):
    checkpoint_utils = _load_checkpoint_utils()

    with pytest.raises(ValueError, match="global_step_<step>"):
        checkpoint_utils.parse_global_step_from_checkpoint_path(checkpoint_path)


class _StubRunner:
    """Expose only the checkpoint helpers and state used by these tests."""

    def __init__(self, critic=None):
        self.critic = critic

    _is_complete_checkpoint = ReasoningRunner._is_complete_checkpoint


class _ImmediateHandle:
    def wait(self):
        return None


class _Actor:
    def save_checkpoint(self, path: str, _step: int):
        os.makedirs(path, exist_ok=True)
        return _ImmediateHandle()


class _Dataloader:
    def state_dict(self):
        return {"offset": 3}


def _write_checkpoint(
    root: Path, step: int, *, complete: bool, with_critic: bool = False
) -> Path:
    checkpoint_dir = root / f"global_step_{step}"
    (checkpoint_dir / "actor").mkdir(parents=True)
    if with_critic:
        (checkpoint_dir / "critic").mkdir()
    if complete:
        data_dir = checkpoint_dir / "data"
        data_dir.mkdir()
        (data_dir / "data.pt").write_bytes(b"dataloader-state")
    return checkpoint_dir


def _resolve_auto_resume(log_path: Path, *, critic=None) -> str | None:
    cfg = OmegaConf.create(
        {"runner": {"resume_dir": "auto", "logger": {"log_path": str(log_path)}}}
    )
    runner = _StubRunner(critic=critic)
    runner.cfg = cfg
    runner.init_rollout_workers = lambda: None
    runner.init_actor_critic_workers = lambda: None

    ReasoningRunner.init_workers(runner)
    return cfg.runner.resume_dir


def _saving_runner(tmp_path: Path) -> _StubRunner:
    runner = _StubRunner()
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "output_dir": str(tmp_path),
                "experiment_name": "experiment",
            }
        }
    )
    runner.global_steps = 8
    runner.actor = _Actor()
    runner.train_dataloader = _Dataloader()
    return runner


@pytest.mark.parametrize(
    "completeness,expected_step",
    [
        pytest.param({40: True, 80: False}, 40, id="skips-the-incomplete-newest"),
        pytest.param({40: True, 80: True}, 80, id="takes-the-newest-complete"),
        pytest.param({40: False}, None, id="starts-fresh-when-none-is-complete"),
    ],
)
def test_auto_resume_selects_the_newest_complete_checkpoint(
    tmp_path, completeness, expected_step
):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    for step, complete in completeness.items():
        _write_checkpoint(checkpoints_dir, step, complete=complete)

    expected = (
        None
        if expected_step is None
        else str(checkpoints_dir / f"global_step_{expected_step}")
    )
    assert _resolve_auto_resume(tmp_path) == expected


def test_checkpoint_requires_the_critic_only_when_configured(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    checkpoint = _write_checkpoint(checkpoints_dir, 40, complete=True)

    assert _StubRunner()._is_complete_checkpoint(str(checkpoint))
    assert not _StubRunner(critic=object())._is_complete_checkpoint(str(checkpoint))


def test_dataloader_state_is_published_atomically(tmp_path, monkeypatch):
    runner = _saving_runner(tmp_path)
    written_paths = []

    def save(_state, path):
        written_paths.append(path)
        Path(path).write_bytes(b"complete")

    monkeypatch.setattr("rlinf.runners.reasoning_runner.torch.save", save)

    ReasoningRunner._save_checkpoint(runner)

    checkpoint = tmp_path / "experiment" / "checkpoints" / "global_step_8"
    final_path = checkpoint / "data" / "data.pt"
    assert written_paths == [f"{final_path}.tmp"]
    assert final_path.read_bytes() == b"complete"
    assert not Path(f"{final_path}.tmp").exists()
    assert runner._is_complete_checkpoint(str(checkpoint))


def test_interrupted_dataloader_save_does_not_publish_completion(tmp_path, monkeypatch):
    runner = _saving_runner(tmp_path)

    def interrupted_save(_state, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr("rlinf.runners.reasoning_runner.torch.save", interrupted_save)

    with pytest.raises(RuntimeError, match="interrupted"):
        ReasoningRunner._save_checkpoint(runner)

    checkpoint = tmp_path / "experiment" / "checkpoints" / "global_step_8"
    final_path = checkpoint / "data" / "data.pt"
    assert not final_path.exists()
    assert not Path(f"{final_path}.tmp").exists()
    assert not runner._is_complete_checkpoint(str(checkpoint))


_REDIRECTED_ENTRYPOINT = textwrap.dedent(
    """
    import hydra

    from rlinf.scheduler import Cluster
    from rlinf.utils.utils import output_redirector


    @hydra.main(version_base="1.1", config_path=None)
    @output_redirector
    def main(cfg):
        Cluster(num_nodes=1)
        print("entrypoint ran")
        if cfg.outcome == "raise":
            raise RuntimeError("entrypoint failed")


    main()
    """
)


def _run_redirected_entrypoint(tmp_path, outcome):
    script = tmp_path / "entrypoint.py"
    script.write_text(_REDIRECTED_ENTRYPOINT)
    return subprocess.run(
        [
            sys.executable,
            str(script),
            f"+outcome={outcome}",
            f"+runner.output_dir={tmp_path}",
            "+runner.experiment_name=exp",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_redirected_entrypoint_exits_zero_and_keeps_its_log(tmp_path):
    result = _run_redirected_entrypoint(tmp_path, "return")

    assert result.returncode == 0, result.stderr
    assert "entrypoint ran" in (tmp_path / "exp" / "log" / "main.log").read_text()


def test_redirected_entrypoint_failure_is_not_reported_as_success(tmp_path):
    # Hydra catches the exception and calls sys.exit(1) itself, so the failure
    # never reaches sys.excepthook.
    result = _run_redirected_entrypoint(tmp_path, "raise")

    assert result.returncode == 1, result.stderr
    assert "entrypoint failed" in (tmp_path / "exp" / "log" / "main.log").read_text()
