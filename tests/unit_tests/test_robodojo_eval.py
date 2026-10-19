# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-layout plans and native-result auditing, with no policy/GPU execution."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from toolkits.rlt import robodojo_eval as evaluation
from toolkits.rlt import robodojo_resume as recovery
from toolkits.rlt import robodojo_runner as runner


@pytest.fixture
def config(tmp_path, monkeypatch):
    value = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / "evaluations/robodojo/arx_x5.yaml"
        ).read_text()
    )
    value.update(
        robodojo_root=str(tmp_path / "RoboDojo"),
        sim_python=sys.executable,
        policy_python=sys.executable,
        episodes=2,
    )
    value["policy"].update(
        backend="openpi_reference", checkpoint_path=str(tmp_path / "checkpoint/1000")
    )
    root = Path(value["robodojo_root"])
    files = {
        "task/RoboDojo/config/_task.yml": "common: {robot_config: dual_x5, eval_nums: 50}\ntasks: {}\n",
        "task/RoboDojo/config/stack_bowls.yml": "{}",
        "Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_10.json": "{}",
        "Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_2.json": "{}",
        "Assets/Robots/asset.txt": "fixture",
        "Assets/Object/RoboDojo/asset.txt": "fixture",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (root / "XPolicyLab/policy").mkdir(parents=True)
    checkpoint = Path(value["policy"]["checkpoint_path"])
    normalization = checkpoint / "assets/arx_x5_sim/norm_stats.json"
    normalization.parent.mkdir(parents=True)
    normalization.write_text(
        json.dumps(
            {
                "norm_stats": {
                    name: {
                        "mean": [0.0] * 14,
                        "std": [1.0] * 14,
                        "q01": [-1.0] * 14,
                        "q99": [1.0] * 14,
                    }
                    for name in ("state", "actions")
                }
            }
        )
    )
    (checkpoint / "params").mkdir()
    # Git is the external source dependency; policy/simulator APIs are never mocked
    # in preparation because they must not be imported or called at all.
    monkeypatch.setattr(evaluation, "_source_check", lambda *args: None)
    return value


def write_result(plan, *, outcomes=((0, True, 1.0), (1, False, 0.5))):
    evaluation.install_policy(plan["config"])
    path = Path(plan["expected_result"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "details": {
            str(i): {"layout_id": layout, "success": success, "score": score}
            for i, (layout, success, score) in enumerate(outcomes)
        },
        "eval_time": len(outcomes),
        "success_rate": sum(success for _, success, _ in outcomes) / len(outcomes),
        "score": 100 * sum(score for _, _, score in outcomes) / len(outcomes),
    }
    path.write_text(json.dumps(result))
    return path


@pytest.fixture
def interrupted_run(config, tmp_path):
    config["episodes"] = 3
    config["gpu"] = 2
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        config["port"] = reservation.getsockname()[1]
    root = Path(config["robodojo_root"])
    (root / "Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_20.json").write_text("{}")
    plan = evaluation.prepare(config)
    result_path = write_result(plan)
    result = evaluation.read_json(result_path)
    manifest = {
        "run_id": config["run_id"],
        "save_dir": str(result_path.parent.relative_to(root)),
        "task_name": config["task"],
        "policy_name": config["policy"]["policy_name"],
        "config_name": config["policy"]["env_cfg_type"],
        "eval_seed": config["seed"],
        "additional_info": config["group"],
        "success_nums": 1,
        "fail_nums": 1,
        "unstable_nums": 0,
        "total_score": 1.5,
        "completed_layout_ids": [0, 1],
        "abandoned_layout_ids": [],
        "details": result["details"],
        "restart_count": 0,
    }
    manifest_path = result_path.parent.parent / f"_resume_{config['run_id']}.json"
    manifest_path.write_text(json.dumps(manifest))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    return plan_path, result_path, manifest_path


def save_receipt(plan_path):
    path = plan_path.with_name("recovery.json")
    path.write_text(json.dumps(recovery.prepare_resume(plan_path)))
    return path


def test_resume_keeps_completed_failures_and_leaves_artifacts_untouched(
    interrupted_run,
):
    originals = {path: path.read_bytes() for path in interrupted_run}
    receipt_path = save_receipt(interrupted_run[0])
    report = recovery.verify_resume(receipt_path)
    assert report["completed_layout_ids"] == [0, 1]
    assert report["remaining_layout_ids"] == [2]
    assert report["success_rate"] is None and not report["gpu_started"]
    assert originals == {path: path.read_bytes() for path in interrupted_run}
    partial = recovery.audit_continuation(receipt_path)
    assert partial["status"] == "incomplete" and partial["success_rate"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "another_run"),
        ("eval_seed", True),
        ("save_dir", "/tmp/unrelated-result"),
        ("success_nums", 2),
        ("fail_nums", True),
        ("unstable_nums", 1),
        ("total_score", float("nan")),
        ("restart_count", -1),
        ("completed_layout_ids", [0, 0]),
        ("completed_layout_ids", [0, True]),
        ("abandoned_layout_ids", [2]),
        ("details", {}),
    ],
)
def test_resume_rejects_inconsistent_native_manifest(interrupted_run, field, value):
    plan_path, _, manifest_path = interrupted_run
    data = evaluation.read_json(manifest_path)
    data[field] = value
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        recovery.prepare_resume(plan_path)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_resume_rechecks_all_original_bytes_after_waiting(interrupted_run, index):
    receipt_path = save_receipt(interrupted_run[0])
    path = interrupted_run[index]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Changed since"):
        recovery.verify_resume(receipt_path)


def test_resume_rechecks_layout_hashes(interrupted_run):
    receipt_path = save_receipt(interrupted_run[0])
    plan = evaluation.read_json(interrupted_run[0])
    Path(plan["layouts"][2]["path"]).write_text('{"changed": true}')
    with pytest.raises(ValueError, match="partial evaluation"):
        recovery.verify_resume(receipt_path)


def test_resume_rejects_modified_commands(interrupted_run):
    plan_path, _, _ = interrupted_run
    plan = evaluation.read_json(plan_path)
    plan["commands"]["env"]["ROBODOJO_RUN_ID"] = "different_run"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="commands"):
        recovery.prepare_resume(plan_path)


@pytest.mark.parametrize("replace_failure", [False, True])
def test_continuation_requires_complete_set_and_preserves_old_outcomes(
    interrupted_run, replace_failure
):
    plan_path, result_path, manifest_path = interrupted_run
    receipt_path = save_receipt(plan_path)
    result = evaluation.read_json(result_path)
    result["details"]["2"] = {"layout_id": 2, "success": True, "score": 1.0}
    if replace_failure:
        result["details"]["1"].update(success=True, score=1.0)
    rows = result["details"].values()
    result.update(
        eval_time=3,
        success_rate=sum(row["success"] for row in rows) / 3,
        score=100 * sum(row["score"] for row in rows) / 3,
    )
    result_path.write_text(json.dumps(result))
    manifest_path.unlink()  # Native evaluator removes it after normal completion.
    report = recovery.audit_continuation(receipt_path)
    assert report["preserved_episodes"] == 2
    if replace_failure:
        assert report["status"] == "failed" and report["success_rate"] is None
        assert "outcome changed" in report["errors"][0]
    else:
        assert report["status"] == "passed" and report["success_rate"] == 2 / 3


def test_receipt_cannot_relabel_remaining_layouts(interrupted_run):
    path = save_receipt(interrupted_run[0])
    data = evaluation.read_json(path)
    data["remaining_layout_ids"] = [1, 2]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="partition"):
        recovery.verify_resume(path)


def test_runner_refuses_implicit_resume(interrupted_run):
    with pytest.raises(ValueError, match="explicit recovery"):
        runner.validate_launch(interrupted_run[0])
    runner.validate_launch(interrupted_run[0], save_receipt(interrupted_run[0]))


def test_runner_requires_completed_outcomes_not_just_exit_zero(interrupted_run):
    plan = evaluation.read_json(interrupted_run[0])
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.verify_result(plan)
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.verify_result(plan, save_receipt(interrupted_run[0]))


def test_runner_lease_cannot_be_shared_and_is_released_on_close(tmp_path):
    path = tmp_path / "gpu.lock"
    with path.open("a") as first, path.open("a") as second:
        assert runner.try_acquire_lease(first)
        assert not runner.try_acquire_lease(second)
        first.close()
        assert runner.try_acquire_lease(second)


def test_runner_listener_requires_process_ownership():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert runner.owns_listener(os.getpid(), listener.getsockname()[1])
        assert not runner.owns_listener(os.getpid(), 0)


def test_renderer_mapping_uses_unique_kit_uuid():
    table = (
        "| 0   | NVIDIA L40 |\n|     | 26b5 | aaaa0000.. |\n"
        "| 2   | NVIDIA L40 |\n|     | 26b5 | cccc0000.. |\n"
    )
    assert runner.renderer_index(table, "GPU-cccc0000-dddd") == 2
    with pytest.raises(ValueError, match="missing or ambiguous"):
        runner.renderer_index(table, "GPU-bbbb0000-dddd")


@pytest.mark.parametrize("stop_mode", ["STOP", "SIGTERM"])
def test_busy_runner_waits_without_launching_children_and_can_stop(
    interrupted_run, tmp_path, stop_mode
):
    # Use a real lock and a separate CLI process; no simulator or GPU is invoked.
    lock_path = tmp_path / "rlt-research-gpu2.lock"
    receipt_path = save_receipt(interrupted_run[0])
    output = tmp_path / "run"
    # The subprocess checks real native Git revisions; this fixture's fake Git
    # boundary is injected only for this integration test of process ownership.
    program = (
        "import runpy; from toolkits.rlt import robodojo_eval; "
        "robodojo_eval._source_check = lambda *a: None; "
        "runpy.run_module('toolkits.rlt.robodojo_runner', run_name='__main__')"
    )
    with lock_path.open("a") as lease:
        assert runner.try_acquire_lease(lease)
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                program,
                "--plan",
                str(interrupted_run[0]),
                "--resume-receipt",
                str(receipt_path),
                "--output",
                str(output),
                "--wait-idle",
                "--lease-directory",
                str(tmp_path),
                "--vulkaninfo",
                "/usr/bin/true",
                "--renderer-table",
                str(interrupted_run[0]),
            ],
            env=dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", CUDA_VISIBLE_DEVICES=""),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 15
            while not (output / "status.json").exists():
                if process.poll() is not None:
                    pytest.fail(process.communicate()[1])
                if time.monotonic() > deadline:
                    pytest.fail("Runner failed to record waiting state")
                time.sleep(0.05)
            status = evaluation.read_json(output / "status.json")
            assert status["state"] == "waiting_lease"
            assert status["wait_deadline"] is None
            assert "server_pid" not in status and "client_pid" not in status
            assert not (output / "server.log").exists()
            if stop_mode == "STOP":
                (output / "STOP").touch()
            else:
                process.terminate()
            process.communicate(timeout=10)
            assert evaluation.read_json(output / "status.json")["state"] == "stopped"
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


def test_preparation_matches_native_numeric_layout_order_without_gpu(config):
    plan = evaluation.prepare(config)
    assert plan["status"] == "prepared" and not plan["runtime_verified"]
    assert [Path(row["path"]).name for row in plan["layouts"]] == [
        "stack_bowls_2.json",
        "stack_bowls_10.json",
    ]
    assert [row["layout_id"] for row in plan["layouts"]] == [0, 1]
    command = plan["commands"]["client_argv"]
    assert "--headless" in command and "--enable_cameras" in command
    assert command[command.index("--device_id") + 1] == "0"
    assert plan["commands"]["env"]["EVAL_NUM"] == "2"
    assert not Path(plan["expected_result"]).parent.exists()
    assert not (Path(config["robodojo_root"]) / "XPolicyLab/policy/RLT_Pi05").exists()


def test_missing_runtime_and_backend_produce_blocked_report(config):
    config["sim_python"] = "/missing/isaac-python"
    config["policy"]["backend"] = "unconfigured"
    report = evaluation.prepare(config)
    assert report["status"] == "blocked"
    assert len(report["errors"]) == 2


def test_nonzero_gpu_separates_vulkan_and_masked_cuda_devices(config):
    config["gpu"] = 2
    commands = evaluation.prepare(config)["commands"]
    assert commands["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert commands["env"]["LD_LIBRARY_PATH"] == ""
    assert commands["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    client = commands["client_argv"]
    assert client[client.index("--device_id") + 1] == "0"
    flags = client[client.index("--kit_args") + 1].split()
    assert "--/renderer/activeGpu=2" in flags
    assert "--/renderer/multiGpu/enabled=false" in flags
    assert "--/physics/cudaDevice=0" in flags


def test_runtime_inventory_never_claims_simulator_verified(config):
    # This uses the real test Python, which need not contain Isaac or OpenPI.
    result = evaluation.runtime_inventory(config)
    assert not result["runtime_verified"]
    assert set(result["interpreters"]) == {"sim_python", "policy_python"}
    assert "isaacsim" in result["interpreters"]["sim_python"]["packages"]


def test_runtime_inventory_missing_interpreter(config):
    config["sim_python"] = "/missing/isaac-python"
    result = evaluation.runtime_inventory(config)
    assert result["status"] == "blocked"
    assert any("sim_python" in error for error in result["errors"])


def test_reference_probe_missing_interpreter_leaves_runtime_unverified(config):
    config["policy_python"] = "/missing/openpi-python"
    result = evaluation.probe_reference(config)
    assert result["status"] == "blocked" and result["errors"]
    assert not result["runtime_verified"] and not result["gpu_started"]
    assert not result["parameters_loaded"]


def test_reference_probe_does_not_import_arbitrary_factory(config):
    config["policy"].update(backend="factory", backend_factory="unknown:load")
    result = evaluation.probe_reference(config)
    assert result["status"] == "blocked"
    assert "openpi_reference" in result["errors"][0]


@pytest.mark.parametrize("change", [None, "libero", "horizon", "normalization"])
def test_native_reference_imports_and_config_without_weights(config, change):
    interpreter = os.environ.get("RP_ROBODOJO_POLICY_PYTHON")
    if not interpreter:
        pytest.skip(
            "Set RP_ROBODOJO_POLICY_PYTHON to the isolated native OpenPI Python"
        )
    config["policy_python"] = interpreter
    if change == "libero":
        config["policy"]["train_config_name"] = "pi05_libero"
    elif change == "horizon":
        config["policy"]["action_horizon"] = 1000
    elif change == "normalization":
        config["policy"]["model_contract"]["normalization_id"] = "wrong_robot"
    report = evaluation.probe_reference(config)
    if change:
        assert report["status"] == "blocked" and report["errors"]
    else:
        assert report["status"] == "reference_imports_passed", report["errors"]
        assert report["reference"]["padded_action_dim"] == 32
        assert report["reference"]["robot_action_dim"] == 14
    assert not report["parameters_loaded"] and not report["runtime_verified"]
    assert not report["gpu_started"]


def test_seven_dimensional_norm_stats_block_mislabeled_checkpoint(config):
    path = (
        Path(config["policy"]["checkpoint_path"]) / "assets/arx_x5_sim/norm_stats.json"
    )
    data = json.loads(path.read_text())
    data["norm_stats"]["actions"]["mean"] = [0.0] * 7
    path.write_text(json.dumps(data))
    report = evaluation.prepare(config)
    assert report["status"] == "blocked"
    assert "14-D" in report["errors"][0]


def test_partial_artifact_download_blocks_preparation(config, tmp_path):
    path = tmp_path / "download-report.json"
    config["artifact_download_report"] = str(path)
    path.write_text(json.dumps({"complete": False, "errors": []}))
    assert evaluation.prepare(config)["status"] == "blocked"
    path.write_text(json.dumps({"complete": True, "errors": []}))
    assert evaluation.prepare(config)["status"] == "prepared"


@pytest.mark.parametrize("change", ["budget", "embodiment", "duplicate_order"])
def test_native_contract_mismatch_blocks_preparation(config, change):
    root = Path(config["robodojo_root"])
    if change == "budget":
        config["episodes"] = 100
    elif change == "embodiment":
        (root / "task/RoboDojo/config/_task.yml").write_text(
            "common: {robot_config: another_robot, eval_nums: 50}\n"
        )
    else:
        (root / "Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_02.json").write_text(
            "{}"
        )
    assert evaluation.prepare(config)["status"] == "blocked"


def test_reference_cannot_be_labelled_q_bc(config):
    config["group"] = "q_bc"
    with pytest.raises(ValueError, match="reference"):
        evaluation.prepare(config)


def test_plugin_installation_is_explicit_exclusive_and_configured(config):
    target = evaluation.install_policy(config)
    deploy = yaml.safe_load((target / "deploy.yml").read_text())
    assert deploy["eval_batch"] is True and deploy["host"] == "127.0.0.1"
    assert deploy["checkpoint_path"] == config["policy"]["checkpoint_path"]
    assert "robodojo_policy import Model" in (target / "model.py").read_text()
    with pytest.raises(FileExistsError):
        evaluation.install_policy(config)


def test_audit_distinguishes_full_success_from_partial_score(config):
    plan = evaluation.prepare(config)
    report = evaluation.audit(plan, write_result(plan))
    assert report["status"] == "passed"
    assert report["success_rate"] == 0.5
    assert report["partial_score_percent"] == 75.0
    assert evaluation.prepare(config)["status"] == "blocked"  # No implicit resume.


def test_missing_or_partial_results_are_not_zero_success(config):
    plan = evaluation.prepare(config)
    missing = evaluation.audit(plan, Path(plan["expected_result"]))
    assert missing["status"] == "incomplete" and missing["success_rate"] is None
    partial = evaluation.audit(plan, write_result(plan, outcomes=((0, False, 0.0),)))
    assert partial["status"] == "incomplete" and partial["success_rate"] is None


@pytest.mark.parametrize(
    "outcomes", [((0, True, 1), (0, True, 1)), ((0, True, 1), (2, True, 1))]
)
def test_duplicate_or_replacement_layouts_fail_paired_evaluation(config, outcomes):
    plan = evaluation.prepare(config)
    report = evaluation.audit(plan, write_result(plan, outcomes=outcomes))
    assert report["status"] == "failed" and report["success_rate"] is None


@pytest.mark.parametrize(
    "field,value", [("success_rate", 1.0), ("score", float("nan")), ("eval_time", True)]
)
def test_inconsistent_native_aggregates_fail(config, field, value):
    plan = evaluation.prepare(config)
    path = write_result(plan)
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    assert evaluation.audit(plan, path)["status"] == "failed"


def test_changed_layouts_and_normalization_fail(config):
    plan = evaluation.prepare(config)
    path = write_result(plan)
    Path(plan["layouts"][0]["path"]).write_text('{"changed":true}')
    Path(plan["normalization"]["path"]).write_text('{"changed":true}')
    report = evaluation.audit(plan, path)
    assert report["status"] == "failed" and len(report["errors"]) == 2


def test_wrong_run_identity_and_duplicate_json_keys_are_rejected(config, tmp_path):
    plan = evaluation.prepare(config)
    with pytest.raises(ValueError, match="identity"):
        evaluation.audit(plan, tmp_path / "unrelated.json")
    path = tmp_path / "duplicates.json"
    path.write_text('{"details":{},"details":{}}')
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.read_json(path)


def test_source_revision_check_is_fail_closed(tmp_path):
    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / "evaluations/robodojo/arx_x5.yaml"
        ).read_text()
    )
    config["robodojo_root"] = str(tmp_path)
    report = evaluation.prepare(config)
    assert report["status"] == "blocked"
    assert sum(error.startswith("Source check:") for error in report["errors"]) == 2


def test_inserting_earlier_layout_or_changing_plugin_fails_audit(config):
    plan = evaluation.prepare(config)
    path = write_result(plan)
    root = Path(config["robodojo_root"])
    (root / "Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_0.json").write_text("{}")
    (root / "XPolicyLab/policy/RLT_Pi05/deploy.yml").write_text("changed: true")
    report = evaluation.audit(plan, path)
    assert report["status"] == "failed"
    assert any("ordering changed" in error for error in report["errors"])
    assert any("Installed policy differs" in error for error in report["errors"])


@pytest.mark.parametrize(
    "row",
    [
        None,
        {"layout_id": [], "success": True, "score": 1},
        {"layout_id": 0, "success": 1, "score": 1},
        {"layout_id": 0, "success": False, "score": float("nan")},
    ],
)
def test_malformed_native_episode_is_a_failed_audit(config, row):
    plan = evaluation.prepare(config)
    path = write_result(plan)
    data = json.loads(path.read_text())
    data["details"]["0"] = row
    path.write_text(json.dumps(data))
    assert evaluation.audit(plan, path)["status"] == "failed"
