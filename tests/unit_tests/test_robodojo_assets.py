# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Pinned artifact selection, download integrity, and CPU-only provisioning."""

import fcntl
import hashlib
from pathlib import Path

import pytest

from toolkits.rlt import robodojo_assets as assets
from toolkits.rlt import robodojo_task_assets as task_assets


def record(path="Assets/Robots/x5/model.usd", content=b"robot"):
    return {
        "path": path,
        "size": len(content),
        "hash_kind": "sha256",
        "digest": hashlib.sha256(content).hexdigest(),
    }


def manifest():
    paths = {
        prefix + "/fixture" for prefix in assets.PREFIXES if prefix != assets.LAYOUTS
    }
    paths.update(assets.EXACT_FILES)
    paths.update(f"{assets.LAYOUTS}/stack_bowls_{i}.json" for i in range(25))
    paths.update(
        (
            f"{assets.CHECKPOINT}/params/_METADATA",
            f"{assets.CHECKPOINT}/assets/arx_x5_sim/norm_stats.json",
        )
    )
    return {
        "schema": "robodojo-assets-v1",
        "repo": assets.REPO,
        "revision": assets.REVISION,
        "task": "stack_bowls",
        "seed": 0,
        "backend": "openpi_reference",
        "checkpoint": assets.CHECKPOINT,
        "files": [record(path) for path in sorted(paths)],
    }


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/Assets/Robots/x5/robot.usd",
        "Assets/Robots/../x5/robot.usd",
        "Assets//Robots/x5/robot.usd",
        "Assets/Robots/x5/./robot.usd",
    ],
)
def test_reject_unsafe_paths(path):
    with pytest.raises(ValueError, match="Unsafe"):
        assets.selected(path)


def test_reference_selection_excludes_optimizer_and_other_tasks():
    assert assets.selected(f"{assets.CHECKPOINT}/params/ocdbt.process_0/d/data")
    assert not assets.selected(f"{assets.CHECKPOINT}/train_state/d/data")
    assert not assets.selected(f"{assets.LAYOUTS}/put_apple_0.json")
    assert not assets.selected("Assets/Robots/franka/model.usd")


def test_budget_identity_and_layout_count():
    value = manifest()
    assert assets.validate_manifest(value, max_bytes=1000) > 0
    with pytest.raises(ValueError, match="budget"):
        assets.validate_manifest(value, max_bytes=1)
    value["revision"] = "main"
    with pytest.raises(ValueError, match="identity"):
        assets.validate_manifest(value, max_bytes=1000)
    value = manifest()
    value["files"] = [
        f for f in value["files"] if not f["path"].endswith("stack_bowls_0.json")
    ]
    with pytest.raises(ValueError, match="25"):
        assets.validate_manifest(value, max_bytes=1000)


def test_duplicate_manifest_file_fails():
    value = manifest()
    value["files"].append(value["files"][0])
    with pytest.raises(ValueError, match="duplicate"):
        assets.validate_manifest(value, max_bytes=1000)


def test_inventory_applies_numeric_task_cap_and_excludes_training(monkeypatch):
    records = manifest()["files"]
    records += [record(f"{assets.LAYOUTS}/stack_bowls_{i}.json") for i in range(25, 50)]
    records += [record(f"{assets.CHECKPOINT}/train_state/optimizer")]
    rows = [
        {
            "type": "file",
            "path": r["path"],
            "size": r["size"],
            "lfs": {"oid": r["digest"]},
        }
        for r in records
    ]
    monkeypatch.setattr(assets, "list_tree", lambda session, prefix: rows)
    result = assets.build_manifest(None, max_bytes=1000)
    paths = {r["path"] for r in result["files"]}
    assert f"{assets.LAYOUTS}/stack_bowls_24.json" in paths
    assert f"{assets.LAYOUTS}/stack_bowls_25.json" not in paths
    assert not any("train_state" in path for path in paths)


class Response:
    def __init__(
        self, content=b"robot", status=200, headers=None, rows=None, next_url=None
    ):
        self.content, self.status_code = content, status
        self.headers = headers or {}
        self.rows, self.links = rows, {"next": {"url": next_url}} if next_url else {}
        self.url = "https://huggingface.co/file"

    def raise_for_status(self):
        pass

    def json(self):
        return self.rows

    def iter_content(self, _):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return next(self.responses)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def test_tree_follows_pinned_pagination():
    url = f"https://huggingface.co/api/datasets/{assets.REPO}/tree/{assets.REVISION}/Assets?cursor=2"
    session = Session(
        Response(rows=[{"path": "a"}], next_url=url), Response(rows=[{"path": "b"}])
    )
    assert assets.list_tree(session, "Assets") == [{"path": "a"}, {"path": "b"}]
    assert len(session.requests) == 2


def test_tree_rejects_external_pagination():
    with pytest.raises(ValueError, match="pagination"):
        assets.list_tree(
            Session(Response(rows=[], next_url="https://example.com")), "Assets"
        )


def test_resume_and_verified_publish(tmp_path, monkeypatch):
    item = record()
    target = tmp_path / item["path"]
    target.parent.mkdir(parents=True)
    partial = target.with_name(target.name + ".robodojo-part")
    partial.write_bytes(b"ro")
    session = Session(Response(b"bot", 206, {"Content-Range": "bytes 2-4/5"}))
    monkeypatch.setattr(assets.requests, "Session", lambda: session)
    result = assets.download_file(tmp_path, item)
    assert result["sha256"] == item["digest"]
    assert target.read_bytes() == b"robot"
    assert not partial.exists()
    assert session.requests[0][1]["headers"]["Range"] == "bytes=2-"
    # Existing verified files are reused without another network request.
    assert assets.download_file(tmp_path, item) == result


def test_corruption_does_not_publish(tmp_path, monkeypatch):
    item = record()
    monkeypatch.setattr(assets.requests, "Session", lambda: Session(Response(b"wrong")))
    with pytest.raises(ValueError, match="Hash mismatch"):
        assets.download_file(tmp_path, item)
    assert not (tmp_path / item["path"]).exists()


def test_server_ignoring_range_restarts_partial(tmp_path, monkeypatch):
    item = record()
    target = tmp_path / item["path"]
    target.parent.mkdir(parents=True)
    target.with_name(target.name + ".robodojo-part").write_bytes(b"ro")
    monkeypatch.setattr(assets.requests, "Session", lambda: Session(Response()))
    assets.download_file(tmp_path, item)
    assert target.read_bytes() == b"robot"


def test_rejects_escape_symlink(tmp_path):
    (tmp_path / "Assets").symlink_to(Path("/tmp"), target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        assets.download_file(tmp_path, record())


def test_git_blob_verification(tmp_path):
    target = tmp_path / "metadata"
    target.write_bytes(b"robot")
    item = record()
    item.update(
        hash_kind="git_blob_sha1", digest=hashlib.sha1(b"blob 5\0robot").hexdigest()
    )
    assert assets.verify_file(target, item) == hashlib.sha256(b"robot").hexdigest()


def test_download_records_failures_without_claiming_complete(tmp_path, monkeypatch):
    value = manifest()

    def fetch(root, item):
        if item["path"].endswith("/params/_METADATA"):
            raise ValueError("fixture checksum failure")
        return dict(item, sha256=item["digest"])

    monkeypatch.setattr(assets, "download_file", fetch)
    report = assets.download(value, tmp_path, max_bytes=1000, workers=2)
    assert not report["complete"]
    assert len(report["errors"]) == 1
    assert not (tmp_path / ".robodojo-download.lock").exists()


def test_concurrent_download_refuses_existing_lock(tmp_path):
    (tmp_path / ".robodojo-download.lock").write_text("123")
    with pytest.raises(FileExistsError):
        assets.download(manifest(), tmp_path, max_bytes=1000)


def selected_task_manifest():
    files = []
    for prefix in task_assets.OBJECT_PREFIXES:
        files.append(record(prefix + "/fixture"))
    for task in task_assets.TASKS:
        for index in range(task_assets.EVAL_LAYOUTS_PER_TASK):
            files.append(record(f"{assets.LAYOUTS}/{task}_{index}.json"))
    return {
        "schema": "robodojo-selected-task-assets-v1",
        "repo": assets.REPO,
        "revision": assets.REVISION,
        "tasks": list(task_assets.TASKS),
        "protected_eval_layouts_per_task": task_assets.EVAL_LAYOUTS_PER_TASK,
        "files": files,
    }


def test_selected_task_bundle_requires_every_object_group_and_layout():
    value = selected_task_manifest()
    assert task_assets.validate_manifest(value, max_bytes=10_000) > 0
    value["files"].pop()
    with pytest.raises(ValueError, match="Incomplete protected layouts"):
        task_assets.validate_manifest(value, max_bytes=10_000)


def test_selected_task_bundle_rejects_unselected_task():
    value = selected_task_manifest()
    value["files"].append(record(f"{assets.LAYOUTS}/stack_bowls_0.json"))
    with pytest.raises(ValueError, match="Unexpected"):
        task_assets.validate_manifest(value, max_bytes=10_000)


def test_selected_task_download_reuses_stale_lock(tmp_path, monkeypatch):
    value = selected_task_manifest()
    lock = tmp_path / ".robodojo-selected-task-download.lock"
    lock.write_text("dead-pid")
    monkeypatch.setattr(
        task_assets,
        "download_file",
        lambda root, item: dict(item, sha256=item["digest"]),
    )
    report = task_assets.download(value, tmp_path, max_bytes=10_000, workers=1)
    assert report["complete"] and not lock.exists()


def test_selected_task_download_refuses_live_lock(tmp_path):
    lock = tmp_path / ".robodojo-selected-task-download.lock"
    with lock.open("a+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="owns the bundle"):
            task_assets.download(
                selected_task_manifest(), tmp_path, max_bytes=10_000, workers=1
            )
