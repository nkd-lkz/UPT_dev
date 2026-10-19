# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Provision immutable evaluation assets for the selected RoboDojo task set."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from toolkits.rlt.robodojo_assets import (
    LAYOUTS,
    REPO,
    REVISION,
    download_file,
    list_tree,
)
from toolkits.rlt.robodojo_task_plan import CANDIDATES

TASKS = tuple(candidate.name for candidate in CANDIDATES)
OBJECT_PREFIXES = (
    "Assets/Object/RoboDojo/Geometry/drawer",
    "Assets/Object/RoboDojo/Geometry/cube_cushion",
    "Assets/Object/RoboDojo/Geometry/monitor",
    "Assets/Object/RoboDojo/Geometry/mousemat",
    "Assets/Object/RoboDojo/Geometry/frame",
    "Assets/Object/RoboDojo/Rigid/mouse",
    "Assets/Object/RoboDojo/Rigid/alarm",
    "Assets/Object/RoboDojo/Rigid/keyboard",
    "Assets/Object/RoboDojo/Rigid/garage",
    "Assets/Object/RoboDojo/Geometry/egg_basket",
    "Assets/Object/RoboDojo/Rigid/egg",
    "Assets/Object/RoboDojo/Articulation/egg_holder",
    "Assets/Object/RoboDojo/Geometry/factory_bolt",
    "Assets/Object/RoboDojo/Rigid/factory_nut",
    "Assets/Object/RoboDojo/Rigid/charger",
    "Assets/Object/RoboDojo/Rigid/socket",
    "Assets/Object/RoboDojo/Geometry/key_slot",
    "Assets/Object/RoboDojo/Rigid/key",
    "Assets/Object/RoboDojo/Geometry/stack_base",
    "Assets/Object/RoboDojo/Rigid/stackingblocks",
)
EVAL_LAYOUTS_PER_TASK = 50
LOG = logging.getLogger(__name__)


def _layout_identity(path: str) -> tuple[str, int] | None:
    match = re.fullmatch(
        rf"{re.escape(LAYOUTS)}/({'|'.join(map(re.escape, TASKS))})_(\d+)\.json",
        path,
    )
    return (match.group(1), int(match.group(2))) if match else None


def _selected(path: str) -> bool:
    layout = _layout_identity(path)
    if layout:
        return layout[1] < EVAL_LAYOUTS_PER_TASK
    return any(path.startswith(prefix + "/") for prefix in OBJECT_PREFIXES)


def build_manifest(session: requests.Session, *, max_bytes: int) -> dict:
    """Inventory the fixed six-task object and protected-layout bundle."""
    rows = []
    for prefix in OBJECT_PREFIXES:
        rows.extend(list_tree(session, prefix))
    rows.extend(list_tree(session, LAYOUTS))
    files = {}
    for row in rows:
        if row["type"] != "file" or not _selected(row["path"]):
            continue
        lfs = row.get("lfs")
        record = {
            "path": row["path"],
            "size": row["size"],
            "hash_kind": "sha256" if lfs else "git_blob_sha1",
            "digest": lfs["oid"] if lfs else row["oid"],
        }
        files[record["path"]] = record
    manifest = {
        "schema": "robodojo-selected-task-assets-v1",
        "repo": REPO,
        "revision": REVISION,
        "tasks": list(TASKS),
        "protected_eval_layouts_per_task": EVAL_LAYOUTS_PER_TASK,
        "files": sorted(files.values(), key=lambda item: item["path"]),
    }
    validate_manifest(manifest, max_bytes=max_bytes)
    return manifest


def validate_manifest(manifest: dict, *, max_bytes: int) -> int:
    """Reject task substitutions, incomplete prefixes, and oversized bundles."""
    identity = (
        manifest.get("schema"),
        manifest.get("repo"),
        manifest.get("revision"),
        tuple(manifest.get("tasks", [])),
        manifest.get("protected_eval_layouts_per_task"),
    )
    expected = (
        "robodojo-selected-task-assets-v1",
        REPO,
        REVISION,
        TASKS,
        EVAL_LAYOUTS_PER_TASK,
    )
    if identity != expected:
        raise ValueError("Unsupported selected-task artifact identity")
    seen, total = set(), 0
    layout_counts = dict.fromkeys(TASKS, 0)
    for record in manifest.get("files", []):
        path = record.get("path")
        if not isinstance(path, str) or path in seen or not _selected(path):
            raise ValueError(f"Unexpected or duplicate artifact: {path!r}")
        seen.add(path)
        if type(record.get("size")) is not int or record["size"] < 0:
            raise ValueError("Invalid artifact size")
        digits = {"sha256": 64, "git_blob_sha1": 40}.get(record.get("hash_kind"))
        if not digits or not re.fullmatch(
            rf"[0-9a-f]{{{digits}}}", record.get("digest", "")
        ):
            raise ValueError("Invalid artifact digest")
        layout = _layout_identity(path)
        if layout:
            layout_counts[layout[0]] += 1
        total += record["size"]
    if total > max_bytes:
        raise ValueError(f"Artifact budget exceeded: {total} > {max_bytes}")
    missing = [
        prefix
        for prefix in OBJECT_PREFIXES
        if not any(path.startswith(prefix + "/") for path in seen)
    ]
    if missing:
        raise ValueError(f"Missing object asset groups: {missing}")
    if any(count != EVAL_LAYOUTS_PER_TASK for count in layout_counts.values()):
        raise ValueError(f"Incomplete protected layouts: {layout_counts}")
    return total


def download(manifest: dict, root: Path, *, max_bytes: int, workers: int) -> dict:
    """Download and verify the selected task bundle under an exclusive lock."""
    total = validate_manifest(manifest, max_bytes=max_bytes)
    if workers not in range(1, 5):
        raise ValueError("Use one to four download workers")
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < total + 1024**3:
        raise ValueError("Insufficient storage reserve for the bundle")
    lock = root / ".robodojo-selected-task-download.lock"
    lock_stream = lock.open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_stream.close()
        raise RuntimeError("Another selected-task download owns the bundle") from None
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(str(os.getpid()))
    lock_stream.flush()
    report = {"complete": False, "revision": REVISION, "files": [], "errors": []}
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(download_file, root, record): record
                for record in manifest["files"]
            }
            for future in as_completed(pending):
                record = pending[future]
                try:
                    report["files"].append(future.result())
                except Exception as error:
                    report["errors"].append(
                        {"path": record["path"], "error": str(error)}
                    )
                temporary = root / ".robodojo-selected-task-report.tmp"
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(root / "robodojo-selected-task-report.json")
        report["complete"] = (
            len(report["files"]) == len(manifest["files"])
            and not report["errors"]
        )
        (root / "robodojo-selected-task-report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()
        lock.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "download"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--max-gib", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    budget = args.max_gib * 1024**3
    if budget < 1:
        parser.error("--max-gib must be positive")
    if args.command == "inventory":
        with requests.Session() as session:
            manifest = build_manifest(session, max_bytes=budget)
        with args.manifest.open("x") as stream:
            json.dump(manifest, stream, indent=2)
        LOG.info(
            "Inventoried %d files, %.3f GB",
            len(manifest["files"]),
            sum(item["size"] for item in manifest["files"]) / 1e9,
        )
        return 0
    if args.root is None:
        parser.error("download requires --root")
    report = download(
        json.loads(args.manifest.read_text()),
        args.root,
        max_bytes=budget,
        workers=args.workers,
    )
    return 0 if report["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
