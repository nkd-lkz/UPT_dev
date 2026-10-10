# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Download pinned, bounded RoboDojo reference assets without GPU imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlparse

import requests

REPO = "RoboDojo-Benchmark/RoboDojo"
REVISION = "35efbc7dedfdbeeb6e95fb749bd885d73d483e41"
CHECKPOINT = "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999"
LAYOUTS = "Assets/Eval_Layout/RoboDojo/arx_x5/0"
PREFIXES = (
    "Assets/Robots/x5",
    "Assets/Object/RoboDojo/Rigid/bowl",
    "Assets/Object/RoboDojo/Geometry/camera_stand/00000",
    "Assets/Room/Simple_Room_nolight",
    "Assets/Material/Base",
    "Assets/Material/Templates",
    "Assets/Material/material_0122",
    "Assets/Material/material_0564",
    "Assets/Sensor/Camera",
    LAYOUTS,
    f"{CHECKPOINT}/params",
    f"{CHECKPOINT}/assets",
)
EXACT_FILES = (
    "Assets/Background/brown_photostudio_02_4k.hdr",
    f"{CHECKPOINT}/_CHECKPOINT_METADATA",
)
LOG = logging.getLogger(__name__)


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not isinstance(value, str)
        or not value
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or "\\" in value
    ):
        raise ValueError(f"Unsafe artifact path: {value!r}")
    return value


def selected(path: str) -> bool:
    """Select the stack_bowls seed-0 reference bundle, excluding training state."""
    _safe_relative(path)
    if path.startswith(LAYOUTS + "/"):
        return (
            re.fullmatch(re.escape(LAYOUTS) + r"/stack_bowls_\d+\.json", path)
            is not None
        )
    return path in EXACT_FILES or any(
        path.startswith(prefix + "/") for prefix in PREFIXES if prefix != LAYOUTS
    )


def list_tree(session: requests.Session, prefix: str) -> list[dict]:
    """Read all HF tree pages, retaining a fixed dataset revision."""
    base = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/"
    url = base + quote(prefix, safe="/") + "?recursive=true&limit=1000"
    rows, visited = [], set()
    while url:
        if not url.startswith(base) or url in visited:
            raise ValueError("Unexpected or cyclic dataset pagination URL")
        visited.add(url)
        response = session.get(url, timeout=120)
        response.raise_for_status()
        rows.extend(response.json())
        url = response.links.get("next", {}).get("url")
    return rows


def build_manifest(session: requests.Session, *, max_bytes: int) -> dict:
    """Resolve file sizes and upstream hashes before permitting any downloads."""
    rows = []
    for prefix in PREFIXES:
        LOG.info("Inventory %s", prefix)
        rows.extend(list_tree(session, prefix))
    for parent in dict.fromkeys(str(PurePosixPath(p).parent) for p in EXACT_FILES):
        rows.extend(list_tree(session, parent))
    files = {}
    for row in rows:
        if row["type"] != "file" or not selected(row["path"]):
            continue
        lfs = row.get("lfs")
        record = {
            "path": row["path"],
            "size": row["size"],
            "hash_kind": "sha256" if lfs else "git_blob_sha1",
            "digest": lfs["oid"] if lfs else row["oid"],
        }
        if record["path"] in files and files[record["path"]] != record:
            raise ValueError("Conflicting artifact metadata")
        files[record["path"]] = record
    # Native stack_bowls evaluation caps seed 0 at the first 25 numeric layouts.
    layouts = sorted(
        (p for p in files if p.startswith(LAYOUTS + "/")),
        key=lambda p: int(PurePosixPath(p).stem.rsplit("_", 1)[1]),
    )
    for path in layouts[25:]:
        del files[path]
    result = {
        "schema": "robodojo-assets-v1",
        "repo": REPO,
        "revision": REVISION,
        "task": "stack_bowls",
        "seed": 0,
        "backend": "openpi_reference",
        "checkpoint": CHECKPOINT,
        "files": sorted(files.values(), key=lambda f: f["path"]),
    }
    validate_manifest(result, max_bytes=max_bytes)
    return result


def validate_manifest(manifest: dict, *, max_bytes: int) -> int:
    """Reject substitutions, omitted required groups, and unbounded downloads."""
    expected = (
        "robodojo-assets-v1",
        REPO,
        REVISION,
        "stack_bowls",
        0,
        "openpi_reference",
        CHECKPOINT,
    )
    keys = ("schema", "repo", "revision", "task", "seed", "backend", "checkpoint")
    if tuple(manifest.get(k) for k in keys) != expected:
        raise ValueError("Unsupported artifact manifest identity")
    seen, total = set(), 0
    for record in manifest["files"]:
        path = record["path"]
        if not selected(path) or path in seen:
            raise ValueError(f"Unexpected or duplicate artifact: {path}")
        seen.add(path)
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError("Invalid artifact size")
        digits = {"sha256": 64, "git_blob_sha1": 40}.get(record["hash_kind"])
        if not digits or not re.fullmatch(rf"[0-9a-f]{{{digits}}}", record["digest"]):
            raise ValueError("Invalid artifact digest")
        total += record["size"]
    if total > max_bytes:
        raise ValueError(f"Artifact budget exceeded: {total} > {max_bytes}")
    if not all(any(p.startswith(prefix + "/") for p in seen) for prefix in PREFIXES):
        raise ValueError("Missing required artifact group")
    if not set(EXACT_FILES).issubset(seen):
        raise ValueError("Missing required single-file assets")
    if len([p for p in seen if p.startswith(LAYOUTS + "/")]) != 25:
        raise ValueError("Require exactly 25 stack_bowls layouts")
    norm = f"{CHECKPOINT}/assets/arx_x5_sim/norm_stats.json"
    if norm not in seen or f"{CHECKPOINT}/params/_METADATA" not in seen:
        raise ValueError("Missing checkpoint metadata or ARX X5 normalization")
    return total


def verify_file(path: Path, record: dict) -> str:
    """Verify byte count and upstream digest, returning a local SHA-256."""
    if path.stat().st_size != record["size"]:
        raise ValueError(f"Size mismatch: {path}")
    sha256, sha1 = hashlib.sha256(), hashlib.sha1()
    sha1.update(f"blob {record['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            sha1.update(chunk)
    digest = sha256.hexdigest() if record["hash_kind"] == "sha256" else sha1.hexdigest()
    if digest != record["digest"]:
        raise ValueError(f"Hash mismatch: {path}")
    return sha256.hexdigest()


def download_file(root: Path, record: dict) -> dict:
    """Resume an owned partial file and atomically publish only verified bytes."""
    relative = _safe_relative(record["path"])
    destination = root / relative
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes storage root via a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return dict(record, sha256=verify_file(destination, record))
    partial = destination.with_name(destination.name + ".robodojo-part")
    if partial.is_symlink():
        raise ValueError("Partial artifact must not be a symlink")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(relative, safe='/')}"
    with requests.Session() as session:
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == record["size"]:
                break
            if offset > record["size"]:
                raise ValueError("Partial file exceeds expected size")
            try:
                # A fresh query avoids reusing expired signed redirect URLs.
                with session.get(
                    url,
                    params={"download": "true", "request": time.time_ns()},
                    headers={
                        "Range": f"bytes={offset}-",
                        "Accept-Encoding": "identity",
                    },
                    stream=True,
                    timeout=(30, 180),
                ) as response:
                    response.raise_for_status()
                    if urlparse(response.url).scheme != "https":
                        raise ValueError("Refuse a non-HTTPS artifact redirect")
                    if response.status_code == 206:
                        expected = (
                            f"bytes {offset}-{record['size'] - 1}/{record['size']}"
                        )
                        if response.headers.get("Content-Range") != expected:
                            raise ValueError("Unexpected artifact range")
                    elif response.status_code == 200:
                        offset = 0
                    else:
                        raise ValueError("Unexpected artifact response")
                    with partial.open("ab" if offset else "wb") as stream:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            offset += len(chunk)
                            if offset > record["size"]:
                                raise ValueError(
                                    "Download exceeds declared artifact size"
                                )
                            stream.write(chunk)
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
    digest = verify_file(partial, record)
    partial.replace(destination)
    return dict(record, sha256=digest)


def download(manifest: dict, root: Path, *, max_bytes: int, workers: int = 3) -> dict:
    """Download a fixed bundle; leave resumable partials and a truthful report."""
    total = validate_manifest(manifest, max_bytes=max_bytes)
    if workers not in range(1, 5):
        raise ValueError("Use one to four download workers")
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < total + 1024**3:
        raise ValueError("Insufficient storage reserve for the bundle")
    lock = root / ".robodojo-download.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    report = {"complete": False, "revision": REVISION, "files": [], "errors": []}
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(download_file, root, f): f for f in manifest["files"]
            }
            for future in as_completed(pending):
                record = pending[future]
                try:
                    report["files"].append(future.result())
                    LOG.info(
                        "Verified %d/%d %s",
                        len(report["files"]),
                        len(pending),
                        record["path"],
                    )
                except Exception as error:
                    report["errors"].append(
                        {"path": record["path"], "error": str(error)}
                    )
                    LOG.error("Artifact failed: %s: %s", record["path"], error)
                temporary = root / ".robodojo-download-report.tmp"
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(root / "robodojo-download-report.json")
        report["complete"] = (
            len(report["files"]) == len(manifest["files"]) and not report["errors"]
        )
        (root / "robodojo-download-report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    finally:
        lock.unlink()
    return report


def main() -> int:
    """Inventory or download assets; never start Isaac, a policy, or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "download"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--max-gib", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.max_gib < 1:
        parser.error("--max-gib must be positive")
    budget = args.max_gib * 1024**3
    if args.command == "inventory":
        with requests.Session() as session:
            manifest = build_manifest(session, max_bytes=budget)
        with args.manifest.open("x") as stream:
            json.dump(manifest, stream, indent=2)
        LOG.info(
            "Inventoried %d files, %.3f GB",
            len(manifest["files"]),
            sum(f["size"] for f in manifest["files"]) / 1e9,
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
