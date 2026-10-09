# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Download and verify the pinned public AlphaBrain Pi0.5 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

REPO = "AlphaBrainGroup/paligemma-pi0-libero-all4suite"
REVISION = "482fcfea4d6a5cee97456f50b1f5aeeae539a1c8"
SOURCE_COMMIT = "604924beb77b04b0da49326dfae6ea423a27d28a"
MODEL_SHA256 = "a56e6b36524a0c445074f15e56a5912b765f4330ee740124cfcc9ab994439834"


def file_hash(path: Path, algorithm: str = "sha256", *, git_blob: bool = False) -> str:
    """Hash a release file, optionally using Git's blob header."""
    value = hashlib.new(algorithm)
    if git_blob:
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(destination: Path) -> dict:
    """Resume downloads and publish a manifest only after all hashes match."""
    with urllib.request.urlopen(
        f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true",
        timeout=30,
    ) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise ValueError("Release revision mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe release filename")
        path = destination / relative
        expected = item.get("lfs", {}).get("sha256", item["blobId"])
        is_lfs = "lfs" in item

        def valid(candidate: Path) -> bool:
            return (
                candidate.is_file()
                and candidate.stat().st_size == item["size"]
                and file_hash(
                    candidate, "sha256" if is_lfs else "sha1", git_blob=not is_lfs
                )
                == expected
            )

        if path.exists() and not valid(path):
            raise ValueError(f"Existing release file failed verification: {path}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            print(f"Downloading {name}: {item['size']} bytes", flush=True)
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "5",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "7200",
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",
                ],
                check=True,
                timeout=7500,
            )
            if not valid(partial):
                raise ValueError(f"Downloaded file failed verification: {partial}")
            partial.replace(path)
        files[name] = {"size": item["size"], "sha256": file_hash(path)}
    if files["model.safetensors"]["sha256"] != MODEL_SHA256:
        raise ValueError("Unexpected model digest")
    manifest = {
        "complete": True,
        "repository": REPO,
        "revision": REVISION,
        "training_scope": "public four-suite LIBERO SFT; not the few-demo RLT release",
        "files": files,
    }
    temp = destination / "verified.tmp.json"
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    temp.replace(destination / "verified.json")
    return manifest


def main() -> None:
    """Download into an explicit large-volume directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    download(args.destination)


if __name__ == "__main__":
    main()
