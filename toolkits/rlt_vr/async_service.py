# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Durable RPC admission with one owner for GPU inference and learning."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
from collections import deque
from concurrent.futures import Future
from pathlib import Path

from .online_service import (
    ONLINE_PROTOCOL,
    OnlineService,
    observation_id,
    validate_transition,
)


def atomic_json(path: Path, value: dict) -> None:
    """Commit a JSON file before acknowledging it, on Windows and POSIX."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class AsyncOnlineService:
    """Acknowledge durable receipts independently of feature extraction.

    Only the worker touches the model, learner or core service. Frontend threads
    validate and journal requests under a condition lock. Journals are retained
    under a byte quota; a full pending budget applies explicit backpressure.
    Checkpoints name all journal roots and a processed ID, so receipts after a
    checkpoint can be replayed on restart without discarding acknowledged data.
    """

    def __init__(
        self,
        service: OnlineService,
        *,
        max_pending: int = 128,
        max_journal_bytes: int = 4 * 1024**3,
        reserve_bytes: int = 1024**3,
        restored: dict | None = None,
    ) -> None:
        if min(max_pending, max_journal_bytes) <= 0 or reserve_bytes < 0:
            raise ValueError("Invalid inbox limits")
        self.service = service
        self.max_pending, self.max_bytes, self.reserve = (
            max_pending,
            max_journal_bytes,
            reserve_bytes,
        )
        self.condition = threading.Condition()
        self.session: str | None = None
        self.sequence, self.digest, self.episode = -1, "", -1
        self.last_next, self.done = "", False
        self.pending: deque[tuple[int, Path]] = deque()
        self.predictions: deque[tuple[dict, Future]] = deque()
        self.active = False
        self.learning_active = False
        self.closing = False
        self.fault: str | None = None
        self.processed_id = -1
        self.roots: list[Path] = []
        self.total_bytes = 0
        metadata = (restored or {}).get("transport", {})
        if restored and not metadata:
            raise ValueError("Resume requires an asynchronous schema-2 checkpoint")
        self.processed_id = metadata.get("processed_id", -1)
        self.roots = [Path(p) for p in metadata.get("journal_roots", [])]
        records: dict[int, Path] = {}
        for root in self.roots:
            if not root.is_dir():
                raise FileNotFoundError(f"Retain the acknowledged journal: {root}")
            for path in root.glob("*.json"):
                index = int(path.stem)
                if index in records:
                    raise ValueError("Duplicate durable journal ID")
                records[index] = path
                self.total_bytes += path.stat().st_size
        self.next_id = max(records, default=-1) + 1
        if self.processed_id >= self.next_id:
            raise ValueError("Checkpoint refers to missing durable records")
        for index in range(self.processed_id + 1, self.next_id):
            if index not in records:
                raise ValueError("Missing acknowledged journal entry")
            self.pending.append((index, records[index]))
        self.root = service.directory / "inbox"
        self.root.mkdir(exist_ok=False)
        self.roots.append(self.root)
        self.snapshot = service({"op": "status"})
        # The wrapper checkpoints after advancing its own processed watermark.
        service.checkpoint_callback = lambda: None
        self._checkpoint()
        self.thread = threading.Thread(
            target=self._work, name="rlt-learner-owner", daemon=True
        )
        self.thread.start()

    def _status(self) -> dict:
        return {
            **self.snapshot,
            "online_protocol": ONLINE_PROTOCOL,
            "received_sequence": self.sequence,
            "sequence": self.snapshot["sequence"]
            if self.snapshot["session"] == self.session
            else -1,
            "receipt_id": self.next_id - 1,
            "receipt_run": str(self.service.directory.resolve()),
            "pending_learning": len(self.pending) + int(self.learning_active),
            "journal_bytes": self.total_bytes,
            "max_pending_learning": self.max_pending,
            "transport_fault": self.fault,
            "faulted": self.fault is not None or self.snapshot["faulted"],
        }

    def __call__(self, message: dict) -> dict:
        """Admit bounded requests; only predict waits for the model owner."""
        future = None
        with self.condition:
            op = message.get("op")
            if op in {"health", "status"}:
                return self._status()
            if self.closing or self.fault:
                raise RuntimeError("Service closing/faulted; retain local journal")
            if op == "begin":
                if message.get("online_protocol") != ONLINE_PROTOCOL:
                    raise ValueError(
                        "Update both client and server: online protocol mismatch"
                    )
                session = message.get("session")
                if not isinstance(session, str) or not 8 <= len(session) <= 64:
                    raise ValueError("Invalid session ID")
                if self.session is not None and session != self.session:
                    raise ValueError(
                        "Another client owns the run; end it or resume the server"
                    )
                if self.session is None:
                    if self.pending or self.active or self.predictions:
                        return {**self._status(), "busy": True}
                    self.session = session
                    self.sequence, self.digest, self.episode = -1, "", -1
                    self.last_next, self.done = "", False
                return {**self._status(), "busy": False}
            if self.session is None or message.get("session") != self.session:
                raise ValueError("Begin the matching client session first")
            if op == "end":
                if self.predictions:
                    return {**self._status(), "busy": True}
                self.session = None
                return {**self._status(), "busy": False}
            if op == "predict":
                if len(self.predictions) >= 1:
                    raise RuntimeError("Prediction already queued")
                future = Future()
                self.predictions.append((dict(message), future))
                self.condition.notify_all()
            elif op == "observe":
                clean = {
                    k: v for k, v in message.items() if k not in {"token", "request_id"}
                }
                digest = hashlib.sha256(
                    json.dumps(clean, sort_keys=True, allow_nan=False).encode()
                ).hexdigest()
                sequence = clean.get("sequence")
                if sequence == self.sequence:
                    if digest != self.digest:
                        raise ValueError("Conflicting duplicate durable receipt")
                    return {**self._status(), "duplicate": True, "busy": False}
                if type(sequence) is not int or sequence != self.sequence + 1:
                    raise ValueError("Out-of-order durable receipt")
                if len(self.pending) + int(self.learning_active) >= self.max_pending:
                    return {**self._status(), "busy": True}
                obs, nxt = validate_transition(clean)
                version = clean.get("policy_version")
                if (
                    type(version) is not int
                    or not -1 <= version <= self.snapshot["metrics"]["policy_version"]
                ):
                    raise ValueError("Invalid behavior version")
                episode = clean["episode"]
                if episode < self.episode or (
                    episode == self.episode
                    and (self.done or observation_id(obs) != self.last_next)
                ):
                    raise ValueError("Noncontiguous durable transition")
                size = len(json.dumps(clean, allow_nan=False).encode())
                if (
                    self.total_bytes + size > self.max_bytes
                    or shutil.disk_usage(self.root).free < self.reserve + size
                ):
                    return {**self._status(), "busy": True, "storage_full": True}
                path = self.root / f"{self.next_id:012d}.json"
                try:
                    atomic_json(path, clean)
                except Exception:
                    self.fault = "journal_write_failed"
                    raise
                self.total_bytes += path.stat().st_size
                self.pending.append((self.next_id, path))
                self.next_id += 1
                self.sequence, self.digest, self.episode = sequence, digest, episode
                self.done, self.last_next = (
                    clean["terminated"] or clean["truncated"],
                    observation_id(nxt),
                )
                self.condition.notify_all()
                return {**self._status(), "duplicate": False, "busy": False}
            else:
                raise ValueError("Unknown online operation")
        return future.result(timeout=25)

    def _activate(self, session: str) -> None:
        if self.service.session != session:
            self.service.session = None
            self.service.sequence, self.service.episode = -1, -1
            self.service.digest, self.service.last_next, self.service.done = (
                "",
                "",
                False,
            )
            self.service({"op": "begin", "session": session})

    def _checkpoint(self) -> None:
        with self.condition:
            metadata = {
                **self.service.metadata(),
                "transport": {
                    "journal_roots": [str(p.resolve()) for p in self.roots],
                    "processed_id": self.processed_id,
                },
            }
        self.service.learner.save(self.service.directory / "learner.pt", metadata)

    def _work(self) -> None:
        last_prediction = False
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.closing or self.pending or self.predictions
                    )
                    if self.closing:
                        break
                    prediction = bool(self.predictions) and (
                        not last_prediction or not self.pending
                    )
                    item = (
                        self.predictions.popleft()
                        if prediction
                        else self.pending.popleft()
                    )
                    self.active = True
                    self.learning_active = not prediction
                if prediction:
                    message, future = item
                    try:
                        self._activate(message["session"])
                        result = self.service(message)
                        with self.condition:
                            self.snapshot = self.service({"op": "status"})
                        future.set_result(result)
                    except Exception as exc:
                        future.set_exception(exc)
                else:
                    index, path = item
                    message = json.loads(path.read_text())
                    self._activate(message["session"])
                    self.service(message)
                    self.processed_id = index
                    if (
                        self.service.learner.accepted
                        % self.service.learner.cfg["checkpoint_interval"]
                        == 0
                    ):
                        self._checkpoint()
                with self.condition:
                    self.snapshot = self.service({"op": "status"})
                    self.active = False
                    self.learning_active = False
                    self.condition.notify_all()
                last_prediction = prediction
            self._checkpoint()
        except BaseException as exc:
            logging.exception("Online worker failed; durable journal retained")
            with self.condition:
                self.fault = f"worker_{type(exc).__name__}"
                self.active = False
                self.learning_active = False
                self.condition.notify_all()
        finally:
            with self.condition:
                for _, future in self.predictions:
                    future.set_exception(RuntimeError("Learner worker stopped"))
                self.predictions.clear()

    def close(self, timeout: float = 60) -> None:
        """Finish the active operation, checkpoint, and retain pending journals."""
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError(
                "Learner still finishing; do not delete acknowledged journals"
            )
        if self.fault:
            raise RuntimeError(
                f"{self.fault}: use last good checkpoint and retain inbox"
            )
