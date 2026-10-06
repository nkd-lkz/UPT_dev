# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Bounded transition uploads that keep network work off the VR/UI thread."""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

from .protocol import CAMERAS, CONTRACT, encode_image, request


def encode_observation(observation: dict) -> dict:
    """Encode the original model images and qpos with the shared contract."""
    return {
        "contract": CONTRACT,
        "state": observation["state"].tolist(),
        **{key: encode_image(observation[key]) for key in CAMERAS},
    }


class TransitionUploader:
    """Upload in sequence; retry lost acknowledgements without changing IDs.

    The caller must persist raw transitions before submit. When capacity is
    reached it pauses simulation; pending samples are never silently dropped.
    """

    def __init__(
        self, port: int, token: str, session: str, capacity: int = 16, recorder=None
    ) -> None:
        self.port, self.token, self.session = port, token, session
        self.queue: queue.Queue = queue.Queue(maxsize=capacity)
        self.error: str | None = None
        self.metrics: dict = {}
        self.accepted_sequence = -1
        self.processed_sequence = -1
        self.server_pending = 0
        self.recorder = recorder
        self.capacity = capacity
        self._outstanding = 0
        self._lock = threading.Lock()
        self.storage_full = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="rlt-upload"
        )
        self._thread.start()

    @property
    def ready(self) -> bool:
        """Whether another control step can be recorded without dropping data."""
        return (
            self.error is None
            and self.outstanding < self.capacity
            and not self.storage_full
        )

    @property
    def outstanding(self) -> int:
        with self._lock:
            return self._outstanding

    def submit_path(self, path: Path) -> None:
        """Queue only a durable filename; raw images stay on local disk."""
        with self._lock:
            if self._outstanding >= self.capacity:
                raise RuntimeError("Local outbox quota reached")
            self._outstanding += 1
        self.queue.put_nowait(path)

    def submit(
        self, sequence: int, observation: dict, next_observation: dict, **fields
    ) -> None:
        """Queue an executed, locally journaled step without blocking the UI."""
        item = {
            "sequence": sequence,
            "observation": observation,
            "next_observation": next_observation,
            **fields,
        }
        with self._lock:
            if self._outstanding >= self.capacity:
                raise RuntimeError("Local outbox quota reached")
            self._outstanding += 1
        self.queue.put_nowait(item)

    def _status(self, response: dict) -> None:
        self.metrics = response["metrics"]
        self.processed_sequence = response.get("sequence", -1)
        self.server_pending = response.get("pending_learning", 0)
        self.storage_full = response.get("storage_full", False)
        if response.get("faulted"):
            raise RuntimeError("Server learner faulted; retain local journal")

    def _run(self) -> None:
        while not self._stop.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                if self.recorder is not None and not self._stop.is_set():
                    try:
                        self._status(
                            request(
                                "127.0.0.1",
                                self.port,
                                self.token,
                                {"op": "status", "request_id": 0},
                                5,
                            )
                        )
                    except (OSError, ConnectionError):
                        pass
                    except Exception:
                        self.error = "server_status_fault; retain local journal"
                        return
                    self._stop.wait(0.5)
                continue
            try:
                if isinstance(item, Path):
                    path = item
                    item = self.recorder.upload_item(path)
                    while item is None and not self._stop.is_set():
                        self._stop.wait(0.05)
                        item = self.recorder.upload_item(path)
                    if item is None:
                        return
                payload = {
                    **item,
                    "op": "observe",
                    "session": self.session,
                    "request_id": item["sequence"],
                    "observation": encode_observation(item["observation"]),
                    "next_observation": encode_observation(item["next_observation"]),
                }
                attempt = 0
                while True:
                    try:
                        response = request(
                            "127.0.0.1", self.port, self.token, payload, 30
                        )
                        self._status(response)
                        if response.get("busy"):
                            if self._stop.wait(0.2):
                                return
                            continue
                        break
                    except (OSError, ConnectionError):
                        if attempt == 2:
                            raise
                        time.sleep(0.2)
                        attempt += 1
                if self.recorder is not None:
                    self.recorder.receipt(item["sequence"], self.session, response)
                self.accepted_sequence = item["sequence"]
                with self._lock:
                    self._outstanding -= 1
            except Exception as error:
                self.error = (
                    f"{type(error).__name__}: upload stopped; raw data retained locally"
                )
                logging.exception(
                    "Online upload failed; pause and inspect local records"
                )
                return
            finally:
                self.queue.task_done()

    def close(self, timeout: float = 35) -> None:
        """Drain pending uploads for a bounded interval; report unsent records."""
        self._stop.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive() or self.error or self.outstanding:
            logging.warning(
                "Upload incomplete; last acknowledged sequence=%d. Keep local records.",
                self.accepted_sequence,
            )
