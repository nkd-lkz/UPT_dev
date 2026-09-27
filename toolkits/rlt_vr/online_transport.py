# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Bounded transition uploads that keep network work off the VR/UI thread."""

from __future__ import annotations

import logging
import queue
import threading
import time

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

    def __init__(self, port: int, token: str, session: str, capacity: int = 16) -> None:
        self.port, self.token, self.session = port, token, session
        self.queue: queue.Queue = queue.Queue(maxsize=capacity)
        self.error: str | None = None
        self.metrics: dict = {}
        self.accepted_sequence = -1
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="rlt-upload"
        )
        self._thread.start()

    @property
    def ready(self) -> bool:
        """Whether another control step can be recorded without dropping data."""
        return self.error is None and not self.queue.full()

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
        self.queue.put_nowait(item)

    def _run(self) -> None:
        while not self._stop.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                payload = {
                    **item,
                    "op": "observe",
                    "session": self.session,
                    "request_id": item["sequence"],
                    "observation": encode_observation(item["observation"]),
                    "next_observation": encode_observation(item["next_observation"]),
                }
                for attempt in range(3):
                    try:
                        response = request(
                            "127.0.0.1", self.port, self.token, payload, 30
                        )
                        break
                    except (OSError, ConnectionError):
                        if attempt == 2:
                            raise
                        time.sleep(0.2)
                self.metrics = response["metrics"]
                self.accepted_sequence = item["sequence"]
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
        if self._thread.is_alive() or self.error:
            logging.warning(
                "Upload incomplete; last acknowledged sequence=%d. Keep local records.",
                self.accepted_sequence,
            )
