# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check a three-port Ray block without replacing existing listeners."""

import socket
from contextlib import ExitStack


def select_ray_ports(requested: int | None = None) -> int:
    """Return a bindable head/client/dashboard block, or reject an explicit one.

    Sockets are released before returning; Ray must still fail closed if another
    process claims the ports before startup. Existing services are never stopped.
    """
    if requested is not None and (
        not 1024 <= requested <= 65533
        or set(range(requested, requested + 3)) & {6379, 6385, 6386, 6387}
    ):
        raise ValueError("Invalid or protected Ray port block")
    for _ in range(128):
        try:
            with ExitStack() as stack:
                head = stack.enter_context(socket.socket())
                head.bind(("0.0.0.0", requested or 0))
                port = head.getsockname()[1]
                if port > 65533 or set(range(port, port + 3)) & {
                    6379,
                    6385,
                    6386,
                    6387,
                }:
                    continue
                for candidate in (port + 1, port + 2):
                    peer = stack.enter_context(socket.socket())
                    peer.bind(("0.0.0.0", candidate))
                return port
        except OSError:
            if requested is not None:
                raise
    raise RuntimeError("Could not find a free three-port Ray block")
