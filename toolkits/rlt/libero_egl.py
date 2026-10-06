# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Match EGL devices to NVIDIA UUIDs without creating a rendering context."""

import ctypes
import uuid


def unique_device_index(devices: list[str | None], gpu_uuid: str) -> int:
    """Fail closed when the selected physical GPU has no unique EGL mapping."""
    target = str(uuid.UUID(gpu_uuid.removeprefix("GPU-")))
    indices = [i for i, value in enumerate(devices) if value == target]
    if len(indices) != 1:
        raise RuntimeError(f"Need one EGL device for {gpu_uuid}; found {indices}")
    return indices[0]


def egl_device_index(gpu_uuid: str) -> int:
    """Resolve EGL_EXT_device_persistent_id; never assume CUDA/EGL order agrees.

    Requires an EGL loader and driver supporting device UUID queries. Missing or
    ambiguous mappings are errors, not a reason to fall back to GPU zero.
    """
    lib = ctypes.CDLL("libEGL.so.1")
    lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    lib.eglGetProcAddress.restype = ctypes.c_void_p
    query_address = lib.eglGetProcAddress(b"eglQueryDevicesEXT")
    binary_address = lib.eglGetProcAddress(b"eglQueryDeviceBinaryEXT")
    if not query_address or not binary_address:
        raise RuntimeError("EGL driver does not expose device UUID queries")
    query = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )(query_address)
    binary = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )(binary_address)
    count = ctypes.c_int()
    if not query(0, None, ctypes.byref(count)) or count.value < 1:
        raise RuntimeError("No EGL devices enumerated")
    handles = (ctypes.c_void_p * count.value)()
    if not query(count.value, handles, ctypes.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    identifiers = []
    for handle in handles:
        value = (ctypes.c_ubyte * 16)()
        size = ctypes.c_int()
        ok = binary(handle, 0x335C, 16, value, ctypes.byref(size))
        identifiers.append(
            str(uuid.UUID(bytes=bytes(value))) if ok and size.value == 16 else None
        )
    return unique_device_index(identifiers, gpu_uuid)
