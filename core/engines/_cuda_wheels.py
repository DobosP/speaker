"""Load the CUDA runtime shipped by pinned Python wheels.

Faster-Whisper's CTranslate2 wheel resolves cuBLAS and cuDNN at runtime.  Pip
installs those libraries below the active environment's ``site-packages``
rather than in the system loader path, so a normal process cannot find them
unless it mutates ``LD_LIBRARY_PATH`` before Python starts.  This module keeps
that process-local boundary explicit: it resolves only the installed
``nvidia.*`` wheel namespaces and loads exact CUDA 12/cuDNN 9 sonames by
absolute path before Faster-Whisper is imported.

No system CUDA directory or ambient library search path is consulted.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable


_WHEEL_LIBRARIES = (
    ("nvidia.cublas", "libcublasLt.so.12"),
    ("nvidia.cublas", "libcublas.so.12"),
    ("nvidia.cuda_nvrtc", "libnvrtc.so.12"),
    ("nvidia.cudnn", "libcudnn.so.9"),
)
# These flags are POSIX-only attributes.  Keep module import harmless on other
# platforms; ``preload`` rejects them before the loader boundary is reached.
_LOAD_MODE = getattr(os, "RTLD_NOW", 0) | getattr(
    os,
    "RTLD_GLOBAL",
    getattr(ctypes, "RTLD_GLOBAL", 0),
)


class _CudaWheelBootstrap:
    """Thread-safe, one-shot loader with injectable boundaries for tests."""

    def __init__(
        self,
        *,
        lib_directory: Callable[[str], Path],
        load_library: Callable[..., Any],
        system: Callable[[], str],
        machine: Callable[[], str],
    ) -> None:
        self._lib_directory = lib_directory
        self._load_library = load_library
        self._system = system
        self._machine = machine
        self._lock = threading.Lock()
        self._complete = False
        # Retain CDLL objects for the lifetime of the process.  Releasing one
        # could unload a dependency still needed by CTranslate2.
        self._handles: list[Any] = []

    def preload(self) -> None:
        if self._complete:
            return
        with self._lock:
            if self._complete:
                return
            if self._system() != "linux" or self._machine() != "x86_64":
                raise RuntimeError(
                    "the CUDA Faster-Whisper verifier requires Linux x86_64"
                )

            directories: dict[str, Path] = {}
            try:
                for package, soname in _WHEEL_LIBRARIES:
                    directory = directories.get(package)
                    if directory is None:
                        directory = self._lib_directory(package)
                        directories[package] = directory
                    library = (directory / soname).resolve(strict=True)
                    if library.parent != directory:
                        raise RuntimeError
                    handle = self._load_library(str(library), mode=_LOAD_MODE)
                    self._handles.append(handle)
            except (OSError, RuntimeError, TypeError, ValueError):
                # Do not expose arbitrary loader details or fall back to a
                # system directory.  A later call may retry after installation
                # has been repaired.
                raise RuntimeError(
                    "the pinned CUDA wheel runtime is incomplete; reinstall "
                    "the Linux x86_64 STT dependencies"
                ) from None
            self._complete = True


def _wheel_lib_directory(package: str) -> Path:
    """Return one wheel-owned ``lib`` directory inside the active prefix."""
    try:
        spec = find_spec(package)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    locations = None if spec is None else spec.submodule_search_locations
    if not locations:
        raise RuntimeError("CUDA wheel namespace is unavailable")

    try:
        prefix = Path(sys.prefix).resolve(strict=True)
    except (OSError, RuntimeError):
        raise RuntimeError("active Python prefix is unavailable") from None

    candidates: list[Path] = []
    for raw_location in locations:
        try:
            directory = (Path(raw_location) / "lib").resolve(strict=True)
            directory.relative_to(prefix)
        except (OSError, RuntimeError, ValueError):
            continue
        if directory.is_dir() and directory not in candidates:
            candidates.append(directory)
    if len(candidates) != 1:
        raise RuntimeError("CUDA wheel namespace is missing or ambiguous")
    return candidates[0]


_BOOTSTRAP = _CudaWheelBootstrap(
    lib_directory=_wheel_lib_directory,
    load_library=ctypes.CDLL,
    system=lambda: sys.platform,
    machine=platform.machine,
)


def preload_cuda_wheel_libraries() -> None:
    """Make pinned wheel-owned CUDA libraries visible to CTranslate2."""
    _BOOTSTRAP.preload()
