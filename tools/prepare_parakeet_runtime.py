"""Create a strict private Parakeet candidate runtime from measured wheels.

This preparation helper intentionally knows no Parakeet or NeMo dependency
versions.  The caller must supply a private, already measured wheel lock and
wheelhouse plus the exact system-Python patch version selected by that lock.
No resolver, installer, download, or candidate import runs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from tools.streaming_stt.runtime_builder import (
    RuntimeBuildError,
    RuntimeBuildPolicy,
    prepare_locked_runtime,
)
from tools.streaming_stt.wheel_lock import (
    PARAKEET_WHEEL_ARCHIVE_POLICY,
    WheelLockError,
    load_wheel_lock,
)


_PYTHON_VERSION_RE = re.compile(
    r"(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)\Z"
)
_SAFE_ERROR = {
    "ok": False,
    "error": "parakeet_runtime_prerequisites_unavailable",
}


def _runtime_policy(python_version: str) -> RuntimeBuildPolicy:
    match = _PYTHON_VERSION_RE.fullmatch(python_version)
    if match is None:
        raise RuntimeBuildError()
    return RuntimeBuildPolicy(
        python_version=python_version,
        python_directory=f"python{match.group('major')}.{match.group('minor')}",
        wheel_archive_policy=PARAKEET_WHEEL_ARCHIVE_POLICY,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a script-free Parakeet candidate runtime from a private, "
            "already measured wheel lock and wheelhouse. No resolver, package "
            "installer, download, or candidate import is used. Its closed "
            "archive policy permits only the exact measured non-runtime, "
            ".pth, compression, and shared-ownership exceptions."
        )
    )
    parser.add_argument("--wheel-lock", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--system-python", type=Path, required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        lock = load_wheel_lock(args.wheel_lock)
        prepared = prepare_locked_runtime(
            lock,
            args.wheelhouse,
            system_python=args.system_python,
            output_root=args.output_root,
            receipt_path=args.receipt,
            policy=_runtime_policy(args.python_version),
        )
        result: Mapping[str, object] = {
            "ok": True,
            "python_version": args.python_version,
            "wheel_lock_sha256": prepared.wheel_lock_sha256,
            "runtime_content_sha256": prepared.expectation.content_digest,
            "runtime_file_count": prepared.expectation.file_count,
            "runtime_total_size_bytes": prepared.expectation.total_size_bytes,
            "runtime_maximum_file_bytes": prepared.expectation.maximum_file_bytes,
            "runtime_receipt_sha256": prepared.receipt.digest,
        }
        code = 0
    except (RuntimeBuildError, WheelLockError):
        result = _SAFE_ERROR
        code = 2
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
