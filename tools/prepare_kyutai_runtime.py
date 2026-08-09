"""Create an exact private Kyutai runtime from an offline wheelhouse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from tools.streaming_stt.runtime_builder import (
    RuntimeBuildError,
    RuntimeBuildPolicy,
    prepare_locked_runtime,
)
from tools.streaming_stt.wheel_lock import (
    KYUTAI_WHEEL_ARCHIVE_POLICY,
    WheelLockError,
    load_wheel_lock,
)


_SAFE_ERROR = {
    "ok": False,
    "error": "kyutai_runtime_prerequisites_unavailable",
}
_RUNTIME_POLICY = RuntimeBuildPolicy(
    python_version="3.12.3",
    python_directory="python3.12",
    wheel_archive_policy=KYUTAI_WHEEL_ARCHIVE_POLICY,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the exact release-owned import runtime from a private, "
            "already downloaded Kyutai wheelhouse. No resolver, package "
            "installer, network access, or candidate import is used."
        )
    )
    parser.add_argument("--wheel-lock", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--system-python", type=Path, required=True)
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
            policy=_RUNTIME_POLICY,
        )
        result: Mapping[str, object] = {
            "ok": True,
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
