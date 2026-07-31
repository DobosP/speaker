"""Private helpers for deterministic isolated streaming-STT tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Mapping, Sequence

from tools.streaming_stt.source_bundle import (
    SourceBundle,
    stage_source_bundle,
    stage_worker_source_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = REPO_ROOT / "tools" / "streaming_stt" / "worker.py"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_file(path: Path, *, allow_path: Path | None = None) -> dict[str, object]:
    selected = allow_path or path
    resolved = path.resolve(strict=True)
    return {
        "path": str(selected),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def pcm_bytes(values: Sequence[float]) -> bytes:
    return b"".join(struct.pack("<f", float(value)) for value in values)


def stage_test_source_bundle(
    scratch: Path,
    worker_path: Path,
) -> SourceBundle:
    destination = scratch / "source-bundle"
    if worker_path.resolve(strict=True) == WORKER_PATH.resolve(strict=True):
        return stage_worker_source_bundle(REPO_ROOT, destination)
    return stage_source_bundle(
        {"worker.py": worker_path},
        destination,
        worker_relative_path="worker.py",
        required_files=("worker.py",),
    )


def resource(
    *,
    rss_mb: float = 64.0,
    threads: int = 1,
    vram_mb: float | None = None,
) -> dict[str, object]:
    return {"rss_mb": rss_mb, "threads": threads, "vram_mb": vram_mb}


def scripted_case(
    *,
    partials: Sequence[tuple[int, str, float, float]],
    final: str,
    elapsed_ms: float = 250.0,
    finalization_ms: float = 50.0,
    compute_ms: float = 25.0,
    deadline_misses: int = 0,
    max_backlog_ms: float = 2.0,
    resources: Mapping[str, object] | None = None,
    hang_sec: float = 0.0,
) -> dict[str, object]:
    return {
        "partials": [
            {
                "after_samples": after_samples,
                "text": text,
                "elapsed_ms": partial_elapsed_ms,
                "decode_ms": decode_ms,
            }
            for after_samples, text, partial_elapsed_ms, decode_ms in partials
        ],
        "final": final,
        "elapsed_ms": elapsed_ms,
        "finalization_ms": finalization_ms,
        "compute_ms": compute_ms,
        "deadline_misses": deadline_misses,
        "max_backlog_ms": max_backlog_ms,
        "resources": dict(resources or resource()),
        "hang_sec": hang_sec,
    }


def write_fixture(
    root: Path,
    case_rows: Sequence[dict[str, object]],
    scripts: Sequence[dict[str, object]],
    *,
    startup_timeout_sec: float = 3.0,
    case_timeout_sec: float = 2.0,
    worker_path: Path = WORKER_PATH,
) -> tuple[Path, Path, list[str]]:
    if len(case_rows) != len(scripts):
        raise ValueError
    corpus_cases: list[dict[str, object]] = []
    artifact_cases: dict[str, object] = {}
    digests: list[str] = []
    for index, (row, script) in enumerate(zip(case_rows, scripts)):
        payload = pcm_bytes(row["values"])
        digest = hashlib.sha256(payload).hexdigest()
        digests.append(digest)
        audio_path = root / f"audio-{index}.f32le"
        audio_path.write_bytes(payload)
        corpus_cases.append(
            {
                "id": row.get("id", f"case-{index}"),
                "file": audio_path.name,
                "sha256": digest,
                "samples": len(row["values"]),
                "expected_text": row["expected_text"],
                "assertion": row.get("assertion", "transcript"),
                "commands": list(row.get("commands", [])),
                "tags": list(row.get("tags", [])),
            }
        )
        artifact_cases[digest] = script

    corpus_path = root / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "deterministic harness test only",
                "cases": corpus_cases,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact_path = root / "fake-script.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ready": {
                    "model_load_ms": 12.5,
                    "resources": resource(rss_mb=48.0, threads=1),
                },
                "cases": artifact_cases,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    python_path = Path(sys.executable)
    manifest_path = root / "worker-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "fake-stream-v1",
                "adapter": "fake-json-v1",
                "python": bound_file(
                    python_path.resolve(strict=True),
                    allow_path=python_path,
                ),
                "worker": bound_file(worker_path),
                "artifacts": [
                    {
                        "name": "fake-script",
                        **bound_file(artifact_path),
                    }
                ],
                "limits": {
                    "startup_timeout_sec": startup_timeout_sec,
                    "case_timeout_sec": case_timeout_sec,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path, corpus_path, digests
