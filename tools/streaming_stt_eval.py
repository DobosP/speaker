"""Aggregate-only streaming-STT evaluation through an isolated JSONL worker.

The controller supports the deterministic fake contract and manifest-selected
real candidates without importing candidate runtimes into this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import secrets
import shutil
import stat
import tempfile
from typing import Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - the benchmark controller is Linux-first
    fcntl = None  # type: ignore[assignment]

from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    opened_directory_nofollow,
)
from tools.streaming_stt.corpus import (
    LoadedCorpus,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.manifest import (
    FASTER_WHISPER_ENDPOINT_ADAPTER,
    FASTER_WHISPER_REQUIRED_MODEL_FILES,
    FASTER_WHISPER_VOCABULARY_FILES,
    MAX_WORKER_BYTES,
    MOONSHINE_ADAPTER,
    NEMOTRON_ADAPTER,
    PARAKEET_REALTIME_EOU_ADAPTER,
    SHERPA_ZIPFORMER_ADAPTER,
    FasterWhisperEndpointConfig,
    NemotronConfig,
    ParakeetRealtimeEouConfig,
    SherpaZipformerConfig,
    WorkerManifest,
    load_worker_manifest,
)
from tools.streaming_stt.metrics import RunRecord, aggregate_metrics
from tools.streaming_stt.protocol import (
    MAX_PCM_BYTES,
    NATIVE_ENDPOINT_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    PcmInput,
    StreamConfig,
    TranscribeRequest,
    encode_message,
    parse_request,
)
from tools.streaming_stt.runtime_receipt import (
    FASTER_WHISPER_MODEL_TREE_LIMITS,
    FASTER_WHISPER_RUNTIME_TREE_LIMITS,
    NEMOTRON_RUNTIME_TREE_LIMITS,
    PARAKEET_RUNTIME_TREE_LIMITS,
    RuntimeTreeReceiptError,
    load_runtime_tree_receipt,
    verify_venv_runtime_location,
)
from tools.streaming_stt.supervisor import StreamingWorker
from tools.streaming_stt.source_bundle import (
    WORKER_SOURCE_FILES,
    SourceBundle,
    stage_worker_source_bundle,
    verify_source_bundle,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SCRATCH = _REPO_ROOT / ".cache" / "streaming-stt"
_HOST_UID = os.getuid() if hasattr(os, "getuid") else -1
_HOST_HOME = Path(pwd.getpwuid(_HOST_UID).pw_dir) if _HOST_UID >= 0 else Path.home()


def _trusted_host_lock_path() -> Path:
    return _HOST_HOME / ".local" / "state" / "speaker" / "streaming-stt-benchmark.lock"


_HOST_LOCK_PATH = _trusted_host_lock_path()
_FIXED_WORKER = (_REPO_ROOT / "tools/streaming_stt/worker.py").resolve(strict=True)
_FAKE_ADAPTER = "fake-json-v1"
_SAFE_ERROR = {
    "ok": False,
    "error": "streaming_stt_prerequisites_unavailable",
}
_PARAKEET_CGROUP_EVIDENCE = {
    "kind": "systemd-user-scope-cgroup-v2",
    "memory_high_bytes": 6_442_450_944,
    "memory_max_bytes": 8_589_934_592,
    "memory_swap_max_bytes": 0,
    "cpu_quota_percent": 200,
    "tasks_max": 128,
    "oom_policy": "kill",
    "verified": True,
}
_EVALUATOR_FILES = (
    "tools/streaming_stt_eval.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/streaming_stt/corpus.py",
    "tools/streaming_stt/manifest.py",
    "tools/streaming_stt/metrics.py",
    "tools/streaming_stt/protocol.py",
    "tools/streaming_stt/runtime_receipt.py",
    "tools/streaming_stt/source_bundle.py",
    "tools/streaming_stt/supervisor.py",
    "tools/streaming_stt/worker.py",
    "tools/streaming_stt/__init__.py",
    "tools/streaming_stt/adapters/__init__.py",
    "tools/streaming_stt/adapters/fake.py",
    "tools/streaming_stt/adapters/faster_whisper_endpoint.py",
    "tools/streaming_stt/adapters/moonshine.py",
    "tools/streaming_stt/adapters/nemotron.py",
    "tools/streaming_stt/adapters/parakeet_realtime_eou.py",
    "tools/streaming_stt/adapters/zipformer.py",
    "tools/__init__.py",
    "tools/recorded_stt_eval.py",
    "core/wer.py",
)


class _BenchmarkRunLock:
    """One fixed host lease plus a pathname-bound advisory lock."""

    def __init__(
        self,
        path: Path,
        descriptor: int,
        parent_descriptor: int,
    ) -> None:
        self._path = path
        self._descriptor = descriptor
        self._parent_descriptor = parent_descriptor
        self._identity = self._file_identity(os.fstat(descriptor))
        self._parent_identity = self._directory_identity(os.fstat(parent_descriptor))

    @staticmethod
    def _file_identity(metadata: os.stat_result) -> tuple[int, int, int]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
        )

    @staticmethod
    def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            stat.S_IFMT(metadata.st_mode),
        )

    @classmethod
    def acquire(cls, path: Path) -> _BenchmarkRunLock:
        if fcntl is None or os.name != "posix":
            raise ValueError
        candidate = Path(os.path.abspath(path))
        descriptor = -1
        parent_descriptor = -1
        try:
            with opened_directory_nofollow(
                candidate.parent,
                create=True,
                require_private=True,
            ) as (parent, stable_parent):
                if parent != candidate.parent:
                    raise ValueError
                parent_descriptor = os.dup(stable_parent)
            fcntl.flock(
                parent_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                candidate.name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            metadata = os.fstat(descriptor)
            pathname_metadata = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or cls._file_identity(metadata) != cls._file_identity(pathname_metadata)
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise ValueError
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = cls(
                candidate,
                descriptor,
                parent_descriptor,
            )
            acquired.assert_bound()
            return acquired
        except (OSError, ValueError, BoundedReadError):
            if descriptor >= 0:
                os.close(descriptor)
            if parent_descriptor >= 0:
                os.close(parent_descriptor)
            raise ValueError from None

    def assert_bound(self) -> None:
        descriptor = self._descriptor
        parent_descriptor = self._parent_descriptor
        if descriptor < 0 or parent_descriptor < 0:
            raise ValueError
        try:
            metadata = os.fstat(descriptor)
            parent_metadata = os.fstat(parent_descriptor)
            parent_path_metadata = self._path.parent.lstat()
            relative_metadata = os.stat(
                self._path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            full_path_metadata = self._path.lstat()
        except OSError:
            raise ValueError from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or self._file_identity(metadata) != self._identity
            or self._file_identity(relative_metadata) != self._identity
            or self._file_identity(full_path_metadata) != self._identity
            or self._directory_identity(parent_metadata) != self._parent_identity
            or self._directory_identity(parent_path_metadata) != self._parent_identity
        ):
            raise ValueError

    def close(self) -> None:
        descriptor = self._descriptor
        parent_descriptor = self._parent_descriptor
        self._descriptor = -1
        self._parent_descriptor = -1
        if descriptor < 0:
            return
        changed = False
        try:
            try:
                self._descriptor = descriptor
                self._parent_descriptor = parent_descriptor
                self.assert_bound()
            except ValueError:
                changed = True
            finally:
                self._descriptor = -1
                self._parent_descriptor = -1
            assert fcntl is not None
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            if parent_descriptor >= 0:
                try:
                    assert fcntl is not None
                    fcntl.flock(parent_descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(parent_descriptor)
        if changed:
            raise ValueError


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError):
        raise ValueError from None


def _write_strict_report(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write one private standards-compliant JSON report."""

    encoded = (_strict_json(payload) + "\n").encode("utf-8")
    selected = path.expanduser()
    selected.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".streaming-stt-report-",
        dir=selected.parent,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, selected)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _evaluator_binding(adapter: str) -> dict[str, object]:
    if adapter not in {
        _FAKE_ADAPTER,
        FASTER_WHISPER_ENDPOINT_ADAPTER,
        MOONSHINE_ADAPTER,
        NEMOTRON_ADAPTER,
        PARAKEET_REALTIME_EOU_ADAPTER,
        SHERPA_ZIPFORMER_ADAPTER,
    }:
        raise ValueError
    files: dict[str, str] = {}
    for relative in _EVALUATOR_FILES:
        path = (_REPO_ROOT / relative).resolve(strict=True)
        path.relative_to(_REPO_ROOT.resolve(strict=True))
        files[relative] = hash_regular_bounded(
            path,
            maximum_bytes=MAX_WORKER_BYTES,
            expected_bytes=path.stat().st_size,
        ).sha256
    return {
        "schema_version": 1,
        "kind": (
            "isolated_streaming_stt_fake_harness"
            if adapter == _FAKE_ADAPTER
            else (
                "isolated_streaming_stt_production_harness"
                if adapter == SHERPA_ZIPFORMER_ADAPTER
                else "isolated_streaming_stt_candidate_harness"
            )
        ),
        "production_model": adapter == SHERPA_ZIPFORMER_ADAPTER,
        "real_candidate_model": adapter
        in {
            FASTER_WHISPER_ENDPOINT_ADAPTER,
            MOONSHINE_ADAPTER,
            NEMOTRON_ADAPTER,
            PARAKEET_REALTIME_EOU_ADAPTER,
        },
        "model_executed": adapter != _FAKE_ADAPTER,
        "files": files,
    }


def _verified_runtime_receipt_digest(manifest: WorkerManifest) -> str | None:
    if manifest.adapter == _FAKE_ADAPTER:
        if "runtime-receipt" in manifest.artifact_by_name:
            raise ValueError
        return None
    if manifest.adapter == SHERPA_ZIPFORMER_ADAPTER:
        if "runtime-receipt" in manifest.artifact_by_name:
            raise ValueError
        return None
    if manifest.adapter not in {
        FASTER_WHISPER_ENDPOINT_ADAPTER,
        MOONSHINE_ADAPTER,
        NEMOTRON_ADAPTER,
        PARAKEET_REALTIME_EOU_ADAPTER,
    }:
        raise ValueError
    artifact = manifest.artifact_by_name.get("runtime-receipt")
    if artifact is None:
        raise ValueError
    limits = {
        FASTER_WHISPER_ENDPOINT_ADAPTER: FASTER_WHISPER_RUNTIME_TREE_LIMITS,
        NEMOTRON_ADAPTER: NEMOTRON_RUNTIME_TREE_LIMITS,
        PARAKEET_REALTIME_EOU_ADAPTER: PARAKEET_RUNTIME_TREE_LIMITS,
    }.get(manifest.adapter)
    try:
        receipt = load_runtime_tree_receipt(
            artifact.path,
            expected_digest=artifact.sha256,
            **({} if limits is None else {"limits": limits}),
        )
        verify_venv_runtime_location(receipt, manifest.python.path)
    except RuntimeTreeReceiptError:
        raise ValueError from None
    if receipt.digest != artifact.sha256:
        raise ValueError
    if manifest.adapter == NEMOTRON_ADAPTER:
        config = manifest.adapter_config
        wheel_lock = manifest.artifact_by_name.get("runtime-wheel-lock")
        maximum_file = max((item.size_bytes for item in receipt.files), default=0)
        if (
            not isinstance(config, NemotronConfig)
            or wheel_lock is None
            or wheel_lock.sha256 != config.wheel_lock_sha256
            or receipt.content_digest != config.runtime_content_sha256
            or receipt.file_count != config.runtime_file_count
            or receipt.total_size_bytes != config.runtime_total_size_bytes
            or maximum_file != config.runtime_maximum_file_bytes
        ):
            raise ValueError
    if manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
        config = manifest.adapter_config
        wheel_lock = manifest.artifact_by_name.get("runtime-wheel-lock")
        maximum_file = max((item.size_bytes for item in receipt.files), default=0)
        if (
            not isinstance(config, ParakeetRealtimeEouConfig)
            or wheel_lock is None
            or wheel_lock.sha256 != config.wheel_lock_sha256
            or receipt.content_digest != config.runtime_content_sha256
            or receipt.file_count != config.runtime_file_count
            or receipt.total_size_bytes != config.runtime_total_size_bytes
            or maximum_file != config.runtime_maximum_file_bytes
        ):
            raise ValueError
    if manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER:
        config = manifest.adapter_config
        maximum_file = max((item.size_bytes for item in receipt.files), default=0)
        if (
            not isinstance(config, FasterWhisperEndpointConfig)
            or receipt.content_digest != config.runtime_content_sha256
            or receipt.file_count != config.runtime_file_count
            or receipt.total_size_bytes != config.runtime_total_size_bytes
            or maximum_file != config.runtime_maximum_file_bytes
        ):
            raise ValueError
    return receipt.digest


def _verified_model_receipt_digest(manifest: WorkerManifest) -> str | None:
    if manifest.adapter != FASTER_WHISPER_ENDPOINT_ADAPTER:
        if "model-receipt" in manifest.artifact_by_name:
            raise ValueError
        return None
    config = manifest.adapter_config
    artifact = manifest.artifact_by_name.get("model-receipt")
    if not isinstance(config, FasterWhisperEndpointConfig) or artifact is None:
        raise ValueError
    try:
        receipt = load_runtime_tree_receipt(
            artifact.path,
            expected_digest=artifact.sha256,
            limits=FASTER_WHISPER_MODEL_TREE_LIMITS,
        )
    except RuntimeTreeReceiptError:
        raise ValueError from None
    maximum_file = max((item.size_bytes for item in receipt.files), default=0)
    paths = set(receipt.file_by_path)
    if (
        receipt.digest != artifact.sha256
        or not FASTER_WHISPER_REQUIRED_MODEL_FILES.issubset(paths)
        or not FASTER_WHISPER_VOCABULARY_FILES.intersection(paths)
        or receipt.content_digest != config.model_content_sha256
        or receipt.file_count != config.model_file_count
        or receipt.total_size_bytes != config.model_total_size_bytes
        or maximum_file != config.model_maximum_file_bytes
    ):
        raise ValueError
    return receipt.digest


def _worker_binding(
    manifest: WorkerManifest,
    source_bundle: SourceBundle,
    *,
    runtime_receipt_sha256: str | None,
    model_receipt_sha256: str | None = None,
) -> dict[str, object]:
    artifacts = [
        {
            "name": artifact.name,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        }
        for artifact in manifest.artifacts
    ]
    binding: dict[str, object] = {
        "manifest_sha256": manifest.digest,
        "model_id": manifest.model_id,
        "adapter": manifest.adapter,
        "python_sha256": manifest.python.sha256,
        "worker_sha256": manifest.worker.sha256,
        "source_bundle_sha256": source_bundle.tree_sha256,
        "source_bundle_files": len(source_bundle.files),
        "artifact_set_sha256": _canonical_digest(artifacts),
        "artifacts": artifacts,
    }
    if manifest.adapter == _FAKE_ADAPTER:
        if (
            manifest.schema_version != 1
            or manifest.adapter_config is not None
            or runtime_receipt_sha256 is not None
            or model_receipt_sha256 is not None
        ):
            raise ValueError
        return binding
    expected_schema = {
        MOONSHINE_ADAPTER: 2,
        NEMOTRON_ADAPTER: 3,
        SHERPA_ZIPFORMER_ADAPTER: 4,
        PARAKEET_REALTIME_EOU_ADAPTER: 5,
        FASTER_WHISPER_ENDPOINT_ADAPTER: 6,
    }.get(manifest.adapter)
    if (
        expected_schema is None
        or manifest.schema_version != expected_schema
        or manifest.adapter_config is None
        or (
            manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER
            and not isinstance(manifest.adapter_config, ParakeetRealtimeEouConfig)
        )
        or (
            manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER
            and not isinstance(manifest.adapter_config, FasterWhisperEndpointConfig)
        )
        or (
            manifest.adapter == SHERPA_ZIPFORMER_ADAPTER
            and runtime_receipt_sha256 is not None
        )
        or (
            manifest.adapter != SHERPA_ZIPFORMER_ADAPTER
            and runtime_receipt_sha256 is None
        )
        or (
            manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER
            and model_receipt_sha256 is None
        )
        or (
            manifest.adapter != FASTER_WHISPER_ENDPOINT_ADAPTER
            and model_receipt_sha256 is not None
        )
    ):
        raise ValueError
    adapter_config = manifest.adapter_config.as_dict()
    expected_config_fields = {
        MOONSHINE_ADAPTER: {
            "package_version",
            "api_version",
            "model_arch",
            "provider",
            "language",
        },
        NEMOTRON_ADAPTER: set(NemotronConfig().as_dict()),
        SHERPA_ZIPFORMER_ADAPTER: set(SherpaZipformerConfig().as_dict()),
        PARAKEET_REALTIME_EOU_ADAPTER: (
            set(manifest.adapter_config.as_dict())
            if isinstance(manifest.adapter_config, ParakeetRealtimeEouConfig)
            else set()
        ),
        FASTER_WHISPER_ENDPOINT_ADAPTER: (
            set(manifest.adapter_config.as_dict())
            if isinstance(manifest.adapter_config, FasterWhisperEndpointConfig)
            else set()
        ),
    }[manifest.adapter]
    if set(adapter_config) != expected_config_fields:
        raise ValueError
    _strict_json(adapter_config)
    binding.update(
        {
            "manifest_schema_version": expected_schema,
            "adapter_config": adapter_config,
        }
    )
    if runtime_receipt_sha256 is not None:
        binding["runtime_receipt_sha256"] = runtime_receipt_sha256
    if model_receipt_sha256 is not None:
        binding["model_receipt_sha256"] = model_receipt_sha256
    return binding


def _validated_parakeet_cgroup_evidence(
    manifest: WorkerManifest,
    value: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Bind only schema-v5 reports to the supervisor-verified hard scope."""

    if manifest.adapter != PARAKEET_REALTIME_EOU_ADAPTER:
        return None
    if manifest.schema_version != 5 or not isinstance(value, Mapping):
        raise ValueError
    if set(value) != set(_PARAKEET_CGROUP_EVIDENCE) or any(
        type(value[name]) is not type(expected) or value[name] != expected
        for name, expected in _PARAKEET_CGROUP_EVIDENCE.items()
    ):
        raise ValueError
    result = {name: value[name] for name in _PARAKEET_CGROUP_EVIDENCE}
    _strict_json(result)
    return result


def _worker_report(
    manifest: WorkerManifest,
    worker_binding: Mapping[str, object],
    runtime: Mapping[str, str],
    cgroup_evidence: Mapping[str, object] | None,
) -> dict[str, object]:
    """Add runtime provenance without changing any legacy report shape."""

    report = dict(worker_binding)
    report["runtime"] = dict(runtime)
    cgroup = _validated_parakeet_cgroup_evidence(manifest, cgroup_evidence)
    if cgroup is not None:
        report["cgroup_evidence"] = cgroup
    return report


def _evidence_binding(
    adapter: str,
    *,
    pace: str,
    adapter_config: (
        FasterWhisperEndpointConfig
        | NemotronConfig
        | SherpaZipformerConfig
        | ParakeetRealtimeEouConfig
        | None
    ) = None,
) -> dict[str, object]:
    if adapter not in {
        _FAKE_ADAPTER,
        FASTER_WHISPER_ENDPOINT_ADAPTER,
        MOONSHINE_ADAPTER,
        NEMOTRON_ADAPTER,
        PARAKEET_REALTIME_EOU_ADAPTER,
        SHERPA_ZIPFORMER_ADAPTER,
    } or pace not in {
        "burst",
        "realtime",
    }:
        raise ValueError
    if adapter == NEMOTRON_ADAPTER:
        if not isinstance(adapter_config, NemotronConfig):
            raise ValueError
        config = adapter_config
        zipformer_config = None
        parakeet_config = None
        faster_whisper_config = None
    elif adapter == SHERPA_ZIPFORMER_ADAPTER:
        if not isinstance(adapter_config, SherpaZipformerConfig):
            raise ValueError
        config = None
        zipformer_config = adapter_config
        parakeet_config = None
        faster_whisper_config = None
    elif adapter == PARAKEET_REALTIME_EOU_ADAPTER:
        if not isinstance(adapter_config, ParakeetRealtimeEouConfig):
            raise ValueError
        config = None
        zipformer_config = None
        parakeet_config = adapter_config
        faster_whisper_config = None
    elif adapter == FASTER_WHISPER_ENDPOINT_ADAPTER:
        if not isinstance(adapter_config, FasterWhisperEndpointConfig):
            raise ValueError
        config = None
        zipformer_config = None
        parakeet_config = None
        faster_whisper_config = adapter_config
    elif adapter_config is not None:
        raise ValueError
    else:
        config = None
        zipformer_config = None
        parakeet_config = None
        faster_whisper_config = None
    candidate_executed = adapter != _FAKE_ADAPTER
    binding: dict[str, object] = {
        "kind": "bounded_pcm_streaming_harness",
        "fake_adapter": adapter == _FAKE_ADAPTER,
        "production_model": adapter == SHERPA_ZIPFORMER_ADAPTER,
        "real_candidate_model": adapter
        in {
            FASTER_WHISPER_ENDPOINT_ADAPTER,
            MOONSHINE_ADAPTER,
            NEMOTRON_ADAPTER,
            PARAKEET_REALTIME_EOU_ADAPTER,
        },
        "model_executed": candidate_executed,
        "replay_pace": pace,
        "accelerated_replay": pace == "burst",
        "conversational_latency_valid": False,
        "latency_evidence": (
            "paced_replay_only" if pace == "realtime" else "accelerated_replay_only"
        ),
        "timing_scope": "adapter_after_pcm_load_to_result",
        "compute_rtf_scope": (
            "generate_wall_minus_pacing"
            if adapter == NEMOTRON_ADAPTER
            else "candidate_calls_only"
        ),
        "partial_source": (
            "provisional_streamer"
            if adapter == NEMOTRON_ADAPTER
            else "candidate_snapshot"
        ),
        "final_source": (
            "authoritative_generate_sequences"
            if adapter == NEMOTRON_ADAPTER
            else "candidate_final"
        ),
        "model_padding_reported": True,
        "end_to_end_rtf": False,
        "pcm_snapshot_io_included": False,
        "controller_ipc_included": False,
        "capture_to_text_latency": False,
        "vad": False,
        "endpointing": adapter == PARAKEET_REALTIME_EOU_ADAPTER,
        "aec": False,
        "live_hardware": False,
        "adoption_authority": False,
    }
    if faster_whisper_config is not None:
        binding.update(
            {
                "streaming_recognizer": False,
                "partial_source": "none_final_only",
                "final_source": "candidate_decode_after_complete_pcm",
                "external_endpoint_required": True,
                "latency_evidence": "complete_pcm_to_candidate_final_only",
                "finalization_latency_scope": (
                    "complete_pcm_available_to_candidate_final"
                ),
                "streaming_deadline_metrics_applicable": False,
                "wire_deadline_fields": (
                    "numeric_protocol_placeholders_reported_as_null"
                ),
                "tail_padding_policy": faster_whisper_config.tail_padding_policy,
                "decode_contract": {
                    "sample_rate_hz": faster_whisper_config.sample_rate,
                    "language": faster_whisper_config.language,
                    "device": faster_whisper_config.device,
                    "device_index": faster_whisper_config.device_index,
                    "compute_type": faster_whisper_config.compute_type,
                    "cpu_threads": faster_whisper_config.cpu_threads,
                    "num_workers": faster_whisper_config.num_workers,
                    "beam_size": faster_whisper_config.beam_size,
                    "vad_filter": faster_whisper_config.vad_filter,
                    "condition_on_previous_text": (
                        faster_whisper_config.condition_on_previous_text
                    ),
                },
                "resource_control": {
                    "one_thread_environment": True,
                    "network_namespace": "unshared",
                    "runtime_tree_read_only": True,
                    "model_tree_read_only": True,
                    "hard_host_memory_limit": False,
                    "hard_vram_limit": False,
                },
            }
        )
    if config is not None:
        binding.update(
            {
                "capture_accrual_included": False,
                "conversational_latency_scope": (
                    "diagnostic_replay_excludes_native_capture_accrual"
                ),
                "deadline_accounting": "post_consumption_per_native_chunk",
                "native_stream_geometry": {
                    "sample_rate_hz": config.native_sample_rate_hz,
                    "hop_length_samples": config.native_hop_length_samples,
                    "n_fft_samples": config.native_n_fft_samples,
                    "win_length_samples": config.native_win_length_samples,
                    "first_chunk_frames": config.native_first_chunk_frames,
                    "chunk_frames": config.native_chunk_frames,
                    "first_window_samples": config.native_first_window_samples,
                    "window_samples": config.native_window_samples,
                    "stride_samples": config.native_stride_samples,
                    "streaming_latency_ms": config.streaming_latency_ms,
                },
            }
        )
    if zipformer_config is not None:
        binding.update(
            {
                "production_streaming_baseline": True,
                "runtime_equivalent_to_production": False,
                "resource_control": {
                    "benchmark_profile": zipformer_config.benchmark_profile,
                    "benchmark_num_threads": zipformer_config.num_threads,
                    "production_device_profile": (
                        zipformer_config.production_device_profile
                    ),
                    "production_num_threads": (zipformer_config.production_num_threads),
                },
                "decode_contract": {
                    "sample_rate_hz": zipformer_config.sample_rate,
                    "feature_dim": zipformer_config.feature_dim,
                    "provider": zipformer_config.provider,
                    "decoding_method": zipformer_config.decoding_method,
                    "max_active_paths": zipformer_config.max_active_paths,
                    "endpoint_rules": [
                        zipformer_config.rule1_min_trailing_silence,
                        zipformer_config.rule2_min_trailing_silence,
                        zipformer_config.rule3_min_utterance_length,
                    ],
                },
            }
        )
    if parakeet_config is not None:
        binding.update(
            {
                "final_source": (
                    "candidate_native_eou_eob_or_declared_tail_exhaustion"
                ),
                "native_endpoint_source": "candidate_eou_eob",
                "response_authority": "complete_source_native_eou_only",
                "endpoint_stop_policy": (
                    "first_native_endpoint_or_declared_tail_exhaustion"
                ),
                "endpoint_ground_truth": False,
                "endpoint_latency_scope": (
                    "declared_source_end_to_observed_native_chunk_boundary"
                ),
                "endpoint_sample_domain": ("model_input_including_terminal_zero_fill"),
                "source_tail_accounting": "worker_reported_supervisor_validated",
                "capture_accrual_included": False,
                "resource_control": {
                    "kind": "systemd-user-scope-cgroup-v2",
                    "proof_source": "supervisor_cgroup_files_before_ready",
                    "receipt_field": "worker.cgroup_evidence",
                    "hard_host_memory_limit": True,
                    "hard_cpu_quota": True,
                    "hard_tasks_limit": True,
                    "hard_vram_limit": False,
                },
                "native_stream_geometry": {
                    "sample_rate_hz": parakeet_config.sample_rate,
                    "chunk_samples": parakeet_config.native_chunk_samples,
                    "chunk_ms": round(
                        parakeet_config.native_chunk_samples
                        * 1000.0
                        / parakeet_config.sample_rate,
                        3,
                    ),
                    "maximum_tail_padding_samples": (
                        parakeet_config.maximum_tail_padding_samples
                    ),
                },
            }
        )
    return binding


def _mark_endpoint_deadline_metrics_not_applicable(
    metrics: dict[str, object],
) -> None:
    """Remove mandatory wire placeholders from final-only report evidence."""

    streaming = metrics.get("streaming")
    if not isinstance(streaming, dict):
        raise ValueError
    for field_name in (
        "deadline_misses",
        "max_backlog_p95_ms",
        "max_backlog_ms",
    ):
        if field_name not in streaming:
            raise ValueError
        streaming[field_name] = None
    streaming["deadline_metrics_applicable"] = False


def _corpus_binding(corpus: LoadedCorpus) -> dict[str, object]:
    case_set = [
        {
            "sha256": case.sha256,
            "samples": case.samples,
            "assertion": case.assertion,
            "commands": len(case.commands),
            "tags": list(case.tags),
        }
        for case in corpus.cases
    ]
    binding: dict[str, object] = {
        "manifest_sha256": corpus.digest,
        "case_set_sha256": _canonical_digest(case_set),
        "cases": len(corpus.cases),
        "audio_bytes": corpus.audio_bytes,
        "purpose_sha256": hashlib.sha256(corpus.purpose.encode("utf-8")).hexdigest(),
    }
    if corpus.schema_version == 1:
        if corpus.provenance is not None:
            raise ValueError
        return binding
    if corpus.schema_version != 2 or corpus.provenance is None:
        raise ValueError
    binding.update(
        {
            "schema_version": 2,
            "provenance": corpus.provenance.as_dict(),
        }
    )
    return binding


def _prepare_scratch(parent: Path) -> Path:
    try:
        with opened_directory_nofollow(
            Path(os.path.abspath(parent)),
            create=True,
            require_private=True,
        ) as (stable_parent, descriptor):
            for _attempt in range(32):
                name = f".streaming-stt-{secrets.token_hex(16)}"
                try:
                    os.mkdir(name, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    continue
                path = stable_parent / name
                with opened_directory_nofollow(
                    path,
                    require_private=True,
                ):
                    pass
                return path
    except (OSError, BoundedReadError):
        raise ValueError from None
    raise ValueError


def _chmod_owned_directory_at(parent_descriptor: int, name: str) -> None:
    metadata = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(metadata.st_mode) or (
        hasattr(os, "geteuid") and metadata.st_uid != os.geteuid()
    ):
        raise ValueError
    os.chmod(
        name,
        0o700,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    current = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        current.st_dev,
        current.st_ino,
        stat.S_IFMT(current.st_mode),
    ) != (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    ):
        raise ValueError


def _unlock_private_directory(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or (
        hasattr(os, "geteuid") and metadata.st_uid != os.geteuid()
    ):
        raise ValueError
    os.fchmod(descriptor, 0o700)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for name in os.listdir(descriptor):
        child_metadata = os.stat(
            name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(child_metadata.st_mode):
            continue
        _chmod_owned_directory_at(descriptor, name)
        child = os.open(name, flags, dir_fd=descriptor)
        try:
            opened = os.fstat(child)
            if (
                opened.st_dev,
                opened.st_ino,
                stat.S_IFMT(opened.st_mode),
            ) != (
                child_metadata.st_dev,
                child_metadata.st_ino,
                stat.S_IFMT(child_metadata.st_mode),
            ):
                raise ValueError
            _unlock_private_directory(child)
        finally:
            os.close(child)


def _remove_private_scratch(
    root: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Remove private scratch without chmod-following links or hard-linked files."""

    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise ValueError
    root = Path(os.path.abspath(root))
    parent = root.parent
    try:
        with opened_directory_nofollow(parent) as (_parent, parent_descriptor):
            initial = os.stat(
                root.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (initial.st_dev, initial.st_ino) != expected_identity:
                raise ValueError
            _chmod_owned_directory_at(parent_descriptor, root.name)
        with opened_directory_nofollow(root) as (_stable_root, root_descriptor):
            root_identity = (
                os.fstat(root_descriptor).st_dev,
                os.fstat(root_descriptor).st_ino,
            )
            _unlock_private_directory(root_descriptor)
            current = root.lstat()
            if (current.st_dev, current.st_ino) != root_identity:
                raise ValueError
    except (OSError, BoundedReadError):
        raise ValueError from None
    shutil.rmtree(root)
    if root.exists() or root.is_symlink():
        raise ValueError


def _validated_stream(
    stream: StreamConfig,
    *,
    protocol_version: int,
) -> StreamConfig:
    probe = TranscribeRequest(
        request_id="stream-validation",
        pcm=PcmInput(
            path=(Path.cwd() / ".stream-validation.f32le").resolve(),
            sha256="0" * 64,
            samples=1,
        ),
        stream=stream,
        protocol_version=protocol_version,
    )
    try:
        parsed = parse_request(encode_message(probe.as_dict()))
    except Exception:
        raise ValueError from None
    if not isinstance(parsed, TranscribeRequest):
        raise ValueError
    return parsed.stream


def _default_stream(manifest: WorkerManifest) -> StreamConfig:
    if manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
        config = manifest.adapter_config
        if not isinstance(config, ParakeetRealtimeEouConfig):
            raise ValueError
        return StreamConfig(
            chunk_samples=config.native_chunk_samples,
            pace="burst",
            partial_interval_ms=(
                config.native_chunk_samples * 1000 + config.sample_rate - 1
            )
            // config.sample_rate,
            tail_padding_samples=config.maximum_tail_padding_samples,
        )
    if manifest.adapter == NEMOTRON_ADAPTER:
        config = manifest.adapter_config
        if not isinstance(config, NemotronConfig):
            raise ValueError
        chunk_samples = config.native_stride_samples
        partial_interval_ms = config.streaming_latency_ms
        if (
            isinstance(chunk_samples, bool)
            or not isinstance(chunk_samples, int)
            or isinstance(partial_interval_ms, bool)
            or not isinstance(partial_interval_ms, int)
        ):
            raise ValueError
        return StreamConfig(
            chunk_samples=chunk_samples,
            pace="burst",
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=0,
        )
    if manifest.adapter not in {
        _FAKE_ADAPTER,
        FASTER_WHISPER_ENDPOINT_ADAPTER,
        MOONSHINE_ADAPTER,
        SHERPA_ZIPFORMER_ADAPTER,
    }:
        raise ValueError
    return StreamConfig(
        chunk_samples=1600,
        pace="burst",
        partial_interval_ms=200,
        tail_padding_samples=0,
    )


def _selected_stream(
    manifest: WorkerManifest,
    stream: StreamConfig | None,
    *,
    chunk_samples: int | None,
    pace: str | None,
    partial_interval_ms: int | None,
    tail_padding_samples: int | None,
) -> StreamConfig:
    protocol_version = (
        NATIVE_ENDPOINT_PROTOCOL_VERSION
        if manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER
        else PROTOCOL_VERSION
    )
    overrides = (
        chunk_samples,
        pace,
        partial_interval_ms,
        tail_padding_samples,
    )
    if stream is not None:
        if any(value is not None for value in overrides):
            raise ValueError
        selected = _validated_stream(
            stream,
            protocol_version=protocol_version,
        )
    else:
        default = _default_stream(manifest)
        selected = _validated_stream(
            StreamConfig(
                chunk_samples=(
                    default.chunk_samples if chunk_samples is None else chunk_samples
                ),
                pace=default.pace if pace is None else pace,
                partial_interval_ms=(
                    default.partial_interval_ms
                    if partial_interval_ms is None
                    else partial_interval_ms
                ),
                tail_padding_samples=(
                    default.tail_padding_samples
                    if tail_padding_samples is None
                    else tail_padding_samples
                ),
            ),
            protocol_version=protocol_version,
        )
    if manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER:
        config = manifest.adapter_config
        if (
            not isinstance(config, ParakeetRealtimeEouConfig)
            or selected.chunk_samples != config.native_chunk_samples
            or selected.tail_padding_samples > config.maximum_tail_padding_samples
        ):
            raise ValueError
    return selected


def _write_pcm_snapshot(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def run_benchmark(
    worker_manifest_path: Path | str,
    corpus_path: Path | str,
    *,
    scratch_parent: Path | str,
    repeats: int = 3,
    stream: StreamConfig | None = None,
    chunk_samples: int | None = None,
    pace: str | None = None,
    partial_interval_ms: int | None = None,
    tail_padding_samples: int | None = None,
) -> dict[str, object]:
    """Run one exact manifest-selected worker/corpus tuple without transcript rows."""

    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or not 1 <= repeats <= 8
    ):
        raise ValueError
    overrides = (
        chunk_samples,
        pace,
        partial_interval_ms,
        tail_padding_samples,
    )
    if stream is not None:
        if any(value is not None for value in overrides):
            raise ValueError
        # Reject geometry that is invalid under every supported protocol before
        # touching caller-supplied inputs. The exact v2/v3 bound is applied
        # after the manifest selects the worker protocol.
        _validated_stream(
            stream,
            protocol_version=NATIVE_ENDPOINT_PROTOCOL_VERSION,
        )
    run_lock = _BenchmarkRunLock.acquire(_HOST_LOCK_PATH)
    try:
        run_lock.assert_bound()
        result = _run_benchmark_locked(
            worker_manifest_path,
            corpus_path,
            scratch_parent=Path(scratch_parent).expanduser(),
            repeats=repeats,
            stream=stream,
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=tail_padding_samples,
        )
        run_lock.assert_bound()
        return result
    finally:
        run_lock.close()


def _run_benchmark_locked(
    worker_manifest_path: Path | str,
    corpus_path: Path | str,
    *,
    scratch_parent: Path,
    repeats: int,
    stream: StreamConfig | None,
    chunk_samples: int | None,
    pace: str | None,
    partial_interval_ms: int | None,
    tail_padding_samples: int | None,
) -> dict[str, object]:
    manifest = load_worker_manifest(worker_manifest_path)
    selected_stream = _selected_stream(
        manifest,
        stream,
        chunk_samples=chunk_samples,
        pace=pace,
        partial_interval_ms=partial_interval_ms,
        tail_padding_samples=tail_padding_samples,
    )
    try:
        if manifest.worker.path.resolve(strict=True) != _FIXED_WORKER:
            raise ValueError
    except (OSError, RuntimeError):
        raise ValueError from None
    runtime_receipt_sha256 = _verified_runtime_receipt_digest(manifest)
    model_receipt_sha256 = _verified_model_receipt_digest(manifest)
    corpus = load_corpus(corpus_path)
    evaluator_binding = _evaluator_binding(manifest.adapter)
    corpus_binding = _corpus_binding(corpus)
    scratch = _prepare_scratch(scratch_parent)
    scratch_metadata = scratch.lstat()
    scratch_identity = (scratch_metadata.st_dev, scratch_metadata.st_ino)
    records: list[RunRecord] = []
    stderr_summary: Mapping[str, object] = {}
    worker: StreamingWorker | None = None
    source_bundle: SourceBundle | None = None
    try:
        source_bundle = stage_worker_source_bundle(
            _REPO_ROOT,
            scratch / "source-bundle",
        )
        worker_binding = _worker_binding(
            manifest,
            source_bundle,
            runtime_receipt_sha256=runtime_receipt_sha256,
            model_receipt_sha256=model_receipt_sha256,
        )
        snapshots: list[Path] = []
        for index, case in enumerate(corpus.cases):
            path = scratch / f"case-{index:04d}.f32le"
            _write_pcm_snapshot(path, case.audio_bytes)
            if (
                hash_regular_bounded(
                    path,
                    maximum_bytes=MAX_PCM_BYTES,
                    expected_bytes=len(case.audio_bytes),
                ).sha256
                != case.sha256
            ):
                raise ValueError
            snapshots.append(path)

        worker = StreamingWorker(manifest, scratch, source_bundle)
        with worker:
            assert worker.ready is not None
            for repeat in range(repeats):
                for case_index, (case, pcm_path) in enumerate(
                    zip(corpus.cases, snapshots)
                ):
                    request = TranscribeRequest(
                        request_id=f"case-{case_index:04d}-r{repeat}",
                        pcm=PcmInput(
                            path=pcm_path.resolve(strict=True),
                            sha256=case.sha256,
                            samples=case.samples,
                        ),
                        stream=selected_stream,
                        protocol_version=(
                            NATIVE_ENDPOINT_PROTOCOL_VERSION
                            if manifest.adapter == PARAKEET_REALTIME_EOU_ADAPTER
                            else PROTOCOL_VERSION
                        ),
                    )
                    records.append(
                        RunRecord(
                            case_index=case_index,
                            repeat=repeat,
                            trace=worker.transcribe(request),
                        )
                    )
            ready = worker.ready
        if manifest.adapter in {
            FASTER_WHISPER_ENDPOINT_ADAPTER,
            NEMOTRON_ADAPTER,
            PARAKEET_REALTIME_EOU_ADAPTER,
        }:
            config = manifest.adapter_config
            if (
                not isinstance(
                    config,
                    (
                        FasterWhisperEndpointConfig,
                        NemotronConfig,
                        ParakeetRealtimeEouConfig,
                    ),
                )
                or ready.runtime.get("python") != config.python_version
            ):
                raise ValueError
        stderr_summary = worker.stderr_summary
        verify_corpus_snapshot(corpus)
        verify_source_bundle(
            source_bundle,
            required_files=WORKER_SOURCE_FILES,
        )
        current_manifest = load_worker_manifest(manifest.path)
        current_runtime_receipt_sha256 = _verified_runtime_receipt_digest(
            current_manifest
        )
        current_model_receipt_sha256 = _verified_model_receipt_digest(current_manifest)
        if (
            current_manifest.digest != manifest.digest
            or current_manifest.worker.path.resolve(strict=True) != _FIXED_WORKER
            or current_runtime_receipt_sha256 != runtime_receipt_sha256
            or current_model_receipt_sha256 != model_receipt_sha256
            or _worker_binding(
                current_manifest,
                source_bundle,
                runtime_receipt_sha256=current_runtime_receipt_sha256,
                model_receipt_sha256=current_model_receipt_sha256,
            )
            != worker_binding
            or _evaluator_binding(current_manifest.adapter) != evaluator_binding
        ):
            raise ValueError
        metrics = aggregate_metrics(corpus, records, ready, repeats=repeats)
        if manifest.adapter == FASTER_WHISPER_ENDPOINT_ADAPTER:
            _mark_endpoint_deadline_metrics_not_applicable(metrics)
        worker_report = _worker_report(
            manifest,
            worker_binding,
            ready.runtime,
            worker.cgroup_evidence,
        )
        result = {
            "ok": bool(metrics["coverage_complete"]),
            "evidence": _evidence_binding(
                manifest.adapter,
                pace=selected_stream.pace,
                adapter_config=(
                    manifest.adapter_config
                    if isinstance(
                        manifest.adapter_config,
                        (
                            NemotronConfig,
                            SherpaZipformerConfig,
                            ParakeetRealtimeEouConfig,
                            FasterWhisperEndpointConfig,
                        ),
                    )
                    else None
                ),
            ),
            "worker": worker_report,
            "corpus": corpus_binding,
            "config": {
                **selected_stream.as_dict(),
                "repeats": repeats,
                "contract_sha256": _canonical_digest(
                    {
                        **selected_stream.as_dict(),
                        "repeats": repeats,
                    }
                ),
            },
            "metrics": metrics,
            "worker_stderr": dict(stderr_summary),
            "evaluator": evaluator_binding,
        }
        _strict_json(result)
        return result
    finally:
        if worker is not None and not worker.fully_stopped:
            raise RuntimeError
        _remove_private_scratch(
            scratch,
            expected_identity=scratch_identity,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate-only isolated streaming-STT harness for an exact "
            "manifest-selected adapter, runtime, artifacts, and corpus."
        )
    )
    parser.add_argument("--worker-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=_DEFAULT_SCRATCH)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-samples", type=int)
    parser.add_argument("--pace", choices=("burst", "realtime"))
    parser.add_argument("--partial-interval-ms", type=int)
    parser.add_argument("--tail-padding-samples", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run_benchmark(
            args.worker_manifest,
            args.corpus,
            scratch_parent=args.scratch_root,
            repeats=args.repeats,
            chunk_samples=args.chunk_samples,
            pace=args.pace,
            partial_interval_ms=args.partial_interval_ms,
            tail_padding_samples=args.tail_padding_samples,
        )
        if args.output is not None:
            _write_strict_report(args.output, payload)
        print(_strict_json(payload))
        return 0 if payload["ok"] else 1
    except Exception:  # noqa: BLE001 - private/native details never reach the CLI
        print(_strict_json(_SAFE_ERROR))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
