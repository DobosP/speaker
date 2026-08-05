"""Aggregate-only evaluation of Speaker's configured production final STT path.

The evaluator deliberately starts after PCM is available.  It reuses
``FileReplayEngine`` so streaming endpoint decisions, the configured offline
recognizer, verifier consensus, and final selection are the same code used by
the local voice runtime.  It does not construct the chatbot, TTS, tools, an
audio device, capture/VAD plumbing, or a live session.

FileReplay cannot supply VAD-owned speech duration or the live capture path's
offline-recovery authorization.  Its selected view is therefore named and
reported as a FileReplay diagnostic, never as complete live-selection evidence.

Only aggregate counters and cryptographic identities may leave this process.
Raw references, hypotheses, PCM, case identifiers, and filesystem paths stay
in memory and are excluded from exception output and reports.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from importlib import metadata as importlib_metadata
import json
import logging
import os
import re
import secrets
import stat
import sys
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping, Sequence

from core.config import (
    apply_device_profile,
    apply_final_stt_profile,
    deep_merge,
    resolve_device,
)
from core.engine import EngineCallbacks
from core.engines.file_replay import FileReplayEngine
from core.engines.sherpa import SherpaConfig
from tools.prepare_public_command_noise_corpus import (
    LOCK_PATH as COMMAND_NOISE_LOCK_PATH,
    _has_git_ancestor,
    _load_recipe_lock,
)
from tools.recorded_stt_eval import (
    _VERIFIER_COMPLETED_OUTCOMES,
    _config_digest,
    _model_digest,
)
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)
from tools.streaming_stt.corpus import (
    LoadedCorpus,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.streaming_stt.final_metrics import (
    FinalDecisionRecord,
    FinalMetricsAccumulator,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAFE_ERROR = {
    "error": "production_final_stt_prerequisites_unavailable",
    "ok": False,
}
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_RECEIPT_BYTES = 256 * 1024
_SAMPLE_RATE_HZ = 16_000
_EXPECTED_CORPUS_SHA256 = (
    "50ec2c1ec7ad7ef2ca58d56d3330ca1a496db3784b78d41be9c8804413c552b9"
)
_EXPECTED_RECEIPT_SHA256 = (
    "bf6cf442e4fb02e489dc7ceef763df3b39c1a4ef2178a5f8f2d2dab664517ac2"
)
_EXPECTED_SOURCE_SET_SHA256 = (
    "3f74cabd5b827ef2c4ef7afc628b3ce0eb5e237fe672d1555b1357940566d246"
)
_EXPECTED_PCM_SET_SHA256 = (
    "b0eeb32c314b5d9b4078fa640331378f8d9e31372436aca178dfdef4c7fbadd4"
)
_EXPECTED_PURPOSE = (
    "Development-only isolated command and noisy word-confusion evaluation; "
    "not capture, VAD/endpoint, AEC, barge-in, speaker identity, tools, live "
    "latency, training-disjoint selection, or runtime-default evidence."
)
_EXPECTED_ASSERTIONS = {
    "transcript": 47,
    "speech_negative": 5,
    "silence": 5,
}
_EXPECTED_TAGS = {
    "command-negative": 26,
    "command-positive": 26,
    "eccc": 27,
    "gsc": 30,
    "noisy": 27,
    "public-command-noise": 57,
    "silence": 5,
    "speech-negative": 5,
}
_EXPECTED_RECEIPT_KEYS = {
    "accepted_terms",
    "audio",
    "counts",
    "kind",
    "production_sources",
    "recipe_lock_sha256",
    "recipe_sha256",
    "schema_version",
    "source_aggregates",
    "source_set_sha256",
}
_EXPECTED_COUNT_SHAPE = {
    "sources": 2,
    "cases": 57,
    "command_positive": 26,
    "command_negative": 26,
    "assertions": _EXPECTED_ASSERTIONS,
}
_SAFE_PROVIDERS = frozenset({"cpu"})
_SAFE_FINAL_BACKENDS = frozenset({"sense_voice", "whisper", "nemo_transducer"})
_SAFE_VERIFIER_BACKENDS = frozenset({"faster_whisper"})
_BASELINE_FINAL_STT_PROFILE = "sense-voice"
_CANDIDATE_FINAL_STT_PROFILE = "parakeet-faster-whisper"
_SAFE_OFFLINE_OUTCOMES = frozenset(
    {"unavailable", "skipped", "error", "decoded", "empty"}
)
_SAFE_VERIFIER_OUTCOMES = frozenset(
    {
        "unavailable",
        "skipped",
        "error",
        *_VERIFIER_COMPLETED_OUTCOMES,
    }
)
_SAFE_SELECTED_SOURCES = frozenset(
    {"none", "streaming", "offline", "verifier_consensus", "established_override"}
)
_SAFE_RUNTIME_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}\Z")
_RUNTIME_DISTRIBUTIONS = (
    ("numpy", "numpy"),
    ("sherpa_onnx", "sherpa-onnx"),
    ("onnxruntime", "onnxruntime"),
    ("faster_whisper", "faster-whisper"),
    ("ctranslate2", "ctranslate2"),
)
_EVALUATOR_SOURCE_FILES = (
    "always_on_agent/acoustic.py",
    "always_on_agent/events.py",
    "always_on_agent/models.py",
    "always_on_agent/speech_analyzer.py",
    "always_on_agent/text.py",
    "core/asr_text.py",
    "core/asr_verifier.py",
    "core/audio_frontend.py",
    "core/config.py",
    "core/contract.py",
    "core/engine.py",
    "core/engines/_acoustic_turn.py",
    "core/engines/_cuda_wheels.py",
    "core/engines/_faster_whisper.py",
    "core/engines/_sherpa_models.py",
    "core/engines/file_replay.py",
    "core/engines/sherpa.py",
    "core/tts_markup.py",
    "core/wer.py",
    "tools/prepare_public_command_noise_corpus.py",
    "tools/production_final_stt_eval.py",
    "tools/recorded_stt_eval.py",
    "tools/streaming_stt/bounded_io.py",
    "tools/streaming_stt/corpus.py",
    "tools/streaming_stt/final_metrics.py",
    "tools/streaming_stt/metrics.py",
    "tools/streaming_stt/public-command-noise-v1.lock.json",
)


class ProductionFinalEvaluationError(RuntimeError):
    """A detail-free corpus, configuration, execution, or output failure."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class _ConfigInputs:
    root: Path = field(repr=False)
    config_path: Path = field(repr=False)
    local_path: Path = field(repr=False)
    config_bytes: bytes = field(repr=False)
    local_bytes: bytes = field(repr=False)
    digest: str


@dataclass(frozen=True)
class _Execution:
    streaming: Mapping[str, object]
    selected: Mapping[str, object]
    decisions: int
    offline_outcomes: Mapping[str, int]
    verifier_outcomes: Mapping[str, int]
    selected_sources: Mapping[str, int]


@dataclass(frozen=True)
class _BoundProfile:
    name: str
    profile_sha256: str
    profile_schema_version: int
    config: SherpaConfig = field(repr=False)
    config_sha256: str
    model_sha256: str


@dataclass(frozen=True)
class _BoundProfilePair:
    baseline: _BoundProfile
    candidate: _BoundProfile
    device_profile: str
    evaluator_source_sha256: str
    runtime_identities: Mapping[str, object] = field(repr=False)


class _DiscardText(io.TextIOBase):
    """Write-only sink which never retains model logs or transcript text."""

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


@contextmanager
def _quiet_model_output():
    """Suppress Python and native-fd output while hypotheses are private."""

    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    sink = _DiscardText()
    null_fd = -1
    saved_stdout_fd = -1
    saved_stderr_fd = -1
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        null_fd = os.open(os.devnull, os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        with redirect_stdout(sink), redirect_stderr(sink):
            yield
    finally:
        if saved_stdout_fd >= 0:
            try:
                os.dup2(saved_stdout_fd, 1)
            finally:
                os.close(saved_stdout_fd)
        if saved_stderr_fd >= 0:
            try:
                os.dup2(saved_stderr_fd, 2)
            finally:
                os.close(saved_stderr_fd)
        if null_fd >= 0:
            os.close(null_fd)
        logging.disable(previous_disable)


def _strict_json(raw: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ProductionFinalEvaluationError()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ProductionFinalEvaluationError()
            ),
        )
    except ProductionFinalEvaluationError:
        raise
    except (OverflowError, UnicodeError, ValueError):
        raise ProductionFinalEvaluationError() from None


def _digest_labeled_bytes(items: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for label, raw in items:
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _load_config_inputs(config_path: Path | str, local_path: Path | str) -> _ConfigInputs:
    config_candidate = Path(config_path).expanduser()
    local_candidate = Path(local_path).expanduser()
    if not config_candidate.is_absolute() or not local_candidate.is_absolute():
        raise ProductionFinalEvaluationError()
    try:
        config = read_regular_bounded(config_candidate, maximum_bytes=_MAX_CONFIG_BYTES)
        local = read_regular_bounded(local_candidate, maximum_bytes=_MAX_CONFIG_BYTES)
        root = config.path.parent
        if local.path.parent != root:
            raise ProductionFinalEvaluationError()
    except BoundedReadError:
        raise ProductionFinalEvaluationError() from None
    return _ConfigInputs(
        root=root,
        config_path=config.path,
        local_path=local.path,
        config_bytes=config.data,
        local_bytes=local.data,
        digest=_digest_labeled_bytes(
            (("config", config.data), ("local", local.data))
        ),
    )


def _verify_config_inputs(snapshot: _ConfigInputs) -> None:
    current = _load_config_inputs(snapshot.config_path, snapshot.local_path)
    if (
        current.root != snapshot.root
        or current.config_path != snapshot.config_path
        or current.local_path != snapshot.local_path
        or current.digest != snapshot.digest
    ):
        raise ProductionFinalEvaluationError()


def _effective_config_mapping(
    inputs: _ConfigInputs,
    device: str | None,
) -> tuple[dict[str, object], str]:
    base = _strict_json(inputs.config_bytes)
    local = _strict_json(inputs.local_bytes)
    if not isinstance(base, dict) or not isinstance(local, dict):
        raise ProductionFinalEvaluationError()
    merged = deep_merge(base, local)
    selected_device = device if device is not None else merged.get("device", "desktop")
    resolved_device, _rationale = resolve_device(merged, selected_device)
    if (
        not isinstance(resolved_device, str)
        or _SAFE_RUNTIME_LABEL_RE.fullmatch(resolved_device) is None
    ):
        raise ProductionFinalEvaluationError()
    effective = apply_device_profile(merged, resolved_device, strict=True)
    if not isinstance(effective, dict):
        raise ProductionFinalEvaluationError()
    return effective, resolved_device


def _sherpa_config_from_effective(
    effective: Mapping[str, object],
    *,
    allow_disabled_verifier: bool,
) -> SherpaConfig:
    sherpa = effective.get("sherpa")
    if not isinstance(sherpa, dict):
        raise ProductionFinalEvaluationError()
    result = SherpaConfig.from_dict(sherpa)
    verifier_backend = str(result.asr_final_verifier_backend or "").strip().lower()
    if (
        result.sample_rate != _SAMPLE_RATE_HZ
        or not str(result.asr_encoder or "").strip()
        or str(result.provider or "").strip().lower() not in _SAFE_PROVIDERS
        or str(result.asr_final_backend or "").strip().lower()
        not in _SAFE_FINAL_BACKENDS
        or (
            verifier_backend not in _SAFE_VERIFIER_BACKENDS
            and not (allow_disabled_verifier and verifier_backend == "")
        )
    ):
        raise ProductionFinalEvaluationError()
    return result


def _effective_config_with_profile(
    inputs: _ConfigInputs,
    device: str | None,
) -> tuple[SherpaConfig, str]:
    effective, resolved_device = _effective_config_mapping(inputs, device)
    return (
        _sherpa_config_from_effective(
            effective,
            allow_disabled_verifier=False,
        ),
        resolved_device,
    )


def _effective_config(inputs: _ConfigInputs, device: str | None) -> SherpaConfig:
    """Compatibility wrapper used by focused configuration tests."""

    return _effective_config_with_profile(inputs, device)[0]


def _profile_binding(metadata: object, expected_name: str) -> tuple[str, str, int]:
    try:
        name = getattr(metadata, "name")
        digest = getattr(metadata, "sha256")
        schema_version = getattr(metadata, "schema_version")
        if (
            type(expected_name) is not str
            or expected_name
            not in {_BASELINE_FINAL_STT_PROFILE, _CANDIDATE_FINAL_STT_PROFILE}
            or type(name) is not str
            or name != expected_name
            or not _is_sha256(digest)
            or type(schema_version) is not int
            or schema_version != 1
        ):
            raise ProductionFinalEvaluationError()
        return name, digest, schema_version
    except ProductionFinalEvaluationError:
        raise
    except Exception:
        raise ProductionFinalEvaluationError() from None


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _configured_model_digest(config: SherpaConfig, root: Path) -> str:
    try:
        with _working_directory(root):
            return _model_digest(config)
    except Exception:
        raise ProductionFinalEvaluationError() from None


def _bind_final_stt_profile_pair(
    inputs: _ConfigInputs,
    device: str | None,
    baseline_name: str,
    candidate_name: str,
) -> _BoundProfilePair:
    """Resolve and hash the fixed pair before either replay cell can start."""

    if (
        type(baseline_name) is not str
        or type(candidate_name) is not str
        or baseline_name != _BASELINE_FINAL_STT_PROFILE
        or candidate_name != _CANDIDATE_FINAL_STT_PROFILE
        or baseline_name == candidate_name
    ):
        raise ProductionFinalEvaluationError()
    try:
        effective, resolved_device = _effective_config_mapping(inputs, device)
        baseline_raw, baseline_metadata = apply_final_stt_profile(
            effective,
            baseline_name,
        )
        candidate_raw, candidate_metadata = apply_final_stt_profile(
            effective,
            candidate_name,
        )
        baseline_binding = _profile_binding(baseline_metadata, baseline_name)
        candidate_binding = _profile_binding(candidate_metadata, candidate_name)
        baseline_config = _sherpa_config_from_effective(
            baseline_raw,
            allow_disabled_verifier=True,
        )
        candidate_config = _sherpa_config_from_effective(
            candidate_raw,
            allow_disabled_verifier=True,
        )
        for config_field in fields(SherpaConfig):
            if config_field.name.startswith("asr_final_"):
                continue
            if getattr(baseline_config, config_field.name) != getattr(
                candidate_config,
                config_field.name,
            ):
                raise ProductionFinalEvaluationError()

        # Keep this ordering explicit: candidate artifacts must be valid and
        # bound before the first baseline sample reaches FileReplay.
        baseline_config_sha256 = _config_digest(baseline_config)
        baseline_model_sha256 = _configured_model_digest(
            baseline_config,
            inputs.root,
        )
        candidate_config_sha256 = _config_digest(candidate_config)
        candidate_model_sha256 = _configured_model_digest(
            candidate_config,
            inputs.root,
        )
        evaluator_source_sha256 = _source_digest()
        runtime_identities = _runtime_identities()
        return _BoundProfilePair(
            baseline=_BoundProfile(
                name=baseline_binding[0],
                profile_sha256=baseline_binding[1],
                profile_schema_version=baseline_binding[2],
                config=baseline_config,
                config_sha256=baseline_config_sha256,
                model_sha256=baseline_model_sha256,
            ),
            candidate=_BoundProfile(
                name=candidate_binding[0],
                profile_sha256=candidate_binding[1],
                profile_schema_version=candidate_binding[2],
                config=candidate_config,
                config_sha256=candidate_config_sha256,
                model_sha256=candidate_model_sha256,
            ),
            device_profile=resolved_device,
            evaluator_source_sha256=evaluator_source_sha256,
            runtime_identities=runtime_identities,
        )
    except ProductionFinalEvaluationError:
        raise
    except Exception:
        raise ProductionFinalEvaluationError() from None


def _source_digest() -> str:
    digest = hashlib.sha256()
    try:
        for relative in _EVALUATOR_SOURCE_FILES:
            snapshot = read_regular_bounded(
                _REPO_ROOT / relative,
                maximum_bytes=8 * 1024 * 1024,
            )
            digest.update(relative.encode("ascii"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(hashlib.sha256(snapshot.data).hexdigest()))
        return digest.hexdigest()
    except (BoundedReadError, UnicodeError, ValueError):
        raise ProductionFinalEvaluationError() from None


def _runtime_identities() -> dict[str, object]:
    """Bind the interpreter and installed native/model Python distributions."""

    try:
        executable = Path(sys.executable).resolve(strict=True)
        executable_snapshot = read_regular_bounded(
            executable,
            maximum_bytes=128 * 1024 * 1024,
        )
        distributions: dict[str, object] = {}
        for label, distribution_name in _RUNTIME_DISTRIBUTIONS:
            distribution = importlib_metadata.distribution(distribution_name)
            version = distribution.version
            record = distribution.read_text("RECORD")
            if (
                not isinstance(version, str)
                or _SAFE_RUNTIME_LABEL_RE.fullmatch(version) is None
                or not isinstance(record, str)
                or not record
                or len(record.encode("utf-8")) > 2 * 1024 * 1024
            ):
                raise ProductionFinalEvaluationError()
            distributions[label] = {
                "version": version,
                "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
            }
        return {
            "python": {
                "version": (
                    f"{sys.version_info.major}.{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
                "executable_sha256": hashlib.sha256(
                    executable_snapshot.data
                ).hexdigest(),
            },
            "distributions": distributions,
        }
    except ProductionFinalEvaluationError:
        raise
    except Exception:
        raise ProductionFinalEvaluationError() from None


def _provider_attestation(config: SherpaConfig) -> dict[str, str]:
    """Fail closed on provider ambiguity for the selected final-STT chain."""

    if str(config.provider or "").strip().lower() != "cpu":
        raise ProductionFinalEvaluationError()
    verifier_backend = str(config.asr_final_verifier_backend or "").strip().lower()
    if verifier_backend not in {"", "faster_whisper"}:
        raise ProductionFinalEvaluationError()
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        if "CPUExecutionProvider" not in providers:
            raise ProductionFinalEvaluationError()
    except ProductionFinalEvaluationError:
        raise
    except Exception:
        raise ProductionFinalEvaluationError() from None
    if not verifier_backend:
        return {
            "streaming_offline_requested": "cpu",
            "streaming_offline_attestation": "explicit_cpu_provider_available",
            "verifier_requested": "disabled",
            "verifier_attestation": "not_configured",
        }
    return {
        "streaming_offline_requested": "cpu",
        "streaming_offline_attestation": "explicit_cpu_provider_available",
        "verifier_requested": "cuda:0-fp16",
        "verifier_attestation": "cuda_only_adapter_completed_decode",
    }


def _require_private_corpus(corpus: LoadedCorpus) -> None:
    """Require the selected manifest, receipt, and PCM to stay owner-private."""

    paths = (
        corpus.path,
        corpus.path.parent / "preparation-receipt.json",
        *(case.source_path for case in corpus.cases),
    )
    try:
        directory = corpus.path.parent.lstat()
        if (
            not stat.S_ISDIR(directory.st_mode)
            or stat.S_IMODE(directory.st_mode) != 0o700
            or corpus.path.parent.resolve(strict=True) != corpus.path.parent
            or (hasattr(os, "geteuid") and directory.st_uid != os.geteuid())
        ):
            raise ProductionFinalEvaluationError()
        for path in paths:
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or path.resolve(strict=True) != path
                or path.parent != corpus.path.parent
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise ProductionFinalEvaluationError()
    except ProductionFinalEvaluationError:
        raise
    except Exception:
        raise ProductionFinalEvaluationError() from None


def _validate_receipt(corpus: LoadedCorpus, *, lock_raw_sha256: str, recipe_sha256: str) -> None:
    try:
        receipt_snapshot = read_regular_bounded(
            corpus.path.parent / "preparation-receipt.json",
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
        value = _strict_json(receipt_snapshot.data)
        if not isinstance(value, dict) or set(value) != _EXPECTED_RECEIPT_KEYS:
            raise ProductionFinalEvaluationError()
        audio = value.get("audio")
        source_aggregates = value.get("source_aggregates")
        if (
            hashlib.sha256(receipt_snapshot.data).hexdigest()
            != _EXPECTED_RECEIPT_SHA256
            or value.get("schema_version") != 1
            or value.get("kind") != "public-command-noise-preparation-receipt-v1"
            or value.get("production_sources") is not True
            or value.get("accepted_terms") != ["CC-BY-4.0"]
            or value.get("recipe_lock_sha256") != lock_raw_sha256
            or value.get("recipe_sha256") != recipe_sha256
            or value.get("source_set_sha256")
            != corpus.provenance.source_set_sha256  # type: ignore[union-attr]
            or value.get("source_set_sha256") != _EXPECTED_SOURCE_SET_SHA256
            or value.get("counts") != _EXPECTED_COUNT_SHAPE
            or not isinstance(audio, dict)
            or set(audio)
            != {"encoding", "sample_rate_hz", "channels", "samples", "pcm_set_sha256"}
            or audio.get("encoding") != "f32le"
            or audio.get("sample_rate_hz") != _SAMPLE_RATE_HZ
            or audio.get("channels") != 1
            or audio.get("samples") != sum(case.samples for case in corpus.cases)
            or not _is_sha256(audio.get("pcm_set_sha256"))
            or audio.get("pcm_set_sha256") != _EXPECTED_PCM_SET_SHA256
            or not isinstance(source_aggregates, list)
            or len(source_aggregates) != 2
            or [item.get("cases") if isinstance(item, dict) else None for item in source_aggregates]
            != [30, 27]
            or any(
                not isinstance(item, dict)
                or set(item) != {"source_id", "cases", "selection_sha256"}
                or not _is_sha256(item.get("selection_sha256"))
                for item in source_aggregates
            )
            or [item["source_id"] for item in source_aggregates]
            != [
                "google-speech-commands-v002-test",
                "english-consistent-confusion-v1.2",
            ]
        ):
            raise ProductionFinalEvaluationError()
    except (BoundedReadError, AttributeError, TypeError):
        raise ProductionFinalEvaluationError() from None


def _validate_exact_corpus(corpus: LoadedCorpus) -> str:
    try:
        recipe = _load_recipe_lock(COMMAND_NOISE_LOCK_PATH)
        provenance = corpus.provenance
        assertions = Counter(case.assertion for case in corpus.cases)
        tags = Counter(tag for case in corpus.cases for tag in case.tags)
        source_assertions = Counter(
            ("gsc" if "gsc" in case.tags else "eccc", case.assertion)
            for case in corpus.cases
        )
        if (
            corpus.schema_version != 4
            or corpus.digest != _EXPECTED_CORPUS_SHA256
            or corpus.purpose != _EXPECTED_PURPOSE
            or provenance is None
            or provenance.kind != "public-command-noise-v1"
            or provenance.suite != "command-noise-v1"
            or provenance.manifest_sha256 != recipe.raw_sha256
            or provenance.metadata_sha256 != _EXPECTED_RECEIPT_SHA256
            or provenance.source_set_sha256 != _EXPECTED_SOURCE_SET_SHA256
            or len(corpus.cases) != 57
            or corpus.audio_bytes != sum(case.samples * 4 for case in corpus.cases)
            or dict(assertions) != _EXPECTED_ASSERTIONS
            or dict(tags) != _EXPECTED_TAGS
            or sum(bool(case.commands) for case in corpus.cases) != 26
            or sum(not case.commands and case.assertion != "silence" for case in corpus.cases)
            != 26
            or sum(case.assertion == "silence" and not case.commands for case in corpus.cases)
            != 5
            or source_assertions
            != Counter(
                {
                    ("gsc", "transcript"): 20,
                    ("gsc", "speech_negative"): 5,
                    ("gsc", "silence"): 5,
                    ("eccc", "transcript"): 27,
                }
            )
            or any(("gsc" in case.tags) == ("eccc" in case.tags) for case in corpus.cases)
        ):
            raise ProductionFinalEvaluationError()
        _require_private_corpus(corpus)
        _validate_receipt(
            corpus,
            lock_raw_sha256=recipe.raw_sha256,
            recipe_sha256=recipe.recipe_sha256,
        )
        return recipe.raw_sha256
    except ProductionFinalEvaluationError:
        raise
    except Exception:
        raise ProductionFinalEvaluationError() from None


def _verify_private_corpus_snapshot(corpus: LoadedCorpus) -> None:
    """Revalidate content, links, ownership, and privacy without leaking detail."""

    try:
        verify_corpus_snapshot(corpus)
        _require_private_corpus(corpus)
    except ProductionFinalEvaluationError:
        raise
    except Exception:
        raise ProductionFinalEvaluationError() from None


def _safe_selected_source(value: object) -> str:
    raw = getattr(value, "value", value)
    if not isinstance(raw, str) or raw not in _SAFE_SELECTED_SOURCES:
        raise ProductionFinalEvaluationError()
    return raw


def _preflight_engine(
    engine: FileReplayEngine,
    *,
    verifier_required: bool = True,
) -> None:
    """Prove the configured final chain exists without inventing a verifier."""

    if (
        getattr(engine, "_tts", None) is not None
        or getattr(engine, "_final_recognizer", None) is None
    ):
        raise ProductionFinalEvaluationError()
    verifier = getattr(engine, "_final_verifier", None)
    if not verifier_required:
        if verifier is not None:
            raise ProductionFinalEvaluationError()
        return
    if verifier is None:
        raise ProductionFinalEvaluationError()
    warm = getattr(verifier, "warm", None)
    if not callable(warm):
        raise ProductionFinalEvaluationError()
    try:
        warm()
    except Exception:
        raise ProductionFinalEvaluationError() from None


def _execute_replay(
    corpus: LoadedCorpus,
    config: SherpaConfig,
    *,
    root: Path,
    repeats: int,
    stratum_tags: Sequence[str],
) -> _Execution:
    import numpy as np

    if not 1 <= repeats <= 3:
        raise ProductionFinalEvaluationError()
    engine = FileReplayEngine(config, asr_only=True)
    streaming_accumulator = FinalMetricsAccumulator(
        corpus,
        repeats,
        stratum_tags=stratum_tags,
    )
    selected_accumulator = FinalMetricsAccumulator(
        corpus,
        repeats,
        stratum_tags=stratum_tags,
    )
    offline_outcomes: Counter[str] = Counter()
    verifier_outcomes: Counter[str] = Counter()
    selected_sources: Counter[str] = Counter()
    decisions_total = 0
    verifier_required = bool(str(config.asr_final_verifier_backend or "").strip())
    try:
        with _working_directory(root), _quiet_model_output():
            try:
                engine.start(EngineCallbacks())
                _preflight_engine(
                    engine,
                    verifier_required=verifier_required,
                )
                for case_index, case in enumerate(corpus.cases):
                    streaming_records: list[FinalDecisionRecord] = []
                    selected_records: list[FinalDecisionRecord] = []
                    samples = np.frombuffer(case.audio_bytes, dtype="<f4").copy()
                    for repeat in range(repeats):
                        result = engine.evaluate_samples(
                            samples,
                            _SAMPLE_RATE_HZ,
                            speech_sec=None,
                        )
                        decisions = tuple(result.decisions)
                        streaming_records.append(
                            FinalDecisionRecord(
                                case_index=case_index,
                                repeat=repeat,
                                decisions=tuple(item.streaming_raw for item in decisions),
                            )
                        )
                        selected_records.append(
                            FinalDecisionRecord(
                                case_index=case_index,
                                repeat=repeat,
                                decisions=tuple(item.selected for item in decisions),
                            )
                        )
                        for item in decisions:
                            if item.offline_outcome not in _SAFE_OFFLINE_OUTCOMES:
                                raise ProductionFinalEvaluationError()
                            if item.verifier_outcome not in _SAFE_VERIFIER_OUTCOMES:
                                raise ProductionFinalEvaluationError()
                            offline_outcomes[item.offline_outcome] += 1
                            verifier_outcomes[item.verifier_outcome] += 1
                            selected_sources[_safe_selected_source(item.selected_source)] += 1
                        decisions_total += len(decisions)
                    streaming_accumulator.add_case(case_index, streaming_records)
                    selected_accumulator.add_case(case_index, selected_records)
            finally:
                engine.stop()
    except ProductionFinalEvaluationError:
        raise
    except Exception:
        raise ProductionFinalEvaluationError() from None
    return _Execution(
        streaming=streaming_accumulator.result(),
        selected=selected_accumulator.result(),
        decisions=decisions_total,
        offline_outcomes=dict(offline_outcomes),
        verifier_outcomes=dict(verifier_outcomes),
        selected_sources=dict(selected_sources),
    )


def _attest_execution(config: SherpaConfig, execution: _Execution) -> None:
    offline = execution.offline_outcomes
    verifier = execution.verifier_outcomes
    verifier_backend = str(config.asr_final_verifier_backend or "").strip().lower()
    invalid = (
        execution.decisions <= 0
        or sum(offline.values()) != execution.decisions
        or sum(verifier.values()) != execution.decisions
        or sum(execution.selected_sources.values()) != execution.decisions
        or int(offline.get("error", 0)) > 0
        or not any(int(offline.get(outcome, 0)) > 0 for outcome in ("decoded", "empty"))
        or int(verifier.get("error", 0)) > 0
        or not str(config.asr_final_backend or "").strip()
        or verifier_backend not in {"", "faster_whisper"}
    )
    if verifier_backend:
        invalid = invalid or not any(
            int(verifier.get(outcome, 0)) > 0
            for outcome in _VERIFIER_COMPLETED_OUTCOMES
        )
    else:
        invalid = invalid or verifier != {"unavailable": execution.decisions}
    if invalid:
        raise ProductionFinalEvaluationError()


def evaluate_production_final(
    corpus: LoadedCorpus,
    config_inputs: _ConfigInputs,
    config: SherpaConfig,
    *,
    repeats: int,
    stratum_tags: Sequence[str],
    device_profile: str = "test-explicit",
    expected_config_sha256: str | None = None,
    expected_model_sha256: str | None = None,
    expected_evaluator_source_sha256: str | None = None,
    expected_runtime_identities: Mapping[str, object] | None = None,
) -> dict[str, object]:
    lock_digest = _validate_exact_corpus(corpus)
    _verify_private_corpus_snapshot(corpus)
    configured_digest = _config_digest(config)
    source_digest = _source_digest()
    runtime_identities = _runtime_identities()
    provider_attestation = _provider_attestation(config)
    model_digest = _configured_model_digest(config, config_inputs.root)
    if (
        (
            expected_config_sha256 is not None
            and configured_digest != expected_config_sha256
        )
        or (expected_model_sha256 is not None and model_digest != expected_model_sha256)
        or (
            expected_evaluator_source_sha256 is not None
            and source_digest != expected_evaluator_source_sha256
        )
        or (
            expected_runtime_identities is not None
            and runtime_identities != expected_runtime_identities
        )
    ):
        raise ProductionFinalEvaluationError()
    execution = _execute_replay(
        corpus,
        config,
        root=config_inputs.root,
        repeats=repeats,
        stratum_tags=stratum_tags,
    )
    _attest_execution(config, execution)
    if (
        execution.streaming["coverage_complete"] is not True
        or execution.selected["coverage_complete"] is not True
        or execution.streaming["decisions"]["zero_decision_evaluations"] != 0
        or execution.selected["decisions"]["zero_decision_evaluations"] != 0
        or execution.decisions < len(corpus.cases) * repeats
    ):
        raise ProductionFinalEvaluationError()
    _verify_private_corpus_snapshot(corpus)
    _verify_config_inputs(config_inputs)
    if (
        _config_digest(config) != configured_digest
        or _configured_model_digest(config, config_inputs.root) != model_digest
        or _source_digest() != source_digest
        or _runtime_identities() != runtime_identities
    ):
        raise ProductionFinalEvaluationError()
    if _SAFE_RUNTIME_LABEL_RE.fullmatch(device_profile) is None:
        raise ProductionFinalEvaluationError()
    result: dict[str, object] = {
        "execution_complete": True,
        "quality_verdict": "diagnostic_only",
        "schema_version": 2,
        "corpus": {
            "schema_version": corpus.schema_version,
            "manifest_sha256": corpus.digest,
            "recipe_lock_sha256": lock_digest,
            "preparation_receipt_sha256": _EXPECTED_RECEIPT_SHA256,
            "source_set_sha256": _EXPECTED_SOURCE_SET_SHA256,
            "pcm_set_sha256": _EXPECTED_PCM_SET_SHA256,
            "cases": len(corpus.cases),
            "audio_bytes": corpus.audio_bytes,
            "audio_seconds": round(
                sum(case.samples for case in corpus.cases) / _SAMPLE_RATE_HZ,
                6,
            ),
        },
        "engine": {
            "effective_config_sha256": configured_digest,
            "model_artifacts_sha256": model_digest,
            "evaluator_source_sha256": source_digest,
            "evaluator_source_files": len(_EVALUATOR_SOURCE_FILES),
            "runtime": runtime_identities,
            "device_profile": device_profile,
            "requested_provider": str(config.provider).strip().lower(),
            "provider_attestation": provider_attestation,
            "asr_final_backend": str(config.asr_final_backend).strip().lower(),
            "asr_final_verifier_backend": str(
                config.asr_final_verifier_backend
            ).strip().lower(),
            "path": "file-replay-production-final-selection",
            "selection_inputs": {
                "vad_owned_speech_duration": False,
                "offline_recovery_authorization": False,
            },
            "transport": {
                "block_ms": 100,
                "synthetic_tail_ms": 600,
            },
            "offline_execution": {
                "requested": True,
                "outcomes": dict(sorted(execution.offline_outcomes.items())),
            },
            "verifier_execution": {
                "requested": bool(str(config.asr_final_verifier_backend or "").strip()),
                "outcomes": dict(sorted(execution.verifier_outcomes.items())),
            },
            "selected_sources": dict(sorted(execution.selected_sources.items())),
        },
        "evaluations": len(corpus.cases) * repeats,
        "repeats": repeats,
        "stratum_tags": sorted(set(stratum_tags)),
        "decisions": execution.decisions,
        "views": {
            "file_replay_streaming_raw": execution.streaming,
            "file_replay_selected_without_owned_timing_recovery": (
                execution.selected
            ),
        },
        "limits": {
            "after_pcm": True,
            "capture": False,
            "vad_endpoint_accuracy": False,
            "native_reader": False,
            "capture_mailbox": False,
            "aec_barge_in": False,
            "speaker_identity_authority": False,
            "chatbot_tools_tts": False,
            "audio_device": False,
            "live_latency": False,
            "runtime_default_promotion": False,
            "synthetic_tail_is_transport_only": True,
            "selection_has_vad_owned_speech_duration": False,
            "selection_has_offline_recovery_authorization": False,
            "native_execution_watchdog": False,
        },
    }
    json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True)
    return result


def evaluate_production_final_pair(
    corpus: LoadedCorpus,
    config_inputs: _ConfigInputs,
    pair: _BoundProfilePair,
    *,
    repeats: int,
    stratum_tags: Sequence[str],
) -> dict[str, object]:
    """Run the fixed control then candidate over one already-loaded corpus."""

    if (
        pair.baseline.name != _BASELINE_FINAL_STT_PROFILE
        or pair.candidate.name != _CANDIDATE_FINAL_STT_PROFILE
    ):
        raise ProductionFinalEvaluationError()
    shared = {
        "expected_evaluator_source_sha256": pair.evaluator_source_sha256,
        "expected_runtime_identities": pair.runtime_identities,
        "repeats": repeats,
        "stratum_tags": stratum_tags,
        "device_profile": pair.device_profile,
    }
    baseline = evaluate_production_final(
        corpus,
        config_inputs,
        pair.baseline.config,
        expected_config_sha256=pair.baseline.config_sha256,
        expected_model_sha256=pair.baseline.model_sha256,
        **shared,
    )
    candidate = evaluate_production_final(
        corpus,
        config_inputs,
        pair.candidate.config,
        expected_config_sha256=pair.candidate.config_sha256,
        expected_model_sha256=pair.candidate.model_sha256,
        **shared,
    )
    streaming_control_match = (
        baseline["views"]["file_replay_streaming_raw"]
        == candidate["views"]["file_replay_streaming_raw"]
    )
    comparison_verdict = (
        "comparable"
        if streaming_control_match
        else "inconclusive_streaming_control_mismatch"
    )
    result: dict[str, object] = {
        "execution_complete": True,
        "quality_verdict": "diagnostic_only",
        "schema_version": 3,
        "comparison_kind": "fixed-final-stt-profile-pair",
        "execution_order": ["baseline", "candidate"],
        "same_loaded_corpus": True,
        "evaluations_per_cell": baseline["evaluations"],
        "total_evaluations": int(baseline["evaluations"])
        + int(candidate["evaluations"]),
        "repeats": baseline["repeats"],
        "stratum_tags": baseline["stratum_tags"],
        "baseline_final_stt_profile": pair.baseline.name,
        "baseline_final_stt_profile_digest": pair.baseline.profile_sha256,
        "candidate_final_stt_profile": pair.candidate.name,
        "candidate_final_stt_profile_digest": pair.candidate.profile_sha256,
        "streaming_control_match": streaming_control_match,
        "comparison": {
            "same_input": True,
            "streaming_control_match": streaming_control_match,
            "valid": streaming_control_match,
            "verdict": comparison_verdict,
            "latency_comparison": False,
            "runtime_default_promotion": False,
        },
        "baseline": baseline,
        "candidate": candidate,
        "limits": {
            "aggregate_only": True,
            "same_loaded_corpus": True,
            "sequential_execution": True,
            "runtime_default_promotion": False,
        },
    }
    json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True)
    return result


def _write_new_private(path: Path | str, payload: bytes) -> None:
    supplied = Path(path).expanduser()
    if (
        not supplied.is_absolute()
        or not supplied.name
        or not isinstance(payload, bytes)
        or not payload
        or len(payload) > 16 * 1024 * 1024
    ):
        raise ProductionFinalEvaluationError()
    candidate = Path(os.path.abspath(supplied))
    descriptor = -1
    parent_fd = -1
    temporary_name = (
        f".production-final-stt-{os.getpid()}-{secrets.token_hex(12)}.tmp"
    )
    temporary_exists = False
    candidate_linked = False
    publication_complete = False
    try:
        if _has_git_ancestor(candidate.parent):
            raise ProductionFinalEvaluationError()
        with opened_directory_nofollow(
            candidate.parent,
            require_private=True,
        ) as (_stable, opened_parent_fd):
            parent_fd = os.dup(opened_parent_fd)
            parent_identity = os.fstat(parent_fd)
            try:
                os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ProductionFinalEvaluationError()
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
            temporary_exists = True
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if not isinstance(count, int) or count <= 0:
                raise ProductionFinalEvaluationError()
            written += count
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise ProductionFinalEvaluationError()
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        remaining = len(payload)
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ProductionFinalEvaluationError()
            digest.update(chunk)
            remaining -= len(chunk)
        if digest.digest() != hashlib.sha256(payload).digest():
            raise ProductionFinalEvaluationError()
        current_parent = candidate.parent.lstat()
        if (
            not stat.S_ISDIR(current_parent.st_mode)
            or stat.S_IMODE(current_parent.st_mode) != 0o700
            or (current_parent.st_dev, current_parent.st_ino)
            != (parent_identity.st_dev, parent_identity.st_ino)
            or candidate.parent.resolve(strict=True) != candidate.parent
            or (
                hasattr(os, "geteuid")
                and current_parent.st_uid != os.geteuid()
            )
        ):
            raise ProductionFinalEvaluationError()
        os.link(
            temporary_name,
            candidate.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        candidate_linked = True
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_exists = False
        os.fsync(parent_fd)
        published = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        current_parent = candidate.parent.lstat()
        if (
            (published.st_dev, published.st_ino) != (metadata.st_dev, metadata.st_ino)
            or published.st_nlink != 1
            or published.st_size != len(payload)
            or stat.S_IMODE(published.st_mode) != 0o600
            or not stat.S_ISDIR(current_parent.st_mode)
            or stat.S_IMODE(current_parent.st_mode) != 0o700
            or (current_parent.st_dev, current_parent.st_ino)
            != (parent_identity.st_dev, parent_identity.st_ino)
            or candidate.parent.resolve(strict=True) != candidate.parent
            or (
                hasattr(os, "geteuid")
                and (
                    published.st_uid != os.geteuid()
                    or current_parent.st_uid != os.geteuid()
                )
            )
        ):
            raise ProductionFinalEvaluationError()
        publication_complete = True
    except ProductionFinalEvaluationError:
        raise
    except Exception:
        raise ProductionFinalEvaluationError() from None
    finally:
        if temporary_exists and parent_fd >= 0:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError:
                pass
        if (
            candidate_linked
            and not publication_complete
            and parent_fd >= 0
            and descriptor >= 0
        ):
            cleanup_fd = -1
            try:
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                cleanup_fd = os.open(candidate.name, flags, dir_fd=parent_fd)
                cleanup = os.fstat(cleanup_fd)
                owned = os.fstat(descriptor)
                if (cleanup.st_dev, cleanup.st_ino) == (
                    owned.st_dev,
                    owned.st_ino,
                ):
                    os.unlink(candidate.name, dir_fd=parent_fd)
                    try:
                        os.fsync(parent_fd)
                    except OSError:
                        pass
            except OSError:
                pass
            finally:
                if cleanup_fd >= 0:
                    try:
                        os.close(cleanup_fd)
                    except OSError:
                        pass
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise ProductionFinalEvaluationError()


def _repeats(value: str) -> int:
    try:
        result = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("expected an integer") from None
    if not 1 <= result <= 3:
        raise argparse.ArgumentTypeError("expected 1..3")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Run the exact public command/noise corpus through Speaker's "
            "configured production final-selection path; write aggregate-only evidence."
        )
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local-config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument(
        "--baseline-final-stt-profile",
        choices=(_BASELINE_FINAL_STT_PROFILE, _CANDIDATE_FINAL_STT_PROFILE),
    )
    parser.add_argument(
        "--candidate-final-stt-profile",
        choices=(_BASELINE_FINAL_STT_PROFILE, _CANDIDATE_FINAL_STT_PROFILE),
    )
    parser.add_argument("--repeats", type=_repeats, default=1)
    parser.add_argument("--stratum-tag", action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        baseline_profile = args.baseline_final_stt_profile
        candidate_profile = args.candidate_final_stt_profile
        pair_requested = baseline_profile is not None or candidate_profile is not None
        if pair_requested and (
            baseline_profile != _BASELINE_FINAL_STT_PROFILE
            or candidate_profile != _CANDIDATE_FINAL_STT_PROFILE
            or baseline_profile == candidate_profile
        ):
            raise ProductionFinalEvaluationError()
        corpus = load_corpus(args.corpus)
        config_inputs = _load_config_inputs(args.config, args.local_config)
        if pair_requested:
            pair = _bind_final_stt_profile_pair(
                config_inputs,
                args.device,
                baseline_profile,
                candidate_profile,
            )
            report = evaluate_production_final_pair(
                corpus,
                config_inputs,
                pair,
                repeats=args.repeats,
                stratum_tags=tuple(args.stratum_tag),
            )
        else:
            config, device_profile = _effective_config_with_profile(
                config_inputs,
                args.device,
            )
            report = evaluate_production_final(
                corpus,
                config_inputs,
                config,
                repeats=args.repeats,
                stratum_tags=tuple(args.stratum_tag),
                device_profile=device_profile,
            )
        payload = (
            json.dumps(
                report,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        if pair_requested:
            cell_reports = (report["baseline"], report["candidate"])
            coverage_complete = all(
                cell["views"]["file_replay_streaming_raw"]["coverage_complete"] is True
                and cell["views"]["file_replay_selected_without_owned_timing_recovery"][
                    "coverage_complete"
                ]
                is True
                for cell in cell_reports
            )
            evaluations = report["total_evaluations"]
            comparison = report["comparison"]
            if (
                not isinstance(comparison, dict)
                or type(comparison.get("valid")) is not bool
                or comparison.get("verdict")
                not in {
                    "comparable",
                    "inconclusive_streaming_control_mismatch",
                }
            ):
                raise ProductionFinalEvaluationError()
            comparison_result: Mapping[str, object] = {
                "comparison_valid": comparison["valid"],
                "comparison_verdict": comparison["verdict"],
            }
        else:
            coverage_complete = (
                report["views"]["file_replay_streaming_raw"]["coverage_complete"]
                is True
                and report["views"][
                    "file_replay_selected_without_owned_timing_recovery"
                ]["coverage_complete"]
                is True
            )
            evaluations = report["evaluations"]
            comparison_result = {}
        _write_new_private(args.report, payload)
        result: Mapping[str, object] = {
            "coverage_complete": coverage_complete,
            "evaluations": evaluations,
            "execution_complete": True,
            "ok": True,
            "quality_verdict": "diagnostic_only",
            "report_sha256": hashlib.sha256(payload).hexdigest(),
            **comparison_result,
        }
        code = 0
    except Exception:  # noqa: BLE001 - no private detail may reach the terminal
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


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())
