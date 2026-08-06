"""Evaluate the real Sherpa capture path on a strict timestamped corpus.

This is a diagnostic entry point, not a second voice-agent application.  It
constructs the configured capture/STT front end without opening an audio device,
feeds :class:`CapturedBlock` objects through the same production loop, and
persists aggregate metrics only.  Model/output paths and transcript rows never
appear in the report or CLI errors.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import tempfile
import threading
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import numpy as np

from always_on_agent.acoustic import AcousticLineage
from core.audio_frontend import compute_input_calibration
from core.config import apply_device_profile, load_config
from core.engine import (
    AcousticSignal,
    CommandDetection,
    EngineCallbacks,
    FinalTranscript,
    PartialTranscript,
    TranscriptAbort,
    TranscriptAbortReason,
)
from core.engines.sherpa import (
    SherpaConfig,
    SherpaOnnxEngine,
    _calibration_has_suspicious_transient,
)
from core.media_session import CaptureControlLane
from tools.capture_replay import (
    LoadedReplayCorpus,
    ReplayCaptureSession,
    ReplayCase,
    ReplayTrack,
    SampleInterval,
    SpeakerInterval,
    WordInterval,
    load_corpus,
    verify_corpus_snapshot,
)
from tools.capture_replay.metrics import (
    ReplayAcousticEvent,
    ReplayRunRecord,
    TimedFinal,
    TimedHypothesis,
    aggregate_metrics,
)
from tools.streaming_stt.bounded_io import (
    BoundedReadError,
    opened_directory_nofollow,
    read_regular_bounded,
)

if TYPE_CHECKING:
    from always_on_agent.logical_turn import LogicalTurnAbortReason
    from always_on_agent.logical_turn_shadow import (
        LogicalTurnCompositionShadow,
        LogicalTurnShadowSnapshot,
        LogicalTurnTerminalLineageSnapshot,
    )


_SAFE_ERROR: Mapping[str, object] = {
    "execution_complete": False,
    "error": "capture_replay_evaluation_unavailable",
}
_MODEL_PATH_FIELDS = (
    "asr_encoder",
    "asr_decoder",
    "asr_joiner",
    "asr_tokens",
    "asr_bpe_vocab",
    "vad_model",
    "asr_final_model",
    "asr_final_tokens",
    "asr_final_verifier_model",
    "denoise_model",
    "punct_model",
    "endpoint_prosody_model",
)
_SAFE_PROVIDERS = frozenset({"cpu", "cuda"})
_SAFE_FINAL_BACKENDS = frozenset(
    {"", "sense_voice", "whisper", "nemo_transducer"}
)
_SAFE_VERIFIER_BACKENDS = frozenset({"", "faster_whisper"})
_MAX_REPEATS = 8
_MAX_WATCHDOG_SECONDS = 7_200
_MAX_RECEIPT_BYTES = 4_096
_WORKER_ENV = "SPEAKER_CAPTURE_REPLAY_WORKER"
_LOWER_HEX = frozenset("0123456789abcdef")
_MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
_MAX_BPE_VOCAB_BYTES = 4 * 1024 * 1024
_MAX_ENDPOINT_PROSODY_MODEL_BYTES = 16 * 1024 * 1024
_EVALUATOR_SOURCE_FILES = (
    "always_on_agent/acoustic.py",
    "core/asr_verifier.py",
    "core/audio_frontend.py",
    "core/config.py",
    "core/engine.py",
    "core/endpointing.py",
    "core/engines/_acoustic_turn.py",
    "core/engines/_aec.py",
    "core/engines/_asr_segment.py",
    "core/engines/_denoiser.py",
    "core/engines/_dtd.py",
    "core/engines/_faster_whisper.py",
    "core/engines/_sherpa_models.py",
    "core/engines/_semantic_hold.py",
    "core/engines/_sherpa_streaming_decode.py",
    "core/engines/_sherpa_streaming_decode_owner.py",
    "core/engines/_speech_evidence.py",
    "core/engines/echo_coherence.py",
    "core/engines/sherpa.py",
    "core/engines/speaker_gate.py",
    "core/media_session.py",
    "core/wer.py",
    "tools/capture_replay/__init__.py",
    "tools/capture_replay/corpus.py",
    "tools/capture_replay/metrics.py",
    "tools/capture_replay/replay.py",
    "tools/capture_replay_eval.py",
)
_LOGICAL_TURN_SHADOW_SOURCE_FILES = (
    "always_on_agent/logical_turn.py",
    "always_on_agent/logical_turn_shadow.py",
    "core/contract.py",
)


class CaptureReplayEvaluationError(RuntimeError):
    """A detail-free configuration, model, replay, or output failure."""


class _DiscardText(io.TextIOBase):
    """Write-only sink that never retains model logs or transcript text."""

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


@contextmanager
def _quiet_model_output():
    """Suppress Python and native-fd chatter while hypotheses are private."""

    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    sink = _DiscardText()
    null_fd = -1
    saved_stdout_fd = -1
    saved_stderr_fd = -1
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        null_fd = os.open(
            os.devnull,
            os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        )
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


@dataclass
class _ReplayInputIdentity:
    """The exact source identity checked by Sherpa for every captured block."""

    generation: int = 1
    actual_samplerate: int = 16_000
    actual_device: str = "capture-replay"

    def request_close(self) -> None:
        return None


def _single_span(lineage: AcousticLineage | None):
    if lineage is None or len(lineage.spans) != 1:
        raise CaptureReplayEvaluationError()
    return lineage.spans[0]


class _CaseCollector:
    """Thread-safe typed callback collector with no persistent text output."""

    def __init__(
        self,
        stop: Callable[[], None],
        logical_turn_shadow: "LogicalTurnCompositionShadow | None" = None,
    ) -> None:
        self._stop = stop
        self._logical_turn_shadow = logical_turn_shadow
        self._lock = threading.Lock()
        self._error = False
        self.partials: list[TimedHypothesis] = []
        self.finals: list[TimedFinal] = []
        self.commands: list[str] = []
        self.acoustic_events: list[ReplayAcousticEvent] = []
        self.abort_reasons = []

    def _shadow_failure(self) -> None:
        observer = self._logical_turn_shadow
        if observer is None:
            return
        try:
            observer.note_internal_error()
        except Exception:  # noqa: BLE001 - shadow cannot alter callbacks
            pass

    def _shadow_abort(self, reason: "LogicalTurnAbortReason") -> None:
        observer = self._logical_turn_shadow
        if observer is None:
            return
        try:
            observer.abort(reason)
        except Exception:  # noqa: BLE001 - shadow cannot alter callbacks
            self._shadow_failure()

    def _shadow_selected_terminal(self, result: FinalTranscript) -> None:
        observer = self._logical_turn_shadow
        if observer is None or not observer.terminal_lineage_enabled:
            return
        try:
            if result.acoustic is None:
                raise CaptureReplayEvaluationError()
            observer.observe_selected_final(result.acoustic, result.revision)
        except Exception:  # noqa: BLE001 - shadow cannot alter callbacks
            self._shadow_failure()

    def _shadow_abort_terminal(self, result: TranscriptAbort) -> None:
        observer = self._logical_turn_shadow
        if observer is None or not observer.terminal_lineage_enabled:
            return
        try:
            observer.observe_typed_abort(
                result.acoustic,
                result.revision,
                result.reason,
            )
        except Exception:  # noqa: BLE001 - shadow cannot alter callbacks
            self._shadow_failure()

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._error

    def _fail(self) -> None:
        with self._lock:
            self._error = True
        self._stop()

    def on_partial(self, result: PartialTranscript) -> None:
        try:
            span = _single_span(result.acoustic)
            if span.emitted_at is None:
                raise CaptureReplayEvaluationError()
            event = TimedHypothesis(
                result.text,
                emitted_at=span.emitted_at,
                speech_start_at=span.speech_start_at,
                utterance_id=span.utterance_id,
            )
        except Exception:  # noqa: BLE001 - callback must stop without leaking text
            self._fail()
            return
        with self._lock:
            self.partials.append(event)

    def on_final(self, result: FinalTranscript) -> None:
        try:
            span = _single_span(result.acoustic)
            if span.emitted_at is None:
                raise CaptureReplayEvaluationError()
            event = TimedFinal(
                result.text,
                emitted_at=span.emitted_at,
                speech_start_at=span.speech_start_at,
                speech_end_at=span.speech_end_at,
                endpoint_committed_at=span.endpoint_committed_at,
                utterance_id=span.utterance_id,
            )
        except Exception:  # noqa: BLE001 - callback must stop without leaking text
            self._fail()
            return
        self._shadow_selected_terminal(result)
        with self._lock:
            self.finals.append(event)

    def on_command(self, result: CommandDetection) -> None:
        try:
            _single_span(result.acoustic)
            text = result.text
            if not isinstance(text, str) or not text or len(text) > 128:
                raise CaptureReplayEvaluationError()
        except Exception:  # noqa: BLE001
            self._fail()
            return
        with self._lock:
            self.commands.append(text)
        if self._logical_turn_shadow is not None:
            try:
                from always_on_agent.logical_turn import (
                    LogicalTurnAbortReason,
                )
                from core.contract import is_stop_command

                shadow_reason = (
                    LogicalTurnAbortReason.STOP
                    if is_stop_command(text)
                    else LogicalTurnAbortReason.CONTROL
                )
                self._shadow_abort(shadow_reason)
            except Exception:  # noqa: BLE001 - baseline callback already succeeded
                self._shadow_failure()

    def on_barge(self, result: AcousticSignal) -> None:
        try:
            _single_span(result.acoustic)
        except Exception:  # noqa: BLE001
            self._fail()
            return
        with self._lock:
            self.acoustic_events.append(ReplayAcousticEvent.BARGE_IN)

    def on_abort(self, result: TranscriptAbort) -> None:
        try:
            _single_span(result.acoustic)
        except Exception:  # noqa: BLE001
            self._fail()
            return
        self._shadow_abort_terminal(result)
        with self._lock:
            self.abort_reasons.append(result.reason)
        if self._logical_turn_shadow is not None:
            try:
                from always_on_agent.logical_turn import (
                    LogicalTurnAbortReason,
                )

                shadow_reason = (
                    LogicalTurnAbortReason.SHUTDOWN
                    if result.reason is TranscriptAbortReason.SHUTDOWN
                    else LogicalTurnAbortReason.TRANSCRIPT_ABORT
                )
                self._shadow_abort(shadow_reason)
            except Exception:  # noqa: BLE001 - baseline callback already succeeded
                self._shadow_failure()

    def callbacks(self) -> EngineCallbacks:
        return EngineCallbacks(
            on_partial_result=self.on_partial,
            on_final_result=self.on_final,
            on_command_result=self.on_command,
            on_barge_in_result=self.on_barge,
            on_transcript_abort=self.on_abort,
        )

    def record(
        self,
        *,
        case_index: int,
        repeat: int,
        wall_seconds: float,
        peak_rss_mb: float | None,
    ) -> ReplayRunRecord:
        if self.failed:
            raise CaptureReplayEvaluationError()
        with self._lock:
            return ReplayRunRecord(
                case_index=case_index,
                repeat=repeat,
                partials=tuple(self.partials),
                finals=tuple(self.finals),
                commands=tuple(self.commands),
                acoustic_events=tuple(self.acoustic_events),
                abort_reasons=tuple(self.abort_reasons),
                wall_seconds=wall_seconds,
                peak_rss_mb=peak_rss_mb,
            )


class _VerifierExecutionProbe:
    """Count verifier execution without retaining its transcript result."""

    def __init__(self, delegate: object) -> None:
        if delegate is None or not callable(getattr(delegate, "transcribe", None)):
            raise CaptureReplayEvaluationError()
        self._delegate = delegate
        self._lock = threading.Lock()
        self._attempts = 0
        self._completed = 0
        self._errors = 0
        self._nonempty = 0

    def transcribe(self, *args, **kwargs):
        with self._lock:
            self._attempts += 1
        try:
            result = self._delegate.transcribe(*args, **kwargs)
            text = getattr(result, "text", None)
            if not isinstance(text, str):
                raise CaptureReplayEvaluationError()
        except Exception:
            with self._lock:
                self._errors += 1
            raise
        with self._lock:
            self._completed += 1
            self._nonempty += bool(text.strip())
        return result

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "attempted_decodes": self._attempts,
                "completed_decodes": self._completed,
                "decode_errors": self._errors,
                "nonempty_decodes": self._nonempty,
            }


def _install_verifier_probe(
    engine: SherpaOnnxEngine,
    config: SherpaConfig,
) -> _VerifierExecutionProbe | None:
    requested = bool(config.asr_final_verifier_backend)
    if not requested:
        if engine._final_verifier is not None:
            raise CaptureReplayEvaluationError()
        return None
    if engine._final_verifier is None:
        raise CaptureReplayEvaluationError()
    probe = _VerifierExecutionProbe(engine._final_verifier)
    engine._final_verifier = probe
    return probe


def _attest_verifier_execution(
    engine: SherpaOnnxEngine,
    probe: _VerifierExecutionProbe | None,
    *,
    text_expected: bool,
) -> Mapping[str, object]:
    if probe is None:
        return {
            "requested": False,
            "available_after_run": False,
            "attempted_decodes": 0,
            "completed_decodes": 0,
            "decode_errors": 0,
            "nonempty_decodes": 0,
        }
    snapshot = probe.snapshot()
    if (
        engine._final_verifier is not probe
        or snapshot["decode_errors"] != 0
        or snapshot["completed_decodes"] != snapshot["attempted_decodes"]
        or (text_expected and snapshot["completed_decodes"] <= 0)
    ):
        raise CaptureReplayEvaluationError()
    return {
        "requested": True,
        "available_after_run": True,
        **snapshot,
    }


def _attest_streaming_decode_execution(
    engine: SherpaOnnxEngine,
    *,
    expected_capture_runs: int,
) -> str:
    session = engine._streaming_decode_session
    if (
        session is None
        or engine._streaming_decode_owner is not None
        or expected_capture_runs <= 0
    ):
        raise CaptureReplayEvaluationError()
    snapshot = session.snapshot()
    if (
        not snapshot.bound
        or not snapshot.closed
        or snapshot.native_call_active
        or snapshot.live_streams != 0
        or snapshot.retained_streams != 0
        or snapshot.orphaned_streams != 0
        or snapshot.streams_created < expected_capture_runs
        or snapshot.streams_retired != snapshot.streams_created
    ):
        raise CaptureReplayEvaluationError()
    return "single-session-synchronous-replay"


def _safe_label(value: object, allowed: frozenset[str]) -> str:
    return value if isinstance(value, str) and value in allowed else "other"


def _json_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        raise CaptureReplayEvaluationError() from None
    return hashlib.sha256(encoded).hexdigest()


def _evaluator_source_files(
    *,
    logical_turn_shadow: bool = False,
) -> tuple[str, ...]:
    if type(logical_turn_shadow) is not bool:
        raise CaptureReplayEvaluationError()
    if logical_turn_shadow:
        return _EVALUATOR_SOURCE_FILES + _LOGICAL_TURN_SHADOW_SOURCE_FILES
    return _EVALUATOR_SOURCE_FILES


def _evaluator_source_digest(
    *,
    logical_turn_shadow: bool = False,
) -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    try:
        for relative in _evaluator_source_files(
            logical_turn_shadow=logical_turn_shadow
        ):
            snapshot = read_regular_bounded(
                root / relative,
                maximum_bytes=_MAX_SOURCE_FILE_BYTES,
            )
            encoded_name = relative.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(4, "big"))
            digest.update(encoded_name)
            digest.update(len(snapshot.data).to_bytes(8, "big"))
            digest.update(hashlib.sha256(snapshot.data).digest())
    except (BoundedReadError, OSError, RuntimeError, ValueError, OverflowError):
        raise CaptureReplayEvaluationError() from None
    return digest.hexdigest()


def _artifact_metadata_digest(config: SherpaConfig) -> str:
    """Bind model metadata plus the bounded active BPE vocabulary bytes."""

    rows: list[tuple[str, str, int, int, int, str]] = []
    for field_name in _MODEL_PATH_FIELDS:
        if field_name == "asr_bpe_vocab":
            unit = str(
                getattr(config, "asr_modeling_unit", "") or ""
            ).strip().lower()
            if not (
                bool(str(getattr(config, "asr_hotwords", "") or "").strip())
                and getattr(config, "asr_decoding_method", "")
                == "modified_beam_search"
                and unit in {"bpe", "cjkchar+bpe"}
            ):
                continue
        if field_name == "endpoint_prosody_model" and not (
            bool(getattr(config, "endpoint_enabled", False))
            and str(getattr(config, "endpoint_detector", "") or "")
            .strip()
            .lower()
            == "prosody"
        ):
            continue
        raw = getattr(config, field_name, "")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            root = Path(raw).expanduser().resolve(strict=True)
            if root.is_file():
                metadata = root.stat()
                content_sha256 = ""
                if field_name in {
                    "asr_bpe_vocab",
                    "endpoint_prosody_model",
                }:
                    snapshot = read_regular_bounded(
                        root,
                        maximum_bytes=(
                            _MAX_BPE_VOCAB_BYTES
                            if field_name == "asr_bpe_vocab"
                            else _MAX_ENDPOINT_PROSODY_MODEL_BYTES
                        ),
                    )
                    content_sha256 = hashlib.sha256(snapshot.data).hexdigest()
                rows.append(
                    (
                        field_name,
                        "file",
                        int(metadata.st_size),
                        int(metadata.st_mtime_ns),
                        1,
                        content_sha256,
                    )
                )
            elif root.is_dir():
                count = 0
                total = 0
                latest = 0
                for candidate in sorted(root.rglob("*")):
                    if candidate.is_symlink():
                        raise CaptureReplayEvaluationError()
                    if not candidate.is_file():
                        continue
                    metadata = candidate.stat()
                    count += 1
                    if count > 10_000:
                        raise CaptureReplayEvaluationError()
                    total += int(metadata.st_size)
                    latest = max(latest, int(metadata.st_mtime_ns))
                if count <= 0:
                    raise CaptureReplayEvaluationError()
                rows.append(
                    (field_name, "directory", total, latest, count, "")
                )
            else:
                raise CaptureReplayEvaluationError()
        except (BoundedReadError, OSError, RuntimeError, ValueError):
            raise CaptureReplayEvaluationError() from None
    if not rows:
        raise CaptureReplayEvaluationError()
    return _json_digest(rows)


def _require_provider_available(provider: str) -> None:
    if provider == "cpu":
        return
    try:
        import onnxruntime

        available = set(onnxruntime.get_available_providers())
    except Exception:
        raise CaptureReplayEvaluationError() from None
    if "CUDAExecutionProvider" not in available:
        # sherpa-onnx otherwise prints a warning and silently falls back to CPU,
        # making a purported GPU benchmark consume the very CPU it was meant to
        # protect and misreporting the executed provider.
        raise CaptureReplayEvaluationError()


def _replay_config(
    configured: SherpaConfig,
    *,
    provider: str | None,
    asr_threads: int | None,
) -> SherpaConfig:
    selected_provider = configured.provider if provider is None else provider
    if selected_provider not in _SAFE_PROVIDERS:
        raise CaptureReplayEvaluationError()
    _require_provider_available(selected_provider)
    threads = configured.asr_num_threads if asr_threads is None else asr_threads
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 0:
        raise CaptureReplayEvaluationError()
    # Output-only and identity-only models cannot affect a no-playback public
    # capture evaluation.  Suppress them to avoid wasting memory and to ensure
    # public voices can never acquire owner authority.  Keep every capture,
    # denoise, VAD, streaming/final ASR, punctuation, endpoint, and AGC setting.
    return replace(
        configured,
        provider=selected_provider,
        asr_num_threads=threads,
        asr_final_async=False,
        tts_model="",
        tts_tokens="",
        tts_data_dir="",
        tts_voices="",
        tts_lexicon="",
        kws_tokens="",
        kws_encoder="",
        kws_decoder="",
        kws_joiner="",
        kws_keywords_file="",
        speaker_embedding_model="",
        speaker_enroll_wav="",
        speaker_enroll_embedding="",
        speaker_gate_input=False,
        record_playback_reference=False,
        record_pre_dsp_reference=False,
    )


def _shift_interval(
    start_sample: int,
    end_sample: int,
    offset: int,
) -> tuple[int, int]:
    if start_sample < offset or end_sample <= start_sample:
        raise CaptureReplayEvaluationError()
    return start_sample - offset, end_sample - offset


def _trim_calibration_prefix(case: ReplayCase, samples: int) -> ReplayCase:
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples <= 0
        or samples % case.frame_samples
        or samples >= case.samples
    ):
        raise CaptureReplayEvaluationError()
    if any(interval.start_sample < samples for interval in case.speech_intervals):
        raise CaptureReplayEvaluationError()
    tracks: dict[str, ReplayTrack] = {}
    for role, track in case.tracks.items():
        raw = track.pcm_bytes[samples * 4 :]
        expected = (track.samples - samples) * 4
        if len(raw) != expected:
            raise CaptureReplayEvaluationError()
        tracks[role] = ReplayTrack(
            role=role,
            path=track.path,
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=track.samples - samples,
            pcm_bytes=raw,
        )
    speech = tuple(
        SampleInterval(*_shift_interval(row.start_sample, row.end_sample, samples))
        for row in case.speech_intervals
    )
    speakers = tuple(
        SpeakerInterval(
            speaker_id=row.speaker_id,
            role=row.role,
            start_sample=_shift_interval(
                row.start_sample,
                row.end_sample,
                samples,
            )[0],
            end_sample=_shift_interval(
                row.start_sample,
                row.end_sample,
                samples,
            )[1],
        )
        for row in case.speaker_intervals
    )
    words = tuple(
        WordInterval(
            speaker_id=row.speaker_id,
            text=row.text,
            start_sample=_shift_interval(
                row.start_sample,
                row.end_sample,
                samples,
            )[0],
            end_sample=_shift_interval(
                row.start_sample,
                row.end_sample,
                samples,
            )[1],
        )
        for row in case.word_intervals
    )
    return replace(
        case,
        tracks=MappingProxyType(tracks),
        speech_intervals=speech,
        speaker_intervals=speakers,
        word_intervals=words,
    )


def _calibrate_case(engine: SherpaOnnxEngine, case: ReplayCase) -> ReplayCase:
    if (
        not engine.config.input_calibrate
        or float(engine.config.input_calibrate_sec) <= 0.0
    ):
        return case
    blocks = max(
        1,
        int(
            round(
                float(engine.config.input_calibrate_sec)
                / float(engine.config.block_sec)
            )
        ),
    )
    calibration_samples = blocks * case.frame_samples
    if calibration_samples >= case.samples or any(
        interval.start_sample < calibration_samples
        for interval in case.speech_intervals
    ):
        raise CaptureReplayEvaluationError()
    raw = np.frombuffer(case.mic.pcm_bytes, dtype="<f4")
    calibration_pcm = tuple(
        np.array(
            raw[index * case.frame_samples : (index + 1) * case.frame_samples],
            dtype="float32",
            copy=True,
        )
        for index in range(blocks)
    )
    calibration = compute_input_calibration(calibration_pcm)
    if (
        _calibration_has_suspicious_transient(calibration)
        or float(calibration.get("clipping_fraction", 0.0) or 0.0) > 0.02
        or engine._calibration_contains_vad_speech(calibration_pcm)
    ):
        raise CaptureReplayEvaluationError()
    engine._last_calibration = calibration
    if engine._input_agc is not None:
        engine._input_agc.noise_floor_rms = float(calibration["noise_floor_rms"])
    engine._install_speech_evidence_profile(calibration, calibration_pcm)
    # Model the startup stream reads that produced this profile.  No CPU work is
    # consumed during the wait, and the subsequent session starts on a fresh
    # perf-counter base exactly as live capture does.
    time.sleep(blocks * float(engine.config.block_sec))
    return _trim_calibration_prefix(case, calibration_samples)


def _reset_case_frontends(engine: SherpaOnnxEngine) -> None:
    engine._reset_capture_frontends_after_gap(capture_sample_rate=16_000)
    engine._last_calibration = None
    engine._speech_evidence_profile = None
    engine._ambient_rms = 0.0
    engine._playback_floor_rms = 0.0
    engine._raw_playback_floor_rms = 0.0
    engine._clip_warned = False
    if engine._input_agc is not None:
        engine._input_agc.reset()
        engine._input_agc.noise_floor_rms = float(
            engine.config.input_agc_noise_floor_rms
        )


def _rss_mb() -> float | None:
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if not math.isfinite(value) or value < 0.0:
            return None
        # Linux reports KiB; macOS reports bytes.
        divisor = 1024.0 if os.uname().sysname != "Darwin" else 1024.0 * 1024.0
        return value / divisor
    except (AttributeError, OSError, ValueError, OverflowError):
        return None


def _execute_case(
    engine: SherpaOnnxEngine,
    case: ReplayCase,
    *,
    case_index: int,
    repeat: int,
    logical_turn_shadow: "LogicalTurnCompositionShadow | None" = None,
) -> ReplayRunRecord:
    if logical_turn_shadow is not None:
        from always_on_agent.logical_turn_shadow import (
            LogicalTurnCompositionShadow,
        )

        if not isinstance(
            logical_turn_shadow,
            LogicalTurnCompositionShadow,
        ):
            raise CaptureReplayEvaluationError()
    started = time.perf_counter()
    session: ReplayCaptureSession | None = None
    control_lane: CaptureControlLane | None = None
    engine._cb = EngineCallbacks()
    try:
        _reset_case_frontends(engine)
        replay_case = _calibrate_case(engine, case)
        source = _ReplayInputIdentity()
        engine._stream_in = source
        engine._capture_sr = replay_case.sample_rate_hz
        engine._resampler = None
        engine._pre_gain_resampler = None
        engine._capture_stopping.clear()
        control_lane = CaptureControlLane()
        engine._capture_control_lane = control_lane
        engine._capture_epoch += 1
        capture_epoch = engine._capture_epoch
        engine._acoustic_turn_tracker = None
        engine._capture_authority_source_generation = source.generation
        engine._capture_authority_source_device = source.actual_device
        collector = _CaseCollector(
            engine._running.clear,
            logical_turn_shadow,
        )
        engine._cb = collector.callbacks()

        has_reference = replay_case.track("far_reference_zero") is not None
        if has_reference:
            with engine._gen_lock:
                engine._speak_gen = 1
                engine._playback_generation = 1
            engine._speaking.set()
        else:
            engine._speaking.clear()
        session = ReplayCaptureSession(
            replay_case,
            on_exhausted=engine._running.clear,
            capture_epoch=capture_epoch,
            source_generation=source.generation,
            capture_generation=1,
            source_device=source.actual_device,
            source_domain="capture-replay",
        )
        engine._capture_media_session = session
        engine._running.set()
        session.start()
        # The evaluator batch reuses one synchronous decode session. Each case
        # creates fresh opaque streams, but the loaded native recognizer is
        # released only after every case/repeat completes on this same thread.
        # Production's dedicated decode-owner thread is source-bound above but
        # deliberately not claimed as replay execution evidence here.
        capture_options: dict[str, object] = {
            "close_decode_session": False,
        }
        if logical_turn_shadow is not None:
            capture_options["logical_turn_shadow"] = logical_turn_shadow
        engine._capture_loop(**capture_options)
        wall_seconds = time.perf_counter() - started
        if not session.naturally_exhausted or session.error is not None:
            raise CaptureReplayEvaluationError()
        return collector.record(
            case_index=case_index,
            repeat=repeat,
            wall_seconds=wall_seconds,
            peak_rss_mb=_rss_mb(),
        )
    finally:
        if logical_turn_shadow is not None:
            try:
                logical_turn_shadow.close()
            except Exception:  # noqa: BLE001 - replay cleanup remains private
                try:
                    logical_turn_shadow.note_internal_error()
                except Exception:  # noqa: BLE001 - legacy cleanup must continue
                    pass
        engine._running.clear()
        engine._speaking.clear()
        if session is not None:
            session.close()
            session.join(timeout=0.0)
        if control_lane is not None:
            control_lane.close()
        engine._capture_control_lane = None
        engine._capture_media_session = None
        engine._stream_in = None
        engine._cb = EngineCallbacks()


def _validate_terminal_lineage_against_metrics(
    *,
    raw_shadow: object,
    terminal_lineage: object,
    metrics: object,
) -> None:
    """Require exact agreement across raw, typed-terminal, and metric views."""

    try:
        if not all(
            isinstance(value, Mapping)
            for value in (raw_shadow, terminal_lineage, metrics)
        ):
            raise CaptureReplayEvaluationError()
        raw_parity = raw_shadow["parity"]
        outcomes = terminal_lineage["outcomes"]
        lineage = terminal_lineage["lineage"]
        streaming = metrics["streaming"]
        if not all(
            isinstance(value, Mapping)
            for value in (raw_parity, outcomes, lineage, streaming)
        ):
            raise CaptureReplayEvaluationError()

        def count(value: object) -> int:
            if type(value) is not int or value < 0:
                raise CaptureReplayEvaluationError()
            return value

        reason_names = {reason.value for reason in TranscriptAbortReason}
        streaming_reasons = streaming["abort_reasons"]
        terminal_reasons = outcomes["abort_reason_counts"]
        matched_reasons = lineage["matched_abort_reason_counts"]
        if (
            not isinstance(streaming_reasons, Mapping)
            or not isinstance(terminal_reasons, Mapping)
            or not isinstance(matched_reasons, Mapping)
            or not set(streaming_reasons).issubset(reason_names)
            or set(terminal_reasons) != reason_names
            or set(matched_reasons) != reason_names
        ):
            raise CaptureReplayEvaluationError()
        expected_reasons = {
            reason: count(streaming_reasons.get(reason, 0))
            for reason in reason_names
        }
        selected = count(streaming["final_events"])
        aborted = sum(expected_reasons.values())
        raw_commits = count(lineage["raw_commits"])
        if (
            count(raw_parity["paired_terminals"]) != raw_commits
            or raw_commits != selected + aborted
            or count(outcomes["selected_finals"]) != selected
            or count(outcomes["typed_aborts"]) != aborted
            or dict(terminal_reasons) != expected_reasons
            or dict(matched_reasons) != expected_reasons
        ):
            raise CaptureReplayEvaluationError()
    except (KeyError, TypeError, ValueError, CaptureReplayEvaluationError):
        raise CaptureReplayEvaluationError() from None


def evaluate_capture_replay(
    corpus: LoadedReplayCorpus,
    configured: SherpaConfig,
    *,
    repeats: int = 1,
    provider: str | None = None,
    asr_threads: int | None = None,
    logical_turn_shadow: bool = False,
    logical_turn_terminal_lineage: bool = False,
) -> Mapping[str, object]:
    """Run the configured production capture stack and return aggregate evidence."""

    if (
        type(logical_turn_shadow) is not bool
        or type(logical_turn_terminal_lineage) is not bool
        or (logical_turn_terminal_lineage and not logical_turn_shadow)
    ):
        raise CaptureReplayEvaluationError()
    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats <= 0
        or repeats > _MAX_REPEATS
    ):
        raise CaptureReplayEvaluationError()
    verify_corpus_snapshot(corpus)
    executed = _replay_config(
        configured,
        provider=provider,
        asr_threads=asr_threads,
    )
    configured_digest = _json_digest(asdict(configured))
    executed_digest = _json_digest(asdict(executed))
    artifact_digest = _artifact_metadata_digest(executed)
    source_files = _evaluator_source_files(logical_turn_shadow=logical_turn_shadow)
    source_digest = _evaluator_source_digest(logical_turn_shadow=logical_turn_shadow)

    build_started = time.perf_counter()
    with _quiet_model_output():
        engine = SherpaOnnxEngine(executed)
        engine._cb = EngineCallbacks()
        engine._build()
        if engine._recognizer is None:
            raise CaptureReplayEvaluationError()
        if bool(executed.asr_final_backend) != (
            engine._final_recognizer is not None
        ):
            raise CaptureReplayEvaluationError()
        engine.warm()
        verifier_probe = _install_verifier_probe(engine, executed)
    model_load_ms = (time.perf_counter() - build_started) * 1000.0

    records: list[ReplayRunRecord] = []
    shadow_snapshots: list["LogicalTurnShadowSnapshot"] = []
    terminal_lineage_snapshots: list[
        "LogicalTurnTerminalLineageSnapshot"
    ] = []
    try:
        with _quiet_model_output():
            for repeat in range(repeats):
                for case_index, case in enumerate(corpus.cases):
                    shadow = None
                    if logical_turn_shadow:
                        from always_on_agent.logical_turn_shadow import (
                            LogicalTurnCompositionShadow,
                        )

                        shadow = LogicalTurnCompositionShadow(
                            terminal_lineage=logical_turn_terminal_lineage,
                        )
                    case_options: dict[str, object] = {
                        "case_index": case_index,
                        "repeat": repeat,
                    }
                    if shadow is not None:
                        case_options["logical_turn_shadow"] = shadow
                    records.append(_execute_case(engine, case, **case_options))
                    if shadow is not None:
                        shadow.close()
                        shadow_snapshots.append(shadow.snapshot())
                        if logical_turn_terminal_lineage:
                            terminal_lineage_snapshots.append(
                                shadow.terminal_lineage_snapshot()
                            )
    finally:
        decode_session = engine._streaming_decode_session
        if decode_session is not None and not decode_session.closed:
            try:
                decode_session.close()
            except Exception:
                raise CaptureReplayEvaluationError() from None
    verify_corpus_snapshot(corpus)
    if (
        _evaluator_source_digest(logical_turn_shadow=logical_turn_shadow)
        != source_digest
    ):
        raise CaptureReplayEvaluationError()
    metrics = aggregate_metrics(corpus, records, repeats=repeats)
    if metrics["coverage"]["complete"] is not True:
        raise CaptureReplayEvaluationError()
    verifier_execution = _attest_verifier_execution(
        engine,
        verifier_probe,
        text_expected=any(bool(case.expected_text.strip()) for case in corpus.cases),
    )
    streaming_decode_execution = _attest_streaming_decode_execution(
        engine,
        expected_capture_runs=len(corpus.cases) * repeats,
    )
    result: dict[str, object] = {
        "execution_complete": True,
        "quality_verdict": "diagnostic_only",
        "schema_version": 1,
        "corpus": {
            "manifest_sha256": corpus.digest,
            "cases": len(corpus.cases),
            "audio_bytes": corpus.audio_bytes,
            "audio_seconds": round(
                sum(case.duration_seconds for case in corpus.cases),
                3,
            ),
        },
        "engine": {
            "configured_config_sha256": configured_digest,
            "executed_config_sha256": executed_digest,
            "artifact_metadata_sha256": artifact_digest,
            "evaluator_source_sha256": source_digest,
            "evaluator_source_files": len(source_files),
            "provider": _safe_label(executed.provider, _SAFE_PROVIDERS),
            "asr_final_backend_configured": _safe_label(
                executed.asr_final_backend,
                _SAFE_FINAL_BACKENDS,
            ),
            "asr_final_verifier_backend_configured": _safe_label(
                executed.asr_final_verifier_backend,
                _SAFE_VERIFIER_BACKENDS,
            ),
            "asr_final_verifier_execution": verifier_execution,
            "final_execution": "inline-deterministic-replay",
            "streaming_decode_execution": streaming_decode_execution,
            "public_identity_models_loaded": False,
            "output_models_loaded": False,
            "model_load_ms": round(model_load_ms, 3),
        },
        "metrics": metrics,
    }
    if logical_turn_shadow:
        from always_on_agent.logical_turn_shadow import (
            aggregate_logical_turn_shadows,
        )

        if len(shadow_snapshots) != len(corpus.cases) * repeats:
            raise CaptureReplayEvaluationError()
        result["schema_version"] = 2
        result["logical_turn_shadow"] = aggregate_logical_turn_shadows(
            shadow_snapshots
        )
    if logical_turn_terminal_lineage:
        from always_on_agent.logical_turn_shadow import (
            aggregate_logical_turn_terminal_lineage,
        )

        if len(terminal_lineage_snapshots) != len(corpus.cases) * repeats:
            raise CaptureReplayEvaluationError()
        terminal_report = aggregate_logical_turn_terminal_lineage(
            terminal_lineage_snapshots
        )
        _validate_terminal_lineage_against_metrics(
            raw_shadow=result["logical_turn_shadow"],
            terminal_lineage=terminal_report,
            metrics=metrics,
        )
        result["schema_version"] = 3
        result["logical_turn_terminal_lineage"] = terminal_report
    # Final serialization is also the aggregate/privacy type gate.
    _json_digest(result)
    return result


def _write_new_private(path: Path | str, payload: bytes) -> None:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    descriptor = -1
    try:
        with opened_directory_nofollow(candidate.parent) as (_stable, parent_fd):
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(candidate.name, flags, 0o600, dir_fd=parent_fd)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            descriptor = -1
            handle.write(payload)
            os.fsync(handle.fileno())
        metadata = candidate.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise CaptureReplayEvaluationError()
    except (
        OSError,
        ValueError,
        OverflowError,
        BoundedReadError,
        CaptureReplayEvaluationError,
    ):
        raise CaptureReplayEvaluationError() from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise CaptureReplayEvaluationError()


def _positive_int(value: str) -> int:
    try:
        result = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive integer") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Run a strict timestamped corpus through the configured Sherpa "
            "capture-loop/VAD/STT/endpoint seam without opening an audio device; "
            "writes aggregate-only evidence."
        )
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--local-config",
        type=Path,
        default=Path("config.local.json"),
    )
    parser.add_argument("--device")
    parser.add_argument("--provider", choices=sorted(_SAFE_PROVIDERS))
    parser.add_argument("--asr-threads", type=_positive_int)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument(
        "--logical-turn-shadow",
        action="store_true",
        help=(
            "add transcript-free raw native-composition shadow evidence; "
            "does not transfer runtime authority"
        ),
    )
    parser.add_argument(
        "--logical-turn-terminal-lineage",
        action="store_true",
        help=(
            "add transcript-free inline selected-final/typed-abort lineage; "
            "requires --logical-turn-shadow and does not cover async delivery"
        ),
    )
    parser.add_argument(
        "--watchdog-seconds",
        type=_positive_int,
        default=1_800,
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _mode_schema_version(
    *,
    logical_turn_shadow: bool,
    logical_turn_terminal_lineage: bool,
) -> int:
    if (
        type(logical_turn_shadow) is not bool
        or type(logical_turn_terminal_lineage) is not bool
        or (logical_turn_terminal_lineage and not logical_turn_shadow)
    ):
        raise CaptureReplayEvaluationError()
    if logical_turn_terminal_lineage:
        return 3
    if logical_turn_shadow:
        return 2
    return 1


def _validate_report_mode(
    report: object,
    *,
    logical_turn_shadow: bool,
    logical_turn_terminal_lineage: bool,
) -> int:
    expected_schema = _mode_schema_version(
        logical_turn_shadow=logical_turn_shadow,
        logical_turn_terminal_lineage=logical_turn_terminal_lineage,
    )
    if (
        not isinstance(report, Mapping)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != expected_schema
        or ("logical_turn_shadow" in report) is not logical_turn_shadow
        or ("logical_turn_terminal_lineage" in report)
        is not logical_turn_terminal_lineage
    ):
        raise CaptureReplayEvaluationError()
    if logical_turn_shadow:
        from always_on_agent.logical_turn_shadow import (
            validate_logical_turn_shadow_report,
        )

        validate_logical_turn_shadow_report(report["logical_turn_shadow"])
    if logical_turn_terminal_lineage:
        from always_on_agent.logical_turn_shadow import (
            validate_logical_turn_terminal_lineage_report,
        )

        validate_logical_turn_terminal_lineage_report(
            report["logical_turn_terminal_lineage"]
        )
        _validate_terminal_lineage_against_metrics(
            raw_shadow=report["logical_turn_shadow"],
            terminal_lineage=report["logical_turn_terminal_lineage"],
            metrics=report["metrics"],
        )
    return expected_schema


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        expected_schema = _mode_schema_version(
            logical_turn_shadow=args.logical_turn_shadow,
            logical_turn_terminal_lineage=(
                args.logical_turn_terminal_lineage
            ),
        )
        corpus = load_corpus(args.corpus)
        raw_config = load_config(
            str(args.config),
            local=str(args.local_config),
        )
        selected_device = (
            args.device if args.device is not None else raw_config.get("device", "auto")
        )
        effective = apply_device_profile(
            raw_config,
            selected_device,
            strict=args.device is not None,
        )
        configured = SherpaConfig.from_dict(effective.get("sherpa", {}))
        report = evaluate_capture_replay(
            corpus,
            configured,
            repeats=args.repeats,
            provider=args.provider,
            asr_threads=args.asr_threads,
            logical_turn_shadow=args.logical_turn_shadow,
            logical_turn_terminal_lineage=(
                args.logical_turn_terminal_lineage
            ),
        )
        _validate_report_mode(
            report,
            logical_turn_shadow=args.logical_turn_shadow,
            logical_turn_terminal_lineage=(
                args.logical_turn_terminal_lineage
            ),
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
        _write_new_private(args.report, payload)
        result: Mapping[str, object] = {
            "execution_complete": True,
            "quality_verdict": "diagnostic_only",
            "report_sha256": hashlib.sha256(payload).hexdigest(),
            "corpus_sha256": corpus.digest,
            "evaluations": report["metrics"]["evaluations"],
            "coverage_complete": report["metrics"]["coverage"]["complete"],
            "schema_version": expected_schema,
            "logical_turn_shadow": args.logical_turn_shadow,
            "logical_turn_terminal_lineage": (
                args.logical_turn_terminal_lineage
            ),
        }
        code = 0
    except Exception:  # noqa: BLE001 - paths/transcript rows stay private
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


def _validated_worker_receipt(
    payload: bytes,
    *,
    returncode: int,
    logical_turn_shadow: bool,
    logical_turn_terminal_lineage: bool,
) -> Mapping[str, object]:
    expected_schema = _mode_schema_version(
        logical_turn_shadow=logical_turn_shadow,
        logical_turn_terminal_lineage=logical_turn_terminal_lineage,
    )
    if not payload or len(payload) > _MAX_RECEIPT_BYTES:
        raise CaptureReplayEvaluationError()
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CaptureReplayEvaluationError() from None
    if returncode == 0:
        if (
            not isinstance(decoded, dict)
            or set(decoded)
            != {
                "execution_complete",
                "quality_verdict",
                "report_sha256",
                "corpus_sha256",
                "evaluations",
                "coverage_complete",
                "schema_version",
                "logical_turn_shadow",
                "logical_turn_terminal_lineage",
            }
            or decoded.get("execution_complete") is not True
            or decoded.get("quality_verdict") != "diagnostic_only"
            or not isinstance(decoded.get("report_sha256"), str)
            or not isinstance(decoded.get("corpus_sha256"), str)
            or len(decoded["report_sha256"]) != 64
            or len(decoded["corpus_sha256"]) != 64
            or any(character not in _LOWER_HEX for character in decoded["report_sha256"])
            or any(character not in _LOWER_HEX for character in decoded["corpus_sha256"])
            or isinstance(decoded.get("evaluations"), bool)
            or not isinstance(decoded.get("evaluations"), int)
            or decoded["evaluations"] <= 0
            or not isinstance(decoded.get("coverage_complete"), bool)
            or type(decoded.get("schema_version")) is not int
            or decoded["schema_version"] != expected_schema
            or type(decoded.get("logical_turn_shadow")) is not bool
            or decoded["logical_turn_shadow"] is not logical_turn_shadow
            or type(decoded.get("logical_turn_terminal_lineage")) is not bool
            or decoded["logical_turn_terminal_lineage"]
            is not logical_turn_terminal_lineage
        ):
            raise CaptureReplayEvaluationError()
        return decoded
    if returncode != 2 or decoded != dict(_SAFE_ERROR):
        raise CaptureReplayEvaluationError()
    return decoded


def guarded_main(argv: Sequence[str] | None = None) -> int:
    """Run the model worker in a killable process with bounded output."""

    if os.environ.get(_WORKER_ENV) == "1":
        return main(argv)
    try:
        raw_argv = list(sys.argv[1:] if argv is None else argv)
        args = _parser().parse_args(raw_argv)
        _mode_schema_version(
            logical_turn_shadow=args.logical_turn_shadow,
            logical_turn_terminal_lineage=(
                args.logical_turn_terminal_lineage
            ),
        )
        if args.watchdog_seconds > _MAX_WATCHDOG_SECONDS:
            raise CaptureReplayEvaluationError()
        environment = dict(os.environ)
        environment[_WORKER_ENV] = "1"
        command = [
            sys.executable,
            "-B",
            "-m",
            "tools.capture_replay_eval",
            *raw_argv,
        ]
        with tempfile.TemporaryFile(mode="w+b") as receipt_file:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=receipt_file,
                stderr=subprocess.DEVNULL,
                env=environment,
                timeout=args.watchdog_seconds,
                check=False,
            )
            receipt_file.seek(0)
            payload = receipt_file.read(_MAX_RECEIPT_BYTES + 1)
        result = _validated_worker_receipt(
            payload,
            returncode=completed.returncode,
            logical_turn_shadow=args.logical_turn_shadow,
            logical_turn_terminal_lineage=(
                args.logical_turn_terminal_lineage
            ),
        )
        code = completed.returncode
    except Exception:  # noqa: BLE001 - parent never exposes paths/model output
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
    raise SystemExit(guarded_main())
