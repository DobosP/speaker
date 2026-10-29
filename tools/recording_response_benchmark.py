"""Bounded local recording -> production answer -> first post-DSP nonzero PCM.

No capture/playback device is opened. FileReplay supplies complete recorded PCM,
including its declared 0.6s endpoint-flush tail; Sherpa's actual synthesis/DSP
writes into a finite null sink. These are observed compute boundaries, never
human speech-end, real-time endpoint, DAC, acoustic, or live latency evidence.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import functools
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

from tools.english_asr_benchmark import (
    load_corpus,
    verify_corpus,
    _pcm,
    _read,
    MAX_WAV_BYTES,
    _absolute,
    _accuracy_pairs,
    _file_digest,
    _json,
)

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "current",
    "vits",
    "vits-fast",
    "vits-asr2",
    "vits-asr8",
    "vits-tts4",
    "vits-cuda",
)
# One literal resource intervention per named experiment. Counts are requested
# native model threads, not total process threads or dedicated/reserved CPUs.
# CUDA is a provider request, not proof of native kernel placement or speed.
_RESOURCE_VARIANTS = {
    "vits-asr2": ("asr_num_threads", 4, 2),
    "vits-asr8": ("asr_num_threads", 4, 8),
    "vits-tts4": ("tts_num_threads", 2, 4),
    "vits-cuda": ("provider", "cpu", "cuda"),
}
_SOURCE_FILES = (
    "tools/recording_response_benchmark.py",
    "tools/english_asr_benchmark.py",
    "tools/english_asr_scoring.py",
    "tools/bench/production.py",
    "tools/conversation_eval/identity.py",
    "tools/conversation_eval/provenance.py",
    "core/app.py",
    "core/config.py",
    "core/runtime.py",
    "core/engine.py",
    "core/metrics.py",
    "core/llm.py",
    "core/llm_decision.py",
    "core/llm_factory.py",
    "core/minicpm_identity.py",
    "core/llm_threads.py",
    "core/persona.py",
    "core/capabilities.py",
    "core/capability_router.py",
    "core/addressing.py",
    "always_on_agent/conversation_admission.py",
    "core/asr_verifier.py",
    "core/asr_text.py",
    "core/engines/_cuda_wheels.py",
    "core/cleanup.py",
    "core/routing.py",
    "core/performance.py",
    "core/voice_model_profile.py",
    "core/voice_model_selection.py",
    "core/contract.py",
    "core/speech_chunking.py",
    "core/tts_markup.py",
    "core/audio_frontend.py",
    "core/engines/file_replay.py",
    "core/engines/_endpoint_lease.py",
    "core/engines/sherpa.py",
    "core/engines/_sherpa_models.py",
    "core/engines/_faster_whisper.py",
    "core/engines/_acoustic_turn.py",
    "core/playback_history.py",
    "core/turn_merge.py",
    "always_on_agent/continuation.py",
    "always_on_agent/events.py",
    "always_on_agent/acoustic.py",
    "always_on_agent/supervisor.py",
    "always_on_agent/tasks.py",
    "always_on_agent/session_actor.py",
    "always_on_agent/event_bus.py",
    "always_on_agent/capabilities.py",
    "always_on_agent/models.py",
    "always_on_agent/speech_analyzer.py",
    "always_on_agent/memory.py",
)
_REPORT_LOCK = threading.Lock()
_WORKER_STAGE = "request"
_STAGES = {
    "request",
    "source",
    "config",
    "corpus",
    "artifacts",
    "ollama",
    "pipeline",
    "verification",
}
_MAX_REPORT = 2 * 1024 * 1024
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_METRICS = (
    "pcm_to_final_seconds",
    "pcm_to_first_answer_pcm_seconds",
    "final_to_first_answer_pcm_seconds",
    "pcm_to_idle_seconds",
    "asr_decode_seconds",
    "source_endpoint_delay_seconds",
    "model_request_to_first_token_seconds",
    "first_token_to_speakable_text_seconds",
    "speakable_text_to_tts_admission_seconds",
    "tts_admission_to_first_answer_pcm_seconds",
)
_CODES = frozenset(
    {
        "ok",
        "failed",
        "case_timeout",
        "cell_timeout",
        "invalid_inputs",
        "inputs_changed",
        "worker_failed",
    }
)


class BenchmarkError(RuntimeError):
    """Fixed codes only; foreign exception messages never cross the worker."""


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def source_binding():
    """Explicit measured seam; no full import closure or bytecode attestation."""
    rows = []
    total = 0
    for relative in _SOURCE_FILES:
        raw = _read(ROOT / relative, 1024 * 1024)
        total += len(raw)
        if total > 8 * 1024 * 1024:
            raise BenchmarkError("invalid_inputs")
        rows.append([relative, len(raw), hashlib.sha256(raw).hexdigest()])
    from tools.conversation_eval.provenance import repository_metadata

    repository = repository_metadata(ROOT)
    revision = repository.get("revision")
    if type(revision) is not str or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise BenchmarkError("invalid_inputs")
    return {
        "sha256": digest(rows),
        "files": len(rows),
        "bytes": total,
        "revision": revision,
        "clean": repository.get("dirty") is False,
    }


def _apply_resource_variant(control, variant):
    """Apply one declared field against the measured VITS control only."""
    field, expected, requested = _RESOURCE_VARIANTS[variant]
    previous = control.get("sherpa", {}).get(field)
    if type(previous) is not type(expected) or previous != expected:
        raise BenchmarkError("invalid_inputs")
    selected = deepcopy(control)
    selected["sherpa"][field] = requested
    # Verify the complete resolved configuration, not a relaxed behavior hash.
    restored = deepcopy(selected)
    restored["sherpa"][field] = previous
    if restored != control:
        raise BenchmarkError("invalid_inputs")
    return selected


def _require_legacy_endpoint_lifecycle(config):
    """Resident rows cannot reuse adapters retired by current FileReplay.stop."""
    sherpa = config.get("sherpa", {})
    if type(sherpa) is not dict:
        raise BenchmarkError("invalid_inputs")
    for field, supported in (
        ("asr_final_backend", {"", "sense_voice", "whisper", "nemo_transducer"}),
        ("asr_final_verifier_backend", {"", "faster_whisper"}),
    ):
        value = sherpa.get(field, "")
        if type(value) is not str or value.strip().lower() not in supported:
            raise BenchmarkError("invalid_inputs")


def resolve_config(config_root: Path, device: str, variant: str):
    from core.config import apply_device_profile, load_config, deep_merge
    from core.performance import apply_performance_mode
    from core.speech_chunking import apply_speech_latency
    from core.voice_model_selection import select_voice_model
    from tools.bench.production import (
        _apply_replay_safety,
        _bind_execution_artifact_paths,
    )

    if device == "auto" or variant not in VARIANTS:
        raise BenchmarkError("invalid_inputs")
    original = apply_device_profile(
        load_config(
            str(config_root / "config.json"),
            local=str(config_root / "config.local.json"),
        ),
        device,
        strict=True,
    )
    original, performance = apply_performance_mode(original, root=config_root)
    original, voice = select_voice_model(original)
    original, _speech = apply_speech_latency(original)
    selected = deepcopy(original)
    selection_sha = digest(
        {
            "variant": variant,
            "performance": performance.sha256 if performance else None,
            "voice": voice.sha256 if voice else None,
        }
    )
    if variant != "current":
        # Verify the complete installed profile, but copy only its TTS family.
        # ASR, verifier, LLM options/residency and authority remain identical.
        responsive, metadata = apply_performance_mode(
            original, "responsive", root=config_root
        )
        tts = {
            key: value
            for key, value in responsive["sherpa"].items()
            if key.startswith("tts_")
        }
        selected = deep_merge(selected, {"sherpa": tts})
        selection_sha = digest(
            {
                "variant": "vits" if variant in _RESOURCE_VARIANTS else variant,
                "responsive_assets": metadata.sha256,
                "tts": tts,
            }
        )
    if variant in _RESOURCE_VARIANTS:
        selected = _apply_resource_variant(selected, variant)
        field, previous, requested = _RESOURCE_VARIANTS[variant]
        selection_sha = digest(
            {
                "variant": variant,
                "vits_control_sha256": selection_sha,
                "field": "sherpa." + field,
                "previous": previous,
                "requested": requested,
            }
        )
    if variant == "vits-fast":
        selected, _speech = apply_speech_latency(selected, "fast")
    # New leased endpoint adapters need a separately qualified per-row model
    # lifecycle. Refuse them before asset binding or any inference starts.
    _require_legacy_endpoint_lifecycle(selected)
    production_sha = digest(selected)
    execution = _bind_execution_artifact_paths(
        _apply_replay_safety(selected), config_root=config_root
    )
    # Session-only memory and disabled providers are explicit harness deltas.
    # Keep admission/cleaner/router/planner and all authority/budget settings.
    execution = deep_merge(
        execution,
        {
            "screen_capture": {"enabled": False, "memorize": False},
            "memory": {
                "backend": "session",
                "embeddings": False,
                "profile_enabled": False,
                "procedural_enabled": False,
                "cross_session_continuity": False,
                "persist_assistant": False,
            },
        },
    )
    return execution, {
        "resolved_config_sha256": production_sha,
        "execution_config_sha256": digest(execution),
        "selection_sha256": selection_sha,
        "asr_settings_sha256": digest(
            {
                k: v
                for k, v in execution.get("sherpa", {}).items()
                if not k.startswith("tts_")
            }
        ),
        "llm_settings_sha256": digest(
            {
                "llm": execution.get("llm"),
                "assistant": execution.get("assistant"),
                "tts_markup": execution.get("sherpa", {}).get("tts_markup"),
            }
        ),
        "safety_policy_sha256": digest(
            {
                "loopback_local_only": True,
                "disabled": [
                    "gui_actions",
                    "obsidian",
                    "reminders",
                    "trusted_apps",
                    "watch",
                    "web_search",
                    "screen_capture",
                ],
                "memory": "isolated_session_no_embedding_or_persistence",
                "agent_actuator": False,
                "router_model": None,
            }
        ),
    }


def _finite(value, *, nullable=False, maximum=7200):
    if value is None:
        return nullable
    return (
        type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum
    )


def _summary(values):
    if not values:
        return None
    values = sorted(values)
    return {
        "count": len(values),
        "min": values[0],
        "p50": values[(len(values) - 1) // 2],
        "p95": values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)],
        "max": values[-1],
    }


class Observation:
    """Private correlation ids never leave this bounded per-case reducer."""

    def __init__(self):
        self.reply_ids = set()
        self.first_pcm = None
        self.nonzero_samples = 0
        self.answer_pcm_samples = 0
        self.answer_audio_seconds = 0.0
        self.reply_text_chars = 0
        self.invalid_chunks = 0
        self.reply_fragments = 0
        self.auxiliary_fragments = 0
        self.final_at = None
        self.source_endpoint_delay = None
        self.final_count = 0
        self.failed_fragments = 0
        self.dropped_fragments = 0
        self.controls = 0
        self.task_failures = 0
        self.task_cancellations = 0
        self._lock = threading.Lock()

    def event(self, event):
        from always_on_agent.events import EventKind

        if event.kind in {
            EventKind.CONTROL_STOP,
            EventKind.CONTROL_MODE,
            EventKind.CONTROL_CONFIRM,
            EventKind.CONTROL_DENY,
        }:
            self.controls += 1
        elif event.kind == EventKind.TASK_FAILED:
            self.task_failures += 1
        elif event.kind == EventKind.TASK_CANCELLED:
            self.task_cancellations += 1

    def diagnostic(self, observation):
        stage = getattr(observation.stage, "value", observation.stage)
        if stage == "playback_admitted":
            with self._lock:
                if observation.playback_kind == "reply":
                    if len(self.reply_ids) >= 512:
                        raise BenchmarkError("invalid_inputs")
                    self.reply_ids.add(observation.correlation_id)
                else:
                    self.auxiliary_fragments += 1

    def pcm(self, fragment_id, samples, clock=time.perf_counter, sample_rate=16000):
        import numpy as np

        values = np.asarray(samples)
        with self._lock:
            if (
                values.ndim != 1
                or values.dtype.kind != "f"
                or values.size > 30 * 192000
                or not np.all(np.isfinite(values))
            ):
                self.invalid_chunks += 1
                raise BenchmarkError("worker_failed")
            if fragment_id in self.reply_ids:
                count = int(np.count_nonzero(values))
                self.nonzero_samples += count
                self.answer_pcm_samples += int(values.size)
                self.answer_audio_seconds += values.size / sample_rate
                if count and self.first_pcm is None:
                    self.first_pcm = clock()


def measured_engine(config, models=None, renderer=None, observation=None):
    """FileReplay ASR + actual Sherpa synthesis/DSP, with zero audio starts."""
    _require_legacy_endpoint_lifecycle(
        {"sherpa": {name: getattr(config, name, "") for name in (
            "asr_final_backend", "asr_final_verifier_backend"
        )}}
    )
    from core.engine import PlaybackOutcome, PlaybackReceipt
    from core.tts_markup import prepare_speech_style
    from core.engines.file_replay import FileReplayEngine
    from core.engines.sherpa import SherpaOnnxEngine
    from core.engines._sherpa_models import create_recognizer_stream

    class MeasuredReplay(FileReplayEngine):
        def __init__(self):
            super().__init__(config)
            self.renderer = renderer or SherpaOnnxEngine(config)
            self.observation = observation or Observation()
            self.reply_classifier = lambda _fragment: False

        def start(self, callbacks):
            # Wrap only the exact selected final handoff; no label substitution.
            on_final = callbacks.on_final_result

            def final(result):
                self.observation.final_at = time.perf_counter()
                self.observation.final_count += 1
                spans = result.acoustic.spans if result.acoustic is not None else ()
                if spans:
                    span = spans[-1]
                    if (
                        span.endpoint_committed_at is not None
                        and span.speech_end_at is not None
                    ):
                        self.observation.source_endpoint_delay = max(
                            0.0, span.endpoint_committed_at - span.speech_end_at
                        )
                if on_final is not None:
                    on_final(result)
                else:
                    callbacks.on_final(result.text)

            callbacks = replace(callbacks, on_final_result=final)
            if models is None:
                super().start(callbacks)
            else:
                self._cb = callbacks
                (
                    self._recognizer,
                    self._final_recognizer,
                    self._final_verifier,
                    self._punct,
                    self._tts,
                ) = models
                self._stream = create_recognizer_stream(self._recognizer, config)
                self._started = True
                self._stopping = False
            if (
                self._tts is None
                or (config.asr_final_model and self._final_recognizer is None)
                or (config.asr_final_verifier_backend and self._final_verifier is None)
            ):
                raise BenchmarkError("invalid_inputs")
            self.renderer._tts = self._tts
            self.renderer._tts_sr = int(getattr(self._tts, "sample_rate", 0) or 22050)

        def warm(self):
            # Actual media warm seeds production normalize-RMS. None of these
            # model objects owns capture/playback streams or speaker identity.
            self.renderer.warm()
            if self._punct is not None:
                self._punct.add_punctuation("ok")
            if self._final_verifier is not None:
                self._final_verifier.warm()

        def speak(self, text, on_done=None):
            # Legacy acknowledgments/intents remain synthetic output work but
            # cannot satisfy the tracked answer-onset measurement.
            try:
                self.renderer._stop_speaking.clear()
                self.renderer._synthesize(
                    text, lambda pcm: self.observation.pcm(None, pcm)
                )
            finally:
                if on_done is not None:
                    on_done()

        def speak_tracked(self, speech, *, on_terminal, on_started=None):
            with self._receipt_lock:
                generation = self._play_generation
            if not self._generate_lock.acquire(blocking=False):
                self.observation.dropped_fragments += 1
                on_terminal(
                    PlaybackReceipt(speech.fragment_id, PlaybackOutcome.DROPPED)
                )
                return
            with self._receipt_lock:
                rejected = (
                    self._stopping
                    or not self._started
                    or generation != self._play_generation
                )
                if not rejected:
                    render_generation = self.renderer._speak_gen
                    self.renderer._stop_speaking.clear()
            if rejected:
                self._generate_lock.release()
                self.observation.dropped_fragments += 1
                on_terminal(
                    PlaybackReceipt(speech.fragment_id, PlaybackOutcome.DROPPED)
                )
                return
            if self.reply_classifier(speech.fragment_id):
                self.observation.reply_ids.add(speech.fragment_id)
            started = False
            speech_lifecycle_invoked = False
            outcome = PlaybackOutcome.FAILED
            text = speech.text
            try:
                prepared = (
                    prepare_speech_style(
                        text,
                        style=speech.style,
                        voices=config.tts_speaker_voices.keys(),
                        emotions=config.tts_emotion_speed_map.keys(),
                    )
                    if config.tts_markup
                    else None
                )
                if prepared is not None:
                    text, style = prepared.text, prepared.style
                else:
                    style = speech.style
                from core.tts_markup import style_to_directives

                directives = style_to_directives(style) if style is not None else None
                self._speaking.set()
                speech_lifecycle_invoked = True
                self._cb.on_speech_start()

                def write(pcm):
                    nonlocal started
                    import numpy as np

                    notify_started = False
                    with self._receipt_lock:
                        if generation != self._play_generation or self._stopping:
                            self.renderer._stop_speaking.set()
                            return
                        # Null-sink acceptance and revocation share one boundary.
                        # External runtime callbacks run after this lock releases.
                        self.observation.pcm(
                            speech.fragment_id, pcm, sample_rate=self.renderer._tts_sr
                        )
                        if not started and np.any(np.asarray(pcm) != 0):
                            started = True
                            notify_started = True
                    if notify_started and on_started is not None:
                        on_started(speech.fragment_id)

                self.renderer._synthesize(
                    text, write, gen=render_generation, directives=directives
                )
                outcome = (
                    PlaybackOutcome.COMPLETED if started else PlaybackOutcome.FAILED
                )
                if generation != self._play_generation or self._stopping:
                    outcome = PlaybackOutcome.INTERRUPTED
                if speech.fragment_id in self.observation.reply_ids:
                    self.observation.reply_fragments += 1
                    self.observation.reply_text_chars += len(text)
            except Exception:
                self.observation.failed_fragments += 1
            finally:
                self._speaking.clear()
                self._generate_lock.release()
                try:
                    if speech_lifecycle_invoked:
                        self._cb.on_speech_end()
                finally:
                    on_terminal(
                        PlaybackReceipt(
                            speech.fragment_id,
                            outcome,
                            safe_text_prefix=text
                            if outcome is PlaybackOutcome.COMPLETED
                            else "",
                        )
                    )

        def stop_speaking(self):
            with self._receipt_lock:
                self.renderer._stop_speaking.set()
                self.renderer._speak_gen += 1
                super().stop_speaking()

        def stop(self):
            with self._receipt_lock:
                self.renderer._stop_speaking.set()
                self.renderer._speak_gen += 1
                super().stop()

    return MeasuredReplay()


def _local_guard():
    """Python connection restriction, not OS/network/native isolation."""
    original = socket.socket.connect
    original_ex = socket.socket.connect_ex

    def allowed(address):
        return (
            isinstance(address, tuple)
            and len(address) >= 2
            and address[0] in ("127.0.0.1", "::1")
            and address[1] == 11434
        )

    def connect(sock, address):
        if not allowed(address):
            raise BenchmarkError("invalid_inputs")
        return original(sock, address)

    def connect_ex(sock, address):
        if not allowed(address):
            raise BenchmarkError("invalid_inputs")
        return original_ex(sock, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = lambda *_a, **_k: (_ for _ in ()).throw(
        BenchmarkError("invalid_inputs")
    )


def child_environment():
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "HOME",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "SYSTEMROOT",
            "WINDIR",
        }
    }
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        SPEAKER_TEST_LOG="0",
        SPEAKER_LIVE="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        NO_PROXY="127.0.0.1,localhost,::1",
    )
    return env


@contextmanager
def quiet_worker():
    logging.disable(logging.CRITICAL)
    with open(os.devnull, "w") as null:
        saved = [os.dup(1), os.dup(2)]
        try:
            os.dup2(null.fileno(), 1)
            os.dup2(null.fileno(), 2)
            yield
        finally:
            for fd, restore in zip((1, 2), saved):
                os.dup2(restore, fd)
                os.close(restore)


def _write_private(path, value, *, exclusive=False):
    with _REPORT_LOCK:
        _write_private_unlocked(path, value, exclusive=exclusive)


def _write_private_unlocked(path, value, *, exclusive=False):
    raw = canonical(value)
    if len(raw) > _MAX_REPORT:
        raise BenchmarkError("worker_failed")
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)


def _api_json(client, method, path, **kwargs):
    raw = bytearray()
    with client._client.stream(method, path, **kwargs) as response:
        response.raise_for_status()
        for part in response.iter_bytes():
            raw.extend(part)
            if len(raw) > 2 * 1024**2:
                raise BenchmarkError("invalid_inputs")
    value = _json(bytes(raw))
    if type(value) is not dict:
        raise BenchmarkError("invalid_inputs")
    return value


def _show_raw(client, alias):
    # Retain newer renderer/parser/system fields omitted by older SDK schemas.
    shown = _api_json(client, "POST", "/api/show", json={"model": alias})
    if any(shown.get(key) for key in ("remote_host", "remote_model")):
        raise BenchmarkError("invalid_inputs")
    return shown


def _show_setting_digests(shown):
    """Canonicalize only Ollama's parameter map and capability-set ordering."""
    from core.minicpm_identity import (
        _parameter_values,
        _modelfile_parameter_values,
        _template,
    )

    modelfile = str(shown.get("modelfile") or "").replace("\r\n", "\n")

    def parameters(values):
        if any(key != "stop" and len(entries) != 1 for key, entries in values.items()):
            raise BenchmarkError("invalid_inputs")
        return {key: sorted(entries) for key, entries in values.items()}

    # Preserve every other byte, including TEMPLATE/LICENSE/SYSTEM bodies.
    retained = []
    delimiter = ""
    for line in modelfile.splitlines(keepends=True):
        stripped = line.strip()
        if delimiter:
            retained.append(line)
            if delimiter in stripped:
                delimiter = ""
            continue
        parts = stripped.split(None, 1)
        if parts and parts[0].upper() == "PARAMETER":
            continue
        retained.append(line)
        if len(parts) == 2:
            value = parts[1]
            if value.count('"""') % 2:
                delimiter = '"""'
            elif value.startswith('"') and not value.endswith('"'):
                delimiter = '"'
    if delimiter:
        raise BenchmarkError("invalid_inputs")
    capabilities = shown.get("capabilities")
    if type(capabilities) is not list or any(
        type(item) is not str for item in capabilities
    ):
        raise BenchmarkError("invalid_inputs")
    behavior = {
        "modelfile_without_parameter_lines": "".join(retained),
        "exact_effective_template": _template(shown),
        "exact_native_template": shown.get("template"),
        "parameters": parameters(_parameter_values(shown)),
        "modelfile_parameters": parameters(_modelfile_parameter_values(modelfile)),
        "capabilities": sorted(capabilities),
    }
    remaining = {
        key: value
        for key, value in shown.items()
        if key
        not in {"modified_at", "modelfile", "parameters", "capabilities", "template"}
    }
    return digest(behavior), digest(
        {"behavior": behavior, "other_exact_show_fields": remaining}
    )


def _ollama_binding(config, model_root=None):
    """Bind actual local multi-layer models; no adoption-verifier relaxation.

    New daemon releases can expose more capabilities and multiple FROM blobs.
    Bind the local manifest and every actual layer/config byte instead of
    pretending the old one-FROM/exact-capability adoption check passed.
    """
    model_root = _absolute(model_root) if model_root is not None else None
    import ollama
    from core.minicpm_identity import (
        MINICPM_Q8_CONTRACT,
        _from_values,
        verify_minicpm_q8_identity,
    )

    client = ollama.Client(
        host="http://127.0.0.1:11434",
        timeout=5.0,
        headers={"authorization": "Bearer speaker-local-evaluation"},
        trust_env=False,
        follow_redirects=False,
    )
    roles = {}
    try:
        installed = client.list().models
        for role in ("main", "fast"):
            alias = config["llm"][role + "_model"]
            if type(alias) is not str or not re.fullmatch(
                r"[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+", alias
            ):
                raise BenchmarkError("invalid_inputs")
            item = next((m for m in installed if (m.model or m.name) == alias), None)
            manifest_digest = (
                str(item.digest).removeprefix("sha256:") if item is not None else ""
            )
            if not _HASH.fullmatch(manifest_digest):
                raise BenchmarkError("invalid_inputs")
            shown = _show_raw(client, alias)
            capabilities = shown.get("capabilities")
            if (
                not isinstance(capabilities, (list, tuple))
                or "completion" not in capabilities
            ):
                raise BenchmarkError("invalid_inputs")
            from_values = _from_values(shown.get("modelfile") or "")
            if not 1 <= len(from_values) <= 4:
                raise BenchmarkError("invalid_inputs")
            from_paths = [Path(value.strip('"')) for value in from_values]
            blobs = from_paths[0].parent
            if (
                not blobs.is_absolute()
                or blobs.name != "blobs"
                or any(
                    path.parent != blobs
                    or not re.fullmatch(r"sha256-[0-9a-f]{64}", path.name)
                    for path in from_paths
                )
            ):
                raise BenchmarkError("invalid_inputs")
            if model_root is not None:
                canonical_blobs = _absolute(model_root / "blobs")
                if any(
                    path.is_symlink()
                    or path.resolve(strict=True) != canonical_blobs / path.name
                    for path in from_paths
                ):
                    raise BenchmarkError("invalid_inputs")
                blobs = canonical_blobs
            name, tag = alias.rsplit(":", 1)
            manifest_path = (
                blobs.parent
                / "manifests"
                / "registry.ollama.ai"
                / "library"
                / name
                / tag
            )
            raw = _read(manifest_path, 2 * 1024**2)
            if hashlib.sha256(raw).hexdigest() != manifest_digest:
                raise BenchmarkError("invalid_inputs")
            manifest = _json(raw)
            if (
                type(manifest) is not dict
                or not {"schemaVersion", "mediaType", "config", "layers"}
                <= set(manifest)
                or set(manifest)
                - {"schemaVersion", "mediaType", "config", "layers", "format", "runner"}
                or any(
                    type(manifest[key]) is not str or not 1 <= len(manifest[key]) <= 256
                    for key in ("format", "runner")
                    if key in manifest
                )
                or type(manifest["schemaVersion"]) is not int
                or manifest["schemaVersion"] != 2
                or type(manifest["layers"]) is not list
                or not 1 <= len(manifest["layers"]) <= 32
            ):
                raise BenchmarkError("invalid_inputs")
            inventory = []
            total = 0
            for layer in [manifest["config"], *manifest["layers"]]:
                if (
                    type(layer) is not dict
                    or not {"mediaType", "digest", "size"} <= set(layer)
                    or set(layer) - {"mediaType", "digest", "size", "from"}
                    or (
                        "from" in layer
                        and (
                            type(layer["from"]) is not str or len(layer["from"]) > 4096
                        )
                    )
                    or type(layer["digest"]) is not str
                    or not re.fullmatch(r"sha256:[0-9a-f]{64}", layer["digest"])
                    or type(layer["size"]) is not int
                    or not 0 < layer["size"] <= 8 * 1024**3
                ):
                    raise BenchmarkError("invalid_inputs")
                checksum = layer["digest"].split(":", 1)[1]
                blob = blobs / ("sha256-" + checksum)
                actual, size = _file_digest(blob, 8 * 1024**3)
                total += size
                if actual != checksum or size != layer["size"] or total > 32 * 1024**3:
                    raise BenchmarkError("invalid_inputs")
                inventory.append([checksum, size])
            if any(
                path.name.removeprefix("sha256-") not in {row[0] for row in inventory}
                for path in from_paths
            ):
                raise BenchmarkError("invalid_inputs")
            if model_root is not None and any(
                path.is_symlink() or path.resolve(strict=True) != blobs / path.name
                for path in from_paths
            ):
                raise BenchmarkError("invalid_inputs")
            config_digest, show_digest = _show_setting_digests(shown)
            canonical = (
                verify_minicpm_q8_identity(
                    show=lambda model: _show_raw(client, model)
                ).ok
                if alias == MINICPM_Q8_CONTRACT.alias
                else None
            )
            roles[role] = {
                "manifest_sha256": manifest_digest,
                "layer_tuple_sha256": digest(inventory),
                "layer_files": len(inventory),
                "layer_bytes": total,
                "effective_config_sha256": config_digest,
                "effective_show_sha256": show_digest,
                "completion": True,
                "vision": "vision" in capabilities,
                "canonical_minicpm_adoption_check": canonical,
            }
        return SimpleNamespace(sha256=digest(roles), role_contracts=roles)
    finally:
        client._client.close()


def process_resources():
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "process_peak_rss_bytes": int(usage.ru_maxrss) * 1024,
        "process_cpu_seconds": usage.ru_utime + usage.ru_stime,
    }


def ollama_residency(config):
    import ollama

    client = ollama.Client(
        host="http://127.0.0.1:11434",
        timeout=5.0,
        headers={"authorization": "Bearer speaker-local-evaluation"},
        trust_env=False,
        follow_redirects=False,
    )
    try:
        version = _api_json(client, "GET", "/api/version").get("version")
        if type(version) is not str or not re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?", version
        ):
            raise BenchmarkError("invalid_inputs")
        models = client.ps().models
        roles = {}
        for role in ("main", "fast"):
            alias = config["llm"][role + "_model"]
            selected = next((m for m in models if (m.model or m.name) == alias), None)
            roles[role] = {
                "resident": selected is not None,
                **{
                    field: int(getattr(selected, native))
                    if selected is not None and getattr(selected, native) is not None
                    else None
                    for field, native in (
                        ("size_bytes", "size"),
                        ("size_vram_bytes", "size_vram"),
                        ("context_length", "context_length"),
                    )
                },
            }
        return {"daemon_version": version, "roles": roles}
    finally:
        client._client.close()


def validate_residency(value):
    if (
        type(value) is not dict
        or set(value) != {"daemon_version", "roles"}
        or type(value["daemon_version"]) is not str
        or not re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?", value["daemon_version"]
        )
        or type(value["roles"]) is not dict
        or set(value["roles"]) != {"main", "fast"}
    ):
        raise BenchmarkError("worker_failed")
    for role in value["roles"].values():
        if (
            type(role) is not dict
            or set(role)
            != {"resident", "size_bytes", "size_vram_bytes", "context_length"}
            or type(role["resident"]) is not bool
        ):
            raise BenchmarkError("worker_failed")
        for key in ("size_bytes", "size_vram_bytes", "context_length"):
            if role[key] is not None and (
                type(role[key]) is not int or not 0 <= role[key] <= 128 * 1024**3
            ):
                raise BenchmarkError("worker_failed")


def run_pipeline(
    corpus, config, *, repeats, startup, case_timeout, publish, clock=time.perf_counter
):
    _require_legacy_endpoint_lifecycle(config)
    import numpy as np
    import ollama
    from always_on_agent.events import Mode
    from core.app import build_runtime
    from core.engines.sherpa import SherpaConfig
    from core.llm import OllamaLLM
    from core.llm_factory import build_llms
    from core.routing import build_router
    from core.metrics import (
        HANDLED_LOCAL,
        LLM_REQUESTED,
        LLM_FIRST_TOKEN,
        TTS_TEXT_READY,
        TTS_ADMITTED,
    )
    from tools.conversation_eval.provenance import LOCAL_OLLAMA_HEADERS

    args = SimpleNamespace(
        llm="ollama",
        model=None,
        fast_model=None,
        ollama_client_headers=dict(LOCAL_OLLAMA_HEADERS),
        ollama_timeout=case_timeout,
    )
    main, fast = build_llms(args, config)
    if type(main) is not OllamaLLM or type(fast) is not OllamaLLM:
        raise BenchmarkError("invalid_inputs")
    for client in (main, fast):
        client._async_client_factory = functools.partial(
            ollama.AsyncClient, trust_env=False, follow_redirects=False
        )
        client._client = ollama.Client(
            **client._client_kwargs(), trust_env=False, follow_redirects=False
        )
    sherpa = SherpaConfig.from_dict(config.get("sherpa", {}))
    models = renderer = None
    provider_counts = {"chunks": 0, "chars": 0, "requests": 0}
    for client in (main, fast):
        original_stream = client.stream

        def counted_stream(*args, _stream=original_stream, **kwargs):
            provider_counts["requests"] += 1
            source = _stream(*args, **kwargs)
            try:
                for piece in source:
                    if piece:
                        provider_counts["chunks"] += 1
                        provider_counts["chars"] += len(piece)
                    yield piece
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    close()

        client.stream = counted_stream
    rows = []
    startup_seconds = None
    warm_finished = False

    def checkpoint(status="ok"):
        result = {
            "status": status,
            "planned": len(corpus.clips) * repeats,
            "completed": len(rows),
            "rows": list(rows),
            "startup_seconds": startup_seconds,
            "warm_plan_finished": warm_finished,
            "failure_stage": _WORKER_STAGE if status != "ok" else None,
            "resources": process_resources(),
            "server_after": None,
        }
        publish(result)
        return result

    checkpoint()
    try:
        for repeat in range(repeats):
            for ordinal, clip in enumerate(corpus.clips):
                observed = Observation()
                engine = measured_engine(sherpa, models, renderer, observed)
                row_config = (
                    config
                    if not rows and startup == "configured"
                    else {**config, "warm_on_start": False}
                )
                runtime = build_runtime(
                    row_config,
                    engine=engine,
                    llm=main,
                    fast_llm=fast,
                    router=build_router(config),
                    start_mode=Mode.ASSISTANT,
                )

                # Runtime owns this exact per-fragment binding before entry.
                # It excludes auxiliary/latency_ack and stale/non-task output.
                engine.reply_classifier = lambda fragment, _runtime=runtime: (
                    fragment in _runtime._playback_metric_turns
                )
                runtime.bus.subscribe(observed.event)

                def timed_out():
                    try:
                        checkpoint("case_timeout")
                    finally:
                        # Receipt failure cannot extend this case's watchdog.
                        os._exit(71)

                watchdog = threading.Timer(case_timeout, timed_out)
                watchdog.daemon = True
                watchdog.start()
                began = clock()
                try:
                    runtime.start(run_bus=True)
                    if not runtime.warm_ready.wait(timeout=case_timeout):
                        raise BenchmarkError("case_timeout")
                    warm_finished = True
                    if models is None:
                        startup_seconds = clock() - began
                        models = (
                            engine._recognizer,
                            engine._final_recognizer,
                            engine._final_verifier,
                            engine._punct,
                            engine._tts,
                        )
                        renderer = engine.renderer
                    # Fresh runtime/session/ASR stream; resident model objects,
                    # TTS scalar gain and native caches remain shared across rows.
                    raw = _read(clip.path, MAX_WAV_BYTES)
                    pcm, _frames = _pcm(raw)
                    samples = (
                        np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
                    )
                    provider_counts.update(chunks=0, chars=0, requests=0)
                    started = clock()
                    engine.replay_samples(samples, 16000)
                    decoded = clock()
                    idle = runtime.wait_idle(
                        timeout=max(0.001, case_timeout - (clock() - began))
                    )
                    finished = clock()
                    runtime.metrics.close_turn()
                    records = runtime.metrics.records()
                    stamps = records[-1].stamps if records else {}

                    def interval(a, b):
                        return (
                            stamps[b] - stamps[a]
                            if a in stamps and b in stamps and stamps[b] >= stamps[a]
                            else None
                        )

                    first, final = observed.first_pcm, observed.final_at
                    from tools.english_asr_scoring import score_english_normalized_pairs

                    pairs = [(clip.reference, engine.last_final)]
                    asr_accuracy = {
                        "legacy": _accuracy_pairs(pairs),
                        "english": score_english_normalized_pairs(pairs),
                    }
                    row = {
                        "repeat": repeat,
                        "ordinal": ordinal,
                        "phase": "first" if not rows else "subsequent",
                        "responded": first is not None,
                        "idle": idle,
                        "final_count": observed.final_count,
                        "reply_fragments": observed.reply_fragments,
                        "auxiliary_fragments": observed.auxiliary_fragments,
                        "invalid_chunks": observed.invalid_chunks,
                        "failed_fragments": observed.failed_fragments,
                        "dropped_fragments": observed.dropped_fragments,
                        "nonzero_samples": observed.nonzero_samples,
                        "answer_pcm_samples": observed.answer_pcm_samples,
                        "answer_audio_seconds": observed.answer_audio_seconds,
                        "reply_text_chars": observed.reply_text_chars,
                        "control_events": observed.controls,
                        "task_failures": observed.task_failures,
                        "task_cancellations": observed.task_cancellations,
                        "handled_local": HANDLED_LOCAL in stamps,
                        "asr_accuracy": asr_accuracy,
                        "provider_text_chunks": provider_counts["chunks"],
                        "provider_text_chars": provider_counts["chars"],
                        "provider_requests": provider_counts["requests"],
                        "pcm_to_final_seconds": final - started
                        if final is not None
                        else None,
                        "pcm_to_first_answer_pcm_seconds": first - started
                        if first is not None
                        else None,
                        "final_to_first_answer_pcm_seconds": first - final
                        if first is not None and final is not None
                        else None,
                        "pcm_to_idle_seconds": finished - started,
                        "asr_decode_seconds": decoded - started,
                        "source_endpoint_delay_seconds": observed.source_endpoint_delay,
                        "model_request_to_first_token_seconds": interval(
                            LLM_REQUESTED, LLM_FIRST_TOKEN
                        ),
                        "first_token_to_speakable_text_seconds": interval(
                            LLM_FIRST_TOKEN, TTS_TEXT_READY
                        ),
                        "speakable_text_to_tts_admission_seconds": interval(
                            TTS_TEXT_READY, TTS_ADMITTED
                        ),
                        "tts_admission_to_first_answer_pcm_seconds": first
                        - stamps[TTS_ADMITTED]
                        if first is not None and TTS_ADMITTED in stamps
                        else None,
                    }
                    if observed.final_count > 1:
                        # Retain observed PCM/counts but refuse single-final
                        # alignment and quality attribution for this failed row.
                        row["pcm_to_final_seconds"] = None
                        row["final_to_first_answer_pcm_seconds"] = None
                        row["asr_accuracy"] = None
                    rows.append(row)
                    checkpoint()
                    if not idle or observed.invalid_chunks or observed.final_count > 1:
                        raise BenchmarkError("worker_failed")
                finally:
                    runtime.stop()
                    watchdog.cancel()
                    if renderer is not None:
                        renderer._stop_speaking.clear()
                del runtime
                del engine
                gc.collect()
        return checkpoint()
    finally:
        for client in (main, fast):
            client._client._client.close()


def validate_request(request):
    if type(request) is not dict or set(request) != {
        "config_root",
        "manifest",
        "source_root",
        "device",
        "variant",
        "startup",
        "repeats",
        "case_timeout",
        "ollama_model_root",
    }:
        raise BenchmarkError("invalid_inputs")
    if request["ollama_model_root"] is not None and (
        type(request["ollama_model_root"]) is not str
        or not 1 <= len(request["ollama_model_root"]) <= 4096
        or not Path(request["ollama_model_root"]).is_absolute()
    ):
        raise BenchmarkError("invalid_inputs")
    for key in ("config_root", "manifest", "source_root"):
        if (
            type(request[key]) is not str
            or not 1 <= len(request[key]) <= 4096
            or not Path(request[key]).is_absolute()
        ):
            raise BenchmarkError("invalid_inputs")
    if (
        type(request["device"]) is not str
        or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", request["device"])
        or request["device"] == "auto"
        or request["variant"] not in VARIANTS
        or request["startup"] not in {"configured", "cold"}
        or type(request["repeats"]) is not int
        or not 1 <= request["repeats"] <= 10
        or not _finite(request["case_timeout"])
        or not 1 <= request["case_timeout"] <= 600
    ):
        raise BenchmarkError("invalid_inputs")


def worker(request, publish):
    global _WORKER_STAGE
    validate_request(request)
    _WORKER_STAGE = "source"
    from tools.english_asr_scoring import scoring_metadata

    scoring_before = scoring_metadata()
    before = source_binding()
    if before["clean"] is not True:
        raise BenchmarkError("invalid_inputs")
    _WORKER_STAGE = "config"
    root = Path(request["config_root"])
    config, settings = resolve_config(root, request["device"], request["variant"])
    _WORKER_STAGE = "corpus"
    corpus = load_corpus(request["manifest"], request["source_root"])
    from tools.bench.production import active_artifact_identity, runtime_identities

    _WORKER_STAGE = "artifacts"
    artifacts = active_artifact_identity(config, config_root=root)
    packages = runtime_identities(config)
    _WORKER_STAGE = "ollama"
    model_identity = _ollama_binding(config, request["ollama_model_root"])
    binding = {
        **settings,
        "source": before,
        "corpus_sha256": corpus.digest,
        "manifest_sha256": corpus.manifest_digest,
        "scoring_sha256": digest(scoring_before),
        "corpus_kind": corpus.kind,
        "corpus_clips": len(corpus.clips),
        "corpus_seconds": corpus.seconds,
        "audio_artifacts_sha256": artifacts[0],
        "audio_artifact_files": artifacts[2],
        "audio_artifact_bytes": artifacts[3],
        "runtime": packages,
        "ollama_sha256": model_identity.sha256,
        "ollama_roles": model_identity.role_contracts,
        "ollama_model_root_sha256": digest(request["ollama_model_root"]),
        "server_before": ollama_residency(config),
    }
    _WORKER_STAGE = "pipeline"
    result = run_pipeline(
        corpus,
        config,
        repeats=request["repeats"],
        startup=request["startup"],
        case_timeout=request["case_timeout"],
        publish=lambda progress: publish({**progress, "binding": binding}),
    )
    _WORKER_STAGE = "verification"
    verify_corpus(corpus)
    after_config, after_settings = resolve_config(
        root, request["device"], request["variant"]
    )
    if (
        after_settings != settings
        or active_artifact_identity(after_config, config_root=root) != artifacts
        or runtime_identities(config) != packages
        or _ollama_binding(config, request["ollama_model_root"]).sha256
        != model_identity.sha256
        or source_binding() != before
        or scoring_metadata() != scoring_before
    ):
        raise BenchmarkError("inputs_changed")
    return {
        **result,
        "binding": binding,
        "inputs_verified_after": True,
        "resources": process_resources(),
        "server_after": ollama_residency(config),
    }


def validate_accuracy(value):
    keys = {
        "clips",
        "exact",
        "empty",
        "word_errors",
        "reference_words",
        "character_errors",
        "reference_characters",
        "wer",
        "cer",
    }
    if type(value) is not dict or set(value) != {"legacy", "english"}:
        raise BenchmarkError("worker_failed")
    for scores in value.values():
        if type(scores) is not dict or set(scores) != keys:
            raise BenchmarkError("worker_failed")
        if any(
            type(scores[k]) is not int or not 0 <= scores[k] <= 16384
            for k in keys - {"wer", "cer"}
        ) or any(not _finite(scores[k], maximum=16384) for k in ("wer", "cer")):
            raise BenchmarkError("worker_failed")
        if (
            scores["clips"] != 1
            or scores["exact"] > 1
            or scores["empty"] > 1
            or not math.isclose(
                scores["wer"],
                scores["word_errors"] / max(1, scores["reference_words"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                scores["cer"],
                scores["character_errors"] / max(1, scores["reference_characters"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise BenchmarkError("worker_failed")


def validate_report(result, request):
    """Exact typed scalar rows; arbitrary worker keys/text are never published."""
    if type(result) is not dict or result.get("status") not in _CODES:
        raise BenchmarkError("worker_failed")
    expected = {
        "status",
        "planned",
        "completed",
        "rows",
        "startup_seconds",
        "warm_plan_finished",
        "failure_stage",
        "resources",
        "server_after",
        "binding",
        "inputs_verified_after",
    }
    if (
        set(result) != expected
        or type(result["rows"]) is not list
        or len(result["rows"]) > 2560
    ):
        raise BenchmarkError("worker_failed")
    for name in ("planned", "completed"):
        if type(result[name]) is not int or not 0 <= result[name] <= 2560:
            raise BenchmarkError("worker_failed")
    if (
        result["completed"] != len(result["rows"])
        or result["completed"] > result["planned"]
    ):
        raise BenchmarkError("worker_failed")
    if (
        type(result["warm_plan_finished"]) is not bool
        or type(result["inputs_verified_after"]) is not bool
        or not _finite(result["startup_seconds"], nullable=True)
    ):
        raise BenchmarkError("worker_failed")
    if result["status"] == "ok" and (
        result["completed"] != result["planned"]
        or result["inputs_verified_after"] is not True
        or result["server_after"] is None
        or result["failure_stage"] is not None
    ):
        raise BenchmarkError("worker_failed")
    if result["failure_stage"] is not None and result["failure_stage"] not in _STAGES:
        raise BenchmarkError("worker_failed")
    resources = result["resources"]
    if (
        type(resources) is not dict
        or set(resources) != {"process_peak_rss_bytes", "process_cpu_seconds"}
        or type(resources["process_peak_rss_bytes"]) is not int
        or not 0 <= resources["process_peak_rss_bytes"] <= 256 * 1024**3
        or not _finite(resources["process_cpu_seconds"], maximum=32 * 86400)
    ):
        raise BenchmarkError("worker_failed")
    if result["server_after"] is not None:
        validate_residency(result["server_after"])
    fields = {
        "repeat",
        "ordinal",
        "phase",
        "responded",
        "idle",
        "final_count",
        "reply_fragments",
        "auxiliary_fragments",
        "invalid_chunks",
        "failed_fragments",
        "dropped_fragments",
        "nonzero_samples",
        "answer_pcm_samples",
        "answer_audio_seconds",
        "reply_text_chars",
        "provider_text_chunks",
        "provider_text_chars",
        "provider_requests",
        "control_events",
        "task_failures",
        "task_cancellations",
        "handled_local",
        "asr_accuracy",
        *_METRICS,
    }
    for row in result["rows"]:
        if (
            type(row) is not dict
            or set(row) != fields
            or row["phase"] not in {"first", "subsequent"}
        ):
            raise BenchmarkError("worker_failed")
        if any(
            type(row[k]) is not bool for k in ("responded", "idle", "handled_local")
        ) or any(
            not _finite(row[k], nullable=True)
            for k in (*_METRICS, "answer_audio_seconds")
        ):
            raise BenchmarkError("worker_failed")
        if row["asr_accuracy"] is None:
            if row["final_count"] <= 1:
                raise BenchmarkError("worker_failed")
        else:
            validate_accuracy(row["asr_accuracy"])
        for name in (
            fields
            - set(_METRICS)
            - {
                "phase",
                "responded",
                "idle",
                "handled_local",
                "answer_audio_seconds",
                "asr_accuracy",
            }
        ):
            if type(row[name]) is not int or not 0 <= row[name] <= 10**9:
                raise BenchmarkError("worker_failed")
    binding = result["binding"]
    if type(binding) is not dict or set(binding) != {
        "resolved_config_sha256",
        "execution_config_sha256",
        "selection_sha256",
        "asr_settings_sha256",
        "llm_settings_sha256",
        "safety_policy_sha256",
        "source",
        "corpus_sha256",
        "manifest_sha256",
        "scoring_sha256",
        "corpus_kind",
        "corpus_clips",
        "corpus_seconds",
        "audio_artifacts_sha256",
        "audio_artifact_files",
        "audio_artifact_bytes",
        "runtime",
        "ollama_sha256",
        "ollama_roles",
        "ollama_model_root_sha256",
        "server_before",
    }:
        raise BenchmarkError("worker_failed")
    # Validate nested public receipts recursively, permitting only fixed schema
    # keys and bounded public version strings from the production identity API.
    for key in (
        "resolved_config_sha256",
        "execution_config_sha256",
        "selection_sha256",
        "asr_settings_sha256",
        "llm_settings_sha256",
        "safety_policy_sha256",
        "corpus_sha256",
        "manifest_sha256",
        "scoring_sha256",
        "audio_artifacts_sha256",
        "ollama_sha256",
        "ollama_model_root_sha256",
    ):
        if type(binding[key]) is not str or not _HASH.fullmatch(binding[key]):
            raise BenchmarkError("worker_failed")
    if (
        binding["corpus_kind"] not in {"recorded-legacy", "owner-scripted"}
        or type(binding["corpus_clips"]) is not int
        or not 1 <= binding["corpus_clips"] <= 256
        or result["planned"] != binding["corpus_clips"] * request["repeats"]
    ):
        raise BenchmarkError("worker_failed")
    for name in ("audio_artifact_files", "audio_artifact_bytes"):
        if type(binding[name]) is not int or not 1 <= binding[name] <= 32 * 1024**3:
            raise BenchmarkError("worker_failed")
    if not _finite(binding["corpus_seconds"]):
        raise BenchmarkError("worker_failed")
    for index, row in enumerate(result["rows"]):
        if (
            row["ordinal"] != index % binding["corpus_clips"]
            or row["repeat"] != index // binding["corpus_clips"]
            or row["phase"] != ("first" if index == 0 else "subsequent")
            or row["responded"] != (row["pcm_to_first_answer_pcm_seconds"] is not None)
        ):
            raise BenchmarkError("worker_failed")
    validate_residency(binding["server_before"])
    source = binding["source"]
    if (
        type(source) is not dict
        or set(source) != {"sha256", "files", "bytes", "revision", "clean"}
        or type(source["sha256"]) is not str
        or not _HASH.fullmatch(source["sha256"])
        or type(source["clean"]) is not bool
        or (result["status"] == "ok" and source["clean"] is not True)
        or type(source["revision"]) is not str
        or not re.fullmatch(r"[0-9a-f]{40,64}", source["revision"])
        or type(source["files"]) is not int
        or source["files"] != len(_SOURCE_FILES)
        or type(source["bytes"]) is not int
        or not 0 < source["bytes"] <= 8 * 1024**2
    ):
        raise BenchmarkError("worker_failed")
    roles = binding["ollama_roles"]
    if type(roles) is not dict or set(roles) != {"main", "fast"}:
        raise BenchmarkError("worker_failed")
    for role in roles.values():
        if (
            type(role) is not dict
            or set(role)
            != {
                "manifest_sha256",
                "layer_tuple_sha256",
                "layer_files",
                "layer_bytes",
                "effective_config_sha256",
                "effective_show_sha256",
                "completion",
                "vision",
                "canonical_minicpm_adoption_check",
            }
            or any(
                type(role[k]) is not str or not _HASH.fullmatch(role[k])
                for k in (
                    "manifest_sha256",
                    "layer_tuple_sha256",
                    "effective_config_sha256",
                    "effective_show_sha256",
                )
            )
            or type(role["layer_files"]) is not int
            or not 1 <= role["layer_files"] <= 33
            or type(role["layer_bytes"]) is not int
            or not 1 <= role["layer_bytes"] <= 32 * 1024**3
            or role["completion"] is not True
            or type(role["vision"]) is not bool
            or (
                role["canonical_minicpm_adoption_check"] is not None
                and type(role["canonical_minicpm_adoption_check"]) is not bool
            )
        ):
            raise BenchmarkError("worker_failed")
    runtime = binding["runtime"]
    if (
        type(runtime) is not dict
        or set(runtime) != {"python", "distributions"}
        or type(runtime["python"]) is not dict
        or set(runtime["python"]) != {"version", "executable_sha256"}
        or type(runtime["python"]["version"]) is not str
        or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", runtime["python"]["version"])
        or type(runtime["python"]["executable_sha256"]) is not str
        or not _HASH.fullmatch(runtime["python"]["executable_sha256"])
    ):
        raise BenchmarkError("worker_failed")
    packages = runtime["distributions"]
    if (
        type(packages) is not dict
        or not {"numpy", "sherpa_onnx", "onnxruntime", "ollama"} <= set(packages)
        or set(packages)
        - {
            "numpy",
            "sherpa_onnx",
            "onnxruntime",
            "ollama",
            "faster_whisper",
            "ctranslate2",
        }
    ):
        raise BenchmarkError("worker_failed")
    for package in packages.values():
        if (
            type(package) is not dict
            or set(package) != {"version", "record_sha256"}
            or type(package["version"]) is not str
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}", package["version"]
            )
            or type(package["record_sha256"]) is not str
            or not _HASH.fullmatch(package["record_sha256"])
        ):
            raise BenchmarkError("worker_failed")


def _stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2.0)


def run_isolated(request, *, scratch, timeout):
    with tempfile.TemporaryDirectory(prefix="response-", dir=scratch) as temporary:
        report = Path(temporary) / "progress.json"
        child = {**request, "report": str(report)}
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                "tools.recording_response_benchmark",
                "--worker",
            ],
            cwd=ROOT,
            env=child_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.communicate(canonical(child), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop(process)
        except BaseException:
            _stop(process)
            raise
        try:
            result = json.loads(_read(report, _MAX_REPORT))
            if set(result) == {
                "status",
                "failure_stage",
                "planned_unknown",
                "confirmed_completed",
            }:
                if (
                    result["status"] != "invalid_inputs"
                    or result["failure_stage"] not in _STAGES
                    or result["planned_unknown"] is not True
                    or type(result["confirmed_completed"]) is not int
                    or result["confirmed_completed"] != 0
                ):
                    raise BenchmarkError("worker_failed")
                return result
            result.setdefault("inputs_verified_after", False)
            if timed_out:
                result["status"] = "cell_timeout"
            elif process.returncode != 0 and result.get("status") == "ok":
                result["status"] = "worker_failed"
            validate_report(result, request)
        except Exception:
            return {
                "status": "cell_timeout" if timed_out else "worker_failed",
                "planned_unknown": True,
                "confirmed_completed": 0,
            }
        return result


def aggregate(result):
    if "rows" not in result:
        return result
    groups = {}
    for phase in ("first", "subsequent"):
        rows = [row for row in result["rows"] if row["phase"] == phase]
        groups[phase] = {
            "cases": len(rows),
            "responses": sum(row["responded"] for row in rows),
            "no_answer_pcm": sum(not row["responded"] for row in rows),
            "handled_local": sum(row["handled_local"] for row in rows),
            "control_events": sum(row["control_events"] for row in rows),
            "task_failures": sum(row["task_failures"] for row in rows),
            "task_cancellations": sum(row["task_cancellations"] for row in rows),
            "metrics": {
                key: _summary([row[key] for row in rows if row[key] is not None])
                for key in _METRICS
            },
        }
    return {**result, "summary": groups}


def compare_paired(baseline, candidate):
    """Deltas of the same responding cases, never sums of component medians."""
    for key in (
        "corpus_sha256",
        "manifest_sha256",
        "corpus_clips",
        "asr_settings_sha256",
        "llm_settings_sha256",
    ):
        if baseline["binding"][key] != candidate["binding"][key]:
            raise BenchmarkError("invalid_inputs")
    if any(
        result["status"] != "ok" or result["inputs_verified_after"] is not True
        for result in (baseline, candidate)
    ):
        raise BenchmarkError("invalid_inputs")
    left = {(r["repeat"], r["ordinal"]): r for r in baseline["rows"]}
    right = {(r["repeat"], r["ordinal"]): r for r in candidate["rows"]}
    if set(left) != set(right):
        raise BenchmarkError("invalid_inputs")
    keys = [key for key in left if left[key]["responded"] and right[key]["responded"]]
    return {
        "attempted_pairs": len(left),
        "responding_pairs": len(keys),
        "baseline_responses": sum(r["responded"] for r in left.values()),
        "candidate_responses": sum(r["responded"] for r in right.values()),
        "response_gains": sum(
            not left[k]["responded"] and right[k]["responded"] for k in left
        ),
        "response_losses": sum(
            left[k]["responded"] and not right[k]["responded"] for k in left
        ),
        "candidate_minus_baseline_first_pcm_seconds": _summary(
            [
                right[k]["pcm_to_first_answer_pcm_seconds"]
                - left[k]["pcm_to_first_answer_pcm_seconds"]
                for k in keys
            ]
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--ollama-model-root",
        type=Path,
        help="explicit canonical local model-cache target; symlink leaves remain forbidden",
    )
    parser.add_argument("--device", default="cpu_laptop")
    parser.add_argument("--variant", choices=VARIANTS, default="current")
    parser.add_argument(
        "--startup", choices=("configured", "cold"), default="configured"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--case-timeout", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        with quiet_worker():
            latest = []

            def publish(value):
                latest[:] = [value]
                _write_private(report, value)

            try:
                request = json.loads(sys.stdin.buffer.read(65537))
                report = Path(request.pop("report"))
                _local_guard()
                result = worker(request, publish)
                _write_private(report, result)
                return 0
            except BaseException as error:
                if latest:
                    failed = {
                        **latest[0],
                        "status": str(error)
                        if type(error) is BenchmarkError and str(error) in _CODES
                        else "worker_failed",
                        "inputs_verified_after": False,
                        "failure_stage": _WORKER_STAGE,
                    }
                    _write_private(report, failed)
                elif "report" in locals():
                    _write_private(
                        report,
                        {
                            "status": "invalid_inputs",
                            "failure_stage": _WORKER_STAGE,
                            "planned_unknown": True,
                            "confirmed_completed": 0,
                        },
                    )
                return 1
    try:
        if sys.platform != "linux":
            raise BenchmarkError("invalid_inputs")
        if (
            any(
                value is None
                for value in (
                    args.config_root,
                    args.manifest,
                    args.source_root,
                    args.scratch,
                    args.out,
                )
            )
            or not 1 <= args.repeats <= 10
            or not 1 <= args.case_timeout <= 600
            or not 1 <= args.timeout <= 7200
        ):
            raise BenchmarkError("invalid_inputs")
        request = {
            "config_root": str(_absolute(args.config_root)),
            "manifest": str(_absolute(args.manifest)),
            "source_root": str(_absolute(args.source_root)),
            "device": args.device,
            "variant": args.variant,
            "startup": args.startup,
            "repeats": args.repeats,
            "case_timeout": args.case_timeout,
            "ollama_model_root": str(_absolute(args.ollama_model_root))
            if args.ollama_model_root is not None
            else None,
        }
        result = run_isolated(
            request, scratch=_absolute(args.scratch), timeout=args.timeout
        )
        payload = {
            "schema_version": 1,
            "kind": "recording_response_after_pcm_null_sink",
            "variant": args.variant,
            "startup": args.startup,
            "method": {
                "tail_silence_seconds": 0.6,
                "audio_devices": False,
                "cloud": False,
                "tool_effects": False,
                "first_pcm": "first_finite_nonzero_production_synthesis_DSP_null_sink_reply_only",
                "source_scope": "explicit_selected_replay_runtime_llm_synthesis_seam_not_full_import_closure",
                "case_state": "fresh_runtime_session_ASR_stream_shared_native_models_and_TTS_gain",
                "warm": "configured_plan_once_before_first_case_or_explicitly_disabled",
                "timing": "complete_source_PCM_available_not_realtime_capture_or_DAC",
                "python_loopback_guard": True,
                "explicit_canonical_model_root": args.ollama_model_root is not None,
                "os_network_isolated": False,
                "native_token_counts_available": False,
                "resources": "Linux_process_rusage_excludes_Ollama_server_peak_residency_samples_not_layer_offload_or_utilization",
                "provider_counts": "text_chunks_and_characters_not_tokenizer_tokens",
                "thread_environment": "OMP_OPENBLAS_MKL_NUMEXPR_1_native_model_options_preserved",
            },
            "result": aggregate(result),
        }
        _write_private(args.out, payload, exclusive=True)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "report_sha256": digest(payload),
                    "completed": result.get("completed", 0),
                },
                sort_keys=True,
            )
        )
        return (
            0
            if result["status"] == "ok" and result.get("inputs_verified_after") is True
            else 1
        )
    except BaseException:
        print('{"status":"invalid_inputs"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
