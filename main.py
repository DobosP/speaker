#!/usr/bin/env python3
"""
Cross-Platform Voice Assistant with Memory

Features:
- Speech-to-text with faster-whisper / pywhispercpp (streaming)
- Text-to-speech with Kokoro / Supertonic / edge-tts (chunked playback)
- NLMS adaptive echo cancellation + Silero VAD for barge-in
- Streaming LLM responses (sentence-by-sentence TTS)
- Multi-layer memory (recent, summaries, vector search)
- Resource profiles for desktop / low-resource devices

Usage:
    python main.py                      # Use defaults
    python main.py --list-devices       # List audio devices
    python main.py --mode controller    # Hybrid streaming + dialogue controller
    python main.py --profile low        # Low-resource preset (tiny models)
    python main.py --llm-model llama3   # Use specific LLM model
    python main.py --tts-backend kokoro # Force Kokoro TTS
    python main.py --no-memory          # Disable persistent memory
"""
# Disable CUDA to avoid cuDNN compatibility issues (must be before torch import)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import signal
import sys
import threading
import json
import queue
import time
from contextlib import nullcontext

from utils.audio import (
    AudioRecorder,
    AudioPlayer,
    list_audio_devices,
    resolve_output_device,
)
from utils.voice_gate import create_speaker_enrollment_wav
from utils.voice_gate import OpenWakeWordGate, list_known_wakewords, validate_wakeword_name
from utils.stt import (
    get_stt_model,
    get_streaming_stt,
    resolve_stt_runtime,
    resolve_partial_stt_config,
    transcribe_audio,
    WHISPERCPP_AVAILABLE,
)
from utils.realtime_pipeline import flush_queue
from utils.llm import get_llm
from utils.dialogue_controller import DialogueController, BargeInInfo
from utils.capabilities import CapabilityRequest, create_default_registry
from utils.conversation_router import (
    ConversationRouter,
    RouteAction,
    RouteContext,
    RouteDecision,
)
from utils.transports import SessionMux, SessionEnvelope, TransportMode
from utils.turn_detector import TurnDetector
from utils import pipeline_log
from utils import tts_debug

# Memory is optional
try:
    from utils.memory import MemoryManager, POSTGRES_AVAILABLE, EMBEDDINGS_AVAILABLE
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    POSTGRES_AVAILABLE = False
    EMBEDDINGS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Resource Profiles
# ═══════════════════════════════════════════════════════════════════════════════

RESOURCE_PROFILES = {
    # ── Tier 1: low  ─────────────────────────────────────────────────────────
    # Target: slow laptops / Raspberry Pi / first install (smallest downloads)
    # STT:  Moonshine tiny  — 26 MB ONNX, English, 0.05 RTF (fastest CPU ASR)
    # TTS:  Piper ONNX      — ~80 MB / voice, 30+ languages, real-time on any CPU
    # LLM:  tinyllama        — 638 MB, fast on CPU
    "low": {
        "stt_model":    "moonshine:tiny",
        "llm_model":    "tinyllama",
        "tts_backend":  "piper",
        "tts_model":    "en_US-lessac-medium",
        "aec_filter_ms": 80.0,
        "adaptive_vad": True,
        "calibrate_on_start": True,
        "streaming_llm": True,
        "chunked_tts": True,
    },
    # ── Tier 2: mid (default) ────────────────────────────────────────────────
    # Target: modern laptop, no GPU — best balance of speed + quality
    # STT:  distil-medium.en — 750 MB, 5× faster than Whisper base, English
    # TTS:  Kokoro 82M ONNX  — 300 MB, high naturalness, English + Japanese
    # LLM:  llama2            — stable, tested, 3.8 GB
    "mid": {
        "stt_model":    "distil-medium.en",
        "llm_model":    "llama2",
        "tts_backend":  "kokoro",
        "tts_model":    None,           # Kokoro picks voice from KOKORO_VOICES map
        "aec_filter_ms": 80.0,
        "adaptive_vad": True,
        "calibrate_on_start": True,
        "streaming_llm": True,
        "chunked_tts": True,
    },
    # ── Tier 3: high ─────────────────────────────────────────────────────────
    # Target: powerful laptop (16+ GB RAM) — best multilingual quality
    # STT:  large-v3-turbo   — 800 MB, 99+ languages, 8× faster than large-v3
    # TTS:  MeloTTS           — 600 MB, 6 languages (EN/ZH/JP/KR/ES/FR)
    # LLM:  local Ollama model selected for this machine
    "high": {
        "stt_model":    "large-v3-turbo",
        "llm_model":    "Agen/gemma-4-26B-A4B-it-uncensored-heretic",
        "tts_backend":  "melotts",
        "tts_model":    "EN-US",       # change to "ZH", "JP", "KR", "ES", "FR" for multilingual
        "aec_filter_ms": 80.0,
        "adaptive_vad": True,
        "calibrate_on_start": True,
        "streaming_llm": True,
        "chunked_tts": True,
        "partial_stt_model": "tiny",
        "partial_stt_backend": "whispercpp",
    },
}
# Backward-compatible aliases
RESOURCE_PROFILES["desktop"] = RESOURCE_PROFILES["mid"]
RESOURCE_PROFILES["server"] = RESOURCE_PROFILES["high"]

# When a deployment profile is selected (CLI --profile or config "profile"),
# these keys are taken from the profile before config.json.  Otherwise a
# leftover config stt_model/llm_model (e.g. "base") overrides --profile low
# and the preset appears "broken".  Override any bundle key via CLI flags.
PROFILE_BUNDLE_KEYS = frozenset(
    {
        "stt_model",
        "llm_model",
        "tts_backend",
        "tts_model",
        "aec_filter_ms",
        "streaming_llm",
        "chunked_tts",
        "partial_stt_model",
        "partial_stt_backend",
    }
)

# Compute-intensive runtime profiles (independent from deployment profile).
RUNTIME_PROFILES = {
    "edge": {
        "stt_model_type": "whispercpp",
        "llm_generation_profile": "edge",
    },
    "balanced": {
        "stt_model_type": "whisper",
        "llm_generation_profile": "balanced",
    },
    "max_quality": {
        "stt_model_type": "whisper",
        "llm_generation_profile": "max_quality",
    },
}

# When config.json omits these keys, defaults depend on runtime_profile.
RUNTIME_CONVERSATION_DEFAULTS: dict[str, dict[str, float | int]] = {
    "edge": {"llm_min_phrase_words": 3, "silence_duration": 1.2},
    "balanced": {"llm_min_phrase_words": 5, "silence_duration": 1.5},
    "max_quality": {"llm_min_phrase_words": 6, "silence_duration": 1.5},
}


def load_config() -> dict:
    """Load configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def validate_runtime_config(config_values: dict) -> list[str]:
    """Return a list of configuration warnings/errors."""
    issues: list[str] = []
    if config_values["barge_in_min_delay_sec"] < 0:
        issues.append("barge_in_min_delay_sec must be >= 0")
    if config_values["barge_in_min_delay_after_ref_sec"] < 0:
        issues.append("barge_in_min_delay_after_ref_sec must be >= 0")
    if config_values["barge_in_min_rms_ratio"] < 1.0:
        issues.append("barge_in_min_rms_ratio should be >= 1.0")
    if config_values["echo_corr_threshold"] <= 0.0 or config_values["echo_corr_threshold"] > 1.0:
        issues.append("echo_corr_threshold must be in (0, 1]")
    if config_values["aec_filter_ms"] < 20.0:
        issues.append("aec_filter_ms should be >= 20ms")
    wakeword_threshold = config_values.get("wakeword_threshold")
    if wakeword_threshold is not None and not (0.0 < wakeword_threshold <= 1.0):
        issues.append("wakeword_threshold must be in (0, 1]")
    speaker_verify_threshold = config_values.get("speaker_verify_threshold")
    if speaker_verify_threshold is not None and speaker_verify_threshold < 0.0:
        issues.append("speaker_verify_threshold must be >= 0")
    wakeword_policy = config_values.get("wakeword_policy")
    if wakeword_policy is not None and wakeword_policy not in {
        "strict_required",
        "hybrid_recovery",
        "legacy_compatible",
    }:
        issues.append("wakeword_policy must be strict_required|hybrid_recovery|legacy_compatible")
    return issues


def validate_profile_transport_config(
    deployment_profile: str | None,
    runtime_profile: str,
    transport_mode: str,
) -> list[str]:
    issues: list[str] = []
    if deployment_profile and deployment_profile not in RESOURCE_PROFILES:
        issues.append(f"Unknown deployment profile: {deployment_profile}")
    if runtime_profile not in RUNTIME_PROFILES:
        issues.append(f"Unknown runtime_profile: {runtime_profile}")
    if transport_mode not in {m.value for m in TransportMode}:
        issues.append(f"Unknown transport_mode: {transport_mode}")
    if runtime_profile == "edge" and transport_mode == TransportMode.WEBRTC.value:
        issues.append("edge runtime_profile is not recommended with webrtc transport")
    return issues


class VoiceAssistant:
    """
    Cross-platform voice assistant that listens, transcribes, and responds.

    Pipeline:
        Mic -> NLMS AEC -> Silero VAD -> STT -> LLM (streaming) -> TTS (chunked) -> Speaker
              ^--- echo reference ---|
    """

    def __init__(
        self,
        llm_model: str = "gemma3:latest",
        stt_model: str = "base",
        input_device: int = None,
        output_device: int = None,
        vad_threshold: float = 0.01,
        silence_duration: float = 1.5,
        tts_voice: str = "en-US",
        tts_backend: str = None,
        tts_model: str = None,
        barge_in_pre_roll_sec: float = 0.3,
        barge_in_min_speech_sec: float = 0.15,
        barge_in_rms_ratio: float = 2.0,
        barge_in_cooldown_sec: float = 0.5,
        barge_in_use_webrtcvad: bool = True,
        mode: str = "controller",
        controller_config: dict | None = None,
        partial_interval_sec: float = 1.0,
        adaptive_vad: bool = False,
        vad_noise_multiplier: float = 2.5,
        vad_noise_floor_min: float = 0.003,
        calibrate_on_start: bool = True,
        calibrate_duration_sec: float = 2.5,
        aec_enabled: bool = True,
        aec_strength: float = 0.3,
        aec_filter_ms: float = 80.0,
        aec_max_ref_sec: float = 20.0,
        simple_voiced_fallback: bool = True,
        barge_in_debug: bool = False,
        barge_in_min_delay_sec: float = 0.5,
        echo_corr_threshold: float = 0.45,
        barge_in_min_delay_after_ref_sec: float = 0.20,
        barge_in_min_rms_ratio: float = 3.0,
        stop_mode: str = "exact",
        stop_phrases: tuple[str, ...] = ("stop", "quit", "exit"),
        enable_memory: bool = True,
        session_id: str = None,
        db_url: str = None,
        memory_smart_save: bool = True,
        memory_flush_interval_sec: float = 180.0,
        memory_enable_embeddings: bool = False,
        memory_persist_assistant: bool = False,
        memory_llm_clean: bool = True,
        memory_config: dict | None = None,
        streaming_llm: bool = True,
        chunked_tts: bool = True,
        runtime_profile: str = "balanced",
        transport_mode: str = "local_lan",
        wakeword_enabled: bool = False,
        wakeword: str | None = None,
        wakeword_threshold: float = 0.5,
        wakeword_timeout_sec: float = 5.0,
        wakeword_model_path: str | None = None,
        wakeword_service_mode: str = "local",
        wakeword_policy: str = "strict_required",
        wakeword_miss_limit: int = 80,
        wakeword_recovery_window_sec: float = 3.0,
        speaker_verify_enabled: bool = False,
        speaker_enrollment_wav: str | None = None,
        speaker_verify_threshold: float = 0.55,
        diagnostics_log_path: str | None = None,
        diagnostics_log_frames: bool = False,
        trace_backends: bool = False,
        stt_transcriber=None,
        llm=None,
        llm_factory=None,
        partial_stt_model: str | None = None,
        partial_stt_backend: str | None = None,
        partial_stt_threads: int | None = None,
        llm_stream_mode: str = "phrase",
        llm_min_phrase_words: int = 6,
        llm_max_phrase_words: int = 12,
        audio_player=None,
        audio_player_factory=None,
        recorder=None,
        recorder_factory=None,
        live_partial_log: bool = False,
        live_partial_mode: str = "overwrite",
        assistant_stream_print: str | None = None,
        streaming_tts_prefetch: bool = True,
        llm_stream_coalesce_min_words: int = 2,
        llm_stream_coalesce_max_words: int = 6,
        llm_stream_coalesce_flush_sec: float = 0.35,
        no_tts: bool = False,
        playback_backend: str = "auto",
        tts_debug_enabled: bool = False,
    ):
        self.llm_model = llm_model
        self.stt_model = stt_model
        self.input_device = input_device
        self.output_device = output_device
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.tts_voice = tts_voice
        self.tts_backend = tts_backend
        self.tts_model = tts_model
        self.barge_in_pre_roll_sec = barge_in_pre_roll_sec
        self.barge_in_min_speech_sec = barge_in_min_speech_sec
        self.barge_in_rms_ratio = barge_in_rms_ratio
        self.barge_in_cooldown_sec = barge_in_cooldown_sec
        self.barge_in_use_webrtcvad = barge_in_use_webrtcvad
        self.mode = mode
        self.controller_config = controller_config or {}
        self.partial_interval_sec = partial_interval_sec
        self.adaptive_vad = adaptive_vad
        self.vad_noise_multiplier = vad_noise_multiplier
        self.vad_noise_floor_min = vad_noise_floor_min
        self.calibrate_on_start = calibrate_on_start
        self.calibrate_duration_sec = calibrate_duration_sec
        self.aec_enabled = aec_enabled
        self.aec_strength = aec_strength
        self.aec_filter_ms = aec_filter_ms
        self.aec_max_ref_sec = aec_max_ref_sec
        self.simple_voiced_fallback = simple_voiced_fallback
        self.barge_in_debug = barge_in_debug
        self.barge_in_min_delay_sec = barge_in_min_delay_sec
        self.echo_corr_threshold = echo_corr_threshold
        self.barge_in_min_delay_after_ref_sec = barge_in_min_delay_after_ref_sec
        self.barge_in_min_rms_ratio = barge_in_min_rms_ratio
        self._stop_mode = stop_mode
        self._stop_phrases = stop_phrases
        self.enable_memory = enable_memory and MEMORY_AVAILABLE
        self.session_id = session_id
        self.db_url = db_url
        self.memory_smart_save = memory_smart_save
        self.memory_flush_interval_sec = memory_flush_interval_sec
        self.memory_enable_embeddings = memory_enable_embeddings
        self.memory_persist_assistant = memory_persist_assistant
        self.memory_llm_clean = memory_llm_clean
        file_mem = {}
        if memory_config and isinstance(memory_config.get("memory"), dict):
            file_mem = dict(memory_config["memory"])
        built_mem = {
            "enabled": memory_smart_save,
            "save_interval_sec": memory_flush_interval_sec,
            "llm_cleanup": memory_llm_clean,
            "llm_gate": memory_llm_clean,
            "cleanup_model": file_mem.get("cleanup_model", "llama3.2:3b"),
            "min_confidence": file_mem.get("min_confidence", 0.55),
            "persist_user_only": not memory_persist_assistant,
        }
        built_mem.update(file_mem)
        self._memory_config_dict = {"memory": built_mem}
        self.streaming_llm = streaming_llm
        self.chunked_tts = chunked_tts
        self.runtime_profile = runtime_profile
        self.transport_mode = transport_mode
        self.wakeword_enabled = wakeword_enabled
        self.wakeword = wakeword
        self.wakeword_threshold = wakeword_threshold
        self.wakeword_timeout_sec = wakeword_timeout_sec
        self.wakeword_model_path = wakeword_model_path
        self.wakeword_service_mode = wakeword_service_mode
        self.wakeword_policy = wakeword_policy
        self.wakeword_miss_limit = wakeword_miss_limit
        self.wakeword_recovery_window_sec = wakeword_recovery_window_sec
        self.speaker_verify_enabled = speaker_verify_enabled
        self.speaker_enrollment_wav = speaker_enrollment_wav
        self.speaker_verify_threshold = speaker_verify_threshold
        self.diagnostics_log_path = diagnostics_log_path
        self.diagnostics_log_frames = diagnostics_log_frames
        self.trace_backends = trace_backends
        self._stt_transcriber = stt_transcriber or transcribe_audio
        self._injected_llm = llm
        self._llm_factory = llm_factory
        self._injected_player = audio_player
        self._audio_player_factory = audio_player_factory
        self._injected_recorder = recorder
        self._recorder_factory = recorder_factory
        self.llm_stream_mode = llm_stream_mode
        self.llm_min_phrase_words = llm_min_phrase_words
        self.llm_max_phrase_words = llm_max_phrase_words
        self._partial_stt_user_model = partial_stt_model
        self._partial_stt_user_backend = partial_stt_backend
        self._partial_stt_user_threads = partial_stt_threads
        self.live_partial_log = live_partial_log
        self.live_partial_mode = (live_partial_mode or "overwrite").strip().lower()
        if self.live_partial_mode not in ("overwrite", "newline"):
            self.live_partial_mode = "overwrite"
        self._last_partial_printed: str | None = None
        self._live_partial_line_dirty = False
        self._live_partial_print_lock = threading.Lock()

        asp = (assistant_stream_print or "").strip().lower()
        if asp not in ("overwrite", "newline", ""):
            asp = ""
        if not asp:
            asp = (
                "newline"
                if (llm_stream_mode or "phrase").lower() in ("token", "word")
                else "overwrite"
            )
        self.assistant_stream_print = asp
        self.streaming_tts_prefetch = streaming_tts_prefetch
        self.llm_stream_coalesce_min_words = llm_stream_coalesce_min_words
        self.llm_stream_coalesce_max_words = llm_stream_coalesce_max_words
        self.llm_stream_coalesce_flush_sec = llm_stream_coalesce_flush_sec
        self.no_tts = no_tts
        self.playback_backend = playback_backend
        self.tts_debug_enabled = tts_debug_enabled
        tts_debug.configure(enabled=tts_debug_enabled)

        self._shutdown_event = threading.Event()
        self._llm = None
        self._recorder = None
        self._player = None
        self._memory = None
        self._speaking_lock = threading.Lock()
        self._response_thread = None
        self._controller = None
        self._partial_audio_queue = None
        self._partial_worker = None
        self._cancel_generation = threading.Event()
        self._stage_metrics = {}
        self._stage_metrics_lock = threading.Lock()
        self._capabilities = create_default_registry()
        self._router = ConversationRouter(
            stop_phrases=tuple(stop_phrases),
            stop_mode=stop_mode,
        )
        self._session_mux = SessionMux(mode=TransportMode(self.transport_mode))
        self._turn_detector = TurnDetector(
            diagnostics_log_path=self.diagnostics_log_path
        )

        import utils.backend_trace as _backend_trace

        _backend_trace.configure(
            enabled=self.trace_backends,
            diagnostics_log_path=self.diagnostics_log_path,
        )
        pipeline_log.configure(
            enabled=bool(self.diagnostics_log_path),
            path=self.diagnostics_log_path,
            session_id=self.session_id or "local",
        )
        self._stt_runtime = resolve_stt_runtime(
            runtime_profile=self.runtime_profile,
            model_id=self.stt_model,
        )
        self._stt_model_type = self._stt_runtime["model_type"]
        self._stt_threads = self._stt_runtime["n_threads"]
        pth = (
            self._partial_stt_user_threads
            if self._partial_stt_user_threads is not None
            else self._stt_threads
        )
        pcfg = resolve_partial_stt_config(
            partial_model=self._partial_stt_user_model,
            partial_backend=self._partial_stt_user_backend,
            final_model_id=self.stt_model,
            n_threads=pth,
        )
        self._partial_stt_model_id = pcfg["model_id"]
        self._partial_stt_backend = pcfg["backend"]
        self._partial_stt_threads = pcfg["n_threads"]
        self._partial_acoustic_only = False
        self._tts_session_seq = 0
        self._active_tts_session = -1
        self._active_tts_queue: queue.Queue | None = None
        self._tts_session_lock = threading.Lock()
        self._utterance_pipeline_lock = threading.Lock()
        self._streaming_controller_tts_active = False
        self._streaming_controller_lock = threading.Lock()
        self._console_print_lock = threading.Lock()
        self._stage_tls = threading.local()

    def _is_junk_stt_text(self, text: str) -> bool:
        """Heuristic: noise / silence hallucinations from STT (esp. on quiet audio)."""
        t = (text or "").strip().lower()
        if not t:
            return True
        if len(t) <= 2 and t in (".", "?", "!", "…"):
            return True
        markers = (
            "[blank_audio]",
            "[blank audio]",
            "(blank audio)",
            "blank_audio",
            "[birds chirping]",
            "(birds chirping)",
            "birds chirping",
            "[music]",
            "(music playing)",
        )
        return any(m in t for m in markers)

    def _print_live_partial(self, text: str) -> None:
        """Echo realtime partial STT to the console (controller mode)."""
        if not self.live_partial_log or self.mode != "controller":
            return
        t = (text or "").strip()
        if not t or self._is_junk_stt_text(t):
            return
        if t == self._last_partial_printed:
            return
        self._last_partial_printed = t
        use_newline = self.live_partial_mode == "newline" or not sys.stdout.isatty()
        with self._live_partial_print_lock:
            if use_newline:
                print(f"You (partial): {t}", flush=True)
            else:
                # Newline first: avoid clobbering the mic/TTS \r status line.
                max_len = 120
                display = t if len(t) <= max_len else t[: max_len - 3] + "..."
                sys.stdout.write(f"\n\r\x1b[KYou (partial): {display}")
                sys.stdout.flush()
                self._live_partial_line_dirty = True
            try:
                self._session_mux.broadcast(
                    SessionEnvelope(
                        session_id=self.session_id or "local",
                        event_type="user_partial",
                        payload={"text": t, "is_partial": True},
                    )
                )
            except Exception:
                pass

    def _clear_live_partial_line_before_final(self) -> None:
        """Start the final user transcript on a fresh line after overwrite partials."""
        if (
            not self.live_partial_log
            or self.live_partial_mode != "overwrite"
            or not self._live_partial_line_dirty
        ):
            return
        with self._live_partial_print_lock:
            sys.stdout.write("\r\x1b[K")
            sys.stdout.flush()
            self._live_partial_line_dirty = False
        self._last_partial_printed = None

    def _mark_stage(self, name: str):
        sink = getattr(self._stage_tls, "stages", None)
        if sink is not None:
            sink[name] = time.perf_counter()
            return
        with self._stage_metrics_lock:
            self._stage_metrics[name] = time.perf_counter()

    def _latency_metrics(self) -> dict:
        stages = getattr(self._stage_tls, "stages", None)
        if stages is None:
            with self._stage_metrics_lock:
                stages = dict(self._stage_metrics)

        def delta_ms(start: str, end: str):
            if start not in stages or end not in stages:
                return None
            return round(max(0.0, stages[end] - stages[start]) * 1000.0, 2)

        metrics = {
            "speech_detected_to_stt_final_ms": delta_ms(
                "utterance_start", "stt_final"
            ),
            "stt_final_to_memory_ready_ms": delta_ms(
                "stt_final", "memory_ready"
            ),
            "stt_final_to_first_llm_sentence_ms": delta_ms(
                "stt_final", "first_llm_sentence"
            ),
            "memory_ready_to_first_llm_sentence_ms": delta_ms(
                "memory_ready", "first_llm_sentence"
            ),
            "first_sentence_to_tts_start_ms": delta_ms(
                "first_llm_sentence", "tts_first_audio"
            ),
            "speech_detected_to_first_audio_ms": delta_ms(
                "utterance_start", "tts_first_audio"
            ),
            "total_turn_ms": delta_ms("utterance_start", "turn_complete"),
        }
        m2 = delta_ms("memory_ready", "llm_response_ready")
        if m2 is not None:
            metrics["memory_ready_to_llm_response_ready_ms"] = m2
        if "llm_response_ready" in stages:
            metrics["stt_final_to_llm_response_ms"] = delta_ms(
                "stt_final", "llm_response_ready"
            )
        return {key: value for key, value in metrics.items() if value is not None}

    def _print_latency_bottleneck_hint(self, metrics: dict) -> None:
        """One-line hint when a single stage clearly dominates (live debugging)."""
        stt = metrics.get("speech_detected_to_stt_final_ms")
        mem = metrics.get("stt_final_to_memory_ready_ms")
        llm_stream = metrics.get("memory_ready_to_first_llm_sentence_ms")
        llm_batch = metrics.get("memory_ready_to_llm_response_ready_ms")
        llm = llm_stream if llm_stream is not None else llm_batch

        if mem is not None and mem >= 800.0:
            print(
                "(latency) hint: memory/context prep is slow (embeddings or DB).",
                flush=True,
            )
            return
        if (
            stt is not None
            and stt >= 2500.0
            and (llm is None or stt >= 0.55 * max(llm, 1.0))
        ):
            print(
                "(latency) hint: final STT dominates — use a smaller/faster "
                "Whisper model or --profile mid / balanced runtime.",
                flush=True,
            )
            return
        if llm is not None and llm >= 4000.0 and (
            stt is None or llm >= 1.8 * max(stt, 1.0)
        ):
            sm = (self.llm_stream_mode or "phrase").lower()
            if sm == "phrase":
                tip = (
                    "or lower --llm-min-phrase-words for earlier first speech chunk."
                )
            elif sm == "sentence":
                tip = "or try --llm-stream-mode phrase/token for earlier chunks."
            else:
                tip = (
                    "or tune --llm-stream-coalesce-* (fewer Kokoro runs); "
                    "GPU helps time-to-first-token."
                )
            print(
                "(latency) hint: LLM dominates — smaller Ollama model, GPU, "
                + tip,
                flush=True,
            )

    def _build_memory_config(self) -> dict:
        """Merge CLI flags into config.json memory section for MemoryWriter."""
        base = dict(self._memory_config_dict or {})
        mem = dict(base.get("memory") or {})
        mem["llm_cleanup"] = self.memory_llm_clean
        mem["llm_gate"] = self.memory_llm_clean
        mem["enabled"] = self.memory_smart_save
        mem["save_interval_sec"] = self.memory_flush_interval_sec
        base["memory"] = mem
        return base

    def _clean_memory_text_with_llm(self, role: str, content: str) -> str | None:
        """
        Cleanup hook used by MemoryManager before buffered DB flush.

        User transcripts are cleaned for long-term memory. Assistant replies
        remain unchanged when explicitly persisted.
        """
        text = (content or "").strip()
        if not text:
            return None
        if role != "user":
            return text
        if self._llm is None:
            return text
        prompt = (
            "Clean this speech-to-text transcript for long-term memory.\n"
            "Rules:\n"
            "- Return only what the user intentionally said.\n"
            "- Fix obvious ASR typos and punctuation.\n"
            "- Remove filler, duplicate words, subtitles, hallucinated thanks, and noise.\n"
            "- If it is not meaningful memory, return exactly EMPTY.\n\n"
            f"Transcript: {text}"
        )
        cleaned = self._llm.get_response(prompt, context=None, history=[])
        cleaned = (cleaned or "").strip().strip('"')
        if cleaned.upper() == "EMPTY":
            return None
        for prefix in ("cleaned:", "memory:", "user:"):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        return cleaned or None

    def _print_startup_latency_notice(self) -> None:
        """Explain slow-stack combos once so logs can be read against expectations."""
        heavy_stt = any(
            x in (self.stt_model or "").lower()
            for x in ("large", "medium", "distil-large")
        )
        lm = (self.llm_model or "").lower()
        heavy_llm = any(
            tok in lm
            for tok in (
                "26b",
                "70b",
                "65b",
                "405b",
                "34b",
                "30b-a",
                "72b",
                "mixtral",
                "command-a",
            )
        )
        if (
            not heavy_stt
            and not heavy_llm
            and self.runtime_profile != "max_quality"
        ):
            return
        print(
            "\nLatency: expect multi-second turns on CPU with large Whisper + big "
            "Ollama models. Each (latency) line splits STT, memory, and LLM/TTS.",
            flush=True,
        )
        sm = (self.llm_stream_mode or "phrase").lower()
        if sm == "phrase":
            print(
                f"  Streaming: phrase buffer uses llm_min_phrase_words="
                f"{self.llm_min_phrase_words} before the first TTS chunk.",
                flush=True,
            )
        elif sm == "sentence":
            print(
                "  Streaming: sentence mode waits for sentence-ending punctuation.",
                flush=True,
            )
        elif sm in ("token", "word"):
            print(
                f"  Streaming: {sm} mode — tokens coalesce to ~"
                f"{self.llm_stream_coalesce_min_words}–{self.llm_stream_coalesce_max_words} "
                f"words (or {self.llm_stream_coalesce_flush_sec:.2f}s) before TTS; "
                "Kokoro synthesis dominates chunk latency.",
                flush=True,
            )
            print(
                f"  TTS prefetch (synth N+1 during playback): "
                f"{'on' if self.streaming_tts_prefetch else 'off'}.",
                flush=True,
            )

    def _alloc_tts_session(self) -> int:
        with self._tts_session_lock:
            self._tts_session_seq += 1
            sid = self._tts_session_seq
            self._active_tts_session = sid
        tts_debug.log("tts", "session_alloc", session_id=sid)
        return sid

    def _invalidate_tts_session(self, *, reason: str = "invalidate") -> None:
        with self._tts_session_lock:
            prev = self._active_tts_session
            self._active_tts_session = -1
        tts_debug.log(
            "tts",
            "session_invalidate",
            reason=reason,
            prev_session=prev,
        )

    def _publish_turn_metrics(self, transcription: str | None = None):
        metrics = self._latency_metrics()
        if not metrics:
            return
        payload = {
            **metrics,
            "runtime_profile": self.runtime_profile,
            "stt_model": self.stt_model,
            "stt_model_type": self._stt_model_type,
            "stt_threads": self._stt_threads,
            "llm_model": self.llm_model,
            "llm_stream_mode": self.llm_stream_mode,
            "llm_min_phrase_words": self.llm_min_phrase_words,
            "streaming_llm": self.streaming_llm,
            "chunked_tts": self.chunked_tts,
        }
        if transcription is not None:
            payload["transcript_chars"] = len(transcription)
        self._session_mux.broadcast(
            SessionEnvelope(
                session_id=self.session_id or "local",
                event_type="turn_metrics",
                payload=payload,
            )
        )
        if self.diagnostics_log_path:
            try:
                parent = os.path.dirname(self.diagnostics_log_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self.diagnostics_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "event": "turn_metrics",
                                "session_id": self.session_id or "local",
                                "timestamp": time.time(),
                                **payload,
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
        cm = self._console_print_lock if self._console_print_lock else nullcontext()
        with cm:
            print(f"\n(latency) {json.dumps(metrics, sort_keys=True)}")
            self._print_latency_bottleneck_hint(metrics)

    def _route_context(self, transcript: str, is_partial: bool = False) -> RouteContext:
        assistant_speaking = bool(
            self._recorder and getattr(self._recorder, "assistant_is_speaking", False)
        )
        barge_in_active = bool(
            self._recorder
            and getattr(self._recorder, "is_barge_in_active", lambda: False)()
        )
        return RouteContext(
            transcript=transcript,
            assistant_speaking=assistant_speaking,
            barge_in_active=barge_in_active,
            is_partial=is_partial,
            mode=self.mode,
            available_capabilities=tuple(self._capabilities.list_capabilities()),
        )

    def _log_route_decision(self, decision: RouteDecision):
        payload = {
            "action": decision.action.value,
            "reason": decision.reason,
            "normalized_text": decision.normalized_text,
            "capability": decision.capability,
            "confidence": decision.confidence,
        }
        self._session_mux.broadcast(
            SessionEnvelope(
                session_id=self.session_id or "local",
                event_type="route_decision",
                payload=payload,
            )
        )
        if self.diagnostics_log_path:
            try:
                parent = os.path.dirname(self.diagnostics_log_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self.diagnostics_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "event": "route_decision",
                                "session_id": self.session_id or "local",
                                "timestamp": time.time(),
                                **payload,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
            except Exception:
                pass
        pipeline_log.emit(
            "pipeline_route",
            action=payload["action"],
            reason=payload["reason"],
            transcript_chars=len(decision.normalized_text or ""),
            capability=payload.get("capability"),
        )

    def _cancel_current_output(self, *, reason: str = "cancel_output"):
        pipeline_log.emit("cancel_output", reason=reason)
        tts_debug.log("tts", "queue_flush", reason=reason)
        tts_debug.console("tts_flush", reason)
        self._cancel_generation.set()
        self._invalidate_tts_session()
        q = self._active_tts_queue
        if q is not None:
            n = flush_queue(q)
            tts_debug.log("tts", "queue_flush_drained", reason=reason, dropped=n)
        if self._player:
            try:
                self._player.stop()
            except Exception:
                pass
        if self._recorder:
            try:
                self._recorder.set_assistant_speaking(False)
            except Exception:
                pass
        with self._streaming_controller_lock:
            if self._controller and self._streaming_controller_tts_active:
                try:
                    self._controller.on_tts_end()
                except Exception:
                    pass
                self._streaming_controller_tts_active = False

    def _capability_response_text(self, name: str, data: dict) -> str:
        if name == "system.time":
            unix_time = data.get("unix_time")
            if isinstance(unix_time, (int, float)):
                return time.strftime("The current time is %H:%M.", time.localtime(unix_time))
            return "I could not read the current time."
        if name == "debug.echo":
            echo = data.get("echo", {})
            text = echo.get("text") if isinstance(echo, dict) else echo
            return f"Echo: {text}" if text else "Echo received."
        return json.dumps(data, sort_keys=True)

    def _execute_route_decision(
        self,
        decision: RouteDecision,
        transcription: str,
        context,
        history,
    ) -> bool:
        """Execute a route decision. Return True when the turn was handled."""
        self._log_route_decision(decision)
        if decision.action == RouteAction.IGNORE:
            return True
        if decision.action == RouteAction.STOP_OUTPUT:
            self._cancel_current_output()
            print("(router) stopped current output")
            return True
        if decision.action == RouteAction.SHUTDOWN:
            self._cancel_current_output()
            print("\nGoodbye!")
            self._shutdown_event.set()
            return True
        if decision.action == RouteAction.CAPABILITY:
            if not decision.capability:
                return True
            req = CapabilityRequest(
                name=decision.capability,
                payload=decision.payload,
                session_id=self.session_id,
            )
            response = self._capabilities.invoke(req)
            if response.ok:
                answer = self._capability_response_text(decision.capability, response.data)
            else:
                answer = f"Capability failed: {response.error}"
            print(f"Assistant: {answer}")
            with open("live_transcript.txt", "a") as f:
                f.write(f"Assistant: {answer}\n")
            self._speak(answer)
            return True
        if decision.action == RouteAction.LLM:
            return False
        return True

    def _on_barge_in(self, info=None):
        """Called when user speaks while assistant is talking (barge-in)."""
        should_stop = True
        if self._controller and info:
            barge_info = BargeInInfo(
                rms=info.get("rms", 0.0),
                threshold=info.get("threshold", 0.0),
                voiced=bool(info.get("voiced", False)),
                duration_sec=info.get("duration_sec", 0.0),
                timestamp=info.get("timestamp", 0.0),
                echo=bool(info.get("echo", False)),
            )
            should_stop = self._controller.should_stop_speaking(barge_info)

        if should_stop:
            self._cancel_generation.set()
            self._invalidate_tts_session(reason="barge-in")
            q = self._active_tts_queue
            if q is not None:
                n = flush_queue(q)
                tts_debug.log("tts", "queue_flush", reason="barge-in", dropped=n)
                tts_debug.console("tts_cancel", "barge-in")
            self._mark_stage("barge_in_confirmed")
        if should_stop and self._player:
            self._player.stop()
        elif self.barge_in_debug and self._controller:
            print(f"Barge-in ignored: {self._controller.last_reason()}")
        pipeline_log.emit(
            "barge_in",
            should_stop=should_stop,
            voiced=None if not info else bool(info.get("voiced")),
            rms=None if not info else round(float(info.get("rms", 0.0)), 4),
        )
        return should_stop

    def _on_speech_detected(self, audio_data):
        """Called when the user finishes speaking."""
        if self._shutdown_event.is_set():
            return
        self._session_mux.broadcast(
            SessionEnvelope(
                session_id=self.session_id or "local",
                event_type="speech_detected",
                payload={"samples": int(len(audio_data))},
            )
        )
        self._response_thread = threading.Thread(
            target=self._utterance_worker_entry,
            args=(audio_data,),
            daemon=True,
        )
        self._response_thread.start()

    def _utterance_worker_entry(self, audio_data):
        """Entry point: full turn (STT is serialized; LLM/TTS may overlap prior playback)."""
        if self._shutdown_event.is_set():
            return
        self._process_and_respond(audio_data)

    def _process_and_respond(self, audio_data):
        """Process audio: hold pipeline lock only for STT + routing, not for LLM/TTS."""
        if not self._utterance_pipeline_lock.acquire(blocking=False):
            print("(queued: finishing previous turn...)", flush=True)
            self._cancel_current_output()
            self._utterance_pipeline_lock.acquire(blocking=True)

        transcription = None
        latency_publish = False
        pipeline_locked = True
        try:
            self._cancel_generation.set()
            self._stage_tls.stages = {"utterance_start": time.perf_counter()}
            with pipeline_log.span(
                "utterance",
                "final_stt",
                model=self.stt_model,
                model_type=self._stt_model_type,
                audio_samples=int(len(audio_data)),
            ):
                transcription = self._stt_transcriber(
                    audio_data,
                    model_id=self.stt_model,
                    model_type=self._stt_model_type,
                    n_threads=self._stt_threads,
                )
            self._turn_detector.on_final_text(transcription or "")
            self._mark_stage("stt_final")

            if not transcription or transcription.strip() == "":
                self._cancel_generation.clear()
                return
            if self._controller and self._controller.should_ignore_transcript(
                transcription
            ):
                self._cancel_generation.clear()
                return
            if self._is_junk_stt_text(transcription):
                self._cancel_generation.clear()
                return

            latency_publish = True
            self._clear_live_partial_line_before_final()
            print(f"\nYou: {transcription}")
            self._session_mux.broadcast(
                SessionEnvelope(
                    session_id=self.session_id or "local",
                    event_type="user_transcript",
                    payload={"text": transcription},
                )
            )

            with open("live_transcript.txt", "a") as f:
                f.write(f"You: {transcription}\n")

            with pipeline_log.span("utterance", "router_route"):
                decision = self._router.route(self._route_context(transcription))
            if decision.action != RouteAction.LLM:
                self._utterance_pipeline_lock.release()
                pipeline_locked = False
                self._cancel_generation.clear()
                self._execute_route_decision(decision, transcription, None, None)
                return

            if self._memory:
                self._memory.add_message("user", transcription)

            if not self._llm:
                self._utterance_pipeline_lock.release()
                pipeline_locked = False
                self._cancel_generation.clear()
                return

            context = None
            history = None
            if self._memory:
                with pipeline_log.span("utterance", "memory_context"):
                    context = self._memory.get_context_for_llm(transcription)
                    history = self._memory.get_chat_history()

            self._mark_stage("memory_ready")

            self._utterance_pipeline_lock.release()
            pipeline_locked = False
            self._cancel_generation.clear()

            if self.streaming_llm:
                self._streaming_respond(transcription, context, history)
            else:
                self._batch_respond(transcription, context, history)

        except Exception as e:
            print(f"Error processing speech: {e}")
            import traceback
            traceback.print_exc()
            self._cancel_generation.clear()
        finally:
            if pipeline_locked:
                try:
                    self._utterance_pipeline_lock.release()
                except RuntimeError:
                    pass
            self._mark_stage("turn_complete")
            if latency_publish and transcription and not self._is_junk_stt_text(
                transcription
            ):
                self._publish_turn_metrics(transcription)

    def _batch_respond(self, transcription, context, history):
        """Non-streaming: generate full response then speak."""
        with pipeline_log.span("llm", "batch_generate", model=self.llm_model):
            response = self._llm.get_response(
                transcription, context=context, history=history
            )
        self._mark_stage("llm_response_ready")
        if self._cancel_generation.is_set():
            return
        print(f"Assistant: {response}")
        if self._memory:
            self._memory.add_message("assistant", response)
        with open("live_transcript.txt", "a") as f:
            f.write(f"Assistant: {response}\n")
        self._speak(response)

    def _enqueue_streaming_tts(self, tts_queue: queue.Queue, item) -> bool:
        while not self._shutdown_event.is_set():
            if item is not None and self._cancel_generation.is_set():
                tts_debug.log("tts", "queue_drop", reason="cancel_generation")
                tts_debug.console("tts_drop", "cancelled before enqueue")
                return False
            try:
                tts_queue.put(item, timeout=0.05)
                if item is not None:
                    sid, text = item if isinstance(item, tuple) and len(item) == 2 else (-1, item)
                    preview = (text or "")[:80]
                    tts_debug.log(
                        "tts",
                        "queue_enqueue",
                        session_id=sid,
                        qsize=tts_queue.qsize(),
                        text_chars=len(text or ""),
                        preview=preview,
                    )
                    tts_debug.console("tts_enqueue", preview)
                return True
            except queue.Full:
                tts_debug.log("tts", "queue_enqueue_blocked", qsize=tts_queue.maxsize)
                continue
        return False

    def _finish_streaming_tts_worker(
        self,
        tts_queue: queue.Queue,
        tts_worker: threading.Thread,
    ):
        if self._cancel_generation.is_set():
            dropped = 0
            while True:
                try:
                    tts_queue.get_nowait()
                    tts_queue.task_done()
                    dropped += 1
                except queue.Empty:
                    break
            if dropped:
                tts_debug.log("tts", "queue_flush", reason="cancel_generation", dropped=dropped)
                tts_debug.console("tts_flush", f"cancelled ({dropped} items)")
        while tts_worker.is_alive() and not self._shutdown_event.is_set():
            try:
                tts_queue.put(None, timeout=0.05)
                break
            except queue.Full:
                if not self._cancel_generation.is_set():
                    continue
                try:
                    tts_queue.get_nowait()
                    tts_queue.task_done()
                except queue.Empty:
                    continue
        tts_worker.join(timeout=30.0)

    _TTS_DEQUEUE_SHUTDOWN = object()
    _TTS_QUEUE_NO_PEEK = object()

    def _streaming_tts_dequeue(
        self,
        tts_queue: queue.Queue,
        *,
        log_sentinel: bool = False,
        prefetch: bool = False,
    ):
        """Block until the next TTS item, queue sentinel (None), shutdown, or cancel."""
        while not self._shutdown_event.is_set():
            if self._cancel_generation.is_set():
                return self._TTS_DEQUEUE_SHUTDOWN
            try:
                item = tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None and log_sentinel:
                tts_debug.log(
                    "tts",
                    "queue_dequeue",
                    item="sentinel",
                    prefetch=prefetch,
                )
            return item
        return self._TTS_DEQUEUE_SHUTDOWN

    def _streaming_tts_worker(self, tts_queue: queue.Queue):
        use_pf = (
            self.streaming_tts_prefetch
            and self.streaming_llm
            and self._player is not None
            and getattr(self._player, "tts_backend", None) is not None
        )
        if use_pf:
            self._streaming_tts_worker_prefetch(tts_queue)
        else:
            self._streaming_tts_worker_serial(tts_queue)

    def _streaming_tts_worker_serial(self, tts_queue: queue.Queue):
        assistant_held = False
        try:
            while not self._shutdown_event.is_set():
                try:
                    item = tts_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if item is None:
                        tts_debug.log("tts", "queue_dequeue", item="sentinel")
                        return
                    if isinstance(item, tuple) and len(item) == 2:
                        sid, sentence = item
                    else:
                        sid, sentence = -1, item
                    tts_debug.log(
                        "tts",
                        "queue_dequeue",
                        session_id=sid,
                        text_chars=len(sentence or ""),
                    )
                    tts_debug.console("tts_dequeue", (sentence or "")[:60])
                    with self._tts_session_lock:
                        valid = self._active_tts_session
                    if valid == -1 or sid != valid:
                        tts_debug.log(
                            "tts",
                            "queue_drop",
                            reason="stale_session",
                            session_id=sid,
                            active_session=valid,
                        )
                        continue
                    if self._cancel_generation.is_set():
                        tts_debug.log("tts", "queue_drop", reason="cancel_generation")
                        return
                    if self._recorder and not assistant_held:
                        self._recorder.set_assistant_speaking(True)
                        assistant_held = True
                    self._speak(sentence, defer_assistant_speaking=True)
                    if self._recorder and self._recorder.is_barge_in_active():
                        print("(LLM generation stopped – barge-in)")
                        self._cancel_generation.set()
                        self._invalidate_tts_session()
                        return
                finally:
                    tts_queue.task_done()
        finally:
            if assistant_held and self._recorder:
                try:
                    if not self._recorder.is_barge_in_active():
                        self._recorder.set_assistant_speaking(False)
                except Exception:
                    pass
            with self._streaming_controller_lock:
                if self._controller and self._streaming_controller_tts_active:
                    try:
                        self._controller.on_tts_end()
                    except Exception:
                        pass
                    self._streaming_controller_tts_active = False

    def _streaming_tts_worker_prefetch(self, tts_queue: queue.Queue):
        """Synthesize chunk N+1 while playing chunk N (when queue supplies text ahead)."""
        from concurrent.futures import ThreadPoolExecutor

        assistant_held = False
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            item = self._streaming_tts_dequeue(
                tts_queue, log_sentinel=True, prefetch=True
            )
            if item is self._TTS_DEQUEUE_SHUTDOWN:
                return
            if item is None:
                tts_queue.task_done()
                return
            if isinstance(item, tuple) and len(item) == 2:
                sid, text_cur = item
            else:
                sid, text_cur = -1, item

            future = None

            while not self._shutdown_event.is_set():
                with self._tts_session_lock:
                    valid = self._active_tts_session
                if valid == -1 or sid != valid:
                    tts_debug.log("tts", "queue_drop", reason="stale_session", session_id=sid)
                    tts_queue.task_done()
                    return
                if self._cancel_generation.is_set():
                    tts_debug.log("tts", "queue_drop", reason="cancel_generation")
                    tts_queue.task_done()
                    return
                tts_debug.log(
                    "tts",
                    "queue_dequeue",
                    session_id=sid,
                    text_chars=len(text_cur or ""),
                    prefetch=True,
                )

                if future is None:
                    future = pool.submit(self._player.prepare_speech_file, text_cur)

                try:
                    path = future.result(timeout=120.0)
                except Exception as e:
                    print(f"TTS prefetch synth error: {e}")
                    tts_queue.task_done()
                    return

                if self._recorder and not assistant_held:
                    self._recorder.set_assistant_speaking(True)
                    assistant_held = True

                self._speak(
                    text_cur,
                    defer_assistant_speaking=True,
                    prepared_path=path,
                )
                tts_queue.task_done()

                if self._recorder and self._recorder.is_barge_in_active():
                    print("(LLM generation stopped – barge-in)")
                    self._cancel_generation.set()
                    self._invalidate_tts_session()
                    return

                try:
                    item_next = tts_queue.get_nowait()
                except queue.Empty:
                    item_next = self._TTS_QUEUE_NO_PEEK

                future = None
                peek_valid = False
                sn = 0
                tn = ""
                if item_next is None:
                    tts_queue.task_done()
                    break
                if item_next is not self._TTS_QUEUE_NO_PEEK:
                    if isinstance(item_next, tuple) and len(item_next) == 2:
                        sn, tn = item_next
                    else:
                        sn, tn = -1, item_next
                    with self._tts_session_lock:
                        v2 = self._active_tts_session
                    if (
                        v2 != -1
                        and sn == v2
                        and not self._cancel_generation.is_set()
                    ):
                        future = pool.submit(
                            self._player.prepare_speech_file, tn
                        )
                        peek_valid = True
                    else:
                        tts_queue.task_done()

                if peek_valid:
                    sid, text_cur = sn, tn
                    continue
                if item_next is not self._TTS_QUEUE_NO_PEEK:
                    if isinstance(item_next, tuple) and len(item_next) == 2:
                        sid, text_cur = item_next
                    else:
                        sid, text_cur = -1, item_next
                    future = None
                    continue

                item = self._streaming_tts_dequeue(tts_queue)
                if item is self._TTS_DEQUEUE_SHUTDOWN:
                    break
                if item is None:
                    tts_queue.task_done()
                    break
                if isinstance(item, tuple) and len(item) == 2:
                    sid, text_cur = item
                else:
                    sid, text_cur = -1, item
                future = None
        finally:
            pool.shutdown(wait=False, cancel_futures=False)
            if assistant_held and self._recorder:
                try:
                    if not self._recorder.is_barge_in_active():
                        self._recorder.set_assistant_speaking(False)
                except Exception:
                    pass
            with self._streaming_controller_lock:
                if self._controller and self._streaming_controller_tts_active:
                    try:
                        self._controller.on_tts_end()
                    except Exception:
                        pass
                    self._streaming_controller_tts_active = False

    def _streaming_respond(self, transcription, context, history):
        """Streaming: generate and speak phrase-by-phrase (or sentence mode)."""
        full_response_parts = []
        _sq = 48 if self.llm_stream_mode in ("token", "word") else 2
        tts_queue = queue.Queue(maxsize=_sq)
        tts_worker = threading.Thread(
            target=self._streaming_tts_worker,
            args=(tts_queue,),
            daemon=True,
        )
        tts_worker.start()
        self._active_tts_queue = tts_queue
        speak_sid = self._alloc_tts_session()
        stream_kw = {
            "stream_mode": self.llm_stream_mode,
            "min_phrase_words": self.llm_min_phrase_words,
            "max_phrase_words": self.llm_max_phrase_words,
            "coalesce_min_words": self.llm_stream_coalesce_min_words,
            "coalesce_max_words": self.llm_stream_coalesce_max_words,
            "coalesce_flush_sec": self.llm_stream_coalesce_flush_sec,
        }
        with pipeline_log.span(
            "llm",
            "streaming_turn",
            model=self.llm_model,
            stream_mode=self.llm_stream_mode,
        ):
            try:
                self._streaming_respond_loop(
                    transcription,
                    context,
                    history,
                    tts_queue,
                    speak_sid,
                    stream_kw,
                    full_response_parts,
                )
            except Exception as e:
                print(f"Streaming error: {e}")
            finally:
                self._active_tts_queue = None
                self._finish_streaming_tts_worker(tts_queue, tts_worker)

        if self._memory and full_response_parts:
            if self.llm_stream_mode in ("token", "word"):
                _full_text = "".join(full_response_parts)
            else:
                _full_text = " ".join(full_response_parts)
            self._memory.add_message("assistant", _full_text)

    def _streaming_respond_loop(
        self,
        transcription,
        context,
        history,
        tts_queue,
        speak_sid,
        stream_kw,
        full_response_parts,
    ):
        sm = (stream_kw.get("stream_mode") or self.llm_stream_mode or "phrase").lower()
        stream_console = sm in ("token", "word")
        console_acc: list[str] = []

        for sentence in self._llm.get_streaming_response(
            transcription,
            context=context,
            history=history,
            should_cancel=lambda: self._cancel_generation.is_set(),
            **stream_kw,
        ):
            if self._cancel_generation.is_set():
                break
            _tls_stages = getattr(self._stage_tls, "stages", None)
            if _tls_stages is not None and "first_llm_sentence" not in _tls_stages:
                self._mark_stage("first_llm_sentence")
            full_response_parts.append(sentence)
            if stream_console:
                console_acc.append(sentence)
                tail = "".join(console_acc)
                if len(tail) > 400:
                    tail = tail[-400:]
                if self.assistant_stream_print == "newline":
                    with self._console_print_lock:
                        print(f"Assistant: {sentence}", flush=True)
                else:
                    with self._console_print_lock:
                        print(f"\rAssistant: {tail}\033[K", end="", flush=True)
            else:
                print(f"Assistant: {sentence}")
            self._session_mux.broadcast(
                SessionEnvelope(
                    session_id=self.session_id or "local",
                    event_type="assistant_sentence",
                    payload={"text": sentence},
                )
            )
            if not stream_console:
                with open("live_transcript.txt", "a") as f:
                    f.write(f"Assistant: {sentence}\n")

            tts_debug.log(
                "tts",
                "llm_chunk",
                session_id=speak_sid,
                text_chars=len(sentence or ""),
                stream_mode=sm,
                preview=(sentence or "")[:80],
            )
            tts_debug.console("tts_llm_chunk", (sentence or "")[:60])
            if not self._enqueue_streaming_tts(tts_queue, (speak_sid, sentence)):
                tts_debug.log("tts", "llm_phrase_dropped", reason="enqueue_failed")
                break

        if stream_console and full_response_parts:
            with self._console_print_lock:
                print()
            with open("live_transcript.txt", "a") as f:
                f.write(f"Assistant: {''.join(full_response_parts)}\n")

    def _speak(
        self,
        text: str,
        *,
        defer_assistant_speaking: bool = False,
        prepared_path: str | None = None,
    ):
        """Speak text with barge-in support.

        When ``defer_assistant_speaking`` is True (streaming TTS worker), the
        worker holds ``assistant_is_speaking`` across all phrases so partial STT
        does not run between sentences. The dialogue controller is notified once
        at the first chunk (``on_tts_start``) and once after the last chunk
        (``on_tts_end`` in the worker ``finally`` or on cancel) so barge-in
        policy stays valid between streamed sentences.
        """
        metrics_sink = getattr(self._stage_tls, "stages", None)

        def _tts_playback_started(audio_data=None, sample_rate=None):
            if metrics_sink is not None:
                if "tts_first_audio" not in metrics_sink:
                    metrics_sink["tts_first_audio"] = time.perf_counter()
            else:
                with self._stage_metrics_lock:
                    if "tts_first_audio" not in self._stage_metrics:
                        self._stage_metrics["tts_first_audio"] = time.perf_counter()
            self._on_tts_start(audio_data=audio_data, sample_rate=sample_rate)

        if self.no_tts:
            preview = (text or "").strip()
            tts_debug.log("tts", "speak_skipped", reason="no_tts", text_chars=len(text or ""))
            tts_debug.console("tts_skip", "no_tts flag")
            if preview:
                print(f"[TTS disabled] {preview[:120]}{'…' if len(preview) > 120 else ''}")
            return
        if not self._player:
            tts_debug.log("tts", "speak_skipped", reason="no_player", text_chars=len(text or ""))
            tts_debug.console("tts_skip", "no audio player")
            return
        if self._player and self._recorder:
            if self._cancel_generation.is_set():
                tts_debug.log("tts", "speak_skipped", reason="cancel_generation")
                return
            tts_debug.log(
                "tts",
                "speak_start",
                text_chars=len(text or ""),
                defer_assistant_speaking=defer_assistant_speaking,
                chunked=self.chunked_tts and not self.streaming_llm,
                prefetch=prepared_path is not None,
                transport_mode=self.transport_mode,
                local_playback=True,
            )
            pipeline_log.emit(
                "tts_speak",
                text_chars=len(text or ""),
                defer_assistant_speaking=defer_assistant_speaking,
                chunked=self.chunked_tts and not self.streaming_llm,
                prefetch=prepared_path is not None,
            )
            with self._speaking_lock:
                if not defer_assistant_speaking:
                    self._recorder.set_assistant_speaking(True)
                if self._controller:
                    self._controller.on_assistant_text(text)
                    if defer_assistant_speaking:
                        with self._streaming_controller_lock:
                            if not self._streaming_controller_tts_active:
                                self._controller.on_tts_start()
                                self._streaming_controller_tts_active = True
                    else:
                        self._controller.on_tts_start()

                try:
                    if prepared_path:
                        ok = self._player.play_prepared_file(
                            prepared_path,
                            on_start=_tts_playback_started,
                        )
                    else:
                        ok = self._player.speak(
                            text,
                            on_start=_tts_playback_started,
                            chunked=self.chunked_tts and not self.streaming_llm,
                        )
                    tts_debug.log(
                        "tts",
                        "speak_end",
                        completed=bool(ok),
                        backend=getattr(self._player, "tts_backend", None),
                    )
                except Exception as exc:
                    tts_debug.log("tts", "speak_error", error=str(exc))
                    tts_debug.console("tts_fail", str(exc))
                    raise
                finally:
                    if not defer_assistant_speaking:
                        if not self._recorder.is_barge_in_active():
                            self._recorder.set_assistant_speaking(False)
                        if self._controller:
                            self._controller.on_tts_end()

    def _on_tts_start(self, audio_data=None, sample_rate=None):
        """Provide TTS audio reference to the recorder for AEC."""
        if self._recorder and audio_data is not None and sample_rate:
            self._recorder.set_echo_reference(audio_data, sample_rate)

    def _on_partial_audio(self, audio_data):
        """Receive partial audio for streaming ASR in controller mode."""
        if not self._partial_audio_queue:
            return
        try:
            self._partial_audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass

    def _partial_transcribe_loop(self):
        """Background worker for streaming ASR partials (small model, not final STT)."""
        streaming_cpp = None
        moonshine_m = None
        whisper_m = None
        if not self._partial_acoustic_only:
            b = self._partial_stt_backend
            mid = self._partial_stt_model_id
            nt = self._partial_stt_threads
            try:
                if b == "whispercpp":
                    streaming_cpp = get_streaming_stt(mid, n_threads=nt)
                elif b == "moonshine":
                    moonshine_m = get_stt_model(
                        mid if str(mid).startswith("moonshine") else "moonshine:tiny"
                    )
                elif b == "whisper":
                    whisper_m = get_stt_model(mid)
            except Exception:
                streaming_cpp = None
                moonshine_m = None
                whisper_m = None

        while not self._shutdown_event.is_set():
            try:
                audio_data = self._partial_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self._partial_acoustic_only:
                    continue
                if streaming_cpp is not None:
                    text = streaming_cpp.transcribe_partial(audio_data)
                elif moonshine_m is not None:
                    text = moonshine_m.transcribe(audio_data)
                elif whisper_m is not None:
                    text = whisper_m.transcribe(audio_data)
                else:
                    text = ""
                self._print_live_partial(text or "")
                if self._controller:
                    self._controller.on_partial_transcript(text)
                self._turn_detector.on_partial_text(text or "")
                decision = self._router.route_partial(
                    self._route_context(text or "", is_partial=True)
                )
                if decision.action in {RouteAction.STOP_OUTPUT, RouteAction.SHUTDOWN}:
                    self._execute_route_decision(decision, text or "", None, None)
            except Exception:
                continue

    def run(self):
        """Run the voice assistant."""
        print("\n" + "=" * 50)
        print("  Cross-Platform Voice Assistant")
        print("=" * 50)
        if self.mode == "controller":
            print("Dialogue Controller: Enabled")

        # Initialize Memory
        if self.enable_memory:
            print("\nInitializing Memory...")
            try:
                self._memory = MemoryManager(
                    db_url=self.db_url,
                    session_id=self.session_id,
                    smart_save=self.memory_smart_save,
                    flush_interval_sec=self.memory_flush_interval_sec,
                    enable_embeddings=self.memory_enable_embeddings,
                    persist_roles=(
                        ("user", "assistant")
                        if self.memory_persist_assistant
                        else ("user",)
                    ),
                    memory_config=self._build_memory_config(),
                )
                stats = self._memory.get_conversation_stats()
                print("Memory initialized")
                if stats.get("total_messages", 0) > 0:
                    print(f"   {stats['total_messages']} messages in history")
            except Exception as e:
                print(f"Warning: Memory initialization failed: {e}")
                self._memory = None
        else:
            print("\nMemory: Disabled")

        # Initialize STT
        print("\nInitializing Speech-to-Text...")
        print(
            "STT runtime: "
            f"profile={self.runtime_profile}, "
            f"backend={self._stt_model_type}, "
            f"model={self.stt_model}, "
            f"threads={self._stt_threads}"
        )
        if self._stt_transcriber is transcribe_audio:
            get_stt_model(self.stt_model)
        else:
            print("Speech-to-Text initialized: injected transcriber")

        if self.mode == "controller":
            pb = self._partial_stt_backend
            pm = self._partial_stt_model_id
            pt = self._partial_stt_threads
            print(
                "Partial STT (realtime): "
                f"backend={pb}, model={pm}, threads={pt}"
            )
            if self._stt_transcriber is transcribe_audio:
                if pb == "whispercpp":
                    if not WHISPERCPP_AVAILABLE:
                        print(
                            "Partial STT: pywhispercpp not installed. "
                            "Using acoustic barge-in only (no live partial text)."
                        )
                        self._partial_acoustic_only = True
                    else:
                        st = get_streaming_stt(pm, n_threads=pt)
                        if st is None:
                            print(
                                "Partial STT: whisper.cpp could not load. "
                                "Using acoustic barge-in only (no live partial text)."
                            )
                            self._partial_acoustic_only = True
                elif pb == "moonshine":
                    try:
                        get_stt_model(
                            pm if str(pm).startswith("moonshine") else "moonshine:tiny"
                        )
                    except Exception as exc:
                        print(
                            f"Partial STT: moonshine unavailable ({exc}). "
                            "Using acoustic barge-in only."
                        )
                        self._partial_acoustic_only = True
                else:
                    try:
                        get_stt_model(pm)
                    except Exception as exc:
                        print(
                            f"Partial STT: could not load model ({exc}). "
                            "Using acoustic barge-in only."
                        )
                        self._partial_acoustic_only = True

        # Initialize LLM
        print("\nInitializing LLM...")
        try:
            llm_generation_profile = RUNTIME_PROFILES.get(
                self.runtime_profile, {}
            ).get("llm_generation_profile", "balanced")
            if self._injected_llm is not None:
                self._llm = self._injected_llm
            elif self._llm_factory is not None:
                self._llm = self._llm_factory(
                    llm_type="local",
                    model=self.llm_model,
                    generation_profile=llm_generation_profile,
                )
            else:
                self._llm = get_llm(
                    llm_type="local",
                    model=self.llm_model,
                    generation_profile=llm_generation_profile,
                )
            print(f"LLM initialized: {self.llm_model}")
            if self.streaming_llm:
                sm = (self.llm_stream_mode or "phrase").lower()
                if sm == "sentence":
                    print("LLM streaming: Enabled (sentence chunks)")
                elif sm == "token":
                    print(
                        "LLM streaming: Enabled (Ollama deltas → immediate TTS; "
                        "use word/phrase if audio is too choppy)"
                    )
                elif sm == "word":
                    print("LLM streaming: Enabled (word-by-word → TTS)")
                else:
                    print("LLM streaming: Enabled (phrase chunks / batched)")
        except Exception as e:
            print(f"Could not initialize LLM: {e}")
            print("   Make sure Ollama is running: ollama serve")
            return

        # Initialize audio
        print("\nInitializing audio...")
        try:
            if self.no_tts:
                print("TTS: disabled (--no-tts); assistant text only")
                self._player = None
            else:
                if self.tts_debug_enabled:
                    print(
                        "TTS debug: enabled (SPEAKER_TTS_DEBUG / tts_debug) — "
                        "grep speaker.tts in logs"
                    )
                player_kwargs = {
                    "output_device": self.output_device,
                    "voice": self.tts_voice,
                    "tts_backend": self.tts_backend,
                    "tts_model": self.tts_model,
                    "playback_backend": self.playback_backend,
                }
                if self._injected_player is not None:
                    self._player = self._injected_player
                elif self._audio_player_factory is not None:
                    self._player = self._audio_player_factory(**player_kwargs)
                else:
                    self._player = AudioPlayer(**player_kwargs)
            if self.mode == "controller":
                self._controller = DialogueController(
                    **self.controller_config,
                    diagnostics_log_path=self.diagnostics_log_path,
                )
                self._partial_audio_queue = queue.Queue(maxsize=2)
                self._partial_worker = threading.Thread(
                    target=self._partial_transcribe_loop, daemon=True
                )
                self._partial_worker.start()
            recorder_kwargs = {
                "callback": self._on_speech_detected,
                "vad_threshold": self.vad_threshold,
                "device": self.input_device,
                "on_interrupt": self._on_barge_in,
                "barge_in_pre_roll_sec": self.barge_in_pre_roll_sec,
                "barge_in_min_speech_sec": self.barge_in_min_speech_sec,
                "barge_in_rms_ratio": self.barge_in_rms_ratio,
                "barge_in_cooldown_sec": self.barge_in_cooldown_sec,
                "use_webrtcvad": self.barge_in_use_webrtcvad,
                "adaptive_vad": self.adaptive_vad,
                "vad_noise_multiplier": self.vad_noise_multiplier,
                "vad_noise_floor_min": self.vad_noise_floor_min,
                "silence_duration": self.silence_duration,
                "aec_enabled": self.aec_enabled,
                "aec_strength": self.aec_strength,
                "aec_filter_ms": self.aec_filter_ms,
                "aec_max_ref_sec": self.aec_max_ref_sec,
                "simple_voiced_fallback": self.simple_voiced_fallback,
                "barge_in_min_delay_sec": self.barge_in_min_delay_sec,
                "barge_in_min_delay_after_ref_sec": self.barge_in_min_delay_after_ref_sec,
                "barge_in_min_rms_ratio": self.barge_in_min_rms_ratio,
                "echo_corr_threshold": self.echo_corr_threshold,
                "partial_callback": (
                    self._on_partial_audio
                    if self.mode == "controller"
                    else None
                ),
                "partial_interval_sec": self.partial_interval_sec,
                "wakeword_enabled": self.wakeword_enabled,
                "wakeword": self.wakeword,
                "wakeword_threshold": self.wakeword_threshold,
                "wakeword_timeout_sec": self.wakeword_timeout_sec,
                "wakeword_model_path": self.wakeword_model_path,
                "wakeword_service_mode": self.wakeword_service_mode,
                "wakeword_policy": self.wakeword_policy,
                "wakeword_miss_limit": self.wakeword_miss_limit,
                "wakeword_recovery_window_sec": self.wakeword_recovery_window_sec,
                "speaker_verify_enabled": self.speaker_verify_enabled,
                "speaker_enrollment_wav": self.speaker_enrollment_wav,
                "speaker_verify_threshold": self.speaker_verify_threshold,
                "diagnostics_log_path": self.diagnostics_log_path,
                "diagnostics_log_frames": self.diagnostics_log_frames,
                "console_print_lock": self._console_print_lock,
            }
            if self._injected_recorder is not None:
                self._recorder = self._injected_recorder
            elif self._recorder_factory is not None:
                self._recorder = self._recorder_factory(**recorder_kwargs)
            else:
                self._recorder = AudioRecorder(**recorder_kwargs)
            print("Audio initialized (NLMS AEC + Silero VAD + barge-in)")
            if self._recorder.wakeword_enabled:
                print(
                    "Wakeword gate: enabled "
                    f"(threshold={self._recorder.wakeword_threshold:.2f}, "
                    f"window={self._recorder.wakeword_timeout_sec:.1f}s)"
                )
                if self._recorder.wakeword:
                    print(f"Wakeword target: {self._recorder.wakeword}")
            else:
                print("Wakeword gate: disabled")
            if self._recorder.speaker_verify_enabled:
                print(
                    "Speaker verification: enabled "
                    f"(threshold={self._recorder.speaker_verify_threshold:.2f})"
                )
            else:
                print("Speaker verification: disabled")
            if self.diagnostics_log_path:
                frame_mode = "full-frame" if self.diagnostics_log_frames else "events-only"
                print(
                    "Diagnostics log: "
                    f"{self.diagnostics_log_path} ({frame_mode})"
                )
        except Exception as e:
            print(f"Could not initialize audio: {e}")
            import traceback
            traceback.print_exc()
            return

        # Startup calibration
        if self.adaptive_vad and self.calibrate_on_start:
            print("\nCalibrating ambient noise (please stay silent)...")
            result = self._recorder.calibrate(self.calibrate_duration_sec)
            if result.get("calibrated"):
                print(
                    f"Calibration complete "
                    f"(noise_floor={result.get('noise_floor'):.4f}, "
                    f"samples={result.get('samples')})"
                )
            else:
                print(f"Warning: Calibration skipped: {result.get('reason')}")

        # Start listening
        print("\n" + "-" * 50)
        print("Listening... (say 'stop' or 'quit' to exit)")
        print("Tip: You can interrupt the assistant by speaking!")
        if self._memory:
            print("Memory: Active (I remember our conversations)")
        print("-" * 50 + "\n")
        self._print_startup_latency_notice()

        self._session_mux.start()
        print(f"Transport mode: {self.transport_mode} ({', '.join(self._session_mux.active_transports()) or 'none'})")
        self._recorder.start()

        try:
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=0.1)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            self._cleanup()

    def _cleanup(self):
        print("\nCleaning up...")
        self._cancel_generation.set()
        if self._player:
            try:
                self._player.stop()
            except Exception:
                pass
        if self._recorder:
            try:
                self._recorder.set_assistant_speaking(False)
            except Exception:
                pass
        if self._recorder:
            self._recorder.stop()
        self._session_mux.stop()
        if self._player:
            self._player.cleanup()
        if self._response_thread and self._response_thread.is_alive():
            self._response_thread.join(timeout=1.0)
        if self._memory:
            stats = self._memory.get_conversation_stats()
            print(
                f"   Session saved ({stats.get('recent_messages', 0)} messages)"
            )
            self._memory.close()
        print("Cleanup complete")

    def shutdown(self):
        self._cancel_generation.set()
        if self._player:
            try:
                self._player.stop()
            except Exception:
                pass
        self._shutdown_event.set()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Platform Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--stt-model", type=str, default=None)
    parser.add_argument("--tts-voice", type=str, default=None)
    parser.add_argument(
        "--tts-backend", type=str, default=None,
        choices=["kokoro", "supertonic", "piper", "melotts"],
        help="Force a local open-source TTS backend",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        default=False,
        help="Disable speech synthesis and speaker playback (text-only assistant)",
    )
    parser.add_argument(
        "--tts-debug",
        action="store_true",
        default=None,
        help="Verbose TTS pipeline logs (or set SPEAKER_TTS_DEBUG=1 / tts_debug in config)",
    )
    parser.add_argument(
        "--playback-backend",
        type=str,
        default=None,
        choices=["auto", "sounddevice", "pygame"],
        help=(
            "Audio output engine for TTS (default auto: sounddevice/PortAudio "
            "so --output-device matches mic routing; pygame/SDL fallback)"
        ),
    )
    parser.add_argument(
        "--tts-model", type=str, default=None,
        help="Voice/model override for the TTS backend "
             "(e.g. 'en_US-lessac-medium' for Piper, 'ZH' for MeloTTS)",
    )
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=None,
        help="Seconds of silence before end-of-utterance (default follows runtime_profile)",
    )
    parser.add_argument(
        "--llm-min-phrase-words",
        type=int,
        default=None,
        help="Minimum words before emitting first streamed phrase to TTS (default follows runtime_profile)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["asr", "controller"], default=None,
    )
    parser.add_argument(
        "--profile", type=str, choices=list(RESOURCE_PROFILES.keys()),
        default=None, help="Resource profile preset (desktop, low, server)",
    )
    parser.add_argument(
        "--runtime-profile",
        type=str,
        choices=list(RUNTIME_PROFILES.keys()),
        default=None,
        help="Runtime compute profile (edge, balanced, max_quality)",
    )
    parser.add_argument(
        "--transport-mode",
        type=str,
        choices=[m.value for m in TransportMode],
        default=None,
        help="Transport mode (local_lan, webrtc, hybrid)",
    )
    parser.add_argument("--partial-interval", type=float, default=None)
    parser.add_argument(
        "--partial-stt-model",
        type=str,
        default=None,
        help="Lightweight model for live partial STT in controller mode (e.g. tiny, base).",
    )
    parser.add_argument(
        "--partial-stt-backend",
        type=str,
        choices=["whispercpp", "whisper", "moonshine"],
        default=None,
        help="Backend for partial STT (default: whispercpp / config).",
    )
    parser.add_argument(
        "--partial-stt-threads",
        type=int,
        default=None,
        help="Thread count for partial whisper.cpp (default: same as final STT).",
    )
    parser.add_argument(
        "--llm-stream-mode",
        type=str,
        choices=["phrase", "sentence", "token", "word"],
        default=None,
        help=(
            "Ollama streaming: phrase (batched words), sentence, "
            "token (each API delta — fastest), word (each word)."
        ),
    )
    parser.add_argument(
        "--live-partial",
        type=str,
        choices=["on", "off"],
        default=None,
        help="Print realtime partial STT lines (default: on in controller mode via config).",
    )
    parser.add_argument(
        "--live-partial-mode",
        type=str,
        choices=["overwrite", "newline"],
        default=None,
        help="How to show partials: overwrite one line (ANSI) or newline each update.",
    )
    parser.add_argument("--controller-min-interrupt-delay", type=float, default=None)
    parser.add_argument("--controller-min-barge-in", type=float, default=None)
    parser.add_argument("--controller-min-partial-chars", type=int, default=None)
    parser.add_argument("--controller-max-partial-age", type=float, default=None)
    parser.add_argument("--controller-echo-similarity", type=float, default=None)
    parser.add_argument("--controller-allow-rms-fallback", action="store_true", default=None)
    parser.add_argument("--controller-require-partial", action="store_true", default=None)
    parser.add_argument("--controller-strong-voiced-multiplier", type=float, default=None)
    parser.add_argument(
        "--controller-interruption-strategy",
        type=str,
        choices=["strict_echo_protect", "balanced", "aggressive_user_takeover"],
        default=None,
    )
    parser.add_argument("--controller-ignore-phrases", type=str, default=None)
    parser.add_argument("--adaptive-vad", action="store_true", default=None)
    parser.add_argument("--vad-noise-multiplier", type=float, default=None)
    parser.add_argument("--vad-noise-floor-min", type=float, default=None)
    parser.add_argument("--calibrate-on-start", action="store_true", default=None)
    parser.add_argument("--calibrate-duration", type=float, default=None)
    parser.add_argument("--aec-enabled", action="store_true", default=None)
    parser.add_argument("--aec-strength", type=float, default=None)
    parser.add_argument("--aec-filter-ms", type=float, default=None)
    parser.add_argument("--aec-max-ref-sec", type=float, default=None)
    parser.add_argument("--simple-voiced-fallback", action="store_true", default=None)
    parser.add_argument("--barge-in-debug", action="store_true", default=None)
    parser.add_argument("--barge-in-min-delay", type=float, default=None)
    parser.add_argument("--echo-corr-threshold", type=float, default=None)
    parser.add_argument("--barge-in-min-delay-after-ref", type=float, default=None)
    parser.add_argument("--barge-in-min-rms-ratio", type=float, default=None)
    parser.add_argument("--stop-mode", type=str, choices=["exact", "prefix"], default=None)
    parser.add_argument("--stop-phrases", type=str, default=None)
    parser.add_argument("--barge-in-pre-roll", type=float, default=None)
    parser.add_argument("--barge-in-min-speech", type=float, default=None)
    parser.add_argument("--barge-in-rms-ratio", type=float, default=None)
    parser.add_argument("--barge-in-cooldown", type=float, default=None)
    parser.add_argument("--barge-in-use-webrtcvad", action="store_true", default=None)
    parser.add_argument("--wakeword-enabled", action="store_true", default=None)
    parser.add_argument(
        "--list-wakewords",
        action="store_true",
        help="List discoverable wakeword labels and exit",
    )
    parser.add_argument("--wakeword", type=str, default=None)
    parser.add_argument("--wakeword-threshold", type=float, default=None)
    parser.add_argument("--wakeword-timeout-sec", type=float, default=None)
    parser.add_argument("--wakeword-model-path", type=str, default=None)
    parser.add_argument(
        "--wakeword-service-mode",
        type=str,
        choices=["local", "process"],
        default=None,
        help="Wakeword detection runtime mode",
    )
    parser.add_argument(
        "--wakeword-policy",
        type=str,
        choices=["strict_required", "hybrid_recovery", "legacy_compatible"],
        default=None,
    )
    parser.add_argument("--wakeword-miss-limit", type=int, default=None)
    parser.add_argument("--wakeword-recovery-window-sec", type=float, default=None)
    parser.add_argument("--speaker-verify-enabled", action="store_true", default=None)
    parser.add_argument("--speaker-enrollment-wav", type=str, default=None)
    parser.add_argument("--speaker-verify-threshold", type=float, default=None)
    parser.add_argument("--diagnostics-log-path", type=str, default=None)
    parser.add_argument(
        "--diagnostics-log-frames",
        action="store_true",
        default=None,
        help="Log per-frame barge-in diagnostics (very verbose)",
    )
    parser.add_argument(
        "--trace-backends",
        action="store_true",
        default=False,
        help=(
            "Log each Ollama chat and STT inference to stderr; "
            "with --diagnostics-log-path also append backend_trace JSON lines"
        ),
    )
    parser.add_argument(
        "--enroll-speaker",
        type=str,
        default=None,
        help="Record speaker enrollment WAV to this path, then exit",
    )
    parser.add_argument(
        "--enroll-duration-sec",
        type=float,
        default=6.0,
        help="Enrollment recording duration in seconds",
    )
    parser.add_argument("--streaming-llm", action="store_true", default=None,
                        help="Enable streaming LLM responses (sentence-by-sentence TTS)")
    parser.add_argument("--no-streaming-llm", action="store_true", default=False,
                        help="Disable streaming LLM responses")
    parser.add_argument("--chunked-tts", action="store_true", default=None,
                        help="Enable chunked TTS playback")
    parser.add_argument("--no-chunked-tts", action="store_true", default=False,
                        help="Disable chunked TTS playback")
    parser.add_argument(
        "--assistant-stream-print",
        type=str,
        choices=["overwrite", "newline"],
        default=None,
        help=(
            "How to print streaming assistant text in token/word mode "
            "(default: newline — avoids collision with mic RMS meter)."
        ),
    )
    parser.add_argument(
        "--streaming-tts-prefetch",
        action="store_true",
        default=None,
        help="Synth next TTS chunk while playing current (streaming LLM; default on).",
    )
    parser.add_argument(
        "--no-streaming-tts-prefetch",
        action="store_true",
        default=False,
        help="Disable streaming TTS prefetch.",
    )
    parser.add_argument(
        "--llm-stream-coalesce-min-words",
        type=int,
        default=None,
        help="Min words before flushing a coalesced token/word chunk to TTS (default 2).",
    )
    parser.add_argument(
        "--llm-stream-coalesce-max-words",
        type=int,
        default=None,
        help="Max words per coalesced chunk before forcing a flush (default 6).",
    )
    parser.add_argument(
        "--llm-stream-coalesce-flush-sec",
        type=float,
        default=None,
        help="Max seconds to hold coalesced text before flushing without punctuation.",
    )
    # Memory
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--new-session", action="store_true")
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--db-url", type=str, default=None)
    parser.add_argument(
        "--memory-flush-interval",
        type=float,
        default=None,
        help="Seconds between buffered smart-save flushes to Postgres.",
    )
    parser.add_argument(
        "--memory-no-smart-save",
        action="store_true",
        default=False,
        help="Persist messages immediately without smart filtering/buffering.",
    )
    parser.add_argument(
        "--memory-enable-embeddings",
        action="store_true",
        default=None,
        help="Enable sentence-transformer embeddings for persisted memory.",
    )
    parser.add_argument(
        "--memory-persist-assistant",
        action="store_true",
        default=None,
        help="Also persist assistant replies. Default persists only user speech.",
    )
    parser.add_argument(
        "--memory-no-llm-clean",
        action="store_true",
        default=False,
        help="Disable LLM cleanup before buffered memory flush.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help=(
            "Record this session (mic + TTS audio per turn) to recordings/. "
            "Run 'python scripts/generate_session_tests.py' afterwards to turn "
            "recordings into regression tests."
        ),
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return
    if args.list_wakewords:
        print("Known wakeword labels:")
        for label in list_known_wakewords():
            print(f"  - {label}")
        probe = OpenWakeWordGate()
        if probe.available and probe.available_labels:
            print("\nDetected labels from local model runtime:")
            for label in probe.available_labels:
                print(f"  - {label}")
        elif not probe.available:
            print("\nopenWakeWord runtime unavailable in this environment.")
        return
    if args.enroll_speaker:
        result = create_speaker_enrollment_wav(
            output_path=args.enroll_speaker,
            duration_sec=args.enroll_duration_sec,
            device=args.input_device,
        )
        if result.get("ok"):
            print(f"Enrollment sample saved: {result.get('path')}")
            print(
                "Use it with: --speaker-verify-enabled "
                f"--speaker-enrollment-wav {result.get('path')}"
            )
            return
        print(f"Enrollment failed: {result.get('reason')}")
        return

    config = load_config()

    # Apply resource profile (lowest priority – overridden by config and args)
    profile = {}
    selected_profile = args.profile or config.get("profile")
    if selected_profile:
        profile = RESOURCE_PROFILES.get(selected_profile, {})

    # Helper to resolve: args > (profile bundle if active) > config > profile > default
    def resolve(arg_val, config_key, profile_key=None, default=None):
        if arg_val is not None:
            return arg_val
        pk = profile_key or config_key
        cfg_val = config.get(config_key) if config_key else None
        prof_val = profile.get(pk) if pk else None
        if (
            selected_profile
            and pk in PROFILE_BUNDLE_KEYS
            and prof_val is not None
        ):
            return prof_val
        if cfg_val is not None:
            return cfg_val
        if prof_val is not None:
            return prof_val
        return default

    llm_model = resolve(args.llm_model, "llm_model", "llm_model", "llama2")
    stt_model = resolve(args.stt_model, "stt_model", "stt_model", "base")
    tts_voice = resolve(args.tts_voice, "tts_voice", None, "en-US")
    tts_backend = resolve(args.tts_backend, "tts_backend", "tts_backend", None)
    tts_model = resolve(getattr(args, "tts_model", None), "tts_model", "tts_model", None)
    no_tts = bool(getattr(args, "no_tts", False) or config.get("no_tts", False))
    tts_debug_enabled = tts_debug.resolve_enabled(
        cli_value=getattr(args, "tts_debug", None),
        config_value=config.get("tts_debug"),
    )
    playback_backend = resolve(
        getattr(args, "playback_backend", None),
        "playback_backend",
        None,
        "auto",
    )
    local_only = bool(config.get("local_only", True))
    if local_only and tts_backend in {"edge-tts", "gtts"}:
        print(
            f"Configured TTS backend '{tts_backend}' is not allowed in local_only mode. "
            "Falling back to local auto-selection."
        )
        tts_backend = None
    vad_threshold = resolve(args.vad_threshold, "vad_threshold", None, 0.01)
    mode = resolve(args.mode, "mode", None, "controller")
    runtime_profile = resolve(
        args.runtime_profile, "runtime_profile", None, "balanced"
    )
    _conv = RUNTIME_CONVERSATION_DEFAULTS.get(
        runtime_profile, RUNTIME_CONVERSATION_DEFAULTS["balanced"]
    )
    transport_mode = resolve(
        args.transport_mode, "transport_mode", None, "local_lan"
    )
    partial_interval_sec = resolve(args.partial_interval, "partial_interval_sec", None, 1.0)
    partial_stt_model = resolve(args.partial_stt_model, "partial_stt_model", None, None)
    partial_stt_backend = resolve(
        args.partial_stt_backend, "partial_stt_backend", None, None
    )
    partial_stt_threads = resolve(
        args.partial_stt_threads, "partial_stt_threads", None, None
    )
    llm_stream_mode = resolve(args.llm_stream_mode, "llm_stream_mode", None, "phrase")
    llm_min_phrase_words = resolve(
        args.llm_min_phrase_words,
        "llm_min_phrase_words",
        None,
        int(_conv["llm_min_phrase_words"]),
    )
    silence_duration = resolve(
        args.silence_duration,
        "silence_duration",
        None,
        float(_conv["silence_duration"]),
    )
    if args.live_partial is not None:
        live_partial_log = args.live_partial == "on"
    else:
        lp_cfg = config.get("live_partial_log")
        if lp_cfg is not None:
            live_partial_log = bool(lp_cfg)
        else:
            live_partial_log = mode == "controller"
    live_partial_mode = (
        args.live_partial_mode
        or config.get("live_partial_mode")
        or "overwrite"
    )
    if live_partial_mode not in ("overwrite", "newline"):
        live_partial_mode = "overwrite"
    adaptive_vad = resolve(args.adaptive_vad, "adaptive_vad", "adaptive_vad", True)
    vad_noise_multiplier = resolve(args.vad_noise_multiplier, "vad_noise_multiplier", None, 2.5)
    vad_noise_floor_min = resolve(args.vad_noise_floor_min, "vad_noise_floor_min", None, 0.003)
    calibrate_on_start = resolve(args.calibrate_on_start, "calibrate_on_start", "calibrate_on_start", True)
    calibrate_duration_sec = resolve(args.calibrate_duration, "calibrate_duration_sec", None, 2.5)
    aec_enabled = resolve(args.aec_enabled, "aec_enabled", None, True)
    aec_strength = resolve(args.aec_strength, "aec_strength", None, 0.3)
    aec_filter_ms = resolve(args.aec_filter_ms, "aec_filter_ms", "aec_filter_ms", 80.0)
    aec_max_ref_sec = resolve(args.aec_max_ref_sec, "aec_max_ref_sec", None, 20.0)
    simple_voiced_fallback = resolve(args.simple_voiced_fallback, "simple_voiced_fallback", None, True)
    barge_in_debug = resolve(args.barge_in_debug, "barge_in_debug", None, False)
    barge_in_min_delay_sec = resolve(args.barge_in_min_delay, "barge_in_min_delay_sec", None, 0.5)
    echo_corr_threshold = resolve(args.echo_corr_threshold, "echo_corr_threshold", None, 0.45)
    barge_in_min_delay_after_ref_sec = resolve(
        args.barge_in_min_delay_after_ref, "barge_in_min_delay_after_ref_sec", None, 0.20
    )
    barge_in_min_rms_ratio = resolve(args.barge_in_min_rms_ratio, "barge_in_min_rms_ratio", None, 3.0)
    stop_mode = resolve(args.stop_mode, "stop_mode", None, "exact")
    stop_phrases = config.get("stop_phrases", ["stop", "quit", "exit"])
    if args.stop_phrases is not None:
        stop_phrases = [p.strip().lower() for p in args.stop_phrases.split(",") if p.strip()]
    barge_in_pre_roll_sec = resolve(args.barge_in_pre_roll, "barge_in_pre_roll_sec", None, 0.3)
    barge_in_min_speech_sec = resolve(args.barge_in_min_speech, "barge_in_min_speech_sec", None, 0.15)
    barge_in_rms_ratio = resolve(args.barge_in_rms_ratio, "barge_in_rms_ratio", None, 2.0)
    barge_in_cooldown_sec = resolve(args.barge_in_cooldown, "barge_in_cooldown_sec", None, 0.5)
    barge_in_use_webrtcvad = resolve(args.barge_in_use_webrtcvad, "barge_in_use_webrtcvad", None, True)
    wakeword_enabled = resolve(args.wakeword_enabled, "wakeword_enabled", None, False)
    wakeword = resolve(args.wakeword, "wakeword", None, None)
    wakeword_threshold = resolve(args.wakeword_threshold, "wakeword_threshold", None, 0.5)
    wakeword_timeout_sec = resolve(args.wakeword_timeout_sec, "wakeword_timeout_sec", None, 5.0)
    wakeword_model_path = resolve(args.wakeword_model_path, "wakeword_model_path", None, None)
    wakeword_service_mode = resolve(
        args.wakeword_service_mode, "wakeword_service_mode", None, "local"
    )
    wakeword_policy = resolve(
        args.wakeword_policy, "wakeword_policy", None, "strict_required"
    )
    wakeword_miss_limit = resolve(
        args.wakeword_miss_limit, "wakeword_miss_limit", None, 80
    )
    wakeword_recovery_window_sec = resolve(
        args.wakeword_recovery_window_sec,
        "wakeword_recovery_window_sec",
        None,
        3.0,
    )
    speaker_verify_enabled = resolve(
        args.speaker_verify_enabled, "speaker_verify_enabled", None, False
    )
    speaker_enrollment_wav = resolve(
        args.speaker_enrollment_wav, "speaker_enrollment_wav", None, None
    )
    speaker_verify_threshold = resolve(
        args.speaker_verify_threshold, "speaker_verify_threshold", None, 0.55
    )
    diagnostics_log_path = resolve(
        args.diagnostics_log_path, "diagnostics_log_path", None, None
    )
    diagnostics_log_frames = resolve(
        args.diagnostics_log_frames, "diagnostics_log_frames", None, False
    )
    trace_backends = bool(args.trace_backends) or bool(
        config.get("backend_trace", False)
    )

    if wakeword_enabled:
        probe = OpenWakeWordGate(model_path=wakeword_model_path)
        ok, reason = validate_wakeword_name(
            wakeword,
            getattr(probe, "available_labels", []),
        )
        if not ok:
            print(f"Configuration error: {reason}")
            print("Tip: run with --list-wakewords")
            return

    runtime_issues = validate_runtime_config(
        {
            "barge_in_min_delay_sec": barge_in_min_delay_sec,
            "barge_in_min_delay_after_ref_sec": barge_in_min_delay_after_ref_sec,
            "barge_in_min_rms_ratio": barge_in_min_rms_ratio,
            "echo_corr_threshold": echo_corr_threshold,
            "aec_filter_ms": aec_filter_ms,
            "wakeword_threshold": wakeword_threshold,
            "speaker_verify_threshold": speaker_verify_threshold,
            "wakeword_policy": wakeword_policy,
        }
    )
    runtime_issues.extend(
        validate_profile_transport_config(
            selected_profile,
            runtime_profile,
            transport_mode,
        )
    )
    if runtime_issues:
        print("Configuration warnings:")
        for issue in runtime_issues:
            print(f"  - {issue}")

    # Streaming LLM
    streaming_llm = resolve(
        args.streaming_llm, "streaming_llm", "streaming_llm", True
    )
    if args.no_streaming_llm:
        streaming_llm = False

    # Chunked TTS
    chunked_tts = resolve(
        args.chunked_tts, "chunked_tts", "chunked_tts", True
    )
    if args.no_chunked_tts:
        chunked_tts = False

    assistant_stream_print = resolve(
        args.assistant_stream_print, "assistant_stream_print", None, None
    )
    streaming_tts_prefetch = resolve(
        args.streaming_tts_prefetch,
        "streaming_tts_prefetch",
        "streaming_tts_prefetch",
        True,
    )
    if args.no_streaming_tts_prefetch:
        streaming_tts_prefetch = False
    llm_stream_coalesce_min_words = resolve(
        args.llm_stream_coalesce_min_words,
        "llm_stream_coalesce_min_words",
        None,
        2,
    )
    llm_stream_coalesce_max_words = resolve(
        args.llm_stream_coalesce_max_words,
        "llm_stream_coalesce_max_words",
        None,
        6,
    )
    llm_stream_coalesce_flush_sec = resolve(
        args.llm_stream_coalesce_flush_sec,
        "llm_stream_coalesce_flush_sec",
        None,
        0.35,
    )

    ignore_phrases = config.get("controller_ignore_phrases", ["", ".", "uh", "um"])
    if args.controller_ignore_phrases is not None:
        ignore_phrases = [p.strip().lower() for p in args.controller_ignore_phrases.split(",")]
    controller_config = {
        "min_interrupt_delay_sec": resolve(
            args.controller_min_interrupt_delay, "controller_min_interrupt_delay_sec", None, 0.2
        ),
        "min_barge_in_sec": resolve(
            args.controller_min_barge_in, "controller_min_barge_in_sec", None, 0.12
        ),
        "min_partial_chars": resolve(
            args.controller_min_partial_chars, "controller_min_partial_chars", None, 3
        ),
        "max_partial_age_sec": resolve(
            args.controller_max_partial_age, "controller_max_partial_age_sec", None, 1.5
        ),
        "echo_similarity_threshold": resolve(
            args.controller_echo_similarity, "controller_echo_similarity_threshold", None, 0.7
        ),
        "allow_rms_fallback": resolve(
            args.controller_allow_rms_fallback, "controller_allow_rms_fallback", None, False
        ),
        "require_partial_for_barge_in": resolve(
            args.controller_require_partial, "controller_require_partial_for_barge_in", None, False
        ),
        "strong_voiced_multiplier": resolve(
            args.controller_strong_voiced_multiplier, "controller_strong_voiced_multiplier", None, 2.0
        ),
        "interruption_strategy": resolve(
            args.controller_interruption_strategy,
            "controller_interruption_strategy",
            None,
            "balanced",
        ),
        "ignore_phrases": tuple(ignore_phrases),
    }

    session_id = None if args.new_session else args.session_id
    memory_smart_save = not args.memory_no_smart_save and bool(
        config.get("memory_smart_save", True)
    )
    memory_flush_interval_sec = resolve(
        args.memory_flush_interval,
        "memory_flush_interval_sec",
        None,
        240.0,
    )
    memory_enable_embeddings = resolve(
        args.memory_enable_embeddings,
        "memory_enable_embeddings",
        None,
        False,
    )
    memory_persist_assistant = resolve(
        args.memory_persist_assistant,
        "memory_persist_assistant",
        None,
        False,
    )
    memory_llm_clean = not args.memory_no_llm_clean and bool(
        config.get("memory_llm_clean", True)
    )

    input_device = args.input_device if args.input_device is not None else config.get("input_device")
    configured_output = (
        args.output_device if args.output_device is not None else config.get("output_device")
    )
    output_device = resolve_output_device(input_device, configured_output)
    if output_device is not None and configured_output is None:
        try:
            import sounddevice as _sd

            out_name = _sd.query_devices(output_device).get("name", output_device)
            print(
                f"TTS output: device {output_device} ({out_name}) "
                "(matched to microphone; set output_device in config.json to override)"
            )
        except Exception:
            print(f"TTS output: device {output_device} (matched to microphone)")

    assistant = VoiceAssistant(
        llm_model=llm_model,
        stt_model=stt_model,
        input_device=input_device,
        output_device=output_device,
        vad_threshold=vad_threshold,
        silence_duration=silence_duration,
        tts_voice=tts_voice,
        tts_backend=tts_backend,
        tts_model=tts_model,
        barge_in_pre_roll_sec=barge_in_pre_roll_sec,
        barge_in_min_speech_sec=barge_in_min_speech_sec,
        barge_in_rms_ratio=barge_in_rms_ratio,
        barge_in_cooldown_sec=barge_in_cooldown_sec,
        barge_in_use_webrtcvad=barge_in_use_webrtcvad,
        mode=mode,
        controller_config=controller_config,
        partial_interval_sec=partial_interval_sec,
        adaptive_vad=adaptive_vad,
        vad_noise_multiplier=vad_noise_multiplier,
        vad_noise_floor_min=vad_noise_floor_min,
        calibrate_on_start=calibrate_on_start,
        calibrate_duration_sec=calibrate_duration_sec,
        aec_enabled=aec_enabled,
        aec_strength=aec_strength,
        aec_filter_ms=aec_filter_ms,
        aec_max_ref_sec=aec_max_ref_sec,
        simple_voiced_fallback=simple_voiced_fallback,
        barge_in_debug=barge_in_debug,
        barge_in_min_delay_sec=barge_in_min_delay_sec,
        echo_corr_threshold=echo_corr_threshold,
        barge_in_min_delay_after_ref_sec=barge_in_min_delay_after_ref_sec,
        barge_in_min_rms_ratio=barge_in_min_rms_ratio,
        stop_mode=stop_mode,
        stop_phrases=tuple(stop_phrases),
        enable_memory=not args.no_memory,
        session_id=session_id,
        db_url=args.db_url,
        memory_smart_save=memory_smart_save,
        memory_flush_interval_sec=memory_flush_interval_sec,
        memory_enable_embeddings=memory_enable_embeddings,
        memory_persist_assistant=memory_persist_assistant,
        memory_llm_clean=memory_llm_clean,
        memory_config=config,
        streaming_llm=streaming_llm,
        chunked_tts=chunked_tts,
        runtime_profile=runtime_profile,
        transport_mode=transport_mode,
        wakeword_enabled=wakeword_enabled,
        wakeword=wakeword,
        wakeword_threshold=wakeword_threshold,
        wakeword_timeout_sec=wakeword_timeout_sec,
        wakeword_model_path=wakeword_model_path,
        wakeword_service_mode=wakeword_service_mode,
        wakeword_policy=wakeword_policy,
        wakeword_miss_limit=wakeword_miss_limit,
        wakeword_recovery_window_sec=wakeword_recovery_window_sec,
        speaker_verify_enabled=speaker_verify_enabled,
        speaker_enrollment_wav=speaker_enrollment_wav,
        speaker_verify_threshold=speaker_verify_threshold,
        diagnostics_log_path=diagnostics_log_path,
        diagnostics_log_frames=diagnostics_log_frames,
        trace_backends=trace_backends,
        partial_stt_model=partial_stt_model,
        partial_stt_backend=partial_stt_backend,
        partial_stt_threads=partial_stt_threads,
        llm_stream_mode=llm_stream_mode,
        llm_min_phrase_words=llm_min_phrase_words,
        live_partial_log=live_partial_log,
        live_partial_mode=live_partial_mode,
        assistant_stream_print=assistant_stream_print,
        streaming_tts_prefetch=streaming_tts_prefetch,
        llm_stream_coalesce_min_words=llm_stream_coalesce_min_words,
        llm_stream_coalesce_max_words=llm_stream_coalesce_max_words,
        llm_stream_coalesce_flush_sec=llm_stream_coalesce_flush_sec,
        no_tts=no_tts,
        playback_backend=playback_backend,
        tts_debug_enabled=tts_debug_enabled,
    )

    # Session recorder (opt-in via --record flag).
    #
    # The recorder (assistant._recorder) is created INSIDE assistant.run(), so
    # it is None here.  We use lazy one-time attachment: the first time
    # _on_tts_start fires the recorder is already live and we hook into it then.
    _session_rec = None
    if args.record:
        try:
            from utils.session_recorder import SessionRecorder

            _session_rec = SessionRecorder(profile=selected_profile or "default")
            _rec_attached = [False]  # mutable flag for closure

            _orig_on_tts_start = assistant._on_tts_start

            def _recording_on_tts_start(audio_data=None, sample_rate=None):
                # Forward to original AEC-reference hook first
                _orig_on_tts_start(audio_data=audio_data, sample_rate=sample_rate)

                rec = assistant._recorder  # now available (we're inside run())

                # One-time lazy attachment to the recorder
                if not _rec_attached[0] and rec is not None:
                    _rec_attached[0] = True
                    _session_rec.recorder = rec
                    # start() already calls _attach_recorder_hooks() when recorder is set;
                    # calling it again would wrap on_interrupt twice and duplicate barge-in events.
                    _session_rec.start()

                    # Wrap set_assistant_speaking(False) to signal TTS end
                    _orig_set_speaking = rec.set_assistant_speaking

                    def _recording_set_speaking(val):
                        _orig_set_speaking(val)
                        if not val:
                            _session_rec.on_tts_end()

                    rec.set_assistant_speaking = _recording_set_speaking

                    sr = getattr(rec, "device_sample_rate", "?")
                    nf = getattr(rec, "_noise_floor", None)
                    print(
                        f"\nSession recording active → recordings/{_session_rec.session_id}/"
                        f"  (SR={sr}, noise_floor={nf})"
                    )

                if _session_rec.recorder is not None and audio_data is not None and sample_rate:
                    _session_rec.on_tts_start(audio_data, sample_rate)

            assistant._on_tts_start = _recording_on_tts_start
            print("[INFO] Session recording enabled — will activate on first TTS turn")

        except Exception as exc:
            import traceback
            print(f"[WARNING] Session recorder could not start: {exc}")
            traceback.print_exc()
            _session_rec = None

    def signal_handler(signum, frame):
        print("\n\nShutdown signal received")
        assistant.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        assistant.run()
    finally:
        if _session_rec is not None:
            _session_rec.stop()
            try:
                saved = _session_rec.save()
                print(f"\nSession saved to: {saved}")
                print("  → python scripts/generate_session_tests.py  (create regression tests)")
                print("  → python scripts/analyze_sessions.py         (view analytics)")
            except Exception as exc:
                print(f"[WARNING] Session recorder save failed: {exc}")


if __name__ == "__main__":
    main()
