"""
Structured diagnostics for the live TTS path (synthesis → queue → playback).

Enable via environment variable ``SPEAKER_TTS_DEBUG=1`` or config ``tts_debug: true``.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

_ENABLED: bool = False
_CONFIGURED: bool = False

LOG_TTS = logging.getLogger("speaker.tts")
LOG_AUDIO = logging.getLogger("speaker.audio")

_CONSOLE_KINDS = {
    "enqueued": "🔊 TTS: enqueued",
    "tts_enqueue": "🔊 TTS: enqueued",
    "dequeued": "🔊 TTS: dequeued",
    "tts_dequeue": "🔊 TTS: dequeued",
    "dropped": "🔊 TTS: dropped",
    "tts_drop": "🔊 TTS: dropped",
    "cancelled": "🔊 TTS: cancelled",
    "tts_cancel": "🔊 TTS: cancelled",
    "playing": "🔊 TTS: playing",
    "tts_play_start": "🔊 TTS: playing",
    "played": "🔊 TTS: played",
    "tts_play_end": "🔊 TTS: playback done",
    "synth_start": "🔊 TTS: synthesizing",
    "tts_synth_start": "🔊 TTS: synthesizing",
    "synth_done": "🔊 TTS: synthesized",
    "tts_synth_done": "🔊 TTS: synthesized",
    "skipped": "🔊 TTS: skipped",
    "tts_skip": "🔊 TTS: skipped",
    "failed": "🔊 TTS: FAILED",
    "tts_fail": "🔊 TTS: FAILED",
    "flush": "🔊 TTS: flush",
    "tts_flush": "🔊 TTS: queue flushed",
    "blocked": "🔊 TTS: blocked",
    "tts_llm_chunk": "🔊 TTS: LLM chunk → queue",
}


def _truthy(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def resolve_enabled(
    config_value: Any = None,
    cli_value: Any = None,
    *,
    config_flag: Any = None,
) -> bool:
    """Resolve debug flag: CLI > config > ``SPEAKER_TTS_DEBUG`` env."""
    cfg = config_flag if config_flag is not None else config_value
    if cli_value is not None:
        return _truthy(cli_value)
    if cfg is not None:
        return _truthy(cfg)
    return _truthy(os.environ.get("SPEAKER_TTS_DEBUG", ""))


def configure(
    enabled: Optional[bool] = None,
    *,
    config_value: Any = None,
    cli_value: Any = None,
) -> bool:
    """Configure loggers and console one-liners. Idempotent."""
    global _ENABLED, _CONFIGURED
    if enabled is None:
        enabled = resolve_enabled(config_value, cli_value)
    _ENABLED = bool(enabled)
    _CONFIGURED = True
    level = logging.DEBUG if _ENABLED else logging.WARNING
    for lg in (LOG_TTS, LOG_AUDIO):
        lg.setLevel(level)
        if _ENABLED:
            if not any(isinstance(h, logging.StreamHandler) for h in lg.handlers):
                h = logging.StreamHandler(sys.stderr)
                h.setFormatter(
                    logging.Formatter(
                        "%(asctime)s %(name)s %(levelname)s %(message)s"
                    )
                )
                lg.addHandler(h)
            lg.propagate = False
        else:
            lg.handlers.clear()
    if _ENABLED:
        LOG_TTS.info("tts_debug enabled")
    return _ENABLED


def is_enabled() -> bool:
    if not _CONFIGURED:
        configure()
    return _ENABLED


def console(kind: str, detail: str = "") -> None:
    """Emit a user-facing one-liner when debug is on."""
    if not is_enabled():
        return
    prefix = _CONSOLE_KINDS.get(kind, f"🔊 TTS: {kind}")
    if detail:
        print(f"{prefix} {detail}", flush=True)
    else:
        print(prefix, flush=True)


def _fmt(extra: dict[str, Any]) -> str:
    parts = []
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        elif isinstance(v, str) and len(v) > 80:
            parts.append(f'{k}="{v[:77]}..."')
        else:
            parts.append(f"{k}={v!r}")
    return " ".join(parts)


def log(
    namespace: str,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Structured log: *namespace* is ``tts`` or ``audio``."""
    if namespace == "audio":
        log_audio(event, level=level, **fields)
    else:
        log_tts(event, level=level, **fields)


def log_tts(event: str, level: int = logging.INFO, **fields: Any) -> None:
    if not is_enabled():
        return
    msg = f"{event}"
    tail = _fmt(fields)
    if tail:
        msg = f"{event} {tail}"
    LOG_TTS.log(level, msg)
    ck = fields.get("console_kind")
    if ck:
        console(str(ck), str(fields.get("console_detail", "")))


def log_audio(event: str, level: int = logging.INFO, **fields: Any) -> None:
    if not is_enabled():
        return
    msg = f"{event}"
    tail = _fmt(fields)
    if tail:
        msg = f"{event} {tail}"
    LOG_AUDIO.log(level, msg)
    ck = fields.get("console_kind")
    if ck:
        console(str(ck), str(fields.get("console_detail", "")))


def log_echo_gate_blocked(**fields: Any) -> None:
    """EchoGuard / barge-in gate blocked user interrupt (not TTS synthesis)."""
    log_audio("echo_gate_blocked", level=logging.DEBUG, **fields)


def log_speech_gate_blocked(**fields: Any) -> None:
    log_audio("speech_gate_blocked", level=logging.DEBUG, **fields)
