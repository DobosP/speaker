"""Configuration for smart Postgres memory persistence."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping


def _nested_get(config: Mapping[str, Any] | None, *keys: str, default=None):
    cur: Any = config or {}
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


@dataclass
class MemoryWriterConfig:
    """Controls filtering, LLM cleanup, and debounced Postgres writes."""

    enabled: bool = True
    save_interval_sec: float = 240.0
    min_confidence: float = 0.55
    llm_cleanup: bool = True
    llm_gate: bool = True
    cleanup_model: str = "minicpm5-1b:q8"
    max_buffer_items: int = 32
    min_chars: int = 3
    dedupe_similarity: float = 0.92
    persist_user_only: bool = True
    save_control_phrases: bool = False
    boilerplate_phrases: tuple[str, ...] = field(
        default_factory=lambda: (
            "stop",
            "quit",
            "exit",
            "shutdown",
            "shut down",
            "goodbye",
            "cancel",
            "pause",
            "uh",
            "um",
            "erm",
            "hmm",
            ".",
            "",
        )
    )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None = None) -> "MemoryWriterConfig":
        mem = _nested_get(config, "memory") or {}
        if not isinstance(mem, dict):
            mem = {}

        phrases = mem.get("boilerplate_phrases")
        if phrases is None:
            bp = cls().boilerplate_phrases
        else:
            bp = tuple(str(p).strip().lower() for p in phrases)

        cleanup_model = (
            mem.get("cleanup_model")
            or os.getenv("MEMORY_CLEANUP_MODEL")
            or os.getenv("OLLAMA_MEMORY_MODEL")
            or cls().cleanup_model
        )

        return cls(
            enabled=bool(mem.get("enabled", True)),
            save_interval_sec=float(
                mem.get("save_interval_sec", mem.get("flush_interval_sec", cls().save_interval_sec))
            ),
            min_confidence=float(mem.get("min_confidence", cls().min_confidence)),
            llm_cleanup=bool(mem.get("llm_cleanup", True)),
            llm_gate=bool(mem.get("llm_gate", True)),
            cleanup_model=str(cleanup_model),
            max_buffer_items=int(mem.get("max_buffer_items", cls().max_buffer_items)),
            min_chars=int(mem.get("min_chars", cls().min_chars)),
            dedupe_similarity=float(
                mem.get("dedupe_similarity", cls().dedupe_similarity)
            ),
            persist_user_only=bool(mem.get("persist_user_only", True)),
            save_control_phrases=bool(mem.get("save_control_phrases", False)),
            boilerplate_phrases=bp,
        )


def config_from_dict(config: Mapping[str, Any] | None = None) -> MemoryWriterConfig:
    """Build writer config from config.json ``memory`` block or top-level keys."""
    if config is None:
        return MemoryWriterConfig()
    if "memory" in config or any(k.startswith("memory_") for k in config):
        nested = dict(config.get("memory") or {})
        for key, value in config.items():
            if key.startswith("memory_"):
                nested[key.replace("memory_", "", 1)] = value
        return MemoryWriterConfig.from_mapping({"memory": nested})
    return MemoryWriterConfig.from_mapping(config)
