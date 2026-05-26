"""Thin Ollama transport shared by the simulated user and the LLM judge.

Deliberately NOT ``utils.llm.LocalLLM``: that class applies a voice-assistant
system prompt and truncates output to ~150 chars / 2 sentences, which would
corrupt a JSON judge verdict and flatten a persona's natural disfluency. Here we
reuse only the transport (``ollama.chat``) and an availability probe modeled on
``LocalLLM.__init__`` + the graceful-degrade pattern in ``tests/conftest.py``.

The assistant *under test* still uses the real ``LocalLLM``.
"""
from __future__ import annotations

import os

DEFAULT_SIM_MODEL = os.environ.get("SPEAKER_SIM_MODEL", "llama3.2:3b")


def ollama_available(model: str = DEFAULT_SIM_MODEL) -> tuple[bool, str]:
    """Return (ok, reason). ok=False means tests should skip (not fail)."""
    try:
        import ollama
    except Exception as exc:  # pragma: no cover - import guard
        return False, f"ollama package missing: {exc}"
    try:
        listed = ollama.list()
    except Exception as exc:
        return False, f"ollama server unavailable: {exc}"
    names = set()
    for entry in listed.get("models", []) if isinstance(listed, dict) else getattr(listed, "models", []):
        name = entry.get("model") or entry.get("name") if isinstance(entry, dict) else getattr(entry, "model", None)
        if name:
            names.add(str(name))
    if model not in names and not any(n.startswith(model) for n in names):
        return False, f"model {model!r} not pulled (have: {sorted(names) or 'none'})"
    return True, "ok"


class OllamaChat:
    """Single-shot chat helper for the user-LLM and the judge."""

    def __init__(self, model: str = DEFAULT_SIM_MODEL, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, messages: list[dict], *, json_mode: bool = False) -> str:
        import ollama

        payload = [{"role": "system", "content": system}, *messages]
        kwargs: dict = {
            "model": self.model,
            "messages": payload,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            kwargs["format"] = "json"
        response = ollama.chat(**kwargs)
        return response["message"]["content"]
