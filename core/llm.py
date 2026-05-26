from __future__ import annotations

from typing import Iterator, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal local-LLM contract used by the runtime's capabilities."""

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str: ...

    def stream(self, prompt: str, *, system: Optional[str] = None) -> Iterator[str]: ...


class EchoLLM:
    """Deterministic fake LLM for tests and the offline console demo."""

    def __init__(self, reply: Optional[str] = None):
        self._reply = reply

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        if self._reply is not None:
            return self._reply
        return f"You said: {prompt}"

    def stream(self, prompt: str, *, system: Optional[str] = None) -> Iterator[str]:
        yield self.generate(prompt, system=system)


class OllamaLLM:
    """Local LLM via Ollama.

    The ``ollama`` package is imported lazily so the rest of the runtime (and
    the test suite) works in environments without it installed.
    """

    def __init__(self, model: str = "gemma3:latest", host: Optional[str] = None):
        self.model = model
        self._host = host
        self._client = None

    def _ensure(self):
        if self._client is None:
            import ollama  # lazy

            self._client = ollama.Client(host=self._host) if self._host else ollama
        return self._client

    def _messages(self, prompt: str, system: Optional[str]) -> list[dict]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        resp = self._ensure().chat(model=self.model, messages=self._messages(prompt, system))
        return resp["message"]["content"]

    def stream(self, prompt: str, *, system: Optional[str] = None) -> Iterator[str]:
        for chunk in self._ensure().chat(
            model=self.model, messages=self._messages(prompt, system), stream=True
        ):
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
