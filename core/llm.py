from __future__ import annotations

from typing import Iterator, Optional, Protocol, Sequence, runtime_checkable

# An "image" is either a path to an image file or raw image bytes. The Ollama
# client accepts both for multimodal models (e.g. Gemma 3 4b/12b/27b). Text-only
# models such as gemma3:1b ignore images.
ImageInput = str | bytes


@runtime_checkable
class LLMClient(Protocol):
    """Minimal local-LLM contract used by the runtime's capabilities.

    ``images`` is optional so the same client serves text-only and multimodal
    callers; implementations backed by a text-only model simply ignore it.
    """

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> str: ...

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> Iterator[str]: ...


class EchoLLM:
    """Deterministic fake LLM for tests and the offline console demo."""

    def __init__(self, reply: Optional[str] = None):
        self._reply = reply

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> str:
        if self._reply is not None:
            return self._reply
        suffix = f" [+{len(images)} image(s)]" if images else ""
        return f"You said: {prompt}{suffix}"

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> Iterator[str]:
        yield self.generate(prompt, system=system, images=images)


class OllamaLLM:
    """Local LLM via Ollama (GPU-accelerated for Gemma 3 on a CUDA host).

    The ``ollama`` package is imported lazily so the rest of the runtime (and
    the test suite) works in environments without it installed. Pass ``options``
    to tune the model server-side, e.g. ``{"num_ctx": 4096}`` for context size
    or ``{"num_gpu": 999}`` to force full GPU offload.

    Multimodal: pass ``images`` (file paths or bytes) to ``generate``/``stream``
    with a vision-capable model (gemma3:4b / 12b / 27b).
    """

    def __init__(
        self,
        model: str = "gemma3:12b",
        host: Optional[str] = None,
        *,
        options: Optional[dict] = None,
        client=None,
    ):
        self.model = model
        self._host = host
        self._options = dict(options) if options else None
        self._client = client

    def _ensure(self):
        if self._client is None:
            import ollama  # lazy

            self._client = ollama.Client(host=self._host) if self._host else ollama
        return self._client

    def _messages(
        self, prompt: str, system: Optional[str], images: Optional[Sequence[ImageInput]]
    ) -> list[dict]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        user: dict = {"role": "user", "content": prompt}
        if images:
            user["images"] = list(images)
        msgs.append(user)
        return msgs

    def _chat_kwargs(self, prompt, system, images, *, stream: bool) -> dict:
        kwargs = {
            "model": self.model,
            "messages": self._messages(prompt, system, images),
            "stream": stream,
        }
        if self._options:
            kwargs["options"] = self._options
        return kwargs

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> str:
        resp = self._ensure().chat(**self._chat_kwargs(prompt, system, images, stream=False))
        return resp["message"]["content"]

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> Iterator[str]:
        for chunk in self._ensure().chat(**self._chat_kwargs(prompt, system, images, stream=True)):
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
