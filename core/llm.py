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


class LlamaCppLLM:
    """On-device LLM via llama.cpp (the mobile/no-Ollama path).

    Ollama is a desktop daemon and does not exist on Android/iOS, so on phone we
    run a quantized GGUF directly through ``llama-cpp-python`` -- same process,
    no server. Use a *small* Gemma 3 here (e.g. gemma3:1b/4b GGUF): on a 12 GB
    phone with no dedicated VRAM, that small model is the intelligence tier.

    The ``llama_cpp`` package is imported lazily so the runtime and tests work
    without the native lib. Pass ``client`` to inject a fake in tests.

    Vision: llama.cpp needs a separate projector (``mmproj``) + a chat handler
    for image input. When ``images`` are passed we format multimodal chat
    content; without a vision-capable build the underlying model ignores them.
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        chat_format: Optional[str] = None,
        options: Optional[dict] = None,
        client=None,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        self._options = dict(options) if options else {}
        self._client = client

    def _ensure(self):
        if self._client is None:
            from llama_cpp import Llama  # lazy

            kwargs = dict(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            if self.n_threads:
                kwargs["n_threads"] = self.n_threads
            if self.chat_format:
                kwargs["chat_format"] = self.chat_format
            self._client = Llama(**kwargs)
        return self._client

    def _messages(
        self, prompt: str, system: Optional[str], images: Optional[Sequence[ImageInput]]
    ) -> list[dict]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        if images:
            content: list[dict] = [{"type": "text", "text": prompt}]
            for img in images:
                url = img if isinstance(img, str) else _to_data_uri(img)
                content.append({"type": "image_url", "image_url": {"url": url}})
            msgs.append({"role": "user", "content": content})
        else:
            msgs.append({"role": "user", "content": prompt})
        return msgs

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> str:
        resp = self._ensure().create_chat_completion(
            messages=self._messages(prompt, system, images), stream=False, **self._options
        )
        return resp["choices"][0]["message"]["content"]

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> Iterator[str]:
        for chunk in self._ensure().create_chat_completion(
            messages=self._messages(prompt, system, images), stream=True, **self._options
        ):
            piece = chunk["choices"][0].get("delta", {}).get("content")
            if piece:
                yield piece


def _to_data_uri(raw: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
