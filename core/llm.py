from __future__ import annotations

import os
import queue
import threading
import time
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
        keep_alive: Optional[str | int] = None,
        client=None,
    ):
        self.model = model
        self._host = host
        self._options = dict(options) if options else None
        # How long Ollama keeps the model resident after a request. A long value
        # (e.g. "30m") or -1 (forever) avoids a cold reload on the next turn --
        # the single biggest win for a snappy first token on a warm box.
        self._keep_alive = keep_alive
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
        if self._keep_alive is not None:
            kwargs["keep_alive"] = self._keep_alive
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


def _openai_messages(
    prompt: str, system: Optional[str], images: Optional[Sequence[ImageInput]]
) -> list[dict]:
    """OpenAI-style chat messages (also used by llama-server, Groq, etc.)."""
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


class OpenAICompatLLM:
    """Streaming LLM over any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    One client covers Groq, Together, Fireworks, SambaNova, Cerebras, OpenAI and
    a local llama.cpp ``llama-server`` -- they differ only in ``base_url``,
    ``model`` and api key. This is the optional "intelligence from a streaming
    source" tier: it is built only when ``llm.cloud.enabled`` is set, so the
    fully-local default is preserved. Rank providers by time-to-first-token.

    The ``openai`` package is imported lazily so the runtime and test suite work
    without it; pass ``client`` to inject a fake. ``api_key_env`` names the env
    var holding the key (so secrets never live in config.json).
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        timeout: float = 30.0,
        options: Optional[dict] = None,
        client=None,
    ):
        self.model = model
        self._base_url = base_url
        self._api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self._timeout = timeout
        self._options = dict(options) if options else {}
        self._client = client

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # lazy

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key or "not-needed",
                timeout=self._timeout,
            )
        return self._client

    def _create_kwargs(self, prompt, system, images, *, stream: bool) -> dict:
        kwargs = {
            "model": self.model,
            "messages": _openai_messages(prompt, system, images),
            "stream": stream,
        }
        kwargs.update(self._options)
        return kwargs

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> str:
        resp = self._ensure().chat.completions.create(
            **self._create_kwargs(prompt, system, images, stream=False)
        )
        return resp.choices[0].message.content or ""

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> Iterator[str]:
        for chunk in self._ensure().chat.completions.create(
            **self._create_kwargs(prompt, system, images, stream=True)
        ):
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            piece = choices[0].delta.content
            if piece:
                yield piece


class HedgeLLM:
    """Race a local LLM against an optional cloud one for the lowest latency.

    Strategies (all keep local as the safety net -- any cloud error/timeout
    falls back to local, honoring the fully-local requirement):

    - ``hedge`` (default): start local now; if it produces no token within
      ``hedge_delay_ms``, also start cloud and stream whichever yields the FIRST
      token. Caps cloud spend/exposure while still racing when local is slow.
      ``hedge_delay_ms=0`` makes it a full race (both start immediately).
    - ``fallback``: start cloud first with a ``ttft_deadline_ms`` first-token
      deadline; on timeout/error, fall back to local.

    Cancellation: the loser worker is signalled to stop and stops consuming
    between tokens; the brain's ``cancel_event`` still cuts the winner's stream
    in the capability layer, so barge-in works unchanged.
    """

    def __init__(
        self,
        *,
        local: "LLMClient",
        cloud: Optional["LLMClient"],
        strategy: str = "hedge",
        hedge_delay_ms: int = 150,
        ttft_deadline_ms: int = 1200,
    ):
        self.local = local
        self.cloud = cloud
        self.strategy = strategy
        self.hedge_delay = max(0.0, hedge_delay_ms / 1000.0)
        self.ttft_deadline = max(0.0, ttft_deadline_ms / 1000.0)

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> str:
        return "".join(self.stream(prompt, system=system, images=images)).strip()

    @staticmethod
    def _worker(client, tag, prompt, system, images, q, stop) -> None:
        try:
            for token in client.stream(prompt, system=system, images=images):
                if stop.is_set():
                    break
                q.put((tag, "tok", token))
            q.put((tag, "done", None))
        except Exception as exc:  # cloud down / rate-limited -> let local win
            q.put((tag, "err", str(exc)))

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> Iterator[str]:
        if self.cloud is None:
            yield from self.local.stream(prompt, system=system, images=images)
            return

        q: "queue.Queue[tuple[str, str, object]]" = queue.Queue()
        stops = {"local": threading.Event(), "cloud": threading.Event()}
        clients = {"local": self.local, "cloud": self.cloud}
        started: set[str] = set()

        def launch(tag: str) -> None:
            if tag in started or clients[tag] is None:
                return
            started.add(tag)
            threading.Thread(
                target=self._worker,
                args=(clients[tag], tag, prompt, system, images, q, stops[tag]),
                daemon=True,
            ).start()

        if self.strategy == "fallback":
            primary, secondary, delay = "cloud", "local", self.ttft_deadline
        else:  # hedge
            primary, secondary, delay = "local", "cloud", self.hedge_delay
        launch(primary)
        live = 1 if primary in started else 0
        deadline = time.monotonic() + delay

        winner: Optional[str] = None
        buffered: list[str] = []
        while winner is None and live > 0:
            secondary_started = secondary in started
            if secondary_started or deadline == float("inf"):
                timeout = None
            else:
                timeout = max(0.0, deadline - time.monotonic())
            try:
                tag, kind, val = q.get(timeout=timeout)
            except queue.Empty:
                # Primary was too slow: bring up the secondary, then wait.
                launch(secondary)
                live += 1 if secondary in started else 0
                deadline = float("inf")
                continue
            if kind == "tok":
                winner = tag
                buffered.append(str(val))
            else:  # this source produced nothing (finished or errored)
                live -= 1
                if not secondary_started:
                    launch(secondary)
                    live += 1 if secondary in started else 0
                    deadline = float("inf")

        if winner is None:
            return  # nothing produced (e.g. no cloud and local empty) -> empty
        for tag, ev in stops.items():
            if tag != winner:
                ev.set()
        for token in buffered:
            yield token
        while True:
            tag, kind, val = q.get()
            if tag != winner:
                continue  # drain the loser's late tokens
            if kind == "tok":
                yield str(val)
            else:
                break


def _to_data_uri(raw: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
