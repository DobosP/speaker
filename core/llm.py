from __future__ import annotations

import logging
import os
import queue
import threading
import time
from contextvars import ContextVar
from typing import Iterator, Mapping, Optional, Protocol, Sequence, runtime_checkable

# Per-turn context published by the capability layer so the LLM stack can
# pick a routing chain (sensitivity / intent_kind / mode) without changing
# the LLMClient protocol. Default is an empty mapping so callers that don't
# set it always land on the safe-default chain.
capability_context: ContextVar[Mapping[str, object]] = ContextVar(
    "speaker_capability_context", default={}
)

_ollama_log = logging.getLogger("speaker.llm.ollama")


def _log_llm_request(
    log: logging.Logger,
    model: str,
    prompt: str,
    system: Optional[str],
    *,
    dt: float,
    out_chars: int,
    tokens: Optional[int] = None,
    ttft: Optional[float] = None,
    streamed: bool,
    cancelled: bool = False,
) -> None:
    """One structured line per LLM call -> feeds the run summary via ``extra``.

    The full prompt goes to the DEBUG file (so a committed log shows exactly what
    was asked); the INFO line carries timings + a short preview."""
    preview = " ".join((prompt or "").split())[:200]
    req = {
        "model": model,
        "prompt_chars": len(prompt or ""),
        "system_chars": len(system or ""),
        "prompt_preview": preview,
        "duration_sec": round(dt, 3),
        "ttft_sec": round(ttft, 3) if ttft is not None else None,
        "out_chars": out_chars,
        "tokens": tokens,
        "streamed": streamed,
        "cancelled": cancelled,
    }
    log.debug("ollama %s full prompt: %s", model, prompt)
    log.info(
        "ollama %s: %.2fs ttft=%s out=%dch tokens=%s%s | %r",
        model, dt,
        f"{ttft:.2f}s" if ttft is not None else "-",
        out_chars, tokens if tokens is not None else "-",
        " CANCELLED" if cancelled else "",
        preview,
        extra={"llm_request": req},
    )

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
        t0 = time.perf_counter()
        resp = self._ensure().chat(**self._chat_kwargs(prompt, system, images, stream=False))
        out = resp["message"]["content"]
        _log_llm_request(
            _ollama_log, self.model, prompt, system,
            dt=time.perf_counter() - t0, out_chars=len(out or ""), streamed=False,
        )
        return out

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> Iterator[str]:
        t0 = time.perf_counter()
        ttft: Optional[float] = None
        tokens = 0
        out_chars = 0
        cancelled = True  # flipped to False only on natural exhaustion
        try:
            for chunk in self._ensure().chat(
                **self._chat_kwargs(prompt, system, images, stream=True)
            ):
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens += 1
                    out_chars += len(piece)
                    yield piece
            cancelled = False
        finally:
            # Runs even if the consumer stops early (barge-in / cancel), so a
            # cut-off generation is still recorded -- useful for "where it stuck".
            _log_llm_request(
                _ollama_log, self.model, prompt, system,
                dt=time.perf_counter() - t0, out_chars=out_chars, tokens=tokens,
                ttft=ttft, streamed=True, cancelled=cancelled,
            )


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
    """Race a local LLM against an optional cloud chain for the lowest latency.

    Strategies (all keep local as the safety net -- any cloud error/timeout
    falls through to the next cloud, finally to local, honoring the fully-
    local requirement):

    - ``hedge`` (default): start local now; if it produces no token within
      ``hedge_delay_ms``, also start the first cloud and stream whichever
      yields the FIRST token. Caps cloud spend/exposure while still racing
      when local is slow. ``hedge_delay_ms=0`` makes it a full race.
    - ``fallback``: start the first cloud with a ``ttft_deadline_ms`` first-
      token deadline; on timeout/error, advance to the next cloud; after the
      chain is exhausted, fall back to local with no deadline.

    The ``cloud`` parameter accepts either a single :class:`LLMClient` (back-
    compat) or a list of them (a failover chain). When any cloud in the chain
    finishes without producing a token (error or empty stream), HedgeLLM
    advances to the next cloud and races it against local with the same
    rules.

    Cancellation: losing workers are signalled to stop between tokens; the
    brain's ``cancel_event`` still cuts the winner's stream in the
    capability layer, so barge-in works unchanged.
    """

    def __init__(
        self,
        *,
        local: "LLMClient",
        cloud: "Optional[LLMClient | Sequence[LLMClient]]",
        strategy: str = "hedge",
        hedge_delay_ms: int = 150,
        ttft_deadline_ms: int = 1200,
    ):
        self.local = local
        if cloud is None:
            self._clouds: list["LLMClient"] = []
        elif isinstance(cloud, (list, tuple)):
            self._clouds = [c for c in cloud if c is not None]
        else:
            self._clouds = [cloud]
        self.strategy = strategy
        self.hedge_delay = max(0.0, hedge_delay_ms / 1000.0)
        self.ttft_deadline = max(0.0, ttft_deadline_ms / 1000.0)

    @property
    def cloud(self) -> "Optional[LLMClient]":
        """Back-compat: the first cloud in the chain (or ``None``)."""
        return self._clouds[0] if self._clouds else None

    @property
    def clouds(self) -> list["LLMClient"]:
        """The full cloud failover chain (empty list when no cloud is set)."""
        return list(self._clouds)

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
        except Exception as exc:  # cloud down / rate-limited -> chain advances
            q.put((tag, "err", str(exc)))

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> Iterator[str]:
        if not self._clouds:
            yield from self.local.stream(prompt, system=system, images=images)
            return

        q: "queue.Queue[tuple[str, str, object]]" = queue.Queue()
        cloud_tags = [f"cloud_{i}" for i in range(len(self._clouds))]
        stops: dict[str, threading.Event] = {
            "local": threading.Event(),
            **{tag: threading.Event() for tag in cloud_tags},
        }
        clients: dict[str, "LLMClient"] = {
            "local": self.local,
            **dict(zip(cloud_tags, self._clouds)),
        }
        started: set[str] = set()
        dead: set[str] = set()

        def launch(tag: str) -> None:
            if tag in started:
                return
            started.add(tag)
            threading.Thread(
                target=self._worker,
                args=(clients[tag], tag, prompt, system, images, q, stops[tag]),
                daemon=True,
            ).start()

        cloud_iter = iter(cloud_tags)

        def next_cloud() -> Optional[str]:
            try:
                return next(cloud_iter)
            except StopIteration:
                return None

        current_cloud = next_cloud()
        if self.strategy == "fallback":
            # Cloud-first: launch the first cloud with a ttft_deadline; on
            # error/timeout, advance the chain; finally fall back to local.
            if current_cloud is not None:
                launch(current_cloud)
            deadline = time.monotonic() + self.ttft_deadline
            local_pending = True
        else:  # hedge
            # Local-first: kick local now; after hedge_delay also launch the
            # first cloud. On cloud error/finish-without-tokens, advance the
            # chain and keep racing against local.
            launch("local")
            deadline = time.monotonic() + self.hedge_delay
            local_pending = False

        winner: Optional[str] = None
        buffered: list[str] = []

        def kick_chain() -> bool:
            """Bring up whatever is next in line. Returns True if launched."""
            nonlocal current_cloud, deadline, local_pending
            if self.strategy == "hedge":
                if current_cloud is not None and current_cloud not in started:
                    launch(current_cloud)
                    deadline = float("inf")
                    return True
            else:  # fallback
                if current_cloud is not None and current_cloud not in started:
                    launch(current_cloud)
                    deadline = time.monotonic() + self.ttft_deadline
                    return True
            if local_pending:
                launch("local")
                local_pending = False
                deadline = float("inf")
                return True
            return False

        while winner is None:
            live = started - dead
            if not live:
                if not kick_chain():
                    break
                continue
            timeout = None if deadline == float("inf") else max(0.0, deadline - time.monotonic())
            try:
                tag, kind, val = q.get(timeout=timeout)
            except queue.Empty:
                # Deadline hit. Hedge: kick the current cloud; fallback:
                # retire the current cloud and advance the chain.
                if self.strategy == "fallback" and current_cloud is not None:
                    stops[current_cloud].set()
                    dead.add(current_cloud)
                    current_cloud = next_cloud()
                kick_chain()
                continue
            if kind == "tok":
                winner = tag
                buffered.append(str(val))
            else:  # this source died (error or empty stream)
                dead.add(tag)
                if tag == current_cloud:
                    current_cloud = next_cloud()
                    kick_chain()
                elif tag == "local":
                    # Local crashed; nothing to do but ride the cloud chain.
                    pass

        if winner is None:
            return  # nothing produced (every source errored or was empty)
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


class SensitivityRouterLLM:
    """Dispatch ``generate``/``stream`` to one of several backing LLMs based
    on the data-sensitivity tag of the current turn.

    Each backing LLM is typically a :class:`HedgeLLM` (local + cloud
    failover chain). The selection happens at call time using a context
    selector that reads :data:`capability_context` -- a ``ContextVar`` set
    by the capability layer before invoking the LLM. This keeps the
    LLMClient protocol unchanged (no extra parameter on ``stream``) while
    letting the routing decision flow from the brain's per-turn context.
    """

    def __init__(
        self,
        chains: Mapping[str, "LLMClient"],
        selector,
        *,
        default_chain: str = "private",
    ):
        if not chains:
            raise ValueError("SensitivityRouterLLM requires at least one chain")
        if default_chain not in chains:
            raise ValueError(
                f"default_chain {default_chain!r} not in chains {sorted(chains)}"
            )
        self.chains: dict[str, "LLMClient"] = dict(chains)
        self.selector = selector
        self.default_chain = default_chain

    def _pick(self) -> "LLMClient":
        ctx = capability_context.get()
        name = self.selector.choose_chain(ctx) if self.selector is not None else self.default_chain
        return self.chains.get(name, self.chains[self.default_chain])

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> str:
        # Pick happens eagerly so any sensitivity-driven routing is logged
        # at the call site (via _pick) rather than at first-yield.
        impl = self._pick()
        return impl.generate(prompt, system=system, images=images)

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> Iterator[str]:
        impl = self._pick()
        return impl.stream(prompt, system=system, images=images)


def _to_data_uri(raw: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
