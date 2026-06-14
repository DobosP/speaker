from __future__ import annotations

import logging
import os
import queue
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Optional, Protocol, Sequence, runtime_checkable

# Per-turn context published by the capability layer so the LLM stack can
# pick a routing chain (sensitivity / intent_kind / mode) without changing
# the LLMClient protocol. Default is an empty mapping so callers that don't
# set it always land on the safe-default chain.
capability_context: ContextVar[Mapping[str, object]] = ContextVar(
    "speaker_capability_context", default={}
)

_ollama_log = logging.getLogger("speaker.llm.ollama")
_hedge_log = logging.getLogger("speaker.llm.hedge")


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

    ``think`` controls Ollama's reasoning-model "thinking" phase. A reasoning
    model (e.g. gemma4) streams a silent chain-of-thought into a SEPARATE
    ``thinking`` field BEFORE any ``content`` token -- which our stream only
    yields content from, so for voice that thinking is pure dead air: measured
    ~9 s of silence before the first spoken word of a *story* on gemma4:12b.
    ``think=False`` skips it (first content token ~1.9 s instead), ``True``
    forces it on, ``None`` (default here) leaves the model's own default. The
    voice factory (:func:`core.llm_factory.build_llms`) defaults it to ``False``
    -- thinking's multi-second latency is unacceptable for a real-time voice
    turn. Passed only when not ``None`` so non-reasoning models / older Ollama
    builds are unaffected.
    """

    def __init__(
        self,
        model: str = "gemma3:12b",
        host: Optional[str] = None,
        *,
        options: Optional[dict] = None,
        keep_alive: Optional[str | int] = None,
        timeout: Optional[float] = 60.0,
        think: Optional[bool] = None,
        client=None,
    ):
        self.model = model
        self._host = host
        self._options = dict(options) if options else None
        # Reasoning-model "thinking" toggle (see the class docstring). Sent as a
        # top-level chat arg only when not None, so the default path is unchanged.
        self._think = think
        # How long Ollama keeps the model resident after a request. A long value
        # (e.g. "30m") or -1 (forever) avoids a cold reload on the next turn --
        # the single biggest win for a snappy first token on a warm box.
        self._keep_alive = keep_alive
        # Socket/read timeout (seconds) for the underlying httpx client. Without
        # it a hung Ollama daemon would wedge the turn forever instead of
        # raising (which the Hedge chain treats as a dead source and advances
        # past). ``None`` disables the timeout (the old behaviour).
        self._timeout = timeout
        self._client = client

    def _ensure(self):
        if self._client is None:
            import ollama  # lazy

            # Always build an explicit Client so the read timeout is applied;
            # the bare ``ollama`` module client has no timeout (hangs forever
            # on a stalled connection).
            kwargs: dict = {}
            if self._host:
                kwargs["host"] = self._host
            if self._timeout is not None:
                kwargs["timeout"] = self._timeout
            self._client = ollama.Client(**kwargs)
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
        if self._think is not None:
            kwargs["think"] = self._think
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
        # A single llama.cpp context can't run two inferences at once, and the
        # lazy build must not double-construct. This serializes both: the startup
        # warm pass and a concurrent first live turn (or two tasks) share one
        # context safely instead of racing into the native lib.
        self._lock = threading.Lock()

    def _ensure(self):
        if self._client is None:
            with self._lock:
                if self._client is None:  # double-checked: build exactly once
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
        client = self._ensure()
        with self._lock:  # one inference at a time on the shared context
            resp = client.create_chat_completion(
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
        client = self._ensure()
        # Held across the whole generation (released when the generator is
        # exhausted or closed early on barge-in) so the single context isn't
        # driven by two threads at once.
        with self._lock:
            for chunk in client.create_chat_completion(
                messages=self._messages(prompt, system, images), stream=True, **self._options
            ):
                piece = chunk["choices"][0].get("delta", {}).get("content")
                if piece:
                    yield piece


def _redact_messages_for_egress(messages: list[dict]) -> list[dict]:
    """Scrub high-confidence PII (cards/SSN/keys/email/phone/secrets) from outbound
    cloud messages -- a §9.7 last-line net INDEPENDENT of the regex sensitivity
    classifier. If a credit card / SSN / API key slips past PRIVATE classification
    (garbled ASR, PII phrased outside the pattern set) and a turn is sent to a
    third-party cloud, this still removes it. Conservative (``redact_pii`` only
    touches Luhn-checked cards, SSNs, known key formats, etc.), so ordinary queries
    are untouched. Applied ONLY to cloud egress -- never to a local model. Text-only
    redaction; image parts (data URIs) pass through (vision egress is gated upstream
    by sensitivity + the local-only captioning rule). Applied to every cloud-chain
    member (OpenAICompatLLM with the flag set); the HedgeLLM local safety-net member
    (Ollama / llama.cpp) is never redacted."""
    from always_on_agent.untrusted import redact_pii  # stdlib-only; core->aoa is allowed

    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({**m, "content": redact_pii(content)})
        elif isinstance(content, list):
            parts = [
                ({**p, "text": redact_pii(p["text"])}
                 if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
                 else p)
                for p in content
            ]
            out.append({**m, "content": parts})
        else:
            out.append(m)
    return out


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


@dataclass(frozen=True)
class ProviderProfile:
    """Per-provider quirks layered on top of the generic OpenAI-compat shape.

    Each major cloud sharing the ``/v1/chat/completions`` endpoint still has
    small departures that, ignored, cause silent failures: Moonshot rejects
    custom temperature; Cerebras requires non-standard params via
    ``extra_body=``; DeepSeek's reasoning models stream the chain-of-thought
    in a separate ``delta.reasoning_content`` field that the generic loop
    drops on the floor; Groq's gpt-oss-120b puts reasoning in ``delta.reasoning``
    and rejects ``n != 1``. This dataclass captures those quirks declaratively
    so :class:`OpenAICompatLLM` consumes them uniformly.

    Names are looked up via :data:`PROVIDER_PROFILES` from a string tag
    stored on the cloud preset in ``config.json`` (e.g. ``"profile":
    "deepseek_reasoning"``).
    """

    name: str
    # Param keys stripped from the generic kwargs dict before calling .create().
    forbidden_params: frozenset[str] = field(default_factory=frozenset)
    # Param keys routed through ``extra_body={...}`` instead of as top-level
    # kwargs (the OpenAI SDK rejects unknown top-level keys).
    extra_body_keys: frozenset[str] = field(default_factory=frozenset)
    # Name of the delta field that carries reasoning tokens (CoT), if any.
    # ``"reasoning_content"`` for DeepSeek V4-Pro; ``"reasoning"`` for Groq
    # gpt-oss-120b; ``None`` for plain chat models.
    reasoning_field: Optional[str] = None
    # When True, reasoning tokens are observed (for metrics) but NOT yielded
    # to the consumer -- the voice assistant shouldn't speak the CoT.
    suppress_reasoning_in_stream: bool = True
    # Hard cap on ``max_tokens`` enforced before sending (Cerebras free tier
    # rejects > 8192).
    max_tokens_cap: Optional[int] = None


# Pre-defined profiles for the cloud providers we ship presets for.
# ``"openai_compat"`` is the safe default for any unrecognized endpoint.
PROVIDER_PROFILES: dict[str, "ProviderProfile"] = {
    "openai_compat": ProviderProfile(name="openai_compat"),
    # Cerebras: free tier caps max_tokens at 8192. Non-OpenAI params (e.g. GLM's
    # ``clear_thinking``, ``reasoning_effort``) must go in extra_body=.
    "cerebras": ProviderProfile(
        name="cerebras",
        extra_body_keys=frozenset({"clear_thinking", "reasoning_effort"}),
        max_tokens_cap=8192,
    ),
    # Groq: ``n`` is fixed at 1 (any other value 400s). gpt-oss-120b streams
    # reasoning in delta.reasoning (separate from delta.content).
    "groq": ProviderProfile(
        name="groq",
        forbidden_params=frozenset({"n"}),
        reasoning_field="reasoning",
    ),
    # DeepSeek non-reasoning models (V4-Flash) -- plain chat.
    "deepseek": ProviderProfile(name="deepseek"),
    # DeepSeek V4-Pro (reasoning): streams delta.reasoning_content ahead of
    # delta.content. The API also rejects echoing reasoning_content back on
    # the next turn -- callers must strip it from prior assistant messages.
    "deepseek_reasoning": ProviderProfile(
        name="deepseek_reasoning",
        reasoning_field="reasoning_content",
    ),
    # Moonshot Kimi: temperature, top_p, n are server-fixed (any value 400s).
    "moonshot": ProviderProfile(
        name="moonshot",
        forbidden_params=frozenset({"temperature", "top_p", "n"}),
    ),
}


class OpenAICompatLLM:
    """Streaming LLM over any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    One client covers Groq, Together, Fireworks, SambaNova, Cerebras, OpenAI and
    a local llama.cpp ``llama-server`` -- they differ only in ``base_url``,
    ``model`` and api key. This is the optional "intelligence from a streaming
    source" tier: it is built only when ``llm.cloud.enabled`` is set, so the
    fully-local default is preserved. Rank providers by time-to-first-token.

    Provider quirks are layered via ``profile`` (a :class:`ProviderProfile`
    instance or a string key into :data:`PROVIDER_PROFILES`): forbidden
    params are stripped, extra-body keys are routed correctly, and
    reasoning-field streaming (DeepSeek ``reasoning_content``, Groq
    ``reasoning``) is consumed and metric-tracked without being yielded to
    the speaker (the assistant shouldn't speak the CoT).

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
        max_tokens: Optional[int] = None,
        options: Optional[dict] = None,
        profile: "ProviderProfile | str | None" = None,
        client=None,
        redact_pii_outbound: bool = False,
    ):
        self.model = model
        self._base_url = base_url
        # §9.7 last-line net: when True (set by the cloud-client factory), scrub
        # high-confidence PII from the outbound prompt before it leaves the device.
        # Default False so a LOCAL OpenAICompat endpoint (llama-server) and existing
        # callers/tests are byte-identical; only cloud providers enable it.
        self._redact_pii_outbound = bool(redact_pii_outbound)
        self._api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        # Socket/read timeout (seconds) handed to the OpenAI client. A small
        # value (BR1) reaps a losing cloud worker still blocked in the first-
        # token read fast -- the HTTP hard-close in stream() is deterministic
        # only after tokens flow, so a pre-first-token loser otherwise holds the
        # socket + billing until this timeout. 30.0 keeps the prior behaviour.
        self._timeout = timeout
        # Optional per-turn output ceiling (BR4). Injected into the merged
        # request kwargs BEFORE the profile max_tokens cap so the profile cap
        # stays authoritative via min() composition (e.g. Cerebras free tier).
        self._max_tokens = max_tokens
        self._options = dict(options) if options else {}
        self._client = client
        if profile is None:
            self.profile = PROVIDER_PROFILES["openai_compat"]
        elif isinstance(profile, str):
            self.profile = PROVIDER_PROFILES.get(profile, PROVIDER_PROFILES["openai_compat"])
        else:
            self.profile = profile
        # Last-call observability: bytes seen on the reasoning field, exposed
        # so HedgeLLM / capabilities can log a "thought N chars before
        # answering" metric without inspecting the raw stream.
        self.last_reasoning_chars = 0

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
        messages = _openai_messages(prompt, system, images)
        if self._redact_pii_outbound:
            messages = _redact_messages_for_egress(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        # Merge caller-supplied options; the profile may then strip / reroute.
        merged: dict = dict(self._options)
        extra_body: dict = {}
        for key in list(merged):
            if key in self.profile.forbidden_params:
                merged.pop(key, None)
            elif key in self.profile.extra_body_keys:
                extra_body[key] = merged.pop(key)
        # Per-turn output ceiling (BR4): inject the configured max_tokens into
        # merged BEFORE the profile-cap block below so the profile cap stays
        # authoritative -- the cap's min()-style composition then keeps whichever
        # is smaller (e.g. a 100-token turn ceiling under an 8192 Cerebras cap
        # stays 100; a 20000 ceiling is clamped to 8192). A caller-supplied
        # options["max_tokens"] still wins (it's already in merged).
        if self._max_tokens is not None and merged.get("max_tokens") is None:
            merged["max_tokens"] = self._max_tokens
        # max_tokens cap (Cerebras free tier).
        cap = self.profile.max_tokens_cap
        if cap is not None:
            requested = merged.get("max_tokens")
            if requested is None or int(requested) > cap:
                merged["max_tokens"] = cap
        kwargs.update(merged)
        if extra_body:
            # Merge with any caller-supplied extra_body rather than clobbering.
            existing = kwargs.get("extra_body") or {}
            existing.update(extra_body)
            kwargs["extra_body"] = existing
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
        self.last_reasoning_chars = 0
        reasoning_field = self.profile.reasoning_field
        suppress = self.profile.suppress_reasoning_in_stream
        # Bind the SDK Stream INSIDE the generator body so a HTTP hard-close on
        # cancel/barge-in (GeneratorExit) runs the finally below, telling the
        # provider to stop streaming + billing instead of leaking the socket
        # until GC (ported from tools/cloudchat.py:181-185). Binding here (not
        # before the loop's caller resumes) keeps a close() BEFORE the first
        # token a no-op: sdk_stream is unbound, so the getattr guard returns
        # None and finally does nothing rather than raising AttributeError (BR6).
        try:
            sdk_stream = self._ensure().chat.completions.create(
                **self._create_kwargs(prompt, system, images, stream=True)
            )
            for chunk in sdk_stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta
                # Reasoning-channel tokens (DeepSeek reasoning_content / Groq
                # gpt-oss reasoning). Count for metrics; yield only if not
                # suppressed (default: suppressed -- assistant shouldn't speak CoT).
                if reasoning_field:
                    reasoning_piece = getattr(delta, reasoning_field, None)
                    if reasoning_piece:
                        self.last_reasoning_chars += len(reasoning_piece)
                        if not suppress:
                            yield reasoning_piece
                piece = getattr(delta, "content", None)
                if piece:
                    yield piece
        finally:
            # Hard-close the HTTP stream on natural exhaustion AND on early
            # consumer close (cancel). Guarded so a pre-first-token close (where
            # .create() never returned, sdk_stream unbound) is a NO-OP, and so a
            # fake/iterator stream without .close() doesn't crash the worker.
            closer = getattr(locals().get("sdk_stream"), "close", None)
            if closer is not None:
                try:
                    closer()
                except Exception:
                    pass


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

    # How long the final-drain ``q.get()`` waits between tokens from a winner
    # that has already produced its first token before giving up and ending the
    # stream cleanly. Without a bound, a winner whose connection stalls
    # mid-stream (TCP black-hole) would wedge the whole turn forever. Generous
    # enough not to truncate a healthy-but-slow generation; the per-token gap on
    # any working stream is far below this.
    DRAIN_IDLE_TIMEOUT = 30.0
    # Wall-clock budget for the PRE-first-token winner-selection wait, derived
    # from ``ttft_deadline`` (see ``_winner_select_budget``). Distinct from
    # DRAIN_IDLE_TIMEOUT, which is the POST-first-token between-token bound: this
    # one caps the time spent waiting for the *first* token from *any* source so
    # a hung source that never yields, errors, or completes cannot wedge the turn
    # before a single token is produced. The hedge code only ever waits on
    # ``deadline`` (hedge_delay / ttft_deadline) when a worker is still live, and
    # in the hedge strategy that becomes ``inf`` once everything is launched -- so
    # without this budget a stalled local source (notably the in-process
    # ``LlamaCppLLM`` tier, which runs a GGUF in-process and therefore has NO
    # socket/read timeout to reap a hung native call) would block the
    # ``q.get(timeout=None)`` forever. This wall-clock budget IS that bound for
    # the llama.cpp tier. The multiplier is generous: each launched source may
    # legitimately consume up to ~``ttft_deadline`` of pre-first-token wait
    # (a fallback chain retires each slow cloud only at its deadline), so the
    # budget scales with the chain length; it must never be unbounded.
    WINNER_SELECT_TTFT_BUDGET_MULT = 4
    # Floor for the winner-selection budget so a tiny configured ttft_deadline
    # (or a zero hedge_delay) still leaves a sane wall-clock window before a hung
    # source is reaped. Capped at a real-time voice-turn budget: this is the max
    # wait for a FIRST token before we give up and end the turn, so it must stay
    # within a conversational latency budget (a 30s floor far exceeds any voice
    # turn and would wedge the turn on a hung source for half a minute).
    WINNER_SELECT_BUDGET_FLOOR = 10.0
    # Total budget the generator's cleanup spends joining worker threads once
    # they have been told to stop. A live worker stops at its next token
    # boundary (sub-ms once signalled) or when its socket read timeout fires;
    # this short bound reaps the common fast-exit case without wedging the turn
    # on a loser still blocked in an uncancellable pre-first-token read (those
    # are daemon threads that their socket timeout reaps shortly after). It must
    # never be unbounded.
    WORKER_JOIN_TIMEOUT = 0.5

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
        # Egress receipt (provenance): which source served the LAST stream --
        # "local" or "cloud_<i>" (the chain index), or None before any call. Lets a
        # caller record/surface whether an answer actually came from the device or a
        # cloud provider (HedgeLLM races, so the winner isn't knowable a priori).
        self.last_source: Optional[str] = None

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
        # Hold the underlying stream so a stop can close it at the next token
        # boundary -- that propagates GeneratorExit into the client's stream(),
        # running its ``finally`` (socket close, metric log) promptly instead
        # of leaving a half-read HTTP body dangling until GC.
        stream = client.stream(prompt, system=system, images=images)
        try:
            for token in stream:
                if stop.is_set():
                    break
                q.put((tag, "tok", token))
            q.put((tag, "done", None))
        except Exception as exc:  # cloud down / rate-limited -> chain advances
            q.put((tag, "err", str(exc)))
        finally:
            # Best-effort close: list/iterator fakes lack .close(); a generator
            # raising on close shouldn't take the worker down.
            closer = getattr(stream, "close", None)
            if closer is not None:
                try:
                    closer()
                except Exception:
                    pass

    def _winner_select_budget(self) -> float:
        """Bounded wall-clock window for the pre-first-token wait.

        Derived from ``ttft_deadline`` and scaled by the number of launched
        sources (local + every cloud) so a healthy fallback chain that retires
        each slow cloud at its own ``ttft_deadline`` never trips it, while a
        hung source still gets reaped. Always finite (never ``inf``)."""
        sources = len(self._clouds) + 1  # + local
        return max(
            self.ttft_deadline * self.WINNER_SELECT_TTFT_BUDGET_MULT * sources,
            self.WINNER_SELECT_BUDGET_FLOOR,
        )

    def stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        hedge_delay_ms: Optional[int] = None,
    ) -> Iterator[str]:
        # Per-turn hedge-delay override (PINNED CONTRACT). ``None`` keeps the
        # constructor's ``self.hedge_delay`` so default behaviour is byte-
        # identical; an int overrides the local-vs-cloud start gap for THIS turn
        # only (the hook for dynamic hedge timing from the routing layer). Only
        # the ``hedge`` strategy uses a hedge delay; ``fallback`` is unaffected.
        hedge_delay = (
            self.hedge_delay
            if hedge_delay_ms is None
            else max(0.0, hedge_delay_ms / 1000.0)
        )
        if not self._clouds:
            self.last_source = "local"  # egress receipt: served on-device, no cloud
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
        threads: dict[str, threading.Thread] = {}

        def launch(tag: str) -> None:
            if tag in started:
                return
            started.add(tag)
            t = threading.Thread(
                target=self._worker,
                args=(clients[tag], tag, prompt, system, images, q, stops[tag]),
                daemon=True,
            )
            threads[tag] = t
            t.start()

        def shutdown() -> None:
            """Signal every worker to stop and bound-join them so the
            generator never leaks threads -- runs on normal completion AND
            when the consumer closes the generator early (barge-in / cancel /
            ``del`` the iterator -> GeneratorExit). Each worker stops at its
            next token boundary or when its socket read timeout fires, then
            closes its underlying stream in _worker's finally. The join shares
            one bounded budget across all workers so a multi-cloud chain can't
            multiply the wait."""
            for ev in stops.values():
                ev.set()
            join_deadline = time.monotonic() + self.WORKER_JOIN_TIMEOUT
            for t in threads.values():
                if t.is_alive():
                    remaining = join_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    t.join(timeout=remaining)
            # Observability for the known leak: WORKER_JOIN_TIMEOUT is below real
            # reap latency (a cloud worker can take up to ~5s; an in-process
            # LlamaCppLLM pre-first-token native call is uncancellable), so a
            # loser can outlive the join and leak a live thread/socket. A barge-in
            # storm accumulates them. Surface the survivor count in the run bundle
            # so the leak is visible instead of silent. Best-effort: never raise
            # from cleanup (this runs in a generator finally, including on
            # GeneratorExit), so guard the whole check.
            try:
                leaked = sum(1 for t in threads.values() if t.is_alive())
                if leaked:
                    _hedge_log.warning(
                        "HedgeLLM shutdown: %d worker thread(s) still alive after "
                        "%.3fs join budget; may leak thread/socket until reaped",
                        leaked,
                        self.WORKER_JOIN_TIMEOUT,
                    )
            except Exception:
                pass

        try:
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
                # chain and keep racing against local. ``hedge_delay`` is the
                # per-call override (or the constructor default when None).
                launch("local")
                deadline = time.monotonic() + hedge_delay
                local_pending = False

            winner: Optional[str] = None
            buffered: list[str] = []
            # Bounded wall-clock budget for the whole pre-first-token wait. The
            # per-iteration ``deadline`` becomes ``inf`` in hedge once every
            # source is launched (and is None-timeout in the q.get below), so a
            # source that hangs without ever yielding/erroring/completing -- e.g.
            # an in-process LlamaCppLLM native call, which has no socket timeout
            # to reap it -- would otherwise block here forever. This caps that
            # wait; on expiry we stop the workers and end the stream cleanly with
            # whatever (nothing) was produced, identical to an all-dead chain.
            select_deadline = time.monotonic() + self._winner_select_budget()

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
                # Wall-clock budget guard: a hung source (no token, no error, no
                # completion) must not wedge the turn pre-first-token. Reap and
                # end cleanly once the budget is spent.
                select_remaining = select_deadline - time.monotonic()
                if select_remaining <= 0:
                    break
                # Clamp the per-iteration wait to the remaining budget so an
                # ``inf`` deadline (hedge: everything launched) still wakes to
                # re-check it instead of blocking forever.
                step = (
                    select_remaining
                    if deadline == float("inf")
                    else min(max(0.0, deadline - time.monotonic()), select_remaining)
                )
                try:
                    tag, kind, val = q.get(timeout=step)
                except queue.Empty:
                    # Either the per-source deadline or the wall-clock budget
                    # elapsed. If the budget is spent, the top-of-loop guard ends
                    # the turn next iteration. Otherwise this is a per-source
                    # deadline -- hedge: kick the current cloud; fallback: retire
                    # the current cloud and advance the chain.
                    if time.monotonic() >= select_deadline:
                        continue
                    if self.strategy == "fallback" and current_cloud is not None:
                        stops[current_cloud].set()
                        dead.add(current_cloud)
                        current_cloud = next_cloud()
                    kick_chain()
                    continue
                if kind == "tok":
                    winner = tag
                    self.last_source = tag  # egress receipt: this source served the turn
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
            # Stop the losers now so they stop billing/streaming promptly;
            # the winner keeps streaming and is joined in shutdown().
            for tag, ev in stops.items():
                if tag != winner:
                    ev.set()
            for token in buffered:
                yield token
            while True:
                try:
                    # Bounded idle wait: a winner whose connection stalls
                    # mid-stream must not wedge the turn forever. On timeout
                    # we stop the winner and end the stream cleanly with what
                    # we already delivered.
                    tag, kind, val = q.get(timeout=self.DRAIN_IDLE_TIMEOUT)
                except queue.Empty:
                    stops[winner].set()
                    break
                if tag != winner:
                    continue  # drain the loser's late tokens
                if kind == "tok":
                    yield str(val)
                else:
                    break
        finally:
            shutdown()


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
        hedge_delay_ms: Optional[int] = None,
    ) -> Iterator[str]:
        impl = self._pick()
        # Transparent dispatch to the chosen per-chain backend. ``hedge_delay_ms``
        # is the per-turn hedge-timing override (PINNED CONTRACT); forward it
        # only when set AND only to a backend that accepts it (HedgeLLM), so the
        # plain ``LLMClient`` protocol stream() of any other backing client is
        # called unchanged -- keeping default (None) behaviour byte-identical.
        if hedge_delay_ms is not None and isinstance(impl, HedgeLLM):
            return impl.stream(
                prompt, system=system, images=images, hedge_delay_ms=hedge_delay_ms
            )
        return impl.stream(prompt, system=system, images=images)


def _to_data_uri(raw: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
