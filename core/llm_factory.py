"""LLM construction factory for the runtime.

This module owns the model wiring the CLI (``core/app.py``) and the remote
worker both need, so neither has to reach into ``core.app`` internals. The
public entry point is :func:`build_llms`; ``_wrap_cloud`` / ``_build_cloud_client``
/ ``_preset_host`` stay public so the cloud-provider tests can exercise them
directly. The construction logic is unchanged from when it lived in
``core/app.py`` (incl. the P3 OpenRouter/PRC/timeout/max_tokens additions).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .llm import (
    EchoLLM,
    HedgeLLM,
    LlamaCppLLM,
    LLMClient,
    OllamaLLM,
    OpenAICompatLLM,
    SensitivityRouterLLM,
)
from .routing import build_chain_selector, order_presets_by_cost
from .llm_threads import auto_llm_threads as _auto_llm_threads

# Keep the historical logger name ("speaker.app") so the cloud-drop INFO logs
# land under the same logger the tests + run-bundle expect after this code
# moved out of core/app.py.
log = logging.getLogger("speaker.app")

__all__ = [
    "build_llms",
    "_build_llms",
    "_auto_llm_threads",
    "_wrap_cloud",
    "_build_cloud_client",
    "_preset_host",
]


def build_llms(args_or_config, config: dict) -> tuple[LLMClient, LLMClient | None]:
    """Return ``(main_llm, fast_llm)``.

    ``main_llm`` is the larger/multimodal model (research/vision); ``fast_llm``
    is a small model for snappy spoken replies. With ``--llm echo`` both are the
    same fake and ``fast_llm`` is ``None``. The backend is chosen by the ``llm``
    config block: ``ollama`` (desktop, GPU) or ``llamacpp`` (on-device GGUF).

    ``args_or_config`` is the parsed CLI namespace (``.llm`` / ``.model`` /
    ``.fast_model`` select the backend and override the config models);
    ``config`` is the (device-profile-applied) config dict.
    """
    args = args_or_config
    if args.llm == "echo":
        return EchoLLM(), None
    llm_cfg = config.get("llm", {})
    options = llm_cfg.get("options")
    # Narrow construction seam for local diagnostic/evaluation callers that
    # must override ambient SDK authentication headers. Normal CLI args do not
    # define it, so production construction remains byte-identical.
    client_headers = getattr(args, "ollama_client_headers", None)
    ollama_timeout = getattr(args, "ollama_timeout", 60.0)
    backend = llm_cfg.get("backend", "ollama")

    if backend == "llamacpp":
        # Resolve both generation and prompt/batch counts at the provider
        # boundary. This leaves bounded worker-count headroom for capture, barge
        # detection, and overlapping TTS; it does not pin/reserve CPUs. It also
        # prevents direct construction paths from silently taking the binding's
        # topology-blind defaults.
        common = dict(
            n_ctx=llm_cfg.get("n_ctx", 4096),
            n_threads=llm_cfg.get("n_threads"),
            n_threads_batch=llm_cfg.get("n_threads_batch"),
            n_gpu_layers=llm_cfg.get("n_gpu_layers", 0),
            chat_format=llm_cfg.get("chat_format"),
            think=llm_cfg.get("think", False),
            options=options,
            # KV-cache quantization (llm-inference-9): optional per-profile.
            type_k=llm_cfg.get("type_k"),
            type_v=llm_cfg.get("type_v"),
        )
        main_path = args.model or llm_cfg.get("main_model_path")
        fast_path = args.fast_model or llm_cfg.get("fast_model_path")
        if not main_path:
            raise SystemExit(
                "llamacpp backend needs llm.main_model_path (a GGUF file), but none "
                "is set. Provision the on-device weights (llm-inference-5):\n"
                "  pip install -r requirements-ondevice.txt\n"
                "  python -m tools.setup_models --gguf   "
                "# fetch the shipped MiniCPM GGUF into models/\n"
                "or pass --model <path/to/model.gguf>."
            )
        # Only the main/reasoning client owns ReAct planning. A distinct fast
        # GGUF must not be forced to verify MiniCPM's tool template merely
        # because the main profile opted in.
        main = LlamaCppLLM(
            main_path,
            tool_format=llm_cfg.get("tool_format"),
            **common,
        )
        # A single-tier profile may deliberately use the same compact GGUF for
        # main + fast.  Share one in-process llama.cpp context in that case:
        # loading identical weights/KV twice wastes RAM and CPU headroom that
        # the always-on capture/barge path needs.
        if fast_path and os.path.abspath(fast_path) == os.path.abspath(main_path):
            fast = main
        else:
            fast = LlamaCppLLM(fast_path, **common) if fast_path else None
        return _tag_local_main(_wrap_cloud(main, llm_cfg), main), fast

    host = llm_cfg.get("host")
    keep_alive = llm_cfg.get("keep_alive")
    # Reasoning-model "thinking" is OFF by default on the voice path: a model
    # like gemma4 streams a silent chain-of-thought before any spoken content
    # (measured ~9 s of dead air before the first word of a story on gemma4:12b),
    # which is unacceptable for a real-time turn. Set llm.think=true to opt back
    # in. None would defer to the model's own default (thinking on for gemma4),
    # so we pass an explicit False unless config overrides it.
    think = llm_cfg.get("think", False)
    main_model = args.model or llm_cfg.get("main_model") or config.get("llm_model", "gemma3:12b")
    fast_model = args.fast_model or llm_cfg.get("fast_model")
    main = OllamaLLM(
        model=main_model,
        host=host,
        options=options,
        keep_alive=keep_alive,
        think=think,
        timeout=ollama_timeout,
        client_headers=client_headers,
    )
    fast = (
        OllamaLLM(
            model=fast_model,
            host=host,
            options=options,
            keep_alive=keep_alive,
            think=think,
            timeout=ollama_timeout,
            client_headers=client_headers,
        )
        if fast_model
        else None
    )
    return _tag_local_main(_wrap_cloud(main, llm_cfg), main), fast


def _preset_host(preset: dict) -> Optional[str]:
    """The hosting jurisdiction of a ``cloud_providers`` entry, if declared.

    OpenRouter (and other US presets) carry a top-level ``"host": "US"``; the
    pre-existing presets stash it inside ``_pricing_usd_per_mtok.host`` (e.g.
    DeepSeek/Moonshot ``"host": "CN"``). Check both so the PRC opt-in gate
    (Decision 2 / BR8) catches CN presets regardless of which form they use."""
    host = preset.get("host")
    if not host:
        pricing = preset.get("_pricing_usd_per_mtok")
        if isinstance(pricing, dict):
            host = pricing.get("host")
    return str(host).upper() if host else None


def build_router_llm(config: dict, fast_llm: "LLMClient | None") -> "LLMClient | None":
    """Optional dedicated model for the CAPABILITY-ROUTER disambiguation slot (P3).

    The capability router's low-confidence disambiguator does a one-word
    ACT/SIMPLE/RESEARCH classification; generic small instruct models are weak at
    it, so a function-calling/instruction-tuned small model (e.g. xLAM-2-3b,
    Qwen3-4B) can do it more reliably. When ``llm.router_model`` is set this builds
    a dedicated LOCAL client for it (the router sees the raw query -- it MUST stay
    on-device, §9.7, so it is NEVER cloud-wrapped); otherwise returns ``fast_llm``
    unchanged (default -> byte-identical). Ollama backend only; other backends fall
    back to ``fast_llm``. Owner pulls the model (``ollama pull ...``) first.

    NOTE: this affects ONLY the router's disambiguator. The ReAct PLANNER's tool
    calls deliberately run on the main tier (a multi-step reasoner wants the bigger
    model, not a 3-4B), so router_model does not change planner tool-calling."""
    llm_cfg = config.get("llm", {}) or {}
    router_model = llm_cfg.get("router_model")
    if not router_model:
        return fast_llm
    if str(llm_cfg.get("backend", "ollama")).lower() != "ollama":
        log.info("llm.router_model set but backend != ollama; using the fast tier for routing")
        return fast_llm
    from core.llm import OllamaLLM  # local import: keep top-level light

    return OllamaLLM(
        model=str(router_model),
        host=llm_cfg.get("host"),
        options=llm_cfg.get("options"),  # respect configured generation options
        keep_alive=llm_cfg.get("keep_alive"),
        think=llm_cfg.get("think", False),  # a one-word/tool-call decision needs no CoT
    )


def _build_cloud_client(
    preset_name: str,
    preset: dict,
    *,
    allow_prc: bool = False,
    timeout_s: float = 30.0,
    max_tokens: Optional[int] = None,
    redact_pii_outbound: bool = True,
) -> Optional[OpenAICompatLLM]:
    """Build one :class:`OpenAICompatLLM` from a ``cloud_providers`` entry.

    Returns ``None`` when the entry's API-key env var isn't set -- so missing
    credentials disable that provider individually (the rest of the chain
    keeps working) rather than crashing the whole runtime.

    PRC opt-in (Locked Decision 2): a PRC-hosted preset (``host == "CN"``) is
    dropped unless ``allow_prc`` is set, keeping US-hosted chains the default.
    The drop is INFO-logged **distinctly** from the missing-API-key drop (BR8)
    so a user who simply forgot ``allow_prc`` isn't silently degraded to local.

    The preset's ``profile`` key (e.g. ``"cerebras"``, ``"deepseek_reasoning"``,
    ``"moonshot"``) selects a :class:`core.llm.ProviderProfile` so per-vendor
    quirks (forbidden params, extra_body routing, reasoning-field streaming,
    max_tokens caps) apply without per-call adapter logic. Unknown profile
    names fall back to the safe generic shape.

    ``timeout_s`` / ``max_tokens`` plumb the BR1 short cloud timeout and the
    BR4 per-turn output ceiling into the client (both construction sites pass
    them so a losing/over-long worker is reaped/capped)."""
    if not isinstance(preset, dict):
        return None
    model = preset.get("model")
    if not model:
        return None
    if not allow_prc and _preset_host(preset) == "CN":
        # Distinct from the missing-key drop below (BR8): the preset is fully
        # configured but PRC-hosted, so it's intentionally held back.
        log.info(
            "cloud preset %r dropped: PRC-hosted (host=CN) and llm.cloud.allow_prc "
            "is not set; set allow_prc=true to opt in",
            preset_name,
        )
        return None
    api_key_env = preset.get("api_key_env")
    if api_key_env and not os.environ.get(api_key_env):
        log.info(
            "cloud preset %r dropped: api key env %s is not set",
            preset_name, api_key_env,
        )
        return None
    return OpenAICompatLLM(
        model=model,
        base_url=preset.get("base_url"),
        api_key_env=api_key_env,
        timeout=timeout_s,
        max_tokens=max_tokens,
        options=preset.get("options"),
        profile=preset.get("profile"),
        redact_pii_outbound=redact_pii_outbound,
    )


def _tag_local_main(wrapped: LLMClient, local: LLMClient) -> LLMClient:
    """Stamp the (possibly cloud-wrapped) main client with the BARE LOCAL handle.

    Privacy-critical callers that must NOT reach a cloud chain -- e.g. visual-memory
    captioning, which encodes raw screen frames (§9.7: raw frames never leave the
    device) -- read ``llm.local_main`` to caption on-device only. When cloud is
    off, ``wrapped is local`` and the attribute just points at itself."""
    try:
        setattr(wrapped, "local_main", local)
    except Exception:  # noqa: BLE001 - a client that rejects attrs still works text-only
        pass
    return wrapped


def _wrap_cloud(local_main: LLMClient, llm_cfg: dict) -> LLMClient:
    """Optionally route the main tier through cloud LLM(s) for lower latency.

    Off unless ``llm.cloud.enabled`` (or a populated ``cloud_providers`` +
    ``cloud_chains``) is configured, so the fully-local default is preserved.
    Only the main/reasoning tier is wrapped -- the fast tier is already
    snappy and stays on-device.

    Two configuration shapes:

    - **Multi-provider sensitivity-routed (preferred).** When
      ``llm.cloud_providers`` and ``llm.cloud_chains`` are present, build
      one :class:`HedgeLLM` per chain (each racing local + the chain's
      ordered list of clouds, with failover on error/timeout) and wrap
      them in a :class:`SensitivityRouterLLM` that picks per turn based on
      ``llm.cloud_routing.sensitivity_to_chain``. Missing API keys cause
      that provider to silently drop out of its chains.

    - **Single-cloud back-compat.** When only ``llm.cloud`` is set (no
      providers/chains), use the existing single-HedgeLLM path.
    """
    cloud_cfg = llm_cfg.get("cloud") or {}
    providers = llm_cfg.get("cloud_providers") or {}
    chains_cfg = llm_cfg.get("cloud_chains") or {}
    strategy = cloud_cfg.get("strategy", "hedge")

    if strategy == "local_only" or not cloud_cfg.get("enabled", False):
        return local_main

    # BR1: a short cloud socket timeout reaps a losing worker still blocked in
    # its first-token read fast (the HTTP hard-close is deterministic only after
    # tokens flow). BR4: an optional per-turn output ceiling. Both plumb into
    # every OpenAICompatLLM built below. allow_prc (Decision 2/BR8) gates the
    # PRC-hosted presets; default False keeps US-hosted chains the default.
    allow_prc = bool(cloud_cfg.get("allow_prc", False))
    timeout_s = float(cloud_cfg.get("timeout_s", 30.0) or 30.0)
    max_tokens = cloud_cfg.get("max_tokens")
    max_tokens = int(max_tokens) if max_tokens is not None else None
    # smart-routing-5: optional cost/ttft-aware chain ordering, default OFF so
    # the configured failover order is unchanged. When on, each chain's preset
    # list is stably reordered by the documentation-only ttft/$-per-Mtok
    # metadata before the HedgeLLM is built (core.routing.order_presets_by_cost
    # is fail-safe: same multiset, original order on any malformed input).
    cost_order = bool(cloud_cfg.get("cost_order", False))
    # §9.7 last-line net (default ON): scrub high-confidence PII from the outbound
    # cloud prompt, independent of the regex sensitivity classifier. Local models
    # are never touched. Off restores byte-identical prior behavior.
    redact_pii_outbound = bool(cloud_cfg.get("redact_pii_outbound", True))

    hedge_kwargs = dict(
        strategy=strategy,
        hedge_delay_ms=int(cloud_cfg.get("hedge_delay_ms", 150)),
        ttft_deadline_ms=int(cloud_cfg.get("ttft_deadline_ms", 1200)),
    )

    # Multi-provider path.
    if providers and chains_cfg:
        # Resolve each provider lazily; drop ones with missing API keys or
        # (without allow_prc) a PRC host.
        resolved: dict[str, OpenAICompatLLM] = {}
        for name, preset in providers.items():
            if isinstance(preset, dict) and name.startswith("_"):
                continue  # skip _comment / metadata keys
            client = _build_cloud_client(
                name, preset,
                allow_prc=allow_prc, timeout_s=timeout_s, max_tokens=max_tokens,
                redact_pii_outbound=redact_pii_outbound,
            )
            if client is not None:
                resolved[name] = client

        hedged_chains: dict[str, LLMClient] = {}
        any_clouds = False
        for chain_name, preset_names in chains_cfg.items():
            if chain_name.startswith("_"):
                continue
            if not isinstance(preset_names, (list, tuple)):
                continue
            # Optional cost/ttft ordering (smart-routing-5), flag-gated. Off by
            # default -> the configured order is preserved byte-for-byte; on, a
            # fail-safe stable reorder by ttft/$-per-Mtok floats the cheaper/
            # faster presets to the front of the failover chain.
            if cost_order:
                preset_names = order_presets_by_cost(preset_names, providers)
            chain_clouds = [resolved[n] for n in preset_names if n in resolved]
            if chain_clouds:
                any_clouds = True
            hedged_chains[chain_name] = HedgeLLM(
                local=local_main, cloud=chain_clouds, **hedge_kwargs
            )

        if not hedged_chains or not any_clouds:
            # Every chain ended up empty (e.g. no API keys set): fall through
            # to the local main tier without surprise wrappers. Otherwise
            # SensitivityRouterLLM would wrap an all-local hedge that adds
            # threading overhead for no benefit.
            return local_main

        routing_cfg = llm_cfg.get("cloud_routing") or {}
        default_chain = str(routing_cfg.get("default_chain", "private") or "private")
        if default_chain not in hedged_chains:
            default_chain = next(iter(hedged_chains))
        selector = build_chain_selector({"llm": llm_cfg})
        # Visibility: cloud egress is now active (the user deliberately set
        # enabled=true AND a key resolved). Log it WARN-level so the privacy
        # boundary change isn't silent -- §9.7 still keeps raw audio local + scrubs
        # outbound PII, but the user should know cloud is on.
        log.warning(
            "cloud LLM egress ACTIVE: %d chain(s) %s, default=%s, redact_pii_outbound=%s "
            "(post-ASR text only; raw audio/STT/TTS stay local per §9.7)",
            len(hedged_chains), sorted(hedged_chains), default_chain, redact_pii_outbound,
        )
        return SensitivityRouterLLM(
            hedged_chains, selector=selector, default_chain=default_chain
        )

    # Single-cloud back-compat path. Same BR1 timeout + BR4 max_tokens plumbing
    # as the multi-provider site above (BR9: both sites must honour them).
    model = cloud_cfg.get("model")
    if not model:
        return local_main
    cloud = OpenAICompatLLM(
        model=model,
        base_url=cloud_cfg.get("base_url"),
        api_key_env=cloud_cfg.get("api_key_env"),
        timeout=timeout_s,
        max_tokens=max_tokens,
        options=cloud_cfg.get("options"),
        redact_pii_outbound=redact_pii_outbound,  # BR9: both sites honor the §9.7 scrub
    )
    # Visibility: same WARN as the multi-chain branch -- cloud egress is now active
    # (the user deliberately set enabled=true AND a model is configured), so the
    # privacy-boundary change is never silent on EITHER path (§9.7).
    log.warning(
        "cloud LLM egress ACTIVE: single-cloud model=%s, redact_pii_outbound=%s "
        "(post-ASR text only; raw audio/STT/TTS stay local per §9.7)",
        model, redact_pii_outbound,
    )
    return HedgeLLM(local=local_main, cloud=cloud, **hedge_kwargs)


# Back-compat alias for the historical private name some tests import.
_build_llms = build_llms
