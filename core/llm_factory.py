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

# Keep the historical logger name ("speaker.app") so the cloud-drop INFO logs
# land under the same logger the tests + run-bundle expect after this code
# moved out of core/app.py.
log = logging.getLogger("speaker.app")

__all__ = [
    "build_llms",
    "_build_llms",
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
    backend = llm_cfg.get("backend", "ollama")

    if backend == "llamacpp":
        common = dict(
            n_ctx=llm_cfg.get("n_ctx", 4096),
            n_threads=llm_cfg.get("n_threads"),
            n_gpu_layers=llm_cfg.get("n_gpu_layers", 0),
            chat_format=llm_cfg.get("chat_format"),
            options=options,
        )
        main_path = args.model or llm_cfg.get("main_model_path")
        fast_path = args.fast_model or llm_cfg.get("fast_model_path")
        if not main_path:
            raise SystemExit("llamacpp backend needs llm.main_model_path (a GGUF file).")
        main = LlamaCppLLM(main_path, **common)
        fast = LlamaCppLLM(fast_path, **common) if fast_path else None
        return _wrap_cloud(main, llm_cfg), fast

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
        model=main_model, host=host, options=options, keep_alive=keep_alive, think=think
    )
    fast = (
        OllamaLLM(
            model=fast_model, host=host, options=options, keep_alive=keep_alive, think=think
        )
        if fast_model
        else None
    )
    return _wrap_cloud(main, llm_cfg), fast


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


def _build_cloud_client(
    preset_name: str,
    preset: dict,
    *,
    allow_prc: bool = False,
    timeout_s: float = 30.0,
    max_tokens: Optional[int] = None,
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
    )


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
    )
    return HedgeLLM(local=local_main, cloud=cloud, **hedge_kwargs)


# Back-compat alias for the historical private name some tests import.
_build_llms = build_llms
