"""Per-query model-tier routing for the two-model (fast/main) split.

The ``Router`` abstraction is replicated from RouteLLM (Apache-2.0,
github.com/lm-sys/RouteLLM): a router maps a query to a scalar in ``[0, 1]``
("how much this query needs the strong/main model") which a threshold turns
into a binary tier choice. We re-implement it here rather than depend on the
library so the default path stays dependency-light and phone-safe; the heavy,
learned variant is opt-in and lazily imports torch/transformers.

This is the TIER router (fast vs main model). It is NOT a duplicate of
``core/capability_router.py``: that module is the unified *action* router
(CONTROL/SIMPLE/RESEARCH/ACT) which *composes* this tier ``Router`` (it passes
one in and adapts it back via ``CapabilityTierRouter``). One decision there
drives both the tier choice here and the ReAct ``escalate`` predicate. See that
module's docstring for the full relationship.
"""

from __future__ import annotations

from typing import Mapping, Optional, Protocol, runtime_checkable

# A model tier: "fast" (small, snappy) or "main" (large/multimodal, slower).
ModelTier = str

FAST: ModelTier = "fast"
MAIN: ModelTier = "main"

# --- Live headroom signal (smart-routing-2) --------------------------------
#
# The router's static per-profile decision is the floor. A *live* signal --
# the local model's recent time-to-first-token (from core.metrics) and an
# optional system-load fraction (from core.sysinfo) -- can NUDGE a borderline
# turn toward the main/cloud tier when the local fast tier is slow or loaded,
# so a busy device offloads to the bigger/cloud model instead of stalling.
#
# Hard guarantee (the gate): the nudge is ADDITIVE-ONLY toward main and
# CLAMPED, and a missing / non-numeric / out-of-range signal contributes
# exactly zero. It can therefore never lower the score and never demote a turn
# the static config would have escalated -- it only ever escalates, and only on
# a *good* signal. A bad or absent signal leaves the static decision untouched,
# so the local tier's static per-profile behavior is the floor and is never
# starved.
#
# The hint is read from ``context['live']`` (a plain mapping) so the Router
# Protocol signature stays unchanged and every existing call site keeps
# working with no live signal at all.
LIVE_CONTEXT_KEY = "live"

# Local TTFT (ms) at/above which the nudge saturates. ~2.5s to first token is
# already a poor local experience; past it, lean on the cloud/main tier.
_TTFT_NUDGE_FLOOR_MS = 800.0    # below this: local is snappy, no nudge
_TTFT_NUDGE_CEIL_MS = 2500.0    # at/above this: full nudge
# System load (0..1, e.g. CPU or GPU utilisation fraction) nudge band.
_LOAD_NUDGE_FLOOR = 0.75        # below this: plenty of headroom, no nudge
_LOAD_NUDGE_CEIL = 0.98         # at/above this: full nudge
# Maximum total escalation the live signal may add to the static score. Kept
# modest (< the 0.5 default threshold span) so the live signal only ever tips
# *borderline* turns -- it cannot, on its own, drag a clearly-fast query to
# main, and it can never subtract.
_MAX_LIVE_NUDGE = 0.25

# Floor (ms) for the dynamic per-call hedge delay. When live routing shortens
# the local-vs-cloud start gap because local is slow/loaded, the delay is
# clamped to this floor so it can never reach zero/negative (which would make a
# *full* race -- starting the cloud with no local head start at all -- on every
# loaded turn, the very starvation the live signal is meant to avoid). A small
# positive floor still lets local win when it is briefly quick.
HEDGE_DELAY_FLOOR_MS = 40


def _as_float(value: object) -> Optional[float]:
    """Best-effort float, returning ``None`` for missing/garbage/non-finite.

    The router's fail-safe rests on this: any signal we can't trust becomes
    ``None`` and contributes no nudge."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return f


def _ramp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to ``[lo, hi]`` then scale to ``[0, 1]`` (0 at/below lo)."""
    if hi <= lo:
        return 0.0
    if value <= lo:
        return 0.0
    if value >= hi:
        return 1.0
    return (value - lo) / (hi - lo)


def live_nudge(live: object) -> float:
    """Compute the additive-only escalation from a live headroom signal.

    ``live`` is whatever sat at ``context['live']`` -- expected to be a mapping
    with optional ``ttft_ms`` (local rolling TTFT) and/or ``load`` (0..1 system
    utilisation) keys. Returns a value in ``[0, _MAX_LIVE_NUDGE]``; anything we
    cannot interpret (wrong type, missing keys, NaN, negative) yields ``0.0``
    so the static decision stands. The two sub-signals are combined by max (not
    sum) so a single hot dimension is enough but two don't double-count past the
    cap."""
    if not isinstance(live, Mapping):
        return 0.0
    ttft = _as_float(live.get("ttft_ms"))
    load = _as_float(live.get("load"))
    ttft_frac = _ramp(ttft, _TTFT_NUDGE_FLOOR_MS, _TTFT_NUDGE_CEIL_MS) if ttft is not None else 0.0
    load_frac = _ramp(load, _LOAD_NUDGE_FLOOR, _LOAD_NUDGE_CEIL) if load is not None else 0.0
    return _MAX_LIVE_NUDGE * max(ttft_frac, load_frac)


def dynamic_hedge_delay_ms(live: object, base_delay_ms: object) -> Optional[int]:
    """Shorten the hedge delay for a slow/loaded local tier (dynamic hedge).

    The live signal (same ``ttft_ms`` / ``load`` shape as :func:`live_nudge`)
    is reused: when local is slow or loaded, the cloud race should start sooner
    so the user isn't stuck behind a stalling local tier. We scale the static
    ``base_delay_ms`` down by the same 0..1 saturation fraction the nudge uses
    (max of the two sub-signals) and clamp the result to
    :data:`HEDGE_DELAY_FLOOR_MS` so it can never reach zero/negative.

    Returns ``None`` (the per-call override is then NOT applied -> the
    constructor's static delay stands) when there is no usable signal, the
    fraction is zero (local is snappy / uninterpretable), or ``base_delay_ms``
    is missing/non-positive. So a missing or good signal leaves hedge timing
    byte-identical; only a genuinely slow/loaded signal pulls the cloud forward,
    and never past the floor."""
    base = _as_float(base_delay_ms)
    if base is None or base <= 0.0:
        return None
    if not isinstance(live, Mapping):
        return None
    ttft = _as_float(live.get("ttft_ms"))
    load = _as_float(live.get("load"))
    ttft_frac = _ramp(ttft, _TTFT_NUDGE_FLOOR_MS, _TTFT_NUDGE_CEIL_MS) if ttft is not None else 0.0
    load_frac = _ramp(load, _LOAD_NUDGE_FLOOR, _LOAD_NUDGE_CEIL) if load is not None else 0.0
    frac = max(ttft_frac, load_frac)
    if frac <= 0.0:
        return None
    shortened = base * (1.0 - frac)
    clamped = max(float(HEDGE_DELAY_FLOOR_MS), shortened)
    # Only an actual shortening is worth overriding; if the clamp floor is at or
    # above the static delay there is nothing to gain (and we must never
    # *lengthen* it), so leave the static delay in place.
    if clamped >= base:
        return None
    return int(round(clamped))


@runtime_checkable
class Router(Protocol):
    """Decides which model tier should answer a query.

    ``score`` returns ``[0, 1]`` (higher = favour the main model); ``choose``
    turns that into a tier via the router's threshold.
    """

    def score(self, query: str, context: Mapping[str, object]) -> float: ...

    def choose(self, query: str, context: Mapping[str, object]) -> ModelTier: ...


class BaseRouter:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, query: str, context: Mapping[str, object]) -> float:
        raise NotImplementedError

    def choose(self, query: str, context: Mapping[str, object]) -> ModelTier:
        return MAIN if self.score(query, context) >= self.threshold else FAST


# Words that signal a query wants reasoning/synthesis rather than a snappy
# lookup. Borrowed in spirit from vLLM Semantic Router's complexity signal
# (Apache-2.0), reduced to a lexical heuristic so it runs on-device with no ML.
_COMPLEXITY_MARKERS = (
    "why",
    "how",
    "explain",
    "compare",
    "comparison",
    "analyze",
    "analyse",
    "evaluate",
    "summarize",
    "summarise",
    "reason",
    "step by step",
    "derive",
    "calculate",
    "prove",
    "design",
    "debug",
    "algorithm",
    "difference between",
    "pros and cons",
    "trade-off",
    "tradeoff",
    "implications",
    "in detail",
)

# Modes whose work is inherently heavyweight (research/synthesis) vs. modes
# that are pure transcription and never need the big model.
_MAIN_MODES = {"research", "search", "meeting"}
_FAST_MODES = {"dictation"}

# IntentKind values (as strings -- the enum is in always_on_agent and we
# don't import it here to keep this module phone-safe / dependency-light;
# values come through ``context['intent_kind']`` as the enum's string value).
_MAIN_INTENTS = {"research", "search"}
_FAST_INTENTS = {"command", "dictation", "meeting_note"}

# Long-form / generation asks -- "tell me a story", "write a poem", "walk me
# through ..." -- want the big model: the fast tier deflects ("I can find one
# online") or gives a shallow one-liner. A single hit escalates strongly (these
# phrases are unambiguous generation requests). Kept DISTINCT from
# ``_COMPLEXITY_MARKERS`` so the calibrated borderline router tests are
# undisturbed (no overlap with "explain ... difference between").
_GENERATION_MARKERS = (
    "story", "poem", "write me", "write a", "tell me a", "tell me about",
    "walk me through", "a list of", "give me a", "essay", "lyrics", "joke",
)


class HeuristicRouter(BaseRouter):
    """Dependency-light router: combines mode, length and complexity signals.

    Runs anywhere (phone included) with no model load. Tuned so short, literal
    asks ("what time is it") stay on the fast tier while reasoning-heavy or
    long queries escalate to the main model.
    """

    def score(self, query: str, context: Mapping[str, object]) -> float:
        q = query.lower().strip()
        n = len(q.split())
        ctx = context or {}
        mode = str(ctx.get("mode", "")).lower()
        intent_kind = str(ctx.get("intent_kind", "")).lower()

        s = 0.0
        if mode in _MAIN_MODES:
            s += 0.6
        elif mode in _FAST_MODES:
            s -= 0.3

        # Intent signal: the deterministic classifier already decided this is
        # a research/search turn, so escalate to main even if the mode hasn't
        # switched yet (and conversely keep COMMAND/DICTATION on fast).
        if intent_kind in _MAIN_INTENTS:
            s += 0.6
        elif intent_kind in _FAST_INTENTS:
            s -= 0.3

        if n >= 40:
            s += 0.4
        elif n >= 20:
            s += 0.25
        elif n >= 12:
            s += 0.12

        hits = sum(1 for marker in _COMPLEXITY_MARKERS if marker in q)
        s += min(0.5, 0.18 * hits)

        # Long-form / generation request -> the big model (one hit is enough).
        if any(marker in q for marker in _GENERATION_MARKERS):
            s += 0.5

        if q.count("?") >= 2:
            s += 0.1

        # The static score above is the floor. A live headroom signal (local
        # slow/loaded) may only ADD to it -- clamped, additive-only -- so a
        # missing/garbage signal leaves the decision unchanged and the local
        # tier is never starved (smart-routing-2).
        s += live_nudge(ctx.get(LIVE_CONTEXT_KEY))

        return max(0.0, min(1.0, s))


class LearnedRouter(BaseRouter):
    """Desktop-only learned router: a BERT sequence classifier.

    Mirrors RouteLLM's ``BERTRouter`` pattern (Apache-2.0). torch/transformers
    are imported here, inside ``__init__``, so simply having this class on the
    import path costs the phone build nothing — it is only constructed when
    ``llm.router.backend == "learned"``.
    """

    def __init__(
        self,
        model: str = "routellm/bert_gpt4_augmented",
        threshold: float = 0.5,
    ):
        super().__init__(threshold)
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - desktop-only path
            raise RuntimeError(
                "LearnedRouter needs torch+transformers (desktop only). "
                "Install them or set llm.router.backend to 'heuristic'."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForSequenceClassification.from_pretrained(model)
        self._model.eval()

    def score(self, query: str, context: Mapping[str, object]) -> float:
        torch = self._torch
        with torch.no_grad():
            inputs = self._tokenizer(query, return_tensors="pt", truncation=True)
            logits = self._model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
        # Convention: the last/highest class is "needs the strong model".
        return float(probs[-1])


def build_router(config: Optional[Mapping[str, object]]) -> Router:
    """Build a router from the ``llm.router`` config block.

    Defaults to the phone-safe ``HeuristicRouter``. Selecting ``"learned"``
    pulls in the desktop-only torch/transformers path lazily.
    """
    llm_cfg = (config or {}).get("llm", {}) or {}
    cfg = llm_cfg.get("router", {}) or {}
    backend = cfg.get("backend", "heuristic")
    threshold = float(cfg.get("threshold", 0.5))
    if backend == "learned":
        model = cfg.get("model", "routellm/bert_gpt4_augmented")
        return LearnedRouter(model=model, threshold=threshold)
    return HeuristicRouter(threshold=threshold)


class ChainSelector:
    """Pick a cloud-provider-chain name from a turn's sensitivity tag.

    The chain selector is the second axis of routing (the first being tier
    ``main`` vs ``fast``): given a query's ``Sensitivity``, return the key
    into ``llm.cloud_chains`` that the brain should use. Defaults route to
    the ``private`` chain whenever the sensitivity is unknown or missing.

    Held as a plain class rather than a Protocol so the runtime can keep a
    single instance and tests can construct one directly with a mapping.
    """

    def __init__(
        self,
        mapping: Optional[Mapping[str, str]] = None,
        *,
        default_chain: str = "private",
    ):
        self.mapping = dict(mapping or {})
        self.default_chain = default_chain

    def choose_chain(self, context: Mapping[str, object]) -> str:
        sensitivity = str((context or {}).get("sensitivity", "") or "").lower()
        if sensitivity and sensitivity in self.mapping:
            return self.mapping[sensitivity]
        return self.default_chain


def build_chain_selector(config: Optional[Mapping[str, object]]) -> ChainSelector:
    """Build a ChainSelector from the ``llm.cloud_routing`` config block.

    Without configuration: a no-op selector that always returns
    ``"private"`` -- the safe default that matches a US-only chain.
    """
    llm_cfg = (config or {}).get("llm", {}) or {}
    routing = llm_cfg.get("cloud_routing", {}) or {}
    mapping = routing.get("sensitivity_to_chain", {}) or {}
    default = str(routing.get("default_chain", "private") or "private")
    return ChainSelector(mapping, default_chain=default)


def _preset_cost_key(preset: object) -> tuple[float, float, float, float]:
    """Sort key for a ``cloud_providers`` preset:
    ``(host_rank, ttft_ms, in_cost, out_cost)``.

    ``host_rank`` is the OUTERMOST dimension so cost/latency ordering stays
    WITHIN a jurisdiction tier and never reorders ACROSS it: a PRC-hosted
    (``host == "CN"``) preset always sorts AFTER every US/unknown-hosted one,
    even when it is cheaper or faster. This keeps the §9.7 / PRC-opt-in intent
    (prefer US jurisdiction) intact under cost ordering -- the bug it fixes is a
    cheap CN provider floating ahead of a US provider the user ordered first.
    The host tag is read from ``preset['host']`` or the pricing block's ``host``
    (the two forms ``_preset_host`` accepts), so both annotated shapes are tiered.

    The remaining dimensions read the documentation-only
    ``_pricing_usd_per_mtok`` metadata (``ttft_ms`` + ``in``/``out`` $/Mtok):
    lower ttft first, ties broken by input then output cost. Any preset missing a
    field sorts *last* on that field (``+inf``) so well-annotated, faster presets
    float to the front of their host group while unannotated ones keep their
    relative place behind them. NB a cost-annotated preset with NO ``ttft_ms``
    (an aggregator like OpenRouter) sorts after ttft-annotated peers in its host
    group; add ``ttft_ms`` to refine its position -- but host_rank still keeps it
    ahead of any CN preset (the reported sink-below-CN bug)."""
    host = None
    pricing = None
    if isinstance(preset, Mapping):
        host = preset.get("host")
        pricing = preset.get("_pricing_usd_per_mtok")
        if not host and isinstance(pricing, Mapping):
            host = pricing.get("host")
    host_rank = 1.0 if str(host or "").upper() == "CN" else 0.0
    if not isinstance(pricing, Mapping):
        return (host_rank, float("inf"), float("inf"), float("inf"))
    ttft = _as_float(pricing.get("ttft_ms"))
    cost_in = _as_float(pricing.get("in"))
    cost_out = _as_float(pricing.get("out"))
    return (
        host_rank,
        ttft if ttft is not None else float("inf"),
        cost_in if cost_in is not None else float("inf"),
        cost_out if cost_out is not None else float("inf"),
    )


def order_presets_by_cost(
    preset_names: object,
    providers: Optional[Mapping[str, object]],
) -> list[str]:
    """Stable-reorder a chain's preset names by host/ttft/cost metadata (smart-routing-5).

    ADDITIVE + fail-safe: the chain's configured order is the floor. Presets are
    stably sorted by ``(host_rank, ttft_ms, in $/Mtok, out $/Mtok)`` (see
    :func:`_preset_cost_key`): CN-hosted presets always sort after US/unknown
    ones (cost optimizes WITHIN a jurisdiction, never across it), then by
    ttft/cost from each entry's ``_pricing_usd_per_mtok`` block in ``providers``
    (the same metadata today documentation-only). A name with no metadata keeps
    its original relative position (sorts last on each missing key). Anything malformed --
    non-sequence input, missing ``providers``, an exploding comparison -- returns
    the input order unchanged, so a bad signal can never reshuffle (let alone
    empty) a chain. Names absent from ``providers`` are preserved in place too.
    The result is the same multiset of names, only reordered."""
    if not isinstance(preset_names, (list, tuple)):
        return []
    names = [n for n in preset_names if isinstance(n, str)]
    if not providers or len(names) < 2:
        return list(names)
    try:
        # ``sorted`` is stable: equal keys (incl. all-unannotated) preserve the
        # original chain order, so failover semantics are unchanged on ties.
        return sorted(names, key=lambda n: _preset_cost_key(providers.get(n, {})))
    except Exception:  # noqa: BLE001 - ordering is best-effort; never break the chain
        return list(names)
