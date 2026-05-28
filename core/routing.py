"""Per-query model-tier routing for the two-model (fast/main) split.

The ``Router`` abstraction is replicated from RouteLLM (Apache-2.0,
github.com/lm-sys/RouteLLM): a router maps a query to a scalar in ``[0, 1]``
("how much this query needs the strong/main model") which a threshold turns
into a binary tier choice. We re-implement it here rather than depend on the
library so the default path stays dependency-light and phone-safe; the heavy,
learned variant is opt-in and lazily imports torch/transformers.
"""

from __future__ import annotations

from typing import Mapping, Optional, Protocol, runtime_checkable

# A model tier: "fast" (small, snappy) or "main" (large/multimodal, slower).
ModelTier = str

FAST: ModelTier = "fast"
MAIN: ModelTier = "main"


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

        if q.count("?") >= 2:
            s += 0.1

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
