"""Shared, backend-neutral recall selector -- the one place that decides *what*
(and *how much*) long-term memory gets injected into a prompt.

This module is the answer to "make memory smart and efficient so it does not blow
up the context". Historically the recall block was bounded by three independent
**blunt char caps** (per-item ``[:150]`` in both the RAM path and the Postgres
path, plus a total ``[:600]`` re-slice at the capability injection site) gated by
**fixed magic-number thresholds** (cosine ``> 0.6``, ``top-3``, keyword
``overlap >= 2``). That stack could (a) cut a fact mid-word, (b) inject a summary
*and* the very messages it already summarizes (double-counting tokens), (c) dump
the entire user-profile unbounded, and (d) silently drop a perfectly good hit
whose similarity happened to be ``0.59``.

Everything here is **pure-Python, stdlib-only** (``re``/``math``/``difflib`` +
the package-local ``.text`` normalizer -- no ``numpy``/``psycopg``/embeddings),
so it is:

* **Tier-0 unit-testable** with hand-built :class:`Candidate` lists (no DB, no
  models);
* **backend-neutral** -- the in-RAM :class:`always_on_agent.memory.SessionMemory`,
  the Postgres :class:`utils.memory.MemoryManager`, and the future SQLite/Dart
  mobile path all gather candidates their own way and hand them to
  :func:`build_block`, so every backend emits a **byte-identical** block for an
  identical candidate list (closing the long-standing RAM-vs-Postgres parity
  gap);
* **dependency-one-way** -- ``utils.memory`` imports *from* here; this module
  imports nothing from ``utils``, so there is no cycle and the brain stays
  Postgres-free.

The replacement is ONE integer **token budget** + a **data-derived adaptive
cutoff**, honoring the owner's standing device-adaptive directive (no fixed
relevance/length constants; the cut is read off the candidate score
distribution, mirroring the self-calibrated pattern in ``core/engines/_dtd.py``).

``Candidate.score`` invariant (load-bearing -- read this):
    The score is **backend-native** -- raw cosine similarity on the Postgres
    path, integer keyword-overlap count on the RAM path -- and is only ever
    ranked *within a single candidate list*. :func:`adaptive_cutoff` is
    **scale-invariant** (it compares *relative* gaps), so the two scales can feed
    the same function without cross-backend normalization. Never merge two
    different score semantics into one list (e.g. flat profile-confidence next to
    cosine similarity) -- run them through :func:`build_block` separately.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Optional, Sequence

from .text import normalize_text

# Mirrors utils.memory_config.MemoryWriterConfig.dedupe_similarity (0.92): the
# WRITE side already collapses near-duplicate utterances at this SequenceMatcher
# ratio, so the recall side uses the same value to stay consistent. This is a
# string-similarity constant (device/room-invariant -- it does NOT vary with the
# mic or acoustics), NOT a relevance decision threshold, so it is exempt from the
# no-magic-number directive that governs the cutoff. Overridable via
# RecallBudget.dedup_ratio.
_DEFAULT_DEDUP_RATIO = 0.92

# Sentence splitter for query-aware compression. Deliberately simple + stdlib:
# a run of non-terminator chars followed by an optional terminator.
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")

# Query scaffold and canonical recall labels carry no subject identity.  They
# are stripped by strong_subject_overlap() before comparing a question with one
# recalled line.  Keep this language-light and deterministic: the helper is a
# routing hint, not a semantic answerer, and must remain portable to the mobile
# reference implementation.
_SUBJECT_SCAFFOLD = frozenset({
    "a", "about", "again", "an", "and", "are", "as", "at", "be", "been",
    "before", "being", "but", "by", "can",
    "conversation", "conversations", "could", "did", "do", "does", "earlier",
    "for", "from", "had", "has", "have", "how", "i", "in", "into", "is", "it",
    "last", "me", "memory", "mine", "my", "of", "on", "or",
    "our", "ours", "past", "please", "previous", "previously", "recall",
    "recalled", "remember", "remind", "said", "screen", "session", "summary",
    "tell", "that", "the", "their", "them", "they", "this", "to", "us",
    "user", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your", "yours", "assistant",
})


def _subject_word_forms(word: str) -> set[str]:
    """Small inflection family used only for conservative subject matching."""
    forms = {word}
    if len(word) > 4 and word.endswith("ies"):
        forms.add(word[:-3] + "y")
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        forms.add(word[:-1])
    if len(word) > 4 and word.endswith("ed"):
        forms.add(word[:-2])
        forms.add(word[:-1])  # named -> name
    if len(word) > 5 and word.endswith("ing"):
        forms.add(word[:-3])
        forms.add(word[:-3] + "e")
    return forms


def _subject_words(text: str) -> list[str]:
    out: list[str] = []
    for word in normalize_text(text).split():
        if word in _SUBJECT_SCAFFOLD or word in out:
            continue
        out.append(word)
    return out


def _looks_like_recall_question(query: str) -> bool:
    normalized = normalize_text(query)
    words = normalized.split()
    if not words:
        return False
    if "?" in query:
        return True
    if words[0] in {
        "are", "can", "could", "did", "do", "does", "how", "is", "remember",
        "recall", "remind", "was", "were", "what", "when", "where", "which", "who",
    }:
        return True
    padded = f" {normalized} "
    return any(
        phrase in padded
        for phrase in (
            " tell me ", " have i ", " have we ", " do you remember ",
            " can you recall ", " remind me ",
        )
    )


def strong_subject_overlap(query: str, recalled: str) -> bool:
    """Whether one recalled line covers every substantive query subject term.

    This is deliberately stricter than the selector that produced ``recalled``.
    Keyword and vector recall may conservatively render a weak candidate, but a
    weak/nonempty block alone must never promote the turn.  A match requires at
    least one substantive query term and requires *one line* (not a union of
    unrelated memories) to contain an inflectional form of every term.

    The result is only a trusted scalar routing hint.  Recalled text remains
    untrusted data and still goes through the normal spotlight/sensitivity
    boundary before either model sees it.
    """
    wanted = _subject_words(query)
    if not wanted or not recalled or not _looks_like_recall_question(query):
        return False
    wanted_forms = [_subject_word_forms(word) for word in wanted]
    for line in recalled.splitlines():
        words = _subject_words(line)
        if not words:
            continue
        available: set[str] = set()
        for word in words:
            available.update(_subject_word_forms(word))
        if all(forms & available for forms in wanted_forms):
            return True
    return False


@dataclass(frozen=True)
class Candidate:
    """One retrievable memory item, normalized across backends.

    ``score`` is backend-native (see module docstring invariant). ``span`` is the
    ``(start_ts, end_ts)`` a summary covers, used to drop the individual messages
    it already subsumes."""

    text: str
    score: float
    kind: str = "message"  # 'message' | 'summary' | 'profile' | 'vision'
    role: Optional[str] = None  # 'user' | 'assistant' | None
    timestamp: float = 0.0
    tags: tuple[str, ...] = ()
    span: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class RecallBudget:
    """The single knob set that replaces every char cap + fixed threshold.

    ``max_tokens`` is the ONE bound on the whole injected block (header
    included). ``chars_per_token`` is the only per-device tuning knob (a
    measurement ratio, not a decision threshold -- mergeable via
    ``device_profiles``). ``cutoff_k``/``dedup_ratio`` default to ``0`` =
    "fully data-derived"; a positive value is an opt-in escape hatch that pins a
    fixed behavior."""

    max_tokens: int = 150
    chars_per_token: float = 4.0
    min_keep: int = 1
    cutoff_k: float = 0.0  # 0 == pure adaptive gap; >0 also requires score >= mean + k*std
    dedup_ratio: float = 0.0  # 0 == _DEFAULT_DEDUP_RATIO; >0 pins a SequenceMatcher ratio
    header: str = "=== Past Conversations ==="
    # Multi-signal recall scoring (Generative-Agents: relevance + recency +
    # importance), ALL default-OFF so the relevance-only block stays byte-identical
    # until opted in. When >0, the native relevance score is scaled by a convex blend
    # of a recency-decay factor and a kind-importance weight, applied multiplicatively
    # so the backend-native score scale (cosine on PG, overlap on RAM) is preserved and
    # adaptive_cutoff stays scale-invariant. The blend is clamped to a true convex
    # [0, 1] (so it only DOWN-weights, never boosts above native, even if the weights
    # sum past 1), and a negative native score (off-topic PG cosine) is left unchanged
    # so it can't be ranked up. NOTE: when recency is on, blocks are NOT byte-identical
    # ACROSS backends (each stamps a different wall-clock for the same logical add), so
    # recency-ranked recall is single-backend by design.
    recency_weight: float = 0.0       # 0 == OFF; weight on the recency-decay factor
    recency_half_life_days: float = 7.0  # age at which the recency factor halves
    importance_weight: float = 0.0    # 0 == OFF; weight on the kind-importance factor


# --- token accounting -------------------------------------------------------


def estimate_tokens(text: str, cpt: float = 4.0) -> int:
    """Tokenizer-free token estimate: ``max(word_count, ceil(chars / cpt))``.

    Dependency-free + deterministic so it runs identically on desktop, the
    no-DB test suite, and the Dart/SQLite mobile path (no ``tiktoken``). Using
    ``max`` of the two heuristics keeps it from *under*-counting either ordinary
    prose (word-dominated) or a few very long tokens (char-dominated), so the
    packed block never silently exceeds the budget. ``0`` for empty text."""
    if not text or not text.strip():
        return 0
    words = len(text.split())
    chars = math.ceil(len(text) / max(cpt, 1.0))
    return max(words, chars, 1)


# --- multi-signal scoring (recency + importance), default-OFF ---------------

# Kind-importance prior (Generative-Agents "poignancy", deterministic + backend-
# neutral so it stays parity-safe): durable profile facts outrank rolling
# summaries, which outrank raw turns; ambient screen memories rank lowest.
_KIND_IMPORTANCE = {"profile": 1.0, "summary": 0.85, "message": 0.6, "vision": 0.5}


def _importance(c: "Candidate") -> float:
    return _KIND_IMPORTANCE.get(c.kind, 0.6)


def _apply_signals(
    cands: Sequence["Candidate"], budget: "RecallBudget", now: Optional[float]
) -> list["Candidate"]:
    """Scale each candidate's native relevance score by a convex blend of recency
    decay + kind importance (Generative-Agents score). No-op (returns the list
    unchanged, byte-identical) when both weights are 0 -- the default -- so the
    relevance-only contract holds until the owner opts in.

    The blend is a true CONVEX factor in ``[0, 1]`` applied MULTIPLICATIVELY to the
    native score (cosine on Postgres, integer overlap on RAM/SQLite), so it only
    ever DOWN-weights a stale/unimportant item (never boosts above native) and the
    per-list score scale is preserved -- :func:`adaptive_cutoff` stays
    scale-invariant and the two score semantics are never mixed into one additive
    sum (the module-docstring invariant).

    A NEGATIVE native score (Postgres cosine can be ``< 0`` for an off-topic
    embedding) is left UNCHANGED: multiplying it by a ``< 1`` factor would move it
    toward zero, i.e. RANK IT UP -- the opposite of down-weighting. Leaving it keeps
    every off-topic item strictly below every on-topic one and preserves their
    relative order."""
    rw = max(0.0, budget.recency_weight)
    iw = max(0.0, budget.importance_weight)
    if rw <= 0.0 and iw <= 0.0:
        return list(cands)
    if now is None:
        now = time.time()
    hl = budget.recency_half_life_days
    base = 1.0 - rw - iw
    out: list[Candidate] = []
    for c in cands:
        if c.score <= 0.0:
            out.append(c)  # never lift an off-topic (negative/zero) score
            continue
        recency = 1.0
        if rw > 0.0 and hl > 0.0 and c.timestamp:
            age_days = max(0.0, (now - c.timestamp) / 86400.0)
            recency = 0.5 ** (age_days / hl)
        imp = _importance(c) if iw > 0.0 else 0.0
        # Clamp to [0, 1]: a true convex blend even when the weights sum past 1.
        blend = max(0.0, min(1.0, base + rw * recency + iw * imp))
        out.append(replace(c, score=c.score * blend))
    return out


# --- adaptive relevance cutoff (no fixed threshold) -------------------------


def adaptive_cutoff(scores: Sequence[float], k: float = 0.0) -> float:
    """Return the keep-**floor**: candidates with ``score >= floor`` are kept.

    The floor is read off the *shape* of the sorted-descending scores. Compute
    the gaps between consecutive scores; the largest gap marks a real "elbow"
    only when it **dominates** the others -- it is at least twice the
    next-largest gap. That dominance ratio is **scale-free** (a cliff that is
    twice as deep as any other drop), so cosine (``0..1``) and integer
    overlap-counts both work without normalization, and it is **FP-robust**: an
    even gradient has near-equal gaps (ratio ~1, never >= 2) so nothing is cut --
    fixing the float-dust mis-cut a variance z-score suffered. It also fires
    correctly at exactly three candidates (two gaps), where a mean+std test is
    mathematically inert. When no gap dominates, nothing is cut and the **token
    budget + dedup** bound the volume instead.

    This is deliberately *conservative* -- it discards only a clearly-detached
    weak tail, never a still-useful hit on a slope -- and has **no fixed
    relevance constant**, so a strong cluster sitting entirely below ``0.6``
    survives. Because :class:`Candidate` scores are ranked only *within one list*
    (the module invariant), the gaps share one scale.

    Degenerate cases: empty -> ``+inf`` (keep nothing); one or two scores -> keep
    them (a single gap has nothing to be judged a cliff against). When ``k > 0``
    the floor is additionally raised to ``mean + k*std`` over the scores (an
    opt-in statistical tightening)."""
    vals = sorted((float(s) for s in scores), reverse=True)
    n = len(vals)
    if n == 0:
        return math.inf
    if n <= 2:  # 0/1 gaps: nothing to call a "cliff" against -> keep all
        gap_floor = vals[-1]
    else:
        gaps = [vals[i] - vals[i + 1] for i in range(n - 1)]
        gi = max(range(len(gaps)), key=lambda i: gaps[i])
        largest = gaps[gi]
        second = max((g for j, g in enumerate(gaps) if j != gi), default=0.0)
        # Dominant cliff: the biggest drop is >= 2x any other drop (and positive).
        gap_floor = vals[gi] if largest > 0 and largest >= 2.0 * second else vals[-1]

    stat_floor = -math.inf
    if k > 0.0:
        mean_v = sum(vals) / n
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / n)
        stat_floor = mean_v + k * std_v

    return max(gap_floor, stat_floor)


# --- redundancy collapse ----------------------------------------------------


def _norm_for_dedupe(text: str) -> str:
    return " ".join(normalize_text(text).split())


def _near_duplicate(a: str, b: str, ratio: float) -> bool:
    na, nb = _norm_for_dedupe(a), _norm_for_dedupe(b)
    if not na or not nb:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= ratio


def collapse(cands: Sequence[Candidate], dedup_ratio: float = 0.0) -> list[Candidate]:
    """Drop redundant candidates so identical information is never injected twice.

    Two passes (both stdlib, O(n^2) over a budget-clamped pool of ~5-15 items):

    1. **Summary subsumes source** -- a ``message`` whose ``timestamp`` falls
       inside an admitted ``summary``'s ``span`` is dropped (the summary already
       carries it). Uses the ``start_time``/``end_time`` already stored on
       summary rows, so it needs no schema change.
    2. **Near-duplicate** -- of any two candidates within ``ratio`` SequenceMatcher
       similarity, only the higher-scored survives (iterate score-descending so
       the first kept is the strongest)."""
    eff_ratio = dedup_ratio if dedup_ratio and dedup_ratio > 0 else _DEFAULT_DEDUP_RATIO
    spans = [c.span for c in cands if c.kind == "summary" and c.span]

    def subsumed(c: Candidate) -> bool:
        if c.kind != "message" or not c.timestamp:
            return False
        return any(s <= c.timestamp <= e for (s, e) in spans if s and e)

    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        if subsumed(c):
            continue
        # Profile rows are distinct key:value facts -- only exact-collapse them
        # (fuzzy near-dup would wrongly merge e.g. "name: Bob" / "nickname: Bobby").
        if c.kind == "profile":
            if any(k.kind == "profile" and _norm_for_dedupe(c.text) == _norm_for_dedupe(k.text) for k in kept):
                continue
        elif any(k.kind != "profile" and _near_duplicate(c.text, k.text, eff_ratio) for k in kept):
            continue
        kept.append(c)
    return kept


# --- rendering --------------------------------------------------------------


VISION_LABEL = "Screen:"  # rendered prefix for a recalled visual (screen) memory


def _render_line(c: Candidate) -> str:
    if c.kind == "summary":
        return f"Summary: {c.text}"
    if c.kind == "profile":
        return f"- {c.text}"
    if c.kind == "vision":
        return f"{VISION_LABEL} {c.text}"
    if c.role == "assistant":
        return f"Assistant: {c.text}"
    if c.role == "user":
        return f"User: {c.text}"
    return c.text


def render(selected: Sequence[Candidate], header: str) -> str:
    """Canonical labelled block (most-relevant first), or ``''`` when empty."""
    if not selected:
        return ""
    ordered = sorted(selected, key=lambda c: c.score, reverse=True)
    return "\n".join([header, *(_render_line(c) for c in ordered)])


# --- value-density packing --------------------------------------------------


def pack(cands: Sequence[Candidate], budget: RecallBudget) -> tuple[list[Candidate], int, Optional[Candidate]]:
    """Greedy value-density knapsack under ``max_tokens`` (header counted).

    Candidates are ordered by **score per rendered token** (the most information
    per unit of context first) and admitted while they fit. Returns
    ``(selected, remaining_room, overflow)`` where ``overflow`` is the
    highest-density candidate that did *not* fully fit -- :func:`build_block`
    compresses it into ``remaining_room`` rather than dropping or hard-slicing
    it, so a single long high-value memory still contributes (truncated to whole
    sentences/words), not nothing.

    Budget accounting measures the **actual rendered block** incrementally (not a
    sum of per-line estimates), so newline/rounding drift can never push the
    final block over ``max_tokens``. ``remaining_room`` is the exact token
    headroom left after the whole items are placed."""
    cpt = budget.chars_per_token

    def density(c: Candidate):
        t = estimate_tokens(_render_line(c), cpt)
        return (c.score / max(t, 1), c.score, c.timestamp)

    lines = [budget.header]
    selected: list[Candidate] = []
    overflow: Optional[Candidate] = None
    for c in sorted(cands, key=density, reverse=True):
        line = _render_line(c)
        if estimate_tokens("\n".join([*lines, line]), cpt) <= budget.max_tokens:
            lines.append(line)
            selected.append(c)
        elif overflow is None:
            overflow = c
    room = budget.max_tokens - estimate_tokens("\n".join(lines), cpt)
    return selected, room, overflow


# --- query-aware compression ------------------------------------------------


def compress(text: str, query: str, token_room: int, cpt: float = 4.0) -> str:
    """Fit ``text`` into ``token_room`` tokens by selecting WHOLE sentences that
    best overlap the query -- never a mid-word cut.

    Picks the highest query-overlap sentences (ties -> earliest) until the room
    is full, then re-emits them in original order. Backstop for a single long
    sentence with no usable boundary (common with run-on STT): take whole words
    up to the room from the most-relevant sentence. Guarantees at least one whole
    word of content when ``token_room >= 1`` and ``text`` is non-empty."""
    text = text.strip()
    if token_room <= 0 or not text:
        return ""
    if estimate_tokens(text, cpt) <= token_room:
        return text

    sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()] or [text]
    qwords = set(normalize_text(query).split())
    scored = [
        (len(qwords & set(normalize_text(s).split())), -idx, idx, s)
        for idx, s in enumerate(sentences)
    ]
    scored.sort(reverse=True)  # highest overlap first; ties -> earliest (-idx)

    chosen: list[tuple[int, str]] = []
    used = 0
    for _, _, idx, s in scored:
        t = estimate_tokens(s, cpt)
        if used + t <= token_room:
            chosen.append((idx, s))
            used += t
    if chosen:
        chosen.sort()
        return " ".join(s for _, s in chosen)

    # Backstop: no whole sentence fits -> whole words from the best sentence.
    best = scored[0][3]
    out: list[str] = []
    for w in best.split():
        if estimate_tokens(" ".join([*out, w]), cpt) <= token_room:
            out.append(w)
        else:
            break
    if not out:  # room smaller than even the first word -> keep one word (min content)
        out = best.split()[:1]
    return " ".join(out)


# --- the entrypoint ---------------------------------------------------------


def build_block(
    cands: Sequence[Candidate], query: str, budget: RecallBudget, *, now: Optional[float] = None
) -> str:
    """Turn a backend's candidate list into the bounded recall block (or ``''``).

    Pipeline: multi-signal re-score (recency + importance, no-op by default) ->
    adaptive relevance cutoff -> redundancy collapse -> value-density pack ->
    compress the overflow item into the leftover room -> render. The returned block
    is guaranteed ``estimate_tokens(block) <= max_tokens`` except in the
    pathological case where a single indivisible word exceeds the whole budget
    (then the minimum viable content is emitted). ``now`` (epoch seconds) is only
    used for recency decay (injectable for deterministic tests); ``None`` ->
    ``time.time()`` when recency is on, ignored when it is off."""
    cands = list(cands)
    if not cands:
        return ""
    # Default-OFF + byte-identical when both signal weights are 0 (returns cands as-is).
    cands = _apply_signals(cands, budget, now)
    cpt = budget.chars_per_token

    floor = adaptive_cutoff([c.score for c in cands], budget.cutoff_k)
    survivors = [c for c in cands if c.score >= floor]
    if not survivors:  # cutoff degenerate-dropped everything -> keep the strongest
        survivors = sorted(cands, key=lambda c: c.score, reverse=True)[: max(1, budget.min_keep)]

    survivors = collapse(survivors, budget.dedup_ratio)
    selected, room, overflow = pack(survivors, budget)

    # The highest-value item that did not fully fit is COMPRESSED into the
    # leftover room (whole-sentence, query-aware) rather than dropped or hard-cut.
    # The compress target reserves both the joining newline AND the render label
    # ("User: "/"Summary: "/"- ") so the *rendered line*, not just its raw text,
    # fits -- then the trial block is exact-guarded against the budget. If even a
    # one-line trial cannot fit (the budget is too small to hold the header plus
    # any content), nothing is added and render() returns '' -- the token bound is
    # honored unconditionally rather than force-emitting an over-budget line.
    if overflow is not None and room > 0:
        label_overhead = max(
            estimate_tokens(_render_line(replace(overflow, text="x")), cpt)
            - estimate_tokens("x", cpt),
            0,
        )
        target = room - 1 - label_overhead
        if target >= 1:
            compressed = compress(overflow.text, query, target, cpt)
            if compressed:
                trial = [*selected, replace(overflow, text=compressed)]
                if estimate_tokens(render(trial, budget.header), cpt) <= budget.max_tokens:
                    selected = trial

    return render(selected, budget.header)


def trim_block_to_tokens(block: str, max_tokens: int, cpt: float = 4.0) -> str:
    """Whole-line secondary cap for an already-built block (never mid-word).

    A cheap, independent safety net at the injection layer: keep the header plus
    as many following lines as fit ``max_tokens``. Returns ``''`` if only the
    header would survive (a header with no items is noise, not context)."""
    if not block or max_tokens <= 0:
        return block
    if estimate_tokens(block, cpt) <= max_tokens:
        return block
    lines = block.split("\n")
    header, body = lines[0], lines[1:]
    kept = [header]
    used = estimate_tokens(header, cpt)
    for ln in body:
        t = estimate_tokens(ln, cpt)
        if used + t <= max_tokens:
            kept.append(ln)
            used += t
        else:
            break
    return "\n".join(kept) if len(kept) > 1 else ""
