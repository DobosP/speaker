"""Tier-0 unit tests for the shared recall selector (``always_on_agent/recall.py``).

No DB, no embeddings, no models -- everything runs over hand-built
:class:`Candidate` lists, so this is the fast logic tier and pins the
context-efficiency contract: a token budget (not char caps), a data-derived
adaptive cutoff (no fixed 0.6/top-3), redundancy collapse, and whole-sentence
query-aware compression that never cuts mid-word.
"""
from __future__ import annotations

import math
import random

from always_on_agent.recall import (
    Candidate,
    RecallBudget,
    adaptive_cutoff,
    build_block,
    collapse,
    compress,
    estimate_tokens,
    pack,
    render,
    trim_block_to_tokens,
)


# --- estimate_tokens --------------------------------------------------------


def test_estimate_tokens_zero_for_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0  # whitespace-only counts as empty


def test_estimate_tokens_at_least_word_count_and_char_bound():
    txt = "alpha beta gamma delta"
    assert estimate_tokens(txt) >= len(txt.split())
    assert estimate_tokens(txt) >= math.ceil(len(txt) / 4.0)


def test_estimate_tokens_monotonic_in_length():
    short = "one two three"
    long = short + " four five six seven eight nine ten"
    assert estimate_tokens(long) > estimate_tokens(short)


def test_estimate_tokens_counts_long_token_by_chars():
    # A single very long token must not be under-counted as 1 token.
    assert estimate_tokens("a" * 40) >= 10


# --- adaptive_cutoff (no fixed threshold) -----------------------------------


def test_adaptive_cutoff_empty_keeps_nothing():
    assert adaptive_cutoff([]) == math.inf


def test_adaptive_cutoff_single_keeps_it():
    assert adaptive_cutoff([0.42]) == 0.42


def test_adaptive_cutoff_two_clusters_keeps_high_cluster():
    scores = [10.0, 9.0, 8.0, 2.0, 1.0]
    floor = adaptive_cutoff(scores)
    kept = [s for s in scores if s >= floor]
    assert kept == [10.0, 9.0, 8.0]


def test_adaptive_cutoff_keeps_strong_survivors_below_0_6():
    # The old code's fixed `similarity > 0.6` would drop ALL of these; the
    # relative-gap cutoff keeps the strong relative cluster. Proves no baked 0.6.
    scores = [0.59, 0.55, 0.20, 0.18]
    floor = adaptive_cutoff(scores)
    kept = sorted((s for s in scores if s >= floor), reverse=True)
    assert kept == [0.59, 0.55]


def test_adaptive_cutoff_flat_distribution_keeps_all():
    scores = [0.5, 0.5, 0.5, 0.5]
    floor = adaptive_cutoff(scores)
    assert all(s >= floor for s in scores)


def test_adaptive_cutoff_k_tightens():
    scores = [0.90, 0.88, 0.86, 0.84]  # smooth, equal gaps -> base keeps all
    base = adaptive_cutoff(scores, k=0.0)
    assert all(s >= base for s in scores)
    # A positive k raises the floor (mean + k*std over the scores).
    assert adaptive_cutoff(scores, k=1.5) > base


# --- collapse ---------------------------------------------------------------


def test_collapse_drops_near_duplicates_keeping_higher_score():
    a = Candidate("my name is Grace and I live in Oxford", 0.9)
    b = Candidate("my name is grace and i live in oxford", 0.4)  # near-identical
    kept = collapse([a, b])
    assert len(kept) == 1 and kept[0].score == 0.9


def test_collapse_keeps_distinct():
    a = Candidate("I like dark roast coffee", 0.9)
    b = Candidate("the capital of france is paris", 0.8)
    assert len(collapse([a, b])) == 2


def test_collapse_summary_subsumes_spanned_messages():
    summ = Candidate("we discussed travel plans", 0.7, kind="summary", span=(100.0, 200.0))
    inside = Candidate("I want to visit Japan", 0.9, kind="message", timestamp=150.0)
    outside = Candidate("unrelated later thought", 0.8, kind="message", timestamp=300.0)
    kept = collapse([summ, inside, outside])
    texts = {c.text for c in kept}
    assert "I want to visit Japan" not in texts  # subsumed by the summary's span
    assert "unrelated later thought" in texts
    assert "we discussed travel plans" in texts


def test_collapse_custom_ratio_is_respected():
    a = Candidate("the quick brown fox", 0.9)
    b = Candidate("the quick brown dog", 0.4)  # ~75% similar
    # default 0.92 ratio keeps both; a loose 0.5 ratio collapses them.
    assert len(collapse([a, b])) == 2
    assert len(collapse([a, b], dedup_ratio=0.5)) == 1


# --- pack (value-density knapsack) ------------------------------------------


def test_pack_respects_token_budget():
    cands = [Candidate(f"fact number {i} " + "word " * 10, score=1.0 / (i + 1)) for i in range(20)]
    budget = RecallBudget(max_tokens=60)
    selected, room, _ = pack(cands, budget)
    block = render(selected, budget.header)
    assert estimate_tokens(block) <= budget.max_tokens
    assert room >= 0


def test_pack_prefers_higher_value_density():
    cheap_high = Candidate("teal", score=0.9)  # tiny + high score -> best density
    expensive_low = Candidate("word " * 40, score=0.95)
    budget = RecallBudget(max_tokens=20)
    selected, _, _ = pack([expensive_low, cheap_high], budget)
    assert cheap_high in selected


def test_pack_overflow_is_the_unfit_high_density_item():
    big = Candidate("word " * 100, score=5.0)
    budget = RecallBudget(max_tokens=15)
    selected, room, overflow = pack([big], budget)
    assert selected == [] and overflow is big and room > 0


# --- compress ---------------------------------------------------------------


def test_compress_returns_text_when_it_fits():
    assert compress("short fact", "anything", 50) == "short fact"


def test_compress_never_cuts_mid_word():
    text = "alphaword betaword gammaword deltaword epsilonword zetaword etaword"
    out = compress(text, "alphaword betaword", token_room=4)
    assert out  # non-empty
    for token in out.split():
        assert token in text.split()  # every emitted token is a whole original word


def test_compress_prefers_query_overlapping_sentence():
    text = "I love hiking in the mountains. My favorite color is teal. Coffee is great."
    out = compress(text, "what is my favorite color", token_room=8)
    assert "teal" in out


def test_compress_within_room_for_normal_text():
    text = "one two three. four five six. seven eight nine. ten eleven twelve."
    out = compress(text, "seven eight", token_room=6)
    assert estimate_tokens(out) <= 6


def test_compress_empty_room_returns_empty():
    assert compress("anything", "q", token_room=0) == ""


# --- build_block (end-to-end) -----------------------------------------------


def test_build_block_empty_candidates_is_empty_string():
    assert build_block([], "q", RecallBudget()) == ""


def test_build_block_is_token_bounded_property():
    rng = random.Random(1234)
    for _ in range(200):
        n = rng.randint(1, 12)
        cands = [
            Candidate(
                text=" ".join(["w" + str(rng.randint(0, 99)) for _ in range(rng.randint(1, 15))]),
                score=rng.random(),
                kind=rng.choice(["message", "message", "summary"]),
                role=rng.choice([None, "user", "assistant"]),
            )
            for _ in range(n)
        ]
        budget = RecallBudget(max_tokens=rng.randint(40, 120))
        block = build_block(cands, "w1 w2 w3", budget)
        assert estimate_tokens(block) <= budget.max_tokens


def test_build_block_min_keep_compresses_when_single_item_too_big():
    big = Candidate("word " * 200, score=1.0, role="user")
    budget = RecallBudget(max_tokens=30)
    block = build_block([big], "word", budget)
    assert block  # something is kept (min_keep=1)
    assert estimate_tokens(block) <= budget.max_tokens
    assert block.startswith(budget.header)


def test_build_block_keeps_strong_hit_below_legacy_threshold():
    # Two hits, both below the old 0.6 cosine floor; the stronger must survive.
    cands = [
        Candidate("my favorite color is teal", 0.59, role="user"),
        Candidate("totally unrelated chatter about weather", 0.12, role="user"),
    ]
    block = build_block(cands, "what is my favorite color", RecallBudget(max_tokens=120))
    assert "teal" in block


def test_build_block_canonical_labels():
    cands = [
        Candidate("hello there", 0.9, kind="message", role="user"),
        Candidate("general kenobi", 0.8, kind="message", role="assistant"),
        Candidate("we talked about star wars", 0.7, kind="summary"),
    ]
    block = build_block(cands, "star wars", RecallBudget(max_tokens=120))
    assert "User: hello there" in block
    assert "Assistant: general kenobi" in block
    assert "Summary: we talked about star wars" in block


def test_build_block_deduped_summary_and_source():
    summ = Candidate("discussed the trip to japan", 0.9, kind="summary", span=(10.0, 20.0))
    src = Candidate("discussed the trip to japan", 0.85, kind="message", timestamp=15.0)
    block = build_block([summ, src], "japan trip", RecallBudget(max_tokens=120))
    # Only one rendering of the fact (the summary), not both.
    assert block.count("discussed the trip to japan") == 1


# --- render + trim ----------------------------------------------------------


def test_render_empty_is_empty():
    assert render([], "=== H ===") == ""


def test_trim_block_to_tokens_whole_line():
    block = "=== Past Conversations ===\nUser: one two three\nUser: four five six\nUser: seven eight nine"
    trimmed = trim_block_to_tokens(block, max_tokens=12)
    assert estimate_tokens(trimmed) <= 12
    # Never a partial line: every retained line is one of the originals.
    for line in trimmed.split("\n"):
        assert line in block.split("\n")


def test_trim_block_header_only_returns_empty():
    block = "=== Past Conversations ===\nUser: " + "word " * 50
    # Budget big enough for the header but not one content line -> drop to ''.
    assert trim_block_to_tokens(block, max_tokens=estimate_tokens("=== Past Conversations ===")) == ""
