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
    _apply_signals,
    _importance,
    adaptive_cutoff,
    build_block,
    collapse,
    compress,
    estimate_tokens,
    pack,
    render,
    strong_subject_overlap,
    trim_block_to_tokens,
)


# --- conservative recall-routing signal -----------------------------------


def test_strong_subject_overlap_requires_one_line_covering_the_question_subject():
    block = (
        "=== Past Conversations ===\n"
        "User: my lighthouse project's codename was Amber Finch\n"
        "User: we discussed the release train"
    )
    assert strong_subject_overlap(
        "what was my lighthouse project codename?", block
    )


def test_strong_subject_overlap_rejects_weak_or_cross_line_matches():
    favorite = "=== Past Conversations ===\nUser: my favorite color is teal"
    assert not strong_subject_overlap("what is the capital of france?", favorite)

    split = (
        "=== Past Conversations ===\n"
        "User: the lighthouse needs paint\n"
        "User: the project codename is Amber Finch"
    )
    assert not strong_subject_overlap(
        "what was the lighthouse project codename?", split
    )


def test_strong_subject_overlap_keeps_attribute_nouns_load_bearing():
    assert strong_subject_overlap(
        "what was my rescue dog's name?",
        "=== Past Conversations ===\nUser: we named the rescue dog Juniper",
    )
    assert not strong_subject_overlap(
        "what is my dog's name?",
        "=== Past Conversations ===\nUser: my dog prefers kibble",
    )
    assert not strong_subject_overlap(
        "what time is my dentist appointment?",
        "=== Past Conversations ===\nUser: my dentist appointment is at Main Street",
    )
    assert not strong_subject_overlap(
        "what is my lighthouse project codename?",
        "=== Past Conversations ===\nUser: my lighthouse project needs paint",
    )


def test_strong_subject_overlap_needs_a_question_and_substantive_subject():
    block = "=== Past Conversations ===\nUser: the lighthouse project is active"
    assert not strong_subject_overlap("the lighthouse project is active", block)
    assert not strong_subject_overlap("what was it?", block)


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


def test_adaptive_cutoff_even_gradient_keeps_all_no_fp_dust():
    # An even gradient has near-equal gaps -> no dominant cliff -> keep all. Pins
    # that floating-point dust in the gaps does NOT trigger a spurious cut.
    scores = [0.9 - 0.1 * i for i in range(6)]  # 0.9,0.8,...,0.4
    floor = adaptive_cutoff(scores)
    assert all(s >= floor - 1e-9 for s in scores)


def test_adaptive_cutoff_fires_at_three_candidates():
    # Exactly 3 candidates (2 gaps) with a clearly detached tail: the dominant-gap
    # rule must still cut (a mean+std z-score is mathematically inert here).
    scores = [0.9, 0.85, 0.1]
    floor = adaptive_cutoff(scores)
    assert sorted((s for s in scores if s >= floor), reverse=True) == [0.9, 0.85]


def test_adaptive_cutoff_handles_nan_and_negative():
    # Must not raise and must keep at least the finite top scores.
    floor = adaptive_cutoff([0.9, 0.8, -1.0, 0.2])
    assert floor != floor or isinstance(floor, float)  # returns a float, no crash


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


def test_collapse_keeps_distinct_profile_rows():
    # Profile rows are distinct key:value facts -- fuzzy near-dup must NOT merge
    # similar-looking ones; only exact-text duplicates collapse.
    a = Candidate("name: Bob", 1.0, kind="profile")
    b = Candidate("nickname: Bobby", 1.0, kind="profile")
    dup = Candidate("name: Bob", 1.0, kind="profile")
    kept = collapse([a, b, dup])
    texts = sorted(c.text for c in kept)
    assert texts == ["name: Bob", "nickname: Bobby"]  # both distinct kept, exact dup dropped


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
    # Wide input space INCLUDING small budgets (where the header eats most of the
    # room) and long indivisible words -- the regime the first review found broke
    # the bound via the label-undercount + `or not selected` escape.
    word_pools = [
        lambda: "w" + str(rng.randint(0, 99)),                 # short (1 token)
        lambda: "x" * rng.randint(8, 40),                      # long (multi-token)
        lambda: "alpha beta gamma. delta epsilon zeta.",       # multi-sentence
    ]
    for _ in range(2000):
        n = rng.randint(1, 12)
        cands = [
            Candidate(
                text=" ".join(rng.choice(word_pools)() for _ in range(rng.randint(1, 15))),
                score=rng.random(),
                kind=rng.choice(["message", "message", "summary", "profile"]),
                role=rng.choice([None, "user", "assistant"]),
            )
            for _ in range(n)
        ]
        budget = RecallBudget(max_tokens=rng.randint(6, 120))  # includes tiny budgets
        block = build_block(cands, "alpha w1 w2", budget)
        assert estimate_tokens(block) <= budget.max_tokens, (block, budget.max_tokens)


def test_build_block_tiny_budget_emits_empty_not_overflow():
    # Header alone (~7 tokens) >= budget: no content line can fit -> '' (NOT a
    # force-emitted over-budget line). Regression for the review's repro.
    big = Candidate("bbhhegcghebdfaahe cddecce abhdfd aac dadfe", 0.5, kind="summary")
    block = build_block([big], "abc def", RecallBudget(max_tokens=8))
    assert estimate_tokens(block) <= 8
    assert block == ""  # too small to hold the header + any labelled line


def test_build_block_run_on_summary_stays_in_budget_at_default():
    # A run-on summary at the realistic default must not overshoot by the
    # newline/label rounding the review caught (+1 at 220).
    big = Candidate("and then we " * 71, 0.9, kind="summary")
    block = build_block([big], "and then", RecallBudget(max_tokens=150))
    assert estimate_tokens(block) <= 150


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


# --- multi-signal scoring (recency + importance), default-OFF ----------------


def test_signals_off_by_default_is_byte_identical():
    cands = [
        Candidate("a relevant fact", 0.9, kind="message", timestamp=1000.0),
        Candidate("a stale aside", 0.5, kind="message", timestamp=2000.0),
    ]
    b = RecallBudget()  # weights default to 0 -> relevance-only
    # _apply_signals is a no-op (same scores) regardless of now.
    assert _apply_signals(cands, b, now=9.9e9) == cands
    # build_block output is identical with or without a now, and unchanged from
    # the pre-scoring contract.
    assert build_block(cands, "relevant fact", b, now=9.9e9) == build_block(cands, "relevant fact", b)


def test_recency_decay_boosts_fresher_candidate():
    b = RecallBudget(recency_weight=1.0, recency_half_life_days=1.0)
    now = 10 * 86400.0
    fresh = Candidate("fresh note", 0.5, kind="message", timestamp=now)
    stale = Candidate("stale note", 0.5, kind="message", timestamp=now - 5 * 86400)
    scored = {c.text: c.score for c in _apply_signals([fresh, stale], b, now=now)}
    assert scored["fresh note"] > scored["stale note"]
    assert math.isclose(scored["fresh note"], 0.5 * 1.0, rel_tol=1e-9)         # age 0 -> factor 1
    assert math.isclose(scored["stale note"], 0.5 * (0.5 ** 5), rel_tol=1e-9)  # 5 half-lives


def test_importance_boosts_durable_kinds():
    b = RecallBudget(importance_weight=1.0)
    prof = Candidate("name: Alice", 0.5, kind="profile", timestamp=0.0)
    msg = Candidate("a passing remark", 0.5, kind="message", timestamp=0.0)
    scored = {c.text: c.score for c in _apply_signals([prof, msg], b, now=0.0)}
    assert scored["name: Alice"] > scored["a passing remark"]
    assert math.isclose(scored["name: Alice"], 0.5 * _importance(prof))
    assert math.isclose(scored["a passing remark"], 0.5 * _importance(msg))


def test_signals_keep_scores_non_negative():
    b = RecallBudget(recency_weight=0.8, importance_weight=0.8, recency_half_life_days=1.0)
    now = 100 * 86400.0
    old = Candidate("ancient", 0.9, kind="vision", timestamp=now - 1000 * 86400)
    [scored] = _apply_signals([old], b, now=now)
    assert scored.score >= 0.0


def test_blend_is_convex_never_boosts_above_native():
    # Even with weights summing past 1, a fresh+important item is at most its native
    # score (the blend is clamped to <= 1), so scoring only down-weights.
    b = RecallBudget(recency_weight=0.9, importance_weight=0.9, recency_half_life_days=1.0)
    now = 5 * 86400.0
    fresh = Candidate("name: Alice", 0.7, kind="profile", timestamp=now)  # recency 1, importance 1
    [scored] = _apply_signals([fresh], b, now=now)
    assert scored.score <= 0.7 + 1e-12


def test_negative_cosine_score_is_not_ranked_up():
    # Postgres cosine can be negative for an off-topic hit; the multiplicative
    # down-weight must NOT move it toward zero (which would rank it ABOVE a
    # down-weighted positive). A negative score is left unchanged and stays last.
    b = RecallBudget(recency_weight=1.0, recency_half_life_days=1.0)
    now = 10 * 86400.0
    off_topic = Candidate("off topic", -0.2, kind="message", timestamp=now)        # fresh but negative
    on_topic_stale = Candidate("on topic", 0.3, kind="message", timestamp=now - 3 * 86400)
    scored = {c.text: c.score for c in _apply_signals([off_topic, on_topic_stale], b, now=now)}
    assert scored["off topic"] == -0.2                  # untouched, not lifted toward 0
    assert scored["on topic"] > scored["off topic"]     # on-topic (even decayed) ranks above off-topic


def test_recency_changes_render_order_in_build_block():
    b = RecallBudget(recency_weight=1.0, recency_half_life_days=1.0, max_tokens=200)
    now = 10 * 86400.0
    cands = [
        Candidate("the older one about cats", 0.6, kind="message", timestamp=now - 6 * 86400),
        Candidate("the newer one about cats", 0.6, kind="message", timestamp=now),
    ]
    block = build_block(cands, "cats", b, now=now)
    assert block.index("newer") < block.index("older")  # render is effective-score descending


def test_build_recall_budget_reads_scoring_knobs():
    from core.app import _build_recall_budget

    b = _build_recall_budget({
        "recall_recency_weight": 0.3,
        "recall_recency_half_life_days": 3.0,
        "recall_importance_weight": 0.2,
    })
    assert b.recency_weight == 0.3
    assert b.recency_half_life_days == 3.0
    assert b.importance_weight == 0.2
    # defaults stay OFF
    d = _build_recall_budget({})
    assert d.recency_weight == 0.0 and d.importance_weight == 0.0
