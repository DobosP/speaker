> **COMPLETED (2026-07-02):** shipped — `endpoint_high_confidence_floor: 0.6` is
> the committed default in `config.json`; rationale + validation absorbed into
> `docs/unified_architecture.md` §13 and the `config.json`
> `_endpoint_high_confidence_comment`. Historical record.

# Cutting the endpoint wait — adaptive confidence-tiered floor (2026-06-01)

Tackling the #1 latency lever identified in `docs/live_validation_capabilities_2026-06-01.md`:
the endpoint trailing-silence wait (SPEECH_END → ASR_FINAL) is ~900ms and dominates
first-audio latency (~70%) uniformly across every LLM tier. This shipped a validated
~110ms cut. A multi-agent workflow (understand → design → judge) scoped the approach; the
value was nailed down by a live on-device A/B.

## The change

The smart endpoint's SHORTEN path commits a complete-reading turn once trailing silence
reaches `endpoint_min_silence_sec` (0.7s) — a uniform floor that exists to clear BOTH the
decoder lookahead (or the last word clips) AND a typical intra-sentence comma pause (or a
run-on splits). The lexical detector emits exactly three completion scores: **0.75** (a
normal ending word — the common, well-formed case), **0.1** (ends on a conjunction/article/
filler → never shortens), **0.4** (very short utterance).

**Adaptive confidence-tiered floor:** let only the **high-confidence (0.75) bin** commit at
a *lower* floor (`endpoint_high_confidence_floor`, 0.6s), while medium-confidence completes
keep the full 0.7s. The 0.75 bin by construction excludes the "…and"/"…the" run-on words, so
it's the safe bin to shorten. Default `0.0` = OFF = byte-identical to before.

```
# core/endpointing.py  AdaptiveEndpointPolicy.decide()
floor = c.min_silence_sec
if c.high_confidence_floor > 0.0 and completion_score >= c.high_confidence_score:
    floor = c.high_confidence_floor
if completion_score >= c.complete_threshold and silence_sec >= floor:
    return True
```

Files: `core/endpointing.py` (EndpointConfig fields + `from_sherpa` + `decide`),
`core/engines/sherpa.py` (SherpaConfig `endpoint_high_confidence_floor` / `_score`),
`config.json` (shipped at 0.6), `tests/test_endpointing.py` (2 pure tests),
`tools/live_session/__main__.py` (`--endpoint-high-confidence-floor` / `--endpoint-min-silence`
for live A/B). Why NOT "partial-stability early-commit": the capture loop's `silence_sec` is
already *time since the partial last advanced* (sherpa.py:919-922) — i.e. partial-stability —
so a second stability path is redundant and pushes commits below the lookahead (clipping).

## Live A/B (inject mode, 4 scenarios incl. the comma-pause canary, n≈23 turns/run)

Inject mode is deterministic except for mild ASR run-to-run jitter, so a **baseline-vs-baseline
control** quantifies that jitter; a candidate floor is safe only if its finals differ from
baseline no more than the control does.

| run | endpoint p50 | first_audio p50 | finals text-diffs vs base1 | "In one sentence, …" canary |
|---|---|---|---|---|
| baseline 0.7 (×2) | 918 / 920 ms | 1235 ms | — / **6 (variance floor)** | full sentence |
| **floor 0.60 (shipped)** | **806 ms (−112)** | **1134 ms (−101)** | **6 — equals variance, no extra splits** | **full ✓** |
| floor 0.55 (rejected) | 808 ms | 1136 ms | 13 — 2× variance | **truncated to "In one sentence"** ✗ |

**Verdict:** floor **0.60** banks the full ~110ms endpoint / ~100ms first-audio cut (the
0.55 win, without its cost) at a finals-diff rate equal to the inject variance floor — no
extra splits or truncations, and the multi-clause comma run-on stays intact. **0.55 truncated
it** (the workflow's proposed value; the live A/B is exactly why we validate before trusting a
number). ~8% off first-audio, uniformly across all tiers, deterministic, no new model, default-
OFF safe, and any premature commit is merged back by the continuation layer.

## What's left on the endpoint lever

0.6 is the safe lexical floor. The next step DOWN (toward ~0.3-0.4s, another ~250-400ms) needs
a **prosody / audio turn-completion model** (Smart Turn v3 style) through the existing
`TurnCompletionDetector` `needs_audio` seam — prosody catches the mid-phrase-trailing the lexical
detector can't, letting the floor drop without splitting. That's the phase-2 big-win, gated on
sourcing a licensed model + a real-audio validation campaign.
