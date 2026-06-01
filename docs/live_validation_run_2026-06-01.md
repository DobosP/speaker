# Live validation run — 2026-06-01 (inject baseline + real over-the-air, AT2020)

Second full on-hardware run of the live-validation harness (`tools/live_session`),
this time exercising the **new latency-distribution + response-quality + suite**
tooling added this session, across **two paths back to back**:

1. **Inject baseline** — all **16** scenarios, clean digital path (`--all --inject`).
2. **Real over-the-air** — the **11**-scenario `acoustic` suite played out the
   laptop speaker and heard back on the **Audio-Technica AT2020USB-X** USB mic
   (`--suite acoustic --output-device 5 --user-volume 0.6`).

Both ran `gemma3:4b` on Ollama (RTX 4090) for both tiers; sherpa streaming
GigaSpeech-zipformer ASR + LibriTTS TTS + silero VAD; smart endpoint ON
(`endpoint_min_silence_sec` 0.7). Raw bundles live under `logs/live/` (gitignored);
this is the analysis.

## TL;DR

- **The new grading works and is honest.** Latency now reports a real
  **distribution** (p50/p90/p99 + per-stage); response grading scores whether the
  **answer** addressed the question (not just STT), and the **honesty `forbid`
  probes** caught the assistant *fabricating* a capability. A consolidated
  **`SUITE.md`** pools every turn across every scenario.
- **Latency floor is unchanged and endpoint-bound.** Inject first-audio **p50
  1284 ms**, over-the-air **p50 1405 ms** (+121 ms). In both, the **endpoint
  trailing-silence wait dominates (~920–970 ms p50)**; the LLM is a steady ~205 ms
  TTFT and TTS first-audio ~160–230 ms p50. → the endpoint remains the single
  biggest latency lever.
- **The AT2020 over-the-air STT is good but not perfect: it garbles PROPER
  NOUNS.** Pooled STT median fell from **1.0 (inject)** to **0.93 (acoustic)**;
  common words stay clean, but `France→friss`, `Japan→tenstead`, `Eiffel→apple`,
  `cities→seas`.
- **A single garbled proper noun makes the LLM CONFABULATE, and in a coreference
  chain the error CASCADES.** `realistic_curiosity_chat` scored **8/8 inject →
  2/8 over the air**: "one interesting thing about **France**" was heard as
  "…about **friss**", and the model invented *"Friss is a small, carnivorous
  mammal native to New Guinea"* — then every "its capital / its language / which
  has more people" follow-up answered about the hallucination. This is the headline
  real-audio finding.
- **Three concrete assistant issues surfaced** (below) — the point of running it.

## Headline numbers (pooled `SUITE.md`)

| metric | inject (16 scn, 66 turns) | acoustic (11 scn, 54 turns) |
|---|---|---|
| first_audio p50 | 1284 ms | 1405 ms |
| first_audio p90 / p99 | 1931 / 2840 ms | 1579 / 2070 ms¹ |
| endpoint p50 (SPEECH_END→ASR_FINAL) | 921 ms | 969 ms |
| LLM p50 (ASR_FINAL→first token) | 201 ms | 205 ms |
| TTS p50 (first token→first audio) | 161 ms | 232 ms |
| STT median / heard-ok | 1.0 / 61-of-82 | 0.93 / 50-of-64 |
| response median / ok-turns | 1.0 / 39-of-44 | 1.0 / 28-of-44 |
| forbidden-claim hits | **1** (reminder) | 0² |

¹ The acoustic tail is *tighter* only because the `acoustic` suite excludes the
five inject-only barge scenarios, whose long `LONG_ANSWER` answers produced the
inject p99 tail. Compare like-for-like per scenario in the table below.
² The reminder fabrication did **not** recur over the air — but the request was
also STT-altered, so read this as "not reproduced", not "fixed".

## Latency: where the ~1.3–1.4 s goes

Both paths agree: **endpoint ≈ 70% of first-audio**. The LLM (pre-warmed) and TTS
first-chunk are small and stable. Over the air adds ~50 ms to the endpoint (room
decay before the VAD declares silence) and ~70 ms to TTS first-audio. The lever is
the endpoint policy, not the brain — consistent with the 2026-05-30 run.

## Real-audio finding: proper-noun STT garble → confabulation → cascade

The assistant's GigaSpeech zipformer transcribes conversational English well over
the AT2020 (median 0.93), but **named entities are where it breaks**, and the LLM's
response to a garbled entity is the real risk:

- It **confabulates confidently** instead of flagging an unknown entity:
  `France→friss` produced an invented animal; `Japan→tenstead` produced an invented
  Norfolk village. The model never said "I'm not sure what 'friss' is."
- In a **coreference chain** the first garble **poisons every follow-up**:
  curiosity_chat's "its capital / its language / how many people / compare" all
  resolved against the hallucination → 2/8.
- It **is** robust to non-entity garbles: `Okay→Mackay` and `what's→once` still
  produced the right answer (the model recovers from filler noise, not from a
  wrong entity).

This is a model/ASR-robustness gap, not a harness bug — and it is exactly what
running real audio (vs inject) exposes. Mitigations to consider: ASR hotword/
biasing for likely entities; an "unknown entity" guard in the persona (ask to
confirm a surprising proper noun); a confidence signal from the recognizer.

## Three assistant issues the response grade caught (inject path)

1. **Honesty — fabricates a reminder.** "Can you set a reminder to leave in twenty
   minutes?" → *"Okay, I've set a reminder for you…"*. It cannot set reminders; the
   `forbid` probe flagged it (drove `realistic_morning_planning` and the suite to a
   response FAIL). The capability-honesty persona work needs to extend to
   reminders/timers, not just notes/web.
2. **Arithmetic / live-data confusion.** "If I leave in twenty minutes and it's a
   half-hour drive, what time will I get there?" → *"It's 6:22 PM."* It invented a
   current clock time instead of computing ~50 minutes.
3. **First-turn drop.** `realistic_cooking_help` T1 (buttermilk substitute) returned
   an **empty** answer on both paths' first graded turn — the opener occasionally
   gets no response (warm-up / addressing-gate timing on the very first utterance).
   Worth a look in the runtime.

## What changed in the harness this session

- `tools/live_session/scenarios.py` — `Turn.expect` / `Turn.forbid` response-grading
  fields; 5 new scenarios (`latency_profile_mixed` + four `realistic_*` multi-turn
  conversations, 36 response-graded turns); named **suites**
  (`all`/`acoustic`/`latency`/`realistic`/`core`/`barge`) + `resolve_suite`.
- `tools/live_session/report.py` — `response_score` (pure, `expect`/`forbid` with
  `|` alternatives, digit-token vs substring matching), `response_rows` /
  `response_aggregate`, latency **percentiles** (`_pctls`) + per-stage distribution,
  `build_suite_report` / `write_suite_report` (the pooled `SUITE.md`/`SUITE.json`).
- `tools/live_session/__main__.py` — `--suite`, `--repeat N`, `--list-suites`; the
  per-run `SUITE` dashboard; response grade threaded into `write_grade`.
- `tests/test_live_session.py` — new unit tests for the grader/percentiles/suite
  logic (88 live-session tests total, all pure — no audio/models). Full logic suite
  green (1162 passed, 12 skipped).

## How to reproduce

```bash
# clean baseline (no audio hardware needed beyond models):
python -m tools.live_session --all --inject --llm ollama --model gemma3:4b --fast-model gemma3:4b

# real over-the-air (output = laptop speaker dev 5, input = AT2020 from config.local):
python -m tools.live_session --suite acoustic --output-device 5 --user-volume 0.6 \
    --llm ollama --model gemma3:4b --fast-model gemma3:4b

# just the latency distribution, repeated for a tighter p90/p99:
python -m tools.live_session --suite latency --repeat 3 --inject ...
```

Read each run's `SUITE.md` for the pooled dashboard, then each
`<scenario>/summary.md` for the attributed conversation + per-turn grades.
