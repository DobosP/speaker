# Live validation — per-capability latency, denoise, search, barge-in (2026-06-01)

A second 2026-06-01 over-the-air session (after
`docs/live_validation_run_2026-06-01.md`) targeting **barge-in, denoise, web
search, and per-capability / per-tier latency** on the real machine
(AT2020USB-X mic → laptop ALC285 speaker, dev 5). New harness toggles this added:
`--denoise`, `--noise-snr <dB>` (controlled noise overlay on the synthetic user),
and the `capability_latency_profile` scenario + `capability` suite. Raw bundles
under `logs/live/` (gitignored); this is the analysis.

## TL;DR — where the next big improvements are

1. **Endpoint trailing-silence is the #1 latency lever, uniformly across EVERY
   tier.** Measured endpoint ≈ **870–1057 ms p50** for fast, main AND research
   turns — ~65–75 % of first-audio. The model tier barely matters (next point), so
   reclaiming the endpoint wait is the single highest-value latency win.
2. **The big model is essentially FREE in responsiveness on this GPU.**
   `gemma3:12b` TTFT (~0.17–0.23 s) ≈ `gemma3:4b` TTFT (~0.19 s) on the RTX 4090.
   The main tier's higher first-audio comes from longer first sentences (more
   TTS), not the LLM. → route to the big model freely; responsiveness is the same.
3. **Research with web disabled returns a useless stub** instead of the model's own
   knowledge — a real capability gap.
4. **Over-the-air barge-in on a single speaker self-interrupts AND hard-crashes
   (PortAudio SIGFPE).** It needs AEC or a second output device; a harness warning
   was added.
5. **GTCRN denoise is a wash** at moderate noise — it recovers some garbled turns
   but over-suppresses others. Keep it default-OFF; it needs tuning.

## 1. Per-capability / per-tier latency (`capability_latency_profile`, real 2-tier)

Run with the REAL split (`--model gemma3:12b --fast-model gemma3:4b`) over the air.
The earlier runs forced both tiers to 4b, so this is the first measurement of the
main/research tiers. Ground truth from the ollama logs (which name the model per
call):

| turn (intent) | routed model | TTFT | total gen | first_audio | endpoint |
|---|---|---|---|---|---|
| capital of Japan (FAST) | gemma3:4b | 0.19 s | 0.22 s | 1271 ms | 934 |
| seven times eight (FAST) | gemma3:4b | 0.19 s | 0.22 s | 1341 ms | 982 |
| why leaves change / chlorophyll (MAIN) | gemma3:12b | 0.17 s | 1.17 s | ~1.5–1.9 s | ~870–980 |
| compare Mars & Earth (MAIN) | gemma3:12b | 0.21 s | — | 1664 ms | 980 |
| short story / lighthouse (MAIN long-form) | gemma3:12b | — | 2.25 s | 1220 ms | 982 |
| summarize exercise benefits (MAIN) | gemma3:12b | 0.23 s | 2.08 s | — | — |
| research WW1 causes (RESEARCH→search) | (corpus stub) | — | — | 1786 ms | 1057 |

**Reads:**
- **TTFT is tier-independent (~0.2 s) on the 4090.** The 12b answers stream their
  first token as fast as the 4b; they just generate *more* (1.2–2.3 s total vs
  0.2 s). For a streaming-TTS assistant, that means the big model's first audio is
  ~as fast — the extra tokens land *after* speech has already begun.
- **Endpoint dominates every tier** (~900–1050 ms). The fast/main first-audio gap
  is driven by the first sentence's length (TTS), not the LLM.
- The two-tier router fired correctly: short literals → 4b; `why`/`explain`/
  `compare`/`in detail` and `tell me a story` → 12b; `research …` → the search
  capability. (`tests/test_live_session.py` pins this with the router scores.)

## 2. Web search / research — gap found

"Research the three main causes of the First World War" routed correctly to
`mode=search capability=web.search`, but with `web_search.enabled=false` (and no
SearXNG running, port 8888 dead) it fell through to the local corpus, missed, and
spoke **"No local result for: the three main causes of the first world war"** — it
did NOT fall back to the LLM's own knowledge. **Improvement:** when web + corpus
return nothing, a research/search turn should answer from the main model instead
of a dead-end stub. (Real web-search latency needs a SearXNG instance — keyless,
`docker run -d -p 8888:8080 searxng/searxng`, then `web_search.enabled=true`.)

Also surfaced: **STT-garble → addressing-gate ingest-drop.** "Name the largest
ocean" was heard as "In the largest ocean" (Name→In); no longer a question, the
addressing gate marked it INGEST and never answered — same fragility class as the
proper-noun confabulation cascade in the first 2026-06-01 doc.

## 3. Denoise A/B (GTCRN, controlled noise overlay at SNR 8 dB)

Same `latency_profile_mixed` (11 turns) played over the air with `--noise-snr 8`,
once without and once with `--denoise`:

| | STT median | heard-ok | response-ok |
|---|---|---|---|
| denoise OFF | 0.946 | 9/11 | 6/11 |
| denoise ON | 0.904 | 9/11 | **8/11** |

**Mixed, not a clear win.** Denoise ON *recovered* some badly-garbled turns
(noise turned "capital"→"castle" OFF, back to "capital" ON; a fully-missed turn
became heard) — which lifted response-ok 6→8 — but it *over-suppressed* others
("square root"→"square leaf", "spell the word Mars"→"These fell the word marsh"),
lowering the STT median. Consistent with the handoff note that GTCRN over-
suppresses speech. **Keep default OFF; it needs tuning/validation before it earns
its place.** Caveat: each run captures a fresh over-the-air pass, so some of the
per-turn delta is acoustic run-to-run variance, not denoise.

## 4. Barge-in over the air — self-interrupt + a CRASH (critical)

`barge_no_barge_control` (a long answer, NO scripted barge) played over the open
speaker with `--barge-in` (so `barge_in_enabled=true`, level-margin gate):

- **The assistant self-interrupted on its OWN TTS.** The log shows `barge-in
  detected` → `ASSISTANT [interrupted]` mid-answer — the 6 dB level-margin gate
  (`barge_in_output_margin_db`) cannot distinguish the assistant's own voice
  leaking into a co-located mic from a real barge. Confirms: **open-speaker
  barge-in without AEC self-interrupts.**
- **Then it hard-crashed: PortAudio SIGFPE (core dumped), EXIT 136.** Reproduced
  with `faulthandler`: the crash is in `sd.play → OutputStream.__init__ →
  start_stream` (`synthetic_user.say`) — the synthetic user opening a playback
  stream **while the assistant's just-barged output stream still holds the
  exclusive-ALSA device 5**. A native FPE in PortAudio's stream-open; the Python
  `Device unavailable` retry can't catch a SIGFPE. **Not a regression** from this
  session's changes (the run used no `--noise-snr`, so the new overlay is a no-op).

**Improvement / mitigation:** over-the-air barge-in on a single exclusive-ALSA
output device is unsafe — it both self-interrupts and can crash. Real acoustic
barge-in needs **AEC** or a **second output device** for the interrupter. The
most promising no-AEC path remains **speaker-gated barge-in** (the assistant's TTS
embeds ~0.01 vs the enrolled user ~0.5+, so a speaker-gated barge fires only on
the user) — but it needs a live-user test, not the synthetic user. A harness
**warning guard** was added for the `acoustic + --barge-in` combination.

## Prioritized next big improvements

1. **Cut the endpoint wait** (uniform ~900 ms, ~70 % of latency, tier-independent).
   The smart-endpoint semantic-completion path is the lever; tune `min_silence` /
   add a confidence-based early commit. Biggest single win.
2. **ASR robustness to named entities / keywords.** Proper-noun garble drives
   confabulation, coreference cascades, and ingest-drops; noise worsens it. Add
   ASR hotword/context biasing and an "unknown/surprising entity or low-confidence
   → confirm" guard so a garbled entity doesn't silently produce a wrong answer.
3. **Research fallback to model knowledge** when web + corpus are empty (no more
   "No local result" dead-ends).
4. **Barge-in: AEC or speaker-gating.** Open-speaker barge-in is unusable (self-
   interrupts + crashes). Pursue speaker-gated barge-in (live-user test) or AEC.
5. **Addressing-gate / first-turn robustness** so STT-garbled questions and openers
   aren't silently ingested.

## Harness additions this session

- `--denoise` (toggle `sherpa.denoise_enabled`) + `--noise-snr <dB>` (deterministic
  broadband noise overlaid on the synthetic-user PLAYBACK only — the saved clip
  stays clean) for the denoise A/B.
- `capability_latency_profile` scenario (fast/main/research tier turns) + the
  `capability` suite; added to the `latency` suite.
- A warning guard for the fragile `acoustic + --barge-in` combination.
- New unit tests incl. a router-scoring test that proves the capability scenario
  actually exercises both tiers (95 live-session tests; full suite green).
