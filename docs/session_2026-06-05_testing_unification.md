# Session 2026-06-05 (pt 2) — Unify the test suite into one marker taxonomy

**Goal:** the owner had testing scattered three ways (stage path-lists, mostly-
unused markers, ad-hoc `importorskip`), some live-on-hardware, some unit; wanted
them **unified + relevant + real-capability**, with real-speaker tests **opt-in,
not always**. Done across 7 commits on `main` (merge tip `e930b92`), full suite
**1350 passed, 14 skipped** at every phase.

## The diagnosis
92 files, ~1,174 tests, **zero** dead-architecture references (refactor stayed
clean). The problem was categorization: a rich marker vocab in `pytest.ini` was
**declared-but-unused** (`hardware`/`audio`/`llm`/`network` = 0 files; only ~12 of
85 carried any marker), `tools/testing/stages.py` selected by **hardcoded path
lists**, and CI gated by **`--ignore=<2 files>`** + the implicit "18 files
`importorskip` + self-skip". Three schemes, none authoritative.

## What landed — one taxonomy → four tiers (driven by markers)
- **Tier 0 unit/logic** = no tier marker (the default, CI gate).
- **Tier 1 integration/sim** = `slow`/`e2e`.
- **Tier 2 real_model** (NEW marker) = real weights over fixtures, no sound; self-
  skips without models. Applied to the 6 genuine real-weight tests (+2 new).
- **Tier 3 live_output** (NEW marker) = **real speakers/mic**, the only tier that
  makes sound; **double-gated** behind the marker + `SPEAKER_LIVE=1` (conftest
  skip mirrors `--postgres`).

Single entry point: extended `tools/run_tests.py` with `unit`/`real_model`/`live`
stages (`fast`→alias of `unit`; the brittle 6-term blocklist became the closed-set
`_NOT_TIERED`). `TestStage` gained `preflight`+`env`; the `live` stage preflights
`tools.live_session --check`, sets `SPEAKER_LIVE=1`, runs `-m live_output`.
Verified end-to-end (`run_tests.py live` played a sentence, 1 passed in 4.5s).

Cleanup: renamed 2 duplicate test-function names. New Tier-0 coverage:
`test_interrupt_race.py` (speech-epoch gate drops stale TTS after barge),
`test_failure_cascades.py` (LLM/capability/memory failure → no wedge, recovers),
`test_multimodal_e2e.py` (image bytes survive the routed/raced main-tier chain).
New real-output coverage: `test_asr_live_quality.py` (real_model: partial→final
ASR consistency, no second-pass hallucination), `test_live_output_smoke.py`
(live_output: real TTS → real speakers). CI: new `perf.yml` `real_model_tests`
job downloads sherpa models + runs `-m "real_model or recorded"`. Docs:
`docs/testing.md` rewritten; CLAUDE.md testing line updated.

Each new logic test was mutation-verified by the authoring agents (defeat the
invariant → the test goes red → restore).

## Commands now
`python tools/run_tests.py list|unit|fast|core|sandbox|memory|cloud|imports|e2e|real_model|live|full`
- `unit` (alias `fast`) = the CI-safe TDD loop.
- `real_model` = real weights over fixtures (no sound; needs models).
- `live` = REAL speakers, opt-in only (preflight + `SPEAKER_LIVE=1`).

## Next steps / findings (pick up here)
1. **Multimodal capability gap (real bug, filed to backlog P1).** The LLM clients
   forward `images=` (now tested), but `core/capabilities.py` never reads
   `context['images']` and nothing populates an image into the turn context — a
   "describe this image" turn silently drops the image. Fix: thread an image
   source → `turn context['images']` → `stream(..., images=...)` in the
   assistant/main-tier capability, then extend `test_multimodal_e2e.py` with a
   `registry.invoke('assistant.answer', …)` case.
2. **Optional marker tidy:** `smoke`/`dev`/`audio`/`backend` remain declared-but-
   unused (left intact; harmless). Apply or drop if you want a leaner `pytest.ini`.
3. **`real_model` in CI is download-heavy** (sherpa ASR+SenseVoice+Smart-Turn+DTLN
   via `setup_models`, cached). Watch the perf.yml `real_model_tests` job's first
   green run; trim the model set if minutes matter.
4. The earlier **voice-demo** work (pt 1) is a separate handoff:
   `docs/session_2026-06-05_voice_demo_live_validation.md` (5 fixes + the 16GB-GPU
   model finding); its open items (confabulation honesty gap, stub search
   backends) are still open.
