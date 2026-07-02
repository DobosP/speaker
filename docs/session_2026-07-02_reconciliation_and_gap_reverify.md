# Session 2026-07-02 — Reconciliation + fresh gap analysis (the SOTA review was stale)

**Headline:** The 2026-07-02 SOTA review (and a first pass at implementing it) were done
against a local checkout that was **125 commits behind `origin/main`**. Once synced,
most of the review's audio/voice recommendations turned out **already implemented**
upstream. This session: synced `main` to `origin/main` (`cc26f3d`→`c5213c0`), re-verified
every recommendation against the *current* code (10-agent fan-out), fixed a **red-CI test**
on main, and hardened Kokoro TTS loading. Branch for the code: `fix/asr-guard-test-and-kokoro-fallback`.

## What happened (so the next session doesn't repeat it)

- The Windows box's local `main` was stale (`cc26f3d`, 2026-06-16). `origin/main` had 125
  newer commits (heavy voice/audio work through 2026-06-30). **Always `git fetch` + check
  `git rev-list --count main..origin/main` at session start** — the bootstrap doesn't.
- The stale-based review recommended (and a first Batch-1 pass implemented) Kokoro TTS,
  loudness normalization, TTS-markup sanitization, a `_speaking` hold-down, and underrun
  counters — **all already on `origin/main`** (`feat/kokoro-tts-backend`, `streaming
  normalize_rms`, `harden TTS markup`, `flush stalled playback tail`, `voice quality
  diagnostics`). That work was discarded (backed up in the session scratchpad).
- `config.local.json` Batch-0 edits were reverted (they were based on the stale analysis
  and unvalidated).

## Verified status vs current `origin/main` (10-agent re-verify)

**DONE upstream** (do NOT re-implement): Kokoro TTS backend (`build_tts` keys Kokoro on
`tts_voices`), streaming per-sentence loudness normalization, TTS markup→voice/emotion +
markup stripping, playback-tail flush before ASR reopens, underrun diagnostics
(`_underrun_blocks` + `playback_underrun` metric), WebRTC-AEC3/APM barge-in, output
leveler, SenseVoice agreement guard, per-device FIFO depth.

**Still OPEN + HEADLESS** (genuine work, no hardware/owner needed):
- **R05 routing** — `HeuristicRouter.score()` gives a plain turn 0.0, so a threshold drop
  can't selectively route content turns to the resident (equal-TTFT) 12b; needs a
  resident-main / min-score lever in `core/routing.py`. Escalation is value-hollow while
  web search is off (planner's `web.search` resolves to a 5-entry corpus,
  `always_on_agent/capabilities.py:117`).
- **R06b prompt order** — recent block is now a suffix (good), but the volatile *recall*
  block still sits ahead of the static system prompt (`core/capabilities.py:446/478`) →
  busts Ollama prefix cache when recall is on. Backlog item `llm-inference-1`.
- **R11 structured history** — history is still rendered as text (`core/conversation.py:203/217`)
  pasted into the *system* string (6 turns/800 chars) instead of role-structured chat
  messages; `num_ctx` mostly unused. Needs an optional messages param on the LLMClient
  protocol (`core/llm.py`) threaded through Ollama/LlamaCpp/cloud.
- **R09 dead-air** — gate + cleanup still run as two serial fast-tier calls; no
  merge-into-one, no concurrency, no ack/filler, no speculative start.
- **R10 cleaner** — a deterministic proper-noun/number-preservation guard in
  `core/cleanup.py` is still open (the agreement guard covers the ASR-hallucination half,
  not the cleaner-rewrite half).
- **R14 ASR** — `build_final_recognizer` still only supports sense_voice/whisper; a
  Parakeet/`nemo_transducer` branch is a headless code+test add (model commit is gated).
- **Voice plan P2 (headless)** — `setup_models --kokoro` fetch; per-device
  `tts_output_lowpass_hz` defaults; profile-gate Kokoro-vs-Piper.
- **Backlog headless** — sha256 model-integrity verification; LiveKit soxr resampling;
  tier-aware load telemetry; dead-knob default cleanups.

**GATED (need owner / live-mic / infra / download):** enabling the memory stack
(deliberate "measure before flipping" decision) + authoring a persona; web search (needs a
self-hosted SearXNG); action tools; the streaming leveler and voice-set finalization
(live-mic/ear); Kokoro model download + listen; the SECURITY P0 (rotate the leaked Gemini
key + history purge — owner-only).

## What this session changed (the PR)

Two safe, high-confidence, fully-headless fixes on current `main`:

1. **Fixed a red-CI test.** `test_asr_final_async::test_finalize_dispatches_upgraded_final_and_backdated_speech_end`
   asserted the *old* "trust the 2nd pass on a long clip" behavior, but the hardened,
   fail-closed `agreement_guard` (commits `65bc852`/`5bc03cb`) now correctly keeps the
   streaming final when the 2nd pass has near-zero overlap (indistinguishable from a
   hallucination). The guard-hardening commits never updated this integration test, so it
   is **deterministically red on `origin/main`** (pure stdlib `difflib`, so CI too). Fixed
   the fixture to an *agreeing* pair (verifies the real punctuation/casing upgrade path) +
   added `test_finalize_keeps_streaming_when_second_pass_disagrees_on_long_clip` to lock
   the fail-closed contract at the finalize/dispatch level.
   > **Tradeoff to note for the owner:** the guard's 0.55 similarity threshold means a
   > *real* low-overlap correction on a long clip is also rejected (kept as streaming). If
   > live data shows good SenseVoice upgrades being dropped, tuning that threshold is a
   > separate, owner-validated decision — not touched here.
2. **Kokoro `build_tts` graceful fallback.** A Kokoro config (`tts_voices` set) whose
   package files are missing now returns `None` with a clear, actionable warning instead
   of the native loader hard-aborting on the capture thread; a general try/except around
   `OfflineTts(...)` fails open for any other build error. Mirrors
   `build_final_recognizer`'s fail-open contract. +4 tests (the Kokoro branch was
   previously untested — the fake `_ModelConfig` had no `.kokoro`).

## Next steps (pick up here)

1. **Land the PR** `fix/asr-guard-test-and-kokoro-fallback` (restores green CI).
2. Highest-value open + headless thread for "doesn't feel smart": **R11 structured chat
   history + R06b prefix-cache-safe prompt order** (both engineering-only, no default
   flips). Changes the model-visible prompt, so do a live listen after.
3. Then the voice-plan P2 headless bundle (setup_models `--kokoro` fetch [code exists in
   the session scratchpad `batch1_backup/`], per-device HF roll-off, profile-gating).
4. Owner-gated, unchanged: SECURITY P0 (rotate leaked key), enable memory + persona (a
   product/latency decision), web search infra, voice-set finalization by ear.
