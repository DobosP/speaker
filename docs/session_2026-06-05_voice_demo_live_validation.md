# Session 2026-06-05 — Live demo of barge-in + smart-routing (mock-user injection)

> **Status (2026-07-02):** immutable dated record — the "both tiers = gemma3:4b in `config.local.json`" setting below was a point-in-time workaround; model tiering is now per-machine (committed `config.json` = gemma3:12b/4b; e.g. gemma4:12b pinned in `config.local.json` on the 4090 box). Do not copy config claims from this doc.

**Goal:** A live, on-machine demo driving the REAL pipeline (sherpa ASR → Ollama
LLM → sherpa TTS) over a mock user, to test response speed, interrupts, multiple
questions, and the smart-routing "answering layer". Used `tools.live_session
--inject` (synthetic user fed straight into the recognizer; the reliable path on
the Windows box). It surfaced 5 real bugs, all fixed + landed.

**Branch/commit map** (all on `main`, tip after merge):
- `8c908b5` fix(routing): stop general-knowledge Qs escalating to the stub tool planner
- `f257151` fix(live_session): faithful + reliable inject-mode demo on a 16GB GPU
- `24aee08` merge of the above; `d809c4a` merge of remote logs chore. Suite **1335 green**.

## What landed
1. **Routing "Wyoming" bug (the answering-layer issue).** `always_on_agent/react.py`:
   "explain X **step by step**" was escalating to the ReAct tool planner, hitting
   the 3-entry stub corpus (`always_on_agent/capabilities.py`: pipecat/wyoming/…),
   and parroting junk ("…Wyoming is a protocol for voice services…"). Dropped
   "step by step" from `_ESCALATE_MARKERS` + planner FINAL now falls back to its
   own knowledge when findings are empty/unhelpful.
2. **Inject-mode latency was silently null.** `tools/live_session/driver.py`
   `_NullOutputStream` only had `write()`, but the engine PULLs audio via the
   PortAudio callback (where TTS_FIRST_AUDIO stamps). Now drives the callback from
   a real-time thread (+ regression test).
3. **Latency dropped on a racy TTS stamp.** `_consume_latency` pairs on
   LLM_FIRST_TOKEN (every answered turn), not the racy TTS_FIRST_AUDIO.
4. **VRAM on the 16GB GPU.** `_free_llm_vram` (evict only a heavy >8GB resident set
   before the sherpa build; warm-ready wait 120s); `__main__` disables AEC in
   inject (no echo → don't mis-apply the residual-floor barge gate). Barge now
   stops 2/2 on most scenarios (was 1/2). config is machine-local (see below).

## Environment on this Windows box (i9-13980HX / RTX 4090 **Laptop, 16GB**)
- **16GB VRAM ceiling (key finding):** gemma3:12b (~10GB) can't reliably share the
  GPU with any second tier + the CPU-side sherpa load — 12b+4b 500s/bad-allocs;
  12b+1b intermittently 500s; and **gemma3:1b as the fast tier gives DEGENERATE
  answers** ("seven times eight"→"Seventeen"). Settled on **both tiers = gemma3:4b**
  in `config.local.json` (gitignored): correct answers, reliable, routing still logs
  its fast/main decision. Alternatives noted in the config comment (12b-alone for
  max quality; 4b+1b for a visible model split at the cost of fast-tier quality).
  Pulled `gemma3:1b` this session.
- **Harness RAM leak:** the live-session runs all 17 scenarios in ONE process and
  builds a fresh runtime each → leaks RAM, sherpa bad-allocs after ~4-5 scenarios.
  **Run `--suite` in small batches**, not all 17 at once.

## Measured (inject, 4b both tiers, --no-input-gate)
- Response speed: first-audio **~2.4–2.6s** = endpoint ~1.1s + LLM ~1.2–1.4s + TTS ~50ms.
- Routing: simple/arithmetic → fast tier; long-form/story/"explain in detail" →
  main tier. WWI causes, a lighthouse-keeper story, rainbow refraction — all correct.
- Barge-in: 3 barge scenarios at **2/2 (rate 1.0, 0 self-interrupts)**.
- Add-on/continuation: "weather today / and also tomorrow" merged into one answer.

## Next steps (pick up here)
1. **Honesty gap:** the assistant CONFABULATES live data it can't fetch (made up a
   Paris weather forecast). The capability catalog / persona should make it decline
   weather/news/time it has no tool for. (self_awareness scenario also probes this.)
2. **Stub search backends:** `search.local`/`web.search` are a 3-entry corpus stub.
   Decide: wire a real backend (SearXNG via `web_search` config) or gate the ReAct
   planner off until one exists. The routing fix mitigates the symptom only.
3. **Short-barge tuning:** a ~0.4s "Stop." is still rejected (`barge_in_min_speech_sec`
   =0.4); lower it for inject, or trigger barge on the coherence detector.
4. **Harness RAM leak:** make the live-session loop run each scenario in a subprocess
   (or fully tear down the runtime) so a full 17-scenario `--suite all` survives.
5. **Addressing gate needs ≥4b:** 1b INGESTs clear questions; only relevant if the
   fast tier is ever set back to 1b.
