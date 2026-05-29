# Engineering Audit — Local-First Real-Time Voice Assistant (`/home/dobo/work/speaker`)

**Audience:** project lead. **Goal anchor:** production-safe and fast, with the best possible *reliable live voice-to-text*. Every finding below is verified against the source on this machine (file:line cited), with severities re-graded after adversarial verification.

---

## 1. Architecture overview

The system is a clean, layered cascaded pipeline (`ASR → LLM → TTS`) with a swappable engine seam and a separate control-plane brain:

- **Audio engine (`core/engines/`)** — `SherpaOnnxEngine` is the production on-device loop: one capture daemon thread (device read → linear resample → `input_gain` → recorder → KWS → VAD/barge-in *or* synchronous ASR decode → endpoint → speaker gate → `on_final`), and one playback daemon thread draining a single bounded queue (maxsize 64) of streamed TTS sentences. `FileReplayEngine` is the headless "measurement twin"; `LiveKitEngine` is the WebRTC transport for the remote path; `ScriptedEngine` is for tests. All STT/TTS/VAD/speaker-ID run on `provider=cpu` so the GPU stays free for the LLM.
- **Runtime + brain (`core/runtime.py`, `always_on_agent/`)** — `VoiceRuntime` owns no DSP/model code; it wires the engine to a priority `EventBus` and the supervisor. Barge-in cancellation is epoch-based and genuinely correct.
- **LLM + routing (`core/llm.py`, `core/routing.py`, `core/capabilities.py`, `core/llm_factory.py`)** — four backends (`EchoLLM`, `OllamaLLM`, `LlamaCppLLM`, `OpenAICompatLLM`) composed into tier routing, latency racing (`HedgeLLM`), and sensitivity-routed cloud chains. Local is structurally always kept in the race.
- **Memory / remote / security (`utils/memory.py`, `remote/`, `core/sensitivity.py`)** — Postgres-backed multi-layer memory off the hot path; a deny-by-default LiveKit token server; a §9.7 local/cloud egress boundary.
- **Reliability / observability (`core/metrics.py`, `core/watchdog.py`, `core/runlog.py`, `core/recorder.py`, `core/sysinfo.py`)** — a live stuck-watchdog, a shared per-turn metrics recorder, an async off-hot-path run-bundle logger, a WAV recorder, and background telemetry.

---

## 2. What is genuinely strong

These are not throwaway compliments — they are load-bearing engineering decisions that are correct and worth protecting:

1. **Epoch-based barge-in cancellation is correct** (`always_on_agent/supervisor.py:114-138`). `cancel_all()` bumps a monotonic `speech_epoch` and sets cancel events under a short-lived lock that is *never* held across an engine call, `stop_speaking`, or a thread join; `tts_request_allowed` drops any older-epoch sentence without depending on the task still being in `active_tasks` (priority ordering guarantees `TASK_COMPLETED` is dequeued before trailing `TTS_REQUEST`s). This is the hardest part of the system and it is right.
2. **The engine seam is clean and shared** (`core/engine.py`, `core/engines/_sherpa_models.py`). Production, replay, scripted, and LiveKit engines share one `AudioEngine`/`EngineCallbacks` contract and one set of model builders; `_supported()` filters kwargs against the installed sherpa signature so old/new builds both work.
3. **Barge-in stop uses `out.abort()`** (sherpa.py:464-483), dropping buffered device audio (~100 ms perceived interrupt vs ~700 ms for `stop()`/drain), stamping `BARGE_IN_STOP` at the true audible-silence instant.
4. **Device recovery is thought-through** (`core/engines/_recovering_input.py`): classifies PortAudio codes from `exc.args[1]`, distinguishes transient-drop-frame from reopen-with-backoff, re-enumerates instead of trusting a stale index, and publishes lifecycle states the watchdog uses to suppress false stall warnings.
5. **Local-first is enforced structurally, not by convention** (`core/llm_factory.py:250-255`, `core/routing.py:64-145`): every `HedgeLLM` keeps local in the race; live-routing nudges are additive-only + clamped and reject NaN/inf so a garbage signal contributes exactly zero.
6. **Observability has real hot-path discipline** (`core/runlog.py:170` no-op `prepare()`, `core/recorder.py:39` copy-and-hand-off, `core/sysinfo.py:134` 10 s sampling). Per-stamp cost is one lock + `dict.setdefault`. `finalize()` is idempotent + atexit-registered so even a crash flushes the bundle.
7. **Memory connection management is correct** (`utils/memory.py` — every call site uses `with self._pool.connection()` short-lived checkout) and degrades gracefully with no Postgres/sentence-transformers.

---

## 3. Verified findings by dimension

> Severities reflect the post-verification grading. Several findings that arrived as P0/P1 were tempered (echo "live demonstrated" → "real latent gap"; some metrics bugs → narrow races); others were confirmed at full strength.

### 3.1 Echo / barge-in / full-duplex (the headline reliability surface)

- **[P0] Input-final speaker gate is pure fail-open by default.** `core/engines/sherpa.py:908-921` (`_should_act_on_final`) returns `True` when gating is off, the gate is `None`, or unenrolled — and `config.json` ships `speaker_embedding_model=""`, so **no gate is even built** (`sherpa.py:292` guards on a non-empty path). Verified live: `speaker_embedding_model=''`, `speaker_gate_input=True` (moot without a model). Unlike the barge-in path (`_looks_like_user`, sherpa.py:873-884, which passes `barge_in_output_margin_db=6.0`), the input path has **no dB-margin fallback at all** — it answers every voice: a TV, a bystander, or its own TTS tail.
- **[P0] No post-speech cooldown.** `_speaking` clears the instant `_play_q.empty()` (`sherpa.py:810-813`) with no trailing guard interval and no recognizer reset on the speaking→listening transition. With no AEC (`sherpa.py:49`), the output-buffer tail + room reverb arrive after ASR re-enables and feed straight into the recognizer. *Verification note:* the cited run-20260529-174902 self-talk is partly a replay artifact (`FileReplayEngine` feeds the whole waveform with no speaking-gate), so this is a **real latent live exposure**, not a fully-demonstrated live loop — but combined with the fail-open gate it is the mechanical root of the self-conversation risk on the default install.
- **[P2] No acoustic echo cancellation on the desktop path** (`sherpa.py:49-53`, `speaker_gate.py:29-48`). The full-duplex story rests on a single global `barge_in_output_margin_db=6.0` validated on one Realtek laptop (`docs/audio_calibration.md`), known-thin at 100% volume. The mobile app does enable platform AEC (`mobile/lib/assistant.dart`), so "no AEC anywhere" is desktop-specific. Enrollment, when present, is a real level-independent guard and the documented primary fix.
- **[P2] VAD is never reset between barge-in windows** (`sherpa.py:589-608`, 30 s ring buffer, `_sherpa_models.py:104`). Empirically confirmed: state latches across un-fed gaps and segments accumulate unpopped over a long session — bounded by the `_looks_like_user` + debounce guards, so a slow-burn reliability drift, not an immediate defect.

### 3.2 Live ASR latency & reliability

- **[P1] Synchronous beam-search decode on the capture thread.** `sherpa.py:616-619` runs `accept_waveform` + `while is_ready: decode_stream` inline on the one thread that also does device read, resample, KWS and VAD. Verified: `asr_decoding_method=modified_beam_search`, `provider=cpu`, threads clamped 2..4 (`_auto_threads`). A decode burst over the 0.1 s block budget backs up `sd.read` → PortAudio overflow `-9981` (handled as a dropped silent frame, `_recovering_input.py:235-239`) → lost mic audio, and delays the next 'stop' poll cross-block. Conditional on hardware (streaming transducer per-chunk decode is normally <0.1 s on a laptop) but a real cliff on phone/CPU. No off-thread decode and no adaptive greedy fallback exist.
- **[P2] `rule2_min_trailing_silence=0.8s` is a single global default** (`_sherpa_models.py:42`, verified `0.8` in `config.json`). Every device profile overrides only thread counts; the 0.8 s is identical on a 4090 and a phone and is the dominant real turn-taking latency. It is also structurally **unmeasured** — `SPEECH_END` is stamped *after* the wait (sherpa.py:639), so `endpoint_latency` reads ~0 in production and replay.
- **[P2] No ASR/TTS warm-up** (`sherpa.py:274-309`, `start()` 345-423). The first utterance pays cold ONNX kernel/arena allocation on top of the cold LLM.
- **[P2] Naive linear resampling** (`_resample_linear`, sherpa.py:188-201) with no anti-alias low-pass on the 44.1/48k→16k capture path — a silent ASR/VAD/speaker-embedding accuracy hit on exactly the mics the fallback chain exists for.
- **[P3] Hotword biasing silently no-ops under greedy** (`_sherpa_models.py:47-48`, sherpa.py:497) while the startup log says it is active.

### 3.3 Runtime / brain / latency

- **[P1] Two blocking fast-LLM calls on the ASR capture thread.** `runtime.py:209` (`addressing.classify`) and `runtime.py:224` (`cleaner.clean`) run serially inside `_on_final` on the audio thread before `bus.publish` — ~0.4-0.6 s of warm-tier latency added to every acted-upon turn, during which ASR decode and the 'stop' fast-path are stalled.
- **[P2] `EventBus` delivers all events on one thread** (`always_on_agent/event_bus.py:59-71`); `runtime._on_event` calls `engine.speak` inline. Fine today (speak enqueues and returns), but no isolation between a misbehaving subscriber and bus liveness.
- **[P2] Watchdog false 'llm stuck' on intent fast-path turns.** `runtime.py:214` marks `ASR_FINAL`, then the intent fast-path returns at 238-240 with no LLM and no `BARGE_IN` exclusion; `watchdog.py:158-165` then warns and `runlog.py:88-89` promotes it to a stuck_hint. The team already guarded the INGEST path against exactly this.
- **[P3] Capacity overflow silently queues turns with no timeout** (`supervisor.py:283-303`).

### 3.4 LLM / routing / cloud egress

- **[P1] No `may_leave_device` gate on the cloud LLM path.** Only `web.search` is egress-gated (`core/websearch.py`); the LLM path's `ChainSelector` picks *which* chain, never *whether* to leave the device. `capabilities.py:226` classifies sensitivity on the bare query, but recall/profile PII is folded into `system_for_call` (line 256) and streamed to a possibly-cloud model (line 320) — verified the system prompt is never inspected by the egress decision. Latent behind `cloud.enabled`/`recall_enabled`/`profile_enabled`/`allow_prc` (all default off), but the data path is built.
- **[P1] In-process `LlamaCppLLM` local tier has no wall-clock bound** (`llm.py:760-762`): when the cloud chain is empty (the pure-local default), `HedgeLLM.stream` short-circuits to `local.stream()` with no `WINNER_SELECT` budget, so a wedged native pre-first-token `generate` hangs the turn forever (cancel is only checked between yielded tokens).
- **[P2] Sensitivity name+money detection is case-sensitive** (`sensitivity.py:98,103-106`, no `re.IGNORECASE`) but runs on unreliable-cased ASR output. Reproduced: `"how many dollars does the CEO michael earn"` classifies PUBLIC. Narrow (needs a public-marker opener + a non-prose-cased name+money phrase) and fails toward third-party financial disclosure, not the user's own credentials.
- **[P2] Per-chain silent degradation** (`llm_factory.py:137-152,216-264`): a sensitivity chain whose only members are PRC-hosted or key-less drops to local-only with just an INFO log, producing sensitivity-dependent latency/quality cliffs.

### 3.5 Memory / remote / security

- **[P2] Recall + profile PII rides to the cloud in the system prompt** — same root as the egress-gate gap above, fully latent behind four default-off gates; the 'Past Conversations' block (`utils/memory.py:919-927`) can also carry prior-turn PII independent of `profile_enabled`.
- **[P2] `apply_retention` nested pool checkout** (`utils/memory.py:1056→1087→794`): deadlocks the retention pass if `pool_max_size=1` or under saturation. De-nest the persist.
- **[P2] Per-connect demo-schema bootstrap runs against `$DATABASE_URL` on every construction** (`utils/memory.py:382`), incl. a `DROP INDEX` on two legacy index names. Verified the live HNSW index it actually uses is `IF NOT EXISTS` and never dropped, so the real residual issue is ungated startup DDL (locks, locked-down roles), not data loss — gate it behind a dev/demo flag (graded P3 after verification).
- **[P2] `/chat` exposes the cloud-hedged main LLM** under `SPEAKER_REMOTE_ALLOW_NOAUTH=1 + SPEAKER_REMOTE_BIND_ALL=1` (`remote/token_server.py`); the per-IP limiter is bypassable behind a proxy/NAT so an open instance can incur cloud spend. Deny-by-default mitigates.

### 3.6 Metrics correctness

- **[P2] `first_token_to_audio` goes negative.** `metrics.py:125-149` opens a turn only on a `_TURN_START` repeat and never closes one when a response finishes (only `app.py` replay calls `close_turn`); `TTS_FIRST_AUDIO` is stamped per playback start (sherpa.py:752-754). On a multi-sentence answer with an inter-sentence gap, a late stamp folds into the next turn (observed -0.28 s). *Verification tempered the scope:* on the live sherpa engine ASR is gated off while `_speaking` is set, so this cannot fire on a turn's trailing sentence — it is a narrow inter-sentence-gap race, not the common case. It still corrupts `summary.json`/bench and the live-routing EWMA on busy sessions.
- **[P2] The dominant 0.8 s endpoint cost is measured nowhere** (`metrics.py:62`, `SPEECH_END` stamped post-wait). The sandbox simulator *does* capture it (`sim_engine.py:86-88`), so the precise bug is a measured-vs-simulated divergence that breaks the metrics module's stated "same shape" goal.
- **[P2] Replay twin emits raw ALL-CAPS text** (`file_replay.py:124-141`, skips `_postprocess_final`), feeding case-sensitive consumers a different distribution than live. *Tempered:* only `_NAME_AND_MONEY` and the missing-punctuation difference actually bite; the intent/addressing classifiers normalize-lowercase first and are insulated.
- **[P3] `fold_local_ttft` cross-turn fold** undershoots the EWMA on cold/slow re-prompted turns (`metrics.py:138-149`) — only matters under the opt-in `live_routing` flag.

### 3.7 Production-safety / CI

- **[P1] `tests/test_memory_pool.py` FAILS (not skips) without `psycopg_pool`.** Reproduced on this machine (`1 failed`, `KeyError 'pool'`): `utils/memory.py:358` `_init_database` returns early on `POSTGRES_AVAILABLE=False` before consulting the injected `pool_factory`. CI (`tests.yml`) installs only pytest+numpy, so this red-lights the green gate the autofix loop depends on.
- **[P2] Run bundles commit verbatim transcripts + full LLM prompts + raw mic WAVs** with no redaction/TTL/opt-out (`runlog.py:109,118,138-143`, `llm.py:40,53`, `.gitignore !logs/runs/*.wav`, 7 tracked bundles). *Tempered:* currently-committed content is benign fixture data on a private repo, so this is a privacy-by-default *design* gap that bites on the first real recorded+committed session, not an active leak.

### 3.8 Concurrency / reliability

- **[P2] FATAL capture state leaves the assistant permanently deaf** with a process that looks alive (`_recovering_input.py:248-279` raises on exhaust → `sherpa.py:670-674` clears `_running`; `runtime.py:288-309` only logs/publishes; no `CAPTURE_STATE` handler in the supervisor; `app.py:232-233` sleeps forever). Conditional on a genuine multi-second device loss, with a log + stuck_hint trail.
- **[P3] `runtime.stop()` never cancels in-flight task threads** (`runtime.py:163-189`); mostly masked behind process exit.
- **[P3] Playback drop-oldest/stop-drain swallow `on_done`** (`sherpa.py:709-722,724-729`) — latent (no production caller passes `on_done`); the "other engines honor it" framing was refuted, they also suppress it on interrupt.
- **[P3] `_playback_level` RMW races a concurrent 0.0 reset** (`sherpa.py:886-897` vs 478); self-healing within a few ms.
- **[P3] WAV recorder silently drops blocks under backpressure** (`recorder.py:50-51`) and never surfaces it in the summary.

---

## 4. SOTA comparison

| Capability | This codebase | SOTA (2025-2026) | Gap |
|---|---|---|---|
| **Turn-commit / endpointing** | Fixed `rule2=0.8s` acoustic silence, content-blind, untuned per device, unmeasured | LiveKit EOU transformer (~50 ms), Pipecat Smart Turn v3 (~8 MB, 12-95 ms CPU), Deepgram Flux native EOT (~260 ms), AssemblyAI adaptive 160 ms-when-confident | **Large** — biggest real latency, no semantic signal |
| **Hiding LLM TTFT** | Strictly sequential; LLM starts only after endpoint + 2 blocking gate calls | LiveKit preemptive generation (ON by default), NVIDIA speculative speech processing, Flux eager-EOT | **Large** — cancel machinery exists, technique unused |
| **Echo / self-talk** | No AEC (desktop); half-duplex mute + fail-open gate + RMS margin | WebRTC AEC3 / Speex / Krisp BVC at transport layer; agent cannot hear itself | **Large** — industry treats open-speaker no-AEC as unsupported |
| **STT decode concurrency** | Synchronous beam-search inline on capture thread, no off-thread, no adaptive fallback | STT an independent streaming stage (LiveKit/Pipecat/NVIDIA worker thread); greedy/ALSD under load | **Medium-Large** — phone-class cliff |
| **Cold start** | Measured 2.75 s first-token; no prewarm by default | Pin + prewarm at process start; warm model assumed in budgets | **Medium** — easily closed |
| **Voice-to-voice budget** | No explicit budget; endpoint reads ~0, `first_token_to_audio` goes negative | Field targets sub-500 ms; Vapi ~465 ms demonstrated; per-stage attribution | **Medium** — can't measure what it doesn't stamp |
| **Self/ambient suppression** | Single fail-open cosine gate, no input dB-margin | Krisp voice isolation, semantic VAD, diarization (Riva), `interrupt_sensitivity` policies | **Medium** |
| **ASR model** | streaming Zipformer2 (sherpa-onnx) | Parakeet-TDT (~6% WER), Moonshine v2 (27-123 MB), Kyutai semantic VAD | **Small-Medium** — current model is reasonable; upgrades exist |
| **Barge-in cancellation** | Epoch-based, deterministic, cancels mid-generation | Comparable | **None — at or above SOTA** |
| **Local-first / egress invariant** | Structurally enforced in routing; PII-first sensitivity | Most stacks are cloud-default | **Ahead** (modulo the system-prompt egress gap) |

**Net read:** the *architecture* is competitive with the best open frameworks and ahead on local-first discipline and barge-in correctness. It trails specifically on the three techniques the field has standardized — AEC, semantic turn detection, and preemptive generation — plus a default-safety packaging gap. None require a rewrite; all fit behind the existing seam.

---

## 5. Phased roadmap

### Phase 0 — Production-safety (do before any real-user deployment)
1. **Fail-safe input gate by default** (P0, S) — thread `barge_in_output_margin_db` + `_playback_level` into `_should_act_on_final`; warn loudly when unenrolled / when a gate is silently dropped under `--llm echo`.
2. **Post-speech ASR cooldown + recognizer reset** (P0, S) — ~0.3 s suppression after `on_speech_end`, reset on every `_speaking` transition.
3. **Fix the psycopg_pool test gate** (P1, S) — restore green CI / the autofix signal.
4. **Wire `may_leave_device` to the LLM path; inspect the injected system prompt** (P1, M) — restore the §9.7 invariant before recall+cloud are ever enabled.

### Phase 1 — Reliability & first-impression latency
5. **Prewarm the LLM (and ASR/TTS) at startup** (P1, S) — kill the measured 2.75 s cold turn-1.
6. **Move ASR decode off the capture thread (or adaptive greedy)** (P1, M) — remove the phone-class dropped-audio/'stop'-stall cliff.
7. **Move addressing + cleaner off the capture thread** (P1, M) — recover ~0.4-0.6 s/turn and un-stall ASR/'stop'.
8. **Fix metrics turn attribution + add a true speech-stop stamp** (P2, S) — make the dominant latency measurable; prerequisite for tuning everything else.
9. **Bound the in-process local generate** (P1, M) — add a wall-clock budget so a wedged `LlamaCppLLM` call cannot hang the turn forever.
10. **Surface FATAL capture state** (P2, M) — spoken/visible notice + a bounded outer re-bring-up loop.

### Phase 2 — Best-in-class live voice-to-text (the headline goal)
11. **Semantic end-of-turn model** (P1, M) — Smart Turn v3 / LiveKit EOU; cut ~0.3-0.55 s off the median turn while reducing cutoffs; expose `rule2` per profile.
12. **Acoustic echo cancellation on the capture path** (P2, L) — WebRTC AEC3 desktop / DTLN-aec phone; the durable full-duplex fix.
13. **Preemptive/speculative LLM generation** (P2, L) — overlap TTFT with trailing speech using the existing epoch-cancel path.
14. **Fix the replay twin** (P2, M) — apply `_postprocess_final` + speaking-gate so the harness measures production input.

### Phase 3 — Quality & hygiene
15. Anti-aliased resampling (P3, S); watchdog intent-turn false positive (P2, S); de-nest `apply_retention` (P2, S); gate the per-connect demo DDL (P3, S); redaction/TTL + default-ignore for run bundles (P2, S); case-insensitive name+money PII (P2, S); recorder drop surfacing + `_playback_level` lock (P3, S).

### Explicitly out of scope
Do **not** chase an end-to-end speech-to-speech model (Moshi/GPT-4o realtime) for the always-on loop — the 2025-26 consensus favors cascaded for tool use, content filtering, debuggability, and on-device control, and a dual-stream model violates the phone-class budget and the raw-audio-never-leaves boundary. If adopted at all, it belongs in the optional cloud thinking tier; record this in `docs/target_architecture.md` §9.7.

---

## 6. One-paragraph verdict

This is strong, deliberate engineering with a correct hard part (barge-in) and disciplined observability. It is held back from production by a handful of **default-safety** decisions — a gate that fails wide open out of the box, no post-speech cooldown, no LLM prewarm, and an egress decision that does not inspect what actually leaves — and from "best reliable live voice-to-text" by the absence of three now-standard techniques (AEC, semantic endpointing, preemptive generation) plus a metrics layer that can't see its own dominant latency. The Phase 0 items are nearly all S-effort and remove the real user-facing failure modes; Phases 1-2 are where the latency and reliability wins live. The codebase is well-positioned to absorb all of it behind the existing engine seam.

---

## Implementation status — branch `perf/production-hardening`

Landed in this branch (full suite green: **848 passed, 12 skipped, 0 failed/errored**;
was 1 failed + 10 errored on a clean machine before):

| # | Fix | Sev | Files | Validated |
|---|-----|-----|-------|-----------|
| 1 | `_init_database` honors the injected `pool_factory` even without psycopg → pool tests skip-or-pass, not fail | P1 | `utils/memory.py` | `pytest tests/test_memory_pool.py` 10/10 |
| 2 | `test_setup_database` `importorskip("psycopg")` → skips on clean machine instead of erroring | P1 | `tests/test_setup_database.py` | full suite 0 errors |
| 3 | Metrics: `first_token_to_audio` guards negative deltas (prior-turn TTS tail mis-attribution) | P2 | `core/metrics.py` | live replay: **0** negative deltas (was −0.28s) |
| 4 | Watchdog: open the metrics turn only after the no-LLM intent fast-path declines → no false "llm stuck" | P2 | `core/runtime.py` | unit + suite |
| 5 | Replay fidelity: shared `postprocess_final` so the bench/replay twin emits the same cased+punctuated text as production | P2 | `core/asr_text.py`, `core/engines/{sherpa,file_replay}.py` | live replay transcript now cased |
| 6 | LLM pre-warm (async) + sherpa ASR/TTS warm (sync, pre-thread) → first turn never cold | P1 | `core/runtime.py`, `core/engines/sherpa.py`, `core/app.py` | live: `LLM warm-up complete in 2.9s`, `audio models warmed` |
| 7 | Input-final speaker gate fail-safe: dB-margin guard when unenrolled + loud one-time "INPUT SPEAKER GATE OFF" warning | P1 | `core/engines/sherpa.py` | unit (`_should_act_on_final`) |
| 8 | Post-speech ASR cooldown (~0.25s) + VAD reset on the speaking→listening transition (echo stopgap, not armed on barge-in/stop) | P0* | `core/engines/sherpa.py` | unit (arming) + live clean start |

\* "P0" reflects the synthesized roadmap priority for the echo/self-conversation
surface; the underlying verified findings were P1/P2 (the risk is real but
conditional on an unenrolled default + acoustics).

### Measured this machine (RTX 4090 laptop, 32c, gemma3:4b)
- ASR streaming decode RTF **0.07–0.10** (mean ~7–9 ms / 100 ms block); beam search ≈ greedy cost → keep beam search; 2 ASR threads ≥ 4 for this zipformer.
- Cold first response 2.9 s → 2.05 s even in a pessimal tight replay loop; sub-second expected live (warm finishes before the user's first utterance).

### Not yet done — roadmap (M/L effort, recommended order)
1. **Semantic end-of-turn** (Pipecat Smart Turn v3 ONNX) to replace the fixed 0.8 s trailing-silence endpoint → ~0.3–0.55 s off the median turn *while* cutting mid-thought interruptions. The single biggest real turn-taking win. *(M)*
2. **True speech-stop metric stamp** at VAD onset-of-silence so the dominant ~0.8 s endpoint cost is actually measured (today it reads ~0). Prerequisite for tuning #1. *(S)*
3. **Acoustic echo cancellation** (WebRTC AEC3 desktop / DTLN-aec ONNX phone) using the already-captured far-end TTS reference → structurally removes self-transcription and enables true full-duplex barge-in. *(L)*
4. **Decouple ASR decode onto a worker thread** + adaptive greedy fallback under load → removes the phone-class dropped-audio/stalled-"stop" cliff. *(M; low urgency on current models per RTF above)*
5. **Preemptive/speculative LLM generation** overlapping the endpoint wait (reuse the epoch-stamped cancellable task path) → hides warm TTFT behind trailing speech. *(L)*
6. **Per-profile endpoint tuning** (`asr_rule2_min_trailing_silence`, decoding method) in `config.json` device profiles. *(S)*
7. **LLM egress gate for injected PII** (recall/profile block) so a misconfigured cloud route can't carry personal data off-device. *(M; latent behind disabled-by-default cloud)*
8. **Run-bundle PII/retention policy** (transcripts + WAVs are committable). *(S–M)*

---

## Stress tests — `tools/stress.py` (run on this machine, 32c / RTX 4090)

New harness hammering the real control plane (runtime → brain → supervisor →
cancel → TTS queue) and the engine queue, plus a real ASR+Ollama load mode.
`python -m tools.stress {turns,bargein,queue,soak,real,all}`.

| Scenario | Load | Result |
|---|---|---|
| turns | 1000 sequential turns (ScriptedEngine+EchoLLM) | **PASS** — 1000/1000, round-trip p50 **10.8ms** / p95 15.7ms, 86 turns/s, RSS +0.8MB, no thread leak |
| bargein | 300 barge-in storms (4 interrupts each, slow streamer) | **PASS** — 300/300, **0 hung**, storm-debounce fired, runtime responsive after |
| queue | 5000-deep playback-queue flood | **PASS** — bounded at 64; surfaced + fixed orphaned `on_done` (now 4936/4936 dropped callbacks fire) |
| soak | 20s continuous | **PASS** — 1751 turns, RSS drift +2.2MB, threads stable |
| real | 18× real sherpa ASR→Ollama(gemma3:4b)→TTS | **PASS** — warm voice-to-first-audio **p50 182ms / p95 301ms**; RSS steps to 800MB then **flat for 11 rounds (no leak)** |

Findings from the stress run, both addressed:
- **Playback-queue `on_done` orphaning** under backpressure — the drop-oldest path
  skipped the dropped sentence's completion callback. Fixed in
  `core/engines/sherpa.py::_enqueue_play` (latent today since the runtime passes
  `on_done=None`, but a silently-skipped callback is a latent hang).
- **Full-pipeline resident memory ≈ 800MB** (ASR+TTS+VAD ONNX + Python + Ollama
  client) and **does not leak** — RSS plateaus after ~7 rounds. The ASR/TTS-only
  path plateaus lower (~538MB). Good capacity-planning number for deployment.

### Additional fixes landed alongside the stress pass
| Fix | Files | Note |
|---|---|---|
| Per-profile endpoint + decode tuning; warm desktop/4090 → `rule2=0.6s` (~0.2s snappier turns), phone stays 0.8s; ASR threads 4→2 (measured ≈ equal, frees cores) | `config.json` | reversible; one number to revert if it clips slow speech |
| Playback-queue `on_done` fires for dropped sentences | `core/engines/sherpa.py` | from the queue stress |
