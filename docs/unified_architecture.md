# Speaker — Unified Architecture

> Single current-truth architecture doc. Sits between docs/architecture.md (as-built) and docs/target_architecture.md (north-star). Last consolidated 2026-06-02.

## Table of contents

- [§0 — Preamble & how to read this doc](#0--preamble--how-to-read-this-doc)
- [§1 — System shape & topology](#1--system-shape--topology)
- [§2 — The control-plane brain (always_on_agent/)](#2--the-control-plane-brain-always_on_agent)
- [§3 — The desktop runtime (core/) & the AudioEngine seam](#3--the-desktop-runtime-core--the-audioengine-seam)
- [§4 — The decision & routing layer (the 4-gate ladder)](#4--the-decision--routing-layer-the-4-gate-ladder)
- [§5 — LLM tiers, cloud routing & the local/cloud boundary (§9.7)](#5--llm-tiers-cloud-routing--the-localcloud-boundary-97)
- [§6 — Memory architecture](#6--memory-architecture)
- [§7 — Real-time quality subsystems](#7--real-time-quality-subsystems)
- [§8 — Capabilities & self-awareness](#8--capabilities--self-awareness)
- [§9 — Optional & experimental tiers (gate inventory)](#9--optional--experimental-tiers-gate-inventory)
- [§10 — Cross-platform contract & mobile](#10--cross-platform-contract--mobile)
- [§11 — Observability, testing & device profiles](#11--observability-testing--device-profiles)
- [§12 — Known gaps & roadmap](#12--known-gaps--roadmap)
- [§13 — Decisions log](#13--decisions-log)

---

## §0 — Preamble & how to read this doc

This document is the **single current-truth architecture** of the speaker project: a snapshot of how the runtime is wired today, sitting between two poles:

- **[`docs/architecture.md`](architecture.md)** — the *as-built* implementation snapshot. Read the intro there for the core shape (one core + many shells, the `AudioEngine` seam, the `AgentEvent`/`Mode` contract).
- **[`docs/target_architecture.md`](target_architecture.md)** — the *north-star* cross-platform strategy. Read §9 for the resolved decisions on the local/cloud boundary (§9.7), device profiles (§10), and why we built both the on-device path (`core/`, `mobile/`) and the remote host+thin-client path (`remote/` + LiveKit).

This doc absorbs the durable truths from ~14 subsystem and design docs (listed in the `docs/` index) and the **verified corrections** below. It does NOT replace the two poles — link to them.

### What this doc defers to

- **[`CLAUDE.md`](../CLAUDE.md)** — session workflow, conventions, run logs, testing, git policy.
- **[`MEMORY.md`](../MEMORY.md)** — the memory system (Postgres on desktop, SQLite on mobile). Detailed in [§6](#6--memory-architecture).
- **[`SETUP.md`](../SETUP.md)** — installation, environment, and first run.
- **[`docs/debugging.md`](debugging.md)** — run artifacts, observability, and how to diagnose failures. Summarized in [§11](#11--observability-testing--device-profiles).
- **[`docs/deployment_profiles.md`](deployment_profiles.md)** — device profiles, engine selection (sherpa/livekit/console), and the `config.json` model. Summarized in [§11](#11--observability-testing--device-profiles).
- **[`docs/testing.md`](testing.md)** — test suite, markers, and when to run which stage. Summarized in [§11](#11--observability-testing--device-profiles).

### Verified corrections

These claims in the old docs are **stale**; the codebase has these truths:

- `always_on_agent/planner.py` is **load-bearing** (imported by `tasks.py`); `planner.py` (explicit `TaskPlan`s) and `react.py` (bounded ReAct loop) are complementary, not duplicates. `always_on_agent/app.py` is a real documented CLI harness, not dead.
- `core/routing.py` (tier router: fast vs main LLM) and `core/capability_router.py` (unified action router: CONTROL/SIMPLE/RESEARCH/ACT) are **composition**, not duplication. The action router composes the tier router. `capability_router` is opt-in (off in base config, on in the desktop profile); byte-identical legacy behavior when off.
- `core/capabilities.py` (LLM-backed provider impls) vs `always_on_agent/capabilities.py` (core-free registry/mechanism) is **interface-vs-implementation** across the shell↔core seam. The brain importing `core/` would break the mobile/remote facade reuse — a load-bearing invariant.
- `tools/stress.py` `scn_real` is fine: `core/runtime.py` declares `warm_on_start` as a real param (prewarm landed after the perf audit). Do NOT carry the old "TypeError / unreproducible baseline" claim.
- `always_on_agent/adapters.py` **no longer exists**; never reference it.

### Experimental tiers (gated, default-off, shipping)

- **DTLN-aec** (`aec_backend='dtln'`): experimental deep ONNX echo canceller (two-stage LSTM, pending tflite→ONNX). Ships with the NumPy FDAF (dependency-free, ~10–20 dB ERLE) as the always-available fallback; DTLN fails open to no-AEC when unavailable. Either tier needs per-device `aec_ref_delay_ms` calibration via `tools/echo_probe.py`. Full gate inventory in [§9](#9--optional--experimental-tiers-gate-inventory); echo-cancellation defense layers in [§7](#7--real-time-quality-subsystems).
- **Smart Turn v3** (`ProsodyTurnCompletionDetector`): prosodic turn-completion detector, gated off by default. Validated on the user's real voice 2026-06-01 (complete turns 0.74–0.98 vs incomplete 0.01–0.56); human-audio only (flat on TTS, so the live floor-lowering A/B is still pending). Ships with the lexical detector as the baseline. Endpointing detail in [§7](#7--real-time-quality-subsystems).

---

## §1 — System shape & topology

**Core + thin per-platform shells** — not a monolith, not independent apps. One portable runtime (`core/`) wraps a shared brain (`always_on_agent/`); thin per-platform UIs talk via a stable event contract. Three deployment topologies, all built: **on-device desktop** (Python `core/` + local LLM), **on-device Android** (Flutter + `flutter_gemma`), and **host+thin-client** (Python brain + LiveKit/WebRTC + web browser).

### Data path

```
┌─────────────────────────────────────────────────────────┐
│ Audio capture  (mic via platform-native / WebRTC)       │
│                          │                               │
│                          ▼                               │
│         ┌─────────────────────────────────────┐         │
│         │    core/engine.py  (AudioEngine)    │         │
│         │  (sherpa-onnx | livekit | scripted) │         │
│         │  VAD, streaming ASR, endpointing    │         │
│         │  barge-in (speaker-ID gated),  TTS  │         │
│         └─────────────────────────────────────┘         │
│  on_partial/on_final/on_barge_in/on_command ▲ speak()  │
│                          │                  │            │
│                          ▼                  │            │
│         ┌─────────────────────────────────────┐         │
│         │   core/runtime.py (VoiceRuntime)    │         │
│         │  orchestrator: engine ↔ brain ↔ TTS │         │
│         └─────────────────────────────────────┘         │
│            STT_FINAL → TTS_REQUEST              ▲       │
│                          │                      │       │
│                          ▼                      │       │
│         ┌─────────────────────────────────────┐         │
│         │  always_on_agent/ (brain)           │         │
│         │  modes · intent · planner · tasks   │         │
│         │  (address gate · cleanup · memory)  │         │
│         └─────────────────────────────────────┘         │
│                          │                              │
│                          ▼                              │
│    core/capabilities.py + core/routing.py              │
│    LLM providers + capability router (tier/action)     │
│                          │                              │
│                          ▼                              │
│         core/llm.py (LLMClient protocol)               │
│         (Ollama | llama.cpp | cloud hedge)             │
└─────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐
│  Per-platform shells             │
├──────────────────────────────────┤
│ • Desktop:                       │
│   core/app.py + Python CLI       │
│   "sherpa" or "livekit" engine   │
├──────────────────────────────────┤
│ • Android (Flutter):             │
│   mobile/ + FFI bindings         │
│   Dart loop → AgentEvent/Mode    │
│   (parallel to desktop, not yet  │
│    sharing the brain)            │
├──────────────────────────────────┤
│ • Host+thin-client:              │
│   remote/token_server.py (FastAPI)
│   remote/worker.py (LiveKit room)│
│   web/ (browser client)          │
│   "livekit" engine + Python brain│
└──────────────────────────────────┘
```

### Component layers

**Audio seam** (`core/engine.py` protocol + implementations; detailed in [§3](#3--the-desktop-runtime-core--the-audioengine-seam)):
- **`SherpaOnnxEngine`** — production, on-device: VAD + streaming STT + endpointing + keyword spotting + TTS. Includes speaker-ID gating so the assistant's own TTS doesn't trigger a self-interrupt.
- **`LiveKitEngine`** — WebRTC transport for the remote path. Same STT/TTS models; audio over a room instead of local mic/speaker.
- **`ScriptedEngine`** — pure-Python fixture replay for tests; zero model deps.
- **`FileReplayEngine`** — headless `.npy`/`.wav` fixture playback for latency benchmarks and CI.

**Orchestrator** (`core/runtime.py` — `VoiceRuntime`; detailed in [§3](#3--the-desktop-runtime-core--the-audioengine-seam)):
- Feeds engine callbacks (`on_partial`, `on_final`, `on_barge_in`, `on_command`) onto the brain's event bus as `STT_PARTIAL`/`STT_FINAL`/`CONTROL_STOP` events.
- Handles the command fast-path: engine's `on_command` keyword match → `AgentEvent(CONTROL_STOP/CONTROL_CONFIRM/CONTROL_MODE)` without invoking the LLM.
- Watches for `TTS_REQUEST` events from the brain and calls `engine.speak(text)` with proper lifecycle (cancel on barge-in).
- Owns the input gate (`AddressingClassifier`), transcript cleanup (`TranscriptCleaner`), and optional ReAct planner.
- Dependency-injects engine + LLM + memory so the same runtime runs with any combination.

**Brain** (`always_on_agent/`; detailed in [§2](#2--the-control-plane-brain-always_on_agent)):
- **`events.py`** — `AgentEvent` / `Mode` (the shell↔core contract, shared across Python, Dart, web).
- **`supervisor.py`** — `AgentSupervisor`: maps STT → intent decision → task start/queue/cancel; owns mode state; emits `TTS_REQUEST`.
- **`tasks.py` + `planner.py`** — cancellable task runtime + explicit step→capability plans (background work like research/reminders).
- **`speech_analyzer.py`** — activation + intent classification *before* the slow LLM.
- **`capabilities.py`** — local registry (`system.time`, notes) — the interface contract; implementations live in `core/capabilities.py`.

**Routing & LLM** (`core/routing.py` + `core/llm.py` + `core/capabilities.py`; routing detailed in [§4](#4--the-decision--routing-layer-the-4-gate-ladder), LLM tiers in [§5](#5--llm-tiers-cloud-routing--the-localcloud-boundary-97)):
- **`HeuristicRouter`** — picks fast vs. main LLM per turn (intent-based, no training).
- **`CapabilityRouter`** (opt-in, off by default) — unified router for both tier choice and action routing (SIMPLE/RESEARCH/ACT).
- **`LLMClient` protocol** — Ollama (desktop GPU), LlamaCppLLM (on-device GGUF), EchoLLM (testing). All accept optional `images=` for multimodal Gemma 3.
- **Capabilities** — LLM-backed providers (fast model answers, main/multimodal model researches, web search, agent action brain).

**Memory** (`utils/memory.py` + `core/`; detailed in [§6](#6--memory-architecture)):
- Postgres-backed on desktop; SQLite on mobile (not yet).
- Shared by supervisor (recent history), capability recall, and the planner.

### Deployment topologies

1. **On-device desktop** (default for this repo):
   - `python -m core --engine sherpa --llm ollama`
   - STT/TTS/VAD/barge-in all local (sherpa-onnx).
   - LLM: Ollama on GPU (`desktop` profile) or llama.cpp (`phone` profile).
   - Raw audio never leaves the device.

2. **On-device Android** (Flutter app, `mobile/`; detailed in [§10](#10--cross-platform-contract--mobile)):
   - `sherpa_onnx` (ASR/TTS) + `flutter_gemma` (Gemma 3 1B, MediaPipe/LiteRT).
   - Fully offline; model downloaded once on first launch, cached on device.
   - `lib/assistant.dart` is a **parallel Dart loop** (command map + streaming TTS) not yet sharing the Python brain.

3. **Host+thin-client** (host runs the Python brain, thin clients over WebRTC; detailed in [§10](#10--cross-platform-contract--mobile)):
   - Host: `python -m remote.worker --engine livekit` joins a room, runs the full Python brain.
   - Web: `remote/token_server.py` (FastAPI) mints LiveKit tokens, serves `web/index.html`, and offers a text `/chat` endpoint for the browser.
   - Audio over WebRTC (LiveKit); the always-on capture loop still runs on the device (phone/browser); STT/TTS route through the same sherpa-onnx engine running on the host.
   - Raw audio traverses the network as compressed WebRTC frames; ASR text + task results flow back.

### The contract

All platforms share the `always_on_agent` **`AgentEvent`/`Mode` model** (defined in `always_on_agent/events.py`):

```python
class EventKind(str, Enum):
    STT_PARTIAL = "stt.partial"         # Transcription in progress
    STT_FINAL = "stt.final"             # Turn complete
    INTENT_DECISION = "intent.decision" # Brain decided what to do
    CONTROL_STOP = "control.stop"       # Fast-path command (no LLM)
    CONTROL_MODE = "control.mode"       # Mode change
    TASK_STARTED = "task.started"       # Background work begun
    TASK_COMPLETED = "task.completed"   # Work done; TTS_REQUEST follows
    TTS_REQUEST = "tts.request"         # Speak this text
    # ... (others for memory, failures, state changes)

class Mode(str, Enum):
    PASSIVE = "passive"         # Listen, don't act
    ASSISTANT = "assistant"     # Conversational Q&A
    COMMAND = "command"         # Fast keyword/intent actions
    SEARCH = "search"           # Local corpus search
    RESEARCH = "research"       # Planner + web search
    DICTATION = "dictation"     # Spoken note-taking
    MEETING = "meeting"         # Call recording + summary
```

This contract is **not** a binary core passed around; it is **reimplemented faithfully per runtime** (Python on desktop/server, Dart on mobile). The brain logic (supervisor, planner, task runner) is the same shape on every platform; platform-specific I/O is the engine (audio + LLM). Shared tests drive both implementations via transcript fixtures and verify they emit the same event sequence (see the golden contract suite in [§10](#10--cross-platform-contract--mobile)).

### Scale: local-first with a hybrid cloud thinking tier

- **Always-on loop** (on-device, every platform): STT → VAD → fast-tier LLM (Gemma 4b-class, answers in ≤2s) → TTS. Raw audio never leaves the device. This loop is always local.
- **Thinking tier** (optional, opt-in via config): main planner / research / multimodal summarize / web search may use cloud (§9.7 of `target_architecture.md`). Only post-ASR text + screen captures + files cross to cloud; only when the thinking tier is invoked. The boundary mechanics are detailed in [§5](#5--llm-tiers-cloud-routing--the-localcloud-boundary-97).

---

## §2 — The control-plane brain (always_on_agent/)

**Module map:** The brain is a supervisor-owned mode and task coordinator above a priority event bus, with no imports from `core/`.

| Module | Purpose |
|--------|---------|
| `events.py` | Event schema (`AgentEvent`, `EventKind`, `Mode`) and constructors for STT, control, and task lifecycle. |
| `models.py` | Speech observations and intent decisions (`IntentKind`, `SpeechObservation`, `IntentDecision`). |
| `speech_analyzer.py` | Deterministic `LiveSpeechAnalyzer`: activates on keywords, normalizes text, decides intent (STOP/CONFIRM/MODE_SWITCH/ASSISTANT/SEARCH/RESEARCH/COMMAND/DICTATION/MEETING_NOTE). |
| `event_bus.py` | `EventBus`: priority queue with `PriorityQueue`, daemon consumer thread, handler subscriptions. Priority order: CONTROL_STOP (0) → CONTROL_MODE/CONFIRM/DENY (5–10) → STT_FINAL (50) → TASK_COMPLETED (60) → TASK_PROGRESS (70) → STT_PARTIAL (90) → default (100). |
| `supervisor.py` | `AgentSupervisor`: owns mode state, active/queued/pending-confirmation tasks, speech epoch (for barge-in). Handles event dispatch, task queueing, timeouts, followups, and continuation merging. |
| `tasks.py` | `TaskRuntime`: spawns cancellable daemon threads per task; `AgentTask` wraps input_text → output_text with cancel events and deadline tracking. |
| `planner.py` | `TaskPlanner`: builds explicit `TaskPlan` (step sequence + metadata) from intent decisions. Complementary to `react.py`. |
| `react.py` | Bounded ReAct loop: LLM plans, capabilities are tools. Imports `CapabilityRegistry` but not `core/`. |
| `continuation.py` | `ContinuationClassifier`: merges or queues follow-up utterances into a single in-flight ASSISTANT turn. |
| `followups.py` | Proactive nudges: timer-based silence cadence, resumption markers ("So, anything else?"). |
| `memory.py` | Session-local transcript store, optionally persisted (see [§6](#6--memory-architecture)). |
| `runtime.py` | `AlwaysOnAgentRuntime`: ~30-line public facade (`ingest_partial`, `ingest_final`, `stop`, `snapshot`, `wait_idle`) for thin clients and audio integration. |
| `bridge.py` | Callback adapter for wiring STT events from the audio stack. |
| `capabilities.py` | Core-free registry and interface; implementation is in `core/capabilities.py` (see [§8](#8--capabilities--self-awareness)). |
| `app.py` | Documented CLI harness and replay driver (not dead code). |
| `diagnostics.py` | Summarization for test output and debugging. |

**Invariant:** The brain imports **nothing from `core/`**. This seam (shell↔brain) preserves reuse for the mobile/remote facades. `core/` may import `always_on_agent` but the reverse is forbidden.

**Planner and ReAct are complementary, not duplicates:**
- `planner.py` (**explicit task plans**): maps intent decisions → fixed step sequences (e.g., RESEARCH → [scope, web.search, synthesize]). Always-on and deterministic. Driven by `TaskRuntime._run_plan` in the worker thread.
- `react.py` (**bounded ReAct loop**): an LLM-backed capability that accepts a decision's text and iteratively calls tools (from the `CapabilityRegistry`) until it emits FINAL. Optional, opt-in, and slower. Sits *inside* a task step (e.g., as the `assistant.answer` capability in the ASSISTANT mode plan).

The planner chooses *which* capability (e.g., `assistant.answer` vs. `web.search` vs. `command.stage`); the ReAct loop is the implementation of `assistant.answer` when enabled. (The decision/escalation ladder that triggers escalation is in [§4](#4--the-decision--routing-layer-the-4-gate-ladder).)

**Task lifecycle:** `LiveSpeechAnalyzer.decide()` emits an intent → `TaskPlanner.plan()` builds a `TaskPlan` → `AgentSupervisor` queues or starts the task → `TaskRuntime.start()` spawns a daemon → `_run_plan()` executes steps and emits `TASK_PROGRESS`/`TASK_COMPLETED`/`TASK_FAILED`. Barge-in (`cancel_all`) sets the task's `cancel_event`, and stepping polls it. Task output is stamped with the `speech_epoch` at start; a later barge-in advances the epoch and stale `TTS_REQUEST`s drop via `tts_request_allowed()`.

**Continuation (add-ons):** When enabled (config `continuation.enabled: true`), a follow-up to a live ASSISTANT turn is:
- **merged** if not yet speaking (one combined reply, one stream),
- **queued behind** if already speaking (strict sequential dependency).

Gated at `_maybe_continue` after control phrases (STOP/CONFIRM/MODE_SWITCH) so real commands cannot be misread.

**AlwaysOnAgentRuntime:** Thin ~30-line facade over `AgentSupervisor` for audio integration. Methods: `ingest_partial(text)`, `ingest_final(text)`, `stop(reason)`, `snapshot()`, `wait_idle()`. Polls the bus drain synchronously; does not spawn its own thread.

**Confirmation staging:** `COMMAND` intent tasks are queued under `pending_confirmations` (not started). User says "confirm" → `CONTROL_CONFIRM` → `_confirm_next()` moves the first confirmation to active. User says "no"/"deny" → clears the queue.

**Timeouts and reaping:** Per-mode wall-clock deadlines (ASSISTANT 25s, RESEARCH 120s, default 60s). The watchdog tick calls `reap_overdue_tasks()` (safe from the watchdog thread: uses `_cancel_lock`, never joins). A reaped task emits a timeout apology and unblocks the queue. (The watchdog itself lives in `core/`; see [§7](#7--real-time-quality-subsystems).)

**Speech epoch invariant (realtime-concurrency-1):** Every `TTS_REQUEST` carries the epoch captured when the task started. A barge-in advances the global epoch under `_cancel_lock` atomically; the supervisor drops any `TTS_REQUEST` stamped with a stale epoch, even if `TASK_COMPLETED` (priority 60) has already removed the task from `active_tasks` before the trailing sentence (priority 100) dequeues. This decouples the liveness check from active-task membership, preventing race-condition audio bleed.

---

## §3 — The desktop runtime (core/) & the AudioEngine seam

**VoiceRuntime** (`core/runtime.py`) is the thin orchestrator between the **AudioEngine** (audio I/O, STT, TTS) and the **AgentSupervisor** (the brain's event loop, modes, intents, cancellation; see [§2](#2--the-control-plane-brain-always_on_agent)). It owns no DSP or model code; instead, it dependency-injects an engine and LLM pair at startup and mediates between them via `EngineCallbacks` and the event bus.

### The AudioEngine seam

`core/engine.py` defines the boundary: one abstract `AudioEngine` class with one callback contract (`EngineCallbacks`), and four production/test implementations:

- **`SherpaOnnxEngine`** (`core/engines/sherpa.py`): production on-device. Pairs `sherpa-onnx` (k2-fsa, VAD + streaming ASR + TTS + keyword spotting) with `sounddevice` for mic/speaker I/O. Captures at the device's native rate (48/32/96 kHz preferred over 44.1) and resamples to 16 kHz via `AudioResampler` (anti-aliased: soxr > scipy.resample_poly > linear fallback).
- **`ScriptedEngine`** (`core/engines/scripted.py`): in-memory test/console engine. Callers inject `partial()`/`final()`/`barge_in()` and read the `.spoken` list.
- **`FileReplayEngine`** (`core/engines/file_replay.py`): headless mode. Replays recorded `.npy`/`.wav` fixtures through the real sherpa-onnx models (not mocked) for latency benchmarks and CI, with timestamps stamped deterministically.
- **`LiveKitEngine`** (`core/engines/livekit.py`): remote transport. Substitutes the mic/speaker for a LiveKit WebRTC room; the full `VoiceRuntime` → `AgentSupervisor` pipeline is reused unchanged (see [§10](#10--cross-platform-contract--mobile)).

### Capture audio front-end stack (production)

**Order of processing** in `SherpaOnnxEngine`:

1. **Resampling** (`AudioResampler`, `core/audio_frontend.py`): Stateful anti-aliased downsampler (16 kHz target). Prefers soxr (carries FIR state across blocks, no per-block seam) → scipy polyphase → naive linear. Shared by the live engine and enrollment recorder. (Quality rationale in [§7](#7--real-time-quality-subsystems).)
2. **Gain + soft-knee limiter** (`apply_gain_soft_limit`, `core/audio_frontend.py`): Mic-level scaling with smooth saturation (no hard clipping distortion). Residual harmonics above 8 kHz are filtered out by the resampler.
3. **AEC** (acoustic echo cancellation, `core/engines/_aec.py`, **default off**, gated by `sherpa.aec_enabled`): Optional NumPy-backed frequency-domain block adaptive filter (FDAF, 512-tap frame, 50% overlap-save). Takes the near-end (mic) block and the far-end reference (TTS being played, teed via `FarEndRing` from the playback thread and read at the configured `aec_ref_delay_ms` speaker→mic delay) and subtracts the loudspeaker→mic echo before any recognizer/VAD/speaker embedder downstream. Double-talk freeze prevents the filter diverging onto the user's voice during simultaneous talk. Passthrough-on-error. Achieves ~10–20 dB real-world ERLE on open speakers (loudspeaker nonlinearity caps the gain); good enough for headsets and near-field rooms. Requires per-device calibration of `aec_ref_delay_ms` via `tools/echo_probe.py`. A deep-learning tier (DTLN-aec, `aec_backend='dtln'`) is reserved but currently fails open to no-AEC, pending tflite→ONNX conversion (see [§7](#7--real-time-quality-subsystems) and [§9](#9--optional--experimental-tiers-gate-inventory)).
4. **Denoiser** (`Denoiser`, `core/engines/_denoiser.py`): Stateful sherpa-onnx `OnlineSpeechDenoiser` (GTCRN, ~523 KB, CPU). Cleans the single 16 kHz block once upstream of recognizer/VAD/speaker embedder so all see the de-noised signal. Passthrough-on-error.
5. **VAD + streaming ASR + optional two-pass final** (`sherpa-onnx` zipformer + offline SenseVoice): Voice activity detection, token-by-token streaming transcription (low-latency partials + acoustic endpoint), and partial/final emission via callbacks. When `asr_final_backend` is set (`'sense_voice'` — the shipped default — or `'whisper'`), the endpointed utterance is RE-transcribed by the offline model for a robust, punctuated final (SenseVoice brings punctuation+casing+ITN built-in, so `_postprocess_final` is skipped on the second pass). Absent model or empty `asr_final_backend` → streaming-only (byte-identical). Measured: ~55 ms second-pass cost (2 threads). See [§7](#7--real-time-quality-subsystems) and `docs/asr_two_pass_2026-06-01.md`.
6. **Speaker embedder** (enrolled speaker gate, see below).
7. **Barge-in gate** (see below).

### Endpointing & barge-in

**Semantic turn-completion** (`core/endpointing.py`): Layered on top of the acoustic silence timer (configurable `asr_rule2_min_trailing_silence`, default 0.8 s). A pluggable `TurnCompletionDetector` protocol (shipped: `LexicalTurnCompletionDetector`, deterministic lexical analysis; `ProsodyTurnCompletionDetector` Smart Turn v3 prosodic ONNX, real-voice scored). An adaptive confidence-tiered floor lets a high-confidence completion commit at a shorter silence. When the partial reads as a complete turn, fire `on_final` early; when it ends mid-phrase ("and", "the", "um"), hold past the timer (up to `endpoint_max_silence_sec`). Falls back to pure acoustic if disabled. Full mechanics in [§7](#7--real-time-quality-subsystems).

**Multi-signal barge-in** (`core/engines/echo_coherence.py`, `core/engines/speaker_gate.py`): The mic stays open during playback, so the assistant's own TTS can leak back and look like a barge-in. The **primary** detector is `EchoCoherenceDetector` (`sherpa.coherence_barge_in_enabled`, default on, needs scipy): it measures magnitude-squared coherence between the played TTS reference and the mic over the voiced band (300–3400 Hz) and fires only on sound the reference can't explain — **volume-independent by algebra** (works at any speaker level) and **never self-interrupts** (the assistant's own echo is fully explained). Its trigger margin is a **self-calibrating EWMA control chart** (learns the room's echo-incoherence mean + variance → zero per-room tuning). When coherence can't decide (no reference yet, or TTS silence), it falls back, strongest-first, to: the **speaker-ID identity gate** (`SpeakerGate`, when enrolled — cosine ≥ 0.5, rejects echo by identity), optional **AEC** (relaxed margin), and a **loudness / output-margin guard** (`barge_in_output_margin_db`, default 6 dB, with optional `input_loudness_margin_db` rescue). Headsets sidestep echo entirely. Full mechanics + calibration in [§7](#7--real-time-quality-subsystems).

### VoiceRuntime: event loop & metrics

**Lifecycle:**
- `start(run_bus=True)` fires the engine and optionally starts the event bus on a background thread (production). Tests use `run_bus=False` and pump via `wait_idle()`.
- `stop()` guards each teardown step (watchdog, supervisor, bus, memory, engine) so an error never prevents the recording flush.

**Engine callbacks → brain:**
- `on_partial(text)` → publishes `AgentEvent.partial()`.
- `on_final(text)` → applies optional input gate (`AddressingClassifier`, `core/addressing.py`), transcript cleanup (`TranscriptCleaner`, drops disfluencies), optional capability router decision, intent fast-path, then publishes `AgentEvent.final()`. (The full gate ladder is in [§4](#4--the-decision--routing-layer-the-4-gate-ladder).)
- `on_barge_in()` → cancels in-flight work (deterministically set before stopping playback, **realtime-concurrency-1**), fires watchdog, publishes `AgentEvent.stop("barge_in")`.
- `on_command(keyword)` → keyword fast-path (maps keyword → "stop"/"confirm"/"deny"/"mode:*"), else intent fast-path, else publishes `AgentEvent.final()`.
- `on_metric(name, at=...)` → feeds the `MetricsRecorder` (see [§11](#11--observability-testing--device-profiles)).
- `on_heartbeat()` → capture-thread liveness signal for the watchdog.
- `on_capture_state(state, message)` → "open"/"recovering"/"fatal"; published as `AgentEvent` and notifies the watchdog to skip false "stalled" warnings.

**Metrics** (`core/metrics.py`): Per-turn latency instrumentation (`TurnRecord`) recording named stage boundaries (`SPEECH_END`, `ASR_FINAL`, `LLM_FIRST_TOKEN`, `TTS_FIRST_AUDIO`, `BARGE_IN`, `BARGE_IN_STOP`) and the deltas between them. Detailed in [§11](#11--observability-testing--device-profiles).

**Watchdog** (`StuckWatchdog`, `core/watchdog.py`): Background daemon (1 s tick) that warns on LLM/TTS stalls, silent capture heartbeat, and barge-in storms, and is wired to `on_tick` (the supervisor's overdue-task reap, see [§2](#2--the-control-plane-brain-always_on_agent)) so a hung task is killed rather than just diagnosed. Detailed in [§7](#7--real-time-quality-subsystems) and [§11](#11--observability-testing--device-profiles).

**Startup pre-warm** (`warm_on_start`): Optional background thread loads answering models (and the engine if it exposes `.warm()`) *before* the user speaks, paying the cold-start cost off the user's first utterance. Per §9.7 (egress gate), only local-answer models are warmed (`_answers_locally` predicate); cloud-backed `HedgeLLM`/`SensitivityRouterLLM` are skipped to avoid billing. The system prompt (from the capability manifest) is used so the KV-cache prefix is real. Signals readiness via `warm_ready` event. Full mechanics in [§7](#7--real-time-quality-subsystems).

**Command fast-path** (`normalize_command`, `core/contract.py`): Keyword-spotted phrases are mapped (case-insensitive) to control actions ("stop", "confirm", "deny", or "mode:<name>"); unmapped keywords fall back to intent fast-path, else normal ASR → brain. (Cross-language contract detail in [§10](#10--cross-platform-contract--mobile).)

### Startup reconciliation & routing

`_reconcile_capabilities()` logs the capability manifest once (from `registry.manifest()`) and warns if the planner is configured with unregistered tools (drift prevention; see [§8](#8--capabilities--self-awareness)).

Optional **capability router** (`core/capability_router.py`): When configured, it drives both the tier choice (fast vs main LLM) and the escalate decision (for the ReAct planner), so a single coherent module decides `SIMPLE/RESEARCH/ACT` + tier. Absent → legacy per-gate routing stands, byte-identical. The full composition is in [§4](#4--the-decision--routing-layer-the-4-gate-ladder).

### Architecture notes

- **Seam design:** One contract (`EngineCallbacks`) + four implementations (sherpa/scripted/file-replay/livekit) lets tests, CLI, benchmarks, and the remote path all speak the same language to the brain without code duplication.
- **Passthrough-on-error DSP:** AEC, denoiser, resampler, and addressing gate all fail open (return input unchanged) so transient model hiccups or edge cases never crash the capture daemon.
- **Per-platform profiles:** `config.json`'s `device_profiles` (desktop/phone) and CLI `--device` flag shallow-merge engine + LLM settings, so mobile can override `engine="livekit"`, disable costly features, etc. (See [§11](#11--observability-testing--device-profiles).)
- **Session recording** (`WavRecorder`, `core/recorder.py`): Background-threaded writer of the exact 16 kHz mono audio the recognizer hears to WAV, so recorded runs can be replayed bit-for-bit and frozen as regression tests.

---

## §4 — The decision & routing layer (the 4-gate ladder)

A user utterance flows through four escalating gates, each able to short-circuit the next. Each gate rejects or answers with the cheapest tier that can handle the input.

### Gate 1: Should we respond at all? (addressing classifier)

**Gate 1a — speaker identity** (`core/engines/sherpa.py::_should_act_on_final`): When enrolled and `sherpa.speaker_gate_input=true`, the engine gates each ASR final on speaker identity *before* the addressing classifier runs — only the enrolled user's voice (cosine ≥ `speaker_threshold`, default 0.5) proceeds, so other speakers, a TV, or read-aloud text are dropped here. An optional `input_loudness_margin_db` rescue re-admits loud near-field speech whose embedding dipped (never overriding an identity *accept*). Unenrolled or `speaker_gate_input=false` → fails open (all finals proceed).

**Gate 1b — addressing classifier:** `core/addressing.py` (`input_gate` config block) classifies each surviving ASR final as **ACT** (addressed to the assistant, reply expected), **INGEST** (ambient speech, read-aloud text, disfluency — remember silently), or **UNSURE** (genuinely ambiguous; caller decides the policy). The classifier runs the fast-tier LLM with a fixed system prompt; failures default to UNSURE so an LLM hiccup never silently flips behavior.

- **Disabled in base config** (`enabled=false`) to keep phones silent.
- **Enabled in the desktop profile** (`input_gate.enabled=true`) where the fast tier has GPU headroom.
- `unsure_acts=true` answers on ambiguity (legacy behavior); `false` is conservative (silent ingest).
- Requires `llm.fast_model` to run (see [§5](#5--llm-tiers-cloud-routing--the-localcloud-boundary-97)).

### Gate 2: Is it a control phrase? (deterministic command fast-path)

`core/intents.py` + the keyword-spotter fast-path bypass the LLM entirely for:

- **Control phrases** ("stop", "cancel", mode switches) — matched by `core/contract.py::is_stop_command`.
- **Literal intent phrases** ("what time is it", "set a timer for 5 minutes") — matched by `IntentGrammar` over low-latency regexes.
- **Custom command phrases** — normalized and matched exactly with no LLM.

Matched intents execute directly (speak the time, set a timer) and return. A miss falls through to gate 3.

### Gate 3: Which model tier? (heuristic + live headroom routing)

`core/routing.py` (`HeuristicRouter`, the tier router) scores the query on `[0, 1]`; a threshold (default `0.3` in base config, raised to `0.55` on slow devices) selects **fast** (small, snappy, e.g. `gemma3:4b`) or **main** (large/multimodal, slower, e.g. `gemma3:12b`). See [§5](#5--llm-tiers-cloud-routing--the-localcloud-boundary-97) for the tier definitions.

**Signals (all lexical, no model):**
- **Mode** (`context['mode']`): research/search/meeting → +0.6 (main); dictation → −0.3 (fast).
- **Intent kind** (from the brain): research/search → +0.6 (main); command/dictation/meeting_note → −0.3 (fast).
- **Query length:** 40+ words → +0.4; 20+ → +0.25; 12+ → +0.12.
- **Complexity markers** (why/how/explain/compare/analyze/step by step/…) → +0.18 per hit, capped at +0.5.
- **Generation markers** (tell me a/write a/poem/story/walk me through) → +0.5 (one hit escalates unambiguously).
- **Double questions** → +0.1.
- **Live headroom signal** (optional, additive-only): if the local fast tier's TTFT is slow (≥800 ms) or system load is high (≥75%), nudge up to +0.25 (clamped, never subtracts). Missing or good signals leave the static decision untouched; the local tier is never starved.

**Profile override:** Each `device_profiles` entry can set its own `llm.router.threshold` (e.g. phones raise it to stay on the fast tier even on long queries).

**Learned router (opt-in, desktop only):** Set `llm.router.backend="learned"` to use a RouteLLM-style BERT sequence classifier for semantic routing (torch/transformers imported lazily, zero cost to phones).

### Gate 4: Does it need multi-step gathering? (ReAct escalation)

`always_on_agent/react.py` (`agent.planner`) detects escalation markers (search/find/look up/research/compare/latest/options/and then/step by step) and query length ≥4 words. Matching queries run a bounded plan→execute loop (default max 4 steps) over the configured tools instead of a one-shot reply. (See [§2](#2--the-control-plane-brain-always_on_agent) for how the planner and ReAct loop compose.)

- **Configured by** `agent.planner` block: `enabled` (default: true), `max_steps` (default: 4), `tools` (default: web.search, search.local, research.scope, research.local).
- **Tool** is any capability in the registry marked `planner_tool=True`; the planner never recurses into `assistant.answer` (see [§8](#8--capabilities--self-awareness)).
- **Barge-in respects** `context['cancel_event']`: a STOP or barge-in aborts the planner mid-step and mid-stream.
- **Deadline renewal** auto-extends the ASSISTANT-mode reap timer (default 30s) so real multi-step turns are not killed prematurely (see [§7](#7--real-time-quality-subsystems)).

### Composition: routing.py vs capability_router.py (NOT duplication)

The **tier router** (`core/routing.py::HeuristicRouter`) and the **unified capability router** (`core/capability_router.py`) are **complementary composition**, not competing implementations.

**Tier router** (`core/routing.py`):
- Decides **which model tier** answers the query (fast vs main).
- Runs gate 3: lexical scoring on mode, intent, length, complexity/generation markers, live headroom.
- Reused by both the one-shot brain path and the escalated planner, so both tiers respect the same heuristic.

**Capability router** (`core/capability_router.py`):
- Decides **what action** the turn takes: CONTROL (instant, no LLM) → SIMPLE (one-shot reply) → RESEARCH (gather with tools) → ACT (perform an action).
- Disabled by default (`enabled=false` in base config); when enabled (e.g. desktop profile: `enabled=true, llm_assist=true`):
  - **Heuristic base** reuses existing signals (tier router's heuristic + `should_escalate` + stop-command detection) for the action choice.
  - **Optional LLM disambiguation** (fast-tier only, memoized) refines the action when the heuristic's confidence is low, but **only on borderline calls** — CONTROL and confident decisions bypass the LLM.
  - **Adapters** (`CapabilityTierRouter`, `escalate_predicate`) wire the unified decision back into the existing brain so gate 3 and gate 4 share one router with **no code duplication** in `always_on_agent` or `core/agent.py`.

**When disabled** (base config): the existing per-gate routing stands — tier router decides tier (gate 3), `should_escalate` decides escalation (gate 4), separate code paths. Byte-identical behavior.

**When enabled** (desktop profile): one router makes all four decisions (addressing, control, escalation, tier). Gate 3 and gate 4 consistency is guaranteed because the planner's escalate predicate is now just `router.route(...).escalates`. The LLM-assist layer adds semantic routing to the heuristic's lexical floor without breaking weak devices (set `llm_assist=false` and pay nothing).

**Load-bearing invariant:** The capability router never imports from `core.agent` (only from `core.routing` and `always_on_agent.react`). The brain stays importable by mobile and remote facades, and `core` stays phone-safe.

---

## §5 — LLM tiers, cloud routing & the local/cloud boundary (§9.7)

The LLM stack is two-tiered: a small fast model for snappy spoken replies (local-only, always) and a main model (research, multimodal, reasoning) that **optionally** hedges or falls back to cloud. The local/cloud boundary enforces a hard security gate: only post-ASR text, screen captures, and given files may cross; raw audio never leaves the device. Egress is classified by sensitivity (personal/code/public) and routed through failure-tolerant cloud chains.

### LLMClient protocol and implementations

All LLM surfaces conform to a single `LLMClient` protocol (`core/llm.py:70-92`): `generate(prompt, *, system, images)` and `stream(prompt, *, system, images)`. This unifies:

- **EchoLLM** — deterministic fake for tests and the offline console demo.
- **OllamaLLM** — GPU-accelerated Ollama daemon on desktop (default). Lazy `ollama` import; configurable `keep_alive` to minimize cold reloads; `timeout` on the socket/read to reap a hung server before the whole turn stalls.
- **LlamaCppLLM** — on-device GGUF on phones and headless machines (no daemon). Lazy `llama-cpp-python` import; thread-locked context so one quantized model can't run two inferences concurrently.
- **OpenAICompatLLM** — streams any OpenAI-compatible `/v1/chat/completions` endpoint (Groq, Cerebras, Together, DeepSeek, Moonshot, OpenRouter, `llama-server`). Lazy `openai` import; built only when `llm.cloud.enabled` or `cloud_providers` is populated.

### Two-model split: fast tier + main tier

`core/llm_factory.py::build_llms` returns `(main_llm, fast_llm)`:

- **main_llm** — the larger/multimodal model (`gemma3:12b` default on Ollama; `gemma3:4b` GGUF on phone). Wrapped by `_wrap_cloud` so it hedges or falls back to cloud.
- **fast_llm** — optional small model (`gemma3:1b`/`4b`, separate path on mobile). Stays local; never calls cloud. Used for input-gate cleanup, spoken confirmations, and snappy replies where the local model has headroom.

When `--llm echo`, both are the deterministic fake (for tests).

### HedgeLLM: local-first racing with failover chains

`HedgeLLM` (`core/llm.py:616-955`) races the local main LLM against an optional cloud chain:

- **hedge** (default) — local starts now; if it doesn't produce a token within `hedge_delay_ms` (~150 ms), also launch the first cloud in parallel. Whichever yields the first token wins; the loser is stopped. Caps cloud spend while racing when local is slow.
- **fallback** — cloud-first with a `ttft_deadline_ms` (default ~1.2 s) first-token deadline. On timeout or error, advance to the next cloud in the failover chain; after exhaustion, fall back to local.

The cloud parameter accepts either a single client (back-compat) or a list (failover chain). Missing API keys cause a provider to silently drop from its chain; a fully-dead chain falls through to local with no additional wait.

**Hard-close and timeout (BR1):** `OpenAICompatLLM.stream` binds the SDK stream inside the generator body (not before). On cancel/barge-in (consumer closes the generator early), `GeneratorExit` runs the `finally`, calling `sdk_stream.close()` to stop billing and release the socket immediately. A pre-first-token close is a no-op (sdk_stream unbound). A short socket timeout `llm.cloud.timeout_s` (default 5 s, configurable) reaps a losing worker stuck in its first-token read before the client-level 30 s timeout would.

**Worker management:** HedgeLLM runs each source (local + each cloud) in a daemon thread, signalling all losers to stop after a winner produces its first token. A bounded join (`WORKER_JOIN_TIMEOUT` 0.5 s) reaps daemon threads promptly so the generator never leaks them on cancel. An unbounded drain idle timeout (`DRAIN_IDLE_TIMEOUT` 30 s) guards against a winner whose connection stalls mid-stream (TCP black-hole), and a wall-clock budget (`_winner_select_budget`) scaled by the chain length bounds the pre-first-token wait so hung sources (e.g., in-process `LlamaCppLLM` native calls with no socket timeout) are reaped before the turn indefinitely blocks.

**Per-turn max_tokens ceiling (BR4):** injected into the merged request kwargs *before* the provider-cap block (e.g., Cerebras free tier's 8192 limit) so `min()` composition keeps the profile cap authoritative. Plumbed via `llm.cloud.max_tokens` (default `None`, set ~512 in cloud-enabled voice profiles).

### SensitivityRouterLLM: context-driven chain selection

`SensitivityRouterLLM` (`core/llm.py:958-1022`) dispatches `generate`/`stream` to one of several backing LLMs based on the turn's data-sensitivity tag, read from a `ContextVar` (`capability_context` at `core/llm.py:16`) set by the capability layer before invoking the LLM. This keeps the `LLMClient` protocol unchanged while letting the routing decision flow from the brain's per-turn context.

**Back-compat:** the `cloud` parameter can be a single `LLMClient` (which HedgeLLM wraps) or a list (failover chain). When a list is given, HedgeLLM is built once per chain, and each chain is wrapped in a separate `HedgeLLM`. Only the multi-provider path is fully implemented in the current config; the single-cloud back-compat site in `_wrap_cloud` still works.

### Sensitivity classification and egress gate

`core/sensitivity.py` classifies every turn into one of three sensitivity tiers:

- **PRIVATE** (default, fail-safe) — personal data (`my <noun>`, possessives), OS-level commands (`COMMAND`/`DICTATION`/`MEETING_NOTE` intents), meeting mode, credentials, addresses, health records. Routed to US-only chains (Cerebras, Groq). Raw audio is never analyzed for sensitivity; only post-ASR text is classified.
- **CODE** — code markers (`function`, `class`, `refactor`, `debug`, language names). Routed to coding-tuned cloud (Cerebras GLM-4.7-coder, Groq). CODE-with-credential queries ("debug this, the api key is sk-…") fail closed via `_is_personal` precedence (security-5, BR5).
- **PUBLIC** — encyclopedic openers (`what is`, `who was`, `how does`) with no personal-data markers. Routed to a cheaper public chain (OpenRouter, DeepSeek V4-Flash, fallback to Cerebras).

The PII detector (`_is_personal`, `core/sensitivity.py:154-170`) is the P0-hardened gate (security-5): any of possessive+noun, personal-action imperatives (`remind me`, `save to`), name+money patterns, PII categories (passwords, SSN, health records, addresses), or street-address shapes triggers PRIVATE. The order matters: if both a CODE marker and a credential appear in the same query, `_is_personal` fires first, routing it to US-only.

`may_leave_device` (`core/sensitivity.py:217-262`) is the egress gate for the web-search surface (BR3): it returns `False` for PRIVATE/MEETING/COMMAND/DICTATION/MEETING_NOTE, blocking any network call to the self-hosted SearXNG backend. Plain public lookups ("weather in Berlin") are permitted *only* when a SEARCH/RESEARCH intent signals the user wants external data. The gate is called *first*, on the raw query (not a trusted `context['sensitivity']` tag, which is only set on the assistant path).

### Web search: SearXNG backend and corpus fallback

`core/websearch.py` registers a `web.search` capability that enforces the order **gate → SearXNG → corpus fallback**. The gate calls `may_leave_device` on the raw query; if denied, or if web search is disabled (`web_search.enabled=false` by default), or if no backend is configured, the corpus answers with `data["egress"]=False`. If permitted, a `SearxngBackend` (lazy `httpx` import) queries `{base_url}/search?q=<query>&format=json` with a bounded connect+read timeout (BR7, ~4 s). Hits are mapped to the corpus shape (`{name, summary}` + `citations`). On empty results, network error, or import error, the fallback returns the corpus result with `ok=True` (a non-ok step aborts the whole plan, so the fallback must never fail), stamped with `data["source"]` / `data["error"]` for audit.

The provider closure never raises and never returns `ok=False`. Web search is opt-in (disabled by default; a user sets `web_search.enabled=true` and configures a `base_url` pointing at their self-hosted SearXNG instance).

### Cloud provider presets and PRC opt-in (Decision 2)

The `llm.cloud_providers` block in `config.json` declares named OpenAI-compatible presets, each with `base_url`, `model`, `api_key_env` (never in config), and a `profile` tag that selects per-vendor quirks:

- **US-hosted (default):** Cerebras (`gpt-oss-120b` $0.50 in / $1.00 out; `glm-4.7-coder` for code), Groq (`gpt-oss-120b` $0.15 in / $0.60 out), OpenRouter (`gpt-oss-120b`, `llama-3.3-70b-instruct`, aggregator for many models).
- **PRC-hosted (opt-in):** DeepSeek (`v4-flash` $0.14 in / $0.28 out; `v4-pro` reasoning), Moonshot Kimi (`k2.6` $0.95 in / $4.00 out, cache-hit $0.16).

The **PRC opt-in gate** (`_build_cloud_client`, `core/llm_factory.py:104-161`) drops any preset with `host=="CN"` unless `llm.cloud.allow_prc=true` is set. The drop is INFO-logged distinctly from a missing-API-key drop (BR8) so a user who forgets `allow_prc` isn't silently degraded to local. US-hosted presets have `"host": "US"` at the top level (OpenRouter) or nested in `_pricing_usd_per_mtok` (existing presets).

### Provider profiles: quirk handling

`core/llm.py::PROVIDER_PROFILES` declares per-vendor quirks layered on top of the generic OpenAI-compatible shape:

- **cerebras** — forbidden params stripped; non-standard params (`clear_thinking`, `reasoning_effort`) routed via `extra_body=`; `max_tokens_cap=8192` (free tier).
- **groq** — `n=1` enforced; `gpt-oss-120b` streams reasoning in `delta.reasoning` (suppressed by default so the assistant doesn't speak the CoT).
- **deepseek** — plain chat (V4-Flash).
- **deepseek_reasoning** — V4-Pro streams `delta.reasoning_content` ahead of `delta.content` (CoT suppressed); the API rejects echoing reasoning on the next turn.
- **moonshot** — `temperature`, `top_p`, `n` are server-fixed (rejected if set).
- **openai_compat** — safe generic fallback.

Reasoning tokens are counted for metrics (`OpenAICompatLLM.last_reasoning_chars`) but not yielded unless `suppress_reasoning_in_stream=False`.

### Cloud chains and sensitivity routing

`llm.cloud_chains` in `config.json` maps sensitivity tiers to ordered failover chains:

```json
"cloud_chains": {
  "private": ["cerebras_gpt_oss_120b", "groq_gpt_oss_120b"],
  "code":    ["cerebras_glm_4_7_coder", "groq_gpt_oss_120b"],
  "public":  ["openrouter_gpt_oss_120b", "deepseek_v4_flash", "cerebras_gpt_oss_120b"]
}
```

Each turn, the capability layer publishes its sensitivity via `capability_context` (a `ContextVar`); `SensitivityRouterLLM._pick()` reads it and returns the corresponding `HedgeLLM` backing. The `llm.cloud_routing.default_chain` (default `"private"`, fail-safe) is used when the context is empty or the selector returns an unknown chain.

Optional **cost/ttft-aware chain ordering** (`smart-routing-5`, `llm.cloud.cost_order=false` by default) stably reorders each chain's presets by documented ttft and $/MTok metadata before the HedgeLLM is built, floating cheaper/faster providers to the front of the failover list. The reorder is fail-safe: malformed input keeps the original order.

### Fully-local default

By default, `llm.cloud.enabled=false`, so no cloud client is built and the runtime returns the raw local main tier (wrapped only by `build_llms`, not by `HedgeLLM` or `SensitivityRouterLLM`). Only when cloud config is populated do the wrappers activate. The always-on capture loop (VAD → partial STT → fast-tier LLM → TTS) runs on-device with no cloud calls; the thinking tier (main planner, research, multimodal) is where cloud optionally races or hedges.

---

## §6 — Memory architecture

The voice assistant remembers through a **backend-neutral Protocol seam** that abstracts away Postgres/pgvector internals. The working memory layer holds recent turns in RAM; tiers (episodic, semantic, summary, profile) are optionally persisted and recalled via the `Memory` contract.

### 6.1 The Memory Protocol

Every memory backend conforms to `always_on_agent.memory.Memory` — a runtime-checkable Protocol with six verbs, no SQL or database details:

- `add(text, tags=())` — ingest (tag = neutral channel: `"user"`, `"assistant_output"`, `"ingested"`, `"meeting"`, etc.)
- `search(query, limit=5)` — semantic recall → neutral `MemoryItem` sequences
- `all()` → recent window (tag-faithful, preserves what was ingested)
- `context_for_llm(query)` — ready-to-prepend block or empty string
- `prune()` — age-TTL retention/eviction; no-op when `_db_available=False`
- `close()` — flush + release resources

Three injection seams type against `Memory`: `always_on_agent.supervisor` (ingests task output), `core.capabilities.assistant()` (ingests user query, reads context), and `core.runtime.VoiceRuntime.__init__` (holds the instance, forwards to `stop()`). The backend choice is transparent; the runtime/brain stay Postgres-free.

### 6.2 SessionMemory (in-RAM default)

`always_on_agent.memory.SessionMemory` is the trivial default for tests and when `DATABASE_URL` is unset:

- **Working-window cap:** `max_items=200` (dropped from front on overflow; Layer-1 RAM working-memory semantics).
- **Keyword recall** (`context_for_llm`): min-overlap relevance gate (≥2 normalized words in common with the query, mirroring Postgres `similarity > 0.6`) + cap at 3 items → injection volume ≈ Postgres top-3. Empty string when irrelevant (no prompt change, zero embedding cost).
- **Stateless:** `prune()` returns 0; `close()` is a no-op. No age-TTL or profile data.

### 6.3 MemoryManagerAdapter (Postgres thin wrapper)

`always_on_agent.memory.MemoryManagerAdapter` wraps the existing Postgres `MemoryManager` (in `utils/memory.py`) behind the Protocol seam, keeping Postgres-isms internal:

**Lazy import:** `from utils.memory import create_memory_manager` happens inside `__init__()` so the brain stays DB-free when no adapter is built.

**Tag-routed add():**
- User/ingested queries → `MemoryManager.queue_user_utterance()` (debounced, cleaned, persisted)
- Assistant output → `MemoryManager.add_message("assistant")` (RAM-only context)
- Meeting notes (R7) → in-RAM ring buffer only unless `meeting_persist=True`

**In-RAM ring buffer (R3):** The adapter keeps its own small `_ring` of raw `(text, tags)` handed to `add()`, so `all()` returns that instead of `MemoryManager.recent_messages` (which drops tags). This ensures tag fidelity: both SessionMemory and MemoryManagerAdapter pass `test_addressing`'s `("ingested",)` tag assertion identically.

**Search and recall:**
- `search()` → wraps `MemoryManager.search_memory()` results (vector sim > 0.6) into neutral `MemoryItem`s with defensive tag extraction (R9: summary rows have no `role`, so tags are `tuple(t for t in (d.get("type"), d.get("role")) if t)`).
- `context_for_llm()` → calls `MemoryManager.get_context_for_llm()` verbatim (top-3 messages + summaries above 0.6 similarity, plus an optional user profile block).

**Retention:** `prune()` calls `MemoryManager.apply_retention()` (summarize-then-evict episodic messages past `episodic_ttl_days`, drop summaries past `summary_ttl_days`, keep user_profile forever). No-op without a live DB.

### 6.4 MemoryManager (Postgres backend)

Lives in `utils/memory.py` and handles all database I/O:

**Three-tier schema:**
- **Layer 1 (working):** `recent_messages` in RAM (capped, warm-started from DB at init).
- **Layer 2 (episodic):** `messages` table (all user speech + optional assistant, ingested via debounced `MemoryWriter`; filtered by `persist_roles=("user",)` by default).
- **Layer 3 (semantic):** `pgvector` embeddings on messages/summaries with per-embedder partial HNSW indexes (unconstrained vector type + `embedding_dim` + `embedder_id` per row, CHECK constraints).

**Long-term:** `summaries` (rolling, folded via `_summary_head` on the background thread) and `user_profile` (durable, deterministic regex at ingest: "my name is X", "call me Z", "I live in Y", "I prefer …", confidence ≥ 0.9).

**Thread-safe I/O:** `psycopg_pool.ConnectionPool` (psycopg3; `min_size=2`, `max_size=5`) replaces the single non-reentrant `psycopg2.connection`. Every call site acquires a short-lived connection via `with self._pool.connection()` so the request thread + writer background thread + summary thread never contend.

**Graceful no-DB degradation:** Constructs safely without `DATABASE_URL` or a live Postgres; `_db_available=False` gates all SQL calls (logs a warning, in-memory-only for that session).

### 6.5 Short-term context aggregation (recent turns)

`core.conversation.build_recent_context(memory, config)` assembles the **immediate conversational history** (this turn, not past sessions) for the answering model to resolve pronouns and anaphora:

- Collects the last `max_turns` (default 6) user/assistant turns from `memory.all()`, filtered to tags `("user",)` and `("assistant_output",)`.
- Excludes ambient/ingested/meeting items and placeholder replies ("Sorry, I don't have an answer…").
- Each turn truncated to `per_turn_chars` (240); the full block capped at `max_chars` (800, dropping oldest turns until it fits).
- Rendered as a `=== Recent conversation (most recent last) ===` block or empty string when disabled/empty.

### 6.6 Configuration (FLAT memory block, R10)

Single-level keys in `config.json` → `memory` so device-profile overrides survive shallow merge:

```json
{
  "memory": {
    "backend": "auto",                    // auto|inmemory|postgres
    "recall_enabled": false,              // Gated; R5 default OFF
    "recall_min_similarity": 0.6,         // MemoryManager threshold
    "recall_max_items": 3,                // Top-3 for injection
    "recall_max_chars": 600,              // Cap on prepend block
    "recent_context_enabled": true,       // Short-term ON by default
    "recent_context_turns": 6,            // Last N turns
    "recent_context_max_chars": 800,      // Total block cap
    "embeddings": false,                  // Enable pgvector
    "max_recent": 20,                     // recent_messages cap
    "profile_enabled": false,             // R8 default OFF, Postgres-only
    "meeting_persist": false,             // R7: meeting notes RAM-only by default
    "episodic_ttl_days": 90,              // Age-TTL for messages
    "summary_ttl_days": 365,              // Age-TTL for summaries
    "save_interval_sec": 240,             // MemoryWriter flush interval
    "min_confidence": 0.55,               // STT confidence floor
    "llm_cleanup": true,                  // Ollama typo fix
    "llm_gate": true,                     // Ollama substantive-content gate
    "cleanup_model": "gemma3:4b",         // Ollama model
    "max_buffer_items": 32,               // MemoryWriter buffer cap
    "dedupe_similarity": 0.92,            // Near-duplicate threshold
    "persist_user_only": true,            // Skip assistant rows
    "save_control_phrases": false         // Skip 'stop' / 'quit' / etc.
  }
}
```

Backend selection (`core.app._build_memory`): if `backend=='postgres'` or (`auto` and `$DATABASE_URL` is set), build `MemoryManagerAdapter` in try/except (catches ImportError/connection), redact connection errors (R12), fall back to `SessionMemory` on failure. Passed to `VoiceRuntime(memory=...)`.

### 6.7 Recall injection workflow

In `core/capabilities.py::assistant()` (lines 260–288):

1. Collect recent turns **before** ingesting the new query (so turn N sees turns 1…N-1, not the fresh query).
2. **Ingest query:** `memory.add(query, tags=("user",))` (R1; makes it findable for future recall).
3. **Fetch recall:** `memory.context_for_llm(query)` if `recall_enabled` → returns a formatted string or `""` (empty = irrelevant, no prompt change).
4. **Fetch recent:** `build_recent_context(memory, recent_cfg)` (cheap keyword filtering).
5. **Float sensitivity:** check each recent turn for private content; float the prompt's `sensitivity` to the most-private (§9.7 — see [§5](#5--llm-tiers-cloud-routing--the-localcloud-boundary-97)).
6. **Compose:** `recall + system + recent` (recall first, system stable in the middle for KV-cache reuse, recent appended).

**Two-layer gate:** config flag `recall_enabled` (short-circuit before any embedding) + relevance gate (backend returns `""` when irrelevant). Never mutates `query` (keeps router/sensitivity inputs clean). Best-effort: errors return empty string, never breaking a turn.

### 6.8 Off-thread producers (R2, R8)

Rolling summary and user profile run **off the bus thread** (never inside `add_message()` / `queue_user_utterance()` synchronously):

**Rolling summary (Layer 2):** When the `recent_messages` token budget (est. `1.3 * word_count`) exceeds `max_context_tokens` and list size > 10, snapshot the older half, trim RAM synchronously (cheap, immediate working-window shrink), and hand the LLM summarize + fold + persist to the background `MemoryWriter` thread (or a one-off daemon thread if no writer). The `_summary_head` accumulates across flushes (rolling, not fragmented). One summary in flight at a time (guarded by `_summary_lock`).

**User profile (Layer 3, Postgres-only):** Deterministic regex patterns (`_PROFILE_PATTERNS`) match "my name is X" / "call me Z" / "I live in Y" / "I prefer …" at ingest time, schedule a background DB write at confidence ≥ 0.9 via `_extract_profile()` (inline cheap regex match + off-thread DB write). Optional fuzzy LLM extraction deferred to P3.

### 6.9 Retention and privacy (R6, R7)

**Age-TTL (apply_retention, invoked at close-time):**
- Episodic `messages` older than `episodic_ttl_days` (default 90): summarize-then-evict (fold into rolling `_summary_head`, persist to `summaries`, DELETE message rows).
- Summaries older than `summary_ttl_days` (default 365): DELETE.
- `user_profile`: never TTL'd (durable).
- Returns the count of rows removed.

**Privacy (§9.7 boundary):**
- Text-only persistence (no audio blobs).
- `persist_roles=("user",)` by default (skip assistant TTS/echoes).
- Meeting notes (R7): routed to in-RAM ring buffer unless `meeting_persist=True` (personal conversations stay local).
- Never log `DATABASE_URL` (R12: use `_redact_db_url` on error messages).
- Continuation turns skip user-memory ingest to avoid double-ingesting the merged prompt.

### 6.10 Lifecycle

**Start:** `VoiceRuntime.__init__` creates/receives `memory` (SessionMemory by default, or MemoryManagerAdapter if `_build_memory` found Postgres). Warm-starts Layer-1 `recent_messages` from DB at init.

**Running:** `supervisor` ingests assistant output (`memory.add(reply, tags=("assistant_output",))`); `capabilities.assistant()` ingests the user query and reads recall/recent context. The MemoryWriter timer flushes debounced utterances on schedule or when the buffer is full.

**Shutdown:** `VoiceRuntime.stop()` calls `memory.close()` → `MemoryManager.close()` → flushes the pending writer + applies retention + closes the pool.

### 6.11 Tests and regression gates

- **Contract test (`test_memory_contract.py`):** Both backends conform to the Protocol; tag fidelity on `all()`; "fact in turn 1 recalled in turn N" on the adapter; default-off latency neutrality (system unchanged); empty recall = no prompt change; max_chars cap; backend selection; no secret-log.
- **Off-thread summary:** `add_message()` returns without invoking the synchronous summarizer (R2).
- **Recall regression:** `build_recent_context()` and `assistant()` end-to-end on both in-RAM and Postgres backends.
- **TTFT gate:** `tools.bench` measures both backends before enabling recall in production.

### 6.12 Deferred to P2b/P3

- **SQLite+sqlite-vec backend:** the Protocol makes it a drop-in; deferred because no consumer needs it now (mobile is Dart, desktop has Postgres).
- **Fuzzy profile LLM:** regex deterministic floor only; soft LLM extraction deferred.
- **Migrations consolidation:** `tools/migrate.py apply` is canonical; `setup_database.py` is a thin wrapper (psycopg3 only, imports only declared deps).

---

## §7 — Real-time quality subsystems

Real-time latency and speech quality depend on stacked subsystems: **two-pass ASR** (streaming zipformer for low-latency partials+endpoint; offline SenseVoice for robust, punctuated finals), **semantic turn-completion endpointing** (commit a final when the user finishes, not after a fixed acoustic timer), the **never-stuck controller** (reap hung tasks and heal on the watchdog tick), **startup pre-warm** (load models before turn 1), **barge-in** (coherence-primary, multi-signal defense against self-interruption), and **STT input-chain quality** (anti-aliased resampling, soft-limit AGC, and optional confidence gating).

### Semantic turn-completion endpointing

The acoustic endpoint (silence-duration rule2, default 0.8 s) was blind to what the user said — it cut off slow speakers mid-thought and added a fixed ~0.8 s tail to turns that obviously ended. `core/endpointing.py` layers a pluggable `TurnCompletionDetector` protocol + `AdaptiveEndpointPolicy` on top:

- **SHORTEN:** when the partial reads as a complete turn, commit early (down to `endpoint_min_silence_sec`, shipped 0.7 s per on-device validation). A **high-confidence** completion (lexical score ≥ `endpoint_high_confidence_score` 0.75 — a normal ending word, never a conjunction/article/filler) commits at the lower `endpoint_high_confidence_floor` (shipped 0.6 s), reclaiming ~110 ms endpoint p50 on the common well-formed turn (on-device A/B 2026-06-01: p50 918→806 ms, no extra splits/truncations; a premature commit is merged back by the continuation layer). The latency win: ~0.8 s → ~0.5 s final wait.
- **EXTEND (bounded):** when it ends mid-phrase ("…and", "…the", "…because"), hold past rule2 up to `endpoint_max_silence_sec` (1.6 s) so a pause isn't mistaken for the end.
- Else the acoustic decision stands.

The **shipped v1 detector** is `LexicalTurnCompletionDetector` — cheap, deterministic, no model: a normalized last-word check against `DEFAULT_INCOMPLETE_ENDINGS` (coordinating conjunctions, articles, fillers, EN + RO). Deliberately conservative: a false-SHORTEN (early commit) is recoverable via the continuation layer (see [§2](#2--the-control-plane-brain-always_on_agent)); a false-EXTEND is not. The protocol also declares `needs_audio`, so the **optional Smart Turn v3 ONNX** (`ProsodyTurnCompletionDetector`, pipecat-ai, ~8 MB, Whisper log-mel + sigmoid, ~15 ms CPU) drops in without re-architecture when `endpoint_detector='prosody'` + `endpoint_prosody_model` + `endpoint_enabled=true`. Both detectors are gated: lexical is the default. The prosody detector is **validated on the user's real voice** (complete 0.74–0.98 vs incomplete 0.01–0.56, margin 0.18, 2026-06-01 via `tools/turn_detect_check`) but is **human-audio only** (flat ~0.97 on TTS, so it can't be A/B'd with the synthetic-user/inject harness); the live floor-lowering on real speech is still pending (gate inventory in [§9](#9--optional--experimental-tiers-gate-inventory)).

The capture loop calls `_decide_endpoint(acoustic_endpoint, partial, silence_sec, samples)` once per block — a pure, side-effect-free decision that is byte-identical to the pure acoustic path when disabled.

### Two-pass ASR (SenseVoice offline refinement)

The streaming zipformer (k2-fsa, `sherpa-onnx`) emits low-latency partials and detects the acoustic endpoint. When `asr_final_backend='sense_voice'` (the shipped default) or `'whisper'` is configured and the model exists, the endpointed utterance audio is RE-transcribed by the offline SenseVoice recognizer (`core/engines/_sherpa_models.py::build_final_recognizer`) for the **final text that reaches the LLM** — robust on run-on/casual speech (it sees the whole utterance, not incremental tokens) and bundling **punctuation + casing + ITN**. The streaming final is used only when (1) the second-pass recognizer is absent or `asr_final_backend=''` forces streaming-only, or (2) the utterance is shorter than `asr_final_min_sec` or the second pass errors / returns empty (graceful fallback). Because SenseVoice output already carries punctuation + casing, `_postprocess_final` (the streaming final's casing/punctuation restore) is **skipped** when the second pass succeeds.

**Shipped default (2026-06-01):** `config.json` ships `asr_final_backend='sense_voice'` at the standard `python -m tools.setup_models --sense-voice` path. Two-pass is **active when the model is present**, **byte-identical streaming-only when absent**. Live-validated on the user's real voice: SenseVoice fixed streaming-model garble (e.g. "HEY IRIC LISTENING TO ME" → "Hey, are you listening to me.") at **~55 ms median second-pass cost** (2 threads) — Whisper was ~2× slower, Moonshine unreliable. It is also English-pinned (`asr_final_language='en'`; auto-detect mis-fired to Chinese on short English). See `docs/asr_two_pass_2026-06-01.md`.

### Never-stuck controller

The supervisor had no wall-clock deadlines — a hung capability (a blocked `generate`, a network read with no timeout) sat "active" forever. `always_on_agent/supervisor.py` + `core/watchdog.py` add:

- **Per-mode task deadlines + reap:** Each `AgentTask` gets a `deadline_at` stamped at `_start_task` from `DEFAULT_TASK_TIMEOUTS` (assistant 25 s, search/command/dictation/meeting 30 s, research 120 s; overridable via `task_timeouts` config; `0` disables a mode). `reap_overdue_tasks()` runs on the watchdog's 1 s tick, cancels and removes any task past its deadline, and republishes `TASK_CANCELLED` to the bus thread (so the supervisor moves on). A reaped turn that would have spoken says "Sorry, that took too long — let's try again." A long-running capability (e.g. the ReAct planner) can renew its deadline via a `renew_deadline` hook in the task context.
- **Watchdog heals:** `StuckWatchdog` has an `on_tick` hook; the runtime wires it to `reap_overdue_tasks()` so a hung task is killed on the watchdog's existing 1 s cadence.
- **ReAct first-token unblock:** An escalated (ReAct) turn does its LLM work inside the planner, which wasn't stamping `LLM_FIRST_TOKEN`, so the watchdog read it as stuck. The planner now fires a `first_token_hook` on its first streamed token; the runtime marks `LLM_FIRST_TOKEN` (idempotent).

The reap mutations happen under `_cancel_lock` exactly like `cancel_all`; `cancel()` only sets an `Event`. Reaped tasks do NOT bump the global speech epoch (that would strand concurrent siblings' TTS); the watchdog also guards all handlers with `try/except` so any future bug degrades to a dropped event, never a dead bus thread.

### Startup pre-warm

Models paid their cold-start cost on turn 1. `core/runtime.py` + `core/engines/sherpa.py` now:

- **Real `engine.warm()`:** Exercises the TTS (synthesize a throwaway "ok" and discard it), the punctuation restorer, and the speaker-ID embedder — the pieces that stay cold until the first *reply* / *final*. ASR/VAD/KWS are JIT'd by the capture loop from the first blocks of ambient audio. All best-effort.
- **Real system prompt:** The runtime warms each local LLM with the capability-aware `system_prompt` it built, so the (now longer) system prefix is prefilled into the KV-cache instead of filled on turn 1's first token. Cloud-hybrid (`HedgeLLM`) only warms its purely-local leg.
- **Gate + cleaner:** The addressing classifier and transcript cleaner run with their own system prefixes; `_warm()` exercises each once so their first live call isn't cold.
- **Readiness signal:** `runtime.warm_ready` (a `threading.Event`) is raised when warm finishes — even if a step failed (raised in a `finally`, so readiness means "warm-up finished"). Set immediately when `warm_on_start=false`.

### Barge-in: coherence-primary multi-signal defense against self-interruption

The assistant keeps the mic open during playback to hear user talk-over. Its own TTS leaks from the speakers into the mic and looks like a barge-in — without a defense the assistant cuts itself off. The detector is **coherence-primary with strongest-first fallbacks**:

1. **Reference-coherence detector (PRIMARY)** (`EchoCoherenceDetector`, `core/engines/echo_coherence.py`; `sherpa.coherence_barge_in_enabled`, default on, needs scipy): Computes the magnitude-squared coherence between the time-aligned TTS reference and the mic over the voiced band (300–3400 Hz), then the **energy-weighted incoherent fraction** — the share of mic energy the reference can't explain. It fires when that fraction clears a **self-calibrating EWMA control chart** (the room's learned echo-incoherence mean + `max(coherence_margin_delta, k·σ)`, σ accumulated on upward excursions only), so the margin auto-widens in reverberant/noisy rooms and tightens in clean ones with **zero per-room tuning**. **Scale-invariant by algebra** — gains cancel in the coherence ratio and the weights, so the same utterance fires identically at any playback/capture volume (`test_decision_is_invariant_to_uniform_volume_scaling`, 100× range). **Structurally never self-interrupts** — the assistant's own echo is fully explained, so the incoherent fraction ≈ 0 at any playback gain (`test_echo_only_never_fires_at_any_playback_gain`). The echo delay is tracked continuously by cross-correlation. When coherence can't decide (no reference yet at session start, or TTS silence) it abstains to the gates below; fails open to the level gate if scipy is missing.
2. **Speaker-ID identity gate** (when enrolled): The `SpeakerGate` (`core/engines/speaker_gate.py`) compares a speaker embedding of detected speech against the enrolled user's voice via cosine similarity (≥ `threshold`, 0.5) — rejecting the assistant's echo by *identity*, independent of volume. Enrollment (`python -m core --enroll`) records a few VAD-trimmed passes and averages the embedding, **pinning `capture_samplerate` to the mic's native rate** (probing 16 kHz first self-mutes USB mics like the AT2020 — measured 2026-06-01: probe-first enrollment self-scored 0.05–0.26 vs 0.76–0.94 when pinned). When unenrolled, the gate fails open.
3. **Acoustic Echo Cancellation (optional):** When `aec_enabled=true`, `core/engines/_aec.py` subtracts the played TTS (the far-end reference, teed from playback into a `FarEndRing` and read at the configured speaker→mic delay) from the mic block *before* any consumer. The shipped backend is the **dependency-free NumPy FDAF** (frequency-domain block adaptive filter, 512-tap, 50% overlap-save): ~10–20 dB real-world ERLE on headsets / near-field rooms. A **deep DTLN-aec ONNX tier** (`aec_backend='dtln'`) is reserved for louder rooms but currently fails open to no-AEC pending tflite→ONNX conversion. Double-talk freeze stops the filter diverging onto the user's voice. `aec_ref_delay_ms` (calibrate with `tools/echo_probe.py`) and `aec_filter_taps` tune per device. When AEC is on, the gate uses the smaller `aec_relaxed_margin_db` (3.0 dB) instead of `barge_in_output_margin_db`.
4. **Loudness / output-margin guard (fallback):** When coherence abstains, the speaker gate is unenrolled, and no AEC is on, `SpeakerGate.accept()` gates on level: detected speech must sit `barge_in_output_margin_db` dB (shipped 6.0) *above* the current playback-buffer RMS. This is device-specific and volume-dependent (mic-capture RMS vs playback-buffer RMS — different scales). Calibrated on a Realtek laptop mic (`tools/echo_probe`): at 0 dB self-interruptions fire 15–21× at every volume; at 6 dB they drop to 0 across 30–100% OS volume. An optional `input_loudness_margin_db` rescue re-admits loud near-field speech whose identity dipped. When playback is silent (`playback_level ≤ 0`), the gate fails open so a real interrupt is never lost.

### STT input-chain quality

The biggest WER leaks are not the model — they're in the input chain before the recognizer. `core/audio_frontend.py` + capture-loop integration + enrollment matching:

- **Enrollment ⇄ capture match:** The enrollment recorder (`core/enroll.py`) and the live capture loop apply identical signal processing in the same order — open the mic at the pinned `sherpa.capture_samplerate` (never probe 16 kHz first → USB-mic self-mute), soft-knee gain, anti-aliased downsample to 16 kHz, then (enrollment only) VAD-trim the silent head/tail — so the enrolled speaker embedding matches what the gate hears live. (See the barge-in section above for the AT2020 self-mute measurement.)
- **Anti-aliased resampler:** The mic on many laptops opens at 44.1/48 kHz only, so every block must downsample to the recognizer's 16 kHz. The old path used `np.interp` linear interpolation with NO anti-alias low-pass, folding content >8 kHz back into the speech band (a real per-block WER hit on fricatives/sibilants). `AudioResampler` (stateful, 0.1 s blocks, carried across resets) prefers `soxr.ResampleStream` (HQ, carries FIR state), falls back to `scipy.signal.resample_poly` (polyphase, stateless), finally `_linear_resample` (linear, no anti-alias). The capture loop applies gain **before** resampling so saturation harmonics are filtered out.
- **Soft-knee AGC:** `apply_gain_soft_limit(samples, gain)` replaces the hard clip (`np.clip(x*gain, -1, 1)`, which on loud phonemes at `input_gain=8` pinned a fraction of samples and produced audible THD). The soft knee leaves levels below the knee (0.8) perfectly linear and saturates peaks smoothly via `tanh`, eliminating square-wave distortion without clipping artifacts.
- **Confidence gate (future):** call `ys_probs`/`tokens` at the endpoint before `reset()`; carry mean/min log-probability on the callback; reject/clarify when `meanLP < ~−0.9 AND words ≤ ~4`. Calibrate on a fixture; prefer the length+confidence conjunction.
- **Decoder levers (future):** `asr_max_active_paths` (shipped 4) can raise to 8 for RTF headroom. Plumb `blank_penalty` and a shallow-fusion LM (`lm`/`lm_scale`, present in sherpa 1.13.2).
- **Model:** Shipped zipformer1. Upgrading to **zipformer2 conversational** (k2-fsa multi-dataset/GigaSpeech) requires updating the model and `tools/bench/models.py` (the live model is still pinned to an older snapshot).
- **Endpoint impact:** Raising rule2 from 0.8 to 1.2–1.6 s fragments fewer short turns but increases latency. The smart endpoint (above) resolves this: commit early on confident-complete turns, extending only on mid-phrase ones.

---

## §8 — Capabilities & self-awareness

The assistant knows what it is and what skills it has, driven by a single **capability manifest** that serves as the source of truth for the controller, the reasoning model, and the runtime reconciliation.

### The manifest

`always_on_agent/capabilities.py` defines `CapabilitySpec`, which describes each registered capability:

| field | meaning |
|---|---|
| `summary` | user/model-facing "what it does" |
| `when_to_use` | planner-facing guidance; falls back to `summary` |
| `egress` | `local` / `cloud` — the §9.7 data-boundary class (see [§5](#5--llm-tiers-cloud-routing--the-localcloud-boundary-97)) |
| `speaks` | produces spoken output (versus a silent side-effect) |
| `side_effecting` | takes an action or changes state |
| `planner_tool` | exposed to the ReAct planner as a callable tool |
| `user_facing` | enumerated in the model's self-description |

`CapabilityRegistry.register(name, provider, *, spec=None)` preserves the existing spec when a provider is re-registered without one — enabling LLM-backed overrides (`assistant.answer`, `research.local`) and the §9.7 `web.search` shadow to keep their metadata. Query via `spec()`, `manifest()`, `planner_tools()`, `describe(names, planner=)`.

### What it drives (preventing drift)

1. **ReAct tool catalog** (`always_on_agent/react.py`) — `_catalog()` renders from `registry.describe(tools, planner=True)` using each spec's `when_to_use`. The hand-typed `_TOOL_DESCRIPTIONS` dict is deleted. `DEFAULT_TOOLS` is asserted equal to `registry.planner_tools()` by test, and `core/runtime.py`'s `_reconcile_capabilities()` warns at startup if a configured planner tool isn't registered (see [§3](#3--the-desktop-runtime-core--the-audioengine-seam)).
2. **The model's self-description** (`core/persona.py`'s `build_system_prompt`) — the answering model's system prompt enumerates user-facing, deliverable skills from the manifest (those with `user_facing=True`). It includes a web-access line that reflects the actual §9.7 egress state: "can search the web" when `web_search.enabled AND base_url` are set, else "have no web access."
3. **Persona** — the optional `assistant` config block (`name`, `persona`, `extra`) gives the model an identity; the ReAct planner's final answer preserves the persona name too.

### Honesty rules (confabulation prevention)

**Silent, mode-gated side-effects are not advertised.** `dictation.clean`, `meeting.note`, and `command.stage` are reachable only via an explicit prefix ("dictate…", "run…") or a prior mode switch chosen deterministically by the analyzer — never by the answering LLM. Advertising them would make the model claim it "took a note" / "ran the command" when the turn was actually a plain text reply with no side-effect. They stay in the manifest (for the planner and reconciliation) with `user_facing=False`.

**Web is gated on real availability.** `web.search` (egress `cloud`) is only listed in the model's "what you can do" block, and the web-access limit line is only used, when `web_search.enabled AND base_url` — matching the condition under which `attach_web_search_capability` actually reaches the network (else it silently falls back to `search.local`).

### Layering

`core/persona.py` owns system-prompt building; `core/capabilities.py` re-exports the byte-identical `DEFAULT_SYSTEM` (pinned by `tests/test_memory_contract.py`). The brain (`always_on_agent`) never imports `core`: the persona name reaches the ReAct planner as a plain string, preserving the core-free facade (see [§2](#2--the-control-plane-brain-always_on_agent)).

### Tests

`tests/test_capability_catalog.py` covers manifest/spec persistence, ReAct-catalog derivation without drift, skill enumeration, §9.7 web gating (unit + e2e runtime), side-effects in the manifest but not advertised, persona identity, `DEFAULT_SYSTEM` byte-identity, and `PersonaConfig.from_dict` robustness.

---

## §9 — Optional & experimental tiers (gate inventory)

| Tier | Config Key / Env | Default | Test File | Status |
|------|------------------|---------|-----------|--------|
| **Capability Router** (unified action/tier decision) | `capability_router.enabled` | OFF (desktop: ON) | `tests/test_capability_router.py` | KEEP, gated |
| **Input Gate** (speaker identity + addressing) | `sherpa.speaker_gate_input` + `input_gate.enabled` | ON (identity) / OFF (addressing; desktop: ON) | `tests/test_speaker_input_gate.py` | KEEP, gated |
| **Smart Turn v3** (prosodic endpoint) | `sherpa.endpoint_detector='prosody'` + `sherpa.endpoint_prosody_model` | OFF | `tests/test_endpointing.py` | KEEP, real-voice scored |
| **SenseVoice two-pass ASR** (offline final recognizer) | `sherpa.asr_final_backend` + `sherpa.asr_final_model` | ON (model present) | `tests/test_asr_final.py` | SHIPPED |
| **Coherence barge-in** (scale-invariant primary) | `sherpa.coherence_barge_in_enabled` (+ scipy) | ON | `tests/test_echo_coherence.py` | SHIPPED |
| **DTLN-aec** (deep echo cancellation) | `sherpa.aec_backend='dtln'` | OFF (default: 'nlms') | `tests/test_aec_seam.py` | KEEP, gated |
| **ReAct Planner** (multi-step planning) | `agent.planner.enabled` | ON (default) | `tests/test_react_planner.py` | KEEP, gated |
| **Continuation** (add-on merging) | `continuation.enabled` | ON (default) | `tests/test_continuation.py` | KEEP, gated |
| **Followups** (proactive listen-ahead) | `followups.enabled` | OFF | `tests/test_followups.py` | KEEP, gated |
| **Live Routing** (dynamic hedge nudging) | `llm.live_routing` or `SPEAKER_LIVE_ROUTING=1` | OFF | `tests/test_core_routing.py` | KEEP, gated |
| **Cost Order** (sensitivity-routed chains) | `llm.cloud.cost_order` | OFF | `tests/test_cloud_providers.py` | KEEP, gated |

### Design & user decision

Every tier below is **kept as a load-bearing gate**. The experimental ones yield **byte-identical behaviour when disabled** — the existing code path runs unchanged, no overhead; the shipped default-on ones (SenseVoice two-pass, coherence barge-in) fail safe (model absent / scipy missing → graceful fallback). The runtime wires each via a deterministic configuration block + optional env override:

- **Capability Router** (`core/capability_router.py`): Disabled default preserves per-gate routing (the tier `Router` + `escalate` predicate stand alone). When enabled, it fuses the decision into one coherent module; the optional fast-LLM assist (`llm_assist=true` + a fast model) only disambiguates low-confidence turns. The desktop profile turns it on (`enabled=true`) with `llm_assist=true` to leverage the extra headroom. (Full composition in [§4](#4--the-decision--routing-layer-the-4-gate-ladder).)
- **Input Gate** (`core/addressing.py`): Disabled default (base config OFF; desktop ON) gates ambient speech silently. Calls the fast tier once per ASR final; on desktop where fast headroom exists, it saves replies to background noise / read-aloud text. (Gate 1 in [§4](#4--the-decision--routing-layer-the-4-gate-ladder).)
- **Smart Turn v3** (`core/endpointing.py` `ProsodyTurnCompletionDetector` + `sherpa.endpoint_detector='prosody'` + `sherpa.endpoint_prosody_model`): Triple-gated — requires `endpoint_detector='prosody'` AND a valid model path AND `endpoint_enabled=true`. Lazy `onnxruntime` import costs nothing on phone; a load error falls back to the lexical detector. Real-voice scored (complete 0.74–0.98 vs incomplete 0.01–0.56) but human-audio only, so the live floor-lowering A/B is still pending. Off by default. (Endpointing detail in [§7](#7--real-time-quality-subsystems).)
- **SenseVoice two-pass ASR** (`core/engines/_sherpa_models.py::build_final_recognizer` + `sherpa.asr_final_backend`): The endpointed utterance is re-transcribed by the offline SenseVoice model for the LLM-facing final. **Default-on** when the model is present (`asr_final_backend='sense_voice'`); byte-identical streaming-only when absent. ~55 ms/utterance; brings punctuation+casing+ITN. (Detail in [§3](#3--the-desktop-runtime-core--the-audioengine-seam) / [§7](#7--real-time-quality-subsystems).)
- **Coherence barge-in** (`core/engines/echo_coherence.py` + `sherpa.coherence_barge_in_enabled`): Scale-invariant reference-coherence detector — the **primary** barge-in signal (default on, needs scipy). Volume-independent by algebra; never self-interrupts; self-calibrating EWMA margin (zero per-room tuning). Falls back to the speaker-ID / loudness gates when it can't decide. (Detail in [§7](#7--real-time-quality-subsystems).)
- **DTLN-aec** (`core/engines/_aec.py`, `build_aec`): `aec_backend='dtln'` would load two ONNX stages (spectral mask + time-domain refine, both carrying LSTM state) but currently fails open to no-AEC when the model paths are invalid or load fails (pending tflite→ONNX conversion; no models shipped). The default `'nlms'` ships the dependency-free NumPy FDAF, byte-identical when `aec_enabled=false`. (Echo defense + calibration in [§7](#7--real-time-quality-subsystems).)
- **ReAct Planner** (`always_on_agent/react.py`): Enabled by default in `config.json` (`agent.planner.enabled=true`); the existing one-shot path runs when a turn doesn't match the `should_escalate` markers (search / look up / find / compare / latest / options / list of / and then / step by step). Bounded to `max_steps=4` with timeout renewal; escalated turns run under the short ASSISTANT budget + extended deadline. (Gate 4 in [§4](#4--the-decision--routing-layer-the-4-gate-ladder).)
- **Continuation** (`always_on_agent/continuation.py`): Enabled by default (`config.json` `continuation.enabled=true`); merges add-ons into in-flight turns deterministically (marker + length heuristic, no LLM). Safe by omission — anything unclear becomes a fresh task. (Mechanics in [§2](#2--the-control-plane-brain-always_on_agent).)
- **Followups** (`always_on_agent/followups.py`): Disabled by default (`config.json` `followups.enabled=false`); cycles a marker cadence so the assistant gently continues after silence. Bounded to `max_followups=3` and not persisted to memory.
- **Live Routing** (`core/routing.py`): Disabled by default (`llm.live_routing=false` or no `SPEAKER_LIVE_ROUTING` env). When on, feeds the tier router an additive-only, clamped live nudge from the recorder's rolling local TTFT EWMA + a cheap SystemMonitor load snapshot; shortens the HedgeLLM's `hedge_delay_ms` so cloud races start sooner when local is slow. A missing/garbage signal is a no-op. (See [§4](#4--the-decision--routing-layer-the-4-gate-ladder) and [§5](#5--llm-tiers-cloud-routing--the-localcloud-boundary-97).)
- **Cost Order** (`core/routing.py::order_presets_by_cost`): Disabled by default (`llm.cloud.cost_order=false`); an optional stable reorder of `cloud_chains` presets by TTFT / $-per-Mtok metadata before the HedgeLLM is built. Fail-safe: same multiset, original order on malformed input. (See [§5](#5--llm-tiers-cloud-routing--the-localcloud-boundary-97).)

Each tier uses the same pattern: disabled → zero cost, enabled → wired deterministically via a config block + optional env / per-device profile. No dead code, no silent fallbacks — the gate inventory is exhaustive.

---

## §10 — Cross-platform contract & mobile

The golden cross-language contract is the linchpin for preventing Python↔Dart runtime drift. It lives in `core/contract.py` (Python) and `mobile/lib/contract.dart` (Dart) — two byte-identical implementations of:

- **Streaming-TTS sentence splitting** (`stream_sentences` / `drain_complete_sentences`): a newline, or a sentence terminator (`.!?`) immediately followed by whitespace. This is what keeps `3.14` and a bare "." from being split mid-stream.
- **Control-command normalization & stop recognition** (`normalize_command` / `is_stop_command`): lowercase, drop non-alphanumeric except spaces, collapse runs, trim. The stop set is `{stop, cancel, quiet, stop talking, be quiet}`; mode/confirm/deny commands are config-driven and desktop-only. (The runtime fast-path that consumes these is in [§3](#3--the-desktop-runtime-core--the-audioengine-seam) and [§4](#4--the-decision--routing-layer-the-4-gate-ladder).)

The contract is **pinned by shared fixtures** in `tests/golden/`:

- `sentence_split.json` — 9 cases covering decimals, newlines, consecutive boundaries, and trailing text.
- `commands.json` — 6 normalize + 9 is_stop cases, including punctuation, extra whitespace, and the canonical stop set.

**Dual-language test runners over the same files:**

- Python: `tests/test_golden_contract.py` (pytest), runs in CI on every push via `.github/workflows/tests.yml`.
- Dart: `mobile/test/golden_contract_test.dart` (flutter test), runs via `.github/workflows/mobile-tests.yml` when `mobile/` or `tests/golden/` changes.

Any change to the contract or fixtures triggers both test suites — the two runtimes cannot silently diverge.

**Mobile usage:** `mobile/lib/assistant.dart` imports `contract.dart` and calls `drainCompleteSentences` in the TTS streaming pipeline (line 385) to emit spoken sentences as they complete while later tokens still generate.

**Remote path:** Two FastAPI endpoints in `remote/token_server.py`:

- `GET /token?identity=&room=` — mints a LiveKit JWT (requires `LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET` in env). Auth: `Authorization: Bearer <token>` matching `SPEAKER_REMOTE_TOKEN`, or dev opt-in via `SPEAKER_REMOTE_ALLOW_NOAUTH=1`.
- `POST /chat {"message": "..."}` — a lightweight text turn through the local LLM (same `core.llm.LLMClient` as the voice path; see [§5](#5--llm-tiers-cloud-routing--the-localcloud-boundary-97)), rate-limited to 30 req/60 s per client.

The live-voice path: `remote/worker.py` is a thin CLI harness that mints a token and forwards to `core.app` with `--engine livekit`. It joins a LiveKit room and uses `core/engines/livekit.py::LiveKitEngine`, which wires sherpa-onnx (ASR/TTS) over WebRTC instead of the local mic/speaker (see [§3](#3--the-desktop-runtime-core--the-audioengine-seam)). The same `VoiceRuntime` orchestrator handles both: the brain is reused, only the audio transport swaps.

**Design decision (from code review §6):** "share the contract + tests, not a binary core." The Python core and Dart mobile shell each re-derive the brain in their own language (`core/capabilities.py` vs `mobile/lib/assistant.dart`'s hot loops for sentence boundaries and the command fast-path). The golden suite (~95% of "one source of truth") is cheaper than FFI/IPC and keeps both shells independent. Drift surfaces immediately in CI.

Mobile is a parallel Dart loop today; the planned convergence onto the `AgentEvent` contract (from `always_on_agent/events.py`, see [§1](#1--system-shape--topology)) is roadmap Phase 2 — it would align the Dart supervisor with the Python one and make the contract duplication disappear (see [§12](#12--known-gaps--roadmap)).

---

## §11 — Observability, testing & device profiles

### Run logs & debugging

Every session writes a committable bundle to **`logs/runs/run-<id>/`** with three independent axes:

- **Always written:** `run-<id>.txt` (full async DEBUG log) + `run-<id>.summary.json` (condensed digest)
- **With `--record`:** `run-<id>.wav` (16 kHz mono audio, replayable)
- **Console verbosity:** `--debug` mirrors DEBUG to the terminal; the file is always full DEBUG regardless

`./session.sh` captures everything (`--debug --record`, sherpa engine; `ENGINE=console ./session.sh` for text-only). Capture is **off the hot path**: logging is fully async (a background `QueueListener` does formatting + disk I/O), recording hands blocks to a writer thread, and telemetry samples on its own thread (10 s interval). So `--debug`/`--record` don't slow the real-time pipeline.

**`run-<id>.summary.json`** has these top-level keys:

- **`meta`** — engine, llm, device, mode, model, fast_model, recording path
- **`stuck_hints`** — plain-English flags from post-hoc checks + live `core/watchdog.py` warnings (LLM/TTS stalls, capture silence, barge-in storms). Start here.
- **`counts`** — llm_requests, turns, transcript_entries, errors, warnings, log_lines_by_level
- **`transcript`** — ordered `[{role: user|assistant, text, mode?, at_sec}]`; `at_sec` is seconds since run start
- **`turns`** — per-turn latencies (seconds): endpoint_latency, final_to_first_token, first_token_to_audio, first_audio_latency, barge_in_latency. `null` means that stage never fired.
- **`llm`** — total_time_sec, avg_time_sec, requests[] with model, prompt_chars, duration_sec, ttft_sec, tokens, streamed, cancelled
- **`system`** — CPU/GPU/RAM telemetry: baseline, final, peak, marks, deltas (final − baseline = run consumption)
- **`errors`** — last 50 WARNING/ERROR records with tracebacks

**`run-<id>.txt`** is the full async DEBUG trace by logger (grep prefixes):

| Logger | Contents |
|--------|----------|
| `speaker.sherpa` | Device names, models, **capture heartbeat with mic RMS** (warns on near-silence), ASR partials/finals, barge-in, playback rate |
| `speaker.runtime` | final→brain, assistant text, TTS requests |
| `speaker.supervisor` | Intent decisions (kind, confidence, reason) |
| `speaker.tasks` | Task start/completed/cancelled/FAILED |
| `speaker.llm` / `speaker.llm.ollama` | Which tier answered, model, **full prompt** (DEBUG), ttft, tokens, cancelled |
| `speaker.sysinfo` | Baseline, periodic, mark, final compute readings |

Bundles are written on clean exit, `Ctrl-C`, and unhandled exception (`finalize()` is idempotent + `atexit`-registered), so crashes still leave artifacts. Test runs write the same shape under **`logs/tests/tests-<id>.{txt,summary.json}`** (disable with `SPEAKER_TEST_LOG=0`).

### Testing & staged runner

**`tools/run_tests.py`** groups the suite into named stages with structured reports under `test-reports/<run_id>/` (with a `latest` pointer):

```
python tools/run_tests.py list      # show all stages
python tools/run_tests.py fast      # quick dev: ~2-4s (skips slow/network/llm/backend)
python tools/run_tests.py core      # runtime + action brain (fast, no models)
python tools/run_tests.py sandbox   # realistic-timing / concurrency sims (slow)
python tools/run_tests.py memory    # smart-memory save/writer + pool contract
python tools/run_tests.py cloud     # cloud LLM: providers, hedge, routing
python tools/run_tests.py imports   # whole-tree import smoke (catches syntax + missing libs)
python tools/run_tests.py e2e       # end-to-end CLI/process tests (subprocess real `core`)
python tools/run_tests.py full      # entire suite
python tools/run_tests.py all-stages # every stage in order with per-stage reports
```

Per-stage reports: `summary.json`, `failures.json` (with tracebacks), `llm-summary.md`. Run-level summary at `test-reports/<run_id>/summary.json`.

**Parallel safety:** the suite is parallel-safe. Run with `pytest-xdist` for ~9 s vs ~35 s serial:

```
python tools/run_tests.py full --pytest-arg=-n --pytest-arg=auto
```

**Postgres integration tests** (`tests/test_memory_postgres_integration.py`, marked `@pytest.mark.postgres`) are collected but skipped by default; opt in with `--postgres`. Requires real PostgreSQL with pgvector.

**Hang guard:** `pytest.ini` sets `--timeout=60 --timeout-method=thread`; any test >60 s is failed with a traceback (thread method tolerates app threads). For legitimately long tests, mark and/or pass `--timeout=N`.

**Test markers** (registered in `pytest.ini`; select with `-m`):

| Marker | Meaning |
|--------|---------|
| `smoke` | Fastest import/config/schema checks |
| `dev` | Critical fast TDD tests |
| `audio` | Deterministic audio, VAD, barge-in, replay, conversation |
| `recorded` | Recorded-session replay |
| `discovery` | Failure-discovery tests (may intentionally fail until bug fixed) |
| `backend` | Optional backend/model integration |
| `slow` | Noticeably slower than the `dev` stage |
| `hardware` | May require real audio hardware |
| `network` | May require network or downloaded assets |
| `llm` | Require a local or remote LLM service |
| `e2e` | Full end-to-end CLI/process (subprocess real `core`) |
| `postgres` | Integration tests needing real PostgreSQL with pgvector |

### Latency instrumentation

**`core/metrics.py`** records per-turn stage timings (seconds, `perf_counter` epoch) shared across the real engine, the file-replay engine, and the sandbox sim engine so measured and simulated numbers land in the same shape and compare against `specsim` budgets. Stages:

- **SPEECH_END** — user stopped speaking (last voiced audio)
- **ASR_FINAL** — recognizer emitted the final transcript
- **LLM_FIRST_TOKEN** — first token streamed from the model
- **TTS_FIRST_AUDIO** — assistant's first audio sample played
- **BARGE_IN** — user spoke over playback
- **BARGE_IN_STOP** — playback actually halted

Deltas (latencies in seconds):
- **endpoint_latency** — SPEECH_END → ASR_FINAL
- **final_to_first_token** — ASR_FINAL → LLM_FIRST_TOKEN
- **first_token_to_audio** — LLM_FIRST_TOKEN → TTS_FIRST_AUDIO
- **first_audio_latency** — SPEECH_END (or ASR_FINAL) → TTS_FIRST_AUDIO
- **barge_in_latency** — BARGE_IN → BARGE_IN_STOP

The recorder is lock-guarded; the hot path computes nothing beyond a dict write.

### Device profiles

`device_profiles` (`device` + selectable via `--device <name>`) are shallow-merged over the base config per section, so a profile only states what differs. Config loads and profile merge in `core/app.py`.

| Profile | LLM backend | Models | Notes |
|---------|-------------|--------|-------|
| `desktop` | `ollama` (GPU) | `gemma3:12b` main + `gemma3:4b` fast | Ollama is desktop-only |
| `phone` | `llamacpp` (GGUF) | `gemma-3-4b` main + `gemma-3-1b` fast | `n_gpu_layers: 0`, `n_ctx: 2048`, STT/TTS threads dialed down; simulates phone-like limits on the Python core |

**The shipped Flutter app** (`mobile/`) is a separate shell and uses `flutter_gemma` (MediaPipe/LiteRT), not these profiles (see [§10](#10--cross-platform-contract--mobile)).

### Engines (transport choice)

`--engine` selects the `AudioEngine` (`core/engine.py`; see [§3](#3--the-desktop-runtime-core--the-audioengine-seam)):

- **`console`** — typed I/O, no audio/models (tests + demo)
- **`sherpa`** — on-device mic/speaker via sherpa-onnx (the local product path)
- **`replay`** — the real pipeline over recorded `.npy`/`.wav` fixtures, headless (no sound card); used for latency benchmarks and CI
- **`livekit`** — audio over a LiveKit/WebRTC room (the remote host+thin-client path)

### Spec simulation & bench

**`tools/specsim/`** renders `test-reports/specsim/index.html` — a model-fit + responsiveness matrix + per-device ASR→LLM→TTS timelines (4090/Mac/Windows/phone/web). Numbers are modelled estimates, not measurements; calibrate `tools/specsim/specs.py` from real runs. Real-model latency benchmark: `python -m tools.bench --fake` is a no-download smoke test; `python -m tools.bench --profile phone --fixtures tests/fixture_audio/virtual_real_world` fetches small Gemma GGUF + sherpa ONNX and runs the REAL ASR→LLM→TTS pipeline over fixtures, writing a measured-vs-specsim report under `test-reports/perf/`.

### Preflight & setup

- **`python -m tools.doctor`** — checks Python, imports, sherpa model paths, Ollama reachability + models, audio devices; prints exact fix commands
- **`python -m tools.setup_models`** — downloads sherpa ASR/VAD/TTS models and wires paths into `config.json`
- **`./install.sh`** — one-command native install (venv + deps + models + doctor); includes `psutil` (telemetry); GPU telemetry uses `nvidia-smi`
- **Quick sysinfo without a run:** `python -m core.sysinfo`

### Source map

- `core/runlog.py` — async logging (`QueueHandler`/`QueueListener`), `RunSummary` aggregation, crash-safe `finalize()`, watchdog-warning → `stuck_hints` promotion
- `core/watchdog.py` — live `StuckWatchdog`: per-second inspection of metrics anchors + heartbeat + barge-in rate; logs WARNINGs on stage stalls (see [§7](#7--real-time-quality-subsystems))
- `core/recorder.py` — background-threaded WAV writer (`WavRecorder`)
- `core/sysinfo.py` — `snapshot()` + `SystemMonitor` (CPU/GPU/RAM)
- `core/engines/sherpa.py` — capture/playback instrumentation + recorder hook + heartbeat callback into the watchdog
- `core/metrics.py` — per-turn stage timings shared across engines + sim
- `tools/run_tests.py` + `tools/testing/` — staged pytest runner with per-stage reports
- `tools/specsim/` — spec simulator + HTML capability report
- `tools/doctor.py`, `tools/setup_models.py`, `install.sh`, `session.sh`
- Tests: `tests/test_runlog.py`, `tests/test_watchdog.py`, `tests/test_recorder.py`, `tests/test_sysinfo.py`, `tests/test_setup_doctor.py`

---

## §12 — Known gaps & roadmap

This section reflects findings from five audit reports (production hardening, perf goal alignment, code review, architecture synthesis, and voice quality improvement), all re-verified against the current codebase (2026-06 HEAD).

### Status of recent user decisions

**Experimental tiers retained and gated** (full inventory in [§9](#9--optional--experimental-tiers-gate-inventory)):
- **DTLN-aec** (`aec_backend='dtln'`) and **Smart Turn v3** (`ProsodyTurnCompletionDetector`) are both implemented and gated behind default-off flags. DTLN currently fails open to the NumPy NLMS AEC (always available, no new dependencies) because the tflite→ONNX conversion is pending. Smart Turn is **scored on the user's real voice** (complete 0.74–0.98 vs incomplete 0.01–0.56) but human-audio only (flat on TTS), so the live floor-lowering A/B is still pending; the lexical detector (`LexicalTurnCompletionDetector`) ships as the default (verified `core/endpointing.py`). **SenseVoice two-pass ASR** (`asr_final_backend='sense_voice'`) and the **scale-invariant coherence barge-in** (`coherence_barge_in_enabled`) are both shipped default-on (model present / scipy present), failing safe to streaming-only / level gates respectively.
- LLM warm-up is real: `core/runtime.py` accepts a `warm_on_start` param (true in `config.json`, pre-warms both tiers at startup) and ASR/TTS warm on sync (verified `core/engines/sherpa.py`). The first turn is no longer cold on desktop (see [§7](#7--real-time-quality-subsystems)).
- `capability_router` is opt-in (default-off in base, enabled in the `desktop` profile with `llm_assist=true` for on-device disambiguation; see [§4](#4--the-decision--routing-layer-the-4-gate-ladder)).

**Module cleanup completed:**
- `always_on_agent/adapters.py` does not exist (confirmed: file not found).

**Load-bearing integrations verified:**
- `always_on_agent/planner.py` is load-bearing (imported by `tasks.py`; planner and react are complementary reachability paths, not duplication; see [§2](#2--the-control-plane-brain-always_on_agent)).
- `always_on_agent/app.py` is a real documented CLI harness (entrypoint `always_on_agent.app:run_demo`), not dead.
- `core/routing.py` (tier router) and `core/capability_router.py` (unified router) compose: capability_router is opt-in, byte-identical legacy behavior when off; both feed the same executor in `core/runtime.py` via the `CapabilityTierRouter` bridge (see [§4](#4--the-decision--routing-layer-the-4-gate-ladder)).
- `core/capabilities.py` (LLM-backed provider impls) vs `always_on_agent/capabilities.py` (core-free registry/mechanism) is interface-vs-implementation; this separation protects mobile/remote reuse.

### Verified corrections — stale claims not to re-raise

1. **`tools/stress.py scn_real` is fine.** The stress harness constructs the runtime with `warm_on_start=True`; `core/runtime.py` declares it as a real param. The earlier "TypeError / unreproducible baseline" claim is stale; do not carry it.
2. **Endpoint latency is now measured.** `SPEECH_END` is stamped at silence-onset (pre-wait) in current code, not post-wait; `endpoint_latency` is no longer systematically 0.000 (fixed per perf-audit roadmap #H1, verified in metrics).
3. **Smart Turn v3 is real-voice scored, not unvalidated.** The model is gated and shipped (never breaks capture — falls back to lexical). Real-voice scoring (2026-06-01 via `tools/turn_detect_check`): complete turns 0.74–0.98, incomplete 0.01–0.56 (margin 0.18). It is human-audio only (flat ~0.97 on TTS), so the synthetic-harness A/B is blocked; the live floor-lowering on real speech is the remaining step.

### Verdicts by dimension (phased roadmap)

**Phase 0 — Production safety (landed):**
- ✅ Fail-safe input gate + loud warning when off.
- ✅ Post-speech ASR cooldown (~0.25 s) + VAD reset on speaking→listening transition.
- ✅ `psycopg_pool` test gate fixed (pool tests skip-or-pass; full suite green, 848 passed).
- ✅ First-turn LLM pre-warm (async), sherpa ASR/TTS warm (sync pre-thread).
- ✅ Watchdog false "llm stuck" on intent fast-path turns suppressed.
- ✅ Metrics `first_token_to_audio` guards negative deltas.

**Phase 1 — Reliability & first-impression latency (in progress):**
- 🔄 **Move ASR decode off the capture thread** — not yet done; the phone-class cliff remains. Stress tests show RTF 0.07–0.10 on 4090 (adequate today; deferred pending CPU-constrained testing).
- ✅ **Move addressing + cleaner off the capture thread** — done via async scheduling in `runtime.py`; the capture thread is no longer stalled.
- ✅ **True speech-stop metric stamp** — `SPEECH_END` now stamped at VAD silence onset (the prerequisite for endpointing tuning landed).
- 🔄 **Bounded local LLM generate** — `LlamaCppLLM` is still unbounded on phone, but desktop `OllamaLLM` has a 60 s wall-clock (closes the hang-forever case).
- 🔄 **Surface FATAL capture state** — currently logs + publishes only; a spoken/visible notice + a bounded outer re-bring-up loop are still open.

**Phase 2 — Best-in-class live voice-to-text (next frontier):**
- 🔄 **Semantic end-of-turn model** — Smart Turn v3 ONNX (`ProsodyTurnCompletionDetector`) built, gated, and **scored on the user's real voice** (complete 0.74–0.98 vs incomplete 0.01–0.56, margin 0.18, 2026-06-01). Human-audio only (flat on TTS), so the live floor-lowering A/B on real speech is still pending. Measured endpoint cost ~0.6 s (true SPEECH_END stamp); Smart Turn promises ~0.3–0.55 s.
- ✅ **Two-pass final ASR** — SenseVoice offline second pass shipped default-on (2026-06-01): streaming zipformer → offline final, ~55 ms, robust punctuated text on run-on/casual speech.
- ✅ **Coherence barge-in** — scale-invariant reference-coherence detector shipped as the primary barge-in signal (2026-06-02): volume-independent, never self-interrupts, self-calibrating margin; legacy speaker-ID / loudness gates are the fallbacks. Field A/B with overlapping real speech still pending.
- ⏸️ **Acoustic echo cancellation** — the NumPy FDAF AEC (`aec_backend='nlms'`, ~10–20 dB real-world ERLE) is implemented; the DTLN-aec deep tier is gated, pending tflite→ONNX conversion. **Both are OFF by default** (`aec_enabled=false`); when enabled, per-device `aec_ref_delay_ms` calibration is required (`tools/echo_probe.py`). The desktop path defaults to no-AEC — the coherence detector (primary), the 6 dB output-margin guard, and speaker-ID enrollment are the stopgaps (see [§7](#7--real-time-quality-subsystems)). Mobile enables platform AEC (`mobile/lib/assistant.dart`).
- ⏸️ **Preemptive/speculative LLM generation** — the architecture supports it (epoch-stamped cancellable tasks, existing cancel machinery); never implemented. Overlaps TTFT with trailing speech.
- ✅ **Fix the replay twin** — `_postprocess_final` is now applied in both sherpa and file_replay; the bench/replay twin emits the same cased+punctuated text as production.

**Phase 3 — Quality & hygiene (lower priority):**
- ⏸️ Anti-aliased resampling shipped; further WER work gated behind a WER harness if pursued.
- ✅ Watchdog intent-turn false positive — suppressed (metrics turn-banking fixed).
- ⏸️ De-nest `apply_retention` pool deadlock — not yet done (P2 mitigation, low urgency).
- ⏸️ Gate the per-connect demo DDL — not yet done; startup still runs an ungated schema bootstrap.
- ⏸️ Redaction/TTL + default-ignore for run bundles — not yet done; transcripts + WAVs are still committable.
- ⏸️ Case-insensitive name+money PII — not yet done (the case-sensitive check remains a narrow edge).
- ⏸️ Recorder drop surfacing + `_playback_level` lock — not yet done (self-heals within ms, low priority).

### Open high-value work (not yet scheduled)

Ranked by field impact:

1. **Smart Turn v3 live floor-lowering A/B (on real speech)** — the prosody model is already real-voice scored (0.74–0.98 vs 0.01–0.56); what remains is validating the actual latency win of dropping `endpoint_min_silence_sec` on high-confidence prosody. The biggest real latency win (~0.3–0.55 s) and the only way to test mid-thought cut-off reduction. Human-audio only, so it can't use the inject harness — needs live recordings (`python -m core --engine sherpa`, then read `logs/runs/*.summary.json` endpoint latency). **(M effort, blocked on hardware access)**
2. **Echo probe ERLE calibration** — `aec_ref_delay_ms` (default 80 ms) and `aec_filter_taps` (default 512) need per-device/room tuning via `tools/echo_probe.py` to hit measured ERLE and minimize self-interrupt risk; the 6 dB `barge_in_output_margin_db` level fallback is validated on one Realtek laptop. Re-calibrate on new hardware. **(M effort, requires field data + echo-probe measurements)**
3. **Tier escalation evidence** — `router.threshold=0.3` + `_GENERATION_MARKERS` exist in the working tree; no recorded session yet shows a real gemma3:12b turn on the 4090. **(S effort, measurement only)**
4. **Phone-class latency validation** — all audited runs are desktop/4090/Ollama. The phone profile (llama.cpp GGUF, unbounded LLM timeout, blocking input_gate on capture) has never been benchmarked. **(M effort, requires phone hardware)**
5. **Multi-participant `remote/` test** — barge-in calls a global `supervisor.cancel_all()` with no session scoping; two participants would cancel each other's work. No test exists. **(M effort, coverage gap)**
6. **LLM egress gate for injected PII** — recall + profile blocks must not carry personal data off-device via cloud LLM chains. A gate exists for web.search; the LLM path is still unvetted. **(M effort, latent behind disabled-by-default cloud; see [§5](#5--llm-tiers-cloud-routing--the-localcloud-boundary-97))**
7. **Run-bundle PII/retention policy** — transcripts + WAVs commit verbatim with no redaction/TTL. **(S effort, hygiene)**

### Known latent risks (not blockers yet)

- **Barge-in detection is unvalidated in the field.** Stress tests and live replay show zero real LLM cancellations (only echo self-interrupts). The cut machinery is correct; the decision path is untested with real overlapping speech.
- **The ASR confidence signal is not yet gated.** Low-confidence finals (`ys_probs` < threshold) feed the LLM unfiltered. Combined with the echo loop, this can drive TTS on garbage fragments.
- **Full-pipeline resident memory** (~800 MB with Ollama; ~538 MB ASR/TTS only) does not leak (verified via stress) but is the sizing baseline for deployment.
- **Capacity counts liveness, not progress.** A thread wedged forever in a capability call counts as healthy; 6 such wedges would report "at capacity, queue the rest" while doing zero work. Not yet observed (peak concurrency 3/6) but untested under a non-returning capability.

### Explicitly out of scope

- **End-to-end speech-to-speech models** (Moshi/GPT-4o realtime) for the always-on loop — cascaded ASR→LLM→TTS is the 2025–26 consensus for tool use, content filtering, and on-device control. If adopted at all, route to the optional cloud thinking tier; document in `docs/target_architecture.md` §9.7.
- **Mobile Rust/FFI core** — defer until drift hurts or iOS-background forces it (Phase 3 roadmap). The current ~4-function pure-function sharing + golden contract suite is adequate for now (see [§10](#10--cross-platform-contract--mobile)).
- **OSS segmentation library** — the current regex is adequate; revisit only if the golden suite surfaces real edge-case failures.

---

## §13 — Decisions log

**An append-only chronological record of locked architectural and implementation decisions, including rationale.** Later phases may supersede earlier entries; the live authority for open items lives in `docs/ultracode_scope.md` (phase pipeline) and `docs/target_architecture.md` (structural decisions). Resolved P0–P5 scope, P2 R1–R12 revisions, P3 BR1–BR8 revisions, and current-session bindings are collected here so archived source docs can retire without losing context.

### Structural decisions (decision record in `docs/target_architecture.md` §9)

1. **Core + thin shells, not a monolith or N independent apps.** One portable Python core (`core/`); per-platform shells (Flutter on Android, web+host path for remote). The `always_on_agent` `AgentEvent`/`Mode` contract (+ tests) is shared by all platforms; the brain is reimplemented faithfully per runtime, not binary-shared. *Rationale:* iOS forbids Python background voice; N apps duplicate logic. `docs/target_architecture.md` §0 (2026-05).

2. **Local-first hybrid boundary (§9.7).** The always-on capture loop (STT/TTS/VAD/speaker-ID + fast LLM) runs fully on-device; raw audio never leaves. The thinking tier (main planner, research, multimodal, web search) may use cloud; only post-ASR text, files, and screenshots cross, and only when invoked. Device profiles tune the local/cloud split per machine class. *Rationale:* preserves privacy, unblocks iOS, allows graceful fallback. `docs/target_architecture.md` §9.7 (2026-05-28).

### Phase P0 — Security & OSS-readiness (2026-05-29)

3. **P0 scope:** auth on `/token`, stop echoing `DATABASE_URL`, remove dev-key defaults (LiveKit), add `LICENSE` + attribution, harden the PII gate, pin the web SDK. *Rationale:* ship-blocker; an unauthenticated remote surface + exposed secrets + missing license violate OSS and cloud safety bars. `docs/review_ultracode.md` § Goal 1 (2026-05-29).

4. **OpenRouter + US-default, PRC opt-in (Decision 2).** Add OpenRouter (`openrouter_*` presets) behind the existing `OpenAICompatLLM`. US-hosted chains (`openai/gpt-oss-120b`, `meta-llama/llama-3.3-70b-instruct`) default; DeepSeek/Moonshot (CN-hosted) are gated behind `allow_prc=false` default, with a silent drop + INFO log on absence. *Rationale:* one key, many models; aligns with the §9.7 data boundary; resolves security-5. `docs/review_ultracode.md` security-5; `docs/ultracode_scope.md` Decision 2 (2026-05-29).

### Phase P1 — Real-time correctness (2026-05-29)

5. **Barge-in + cancellation correctness.** Set `cancel_event` before `stop_speaking()` returns; deterministically drop a stale `TTS_REQUEST` on an active task_id. Convert ReAct plan/step `generate()` to cancel-aware streaming. Bounded idle timeout on the Hedge final-drain. Output-activity barge-in suppression. *Rationale:* a stale sentence post-interrupt + non-cancellable escalated turns degrade the most-distinguishing feature. `docs/review_ultracode.md` realtime-concurrency-1/2 (2026-05-29).

### Phase P2 — Layered memory seam + wiring (2026-05-29)

6. **Memory Protocol seam (P2).** One backend-neutral `Memory` Protocol in `always_on_agent/memory.py` with `add/search/all/context_for_llm/prune/close`. `SessionMemory` = trivial in-RAM default (cap + relevance gate). `MemoryManagerAdapter` wraps the tested Postgres `MemoryManager`. *Rationale:* highest leverage — simultaneously unblocks Goal 2 (memory was unwired), wires P2, makes SQLite-on-mobile a drop-in later. `docs/p2_memory_design.md` §1-3 (2026-05-29).

7. **SQLite defer, Postgres first (P2).** Ship only `SessionMemory` + Postgres adapter + Protocol this cycle. The Python SQLite+sqlite-vec backend (for mobile) is deferred; the Protocol makes it a same-interface drop-in when mobile needs it. *Rationale:* no Python consumer needs it now; migrations are Postgres-only; net-new. `docs/p2_memory_design.md` §9 (2026-05-29).

8. **Memory backend selection.** `_build_memory(config)`: if `backend=='postgres'` or (`auto` + `DATABASE_URL` set) → try the adapter (ImportError/connect → fall back to `SessionMemory`); else `SessionMemory`. Desktop defaults to in-RAM unless `DATABASE_URL` is set. Redact connection errors (never echo the DSN). *Rationale:* graceful degradation; portable desktop default; secrets stay off stdout. `docs/p2_memory_design.md` §8; `docs/ultracode_scope.md` Decision 4 (2026-05-29).

9. **Recall injection (assistant closure only, R1, R5).** The query is ingested as `memory.add(query, tags=("user",))` before the recall prepend; recall is only prepended in the `assistant` closure (the RESEARCH path is excluded, R5). Two-layer gate: config flag + relevance (similarity>0.6 self-gate; '' = no change). Capped top-3 + `max_chars` bound. *Rationale:* solved "recall retrieves nothing" (R1 critical); prevents prompt-quality regression from unmeasured research recall. `docs/p2_memory_design.md` R1, R5 (2026-05-29).

10. **Off-thread rolling summary (R2).** The summary is scheduled on the `MemoryWriter` background thread, never inline in `add_message`. The profile producer is also off-thread, **Postgres-only** (R8). Fail-safe to the lexical endpoint if the module is unavailable. *Rationale:* R2 HIGH — a sync summary in the bus thread would stall TTS/barge-in. `docs/p2_memory_design.md` R2, R8 (2026-05-29).

11. **Setup migration consolidation (Decision 5).** `tools/migrate apply` is canonical (idempotent/additive, reconciles legacy DBs). `setup_database.py` is reduced to a thin `create_database()` + `verify_setup()` wrapper, ported to psycopg3 (R4). `setup.sh` + `SETUP.md` are updated to the single schema path. *Rationale:* one source of truth; removes schema SQL duplication; unblocks tests (R4 import smoke). `docs/p2_memory_design.md` R4, §10; `docs/ultracode_scope.md` Decision 5 (2026-05-29).

### Phase P3 — Real web research + cost-controlled cloud streaming (2026-05-29)

12. **Self-hosted SearXNG + pluggable web.search (Decision 3).** New `core/websearch.py` module; `SearxngBackend` (lazy `httpx` import) behind a `Backend` Protocol. Register the provider at `core/runtime.py:96`. Gate: SearXNG → corpus fallback (never raises). The result mirrors the corpus shape; the audit stamps `egress`/`sensitivity`. *Rationale:* self-hosted egress stays under user control; pluggable allows Tavily/Brave later; corpus fallback when unreachable. `docs/p3_design.md` §1 (2026-05-29).

13. **Sensitivity egress gate (§9.7, BR3).** New `core.sensitivity.may_leave_device(query, *, mode, intent_kind)` predicate: **PRIVATE (PII/possessive) blocks egress; COMMAND/DICTATION/MEETING intents block egress; else permit to SearXNG.** Guarded enum coercion (BR2 HIGH); fail-safe to PRIVATE on a bad `context["mode"]`/`intent_kind`. PRIVATE ⇒ corpus fallback with `egress=False`. *Rationale:* preserves the §9.7 boundary; blocks credential/personal egress; permits plain public queries ("weather in Berlin") to the self-hosted lookup. `docs/p3_design.md` BR2, BR3 (2026-05-29).

14. **Hard close + socket timeout (BR1).** `OpenAICompatLLM.stream` binds `sdk_stream` and closes it in `finally` (replaces lazy GC close). Plumb `llm.cloud.timeout_s` (5 s default, low single digits) to `OpenAICompatLLM(timeout=)` so a losing pre-TTFT worker is reaped fast (stops billing). *Rationale:* BR1 HIGH — a blocking first-token read holds the socket + billing until the client timeout (30 s default). `docs/p3_design.md` BR1 (2026-05-29).

15. **Per-turn max_tokens ceiling (BR4).** Add to `OpenAICompatLLM.__init__`; inject into `merged` (before the profile cap via `min()`) from `llm.cloud.max_tokens` (default `None`). *Rationale:* BR4 low; composes under the profile cap authority. `docs/p3_design.md` BR4 (2026-05-29).

### Current session — experimental tiers + unified doc structure (2026-06-02)

16. **Experimental DTLN-aec tier kept, gated, default-off.** `aec_backend='dtln'` selects the NumPy-based DTLN echo canceller (vs the nlms default). Requires a separate model fetch. Fail-safe: if unavailable or on error, fall back to nlms. *Rationale:* AEC is an experimental hyperparameter space; shipping a fallback chain lets measurement-driven rollout proceed without breakage. Current `config.json` shows `aec_backend: "nlms"` (default on).

17. **Smart Turn v3 (`ProsodyTurnCompletionDetector`) kept, gated, default-off — now real-voice scored.** The prosodic turn-completion model (Whisper log-mel + sigmoid, ~8 MB ONNX, ~15 ms CPU) replaces the cheap lexical endpoint when enabled. Triple-gated: needs `endpoint_enabled=true` AND `endpoint_detector='prosody'` AND a valid `endpoint_prosody_model` path; otherwise the lexical detector is used. Requires onnxruntime (lazy import, desktop-only load-error fallback). Real-voice scoring 2026-06-01 (`tools/turn_detect_check`): complete 0.74–0.98 vs incomplete 0.01–0.56 (margin 0.18); human-audio only (flat on TTS), so the live floor-lowering A/B is still pending. *Rationale:* the model now has real-voice evidence; the gate + fallback prevents capture-path breakage. Current `config.json` shows `endpoint_detector: 'lexical'` (default); `endpoint_enabled: true`; `endpoint_prosody_model: ''` (set in `config.local.json`).

18. **Unified architecture doc + two-pole linking.** `docs/unified_architecture.md` (this doc) absorbs durable truths from ~14 subsystem docs and links authoritatively to `docs/target_architecture.md` (§9, §9.7, §10 structural decisions — north star) and `docs/architecture.md` (as-built snapshot). The decisions log (§13) migrates P0–P5 scope, P2 R1–R12, P3 BR1–BR8, and current bindings so the source docs can retire. *Rationale:* avoids stale claims from archived docs; preserves decision context without duplication.

19. **`AgentEvent` → `AgentBrainEvent` rename (the private `core/agent.py` class).** The private action-brain bridge class in `core/agent.py` is renamed `AgentBrainEvent` to eliminate the collision with the PUBLIC `always_on_agent.events.AgentEvent` platform contract. `AlwaysOnAgentRuntime` is NOT renamed (it remains the public runtime handle). *Rationale:* naming clarity — `AgentEvent` is the durable platform contract (never renamed), `AgentBrainEvent` is internal plumbing. DONE this session (`core/agent.py` now defines `AgentBrainEvent`; `tests/test_core_agent.py` green).

20. **Dead module removal: `always_on_agent/snapshots.py`.** The stale snapshot fixture module is removed; `.agents/backlog.md` is refreshed with current P0–P5 scope and session notes. *Rationale:* `snapshots.py` is imported nowhere; test fixtures now live in `tests/golden/` per the P5 plan. DONE this session via `git rm` (verified: no importers; full suite green at 1205 passed, 10 skipped).

### Composition & invariants

21. **Multi-tier routing composition.** `core/routing.py` (tier router: fast vs main LLM) and `core/capability_router.py` (unified action router: CONTROL/SIMPLE/RESEARCH/ACT) are **complementary composition layers, not duplication**. The capability router (opt-in, off in base, on in the desktop profile) composes the tier router; byte-identical legacy behavior when off. *Rationale:* clean separation of concerns; smart-routing-2/3 avoids over-merging. `docs/review_ultracode.md` smart-routing-1; VERIFIED: `always_on_agent/tasks.py:265-275` invokes both. (2026-05-29). (See [§4](#4--the-decision--routing-layer-the-4-gate-ladder).)

22. **Memory Protocol + capabilities interface seam.** `core/capabilities.py` (LLM-backed provider impls) vs `always_on_agent/capabilities.py` (core-free registry/mechanism) is an intentional interface split across the shell↔core boundary. `always_on_agent` stays core-free (no `import core`) so mobile/remote facade reuse is unbroken. *Rationale:* a load-bearing invariant for mobile (the Dart shell would break if core leaked into always_on_agent). VERIFIED: `always_on_agent/react.py:9-12` documents this; `always_on_agent/` never imports `core/`. (2026-05-29). (See [§2](#2--the-control-plane-brain-always_on_agent), [§8](#8--capabilities--self-awareness).)

23. **Planner + ReAct complementary, not duplicates.** `always_on_agent/planner.py` (explicit TaskPlans) and `always_on_agent/react.py` (bounded ReAct loop with tools) are **both loaded and active** — the planner routes to react for RESEARCH/ACT; react supplies the DEFAULT_TOOLS fallback. *Rationale:* clean task scoping; planner explicit, react the bounded-exploration fallback. VERIFIED: `always_on_agent/tasks.py:229-231` branches on the plan step's `ok`; react tools at `:157` supply the scope. (2026-05-29). (See [§2](#2--the-control-plane-brain-always_on_agent).)

24. **`always_on_agent/app.py` is a real CLI harness, not dead.** A console REPL for manual agent testing and brain validation; used in the sandbox middle-layer tests. *Rationale:* dev/test utility; not shipped but actively used. VERIFIED: imported by `tests/test_sandbox_middle_layer.py`. (2026-05-29).

25. **Pre-warm is real (`tools/stress.py scn_real`).** `core/runtime.py` declares `warm_on_start` as a live param (landed after the perf audit; `config.json` shows `warm_on_start: true`). The `tools/stress.py scn_real` baseline is correct. *Rationale:* the cold-load penalty eats latency headroom; turn-1 TTFT matters. VERIFIED: `core/runtime.py` param; `config.json` `warm_on_start: true` default. (2026-05-29). (See [§7](#7--real-time-quality-subsystems).)

26. **`always_on_agent/adapters.py` does not exist.** Never reference it. (2026-05).

### Voice-stack decisions (2026-06-01 / 2026-06-02)

27. **SenseVoice two-pass final ASR, shipped default-on (2026-06-01).** The streaming zipformer gives partials + the acoustic endpoint; the endpointed utterance is re-transcribed by the offline SenseVoice model (`asr_final_backend='sense_voice'`) for the LLM-facing final. ~55 ms/utterance (2 threads); brings punctuation + casing + ITN, so `_postprocess_final` is skipped on the second pass. Byte-identical streaming-only when the model is absent (`build_final_recognizer` → None). English-pinned (`asr_final_language='en'`). *Rationale:* the streaming-only zipformer garbles run-on/casual speech; SenseVoice is accurate, sherpa-native, and cheap (Whisper ~2× slower, Moonshine unreliable). Live-confirmed on real voice (`docs/asr_two_pass_2026-06-01.md`).

28. **Coherence barge-in as the primary signal (Phase 1, 2026-06-02).** `EchoCoherenceDetector` (`coherence_barge_in_enabled`, default on, needs scipy) measures magnitude-squared coherence between the TTS reference and mic over the voiced band and fires on the energy-weighted incoherent fraction above a self-calibrating EWMA baseline. Volume-independent by algebra; structurally never self-interrupts; zero per-room tuning and zero enrollment. Falls back to speaker-ID / AEC / loudness gates when it abstains. *Rationale:* the user brief was "smart, not loudness-based; any volume; zero setup" — coherence is the only detector meeting all three (`docs/barge_in_coherence_2026-06-02.md`).

29. **Adaptive confidence-tiered endpoint floor (2026-06-01).** A high-confidence lexical completion (score ≥ `endpoint_high_confidence_score`, 0.75) commits at the lower `endpoint_high_confidence_floor` (0.6 s) instead of the full `endpoint_min_silence_sec` (0.7 s), reclaiming ~110 ms endpoint p50 on well-formed turns. On-device A/B 2026-06-01: p50 918→806 ms, no extra splits/truncations; floor 0.55 was rejected (truncated a comma run-on). Shipped in base config; set the floor to 0 to revert to a uniform floor. *Rationale:* endpoint cost is the dominant latency lever; the high-confidence case is safe to shorten because a premature commit is merged back by the continuation layer.

---

*Note: This log supersedes historical rationale scattered across `docs/ultracode_scope.md` (phase scope), `docs/review_ultracode.md` (audit findings), `docs/p2_memory_design.md` (R1–R12), and `docs/p3_design.md` (BR1–BR8). All decisions herein have been verified against live code and config as of 2026-06-02.*

