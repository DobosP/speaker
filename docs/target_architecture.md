# Target Architecture: Fully On-Device, Cross-Platform Voice Assistant

Status: proposal / decision record. Supersedes the implicit architecture in
`main.py`. Written to answer the refactor question: *"rebuild lighter, run on
Android/iOS/Linux/Windows/Mac, fully local, always-listening, mode-based,
multithreaded."*

---

## 1. Goals and constraints

**Product goals (from the brief):**

- Always-listening assistant, fully **on-device** and **fully local** (no cloud).
- Runs on **Linux, Windows, macOS, Android, iOS**.
- **Modes** (e.g. `passive/quiet`, `assistant`, `research`, `command`,
  `dictation`) that change how speech is interpreted and what runs in the
  background.
- **Concurrent tasks** — research/background work runs without blocking
  listening or speaking.
- Low latency; reliably captures the *complete* question; reliably knows
  *when to stop speaking* (barge-in).

**Two constraints that are not engineering problems and must be designed
around, not coded around:**

1. **iOS forbids indefinite background mic capture.** Apple terminates
   continuous background microphone use. "Listens all the time" is achievable
   on Android (foreground service + persistent notification) and on desktop,
   but on iOS it must degrade to **foreground-only or wakeword-gated**
   listening. Plan the UX for this difference from day one.
2. **On-device LLM on phones is modest.** Practical mobile models are small
   quantized LLMs (~1–3B, e.g. Gemma 3 1B / Qwen 2.5 1.5B via `llama.cpp`).
   Running one always-on in the background drains battery and thermally
   throttles. Desktop can run larger models; phones cannot. Mode behavior
   should be tunable per device class.

If either constraint is a dealbreaker, the alternative is **"brain on a host +
thin clients"** (one machine runs the agent; phones/laptops are WebRTC mic+
speaker endpoints). That path sidesteps both limits and reuses more of the
current Python code. This document assumes the on-device path was chosen.

---

## 2. Why the current repo can't be the mobile foundation

The existing app is Python + PortAudio + a hand-rolled real-time audio stack.
Python does not ship as an always-on background voice app on iOS/Android.
Therefore:

- The current repo becomes the **desktop reference implementation and
  prototyping ground** — not the shippable mobile artifact.
- The *shippable* product is built on a **portable core** (native/ONNX) wrapped
  by thin per-platform UIs.
- The work is not wasted: the **orchestration logic** (modes, turn-taking,
  task scheduling) ports cleanly; only the audio plumbing is replaced — and
  that plumbing has to be replaced regardless, because the home-grown version
  is the source of today's latency and reliability pain.

---

## 3. Root cause of today's problems

| Symptom (from the brief) | Where it lives today | Why |
|---|---|---|
| Latency is big | `utils/audio.py` (~3,000 lines) + `main.py` monolith | Hand-rolled NLMS echo canceller, VAD gating, and barge-in scoring run in a Python orchestration loop. DIY real-time DSP in Python is slow and hard to tune. |
| Doesn't reliably catch the exact question | `main.py` partial/final STT + turn detection | Turn-end detection is bespoke and threshold-driven; no battle-tested endpointing. |
| Doesn't reliably know when to stop speaking | `utils/audio.py` `BargeInDetector` + `utils/dialogue_controller.py` | Self-echo vs. real barge-in is solved with manual RMS/correlation thresholds (`config.json` has ~15 barge-in tuning knobs — a sign the heuristic is fragile). |
| Not multithreaded / not smooth | `main.py` `VoiceAssistant` (one class, ~1,800 lines of methods) | TTS workers, partial-transcribe loop, streaming respond, and barge-in are entangled in one object. |

The fix is to **stop hand-rolling the real-time audio layer** and delegate it to
a proven, cross-platform, on-device engine.

---

## 4. Target architecture (layered)

```
+-------------------------------------------------------------+
|  Platform shell (per OS)                                    |
|  - UI, permissions, background/foreground lifecycle         |
|  - Linux/Win/macOS: desktop app   Android/iOS: Flutter/native|
+-------------------------------------------------------------+
                         | FFI / IPC (typed events)
+-------------------------------------------------------------+
|  Portable core (shared, language-agnostic logic)           |
|                                                             |
|   Audio engine            Brain (the "mid layer")          |
|   - VAD                    - Event bus (priority)           |
|   - streaming STT          - Supervisor: modes + tasks      |
|   - endpointing/turn       - Speech analyzer / intent       |
|   - barge-in / AEC         - Cancellable task runtime       |
|   - TTS                    - Capability providers           |
|        |                          |                          |
|        +------ typed AgentEvents --+                         |
|                                                             |
|   LLM runtime              Memory                           |
|   - on-device quantized    - session + persistent store     |
+-------------------------------------------------------------+
```

### Keystone components

| Layer | Chosen engine | Why |
|---|---|---|
| Audio: VAD + STT + TTS + speaker ID | **`sherpa-onnx` (k2-fsa)** | One library that does all of it **on-device**, ONNX-based, with official **Android/iOS/Linux/Windows/macOS** bindings. Replaces the entire `utils/audio.py` + the STT/TTS plumbing. Has built-in VAD and endpointing — the parts we currently hand-roll badly. |
| On-device LLM | **`llama.cpp`** (alt: MLC-LLM / ExecuTorch) | Runs quantized small models on all five platforms. Desktop keeps the option of Ollama as a thin wrapper over the same models. |
| Brain / mid-layer | **`always_on_agent/` (keep)** | Already implements modes, a priority event bus, a supervisor, and cancellable tasks — exactly the requested design. Language-agnostic in shape. |
| UI shell | **Flutter** (recommended) | Single codebase, all five platforms, clean FFI to a native core. (Alt: native per-platform if Flutter is undesired.) |

The repo already points this way: `requirements.txt` pins ONNX components
(`useful-moonshine-onnx`, `kokoro-onnx`, `silero-vad`), and
`always_on_agent/adapters.py` already names Pipecat/LiveKit/Wyoming/Moonshine as
integration boundaries. `sherpa-onnx` is the consolidation of that instinct that
*also* works on phones.

---

## 5. The brain stays — and it's good

`always_on_agent/` is the one subsystem worth carrying forward almost as-is.
It is small, typed, and synchronous (drain-based event bus), which makes it
easy to port and test deterministically:

- `events.py` — `EventKind`, `Mode` (`passive/assistant/command/search/research/dictation/meeting`), `AgentEvent` with priorities.
- `event_bus.py` — priority queue, `drain()`.
- `supervisor.py` — `AgentSupervisor`: maps STT events → intent decisions →
  task start/queue/confirm/cancel; owns mode state; emits `TTS_REQUEST`.
- `speech_analyzer.py` — activation, language hints, intent classification
  *before* slow LLM work.
- `tasks.py` / `planner.py` — cancellable task runtime + step→capability plans
  (this is the "multithreaded background work" requirement).
- `capabilities.py` — local providers (`system.time`, etc.).
- `memory.py` — session memory.

**Required change:** it is currently a synchronous simulation driven by
`drain()`. To run live it needs (a) STT events published from the real audio
engine, and (b) the task runtime backed by real threads/async with true
cancellation. The *interfaces* don't change; the execution backend does.

If/when the core is ported off Python (for mobile), this is the module to
**reimplement faithfully** in the core language (or expose Python core via
embedded interpreter only on desktop). Keep the event/mode contract identical
so tests transfer.

---

## 6. Keep / Replace / Delete map

Grounded in the current tree.

### Keep (carry forward, minimal change)
- `always_on_agent/` — the brain. (Make task runtime real-threaded; wire to live STT.)
- `utils/memory.py`, `utils/memory_writer.py`, `utils/memory_config.py` — memory subsystem (port storage backend per platform; SQLite on mobile instead of Postgres).
- `utils/capabilities.py`, `utils/conversation_router.py` — routing/capability logic (fold into the supervisor's decision path).
- The test *philosophy* in `tests/` (replay/transcript-driven tests) — keep replay harnesses; drop tests tied to the deleted DSP.

### Replace (delete the implementation, keep the responsibility)
- `utils/audio.py` (~3,000 lines: NLMS AEC, EchoGuard, SpeechGate, BargeInDetector) → **`sherpa-onnx`** VAD + endpointing + (its) AEC/echo handling.
- `utils/stt.py`, `utils/turn_detector.py`, `utils/voice_gate.py`, `utils/wakeword_service.py` → **`sherpa-onnx`** STT/endpointing/keyword-spotting; speaker ID via sherpa-onnx speaker models.
- `utils/dialogue_controller.py` (turn state machine) → folded into the supervisor + the engine's endpointing/interruption events.
- `main.py` `VoiceAssistant` orchestration (~1,800 lines) → a thin **runtime adapter** that pumps engine events into the supervisor and plays `TTS_REQUEST`s. Target: a few hundred lines.
- `utils/llm.py` → thin client over `llama.cpp` (desktop may keep Ollama behind the same interface).
- `utils/transports.py` (`SessionMux` local_lan/webrtc) → only relevant if the "thin client" path is ever revived; otherwise drop.

### Delete (dead weight once the DSP is gone)
- The ~15 barge-in tuning knobs in `config.json` (`echo_corr_threshold`, `barge_in_min_rms_ratio`, `aec_filter_ms`, etc.) — these exist to babysit the hand-rolled detector that's being removed.
- Benchmarks/scripts tied to the old AEC/barge-in (`benchmarks/benchmark_realtime.py`'s AEC paths, `scripts/check_latency_slo.py` if its SLOs are DSP-specific).
- Tests bound to deleted modules (`test_audio.py`, AEC/barge-in scenario tests, `test_dialogue_controller.py`, etc.).

### Open question
- `recordings/`, `session_recorder.py`, `analyze_sessions.py` — the session
  recording/replay tooling is genuinely useful for tuning. Keep if we re-target
  it at the new engine's event stream; otherwise archive.

---

## 7. Portable-core module boundaries (target)

```
core/
  audio/        # thin wrapper over sherpa-onnx: start_listening(), events: vad_start,
                # partial(text), final(text), endpoint, barge_in; speak(text)/stop_speaking()
  llm/          # generate(prompt, stream) over llama.cpp; same iface on every platform
  brain/        # ported always_on_agent: event bus, supervisor, modes, tasks, analyzer
  memory/       # session + persistent (SQLite on-device); embeddings optional
  capabilities/ # local tool providers (time, notes, search-local, etc.)
  api/          # the FFI/IPC surface the shells call: one event stream in, one out
```

The shell ↔ core contract is **just the `AgentEvent` stream** that already
exists. Shells send `STT`/`CONTROL` events and audio frames; the core sends
back `TTS_REQUEST`/`TASK_*`/`INTENT_DECISION`. Keeping this contract stable is
what lets desktop (Python) and mobile (native) share the same brain and tests.

---

## 8. Phased migration (each step independently useful)

1. **Prove the engine on desktop (Python).** ✅ *Landed in `core/`.* New lean
   runtime (`VoiceRuntime`): a swappable `AudioEngine` seam with a `sherpa-onnx`
   production implementation (VAD/STT/endpointing/barge-in/TTS) and a scripted
   engine for tests, wired to the existing `AgentSupervisor`, plus real
   LLM-backed `assistant.answer`/`research.local` capabilities (Ollama). Runs
   alongside the old `main.py`. The full path is tested without audio/models
   (`tests/test_core_runtime.py`) and runnable via `python -m core --engine
   console`. **Remaining:** validate the `sherpa-onnx` engine on real hardware
   (mic + model files + Ollama) and tune barge-in (headset or speaker-ID gating,
   since sherpa has no AEC). *Exit criteria:* lower end-to-end latency, reliable
   turn capture, no self-barge-in — without any of the old tuning knobs.
2. **Make the brain real.** Replace the `drain()` simulation with a real
   threaded/async task runtime + true cancellation. Implement the requested
   modes (`quiet`, `assistant`, `research`, ...) and background concurrency on
   the now-reliable foundation.
3. **Define the FFI/IPC core boundary** (`core/api`) on desktop, still in Python,
   so the contract is exercised before the language port.
4. **Port the core** to the shared stack (sherpa-onnx + llama.cpp + reimplemented
   brain) behind that same boundary; wrap in a **Flutter** shell; bring up
   **one** mobile platform end-to-end (Android first — fewer background limits).
5. **iOS + remaining platforms**, accepting the foreground/wakeword-gated
   listening constraint on iOS.

---

## 9. Risks / decisions still open

- **Flutter vs. native shells** — recommend Flutter; confirm.
- **`llama.cpp` vs MLC-LLM vs ExecuTorch** for mobile LLM — decide after a phone
  perf spike; affects model selection and battery story.
- **Core language for the eventual port** — Python-embedded (simplest, but heavy
  on mobile) vs. Rust/C++ core with a Python desktop binding (more work, best
  mobile result). Decide before Phase 4; Phases 1–3 are unaffected.
- **iOS always-on** — confirm acceptable degraded UX (push-to-talk / wakeword)
  or reconsider the "brain on a host" alternative for mobile specifically.
- **Memory on mobile** — SQLite + optional on-device embeddings; confirm whether
  vector search is needed on phones or desktop-only.
```
