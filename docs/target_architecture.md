# Target Architecture: Fully On-Device, Cross-Platform Voice Assistant

> For the consolidated **current-truth** architecture (how it's wired today, with
> this doc's §9/§9.7/§10 decisions woven in), see
> [`unified_architecture.md`](unified_architecture.md). This doc remains the
> authority for the north-star structural decisions.

Status: decision record — the structural choices in §9 are now **resolved**
(2026-05). Supersedes the implicit architecture in `main.py`. Written to answer
the refactor question: *"rebuild lighter, run on Android/iOS/Linux/Windows/Mac,
fully local, always-listening, mode-based, multithreaded."*

---

## 0. One app or many? (the headline decision)

**Neither a single monolith nor independent per-platform apps — one portable
*core* + thin per-platform *shells*.** A monolith can't ship on iOS (Python is
not an always-on background voice app there); N independent apps would duplicate
the brain and break shared tests. So:

- **Shared:** the `always_on_agent` **`AgentEvent` / `Mode` contract** (and its
  tests). The small brain is reimplemented faithfully per runtime — *contract +
  tests are shared, not a binary core.*
- **Per-platform:** a thin UI shell + the native audio/LLM bindings.
- **"Server" is a deployment topology, not a separate app.** Two topologies, both
  built: **on-device** (desktop `core/`; the `mobile/` Flutter app) and
  **brain-on-host + thin clients** (`remote/` + LiveKit/WebRTC + `web/`). The
  product target is the **hybrid**: on-device first, host path as fallback.

Full rationale and the resolved sub-decisions are in §9.

---

## 1. Goals and constraints

**Product goals (from the brief):**

- Always-listening assistant; **local-first with a hybrid cloud thinking tier**
  (resolved 2026-05-28 — see §9.7). The always-on capture loop (STT / TTS / VAD
  / speaker-ID / fast-answering LLM) runs on-device; raw audio never leaves the
  device. The *thinking tier* (main planner, research, multimodal summarize) and
  web search may use cloud — only post-ASR text + screen captures + files cross
  over, and only when the thinking tier is invoked.
- Runs on **Linux, Windows, macOS, Android, iOS** *eventually*. **v1 is
  desktop Linux only** (mobile/multilingual/sync deferred — see §9.10).
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

The complementary path is **"brain on a host + thin clients"** (one machine runs
the agent; phones/laptops are WebRTC mic+speaker endpoints). It sidesteps both
limits and reuses the current Python code as-is. **Both paths are now built** —
on-device (`core/`, `mobile/`) *and* the host path (`remote/` + LiveKit + `web/`)
— and the product target is the **hybrid** of the two (see §0). The on-device
path remains the north star for the always-on capture/respond loop (raw audio
never leaves the device); the cloud thinking tier (§9.7) extends it where local
headroom isn't enough.

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

This is the module to **reimplement faithfully** per runtime — keep the
`AgentEvent`/`Mode` contract identical so tests transfer. **Status:** the
`mobile/` Flutter app currently runs a *parallel Dart loop* (`lib/assistant.dart`)
that re-derives core behavior (command fast-path, streaming TTS) **without** the
brain yet; converging it onto this contract — backed by a cross-language golden
test suite (transcript → expected `AgentEvent` sequence) — is the open work that
keeps "one core, many shells" from drifting into "many apps" (see §9.1).

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

## 9. Decisions (resolved 2026-05)

The forks that were open are now decided. Rationale is grounded in what shipped.

1. **Core-sharing strategy — share the contract + tests, not a binary core.**
   The shared seam is the `always_on_agent` `AgentEvent`/`Mode` model; the small
   brain is **reimplemented per runtime** (Python desktop/server, Dart mobile).
   This *supersedes* the earlier "Rust/C++ FFI core vs. embedded Python" framing:
   the Dart mobile loop already ships, and an FFI core is the most expensive path
   with the least to show. **Risk:** logic drift between the Python and Dart
   brains. **Mitigation:** a cross-language **golden test suite** (transcript →
   expected `AgentEvent` sequence) both runtimes must pass, plus the §5
   convergence work (`mobile/lib/assistant.dart` onto the contract).
2. **Deployment topology — hybrid (both paths first-class).** On-device leads
   for the always-on loop (raw audio + STT/TTS + fast-answering LLM stay
   local); the host + thin-client path (`remote/` + LiveKit + `web/`) is the
   iOS-background story and the instant-reach fallback. The local/cloud
   boundary inside a single deployment is §9.7. No longer hypothetical — it
   is built.
3. **UI shell — Flutter.** Confirmed and built (`mobile/`).
4. **Mobile LLM runtime — MediaPipe/LiteRT via `flutter_gemma`** (Gemma 3 1B) in
   the shipped Flutter app; `llama.cpp`/Ollama remain the runtimes for the
   **Python core**'s desktop/`phone` profiles. (Resolves the
   llama.cpp-vs-MLC-vs-ExecuTorch question with the de-facto, shipped choice.)
5. **iOS always-on — accept push-to-talk / wakeword-gated** listening (OS limit);
   the host path covers continuous-listening needs where required.
6. **Memory on mobile — SQLite** (+ optional on-device embeddings); vector search
   stays desktop-first (`utils/memory.py` Postgres+pgvector), optional on phones.
7. **Local/cloud boundary — local-first with a hybrid cloud thinking tier**
   (resolved 2026-05-28). *Local:* STT, TTS, VAD, speaker-ID, the always-on
   capture loop, the fast/answering LLM tier (gemma3:4b-class), conversation
   memory. *Cloud (optional, opt-in):* the *thinking* tier (main planner,
   research, multimodal summarize) and **web search**. *Invariant:* raw audio
   never leaves the device — only post-ASR text + screen captures + files
   given to the assistant may cross to cloud, and only when the thinking
   tier is invoked. Supersedes the earlier "fully-local, no cloud LLM"
   stance; the always-on loop is still fully local.
8. **Input gate — implicit addressing, speaker-ID gated** (resolved
   2026-05-28). The brain transcribes everything but only *acts* when (a)
   the speaker-ID matches the enrolled user and (b) the model judges the
   utterance is addressed to it (no wake word). This is the central design
   bottleneck and the next concrete PR after this docs landing. Today's
   behavior — every clean ASR final is a query — is explicitly rejected
   (see `logs/runs/run-20260528-004726.summary.json` for the symptom: four
   nonsense transcripts answered as queries because the gate doesn't
   exist).
9. **Microphone is variable; speaker-ID is essential** (resolved
   2026-05-28). v1 must handle headset *and* laptop-mic-plus-speakers in
   the same session. Without speaker-ID, TTS leaks into the mic when
   using speakers and produces barge-in storms (see
   `logs/runs/run-20260528-004726.txt` lines 68–75). The speaker-ID gate
   in `core/engines/speaker_gate.py` is therefore not optional in v1.
10. **v1 scope — desktop-Linux-only** (resolved 2026-05-28; mode set
    since expanded). v1 ships: desktop Linux; English; the originally-scoped
    four modes (quiet/assistant/research/command) have **grown to seven as
    implemented** (`passive`, `assistant`, `command`, `search`, `research`,
    `dictation`, `meeting` — `always_on_agent/events.py`); four background-task
    families (research / summarize / reminders / watch), of which **only
    `research` ships so far**, with per-task delivery
    preference; 2–3 concurrent tasks; spoken confirmation on destructive
    actions only; sherpa-onnx + existing brain + the cloud thinking tier
    from §9.7. **Out of v1:** mobile shell, multilingual, cross-device
    sync. Hardware target: workstation with ~16 GB VRAM GPU + 32 GB RAM
    (the rig already running `gemma3:12b` + `4b` on Ollama).

Mirrored for *intent* in `PROJECT_KICKOFF.md` §§1–7.

---

## 10. Performance plan — balanced cost × speed × intelligence

A tiered routing strategy that the existing code already supports (the
`device_profiles`, `llm.cloud` and `input_gate`/`cleanup` config blocks). The
`tools.recommend_profile` script probes the host and prints the matching
profile so users don't have to read this section first.

### 10.1 Device tiers (specsim modelled estimates, 2026-05)

Numbers from `python -m tools.specsim`. ✓ = inside the budget (≤1.2 s first
audio · ≤0.3 s barge-in stop); ~ = inside the relaxed budget (≤2.5 s / ≤0.5
s); ✗ = miss. Validate per machine with `python -m tools.bench --profile
<name>`.

| Device class | Profile | LLM tier | LLM speed | quick TTFA | research TTFA | barge-in |
|---|---|---|---|---|---|---|
| RTX 4090 / 5090 | `desktop_gpu_4090` | gemma3:12b on GPU | 111 tok/s | 0.96s ✓ | 1.63s ✓ | 0.30s ✓ |
| MacBook M2/M3/M4 (16+ GB) | `macbook_m_series` | gemma3:4b on Metal | 50 tok/s | 1.46s ~ | 2.96s ✗ | 0.30s ✓ |
| Windows/Linux laptop, no dGPU | `cpu_laptop` | gemma3:4b on CPU | 12.5 tok/s | 3.54s ✗ | 9.54s ✗ | 0.35s ~ |
| Android 12 GB | `phone` | gemma3:4b GGUF | 8.3 tok/s | 5.26s ✗ | 14.26s ✗ | 0.40s ~ |
| Low-end phone / web | `phone_lite` | gemma3:1b GGUF | 11.1 tok/s | 5.07s ✗ | 11.82s ✗ | 0.45s ~ |

**The takeaway:** only the dGPU profile hits the snappy budget locally; the
Mac is borderline; every other on-device target falls off a cliff.
Barge-in is fine everywhere — the cancellation path is cheap.

### 10.2 The tiered strategy

The pipeline already has three independent LLM-using surfaces. Tier them by
*how often they run* and *how local they must be*:

| Tier | Where it runs | Latency target | Intelligence | Cost / turn |
|---|---|---|---|---|
| Wake / KWS / intent fast-path | always local (sherpa-onnx KWS, deterministic) | <100 ms | rule-based | 0 |
| Input gate + cleanup | local fast tier *if* the device has headroom | +150–300 ms each | small LM (1b–4b) | 0 |
| Fast spoken reply (`assistant.answer`) | local fast tier | 0.3–2 s | conversational | 0 |
| Main planner / multimodal | local main tier *or* cloud hedge | 1–3 s | strong | 0 or ~$0.0008 |
| Research / web / vision summarize | cloud-only when enabled (§9.7) | 2–5 s | frontier | ~$0.005 |

The fully-local fast-tier loop (STT → fast LM → TTS) is the always-on
contract from §9.7 and §1 — it never leaves the device. The cloud appears
only where the local headroom isn't there.

### 10.3 Profile → strategy mapping

Each profile under `config.json.device_profiles` codifies one row of the
matrix below; flip `llm.cloud.enabled` in a gitignored `config.local.json`
to add a cloud hedge without editing the committed template.

| Profile | Models on-device | Gates ON? | Cloud hedge | Notes |
|---|---|---|---|---|
| `desktop_gpu_4090` | gemma3:12b + 4b | both | off | The fully-local north-star config. |
| `desktop` | gemma3:12b + 4b | off (compat) | off | The existing profile; left alone for backward-compat. |
| `macbook_m_series` | gemma3:4b + 1b on Metal | both | off (recommended for research mode) | M-series Macs hit ~50 tok/s on Metal — fast tier has headroom for the gates. |
| `cpu_laptop` | gemma3:4b + 1b on CPU (Ollama) | **off** | **on, strongly recommended** | CPU LLM TTFT (~0.8 s) leaves no slack for the gates. Cloud hedge keeps research-mode under 3 s. |
| `phone` | gemma3:4b + 1b GGUF | off | optional | Android/iOS class; same cost story as `cpu_laptop`. |
| `phone_lite` | gemma3:1b GGUF only (single-tier) | off | **required for research** | Sub-8 GB hosts thrash on the 4b model. |

### 10.4 Cloud middle layer — providers and pricing (verified May 2026)

When the local main tier can't meet the deadline (CPU laptops, phones, web)
the brain hedges or falls back to a low-latency cloud LLM. The supported
providers, **verified against vendor docs on 2026-05-28** (6 of the 7 IDs
shipped a week earlier had silently deprecated; see commit history):

| Preset key | Model id | Provider | Hosting | $/MTok in | $/MTok out | TTFT |
|---|---|---|---|---|---|---|
| `cerebras_gpt_oss_120b`   | `gpt-oss-120b`         | Cerebras | **US** | $0.50 | $1.00 | ~80 ms |
| `cerebras_glm_4_7_coder`  | `zai-glm-4.7`          | Cerebras | **US** | $0.60 | $1.20 | ~80 ms |
| `groq_gpt_oss_120b`       | `openai/gpt-oss-120b`  | Groq     | **US** | $0.15 | $0.60 | ~100 ms |
| `deepseek_v4_flash`       | `deepseek-v4-flash`    | DeepSeek | CN     | $0.14 | $0.28 | ~400 ms |
| `deepseek_v4_pro`         | `deepseek-v4-pro`      | DeepSeek | CN     | $0.55 | $2.19 | ~500 ms (reasoning) |
| `moonshot_kimi_k2_6`      | `kimi-k2.6`            | Moonshot | CN     | $0.95 ($0.16 cache hit) | $4.00 | ~300 ms |

Each preset declares a `profile` (`cerebras` / `groq` / `deepseek` /
`deepseek_reasoning` / `moonshot`) that maps to a `core.llm.ProviderProfile`
encoding the per-vendor quirks: Cerebras free-tier caps `max_tokens=8192`
and routes non-standard params via `extra_body=`; Groq fixes `n=1` and
streams reasoning in `delta.reasoning`; DeepSeek V4-Pro streams
`delta.reasoning_content` BEFORE `delta.content` (the assistant must not
speak the CoT but must count it for run-summary metrics); Moonshot Kimi
rejects custom `temperature`/`top_p`/`n`. `tools/llm_sanity.py --smoke`
exercises each profile against real keys; `.github/workflows/llm-cloud-smoke.yml`
runs it weekly to catch provider drift before users do.

Sources verified 2026-05-28: [Cerebras models](https://inference-docs.cerebras.ai/models/),
[Cerebras deprecation](https://inference-docs.cerebras.ai/support/deprecation),
[Groq deprecations](https://console.groq.com/docs/deprecations),
[DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing),
[DeepSeek reasoning model](https://api-docs.deepseek.com/guides/reasoning_model),
[Kimi K2.6 quickstart](https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart),
[LiteLLM model registry](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
(vendored at `tools/litellm_model_registry.json` as a build-time validator).

Per-turn cost (quick ≈ 250 tokens; research ≈ 1.3 k; 80/20 in:out split):

| Model | Quick | Research | 100 quick + 20 research / day |
|---|---|---|---|
| qwen-3-coder-480B (Cerebras) | $0.0005 | $0.0026 | **$0.10/day** |
| qwen-3-235B (Cerebras) | $0.0002 | $0.0012 | **$0.04/day** |
| llama-3.3-70b (Groq) | $0.0002 | $0.0009 | **$0.04/day** |
| kimi-k2 (Groq) | $0.0006 | $0.0035 | **$0.13/day** |
| deepseek-v4-flash | $0.00007 | $0.0004 | **$0.014/day** |
| deepseek-r1 | $0.0003 | $0.0017 | **$0.07/day** |

Heavy daily use stays under ~$0.20/day in every realistic mix. Hedging
(see `core/llm.py::HedgeLLM`) makes these an *upper bound* — when local
beats the deadline, the cloud isn't called at all.

### 10.5 Sensitivity-routed cloud chains

Among the providers above, **hosting jurisdiction matters**. Cerebras and
Groq are US-incorporated; DeepSeek and Moonshot route through PRC servers.
The runtime tags every turn with one of three *sensitivity* values
(`core/sensitivity.py`) and dispatches to a named chain accordingly:

| Sensitivity | What triggers it | Chain (config default) |
|---|---|---|
| `private` | `my <noun>`, `IntentKind.COMMAND/DICTATION/MEETING_NOTE`, `Mode.MEETING`, **everything unclassified (safe default)** | `[cerebras_gpt_oss_120b, groq_gpt_oss_120b]` — US-only |
| `code` | code markers (`function`, `class`, `refactor`, `debug`, language names) | `[cerebras_glm_4_7_coder, groq_gpt_oss_120b]` — US-hosted |
| `public` | encyclopedic openers (`what is`, `who was`, `how does`) with no personal-data markers | `[deepseek_v4_flash, cerebras_gpt_oss_120b]` — cheapest first |

`HedgeLLM` tries chain entries in order, falling through on
timeout/error; local is the final fallback. The classifier is a deliberate
floor — pattern-based, fail-safe-to-`private`. A learned classifier
replaces it only if the heuristic mis-routes in practice.

### 10.6 The tiered strategy in summary

The pipeline now has four LLM-using surfaces, ordered by frequency:

| Tier | Where it runs | Latency target | Cost / turn |
|---|---|---|---|
| Wake / KWS / intent fast-path | local (sherpa-onnx KWS, deterministic) | <100 ms | 0 |
| Input gate + cleanup | local fast tier *if* the device has headroom | +150–300 ms each | 0 |
| Fast spoken reply (`assistant.answer`, fast tier) | local fast tier | 0.3–2 s | 0 |
| Main planner / multimodal | local main tier hedged against the sensitivity-routed cloud chain | 1–3 s | $0 or ~$0.0008 |

The always-on capture loop never leaves the device (§9.7). Only post-ASR
text crosses the local↔cloud boundary, and only to the chain selected by
sensitivity.

### 10.7 What this changes in the repo

- `config.json` — adds `llm.cloud_providers` (named OpenAI-compatible
  presets), `llm.cloud_chains` (failover chains by sensitivity), and
  `llm.cloud_routing` (sensitivity → chain mapping). Each `device_profile`
  picks a sensible `cloud` strategy (off / hedge / fallback) plus
  per-tier deadlines.
- `core/sensitivity.py` (new) — heuristic `Sensitivity` classifier.
- `core/llm.py` — `HedgeLLM` now accepts a *list* of clouds (failover
  chain); `SensitivityRouterLLM` dispatches `stream`/`generate` to one
  of several backing LLMs via a `ContextVar`-published per-turn context.
- `core/routing.py` — `HeuristicRouter` factors `IntentKind` (research →
  main, command/dictation → fast); new `ChainSelector` picks chain by
  sensitivity.
- `core/capabilities.py` — `assistant.answer` enriches the context with
  `intent_kind` + `sensitivity`, publishes it via `capability_context`,
  then resets it after the turn.
- `always_on_agent/tasks.py` — task `_invoke` forwards
  `task.intent.value` into the capability context.
- `tools/recommend_profile.py` (existing) — still picks the device
  profile; cloud chains activate automatically once API keys are present.
- Tests: `tests/test_sensitivity.py`, `tests/test_hedge_chain.py`,
  `tests/test_cloud_providers.py`, `tests/test_routing_intent.py`,
  `tests/test_imports_smoke.py` — plus `tools/run_tests.py` gains
  `cloud` and `imports` stages for fast feedback.
