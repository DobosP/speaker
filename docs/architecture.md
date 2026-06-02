# Current Architecture (as built)

A snapshot of how the runtime is wired **today**. For the cross-platform
strategy, roadmap, and resolved decisions see
[`target_architecture.md`](target_architecture.md); for product intent see
[`PROJECT_KICKOFF.md`](PROJECT_KICKOFF.md). For the **consolidated current-truth
overview** that ties all subsystems together (and absorbs the dated subsystem
docs), see [`unified_architecture.md`](unified_architecture.md).

> This **supersedes the pre-refactor pipeline** (`main.py` + `utils/audio.py`'s
> hand-rolled AEC/VAD/barge-in, `utils/stt.py`, `utils/dialogue_controller.py`,
> `utils/transports.py`, …), which was **deleted**. Don't look for those modules.

## Shape: one core, many shells

One portable **core** (audio + brain + LLM + memory) wrapped by thin
**per-platform shells**, sharing the `always_on_agent` `AgentEvent`/`Mode`
contract. Two deployment topologies, both built: **on-device** and
**host + thin-client**.

```
 mic ─▶ AudioEngine ───────────────▶ VoiceRuntime ─▶ always_on_agent "brain"
        (sherpa-onnx | livekit |      (core/        (modes · intent · planner ·
         scripted)                     runtime.py)   cancellable threaded tasks)
        on_partial/on_final/                │                 │
        on_command/on_barge_in              │       capabilities (LLM-backed,
        ▲ speak()/stop_speaking()           │       cancellable) ─▶ LLMClient
        └────────── TTS_REQUEST ◀───────────┘                      (Ollama | llama.cpp)
```

## Components

### `core/` — the runtime
- **`engine.py`** — the `AudioEngine` seam. `start(callbacks)/stop()/speak()/
  stop_speaking()`; `EngineCallbacks` = `on_partial`, `on_final`,
  `on_barge_in`, `on_speech_start/end`, `on_command` (keyword fast-path).
- **`engines/sherpa.py`** — production, on-device: sherpa-onnx VAD + streaming
  STT + endpointing + keyword spotting + TTS. `engines/_sherpa_models.py` holds
  the shared model builders; `engines/speaker_gate.py` is the speaker-ID barge-in
  gate (no AEC needed).
- **`engines/scripted.py`** — pure-Python engine for tests/console (zero deps).
- **`engines/file_replay.py`** — replays recorded `.npy`/`.wav` fixtures through
  the real pipeline, headless (latency benchmarks / CI).
- **`engines/livekit.py`** — WebRTC transport: same STT/TTS models, audio over a
  LiveKit room instead of the local mic/speaker (the remote path).
- **`llm.py`** — `LLMClient` protocol + `EchoLLM` (fake), `OllamaLLM` (desktop
  GPU), `LlamaCppLLM` (on-device GGUF). All accept optional `images=` (Gemma 3).
- **`routing.py`** — `HeuristicRouter` (dependency-light) / `LearnedRouter`
  (lazy torch) picks the fast vs. main model per turn.
- **`capabilities.py`** — LLM-backed cancellable providers (fast model answers,
  main/multimodal model researches).
- **`runtime.py`** — `VoiceRuntime`: wires engine callbacks → brain event bus →
  TTS. Owns no audio or model code (dependency-injected engine + LLM).
- **`app.py`** — CLI: builds engine (`--engine console|sherpa|livekit`) and LLM
  (`--llm echo|ollama|llamacpp`) from `config.json`, applies the `--device`
  profile.

### `always_on_agent/` — the brain (the shared contract)
- **`events.py`** — `EventKind`, `Mode` (`passive/assistant/command/search/
  research/dictation/meeting`), `AgentEvent` with priorities. **This is the
  shell↔core contract** every platform shares.
- **`event_bus.py`** — priority queue + `drain()`.
- **`supervisor.py`** — `AgentSupervisor`: STT → intent decision → task
  start/queue/confirm/cancel; owns mode state; emits `TTS_REQUEST`.
- **`speech_analyzer.py`** — activation + intent classification *before* slow LLM
  work. **`tasks.py`/`planner.py`** — cancellable task runtime + step plans.
- **`capabilities.py`** — local providers; **`memory.py`** — session memory.

### `utils/memory*.py` — persistent memory
Postgres-backed smart memory (recent RAM → Postgres history + summaries →
optional pgvector search), with an in-memory fallback. Moves to **SQLite** on
mobile. See [`../MEMORY.md`](../MEMORY.md).

### `remote/` + `web/` — host + thin-client path
- **`token_server.py`** — FastAPI: `GET /healthz`, `GET /token` (LiveKit JWT),
  `POST /chat` (a text LLM turn for the web box), and serves `web/`.
- **`worker.py`** — joins a LiveKit room running the full Python brain
  (`--engine livekit`). **`web/index.html`** — the browser client.
- Optional: `requirements-remote.txt` + `LIVEKIT_URL/API_KEY/API_SECRET`.

### `mobile/` — on-device Android app (Flutter)
ASR/LLM/TTS fully local: `sherpa_onnx` (streaming + offline-revision ASR, TTS) +
`flutter_gemma` (Gemma 3 1B, MediaPipe/LiteRT). Today `lib/assistant.dart` is a
**parallel Dart loop** that re-derives core behavior (a local command map ≈ the
desktop fast-path; sentence-streaming TTS) and does **not** yet use the shared
brain — convergence onto the `AgentEvent` contract is tracked in
`target_architecture.md` §5/§9. See [`../mobile/README.md`](../mobile/README.md).

## Config model

`config.json` drives everything (see [`deployment_profiles.md`](deployment_profiles.md)):
`device` + `device_profiles` (`desktop`/`phone`, shallow-merged via `--device`),
the `llm` block (`backend`, `main_model`/`fast_model`, `router`), `commands` (the
keyword fast-path map), `sherpa` (model paths/threads), `remote`, `memory`, and
`agent_brain` (optional Open Interpreter action brain for command mode).

## Run it

```bash
python -m core --engine console --llm echo     # no audio/models — exercises the brain
python -m core --engine sherpa                  # on-device audio (needs sherpa models + mic)
python -m remote.worker                         # host+thin-client (needs LIVEKIT_*)
uvicorn remote.token_server:app --port 8080     # token endpoint + web client
```
