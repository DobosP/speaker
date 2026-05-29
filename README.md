# Real-Time Local Voice Assistant

A fully-local, always-listening voice assistant (`ASR → LLM → TTS`) with
barge-in and a mode-based control plane. Open-source components only; no cloud
dependency. Target: on-device across Linux/Windows/macOS and (later) Android/iOS.

> **Architecture & roadmap:** see [`docs/target_architecture.md`](docs/target_architecture.md).
> **Working notes & open decisions:** [`docs/PROJECT_KICKOFF.md`](docs/PROJECT_KICKOFF.md).

## Design

```
 mic ─▶ AudioEngine (sherpa-onnx: VAD, streaming STT, endpointing, barge-in, TTS)
            │  on_partial / on_final            ▲ speak()
            ▼                                    │
        VoiceRuntime ──▶ always_on_agent "brain" │
                         (modes · intent · planner · cancellable threaded tasks)
                                   │
                         capabilities (LLM-backed, cancellable) ─▶ Ollama (local)
```

- **`core/`** — the runtime. `engine.py` is the `AudioEngine` seam; swap
  `engines/sherpa.py` (on-device production), `engines/scripted.py` (tests), or
  `engines/livekit.py` (remote/WebRTC).
- **`always_on_agent/`** — the control-plane brain: modes
  (`passive/assistant/command/search/research/dictation/meeting`), a priority
  event bus, a supervisor, and cancellable tasks that run on their own threads.
  Its `AgentEvent`/`Mode` contract is what every platform shell shares.
- **`core/engines/speaker_gate.py`** — speaker-ID barge-in gate so the
  assistant's own TTS can't self-interrupt (no AEC required).
- **`utils/memory*`** — Postgres-backed smart memory (see [`MEMORY.md`](MEMORY.md)).
- **`mobile/`** — on-device **Android app** (Flutter): `sherpa_onnx` +
  `flutter_gemma`, fully local. See [`mobile/README.md`](mobile/README.md).
- **`remote/`** + **`web/`** — optional **host + thin-client** path: run the brain
  on one machine; browsers/phones connect over LiveKit/WebRTC.

## Quick start

One command sets up everything (a clean `.venv` with pip, the lean runtime
deps, the speech models, and a preflight check). It's the same cross-platform
code path on every OS — pick your shell:

```bash
# Linux / macOS
./install.sh

# Windows (PowerShell)
.\install.ps1
#   or double-click install.bat  (cmd.exe)
```

Useful flags (all platforms): `--dry-run` (show the plan, change nothing),
`--recreate` (rebuild the venv from scratch — the fix if a conda/venv mix gives
`No module named pip`), `--skip-models` (deps only). Then activate the venv it
created (`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1` on Windows)
and add the local LLM:

```bash
ollama pull gemma3:12b && ollama pull gemma3:4b    # https://ollama.com
```

Not sure what's missing? `python -m tools.doctor` prints a READY/NOT-READY
report with the exact fix command (OS-aware) for each line.

Run the console (no audio/models/Ollama needed — type to talk, exercises the brain):

```bash
python -m core --engine console --llm echo
```

Run the on-device pipeline (needs sherpa-onnx model files + a mic; point the
`sherpa` block in `config.json` at your models):

```bash
python -m core --engine sherpa --llm ollama --model gemma3:latest
```

Flags: `--engine {console,sherpa,replay,livekit}`, `--llm {echo,ollama,llamacpp}`,
`--model NAME`, `--device {desktop,phone}`, `--mode {passive,assistant,research,...}`.

For the host + thin-client path (browser/phone as endpoints), see
[`docs/deployment_profiles.md`](docs/deployment_profiles.md):
`python -m remote.worker` + `uvicorn remote.token_server:app --port 8080`.

## Models (on-device, ONNX)

`sherpa-onnx` provides VAD, streaming ASR, endpointing, TTS, and speaker
embeddings from ONNX files on all desktop/mobile platforms. Download a
streaming ASR model, a Silero VAD model, a TTS model, and (for the barge-in
gate) a speaker-embedding model, then set their paths in `config.json`:

```json
"sherpa": {
  "asr_encoder": "...", "asr_decoder": "...", "asr_joiner": "...", "asr_tokens": "...",
  "vad_model": "silero_vad.onnx",
  "tts_model": "...", "tts_tokens": "...",
  "speaker_embedding_model": "...", "speaker_enroll_wav": "you.wav", "speaker_threshold": 0.5
}
```

Without a speaker model the gate fails open (any voice can barge in — use a
headset for clean barge-in). With enrollment, only your voice interrupts.

## Tests

```bash
python -m pytest tests -q
```

- `tests/test_core_runtime.py` — fast logic (scripted engine + fake LLM).
- `tests/test_sandbox_middle_layer.py` — **realistic-timing** scenarios across
  device profiles (slow phone → fast desktop), modeling STT partial cadence,
  LLM time-to-first-token/per-token latency, and TTS playback. Catches
  concurrency bugs (e.g. barge-in during LLM generation).
- `tests/test_speaker_gate.py` — speaker-ID gate policy.
- `tests/test_always_on_agent.py` — brain unit tests.

All run with no audio hardware, no models, and no Ollama.

## License

MIT.
