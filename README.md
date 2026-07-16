# Real-Time Local Voice Assistant

A local-first, always-listening voice assistant (`ASR → LLM → TTS`) with
barge-in and a mode-based control plane. The always-on loop is **fully local**:
STT, TTS, VAD, speaker-ID, and the fast answering LLM run on-device, and raw
audio never leaves the machine. An optional *thinking tier* (research / web
search) may use a cloud LLM — **off by default**, deliberately opt-in, and only
post-ASR text crosses (the [`docs/target_architecture.md`](docs/target_architecture.md)
§9.7 boundary). Open-source components. Target: on-device across
Linux/Windows/macOS and (later) Android/iOS.

> **Current architecture:** [`docs/unified_architecture.md`](docs/unified_architecture.md) — the single current-truth overview.
> **North-star & roadmap:** [`docs/target_architecture.md`](docs/target_architecture.md).
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
- **`core/engines/speaker_gate.py`** — auxiliary speaker identity for normal
  finals and the opt-in multi-voice word-cut filter. Current Linux open-speaker
  capture uses the PipeWire route prepared by `./live.sh`; physical exact Stop
  remains live-red (see `STATUS.md` and ADR-0072/0075).
- **`utils/memory*`** — Postgres-backed smart memory (see [`MEMORY.md`](MEMORY.md)).
- **`mobile/`** — on-device **Android app** (Flutter): `sherpa_onnx` +
  `flutter_gemma`, fully local. See [`mobile/README.md`](mobile/README.md).
- **`remote/`** + **`web/`** — optional **host + thin-client** path: run the brain
  on one machine; browsers/phones connect over LiveKit/WebRTC.

## Quick start

Installation has two fail-closed stages ([ADR-0063](docs/adr/0063-fail-fresh-install-readiness-closed.md)).
The first command creates a clean `.venv`, installs the local audio runtime,
downloads the selected speech stack (streaming ASR/VAD, SenseVoice, GTCRN,
Kokoro, and speaker-ID), atomically writes `config.local.json`, and runs a base
preflight without contacting Ollama:

```bash
# Linux / macOS
./install.sh

# Windows (PowerShell)
.\install.ps1
#   or double-click install.bat  (cmd.exe)
```

Stage one succeeds only with a `BASE READY (Ollama deferred)` result. Useful
flags on every platform are `--dry-run` (show the plan, change nothing) and
`--recreate` (rebuild the venv after a broken conda/venv mix). `--skip-models`
installs dependencies only and exits 2 as deliberately incomplete.

For stage two, activate the environment (`source .venv/bin/activate`, or
`.venv\Scripts\Activate.ps1` on Windows) and, with Ollama running, provision both
local Ollama roles:

```bash
ollama pull gemma3:12b                              # vision/complex main tier
python -m tools.setup_minicpm                       # MiniCPM5-1B answering tier
```

On the Linux OS-EC path, `./live.sh` prepares the transient audio route and then
requires full doctor `READY` before opening the microphone. A standalone
`python -m tools.doctor` can issue that verdict only when the required route is
already prepared. `python -m tools.doctor --defer-ollama` is a base-only
diagnostic and can never issue full `READY`; each failing line includes its fix.

Run the console (no audio/models/Ollama needed — type to talk, exercises the brain):

```bash
python -m core --engine console --llm echo
```

Run a private recorded physical session on Linux. This starts/reuses Ollama and
PipeWire echo cancellation, requires doctor READY, and restores session-owned
state when you press Ctrl-C ([ADR-0075](docs/adr/0075-make-recorded-linux-live-session-one-command-and-reversible.md)):

```bash
./live.sh
```

The portable low-level entry point remains available when platform audio is
already prepared (needs sherpa-onnx model files + a mic):

```bash
python -m core --engine sherpa --llm ollama --model gemma3:latest
```

Flags: `--engine {console,sherpa,replay,livekit}`, `--llm {echo,ollama,llamacpp}`,
`--model NAME`, `--device {desktop,phone}`, `--mode {passive,assistant,research,...}`.

For the host + thin-client path (browser/phone as endpoints), see
[`docs/deployment_profiles.md`](docs/deployment_profiles.md):
`python -m remote.worker` + `uvicorn remote.token_server:app --port 8080`.

## Models (on-device, ONNX)

`sherpa-onnx` provides VAD, streaming ASR, endpointing, TTS, and optional
speaker embeddings from ONNX files on all desktop/mobile platforms. Download a
streaming ASR model, a Silero VAD model, and a TTS model. Add a speaker-embedding
model and enrollment when normal finals need identity gating or when a
multi-voice room should restrict word-cut barge-in:

```json
"sherpa": {
  "asr_encoder": "...", "asr_decoder": "...", "asr_joiner": "...", "asr_tokens": "...",
  "vad_model": "silero_vad.onnx",
  "tts_model": "...", "tts_tokens": "...",
  "speaker_embedding_model": "...", "speaker_enroll_wav": "you.wav", "speaker_threshold": 0.5,
  "speaker_gate_input": true,
  "barge_word_cut_require_speaker": true
}
```

Word-cut barge-in is enrollment-free by default: it admits a novel exact
Stop/cancel control or at least four novel non-own words. Enrollment alone does
not filter word-cut; set `barge_word_cut_require_speaker=true` to require owner
identity for generic four-word overrides in a multi-voice room. Novel exact
Stop/cancel remains an open fail-safe control in either mode. Controls that
overlap current or recent TTS retain the fail-closed enrolled-speaker ambiguity
check. `speaker_gate_input` separately controls identity gating for normal
finals. See [ADR-0072](docs/adr/0072-make-word-cut-enrollment-optional.md)
and [ADR-0042](docs/adr/0042-attested-canonical-stop-repair.md).
For bare-laptop-speaker testing on the current Linux route, use `./live.sh`; do
not combine it with `--device open_speaker`, which selects the separate in-app
APM fallback. Physical exact Stop remains live-red in `STATUS.md`; the launcher
captures evidence but does not itself validate barge-in.

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
