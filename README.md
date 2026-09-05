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
> **Current truth:** [`STATUS.md`](STATUS.md) · **Agent contract:** [`AGENTS.md`](AGENTS.md)

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

Optional local tools are setup-time additions to the same chatbot. They can be
granted during installation or changed later without a tool-specific launcher:

```bash
./install.sh \
  --obsidian-vault ~/work/dobo-brain/paul-brain \
  --enable-reminders \
  --trust-app obsidian=obsidian.desktop

# Equivalent after installation:
.venv/bin/python -m tools.setup_assistant \
  --obsidian-vault ~/work/dobo-brain/paul-brain \
  --enable-reminders \
  --trust-app obsidian=obsidian.desktop
```

The setup command writes only machine-local `config.local.json`. It validates
the vault directory without reading notes and records an exact desktop ID
without launching it. Use `--disable-obsidian`, `--disable-reminders`, or
`--untrust-app ALIAS` to remove grants. Trusted-app v1 opens an allowlisted app;
it does not generalize to shell commands, URLs, files, typing, or clicks.

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
python -m core --session console --llm echo
```

Run a private recorded physical session on Linux. For an Ollama profile this
starts/reuses loopback Ollama; every profile gets PipeWire echo cancellation,
the applicable doctor gate, four continuous private PCM16 stage tracks plus an
exact lossless f32le final-input spool, and restoration of session-owned state
when you press Ctrl-C
([ADR-0075](docs/adr/0075-make-recorded-linux-live-session-one-command-and-reversible.md),
[ADR-0077](docs/adr/0077-capture-aligned-pre-dsp-live-stt-evidence.md),
[ADR-0108](docs/adr/0108-bind-exact-final-model-inputs-to-private-stt-replay.md),
[evidence guide](docs/voice_evidence.md)):

```bash
./live.sh
```

That one session includes every capability enabled at setup. For example, say
`search in my vault for speaker`, `show my active reminders`, `remind me to
stretch in ten minutes`, or `open obsidian`. Reminder changes and app opening
are read back and require a later spoken `confirm`; vault search and reminder
listing are read-only. There are no vault/reminder/app variants of `live.sh`.

The portable low-level entry point remains available when platform audio is
already prepared (needs sherpa-onnx model files + a mic):

```bash
python -m core --session local --llm ollama
```

Sessions: `--session {console,local,replay,trusted-lan}`. See
[ADR-0097](docs/adr/0097-unify-voice-sessions-and-gate-audio-egress.md) for
topology policy and promotion state.
Other flags include `--llm {echo,ollama}`,
`--model NAME`, `--device {desktop,phone}`, `--mode {passive,assistant,research,...}`.

For the host + thin-client migration and its current promotion gate, see
[ADR-0096](docs/adr/0096-introduce-publisher-bound-livekit-agents-adapter.md)
and [ADR-0097](docs/adr/0097-unify-voice-sessions-and-gate-audio-egress.md).

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

The Tier-0 logic suite: no audio hardware, models, or Ollama. Tiers, markers and
the staged runner: [`docs/testing.md`](docs/testing.md); agent gates and the
before-commit checklist: [`docs/agent-testing.md`](docs/agent-testing.md).

## Documentation

Every non-history document is one hop from this table. Decisions live in
`docs/adr/` (append-only); dated journals are history.

| Doc | Read it for |
|---|---|
| [`AGENTS.md`](AGENTS.md) | The operating contract: read-first order, commands, safety, docs discipline. |
| [`STATUS.md`](STATUS.md) | Current truth: state, open gates, next, verification record. |
| [`WORKLOG.md`](WORKLOG.md) | Dated history and the frozen receipts trimmed from `STATUS.md`. |
| [`docs/agent-map.md`](docs/agent-map.md) | Entry points, task routes, do-not-load list, pitfalls. |
| [`docs/agent-testing.md`](docs/agent-testing.md) | Gate commands with expected output, before-commit checklist, known flaky. |
| [`docs/testing.md`](docs/testing.md) | Test tiers, markers, staged runner, CI, per-feature gate matrix. |
| [`docs/dev_guide.md`](docs/dev_guide.md) | Day-to-day development loop, layout, environment notes. |
| [`docs/debugging.md`](docs/debugging.md) | Run logs, bundles, PII scrubbing before `git add`. |
| [`docs/deployment_profiles.md`](docs/deployment_profiles.md) | Device profiles and what transport each selects. |
| [`docs/docker_quickstart.md`](docs/docker_quickstart.md) | Container path for the cloud middle layer. |
| [`docs/audio_pipeline.md`](docs/audio_pipeline.md) | Current capture/playback guide (AEC, DSP, TTS output). |
| [`docs/asr_biasing.md`](docs/asr_biasing.md) | Streaming hotwords and contextual biasing. |
| [`docs/voice_evidence.md`](docs/voice_evidence.md) | Recorded and live evidence protocol. |
| [`docs/public_voice_evaluation_matrix.md`](docs/public_voice_evaluation_matrix.md) | Public STT evaluation matrix: tracks, sources, exclusions. |
| [`docs/public_voice_regression.md`](docs/public_voice_regression.md) | Public voice regression procedure. |
| [`docs/evaluation_runbooks.md`](docs/evaluation_runbooks.md) | Protected benchmark and diagnostic runbooks, harness semantics. |
| [`docs/unified_architecture.md`](docs/unified_architecture.md) | Current architecture overview. |
| [`docs/target_architecture.md`](docs/target_architecture.md) | North star, §9 structural decisions, §9.7 local/cloud boundary. |
| [`docs/PROJECT_KICKOFF.md`](docs/PROJECT_KICKOFF.md) | Product intent and open product questions. |
| [`MEMORY.md`](MEMORY.md) | Postgres-backed smart memory design and operation. |
| [`SETUP.md`](SETUP.md) | Database and environment setup. |
| [`CREDENTIALS.md`](CREDENTIALS.md) | Secret and token names (values never live in git). |
| [`SECURITY.md`](SECURITY.md) | Threat model, egress boundary, reporting. |
| [`always_on_agent/README.md`](always_on_agent/README.md) | Control-plane brain: modes, event bus, supervisor. |
| [`mobile/README.md`](mobile/README.md) | Flutter Android shell: build and on-device models. |
| [`tools/autotest/README.md`](tools/autotest/README.md) | Autonomous voice/barge stress harness. |
| [`.agents/backlog.md`](.agents/backlog.md) | Work queue read by `tools/session_bootstrap.py`; `STATUS.md` Next wins. |
| [`docs/adr/`](docs/adr/) | Decisions, append-only. |
| [`docs/archive/`](docs/archive/) | Superseded documents, kept as history. |

Design plans (snapshots; verify against [`STATUS.md`](STATUS.md)):

- [`docs/cross_device_audio_quality.md`](docs/cross_device_audio_quality.md) — output-side loudness/limiter plan (2026-06-22).
- [`docs/memory_layers_enhancement_plan.md`](docs/memory_layers_enhancement_plan.md) — memory layers ([ADR-0009](docs/adr/0009-memory-layers-approved-deferred.md)).
- [`docs/voice_upgrade_plan.md`](docs/voice_upgrade_plan.md) — voice upgrade ([ADR-0010](docs/adr/0010-kokoro-tts-adopted.md)).

Dated journals (`docs/session_*.md`, `docs/*_2026-*.md`, `docs/2026-*.md`) are
history and are deliberately not indexed.

## License

MIT.
