# Developer guide — working practices, layout, tooling, CI

Living doc (extracted from CLAUDE.md 2026-07-02). Architecture truth is
[`unified_architecture.md`](unified_architecture.md); test tiers are
[`testing.md`](testing.md); run-log debugging is [`debugging.md`](debugging.md);
device/LLM config is [`deployment_profiles.md`](deployment_profiles.md).

## Session bootstrap (run first every session)

This project is worked on from **multiple machines**, and per-user Claude memory
does **not** travel between them — prior-session state lives in the repo. At the
start of a session, reconstruct where things left off **before touching code**:

```
python -m tools.session_bootstrap   # pure-local, stdlib-only, <1s, no deps
```

It prints a one-page briefing assembled in this read order, then a
**Recommended working strategy** block:

1. `.agents/status.json` — machine profile + last test verdict (green/red + counts)
2. `docs/session_*.md` (newest by filename) — headline, branch, first 3 next-steps
3. `logs/runs/*.summary.json` (3 newest) — per-run `stuck_hints`, errors, slow turns
4. `.agents/backlog.md` — OPEN P0 items only

The briefing is **advisory** — it sets direction, it does not change config or
git state. If the tool is unavailable, walk the read order above by hand. A copy
is written to `logs/session_<ts>_bootstrap.md` (gitignored).

**At session END**, if meaningful work landed, refresh `.agents/status.json`
(machine + `last_verdict` + `next`) and write a
`docs/session_<YYYY-MM-DD>_<slug>.md` handoff (header / branch-commit map /
what-landed / environment-on-`<machine>` / **Next steps (pick up here)**) so the
next session's bootstrap reads fresh state. See `tools/session_bootstrap.py` for
the exact fields it parses.

## Repo layout

- `core/` — **the runtime (all new work goes here).** `engine.py` (the
  `AudioEngine` seam), `engines/sherpa.py` (production, on-device; CPU STT/TTS
  with auto-tuned threads + explicit `provider`), `engines/scripted.py`
  (tests/console), `engines/livekit.py` (WebRTC transport for the remote
  host+thin-client path), `engines/speaker_gate.py` (speaker-ID barge-in gate),
  keyword-spotter **command fast-path** (sherpa KWS runs alongside ASR and fires
  `on_command`; the runtime maps it to a control event via the `commands` config
  block — instant actions like "stop" with no LLM in the loop), `llm.py` (the
  `LLMClient` protocol + `EchoLLM` fake, `OllamaLLM` for desktop GPU,
  `LlamaCppLLM` for on-device GGUF; all accept optional `images=` for multimodal
  Gemma 3), `capabilities.py` (LLM-backed cancellable providers; two-model split
  — fast model answers, main/multimodal model researches), `runtime.py`
  (`VoiceRuntime` orchestrator), `app.py` (CLI; builds models from the `llm`
  config block and applies the selected device profile).
- `always_on_agent/` — the **control-plane "brain"** (modes, priority event bus,
  supervisor, cancellable threaded tasks, intent analyzer). The keeper. See its
  `README.md` and `docs/archive/always_on_agent_layer.md`. Its `events.py`
  (`AgentEvent`/`Mode`) is **the shell↔core contract** every platform shares.
- `mobile/` — **on-device Android app** (Flutter): ASR/LLM/TTS fully local via
  `sherpa_onnx` + `flutter_gemma` (Gemma 3 1B, MediaPipe/LiteRT). Today it is a
  **parallel Dart loop** (`lib/assistant.dart`) that re-derives core behavior
  (command fast-path, streaming TTS) rather than sharing the brain — the planned
  convergence is onto the `AgentEvent` contract. See `mobile/README.md`.
- `remote/` — **host + thin-client path** (optional; `requirements-remote.txt` +
  `LIVEKIT_*`). `token_server.py` (FastAPI: mints LiveKit tokens, a text
  `/chat`, serves `web/`), `worker.py` (joins a LiveKit room running the full
  Python brain via `--engine livekit`). `web/index.html` is the browser client.
- `utils/memory.py` (+ `memory_writer.py`, `memory_config.py`) — Postgres-backed
  smart memory (the only surviving `utils/` modules). See `MEMORY.md`. Keep;
  will move to SQLite on mobile.
- `tests/` — pytest. `tests/sandbox/` is the device-simulation harness
  (latency/LLM-weight profiles + simulated engine/LLM) for middle-layer tests;
  `test_core_runtime.py` is fast logic; `test_sandbox_middle_layer.py` is
  realistic-timing/concurrency. No audio/model deps.
- `tools/` — dev tooling (no app code). `run_tests.py` + `testing/` (staged
  pytest runner with reports under `test-reports/`); `specsim/` (machine-spec
  simulator that renders an HTML capability report); `cloudchat.py` (parallel
  cloud-LLM REPL: fires N prompts in parallel at the endpoint in
  `config.llm.cloud`, streams them with `[Qn]` prefixes, `/cancel` hard-closes
  the HTTP stream so the provider stops billing; needs `openai`);
  `recommend_profile.py` (stdlib hardware probe → prints which
  `device_profiles` entry to use; see `docs/target_architecture.md` §10).
- `config.json` — runtime config (see `deployment_profiles.md`). `docs/` —
  architecture and subsystem docs.

> The legacy stack (`main.py`, `utils/audio.py`, the hand-rolled STT/TTS/LLM
> plumbing, `benchmarks/`, `scripts/`, and their tests) was **deleted** in the
> refactor (`docs/adr/0002`) — don't try to import it. The refactor replaced the
> hand-rolled audio stack with `sherpa-onnx` and kept `always_on_agent/` as the
> real brain.

## Running the app

- `python -m core --engine console --llm echo` — no audio/models (try-it path).
- `python -m core --engine sherpa` — on-device audio.
- `python -m core --engine replay --replay-dir <dir>` — the real engine headless
  over recorded `.npy`/`.wav` fixtures (no sound card).
- `python -m remote.worker` — host+thin-client path (joins a LiveKit room; needs
  `LIVEKIT_*`).
- Engines: `--engine {console,sherpa,replay,livekit}`; LLM:
  `--llm {echo,ollama,llamacpp}`; profile: `--device <name>` (default from
  `config.device`; profiles + the `llm` block: `deployment_profiles.md`).
- Latency instrumentation (`core/metrics.py`) and run-log capture are described
  in `unified_architecture.md` §11 and `debugging.md`.

## Bench & specsim

- `python -m tools.bench --fake` — no-download plumbing smoke test.
- `python -m tools.bench --profile phone --fixtures
  tests/fixture_audio/virtual_real_world` — fetches the ADR-0020 MiniCPM5 GGUF
  (optionally authenticated by `$HUGGINGFACE_TOKEN`) + sherpa ONNX and runs the
  REAL ASR→LLM→TTS pipeline
  over fixtures, writing a measured-vs-`specsim`-budget report under
  `test-reports/perf/`. Model coordinates are overridable via a
  `--models-manifest` JSON / `SPEAKER_BENCH_*` env vars.
- `python -m tools.specsim` renders `test-reports/specsim/index.html`
  (model-fit + responsiveness matrix + per-device ASR→LLM→TTS timelines across
  4090/Mac/Windows/phone/web). Numbers are modelled estimates, not measurements
  — calibrate `tools/specsim/specs.py` from real runs before trusting absolutes.

## CI workflows

- `.github/workflows/tests.yml` runs the logic suite (`python -m pytest tests`,
  audio/model-dep tests excluded) on every push to `main` and every pull
  request. Keep it green; it is the gate that lets the autofix loop below know
  when a change is safe.
- `.github/workflows/perf.yml` is the (heavier) real-model latency benchmark. It
  runs on every push to `main` (post-merge), on manual `workflow_dispatch`, and
  on PRs labelled `perf`. It downloads models (cached; uses the `HF_TOKEN`
  Actions secret), runs `python -m tools.bench`, uploads the report as an
  artifact + the headline table to the job summary, and (on main/dispatch)
  publishes the full HTML report to GitHub Pages at
  https://dobosp.github.io/speaker/ . A GitHub CPU is a repeatable baseline, not
  phone silicon — read it as calibration against `specsim`.
- `android-apk.yml` builds + publishes the mobile APK on pushes touching
  `mobile/**`; `publish-model.yml` republishes the gated Gemma model (see
  `CREDENTIALS.md`).

## Self-monitoring / autofix loop

Put work in a PR, then a Claude session can `subscribe` to that PR's activity to
receive its CI results (from `tests.yml`) and review comments as events, and
push fixes until checks pass. The loop is driven by the live session
(subscription is per-session, not stored state); on merge/abandon, unsubscribe.
Claude can edit pipelines (`.github/workflows/*`) and watch runs, but cannot
trigger/re-run a workflow or create Actions secrets from its in-session toolset
— those need the API with an out-of-band token (`tools/gh_admin.py` +
`CREDENTIALS.md`).

## Environment notes

- Web sessions run in an ephemeral container; commit anything worth keeping
  (commit only — pushing stays gated by the fleet git policy, ADR-0007).
- Pushes may be blocked if the session was provisioned read-only
  (`403 Permission denied`). Surface it — it's an environment permission, not a
  code problem (and `GIT_HUB_TOKEN` does not change it; see `CREDENTIALS.md`).
- `GIT_HUB_TOKEN` (maximum-access GitHub key) + the `tools/gh_admin.py` wrapper:
  fully documented in `CREDENTIALS.md`. Never print tokens.
- Windows landing procedure (guard hook, SSH identity, PR flow):
  `windows_landing_workflow.md`.
