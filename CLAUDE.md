# CLAUDE.md

Guidance for Claude Code working in this repo. Read this first every session.

## What this project is

A local-first, real-time voice assistant (`ASR → LLM → TTS`) with barge-in and
a mode-based control plane. The desktop Python runtime in `core/` is the
reference implementation; an on-device **Android app** lives in `mobile/`
(Flutter), and a **host + thin-client** path (`remote/` + `web/`, over
LiveKit/WebRTC) lets a browser or phone talk to one running brain. The **goal**
is an always-listening, mode-based assistant across
Linux/Windows/macOS/Android/iOS; the always-on capture/respond loop runs
on-device (raw audio never leaves), with an optional cloud *thinking tier*
(main planner / research / multimodal summarize / web search) for work that
exceeds local headroom. The boundary is `docs/target_architecture.md` §9.7.

> **Cross-platform shape (decided):** one portable **core** + thin **per-platform
> shells** — *not* a monolith, *not* independent apps. Platforms share the
> `always_on_agent` **`AgentEvent`/`Mode` contract** (plus its tests); the small
> brain is reimplemented per runtime (Python on desktop/server, Dart on mobile).
> Deployment topology is **hybrid**: on-device first for the always-on loop
> (STT/TTS/VAD/speaker-ID + the fast LLM tier stay local; raw audio never
> leaves), with the `remote/` host path as the iOS-background story and the
> instant-reach fallback. The thinking tier (main LLM / research / multimodal
> summarize / web search) may use cloud — see §9.7. Full rationale + the
> resolved decisions live in **`docs/target_architecture.md`** §9 — read it
> before structural changes.
>
> The refactor removed the hand-rolled audio stack in favour of `sherpa-onnx`;
> the `always_on_agent/` brain is kept and made real; the old `main.py` monolith
> is **deleted**. The desktop runtime is `core/` (`VoiceRuntime`): a swappable
> `AudioEngine` (sherpa-onnx production, scripted for tests, LiveKit for remote)
> wired to the brain with real LLM-backed, cancellable capabilities. Try it
> without audio: `python -m core --engine console --llm echo`.

## Layout

- `core/` — **the runtime (all new work goes here).** `engine.py` (the
  `AudioEngine` seam), `engines/sherpa.py` (production, on-device; CPU STT/TTS
  with auto-tuned threads + explicit `provider`), `engines/scripted.py`
  (tests/console), `engines/livekit.py` (WebRTC transport for the remote
  host+thin-client path), `engines/speaker_gate.py` (speaker-ID barge-in gate),
  keyword-spotter **command fast-path** (sherpa KWS runs alongside ASR and
  fires `on_command`; the runtime maps it to a control event via the
  `commands` config block — instant actions like "stop" with no LLM in the loop),
  `llm.py` (the `LLMClient` protocol + `EchoLLM` fake, `OllamaLLM` for desktop
  GPU, `LlamaCppLLM` for on-device GGUF; all accept optional `images=` for
  multimodal Gemma 3), `capabilities.py` (LLM-backed cancellable providers;
  two-model split — fast model answers, main/multimodal model researches),
  `runtime.py` (`VoiceRuntime` orchestrator), `app.py` (CLI; builds models from
  the `llm` config block and applies the selected device profile). Run:
  `python -m core --engine console --llm echo`.
- `always_on_agent/` — the **control-plane "brain"** (modes, priority event bus,
  supervisor, cancellable threaded tasks, intent analyzer). The keeper. See its
  `README.md` and `docs/always_on_agent_layer.md`. Its `events.py`
  (`AgentEvent`/`Mode`) is **the shell↔core contract** every platform shares.
- `mobile/` — **on-device Android app** (Flutter): ASR/LLM/TTS fully local via
  `sherpa_onnx` + `flutter_gemma` (Gemma 3 1B, MediaPipe/LiteRT). Today it is a
  **parallel Dart loop** (`lib/assistant.dart`) that re-derives core behavior
  (command fast-path, streaming TTS) rather than sharing the brain — the planned
  convergence is onto the `AgentEvent` contract. See `mobile/README.md`.
- `remote/` — **host + thin-client path** (optional; `requirements-remote.txt` +
  `LIVEKIT_*`). `token_server.py` (FastAPI: mints LiveKit tokens, a text `/chat`,
  serves `web/`), `worker.py` (joins a LiveKit room running the full Python brain
  via `--engine livekit`). `web/index.html` is the browser client.
- `utils/memory.py` (+ `memory_writer.py`, `memory_config.py`) — Postgres-backed
  smart memory (the only surviving `utils/` modules). See `MEMORY.md`. Keep;
  will move to SQLite on mobile.
- `tests/` — pytest. `tests/sandbox/` is the device-simulation harness
  (latency/LLM-weight profiles + simulated engine/LLM) for middle-layer tests;
  `test_core_runtime.py` is fast logic; `test_sandbox_middle_layer.py` is
  realistic-timing/concurrency. No audio/model deps.
- `tools/` — dev tooling (no app code). `run_tests.py` + `testing/` (staged
  pytest runner with reports under `test-reports/`); `specsim/` (machine-spec
  simulator that renders an HTML capability report — see Conventions);
  `cloudchat.py` (parallel cloud-LLM REPL: fires N prompts in parallel at the
  endpoint in `config.llm.cloud`, streams them with `[Qn]` prefixes, `/cancel`
  hard-closes the HTTP stream so the provider stops billing; needs `openai`);
  `recommend_profile.py` (stdlib hardware probe → prints which
  `device_profiles` entry to use; see `docs/target_architecture.md` §10).
- `config.json` — runtime config. `docs/` — architecture and subsystem docs.

> The legacy stack (`main.py`, `utils/audio.py`, the hand-rolled STT/TTS/LLM
> plumbing, `benchmarks/`, `scripts/`, and their tests) was deleted in the
> refactor. Don't try to import them.

## Conventions

- Python, standard `pytest`. Run tests: `python -m pytest tests -q`. For staged
  runs with structured reports (per-stage + a tabular run summary under
  `test-reports/`), use `python tools/run_tests.py list|core|sandbox|memory|cloud|imports|full`.
  The `imports` stage is a whole-tree import smoke that catches syntax errors
  and missing optional libs across `core/`/`always_on_agent/`/`remote/`/`tools/`
  before any logic test runs — use it as a "does the code compile and are the
  libraries present" preflight.
- Run the app: `python -m core --engine console --llm echo` (no audio/models);
  `python -m core --engine sherpa` for on-device audio;
  `python -m core --engine replay --replay-dir <dir>` to run the real engine
  headless over recorded `.npy`/`.wav` fixtures (no sound card);
  `python -m remote.worker` for the host+thin-client path (joins a LiveKit room;
  needs `LIVEKIT_*`). Engines: `--engine {console,sherpa,replay,livekit}`; LLM:
  `--llm {echo,ollama,llamacpp}`.
- Latency instrumentation: `core/metrics.py` records per-turn stage timings
  (`speech_end → asr_final → llm_first_token → tts_first_audio`, plus
  `barge_in → barge_in_stop`) via `runtime.metrics`. The real engine, the
  file-replay engine, and the sandbox sim engine all feed it through the
  `on_metric` callback, so measured and simulated numbers share one shape.
- **Run logs & debugging — read [`docs/debugging.md`](docs/debugging.md) when the
  user reports a failed/stuck/slow run or attaches files.** Every run writes a
  committable bundle under `logs/runs/run-<id>.{txt,summary.json}` (+ `.wav` with
  `--record`): the `summary.json` has `stuck_hints`, per-turn latencies, every
  LLM request, the conversation transcript, CPU/GPU/RAM telemetry, and errors;
  the `.txt` is the full async DEBUG trace. `./session.sh` captures everything in
  one go (`--debug` = console only; `--record` = audio). Recorded WAVs replay via
  `python -m core --engine replay --replay-dir logs/runs` and become regression
  tests. Preflight with `python -m tools.doctor`. `pytest` runs write the same
  shape under `logs/tests/`. Capture is built to stay off the hot path: logging
  is fully async (a queue handler that defers all formatting + I/O to one
  background listener thread), recording uses a writer thread, and telemetry is
  sampled on its own thread — so `--debug`/`--record` don't slow the real-time
  pipeline. See `docs/debugging.md` §Performance for the reliability tradeoffs.
- Real-model latency benchmark: `python -m tools.bench --fake` is a no-download
  plumbing smoke test; `python -m tools.bench --profile phone --fixtures
  tests/fixture_audio/virtual_real_world` fetches small Gemma GGUF (via
  `$HUGGINGFACE_TOKEN`) + sherpa ONNX and runs the REAL ASR→LLM→TTS pipeline
  over fixtures, writing a measured-vs-`specsim`-budget report under
  `test-reports/perf/`. Model coordinates are overridable via a
  `--models-manifest` JSON / `SPEAKER_BENCH_*` env vars.
- LLM/device config (`config.json`): the `llm` block selects a `backend`
  (`ollama` desktop-GPU, or `llamacpp` on-device GGUF) plus a `main_model`
  (large/multimodal) and `fast_model` (snappy replies). `device_profiles`
  (`desktop`, `phone`, …) are shallow-merged over the base per section; pick one
  with `--device <name>` (default from `config.device`). Desktop runs
  gemma3:12b + 4b on Ollama/GPU; the `phone` profile runs the **Python core**
  under phone-like limits (small Gemma 4b/1b GGUF on `llama.cpp`, STT/TTS threads
  dialed down) — Ollama is desktop-only. Note: the **shipped Flutter app**
  (`mobile/`) is separate from these profiles and uses `flutter_gemma`
  (MediaPipe/LiteRT), not `llama.cpp`, for its on-device LLM.
- Simulate specs without hardware: `python -m tools.specsim` renders
  `test-reports/specsim/index.html` (model-fit + responsiveness matrix + per-
  device ASR→LLM→TTS timelines across 4090/Mac/Windows/phone/web). Numbers are
  modelled estimates, not measurements — calibrate `tools/specsim/specs.py` from
  real runs before trusting absolutes.
- Keep new control-plane logic in `always_on_agent/`, typed and testable, not in `main.py`.
- Prefer replay/transcript tests over tests that require live audio devices.
- **Local/cloud boundary (`docs/target_architecture.md` §9.7).** STT, TTS,
  VAD, speaker-ID, the always-on capture loop, and the fast/answering LLM
  tier stay **on-device** — raw audio must never leave the device. The
  *thinking tier* (main planner / research / multimodal summarize) and web
  search **may use cloud**; only post-ASR text + screen captures + files
  given to the assistant may cross over, and only when invoked. The earlier
  "no cloud STT/LLM/TTS by default" stance is superseded by this boundary;
  the always-on loop is still fully local.
- CI: `.github/workflows/tests.yml` runs the logic suite (`python -m pytest tests`,
  audio/model-dep tests excluded) on every push to `main` and every pull request.
  Keep it green; it is the gate that lets the autofix loop below know when a
  change is safe. `.github/workflows/perf.yml` is the (heavier) real-model
  latency benchmark. It runs on every push to `main` (post-merge), on manual
  `workflow_dispatch`, and on PRs labelled `perf`. It downloads models (cached;
  uses the `HF_TOKEN` Actions secret), runs `python -m tools.bench`, uploads the
  report as an artifact + the headline table to the job summary, and (on
  main/dispatch) publishes the full HTML report to GitHub Pages at
  https://dobosp.github.io/speaker/ . A GitHub CPU is a repeatable baseline, not
  phone silicon — read it as calibration against `specsim`.
  `android-apk.yml` builds + publishes the mobile APK on pushes touching
  `mobile/**`; `publish-model.yml` republishes the gated Gemma model (see
  Environment / git).

## Environment / git

- `main` is the integration branch and holds the latest work. Do feature work
  on a short-lived branch and merge back to `main`.
- **Standing session workflow (durably authorized — do not ask per action).**
  When a session's work is complete and the logic suite is green
  (`python -m pytest tests -q`), Claude may, without stopping to ask:
  commit, push, **merge the short-lived branch into `main`, push `main`, and
  delete that now-merged branch** (local; also remote if it was pushed). This
  supersedes the older "only commit/push/merge when the user explicitly asks"
  stance — a successful session ends by landing on `main` and cleaning up its
  branch. **Still pause and confirm** only when: tests are red or the work is
  not actually finished; the operation is destructive *beyond* deleting the
  just-merged branch (e.g. force-push, history rewrite, deleting an *unmerged*
  or shared branch); or `main` itself would regress. Surface a one-line summary
  of what was merged/pushed/deleted so the user can see it after the fact.
- Web sessions run in an ephemeral container; commit anything worth keeping.
- **Secrets & tokens live in [`CREDENTIALS.md`](CREDENTIALS.md)** — the single
  source of truth for every credential (the CI `HF_TOKEN` Actions secret, the
  session `HUGGINGFACE_TOKEN` and `GIT_HUB_TOKEN` env vars, and `LIVEKIT_*`):
  where each comes from, what it unlocks, and how it's consumed. **Golden rule
  for all of them:** read from the env at runtime; **never** hard-code, echo, or
  commit a token — reference it only as `$VAR`.
- `GIT_HUB_TOKEN` is the **maximum-access** key: it performs the privileged ops
  the session harness blocks — branch deletion, `workflow_dispatch`, re-running
  Actions, reading/writing Actions secrets. The `tools/gh_admin.py` helper wraps
  these (dry-run by default; `--execute` to send, `--yes` for destructive ops)
  and never prints the token; raw `curl` recipes are in `CREDENTIALS.md`.
- NOTE: pushes may be blocked if the session was provisioned read-only
  (`403 Permission denied`). If so, surface it — it's an environment permission,
  not a code problem (and `GIT_HUB_TOKEN` does not change it).
- Self-monitoring / autofix loop: put work in a PR, then a Claude session can
  `subscribe` to that PR's activity to receive its CI results (from
  `tests.yml`) and review comments as events, and push fixes until checks pass.
  The loop is driven by the live session (subscription is per-session, not
  stored state); on merge/abandon, unsubscribe. Claude can edit pipelines
  (`.github/workflows/*`) and watch runs, but cannot trigger/re-run a workflow
  or create Actions secrets from its in-session toolset — those need the API
  with an out-of-band token.

## When unsure

Ask clarifying questions before large changes. `docs/PROJECT_KICKOFF.md` is the
running list of product decisions; check it for current intent and open items.
