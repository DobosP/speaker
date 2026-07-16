# Debugging with run logs, recordings & telemetry

This is the **debugging contract** for the desktop runtime: what artifacts a run
produces, what's in them, and how to use them to diagnose a problem (or build a
regression test from a real session). When the user reports "it got stuck / it
failed / it's slow" and attaches files, **read this, then read their files.**

Everything a run produces is grouped in one selected bundle directory by a
shared run id. Direct runs default to `logs/runs/`; `./live.sh` creates a unique
private directory under `logs/live/`:

```
<bundle>/run-<id>.txt           full DEBUG log (always written)
<bundle>/run-<id>.summary.json  condensed digest   (always written)
<bundle>/run-<id>.wav           processed mic audio (with --record)
<bundle>/run-<id>.ref.wav       aligned playback reference (when selected)
```

New run/test/live bundles are ignored and stay on-device. They can be replayed
and diagnosed directly from their local directory.

> **Privacy — raw voice never leaves the device (architecture §9.7).** A run
> bundle can contain *raw voice* (`run-<id>.wav`), *verbatim transcripts*
> (`summary.json` → `transcript[].text`), and *full prompts* (the `.txt` DEBUG
> trace). Do **not** stage, push, upload, or paste a real live bundle. Give a
> local agent only its directory name and inspect the summary first. A separately
> synthesized or non-sensitive fixture may be reviewed for publication under an
> explicit decision; the private original remains unchanged and local.

---

## How to capture a run (what the user runs)

Three independent axes, one shared bundle:

| Axis | Flag | Effect |
| --- | --- | --- |
| Run bundle | *(always)* | writes `run-<id>.txt` + `run-<id>.summary.json` |
| Console verbosity | `--debug` (or `SPEAKER_DEBUG=1`) | mirrors DEBUG to the terminal; **file is always full DEBUG regardless** |
| Audio | `--record` | also writes `run-<id>.wav` (16 kHz mono, replayable) |

One-command physical Linux capture: **`./live.sh`**. It locks the session,
starts/reuses loopback Ollama only when the selected profile uses that backend,
prepares the PipeWire EC route, and runs the applicable shared preflight before
recording both mic and aligned playback reference in a private `logs/live/`
directory. Normal Ollama and llama.cpp profiles require full doctor `READY`;
explicit echo uses the applicable base/deferred preflight. Cleanup restores
only launcher-owned state (ADR-0075).

**`./session.sh`** remains the lower-level `--debug --record` wrapper for an
already-prepared route (`ENGINE=console ./session.sh` for text). Direct
`python -m core` remains portable and performs no automatic host-service or
default-audio-route provisioning.

The bundle is written on a clean exit, on `Ctrl-C`, **and on an unhandled
exception** (`finalize()` is idempotent and also runs at interpreter exit), so a
crash still leaves artifacts.

Performance note: logging and recording are **off the hot path** — the
`speaker` logger only enqueues (a background `QueueListener` thread does the disk
writes + summary aggregation), and the recorder hands audio blocks to a writer
thread over a queue. So capturing does not slow the real-time pipeline.

---

## Performance & reliability (why capturing is cheap)

Capturing is designed to **never block the real-time audio/LLM threads**:

- **Logging is fully async.** The `speaker` logger holds a custom
  `_ThreadQueueHandler` whose only job on the hot path is to drop the raw
  `LogRecord` onto an in-memory queue — no string interpolation, no traceback
  rendering, no disk I/O (we override the stock `QueueHandler.prepare()` that
  would otherwise format on the calling thread). A single background
  `QueueListener` thread does *all* formatting + disk writes + summary
  aggregation.
- **Recording is async.** The capture loop copies a block (~6 KB) and enqueues
  it; a dedicated writer thread does the float32→int16 conversion and the WAV
  writes. A bounded queue drops (and counts) under backpressure rather than
  stalling capture — it never does in practice at 16 kHz mono.
- **Telemetry is sampled, not continuous.** `SystemMonitor` runs on its own
  thread at a 10 s interval (plus one-off baseline/mark/final reads), so
  `nvidia-smi`/`psutil` cost stays off the main thread and infrequent.
- **The summary is written once**, at `finalize()` — not per event.

Reliability tradeoffs we chose: the background listener flushes **per record**
(not buffered into large chunks), so a `Ctrl-C` or crash still leaves a complete
`.txt`; and `finalize()` is idempotent + `atexit`-registered so the
`.summary.json` is written on clean exit, interrupt, or unhandled exception.
The only thing a hard `SIGKILL` can lose is the in-flight queue tail. Net: hot
path pays ~a queue append per log line; everything heavy is on background
threads that communicate via queues/events only.

---

## `run-<id>.summary.json` — read this first

Top-level keys:

- **`meta`** — `engine`, `llm`, `device`, `mode`, `model`, `fast_model`,
  `recording` (path) when `--record` was used, and `playback_reference` when the
  aligned reference was selected.
- **`stuck_hints`** — plain-English flags computed for you. Two sources feed
  this list:
  - *post-hoc* checks on the whole bundle: *"no LLM request was ever issued
    (ASR never produced a final?)"*, *"every LLM request was cancelled"*,
    *"empty transcript"*, *"N error(s) — see the .txt traceback"*;
  - *live watchdog* (`core/watchdog.py`) warnings promoted at finalize time:
    *"LLM stalled mid-turn"*, *"TTS stalled mid-turn"*, *"capture thread went
    silent"*, *"barge-in gate flapping"*. The watchdog runs on a daemon
    thread alongside the session and logs a WARNING the moment it detects a
    stalled stage — so a run that froze for 15s mid-turn no longer looks
    clean. **Start here.**
- **`counts`** — `llm_requests`, `turns`, `transcript_entries`, `errors`,
  `warnings`, and `log_lines_by_level`.
- **`transcript`** — ordered conversation: `[{role: user|assistant, text,
  mode?, at_sec}]`. `at_sec` is seconds since run start (the "stage timing").
- **`turns`** — per-turn latencies (seconds): `endpoint_latency`,
  `final_to_first_token`, `first_token_to_audio`, `first_audio_latency`,
  `barge_in_latency`. `null` means that stage never fired (itself a clue).
- **`llm`** — `total_time_sec`, `avg_time_sec`, and `requests[]`, one per call:
  `model`, `prompt_chars`, `prompt_preview`, `duration_sec`, `ttft_sec`
  (time-to-first-token), `out_chars`, `tokens`, `streamed`, `cancelled`
  (`true` when a barge-in cut generation off mid-stream).
- **`system`** — compute telemetry (see below).
- **`errors`** — last 50 WARNING/ERROR records: `{t, level, logger, message,
  exc}` (`exc` is the full traceback when present).

### `system` block (CPU / GPU / RAM)

`{baseline, final, peak, marks, samples, deltas}`. Each reading has system
`cpu_percent`, `ram_used_mb` / `ram_total_mb` / `ram_percent`, this process's
`proc_rss_mb` / `proc_cpu_percent`, and `gpu` (a list, per device:
`util_percent`, `mem_used_mb`, `mem_total_mb`, `temp_c`).

- `baseline` = before models load; `marks.after_build` = after clients/engine
  built; `final` = at shutdown; `peak` = high-water marks across the run.
- `deltas` = `final - baseline` for `ram_used_mb`, `proc_rss_mb`,
  `gpu_mem_used_mb` → **how much the run consumed**.
- **The LLM runs in the Ollama process, not ours** — so `gpu_*` is system-wide
  (reflects Ollama), while `proc_*` is this Python process (sherpa STT/TTS +
  runtime). Missing fields mean `psutil` or `nvidia-smi` wasn't available.

Quick one-off reading without a run: `python -m core.sysinfo`.

---

## `run-<id>.txt` — the full trace

Async DEBUG log from these loggers (grep by prefix):

| Logger | What it tells you |
| --- | --- |
| `speaker.sherpa` | input/output device names, which models loaded, **capture heartbeat with mic RMS level** (warns when input is ~silent), ASR partials/finals, barge-in, playback rate, recorder path, and **worker-thread tracebacks** |
| `speaker.runtime` | `final -> brain` (user utterance), `assistant` (spoken text), TTS requests |
| `speaker.supervisor` | intent **decisions** (`kind`, `confidence`, `reason`) — "how the brain thinks" — and `cancel_all` |
| `speaker.tasks` | task start / completed (with duration) / cancelled / **FAILED (with error)** |
| `speaker.llm` / `speaker.llm.ollama` | which tier answered; per-request model, **full prompt** (DEBUG), ttft, duration, tokens, cancelled |
| `speaker.sysinfo` | baseline / periodic / mark / final compute readings |

---

## Debugging playbook (how to use these)

1. **Open `summary.json`.** Read `stuck_hints` and `errors` first, then `counts`,
   `turns` (latencies), `llm.requests` (timings/cancellations), and
   `system.deltas` (resource pressure).
2. **Map symptom → likely cause:**
   - silent / "does nothing" → `models: recognizer=False` in `.txt`, or capture
     heartbeat `avg_rms≈0` (wrong/muted mic) → run `python -m tools.doctor`.
   - heard but no answer → `transcript` has user lines but no assistant; check
     `tasks` for a `FAILED` (Ollama down/model missing) or a task that started
     but never completed (LLM hang).
   - slow → `turns[].final_to_first_token` / `llm.requests[].ttft_sec` (model
     cold? `keep_alive`?) and `system.peak` (VRAM/CPU saturation).
   - cut off → `llm.requests[].cancelled = true` (barge-in storm).
3. **Open `.txt`** around the relevant timestamps for the full trace + traceback.
4. **Reproduce from the recording** (if `.wav` present): replay the exact audio
   through the real pipeline, no mic needed —
   `python -m core --engine replay --replay-dir <bundle-dir> --debug`.
5. **Freeze it as a test:** the recorded WAV is the same format the replay
   loader (`core/engines/file_replay.py: load_waveform`) consumes, so a captured
   session becomes a deterministic regression fixture (see
   `docs/testing.md`).

---

## Reproducing a stuck-state run

The hard ones to debug are runs that *worked but felt wrong*: the model went
quiet for 10 seconds; the assistant talked over you and wouldn't stop; the
mic seemed to give up. The watchdog above flips those into evidence — but
**only if the session is long enough to surface the failure**. A clean
55-second test like `run-20260528-004726.summary.json` won't catch a stuck
case that needs a few minutes of real talking to trigger.

When the assistant has misbehaved in real life and you want to debug it:

1. **Capture a long-form physical session.** `./live.sh` (reversible setup plus
   mic/reference recording). Use `./session.sh` only when the route is already
   prepared.
   Run at least 3 minutes; longer is better. Don't stop early to "save the
   bundle" — the bundle is written on Ctrl-C and on crash, so let the
   misbehavior actually happen.
2. **Stress the patterns you've seen fail** during that session:
   - barge in mid-reply repeatedly (this is what would trip the barge-in
     storm watcher);
   - cause background noise while the assistant is speaking (TTS leaking
     into the mic also trips the storm watcher);
   - ask a long, slow query that should take a few seconds on the main
     model (catches LLM stalls);
   - if you can reproduce a "freeze," let it freeze — don't Ctrl-C; the
     watchdog will warn at the deadline.
3. **Keep the original bundle local.** Record its directory name, then replay it
   with `python -m core --engine replay --replay-dir <bundle-dir> --debug`.
   Preserve the mic/reference pair byte-for-byte. If a regression fixture is
   needed, create a separate synthetic/non-sensitive derivative; never stage or
   push the real-voice original.
4. **Open `summary.json` → `stuck_hints` first.** If the list is non-empty
   the watchdog or post-hoc check has named the failure; if it's still
   empty, the watchdog deadlines may be too generous for what you saw — the
   thresholds live in `core/watchdog.py` (`LLM_FIRST_TOKEN_DEADLINE_SEC`,
   `TTS_FIRST_AUDIO_DEADLINE_SEC`, `CAPTURE_SILENT_DEADLINE_SEC`,
   `BARGE_IN_STORM_*`) and are safe to tune.

---

## Preflight & setup (when a run won't even start)

- **`python -m tools.doctor`** — checks Python, required imports, sherpa model
  paths, selected local-backend readiness, and audio devices; prints the exact fix
  command for each failure and a `READY` / `NOT READY` verdict. On the Linux
  OS-EC path, standalone doctor assumes the transient route is already prepared;
  `./live.sh` prepares it before invoking the same check.
- **`python -m tools.setup_models`** — downloads the sherpa ASR/VAD/TTS models
  and wires their paths into `config.json` (the native path needs this).
- **`./install.sh`** — one-command native install (clean venv + deps + models +
  doctor). Deps include `psutil` (telemetry); GPU telemetry uses `nvidia-smi`.

---

## Test-run logs

Every local `pytest` run also writes an ignored local bundle under **`logs/tests/`**:

```
logs/tests/tests-<id>.txt           full DEBUG log of the session
logs/tests/tests-<id>.summary.json  {counts, failures (with tracebacks), slowest}
```

Disable with `SPEAKER_TEST_LOG=0`. When a test fails only on one machine, inspect
the local summary first; share only a deliberately sanitized extract.

---

## Source map

- `core/runlog.py` — async logging (`QueueHandler`/`QueueListener`), `RunSummary`
  aggregation, crash-safe `finalize()`, watchdog-warning → `stuck_hints` promotion.
- `core/watchdog.py` — live `StuckWatchdog`: per-second inspection of metrics
  anchors + heartbeat + barge-in rate; logs WARNINGs the moment a stage stalls.
- `core/recorder.py` — background-threaded WAV writer (`WavRecorder`).
- `core/sysinfo.py` — `snapshot()` + `SystemMonitor` (CPU/GPU/RAM).
- `core/engines/sherpa.py` — capture/playback instrumentation + recorder hook
  + heartbeat callback into the watchdog.
- `tools/doctor.py`, `tools/setup_models.py`, `tools/live_launcher.py`,
  `install.sh`, `live.sh`, `session.sh`.
- Tests: `tests/test_runlog.py`, `tests/test_watchdog.py`, `tests/test_recorder.py`,
  `tests/test_sysinfo.py`, `tests/test_setup_doctor.py`,
  `tests/test_live_launcher.py`.
