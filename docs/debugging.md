# Debugging with run logs, recordings & telemetry

This is the **debugging contract** for the desktop runtime: what artifacts a run
produces, what's in them, and how to use them to diagnose a problem (or build a
regression test from a real session). When the user reports "it got stuck / it
failed / it's slow" and attaches files, **read this, then read their files.**

Everything a run produces is grouped under **`logs/runs/`** by a shared run id:

```
logs/runs/run-<id>.txt           full DEBUG log (always written)
logs/runs/run-<id>.summary.json  condensed digest   (always written)
logs/runs/run-<id>.wav           session audio       (only with --record)
```

`.txt` / `.json` / `.wav`-under-`logs/runs/` are git-committable on purpose (the
global `*.wav` ignore is overridden there), so the user can push a failing run
and we can replay it.

---

## How to capture a run (what the user runs)

Three independent axes, one shared bundle:

| Axis | Flag | Effect |
| --- | --- | --- |
| Run bundle | *(always)* | writes `run-<id>.txt` + `run-<id>.summary.json` |
| Console verbosity | `--debug` (or `SPEAKER_DEBUG=1`) | mirrors DEBUG to the terminal; **file is always full DEBUG regardless** |
| Audio | `--record` | also writes `run-<id>.wav` (16 kHz mono, replayable) |

One-command full capture: **`./session.sh`** (= `--debug --record`, sherpa engine;
`ENGINE=console ./session.sh` for a text session). Extra args pass through.

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

- **`meta`** — `engine`, `llm`, `device`, `mode`, `model`, `fast_model`, and
  `recording` (path) when `--record` was used.
- **`stuck_hints`** — plain-English flags computed for you, e.g.
  *"no LLM request was ever issued (ASR never produced a final?)"*,
  *"every LLM request was cancelled"*, *"empty transcript"*,
  *"N error(s) — see the .txt traceback"*. **Start here.**
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
   `python -m core --engine replay --replay-dir logs/runs --debug`.
5. **Freeze it as a test:** the recorded WAV is the same format the replay
   loader (`core/engines/file_replay.py: load_waveform`) consumes, so a captured
   session becomes a deterministic regression fixture (see
   `docs/testing_audio.md`).

---

## Preflight & setup (when a run won't even start)

- **`python -m tools.doctor`** — checks Python, required imports, sherpa model
  paths, Ollama reachability + models, and audio devices; prints the exact fix
  command for each failure and a `READY` / `NOT READY` verdict.
- **`python -m tools.setup_models`** — downloads the sherpa ASR/VAD/TTS models
  and wires their paths into `config.json` (the native path needs this).
- **`./install.sh`** — one-command native install (clean venv + deps + models +
  doctor). Deps include `psutil` (telemetry); GPU telemetry uses `nvidia-smi`.

---

## Test-run logs

Every local `pytest` run also writes a committable bundle under **`logs/tests/`**:

```
logs/tests/tests-<id>.txt           full DEBUG log of the session
logs/tests/tests-<id>.summary.json  {counts, failures (with tracebacks), slowest}
```

Disable with `SPEAKER_TEST_LOG=0`. Useful when a test fails only on the user's
machine — they push the bundle and the summary names the failing test + the
slowest tests.

---

## Source map

- `core/runlog.py` — async logging (`QueueHandler`/`QueueListener`), `RunSummary`
  aggregation, crash-safe `finalize()`.
- `core/recorder.py` — background-threaded WAV writer (`WavRecorder`).
- `core/sysinfo.py` — `snapshot()` + `SystemMonitor` (CPU/GPU/RAM).
- `core/engines/sherpa.py` — capture/playback instrumentation + recorder hook.
- `tools/doctor.py`, `tools/setup_models.py`, `install.sh`, `session.sh`.
- Tests: `tests/test_runlog.py`, `tests/test_recorder.py`,
  `tests/test_sysinfo.py`, `tests/test_setup_doctor.py`.
