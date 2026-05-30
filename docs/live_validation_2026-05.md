# Live on-hardware validation harness (2026-05)

A self-driving spoken conversation test: a **synthetic user** (a TTS voice
distinct from the assistant's) speaks scripted lines aloud through your real
speakers; the **real assistant** hears them over the air via your real mic,
thinks, and answers aloud. The harness records an **attributed timeline** (every
turn labelled user vs assistant, with the exact audio file + timestamps) and
**per-turn latency**, so you can validate the capabilities end-to-end on your own
machine.

It runs **only on request** — it needs real ASR/TTS/LLM models and audio
hardware, so it is **not** part of the pytest logic suite. `tools/live_session`.

## Run it

```bash
python -m tools.live_session --check        # preflight: are models + audio ready?
python -m tools.live_session --list          # list scenarios
python -m tools.live_session --list-devices  # your audio devices (pick input/output)

# one scenario, or all:
python -m tools.live_session --scenario baseline_latency_single_turn_qa
python -m tools.live_session --all
```

Useful flags: `--device <profile>` (config.json profile), `--llm ollama --model
... --fast-model ...`, `--input-device/--output-device <id>`, `--user-speaker-id
/ --user-speed` (the synthetic user's voice), `--no-assistant-audio`,
`--response-timeout <s>`, `--inject`, `--no-input-gate`.

**Setup:** the assistant captures the real mic, so put the speakers and mic
**near each other** at a **sane volume** so the synthetic user is heard clearly.
The assistant gates ASR during its own playback, so it won't transcribe itself.
A barge-in scenario deliberately speaks over the assistant — keep the room quiet
otherwise.

### `--inject` (clean pipeline test, no acoustic loop)

Over-the-air needs a **well-coupled** mic+speaker. On a laptop's built-in
speaker→mic the STT garbles badly and the mic re-hears the assistant's own TTS
(feedback) — so on such hardware use **`--inject`**: the synthetic-user audio is
fed **straight into the recognizer** (the real mic is never opened) and the
assistant's TTS goes to a **null sink**. You still exercise the **real**
STT→LLM→TTS pipeline + the brain on **clean** audio — latency comes from metrics,
the assistant track is re-synthesized for the artifact — but with no acoustic
degradation, no feedback, and no flaky-device crashes. `--inject` does **not**
test the literal mic/speaker or barge-in (interruption needs live audio during
playback + AEC). See `docs/live_validation_run_2026-05-30.md` for the first run's
analysis (and why each path was needed). `--no-input-gate` disables the
ACT/INGEST addressing gate, useful when garbled over-the-air STT gets INGEST'd.

## What it produces (per scenario, under `logs/live/<run-id>/<scenario>/`)

- `timeline.json` — the attributed timeline. Every event has `speaker`
  (`user`/`assistant`), the **exact `audio` file** that produced/played it,
  timestamps, and — for assistant turns — the latency breakdown. A user event's
  `asr_final` is **what the assistant heard** (the transcript the user's audio
  produced), closing the loop played-audio → recognition → answer.
- `latency.json` — per-turn `first_audio_latency` (SPEECH_END → first assistant
  audio) + the stage breakdown (endpoint / final→token / token→audio), and the
  aggregate floor.
- `summary.md` — a human-readable, **gradeable** report: the attributed
  conversation, the latency table, and the scenario's expected behavior / pass
  signals / failure modes.
- `user/NN.wav` — the **exact** synthesized user audio that was played.
- `assistant/NN.wav` — a clean re-synthesis of what the assistant said (the
  separable assistant track; `--no-assistant-audio` to skip).
- `heard_over_air.wav` — the **ground-truth** mic recording (what the assistant
  actually heard over the air — the user + the assistant's own voice mixed).

## Differentiating the two speakers

Attribution never relies on the acoustic difference: the harness *controls* the
user audio (it synthesizes + plays it), so it always knows which side is which.
The distinct user voice (different speaker id / speed) is so a *human* listening
to `heard_over_air.wav` can tell them apart. The `timeline.json` is the source of
truth: each event is labelled and points at its exact audio file.

## The scenarios (what each validates)

| Scenario | Capability |
|---|---|
| `baseline_latency_single_turn_qa` | latency floor + clean attribution on the easy path |
| `context_aggregation_its_population` | short-term memory ("what's *its* population?") |
| `addon_continuation_merge_and_queue` | ADD-ON / continuation (one answer, not two racing) |
| `self_awareness_enumerate_do_decline` | knows its real skills, doesn't confabulate |
| `smart_endpoint_hold_vs_crisp` | endpoint hold-on-pause vs crisp (enable `sherpa.endpoint_enabled`) |
| `barge_in_interrupt_stop` | interrupt promptly, no stale audio after stop |
| `never_stuck_heavy_then_recover` | heavy turns recover; controller never wedges |

Timing of each user line (how it's scheduled relative to the assistant):
`wait_for_response` (speak, wait until idle) · `immediately` (tack on an add-on
before the answer) · `barge_in` (speak over the assistant mid-answer) ·
`pause:<N>` (silent gap, then wait).

## Caveats

- Over-the-air acoustics vary by room/hardware; treat absolute latency numbers as
  *your* machine's, and use them as a baseline to track regressions, not a
  universal spec.
- `smart_endpoint` only changes behavior when `sherpa.endpoint_enabled` is true
  (default off / experimental); otherwise the scenario just measures the
  fixed-timer endpoint latency.
- A true capability *hang* is hard to force live; `never_stuck` mainly confirms
  the controller stays responsive across a long run.
- The assistant track is a faithful **re-synthesis** of the spoken text (same
  voice), not the exact played PCM; `heard_over_air.wav` is the ground truth.
- The synthetic user opens its own output stream while the assistant's playback
  stream is also open. Most backends mix concurrent output streams fine; on an
  exclusive-mode backend, point them at the same `--output-device` (a per-scenario
  failure is caught and reported, not fatal to the run).
- Assistant turns are captured from what the engine actually `speak()`s, so a
  **barged/interrupted** answer IS recorded (flagged `interrupted: true` in the
  timeline) — the start of the story you cut off shows up.
```
