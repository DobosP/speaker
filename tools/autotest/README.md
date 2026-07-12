# `tools.autotest` — autonomous (no-human) test harness

Drives the **real** runtime end-to-end with **no person at the mic**, so a
session can self-verify STT / TTS / barge-in / memory and emit a committable
verdict. Sits between `tools.live_session` (needs a human) and `--engine replay`
(only re-plays recordings).

```
python -m tools.autotest all                         # memory + voice + suite + scorecard
python -m tools.autotest memory  [--llm echo|ollama] [--model minicpm5-1b:q8]
python -m tools.autotest voice   [--acoustics cable|delay|speaker] [--utterances DIR] [--make-sound]
python -m tools.autotest replay  [--bundle logs/runs/run-<id>.wav]
python -m tools.autotest suite                       # existing headless pytest gates
```

Use the shipped answering LLM (default `minicpm5-1b:q8`) — this tests the audio/memory
plumbing, not model intelligence. Run from the repo root with the venv python:
`.venv/bin/python -m tools.autotest …`. Reports land under `logs/autotest/`
(gitignored); the `voice` tier also drops a normal `logs/runs/` bundle.

## The `voice` tier: real injection + three acoustic paths

It runs the real `sherpa` engine and injects "user" utterances on a real
timeline — either the engine's **synthesized** voice (default) or **your real
recordings** (`--utterances DIR`). STT is scored by **WER** (word error rate)
against each clip's ground-truth transcript.

The acoustic path (`--acoustics`) is pluggable — and which one to use depends on
what you're testing, because the assistant's own audio interacts with the mic:

| mode | echo? | sound? | best for |
|------|-------|--------|----------|
| `cable` (default) | no (playback → dead sink) | silent | **STT accuracy + round-trip** — clean, reproducible (digital injection = a perfect near-field user). Does **not** test self-interrupt/barge (no echo). |
| `delay` | yes (loopback + ~260 ms air-gap) | silent | **self-interrupt + barge-in** with the AEC reference aligned the way it is on a real speaker. |
| `speaker` | yes (real speaker + mic) | **audible** | **true over-the-air** — real ~260 ms acoustic delay + room/speaker coloring. The genuine open-speaker condition; calibrates STT/TTS under real echo. Needs `--make-sound`. |

Why the split: in a loopback/over-the-air path the assistant's TTS raises the
engine's learned **echo floor**, which then drops quiet user clips as
"echo/ambient". A real user is near-field (loud) so they pass; an injected clip
must either avoid the echo (cable) or out-shout it (the gain boost in
delay/speaker). And barge-in/self-interrupt only *exist* when there's echo — so
they're tested in `delay`/`speaker`, while `cable` gives the clean STT number.

For the **truest STT number that predicts your app**, inject your real
recordings in `cable` mode (clean near-field). For **realism** (echo, barge,
self-interrupt), use `speaker`.

**Near/far over-the-air** (`speaker` mode): by default the assistant and the
injected user clips share one speaker (both far-field). Pass `--inject-sink
<pactl-sink>` to play the user clips through a **different** speaker — e.g. the
assistant on an external speaker set as the default sink, and the user clips on
the laptop speaker next to the mic — for real near/far separation (the faithful
open-speaker scenario). Mind the levels: capturing speaker *playback* through the
mic needs more gain than capturing a near-field voice, and OTA re-capture
degrades STT (judge STT from `cable`, not `speaker`). See
`docs/session_2026-06-19_real_voice_ota_setup.md` for a worked JBL setup.

A hot mic ruins everything (clipping garbles STT, unrecoverably). On PipeWire the
knob that sticks is the **source volume**, not the ALSA mixer:
`pactl set-source-volume @DEFAULT_SOURCE@ 40%`. Verify with `record --check`.

## Recording your own utterances

Easiest — the **guided recording studio**: it shows each line, counts down
`3..2..1..GO`, records while you speak and auto-stops when you finish, trims the
silence, takes a short break, and writes `manifest.json` for you:

```
.venv/bin/python -m tools.autotest record --out recordings/owner
.venv/bin/python -m tools.autotest record --group questions --group commands   # subset
.venv/bin/python -m tools.autotest record --review                              # keep/redo each take
.venv/bin/python -m tools.autotest record --dry-run                            # just print the script
```

The built-in script covers questions, instant commands, long prompts, barge-in
talk-overs, a memory fact/recall pair, harder/natural speech, and a
**corrections** group — self-corrections ("five — no, ten minutes"),
repeats/stutters ("what what time is it"), and stretched words ("Lonnndon") that
test the final text fed to the LLM and whether the system makes sense of messy
input (the report prints each final text → the assistant's reply so you can
judge). Recordings go to a **gitignored** dir (your raw voice stays local, per
architecture §9.7). Then point the voice tier at it: `--utterances recordings/owner`.

You can also hand-build the directory: one sentence per file, **mono WAV**
(16 kHz or 48 kHz), with a `manifest.json`:

```json
{"clips": [
  {"file": "capital.wav", "text": "what is the capital of france", "role": "round_trip"},
  {"file": "weather.wav",  "text": "what's the weather like today",  "role": "round_trip"},
  {"file": "story.wav",    "text": "tell me a short story about a sailor", "role": "speak"},
  {"file": "planets.wav",  "text": "tell me about the planets",      "role": "speak"},
  {"file": "barge.wav",    "text": "wait stop for a second",         "role": "barge"},
  {"file": "stop.wav",     "text": "stop",                           "role": "command"}
]}
```

Roles: `round_trip` (ask → expect a reply; record several for WER stats),
`speak` (longer prompts — `speak[0]` drives the self-interrupt window,
`speak[1]` the barge-in window), `barge` (a talk-over phrase; not WER-scored —
it deliberately overlaps), `command` (a KWS word). `text` is the ground truth.
Any omitted role falls back to a synthesized clip.

## Tiers at a glance

| tier | proves | needs |
|------|--------|-------|
| `memory` | a fact in turn 1 is recalled at turn N — injected into the prompt AND used in the answer | in-RAM memory (no DB); small LLM |
| `voice`  | real engine round-trips STT→LLM→TTS; WER; barge-in cuts; self-interrupt | PipeWire, sherpa models, small LLM |
| `replay` | delay-independent self-interrupt + AEC ERLE/delay over a bundle | a `run-<id>.wav` + `.ref.wav` |
| `suite`  | the existing headless barge/sandbox/memory pytest gates | nothing extra |

Verdict gates on the robust signals (audio flowed + round-trip + bundle). All
PipeWire devices it creates are reversible (unloaded on exit); the system
default sink/source and your config are restored verbatim.
