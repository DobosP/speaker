# Session 2026-05-31 — acoustic loop fixed + real-voice speaker-ID

Handoff doc for the over-the-air / acoustic work landed on `main`. Written as a
durable in-repo record (the prior account lived only in Claude's per-user
memory) so the next machine — including a **Windows** continuation — can pick up
without re-deriving any of it.

All code described here is **already on `main`** (and pushed to `origin/main`):

| Commit    | What |
|-----------|------|
| `455348c` | acoustic loop: native-rate mic pin + shared output stream + `tools/audio_mix.py` |
| `5f49a85` | `live_session --barge-in` flag (measure over-the-air self-barge) |
| `95bf6f0` | `tools/voice_id_check.py` — real-voice speaker-ID separability test |

This session also commits the **validation artifacts** those runs produced:
`logs/runs/run-20260530-*` / `run-20260531-*` bundles, `logs/real_usage/`
reports, and `logs/voice_id/s1/` enrollment clips (see *Committed artifacts*).

---

## The headline (resolves weeks of acoustic debugging)

**The entire over-the-air STT failure was the laptop's built-in ALC285 mic — not
the app.** Plugging in an **Audio-Technica AT2020USB-X** condenser mic took
over-the-air STT from garbage/silence to **perfect**: play "what is the capital
of france" out the speaker → the AT2020 hears it → `WHAT IS THE CAPITAL OF
FRANCE` (clean, rms ~0.03, no clip). The pipeline was always fine. Baseline
over-the-air QA went from STT accuracy **0.0 / FAIL** to **1.0 / full_duplex
ok**.

## Two engine root-causes found + fixed (commit `455348c`)

Both fixes are config-gated and default to the safe/old behaviour.

1. **Native-rate mic pin — `sherpa.capture_samplerate`** (`core/engines/sherpa.py`).
   The AT2020 self-muted because the capture fallback chain opened the mic at
   48000 Hz (a *non-native* rate → USB altsetting reconfigure → the mic's
   touch-mute self-engages). The new knob pins the capture rate and never probes
   other rates. Set to **44100** (the AT2020's native rate) in
   `config.local.json`; soxr resamples 44100→16000. Opening at the native rate
   never reconfigures the device, so the mic stays live.

2. **Release output when idle — `sherpa.release_output_when_idle`**
   (`core/engines/sherpa.py`). Once the assistant actually spoke, the loop hit
   `Device unavailable [-9985]`: this box's PortAudio is **exclusive ALSA** (no
   mixing layer), so the engine's TTS output stream and a synthetic user's
   playback can't both hold the one output device. The new flag makes the engine
   close its output stream when the play queue drains (**default OFF** so the
   real app keeps its low-latency held stream; `live_session` enables it for
   non-inject runs). Plus a busy-retry in `synthetic_user.say()`. Turn-taking
   now hands the device back and forth cleanly.

3. **`tools/audio_mix.py`** — combine 2+ audio files into ONE buffer and play
   them as a single stream. The correct way to do competing-voice tests on a box
   where two concurrent OUTPUT streams fail.

## Acoustic scenarios re-run over the air

- `baseline_latency_single_turn_qa`: STT **1.0**, full_duplex ok.
- `context_aggregation`: STT **1.0**, full_duplex ok (context memory works).
- `self_awareness`: full_duplex ok but STT **0.67** — the loop is robust (no
  collisions/mutes) but STT **degrades on long/fast conversational sentences**
  ("double check the price" → "dulce your surprise"). Short QA stays perfect;
  long sentences lose accuracy (SNR / endpointing, not a loop bug).

## Acoustic barge-in (`--barge-in`, commit `5f49a85`)

New `tools/live_session --barge-in` forces `sherpa.barge_in_enabled=True` for a
run to **measure** over-the-air self-barge. Finding: with a co-located speaker
the assistant **self-interrupts on its own TTS** even on the clean AT2020 (the
6 dB level-margin gate can't hold); a latch caps it at 1 self-interrupt/run (vs
the old 12×/6s storm). **Key insight:** mixing the interrupter *into the one
output stream* can NOT enable detection — the level gate sees the mixed
interrupter as part of the assistant's own playback. A true acoustic barge-in
needs either the interrupter on a **separate output device** (a 2nd speaker,
louder than the assistant at the mic — a separate PortAudio stream is allowed)
**or AEC** — **or** the speaker-gated path below.

## Real-voice speaker-ID — the big result (`tools/voice_id_check.py`, commit `95bf6f0`)

This **reverses** the earlier *synthetic* speaker-ID result and opens a
**no-AEC** barge-in path.

`tools/voice_id_check.py` does onset-gated mic capture (pins the native rate),
with enroll / probe / report phases, reusing `core.enroll`. The user read 4
enroll + 4 probe lines into the AT2020. On **clean real audio** the speaker-ID
embedder (CAMPPlus) is **SEPARABLE**:

| Comparison | cosine |
|---|---|
| user held-out probes vs their own enrollment | **0.515 – 0.566** |
| assistant TTS voice vs the user | **≈ 0.01** (orthogonal!) |
| other real humans (`tests/voice_samples`) | **≤ 0.284** |

Margin **0.231**; recommended threshold **~0.4** (the default 0.5 also works —
user floor 0.515 > 0.5). The numbers are in `logs/voice_id/s1/report.json`. The
laptop mic gave 0.30–0.46 (it *rejected* the user); the AT2020 clears 0.5. So
**clean audio fixes the gate for the real user**, and the earlier synthetic
"inversion" was purely a CAMPPlus-vs-libritts-TTS artifact (TTS voices map into a
degenerate region of a model trained on human VoxCeleb).

**Implication for barge-in:** because the assistant's own TTS embeds ≈0.01 (far
below any threshold), a **speaker-gated barge-in** (fire only on the *enrolled
user*) would NOT self-trigger on the assistant's own TTS — an acoustic barge-in
path **without AEC**, and the cheaper experiment to try first.

**Caveat before declaring it solved:** real-time barge embeds **short ~0.2s
windows** (worse than these ~8s probes), and the mic hears TTS *acoustically*
(through the speaker, room-coloured), not the clean TTS — so the
short-window, over-the-air separation still has to be tested.

---

## Machine audio state (this Linux box — `config.local.json` is gitignored)

`config.local.json` is **machine-local and never committed**, so a Windows
continuation starts from `config.json` and must re-establish its own device
setup. What this Linux box used, for reference:

- `sherpa.input_device = "AT2020USB-X"` (by NAME — robust to USB index shifts).
  `input_gain = 2.0` (the AT2020 is sensitive; the laptop mic needed ~8).
- `sherpa.capture_samplerate = 44100` (native; **required** — see fix #1).
- Output device 4 = `HDA Intel PCH: ALC285 Analog` = laptop speaker (the system
  default is HDMI → always pass `--output-device 4` on this box).
- `denoise_enabled = false` (GTCRN front-end exists but **over-suppresses** the
  self-audio over-the-air case; it's for genuine background noise — validate
  separately).
- `speaker_gate_input = false`, `barge_in_enabled = false` — both must stay OFF
  on open speakers without AEC.
- OS mixer (not persisted across reboot — `sudo alsactl store` to keep): laptop
  Speaker output had been **OFF** (a real blocker — re-enable it); analog Capture
  was **+23 dB (clipping)** → lowered to ~+12 dB; AT2020 Mic capture ~+17 dB.

> **AT2020 quirk:** it has a **touch mute button** that self-mutes after a USB
> reset (kernel logs `reset high-speed USB device`). The user manages it
> **manually** — do NOT try to toggle/reset it in software; open it ONLY at
> 44100. If capture reads rms ~0.000 across `arecord`/`pw-record`/PortAudio,
> it's the hardware mute, not the app. Confirm it's live with:
> ```bash
> python3 -c "import sounddevice as sd,numpy as np,time; b=[]; \
> s=sd.InputStream(device='AT2020USB-X',channels=1,samplerate=44100,callback=lambda d,f,t,st:b.append(d.copy())); \
> s.start(); time.sleep(1.5); s.stop(); a=np.concatenate(b); print('LIVE' if abs(a).max()>1e-3 else 'muted')"
> ```

## Committed artifacts (this commit)

- `logs/voice_id/s1/` — the real-voice enrollment + probe clips (`.npy`) and
  `report.json` from the speaker-ID test above. Committed at the user's request
  so the **speaker-gate experiment can resume on the next machine**. NOTE: these
  are NOT yet wired to the live `speaker_enroll_embedding` path; the gate is not
  enabled. (Privacy note: raw voice now lives in git history.)
- `logs/runs/run-20260530-*`, `run-20260531-*` — run bundles
  (`.txt` async DEBUG trace + `summary.json`; `run-20260530-232229.wav` is
  replayable via `python -m core --engine replay --replay-dir logs/runs`).
- `logs/real_usage/<ts>/report.{json,md}` — `tools/real_usage` grading reports.

## Next steps (pick up here)

1. **Speaker-gated barge-in without AEC** — the cheapest next experiment. Wire
   `logs/voice_id/s1/` enrollment into the live `speaker_enroll_embedding` path,
   enable the gate, and test whether short (~0.2s) over-the-air windows still
   separate the enrolled user from the assistant's room-coloured TTS. If they
   do, barge-in works on open speakers with no AEC.
2. **2nd speaker → fully acoustic barge-in test** — assistant on speaker A,
   interrupter on speaker B, both heard by the AT2020 (separate PortAudio
   streams are allowed; two streams on *one* device are not, on this box).
3. **AEC + AGC** — the general unlock for barge-in on open speakers and a robust
   over-the-air loop across volume levels. The bigger build if the speaker-gated
   path proves insufficient.
4. On Windows: re-establish `config.local.json` device setup (own mic/speaker
   indices, gain), and re-enroll the speaker-ID voice with the Windows mic — the
   committed Linux clips are reference data, not portable enrollment.
