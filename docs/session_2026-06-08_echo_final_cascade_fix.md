# Session 2026-06-08f — Break the open-speaker self-interrupt echo-final cascade (device-adaptive)

**Branch → main:** `fix/echo-final-cascade` → `main`. Full Tier-0 suite green
(**1416 passed**, 14 skipped, 0 failed). Built via a Map→Design→Build→Verify
fan-out workflow; the design/asr_text layer came from the workflow, the engine
layer + the corrected gate semantics were implemented + verified in-loop.

## What the owner reported
"It was working in an acceptable way; after a rework it's **shutting it down
again**." = the **self-interrupt + "two outputs one after another" cascade** that
appeared on the first live `--engine sherpa` run on the **Windows** side
(`logs/runs/run-20260608-181250`). Barge-in had been validated only on the Linux
side; this was the first Windows live run.

## Root cause (conclusive from the trace)
The **final-dispatch seam had no energy/quality floor**. The streaming recognizer
turns the assistant's own open-speaker **residual echo / ambient noise** into
"finals" (`asr final: 'I.' (raw 'BEING')` at `avg_rms=0.0066`, `speaking=False`),
and `runtime._on_final` dispatched them to the brain exactly like real speech
(real speech is `~0.2-0.4`, echo `~0.005-0.018`). Each spurious response's TTS
echo then self-interrupted, and the cancelled tail's echo became the next garbage
final → runaway. On the Windows box `asr_final_backend=sense_voice` amplified it
by **hallucinating** plausible sentences from echo (`'BEING'→'I.'`, `'THE LOW IS
THIS CORDOOR KING'→'Hello, is this code working?'`).

This is **not** a regression from the recent think=false / smart-routing / P2
work (those never touch AEC/barge/ASR). The Windows stale `aec_ref_delay_ms`
*leaks* more echo, but the durable fix is to make the **code** robust so a
mis-calibrated machine can never cascade — on **any** device.

## The fix — 3 device-adaptive, additive, off-switchable layers
Owner hard directive honored: **no fixed magic-number thresholds in the energy
trigger path** — every bar is RELATIVE to a LEARNED per-device floor (same
principle as the AdaptiveDTD / `loudness_admits`). The working barge-in is
untouched.

| | what | where | device-adaptive basis | default |
|--|--|--|--|--|
| **L1** | drop a final at/near the learned echo/quiet floor | `_final_above_floor` wired at the capture-loop final seam (before `_should_act_on_final`, so it also guards the identity-free barge config) | `rms(seg) ≥ max(_ambient_rms, _playback_floor_rms) · margin_dB` via `loudness_admits` | `final_floor_margin_db` = **6.0** (config.json); dataclass **0.0** = off |
| **L2** | suppress a RE-fired barge on the just-cancelled echo tail | `_in_post_speaking_refractory` + `_last_speaking_end` stamped at both `_speaking→clear` sites; ORed into the **barge** debounce only | a fixed clearance *time* window (allowed; not an energy threshold) | `barge_in_refractory_sec` = **0.5** |
| **L3** | demote a short-clip SenseVoice 2nd-pass hallucination | `agreement_guard` (core/asr_text.py) wired into `_final_transcribe` | clip **length** (text/duration), never energy | on with the 2nd pass |

### Why it does not regress the working barge-in (the things that matter)
- **L1 only "bites" when a floor is learned.** `_ambient_rms` is tracked only when
  `input_loudness_margin_db>0`; `_playback_floor_rms` only when AEC is on. With
  neither, `max(...)=0` → L1 **fails open**. So L1 activates exactly on the
  AEC/open-speaker configs that have the echo problem, and never silently drops
  finals on a minimal config. A real talk-over (`~0.2-0.4`) is 12-80× the echo
  floor — far past the 6 dB bar — so it still barges AND still dispatches.
- **L2 is on the barge path only, never the final seam.** A real barged-in user's
  request final is gated solely by L1 (it's loud → admitted). `_speaking.clear()`
  fires **per-reply, not per-sentence** (guarded by `if self._play_q.empty():` in
  `_playback_loop`), so the refractory only arms at reply boundaries — it does not
  touch mid-reply barge-in.
- **Off-switch parity.** `final_floor_margin_db=0` and `barge_in_refractory_sec=0`
  are byte-identical to the prior behavior; `_should_act_on_final` is **unchanged**
  (all 9 existing input-gate tests untouched).

## Tests (+30; suite 1416 green)
- `tests/test_asr_text_agreement.py` — the hallucination pairs + the legit
  long-correction case, straight from the live run.
- `tests/test_speaker_input_gate.py` — L1 echo-vs-speech against the **real** RMS
  numbers (echo 0.008/0.018 dropped, speech 0.3/0.5 passed, `max(quiet,playback)`,
  cold-start fail-open, off-by-default); L2 refractory active/expired/off/inert.
- `tests/test_asr_final.py` — L3 `_final_transcribe` wiring (short hallucination
  rejected, long correction kept).

## Validation notes
- **Headless replay** of the failing WAV (`python -m core --engine replay
  --replay-dir <tmp> --llm echo`) runs clean (no crash) and shows the raw
  recognizer dispatching ~31 garbage finals — the pre-gate baseline the layers
  target. **Caveat:** `FileReplayEngine.replay_samples` calls `on_final()`
  directly and **bypasses** `_final_transcribe`/`_final_above_floor`/the seam, so
  the replay does **not** exercise L1/L2/L3. The production decision logic is
  validated by the deterministic unit tests above (they encode the real failing
  RMS numbers). The live capture-loop seam cannot be driven headlessly (no mic).

## Next steps (pick up here)
1. **P0 remainder — needs the mic.** The FIRST self-interrupt (the Windows DTD
   firing on echo) is the stale-AEC calibration: run `python -m tools.echo_probe`
   (echo-only) on the Windows box, pick the ERLE-max `aec_ref_delay_ms` (don't
   assume 0), target `self_interruptions=0`; replay `run-20260608-181250.wav`
   through it to iterate without re-talking. This commit makes that miscalibration
   **non-catastrophic** (no runaway) until it's done.
2. **Live-tune the new knobs** on the open speaker: `final_floor_margin_db` (raise
   if echo finals still slip; lower if quiet real speech is dropped — watch the
   `echo_floor_rejected_final` metric in run bundles) and `barge_in_refractory_sec`.
3. Re-assess prosody turn-taking once the self-interrupt is gone (it was
   unreadable under the cascade chaos).
