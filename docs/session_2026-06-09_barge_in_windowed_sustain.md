# Session 2026-06-09 — Barge-in windowed-sustain fix + deterministic test suite

**Headline:** A live debug session on the Linux box surfaced that open-speaker
barge-in had regressed to "needs a shout." Root-caused it to a starved mic **plus**
a flicker-starved capture-loop integrator, built a deterministic audio/model-free
test suite from the recorded failure, then fixed the integrator with a bounded
windowed sustain (`BargeSustain`). Merged to `main` `@0a1cdf5`; **1445 passed, 14
skipped, 0 failed, 0 xfailed.**

**Branch → commit map:** `test/barge-in-capability` (deleted, merged)
- `bedc98b` test(barge): deterministic open-speaker barge-in suite from the live failure
- `7966d70` fix(barge): windowed sustain so a normal-volume talk-over cuts without a shout
- `0a1cdf5` Merge test/barge-in-capability → main

## What happened
Started a live `--engine sherpa` session (laptop mic + laptop speaker = the
open-speaker hard case; the AT2020 USB mic was **unplugged**). The owner reported
**"i need to scream loud to stop"** — a normal-volume talk-over did not cut the
assistant's TTS. Captured as `run-20260609-203236` (`.txt`/`.summary.json`/`.wav`).

## Root cause (two compounding layers)
1. **Mic starvation (dominant for STT, not the barge blocker).** The far-field
   ALC285 at `input_gain=2.0` captured the owner at `raw RMS ~0.002–0.01` (max over
   the whole run `0.0704`, a full scream) vs a historical real barge `~0.4`;
   repeated `input is ~silent (avg_rms=0.00008)` warnings. The 2026-06-08 "feels
   good, no shout" result was on the **close AT2020 mic**, which was absent.
2. **Decision-layer (the actual barge blocker, device-independent).** The
   `AdaptiveDTD` *fired* on the normal talk-over (turn-2: 3 of 5 blocks), but the
   capture-loop integrator — `voiced_run += 0.1` on a fire, `*= 0.5` on a miss, fire
   at `barge_in_min_speech_sec` — could not accumulate the **flicker** (breath/pauses
   + DTLN suppressing the user mid-double-talk). It peaked at `~0.15s`, below the
   `0.3s` bar; only the 10× shout (`raw 0.0704`) sustained a cut. The turn-3
   *pre-shout* portion had only **2** DTD fires, so no decay value reaching a 3-block
   (0.3s) threshold could ever fire it — the threshold had to drop *and* the
   mechanism had to tolerate flicker.

## The fix — `core/engines/_dtd.BargeSustain`
A bounded, windowed temporal-confirmation integrator that replaces the leaky
accumulator: it cuts when at least `barge_in_min_speech_sec` of eligibility lands
within the trailing `barge_in_sustain_window_sec` (new `SherpaConfig` knob, default
`0.5`; `min_speech` default stays `0.2` → **2 of the last 5 blocks**).
- **Responsive to flicker:** the recorded turn-2 normal talk-over now cuts at idx 11
  (no shout); the turn-3 pre-shout cuts at idx 189 (not the idx-196 peak).
- **Echo-safety is STRUCTURAL:** the bounded window means a sporadic single-frame
  echo leak over a long reply can never accumulate to a self-interrupt (only a
  sustained talk-over packs enough eligible blocks into one window) — the owner's
  other HARD requirement.
- Extracted as an **importable class**, so the test driver
  (`tests/barge_fixtures.run_frames`) drives the **real** `BargeSustain` rather than
  a mirror — closing the one "mirrored, not imported" caveat from the test review.

`barge_in_min_speech_sec` default is already `0.2` in `config.json` (the shipped
fix). The machine-local `config.local.json` override was lowered `0.3 → 0.2`
(gitignored, so not committed).

## The test capability (built first, via a 5-phase ultracode workflow)
Deterministic, **no audio / no models**: the recorded DEBUG trace already carries
the exact per-frame `(raw, resid, incoherence)` the live detector saw, so replaying
them through the **real** `AdaptiveDTD` + **real** `BargeSustain` reproduces the
live behavior exactly.
- `tests/barge_fixtures.py` — loader/driver (reads the committed
  `tests/fixtures/barge_in/`, not the prunable `logs/runs/`).
- `tests/test_barge_trace_replay.py` — reproduction/baseline (now: cuts on turn-2).
- `tests/test_barge_requirement.py` — owner-requirement **guards** (were strict-xfails
  pinning the bug; now passing — they go red if barge-in regresses).
- `tests/test_barge_echo_must_not_fire.py` — echo self-interrupt guardrails.
- `tests/test_barge_gate_conversion.py` — real `SherpaOnnxEngine` gate-seam wiring.
- `tests/test_dtd.py` — +8 `BargeSustain` unit tests (flicker tolerance, bounded
  echo-safety, reset, diagnostics).
- `tests/fixtures/barge_in/` — evidence: `…frames.json` (204 parsed frames),
  `…trace.txt`, `…summary.json` (the trimmed `…talkover.wav` is gitignored —
  `*.wav` — and not depended on by any test).

## Environment on the Linux box (i9-13980HX / RTX 4090 / ALC285)
- Live config (`config.local.json`, gitignored): `gemma4:12b` main / `gemma3:4b`
  fast on Ollama; `think=false`; `tts.streaming=true`; AEC `dtln` @ `ref_delay=0`;
  `coherence_barge_in_enabled=true`; `input_gain=2.0`; `barge_in_min_speech_sec` now
  `0.2`; `input_device=AT2020USB-X` (absent — overridden to ALC285 via CLI this
  session); `output_device=ALC285 Analog`.
- **PortAudio enumeration race:** the core process occasionally starts with **no
  input devices** (`sd.default.device == [-1, 0]`, empty input list) → `No input
  device matching '…'` + `Error querying device -1`. A plain relaunch fixed it (a
  `sd._terminate(); sd._initialize()` cycle also clears it). Worth a startup
  retry/re-init in the engine if it recurs.

## Next steps (pick up here)
1. **LIVE re-validate the barge fix (NEEDS THE MIC).** Owner chose "merge now,
   validate later." Run
   `python -m core --engine sherpa --input-device 'ALC285 Analog' --output-device 'ALC285 Analog'`,
   talk over a reply at **normal volume**, and confirm it **cuts without a shout**
   and does **not** self-interrupt. The deterministic proof is strong (the normal
   talk-over already fired the DTD), but the owner requirement is a live behavior.
   If it cuts too eagerly on echo, raise `barge_in_min_speech_sec` (→0.3) or shrink
   `barge_in_sustain_window_sec`; if it still needs pushing, the levers are the same
   two knobs.
2. **Mic gain (problem #1, for STT robustness — separate from barge).** Raise the
   ALC285 capture level / `input_gain` so a normal voice reaches `raw ~0.1+`, or plug
   in the AT2020 USB mic (watch the touch-mute gotcha).
3. Carryover from prior sessions still open: Windows AEC echo_probe recalibration
   (non-catastrophic now); prosody-endpointing live tuning; `input_gate` re-enable
   for always-on; `logs/runs` committed-bundle churn.
