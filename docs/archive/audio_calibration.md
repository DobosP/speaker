> **SUPERSEDED (2026-07-02):** the level-margin gate calibrated here is
> fallback-only now — barge-in fires on the self-calibrating `AdaptiveDTD`
> (`docs/adr/0004`) with echo-probe-calibrated/auto AEC ref delay
> (`docs/adr/0005`) and WebRTC APM as the production open-speaker AEC
> (`docs/adr/0006`). `barge_in_output_margin_db=6` survives only as the
> committed no-AEC fallback default. Historical record.

# Audio self-interruption (TTS-echo) calibration

The always-on assistant keeps the mic **open while it speaks** (barge-in needs
that). Without acoustic echo cancellation, the assistant's own TTS leaks from the
speakers into the mic and can be mistaken for a user barge-in — the assistant
**cuts itself off** (`realtime-concurrency-5`). The unenrolled fallback gate
(`sherpa.barge_in_output_margin_db`) requires detected speech to stand a margin
**above the current playback level** before it counts as barge-in. We shipped the
margin at **0 (OFF / legacy fail-open)** pending on-device calibration, because
the gate compares mic RMS (post speaker-volume + room coupling) against the TTS
**buffer** RMS (pre-volume) — different scales, so the right margin is
volume-dependent.

## Probe

`python -m tools.echo_probe --margin-db <db> [--sentences N] [--gain G]` drives the
**real** sherpa engine (plays TTS out loud, mic open), records how many times the
assistant's own speech trips barge-in (`self_interruptions`), and the
mic-vs-playback ratio the gate evaluates. Reusable on any machine to find that
machine's margin. (Vary the OS master volume between runs — that is the axis that
moves the echo relative to the buffer reference.)

## Results — this laptop (Realtek mic-array + speakers, unenrolled)

Margin sweep at 70% volume, ~12 s of TTS:

| `margin_db` | self-interruptions | gate passed / flagged | max echo dB |
|---|---|---|---|
| **0 (shipped)** | **21** | 42 / 42 (fail-open) | +4.0 |
| **6** | **1** | 4 / 119 | +2.4 |
| 12 | 1 | 4 / 117 | +2.4 |

Volume × margin sweep:

| OS volume | `margin_db` | self-interruptions | max echo dB |
|---|---|---|---|
| 30% | 0 | 15 | −10.8 |
| 30% | 6 | **0** | −9.5 |
| 100% | 0 | 16 | +6.4 |
| 100% | 6 | **0** | +6.0 |

**Findings:**
- `margin_db=0` (the shipped default) self-interrupts **15–21×** at *every* volume.
- The echo scales with OS volume **~+17 dB from 30% → 100%** (max −10.8 → +6.4 dB
  relative to the buffer) — confirming the cross-scale dependency.
- **`margin_db=6` drove self-interruption to 0 across 30–100% volume.** At 100% the
  echo flirts with the margin (max +6.4 dB, 3 transient frames) so headroom is thin
  at max volume; 12 dB adds nothing over 6.

## Recommendations

1. **Enroll speaker-ID — the robust, hardware-independent fix.** Run
   `python -m core --enroll`. The gate then accepts only the enrolled user's voice;
   the assistant's own echo is rejected by *identity*, independent of volume, so no
   level margin tuning is needed. (The probe runs unenrolled, hence fail-open.)
2. **Unenrolled fallback:** `barge_in_output_margin_db = 6` is now the **shipped
   default** (`config.json` + `SherpaConfig`), based on this calibration (0
   self-interruptions, 30–100% volume). It is **device-specific** — re-run the
   probe on other hardware (set `0` to restore the legacy fail-open). **Caveat:** a genuine
   barge-in must still clear the margin; verify a real "stop" still interrupts at
   the chosen margin before relying on it (speak over the assistant during a
   `--engine sherpa` run). For very loud speakers, prefer enrollment or ~8 dB.
3. Recorded-fixture replays (`python -m core --engine replay`) can freeze a captured
   echo session into a regression once enrollment/AEC lands.
