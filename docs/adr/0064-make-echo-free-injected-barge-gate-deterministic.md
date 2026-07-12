# ADR-0064: Make the echo-free injected barge gate deterministic

Date: 2026-07-12
Status: accepted

## Decision

Treat `tools.live_session --inject` as an explicitly clean, echo-impossible
capture topology. Its shared profile disables physical AEC/voice-route and
word-cut authority plus playback-reference coherence, DTD, output-level,
word-confirmation, and denoising gates. The real Silero VAD, bounded sustain,
capture callback, task cancellation, and playback FIFO remain active; one
eligible 100 ms block is required. `--denoise` may re-enable GTCRN only as a
separately labelled stress diagnostic. Do not change any ordinary acoustic or
production setting.

Make every injected run return nonzero when its machine-owned full-duplex,
barge, or response grade fails, or when the emitted barge/response row counts do
not exactly cover the selected scenario. Continue writing all per-run and suite
artifacts before returning so a red run remains diagnosable.

## Context / why

The exact three-repeat gate at `203334` cut 1/2, 2/2, and 1/2 intended barges,
yet the CLI returned zero. The missed input was the identical 314 ms synthetic
`Stop`; the longer redirect always cut. The injected mic contains no assistant
audio, but the inherited profile still compared the short word with a varying
TTS reference and retained a 6 dB playback-level fallback. Removing those
inapplicable echo discriminators made the CLI honestly return 1 at `204845`, but
GTCRN still left the short word below Silero's eligible window in all repeats.

Clean injected audio has no noise for GTCRN to remove. A controlled no-denoise
run at `205151` cut 2/2, and the final exact three-repeat run at `205351` cut 2/2
in every repetition with zero self-interrupts. Stretching the word, lowering
production thresholds, or claiming this as acoustic evidence were rejected:
they would either weaken the requirement or confuse a control-flow harness with
the mandatory bare-speaker gate.

## Consequences

- The documented injected command can no longer false-green through exit status
  or through an empty/partial grade that happens to say `ok`.
- The gate deterministically covers clean capture continuity, VAD/sustain,
  cancellation, and FIFO interruption; GTCRN can still be tested explicitly.
- Production AEC/coherence/DTD, two-block sustain, speaker authority, denoising,
  and word-cut behavior are unchanged.
- Physical echo rejection, current-room short Stop, v5 owner enrollment, and
  audible stop quality still require the bare-laptop live acceptance gate.
