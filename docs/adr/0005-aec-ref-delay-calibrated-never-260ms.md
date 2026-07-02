# ADR-0005: `aec_ref_delay_ms` is echo-probe-calibrated or auto — NEVER hardcode 260 ms

Date: 2026-06-10
Status: accepted

## Decision
Set the AEC far-reference delay per machine, never globally: calibrate
`aec_ref_delay_ms` with `tools/echo_probe.py` (pick the ERLE-maximizing value)
or let the runtime track it with `aec_auto_delay` (cross-correlation feedback
in the capture loop). Do not hard-set 260 ms — or any other borrowed number —
on a machine it was not measured on.

## Context / why
The "real ~260 ms acoustic delay" figure from the 2026-06-05 live analysis was
a single-machine, pre-FIFO observation that leaked into CLAUDE.md as if it
were a constant. On this box the calibrated value moved 19→105 ms
(2026-06-10 live session, 30.3 dB ERLE); on the 2026-06-08 rig the correct
setting was 0 (the playback FIFO already aligned the far reference). A wrong
delay silently collapses ERLE, which re-opens self-interruption — the exact
failure the calibration exists to prevent. Why not one shipped default: the
speaker→mic path depends on device buffers, OS mixer, and topology (e.g. the
OTA/JBL rig), so any constant is wrong somewhere.

## Consequences
- Per-machine values live in `config.local.json`, never in the committed base
  config.
- New machine or changed audio topology ⇒ re-run `tools/echo_probe.py` (or
  rely on `aec_auto_delay`), then verify self_interruptions=0 in the run
  bundle.
- The `aec_auto_delay` capture-loop feedback still lacks an engine-level test
  (backlog P2/P3) — add one before trusting it blind on new hardware.
