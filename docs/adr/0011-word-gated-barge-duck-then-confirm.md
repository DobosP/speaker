# ADR-0011: word-gated barge-in (duck-then-confirm) over acoustic-only firing

Date: 2026-07-02
Status: accepted

## Decision

An acoustic barge trigger (BargeSustain trip) no longer hard-cuts playback.
With `barge_confirm_enabled`, the engine DUCKS playback (`barge_confirm_duck_gain`,
in-place in `_audio_cb` before the far-end/coherence tees), feeds the streaming
recognizer for `barge_confirm_window_sec`, and hard-fires only on
`barge_confirm_min_words`+ NEW transcribed words that don't read as the
assistant's own sentences (`_now_playing` + the `_recent_spoken` ring), or on a
stop command. An unconfirmed window restores volume, resets the stream, arms
`barge_confirm_retry_suppress_sec`, and TEACHES the DTD echo charts its
(verified-echo) per-block levels. Shipped default OFF (legacy hard-fire is
byte-identical); enabled per-machine in `config.local.json`.

## Context / why

Live Windows session 2026-07-02 (runs 212109/220207/223217, open nonlinear
laptop speaker, APM backend): the acoustic detectors are inherently ambiguous
there. With `dtd_coherence_echo_veto=true` (default since ~2026-06-27) the
coherence detector read even genuine talk-over as echo-only and vetoed 468/481
correct DTD fires → barge-in dead. With the veto off, the DTD fired on the
assistant's own residual echo (z_resid 100-855 over K=5) → self-interrupt, plus
a trigger flood (14/45s → audible volume pumping) because the DTD charts starve
on this box (TTS echo always reads as VAD-speech, so the VAD-quiet
`observe_echo` tap never feeds them). "Why not X": tuning the veto/margins is a
knife-edge per machine+volume (both prior sessions landed on opposite failure
modes); switching AEC backends alone (DTLN→APM) reduced but did not eliminate
echo-triggered fires. The word gate (LiveKit `min_interruption_words` /
Pipecat MinWords pattern) removes the requirement that any single acoustic
detector be perfect: a false trigger costs a ~1.5 s volume dip that self-heals
and self-teaches; a real talk-over cuts with the user's words already captured
(free pre-roll for the next final).

## Consequences

- Barge-in works on the open speaker without headphones (D-A, ADR-0008 upheld)
  and without per-room acoustic tuning; the confirm layer is the guard, so
  `dtd_coherence_echo_veto=false` is safe where the gate is on.
- Interrupt latency gains the ASR confirm time (~0.3-0.9 s after the duck) —
  the duck itself is immediate, so the assistant audibly "yields" fast.
- Echo that garbles into ASR words is filtered by token-overlap vs
  `_now_playing`/recent sentences; a user whose talk-over consists ONLY of
  words the assistant just spoke may need a reworded interjection ("stop"
  always works).
- Unconfirmed-window chart teaching means the trigger flood decays within a
  reply or two; if APM/AEC improves further, the gate simply idles.
- Revisit: word-confirm on the *raw-mic* stream + KWS hotwords during the
  window; surfacing `barge_in_duck/confirmed/unconfirmed` metrics in
  `tools/diagnose_run.py`.
