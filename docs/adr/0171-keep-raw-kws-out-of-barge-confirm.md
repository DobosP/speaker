# ADR-0171: Keep raw KWS out of barge confirmation and close KWS terminals

Date: 2026-08-10
Status: accepted

## Decision

Do not add a raw-microphone keyword-spotter effect to the duck-confirm path.
The existing confirmation decoder already selects the application pre-AEC
microphone tap when the active canceller masks near-end speech, and every
primary streaming-ASR stream already receives the configured contextual
hotwords. Keep continuous KWS on its established processed/OS-echo-cancelled
input and verified-route
command path. When that independent KWS path terminates an acoustic turn while
a duck-confirm window is open, close the window in the terminal branch before
primary-ASR rotation and the next block so its gain, buffered PCM, and
playback-generation state cannot survive the terminal.

This changes no option, default, model, keyword threshold, command mapping, or
effect authority. A future raw-KWS experiment must start as a separate-stream,
aggregate-only shadow and requires owner bare-speaker A/B before it may affect a
cut.

## Context / why

ADR-0011 left “raw-mic word-confirm + KWS hotwords” as a revisit item. Later
work implemented both relevant non-KWS pieces: masking cancellers set
`_resid_blind`, which makes `_barge_confirm_step` decode the application pre-AEC
`mic_raw` tap, and the stream factory applies configured ASR hotwords to the
same primary stream used
inside the confirm window. Sherpa-ONNX 1.13.3 also supports multiple independent
KWS streams and per-stream extra keywords, but that API does not make an
echo-bearing raw stream safe control evidence.

The configured KWS phrases collapse several spoken forms to shared result
labels (`BE QUIET` to `stop`, `HOLD ON` to `wait`). A result-label comparison
therefore cannot reliably recognize the assistant's own raw-mic echo. Existing
live evidence also recorded phantom playback-time KWS hits at the current
threshold on a broken-quiet microphone. Feeding raw audio into the generic KWS
callback would be worse: that callback can map arbitrary labels to STOP,
confirmation, mode, or ordinary final-text behavior and is not the closed
STOP-only barge confirmation contract.

The audit found one independent cleanup. KWS is evaluated before the confirm
step and wins the acoustic terminal. Shipped stop labels synchronously re-enter
`stop_speaking()`, which happened to close the confirm window, but a custom,
non-stop, or unmapped KWS result need not stop playback. The capture loop now
closes the window explicitly before resetting the primary ASR turn.

## Consequences

- After a KWS callback returns consumed, capture closes any open confirm window
  before primary-ASR reset and before processing another block. A blocking or
  raising callback remains part of the pre-existing KWS risk surface.
- Headless tests bind raw-vs-processed confirmation selection and KWS-terminal
  precedence. They do not validate a microphone, KWS model, open-speaker echo,
  recognition quality, latency, or physical barge behavior.
- No raw-KWS stream, shadow metric, command, score, config field, or model path
  is added. Continuous KWS retains its pre-existing authority and risks.
- Before any effectful raw-KWS work, bind the full detected phrase rather than
  its collapsed label, make STOP-only authority explicit, atomically claim the
  captured playback/source generation before callback effects, and apply the
  ADR-0042 own-TTS ambiguity/speaker rule. Those are follow-up work, not claims
  of this cleanup.

## Verification

- Focused fake/headless barge, KWS, word-cut, runtime, and setup gate: `240`
  passed.
- Complete local Sherpa decode-owner gate: `327` passed.
- APM/DTD regression: `6` passed.
- No model, microphone, audio device, open-speaker, GPU, or live path ran.
