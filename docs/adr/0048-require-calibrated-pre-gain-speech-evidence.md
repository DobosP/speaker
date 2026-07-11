# ADR-0048: Require calibrated pre-gain speech evidence for ordinary finals

Date: 2026-07-11
Status: accepted

Supersedes ADR-0040 while retaining its transient thresholds, single-retry
bound, four-window read cap, and configured-floor fallback.

## Decision

Enable complete startup ambient calibration by default for normal runtime and
production enrollment. Build an immutable ordinary-final profile only from a
complete, stable, finite, positive calibration in the current pre-gain capture
domain. Calibration needs at least 200 ms of nondegenerate PCM. Its normalized
20-band ambient spectrum must be stable; the novelty threshold is
`max(0.12, ambient_q99_distance + 0.05)`. Reject clipped or VAD-detected
speech/TV-contaminated startup windows, retry one complete window, then abstain
if the replacement is also unsuitable. Preserve ADR-0040's one-retry/read-cap
contract and emit the clipping diagnostic even when the hot window is rejected.
Measure a parallel model-rate PCM tap before InputAGC/static gain with independent
resampler state.

A normal VAD onset opens one evidence epoch through the ASR endpoint even if VAD
flickers afterward. Split its pre-gain PCM into 20 ms frames. Gate energy with the
trailing 20→40→60 ms raw RMS so low-fundamental phase cannot alternate around the
threshold. A frame qualifies only when that RMS clears calibrated `ambient_rms`
by more than 6 dB, its mean-removed spectrum is finite/nondegenerate, its
zero-crossing rate does not exceed 0.45, and it is novel against calibrated
ambient or the preceding qualified frame in the uninterrupted run.

A dynamic short pattern at shipped minima needs four qualifying frames total and
one joint run of four (80/80 ms) containing a spectral change above 0.05 whose
zero-crossing rate is at least 0.015. Raising minimum-qualified raises the total;
raising minimum-contiguous raises the joint run. A steady pattern needs one run
of `max(those configured frame counts, 6)` (at least 120 ms). Either run must
contain at least two qualified trailing-
60-ms windows whose YIN-style periodicity score is above 0.65 over 50–400 Hz
pitch lags. Dynamics and periodicity must belong to the same run, so unrelated
artifacts cannot compose. The 0.015 guard prevents 50/60/90 Hz phase leakage from
unlocking the short path.
The 6 dB/80 ms/80 ms/120 ms floors are hard minima configuration cannot lower.
Use no absolute RMS floor and never use InputAGC's headroom-expanded/clamped
floor.

When an ordinary profile is armed, withhold normal partial callbacks while its
disposition is `INSUFFICIENT`, because a partial can cancel valid runtime work
before its eventual final. Snapshot the typed
`SATISFIED`, `INSUFFICIENT`, `UNAVAILABLE`, or `BYPASSED` disposition at the
capture endpoint. Reject only `INSUFFICIENT`, before async queueing, second-pass
ASR, speaker embedding, or runtime dispatch. Missing, partial, suspicious,
clipped, speech-contaminated, too-short, spectrally unstable/degenerate,
zero/non-finite, disabled, no-VAD, or pending changed-domain calibration
abstains/fails open. Word-cut handoffs use `BYPASSED` only with their explicit
bounded PCM prefix. Duck/confirm handoffs now publish bounded, capture-generation-
bound PCM with a short deadline; the next normal block adopts that PCM regardless
of instantaneous VAD. A stale latch cannot bypass a later unrelated epoch. These
handoffs retain their separate lexical/speaker contracts. Evidence establishes
acoustic existence only; it cannot mint owner verification, barge authority, or
action/tool trust.

Preserve a stable profile across a same-domain capture reopen while resetting
the per-VAD-epoch accumulator and both resamplers. Invalidate it immediately on
a changed domain; ordinary evidence fails open while recalibration is pending,
but speaker/action/word-cut authority remains cleared. Recovery accepts only a
contiguous idle window outside any open speech
epoch, replays the aggregate through VAD, retries one contaminated/unstable
window, and keeps authority cleared after a failed replacement. Re-arm only from
a complete stable replacement. Calibrate after recovery even when InputAGC is
disabled.

## Context / why

Run `20260711-200747` had 115 turns and 99 LLM requests with no process error,
but ordinary ASR admitted stale and fresh garble while `speaker_gate_input` was
locally disabled and the configured final-floor gate had no learned source. It
also produced zero active word cuts. ADR-0046 prevents decoder text crossing a
VAD epoch and ADR-0047 prevents below-floor PCM from inheriting stale AGC gain,
but neither proves that fresh post-front-end VAD/ASR text had a corresponding
current pre-gain signal. Just-above-floor noise may still be boosted, and a
denoiser/recognizer may turn one energetic transient into tokens.

The original six-consecutive-frame proposal imposed 120 ms on every control;
the first four/three-frame proposal then admitted energetic DC and a stationary
100 ms tone. The adopted two-speed rule rejects those cases, a real one-frame
impulse, broadband noise-like frames, and a gain-scaled copy of calibrated
ambient. Periodicity/run binding rejects a 156-case AR(1) sweep (100/120/200 ms,
four coefficients, 13 seeds) and a separated periodic-plus-dynamic artifact. A
50–400 Hz steady harmonic-voice grid, a dynamic 100 ms synthetic `YES`, an
armed-profile `STOP` transcript, and moving voice over overlapping fan harmonics
pass. A committed failure-discovery replay admits short `YES`/`NO`/`STOP` and
laptop-fan speech while rejecting the tested bump, scrape, door tail, bass,
alarm, and unchanged-noise fixtures. No isolated owner `YES` recording exists,
so quiet owner `YES` and real `STOP` remain live checks.

The existing final-floor gate is post-front-end, inactive on the current OS-EC
configuration, and can be raised by the same processing under test. Speaker-ID
answers *who*, not whether current speech existed, and the local input gate is
disabled after short/quiet owner scores proved unreliable. A lexical minimum
would reject valid `yes`, `no`, names, and commands. Energy plus amplitude-
invariant ambient/temporal spectral novelty, a broad noise ceiling, and bounded
YIN-style periodicity are deterministic and cheap. They separate application
gain, degenerate offsets, stationary one-read tones, tested AR-noise motion, and
louder stationary ambient from more plausible current speech, without claiming
to classify identity or semantics.

## Consequences

- With an armed ordinary profile, normal garbage without periodic evidence plus either the dynamic 80 ms
  total/80 ms contiguous pattern or the steady ≥120 ms pattern in pre-gain
  model-band PCM cannot publish a partial, enter SenseVoice, or create an LLM turn.
- A clean install now spends the configured 1.5 s measuring its device/room at
  startup. Production enrollment also records a matched ambient operating point
  when InputAGC is off; runtime startup independently builds the evidence profile.
- With an armed ordinary profile, first partial publication cannot occur before
  80 ms of qualifying evidence
  and may wait longer for dynamics, a steady fallback, or decoder output.
  Real speech below +6 dB, spectrally too similar to calibration, or too short can
  be rejected; quiet `YES`, low-sensitivity speech, and STOP therefore remain
  live gates before landing.
- The parallel resampler adds one lightweight capture-rate-to-model-rate pass.
  Its immutable domain and calibration generations prevent cross-device reuse.
- A periodic spectrally novel fan/music tone, TV/other voice, or post-playback
  echo tail may still satisfy this existence check. Speaker identity and tail
  provenance remain separate layers; this decision does not make live barge green.
- Same-route reopen preserves calibration. An OS sensitivity change that does not
  change the capture-domain fingerprint is corrected at the next process startup,
  not inferred during that reopen; low-sensitivity live A/B remains required.
