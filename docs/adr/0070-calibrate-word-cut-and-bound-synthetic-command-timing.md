# ADR-0070: Calibrate word-cut and bound synthetic command timing

Date: 2026-07-14
Status: accepted

## Decision

Keep calibrated energy admission for word-cut default-off. It may substitute for
a false Silero decision only when InputAGC exists, a complete current pre-gain
calibration is valid and unclipped, the word-cut route is verified, and the
post-OS-EC/pre-application-gain block stays above the calibrated floor by the
configured margin for the configured consecutive-block count. A VAD-positive
block resets rather than warms this fallback. The private synthetic delay profile
alone initially enables it at +6 dB for three blocks; production keeps the
library default of disabled pending physical acceptance. The fallback admits the
normal processed ASR block and does not weaken lexical, own-speech, route,
speaker-authority, or one-cut gates. Reset its debounce and private onset edge at
every reply, recovery, route loss, revocation, and regrant boundary.

Permit one bounded detector-only clock advance after a nonempty voiced word-cut
candidate. On the first genuinely quiet capture block, feed the throwaway
detector the real block followed by zeros up to 0.4 seconds, with an absolute
configured target cap of 0.5 seconds; feed the second quiet block literally. At
the default three-block debounce, reset before feeding the third. With a longer
debounce, blocks three through N-1 remain inert and block N resets. With debounce
one, reset immediately. Padding and quiet PCM never enter VAD, energy/calibration,
candidate or observed-sample clocks, speaker evidence, pending PCM, or normal
ASR. Empty, own-folded, speaker-rejected/ambiguous, route-revoked, guarded, or
energy-debounce-pending state cannot regain authority through this flush. Report
total flush and synthetic-padding duration separately.

For the private synthesized delay scenario, inject the exact canonical command
`quiet` at speed 0.8 and keep its single playback stream alive with 600 ms of
trailing silence. Build only autotest VITS inputs with acoustic and duration
noise scales set to zero; runtime TTS retains native stochastic defaults. Grade
from the engine's first calibrated capture-onset marker to its barge marker, keep
source-onset latency as a diagnostic, and allow at most 1.4 seconds only for the
exact `delay + synth + command` tuple. Recorded-owner, physical speaker, generic
barge, and stress paths retain the 1.0-second limit. Missing, nonfinite, reversed,
or misordered timing evidence fails closed.

## Context / why

The native virtual EC delivered command energy that repeatedly exceeded the
calibrated pre-gain floor while Silero stayed false, starving the isolated
recognizer. A fixed absolute RMS threshold would recreate device-specific mic
tuning, while enabling a relative fallback everywhere before physical testing
would broaden production authority. A current calibration plus sustained margin
is the device-adaptive evidence already available at this boundary.

Even after voiced admission, the streaming transducer sometimes published a
two-word command only after more acoustic clock arrived. Adding quiet to the
candidate or speaker window would manufacture identity duration; widening the
capture debounce would let separate bursts accumulate. A detector-only bounded
zero suffix advances the disposable decoder without changing capture time or
authority evidence.

The first apparently green command run was not repeatable because VITS generated
a different waveform and duration for every harness invocation: the next clip
decoded nothing, and a later one decoded only `YET`. After deterministic render,
the two-word phrase still sometimes arrived as non-command `VERY QUIET`; the exact
one-word `quiet` removes that unstable first-token decision while retaining the
production command grammar. Reusing post-lexical latency would have hidden ASR/
admission regressions, and widening every barge SLO would have weakened recorded
and physical acceptance. Deterministic synthesis plus the exact private clock/SLO
isolates that validation artifact.

## Consequences

- Fresh physical-device-free runs `20260714-041032` and `20260714-041156` both
  passed: zero self-cuts, one causal cut, capture-to-cut 0.509/0.818 seconds,
  source-to-cut 0.758/1.005 seconds, identical command SHA-256
  `dcdca42671e60d2a13c694afc03beec3461247590cf76b92bd6c76b34271ee89`, and all
  topology/capture/duplex/correlation/child-exit/cleanup proofs green.
- Run `041032` published exact `QUIET` only after both bounded quiet detector
  feeds (`496` ms total, `304` ms synthetic padding); `041156` cut earlier on
  voiced input. The repeated gate therefore covers both timing outcomes.
- The deterministic implementation is covered by 146 word-cut tests, focused
  TTS construction tests, and a clean-checkout full result of 4049 passed/31
  skipped/9 warnings.
- The private delay gate now diagnoses route, calibrated admission, decoder
  timing, and control-plane interruption reproducibly. It still cannot prove
  owner identity, physical echo cancellation, room acoustics, audible voice
  consistency, or bare-speaker latency.
- Production fallback remains disabled and physical v5 acceptance remains red;
  enabling it outside the private profile requires a new decision backed by the
  recorded-owner and manual bare-speaker gates.
