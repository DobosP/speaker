# ADR-0181: Bind echo probe to terminal playback receipts

Date: 2026-08-10
Status: accepted

## Decision

Require `tools.echo_probe` to pace every stimulus from a typed, matching
sink-terminal playback receipt. Immediately after engine construction and before
diagnostic instrumentation or `engine.start()` can open the microphone, require
an exact true `playback_capabilities.tracked_terminal` capability. Missing,
malformed, or raising capability access returns the identity-free
`probe-preflight-failed` error with exit `1` and invokes neither start nor stop;
preserve `KeyboardInterrupt`. The tracked helper also checks the capability
defensively after start. Submit each sentence as a `TrackedSpeech` with a unique
ordinal-only, privacy-safe fragment identifier and an exact `on_terminal`
callback. Do not pin the identifier's spelling. Submit the next sentence only
after the current sentence has produced exactly one valid terminal receipt; on
the normal path, attempt `stop()` only after the final sentence's terminal
receipt. The existing lifecycle owner still attempts exactly one stop after any
invoked start when the probe fails.

A typed receipt with the expected fragment identifier and
`PlaybackOutcome.COMPLETED` is a completed terminal observation. A typed,
matching `PlaybackOutcome.INTERRUPTED` receipt is a receipt-attested
`INTERRUPTED` terminal and remains reportable because self-interruption is what
this diagnostic observes.
Fail closed on `DROPPED`, `FAILED`, an unrecognized outcome, a wrong receipt
type, fragment mismatch, duplicate terminal, bounded-wait timeout, or submission
failure. These post-start failures return `1` through the existing identity-free
`probe-lifecycle-failed` result and never publish a normal diagnostic, stimulus
identity, or playback counters. The preflight capability failure has the same
identity-free exit meaning and the exact `probe-preflight-failed` error without
claiming lifecycle ownership.

Every full successful report adds the exact scalar
`playback_terminal_protocol = "tracked-sink-terminal-v1"` and exact integer
counters `sentences_submitted`, `sentences_completed`, and
`sentences_interrupted`. On success, submitted equals the normalized stimulus
count and completed plus interrupted equals submitted. Preserve the established
success-only `sentences_spoken` key as a compatibility alias equal to
`sentences_submitted`; despite its name, it records submissions and is not
evidence of audible or completed playback. The counters record typed receipt
outcomes, not sample measurements or human-audible playback. Bind each exact text into the
stimulus-identity builder immediately after `speak_tracked` returns normally.
Only a full normal report exposes the resulting identity or counters.

Keep the one-second capture warm-up, but remove the legacy 0.4-second
per-sentence and 0.5-second final timing heuristics. The sink-terminal wait now
owns sequencing. Reports produced by the old enqueue/synthesis-callback pacing
and reports carrying `tracked-sink-terminal-v1` must not be pooled, even when
their `stimulus_id` values are equal. Equality of both protocol and stimulus
identity remains necessary rather than sufficient because device, gain,
strategy, TTS, configuration, route, room, and the other evidence dimensions
still apply.

Carry forward ADR-0180's exact normalized cyclic stimulus plan. For
`max(1, args.sentences)`, cycle these exact UTF-8 sentence bytes in order:

1. `This is a live audio calibration test of the speaker assistant.`
2. `I am checking whether my own voice is captured back by the microphone.`
3. `The barge in gate should not interrupt me while I am still speaking.`
4. `This final sentence completes the quiet playback portion of the diagnostic.`

Build `stimulus_id` with the prefix
`echo-probe-spoken-text-v1:sha256:`. Initialize SHA-256 with the exact domain
`b"speaker.echo-probe.spoken-text.v1\0"`, then append each submitted text's
exact UTF-8 byte length as an unsigned eight-byte big-endian integer followed by
those bytes. Do not normalize text or append a newline. The default
three-sentence interrupt-suite identity remains
`echo-probe-spoken-text-v1:sha256:27c744fbb14bbcb74f9256a2a4db2af25c6e7ad6c70e5ea224c3c8708a06a3c6`.
The default four-sentence identity remains
`echo-probe-spoken-text-v1:sha256:4366d6a869714ed767964bda1dc682b5264c109d5d3f12260f6e07276c502704`.
ADR-0179's old-fourth-sentence identity
`echo-probe-spoken-text-v1:sha256:be277009edac56b0473730c1e5326a1b765976386aca9f26c500eab9f00822eb`
remains a separate, non-poolable population. The identifier is self-attested
submitted-text provenance; it does not authenticate a hostile child or by
itself prove synthesis, audible playback, or physical acceptance.

Continue to preserve an additive raw child report in `tools.interrupt_suite`
without validating or using the terminal protocol, counters, or stimulus
identity in summary, coupling, classification, or exit logic. Old and new raw
rows remain non-poolable across their protocol boundary. Carry forward the
descriptive coherence, adaptive-DTD, uncoupled, and top-level guidance: do not
recommend tuning from one run, headphones, or a low-level core session.
Physical acceptance still requires a real assistant session started through
`./live.sh`, a human talk-over on the bare laptop speaker, and
`python -m tools.live_audio_ab logs/runs/run-<id>.txt` against that exact retained
run log. Missing or weak coupling remains inconclusive and non-authoritative.

Repair only the probe's diagnostic wrappers to match the current engine seams:
forward the capture-time `playback_level` keyword through `_looks_like_user` and
record that same value, and forward the synthesis generation and directives
through the gain-scaling `_synthesize` wrapper. These are diagnostic-hook parity
repairs, not core engine or audio-policy changes.

Preserve metric definitions, classification logic, exit meanings, one-stop
lifecycle ownership, AEC-delay snapshot rules, the DSP path, configuration,
defaults, thresholds, backend selection, runtime control behavior, and
interruption, identity, and tool authority. This ADR fully supersedes ADR-0180
while carrying its sentence, digest, guidance, and live-gate decisions forward.

## Context / why

The probe used legacy `engine.speak(text, on_done=...)` callbacks as its pacing
signal. For Sherpa, that callback can mean synthesis/enqueue completion rather
than completion or interruption at the playback sink. The probe therefore
incremented `sentences_spoken`, advanced to its next sentence after an
enqueue-side callback or timeout, and later reported success without possessing
the terminal observation needed to distinguish completed, interrupted, dropped,
or failed playback. Its deterministic sleeps also made report timing an
unversioned part of the observation population.

The engine boundary already supplies exactly-once tracked terminal receipts, and
the normal `VoiceRuntime` already uses them. Consuming that existing contract in
the diagnostic closes the probe defect without changing the normal reply path.
It does not prove that ordinary replies were cut or establish the cause of the
owner's previously observed tail cut; that remains a separate bare-speaker
hardware investigation.

An interrupted terminal cannot be treated as a tool failure: it is a valid
measurement of the self-interruption this probe is designed to expose. A dropped
or failed terminal, missing receipt, or unverifiable identity cannot support the
same diagnostic, so continuing or publishing partial normal fields would be
fail-open.

## Consequences

- Each new sentence waits for the prior sentence's sink-terminal state, and a
  successful lifecycle reaches stop only after all submitted sentences have
  terminalized. Failure cleanup still attempts the owned stop immediately.
- Missing, malformed, or raising tracked-terminal capability access is rejected
  immediately after construction and before instrumentation, engine/microphone
  start, or stop; `KeyboardInterrupt` is preserved. The post-start failure paths
  retain exactly one stop attempt.
- New success reports make submission, completion, and interruption distinct.
  `sentences_spoken` remains readable by old consumers but must never be cited as
  audible-completion evidence.
- Old reports remain records, but are not poolable with
  `tracked-sink-terminal-v1` reports even when the exact `N=3` or `N=4`
  `stimulus_id` matches. ADR-0179's old `N=4` stimulus is separately non-poolable.
- The interrupt suite remains a raw-report carrier and its established reduction
  and outcome semantics do not acquire receipt-validation authority.
- The probe's gain and barge instrumentation keeps parity with current
  generation/directive and playback-level call signatures.
- The exact focused `tests/test_echo_probe.py tests/test_interrupt_suite.py`
  gate passed 145 tests in 0.35s. Isolated, echo probe passed 80 in 0.28s
  and interrupt suite passed 65 in 0.14s.
- The exact adjacent `tests/test_sherpa_playback.py tests/test_sherpa_streaming_decode_session.py`
  receipt/drain/stop gate passed 111 tests in 2.20s.
- The exact five-file `tests/test_interrupt_suite.py tests/test_echo_probe.py tests/test_barge_self_interrupt.py tests/test_autotest_verdicts.py tests/test_diagnose_run.py`
  consumer-adjacent gate passed 312 tests in 1.35s.
- The standard APM/DTD regression passed 6 tests in 0.65s.
- Scoped Ruff lint is green for the exact command
  `/home/dobo/.local/bin/ruff check --no-cache tools/echo_probe.py tests/test_echo_probe.py tests/test_interrupt_suite.py`.
  Ruff format-check on the two changed tests reports `2 files already formatted`;
  AST parse,
  `git diff --check`, and the 100-line STATUS check are green. This supplies no
  whole-tool Ruff format claim.
- No model, network, GPU, microphone, audio device, physical suite,
  bare-speaker session, `./live.sh`, or `tools.live_audio_ab` run is claimed.
  No live result, tail-cut diagnosis, quality conclusion, core behavior,
  configuration/default/threshold/backend change, or new authority follows.
