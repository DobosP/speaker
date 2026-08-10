# ADR-0180: Neutralize the echo-probe fourth stimulus

Date: 2026-08-10
Status: superseded-by ADR-0181

## Decision

Carry forward ADR-0179's complete submitted-text identity, guidance, and live-gate
decision. Every full successful `tools.echo_probe` report still carries the
top-level scalar
`stimulus_id = "echo-probe-spoken-text-v1:sha256:<64 lowercase hex>"`. Build it
from the same frozen `max(1, args.sentences)` cyclic plan submitted to returned
`engine.speak` calls: initialize SHA-256 with the exact domain
`b"speaker.echo-probe.spoken-text.v1\0"`, then append each exact UTF-8 byte
length as an unsigned eight-byte big-endian integer followed by those exact
bytes. Do not normalize text or append a newline.

Narrowly supersede only ADR-0179's requirement to preserve the fourth stimulus
sentence. Replace the exact old text
`If you hear this whole message without it cutting off, suppression works.`
with the exact neutral text
`This final sentence completes the quiet playback portion of the diagnostic.`
Preserve the first three sentence bytes, sentence order, cycling, count
normalization, playback control flow, and wait/deadline/sleep policy. Only the
fourth submitted text, its synthesized stimulus and likely duration, and the
derived four-sentence identifier change.

The interrupt-suite default requests three sentences and therefore does not
submit the changed fourth sentence. Its exact identifier remains
`echo-probe-spoken-text-v1:sha256:27c744fbb14bbcb74f9256a2a4db2af25c6e7ad6c70e5ea224c3c8708a06a3c6`.
The echo-probe default requests four sentences and now identifies the changed
population as
`echo-probe-spoken-text-v1:sha256:4366d6a869714ed767964bda1dc682b5264c109d5d3f12260f6e07276c502704`.
Its prior ADR-0179 four-sentence identifier
`echo-probe-spoken-text-v1:sha256:be277009edac56b0473730c1e5326a1b765976386aca9f26c500eab9f00822eb`
belongs to a different population; the two must not be pooled.

The identifier remains self-attested submitted-text provenance. It binds only
the exact ordered text arguments whose `engine.speak` calls returned on the
normal-report path; it proves neither synthesis nor completed or audible
playback and does not authenticate a hostile or modified child. Lifecycle,
setup, and serialization failures remain identity-free. The interrupt suite
continues to preserve an additive raw child value without validating or using
it in summary, coupling, classification, or exit logic. Missing or unequal
identifiers remain non-poolable; equality remains necessary rather than
sufficient because device, gain, strategy, TTS, configuration, route, room, and
other evidence dimensions still apply.

Carry forward ADR-0178's guidance as incorporated by ADR-0179. Echo-probe
coherence, adaptive-DTD, uncoupled, and top-level guidance remains descriptive,
must not recommend tuning from one run, and must not suggest headphones or the
low-level core session. Physical acceptance still requires a real assistant
session launched through `./live.sh`, a human talk-over on the bare laptop
speaker, and
`python -m tools.live_audio_ab logs/runs/run-<id>.txt` against that exact retained
run log. Missing or weak coupling remains inconclusive and non-authoritative.

Preserve report keys and types, metric definitions and fields, classification
logic, exit codes, lifecycle rules, AEC-delay snapshot rules, the DSP path,
configuration, defaults, the runtime control path, backend selection, and
interruption/tool-authority rules. This ADR fully replaces ADR-0179 while changing
the fourth submitted text, its resulting synthesized stimulus and likely duration,
and the deterministic four-sentence identity. It continues to implement
ADR-0008's open-speaker requirement, ADR-0012's self-calibration boundary, and
ADR-0176's exact live-evidence binding.

## Context / why

ADR-0179 intentionally versioned the submitted text before this wording change.
The old fourth sentence says that the sequence ensures suppression works, which
can sound like a successful verdict even though the probe is only a quiet
echo-only diagnostic. A neutral completion sentence removes that false success
implication without turning the probe into a different control or acceptance
path.

Changing only the fourth item keeps the interrupt-suite's three-sentence
stimulus stable while allowing the identity contract to separate old and new
four-sentence reports. Reusing the old identifier or treating a repository
revision as an implicit migration would erase the exact report-local provenance
that ADR-0179 added.

## Consequences

- The fourth submitted text, its synthesized stimulus and likely duration, and
  the derived default-four identity change. Playback control flow and the
  wait/deadline/sleep policy do not.
- New default four-sentence reports form a distinct submitted-text population;
  prior four-sentence reports remain valid records but are not poolable with it.
- Future observed echo, ERLE, coupling, and interruption metric values may differ
  because the synthesized stimulus and likely duration differ. The unchanged
  metric definitions and classification logic do not make old and new `N=4`
  observations comparable.
- The default three-sentence interrupt-suite text and identifier are unchanged.
  Its reduction and outcome contract remain unchanged as well.
- The neutral sentence does not claim suppression, barge-in, coupling, playback,
  or physical acceptance succeeded.
- Focused four-file echo/consumer gate: 190 passed, 1 skipped because the DTLN
  ONNX is absent (`tests/test_aec_seam.py:930`) in 2.06s.
- Adjacent AEC/capture/teardown gate: 214 passed, 1 skipped for the same absent
  DTLN ONNX in 3.84s.
- Interrupt-suite single-file gate: 64 passed in 0.14s. The five-file consumer
  gate passed 284 in 1.36s; `tests/test_interrupt_suite.py` is unchanged.
- Standard APM/DTD regression: 6 passed in 0.67s.
- Scoped `/home/dobo/.local/bin/ruff` lint is green for the tool and both tests;
  both test files pass Ruff format, and AST, whitespace, and the 100-line STATUS
  check are green. Whole-tool Ruff format remains red only for inherited debt;
  the changed sentence region is clean.
- No model, network, GPU, microphone, audio device, physical suite,
  bare-speaker session, `./live.sh`, or `tools.live_audio_ab` run is authorized
  or claimed by this change. No live result, quality conclusion,
  configuration/default change, backend change, or new authority follows.
