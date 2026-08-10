# ADR-0179: Bind exact echo-probe submitted-text identity

Date: 2026-08-10
Status: accepted

## Decision

Carry forward ADR-0178's complete guidance and live-gate decision. Echo-probe
coherence, adaptive-DTD, uncoupled, and top-level guidance remains descriptive,
must not recommend tuning from one run, and must not suggest headphones or the
low-level core session. Physical acceptance still requires a real assistant
session launched through `./live.sh`, a human talk-over on the bare laptop
speaker, and
`python -m tools.live_audio_ab logs/runs/run-<id>.txt` against that exact retained
run log. Missing or weak coupling remains inconclusive and non-authoritative.

Narrowly supersede ADR-0178's no-schema-change clause. Add one top-level scalar
to every full successful `tools.echo_probe` report:

`stimulus_id = "echo-probe-spoken-text-v1:sha256:<64 lowercase hex>"`.

Normalize the requested sentence count once as `max(1, args.sentences)` and
snapshot `tuple(SENTENCES)` once. The same frozen count and tuple must drive both
the identifier and the ordered cyclic text calls submitted to `engine.speak`.
Initialize SHA-256 with the exact domain
`b"speaker.echo-probe.spoken-text.v1\0"`. For each submitted sentence in order,
append its UTF-8 byte length as one unsigned eight-byte big-endian integer and
then its exact UTF-8 bytes. Do not normalize text or append a newline.

The identifier binds only the exact ordered text arguments whose `engine.speak`
calls returned on a normal-report path. It does not claim that synthesis,
playback, or audibility completed. Lifecycle, setup, and serialization failures
remain identity-free error results and cannot be compared as normal reports.
`tools.interrupt_suite` preserves the additive field in each raw child object but
does not validate it or use it in summary, coupling, classification, or exit
logic.

Preserve all four current `SENTENCES` strings byte-for-byte, including the fourth
sentence's `suppression works` wording. Preserve playback order and sleeps, JSON
keys and types other than the one additive success-only scalar, numeric metrics,
classification, exit codes, lifecycle, AEC-delay snapshot rules, DSP,
configuration, defaults, runtime behavior, backend selection, and
interruption/tool authority. Any neutral fourth-sentence wording is a separate
future ADR and branch after this identity contract lands.

This ADR fully replaces ADR-0178 while changing only its no-schema-change clause.
It continues to implement ADR-0008's open-speaker requirement, retain ADR-0012's
self-calibration boundary, and apply ADR-0176's exact live-evidence binding.

## Context / why

ADR-0178 corrected prescriptive and contradictory serialized guidance while
deliberately preserving the audible stimulus and result shape. Its fourth
sentence still says `suppression works`, which can sound like an acceptance
verdict. Changing that text before reports identify their submitted text would
make acoustically different probe populations indistinguishable.

Historical reports contain a sentence count but no binding to the exact ordered
text submitted for that run. A repository revision is not a report-local
binding, a catalog-only digest would include unused sentences and omit cyclic
count, and copying the public sentence text into every report is unnecessary.
The domain-separated, length-framed sequence digest is compact and unambiguous
across text, order, repetition, and UTF-8 byte boundaries.

The field is self-attested provenance, not authentication against a hostile or
modified child process. Equal submitted text also does not make TTS voice,
device, route, gain, strategy, configuration, room, or other conditions equal.
It therefore gates only the stimulus-text dimension of comparability.

## Consequences

- A legacy report without `stimulus_id` must not be pooled with an
  identity-bearing report, and historical reports must not be backfilled from an
  assumed source revision. Reports with unequal identifiers must not be pooled.
- Equal identifiers are necessary, not sufficient, for comparison. Existing
  device, strategy, gain, TTS, configuration, route, and evidence boundaries
  still apply.
- The current interrupt-suite default count of three intentionally excludes the
  unused fourth sentence; its identifier is unchanged by a future edit to that
  sentence. The echo-probe default count of four includes it and must receive a
  different identifier when that later text change lands.
- Headless tests pin the four unchanged sentence bytes, count normalization and
  cycling, domain and unsigned length framing, exact current three- and
  four-sentence identifiers, the shared frozen plan, success-only publication,
  identity-free errors, raw child preservation, and unchanged suite reduction.
- Focused four-file echo/consumer gate: 190 passed, 1 skipped because the DTLN
  ONNX is absent (`tests/test_aec_seam.py:930`) in 2.07s.
- Adjacent AEC/capture/teardown gate: 214 passed, 1 skipped for the same absent
  DTLN ONNX in 3.83s.
- Interrupt-suite single-file gate: 64 passed in 0.14s. The five-file consumer
  adjacent gate passed 284 in 1.34s.
- Standard APM/DTD regression: 6 passed in 0.69s. Scoped Ruff lint is green for
  the tool and both tests; both test files pass Ruff format, and whitespace and
  AST checks are green. The only newly introduced blank-line formatter hunk was
  corrected, so the changed helper, report, and test regions are formatted while
  the inherited whole-tool Ruff-format debt remains.
- No model, network, GPU, microphone, audio device, physical suite, bare-speaker
  session, `./live.sh`, or `tools.live_audio_ab` run is authorized or claimed by
  this change. No audible-stimulus, metric, classification, exit, DSP,
  configuration, default, runtime, backend, authority, or live result follows.
