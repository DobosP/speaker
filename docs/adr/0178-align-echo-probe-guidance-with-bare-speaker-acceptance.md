# ADR-0178: Align echo-probe guidance with bare-speaker acceptance

Date: 2026-08-10
Status: superseded-by ADR-0179

## Decision

Apply ADR-0008, ADR-0012, and ADR-0176 to the affected serialized guidance in
`tools.echo_probe`. Keep the coherence and adaptive-DTD hints descriptive: they
may explain their existing fields, but one probe run must not recommend changing
a threshold or configuration value. Bind their physical-acceptance instruction
only to a real assistant session launched through `./live.sh`, a human talk-over
on the bare laptop speaker, and
`python -m tools.live_audio_ab logs/runs/run-<id>.txt` against that exact retained
run log.

Replace headphone and low-level `python -m core --session local` suggestions in
the affected coherence, adaptive-DTD, uncoupled, and top-level guidance. Missing
or weak coupling remains inconclusive and non-authoritative.

Preserve the audible probe stimulus, JSON keys and types, numeric metrics,
classification, exit codes, lifecycle, AEC-delay snapshot rules, DSP,
configuration, defaults, runtime behavior, and interruption/tool authority. This
implements ADR-0008's open-speaker requirement, retains ADR-0012's runtime
self-calibration boundary, and applies ADR-0176's exact live-evidence binding to
echo-probe guidance. It supersedes no decision.

## Context / why

ADR-0176 removed headphone fallback from the interrupt-suite presentation and
defined the only complete physical gate, but the child echo probe still
serialized headphone explanations, invited manual coherence/DTD tuning from one
diagnostic, and directed a real-barge check through the low-level core session.
Those strings could make a quiet-control or echo-only observation look like
configuration or physical-acceptance authority that ADR-0008, ADR-0012, and
ADR-0176 explicitly withhold.

Changing the metrics or test stimulus is rejected. The measurements remain useful
for diagnosing coupling and self-interruption risk, and existing consumers depend
on the stable result shape. The defect is the prescriptive and contradictory
serialized guidance, not the probe's stimulus, result algebra, or engine path.

## Consequences

- Existing JSON consumers retain the same keys, types, measurements, and outcome
  behavior; only the affected serialized guidance values change.
- A quiet echo-probe or interrupt-suite result remains diagnostic. Physical
  acceptance still requires the production launcher, human bare-speaker
  talk-over, exact retained log, and explicit analyzer invocation above.
- Headless tests cover the descriptive-only tuning boundary, forbidden headphone
  and low-level-session suggestions, exact launcher/analyzer binding, and
  unchanged interrupt-suite consumption.
- Focused echo/consumer gate: 177 passed, 1 absent-DTLN-ONNX case skipped in
  2.08s.
- Adjacent echo/self-interrupt/verdict/run-analysis gate: 271 passed in 1.30s.
- Standard APM/DTD regression: 6 passed in 0.68s. Scoped Ruff lint passes for
  the tool and both tests; both test files pass Ruff format, the whole-file
  echo-probe format check remains red on pre-existing debt, changed
  guidance/test regions are formatted, and whitespace is green.
- No model, network, GPU, microphone, audio device, physical suite, bare-speaker
  session, `./live.sh`, or `tools.live_audio_ab` run occurred for this change. No
  audible-stimulus, schema, metric, classification, exit, DSP, configuration,
  default, runtime, backend, authority, or live result follows.
