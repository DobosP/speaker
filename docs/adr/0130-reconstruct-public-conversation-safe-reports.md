# ADR-0130: Reconstruct public-conversation safe reports

Date: 2026-08-05
Status: accepted

Refines: ADR-0127, ADR-0129

## Decision

Replace the public-conversation qualification v1 result with schema-v2 kind
`public-conversation-canonical-strata-qualification-v2`. Keep the generic
suite checkpoint private and reconstruct a fixed
`public-conversation-safe-aggregate-suite-v1` view from it. Admit only the
adapters selected by the two validated worker manifests from the harness
allow-list, bind each worker's adapter and fully resolved stream configuration
into the outer contract, and validate exact fixture,
worker, evaluator, configuration, denominator, stratum, endpoint, report,
binding, set, and matrix relations. Retain source-report and source-binding
digests under distinct fields, while computing new digests over the reconstructed
safe reports and bindings. Drop raw transcript-like fields, paths, worker text,
stderr, and unbounded evidence; retain only fixed aggregate metrics and bounded
resource observations.

Remove caller injection of the suite runner from the public API. Keep one
private production-runner seam for deterministic tests, and expose
`validate_qualification_report` for serialized retained reports. Recheck the
controller implementation, fixture, worker manifests, adapter/config resolution,
private scratch identity/mode, checkpoint identity/content, and absent output
after the runner and again after report reconstruction. Bound every checkpoint
and report read. Publish canonical JSON plus one newline by identity-safe,
failure-atomic no-clobber linking; make `write_new_report` return the exact
published-byte SHA-256 and print it from the CLI.

Continue to label success as coverage-only, development-only, non-promotional,
and without a quality threshold or runtime authority.

## Context / why

The v1 controller revalidated the generic suite result but then republished its
nested reports. A substituted runner could therefore supply internally
self-consistent transcript text, extra evidence, or forged aggregate counts and
have those values reach the public output. Generic suite schema identifiers also
did not describe the narrower retained surface. Default-derived stream settings
and adapter identity were not independently carried by the outer contract, and
one early scratch check did not cover mutation during reconstruction.

The qualification report crosses a trust boundary: production model workers and
private corpora produce the raw checkpoint, while the retained artifact must be
aggregate-only and independently reviewable. Validating caller-shaped output is
insufficient at that boundary; the controller has to reconstruct the permitted
surface and bind every denominator and implementation choice itself.

## Consequences

Existing v1 reports remain historical and do not validate as v2. Library callers
must stop passing `run_validated_suite_fn`, validate retained JSON through
`validate_qualification_report`, and consume the SHA-256 returned by
`write_new_report` instead of a `None` return. The file digest is an external
binding for the complete serialized report; it cannot be embedded into that same
report without a self-reference.

Coordinated nested rewrites now fail unless they preserve the fixed safe schema,
resolved worker contract, source/safe digest domains, exact aggregate arithmetic,
and outer matrix closure. A runner failure or close-time mutation leaves neither
the advertised report nor an owned temporary publication. The raw suite
checkpoint remains available only inside the owner-private scratch tree for
local diagnosis.

This hardens evidence handling but does not add model-quality thresholds, choose
an STT model, acquire EdAcc or Common Voice, run a model or GPU, validate a
microphone/device, or authorize live/runtime behavior.
