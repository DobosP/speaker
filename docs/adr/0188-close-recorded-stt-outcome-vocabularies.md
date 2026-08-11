# ADR-0188: Close recorded-STT outcome vocabularies

Date: 2026-08-12
Status: accepted

Refines: ADR-0078, ADR-0080

## Decision

Close `tools.recorded_stt_eval` offline outcomes to `unavailable`, `skipped`,
`error`, `decoded`, and `empty`. Close verifier outcomes to `unavailable`,
`skipped`, `error`, `consensus`, `empty`, `tie`, `no_quorum`,
`control_guard`, `attested_control`, `empty_veto`, and
`empty_streaming_guard`. Treat these as explicit evaluator-boundary
vocabularies rather than deriving them dynamically from configuration or
recognizer objects.

Validate every terminal decision outcome before aggregate counting. Accept
only exact built-in strings in the relevant vocabulary; do not coerce values,
call their string or representation hooks, or preserve exception detail.
Snapshot each aggregate map into detached read-only state, accepting only
exact built-in string keys and exact nonnegative built-in integer counts, and
omit zero-count entries.
Offline and verifier accounting are
independently complete only when their validated count sums each equal the same
exact positive terminal-decision count. Recorded baseline acceptance and
candidate promotion require both complete outcome maps in addition to the
existing selected-source closure.

Keep outcome accounting distinct from configured-model quality. A disabled
offline or verifier backend still bypasses its existing decode-quality gate,
but its per-decision `unavailable` outcomes remain part of complete aggregate
accounting. An enabled offline backend still requires at least one `decoded`
or `empty` outcome and no `error`; an enabled verifier still requires at least
one completed safe outcome and no `error`. No outcome value changes final-text
selection or supplies recognizer authority.

## Context / why

ADR-0124 closed and detached selected-source accounting, but deliberately did
not cover the older offline and verifier maps. Those maps still accepted raw
decision strings into `Counter` objects and report payloads. An unknown or
hostile value could therefore cross the aggregate boundary, and a partial map
could satisfy the configured-model quality check without proving that every
terminal decision was accounted for.

The offline vocabulary is production-local state, while verifier state
combines production lifecycle markers with the finite acoustic-consensus
outcomes. Reflecting enum members or accepting arbitrary strings would let
producer drift silently redefine retained report schema. Explicit boundary
sets make such drift a reviewed change and turn malformed or new outcomes into
one detail-free prerequisite failure before report publication.

## Consequences

Recorded evaluation reports retain only closed aggregate outcome counts, with
no new serialized completeness fields. A caller cannot mutate accepted maps
afterward. Unknown keys, boolean or non-integer counts, negative counts,
hostile mappings, or hostile value hooks raise the evaluator's detail-free
prerequisite failure. A valid but incomplete or sum-mismatched map remains
constructible and serializable for generic selector/consensus compatibility,
but cannot satisfy recorded acceptance or promotion.

This refines ADR-0078's aggregate recorded gate and ADR-0080's required-model
completion rule. It is follow-up hardening, not an expansion of ADR-0124's
selected-source slice, and supersedes no prior decision. Models, selectors,
profiles, thresholds, defaults, authority, report transcript privacy, and
entry points remain unchanged.

Frozen headless verification passes 175 focused tests in 1.66s, 221
recorded-plus-route tests in 1.89s, 140 downstream evaluator-consumer tests
plus one skip in 2.22s, 421 broad exact-private consumer tests plus one skip in
7.96s, and 6 APM/DTD tests in 1.08s. Independent audit is GO after reproducing
221 focused-plus-route, 82 generic selector/consensus/guided, 109
production-final/FileReplay plus two expected skips, and 6 APM/DTD tests; its
inline vocabulary/report/detachment/mismatch probe, AST3, scoped Ruff, and
diff-check are green. Frozen SHA-256 values are tool
`98b9bf9250818361ba3fa10621abc16757b58eef8d19d71428016b3a5f8dc8ad`,
recorded test
`de093890c5951790fc11cacfbe39733916ce589e0a9e52f7fb7fd06038700f44`, and
route test
`fadc282c61b47d00997a168d8c7f455df21866cb811782903c5b4aff768487dc`.
These tests use generated aggregate values and fake selectors only; they
provide no owner corpus, transcript, model, GPU, network, capture, microphone,
audio device, quality, latency, or live evidence.
