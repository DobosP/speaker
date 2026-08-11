# ADR-0187: Bind web-search egress to exact turn sensitivity

Date: 2026-08-12
Status: accepted
Refines: ADR-0003

## Decision

Before `web.search` coerces mode/intent, invokes its injectable raw-query
classifier, or calls a backend, treat a supplied turn-sensitivity field as a
monotonic restriction. Preserve the historical raw-query gate when the field
is absent because deterministic SEARCH/RESEARCH plans do not always carry one.
That exception applies only at the direct web capability boundary: ReAct marks
an absent entry field invalid because Assistant-owned escalations always carry
the classified turn before a planner can generate a replacement query.
The capability registry must distinguish only `None` from a supplied context;
it must not discard an explicit empty or falsey dict by invoking truthiness.
Opt-in invocation observers copy only an exact built-in dict and publish only
exact-string keys whose values are also exact strings. This observer snapshot
is isolated from the context passed to the provider, so diagnostic field
extraction cannot invoke context or value hooks before the provider boundary.
The web provider accepts absent-field compatibility only from an exact built-in
dict, takes one built-in shallow snapshot, and uses it for sensitivity, mode,
and intent admission; dict subclasses fail closed before virtual lookup hooks
can hide a field.
ReAct likewise scans exact-string keys through built-in iteration and marks a
non-exact entry Mapping or non-string key invalid before any virtual lookup or
tool-context copy, so direct planner calls cannot manufacture absence.
Non-string context keys also fail closed before dictionary equality hooks run.
When it is present, accept only the exact canonical strings `public` and
`code` as permission to continue to the unchanged raw-query/mode/intent gate.
Exact `private`, null, non-string, unknown, empty, or case-variant values must
fall back locally before classifier or backend work. Inspect type and exact
membership without invoking arbitrary equality, hashing, or string-coercion
hooks.

From that accepted snapshot, mode and intent retain their historical `None`
semantics: actual enum members pass through, only exact built-in strings enter
enum construction, and `None`, unknown strings, or non-string/hostile values
abstain as `None`. This precheck does not call hostile hash, equality, or string
hooks, but malformed or absent mode/intent is not a new context veto; the raw
query and remaining gate inputs still decide.

Neither `public` nor `code` grants egress. The exact generated tool query must
still pass `may_leave_device`, including its personal/PII, mode, intent, and
classifier-failure fences. A denial returns the existing successful local
corpus fallback with `egress=false`, `source=corpus`, and PRIVATE sensitivity;
it never calls SearXNG. `web_search.enabled` remains mandatory even when a
backend is injected, so disabled configuration cannot be bypassed through the
test/custom-backend seam. Keep the default-off SearXNG configuration, bounded
per-phase connect/read/write/pool inactivity timeouts (not a total wall-clock
deadline), normal result mapping, and empty/error fallback otherwise unchanged.
Non-mapping backend hit entries are skipped. If no usable hit remains, return
the successful corpus fallback with `egress=true`; if backend iteration or
mapping/value normalization raises, do the same with the exception type in the
error receipt. Payload shape cannot abort the plan or erase the fact that the
backend call already occurred. Normalize only the configured `max_results`
prefix rather than materializing an injected backend's whole result iterable.

This refines ADR-0003 without superseding its self-hosted SearXNG, cloud-host,
or memory-backend decisions. It replaces only the query-only web admission
rule preserved as historical BR3 in `docs/archive/p3_design.md`; that archived
record remains unchanged. ADR-0097's audio-egress topology and the separate
sensitivity-routed cloud LLM chains are unchanged, so no existing ADR status
changes.

## Context / why

The answering capability classifies the current turn, floats it toward PRIVATE
over bounded recent conversation, and publishes that same context while an
escalated ReAct planner runs. The planner may then generate an innocuous-looking
`web.search` argument that omits the private material responsible for the turn
classification. The historical gate reclassified only that generated argument,
so it could permit a backend call despite the enclosing exact turn remaining
PRIVATE.

Trusting the turn tag as positive authorization would create the inverse bug:
a forged or stale `public` value could declassify raw PII. Requiring the field
unconditionally would also disable deterministic SEARCH/RESEARCH calls whose
task context legitimately lacks it. The field is therefore veto-only when
present, while absence is compatibility only with the independently mandatory
raw gate. The canonical strings are structural labels, not opaque or
authenticated provenance. Malformed present values fail closed because they
cannot prove one of the two canonical non-private states.

The registry previously replaced any falsey supplied context with a new empty
dict. A hostile falsey dict subclass could therefore lose its restrictive
field before the web provider inspected it. Even when preserved, virtual
lookup hooks or a non-string key comparing equal to `sensitivity` could conceal
or counterfeit absence. One exact built-in snapshot and exact key types keep
the compatibility exception structural and hook-free. On the output side, a
backend can return non-mappings or mapping values whose access or string
normalization raises; those untrusted shapes must degrade to the local fallback
after recording that the backend call already happened.

## Consequences

- A PRIVATE recent-context turn cannot reach SearXNG merely because its planner
  emits a public-looking query; falsey dict subclasses, other non-exact or
  malformed sensitivity containers, and non-string keys also stay local
  without user-defined admission hooks.
- Context-free deterministic search preserves its existing behavior, and a
  canonical non-private turn still cannot bypass query, mode, intent, or
  classifier denial.
- Registry observers receive only isolated exact-string metadata, and direct
  ReAct calls invalidate missing sensitivity, a non-exact Mapping, or any
  non-exact-string key before tool-context copying. This does not sandbox a
  same-process provider or other code retaining an alias to the context or to
  objects reachable through its shallow values; later mutation at that trust
  boundary remains out of scope.
- Malformed backend hits cannot fail the plan. Skipped/failed normalization
  retains a successful corpus fallback and a truthful post-call egress/error
  receipt.
- Existing raw-query classification remains heuristic and can have false
  negatives. Once a blocking injected/custom backend call begins it is not
  forcibly cancellable, and this change adds no bound on the transport's parsed
  response body, trusted configured `max_results`, or individual result-value
  size. Provider normalization does consume only the configured result prefix;
  the shipped SearXNG timeout remains per-phase inactivity, not total wall time.
- This closes only the web-search defense-in-depth gap. PRIVATE text may still
  use the separately configured US cloud LLM chain, and the broader injected-PII
  cloud-LLM review item remains open.
- Frozen low-priority headless receipts are: focused three-file `96 passed in
  0.85s`; policy/context adjacent `111 passed in 0.75s`; capability/observer
  adjacent `161 passed in 7.58s`; ReAct/untrusted/cancellation adjacent `150
  passed in 0.82s`; and APM/DTD `6 passed in 1.65s`. Six-file AST parsing and
  diff-check are green. Scoped Ruff is green while ignoring only React's
  identical inherited F401; all 37 remaining formatter hunks are base-identical
  with zero task-line overlap. Frozen SHA-256 is
  `257e8e0ec6be708b01a987f403cb527658ee70a1f9d9c1bdb5109ac12a602f0d`,
  `630cc8ff679ec32f996558164a0dae024c20ff42e0a1de32cc1d7c20f2c6e4c5`,
  `3a91433199662d1f8acbf65a5e140388771c13f2c0ab4c782957b04eb09887d1`,
  `7e4d65e9b59e5a32023f4a16446bea02a44148ee433d03cf530388ccaa8e36a8`,
  `bf3dc820e6646bb7d0343324257ba4b78654144c2bde1fc039e24c90c925de96`,
  and `c246d309e146302e8884f7b4ce5e521d0bbd8ee5e922240f566d21f9c1ee377f`
  for capabilities, ReAct, sensitivity, websearch, websearch tests, and ReAct
  tests respectively. No network, SearXNG, cloud/model, microphone, audio
  device, or live path ran.
