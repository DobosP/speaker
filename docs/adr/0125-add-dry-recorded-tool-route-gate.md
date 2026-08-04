# ADR-0125: Add a dry recorded tool-route gate

Date: 2026-08-05
Status: accepted

Refines: ADR-0124

## Decision

Add an opt-in `--tool-route-gate` mode to `tools.recorded_stt_eval` for the
canonical private schema-v3 exact-input corpus. Require every gated case to
carry exactly one closed annotation:
`expected-tool.none`, `expected-tool.vault.search`,
`expected-tool.web.search`, `expected-tool.reminder.create`,
`expected-tool.reminder.list`, `expected-tool.reminder.cancel`, or
`expected-tool.app.open`. Reserve the entire `expected-tool.` prefix; reject a
missing, unknown, or multiple annotation before model startup. Reject legacy
corpora when the gate is requested. Require exactly one nonempty selected
terminal decision per case and never join decisions for routing.

Classify recognized text through the production deterministic
`LiveSpeechAnalyzer`, `TaskPlanner`, and
`DeviceToolCommandDispatcher.match` contracts in assistant mode. Obtain vault
and web routes from the closed plan steps and typed reminder/app routes from
the prepared `DeviceToolCommand`; keep control, generic-action, other-action,
and unrouted outcomes closed. Model optional setup availability only through
explicit `--tool-route-vault-enabled`,
`--tool-route-reminders-enabled`, and repeatable
`--tool-route-app-alias` inputs. Bind their normalized semantics, the fixed
clock, and the implicit web-planning profile into a domain-separated digest;
emit only that digest.

Construct no production provider or reminder store. Reminder parsing uses a
fixed aware UTC clock and empty read-only state. The trusted-app launcher and
capability registry raise if reached. Call only dispatcher `match`: never call
`dispatch`, registry `invoke`, a provider, supervisor/task runtime, ReAct/LLM,
or any capability. Reduce each private case immediately to aggregate closed
counters. Baseline success requires a complete gate with at least one positive
and one negative case. Candidate promotion requires all existing accuracy,
source, offline, and verifier safety checks for both configurations plus a
complete non-regressing candidate route gate. A candidate may use correction
of a red baseline route gate as its improvement; a green-to-green route tie is
not an improvement.

## Context / why

ADR-0124 retained exact case tags and one slot per final selector decision, but
the evaluator stopped at transcription accuracy. The owner therefore could
not tell whether STT output from phrases such as “search”, “go”, or “find” in
the vault would reach the same deterministic route used by the voice agent.
Joining multiple terminal strings would hide segmentation failures, while
constructing `VoiceRuntime` or real tool providers inside an evaluator would
turn development evidence into an action surface. Reading machine-local tool
configuration would also make reports irreproducible and risk disclosing
private roots, aliases, or provider state.

## Consequences

One exact-input A/B run can now measure final transcript accuracy, selected
source accounting, and proposed recognized-text routing without a second app
entry or any tool execution authority. Reports contain only aggregate counts,
closed route names, and a profile digest; transcripts, tags, aliases, reminder
identities, vault paths, and per-case rows remain private. Invalid annotations,
profiles, terminal boundaries, route drift, and malformed aggregate contracts
fail through the existing detail-free boundary.

This is an after-endpoint dry proposal check. It excludes capture and VAD
timing, AEC, barge-in, enrollment/identity, origin and direct-live authority,
confirmation, provider readiness, reminder state/identity, vault reads,
cleaning/addressing/continuation, ReAct/LLM routing, external effects, device
behavior, and live latency. It changes no runtime, public app entry, provider,
model, setup default, or capability authority. Owner-labelled recordings and
physical open-speaker A/B remain required.
