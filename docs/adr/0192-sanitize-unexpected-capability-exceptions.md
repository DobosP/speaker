# ADR-0192: Sanitize unexpected capability exceptions

Date: 2026-08-12
Status: accepted
Refines: ADR-0021, ADR-0033, ADR-0051, ADR-0076, ADR-0086, ADR-0189
Supersedes: none

## Decision

Publish one process-stable capability failure code,
`CAPABILITY_PROVIDER_FAILED = "capability_provider_failed"`. At the common
`CapabilityRegistry` provider boundary, catch an ordinary `Exception` without
binding, inspecting, stringifying, typing, logging, or retaining the exception
object. Return exactly `CapabilityResult(False, "",
error=CAPABILITY_PROVIDER_FAILED)`, with the dataclass defaults of empty data
and citations. This replacement must exist before an opt-in finished observer,
a task step, ReAct's textual/native observation, or a later local/cloud planner
prompt can consume the failure.

Validate a normal provider return with exact type identity before any observer
access: only `type(result) is CapabilityResult` is the typed success/failure
contract. A foreign object, proxy, or `CapabilityResult` subclass becomes the
same exact failed result without reading its attributes or invoking virtual
hooks. Each replacement receives the dataclass's fresh empty data dictionary.

Keep the normal typed-return contract separate. When a provider returns an exact
`CapabilityResult`, return that object unchanged, including its provider-authored
`ok`, text, data, citations, and error. Existing observer detachment/freezing
still applies only to the observer copy. Preserve the existing missing-capability
result and authority refusal. Thus this decision sanitizes an exception that
escapes a provider or an invalid foreign return; it does not reinterpret an
explicit domain error such as `cancelled`, `web offline`, or a truthful egress
receipt.

Keep direct process-control behavior explicit. A `BaseException` is not caught
by the common registry boundary and still propagates from a direct registry
call. Without observers it propagates directly. With opt-in observers, publish
one fixed finished observation and then bare-reraise the original; the observer
must not receive or inspect it. When the existing `TaskRuntime` provider thread
contains such an object, the coordinator must pop it from its outcome, clear
its reference, and return the same exact empty failed `CapabilityResult`. The
ordinary task-step failure path therefore publishes bare
`capability_provider_failed`, never a synthetic exception/log or the original
class, type name, arguments, string, repr, or traceback chain. Cancellation
precedence, provider-slot ownership, task
identity, ReAct failed-tool retry suppression/fallback, failure apology, and
all egress and action-authority gates remain unchanged.

This contract covers providers invoked through the canonical registry and the
assembled task path. A provider called directly, a provider that catches its
own exception and deliberately returns its details in an exact
`CapabilityResult`, and arbitrary same-process code remain outside it. So do a
provider's earlier logs/side effects, finalizers, observer `BaseException`, and
ordinary non-registry TaskRuntime/controller failures. The stable code is not
source classification, authenticated provenance, process isolation, memory
erasure, or proof that the failed provider stopped external/native work.

## Context / why

ADR-0189 fenced enumerated retained local context from cloud model and web
egress but recorded unexpected private-tool exception strings without source
tags as a residual. `CapabilityRegistry` converted a raised exception with
`str(exc)`. That string then looked identical to an intentional returned error:
an observer could record it, a task could publish it, and ReAct could insert it
as `Tool failed: ...` into its next planning/final prompt. An unvalidated foreign
return could likewise reach observer attribute access. Backend messages can
contain paths, queries, hostnames, configuration, account details, or other
local state, while hostile exception/result attributes and
`__str__`/`__repr__` implementations are executable same-process hooks.

[MITRE CWE-209](https://cwe.mitre.org/data/definitions/209.html) recommends
handling exceptions internally and exposing only minimal error information.
[OWASP LLM02:2025](https://genai.owasp.org/llmrisk/llm022025-sensitive-information-disclosure/)
also identifies application context, error, and configuration detail as
possible sensitive-information disclosure into an LLM system. These references
motivate a deterministic detail-free boundary; they are not runtime or
network-backed verification of this implementation.

Why not preserve only the exception type: module and class names can themselves
identify a backend or internal structure, and callers need one failure fact,
not a guessed diagnostic taxonomy. Why not redact exception text: a heuristic
cannot know which fragment is private and must first invoke the potentially
hostile conversion hook. Why not sanitize every explicit returned error:
`CapabilityResult` is the provider's typed compatibility surface and existing
controllers depend on deliberate cancellation, validation, fallback, and
egress receipts. Providers remain responsible for making those explicit errors
safe; silently changing them would collapse domain behavior into an unexpected
crash.

## Consequences

- An ordinary provider exception or non-exact/foreign provider return has one
  exact empty failed result on direct registry, observer, ReAct, and assembled
  task paths. ReAct may expose only the stable literal to a later
  otherwise-authorized model call; it cannot expose the original exception or
  foreign-object detail.
- A contained non-`Exception` provider abort reaches the task terminal path as
  the same exact failed result, so `TASK_FAILED.error` is the bare stable code.
  Direct registry callers retain Python's ordinary `BaseException` propagation
  semantics; subscribed observers receive a fixed finished receipt before the
  original is reraised.
- External diagnostics lose accidental exception detail. A provider that needs
  a useful caller-visible failure must deliberately return a safe typed
  `CapabilityResult`; any separate diagnostic logging must obey its own privacy
  policy outside this boundary.
- Provider-authored errors remain a trust boundary. A buggy or hostile provider
  can still return sensitive text/data explicitly, retain an exception itself,
  log or act before failure, mutate shared same-process objects, trigger
  finalizers, or bypass the registry through a direct call. Observer process
  controls and unrelated TaskRuntime/controller exceptions are not normalized.
- The failure replacement does not cancel or kill a provider, close a socket,
  release native resources, stop billing, alter retry/cancellation ordering, or
  grant any model, web, tool, audio, or device authority.

## Verification

Frozen verification used the low-priority/no-cache prefix
`/usr/bin/time -p env SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19
/home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider`.
Results are:

- `tests/test_capability_exception_sanitization.py
  tests/test_capability_context_isolation.py::test_escalated_turn_resets_context_on_failure_path
  tests/test_failure_cascades.py::test_llm_stream_raising_mid_token_does_not_wedge_the_runtime
  tests/test_failure_cascades.py::test_capability_raising_fails_the_turn_then_runtime_serves_the_next
  -q`: `14 passed in 0.38s` (`real 0.55s`, `user 0.45s`, `sys 0.03s`).
- The adjacent seven-file set (`tests/test_capability_exception_sanitization.py`,
  `tests/test_capability_context_isolation.py`, `tests/test_failure_cascades.py`,
  `tests/test_react_planner.py`, `tests/test_minicpm_tools.py`,
  `tests/test_conversation_eval.py`, and `tests/test_session_actor.py`) passes
  `262 in 7.34s` (`real 7.50s`).
- The exact current eleven-file cloud stage passes `319 in 10.18s`
  (`real 10.33s`). Its ordered paths are `tests/test_hedge_source_owner.py`,
  `tests/test_llm_egress_policy.py`, `tests/test_websearch.py`,
  `tests/test_multi_provider_llm.py`, `tests/test_hedge_chain.py`,
  `tests/test_hedge_chain_advanced.py`, `tests/test_cloud_providers.py`,
  `tests/test_routing_intent.py`, `tests/test_sensitivity.py`,
  `tests/test_cloud_integration.py`, and
  `tests/test_capability_context_isolation.py`.
- The task/cancellation set (`tests/test_never_stuck.py`,
  `tests/test_interrupt_race.py`, `tests/test_pretoken_cancellation.py`,
  `tests/test_capability_stream_close.py`,
  `tests/test_capabilities_cross_tier_retry.py`, and
  `tests/test_always_on_agent.py`) passes `88 in 7.66s` (`real 7.82s`).
- `tests/test_imports_smoke.py -q` passes `315 in 1.85s` (`real 2.11s`),
  and `tests/test_apm_double_talk.py -q` passes `6 in 0.74s`
  (`real 0.93s`).

AST parsing for the five Python paths, scoped Ruff with exactly eight
base-identical findings, inherited-only formatter debt, `git diff --check`, and
the exact 100-line STATUS are green. Receipt audit and architecture review are
GO. Final independent landing audit is also GO on the exact frozen 11 paths;
its 30 relevant tests/hostile probes, STATUS100, diff-check, and shared-base
cleanliness are green.

The exact inventory is two production paths, three test paths, and
six durable document paths: `always_on_agent/capabilities.py`,
`always_on_agent/tasks.py`, `tests/test_capability_exception_sanitization.py`,
`tests/test_capability_context_isolation.py`, `tests/test_failure_cascades.py`,
`STATUS.md`, `.agents/backlog.md`, `docs/agent-map.md`,
`docs/agent-testing.md`, `docs/unified_architecture.md`, and this ADR.

Frozen SHA-256 values are:

- `always_on_agent/capabilities.py`:
  `713f370beb4f49c64cffedf054510e7831153f078d845e01ab3137d9f82acbca`
- `always_on_agent/tasks.py`:
  `2aa0813e9f1b75bcdbe3012fffe03086984fff0b4ef101ffa24639837ba9d160`
- `tests/test_capability_exception_sanitization.py`:
  `8569f0c3a5dcdbf92b23264941fdd2dd5bb3887d555572aee93992876e81e8a8`
- `tests/test_capability_context_isolation.py`:
  `14273b2a8002168e67c44bae49397316bbb8c5cb81fa311522f06e026b318208`
- `tests/test_failure_cascades.py`:
  `31eb12329783ad6442e872ed15fe3942e56023e67259e02966f8c5e9c65d9c70`

No network, provider, model, GPU, audio, microphone, device, billing-bearing, or
live path ran. These receipts prove deterministic Python control-plane behavior
only; the explicit trust boundaries and lifecycle/resource nonclaims above
remain.

## Re-verification on current `main` (2026-09-07)

The decision above was written against `0223a2f` (2026-08-12) and landed only on
2026-09-07 against `main` `523f22f`. Main did not touch
`always_on_agent/capabilities.py`, `always_on_agent/tasks.py`,
`tests/test_capability_context_isolation.py` or `tests/test_failure_cascades.py`
in between, so all five frozen SHA-256 values reproduce byte-for-byte on the
merged tree.

Re-run with `env SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider`:

- the focused four-selector set: `14 passed` (unchanged);
- the adjacent seven-file set: `262 passed` (unchanged);
- the eleven-file cloud stage: `397 passed` — main grew that stage from the
  frozen `319`; `python tools/run_tests.py cloud` reports the same `397`;
- the six-file task/cancellation set: `89 passed` — main added one test to
  `tests/test_always_on_agent.py` since the frozen `88`;
- `tests/test_imports_smoke.py -q`: `322 passed` (frozen `315`);
- `tests/test_apm_double_talk.py -q`: `6 passed` (unchanged);
- the three-file gate now published in `docs/agent-testing.md`
  (`tests/test_capability_exception_sanitization.py`,
  `tests/test_capability_context_isolation.py`,
  `tests/test_failure_cascades.py`): `30 passed`.

The "exact 100-line STATUS" check in the frozen receipts is retired: `STATUS.md`
is now governed by the ≤120-line budget in `AGENTS.md`. The `STATUS.md`,
`docs/agent-map.md` and `docs/agent-testing.md` hunks written for this decision
were discarded rather than applied, because `523f22f` rewrote all three into the
fleet doc convention; equivalent content was rewritten in that convention instead.
