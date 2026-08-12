# ADR-0190: Bound Hedge workers by exact source ownership

Date: 2026-08-12
Status: accepted
Refines: ADR-0021, ADR-0030, ADR-0189
Supersedes: none

## Decision

Give every factory-created `HedgeLLM` source an exact, stable key in one
process-wide registry. The local main member uses `factory/LOCAL_MAIN`; the
single-cloud compatibility member uses `factory/SINGLE_CLOUD`; and a configured
named cloud member uses `factory.named_cloud/<exact preset name>`. Reusing a
local main client or named preset across sensitivity chains or rebuilding the
factory therefore reuses its source key. Different exact preset names remain
different keys even when they alias the same client or endpoint. A directly
constructed Hedge instead owns an instance-private registry with `direct/LOCAL`
and index-stable `direct/CLOUD_<n>` keys; separate direct wrappers do not claim
process-wide exclusion.

Before an asynchronous Hedge member can construct a thread, publish one exact
`HedgeSourceOwner` for its key. Registry reservation, owner publication, thread
construction, and start-attempt state are distinct: publication precedes both
thread construction and `Thread.start()`, and an escaping construction/start
failure retains the owner unless the exact published target later proves its
return. Do not wait for a busy source, enqueue a successor, construct another
thread/client, or retry that key. Treat BUSY as an unavailable chain member and
advance immediately through the remaining distinct cloud keys and then the
local key according to the existing hedge/fallback order. A late queue item
from a retired member cannot make it a winner.

The owner's finite Python payload is the exact source client, prompt, optional
system text, images, history, output queue, stop event, exact captured turn
context and `ContextVar`, plus a diagnostic tag. The static thread target
receives only that owner. On return, the worker invokes any available stream
close seam, restores the context, and clears every payload and stream alias
before publishing return.
Only the exact owner may then reap: return and reference cleanup must be proven,
the exact thread must be observed dead and accept a zero-time join, and registry
release must match owner identity. A close/reset/join/liveness/registry-release
failure remains retained. Provider exceptions publish only their exact type
name; they do not copy an arbitrary exception string into the coordinator.

Use the same exact source key for a synchronous local-only/no-cloud call, but
claim a no-thread `HedgeSourceLease`. A busy key yields an empty stream without
calling the source. Release the lease only after any available local-stream
close seam returns, the captured context is restored, and call-payload aliases
are cleared; retain it after ambiguous cleanup. Thus ADR-0189's local-only
branch bypasses cloud racing and a shared queue without bypassing source
ownership.

Keep Hedge coordinator cleanup bounded to the existing shared
`WORKER_JOIN_TIMEOUT = 0.5` seconds for all owners. Request stop and invoke only
an explicitly advertised cross-thread-safe, nonblocking stream cancel seam,
then—after those contract-bearing calls return—spend at most that one shared
budget trying to reap the owners. The outer Hedge call and ADR-0021
task/provider-capacity slot may retire after the budget while an exact source
owner remains. That durable owner, rather than an indefinite coordinator wait,
prevents every same-key successor from starting; unrelated keys remain
available. This deliberately refines ADR-0021's outer-slot retention semantics
for Hedge members. A stream that falsely advertises nonblocking cancellation can
still block the coordinator before the join budget begins; the owner does not
turn arbitrary callbacks into preemptible work.

Keep the existing finite pre-first-token winner-selection budget, 30-second
post-first-token idle bound, cloud-egress admission and sensitivity policy,
provider ordering, output behavior, and shared `last_source` diagnostic. Add
the exact owner test to the canonical `tools/run_tests.py cloud` stage. This
decision changes lifecycle ownership only; it does not authorize new egress or
change model, provider, timeout, token, billing, audio, or device policy.

## Context / why

ADR-0021 bounds an abandoned synchronous provider by its outer provider slot,
and ADR-0030 adds an audited cooperative abort/reset seam to direct CPU
llama.cpp inference. `HedgeLLM`, however, creates an additional thread per local
and cloud member. Its old 0.5-second join budget let a stalled loser outlive the
Hedge wrapper; a rebuilt wrapper or a concurrent sensitivity chain could then
start another worker against the same local model/client or named cloud source.
Repeated turns could multiply Python threads, shared-client/native contention,
retained request objects, sockets, and provider work even though each individual
Hedge call returned.

Keying by transient chain index or wrapper identity would not close that gap.
The same local main object is installed in every factory sensitivity chain, and
one named provider preset can be present in several chains and factory rebuilds.
Conversely, a fixed single global permit would make one stalled provider block
unrelated local and cloud sources and erase failover. Exact stable per-member
factory keys preserve the intended independent chain progression while bounding
same-source multiplication.

Do not indefinitely retain the Hedge coordinator and its outer ADR-0021 slot
for a stream that advertises cooperative cancellation. An advertised
cancel/close hook can itself fail, block, or return while native/provider work
remains uncertain. The durable exact-source owner supplies the narrower proof
that matters—no same-source successor—while the bounded coordinator lets other
admitted work continue after contract-conforming cancel calls return.

Like ADR-0185's KWS owner, this uses publish-before-start, conservative ambiguous
start retention, alias clearing, and exact reap proof. ADR-0185 is analogous
proof shape only: its single process-wide KWS permit, 50 ms capture deadline,
speaker authority, restart refusal, and callback rules do not apply to Hedge.

## Consequences

- At most one may-start, running, returned-but-unreaped, or cleanup-uncertain
  Hedge invocation exists for each exact factory source key. There is no global
  Hedge-worker cap: distinct named providers, the local main source, and the
  single-cloud compatibility source can each have one owner, and newly invented
  preset names create newly independent keys.
- A busy local/cloud member fails fast and fallback/hedge progression can use a
  different key. A retained local key can therefore leave cloud keys usable;
  one retained cloud key can be skipped for the next cloud or local member.
- One hung owner can retain its client/model, prompt, system text, history,
  images, captured context, stream/socket, native CPU state, SDK resources, and
  provider request. A cloud request can keep accruing charges until it actually
  returns or the process exits. A different provider key can also run and bill;
  the registry does not provide an account-wide spend limit. The payload has a
  fixed field set, not a byte/object-graph cap; retained client, image, and
  history objects may themselves own larger graphs or native allocations.
- Stop/cancel/close are cooperative requests only. Python cannot force a native
  or SDK call to return, and an entered call may release the GIL or hold it
  depending on provider/build. This decision claims neither GIL availability,
  native-thread/storage/buffer release, transport closure, billing cessation,
  nor a hard process kill. The coordinator's 0.5-second join budget also cannot
  preempt an advertised cancel callback that violates its nonblocking contract.
  If a retained call never returns, process exit is the only universal recovery
  boundary.
- Factory keys cover only the canonical main Hedge assembly. The raw
  `local_main`/`fast_llm` aliases, directly invoked clients, custom providers,
  directly constructed instance-private Hedges, and the same endpoint/client
  configured under distinct exact preset names can bypass or partition this
  ownership fence. Same-process mutation and `last_source` races remain outside
  the claim.
- Deterministic fake-thread/client tests can prove registry, ordering, BUSY,
  fallback, ambiguous-start, cleanup, alias-release, and factory-rebuild
  contracts. They cannot prove a provider/model/native call returns, the GIL is
  released, a socket closes, billing stops, or live/device behavior.

## Verification

Frozen verification used the low-priority/no-cache prefix
`/usr/bin/time -p env SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19
/home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider`.
Results are:

- `tests/test_hedge_source_owner.py -q`: `44 passed in 0.27s`
  (`real 0.40s`, `user 0.34s`, `sys 0.05s`).
- `tests/test_hedge_source_owner.py tests/test_hedge_chain.py
tests/test_hedge_chain_advanced.py tests/test_multi_provider_llm.py
tests/test_llm_egress_policy.py tests/test_llm_sanity.py
tests/test_llm_thread_budget.py -q`: `256 passed in 9.07s`
  (`real 9.22s`, `user 0.90s`, `sys 0.10s`).
- The exact current `tools/testing/stages.py` cloud inventory—the owner test
  followed by ADR-0189's ten paths—passes `319 in 10.29s`
  (`real 10.45s`, `user 1.44s`, `sys 0.17s`).
- `tests/test_ollama_async_cancel.py tests/test_cloud_pii_egress.py
  tests/test_openrouter_prc.py tests/test_core_routing.py
  tests/test_p2_consent_provenance.py tests/test_readiness.py
  tests/test_visual_memory.py -q`: `157 passed, 7 warnings in 5.02s`
  (`real 5.25s`, `user 1.41s`, `sys 0.11s`). The warnings are the inherited
  Pillow `Image.getdata` deprecation in visual-memory tests.
- `tests/test_imports_smoke.py -q`: `315 passed in 2.11s`
  (`real 2.41s`, `user 2.14s`, `sys 0.17s`).
- `tests/test_apm_double_talk.py -q`: `6 passed in 1.02s`
  (`real 1.24s`, `user 0.93s`, `sys 0.15s`).

AST parsing for all five Python paths, scoped Ruff
`E9,F63,F7,F82`, `git diff --check`, and the exact 100-line STATUS are green.
The two new Python files are Ruff-formatted. Read-only formatter audit proves
every remaining proposed edit in the three legacy files is inherited with zero
task-line overlap; full-repo Ruff reproduces the same 200 inherited diagnostics
as base. Independent lifecycle review is GO and specifically rechecked
publication before construction/start, exact retain/reap proof, direct lease
cleanup, BUSY progression, stable factory keys, and the advertised-cancel
contract boundary.

The candidate inventory is exactly five Python paths plus six durable document
paths: `core/_hedge_source_owner.py`, `core/llm.py`, `core/llm_factory.py`,
`tests/test_hedge_source_owner.py`, `tools/testing/stages.py`, `STATUS.md`,
`.agents/backlog.md`, `docs/agent-map.md`, `docs/agent-testing.md`,
`docs/unified_architecture.md`, and this ADR.

Frozen SHA-256 values are:

- `core/_hedge_source_owner.py`:
  `30fa66b6e687c208d02557b4a9119d91c8e915b5d550c9247fc5bfa596a3c508`
- `core/llm.py`:
  `df20e9bee797f766de37e539deb3ae798718b499308e353d297ee2e81d9f0d37`
- `core/llm_factory.py`:
  `b915672ed8a328fb76f3ede683e8818cd93efbf43f23684a2cf7a9ff1d655953`
- `tests/test_hedge_source_owner.py`:
  `090c70350e1d4ed22bcb69058d0677e56416710953dad3d4108baa9872bb6d43`
- `tools/testing/stages.py`:
  `2e53ffbfa3187115f39dad97aaff8fd8a6a342188c6794e67eb025f88e86b095`

No network, provider, model, GPU, audio, microphone, device, billing-bearing, or
live path ran. The receipts prove Python lifecycle logic only; the retained
resource, cancellation, native-return, and possible billing risks above remain.
