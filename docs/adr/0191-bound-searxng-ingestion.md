# ADR-0191: Bound SearXNG response ingestion

Date: 2026-08-12
Status: accepted
Refines: ADR-0003, ADR-0187, ADR-0189
Supersedes: none

## Decision

Keep ADR-0003's opt-in, self-hosted SearXNG backend and ADR-0187/0189's
monotonic context, sensitivity, and raw-query egress gates. Harden only the
shipped HTTP ingestion and its configuration boundary. Preserve the original
one-argument `Backend.search(query)` protocol for injected/custom backends.

Pin `httpx==0.28.1` directly in `requirements.txt`. The shipped backend lazily
imports it only after web egress has passed every earlier gate and the feature
is enabled. Its GET uses `{base_url}/search`, exact `q` and `format=json`
parameters, `Accept: application/json`, `Accept-Encoding: identity`, and an
`httpx.Timeout` whose one configured value applies to the connect, read, write,
and pool inactivity phases. Consume the reply through the response context
manager and `iter_raw()` without a requested chunk size, so successful entry
closes response-owned resources on normal return and on header, status, read,
decode, or payload failure. An injected client's lifetime remains its caller's
responsibility.

Make `WebSearchConfig` frozen and normalize direct construction exactly as its
JSON factory does. Only exact built-in `True` enables web search; every other
value disables it. Only an exact built-in string is a base URL, and it is
stripped. A non-exact configuration mapping, a non-string key, or any failed
copy/scan yields the complete safe defaults without invoking virtual mapping,
equality, hashing, or coercion hooks. Numeric values are accepted only at the
exact built-in types described below; malformed or non-finite values use the
default, while finite in-range/out-of-range values survive/clamp as follows:

| Field | Default | Accepted exact type | Clamp |
|---|---:|---|---:|
| `timeout_s` | `4.0` | `int` or `float`, never `bool` | `0.05..15.0` |
| `total_timeout_s` | `8.0` | `int` or `float`, never `bool` | `0.05..20.0` |
| `max_response_bytes` | `262144` | `int`, never `bool` | `1024..1048576` |
| `max_results` | `5` | `int`, never `bool` | `1..8` |
| `max_title_chars` | `256` | `int`, never `bool` | `1..512` |
| `max_summary_chars` | `2048` | `int`, never `bool` | `1..4096` |
| `max_url_chars` | `2048` | `int`, never `bool` | `1..2048` |
| `max_output_chars` | `16384` | `int`, never `bool` | `1..32768` |

Before opening transport, and at every point at which synchronous control has
returned, check one monotonic `total_timeout_s` deadline and an exact
`threading.Event` instance as the cancellation token. Check after stream entry,
before and after every raw iterator step, after end-of-stream and context exit, around
UTF-8 decode and JSON parse, and while validating the consumed result prefix.
A pre-attempt cancellation/deadline has `egress=false`; once the stream factory
has been called, failures and empty results retain `egress=true`. This total
budget is cooperative: it cannot interrupt a synchronous stream-factory call,
context entry, raw read, context exit, or active JSON parse that has not
returned control.

Treat the HTTP reply as untrusted bounded input. Require exactly one
`Content-Type` header whose media type is `application/json`. Permit no
`Content-Encoding` or exactly one `identity` value. Permit no
`Content-Length` or exactly one nonempty ASCII-decimal value; reject duplicates,
malformed values, and a declared value above the configured body cap without
converting an attacker-sized decimal to an integer. Accept only exact `bytes`
raw chunks and reject a chunk before appending it when the accumulated body
would exceed the cap. Reject a UTF-8 BOM and decode strict UTF-8 only.

Parse the bounded body with duplicate-object-key rejection and rejection of
JSON `NaN`, `Infinity`, and `-Infinity`; map parser recursion overflow from
excess nesting to the stable `invalid_json` refusal. Require an exact root
object with an exact `results` list. The shipped backend validates and copies
only `results[:max_results]`: each consumed hit is an exact dict and each
`title`, `content`, and `url` value is missing/null or an exact string. A
malformed suffix outside that prefix cannot poison it. The whole bounded JSON
body is necessarily parsed before this prefix-only validation; there is no
streaming JSON parser claim.

At the provider boundary, consume no more than the configured result prefix
from either shipped or custom iterables. Admit only exact dict hits with
exact-string keys and exact-string/null title, content, and URL fields; skip
other shapes without string coercion or user hooks. Apply title, summary, and
URL character caps before constructing results/citations, and cap the joined
result text separately with `max_output_chars`. Empty/all-skipped hits retain
the successful local corpus fallback with truthful post-attempt egress.

Use detail-free failure receipts and logs. Shipped input refusals expose only a
stable code such as `cancelled`, `deadline_exceeded`, `invalid_content_type`,
`unsupported_content_encoding`, `invalid_content_length`,
`response_too_large`, `invalid_utf8`, `invalid_json`, or `invalid_payload`.
Other failures expose only the exact exception type name, never its string or
response/query/body detail. An injected/custom backend cannot forge a shipped
pre-attempt receipt by raising the internal error type; after custom invocation
its fallback is conservatively `egress=true`.

The exact shipped-backend path is recognized by exact type, not `isinstance`.
Only it receives the cooperative cancel event and configured result limit.
Custom backends, including subclasses of `SearxngBackend`, retain their legacy
one-positional-argument call shape. This compatibility seam intentionally does
not import the shipped transport guarantees into arbitrary same-process code.

## Context / why

ADR-0187 made the surrounding turn receipt a veto and hardened the provider's
post-call result normalization, while explicitly leaving parsed-response,
trusted-limit, value-size, and blocking-backend bounds open. ADR-0189 added a
retained-context/top-level-scope veto but did not change the HTTP reader. The
shipped SearXNG path still used an eager convenience GET/JSON path: a server or
misconfigured endpoint could return a large/compressed/malformed body, consume
unbounded normalization work, or leak arbitrary exception details into logs
and receipts. A single per-phase timeout did not express an end-to-end budget.

The response must therefore be bounded before decode, parsed strictly, and
normalized under independent result/field/output limits. Exact types and
detail-free failures preserve the same hostile-object posture as ADR-0187.
Context-managed raw identity streaming makes the ownership and byte-counting
boundary explicit. Cooperative checkpoints add an end-to-end refusal once
Python regains control without claiming that a synchronous networking or JSON
operation can be killed from the same thread.

Why not replace the pluggable backend protocol with transport-specific keyword
arguments: existing tests and same-process integrations may provide a simple
one-argument backend. Exact-type dispatch keeps that compatibility while
making clear that only the shipped implementation has the new HTTP contract.

Why not stream-parse JSON: the hard response-byte cap already bounds input
bytes, and SearXNG's result shape is simple. Whole-body strict parsing is a
smaller implementation, but it retains the parser-graph amplification residual
below and therefore is not described as a total memory bound.

## Consequences

- Default configuration remains corpus-only. Web access requires both exact
  built-in `enabled=true` and a nonempty exact-string `base_url`, after the
  unchanged ADR-0187/0189 context/sensitivity/raw-query gates.
- The shipped reader has hard configured/clamped response, prefix, field, and
  rendered-output caps. A declared or observed body over the cap is refused;
  malformed headers, encoding, UTF-8, JSON, payload, hit, or field shapes
  degrade to the successful corpus fallback without losing the egress fact.
- HTTP phase timeouts and the cooperative total deadline complement each
  other. Neither can preempt an already-entered synchronous factory call,
  context enter/read/exit, nor active JSON parse. An implementation requiring a
  hard wall deadline needs a separately owned interruptible process or a
  transport with a proved cancellation boundary.
- `max_response_bytes` bounds the accumulated response body only. It excludes
  DNS/TLS and HTTP parser/header allocations, proxy/client state, one raw
  transport chunk before rejection, simultaneous bytearray/bytes/decoded-text
  copies, and JSON object-graph amplification. The whole bounded JSON document
  is parsed before only the configured result prefix is validated/copied.
- Field and rendered-output limits count Python string code points, while the
  citation tuple is bounded indirectly by `max_results`. Strict UTF-8 applies
  to response bytes; JSON escapes may still produce a lone UTF-16 surrogate
  code unit that is bounded but not Unicode-scalar-normalized or guaranteed to
  encode in every downstream sink.
- The exact `threading.Event` is a trusted same-process cancellation primitive,
  not a hostile-hook boundary. Same-process monkeypatching can make `is_set()`
  fail; the provider catches that failure with a detail-free, truthful egress
  receipt, but this decision does not make arbitrary in-process mutation safe.
- Injected/custom backends are executable same-process code. They may block,
  allocate, mutate aliases, yield hostile values, ignore cancellation, or
  perform their own egress; only prefix consumption and provider-side exact
  hit/field/output normalization apply after they return/yield. Direct calls to
  `SearxngBackend` likewise bypass the registered capability's privacy gates.
- The operator-selected endpoint and environment-derived HTTP proxy remain
  trusted configuration. The code does not authenticate self-hosting, require
  loopback/TLS, pin an origin/certificate, disable environment proxy routing,
  or provide process isolation. Raw identity streaming prevents transparent
  content decoding on the accepted path; it does not make the remote endpoint,
  proxy, HTTPX, DNS, TLS, or same process trustworthy.
- The raw-query classifier remains heuristic, and ADR-0187/0189's direct/raw,
  same-process mutation, and provenance residuals remain. This decision changes
  no cloud-model, provider, audio, device, or live behavior.

## Verification

The final low-priority/no-cache command prefix was
`/usr/bin/time -p env SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19
/home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider`.
With only the named paths appended, final rebased results are:

- `tests/test_websearch.py -q`: `136 passed in 0.59s` (`real 0.77s`,
  `user 0.53s`, `sys 0.03s`);
- `tests/test_websearch.py tests/test_capability_context_isolation.py
  tests/test_react_planner.py -q`: `186 passed in 0.84s` (`real 1.01s`,
  `user 0.71s`, `sys 0.05s`);
- `tests/test_sensitivity.py tests/test_llm_egress_policy.py -q`: `108 passed
  in 0.57s` (`real 0.75s`, `user 0.61s`, `sys 0.04s`);
- `tests/test_imports_smoke.py -q`: `315 passed in 2.02s` (`real 2.31s`,
  `user 2.16s`, `sys 0.13s`); and
- `tests/test_apm_double_talk.py -q`: `6 passed in 0.81s` (`real 1.02s`,
  `user 0.88s`, `sys 0.13s`).

`config.json` parses, both changed Python files parse as ASTs, the installed
HTTPX version equals the direct pin, scoped Ruff and `git diff --check` are
green, and `core/websearch.py` is Ruff-formatted. Read-only format comparison
finds five remaining `tests/test_websearch.py` hunks, all base-identical with
zero task-line overlap; no whole-test-file format-clean claim follows. The
exact 100-line STATUS check, frozen four-file hashes, and independent rebased
code/security audit are green.

All tests use in-process HTTPX transports/raw streams, fake clocks,
`threading.Event`, fake registries/backends/classifiers, and local corpus data.
No network, SearXNG, provider, model, GPU, audio, microphone, device, or live
path ran.

The exact intended inventory is ten paths: `core/websearch.py`,
`tests/test_websearch.py`, `config.json`, `requirements.txt`, `STATUS.md`,
`.agents/backlog.md`, `docs/agent-map.md`, `docs/agent-testing.md`,
`docs/unified_architecture.md`, and this ADR. Frozen non-document SHA-256 values
are:

- `core/websearch.py`:
  `c1c8085bdf8b45ec42d656a9087f2bf14b13c875ec2e907fa4a8de05aae34223`
- `tests/test_websearch.py`:
  `b97ddee2303d9d18f9fdcbef37eb54bfe5923f5166d983f28c148b52d7d33802`
- `config.json`:
  `47c3036eb1b51d88c1e30404cb8ed3423f09764445a42988e7d968296d042c14`
  (the base-`0223a2f` value was
  `0ffe35d339c192de47801c0caa5da6932d90a436cd35fc6dd7532d2a124d6c49`; the file
  now also carries main's `tts_output_leveler` and
  `barge_word_cut_require_speaker` device-profile keys from `3173d07`, which
  this decision does not touch)
- `requirements.txt`:
  `7672bea6122a301685c609e86d43da77e2bfed800971ffad9cdc3da837f8b18d`

## Re-verification on current `main` (2026-09-07)

The decision above was written against `0223a2f` (2026-08-12) and landed only on
2026-09-07 against `main` `523f22f`. Main did not touch `core/websearch.py`,
`tests/test_websearch.py` or `requirements.txt` in between, so both frozen Python
hashes and the `requirements.txt` hash reproduce byte-for-byte on the merged
tree; only `config.json` differs, for the reason recorded above.

Re-run with `env SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider`:

- `tests/test_websearch.py -q`: `136 passed` (unchanged);
- `+ tests/test_capability_context_isolation.py tests/test_react_planner.py`:
  `186 passed` (unchanged);
- `tests/test_sensitivity.py tests/test_llm_egress_policy.py -q`: `108 passed`
  (unchanged);
- `tests/test_imports_smoke.py -q`: `322 passed` — main added seven importable
  modules since the frozen `315`;
- `tests/test_apm_double_talk.py -q`: `6 passed` (unchanged).

`config.json` still parses and `WebSearchConfig.from_dict` returns the clamped
shipped defaults. The "exact 100-line STATUS" check in the frozen receipts is
retired: `STATUS.md` is now governed by the ≤120-line budget in `AGENTS.md`.
The three agent-facing document hunks written for this decision were discarded
rather than applied, because `523f22f` rewrote `STATUS.md`, `docs/agent-map.md`
and `docs/agent-testing.md` into the fleet doc convention; equivalent content was
rewritten in that convention instead.
