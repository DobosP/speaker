# ADR-0224: Move the dormant web serving boundary to Go

Date: 2026-10-05
Status: accepted

## Decision

Move the optional rollback `/healthz`, `/token`, `/chat` and static web listener
from FastAPI/Uvicorn to a stdlib-only Go HTTP binary. Go owns bearer admission,
LiveKit HS256 token generation, request/reply limits, deadlines, peer-rate and
source admission, static delivery, shutdown and probes. Remove Python HTTP
serving and its required framework packages. Keep native/Python audio,
inference, action and session owners unchanged. The web facade stays dormant;
this is neither a new default audio architecture nor deployment authorization.

Keep `/chat` unavailable unless an explicit local Python interpreter/config/repo
pipe adapter is selected. That adapter accepts one bounded text document,
generates one stateless local inference reply and exits; it owns no tools,
conversation memory, mode/control plane, microphone, TTS or action authority.
Python `core.llm` remains the real inference implementation. No Python HTTP
listener or HTTP proxy is retained. Limit the Ollama endpoint to literal
loopback (normalize localhost), disable environment proxies/redirects, discard
provider diagnostics and omit ambient provider/transport credentials from the
child environment. The synthetic echo worker is a separate explicit test flag.

Allow at most one active source per serving instance, with no wait queue and a
30-second HTTP deadline. A timed-out/disconnected uncooperative source retains
BUSY until actual source return; a process source retains it through `cmd.Wait`.
Kill and wait the Python child on cancellation/oversized output. This proves
local child settlement only: a separate Ollama server computation or arbitrary
native descendant termination is not attested. The IPC and HTTP request caps
are 16 KiB; replies including JSON framing/escaping are capped at 64 KiB.
Retain the 30-per-minute chat peer budget; bound peer retention to 4,096 and
ignore forwarded headers. Return fixed errors without backend/path detail.

Disable `/token` by default. An explicit `--rollback-voice` permits only a
loopback ws/wss LiveKit URL. Refuse LAN/public/cloud token admission even if a
setup grant exists: the canonical trusted-LAN selection and owner live A/B gates
in ADR-0097/0164 remain open. A browser/room identity grants no owner or action
authority. This tighter facade policy does not change the retained Python
rollback worker or mint helper, and does not itself qualify that audio path.

Keep all Compose services behind explicit independent `rollback-web` and
`rollback-audio` profiles; ordinary Compose selects none. Give Go web its own
non-root scratch image with no Python, model mount or provider environment.
Keep a read-only Go development build, and manual-only qualification workflow
under ADR-0223. Bind loopback by default; an explicit bind-all additionally
requires configured bearer auth and does not enable audio/token admission.

## Context / why

The optional Python listener duplicated deployment dependencies despite owning
only HTTP/token/static transport and a stateless local text call. A Go reverse
proxy would retain those dependencies and would not migrate the serving owner.
The original listener buffered bodies before checking size, lacked reply/source
bounds, retained unbounded peer keys and did not apply the trusted-LAN audio
fences to token minting. Framework replacement must not activate this dormant
rollback path or create another assistant/tool plane.

Independent original-case mapping, native HTTP/image qualification, retained
Python boundary tests, measured scope and residual gaps live in
[`go_serving_boundary.md`](../go_serving_boundary.md). The exact reviewed Python
LiveKit SDK mint chain remains tested for the retained worker; Go tests verify
real JWT bytes/signature, exact one-hour claims and reviewed grant fields from
the [official pinned API 1.2.0 source](https://github.com/livekit/python-sdks/blob/api-v1.2.0/livekit-api/livekit/api/access_token.py).
Historical SDK/audio receipts remain historical and no source-hash refresh is
presented as acoustic evidence.

## Consequences

Ordinary HTTP builds/tests/CLI/probes and the scratch image require no Python
or Rust. Explicit text inference still costs one Python process per nonempty
turn and retains its backend/native dependencies. No loaded-model, live audio,
LAN, production, phone, provider or minor-data qualification follows from fake
worker and synthetic-loopback tests. The missing FastAPI/Uvicorn baseline
prevents an equivalent old/new resource comparison on this host; native-only
measurements do not authorize cost/resource-tier claims. Returning to the old
HTTP framework requires reverting this boundary change; it must not bypass the
independent audio/consent/live gates.
