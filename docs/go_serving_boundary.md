# Optional Go serving qualification

Boundary/status: [ADR-0224](adr/0224-move-dormant-web-serving-boundary-to-go.md),
[STATUS](../STATUS.md). Audio topology and its still-open live gates:
[ADR-0097](adr/0097-unify-voice-sessions-and-gate-audio-egress.md) and
[ADR-0164](adr/0164-bind-compatible-livekit-release-closure.md).

## Executable inventory

| Surface | Current executable/dependency | Retained scope |
|---|---|---|
| HTTP auth/token/chat/static | `remote/serving/cmd/speaker-web` (Go 1.27.1 stdlib) | Optional rollback facade; no audio/tool plane |
| Ordinary test/build/probe | `go test`, `go test -race`, `go vet`, `go build`; binary `--probe-url` | No Python/Rust/model/provider requirement |
| Actual HTTP/resource qualification | `remote/serving/cmd/qualify-serving` | Synthetic loopback, native ordinary path |
| Explicit text computation | `remote.text_backend` over private stdin/stdout | Python `core.llm`, one stateless local turn/process |
| Text adapter qualification | native qualifier `--chat-python`; `tests/test_remote_text_backend.py` | Explicit synthetic worker and retained Python contracts |
| Ordinary image | `docker/Dockerfile.web` runtime | Scratch, uid 10001, binary/web/config only |
| Development image/launch | same Dockerfile `dev`; `docker-compose.dev.yml` | Read-only source/assets/config; rebuild on restart |
| Optional orchestrator | `docker-compose.yml` | No services without profiles; independent web/audio rollback |
| Retained audio image/worker | `docker/Dockerfile`, `python -m remote.worker` | Existing local audio/SDK/native core; separately gated |
| Retained SDK mint helper | `remote/token_server.py:create_access_token` | Python worker token only; module starts no HTTP listener |
| Env/config example | `docker/.env.example` | Existing names only, no credential defaults; audio profile has independent gates |
| CI | `.github/workflows/go-serving.yml` | Manual-only native tests/HTTP/image/dev qualification |
| Retired HTTP tooling | FastAPI, Uvicorn, ASGI `create_app`/`app` | Removed from code and serving/worker dependency lists |

No serving jobs/automations or public deployment were created. Existing audio,
Flutter, core tests/tooling and fleet Python governance tools are outside this
bounded web port.

## Contract mapping

All eight original `tests/test_token_server.py` collected cases are accounted
for below. Counts alone do not establish equivalence.

| Original behavior | Independent current evidence |
|---|---|
| Room sanitation/defaults | Go `TestSanitizeRoomNameFromPythonCases`, original exact rows |
| Identity sanitation/defaults | Go `TestSanitizeIdentityFromPythonCases`, original exact rows |
| llama.cpp bounded thread pair | Python `test_llamacpp_text_server_uses_bounded_thread_pair` in adapter tests |
| Explicit llama.cpp batch override | Python `test_llamacpp_text_server_preserves_explicit_batch_override` in adapter tests |
| Reviewed SDK mint chain/exact TTL | Original Python SDK tests retained; Go `TestMintAccessTokenObservableSDKContractAndExactTTL` verifies real HS256/signature/issuer/subject/name/grants/time |
| TTL failure publishes no JWT | Original Python test retained; Go `TestMintAccessTokenConfigurationAndTTLFailuresPublishNoToken` |
| Generic HTTP mint error | Go `TestTokenErrorDetailFromOriginalPythonContractIsGeneric` |
| Generic HTTP inference error | Go `TestChatErrorDetailFromOriginalPythonContractIsGeneric` |
| Installed SDK one-hour/grant closure | Python `test_exact_installed_rollback_token_ttl_is_headless`; its HTTP/ASGI health half moved to Go `TestHealthIsPublicAndDoesNotConstructBackend` |

New Go cases cover real HTTP auth precedence, unset/whitespace tokens,
malformed/duplicate authorization, exact method/status/JSON shapes, explicit
loopback rollback, blocked LAN/cloud tokens, streaming and exact-limit bodies,
ASCII/Unicode/escaped text, duplicate/trailing JSON, concurrent rate/admission,
bounded peer retention, public health, backend panic/error/timeout/cancellation,
stale-source BUSY, malformed/null/oversized private output, child environment,
process kill/wait and static allowlist/confinement/byte identity/CSP/SDK SRI.
Python cases independently cover protocol/config/input/output failures, local
endpoint/proxy/redirect limits, provider/native diagnostics and exit exceptions.

Intentional tightened behavior is ADR-0224's disabled-by-default voice tokens,
loopback-only rollback, optional-only text backend, authenticated bind-all,
strict non-string/duplicate JSON refusals, bounded replies and source retention.
Empty/null/missing/non-object chat input retains the empty reply contract.
Static files are limited to the shipped three filenames; arbitrary root files,
directory listings and symlink substitutions are refused. The FastAPI
framework-binding/ASGI mechanism is retired, with its observable HTTP behavior
covered independently in Go.

## Reproduce ordinary native qualification

From `remote/serving`, with Go 1.27.1 on the executable path and build/cache
output in the task's scratch directory:

```sh
go test -count=1 ./...
go test -count=1 -race ./...
go vet ./...
CGO_ENABLED=0 go build -mod=readonly -trimpath -o "$TASK_SCRATCH/speaker-web" ./cmd/speaker-web
CGO_ENABLED=0 go build -mod=readonly -trimpath -o "$TASK_SCRATCH/qualify-serving" ./cmd/qualify-serving
"$TASK_SCRATCH/qualify-serving" --binary "$TASK_SCRATCH/speaker-web" --repo ../..
```

The qualifier launches only a short-lived synthetic loopback fixture with
synthetic auth/signing values, checks actual health/auth/token/JWT/static/error
behavior, performs 800 sequential keepalive requests (80% health, 20% shipped
index) after warm-up, reports measured Linux RSS/CPU/latency/throughput and
terminates its process. It uses no model, provider, audio device or LiveKit
server. Metrics describe that small warm workload, not production headroom.

`--probe-url http://127.0.0.1:8080/healthz` is the native health probe. It accepts
only literal loopback HTTP `/healthz`, caps response/time, and follows neither
proxies nor redirects. Health proves HTTP liveness, not backend/audio readiness.

To qualify the separately retained Python pipe, add an explicit interpreter:

```sh
"$TASK_SCRATCH/qualify-serving" --binary "$TASK_SCRATCH/speaker-web" --repo ../.. \
  --chat-python /home/dobo/work/speaker/.venv/bin/python
SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 /home/dobo/work/speaker/.venv/bin/python \
  -B -m pytest -p no:cacheprovider ../../tests/test_remote_text_backend.py -q
```

The qualifier chooses synthetic `--chat-echo`; it proves pipe assembly, not real
model behavior. Actual text inference is selected only with explicit
`--chat-python`, `--config` and `--repo-dir` paths on `speaker-web`. No HTTP
route accepts a command, interpreter path or authority field. Unknown backend,
remote Ollama endpoint, provider diagnostics, malformed/oversized reply or
ambiguous/cancelled source fails closed. A successful wait settles the local
child, not a separate model server's computation. Model throughput/RSS/cold
startup and long-lived inference composition remain unmeasured.

## Images, launch and rollback

Build the Go-only runtime with
`docker build -f docker/Dockerfile.web --target runtime -t speaker-web:qualify .`.
The manual workflow checks exported contents for Python/core/models and invokes
the image's own native probe under `--network none`, a read-only filesystem,
dropped capabilities and no new privileges. No host port is published.

`docker compose --env-file /dev/null config --services` selects nothing.
`--profile rollback-web` selects only the Go token-server; its ordinary image
has no text interpreter and its command enables no voice token minting. Read
ADR-0224 before an optional invocation. The dev overlay compiles current Go
source on restart; `/tmp` holds build caches and one executable 32 MiB
`/run/speaker` tmpfs holds the binary. All input mounts stay read-only. The
`rollback-audio` profile is a distinct retained composition and is not selected
or qualified by any web test.

The former `python -m remote.token_server`/Uvicorn entrypoints are retired.
Reverting the boundary commit restores historical HTTP code; it does not close
or override any audio/consent/owner live-A/B gate. No cutover/deployment occurred.

## Evidence and limits

Observed commands, native measurements, failed-environment receipts, image
identity and exact revision qualification are recorded in [WORKLOG](../WORKLOG.md)
and the local ignored task result. The original Python web baseline cannot be
measured on this host because FastAPI/Uvicorn are absent; neither resource
savings nor a monthly cost/tier change is claimed. Installed SDK and real
voice/LAN/model/provider/phone qualification remain independent open gates.
