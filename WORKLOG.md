# Worklog — speaker

Append-only dated history, newest first (agent-ops doc governance §2). `STATUS.md` is current truth; entries here are records and are never edited after landing.

## 2026-10-09 — explicit console and desktop-first implementation scope

Valid until: later session assembly or owner scope supersedes this receipt — then treat as history.

- Owner clarified Linux/Windows/macOS desktop core first. Mobile PCM work is preserved on
  `codex/mobile-pcm-streaming` at `dfcef686aff2ce3b69eff41803acbfbb019cb425`, unmerged and
  incompletely qualified; it is excluded from desktop integration. Its proposed ADR-0229 is
  on that branch only. Desktop TTS callback failure handling replaces the active media stage.
- Explicit console assembly keeps dependency-free echo/text operation without a learned
  ambient classifier. An enabled speech classifier missing its local model still refuses.
  This adds no owner/live-origin/tool authority. Factory/session gates: **126 passed in 4.71 s**.
- A synthetic six-text candidate-inspection microbenchmark (7 blocks, 6000 calls each) measured
  median block mean **4.519 microseconds**, max block mean **4.889 microseconds** on this Linux
  host. This is only Python cue inspection, not callback-tail, model, acoustic or whole-turn
  performance; it provides no reason for a language rewrite of this small state machine.

## 2026-10-09 — decision collector boundary review repairs

Valid until: later decision collector evidence supersedes this receipt — then treat as history.

- Root review found that empty chunks consumed list entries without increasing the character
  bound. They are now skipped; a 100,000-empty-chunk regression asserts peak traced collection
  memory stays below 128 KiB. Pre-cancelled direct compatible-API requests now raise the same
  cancellation signal before their unsupported-client refusal.
- Final focused decision/router/async-cancellation gate: **125 passed in 3.81 s**, native-free,
  using the preceding receipt's environment and task-local basetemp. No model/live claim.

## 2026-10-09 — conversational admission and priority preservation

Valid until: later source or owner acoustic evidence supersedes this headless receipt — then treat as history.

- ADR-0227 adds a typed candidate gate before ambient partial/final preemption, current rendered-question
  windows and exact acoustic cue lifetime. Controls, continuations, typed input and non-assistant modes retain
  their existing authority. Ambient memory is bounded/droppable DATA; no model or device-manager call runs
  on this early capture seam. ADR-0228's request limits are wired into addressing.
- Independent review reproduced broad-question false negatives, ambient control-queue overload, stale/auxiliary
  question windows and unavailable-model ambiguity; all were corrected before commit. The old 30-second
  partial ticket renewal was also removed, with exact unheard-input restoration and capture recovery fencing.
- Targeted command: standard single-thread headless prefix, pytest -p no:cacheprovider over conversation
  admission, addressing, acoustic lineage, post-barge, final-preprocessing cancellation, event bus, core runtime
  and playback history plus factory/replay wiring: **385 passed in 12.39 s**, including the capture-manager avoidance regression.
  Earlier combined factory/authority gate: **303 passed in 9.89 s**. These are deterministic state/ownership
  checks; no recognition/model/GPU/phone/microphone result follows. Final integration receipt will follow.

## 2026-10-09 — bounded local decision requests (ADR-0228)

Valid until: a later decision/provider contract supersedes this receipt — then treat as history.

- Added independent per-request 16-token/256-character/three-second decision bounds with
  Ollama JSON string-enum and llama.cpp GBNF, exact complete-label reduction, local-only
  factory routing and unchanged answer/model/context/residency settings. Direct compatible-
  API clients abstain rather than start a transport whose local/cancel contract is unproved.
- Capability-router disambiguation uses the helper; addressing wiring belongs to the separate
  conversational-admission branch. Format correctness is not semantic/reply quality.
- Final native-free gate: **326 passed in 8.84 s**, no audio/models/daemon/cloud. Command used
  `SPEAKER_TEST_LOG=0 SPEAKER_NO_LOCAL_CONFIG=1 SPEAKER_LIVE=0 PYTHONDONTWRITEBYTECODE=1`,
  task-local TMPDIR/basetemp and one-thread BLAS, then the main venv Python `-B -m pytest
  -p no:cacheprovider -q` on `test_llm_decision.py test_capability_router.py
  test_ollama_async_cancel.py test_llamacpp_cancel.py test_llamacpp_options.py
  test_llamacpp_thinking.py test_llamacpp_tool_chat.py test_llm_egress_policy.py
  test_multi_provider_llm.py test_pretoken_cancellation.py` under `tests/`.
- Earlier expanded attempts exposed two test-fixture incompatibilities (keyword-only Hedge
  construction and missing optional image/history arguments); corrected fixtures and final
  all-source gate above passed. Wrapped native option propagation has its own regression.
- Deadline tests cancel the existing async provider before first token and verify owned close;
  cooperative limits cannot preempt a foreign blocking iterator or non-aborting native call.
  No new timeout thread, model/default promotion, enrollment/private-recording mutation, live
  result, phone/thermal claim or microphone/doctor call follows.

## 2026-10-05 — complete addressing-decision validation

Valid until: later addressing/source/owner evidence supersedes this repair — then treat as history.

- A public 24-case current/short-rules/short-examples probe retained all 72 outputs. Every condition produced
  zero exact labels at seed 0, temperature 0, context 4096, output 64. Manual inspection found ACT-leading
  instruction recitations, and the historical first-word parser admitted 12/12 negative current-prompt cases.
  Shorter prompts lost legitimate admissions, so no replacement prompt was selected. The owned daemon stopped.
- ADR-0226 changes only complete normalized-token validation in core.addressing. Labels, bounded aliases,
  existing punctuation/case cleanup, explicit shortcuts and caller UNSURE policy remain. Lists, explanations,
  competing labels and reasoning cannot grant ACT through their first word.
- Worker 6f5501d passed `tests/test_addressing.py`: **47 passed in 0.55 s**. Adjacent `test_core_runtime.py`,
  `test_speaker_input_gate.py`, `test_final_preprocessing_cancel.py`, `test_pretoken_cancellation.py`,
  `test_ollama_async_cancel.py`, `test_cleanup.py`: **227 passed in 9.86 s**. Prefix: no test log, no local config,
  no live flag, bytecode off, one-thread BLAS, low priority, task-local tmp/basetemp. Independent source review GO.
- An original-source overlay substituted only e3e419b's historical parser in the new runtime regressions:
  all four malformed-output cases failed as expected; clean ACT passed in the same 0.26 s run. No private file,
  model, GPU, microphone or hardware path ran in those regression/proof commands. Rebase onto docs-only 8f80d77
  preserved source; docs/decision are amended into the same final commit.
- Separate public streaming JSON-string-enum diagnostic, six requests/six negatives per model: format 12/12
  for both; semantic MiniCPM 6/12 (all negatives ACT), Gemma3 12/12. These manually inspected 24 calls used
  the same seed/temperature/context/output bounds and no ambiguous UNSURE cases. Subsequent-request median
  elapsed was 181 ms vs 532 ms, eleven calls each, isolated classification only. No whole-agent latency/ranking,
  default selection or live acceptance claim. Owned daemon stopped; no model/config/source change by that probe.
- Actual exact-ACT noise misclassification and reply copying remain separate open failures. Candidate compatible,
  primary pointer historical, no promotion. Recordings/native receipts stay in the preserved private test lane.
- Owner chose stop live testing for now after the headless comparison. No parser live retry or stronger-tier
  live trial occurred. Mic/doctor/live remain stopped until explicit owner resume; candidate, historical source,
  backup and private evidence remain preserved, with no promotion or primary pointer change.

## 2026-10-05 — fresh candidate compatibility and remaining ambient activation

Valid until: new owner acoustic/addressing/reply evidence supersedes this retry — then treat as history.

- After independently reviewed e3e419b landed, the preserved candidate lane was fast-forwarded to that source
  without preparing again or altering its candidate/backup/primary lineage. `./live.sh --device desktop_gpu_4090
  --performance responsive --run-label responsive-enrolled-repaired-20261005` passed full READY and accepted
  the fresh reference against the actual current capture front end. Speaker warm-up, capture and streaming RMS
  playback started with enrolled word-cut authority preserved. Primary enrollment remains the historical reference.
- Owner reported only random room noises before unsolicited speech. A short recognized fragment was classified
  ACT and routed to assistant.answer; output again copied system instructions and made ungrounded tool claims.
  This is a separate remaining noise/addressing/reply-quality failure, not a no-thinking-template pass for behavior.
  No owner-labelled intentional question, physical STOP/talk-over or Current/Compact acceptance was obtained.
- Root stopped with Ctrl-C: exit 130, complete clean-shutdown private diagnostic bundle, owned audio defaults
  restored and temporary Ollama stopped. Actual-host inspection afterward found no active playback sink inputs;
  remaining Python processes did not identify as speaker-directory/voice-entry actors. No further mic run started.
- All recordings, embeddings, transcripts and native receipts remain local in the preserved candidate lane;
  no primary pointer change, promotion, model download or default performance change. Next work is bounded
  headless addressing/reply diagnosis before another owner trial. Candidate compatibility is verified, full live
  acceptance remains open, and the original private evidence and protected backup must be retained.

## 2026-10-05 — owner Linux live trial, isolated enrollment and MiniCPM no-thinking repair

Valid until: later owner live/model/route evidence supersedes this continuation — then treat as history.

- Owner explicitly lifted Linux microphone/doctor deferral. Source 268bf60 ran 4090 Responsive via `./live.sh`:
  full READY (including final ASR and CUDA FP16 verifier) then exit 1 after capture/calibration rejected the older
  frontend enrollment. Original route defaults restored; temporary Ollama stopped. This was not a no-capture result.
- Owner selected identity-off trial. 4090 correctly refused the conflicting identity-required policy before host
  setup (exit 2). Supported Desktop Responsive identity-off ran, then spoke internal instructions. Root stopped it
  with Ctrl-C (exit 130); private diagnostic bundle completed and route/daemon cleanup succeeded. This supplies no
  physical STOP verdict or whole-agent performance qualification. No raw audio/transcript/native bundle is committed.
- Owner then authorized fresh enrollment. Guarded `tools.prepare_enrollment` completed with an independent backup,
  empty reserved v5 candidate and regular mode-600 task config. The recording wrapper held the existing host lock
  and exact EchoRouteLease, used the printed core enrollment path with explicit 4090/Responsive selection and
  three 12-second clips, and restored the route. Exit 0; current signal-AGC/GTCRN frontend candidate created.
  Historical enrollment and primary pointer unchanged; candidate and all private evidence retained in the test lane.
- Synthetic public-text probe on installed pinned Q8 compared baseline chat/raw rendering to OpenBMB's closed
  no-thinking prefill. Baseline arithmetic emitted reasoning; prefill emitted a short correct answer without markup.
  The public voice-persona baseline exhausted 256 output tokens with reasoning; prefill finished with a 35-character
  incorrect answer, 4.2. Question addressing became exact ACT; ambient statement remained incorrectly ACT.
- ADR-0225 adds only the explicit closed-thinking prefill to the canonical/deployed template. Worker code d2cda31
  passed focused `test_setup_minicpm.py test_setup_doctor.py`: **234 passed, 2 warnings in 7.02 s**, plus inherited exit
  warning. Actual stdlib Go rendering ran with the retained Go 1.27.1 toolchain; no Go-test skip. Independent code
  review GO; diff clean. Ruff was unavailable; no install was performed. No capture/model path ran in that review.
- Root adjacent command at that code, with SPEAKER_TEST_LOG=0, SPEAKER_NO_LOCAL_CONFIG=1, SPEAKER_LIVE=0,
  PYTHONDONTWRITEBYTECODE=1, task-local TMPDIR/basetemp and the four one-thread BLAS environment settings:
  `ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider
  tests/test_addressing.py tests/test_ollama_async_cancel.py tests/test_multi_provider_llm.py
  tests/test_local_model_residency.py tests/test_llm_sanity.py -q`: **124 passed in 2.65 s**.
- Rebuilt the same alias through `tools.setup_minicpm --no-pull`, after checking the cached official blob, on a
  temporary loopback Go-template daemon. Exact identity passed; actual `OllamaLLM.generate` and `.stream` both
  produced brief markup-free but incorrect voice-persona arithmetic (4.2) and exact question labels. The initial
  arithmetic substring assertion was too loose; root inspected the public outputs and rejected that false pass.
  Ambient-statement probes also failed expected INGEST. Daemon stopped; corrected template alias retained.
  No model download, new model selection, performance default change, private-audio replay or enrollment promotion.
- Stricter public persona comparison retained all failures: MiniCPM answered worded/digit arithmetic correctly but
  the addressed variant as 4.2 (2/3); Gemma3 answered the digit variant correctly but both worded forms as six (1/3).
  The ambient statement was ACT on MiniCPM and INGEST on Gemma3. Single seed-0, temperature-0 trials with 4096
  context/256 output tokens are diagnostic only, not a general quality/ranking or whole-session speed verdict.
  All outputs were manually inspected; the comparison-owned daemon was stopped. Both tiers need actual owner testing.

## 2026-10-05 — Linux bounded headless voice preparation

Valid until: the host environment, assets, route or owner deferral changes — then treat as history.

- Independently reviewed Windows docs-only `dfa53b29fc2f2f45179a907902ced1462e2a8c7b` was fast-forwarded onto
  `main` from `0a88fb10e4fee2357108bbabda9b12d57e746630` and pushed without rewriting provenance.
  Review: GO; scoped diff clean; docs gate `files=38 dead_links=0 stale_terms=0 retired_verbs=0 orphans=0`.
  The verified-landed local integration lane/worktree and original remote Windows task branch were removed.
- At that source, from `docs/linux-voice-preparation-20261005`, the following actual-host synthetic gate passed
  **666**, with **2** inherited SWIG warnings, in **6.87 s** (plus inherited swigvarlink exit warning):

  ```sh
  env SPEAKER_TEST_LOG=0 SPEAKER_NO_LOCAL_CONFIG=1 SPEAKER_LIVE=0 PYTHONDONTWRITEBYTECODE=1 \
    TMPDIR=/home/dobo/work/_temp/docs__linux-voice-preparation-20261005 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider \
    --basetemp=/home/dobo/work/_temp/docs__linux-voice-preparation-20261005/pytest \
    tests/test_enroll.py tests/test_prepare_enrollment.py tests/test_promote_enrollment.py \
    tests/test_speaker_input_gate.py tests/test_sherpa_playback.py tests/test_barge_onset_grace.py \
    tests/test_playback_receipts.py tests/test_engine_playback_receipts.py tests/test_live_launcher.py \
    tests/test_setup_doctor.py tests/test_performance_modes.py tests/test_apm_double_talk.py -q
  ```

- Native venv CPython 3.12.3: `python -I -B -m pip check` exit 0 / no broken requirements; declared python-dotenv
  is absent, so complete install closure is not claimed. Metadata reports Sherpa 1.13.3, Faster-Whisper 1.2.1,
  CTranslate2 4.8.1, CUDA cuBLAS 12.9.2.10/cuDNN 9.24.0.43/NVRTC 12.9.86, sentencepiece 0.2.1 and httpx 0.28.1.
- Synthetic `python -B -m core --session console --llm echo --device desktop --performance current` exit 0:
  expected `You said:` reply and STOP control completion. First child took 0.345 s; wrapper had the wrong prefix
  assertion, corrected rerun passed in 0.290 s. Both skipped the private overlay, removed DATABASE_URL only in
  the child, and used fresh `_temp` logs with pruning disabled; no existing logs or persistent database were accessed.
- Read-only `resolve_check_config(..., 'desktop_gpu_4090')`, `apply_performance_mode(..., root=host)` and
  `check_sherpa_models` pass selected asset paths/metadata for all three modes and exact candidate hashes for
  Responsive/Compact. Selected final backend remains NeMo/Parakeet with Faster-Whisper. No native model load/warm.
- Sandbox service inspection was unavailable. Actual-host `probe_pipewire_state` succeeds but no echo module
  is loaded and `_check_pipewire_echo_route` refuses the selected unbound route. No route/default mutation.
  The normal loopback Ollama endpoint is stopped. A temporary installed `ollama serve` on loopback port 11435
  without the launcher's Go-template setting refuses MiniCPM identity; the ADR-0209 `OLLAMA_GO_TEMPLATE=1`
  daemon passes `check_ollama` for Gemma3 presence and pinned MiniCPM Q8 identity. Every owned daemon was stopped;
  no generation, pulls, alias repair, downloads, install or default change occurred.
- Real retained enrollment/private audio remain unopened. Preparation/promotion tests use generated private
  fixtures only. Physical STOP/talk-over, compatible isolated enrollment, CUDA FP16 warm, doctor, Current/
  Responsive/Compact A/B, LocalVQE comparison and phone gates remain open. The existing guide now has the exact
  prerequisite checklist; older testing-guide wording no longer describes Windows doctor as audio-free.
  The 11,701-test and Go migration receipts were not rerun; no source/audio behavior changed.
- Independent read-only review: GO after clarifying that schema v2 describes the preparation marker,
  while the isolated enrollment candidate is v5. Final diff/links gate clean (`files=38`, all findings zero);
  STATUS is 120 lines. No remaining review blocker in this bounded preparation/documentation scope.

## 2026-10-05 — Windows voice preparation and documentation freeze

Valid until: the destination environment, local assets or owner microphone preference changes — then treat as history.

- Existing native Windows CPython 3.10.11 executed `python -B -m core --session console --llm echo --device desktop --performance current`
  on task source based on 0a88fb10e4fee2357108bbabda9b12d57e746630: exit 0 in 1.890 s with the expected synthetic echo.
  Private config/persistent database access was excluded and new synthetic logs were isolated without pruning existing evidence.
- `python -I -B -m pip check` returned exit 0 / no broken installed requirements. This is not a complete repository pin gate:
  Sherpa is 1.13.2 rather than 1.13.3; sentencepiece/python-dotenv and bound optimized-mode assets are incomplete.
- Paul explicitly deferred microphone checks. Windows doctor opens/initializes the communications endpoint to snapshot effects, so neither
  full nor deferred-LLM doctor ran. No Windows READY, enrollment compatibility, AEC effectiveness, physical/phone pass or default change.
- Enrollment preparation/persistence/promotion still depends on POSIX private-file/owner/locking APIs absent on Windows. A future adapter
  must preserve privacy, exact lineage, atomic durability and fail-closed semantics before capture; existing enrollment/audio remain local.
- Linux retained-enrollment refresh, local LLM/echo-route readiness and owner physical STOP/talk-over/A-B remain open. Read the existing
  restart section in `docs/local_voice_performance.md`; no private receipts, machine authority or recordings are transported by this record.
- Documentation-only freeze: no new tests/runtime probes, installation, models/downloads, microphone/private audio or WSL activity.
  Only STATUS, this worklog and the existing continuation guide are committed; origin publication leaves the task branch intact.
## 2026-10-05 — coordinator dormant serving qualification

- Independent review reproduced and re-reviewed HTTP unread-body rejection and Unicode
  scalar repairs at clean3c71061. Original proofs and focused TCP/Unicode race tests passed
 13.765s; host binary/clean revision and both runtime/development image IDs, labels and UID
  matched corrected receipts. No remaining concrete blocker in this bounded web scope.
- The dormant Go facade is qualified for integration; no voice, model, provider, LAN or
  action activation occurred. Audio/core Python/native ownership and live SDK/A-B/Ollama
  computation/descendant-cleanup limits remain as documented in the serving receipt.





## 2026-10-05 — root-review HTTP refusal and Unicode repairs

Valid until: newer serving verification supersedes this corrected branch — then treat as history.

Root's independent review of93f14f3 reproduced two P2 defects after the prior
internal clean review: unauthorized responses could wait for net/http's unread
request-body drain, and Go JSON decoding silently repaired lone UTF-16 surrogate
escapes to U+FFFD. The original root proof overlay is unchanged and all three
negative proofs now pass. This receipt supersedes the earlier clean-review
conclusion; those historical records remain unchanged.

- Pre-body auth/method/rate/declared-size refusals and routes ignoring bodies
  announce Connection:close and expire reads before writing, without calling
  Body.Close (which could itself drain). Admitted chat still uses its bounded
  body reader. Seven actual TCP tests/21 wire scenarios send content-length or
  chunked headers and withhold all body bytes; full response/wire close/EOF must
  arrive within400ms with zero underlying Body.Read and zero backend calls.
  The original93f14f3 failure reproduced0.406s; corrected race slice passed1.052s.
- One bounded shared raw-JSON scalar validator runs before HTTP and IPC decoding.
  It checks every string, including nested/ignored fields and keys, rejecting
  unpaired/invalid surrogate escapes and invalid raw UTF-8. JSON structure stays
  with encoding/json. Valid pairs, literal replacement/nonBMP characters and
  escaped backslashes/quotes preserve exact text. Eight tests include compiled
  fake-private-pipe negative/positive protocol cases and HTTP no-backend refusals.
- Uncached corrected Go suite passed; race serving25.761s/CLI1.018s, vet clean.
  Root's unmodified overlay negative proofs passed0.509s. Native qualifier now
  checks the compiled listener's header-only refusal and scalar-negative cases;
  explicit fake Python pipe also checks valid pair/U+FFFD/backslash preservation.
- Python/core/audio/requirements are unchanged; no redundant full Python rerun.
  The prior11701-pass hermetic gate remains its receipt, with the exact installed
  SDK/model/voice/LAN/descendant/Ollama-termination gaps unchanged. No real data,
  provider/model/audio, remote/LAN listener, publication or authority change.
  Corrected exact-head binaries/images and final tests are in ignored TASK_RESULT.

## 2026-10-05 — share facade auth configuration admission

Valid until: later serving/auth verification supersedes this revision — then treat as history.

Final parity audit found that retaining Python whitespace stripping in the
handler also requires the CLI bind-all guard to use that same predicate.
`HasRemoteToken` now shares that configuration test; empty, information-separator
and Unicode-space tokens all refuse bind-all even with the dev no-auth flag.
The new CLI regression table and normal Go suite pass. This closes the posture
mismatch before publication; no remote listener or credentials were used.

## 2026-10-05 — final web parity and hermetic gate

Valid until: later serving verification supersedes this revision — then treat as history.

- At committed source `345e859a7214df61da2a12871d02d67f8a432f4a`, the same CI-style full command with
  `--basetemp=/var/tmp/speaker-go-full-synthetic-345e859` outside Git ancestry passed:
  `11701 passed, 40 skipped, 9 warnings in 417.14s`. No private guard or Git marker was weakened.
  No subsequent Python/requirements/core/audio sources changed; final changes preserve Go string parity and qualify its tooling.
- Final parity review independently replayed original Python synthetic cases. Go now preserves Python information-separator whitespace,
  dotted-I lower expansion before room ASCII filtering, and exact multiword bearer bytes. Panic completion is fail-closed even for nil
  panic under the legacy runtime setting. Native qualifier settles its ordinary fixture before admitting its separate pipe fixture;
  its report identifies server-only resource scope plus driver/probe/optional inference process counts. No unresolved review findings.
- After string fixes: uncached Go test passes, race serving14.592s/CLI1.013s and vet clean. Final revision Go/native/image results and
  exact command/head/binary identities are in ignored `TASK_RESULT.md`; these later checks include the added nil-panic regression.
- Exact-head345e859 local image qualification (not registry publication): runtime image ID
  `sha256:8c1cfe4374e4c2f56c10aaa80787b88fd80ff95153f7f6fcbad1fe752562c05e`, 7993155bytes;
  dev `sha256:c20a1327b99112ec245b2c199c78e8310becd4263bb7ff3518e221c2930a5c04`, 844928962bytes.
  Both OCI revision labels matched that full source SHA. Runtime exported19 entries, no Python/shell/core/models, uid10001:10001;
  own native health passed under network-none/read-only/cap-drop/no-new-privileges. Dev rebuilt read-only mounted inputs into bounded
  executable tmpfs and passed native health. Temporary containers removed; zero default Compose services, only token-server under rollback-web.
  Final revision rebuild preserves these same isolation checks with its own exact label/results in the local task result.
- Explicit installed SDK gate refused collection. This host has livekit1.1.10; livekit-api/livekit-agents/livekit-protocol are absent.
  The ADR-0164 pins remain unchanged; this web migration neither installs them nor claims installed/live closure qualification.

## 2026-10-05 — dormant Go web serving boundary (ADR-0224)

Valid until: a later serving/qualification decision supersedes these receipts — then treat as history.

- Scope: actual Go HTTP/auth/HS256/token/text transport/static listener, CLI/probe/native qualifier, Go-only runtime/dev images,
  dormant independent Compose profiles, manual CI and exact original-case mapping. Python HTTP/FastAPI/Uvicorn removed; Python/native
  audio/inference/action core and retained SDK mint helper remain. No push/merge/deploy, LAN listener, model/provider/audio session or minor data.
- Source base `5800a94`; Go 1.27.1. Official downloaded task compiler SHA256 `63d339f0da5ab53635a56f2490a7984dfe12dfcff22ad749f63edaf590168445`
  matched go.dev; later gates used the existing read-only `/mnt/data/decision-lab-runtime/kev-native/toolchain/go/bin/go`, also 1.27.1.
  `GOTOOLCHAIN=local GOENV=off`, task scratch GOCACHE/GOTMPDIR; ordinary module has no external Go dependencies.
- Final source-tree Go `test -count=1 ./...` passed; `test -count=1 -race ./...`: serving 14.579s, CLI 1.022s; `vet ./...` clean.
  41 top-level cases plus table scenarios include original known answers, JWT signatures, wire/IPC caps, stale-source BUSY, process
  cancellation/wait, no credential/proxy inheritance and static identity/confinement. The original eight Python cases are fully mapped in
  `docs/go_serving_boundary.md`; two SDK tests remain and two llama thread-pair contracts moved to the new Python worker tests.
- Python targeted SDK/adapter/session/device/entrypoint gate: `166 passed, 1 skipped in 1.28s`; adapter alone `73 passed in 0.44s`.
  Exact installed SDK opt-in gate was attempted and refused collection (`livekit-agents` distribution missing); no package install or
  claimed installed-SDK/live qualification. Scoped Ruff unavailable; six changed Python sources parse cleanly. Docs: files=37,
  dead_links=0 stale_terms=0 retired_verbs=0 orphans=0; whitespace clean; STATUS120/map59 line budgets hold.
- Initial CI-style Python full gate under task scratch: `103 failed, 11595 passed, 40 skipped, 9 warnings in 358.42s`.
  Failures are synthetic private-artifact guards under the host's `/home/dobo/work/.git` marker, the known ADR-0215 environment boundary.
  No guard/marker was changed. Framework-generated synthetic fixtures are rerun outside Git ancestry; source/reports stay in task worktree/scratch.
- Native actual HTTP qualifier passed ordinary HTTP/auth/JWT/static/error/probe contracts with Python absent from its required path;
  explicit `--chat-python /home/dobo/work/speaker/.venv/bin/python` also passed the synthetic echo-pipe assembly. No Python HTTP proxy.
  One Go HTTP process; explicit text adds one bounded Python child per nonempty admitted turn, with no wait queue. Native-server return/child
  wait is not separate Ollama computation termination or arbitrary native-descendant proof; real model costs remain unmeasured.
- Warm sequential 800-request native-only mix (80% health/20% shipped index, keepalive, concurrency1, 399360 response-body bytes):
  idle RSS9252KiB, peak12540KiB, CPU8 ticks at100Hz =0.100ms/request; p50/p95/p99 0.113/0.196/0.297ms, 7725.27 requests/s.
  Another run with separately requested pipe qualification measured ordinary HTTP 9344/12216KiB RSS, 0.0875ms CPU/request,
  0.107/0.190/0.295ms and 8433.59 requests/s. These are local small warm-loopback measurements, not production/latency/headroom claims.
  Python HTTP baseline unmeasured because FastAPI/Uvicorn are absent; no language savings, cost, container tier or model-performance claim.
- Evolving-tree Docker runtime and development builds passed. Runtime exported19 entries, uid10001:10001, no Python/shell/core/models;
  native health under `--network none`, read-only rootFS/dropped caps/no-new-privileges passed. Read-only Go dev compiled in tmpfs and
  reached private health. Both temporary containers removed; final-head image qualification is recorded in the ignored task result.
  Compose selected zero default services and only token-server under rollback-web; YAML/six shell-run steps pass. GitHub workflow unrun.
- Independent contract and privacy/lifecycle review found serialized-output overflow, exact ASCII/Unicode IPC cap regressions, null reply
  admission, provider exit diagnostic leakage and static replacement identity risk. All were repaired and independently reproduced as fixed,
  with regression cases. No unresolved code-review findings. Audio/trusted-LAN/owner A/B/phone/model qualification remains outside this gate.

## 2026-09-05 — docs refresh: STATUS.md receipts moved here verbatim

Valid until: superseded by newer ADR receipts — then treat as history.

The bullets below are the verbatim `STATUS.md` body as of `Last verified: 2026-08-21 on Linux ROG` (commit `3173d07`, ADR-0209), moved unchanged during the fleet doc-convention refresh. Each bullet's ADR citations are the authority; the condensed current state is in `STATUS.md`.

### Baselines and gate receipts (verbatim)

- Last broad non-real baseline at `dfb5163`: 7,897 low-priority host-side passes plus the bounded-read race isolated; 14 skipped, 24 model-only deselected, 11 warnings. Current receipt-v2 gates: exact-private 362 + 1 skipped; adjacent consumers 448 + 1 skipped plus the race isolated; dry tool-route 1,287; safe canonical conversation 214, public-fixture lock 214, and full public-conversation 252; phase-resource 210; AMI proxy focused 206; broad streaming 1,015 + 1 skipped + 5 deselected; APM/DTD 6. Current EdAcc safe-extraction gates: focused 121, public conversation 325, canonical conversation 214, bounded-read race 1, and APM/DTD 6. Current Smart Turn correction gates: focused 99 + 3 model-only deselected, setup/capture adjacent 173, exact CPU-model contract smoke 2, and APM/DTD 6. Current causal LiveKit gates: headless (`-m "not real_model and not slow and not backend"`) 141 passed + 5 deselected; exact source 1, exact model 1, adjacent endpoint-source admission 69, adjacent turn/Sherpa 37, and APM/DTD 6. Current atomic final-STT profile gates: integrated affected 508; adjacent authority/default 204 + 2 skipped; exact-private evidence 426 + 1 skipped; real-process CLI 1; exact real-model readiness 3 (SenseVoice decode, Parakeet decode, Faster-Whisper CUDA FP16 warm); paired command/noise focused 48 + 1 skipped, integrated 352 + 2 skipped, independent audit 90 + 1 skipped, exact corpus 1, and exact real pair 114 evaluations; production whole-turn focused 42, expanded adjacent 298 + 1 skipped, actual 13-root/367-file/699,183,389-byte SenseVoice closure, and no-audio warm PASS; the exact retained replay failed closed at its first multi-final row and published no report; APM/DTD 6. Current semantic reset-and-accumulate gates: broad non-real 8,769 + 15 skipped + 25 deselected, focused 168, expanded adjacent 373 + 4 skipped, EdAcc contracts 103, APM/DTD 6, scoped Ruff/whitespace green, two independent audits GO, and exact 24-row/three-cell replay green; owner live A/B remains pending. Current inactive logical-turn kernel: focused 16, expanded 219, broad non-real 8,785 + 15 skipped + 26 deselected with the documented bounded-read race green in isolation, APM/DTD 6, and scoped Ruff/format/whitespace green. Current opt-in raw-composition shadow: focused 88, adjacent 226, broad non-real 8,825 + 16 skipped + 25 model-only deselected (8,795 low-priority host shard plus 30 log/loop cases isolated), APM/DTD 6, scoped Ruff/whitespace green, and both semantic and exhaustive-state audits GO; the EdAcc opt-in extension has 104 focused and 173 combined replay/lifecycle passes with scoped Ruff/whitespace green, and its exact 24-row/three-cell schema-3 replay closed cleanly with five reset-cell multi-epoch commits. Current session-only identity-off gates: affected headless 339, authority/word-cut 203, APM/DTD 6, help surface and whitespace green; owner live A/B remains pending. Current continuation/resume lineage gates: focused 263 and APM/DTD 6. Earlier repeat/memory 105, adjacent 304 + 3 skipped, conversation 213, candidate/shared 464 + 1 isolated, and public command/noise 246 remain green.
- Current opt-in terminal-lineage gates: inline has 259 focused passes and exact clean `dac50c9` EdAcc schema-4 replay matching all 35/35/32 raw commits to 31/31/29 selected finals plus 4/4/3 typed input-rejection aborts; post-capture async delivery has 295 focused + 1 skipped, 522 adjacent Sherpa/APM, and 318 EdAcc/logical-turn passes. Its exact clean `ab4f4eb` schema-5 replay accepted, worker-delivered, and finished all 31/31/29 selected finals across 24 clean drains/cell, kept 4/4/3 prequeue aborts separate, had zero async health errors, and received independent implementation/evidence GO reviews (ADR-0155/0156).
- Earlier landed component baselines: streaming family 653 passed, 3 model-only deselected; structured lifecycle 270; capture-replay 71. Moonshine Tiny/Small/Medium, Nemotron, and Parakeet NeMo stay rejected (ADR-0090/0091/0099/0115).
- Post-ASR mailbox/adjacent 221; LiveKit Agents seam/manual-session 78, adjacent baseline 224; unified-session 55; playback/actor/runtime 150; public-v3 81 file/145 combined.
- Stable evidence: both conversation pairs 42/42, semantic-memory PASS, owner replay 9/9, two synthetic-delay passes (ADR-0051/0065/0067/0068/0070/0080).
- Current Hedge source-owner verification is frozen and independently GO: 44 focused owner, 256 adjacent Hedge/provider, 319 canonical cloud-stage, 157 cancellation/routing adjacent with 7 inherited Pillow warnings, 315 import, and 6 APM/DTD tests pass; `tools/run_tests.py cloud` includes the owner matrix. AST5, scoped Ruff, inherited formatter/lint parity, diff-check, and STATUS100 are green. No network/provider/model/GPU/audio/device/live path ran (ADR-0190).

### Runtime (verbatim)

- `python -m core --session` is the one public core entry; `VoiceSession` owns one injected `VoiceRuntime`, and `build_runtime` remains the sole tool/authority plane. Repeat-previous admits built-in memory items only after a process-local ordinal fence; foreign backends require a finite, strictly newer wall time. An accepted continuation rebinds only ephemeral resume-query text to its exact composed task after successful start and exact epoch/generation validation; synthetic resumes remain unverified, response-only, and excluded from user memory (ADR-0123/0154).
- Audio defaults to device-only. The exact-closure manual-RoomIO trusted-LAN wrapper passes fake-SDK and exact-installed headless isolation but stays unselectable until owner live A/B; legacy remote is rollback only (ADR-0096/0097/0164).
- `./live.sh` is the single Linux physical entry. It owns the host lock, reversible echo-control route, conditional Ollama, doctor, and aligned private pre-DSP/processed-mic/playback-reference evidence. Explicit `sense-voice` and `parakeet-faster-whisper` final-STT profiles replace one complete validated tuple in memory, reach doctor and core identically, fail closed on required recognizer/decode/verifier construction instead of degrading, and bind their safe name/digest in the private summary; no option preserves the configured path. `--no-speaker-enrollment` masks both enrollment references only in the selected local process after those profiles, records `speaker_identity_policy=off_for_session`, preserves files/model/policies, and rejects active identity-required multi-voice word-cut before host setup. It is an identity downgrade: audible speakers retain dialogue/read/web and direct-live-confirmed tool access, while the selector grants no owner trust. Production whole-turn replay binds active model paths to the selected config root before runtime construction. Its exact 24-row run at clean `78aa3fc` passed warm-up but failed closed at row 9 on two finals/records and published no report; an exploratory ASR-only sweep found 8/24 split rows at the current 0.8-second hold, while only a non-authoritative 2.4-second hold removed every split (ADR-0075/0077/0144/0146/0152).
- `./live.sh --guided-stt-capture` requires one complete final-STT profile and publishes an immutable 16-case close/far plan plus a private configuration contract before echo-route or microphone setup. Its `stt_capture_only` core purpose forces process-local identity off, keeps input/DSP/VAD/final-STT/verifier/punctuation/diagnostics, warms STT before mic/workers, and constructs no assistant/control plane, TTS, KWS, speaker gate, playback worker, or application output device. Exactly one typed live final may consume each explicit arm; abort, recovery, extra/reused lineage, timeout, stop-time terminal, config drift, or anything short of 16/16 plus one strict post-stop diagnostic manifest is red. Recognized text and recognized-text-bearing final-path exception details are suppressed from capture-purpose logs. Broad non-real has 9,094 passes + 15 skips + 25 model-only deselections; adjacent core has 494 passes + 1 skip with two dependency warnings; plan/public-launcher has 116 passes; direct capture isolation has 8; APM/DTD has 6; and independent implementation/app-readiness reviews are GO. No microphone run, STT-quality verdict, profile promotion, or default change follows (ADR-0157).
- `tools.guided_stt_pair_attestor` is the only paired qualifier for two completed guided bundles. It fixes SenseVoice-control then Parakeet/Faster-Whisper-candidate crossover on each bundle's exact ordered 16-case final-input PCM, requires equal non-final capture protocol/plan/configuration/device/input-gain and route-policy bindings while validating each profile-specific Sherpa binding, runs only the inert dry tool-route gate, rechecks both private bundles, and atomically publishes one aggregate-only no-clobber schema-v1 report. Its gate uses fail-closed no-invoke/no-open sentinels and constructs no execution-capable provider/runtime/capability/effect/audio path; the attestor never mutates the inputs. Final headless verification is 479 affected integrated passes, 235 adjacent configuration passes plus 6 APM/DTD passes, and 9,146 broad non-real passes + 15 skips + 25 deselections + 9 warnings; scoped Ruff, format, and whitespace are green, and two final independent audits are GO. Exit zero means attestation completed, not that a profile won; no live/model/GPU/device run, owner verdict, promotion, or default change exists (ADR-0158).
- Desktop MiniCPM Q8 is the local text tier; Gemma3 is complex/vision. Phone Q4 uses native XML tools, and phone thermal behavior remains unvalidated. ADR-0186/0201/0202/0203 retain their exact playback, Gemma, reply-subscription, and ASR/listening owners and Dart-only proof boundaries; ADR-0204 retains the app-root `AgentSession` and canonical-UI-isolate speech owner. ADR-0205/0206 fix and materialize the five-domain English packet. ADR-0207's exact 74,207,237-byte mobile-hybrid Zipformer remains the sole endpoint/control authority: its strictly loaded report `264f4396…e796` records 129 source-complete/endpoint-terminated evaluations and 198 reset-committed epochs (139 literal-nonempty/59 literal-empty) across five unpooled cells. ADR-0208 accepts only an offline endpoint-epoch comparison against the exact 160,626,066-byte MobileWhisper base.en INT8 tuple; its 19,574-byte mode-0600/single-link report `0ae9424d…e625` binds all 198 inherited epochs and has zero candidate app admissions. Candidate versus baseline has command targets 18/26 for both with 2 gained/2 lost, forbidden hits 0/355→7/355, command-subset WER `.6809`→`2.3404`, DEMAND WER `.9149`→`.7409`, NOTSOFAR WER `.4423`→`.4038` but CER `.2638`→`.3216` and pair agreement 4/9→1/9, isolated WER `.6154`→`.5385`, and custom overlap WER `.3964`→`.1892`; all 59 baseline literal-empty epochs become candidate literal-nonempty and normalized text agrees on 39/198. The successful v3 desktop-CPU unit used 155.771 CPU-seconds/623.803 seconds wall, direct live cgroup peak 994,983,936 bytes, zero swap/events, and at most six tasks; its systemd footer displayed `Memory peak: 2.1M` and under-observed the active cgroup. V1/v2 failed closed with no report before scratch-witness and literal-versus-normalized-empty fixes. V3 package/replay gates pass 85 focused, 150+1 skip composed, 354+1 skip adjacent, 6 APM/DTD, 5 targeted, Ruff11, scoped format2, diff and two independent audits. License/provenance for the converted candidate artifact remains unverified beyond pinned upstream MIT/checkpoint-map evidence. Neither report supplies pooled/model-quality/default/promotion/qualification, held-out/training-disjoint, latency/RTF, Flutter/native/phone/device/microphone/thermal/live, endpoint/control/tool/agent-session, network/GPU, or Romanian authority. Full Python-plane convergence, native cleanup, a disjoint owner holdout, and physical-phone validation remain open. Frozen ADR-0200 is unlanded provenance; ADR-0201 through ADR-0207 remain byte-stable (ADR-0020/0186/0201/0202/0203/0204/0205/0206/0207/0208).
- The stopped 2026-08-21 desktop-GPU smoke is failure evidence, not acceptance. ADR-0209 makes launcher-owned Ollama select the verified MiniCPM Go template and rejects any selected capability set other than exact completion-only; malformed addressing output still abstains. The 4090 profile uses the streaming-compatible RMS path instead of whole-clip output leveling, derives its first-fragment gain from the discarded post-DC warm clip under the TTS lock, grows FIFO lead on both partial underruns and dry gaps, and reports native output underflow separately. Generic word-cut now requires compatible enrolled-speaker authority while the existing novel exact STOP policy remains; pre-endpoint and nested-barge diagnostics bind a real capture span before terminal abort. Headless gates pass 796 affected, 392 adjacent, and 6 APM/DTD tests; scoped Ruff is green under the identical inherited E731/E402/F401 exclusions and the inherited formatter-debt file set is unchanged. The retained enrollment is capture-front-end incompatible, so the next 4090 start intentionally fails closed until isolated enrollment is repeated on the active route. No repaired microphone/GPU/audio-device/live A/B has run.
- Setup may enable bounded PRIVATE vault search, reminders, and exact trusted apps. Mutations require unchanged direct speech plus confirmation and stay outside planners. `CapabilityRegistry` preserves supplied contexts while observers snapshot only exact-string metadata; ReAct invalidates missing-sensitivity, non-exact, or non-string-key context before copying. `web.search` preserves absent top-level scope for canonical deterministic SEARCH/RESEARCH and direct compatibility; if scope is present, only exact `current_turn_only` may reach the still-required sensitivity/raw gates, while `local_only`/malformed scope, non-exact metadata, non-string metadata keys, or retained-key presence veto before classifier/backend. Unknown exact-string metadata remains ignored. Disabled config blocks injected backends; malformed post-call hits fall back truthfully after consuming only the configured result prefix. SearXNG timeouts are per-phase inactivity, not wall deadlines. Earlier frozen web gates pass 96 focused, 111 policy/context, 161 capability/observer, 150 ReAct/untrusted/cancellation, and 6 APM/DTD; AST6/diff-check and scoped Ruff excluding one identical inherited React F401 are green, while 37 formatter hunks are base-identical with zero task-line overlap. ADR-0189 now covers retained-context model/web egress: capability entry mints exact typed `current_turn_only` only for an absent scope key and never upgrades a present restriction or malformed value; current-turn post-ASR PRIVATE retains its US model chain, while composed/prepublished recent context, recall/profile/last-session/screen recall, procedural rules, PRIVATE local/vault findings, accepted substantive contextful cleaner rewrites, and synthetic resume query/spoken-tail prompts force monotonic `local_only`. Runtime/Supervisor mint and propagate exact-True retained provenance through reservation/continuation lineage into one task and then clear ambient state; model/web consumers treat reserved-key presence as restrictive. Generic current/post-barge text, case/spacing/ASCII-punctuation-only or context-free cleaner rewrites, explicit file/image and ambient/current images, public already-egressed web findings, and static configured prompts retain existing policy. Direct/custom/raw providers, aliases and arbitrary same-process mutation remain outside the claim; the deliberate cleaner byte/covert channel, non-hook-free composer surfaces, shared `last_source` race, untagged unexpected private-tool exception strings, blocking-backend cancellation, response/trusted-limit/value caps, heuristic raw-gate false negatives, and a dedicated file-path test remain residuals. Frozen fake/headless gates pass 262 focused changed-surface, 275 exact cloud-stage, 101 declared-adjacent, 1,196 broad policy/context/ReAct/vault, 103 resume-integration, 314 import, and 6 APM/DTD tests; ADR-0189 records exact timings and all 13 Python hashes. AST13, scoped Ruff `E9,F63,F7,F82`, diff-check, STATUS100, hash stability, and the exact 19-path inventory are green; security and architecture/adversarial reviews are GO. Managed policy rejected three documented task-owned mechanical format suggestions, so no bytes changed and no full-format-clean claim follows. No network, SearXNG, cloud provider/model, GPU, audio, microphone, device, or live path ran (ADR-0003/0060/0073/0074/0076/0187/0189).
- Factory-built Hedges now publish one exact owner before constructing/starting each source worker. Stable process-wide keys are the local main member, single-cloud compatibility member, and each exact named cloud preset across chains/rebuilds; direct wrappers instead use instance-private local/cloud-index keys. Same-key synchronous local-only calls use a no-thread lease. BUSY never waits or spawns: hedge/fallback skips that member and advances through distinct keys. Only exact worker return, cleared prompt/system/history/images/context/client/stream aliases, dead-thread proof, zero-time join, and identity-matched registry release free a key; ambiguous start or cleanup retains it. After advertised nonblocking cancel calls return, coordinator joining remains one shared 0.5-second budget, so its outer task/provider slot may retire while the durable owner keeps same-source successors busy and unrelated keys usable. This is per-key, not a global cap: raw/direct aliases and the same endpoint under different names remain bypasses. One hung owner can retain prompt/context, client/model, socket/native/SDK resources, and a possibly billable cloud request until return or process exit; a cancel callback violating its nonblocking contract can still block before the join budget, and cooperative cancel/close supplies no GIL, native-return, transport-close, billing-stop, or hard-kill guarantee. Frozen fake/headless and static receipts plus independent lifecycle review are GO; no network/provider/model/GPU/audio/device/live path ran (ADR-0021/0030/0189/0190).
- Model setup routes punctuation, SenseVoice, Parakeet, Kokoro, and KWS archive writes through one bounded exact-size extractor; POSIX publication is descriptor-relative/no-follow, the fallback rejects stable symlinks/reparse points and Windows aliases, and both atomically replace leaf links instead of writing through them. Family-aware configuration staging now preserves every complete on-disk ASR/TTS/final/verifier/KWS or singleton selection unless that exact family has a complete successful explicit replacement, while invalid families repair only as whole groups and failed best-effort Kokoro cannot publish a Piper/Kokoro hybrid. The focused combined gate passes 262 + 2 skipped, all setup/TTS tests pass 344 + 2 skipped with two existing dependency warnings, and APM/DTD passes 6; no archive download, model, GPU, audio device, runtime/default, or live path ran (ADR-0165/0167).
- Final-ASR selection may improve dialogue text, but every nontrivial offline, punctuation, recovery, custom, or verifier rewrite loses direct-live and owner action authority before typed publication; only case, surrounding ordinary spaces, and one trailing terminator remain trusted (ADR-0133). Transcript cleanup now receives only the newest four exact canonical user-tagged utterances; assistant output and every mixed/non-user channel fail closed before the cleanup prompt, while the answering model keeps its separate bounded user-plus-assistant history. Optional context failure still runs context-free cleanup, and the existing overreach/own-speech/authority guards remain. Headless gates pass 108 focused, 273 affected, and 6 APM/DTD with scoped static checks and two independent audits green; no cleaner model, microphone, device, GPU, network, latency, quality, or live path ran (ADR-0173).

### Voice reliability now implemented (verbatim)

- ADR-0185 is accepted, implemented, landed, and pushed on `main` at `b6693ab`, with frozen headless/static receipts; this is not main-live evidence. It narrowly supersedes ADR-0183's synchronous capture-owner inference/shutdown clauses plus ADR-0184's no-move clause while carrying every other authority/resource fence forward: only an available compatible enrolled/warmed receipt enables exact post-front-end KWS PCM and pinned v1.13.3 40 ms timestamps under the full capture/media/source/authority-source/route/speaking/speak/playback and gate/model/enrollment/threshold-policy/warm generations; unavailable authority retains zero PCM/no timestamp work; every one/two-word ambiguous phrase still needs independent current `ACCEPT` at the unchanged 0.10-second voiced minimum; and the default ledger remains 32,000 float32 samples/128,000 bytes/twenty normal chunks plus the 256-chunk ceiling, sample-first/oldest-whole-chunk trim, retained-tail abstention, one owned read-only phrase root, validated geometry, independent voiced PCM, and callback-time old-reference release. The new lifecycle requires exactly one unfinished may-start-or-running KWS task and worker process-wide, no successor/replacement/retry, and permit acquisition before publication/start. `_KWS_SPEAKER_INFERENCE_TIMEOUT_SEC = 0.050` sets one code-owned 50 ms total action deadline beginning before task claim and rechecked through callback admission; it is a headless fail-closed safety bound, not speaker-model/live latency calibration. Its payload is only the gate, one/two owned clips, sample rate, and exact ticket metadata—never engine, callback, expected speaker authority, or KWS control authority. Before each word a lifecycle-Condition `enter_native_step` is the timeout/stop versus native-entry ordering point: abandonment first forbids the call; admission first may finish after later abandonment only into discard. After clearing gate/clip references, the worker may publish only a raw immutable `SpeakerSimilarityBatchReceipt`/error/busy; capture alone applies deadline/threshold/status reduction, revalidates full authority, claims, and callbacks. Only the reaper releases the permit and registry slot, after worker return, Python reference cleanup, thread-death proof, and a zero-time join. Timeout/stop permanently abandons the task, no late result callbacks, stop never joins the native worker, and rebuild/start refuses before mutation until return/reap; current-epoch timeout, worker/error, and malformed/nonfinite receipt latch KWS speaker inference unavailable for that epoch, unlike stale/reject/defer/busy abstention. Exact final verification, warm-up, legacy-WAV runtime enrollment, and unchanged synchronous word-cut try-scoring share the nonblocking process permit and may first opportunistically reap a proven-returned local or cross-engine owner without waiting: busy stays `UNKNOWN`/never `VERIFIED`, cold, unavailable/unenrolled after revoking any old WAV enrollment, or abstaining respectively. A legacy/custom live final or word-cut gate missing its process-permit try seam abstains instead of invoking a blocking fallback. Idle/novel/non-ambiguous KWS creates no task; label/effect/threshold/model/config/default policy is unchanged. Headless receipts are isolated owner `14 passed in 0.19s`, isolated virtual `167 passed in 1.55s`, exact six-file lifecycle `460 passed in 4.60s`, carried ten-file authority/resource `643 passed in 5.91s`, adjacent owner/activation `290 passed, 2 warnings in 3.46s` plus the existing exit-line `swigvarlink` `DeprecationWarning`, exact complete-Sherpa `538 passed in 5.34s`, and APM/DTD `6 passed in 1.08s`. Frozen production SHA-256 is `d3a4015b306cea183ae59cbe4345d2f4be8e06545c399f7860fa6c1ec4fbfffc` for the owner, `ff168dfa636dbeb4f4ed283c76d3aff990d9168f1f1b9faadf1b6bee80d88b45` for `speaker_gate.py`, and `9669c92d458e4defb93de1bf5eb3349c454110e33ee431a65fcdbdb9e3f09dfe` for `sherpa.py`; scoped lint, AST9, Ruff AST equivalence, diff-check, exact STATUS100, zero changed-line formatter overlap, and frozen-hash adversarial review are green, while whole-nine-file lint retains exactly Sherpa's inherited 23 findings and remaining formatter debt matches `main`. Read-only static inspection of the installed CPython-3.12/Linux-x86_64 sherpa-onnx 1.13.3 wheel found `SpeakerEmbeddingExtractor.compute` bracketing C++ compute with `PyEval_SaveThread`/`PyEval_RestoreThread`; this exact-wheel/ABI evidence proves only that wrapper branch releases the GIL, and no extractor/model/compute/device/live path ran. Historical ADR-0184 receipts remain 156/613/280/522/6 and ADR-0183 receipts 607/280/516/6; they do not prove ADR-0185. Hostile finite configuration bytes, native return/cancellation/internal-thread/storage/buffer bounds, other wheel builds, owner bare-speaker A/B, device latency/quality, and live evidence remain separate and pending (ADR-0042/0072/0137/0152/0171/0172/0183/0184/0185).
- Open-speaker barge-in must work without enrollment. Generic four-novel-word cuts are identity-optional; optional multi-voice mode and own-TTS-ambiguous STOP require compatible speaker authority. The live speaker-ID model stays unallocated unless an enrollment file exists or the active no-in-app-AEC word-cut path explicitly requires identity; inactive missing models are advisory, while active paths remain strict. A session-only identity-off live test may ignore stored references without changing the next configured session, but it never weakens explicit multi-voice policy or tool authority. The pure `semantic-interruption-policy-v1` contract now classifies never-reused-token-bound scored candidate evidence as take-floor/non-interrupting/abstain across a representable threshold dead band, but has no detector, callback, runtime effect, config/default, STOP/identity/tool authority, data, or live claim. Its headless gates pass 95 focused, 422 adjacent + 3 skipped, and 6 APM/DTD; scoped Ruff/format/whitespace and two independent audits are green. The no-download `anyreach-semantic-turn-taking-q8-synthetic24-v1` contract pins a declared four-file 507,145,644-byte Q8 source closure and zero-based physical benchmark rows 36–59 (12 `start_listening`, then 12 `continue_speaking`), but the locked candidate set was not retained or provisioned, no model/tokenizer/Parquet object was acquired, no Parquet row decoded, and no runtime/score/effect/authority path ran. Its headless gate passes 148 (53 new plus 95 policy), the standard APM/DTD gate passes 6, scoped Ruff/format/whitespace is green, and independent source/security audits are GO. The metadata-only Anyreach CPU source/four-way protocol freezes 20 official CPython-3.12/Linux-x86_64 wheels (46,744,727 published bytes), exact CL/SS/SLi/CS logits, explicit ties, and an aggregate synthetic24 2-by-4 reducer; its gates pass 107 focused, 255 affected, and 6 APM/DTD, but no wheel/model/tokenizer/Parquet byte was acquired or executed and no worker/runtime/effect/default/live authority exists. The raw-confirm/KWS audit verifies that masking-canceller confirmation already decodes the application pre-AEC mic tap and its primary stream already carries configured ASR hotwords; no raw-KWS path was added. After a consumed KWS callback returns, capture now closes an open duck-confirm window before primary-ASR rotation and the next block. Its current headless gates pass 306 focused, 402 complete-Sherpa, and 6 APM/DTD; no model, microphone, audio-device, threshold, default, or live claim follows (ADR-0008/0042/0072/0137/0152/0168/0169/0170/0171). Existing KWS and duck-confirm native work is now finite after native calls return: one accepted feed permits at most 64 `decode_stream` calls plus one final readiness probe, and validated capture geometry derives the per-feed sample cap plus each confirm window's independent primary/alternate cumulative and invocation caps. Confirm start time must be finite and non-negative; each step must be finite and strictly advance. Valid-feed KWS native fault/exhaustion recreates KWS and abstains without a terminal; invalid, empty, or oversized KWS PCM becomes a controlled capture gap. Confirm structural/native failure closes and discards all window state before exact-owner whole-continuity recovery and cannot teach DTD, emit an unconfirmed result, arm retry, claim a barge, or publish a handoff; engine stop closes the same state. Its headless gates pass 139 focused, 258 adjacent, 402 complete-Sherpa, and 6 APM/DTD; scoped Ruff, changed-region format, and whitespace are green while pre-existing whole-file formatter debt is unchanged from `main`. No model, keyword, threshold, default, source selection, authority, or healthy confirmation policy changed; no microphone, device, latency, quality, or live path ran, and one already-entered native call/callback remains non-preemptible (ADR-0172). Playback-time KWS effects now require the code-pinned exact ordered shipped six-row phone-token/raw-label binding, whose row order is setup provenance rather than authenticated natural-language evidence. Accepted `stop`/`wait` hits canonicalize to the typed STOP callback outside the mutable command map only while the exact capture epoch/media/source/authority-source/route/speaking/speak/playback authority remains current and atomically claimed against playback start; a missing typed callback abstains before native work, and a live STOP claim rejects new playback admission. Own-TTS ambiguity checks every collapsed-label alias and abstains, preserving ADR-0082 and the speaker-aware word-cut fallback; idle generic KWS is unchanged, and tracker terminalization occurs only inside admitted callback delivery. Headless gates pass 349 focused, 218 adjacent, 449 complete-Sherpa, and 6 APM/DTD; scoped Ruff, new-file/changed-region format, AST, whitespace, and the 100-line STATUS check are green, and independent adversarial review is GO. No raw-KWS, speaker acceptance, model, threshold, boost, default, device, or live claim follows (ADR-0182).
- Sherpa native reads use a 300 ms/eight-frame capture-only MediaSession. Overload retires stale PCM/lineage; atomic reader context and one rational playback clock prevent time/reference drift. Priority recovery preserves fatal state; same-domain gaps preserve learned evidence, recreate KWS, atomically claim confirmation, and guard first-block effects (ADR-0088). Every admitted engine rebuild now neutralizes the eight conditionally built AEC/coherence/DTD roles after live-owner guards and re-derives them only from the current builders, so a current fail-open/disabled path cannot inherit prior APM ownership, masking, alternate ASR, delay, or detector state. Headless gates pass 46 focused + 1 missing-DTLN skip, 133 adjacent + 1 missing-DTLN skip, and 6 APM/DTD; no model, device, quality, latency, or live evidence follows (ADR-0174). Echo probe now preserves `strategy.aec_ref_delay_ms` as the configured seed and adds a scalar `aec_reference_delay` state only after bounded shutdown proves the capture writer quiesced; explicit acceptance distinguishes a measured value even when it equals the seed, while retained ownership with active AEC yields unavailable rather than racing. Its headless gates pass 89 focused + 1 missing-DTLN skip, 134 adjacent + the same skip, and 6 APM/DTD; no config write, automatic suggestion, DSP/default/authority change, physical delay, ERLE-quality, device, or live evidence follows (ADR-0175). Interrupt suite now publishes `diagnostic_outcome` as `quiet-control-candidate`/0 on any valid coupled zero cell, `no-safe-candidate`/1 on valid coupled evidence with none zero, or `inconclusive`/2 without valid coupled evidence; other errors do not erase valid evidence, every unique candidate label is listed lexically without a winner, and `live_validation_required` stays true. `error_cells` counts malformed or explicit-error rows plus one terminal error if report publication fails; valid uncoupled rows are not errors, and `matrix_partial` is true exactly when that count is positive. Report-publication failure forces `inconclusive`/2, clears `candidate_labels`, increments `error_cells`, and marks the matrix partial while descriptive valid/uncoupled counts may remain. Its backend-neutral AEC request is `configured-aec`. Headless gates pass 63 focused, 183 adjacent, and 6 APM/DTD; scoped Ruff/format/whitespace is green. Bare-speaker acceptance still requires `./live.sh` plus `python -m tools.live_audio_ab logs/runs/run-<id>.txt` against that explicit retained run log. Echo probe now gives every invoked start exactly one stop attempt across post-start setup, warm-up, body, and tail. Ordinary and post-start `SystemExit` failures return `1` with only an error object; a stop failure is secondary to an existing primary, `KeyboardInterrupt` is preserved after cleanup, and no failed lifecycle builds the normal report or runtime snapshot. Its single strict JSON emitter fails closed on invalid serialization. With active AEC, a normally returned bounded stop with retained ownership remains an exit-`0` diagnostic with `snapshot_unavailable`, not proof of native cleanup. Headless gates pass 176 focused + 1 missing-DTLN skip, 214 adjacent + the same skip, and 6 APM/DTD; no model, network, GPU, microphone, audio device, physical suite, runtime/config/default/backend change, or live result follows (ADR-0008/0175/0176/0177). The affected serialized echo-probe coherence, DTD, uncoupled, and top-level guidance keeps one-run metrics descriptive, omits headphone and low-level core-session suggestions, and binds physical acceptance only to `./live.sh` followed by `python -m tools.live_audio_ab logs/runs/run-<id>.txt` on that explicit retained run log. Successful reports now additionally carry `stimulus_id=echo-probe-spoken-text-v1:sha256:<hex>`, a domain-separated, unsigned-length-framed digest of the single frozen normalized cyclic text plan submitted to returned `engine.speak` calls; lifecycle and serialization errors remain identity-free. Interrupt-suite raw rows preserve the self-attested value without validating or reducing it. Missing or unequal identifiers are not poolable, while equality is necessary rather than sufficient and proves neither completed audio nor live acceptance. ADR-0179 gates pass 190 focused + 1 absent-DTLN-ONNX skip at `tests/test_aec_seam.py:930` in 2.07s, 214 adjacent + the same skip in 3.83s, 64 interrupt-suite in 0.14s, 284 consumer-adjacent in 1.34s, and 6 APM/DTD in 0.69s; scoped Ruff lint, both test-file format checks, whitespace, and AST are green, changed helper/report/test regions are formatted, and inherited whole-tool Ruff-format debt remains. ADR-0180 changes only the fourth stimulus to `This final sentence completes the quiet playback portion of the diagnostic.`: the default interrupt-suite `N=3` identity remains `27c744fbb14bbcb74f9256a2a4db2af25c6e7ad6c70e5ea224c3c8708a06a3c6`, while the default echo-probe `N=4` identity becomes `4366d6a869714ed767964bda1dc682b5264c109d5d3f12260f6e07276c502704` and is not poolable with the prior `be277009edac56b0473730c1e5326a1b765976386aca9f26c500eab9f00822eb` population. ADR-0180 gates pass 190 focused + 1 absent-DTLN-ONNX skip at `tests/test_aec_seam.py:930` in 2.06s, 214 adjacent + the same skip in 3.84s, 64 interrupt-suite in 0.14s, 284 consumer-adjacent in 1.36s, and 6 APM/DTD in 0.67s; scoped `/home/dobo/.local/bin/ruff` lint, both test-file format checks, AST, whitespace, and the 100-line STATUS check are green, `tests/test_interrupt_suite.py` is unchanged, and whole-tool format remains red only for inherited debt while the changed sentence region is clean. The implementation change is limited to the fourth submitted text, its synthesized stimulus and likely duration, and its derived `N=4` identity; playback control flow, wait/deadline/sleep policy, report schema and metric definitions, classification logic, exits, DSP, configuration, defaults, runtime control path, backend, and authority remain unchanged. Observed echo, ERLE, coupling, and interruption values may differ, so old and new `N=4` results remain non-poolable; no physical validation follows (ADR-0012/0180).
- ADR-0181 supersedes ADR-0180's pacing/report contract while carrying its sentence, digest, guidance, and live gate forward. Echo probe now rejects missing/malformed/raising tracked-terminal capability immediately after construction and before instrumentation/start as identity-free `probe-preflight-failed` rc1 with no stop while preserving `KeyboardInterrupt`, defensively rechecks after start, submits ordinal-only `TrackedSpeech`, waits for one typed matching `COMPLETED` or receipt-attested `INTERRUPTED` sink terminal before the next sentence and before normal stop, and fails closed with the existing identity-free rc1 lifecycle error on dropped/failed/malformed/mismatched/duplicate/missing/timed-out/submission failures after start and exactly one owned stop. Successful reports add `playback_terminal_protocol=tracked-sink-terminal-v1` and exact submitted/completed/interrupted receipt-outcome counts without sample or audible-playback proof; legacy `sentences_spoken` equals submitted only and is not audible-completion evidence. The one-second warm-up remains while the old per-sentence/final sleeps are removed, so old enqueue-paced rows are not poolable even when their exact `N=3` or `N=4` stimulus identity matches. The probe-only `playback_level` and synthesis generation/directive wrappers regain current signature parity. Headless gates pass 145 focused in 0.35s (80 echo-only in 0.28s; 65 interrupt-only in 0.14s), 111 adjacent Sherpa receipt/drain/stop in 2.20s, 312 five-file consumer-adjacent in 1.35s, and 6 APM/DTD in 0.65s. Exact scoped Ruff lint, changed-test Ruff format (`2 files already formatted`), AST parse, whitespace/diff, and the 100-line STATUS check are green; no whole-tool Ruff format claim follows. Normal `VoiceRuntime` already used tracked receipts, so this headless tool fix does not prove ordinary replies were cut; no core behavior, config/default/threshold/backend/authority, model, network, GPU, microphone, device, physical, or live evidence follows (ADR-0012/0181). Async endpoint finalization crosses a route-neutral, process-local bounded media stage with explicit capture epoch/generation, immutable owned PCM, and per-item cancellation. Overload never decodes inline; shutdown releases queued work without depending on diagnostics, and callback-owned stop defers retained audio cleanup until its capture-effect lease unwinds. Identities, entry points, tools, and model defaults stay unchanged (ADR-0107).
- The default bounded Sherpa path now runs its complete DSP/VAD/primary-ASR/playback-confirm/word-cut/endpoint/capture-loop callback state machine on one dedicated exact-thread decode owner behind the existing capture mailbox. Immutable PCM/reference snapshots, cancelable effect leases, whole-scope loss rotation, whole-role native-error recovery, owner-before-reader readiness, rollback-safe tail workers, and deferred re-entrant cleanup are headlessly covered. The inline zero-queue path remains compatibility only (ADR-0110/0111).
- Capture replay uses one synchronous decode session and binds the dedicated-owner source in its digest, but explicitly rejects and does not execute the production owner thread. It is not native-reader, mailbox-overload, effect-lease, device, WER, latency, or live evidence (ADR-0111).
- Sherpa carries immutable acoustic identity through partial, final, barge, command, and abort paths; FileReplay does so for deterministic replay. Typed stale/duplicate ingress is rejected before effects; command PCM cannot also finalize; accepted revisions continue through task, TTS, and playback (ADR-0084/0086).
- VAD owns live ASR segments and acoustic time. Pre-VAD text cannot publish;
  calibrated speech evidence gates ordinary partials/finals, while unavailable
  bounded handoffs abstain or bypass as specified (ADR-0046/0048).
- Capture recovery rebinds rate/resampling and preserves the first timed block.
  Same-domain recovery preserves evidence; changed domains relearn outside
  complete speech epochs (ADR-0043/0048).
- Owner verification is distinct from admission. Only a finite enrolled final
  match can mint owner trust; advisory, mixed, rescue, and generic rewrite paths
  cannot grant device-action authority (ADR-0027/0041/0051).
- The opt-in Linux final pair remains checksum-pinned Parakeet Unified English plus Faster-Whisper Small. Faster-Whisper stays an independent exact-consensus verifier with a one-thread explicit-profile bound, protected controls fail closed, and SenseVoice/configured defaults remain unchanged. Recorded A/B requires an explicit complete control/candidate profile pair, so ambient self-comparison and backend/artifact mixing fail closed. Its offline outcomes are closed to `unavailable`/`skipped`/`error`/`decoded`/`empty`, and verifier outcomes to `unavailable`/`skipped`/`error`/`consensus`/`empty`/`tie`/`no_quorum`/`control_guard`/`attested_control`/`empty_veto`/`empty_streaming_guard`; exact detached maps must each account for every positive terminal decision before recorded acceptance or promotion. Frozen headless gates pass 175 focused, 221 recorded-plus-route, 140 downstream plus 1 skip, 421 broad exact-private consumers plus 1 skip, and 6 APM/DTD. Independent audit is GO with 221 focused-plus-route, 82 generic selector/consensus/guided, 109 production-final/FileReplay plus 2 expected skips, 6 APM/DTD, and green inline/static probes on tool `98b9bf92…c8ad`, recorded test `de093890…00f44`, and route test `fadc282c…487dc`. No owner corpus, model, GPU, network, capture, microphone, audio device, quality, latency, or live result is claimed (ADR-0078/0080/0144/0188).
- Streaming hotwords now require explicit model context. Isolated opt-in setup pins and atomically selects a complete, self-contained English Zipformer ASR/BPE family plus digest/case policy but adds no phrases or capabilities; active local/replay/LiveKit paths fail closed and replay binds the consumed vocabulary bytes (ADR-0114).
- Smart Turn v3.2 remains opt-in and lexical remains the endpoint-detector default. Its strict caller-provided 16 kHz short-turn path follows upstream last-8-second, left-pad-before-normalize, end-aligned preprocessing; setup pins and atomically verifies one exact CPU artifact, startup contract/rate failure selects lexical once, and diagnostics share production math. Earlier owner scores remain invalid; corrected LiveKit replay is complete and diagnostic-only. The exact clean-revision EdAcc capture-loop diagnostic completed both 24-row cells: acoustic and HOLD-only Smart Turn each emitted 31 finals with the exact same 19 single/5 multi row shape. Candidate HOLD fired 34 times across five rows but repaired zero; WER was 0.5729 versus 0.5779, while endpoint-delay p95 regressed to 1,700 ms from 1,200 ms. Preserve the 10,343-byte aggregate report SHA-256 `9561b6f11ddddeef4cba04caf777f93d0eb9a128769686e2bcecc9b1ef6e04d6`. The opt-in, default-false native reset-and-accumulate path resets only a held Sherpa stream while preserving one VAD/acoustic turn, complete PCM, total-silence/rule-3 bounds, and transcript-free replay-v2 evidence; it fails startup without VAD. Its exact three-cell replay changed 19 single/5 multi rows to 20/4, repairing one split with no single-to-multi regression; WER/CER improved to 0.4874/0.3237 from acoustic 0.5779/0.3919 and HOLD-only 0.5729/0.3896, while endpoint-delay p50/p95 was 900/1,700 ms versus acoustic 800/1,200 and HOLD-only 800/1,700. Exact rows stayed four. Preserve the mode-0600, single-link, 14,995-byte report SHA-256 `98ea171dfcd9ad553604992a6e4bfb450c8340af5be57dbcb89c24cce142f5f3`. This is partial diagnostic evidence; owner bare-speaker A/B is pending, so thresholds, models, entry points, and defaults remain unchanged. The bounded ADR-0149 lifecycle kernel remains inactive; an opt-in synchronous capture-replay observer compares only native-epoch composition with the legacy raw final and emits transcript-free aggregates. Capture replay uses schema 2 over its unchanged schema-1 default; EdAcc uses conditional schema 3 over its unchanged schema-2 default and requires clean matched controls plus positive reset-cell multi-epoch parity. Its exact `92c2456` run closed all 72 row observers with zero health/parity errors: controls matched 35 raw commits each without multi-epoch evidence, while reset matched 32 including five exact multi-epoch commits. Downstream callbacks recorded 31/31/29 selected finals and, separately, 4/4/3 typed input-rejection aborts; cell-level totals equal 35/35/32 raw commits, and schema 3 remains aggregate-only. The opt-in capture schema 3/EdAcc schema 4 mode binds each raw commit in memory by exact ordered acoustic keys plus revision to one inline selected final or typed abort, erases identities on close, and exports counters/reason marginals only; its guarded workers reject schema/flag disagreement. The exact clean `dac50c9` replay matched and bound all 35/35/32 raw commits to 31/31/29 selected finals plus 4/4/3 typed input-rejection aborts with zero health errors. Preserve its mode-0600, single-link, 23,350-byte report SHA-256 `1c03d64295fe9443d057c854b034fbc73723420641088f918f242a79cb1f5230`. A separate capture schema 4/EdAcc v5 mode keeps capture synchronous, then drains each fresh accepted final stage through the real final worker on one persistent evaluator thread before observer close; it separates capture-side prequeue aborts, enforces exact callback-after-worker-item lineage, and adds only stage/helper files to its conditional closures. It does not prove simultaneous capture/finalizer concurrency, inline text parity, overflow, production stop/shutdown, dispatcher, supervisor, runtime, defaults, authority, device, latency, live behavior, or quality; the production owner path is unchanged (ADR-0135/0136/0147/0148/0149/0150/0153/0155/0156).
- Public matrix v5 has 8 tracks/19 sources/11 exclusions. One self-digested, transcript/path/identity-free lock binds four separate private schema-v2 corpora at exactly 24 cases each: HarperValleyBank, EdAcc test, paired AMI ES2004a close/far, and a 6x4 Common Voice SPS v4 projection. Source-specific materializers recompute checkout/archive/audio evidence; the CV parent receipt/corpus and every generic receipt/reference/PCM are cross-bound. The two-role canonical-strata entry reconstructs a schema-v2 fixed aggregate-only report, binds exact adapters/resolved streams and source/safe digest domains, rejects public runner injection, and exposes strict retained-report validation plus the exact published-byte digest; green remains coverage-only/non-promotional. The AMI materializer additionally publishes a private sidecar-bound, aggregate-only forced-aligned last-word endpoint proxy for its 16 isolated windows and only the Parakeet Realtime EOU/parakeet.cpp adapters; it is not acoustic ground truth and defines no threshold. EdAcc owns strict raw-header, descriptor-safe private extraction plus exact layout/source/output rebinding; canonical bounded base-256 is admitted only for discarded UID/GID, and the root README is required and bounded. Exact segment identifiers remain WAV authority; an optional terminal one-digit participant suffix projects only to the strict `conv.list` conversation-family namespace, with both recording and family split disjointness enforced. The pinned linguistic-background participant header is admitted only as exact `PARTICIPANT_ID`, without case or whitespace normalization; accent, duplicate, and row parsing stay unchanged. Caller trees remain synthetic-only. The unchanged 5,916,732,170-byte archive passed the complete no-extraction scan: publisher MD5 `146b4b8026b5d0ce9611667c708456b3`, local SHA-256 `428e5b5d678ee2dba9ec7362878324a2e5fac1f13ac7b7266cacbcade52bb61b`, layout SHA-256 `a4deda10b3246d8795304a4cb11365f3401e1acbb11fe6210ace61369b54e01b`, and 98 members/76 WAVs. After the failed conversation-family extraction and failed exact-header preflight were preserved, a new exact-source preflight and fresh production run matched metadata digest `e8d06327a595c6e7882bfddd4d9747be7242599fda7a9a09124ba41eb426bc45` and selection digest `d8b7e3e8e9dc4f43168ff96d700961261aea028e9d063d730851e53bbd25cd3b`; their corpus digests differ because the preflight remains non-production. The production extraction retained 94 files/8,220,494,044 bytes and published `production_evidence=true`: 26 output files/4,821,792 bytes, 24 cases, and 4,790,400 PCM bytes, with every retained directory mode 0700 and file mode 0600/single-link. Corpus SHA-256 is `4da392c39a0b6bd18057f63c96b4f67c0dfdaf4c14e1d2d76176cdbee0920772`; receipt SHA-256 is `3c516b6fbb8ce139498cd2e01d83a4faf825e58bf2b9abcaa52896bdab4c853f`. Preserve all retained evidence. A separate fixed EdAcc-only 2x1 wrapper pins those bytes, the exact 24-case 8/8/8 marginal, Zipformer-then-Faster-Whisper order, three-repeat 1,600-sample burst/200 ms/zero-tail geometry, sequential delegation, repeated bindings, and aggregate-only non-promotional publication; its exact private run is complete, results are recorded below, and it does not alter the canonical 2x4 gate. Common Voice remains open. The fresh no-overwrite AMI rematerialization retains 24 cases/3,802,880 PCM bytes with corpus SHA-256 `0b5e4d78847f886785a814f68055eab1f3007319c028ce92d42388b822e25772` and preparation-receipt SHA-256 `b7ef76fc528add6bf8017a3532d649102cdedd4b70001141a1f8c835dd609585`; its completed Faster-Whisper Small result is recorded below. Exact Harper validates at 24 cases/2,956,800 bytes (ADR-0098/0101/0109/0113/0120/0122/0127/0129/0130/0131/0132/0142/0143/0151).
- Endpoint catalog v1 separately pins the CC-BY-4.0 LiveKit English EoT shard without changing matrix v5 or its four-source lock. The exact private shard passed the isolated PyArrow 25 inventory at 400 rows/2 row groups, 1,250 silence spans (850 HOLD/400 EOT), 161,326,412 exact PCM16 WAV bytes, and 80,654,406 samples. Duration exact/quantized rows are 394/6 and final-gap zero/one-sample rows are 394/6; its inventory report SHA-256 is `4c6c53d6324741006fa34da7bc6ecb02069ca68e2d5efffd166661b8b2f610e5`. The hardened parent cross-validates conservative count/sum/min/max, no-censor equality, and percentile order. The exact descriptor-bound causal diagnostic completed with a mode-0600/single-link 13,966-byte aggregate report SHA-256 `ee07bfbe0f1b16a4fafee44c4f4901a3799adcdf7bffa46ad8df775ce833e9e2` and absent scratch. Full-source candidate HOLD early cuts were 57/850 across 32 rows versus no-partial acoustic fallback 93/850 across 49; candidate EOT committed 350/400, right-censored 50, and had conservative p50/p95 700/1,600 ms versus fallback 400/400 at 800 ms. Strictly-before-rule3 candidate results were HOLD 46/646 and EOT 300/332 with 32 censored and conservative p50/p95 700/1,600 ms. Output remains diagnostic-only, `real-source-offline-endpoint-counterfactual`, and not production-runtime evidence; lexical remains default. This conditional opaque-partial tradeoff supplies no threshold/model promotion, Sherpa/VAD/STT, device, latency, or live claim (ADR-0134/0136).
- The exact public command/noise lock and no-download preparer bind 57 private schema-v4 cases from official Speech Commands v0.02 and ECCC v1.2. Typed targets distinguish recall, precision, false positives, speech negatives, and silence without committing private rows or paths. The aggregate-only ASR FileReplay route binds exact corpus/config/model/source/runtime/provider; its atomic schema-v3 pair mode now resolves SenseVoice and Parakeet/Faster-Whisper from one base, prebinds both closures, requires profile-correct verifier evidence, preserves the schema-v2 configured path, and reports control mismatch as inconclusive. The exact pair completed 114 evaluations with matching streaming controls and rejected candidate promotion; neither mode supplies live VAD-owned timing or recovery authorization (ADR-0117/0118/0145).
- The isolated harness now has a receipt-bound CPU-only parakeet.cpp v0.5.0 candidate: schema v8 binds its closed source/build/model/ELF set and Speaker bridge, protocol v4 preserves typed EOU/EOB observations and first-EOU text/terminal policy, and Bubblewrap plus a verified one-CPU, 2/3-GiB, zero-swap scope exposes no network or NVIDIA nodes. Applicable scoped workers additionally expose ordered startup, first-use, and resident whole-scope cgroup memory/CPU observations; cache state is uncontrolled, and these are neither cold/warm nor acceptance evidence. It has no runtime, tool, command, or adoption authority (ADR-0119/0128).
- The quarantined schema-v9 Kyutai 1B Candle candidate is benchmark-only: one whole-buffer noncausal 16-to-24-kHz resample precedes the true 80 ms Mimi/Moshi frame loop and duplicate-first-frame LM prime; its four-by-six finite semantic heads are diagnostic-only, finalization aggregates are not applicable, and endpoint/tool/live/default authority is absent. Networkless read-only isolation requires an exact 12-GiB host-availability prelaunch guard and a verified 5/6-GiB, one-CPU, zero-swap cgroup. A real private no-import/no-load provision produced path-specific manifest SHA-256 `7fb4cff5702dd9de48738410447fa36bb809de45fcfb46bc806bf282932cf970` and artifact-set SHA-256 `7fa16c0f3f1310c0dac828d71b62463226b349799b0983344c3ffb9f94040138`; no candidate import, model/GPU allocation, inference, quality/latency result, device run, or live validation occurred. Focused contracts have 87 passes, adjacent manifest/runtime/wheel/sandbox/supervisor/evaluator contracts have 382 passes + 1 skip, and APM/DTD has 6 passes (ADR-0160).
- The opt-in conversation flow-v1 gate composes unchanged v4 semantics with exact 4x3 deterministic journeys for typed incremental turns, scripted interruption recovery, delayed read tools/follow-up, and confirmed synthetic-action stop/restart fencing. It uses production runtime/session ownership but explicitly excludes microphone/VAD wiring, STT/WER, real TTS/audio, AEC/room, physical audibility, external effects, and live latency (ADR-0112).
- Capabilities use actor-issued per-task `TurnHandle`s and five-field bindings. Task/provider/tool/playback ownership registers before start; cancel fences new children and bounded drain reports providers still exiting (ADR-0094).
- Same-ID reuse cannot consume stale terminal/TTS/playback cleanup. Tools do not retry; failed web gets one local fallback (ADR-0021/0030/0051/0086/0094).
- Terminal receipts govern spoken history with fail-closed identity and exact
  ownership. Diagnostic schema v2 binds four continuous private PCM16 tracks, a lossless f32le final-input spool, endpoint replay, and causal playback.
  An ordered private owner plan can bind one-to-one to every validated final-input receipt without inferred transcript truth; private labels export corpus schema v3 with mandatory receipt v2, which cross-binds the raw-label digest and a domain-separated exact ordered case-surface digest before aggregate final-selector/tool-route replay and close-time rechecks, while public corpora stay v2
  (ADR-0028/0029/0038/0086/0100/0108/0124/0126/0138).

### Live evidence and limits (verbatim)

- Exact physical STOP is red: `192151`/`193713` failed with enrollment on/off; v5 was rejected and route settling is unproven (ADR-0072).
- The 2026-07-16 vault run admitted six unclipped windows without capture,
  decode, finalizer, or echo-separation failure, yet recognized `vault` 0/6.
  Only post-GTCRN mic audio was retained, so the failure seam is unknown (ADR-0077).
- Private replay WER 0.00 is non-disjoint. The retained aggregate-only BPE candidate tied all six clips (streaming .20, selected 0.00) but had zero keyword attempts and no separate command/exit receipt, so it is not promotable or live evidence. Public-v3 trusts local PyArrow; its code-bound Small
  control WER 0.6685 stays rejected. These are development—not streaming, held-out, live, or adoption—results (ADR-0087).
- Common Voice SPS v4 real-archive/model compatibility is unrun. VoxPopuli's first real preparation failed closed on its superseded PCM16 assumption; aggregate inspection found the exact float32 container and a feasible 24-case selection, but no real corpus/model run exists and the slice is not command, domestic, capture/AEC, tool, live, or training-disjoint evidence (ADR-0101/0106).
- Strict sequential clean/noisy WER is .6685/.6757 for final-only Faster-Whisper Small, .6848/.6830 for Turbo, and .7283/.9076 for CPU Zipformer. Small wins this 14-source development slice but has no endpoint, live, or adoption validity. The fixed EdAcc 2x1 run at `96c36fe` completed 72 evaluations per cell over three repeats with zero final disagreements: Zipformer WER/CER .6583/.4884 versus Faster-Whisper Small .3719/.2464, a 28.64-point (about 43.5% relative) WER improvement on this slice. Its mode-0600/single-link 20,956-byte report SHA-256 is `7edcd8a6e2c8422799a94f5b852314517170959167d2821c510a73cff56ed0ba`; preserve it. The fresh `a46942c` Faster-Whisper Small CUDA/FP16 AMI run completed 72/72 evaluations with zero disagreement: overall WER/CER .3797/.2905, close .2025/.1376, and far .5570/.4434; after-complete-PCM finalization p50/p95 was 28.562/80.935 ms, but the adapter emitted no partials. Preserve its mode-0600/single-link 16,062-byte report SHA-256 `06e51d89536e19d7c2e232d7c11e9857c3bb95eab535cec79b2361ac535dfc47`. The separate bounded NOTSOFAR fixture published 18 cases from nine isolated windows, three speakers, and two far-field channels (2,309,120 PCM bytes): corpus `bdff6983416fb276c0d5060aeb7f3fa07d2ffcd4e0da38f50466d1b72a1b9c71`, receipt `45aceac93a16128d287371e3ff46f214227d8023dd355e99a79494e687ca2c0f`, selection `f83ff910971b6ee51717e6e4eefbe995c426c8044ab5a5530676fb1208683c80`. Its one-repeat Faster-Whisper Small cell completed 18/18 at WER/CER .1538/.1030, 13 exact, and .0638 RTF; one-thread Zipformer completed 18/18 at .5096/.3467, zero exact, and .1124 RTF. Preserve the mode-0600 reports `df7fadd455f9de38639ea99fd69ec057d25dc7a22700a11385dee3fe1fb7920d` and `46d8437310732336605443f38aec9a68fb4474235af36413d6cc18e5f22bf5c1`. These are development-only, non-promotional after-PCM accuracy results: the fixture has zero overlap cases and supplies no identity, capture, endpoint, live, or default evidence; timing/RTF are not conversational latency, reported peak VRAM is null, and no active interval was externally sampled (ADR-0102/0109/0143/0151/0159). The bounded PriMock57 slice admits 29,358,062 exact source bytes and separately published three locked isolated hard-WER cases (manifest/receipt `769f0bea…ace88`/`ac4f98cb…9d2b6`) plus three two-role natural-timing synthetic-mix diagnostics (`e69c8535…43b4`/`69ec7df3…f507`); all 12 derived PCM hashes are immutable lock pins, narrow imports stay lazy, and the source/output directory and terminal receipt boundaries fail closed. `tools.primock57_isolated_eval` is the sole isolated-result wrapper; a separate self-digested lock and `tools.primock57_overlap_eval` now bind only the exact v4 overlap bundle, require source-complete sequential finals, and can publish only aggregate stem WER plus the custom `two-utterance-min-order-wer-v1` diagnostic. This is not ORC-WER/tcpWER/tcORC and supplies no ordinary-WER, endpoint, latency, qualification, promotion, identity, tool, device, live, or default authority. No model, GPU, audio-device, quality, or live run occurred (ADR-0161/0162).
- Current PriMock overlap gates pass 163 focused and 250 adjacent headless tests including APM/DTD; an independent settled-byte rerun also passes the 163-test gate. The retained production v4 bundle passes the exact evaluator preflight, scoped Ruff lint/format and whitespace are green, and the final adversarial implementation audit is GO (ADR-0162). ADR-0194's separate offline strict-loader gates pass 70 focused and 155 adjacent tests; its exact-source private v5 bundle has seven two-reference/two-original-device windows, 14 leaves, and 2,416,640 f32 bytes, manifest/receipt `898e291c…da7b`/`29b5d0f1…e2ba`, with creation-time `overlap_metric=not-implemented`; type/security, materialization, and independent final audits are GO. ADR-0196's focused/adjacent fake gates pass 111/440 and static security review is GO. Its retained Faster-Whisper Small CUDA/FP16 diagnostic completed all 14/14 source-complete burst evaluations over seven windows, two anonymous channels, and one repeat: custom `two-utterance-min-order-wer-v1` is .4231 (77/182 word errors; 70 deletions, 7 substitutions, 0 insertions), anonymous channel A/B are .4505/.3956, selected-order agreement is 7/7, and exact normalized-hypothesis agreement is 1/7. Preserve the mode-0600/single-link 7,946-byte report SHA-256 `18ec9f583190752bd683f07341e8205d9b511e6823081b5173405a8fe0df45bd`. Source-complete is execution accounting, not corpus/turn coverage; this is custom after-PCM final-only evidence, not ordinary WER, ORC-WER/tcpWER/tcORC, diarization, endpoint, latency, identity, held-out quality, qualification, promotion, device/default, or live authority. The report contains no cgroup/resource observations; collected-unit and zero-compute-PID checks are separate operational evidence (ADR-0194/0196).
- The bounded Microsoft AEC component slice locks 32 cases/128 WAV payloads plus the exact 10,000-row metadata file (44,209,144 bytes) behind five explicit term acceptances and a private aggregate-only CPU replay. Final gates pass 49 fixture/lock, 83 evaluator/metrics, 138 combined with APM/DTD, and 116 adjacent with 2 skips; after binding the compatible four-package remote closure, the 53-file/1,784,000-byte evaluator closure is `894fe05a7008755e260ecf68f4f8d8022b8ec02cb3547de67cc8aa3ccfd0dc54`, scoped Ruff/format/whitespace are green, and independent settled-byte review is GO. No source WAV, official fixture, exact-1.1.14 production replay, model, GPU, microphone, device, quality, latency, or live run occurred; the installed LiveKit 1.1.10 exact-version rejection and one all-zero-frame non-production APM smoke confer no evidence. Voice/runtime/tool/default authority and public matrix v5 remain unchanged (ADR-0163/0164).
- Parakeet clean/noisy WER is .6576/.5429; paced clean first/stable partial p50 is 1.692/5.612 s with two misses/120 ms backlog. Native EOU ended 85/126 noisy evaluations early and one source disagreed across repeats, so it remains rejected after-PCM evidence (ADR-0099/0109).
- Exact Moonshine 0.1.0 is benchmark-only. Medium's matched legacy burst WER/CER/RTF is .6685/.6342/.2971. Its stock external-presegmented threshold-zero profile is now reproducible but rejected: WER/CER/RTF .8750/.8373/.9950, 10/14 nonempty finals, finalization p50/p95 .608/4.907 s, seven missing stable partials, and 1,635 MiB reported RSS. A stock internal/external causal pair is invalid because internal VAD leaks recurrent state and can drop segment tails. No command, Speaker endpoint, or live evidence exists (ADR-0090/0115/0116).
- On the isolated 57-case command/noise gate, Faster-Whisper Small reached 18/20 clean commands, 1/6 noisy ECCC commands, and 11/21 ECCC false activations; one-second-context Zipformer reached 16/20, 1/6, and 0/21. These remain distinct development comparators, not endpoint/latency evidence (ADR-0117).
- The atomic same-PCM profile pair reproduced identical streaming controls. SenseVoice selected 17/26 commands with 1/31 monitored negative false activations, 0.8095 target precision, and 0.8936 WER. Parakeet/Faster-Whisper improved clean GSC from 16/20 at 0.30 WER to 18/20 at 0.10, but noisy ECCC stayed 1/6 at 1.3333 WER while false activations rose from 1/21 to 5/21; overall precision fell to 0.7308. The candidate remains rejected and SenseVoice remains the live control; no live validation or default change occurred (ADR-0118/0145).
- On the same 57-case command/noise corpus, the selected parakeet.cpp tail-8,000 cell completed 171 burst evaluations over three repeats with zero final disagreement: WER/CER .5745/.5424, 63/141 exact matches, .8077 command recall, .2581 negative final-case FPR, 24 accepted first-EOU endpoints, 147 tail exhaustions, .4826 source-audio RTF, .3315 model-input RTF, and 306.027 MiB maximum reported RSS. Matched paced replay preserved exact accuracy with .5592/.3841 RTF, first-partial p50/p95 772.961/1,104.004 ms, six deadline misses, 30.43 ms maximum backlog, and 306.387 MiB maximum reported RSS. This remains after-PCM development evidence, not capture/VAD/AEC/device/live/tool/default authority (ADR-0119).
- On the locked AMI source alone, the same candidate completed 72 burst evaluations with zero disagreement and WER/CER .4684/.4113, but close/far WER split .2278/.7089 and only 3/12 far cases produced nonempty finals per repeat. Paced accuracy matched exactly; source/model-input RTF were .4367/.3677, first-partial p50/p95 622.543/2,369.120 ms, ten stable partials were missing, two deadlines missed, and maximum reported RSS was 308.105 MiB. NVIDIA discloses AMI in training/evaluation without exact membership, so disjointness is unproved; this single-source direct run is not four-source fixture, live, or promotion evidence (ADR-0121).
- On the locked Harper caller source alone, the same candidate completed 72 burst evaluations with zero disagreement and WER/CER .1562/.0677; paced accuracy matched exactly at 15/24 exact and 23/24 nonempty finals. Sixteen cases accepted a post-source EOU and eight exhausted the 500 ms tail; paced source/model-input RTF were .4736/.3886, first-partial p50/p95 615.498/788.595 ms, three stable partials were missing, four deadlines missed, and maximum reported RSS was 304.012 MiB. Harper is absent from NVIDIA's disclosed dataset list, which does not prove disjointness; this clean single-channel direct run is not four-source, live, or promotion evidence (ADR-0122).
- Nemotron burst WER/CER/RTF: LA0 .7065/.6606/.4331; LA3 .7065/.6592/.1395;
  LA6 .6957/.6551/.0859; LA13 .6902/.6467/.0556. LA6 real-time was
  .6957 WER/.0992 RTF, first-partial p50 1.666 s, one miss, 10.439 ms backlog.
  Every mode is rejected; this excludes capture/VAD/endpoint/AEC/live validity
  and leaves `.venv`, defaults, and `./live.sh` unchanged (ADR-0091).
- Corrected legacy mic-only AMI remains poor recognition at .852 overall including order-ambiguous overlap (ADR-0092). The exact natural-turn fixture privately materialized four 128,000-sample cases from two manual-DA windows projected to close/far channels.
  Strict reopening binds 2,048,000 PCM bytes plus manifest `378ebdcd…24dd`, private metadata `7d69f72e…77fe`, terminal receipt `e11ab9cf…9725`, lock recipe `9fe99f0b…fa81`, and preparer closure `336abd92…ac85`; root/leaves remain 0700/0600, current-owner, single-link files.
  Its model diagnostic, launched through systemd with requested one-CPU/4-GiB limits after the RAM/idle-GPU preflight, completed 12/12 source-complete evaluations: 9 typed finals, 3 input-rejected aborts, 9/9 verifier decodes, transition WER .9167, and custom minimum-order overlap WER .7500; ordinary overlap WER remains null. The mode-0600/single-link 5,687-byte aggregate report SHA-256 is `ca1625423f168378eb01ea941b208a2422565eeac14a3669a5ce5dfaf544d8d3`.
  Wall/audio was 1.0046 over 96 s, peak RSS 1,973.219 MiB, and reported VRAM null. Final gates remain 193 focused, 251 adjacent, and 6 APM/DTD with independent audits GO; metrics are descriptive only and provide no device, live/conversational-latency, quality, promotion, or default authority (ADR-0166).
- Windows communications capture verifies OS AEC/NS/beamforming; owner physical talk-over/STOP tests remain (ADR-0081/0082).
- Diagnostic replay fails closed on malformed heartbeats/silent playback references; GitHub raw-audio PR refs remain owner follow-up (ADR-0083).
- The exact LiveKit 1.1.14/Agents 1.6.8/API 1.2.0/protocol 1.1.21 closure is source-audited, pinned, and headlessly installed. A private Linux-x86_64/CPython-3.12 PEP-751 resolution selected 72 wheels / 122,299,933 bytes (`11acbdea…cc4a` lock; measured lock `514b0102…d41c`); strict archive verification covered 5,642 members and 5,379 release files / 296,914,949 bytes. A fresh no-pip venv reproduced exactly 72 distributions from that wheelhouse, passed both dependency checkers, and loaded the reviewed SDK/token/app surfaces with networking disabled. The same verified runtime plus five pinned pytest-only distributions passed the opt-in real-SDK manual-RoomIO/AgentSession, one-frame tracked speech, one-hour token, and in-process health contract 2/2 and the combined real/fake SDK gate 80/80 in networkless Bubblewrap; the managed seccomp sandbox itself denies the RTC native wakeup. No server connection, hosted inference, model, GPU, audio device, real credential, or live run occurred, and trusted-LAN remains unselectable until owner live A/B (ADR-0164).
- The opt-in exact-input tool-route gate uses receipt-bound schema-v3 tags and one selected terminal final to score the production deterministic analyzer/planner/device matcher without invocation; its explicit inert profile is digest-bound and reports aggregates only. No owner-labelled run, provider/tool execution, device, model, or live authority exists (ADR-0108/0124/0125/0126).

### Next as of 2026-08-21 (verbatim)

- Run the two guided `./live.sh` profiles through the owner's close/far plan, retain both private bundles, and accept a narrow profile verdict only from fresh paired attestation plus owner review. Add disjoint command/multi-voice strata, native-reader/gap, bare-speaker barge grading, and live latency/natural-conversation validation before any default change. Only after all five Microsoft AEC terms are explicitly accepted, materialize its private official fixture, provision exact LiveKit 1.1.14 alone, and run the bounded component replay. Preserve the exact remote closure and run its self-hosted owner live A/B; it remains unselectable until that succeeds. Kyutai stays quarantined until its bounded resource gate. Keep PriMock overlap aggregate-only and diagnostic-only, and use the completed AMI natural-turn report only to guide new tests. Preserve ADR-0207/0208's exact packet, both model provisions, reports, scratch, package receipts, and every failed attempt unchanged. Treat both retained mobile-ASR reports as descriptive config-faithful desktop CPU evidence only; do not pool, select, qualify, promote, wire the candidate, change defaults, or infer latency/device/live behavior. The next mobile ASR evidence is a new private owner-recorded holdout kept disjoint between tuning and verdict, followed by actual Flutter/native phone CPU/RSS/thermal/microphone/lifecycle/live validation. The current Zipformer stays the English endpoint/control owner. Control normalization retains only ASCII `a-z` plus spaces; defer Romanian until a separate decision pins a multilingual model/export, language/tokenizer/Unicode contract, metrics, compatible public strata, owner recordings, and physical-phone gates. Continue Common Voice term/token acceptance and publisher/mobile/remote work separately (ADR-0092/0099/0100/0109/0114/0121/0122/0129/0134/0136/0144/0147/0148/0149/0150/0151/0153/0154/0155/0156/0157/0158/0159/0160/0161/0162/0163/0164/0165/0166/0203/0205/0206/0207/0208).


Valid until: a newer verification run supersedes these receipts — then treat as history.

## Verification record (2026-09-07)
| Check | Command | Result |
|---|---|---|
| Docs | `python3 ~/work/agent-ops/scripts/check_docs.py .` | `files=34 dead_links=0 stale_terms=0 retired_verbs=0 orphans=0` |
| APM/DTD | `~/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q` | `6 passed` |
| Staged runner | `~/work/speaker/.venv/bin/python tools/run_tests.py list` | 11 stages, `core` through `full` |
| Entry points | `-m tools.doctor --help`; `-m core --help`; `-m tools.session_bootstrap` | all exit 0; flags per `AGENTS.md` |
| Doc parsers | `~/work/speaker/.venv/bin/python -m pytest tests/test_session_bootstrap.py tests/test_golden_contract.py tests/test_diagnose_run.py -q` | `100 passed`; the `Read this when:` + `## Contents` header added to `.agents/backlog.md`, `docs/target_architecture.md` and `docs/public_voice_evaluation_matrix.md` leaves `open_p0` at the same 3 items |
| Bounded SearXNG ingestion | `pytest tests/test_websearch.py -q`; `+ test_capability_context_isolation test_react_planner`; `test_sensitivity + test_llm_egress_policy` | `136 passed` / `186 passed` / `108 passed` (ADR-0191) |
| Capability exception sanitization | `pytest tests/test_capability_exception_sanitization.py tests/test_capability_context_isolation.py tests/test_failure_cascades.py -q`; adjacent 7-file set | `30 passed` / `262 passed` (ADR-0192) |
| Cloud stage / task set / imports | `tools/run_tests.py cloud`; ADR-0192's 6-file task set; `tests/test_imports_smoke.py` | `397` / `89` / `322 passed` |
| Whitespace | `git diff --check` | no output |
| Budgets | `wc -l` AGENTS/STATUS/agent-map/agent-testing; CLAUDE.md non-blank | 79 / 120 / 59 / 52; 5 non-blank (STATUS budget 120) |
| Not re-run | broad non-real suite, `real_model`, `live` | last receipts 2026-08-21, verbatim in `WORKLOG.md` |

Valid until: a newer run supersedes these receipts — then treat as history.

## Local voice review and repair verification — 2026-10-04

Four separate behavior repairs: rendered playback onset ownership (ADR-0210), one mobile TTS
lookahead (ADR-0211), bounded capture coalescing with original endpoint boundaries (ADR-0212),
and media-first warming on the existing worker (ADR-0213). ADR-0214 records the native-media
architecture/rewrite criterion; ADR-0215 repairs existing prospective evaluator gate debt.

Python validation used CPython 3.12.3 and the shared speaker .venv with:
`SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19 ~/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests -q --ignore=tests/test_livekit_audio.py --ignore=tests/test_livekit_engine.py --basetemp=<fresh-private-framework-fixture-root>`.
The final full gate passed **11268**, skipped **42**, and emitted **9** dependency warnings in
**429.71 seconds**. The two excluded files match CI. Skips/model absence do not establish native
model quality or a physical route. Warnings are Sherpa SWIG and inherited Pillow deprecations.

The first broad run using default /tmp failed **89**, passed **11179**, skipped **42**, with
**9** warnings in **524.07 seconds**. Paired pristine-dab5e15/current tests proved two stale eager
import lists and public repository lock permission assumptions; ADR-0215's focused six and
complete AMI/Microsoft/Parakeet three-module gate passed **6**/**187** after repair. Default
framework fixtures also encountered /tmp/.git and /home/dobo/work/.git markers. The unchanged
Anyreach file passed **36/36 on each tree** using fresh private hermetic fixtures outside those
markers. Privacy/source/model guards and old locks/receipts were not relaxed. Completed
synthetic framework fixtures were moved into canonical _temp scratch for cleanup; task source,
reports and commits stayed in the prescribed worktrees.

Focused desktop audio gate passed **479**, skipped **1** missing DTLN model; final callback-slot
retest passed **60**, skipped **1**. Independent diagnostic observation and all-Sherpa sets
passed **7**/**238**. These use synthetic PCM and no microphone or speaker. The startup
readiness/runtime set passed **70**. The APM/double-talk six cases are included in the audio
and full gates. Scoped Ruff used `~/.local/bin/ruff check --no-cache --select E9,F63,F7,F82`
on changed Python implementation/tests; it and git diff --check were clean.

Mobile used `/home/dobo/flutter/bin/flutter` **3.44.2** and Dart **3.12.2**. The initial TTS
full suite passed **243**; the combined TTS/ASR suite passed **252** in **7 seconds** of reported
test execution. Independent ASR owner **52** and TTS/path **26** cases passed. Full
`flutter analyze` reported **No issues found**. These are Dart/widget/adapter checks, not native
plugin, model, microphone, physical-phone, audibility or thermal validation.

`python3 ~/work/agent-ops/scripts/check_docs.py .` returned
`files=34 dead_links=0 stale_terms=0 retired_verbs=0 orphans=0`. Compact-file budgets are
AGENTS79 / STATUS120 / agent-map59 / agent-testing52. `tools.doctor --defer-ollama` returned
**BASE NOT READY**: no visible device for the selected CUDA verifier, input/output query
failures and unavailable pactl echo-route inspection. No route changed and no repaired live
A/B, offline physical-device acceptance, WER/CER, end-to-end acoustic latency, CPU/RSS/PSS,
battery or thermal result is claimed.

Valid until: a new model/runtime/profile or disjoint physical-device run supersedes these receipts — then treat as history.

## English model research and retained-recording trials — 2026-10-04

Scope: English only, explicitly selected by the owner. Preserve all source recordings/labels,
original model/runtime locks, historical rejections, active config and production venv. New
model assets and native Moonshine runtime live under ignored pretrained_models/sherpa/benchmarks/
english-2026-10-04. Detailed aggregate reports, diagnostics, source snapshots and the public
Kitten sample are retained under ignored logs/runs/english-model-benchmarks-20261004.

Inputs:37 scripted owner references/101.1 seconds; five-item subset is duplicate and excluded.
Separately6 older hash-pinned microphone clips/12.9 seconds; no37/6 WER pooling. The owner
manifest cannot distinguish real capture from simulation;31 session turn JSONs have no usable
reference text. No transcript was invented as truth or copied into committed evidence.

ASR:8 model choices,3 repeats, CPU, requested two native threads/two-CPU caller mask; final
Moonshine cells explicitly use MOONSHINE_ORT_SINGLE_THREAD=1 and sample native-thread masks.
Initial6 ordinary cells completed111 calls each; stock Moonshine cells failed. Archived
asr-owner-stock.json and source eebf3085... are diagnostic; streaming rows lacked the official
flush and are excluded from the final table. Corrected owner four-cell run (FP32/INT8 streaming
and two Moonshine sizes) and fresh legacy eight-cell run completed all37 and6 cases,3 repeats.
Corrected Zipformer appends10560float32 zeros before input_finished; original PCM/files unchanged.
The summary binds source-report hashes and does not pretend the derived owner table is atomic.

CPU owner WER:FP32/INT8 Zipformer19.81%/19.81%, SenseVoice14.15%, Parakeet Unified8.02%,
Parakeet TDTv3 7.55%, Faster-Whisper Small10.85%, Moonshine Tiny/Small25.00%/16.51%.
Mic6 WER:20%/24%,0%,0%,0%,0%,40%/28%, respectively. INT8 saves roughly54% process RSS on
owner vsFP32 but adds one word error on Mic6. Current SenseVoice is the fastest final model;
Faster-Whisper CPU Small has RTF>1. No default promotion follows these development scores.

TTS:4 sequential native CPU models, speaker0/speed1,3 repeats of37 phrases (111 calls/model),
no output DSP or audio device. p50first nonzero callback:Kokoro2325ms, VITS134ms, Kitten693ms,
Supertonic1950ms. Isolated process peaks470.4/253.1/193.3/296.9MiB. All waveforms finite/nonzero; Kitten4/5,548,840samples at/above full scale, others0. These are callback/synthesis metrics, not subjective quality or audibility.
Supertonic OpenRAIL-M weight license is retained separately from bundled MIT exporter license;
upstream is archived. The two installed Kokoro aliases have identical relevant asset hashes.

LLM:two GGUFs on llama-cpp-python0.3.33, CPU/GPU layers0, context1024, maxreply64, greedy,
thinking requestedfalse through each template; BOS handling follows formatter.added_special.
Eight private reference prompts plus four public canaries,3 repeats (36 generations/model).
MiniCPM1BQ8 p50visible text377ms/completion1328ms/peak1285.1MiB; newer2BQ4 p50visible895ms/
completion2717ms/peak2601.5MiB. Canonical model files resolved explicitly after a rejected
symlinked storage path. Original canary formatting mismatched open question wording; counts
are not intelligence rankings. Separate explicitly formatted public-only run:12 generations
permodel, both9/12strict cases; no tool/memory/factual-dialogue promotion. Source snapshot retained.

VAD:installedv4-era vs canonical6.2weights packaged in6.2.3. Both real constructors and111clip
calls pass. The Python1.13.3 neg_threshold setter is unavailable; original load-failure receipt
retained and compatible retry binds native-default-unsettable. FrameAPI p50.173/.144ms,
process peaks75.6/79.9MiB, segments123/114 acrossrepeats. No onset/frame/endpoint/speaker labels,
so no VAD quality verdict. Native576context/residual tail coverage limitations are explicit.

Benchmark contracts:208native-free tests passed; shared archive extraction51cases additionally
passed in the TTS gate. Scoped Ruff/format/diff and docs checks were clean at report preparation.
No physical capture/playback, live echo, GPU speed, phone thermals/battery, new enrollment,
repaired bare-speaker A/B or fully offline physical-device acceptance ran.

CLI entry points:python -m tools.english_asr_benchmark; english_tts_benchmark;
english_llm_benchmark; english_vad_benchmark. Use each --help for exact manifest/config/model/
scratch/output parameters. Runtime thread fairness, private scalar validation, native output
suppression, abnormal-exit group cleanup and immutable input bindings are covered by tests.
Moonshine stock diagnostic:117threads/32CPUunion versus caller2;115masks outsidebudget,
180CPU-second termination after14completed clips. Fresh selected-ordinal stock call1193ms/
18.87CPU seconds, then closure hitlimit; explicitORT1 call517ms/.513CPU seconds,2threads,
2CPUs and cleanclose. These are resource diagnostics, not formal corpus rankings. Prior
0.1.0 rejects unchanged; installed/new export identities and licenses are in ADR-0216/report.


Valid until: the model/runtime/profile or fresh disjoint physical evidence changes — then treat as history.

## 2026-10-04 — explicit Kitten runtime integration and combined gate (ADR-0217)

Kitten Nano 0.8 INT8 now has an explicit `tts_backend="kitten"` production factory
path, with bounded metadata/asset/voice-table preflight and setup/readiness preservation.
Legacy VITS/Kokoro selection remains the empty-selector behavior. The ignored model-cache
overlay binds speaker 0 locked, speed 1, two TTS threads and shared CPU provider; active
config.local.json and the production venv were not altered.

Actual core `build_tts(SherpaConfig(...))` construction succeeded in Sherpa 1.13.3 and
synthesized a public phrase without a microphone/playback device. Retained local sample:
logs/runs/english-model-benchmarks-20261004/kitten-public-sample.wav, mono 24 kHz,
6.566 seconds, finite PCM. No subjective quality, audibility or phone result is claimed.
Known loader conditions are guarded; arbitrary graph/native failures are not certified.

Focused runtime/setup/readiness gates: **262 passed, 2 optional-model skips**. Combined
CI-style full gate on the integrated benchmark/Kitten sources: **11543 passed, 42 skipped,
9 warnings in 559.12 seconds**, using the shared production venv, `-p no:cacheprovider`,
`--ignore=tests/test_livekit_audio.py --ignore=tests/test_livekit_engine.py`, private fixtures
under a fresh non-repository /var/tmp root, then moved to canonical task scratch. Warnings
were SWIG type metadata and existing Pillow getdata deprecations. No tests or privacy
contracts were weakened. Raw audio, original labels, model locks and prior failed receipts
are retained unchanged; all committed benchmark artifacts are aggregate only.

Final documentation gate: `files=36 dead_links=0 stale_terms=0 retired_verbs=0 orphans=0`;
compact budgets AGENTS79 / STATUS120 / agent-map59 / agent-testing52. Whitespace clean.
Changed runtime/setup/test files have **0 introduced Ruff findings** against HEAD (24
pre-existing E402/F401/E731 findings remain); no unrelated formatting/lint repair was made.



Valid until: the DSP/library/platform or physical-route evidence changes — then treat as history.

## 2026-10-04 — existing native streaming low-pass kernel (ADR-0222)

Targeted DSP/APM gate: 139 passed in 4.17 s; scoped lint/whitespace clean. The accepted
single-section SciPy SOS path preserves tested float32 output and float64 state exactly
across four rate/cutoff/Q matrices, single/irregular chunks, 300-chunk streams, reset,
failed optional kernels, nonfinite inputs and bypass. The lfilter prototype was rejected
for last-bit state changes. Implemented 300 x 100-ms synthetic profile: scalar p50/p95
871.2595/1002.68345 us, native SOS 101.0415/131.5884 us (8.62x p50). Tracemalloc transient
peaks 10,157/40,779 bytes; cold import excluded. Receipt remains in canonical task scratch
until preserved at integration. No Rust build, private recording, native model, audio
capture/playback, whole-session CPU/RSS or phone validation was part of this DSP probe.


Valid until: the integrated source/runtime/profile or physical-device evidence changes — then treat as history.

## 2026-10-04 — complete performance-mode integration verification

Merged the four separately documented behavior/resource branches into the isolated mode
branch, retaining every worklog record. The TTS overlap explicitly preserves both scoped
Amy-low resolution and tagged thread propagation/shared native configuration. Independent
policy/CLI/residency and mobile source reviews approved the final integration with no
blockers. The verifier covers known preflight consistency, not later native-file mutations.

Full CI-style Python gate: **11631 passed, 42 skipped, 9 warnings in 460.43 s**, using the
shared venv, no pytest cache, the two CI LiveKit exclusions, fresh private synthetic fixtures
in /var/tmp, then moved to canonical task scratch. Full current-lock Flutter: **275 passed**;
analysis **No issues found** in 3.3 s; integrated compiled Compact + asset tests **23 passed**.
Native/device calls were absent from these Dart/fake gates. Initial unsupported Flutter
`test --offline` invocation was corrected to cached `pub get --offline` then `test --no-pub`.
Dependency versions were retained, not upgraded. Final changed-Python Ruff comparison:
19 inherited findings and **0 introduced**; whitespace clean. Source behavior was frozen
for the full gate; the later Whisper comment correction is documentation only.

The first pair probe is retained as media-pair-probe-v1.json. The final source-bound probe
retains model/core/config hashes before/after each process, all three completed, SDK1.13.3,
ASR2/TTS2 with two-CPU caller masks; five process threads sampled/cell, zero unknown/reading
failures/outside masks. Steady PSS current **805.6748**, responsive **379.9053**, compact
**355.7949 MiB**; process peaks **816.0391/390.2227/368.0078 MiB**. Three public phrase full
synthesis timings per model include cold first calls and do not measure audibility. All
waveforms finite/nonzero. Only streaming ASR+TTS were resident: final/verifier/LLM/DSP and
physical devices were excluded, so this is not whole-app/phone RSS or a quality promotion.
The aggregate-only committed receipt SHA256 is **0c661c37ed560d574ce8a947405679536cf2a07b21ef5f8add4af2d0241663d9**; actual receipt and probe
source remain in ignored logs/runs/voice-performance-modes-20261004. Raw recordings, old
labels/models/configuration and every original failed/diagnostic log remain unchanged.

Final docs gate: files=37, dead_links=0, stale_terms=0, retired_verbs=0, orphans=0.
Compact budgets: AGENTS79 / STATUS120 / agent-map59 / agent-testing52. No new whitespace
errors; no introduced Ruff findings against e337c20. All behavior/source checks passed
before landing. Publication remains gated by the earlier automatic approval rejection;
no push or post-push branch/worktree cleanup is claimed without explicit resolution.


Valid until: the next source/evidence revision supersedes these restored receipts — then treat as history.

## 2026-10-04 — restore focused branch verification records

The worklog conflict resolver retained the final DSP/integration records but dropped four
new append sections. Their original commits retained exact copies; restored verbatim below.
No code, decision, runtime configuration, recording, native evidence or successful gate changed.


Valid until: the profile/model/runtime or fresh physical-device evidence changes — then treat as history.

## 2026-10-04 — explicit resource modes and component probes (ADR-0218)

Current is a no-op; responsive/compact bind the actual English INT8 streaming tuple
and VITS/Kitten TTS assets, respectively, without altering final/verifier, provider,
DSP, endpoints, tool/privacy/identity, expressive-voice or context/output policy.
24 final policy tests passed; earlier wider launcher/readiness/device gate passed 329.
Both optimized presets use the separate ADR-0219 residency implementation at integration.
Caller-relative paths resolve against the same root in core, doctor and live launcher.
Mode snapshots and descriptor/path/tree identities prevent known preflight mutation
bypasses; native opening after verification still has a later-filesystem-change limit.

Actual sequential native CPU pair probe under logs/runs/voice-performance-modes-20261004:
streaming ASR plus TTS resident, synthetic silence plus three public-phrase generations,
two threads per engine/two-CPU caller mask; no final recognizer/verifier/LLM/DSP/device.
Steady PSS current805.3701/responsive380.1309/compact355.1533 MiB; process peaks
816.1289/390.8242/367.9492 MiB. All generated waveforms finite/nonzero; detailed receipts
bind source/model bytes and sampled thread masks. Public-phrase synthesis timings include
cold first calls and are diagnostic, not live TTFA/RTF or a corpus speed ranking. No audio
capture/playback, repaired echo/barge acceptance, phone thermal/battery or promotion.
Combined integration Python/Flutter gates are recorded below after they actually complete.


Valid until: the model/runtime/profile or physical-device evidence changes — then treat as history.

## 2026-10-04 — Keep heavy local models available without compulsory startup residency (ADR-0219)

Residency gate: 218 native-free affected tests passed in 4.74 s, including multimodal,
egress, role retention and APM regressions; scoped lint/whitespace clean. Synthetic loaders
reported all=2 generation calls versus fast=1 (148.14/36.14 ms, fake 72/8 MiB buffers).
These numbers model skipped startup work and are not native model savings. The sole/shared
answering model is still warmed and cold main research/images remain callable. The existing
GGUF lifecycle/context is unchanged; Ollama retention takes effect on subsequent model requests.


Valid until: the model/runtime/profile or physical-device evidence changes — then treat as history.

## 2026-10-04 — Stage and bundle only the active mobile speech assets (ADR-0220)

Mobile footprint: 262 Flutter tests passed with cached Sherpa 1.13.3/Gemma 0.16.5;
10 final asset tests and full analysis clean. Expanded fake download/bundle gate: 13 passed
in 0.92 s; shell syntax, scoped lint/format and whitespace clean. No native/model downloads
or device runs. Default bundling selects the four locked ASR filenames; optional Whisper
selects its three locked filenames. Available rights/readme files survive. Cached unused
precisions and public examples are retained on disk. Locked active ASR 74,207,237 bytes;
omitted Whisper 160,626,066 bytes, not an APK-size or phone-RSS result.


Valid until: the model/runtime/profile or physical-device evidence changes — then treat as history.

## 2026-10-04 — Freeze mobile thread requests before service construction (ADR-0221)

Mobile startup budgets: 79 native-free ownership/config tests passed. Explicit
compiled compact gate passed 13 tests; full current-lock Flutter suite and analysis were
green (combined gate recorded at integration). Pure config tests compare all non-thread
native fields unchanged and cover exact ASR payload reconstruction, TTS direct/worker
parity and invalid mode refusal before startup. No phone/plugin/native/resource run.


Valid until: the capture route/enrollment or a completed owner live run changes — then treat as history.

## 2026-10-04 — actual-host monitored live startup and continuation

The owner requested an immediate monitored test. Outside the execution sandbox the host
GPU and PipeWire were accessible: RTX4090 Laptop, 16,376 MiB total/7,749 MiB used GPU
memory at preflight; available RAM 26,454 MiB. The configured inference remained local_only
with cloud disabled. Responsive + required SenseVoice final was selected for the attempted
comparison; saved configuration and enrollment were not changed.

The sole physical `./live.sh` entry created temporary Ollama and the reversible OS echo
route. Full doctor reached READY. Once the actual microphone domain became known, startup
failed closed at the existing compatible-speaker-enrollment word-cut guard (ADR-0209).
There were zero capture heartbeats and zero assessed turns; the final diagnostic was
incomplete with unclean_shutdown. No STT quality, reply latency, actual audibility, barge-in,
fully offline whole-session or phone result follows from this attempted startup.
The launcher restored original audio defaults and stopped its temporary server; no test
was left running. Private console/WAV/diagnostic artifacts remain in ignored logs/live.

A concrete session-only desktop --no-speaker-enrollment fallback was statically validated
without starting it. Owner-verified actions would be unavailable. That choice was offered,
not authorized or run; the owner instead requested publication to origin/main and later
continuation. Next: refresh compatible enrollment on the active OS echo route, or obtain
explicit selection of the temporary non-enrolled voice trial, then run the same benign
questions/STOP/talk-over cases in Responsive and Compact with aggregate monitoring. Keep
all original private recordings and validate final evidence coverage before any verdict.


Valid until: a newer source/trigger policy changes this integration — then treat as history.

## 2026-10-04 — integrate newer origin manual-workflow policy

Origin advanced to 4422261 while the performance work was local. Integrated its explicit
owner-requested workflow_dispatch-only GitHub Actions policy and ADR-0223. Runtime code,
job bodies/test commands, safety gates and permissions are preserved. STATUS combines
both sessions' facts within its 120-line budget. The existing green local application
results remain valid; no automatic workflow dispatch was added or invoked.


Valid until: the reply streaming/cancellation contract changes — then treat as history.

## 2026-10-10 — Observe reply revocation between emitted sentences

A deterministic public-only provider token containing two sentences reproduced
continued emission and cancelled=False when the first emitter revoked the task.
The real TaskManager emitter independently protects audio; this establishes a
capability cancellation-metadata/cleanup defect, not a proven audio escape.
`_stream_and_speak` now checks revocation before each sentence, after exhaustion
and after the final tail, then closes the provider through the existing seam.
Uncancelled text and playback-grounded history semantics are unchanged.

Headless gate: `tests/test_stream_speech_cancellation.py`,
`test_streaming_tts.py`, `test_capability_stream_close.py`,
`test_capability_context_isolation.py`, `test_pretoken_cancellation.py`,
`test_playback_history.py`: 50 passed in 3.42 s. No audio/model/live run.


Valid until: the selected model, template, persona or device configuration changes — then treat as history.

## 2026-10-10 — Explicit small desktop model and spoken prompt (ADR-0232)

One approved official Qwen2.5-1.5B Q4_K_M asset was pinned and imported locally;
source and native rewritten-container pins are separate. All 339 tensor payloads
and metadata matched; no other model was downloaded. Matched ChatML/raw and
direct/core controls did not establish a MiniCPM transport bug. Full/minimal
Qwen public canaries scored 22/24 and 24/24 with zero instruction recitations.

Final 4090 factory QA preserved caps of 8192 context and 512 output tokens with
num_gpu=999, requested fast threads=2, actual spoken runtime system and fixed
public Iris persona: 32/32 exact answers (8 development, 8 evaluation,
8 confirmation, 8 new holdout), no empty/recitation/think-markup outputs.
Split answer medians were 187–215 ms, max 261 ms. Resident bounded helper:
12/12 correct, zero false ACT, median 336 ms, max 391 ms. Explicit warm-up was
outside those timings. Across 45 post-call resource samples, daemon and
all-thread descendants reached 688431104 RSS bytes, 2 processes/59 threads,
with zero read failures; Ollama reported 1360758046 resident VRAM bytes.
No continuous peak, resource reservation or physical latency claim follows.
All owned daemons stopped.

The first factory run kept the committed anonymous persona while its benchmark
expected Iris; preserve it as mismatched identity-scoring evidence, not a model
quality failure. Native-measured source is archived beside aggregate receipts;
later typed-scope/CLI privacy/tooling edits do not relabel those old hashes.
Earlier root-child-only RSS samples omitted descendants and are not full RSS.
The initial decision run with differing native thread options returned unavailable
at 3 seconds; matched resident settings passed without changing the deadline.

Final native-free gate: 511 passed in 3.50 s across profile/setup/persona/quality,
residency/threads/catalog/goal/memory/context/stream/decision/router and provider/
privacy files. Ruff has no introduced findings; readiness retains its existing
E731 lambda. Diff and docs checker pass. The committed quality tool now exposes
fixed factory settings and the eight public holdout questions. Exact setup uses
cached bytes and verifies the existing alias without replacing it. Explicit
current default/rollback remains; main/vision/tools/caps are retained. No cloud,
microphone, doctor, audio route or owner recording was used. CPU-only and
macOS/Windows/native/live/owner gates remain pending.


Valid until: the pinned model identity parser changes — then treat as history.

## 2026-10-10 — Independent identity review follow-up

Independent review demonstrated that a second differing TEMPLATE directive
could be omitted by the shared first-template parser. The project verifier now
requires exactly one FROM and one TEMPLATE before hashing effective behavior.
Parameter order remains normalized; duplicate parameters remain bound. Public
fixture regression rejects the later override. This validation-only tightening
does not change the measured valid native alias or any model request.

Focused identity gate: 31 passed in 0.23 s; Ruff/diff pass.


Valid until: the fixed public factory benchmark protocol changes — then treat as history.

## 2026-10-10 — Define bounded CPU factory qualification

The committed public benchmark accepts only desktop_gpu_4090 or cpu_laptop
factory settings; cpu_laptop explicitly requests num_gpu=0 and fast threads=2
while retaining its 2048-context/256-output caps. Eight fixed resident decisions
retain the 16-token/3-second production helper. Separate different-prompt warm-up
is outside scoring; no graded answer is precomputed. First nonempty text timing
is model text latency, not audio. Optional owned-daemon Linux resource samples
follow descendants from every thread and return numeric counts/RSS only; they
are post-call observations, not continuous peaks. Synthetic gate:26 passed in
0.23 s; Ruff/diff pass. No CPU inference result is claimed yet.


Valid until: the CPU model, addressing prompt or warm/cache policy changes — then treat as history.

## 2026-10-10 — CPU resident pass and cold mixed-prefix failure (ADR-0232)

The committed public factory protocol forced num_gpu=0 and requested threads=2
with unchanged cpu_laptop caps (2048 context/256 output). Eight fresh public
holdout answers were exact, with no empty/recitation/markup outputs. Median
first model text was 393 ms (max 452 ms); complete short answer 558 ms (max
614 ms). Eight resident addressing decisions were available/correct with zero
false ACT, median 591 ms/max 674 ms, within the unchanged 16-token/3-second
helper. Answer warm 2.624 s and matched-format classifier warm 5.714 s were
excluded from scoring. These are grouped resident results, not cold readiness.

Four further fixed answer→decision cycles had no classifier rewarm: four
answers were exact, but only three decisions were available/correct; one was
unavailable at 3.009 s. No false ACT. Answer max grew to 1.860 s, first model
text max 1.759 s. This fails cold mixed-prefix CPU qualification. A longer
startup prefill may help the first classification, but the grouped result alone
cannot justify CPU defaults; the shorter profile-specific addressing prompt or
cache policy must be qualified without raising the per-turn deadline. No runtime
prompt/model/default was changed after this result.

Post-call samples followed daemon descendants through every process thread.
Grouped max summed RSS 1401872384 bytes across 17 samples; mixed max 1401892864
bytes across 9 samples, two processes, zero read failures. Ollama reported zero
resident VRAM in both runs (resident model size 1240433949 bytes). These are
sampled sums/API measures, not continuous peaks or full application memory.
The unchanged CPU main/vision Gemma3:4b client was constructed but never loaded
or called. No CPU READY, Mac/Windows, audio, live or native main-tier claim.
All owned daemons/workers stopped and preexisting services remained untouched.

Aggregate reports and source/binding receipts remain separate for the grouped
and mixed protocols; the benchmark includes a fixed --interleave-checks flag.
Synthetic benchmark tests: 27 passed in 0.20 s; Ruff/diff pass. No raw input,
output or recording is in those aggregate reports, and no other model downloaded.


## 2026-10-10 — Shared desktop model selection

Valid until: model/profile or entrypoint changes — then treat as history.

Core, doctor and the Linux launcher now share `select_voice_model`; explicit
model overrides reach both readiness and runtime. Conflicting fast selection
refuses before host setup. Current rollback uses the original config and
refuses an already resolved candidate. Direct readiness resolves the profile
marker; runtime passes the selected persona to capabilities. No default switch
or microphone/doctor execution occurred.

Fake-only gate: voice_model_selection, voice_model_profile, live_launcher,
setup_doctor and performance_modes: **365 passed, 2 warnings in 2.35s**.
The first run had two test-fixture errors (an empty strict device profile),
corrected without changing device validation. No native model run in this gate.

Valid until: resumed-turn identity or conversational-admission behavior changes — then treat as history.

## 2026-10-10 — Preserve answers to rendered resumed questions

The enabled ResumeConfig controller branch committed its synthetic-resume input
generation without binding conversational admission. A deterministic public fixture
reproduced the real resume event: after barge/continue, the resumed "Which city?"
completed, but "Paris" never reached the addressing classifier. That regression failed
on cd443a0, then passed after the five-line current-generation binding. A held resumed
question still grants no answer window before playback completion. Its published final
keeps unknown origin, owner_verified=false, response-only and skip-user-memory metadata.

Low-priority headless command from docs/agent-testing.md: tests/test_conversation_admission.py,
tests/test_resume.py, tests/test_post_barge_response.py, tests/test_continuation.py —
112 passed in 4.09 s. Scoped E9/F63/F7/F82 Ruff passes. The runtime persona wiring in the
orchestrator's separate worktree was untouched. No native model, doctor, microphone,
phone, route change, recording, cloud call or action-authority promotion occurred.


Valid until: the setup transport, asset publication or alias API changes — then treat as history.

## 2026-10-10 — Independently review pinned voice-model installation

Read-only review found Ollama SDK redirects enabled on loopback metadata calls
and parent mkdir occurring before symlink refusal. Setup now disables redirects
and checks parents before creating directories; quoted/control-character paths
cannot inject Modelfile directives. A cooperating O_EXCL fail-busy lock spans
alias recheck/create/post-identity and removes only its own regular inode.
Source bytes remain exclusively published and existing mismatches are refused.
External Ollama writers and separate user/temp namespaces are not covered by
daemon CAS; no atomic alias-create claim is made. Windows path syntax uses
quoted forward slashes, but no Windows native run occurred.

Thirty fake installer tests passed in 0.37 s; Ruff/diff pass. They cover
redirect/proxy client construction, no redirected mkdir, concurrent busy lock,
recheck/import/identity ownership, failed import, replaced-lock preservation and
Modelfile path refusal before I/O. No new download, alias import or native run
was performed for these validation-only setup repairs.


Valid until: the addressing, model, startup or context policy changes — then treat as history.

## 2026-10-10 — Reject compact addressing and retain scoped explicit Qwen (ADR-0235)

One492-character system plus JSON-input candidate failed all16 negative
component cases with false ACT, both full classifier and forced helper. CPU
mixed decisions1/4 correct (3available), GPU2/4 correct. Rejected source and
aggregates are retained; neither it nor a new prompt policy is shipped.

Restored legacy1546-character system/format and actual3-second classify-hi
startup warm: CPU mixed4 answers/decisions correct, but max answer11.226s;
CPU semantic cell max answer14.410s. Aggregate-only old receipts do not identify
which ordinal was slow. Component matrix: positives7/8 (one semanticUNSURE),
negatives10/16 (six falseACT, zero shortcuts); context short-answer full path
timed out while later forced helper succeeded, and context quote forced helper
was unavailable. No invented ground truth for the four ambiguous cases.
ConversationAdmission independently rejects all16 idle negative fixtures, so
component failures do not establish whole-runtime activated replies.

GPU actual-warm legacy mixed-context4 answers/decisions passed, decision max
629ms. One approved20-second benchmark-only direct-prefill diagnostic expired
at20.012s. Later mixed4 passed: answers1.31–1.37s, decisions1.65–2.70s; all
per-ordinal timings retained without text. No successful bounded prefill, warm
API, per-turn deadline increase, CPU default or semantic promotion follows.
Explicit Qwen reply profile remains; CPU's existing input_gate=false unchanged.
No more native experiments authorized in this iteration; all owned daemons
retired and user services untouched. Model text latency is not physical audio.


## 2026-10-10 — Desktop integration preflight and first broad gate

Valid until: changed integration source or a new full gate — then treat as history.

Actual runtime/persona/classifier assembly and installer/profile tests passed
**61 tests in 2.00s**. The first CI-style full run was interrupted for repair
after **3 failed, 4323 passed, 19 skipped in 263.11s**. One failure was a
prospective AMI source inventory missing the new output-cleanup module. Two
assertions reported the same stop/restart scenario regression: admission omitted
the canonical run/execute command verbs before command.stage could confirm them.
The complete failure log and synthetic fixtures remain private task evidence.

The real `python -m core --session console --llm echo --voice-model current`
entrypoint returned exit 0 and the expected reply to public text, using a scratch
current-directory config, forced in-memory storage and no warm/audio/model calls.
An initial unsupported `--config` invocation returned argparse exit 2 before
startup; both receipts are preserved. Console final process RSS was 38.8 MB, which
is not full voice-agent memory. Its system samples showed 94–96% host CPU use;
these occupied-host observations are not hardware-isolated model performance.
No unrelated process was stopped and no microphone/doctor/live check ran.

Valid until: the explicit benchmark source list, model or measurement protocol changes — then treat as history.

## 2026-10-10 — Bind future public voice benchmarks to the actual source seam

Root integration review found the benchmark guarded only its own file/persona
while classification/factory/native request behavior depended on other modules.
The prospective fixed manifest includes the harness, identity adapters, Ollama
client/decision/factory/model-profile/classifier/persona/config helpers and
runtime prompt assembly/registry/model enums: 15 source paths, at most 1 MiB
each and 4 MiB total plus one detection byte. It hashes before any model
metadata call and after every measured/metadata call, refuses unavailable or
changed source with coarse fixed codes, and publishes only public relative
filenames, per-file hashes/counts and canonical before/after digest. No dynamic
import crawl, whole audio closure or loaded-bytecode/native attestation claim.

Original native aggregate reports, wrapper bindings and retained source remain
unchanged in the durable ignored archive. This adds prospective evidence only;
no model/native/audio/live run or download occurred. Deterministic tests mutate
each bound source during a fake run, reject invalid/oversized sources before
model metadata, refuse disappearance after measurement, verify complete typed
hash/count receipts and ensure coarse CLI failures publish no report or paths.

Verification: `tests/test_voice_model_quality.py`, `test_addressing_profile.py`
and `test_voice_model_profile.py`: 99 passed in 1.82 s using the shared venv
with native/log/config access disabled and fresh owned scratch. Scoped Ruff,
`git diff --check` and docs checker pass; STATUS stays at 120 lines.

Valid until: the evaluator import closure or declared source qualification scope changes — then treat as history.

## 2026-10-10 — Bind prospective media evaluators to output cleanup

The first integrated full gate stopped with 3 failed, 4323 passed and 19 skipped
in 263.11 s. One failure was AMI's exact clean-import closure: Sherpa now imports
`core/engines/_output_cleanup.py`, absent from the prospective list. The other
two conversation-flow failures are repaired separately. Five media evaluator
inventories now hash the helper; AMI derives its inventory from generic replay.
AMI's exact count changes 58 to 59 and generic schema-4 async replay 32 to 33.
The new final-STT control proves changed helper bytes change its source digest.
Historical locks, recipes, source receipts and native results were not edited.

Audit found no related curated whole-agent list to extend: conversation
qualification binds clean Git revision, and the legacy English LLM harness
directly measures llama.cpp rather than the application factory. Unrelated
control-plane modules remain outside the media inventories. The prospective
voice-model factory manifest is a separate worker change.

The first six selected checks passed in 15.42 s. An added async test's initial
attempt failed at pytest setup because its task scratch parent did not exist;
creating that parent under `~/work/_temp/` allowed the final seven-test gate:

```sh
SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_ami_natural_turn_capture_replay_eval.py::test_exact_clean_import_closure_matches_persisted_source_list tests/test_microsoft_aec_apm_dtd_eval.py::test_clean_process_eager_local_imports_equal_the_bound_closure tests/test_capture_replay_eval.py::test_evaluator_provenance_includes_streaming_decode_sources tests/test_capture_replay_eval.py::test_evaluator_async_mode_uses_schema_four_and_real_worker_harness tests/test_production_final_stt_eval.py::test_source_identity_binds_newly_imported_output_cleanup tests/test_edacc_endpoint_integrity_eval.py::test_shadow_source_binding_is_strictly_conditional tests/test_livekit_causal_endpoint_eval.py::test_execution_closure_reader_admits_current_sherpa_above_public_limit -q --basetemp=/home/dobo/work/_temp/codex__resume-question-admission/pytest-inventory-qualified
```

Result: **7 passed in 12.26 s**. Scoped E9/F63/F7/F82 Ruff and whitespace pass;
the documentation checker reports 38 files and zero issues, with STATUS at 120 lines.
These are source-binding/headless checks, with no private corpus read, model
inference, device/doctor/live run, network call or change to the live-stop gate.

Valid until: conversational admission or canonical request grammar changes — then treat as history.

## 2026-10-10 — Repair canonical request admission and stop/restart flow (ADR-0227)

Reproduced the two integration failures unchanged: stop/restart confirmation and
the strict three-run flow report were red (`2 failed in 51.34 s`). The early cue
rejected the shipped Run/Execute command forms before staging; deterministic
models already implemented stream. No fixture/deadline/assertion changes were
needed. Audited and retained existing dictation/controller and vault request
verbs plus bounded greeting/vocative/courtesy inspection. Original addressing
input and downstream confirmation/provenance remain unchanged. Idle quotes stay
ambient; a frozen exact heard-answer ticket preserves a quoted final across
arrival-time window closure. Hot-path phrase storage/parsed-word reuse retained.

```sh
SPEAKER_TEST_LOG=0 SPEAKER_NO_LOCAL_CONFIG=1 SPEAKER_LIVE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_conversation_admission.py tests/test_conversation_flow.py tests/test_speech_analyzer_contract.py tests/test_addressing.py tests/test_post_barge_response.py tests/test_continuation.py tests/test_resume.py tests/test_voice_session.py -q
```

Final focused result: **264 passed in 14.20 s**. Scoped Ruff, whitespace and docs
checks pass. The original production flow and all its assertions/timing bounds
are unchanged. These are fake model/device control-plane receipts; no native,
private recording, microphone, download, cloud or physical validation occurred.


## 2026-10-10 — Complete prospective inventory regression gate

Valid until: evaluator imports or source inventories change — then treat as history.

The second full attempt stopped at one remaining AMI fake-report expectation:
**1 failed, 225 passed, 11 skipped in 82.54s**. Its expected prospective source
count was still 58, although the exact closure was correctly 59. Corrected that
expectation without changing guards or historical reports. All six affected
evaluator modules (AMI, generic capture, EdAcc endpoint, LiveKit causal endpoint,
Microsoft AEC and production final STT) then passed **460 tests, 3 skips in
43.16s**, with private hermetic synthetic fixtures.

The final full logic rerun uses ordinary CI scheduling with the existing
single-thread native environment, no bytecode/cache, live audio disabled and
the two CI LiveKit exclusions. Test wall time is not a performance comparison
with the earlier low-priority receipts. Failed logs/fixtures are preserved.


Valid until: private-writer boundary or recognized-Git fixture semantics change — then treat as history.

## 2026-10-10 — Repair inherited final-STT recognized-Git fixture

Third full attempt: **1 failed, 7902 passed, 32 skipped in 233.74 s** at
`test_private_writer_is_mode_600_no_clobber_and_refuses_git_and_symlink`.
The runtime helper is unchanged from pristine 734713d: a directory marker needs
HEAD, whereas an empty `.git` directory is inert. The test made an empty marker
and default-mode parent. A 0755 parent caused unrelated private-parent refusal,
masking the defect in usual focused runs; root's full runner uses umask 077 and
therefore made the parent 0700, exposing the incorrect Git-refusal expectation.

Paired exact test, no source edits: pristine main 734713d passed in 0.73 s and
integrated 2f3dd90 passed in 0.95 s with the default umask; under explicit `umask 077`
both failed with DID NOT RAISE in 0.71 s and 0.90 s respectively. No ordering-induced
production change is needed. Shared main stayed clean; test logs/bytecode/cache
were disabled and synthetic fixtures remained in the task scratch directory.

The test-only repair sets its Git parent explicitly to 0700 and creates `.git/HEAD`
using the existing preparer fixture pattern. No production guard is relaxed.
Adjacent scan covered all six affected media evaluator tests, the command/noise
preparer and conversation qualification: other recognized markers already have
HEAD or nonempty gitdir-files, or use the actual checkout; no second misuse found.

```sh
umask 077
SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_production_final_stt_eval.py tests/test_prepare_public_command_noise_corpus.py -q --basetemp=/home/dobo/work/_temp/codex__resume-question-admission/pytest-writer-fixed-077
```

**70 passed, 1 optional-corpus skip in 2.99 s**. The repaired exact test additionally
passed under explicit `umask 022` in 0.85 s with basetemp `pytest-writer-fixed-022`.
Scoped Ruff/whitespace pass; docs report 38 files and zero issues, STATUS 120 lines.
No recording, native model, network, doctor,
microphone, hardware or live execution occurred; historical receipts are intact.


Valid until: negative-final retirement or input-continuation ownership changes — then treat as history.

## 2026-10-10 — Retire matching rejected typed partials (ADR-0227)

The combined integration gate before this repair was reported green at 12,187
passed, 41 skipped, 9 warnings in 523.23 s; that receipt does not include this
negative-final repair. A targeted actual-runtime probe then proved a matching
quoted rejection left both a partial fence and arrival reservation alive, causing
the next short add-on's model request to inherit the old unheard request. Matching
abort cleared both. Five new typed-runtime cases reproduced `2 failed, 3 passed`
in 3.58 s before the fix. Applied exact acoustic-key retirement only: commit the
existing partial generation and clear its matching continuation, with no automatic
restoration/reissue and no unmatched/unkeyed effects. Heard quoted answers retain
their existing admitted path.

```sh
SPEAKER_TEST_LOG=0 SPEAKER_NO_LOCAL_CONFIG=1 SPEAKER_LIVE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_rejected_partial_final.py tests/test_conversation_admission.py tests/test_continuation.py tests/test_resume.py tests/test_post_barge_response.py tests/test_conversation_flow.py tests/test_core_runtime.py tests/test_acoustic_lineage.py tests/test_final_preprocessing_cancel.py -q
```

Observed **342 passed in 32.66 s**; two synthetic watchdog diagnostics appeared
without failures. Ruff/whitespace/docs checks pass. No native/full-suite/device/
audio/model/network experiment ran in this repair lane. Final combined integration
verification follows separately; original flow assertions and deadlines remain.


## 2026-10-10 — Final combined desktop implementation verification

Valid until: source, dependency, platform or physical-route changes — then treat as history.

Final tested source: 259448c180857ba588ec648e43d2445817496f24. The CI-style
logic gate passed **12192 tests, 41 skipped, 9 warnings in 562.40 s** after the
exact rejected-final retirement fix. Flags: `SPEAKER_TEST_LOG=0`,
`SPEAKER_NO_LOCAL_CONFIG=1`, `SPEAKER_LIVE=0`, `PYTHONDONTWRITEBYTECODE=1`,
OMP/OpenBLAS/MKL/NumExpr threads 1; shared venv Python `-B -m pytest
-p no:cacheprovider tests -q --maxfail=1 --ignore=tests/test_livekit_audio.py
--ignore=tests/test_livekit_engine.py` with a fresh private `/var/tmp` fixture
root, moved into task scratch afterward. This final run used ordinary
scheduling and umask 077. The warnings are the two Sherpa SWIG and seven
existing Pillow deprecations; the separate interpreter-exit SWIG warning stays
historical dependency output, not a failure.

The earlier complete pre-fix receipt, 12187/41 skips/9 warnings in 523.23 s, is
retained separately with source 26c4938494dea6faae68c8471c7b9fddaf5c7991.
No failed, rejected or earlier native receipt was relabeled as this final run.
All 58 final changed Python files pass scoped fatal-error Ruff checks. The
repository documentation checker reports 38 files with zero findings, and
`git diff --check` is clean. No microphone/doctor/live validation or model rerun accompanies this
logic gate. Mobile source is unchanged; dfcef68 is not an integration ancestor.

Durable ignored archives: `logs/runs/desktop-core-integration-20261010`,
`logs/runs/voice-model-quality-20261010-d2157e2`,
`models/candidates/reference-ring/ac0897d`, and
`models/candidates/localvqe/evidence`. The model archive preserves 139 files
(4,318,917 bytes), manifest SHA256
556b511e4b8762d35ae6ea2f6c9843ffbf1a84423dfaf3ed94fc655ccaa1dcbe.
Only aggregate findings enter Git. The protected live candidate/backup/recordings
and deferred mobile branch stay separate from desktop landing/cleanup.

## Desktop second pass: first-fragment scheduling (2026-10-10)

Valid until: chunking/stream lifecycle changes or physical evidence — then treat as history.

ADR-0239 adds explicit normal/fast delivery without changing defaults or shared sentence semantics. Public scheduling probe: long-clause first speakable step 25→10, unpunctuated 56→29, short/tagged/numeric unchanged; this is not audible latency. Initial adjacent gate 389 passed with one invalid empty-profile test fixture, corrected; final new module 34 passed. Full integration receipt follows after source freeze. No microphone, doctor, models or native audio were run.

Valid until: startup warm ownership, provider cancellation or inference policy changes — then treat as history.

## 2026-10-10 — Foreground input retires speculative warm ownership (ADR-0237)

A public synthetic Event/lock reproducer on 70345ae observed no warm cancellation
context, a foreground stream blocked by warm, unfinished warm after stop, and
one helper call beginning after stop. No model/audio/route was used. The repair
sets a runtime-owned retirement signal at accepted generation allocation and
shutdown, uses the existing streaming cancellation seam, and checks every
successor warm stage. Media-first order, selected residency and foreground
models/capabilities/options/deadlines are retained. Legacy generate-only APIs
are adapted before entry; an entered TypeError cannot replay warm inference.

Built-in direct cloud clients and declared cloud helper owners are excluded,
with zero-transport tests. Context snapshots are hook-free, LOCAL_ONLY and
preserve inherited cancellation; foreign providers remain responsible for their
locality/cancellation. Noncooperative generators/media warm may outlive retirement;
readiness remains unfinished until true warm-worker return. No native-preemption,
quiet-host CPU, Windows/macOS or physical-first-audio claim follows. Prior loaded
host experiments remain unchanged. Handy primary code informs ownership/cleanup
semantics, not imported implementation or performance evidence.

Final scoped command: shared venv pytest with SPEAKER_TEST_LOG=0, local-config/
live disabled, bytecode/cache disabled and fresh owned basetemp; files:
startup_warm_priority, readiness, local_model_residency, ollama_async_cancel,
llamacpp_cancel, final_preprocessing_cancel, pretoken_cancellation. Initial
strict-context scope: 207 passed in 13.36 s; final nominal built-in warm
selection gate is recorded below. Full new-test/readiness Ruff, scoped runtime
lint, diff check and docs checker pass; STATUS remains at 120 lines.

Final nominal-selection scope: 207 passed in 13.29 s. Source is frozen for
review/landing; no native benchmark, model download, audio route, microphone,
doctor or live run occurred. Only one compact result and the baseline
synchronization reproducer are retained in owned scratch; superseded test
transients may be pruned under the current human retention instruction.

Valid until: stage-boundary semantics, metric token or playback receipt ownership changes — then treat as history.

## 2026-10-10 — Add separate observed pipeline latency (ADR-0240)

Based on clean 70345ae, this instrumentation lane adds request/text/admission/
onset-observation boundaries and a separate scalar-only run-summary block. Existing
five-key metrics, first-audio callbacks, EWMA, watchdog, routing and authority are
unchanged. A locked snapshot requires captured current turn tokens; invalid
timestamps and reversed known boundary order are refused. Exact tracked playback
owns a maximum 64-entry fragment-to-metric map, installed before synchronous sink
callbacks and retired through the existing receipt/cancellation lifecycle.

The output endpoint is runtime **receipt-dispatch observation**, not DAC, first
PCM, pure GPU, synthesis-only or queue-only time. Auxiliary/latency-ack, legacy,
nonexact, missing-token and stale paths remain unknown. The new summary retains
only five named intervals and aggregates, with no text/audio, identities or raw
timestamps. Root integrates model-request/text producer and CLI finalize call
sites separately; the isolated lane's producer boundaries are deterministic fakes.

First focused run: 1 failed, 74 passed in 2.95 s. The new fixture replaced the
recorder after engine.start had captured the old metric callback; moving fake
recorder setup before start corrected the test, with no production workaround.
Expanded focused gate: **79 passed in 2.43 s**. Adjacent command:

```sh
SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_metrics.py tests/test_runlog.py tests/test_latency_stage_runtime.py tests/test_playback_receipts.py tests/test_engine_playback_receipts.py tests/test_playback_history.py tests/test_core_runtime.py tests/test_watchdog.py tests/test_core_routing.py tests/test_streaming_tts.py tests/test_resume.py tests/test_continuation.py tests/test_tts_markup.py -q --basetemp=/home/dobo/work/_temp/codex__latency-contributions/pytest-adjacent
```

**391 passed in 8.79 s**. Formatting only the wholly new receipt module was
followed by its standalone gate: **20 passed in 2.45 s**. Scoped fatal-error Ruff
and whitespace pass. The host had 169 GiB available before the adjacent gate.
No private recording, native model, microphone, doctor, network or device ran.
Protected live/mobile worktrees and original baseline/source branches are intact.

Self-review found that the new field's reused statistics.median could overflow
when two individually finite intervals were 1e308. A separate interpolation-based
stage aggregator now keeps its result finite, with a strict-JSON regression;
the legacy aggregate implementation is restored exactly. The same adjacent
command with fresh basetemp `pytest-qualified` passed **392 tests in 9.69 s**.
Docs checker: 38 files, zero issues; STATUS remains 120 lines. Superseded owned
synthetic fixture trees were pruned only after the successor qualified; one
rolling qualified fixture and this compact reproducible result remain.

Valid until: AEC delay history/estimator or its benchmark environment changes — then treat as history.

## 2026-10-10 — Bound calibration history without changing delay math (ADR-0238)

Pinned 70345ae baseline kept in Git. Paired fixed-ring storage retains byte-identical
chronological windows and unchanged estimator/recalc/clamp AST. Original compact
baseline/prototype result and one verified current result are retained; no model,
private capture or snapshot tree was copied. Disk check reported 170 GiB available.
Adjacent alternating unpaced synthetic observations measured baseline p50 21.965–
27.806 us versus ring 15.022–19.273 us. Host-load variation limits absolute timing.
Complete traced peaks: idle 309312→201056 bytes, energetic 496180→483156 bytes;
post-warm incremental peak 308072→7704 bytes. Steady window storage 204800→192000 bytes;
rolling copies 204800→12800 bytes/block. No process RSS or whole-agent claim.

```sh
SPEAKER_TEST_LOG=0 SPEAKER_NO_LOCAL_CONFIG=1 SPEAKER_LIVE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_aec_delay_windows.py tests/test_aec_seam.py tests/test_input_calibration.py tests/test_denoise.py tests/test_sherpa_playback.py tests/test_sherpa_media_session.py tests/test_reference_recording.py tests/test_audio_frontend.py tests/test_apm_double_talk.py tests/test_barge_word_cut.py -q
```

Observed 467 passed/1 optional-model skip in 10.11s. Source pin/exact-window/default
synthetic delay and reset comparisons pass. No native models, devices, microphone,
private recordings, network model transfer, live probe or acoustic default change.

## Rolling retention applied (2026-10-10)

Valid until: keeper migration or another explicit retention instruction — then treat as history.

ADR-0236 supersedes earlier accumulation of owned full captures. Verified the retained 12192-pass log hash and canonical copies of all duplicated synthetic identity arrays, then removed prior integration worktree-logs and worker-logs: 657 files / 49,690,305 logical bytes. The old trees are gone; retention-pruned-20261010.json records history only. Compact original baseline receipts, model/evidence keepers, original recordings, protected live setup and unmerged mobile packet remain. Other owners’ storage changes are not attributed to this cleanup.

## Second-pass integration source freeze (2026-10-10)

Valid until: source changes or a new platform/physical result — then treat as history.

Merged startup96b5ab7, AEC1b7626f and latency8e9d39e with first-fragment7900835. Independent ownership reviews returned GO. Combined focused gate:514passed/2dependency warnings in6.71s. Review then reproduced/fixed Unicode-whitespace directive promotion; speech/markup/cancellation gate:121passed in1.53s. Actual producer/finalizer review required captured dispatch metadata for new stages, preserving legacy fallback, and found nonstream silent-cancellation at exhaustion. Final producer/close/cancellation gate:57passed in0.58s, including48 new cases and real console Echo finalization/fallback. These are headless tests, no model/audio/hardware claims. Full source qualification follows.

## Second-pass full qualification (2026-10-10)

Valid until: source changes or new physical/platform evidence — then treat as history.

Source ca2fe03375a1cf6b0b07cda9681387d22b81eda7: 12369 passed, 41 skipped, 9 dependency warnings, 225.76 s. CI LiveKit exclusions retained; no microphone/doctor/native platform run. New coverage includes actual console Echo finalization and unavailable-stage fallback. Changed 23 Python files pass scoped fatal-error Ruff; 38 docs checked with no findings; whitespace clean. Ignored current keeper logs/runs/desktop-pass-two-20261010 contains source-bound receipt, current runner, compact logs and public scheduling/baseline results; fresh hermetic pytest fixtures removed after result. Source-qualified worker branches merge into the integration ancestry; original live/mobile keepers remain excluded from cleanup.

## Third-pass long-session watchdog (2026-10-10)

Valid until: recorder/watchdog source or monitoring behavior changes — then treat as history.

Baseline 1bed9e2 rescans full metrics every tick. ADR-0241 uses per-observer monotonic cursors and exact unresolved/current snapshots, preserving full metric export, deadlines and maintenance callbacks. A measured initial-copy prototype regression was removed before acceptance; current source-pinned benchmark includes both initial and steady allocations. At 50,000 settled synthetic turns, repeated p50 20.37–22.06ms becomes 6.60–6.89us; traced steady400232→624B, initial400232→808B. Not agent latency/RSS. Existing+new gate113 passed in5.37s; no audio, native model or doctor. Independent review and combined qualification follow.

Watchdog review correction: cursor now acknowledges completed records only, preserving current→banked phase changes between ticks. Both reproduced missed-warning regressions pass; independent re-review GO. Final focused115 passed3.29s. Final compact benchmark replaces the superseded transient report:50kturns baseline6.23–6.86ms/current2.57–2.71us p50;steady400232→592B;initial400232→776B. Prior numbers remain historical, with host-load-sensitive absolute times. Current source hashes bind the replacement receipt.

Valid until: final-ASR model boundaries, work ownership or benchmark environment changes — then treat as history.

## 2026-10-10 — Retire obsolete final-ASR model stages (ADR-0243)

On pinned main 1bed9e2, actual `_finalize_and_dispatch` plus fakes reproduced:
revocation during punctuation still admitted offline create/accept/decode and
verifier calls; final callbacks were correctly suppressed only afterwards.
The exact existing work-current predicate now fences model boundaries, unwinding
without fail-open selection or verifier-health mutation after observed revocation.
An entered native call remains owned until return; no preemption/thread/model,
endpoint, threshold, raw/provenance or control-authority change is claimed.

Focused ASR/provenance/capture/diagnostic gates: **914 passed, 1 optional-model
skip in 22.76s**. Earlier narrow gate: 222 passed/1 skip in 5.51s; exact stream-release-count tightening then passed the 128-case new module in 1.45s. New deterministic
cases cover every model boundary with success/error return, exact stage retirement,
blocked-call ownership, stream release on the same worker, sole-worker successor
progress, foreign scope isolation and healthy exact decisions. Two initial test
fixtures were corrected to supply acoustic lineage for revisions and retain the
established offline selector's lowercase text; no production policy was changed.

The one reusable `tools.bench_asr_final_cancellation` setup and compact receipt
bind baseline helper Git source plus current helper hashes. Healthy decisions
match 48/48 baseline cases. Post-revocation calls at punctuation/create/accept/
decode fall 4/3/2/1 to zero. A synthetic verifier allocating exactly 1MiB after
offline-decode revocation falls from 1050248 to 2096 traced-peak bytes and from
242.811 to 21.153us paired p50. This is deliberately a fake workload, not native
latency/RSS or an agent speed claim. No-cost healthy fake p50 rises 67.376 to
70.944us; that predicate is a lambda/Event check, not full-engine native timing.
Four BLAS/OpenMP environment requests were one; CPU isolation was not established.
Only one current compact scratch receipt is retained, with pinned baseline in Git.

```sh
SPEAKER_TEST_LOG=0 SPEAKER_NO_LOCAL_CONFIG=1 SPEAKER_LIVE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_asr_final_cancellation.py tests/test_asr_final.py tests/test_asr_final_async.py tests/test_asr_segment.py tests/test_asr_verifier.py tests/test_asr_postprocess.py tests/test_asr_text_agreement.py tests/test_asr_text.py tests/test_final_trust_lineage.py tests/test_final_stt_profiles.py tests/test_final_preprocessing_cancel.py tests/test_sherpa_streaming_decode_owner.py tests/test_sherpa_streaming_decode_session.py tests/test_sherpa_media_session.py tests/test_sherpa_vad_final_gate.py tests/test_capture_replay_async_delivery.py tests/test_diagnostic_bundle.py tests/test_barge_confirm.py tests/test_barge_word_cut.py -q
```

No model, microphone, doctor, device, original recording, model transfer or live
run occurred. Native/platform/physical acceptance and protected owner evidence
remain open and untouched.

Valid until: TTS generation cancellation, DSP or native ownership changes — then treat as history.

## 2026-10-10 — Retire stopped synthesis before native/DSP work (ADR-0242)

A public synthetic Event/lock probe on 1bed9e2 measured one obsolete native entry
after model-lock wait, one cancelled callback DSP/direct sink offer and one
cancelled whole-clip DSP pass. The production FIFO independently prevented
stale audible output; these counters are wasted wrapper work, not audio escape.
Exact generation/stop checkpoints now reject those stages, giving zero for each
counter on the same synthetic probe. The compact record is
`docs/evidence/tts-generation-retirement.json`, with the protected baseline,
probe hash and synthesis/file source hashes; no full source/cache snapshot added.

Successor progress no longer needs an obsolete fake native render to finish.
Entered native cleanup retains its lock until true return; errors propagate once
and cannot trigger regeneration. Healthy PCM is bitwise identical under both
speaker-lock policies with unchanged emotion speed. Existing FIFO/receipt/fade/
resampling/cleanup ownership and live model/option selection are unchanged.
Native kernels and already-entered DSP remain nonpreemptible; later checkpoints
withhold known retired work/carry. No native RTF/physical latency claim follows.

Affected headless gate: tts_generation_retirement, sherpa_playback, tts_markup,
streaming_tts, audio_frontend: 224 passed in 3.06 s. The two old tests intentionally
expecting a stale first chunk were corrected to assert zero native entry/write;
no other contract was relaxed. New-test Ruff, scoped engine/playback lint and
diff checks pass. Independent read-only review GO. No native model, private
recording, microphone, doctor, route change, download or live run occurred.

Valid until: Ollama bridge completion, queue or SDK cleanup semantics change — then treat as history.

## 2026-10-10 — Remove local Ollama terminal poll (ADR-0244)

Audit of main 1bed9e2 found the local async bridge already bounds queued items,
isolates clients/loops and owns cancellation cleanup. Its natural done Event did
not notify Queue.get, however. Five public fake-local cases with response/client
cleanup already complete each paid 50.168–50.239 ms for terminal next(), p50
50.208 ms. The fix publishes a private notification after exact private loop
return and drains known-complete queues nonblockingly; full queues need no
notification slot or dropped output. A timed-empty race rechecks queued data/error
after observing done. No model/provider/pool/cap/locality change follows.

The first focused run exposed two error-settlement assertions: 2 failed,
26 passed in 2.07 s. Moving done to loop return allowed a queued provider error
to reach the caller just before done; terminal errors now wait for that same
owner completion, preserving the original exception object and allowing
cancellation to win. Focused rerun: **28 passed in 1.23 s**. Independent final
bridge review from desktop_review is GO, including this error boundary.

```sh
SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_ollama_terminal_wakeup.py tests/test_ollama_async_cancel.py tests/test_hedge_source_owner.py tests/test_hedge_chain.py tests/test_multi_provider_llm.py tests/test_llm_decision.py tests/test_llamacpp_cancel.py tests/test_startup_warm_priority.py tests/test_pretoken_cancellation.py tests/test_capability_stream_close.py tests/test_stream_speech_cancellation.py tests/test_llm_egress_policy.py tests/test_capability_context_isolation.py -q --basetemp=/home/dobo/work/_temp/codex__ollama-terminal-wakeup/pytest-qualified
```

**302 passed in 7.24 s**. Fake native-abort tests do not load native models.
The compact current paired result is docs/evidence/ollama-terminal-wakeup-2026-10-10.json;
the reproducible setup is tools/bench_ollama_terminal_wakeup.py. It extracts only
the two pinned Git bridge classes into the same public fixture environment and
binds source hashes, with five alternating adjacent cases per variant. After
known cleanup, terminal next() p50 was baseline 50.1314 ms versus current 0.0090 ms.
This is not model, whole-turn, physical audio or phone performance; no CPU
isolation was established. Reproduce with a fresh output path:

```sh
SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m tools.bench_ollama_terminal_wakeup --output <fresh-json>
```

No SDK request, daemon, network/cloud, private data, model download/inference,
microphone, doctor or device ran. Original baselines, source branches and live/
mobile keepers remain untouched; only compact public evidence enters Git.
Scoped fatal-error Ruff, new-file formatting and whitespace pass; docs report
38 files with zero findings, and STATUS remains at 120 lines.
