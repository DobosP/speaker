Valid until: this branch lands or is superseded — then treat as history.

# Task result — bounded session actor

## Summary

Implemented one core-free, lock-protected `SessionActor` for the Python
control plane. It allocates session/turn identity after accepted STT finals,
carries acoustic revision and speech-cancel generation through executable
tasks and production TTS, and bounds task, mutating-tool, playback, remembered
task, and idempotency admission.

Mutating `CapabilitySpec.side_effecting` calls now receive generic
`tool_call_id` and `idempotency_key` values. Their explicit tool cancellation
Event becomes independent after provider start: speech cancellation prevents
an unstarted mutation from launching, while a provider-started mutation may
finish with stale speech suppressed. The bounded session idempotency record
prevents the same turn/step from executing again. A winning task timeout
atomically retires future/unstarted tool work without overriding a provider or
completion that already won.
Timeout apology identity/publication is best-effort: capacity failure omits the
speech claim but never blocks cancellation lifecycle publication or queue drain.

Task bindings are reclaimed only after explicit controller release, including
queued, confirmation, pending-output, and auxiliary ownership. Playback
requires an exact four-field task binding; partial, malformed, or forged
identity fails closed. A single lease covers preparation through legacy sink
return or durable history transfer, including exact task-owner retention and
stage/receipt failure cleanup. Malformed lifecycle identity remains explicit
through TTS and cannot enter the all-absent compatibility path.
Terminal shutdown fences new work, signals every tool, and boundedly drains
cooperative providers before watch, reminder, or memory dependencies close.
Concurrent stop callers wait for that whole teardown. Receipt
outcome/text/sample fields are validated before fragment mutation.

Capture, VAD, endpointing, recognition, and acoustic revision gating remain
outside the actor. No monolith, device/audio I/O, model, or central event-bus
rewrite was added.

## Files changed

- `always_on_agent/session_actor.py` — typed identities and bounded ownership.
- `always_on_agent/tasks.py` — task binding, tool admission/cancellation, and
  lifecycle/TTS identity propagation.
- `always_on_agent/supervisor.py` — one actor per session, accepted-final turn
  binding, actor-owned speech generation, auxiliary TTS/playback admission,
  explicit binding release, bounded tool drain, and tool cancellation seam.
- `always_on_agent/capabilities.py` — tool call/idempotency fields in opt-in
  invocation observation.
- `always_on_agent/__init__.py` — public control-plane identity exports.
- `core/runtime.py`, `core/playback_history.py` — serialized teardown, bounded
  sink admission, guarded ownership transfer, forced-failure receipt cleanup,
  and playback identity retention.
- `tests/test_session_actor.py`, `tests/test_playback_receipts.py`,
  `tests/test_core_runtime.py` — deterministic ownership/bounds, task-binding
  lifecycle, start/cancel races, shutdown drain, malformed identity/receipts,
  interruption/idempotency, and receipt-release regressions.
- `docs/adr/0086-bound-post-recognition-session-ownership.md` — decision,
  tradeoffs, and deferred work.
- `STATUS.md`, `docs/agent-map.md` — current truth and routing.

## Commands run and exact results

Final focused headless gate:

```text
SPEAKER_TEST_LOG=0 PYTHONPYCACHEPREFIX=/tmp/speaker-session-pycache \
nice -n 10 /home/dobo/work/speaker/.venv/bin/python -m pytest \
-p no:cacheprovider \
tests/test_session_actor.py tests/test_pretoken_cancellation.py \
tests/test_owner_verified_actions.py tests/test_interrupt_race.py \
tests/test_playback_history.py tests/test_playback_receipts.py \
tests/test_final_preprocessing_cancel.py tests/test_device_tools_runtime.py \
tests/test_reminders.py tests/test_trusted_apps.py tests/test_core_runtime.py \
tests/test_always_on_agent.py tests/test_never_stuck.py \
tests/test_bounded_queues.py tests/test_confirmation_ttl.py \
tests/test_load_elastic_admission.py -q

321 passed in 14.50s
```

Combined post-rebase focused gate (SessionActor, public provenance, and APM):

```text
SPEAKER_TEST_LOG=0 PYTHONPYCACHEPREFIX=/tmp/speaker-session-pycache \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false nice -n 19 \
/home/dobo/work/speaker/.venv/bin/python -m pytest -p no:cacheprovider \
<session-focused files> tests/test_public_voice_fixtures.py \
tests/test_public_stt_eval.py tests/test_apm_double_talk.py -q
391 passed in 14.96s
```

Fresh full combined repository gate, pinned to four CPUs:

```text
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
taskset -c 28-31 nice -n 19 \
/home/dobo/work/speaker/.venv/bin/python -m pytest -p no:cacheprovider \
-m "not real_model" -q
5801 passed, 13 skipped, 20 deselected, 9 warnings in 112.57s
```

Static/import/whitespace gate:

```text
PYTHONPYCACHEPREFIX=/tmp/speaker-session-pycache nice -n 10 \
/home/dobo/work/speaker/.venv/bin/python -m py_compile \
always_on_agent/session_actor.py always_on_agent/tasks.py \
always_on_agent/supervisor.py always_on_agent/capabilities.py \
core/playback_history.py core/runtime.py tests/test_session_actor.py \
tests/test_interrupt_race.py tests/test_never_stuck.py \
tests/test_playback_receipts.py tests/test_core_runtime.py
/home/dobo/.local/bin/ruff check --no-cache \
always_on_agent/tasks.py always_on_agent/session_actor.py \
always_on_agent/supervisor.py core/runtime.py core/playback_history.py \
tests/test_session_actor.py tests/test_interrupt_race.py \
tests/test_never_stuck.py tests/test_playback_receipts.py \
tests/test_core_runtime.py --select E731,F841
git diff --check
if rg -n "^(from|import) core" always_on_agent; then exit 1; fi

exit 0; Ruff: All checks passed; other checks: no output
```

Incremental compatibility gates also completed green at 204 passed, 122 passed,
93 passed, 20 passed, and smaller targeted subsets. The first pytest attempt
could not create worktree-local `logs/tests` under the execution sandbox; final
runs intentionally used `SPEAKER_TEST_LOG=0` and disabled the pytest cache. One
early focused run exposed a direct-invocation compatibility failure; it was
fixed before the recorded green gates.

No real-model, recorded-audio, network, GPU, or live-hardware test was run.

## Risks and manual review

- Generic idempotency is bounded and session-local. Process restart or eventual
  eviction requires mutating backends to retain their own durable key, as the
  reminder and trusted-app implementations already do.
- A duplicate generic call is rejected before provider execution and currently
  becomes a controlled task failure; arbitrary provider result bodies are not
  cached in the actor.
- A provider-started synchronous/native tool may ignore explicit or shutdown
  cancellation until its own timeout/return. Actor tool capacity and the
  existing provider semaphore bound survivors, but Python cannot force-kill
  them; shutdown logs survivors after its short bounded drain.
- The central event bus remains unbounded, and remote/mobile brains do not yet
  implement this actor contract.
- Manual live A/B still needs to check natural barge timing, real sink receipts,
  STT quality during/after playback, multiple voices, and physical behavior
  while a confirmed mutation completes. No live claim is made here.

## Merge recommendation

The branch is rebased onto public-v2 `main`; its focused and fresh full
deterministic gates are green. ADR-0086 and STATUS are updated. It is ready for
fast-forward landing; no worker pushed the branch.
