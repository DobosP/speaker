# ADR-0201: Bound mobile Gemma generation and chat lifecycle

Date: 2026-08-14
Status: accepted
Refines: ADR-0020, ADR-0186
Supersedes: none

## Decision

Make one shared pure-Dart `GemmaGenerationOwner` the app-lifetime generation
and chat owner used by `GemmaService.instance` in the shipped app's canonical
Flutter UI isolate. This is not an operating-system owner: another
FlutterEngine, Dart isolate, or direct plugin caller can bypass its static
registry.

Validate every prompt before listening and cap it at 16 KiB of UTF-8. Reserve
one opaque exact generation synchronously when its reply stream is first
listened to, before chat construction, prompt admission, or generation entry.
Creating but never listening to a stream performs no chat/model work.

Admit one active generation and at most one latest pending replacement. A newly
listened generation immediately becomes the latest outward authority, fences
the old consumer stream, revokes the active generation, and replaces any older
pending request. There is no FIFO or unbounded waiter population; a displaced
pending request performs no factory work. Only the still-latest pending request
may construct a chat, and only after the exact predecessor proves release.

Consumer cancellation and explicit `GemmaService.cancelCurrent()` synchronously
fence outward output while cleanup continues independently. Ordinary stream
pause is backpressure, not cancellation: keep draining the exact upstream
source into the owner's bounded paused-text buffer, flush it on resume only if
the generation is still authoritative, and drop it on cancellation or
supersession.

For a revoked generation that entered generation, invoke exactly one memoized
explicit stop through its `GemmaChatPort`, then await the exact source's real
`done` terminal; a source error is not a terminal receipt. Close the exact chat
once after that terminal and release only on exact close success. A settled
explicit-stop error may recover only when an independent true source terminal
arrives and the exact chat subsequently closes successfully.

Treat ambiguous chat construction, generation entry/listening, terminal state,
or chat close as retained owner poison. A code-owned 120-second lifetime starts
at admission. Expiry during active startup, prompt admission, generation,
stop/drain, or chat close fences output and permanently poisons this owner even
if late cleanup later returns; the timer cannot fabricate terminal or close
proof. Expiry of a pending request that never started settles that request
cleanly and performs no factory work.

The owner observes the adapter's `ModelResponse` data events, including
non-text responses. Cap those owner-observed events at 2,048, each text event at
4 KiB UTF-8, and total forwarded or pause-retained text at 64 KiB UTF-8. The
first excess event is not published and enters the exact stop/drain/close path.
These are adapter/Dart retention bounds, not counts of native callbacks and not
SDK/native memory, compute, thread, token, or return-time bounds.

Use one fresh chat and the existing non-greedy sampling settings for each
request that becomes active and reaches chat construction. Serialize readiness
and make service disposal one-shot. Owner
closure fences late model activation; a late model must close exactly or remain
retained as uncertainty. Do not close the shared model until generation cleanup
is proven. Treat every thrown GPU or CPU activation as permanently ambiguous:
the native engine may have been allocated without returning a closable Dart
model, so latch same-isolate service unavailability, do not try the other
backend, refuse re-readiness, and make disposal report unproven activation
cleanup rather than claiming or attempting model close. Only an actual clean
`null` GPU result may fall back to CPU. A failed model close is retained and not
retried through another dispose caller. Model preparation, download, install,
activation, and final model close have no code-owned hard deadline and remain
outside the generation's 120-second timer.

The cached dependency source inspected for this slice is
`flutter_gemma` 0.16.5, resolved under the declared `^0.16.1` constraint and an
ignored `pubspec.lock`; 0.16.5 is evidence for this checkout, not a durable
build pin. In that cached source, activation can throw after native engine
allocation without returning a closable Dart handle, and `.task` session close
defensively attempts platform stop again and may swallow Android cancellation
errors before closing. Therefore activation throws retain uncertainty, the
generation contract is one explicit owner-port stop rather than one total
native stop, and no native termination follows from the fake evidence.

Do not change `mobile/lib/assistant.dart` in this slice. Its turn/dispose fences
stop consuming only when the existing `await for` next receives a token, error,
or terminal. Immediate STOP or widget-dispose cancellation of a stalled
no-token Gemma source is not claimed. The later ADR-0200 composition must
explicitly own and cancel the reply subscription before making that claim.

The separately prepared ADR-0200 mobile session/process-TTS integration is not
landed. Its pre-slice `mobile/lib/llm.dart` and
`mobile/test/llm_glue_test.dart` were byte-identical to this branch's base, so
later code composition has a clean base; documentation and combined gates still
require explicit reconciliation. Their respective base SHA-256 digests were
`ab1e604fe5cbf25f36a31198fbe7451e9e98c853e1b55902ec84c65892dccb90`
and `adb775ca0b5c6e543bf748223ce4b8919f13a3ed22003a735566386114c5d7b9`.

## Context / why

The old `GemmaService.reply()` async generator created a fresh SDK chat, added
the prompt, and yielded its response stream, but never stopped or closed that
chat. Abandoning the Dart consumer fenced only later UI mutation; it did not
prove the SDK stream terminal or release the chat. A replacement turn could
therefore enter another chat while the old SDK/native generation remained.

A BUSY-only drop loses the newest valid turn, while a mutex/FIFO retains an
unbounded population of stale prompts. Latest-wins active-plus-one-pending
admission preserves the newest request without starting it beside an uncertain
predecessor. Cancelling the upstream subscription at ordinary pause was also
rejected because `await for` may pause during normal loop-body backpressure;
bounded owner-side draining preserves that normal streaming behavior.

Best-effort stop, source error, timeout, or consumer cancellation cannot prove
native return. Closing before the exact source terminal can confuse lifecycle
state, while releasing without successful exact chat close can admit overlap.
The conservative terminal/close receipts and sticky poison choose bounded
resource uncertainty over silent reuse.

## Consequences

- The canonical UI isolate has at most one active chat and one latest pending
  prompt. Repeated supersession is bounded and stale pending prompts do no work.
- Cancellation/supersession prevents stale publication immediately, but an
  entered cooperative SDK call may head-of-line block the latest replacement
  until source terminal and chat close—or poison the owner at the deadline.
- Ordinary pause keeps upstream draining and retains at most the same 64 KiB
  text envelope; resumption does not revive output after supersession.
- Natural source completion closes its exact chat without an owner stop. A
  revoked generation has one explicit owner-port stop; SDK close may
  defensively perform another native stop internally.
- The 120-second deadline is a fail-closed availability bound, not a hard kill.
  Active expiry disables later same-isolate replies until app restart.
- A thrown activation permanently disables same-isolate readiness and forbids
  CPU fallback or model-close claims; only a clean GPU `null` permits CPU.
- Preparation/readiness/model close have no code-owned hard deadline; another
  engine/isolate remains outside this owner.
- Full mobile `AgentEvent`, task, tool, confirmation, memory, process-TTS, and
  Assistant subscription convergence remain open.
- Headless fakes prove Dart ordering and retention only; they cannot establish
  plugin/native cancellation, session destruction, GPU behavior, latency,
  thermal behavior, quality, or audible/live behavior.

## Verification

- **Frozen focused receipt:** from `mobile/`,
  `SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 19 ionice -c 3 /home/dobo/flutter/bin/flutter test --no-pub test/llm_generation_owner_test.dart test/llm_glue_test.dart -r compact`
  passed `44/44` (`29` owner plus `15` service/bootstrap; Flutter-reported
  `00:05`). Fakes cover natural release,
  cancellation, ordinary pause, latest-wins/reentrant supersession, one explicit
  stop, stop-error recovery, error-versus-terminal separation, exact close,
  startup/close/deadline poison, pending expiry, all four bounds, serialized
  bootstrap, disposal fences held `hasModel`, download, model-path, install,
  and activation stages, clean-null-only GPU-to-CPU fallback,
  thrown-activation uncertainty, late model retirement, and one-shot disposal.
- **Frozen scoped-analyzer receipt:** from `mobile/`,
  `SPEAKER_TEST_LOG=0 /home/dobo/flutter/bin/cache/dart-sdk/bin/dart analyze lib/llm_generation_owner.dart lib/llm.dart test/llm_generation_owner_test.dart test/llm_glue_test.dart`
  reported `No issues found!`.
- **Frozen format receipt:** from `mobile/`,
  `/home/dobo/flutter/bin/cache/dart-sdk/bin/dart format --output=none --set-exit-if-changed lib/llm_generation_owner.dart lib/llm.dart test/llm_generation_owner_test.dart test/llm_glue_test.dart`
  checked `4` files with `0` changes in `0.05s`.
- **Frozen full Flutter receipt:** from `mobile/`, the same low-impact
  environment and scheduling prefix with
  `/home/dobo/flutter/bin/flutter test --no-pub --concurrency=1 -r compact`
  passed `74/74` (Flutter-reported `00:08`).
- **Frozen full-analyzer receipt:** from `mobile/`,
  `SPEAKER_TEST_LOG=0 nice -n 19 ionice -c 3 /home/dobo/flutter/bin/flutter analyze --no-pub`
  reported no issues in `1.8s`.
- **Frozen repository regression receipt:** from the repository root,
  `SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 19 ionice -c 3 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_apm_double_talk.py -q`
  passed `6/6` in `3.48s`.
- `git diff --check` was clean after applying the documentation patch.
- Read-only cached-source inspection covered resolved `flutter_gemma` 0.16.5
  under the declared `^0.16.1` range and ignored lock; it does not pin future
  dependency resolution or execute the package.
- No Flutter plugin runtime, model, GPU, network, microphone, audio device, APK,
  emulator, physical phone, native session, latency, quality, thermal, or live
  path ran.
