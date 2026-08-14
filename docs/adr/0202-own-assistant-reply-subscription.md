# ADR-0202: Own the Assistant reply subscription exactly

Date: 2026-08-14
Status: accepted
Refines: ADR-0186, ADR-0201
Supersedes: none

## Decision

Give each canonical Flutter `AssistantScreen` state one pure-Dart
`AssistantReplyOwner`. Inject `GemmaService.instance.reply` as its reply opener;
the opener must do no generation/chat work until the returned stream is
listened to. Keep the ADR-0201 `GemmaGenerationOwner` as the separate
app-lifetime lower owner in the canonical UI isolate.

At the widget boundary, admit one active exact `StreamSubscription<String>` and
at most one latest pending replacement. A third request replaces the pending
request without opening or listening to a backend stream. Publish an opaque
owner-keyed generation synchronously, and defer even hostile synchronous stream
callbacks until `start()` has returned so `Assistant` can publish that exact
generation before any callback can mutate UI or TTS state.

Require every token callback to match all four authorities: the callback's
exact reply generation, the widget's current reply generation, the reply
owner's current authority, and the existing exact LLM-turn plus TTS-playback
generation fence. Clear the turn-local TTS buffer when reply authority ends.
Flush its remaining tail only for natural completion that is still exact and
current; cancellation, supersession, failure, deadline, disposal, or stale
completion cannot enqueue that tail. Surface and log bounded failure codes
only, never raw provider exceptions.

On a new acoustic endpoint, typed replacement, STOP, `_stopSpeaking()`,
listening shutdown, or widget disposal, advance/fence reply authority and begin
exact reply cancellation before playback supersession, playback waits, plugin
cleanup, or playback-owner cleanup can suspend. Do this even when playback stop
is already deduplicated. Flutter `dispose()` is synchronous, so it begins the
memoized reply-owner close before TTS cleanup but does not claim to await its
receipt.

Cancel only the exact retained Dart subscription. Do not call provider-global
`GemmaService.cancelCurrent()` from `Assistant`: an old widget owner could
otherwise cancel a newer widget's generation. Cancelling the exact lower reply
subscription invokes that exact ADR-0201 stream's `onCancel`. The lower owner
prevents successor Dart chat construction until the predecessor's true source
terminal plus successful exact chat close; a revoked entered predecessor
additionally gets one explicit owner-port stop attempt. None proves native
cleanup, termination, or return. The upper owner's `done` receipt means immediate loss of UI
authority, while its later `cleanup`/`close` receipts prove only exact Dart
subscription settlement. They never prove lower/native stop, terminal, chat
close, model cleanup, or plugin return.

Keep upper receipts aggregate-only: outcome, ordinal, bounded counters, and
boolean lifecycle facts retain no prompt, token, or raw error. This is not an
app-wide privacy claim; the existing Assistant still retains `_lastUtterance`,
the rendered `_answer`, and bounded diagnostic logs for its separate UI and
debugging behavior.

Start one 120-second maximum lifetime at upper admission, including time spent
pending, and do not reset it on promotion. Give exact subscription cancellation
a separate 10-second deadline. Treat ambiguous listen, cancellation failure or
expiry, active lifetime expiry, and timer/ordinal uncertainty as sticky poison
for that widget owner. A pending request that expires or is displaced before
opening does no backend work and settles cleanly. Preserve the defensive upper
limits of 16 KiB prompt, 4 KiB token chunk, 64 KiB aggregate delivered text,
and 2,048 source data events. These duplicate the lower adapter envelope at a
different ownership seam; they do not bound SDK/native callbacks, tokens,
compute, memory, threads, return time, or plugin behavior.

The legacy mobile `_looksLikeStop` substring recognizer remains unchanged and
broader than the shared exact command contract. The separately prepared
ADR-0200 `AgentSession` plus process-TTS composition remains unlanded. This
slice resolves ADR-0201's stalled-source Assistant subscription residual only;
it does not complete mobile `AgentEvent`, task/tool/confirmation/memory, or
process-TTS convergence.

## Context / why

ADR-0201 made the lower Gemma stream/chat lifecycle conservative, but the old
`await for` in `Assistant` did not retain its `StreamSubscription`. Advancing a
turn fenced later UI mutations only after the stalled stream next emitted,
failed, or completed. A no-token source could therefore remain subscribed after
STOP or widget disposal, delaying ADR-0201's exact `onCancel` cleanup path.

Using the service-wide cancel method looked simpler but is not widget-safe. A
stale disposed widget and a newly constructed widget share the app-lifetime
Gemma service; the stale widget must cancel only its own subscription, not
whichever lower generation is globally newest. Exact subscription ownership is
the narrow seam that preserves replacement safety across widget owners while
leaving lower/Dart chat-admission gating with ADR-0201. Neither owner proves
native cleanup, termination, or return.

Awaiting lower cleanup in the UI authority path was also rejected. UI/TTS must
lose authority immediately even when cooperative plugin/native cleanup stalls.
Separating `done` from exact Dart-subscription cleanup lets replacement and
failure UI settle promptly while retaining conservative cleanup evidence.

## Consequences

- Every canonical widget has at most one active exact reply subscription and
  one latest pending request; displaced pending prompts do no backend work.
- STOP, replacement, listening shutdown, and disposal fence reply callbacks
  before playback/plugin waits. A stalled no-token lower reply receives exact
  Dart subscription cancellation instead of waiting for another source event.
- An old widget cancels only its own subscription. A two-owner headless test
  proves that this Dart action cannot cancel the newer owner's subscription;
  it is not a cross-widget native no-overlap proof.
- Upper exact-subscription settlement can precede ADR-0201's lower/Dart gate.
  The lower owner prevents successor Dart chat construction until the
  predecessor's true source terminal plus successful exact chat close; a
  revoked entered predecessor additionally gets one explicit owner-port stop
  attempt. Otherwise it poisons admission. None proves native cleanup,
  termination, or return.
- Widget disposal begins but cannot synchronously await upper close. Ambiguous
  upper cleanup poisons that owner; another widget owner remains a separate
  scope and still relies on ADR-0201 for shared lower serialization.
- The unchanged playback owner retains ADR-0186's cross-widget native-player
  uncertainty. The legacy substring STOP rule and the broader ADR-0200
  session/process-TTS composition remain open.
- Headless fakes and source-contract tests prove Dart ordering and retention
  only. No Flutter plugin, model, network, GPU, microphone, audio-device,
  mobile-native, APK, emulator, phone, latency, thermal, quality, or live
  behavior follows; APM/DTD used synthetic headless PCM.

## Verification

- **Combined focused receipt:** from `mobile/`, with
  `SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1` and low-impact
  `ionice -c 3 nice -n 19`,
  `/home/dobo/flutter/bin/flutter test --no-pub --concurrency=1 -r compact test/llm_generation_owner_test.dart test/llm_glue_test.dart test/assistant_reply_owner_test.dart test/assistant_reply_integration_contract_test.dart`
  passed `77/77`; the
  `/usr/bin/time -f 'wall=%e user=%U sys=%S rss_kib=%M'` wrapper reported wall
  `9.26s`, user `9.04s`, system `2.19s`, and maximum RSS `504600 KiB`.
- **Full mobile receipt:** the same low-impact environment/scheduling prefix
  with `/home/dobo/flutter/bin/flutter test --no-pub --concurrency=1 -r compact`
  passed `107/107`; wall `12.37s`, user `10.37s`, system `3.49s`, maximum RSS
  `467448 KiB`.
- **Full analyzer receipt:** `SPEAKER_TEST_LOG=0 nice -n 19 ionice -c 3 /home/dobo/flutter/bin/flutter analyze --no-pub`
  reported no issues in analyzer `1.5s`; wall `2.55s`, maximum RSS
  `203684 KiB`.
- **Non-writing format receipt:** Dart format checked all `8` code/test files
  with `0` changes in `0.08s`.
- **Repository regression receipt:** the standard low-priority APM/DTD command
  passed `6/6` in pytest `3.59s`; wall `4.99s`, maximum RSS `120672 KiB`.
- `git diff --check` was clean. Independent final owner and Assistant-wiring
  audits both returned GO.
- Frozen implementation/test SHA-256 values are:
  `1033c91ed7c10d281098d0e8ada98bd996db9144d8a0278f97ca23a43563cbb7`
  (`assistant.dart`),
  `6dc0955611eea1c3e8b9a48f9b62a0a265170b48482d413429ac0e0408c37218`
  (`assistant_reply_owner.dart`),
  `1fc1c010e35f51bb64a1779d91226c6238bf8d3a90581962dcc41d884fe14563`
  (`llm.dart`), and
  `2e5b0cf1e0068fb6f6aa2a0084a91aad7dff8a3175b40dba5bc75153eb08319c`
  (`llm_generation_owner.dart`); test hashes are
  `79a0d4af02e9268be68e624e6a6ece1029c8bf0743b0027a9243c99d135cbf34`,
  `3605e036146c1b6c99af3f52c5976c41f972eb9e63c19c70bf886cac28b0cf25`,
  `b69ee350546b03f4baf3afdd65f4a6e0460064e4f8b2fde108ab860217c8df13`,
  and `3cdcaa9defeddd0723b90fbe2609e7f6a4ec3ff4b54d51345aab8723b927b5e4`
  in the corresponding integration/upper-owner/lower-owner/glue order.
- The durable change inventory is exactly `16` paths: `8` code/test paths,
  `6` reconciled existing documents, ADR-0201, and this ADR. No task/handoff
  artifact is part of Git inventory.
