# ADR-0203: Own exact mobile ASR sessions and serialize Assistant listening

Date: 2026-08-14
Status: accepted
Refines: ADR-0020, ADR-0186, ADR-0201, ADR-0202
Supersedes: none

## Decision

Keep `AsrService.instance` as the app-lifetime ASR owner used by the shipped
app's canonical Flutter UI isolate, and issue one opaque `AsrSession` for each
listening generation. The object identity plus a service-private owner key—not
its bounded ordinal—is the authority for `ready`, `feed`, `endSession`, and
`cleanup`. Another FlutterEngine, Dart isolate, or direct plugin/worker caller
remains outside this owner.

`beginSession` synchronously revokes the previous session's Dart callbacks and
publishes the new exact session before asynchronous startup. Actual worker
admission is serialized behind the predecessor's exact cleanup receipt. A
successor may send its reset only after the predecessor settled true; a false
or timed-out receipt freezes uncertainty, prevents promotion, and makes later
same-service admission fail closed. The worker accepts only increasing bounded
session ordinals and exact increasing audio sequences, so stale reset, audio,
end, partial, endpoint, and credit messages cannot regain authority.

Fence worker launch with a one-way permit checked after model configuration
returns and before spawn, when the worker Hello arrives, and after a late spawn
becomes observable. Closing the app-lifetime service revokes that permit
synchronously. A late spawned transport is retired through its one memoized
close operation where it can be observed; a failed or timed-out close remains
false and is never retried as if it were clean.

Copy every admitted PCM16 chunk and bind it to its exact session and sequence.
Retain the existing worker envelopes: nonempty even-length chunks of at most
32,768 bytes, at most four unacknowledged chunks, at most 2,048 decode steps per
chunk, and at most 16 KiB UTF-8 for a partial or endpoint. Output is emitted
before its exact audio credit. A bound, callback, send, worker, or lifecycle
failure fences outward callbacks and publishes only bounded failure codes.

Treat `AsrSession.cleanup == true` as exact-session settlement evidence, with
three explicit branches:

- If the app ends or fails a session before its reset is sent, or no worker
  session exists for that ordinal, it revokes the callbacks and completes
  cleanup true locally. This is clean Dart no-worker-session settlement: no
  EndAck or session-specific worker release occurred, and it does not by itself
  settle a separately pending or late app-lifetime worker startup.
- On the normal reset-sent end path for an admitted stream, the app receives
  the exact `AsrWorkerEndAck(released: true)` only after that session's
  worker-side `OnlineStream.free()` call returned. The app-side callbacks were
  already revoked before the end command was sent.
- A typed `AsrWorkerSessionFailure(sessionReleased: true)` may instead report a
  clean pre-resource failure or a worker-side failure whose exact session
  stream-release call returned. This is intentionally not called an EndAck.
  `sessionReleased: true` alone does not authorize a higher ordinal: retry also
  requires `workerHealthy: true` and a service/transport that remains
  unpoisoned.

None of these branches proves a C/C++ destructor completed, native work
terminated, native memory was returned, or plugin threads stopped. A false receipt, an
untyped/ambiguous failure, or a deadline remains frozen uncertainty and blocks
successor promotion. App-lifetime cooperative close additionally orders any
current stream free before recognizer free and emits `AsrWorkerShutdownAck`
only after both Dart wrapper calls returned; even that is not native
destruction or termination proof. A widget must not close the app-lifetime ASR
service.

Give each canonical `AssistantScreen` state one
`AssistantListeningOwner<AsrSession, _AssistantListeningCapture>`. It owns one
active reconciliation and at most one latest pending ON/OFF intent. Replacement
publishes the latest desired state and synchronously revokes the old exact
generation; a displaced pending intent performs no permission, route, ASR,
recorder, subscription, or callback work. Use one 30-second admission lifetime
and one 10-second cleanup deadline. Active ambiguity or cleanup expiry poisons
that widget owner and retains uncertain resources instead of admitting a
successor; an unstarted displaced or expired intent settles without resources.

Keep plugin-specific state in a cleanup-capable adapter that owns the exact
`AudioRecorder` and exact ASR session/capture lanes, borrows the widget's
playback owner only to coordinate the audio route, and retains only a
`WeakReference` to widget state. It must not retain transcripts, answers, or UI
callbacks, and it does not own or close the app-global TTS service. Admit
listening in this order:
permission, voice route, exact ASR session plus cleanup watch, exact session
ready, capture-lane publication, recorder `startStream`, then exact Dart stream
subscription. Publish the capture lane before calling `startStream` so a throw
or other ambiguous start can be recovered only through that exact lane.

Close every capture callback over its immutable listening generation,
`AsrSession`, and capture lane. Before feeding PCM or mutating UI, require the
same lane, the same session, the current listening generation, and current
owner authority. Never redirect stale capture work through a mutable
"current session." A rejected ASR feed or recorder source error synchronously
fences that exact generation and starts its memoized exact subscription
cancellation. Publish a failed capture terminal only after that cancellation
returns true. A failed terminal is a cleanup trigger, not independent
subscription-release evidence; a natural source `done` may be the terminal
evidence for that exact source.

On revoke, replacement, OFF, source failure, or widget close, synchronously
request exact ASR session end, exact capture-subscription cancellation, and
exact recorder stop—in that order and all before awaiting any of them. Memoize
each operation so reentrant and concurrent cleanup joins the same receipt. A
listening revoke callback also synchronously fences reply and playback before
checking whether its UI generation is still current; displaced listening
authority cannot preserve a stale outward effect. A
listening cleanup receipt is true only when:

- the exact capture source ended naturally or its exact Dart subscription
  cancellation returned successfully;
- the exact recorder `stop()` call returned successfully; and
- the exact ASR end request was accepted and that exact
  `AsrSession.cleanup` receipt settled true.

This proves only local Dart session/subscription/recorder call settlement. It
does not prove recorder-plugin return below Dart, microphone/device release,
audio-route restoration, Sherpa/native teardown, or live behavior. Failed or
expired cleanup stays false, poisons that owner, and retains the exact
uncertain lane for diagnosis.

On Flutter `dispose`, clear reply and listening UI authority and synchronously
start exact reply-owner close, listening-owner close, and playback-owner close
before any asynchronous wait. Dispose the recorder only if listening close
reports `exactResourcesSettled`; otherwise retain it rather than manufacture a
cleanup claim. The widget does not call `AsrService.close()` or redirect a
stale widget through provider-global cancellation. ADR-0203 also does not call
or own app-global `TtsService.dispose()`; process-TTS ownership remains in the
separate unlanded ADR-0200 composition candidate. This ordering does not prove
arbitrary cross-widget/process recorder or native-player exclusion: a new
widget may exist while an old plugin resource remains uncertain, while the
shared ASR service supplies only its narrower exact-session serialization.
The widget's unawaited `TtsService.instance.ensureReady()` request when
listening starts is best-effort warming outside the listening adapter; it is
neither admission evidence nor service ownership or cleanup authority.

Resolve ADR-0202's open legacy STOP residual at the completed/typed command
seam: both ASR endpoints and typed submissions use only the shared exact
`isStopCommand` contract. Remove substring STOP matching. A nonempty partial of
at least two characters remains the separate generic transcription-barge
signal and is not a typed STOP decision. Extend the shared exact STOP fixture
with these eight required negatives: `nonstop`, `stopwatch`, `pit stop`,
`stop sign`, `do not stop talking`, `quiet room`, `cancel culture`, and
`don't stop`. None may gain STOP authority by substring.

Keep listening and ASR receipts aggregate/content-free. They may contain a
bounded ordinal, outcome, counters, booleans, and bounded failure code, but no
PCM, prompt, transcript, token, or raw provider/plugin exception. This is not
an app-wide transcript-privacy claim: `Assistant` still retains and renders its
current partial, `_lastUtterance`, `_answer`, and bounded diagnostic logs.

Do not treat this lifecycle slice as multilingual support or ASR-quality
evidence. The shipped mobile streaming Zipformer is English-only. Shared
control normalization lowercases, retains only ASCII `a-z` plus spaces, then
collapses spaces and trims; digits and non-ASCII characters are discarded. A
future bounded mobile ASR packet should reuse the repository's
existing/retained GSC, ECCC, DEMAND, NOTSOFAR,
and PriMock assets and their
current source locks/materializers, with fixed English-only case/byte bounds,
outside-Git publication, exact provenance, and aggregate results. External
sources were researched for that future packet, but this slice downloads or
materializes none. A Romanian packet is blocked until a separate decision pins
a multilingual model/tokenizer, language labels, Unicode normalization, and
metrics; do not reuse the English-only model to imply Romanian coverage.

The broader mobile `AgentEvent`, mode, task/tool/confirmation, memory, and
ADR-0200 `AgentSession`/process-TTS composition remains open. ADR-0186's
cross-widget native-player uncertainty and ADR-0201/0202's native Gemma limits
also remain unchanged.

## Context / why

The previous mobile ASR service exposed mutable process-wide callbacks plus
reset/feed/stop operations. A rapid ON/OFF/ON sequence, source error, or widget
reconstruction could let late worker or recorder callbacks consult a newer
mutable session and could begin a successor before the predecessor's exact
cleanup settlement was known. Manual `_audioSub.cancel()`, recorder stop, and ASR
reset also supplied no single exact generation tying those cleanup results
together.

Closing the whole ASR service from a widget was rejected because the worker is
app-lifetime shared state: a stale widget must not destroy the service under a
new widget. Treating recorder `onError` as release was rejected because an
error says nothing about subscription cancellation. Treating `stop()`,
`free()`, an isolate exit, or a timeout as native destruction was rejected
because each is only observable Dart/port-level evidence. The conservative
session token, serialized admission barrier, exact capture lane, and sticky
uncertainty preserve a clear proof boundary.

## Consequences

- The canonical widget has one active listening reconciliation plus one latest
  pending desired state; repeated toggles cannot build a FIFO of plugin work.
- A successor may head-of-line wait for exact predecessor cleanup. A timeout or
  ambiguous release chooses unavailable/poisoned listening over possible
  overlap.
- Stale ASR, recorder, and UI callbacks lose authority synchronously. Exact
  callback identity is independent of display booleans such as `_alwaysOn` and
  `_listening`.
- Normal session cleanup and app-lifetime worker close now have explicit Dart
  receipts and ordering. They remain deliberately weaker than native teardown.
- Recorder disposal is conditional on exact local cleanup; uncertainty may
  retain plugin objects until process exit. Cross-widget/process plugin
  exclusion still needs a plugin contract or a separate shared owner.
- Completed/typed STOP matches the shared Python/Dart golden contract. Generic
  partial barge-in remains intentionally broader and does not grant command
  authority; eight substring-bearing negative fixtures remain non-STOP.
- The future bounded English data packet can exercise command/noise,
  far-field, and two-role strata without new downloads, but it is separate
  from this lifecycle proof and from any model/device/live qualification.
- Full mobile semantic/runtime convergence remains open.

## Verification

The receipts below are frozen against the final integrated code/test bytes.

- **Focused mobile lifecycle gate:** from `mobile/`, with
  `SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1` and low-impact
  `ionice -c 3 nice -n 19`, run
  `/home/dobo/flutter/bin/flutter test --no-pub --concurrency=1 -r compact test/asr_session_owner_test.dart test/assistant_listening_owner_test.dart test/assistant_reply_owner_test.dart test/assistant_reply_integration_contract_test.dart test/llm_generation_owner_test.dart test/llm_glue_test.dart test/golden_contract_test.dart`.
  It passed `170/170`; wall `8.39s`, user `7.81s`, system `2.45s`, maximum RSS
  `481872 KiB`.
- **Full mobile gate:** the same environment/scheduling prefix with
  `/home/dobo/flutter/bin/flutter test --no-pub --concurrency=1 -r compact`.
  It passed `198/198`; wall `9.31s`, user `8.37s`, system `3.06s`, maximum RSS
  `491632 KiB`.
- **Full analyzer:**
  `SPEAKER_TEST_LOG=0 nice -n 19 ionice -c 3 /home/dobo/flutter/bin/flutter analyze --no-pub`.
  It reported no issues in analyzer `1.7s`; wall `2.52s`, maximum RSS
  `306516 KiB`.
- **Shared Python command-contract gate:** from the repository root, run
  `SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_golden_contract.py -q`.
  Together with the Dart `test/golden_contract_test.dart` case in the focused
  gate, this checks both consumers of `tests/golden/commands.json`. It passed
  `34/34` in pytest `0.13s`; wall `0.45s`, maximum RSS `35584 KiB`.
- **Non-writing format:** run Dart format with
  `--output=none --set-exit-if-changed` over the exact final task-owned Dart
  code/test inventory. It checked `12` files with `0` changes in `0.10s`; wall
  `0.12s`, maximum RSS `71168 KiB`.
- **Repository regression:** run the standard low-priority APM/DTD gate from
  the repository root. It passed `6/6` in pytest `1.94s`; wall `2.68s`, maximum
  RSS `121252 KiB`; it used synthetic headless PCM only.
- `git diff --check`, the conflict-marker scan, and the exact `100`-line
  `STATUS.md` gate are clean. The shared `main` checkout is clean. Independent
  ASR and final integration audits returned GO; the latter audited exact patch
  SHA-256
  `27747d1226f620c6fd9af00c42154fba3a8e594a532e1473111db16faf1bbba7`.
- Frozen implementation/test SHA-256 values are recorded below. ADR-0201 and
  ADR-0202 remain byte-for-byte frozen at their listed hashes; final hashes for
  ADR-0203 and the six reconciled current-truth documents are recorded in the
  ignored task result so ADR-0203 does not contain a self-hash.

  ```text
  3f711163a8834426edf396d4c65c86cb85a7b601b03a4e0b1af16cc2b0aee1df  mobile/lib/asr_isolate.dart
  2908ce2f7ebd9a5db1d8ec09722ec665e610764a74f3c64999c61cf9f5abc5b3  mobile/lib/assistant.dart
  1fc1c010e35f51bb64a1779d91226c6238bf8d3a90581962dcc41d884fe14563  mobile/lib/llm.dart
  b76b1e2a5f7d18b986f3f14a5926f3b54a1a74d5e2b90d2f0649267fd7e2ebfe  mobile/lib/assistant_listening_owner.dart
  6dc0955611eea1c3e8b9a48f9b62a0a265170b48482d413429ac0e0408c37218  mobile/lib/assistant_reply_owner.dart
  2e5b0cf1e0068fb6f6aa2a0084a91aad7dff8a3175b40dba5bc75153eb08319c  mobile/lib/llm_generation_owner.dart
  92e2e7df3ddc232505f8f28f6b6db988664b5b13dd5410ebaf6c33d573b74caf  mobile/test/asr_session_owner_test.dart
  5bf4d5faac03c8f6a3467f615d810b2d8572d728807b4993cdfada2222d0d5e0  mobile/test/assistant_listening_owner_test.dart
  c4d7b52d318ebc6f2f80963715ca8b7690d8f09c048091e476f22931f507daf0  mobile/test/assistant_reply_integration_contract_test.dart
  3605e036146c1b6c99af3f52c5976c41f972eb9e63c19c70bf886cac28b0cf25  mobile/test/assistant_reply_owner_test.dart
  b69ee350546b03f4baf3afdd65f4a6e0460064e4f8b2fde108ab860217c8df13  mobile/test/llm_generation_owner_test.dart
  3cdcaa9defeddd0723b90fbe2609e7f6a4ec3ff4b54d51345aab8723b927b5e4  mobile/test/llm_glue_test.dart
  dc344eb7e56b95be37a75588703c4f187987197137e34d07dc0fbd04bbce9ebb  tests/golden/commands.json
  eab77f714eb7199902077aa4126eadad2ececbd8e53ca85755805fb0297e47ef  docs/adr/0201-bound-mobile-gemma-generation-lifecycle.md
  017bf3cf8c7fd982ececb24f9917ebd45aee7a4d04d263828c51ded9a69c41be  docs/adr/0202-own-assistant-reply-subscription.md
  ```

- The durable change inventory is exactly `22` paths: `12` Dart code/test
  paths, the shared golden JSON, `6` reconciled existing current-truth
  documents, ADR-0201, ADR-0202, and ADR-0203. The ignored task brief/result and
  all `/tmp` artifacts are outside that inventory.
- No Flutter plugin runtime, Sherpa model, Gemma model, network, download, GPU,
  microphone, recorder device, audio output device, APK, emulator, physical
  phone, latency, thermal, quality, multilingual/Romanian, or live path ran.
  APM/DTD, if included in the final receipt, uses synthetic headless PCM only.
