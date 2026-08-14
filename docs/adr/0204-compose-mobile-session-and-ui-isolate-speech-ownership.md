# ADR-0204: Compose mobile session authority with UI-isolate speech ownership

Date: 2026-08-14
Status: accepted
Refines: ADR-0020, ADR-0186, ADR-0201, ADR-0202, ADR-0203
Supersedes: none

## Decision

Make one app-root `AgentSession` the authoritative mobile
transcript-to-decision boundary and keep it alive across canonical Assistant
widget replacement. Compose each Assistant instance beneath that session with a
nullable exact `TtsProcessLease` from the single no-wait speech-output registry
used by shipped paths in the canonical Flutter UI isolate. Semantic input is
authoritative independently of optional speech output: a missing, BUSY,
revoked, or poisoned lease never suppresses partial/final/typed observation,
STOP, mode switching, ignore, or a typed unavailable result.

Keep the mobile `AgentEvent`/decision surface a closed, bounded projection, not
the Python runtime bus. It contains seven modes, eleven intent kinds, bounded
scalar events, exact input/event ordinals, and an atomic trace of STT, intent,
and optional control. Validate raw input against the 16 KiB UTF-8 bound before
trimming or retaining it. Construct and validate the whole transition before
one state commit publishes the resulting mode, generation, sequence, and
opaque current `AgentTurn`; retain no transcript or event history in the
session itself.

Every mobile partial is a semantic `ignore`, including partial text that
normalizes to STOP, mode, confirmation, denial, or work syntax. While speech is
active, the existing nonempty at-least-two-character partial path may still
request generic acoustic barge-in after semantic observation; that path owns no
command classification. Only completed ASR and typed input may use the shared
ASCII command contract. After lowercasing, removing every character except
`a-z` and space, collapsing spaces, and trimming, its exact STOP set is:

```text
stop
cancel
cancel that
quiet
stop talking
stop speaking
be quiet
```

Retain the shared negative lock: `nonstop`, `stopwatch`, `pit stop`,
`stop sign`, `do not stop talking`, `quiet room`, `cancel culture`, and
`don't stop` remain non-STOP. Digit-only, full-width-only, and the listed
Romanian forms gain no mobile control authority. No broader Unicode
equivalence or support claim follows.

Do not claim partial parity with desktop. The Python runtime deliberately keeps
exact STOP partial-authoritative, while a separate desktop safeguard rejects
all other partial mode, confirmation, denial, and task decisions before they
gain authority. The shared `tests/golden/commands.json` contract covers the
narrow completed/typed ASCII STOP slice. The existing
`tests/golden/speech_analyzer_contract.json` remains a desktop golden: do not
overlay or extend it with the frozen candidate version, and do not claim it as
mobile or general cross-runtime analyzer parity.

Apply every nonempty admitted final (completed ASR or typed) in this order:

For an admitted final, `nonempty` refers to the raw string passed to
`AgentSession`. Punctuation-only input remains raw-nonempty and therefore
crosses this fence. An empty endpoint and trim-empty typed input are rejected
by Assistant before admission; direct
`AgentSession.acceptFinal('')` is observation-only and preserves the input
generation and current turn.

1. enforce the raw 16 KiB UTF-8 bound;
2. commit `AgentSession.acceptFinal` before cleanup waits or effect dispatch;
3. synchronously clear and cancel the exact prior reply authority;
4. synchronously supersede playback when a playback owner exists;
5. dispatch STOP, mode, ignore, or typed-unavailable without new backend work;
6. for a reply, validate its turn and model, acquire the lease without waiting,
   and mint an exact playback generation or interrupt the new turn;
7. await that playback generation's exact stop fence, then recheck reply,
   session-transition, turn, playback, lease, mounted, and disposal authority;
8. only then admit the reply owner and lower reply generation.

For an admitted reply, pass the trimmed bounded `transition.decision.text` to
the reply owner rather than reconstructing the prompt from raw input. This keeps
the app session authoritative for transformations such as passive wake-word
stripping while the raw text was still the value preflight-bounded above.

STOP interrupts the current exact session turn and cancels reply authority
before playback-stop deduplication. Mode and ignore effect branches admit no
new backend generation. Search, research, command, dictation, and meeting work
decisions remain typed-but-unavailable on mobile and gain no Python task, tool,
capability, or memory authority. Confirm/deny remain bounded projection kinds,
but the shipped session grants no pending-confirmation authority. Only an
`assistant` decision can mint a reply-capable turn.

Require every token, UI, and TTS mutation from an Assistant reply to retain all
of these exact authorities: its `AssistantReplyGeneration`, current reply-owner
authority, current app-session `AgentTurn`, current ADR-0186
`TtsPlaybackGeneration`, live exact `TtsProcessLease`, and a mounted,
non-disposing widget. Preserve ADR-0201's lower Gemma generation/chat owner and
ADR-0202's exact upper subscription owner. Every nonempty admitted final first
fences and cancels the predecessor exact reply before effect dispatch; that
mandatory predecessor cancellation and any lower cleanup may continue. If the
new effect is a reply
and has no current lease or cannot publish a playback generation, interrupt its
new exact turn with a bounded speech-output-unavailable reason and return before
starting a new reply factory/listen, Gemma generation, TTS synthesis,
AudioPlayer/playback admission, plugin, or native work. Missing or BUSY output
therefore hides no committed semantic result and admits no new reply effect.

Reserve the Assistant lease before constructing any `AudioPlayer`, playback
owner, or TTS-isolate work. A pure player adapter may exist without entering a
plugin. With a live lease the listening adapter serializes the global
AudioPlayer route request through that adapter and rechecks
generation/session/lease authority after the await. Without a lease it admits a
capture-only route: it constructs no `AudioPlayer` or playback owner and calls
neither an AudioPlayer route API nor `TtsService`; the recorder's later
`voiceCommunication` capture configuration is only a capture request and is
not evidence that a global playback route was configured or became live.
Existing ADR-0203 ASR-session, listening-owner, capture-lane, and recorder
cleanup authority remains unchanged.

Cover the direct Speak/replay screen with the same registry before it constructs
or enters TTS, synthesis, player, plugin, or native work. That screen remains
outside `AgentSession` semantic authority. Acquisition is no-wait: BUSY creates
no waiter, replacement, player, TTS request, or native entry. The canonical
Assistant and direct Speak/replay paths therefore cannot concurrently own
shipped UI-isolate speech output when their Dart receipts are exact. Direct
Speak registry participation is source-inspected and compile/analyze-covered;
this slice adds no dedicated source-level wiring test for that screen.

Bind `TtsService` work to the exact lease and one `TtsIsolateLifecycle` epoch.
Publish the epoch before spawn; bind handshake, ready, request, result, and
shutdown traffic to it; deterministically settle pending callers on close; and
ignore stale callbacks. Readiness, request, and shutdown deadlines fence the
epoch and retain uncertainty rather than inventing a return receipt. Isolate
kill and port close are cleanup steps, never native/plugin return proof.

On Assistant disposal, first mark the widget disposing, clear outward
authorities, interrupt its exact current turn, and leave the app-root session
open for a successor widget. Synchronously start reply-owner and
listening-owner close, revoke its exact lease, revoke the player adapter, and
then start playback-owner close before awaiting any of them. Dispose the
recorder only after the exact listening close permits it. Run lease cleanup in
the strict order playback owner, player, then `TtsService.dispose(lease)`; the
player's close must await its serialized operation tail. Release the lease only
when every owned component reports exact success. False, thrown, timed-out,
ambiguous, or never-returning cleanup retains the lease or same-isolate poison
and cannot authorize a successor. A successful `TtsService.dispose(lease)` is
only the exact Dart/coordinator receipt defined by that service; alone or in
the all-component receipt it proves neither native shutdown nor isolate
termination.

Keep every claim at its observable Dart boundary. “Process TTS owner” means the
shipped canonical Flutter UI isolate and covered UI paths, not an
operating-system process lock. Another FlutterEngine, headless engine, Dart
isolate, direct service/plugin caller, or unregistered path can bypass it. No
stop, dispose, shutdown acknowledgement, isolate exit, timeout, or lease
receipt proves native destruction, native return, preemption, allocator/thread
release, audio-route restoration, or device release. Cross-engine and arbitrary
native recorder/player exclusion remain open. The standalone `AsrScreen`
bypasses Assistant's ASR/listening owner, so cross-widget/process recorder/ASR
exclusion during asynchronous cleanup is also outside this claim. The ADR-0186
sentence-queue count remains unbounded.

Treat the frozen file named ADR-0200 only as provenance from the unlanded
`feat/mobile-agent-tts-integration` candidate. It was never accepted on `main`,
predates the landed ADR-0201/0202/0203 owners, and contains obsolete partial and
composition assumptions. Do not overlay its code, tests, golden, documents, or
decision record. ADR-0204 is the first accepted record of the hand-composed
landing; ADR-0201, ADR-0202, and ADR-0203 remain byte-for-byte unchanged.

## Context / why

ADR-0201, ADR-0202, and ADR-0203 separately closed the lower Gemma lifecycle,
upper reply subscription, and ASR/listening lifecycle, while ADR-0186 retained
an instance-local playback owner. None supplied an app-lifetime semantic token
across Assistant reconstruction or prevented a new canonical UI speech path
from starting beside an uncertain predecessor. The direct Speak tab was also a
separate player/TTS path.

The two frozen candidates explored those missing layers against older bytes,
but neither was safe to merge mechanically. The old session candidate granted
semantic authority to partial STOP and broader grammar no longer allowed by the
landed exact-command decision. The old TTS candidate predated the exact
reply/listening/Gemma owners. Overlaying either Assistant file could resurrect
stale cancellation, disposal, routing, or partial behavior. Hand composition
preserves the current owners and places semantic truth above nullable output
resources.

Making the session conditional on TTS was rejected because output failure must
not disable input or controls. Allowing a reply model to run without a speech
lease was rejected because the shipped voice effect would have no exact output
authority. A queued lease or release on timeout/isolate kill was rejected
because an already-entered plugin/native call has no hard cancellation proof.
Configuring the global playback route on the no-lease path was rejected because
capture-only operation needs no player and must not imply speech ownership.

## Consequences

- One app-root session preserves bounded mode/input/turn authority across
  Assistant reconstruction while retaining no transcript history.
- All mobile partials are semantic ignore. Generic acoustic barge remains
  available while speaking, but only completed/typed input can exercise the
  seven exact ASCII STOP commands; the eight substring-bearing negatives
  remain non-STOP.
- Desktop remains intentionally asymmetric: exact partial STOP is authoritative
  there, while every other partial decision is rejected. No broad partial or
  analyzer parity follows.
- Missing/BUSY/poisoned speech output leaves STOP, mode, ignore, and unavailable
  semantics working. Every nonempty admitted final still fences/cancels its
  predecessor exact reply. A new reply effect without exact speech authority
  admits no new reply
  factory/listen, Gemma generation, TTS synthesis, or AudioPlayer/playback work;
  predecessor cancellation/lower cleanup may continue.
- With no lease the Assistant can still listen through a capture-only path that
  makes no AudioPlayer/TTS call. With a lease, route/player/TTS work remains
  exact-authority-gated.
- Assistant and direct Speak/replay share one no-wait speech-output lease in the
  canonical UI isolate; direct Speak remains semantically outside the session.
  Its registry wiring is source-inspected and compile/analyze-covered, not
  protected by a dedicated source-level wiring test in this slice.
- Exact cleanup may admit a later same-isolate owner. Ambiguity deliberately
  retains BUSY/poison until app restart while semantic input remains available.
- Full Python priority-bus, supervisor/task, tools/capabilities, staged
  confirmation, memory, wire payload, and general Unicode convergence remains
  open.
- Existing Assistant UI/diagnostics still retain and can export utterance,
  partial, answer, and bounded log text; the session's no-history property is
  not an app-wide privacy guarantee.
- Headless/static evidence cannot establish plugin execution, native cleanup,
  device routing, audibility, latency, thermal behavior, model/ASR quality,
  multilingual/Romanian support, or live barge behavior. Standalone `AsrScreen`
  bypass and cross-widget/process recorder/ASR exclusion remain unproved.
- This branch adds no acoustic or corpus artifact. The `commands.json` cases are
  unchanged; only their description is reconciled. The already-open bounded
  English packet and separate Romanian opt-in/model/tokenizer/normalization/
  metric plan remain future work, with no dataset-based promotion claim here.

## Production and test inventory

The exact final durable inventory—including the desktop partial safeguard,
`tests/test_speech_analyzer_contract.py`, and the description-only
`tests/golden/commands.json` truth-up—is frozen below. Implementation/test
SHA-256 values may be embedded without a self-reference; the complete
implementation/test/document manifest, including this ADR and the six current-
truth documents, is frozen in ignored `TASK_RESULT.md` and the external closure
artifact. The existing desktop `tests/golden/speech_analyzer_contract.json` is
not a changed path:

```text
31 paths:
.agents/backlog.md
.github/workflows/mobile-tests.yml
STATUS.md
always_on_agent/speech_analyzer.py
core/contract.py
docs/adr/0204-compose-mobile-session-and-ui-isolate-speech-ownership.md
docs/agent-map.md
docs/agent-testing.md
docs/target_architecture.md
docs/unified_architecture.md
mobile/lib/agent_decision.dart
mobile/lib/agent_event.dart
mobile/lib/agent_session.dart
mobile/lib/assistant.dart
mobile/lib/contract.dart
mobile/lib/main.dart
mobile/lib/tts.dart
mobile/lib/tts_isolate.dart
mobile/lib/tts_isolate_lifecycle.dart
mobile/lib/tts_playback_owner.dart
mobile/lib/tts_process_owner.dart
mobile/test/agent_decision_contract_test.dart
mobile/test/agent_session_test.dart
mobile/test/assistant_reply_integration_contract_test.dart
mobile/test/assistant_session_root_contract_test.dart
mobile/test/tts_isolate_lifecycle_test.dart
mobile/test/tts_playback_owner_test.dart
mobile/test/tts_process_owner_test.dart
tests/golden/commands.json
tests/test_always_on_agent.py
tests/test_speech_analyzer_contract.py
Implementation/test SHA-256:
ffcfeee291d37d45ac63d61bdecac103dde344db0b04933a5281060278daa209  .github/workflows/mobile-tests.yml
5ef3425f7ba1356a1a0932b33e3607e46d7e9eec5d95458f00d300cd9a6a8610  always_on_agent/speech_analyzer.py
13b9f31feab738db41ca59b273a608e0678d9a2b591d155abd5f260713dd8c46  core/contract.py
f639f4c2445897e9c79824473b7f4f8c21b2511b3d415a68491b47a2347a238a  mobile/lib/agent_decision.dart
ba941be6633c16ddb2bbf635a24fc35d08f16716a47f1cf8b69a70d416d2337c  mobile/lib/agent_event.dart
c24c0572f19d2d08d999401c1d8f977da9d1ee8aabe0195a330ca70f5c547572  mobile/lib/agent_session.dart
56b322b4d9b3d7f2cf185ce3e4090a7fb1e35d9938a883829b10a21080bea1cc  mobile/lib/assistant.dart
6b4058074d396efac59cc25beba37fbd19f032df1e78f1dee7e62d3f41648e21  mobile/lib/contract.dart
844f9fcd41cece91b2898a8b147878828a75fb02d48fda2b75276dd57f190312  mobile/lib/main.dart
7baad36c5ba538d70849e960efcb1713dbc66a1cf39d24d546c21390b9cddd56  mobile/lib/tts.dart
ce805ac463b587690e144811bcecfe7c71d3c4e6b5ecb4fea9ff7767750dc40c  mobile/lib/tts_isolate.dart
2c63d2ab4a65353e352ef0148e2cc6dc96458cef3521938c1a02bd7178fa5953  mobile/lib/tts_isolate_lifecycle.dart
42b8eda952e3fb3ff2d19baa72c8f56105035ba59584c02253f4ba36b86aae45  mobile/lib/tts_playback_owner.dart
868cff5b41d1717215912e9ff2675a0404289cc7179fef52f65e37ed1f404d5b  mobile/lib/tts_process_owner.dart
d493dd1abfe37fdf90070ba17da6be6cd558b042887c8441f422b866133912ff  mobile/test/agent_decision_contract_test.dart
407c6dfc059489e3663dbcbe4065521dfdb6cabdb583332b4f90994abb6dc692  mobile/test/agent_session_test.dart
a1cbc96ca1570e06803603b06e1f4664f8cd279e93aecbfcc0a5ea622ddf37ed  mobile/test/assistant_reply_integration_contract_test.dart
9d033e8fdf315d8900adaa5d14db9924d7d2e17dbd6c5c31a412715f6dc9bc50  mobile/test/assistant_session_root_contract_test.dart
2b4d690efdf35b920bd9107ab890b547b6f01161f5ec744a8450d24ab3efc75b  mobile/test/tts_isolate_lifecycle_test.dart
a8bf9ef719a828de2f9b35e32a4199e73019aea46da87cdd1f21a3fb4822f960  mobile/test/tts_playback_owner_test.dart
7df5899119f3f6761d6f744223f2db49f7d1699fd035fb8076069fb3d0621f16  mobile/test/tts_process_owner_test.dart
998fba2389c9bcd451346d9b8f36cf8048c650c7aa41c9e9fca6fd7e118196c7  tests/golden/commands.json
1ab38c676673ddd5862aaef84a30b80cefba387d15ca4f524be76c25e60cca48  tests/test_always_on_agent.py
5ddf9871fd6b615f9f86a3ca8015da65c913a4eafd4a4a2245dfb3c114152bf0  tests/test_speech_analyzer_contract.py
```

The ignored `TASK_BRIEF.md`, ignored `TASK_RESULT.md`, and `/tmp` closure
artifacts are not part of the durable Git inventory. The accepted historical
ADRs must recheck byte-for-byte as:

```text
eab77f714eb7199902077aa4126eadad2ececbd8e53ca85755805fb0297e47ef  docs/adr/0201-bound-mobile-gemma-generation-lifecycle.md
017bf3cf8c7fd982ececb24f9917ebd45aee7a4d04d263828c51ded9a69c41be  docs/adr/0202-own-assistant-reply-subscription.md
74846aeafdf9d56560d2a803c9509a1ca672658dde2c4cde4880bd59baf93639  docs/adr/0203-own-mobile-asr-session-and-listening-lifecycle.md
```

## Verification

The receipts below are frozen at the stated byte boundary. Flutter commands ran
from `mobile/`; path-qualified Dart and Python/static commands ran from the
repository root.

- **Focused mobile composition:** `env SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -f 'focused_wall_s=%e focused_max_rss_kib=%M' ionice -c 3 nice -n 19 /home/dobo/flutter/bin/flutter test --no-pub --concurrency=1 test/agent_decision_contract_test.dart test/agent_session_test.dart test/assistant_reply_integration_contract_test.dart test/assistant_session_root_contract_test.dart test/assistant_reply_owner_test.dart test/assistant_listening_owner_test.dart test/tts_process_owner_test.dart test/tts_isolate_lifecycle_test.dart test/tts_playback_owner_test.dart test/golden_contract_test.dart` passed
  `140/140`; `focused_wall_s=10.03; focused_max_rss_kib=471088`.
- **Full mobile:** `env SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -f 'full_wall_s=%e full_max_rss_kib=%M' ionice -c 3 nice -n 19 /home/dobo/flutter/bin/flutter test --no-pub --concurrency=1 -r compact` passed
  `237/237`; `full_wall_s=15.93; full_max_rss_kib=520740`.
- **Flutter analyzer:** `clean; reported 0.8s; analyze_wall_s=1.46; analyze_max_rss_kib=218936`.
- **Non-writing Dart format:** `18 changed Dart files / 0 changes; final comment-only file recheck 1/0` over
  `the exact 18 Dart paths in the 24-path implementation manifest`.
- The focused, full, and analyzer receipts ran immediately before a comment-only
  proof-boundary correction changed `mobile/lib/tts_isolate.dart` from
  `fd4415a...` to final `ce805ac...`; no executable statement changed. The final
  file then passed the 1/0 format recheck, diff/hash binding, and exact-current
  audit.
- **Desktop partial-order safeguard:** `env SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /usr/bin/time -f 'python_policy_wall_s=%e python_policy_max_rss_kib=%M' ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_always_on_agent.py::test_non_stop_partials_cannot_mutate_mode_confirm_or_start_work tests/test_always_on_agent.py::test_only_scoped_partial_stop_cancels_research tests/test_speech_analyzer_contract.py` passed
  `6/6; 0.38s pytest; python_policy_wall_s=0.73; python_policy_max_rss_kib=47640`; it proves exact partial STOP plus rejection of
  every other partial mode/confirmation/task decision, not mobile parity.
- **Shared completed/typed command contract:** `env SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /usr/bin/time -f 'python_golden_wall_s=%e python_golden_max_rss_kib=%M' ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_golden_contract.py`
  passed `34/34; 0.07s pytest; python_golden_wall_s=0.36; python_golden_max_rss_kib=36352`; the paired Dart consumer is included in
  `the focused 140/140 mobile gate`.
- **Repository APM/DTD regression:** `6/6; 2.01s pytest; apm_dtd_wall_s=2.68; apm_dtd_max_rss_kib=125916`; synthetic headless
  PCM only.
- **Scoped Python lint:** `clean on always_on_agent/speech_analyzer.py, core/contract.py, tests/test_always_on_agent.py, and tests/test_speech_analyzer_contract.py`. This is a Ruff check claim only;
  the four legacy Python files are not claimed globally Ruff-format-clean.
- `git diff --check`, conflict-marker, exact STATUS line-count, accepted-ADR
  stability, inventory, and hash gates: `scoped Ruff and git diff --check clean; conflict-marker scan empty; STATUS.md 100 lines; ADR-0201/0202/0203 byte hashes stable; exact 31-path inventory and code/docs native replays exact`.
- Independent final audits and exact integrated patch digest:
  `Assistant composition architecture audit GO; exact-current semantic/resource audit GO (ordered 29-path manifest SHA-256 385af34914645002c84bd90736c0494b276d7e3ba0621f0e8c698c4fc4a88c8f); independent data audit GO; replayed code-only native patch SHA-256 46f4cd523d6c3637055db81d2a98c3670f2ca77ee183450a0c94dd302a835ba0`.

No Flutter plugin runtime, Sherpa model, Gemma model, network, download, GPU,
microphone, recorder/audio device, APK, emulator, physical phone, native
cleanup, audibility, latency, thermal, quality, multilingual/Romanian, or live
path ran. No new acoustic or corpus artifact was added or evaluated;
`commands.json` changed description only, not cases.
