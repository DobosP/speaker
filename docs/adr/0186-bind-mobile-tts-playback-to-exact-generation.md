# ADR-0186: Bind mobile TTS playback to an exact generation

Date: 2026-08-11
Status: accepted

## Decision

Make one pure-Dart `TtsPlaybackOwner` the sole queue, pump, and native-player
lifecycle owner for streaming TTS in the Flutter `AssistantScreen`. Give each
admitted reply an opaque `TtsPlaybackGeneration` created by that exact owner.
Its monotonic ordinal is diagnostic only; authority requires the exact token
object and its private owner identity.

Advance the generation synchronously on every superseding endpoint, typed
turn, speech interrupt, and close. Clear old queued sentences and complete only
the exact active clip's interrupt before a caller can enqueue replacement work.
Keep the unfinished sentence buffer local to its exact `_answerUtterance`
invocation and pass the generation on every flush. Reject empty, foreign-owner,
closed-owner, poisoned-owner, or non-current enqueue attempts, so a token or
sentence continuation that returns after barge-in cannot resume old speech.

Run exactly one synth/play pump across all generations. Do not cancel synthesis
that has already entered its native isolate seam. Recheck its generation after
return and discard a stale or null path before player admission. Before each
play, also require the exact predecessor-cleanup fence to have succeeded and
recheck current authority. A blocked old synthesis therefore head-of-line
blocks replacement speech but cannot play stale audio or start a second pump.

Replace the shared `AudioPlayer` and its untagged completion stream with one
fresh `AudioPlayer` per clip. After constructing that player, synchronously
install its exact adapter state and `TtsPlaybackClip` handle before admitting
the per-player route or `play` calls; return the handle without awaiting those
calls. `AudioPlayer()` construction precedes handle installation and may itself
begin plugin/native creation, so this decision does not claim that the handle
exists before every native operation. The handle exposes a started receipt,
one typed `completed`/`interrupted`/`failed` terminal receipt, and one memoized
stop-and-dispose cleanup receipt. Normalize started, terminal, and cleanup
failures into immutable result data immediately; never expose an erroring
Future whose failure can escape before the owner installs its waiter. Subscribe
to the fresh player's completion stream before route/play admission. Do not
infer clip identity from a process-shared or reused player's raw terminal event.

Preserve the Android voice-communication route in both scopes required by the
fresh-player design: configure `AudioPlayer.global` for the capture/playback
session, and set the same speakerphone/speech/voice-communication/focus context
on every exact clip player before `play`.

Serialize every clip start and cleanup transaction on one owner operation
lane. Reserve the start on that lane before its first asynchronous wait; a
concurrent supersession queues cleanup after an already-admitted start.
Replacement play waits for the cleanup fence associated with its generation,
so a late old stop/dispose cannot kill or overlap a new clip. Native synthesis
and player start that have already been entered remain non-preemptible: either
may head-of-line block replacement playback and orderly idle waiting until it
returns.

Derive public speech activity from two separate facts: current logical work and
possibly active physical playback. Mark physical activity before awaiting the
started receipt, and retain it through starting, playing, stopping, disposing,
and any uncertain cleanup outcome. Supersession clears logical work but cannot
publish inactive while old physical state remains possible. A successful exact
cleanup receipt clears physical activity. An exact natural `completed` terminal
may independently prove audible silence before cleanup finishes, but cleanup
still must dispose the player before another clip can be admitted.

Treat stop-and-dispose cleanup as a fail-closed resource boundary. Always try
dispose even when stop fails. Dispose success is an exact release receipt and
may carry a recovered stop or subscription-cancel error as diagnostic data.
Dispose failure poisons this exact `TtsPlaybackOwner`/`_AssistantScreenState`
instance for its remaining lifetime, clears queued logical work, and rejects
every later enqueue through that instance. Without an earlier exact natural
completion, keep physical activity conservatively true after that failure;
with natural completion, audible silence is proven but resource admission
remains poisoned. No diagnostic callback can clear that instance's poison or
reopen its playback.

Do not treat that poison or restart refusal as process-wide. Flutter may dispose
and reconstruct the Assistant tab, creating a fresh state, owner, adapter, and
player admission domain while an old state's native start or cleanup remains
hung, failed, or physically uncertain. This slice has no process registry or
cross-widget fence and makes no no-overlap claim across reconstruction. Retain
that as an explicit unresolved plugin/live lifecycle risk; only a later
process-wide owner/fence or verified plugin lifecycle contract can close it.

Publish activity and playback-start callbacks only for the current exact
generation. Tag every error with the generation whose operation observed it,
but treat errors from stale start/cleanup as diagnostics only, never current
reply or playback authority. `AssistantScreen` independently checks current
generation before stateful activity/start handling; diagnostic logging may
retain a stale generation label without mutating authority.

Make barge-in revoke both domains synchronously: advance the playback
generation and increment the LLM turn before awaiting cleanup. Every token,
terminal flush, exception path, and `finally` mutation rechecks the exact turn,
generation, mounted state, and disposing fence. `dispose()` sets that fence and
advances the turn before asynchronous shutdown, so a late reply cannot call
`setState`, append speech, or clear replacement/disposed state.

This is an additive mobile lifecycle decision. It does not supersede another
ADR, migrate the Dart loop to the shared `AgentEvent` contract, change sentence
splitting, model selection, synthesis audio, barge thresholds, or authorize a
device/live conclusion.

## Context / why

The prior `AssistantScreen` stored one shared sentence buffer, speech queue,
`_draining` boolean, player, completion stream, and `_playInterrupt`. A new
endpoint launched `_stopSpeaking()` without waiting; that stop cleared the
queue and `_draining` before its awaited player stop returned. A replacement
answer could start a second drain while old synthesis or player work remained.
Old synthesis had no pre-play generation check, and an old drain's `finally`
could clear state belonging to the replacement reply.

The first generation-owner draft closed those queue races but still consumed
raw events from one shared `onPlayerComplete` stream. A stale event arriving
during a later start could complete the wrong clip, and a stream error before
the owner began waiting could surface as an unhandled asynchronous error. It
also published inactive as soon as logical work was superseded, even while
stop was blocked or had failed. That disabled energy barge detection and could
admit contaminated microphone chunks into ambient-floor calibration while old
audio might still be audible.

A boolean generation, one completion Future reused across clips, or another
drain per generation cannot establish exact native-player identity. Awaiting
stop only in typed submission also cannot revoke endpoint/partial/energy barge
work synchronously. A fresh player plus exact state/handle installation before
route/play admission make terminal and cleanup resources exact within one
widget-owner instance; one cross-generation pump and one operation lane choose
bounded concurrency and stale-audio safety over progress when an old native
call is blocked. They do not fence a later reconstructed widget from that old
instance.

## Consequences

- At most one pump, one synthesis invocation, one clip wait, and one serialized
  player start/cleanup transaction are admitted by this owner. The adapter owns
  a fresh player and terminal stream for each clip.
- Supersession immediately removes old reply/enqueue authority. An entered
  synth or start may retain native resources and delay replacement speech until
  return, but stale completion cannot authorize playback or UI mutation.
- Derived activity stays true throughout queued logical work and every starting,
  playing, stopping, or uncertain physical interval. This keeps energy barge
  detection active and ambient-floor observation closed until silence is
  established.
- Exact natural completion proves silence only for its clip. A successful
  dispose proves resource release; a recovered stop failure remains diagnostic.
  Failed dispose poisons future admission even after natural silence.
- Per-clip terminals cannot be confused with another player's raw completion,
  and started/terminal/cleanup failures are consumed as data rather than
  escaping before a waiter exists.
- Generation-tagged stale errors remain observable diagnostics but confer no
  current authority. Close is memoized and permanently rejects work in that
  exact owner; dispose invalidates its late reply mutations before asynchronous
  cleanup begins.
- Instance-local poison is not a process restart fence. Widget reconstruction
  can admit a fresh owner while the old player remains uncertain, so native
  no-overlap across reconstruction is unproved and remains a plugin/live risk.
- The current-generation sentence queue retains its prior unbounded-count
  behavior. This ADR bounds concurrency and identity, not queued text, native
  synth/start duration, plugin-internal resources, or return.
- The BargeCalibrator's missing pre-partial ASR speech-start signal remains
  deferred to wider mobile `AgentEvent` convergence. This slice closes the
  post-partial/playback activity boundary only.
- Headless fakes cannot prove plugin/device latency, audible ordering, audio
  quality, Android route behavior, native return, or live barge behavior.

## Verification

- From `mobile/`, `ionice -c 3 nice -n 15 /home/dobo/flutter/bin/flutter test test/tts_playback_owner_test.dart --no-pub`
  passed `18` pure exact-clip owner
  tests (Flutter display `00:00`; tool wall 5.2 s). Controlled receipts/futures
  cover exact handle state, direct foreign-owner token rejection,
  started/terminal/cleanup data, old synth/start head-of-line schedules, one
  pump/lane/FIFO, supersede/stale rejection, cleanup fences, natural
  completion, current/stale errors, conservative activity, recovered stop,
  failed-dispose poison, and memoized close.
- From `mobile/`, `ionice -c 3 nice -n 15 /home/dobo/flutter/bin/flutter test --no-pub`
  passed all `31` tests (Flutter display `00:01`; tool wall 7.3 s).
  `flutter pub get --offline` had resolved 99 cached packages beforehand; no
  network ran.
- From `mobile/`, `ionice -c 3 nice -n 15 /home/dobo/flutter/bin/flutter analyze --no-fatal-infos --no-fatal-warnings`
  reported `No issues found!` (analyzer
  1.1 s; tool wall 4.6 s). Fresh `AudioPlayer`, global plus per-player
  voice-communication context, barge-turn invalidation, and dispose-time reply
  guards are static/analyzer-reviewed integration, not fake-executed plugin
  behavior.
- Independent adversarial review found no remaining within-instance blocker on
  the settled semantic contract. It explicitly does not close the cross-widget
  reconstruction risk: `AudioPlayer()` construction may begin plugin/native
  creation before exact handle installation, and an old uncertain native player
  is not process-fenced from a new Assistant state.
- No Flutter plugin runtime, TTS/LLM/ASR model, GPU, microphone, audio device,
  network, APK, emulator, physical phone, latency, quality, or live path ran.
