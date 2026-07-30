# Task Result — typed acoustic ingress prerequisite

Valid until: ADR-0084 is superseded or the session-boundary redesign lands —
then treat as history.

## Outcome

- Recognizer observations now carry immutable, PII-free acoustic lineage and
  monotonic revisions through partial, final, barge, command, abort, replay,
  turn-merge, and runtime-ingress paths.
- Typed stale or duplicate events are rejected before runtime effects.
- A mapped keyword command is terminal at the source. Its PCM and normal
  recognizer state are retired in the same capture iteration, so it cannot
  later become a second ASR final.
- Barge-in is revision-gated before cancellation or playback interruption.
- Bounded async-final overflow, recovery, rejection, empty final, and decode
  failures have explicit typed abort paths. Shutdown retires source ownership
  locally after external callbacks are fenced.
- Typed post-barge response permission is acoustic-turn-bound; an unrelated
  final cannot inherit it. A matching aborted partial can restore the valid
  unheard final it displaced.
- This is not the new session actor. Task, tool, TTS, and playback lifecycles
  still use legacy generations, and capture/recognition is still synchronous.

## Verification

- Deterministic repository gate: `5722 passed, 13 skipped, 20 deselected`,
  with 9 pre-existing warnings.
- Focused typed-ingress, Sherpa, replay, and APM gate:
  `354 passed, 2 skipped`.
- Added a direct capture regression proving a KWS-consumed frame never reaches
  normal ASR.
- Ruff found no new diagnostics after excluding the repository's pre-existing
  import-order and unused-import findings.
- `git diff --check` passed.

## Not validated

- No microphone, speaker, echo-route, owner-voice, or natural-conversation live
  test ran.
- No CUDA model benchmark ran; the managed sandbox did not expose
  `/dev/nvidia*`.
- No production STT/TTS/model selection changed.

## Next

Build one bounded `SessionActor` that carries session/turn/revision and separate
speech/tool cancellation through tasks, TTS, and playback. Keep device capture
as a thin producer. Public voice-corpus work belongs on a separate branch with
the existing FSDD license mismatch resolved before any redistribution.
