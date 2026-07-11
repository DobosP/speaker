# ADR-0037: Keep streamed TTS voice continuous within one reply

Date: 2026-07-11
Status: superseded-by ADR-0041

## Decision
The runtime owns a reply-scoped voice lease keyed by `(task_id, speech_epoch)`.
For adapters that explicitly advertise typed-style support, after a TTS request
passes stale/auxiliary admission the runtime strips its leading expressive tag
and snapshots an immutable `SpeechStyle` on the `TrackedSpeech` fragment. A
valid explicit voice establishes or replaces that reply's voice; untagged later
fragments inherit it. Emotion and rate remain fragment-local. Auxiliary
acknowledgements/apologies are unscoped, and reply state closes on
`TTS_STREAM_END`, task cancellation/failure, barge-in, stop, or shutdown.
Sherpa and FileReplay resolve the typed hint against their local TTS
configuration; legacy `speak()` overrides keep receiving raw markup.

## Context / why
Live run `run-20260711-130601` emitted one story as sentence-level TTS requests.
The first sentence carried `[voice:warm]` and resolved to speaker 16, but every
later untagged sentence independently resolved to default speaker 0. This was a
contract mismatch: prompt guidance deliberately says tags are sparse, while the
engine parser treated every sentence as an unrelated utterance.

Repeating the tag in every model-generated sentence is brittle and wastes model
output. An engine-global "last voice" would leak across interleaved replies and
auxiliary speech; resolving it when the playback worker dequeues would also let
a later switch restyle already-queued text. `TASK_COMPLETED` cannot close the
lease because its priority 60 event overtakes queued sentence requests at
priority 100. The explicit priority-110 `TTS_STREAM_END` is the safe normal
reply boundary.

## Consequences
- Voice identity remains stable for a streamed reply; a later valid voice tag
  switches only that reply from that fragment forward.
- Concurrent replies and auxiliary speech cannot borrow or mutate one another's
  voice. Queue entries retain their admission-time style snapshot.
- Markup is sanitized before playback history/resume registration, so receipts
  and remembered spoken text use listener-visible text.
- Stream end tombstones the completed producer so a late fragment cannot reopen
  a playback-history group with no future close marker.
- A non-streamed reply applies one leading tag to its whole TTS fragment; prompt
  guidance forbids tags between its sentences, and defensive sanitization strips
  accidental later tags instead of speaking them. Default-off behavior is
  unchanged. A new reply starts from the configured default; there is no
  within-reply `voice:default` reset syntax.
- The lifecycle and adapter behavior are headless-testable; no live audio A/B is
  required to prove control-plane continuity, though voice quality remains an
  owner-ear judgment.
