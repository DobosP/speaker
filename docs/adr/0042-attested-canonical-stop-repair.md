# ADR-0042: Repair one attested canonical STOP transcript during word-cut

Date: 2026-07-11
Status: accepted

## Decision

Classify playback-time STOP evidence with one typed decision shared by the
active word-cut, reply-tail, and continuation boundaries. Preserve every exact
stop-class command. Additionally repair only the normalized, owner-attested
tuple `("of", "he", "stop")` observed in live run `20260711-154451`. Every
other non-command—including ordinary noun phrases, sentences ending in stop,
and negations—remains ordinary talk-over text; do not infer control semantics
from a generic prefix length or terminal token.

Fuse that lexical evidence with the existing own-speech comparison. Novel
canonical STOP evidence may cut playback immediately, as exact STOP already
did. If the candidate resembles current or recent TTS, require the existing
enrolled-speaker decision over at least the short `0.10` second voiced window;
missing, cold, busy, rejected, errored, or insufficient authority does not cut.
For the attested repair, token `stop` anywhere in current or recent TTS forces
that ambiguity even when ASR corruption makes ordinary token overlap too low.
Do not change the general `0.30` accept or `0.22` reject thresholds. This
playback-only decision never creates owner verification, action trust, private
context access, tool authority, or continuation lineage.

## Context / why

In live run `20260711-154451`, the owner said STOP during `sid=16` TTS. The
word-cut recognizer advanced through a burst ending in the three-word suffix
`OF HE STOP`, then three quiet blocks reset it with zero cuts. The implementation
called exact `is_stop_command(new_text)`, so it treated the suffix as ordinary
audio-first talk-over; its short burst could not satisfy the normal `0.35`
second identity window. Nearby identity samples were clear rejects or ambiguous
(`0.16-0.23` against reject `0.22` / accept `0.30`), which is not evidence for
lowering the general threshold. A prior live run had recognized and cut one
exact STOP, isolating the regression to corrupted lexical framing.

Arbitrary suffix scanning was rejected after review showed it would reinterpret
`BUS STOP`, `NEXT STOP`, `FULL STOP`, `IT WILL STOP`, and negated phrases as
controls. A special lower identity threshold was also rejected because the
failed burst did not retain a score that proves such a threshold, and word-cut
identity must not leak into downstream trust.

## Consequences

- The exact observed `OF HE STOP` shape cuts before its quiet reset; no other
  corrupted transcript is repaired without a new attested decision.
- Current TTS containing STOP anywhere, such as `The app may stop responding`,
  still forces speaker authority when ASR emits the attested repair, and a
  decisive own-TTS score does not self-interrupt.
- Ordinary and negated STOP-ending phrases remain ordinary talk-over text and
  cannot use the canonical STOP bypass.
- Headless replay proves the state-machine behavior only. The same owner STOP,
  a silent-control reply, and an own-TTS stop phrase still require live A/B on
  the open laptop speakers before barge-in can be called validated.
