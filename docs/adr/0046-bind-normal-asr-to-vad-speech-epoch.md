# ADR-0046: Bind normal ASR decoder state to the VAD speech epoch

Date: 2026-07-11
Status: accepted

## Decision

Keep ADR-0017's VAD-backed final admission, acoustic clock, and bounded owned
PCM, and additionally scope normal streaming-decoder hypotheses to the current
VAD speech episode. While a configured VAD has not observed speech, decoder text
may remain internal model lookback but cannot publish a partial or acquire an
ASR-segment text timestamp. At the first VAD speech onset, reset the normal
recognizer stream and replay only the segment's bounded pre-roll before feeding
the current onset block once. Use the alternate relaxed/NS-off pre-roll when that
is the live streaming-ASR domain; primary PCM remains finalizer/floor/speaker
authority.

Rebase once per segment, not on a mid-utterance pause or resume. A word-cut
handoff already marks the segment as speech and has its own reset/replay lineage,
so it bypasses this normal-onset rebase. A successfully confirmed legacy
duck-window barge already left its command head in the normal stream; adopt that
explicit one-shot handoff without rebasing, then clear it at onset, endpoint,
error, expiry, recovery, stop, or start. Recheck the capture epoch after the
external barge callback and publish under the same lifecycle lock that stop uses
to invalidate it. No other pre-onset stream state gets this exception.
If a VAD episode ends with no decoder text from that episode and sherpa emits no
endpoint, reset its normal stream and segment at `endpoint_max_silence_sec`.
Word-cut prefixes retain their existing bounded endpoint/offline-recovery
contract. Keep the no-VAD fallback unchanged.

Do not add an absolute RMS threshold, a one/two-word blacklist, or special-case
STOP. Quiet genuine speech and canonical controls use the same fresh-epoch path.

This supersedes ADR-0017 while retaining its VAD admission, bounded pre-roll,
complete-utterance ownership, duration, pause learning, and no-VAD behavior.

## Context / why

Live run `20260711-200747` produced streaming `And` / `And truth` around
`20:08:00-05` at post-front-end heartbeat RMS `0.0002-0.0004`. No newer partial
was logged, but raw `AND TRUTH` finalized at `20:20:04`, about twelve minutes
later, when later acoustic activity occurred. The configured final-floor gate
was explicitly inert and speaker input gating was disabled, so the stale final
reached the runtime. Later ordinary-listening garble generated most of the run's
115 turns and 99 LLM requests; the run recorded zero active word-cut cuts.

ADR-0017 required VAD speech somewhere in the segment, but `speech_seen` and the
normal recognizer stream both remained live until a recognizer endpoint. A later
VAD blip could therefore lend new acoustic admission to an arbitrarily old
decoder hypothesis, and the pre-VAD partial callback could already cancel or
reshape runtime work before the final gate rejected it. Bounded PCM alone did not
bound decoder state or callback authority.

## Consequences

- Pre-VAD hallucinations cannot enter partial-driven cancellation/continuation,
  and a later VAD onset cannot turn their old text into a final.
- The fresh decoder sees at most bounded pre-roll plus the current onset block;
  neither the recognizer nor final-owned PCM receives a duplicate current block.
- Confirmed duck-window STOP/word heads remain in their already-authorized normal
  stream and finalize once; an unconfirmed or stale window cannot bypass rebase.
- A no-text false VAD blip cannot leave an episode armed indefinitely when the
  recognizer omits its endpoint; the reset is observable as
  `vad_abandoned_epoch_reset`.
- Headless capture tests retain fresh quiet `YES` and `STOP`, prove stale text is
  absent, and compare distinct replay/final PCM blocks exactly. Normal pause,
  recovery, word-cut, endpoint, and no-VAD suites remain required gates.
- This closes only stale cross-epoch reuse. Fresh-epoch garbage, the inert
  OS-route final floor, disabled local speaker input gate, and reply-tail
  lexical/acoustic provenance remain separate live-red issues.
