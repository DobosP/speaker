# ADR-0036: Authorize no-duck barge-in from enrolled-speaker audio

Date: 2026-07-11
Status: accepted

## Decision

Keep ADR-0013's verified OS/PipeWire echo-cancel route and continuous no-duck
playback-time capture, but replace word count as the production cut authority.
When word-cut is active, require a compatible enrollment and synchronously warm
the speaker model before starting capture. A canonical stop remains immediate,
except when it reads as the assistant's currently playing standalone stop; that
ambiguous case requires at least `0.10` seconds of speaker-authorized voiced PCM.

For other interruptions, retain bounded candidate PCM and score the newest at-most
two-second energy-voiced window after at least `0.35` seconds of actual voiced
frames, regardless of whether playback-time ASR produced zero, one, or garbled
words. On hosts where normal finals are not identity-gated, accept at
`barge_word_cut_speaker_threshold=0.30`, decisively reject at or below `0.22`, and
discard the PCM behind an ambiguous score before retrying on a fresh authority
window after at least `0.25` seconds of additional observed PCM. When final input
speaker-gating is active, its threshold remains the minimum accept threshold.
Missing, cold, incompatible, busy, errored, or non-finite authority fails closed
without blocking capture. Identity-free legacy mode retains the four-word rule.

Promote only the exact voiced/model-bounded PCM that speaker-ID scored, replay it
once into normal ASR, and splice it once into finalization. If streaming ASR stays
empty through the endpoint, this finite authorized handoff may invoke offline ASR
directly; ordinary VAD-only empty clips may not. PCM captured while an
onset/suppression/refractory guard is active is discarded and cannot ride into a
later authorization. Clear decisive non-owner bursts and identity-bound ambiguous
PCM. Apply the same authority at reply-tail and fresh post-playback continuation
boundaries. Keep exact-route readiness, per-reply funnel telemetry, and
self-echo-aware diagnosis mandatory.

Keep final admission separate from owner trust. Engine finals carry an explicit
`UNKNOWN`/`VERIFIED`/`REJECTED` speaker verdict and live-audio origin. Gate-off,
unavailable, error fail-open, loudness rescue, the double-talk-only `0.30` cut
threshold, ambiguous prefixes, and unscored post-cut voiced continuations never
set `owner_verified`. Only the enrolled final-input gate's clean accept may do so;
turn merging combines verification with fail-closed AND semantics.

## Context / why

Live run `20260711-130601` reproduced the owner's report that barge-in did not
work. The first two replies retained 17 and 9 VAD-active blocks, including near-end
RMS around `0.018-0.022`, but playback ASR produced zero words, so speaker-ID was
never consulted. During a story it emitted unstable one-, two-, and three-word
bursts that reset before ADR-0013's four-word floor. The sole four-word burst was
own TTS and correctly scored `0.15` as non-owner.

Recorded probes separate the acoustic classes despite corrupted text: own TTS
scored `0.15-0.179`, while available owner double-talk slices scored
`0.387-0.537`. Reusing the clean-final threshold `0.50` would reject known owner
samples. Text is therefore useful for a canonical control command and downstream
ASR, but not as the identity authority during double-talk.

## Consequences

- A real owner can cut while playback-time ASR is empty or fragmented; no duck or
  volume pumping is introduced.
- Clear own-TTS residuals do not cut, ambiguous scores wait for more evidence, and
  inference retry rate/window/memory remain bounded.
- Zero-word authorized audio reaches offline transcription without converting the
  purpose-specific `0.30` cut score into owner/action trust.
- Threshold evidence is from one device, route, room, and enrollment. Other-human,
  cloned-voice, phone-cost, and thermal distributions remain unvalidated; the
  thresholds stay configurable and require a silent-control/owner live A/B.
- The rule cannot prove identity against a high-quality clone of the enrolled
  voice. Exact standalone-stop ambiguity intentionally favors safety.
- This supersedes ADR-0013's four-word production authority. It does not fix the
  separate runtime input-calibration, addressing, TTS continuity, or shutdown
  defects exposed in the same live session.
