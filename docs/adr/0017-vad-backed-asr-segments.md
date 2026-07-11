# ADR-0017: VAD-backed ASR segment ownership and endpoint clock

Date: 2026-07-10
Status: superseded-by ADR-0046

## Decision

When a VAD model is available, the live desktop ASR path requires that VAD to
observe speech before dispatching a final, derives trailing silence and learned
mid-utterance pauses from VAD speech/quiet transitions, and permits a semantic
early endpoint only after verified VAD quiet. The offline finalizer retains a
bounded model-lookback pre-roll until speech starts, then owns the complete
utterance through the configured rule-3 ceiling and endpoint tail. When VAD is
unavailable, final admission remains fail-open but semantic early commits are
disabled because decoder inactivity is not acoustic silence.

## Context / why

On the first real host run of 2026-07-10 (`run-20260710-084939`, no audio
recording and nobody speaking), the fully loaded runtime emitted raw/final
`"AND"` every roughly two seconds from near-zero input (`avg_rms` about
0.000008–0.000016). The existing floor gate was inert on the OS-capture path,
so each hallucination reached addressing and consumed local LLM work. Separately,
the engine treated the time since the decoder's last text change as trailing
silence and as a speaker pause; stable tokens or a compute stall could therefore
end ongoing speech early. Its unconditional rolling PCM buffer also made a short
request after idle look like a ten-second clip while truncating the head of a
valid 15–20 second utterance.

Why not use a fixed RMS threshold: it is device/route/gain dependent and would
repeat the hard-coded operating-point failure rejected by ADR-0012. Why not trust
the recognizer endpoint or token cadence alone: the observed endpointed `"AND"`
storm proves a recognizer can produce text and an endpoint with no speech, and
token cadence measures decoding rather than acoustics.

## Consequences

- Idle/silence hypotheses are dropped before SenseVoice, speaker-ID, addressing,
  or the LLM; `vad_rejected_final` makes the rejection observable.
- SenseVoice's short-clip guard receives actual VAD speech duration, independent
  of pre-roll and endpoint padding, and the speaker/floor gates receive the owned
  utterance instead of idle-diluted or head-truncated PCM.
- The adaptive pause model learns real resumed pauses, not decoder update gaps.
- VAD becomes load-bearing when configured. Headless tests cover silence reject,
  speech admit, stable-partial/no-early-endpoint, idle pre-roll, and a 15-second
  head-preservation case. A live owner-voice A/B must still confirm quiet speech
  is admitted and that mid-thought pauses are not split; no such human validation
  is claimed by this ADR.
