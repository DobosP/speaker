# ADR-0052: Separate the echo-free inject barge profile

Date: 2026-07-12
Status: accepted

## Decision

Keep the physical coherence detector's conservative five-frame warm-up and
two-frame confirmation unchanged. In explicit `tools.live_session --inject`
mode only, disable physical AEC and OS word-cut authority as before, and use
zero coherence warm-up, one detector confirmation, and zero physical delay
search because assistant output is structurally absent from the injected capture
domain. Retain the shipped two-block `BargeSustain` requirement. Pace injected
capture against absolute device deadlines so capture processing time is deducted
from the next read wait. Count an answer as stopped only when both the stop call
and a `BARGE_IN_STOP` FIFO-cut stamp are present. Record user timing as enqueue
metadata, bind each assistant answer to `response_to_user_idx`, and render
scenario pass/failure signals as unchecked manual criteria. Use distinct
long-answer prompts for independent interruptions in one scenario so EchoLLM's
deliberately verbatim response cannot make the runtime's own-echo guard consume
the second target request.

## Context / why

The first exact injected replay stopped the longer redirect but missed the
0.395-second `Stop.` clip. Inject mode routes assistant playback to a null sink,
so no playback echo reaches capture to consume or seed the physical detector's
five warm-up frames. The short user's entire voiced region consequently became
warm-up and was discarded before the runtime received any barge callback. This
was a synthetic-topology mismatch, not a cancellation failure. Privately marking
the fake input as a verified OS echo-cancel route was rejected because it would
misstate capture provenance and switch the test to production word-cut without
valid enrollment authority. Weakening physical detector defaults or lengthening
the canonical stop stimulus was also rejected.

The first inject-only adjustment retained the physical 400 ms delay search. That
left the exact Stop clip at the two-block speech floor after reference fill and
made success sensitive to block phase. With no capture-side assistant echo there
is no physical propagation delay to estimate, so the inject profile removes that
search while the production 400 ms setting remains unchanged.

The fake input stream additionally slept one full block after every read. GTCRN
and ASR processing time accumulated on top of that period, so a prior artifact
recorded only 85% of wall time and falsely failed continuous-capture coverage.
Finally, the scenario repeated an identical long prompt. With deterministic
EchoLLM, the assistant had just spoken those exact words, so the production
self-echo guard correctly rejected the repeat and left no second answer to cut.
The original report also paired per-user latency ordinally, so the no-answer Stop
shifted later latency rows, and labelled non-blocking enqueue return as if user
audio had finished. Explicit response links and enqueue-only metadata prevent
those artifacts. A generic `stop_speaking()` call was insufficient proof of a
cut; `BARGE_IN_STOP` now attests that a live FIFO was actually cut.

## Consequences

The echo-free inject profile exercises real Sherpa ASR, VAD, denoise, TTS
playback workers, capture continuity, barge callback, cancellation, recovery,
and post-cut input recovery without opening microphone or speaker hardware. It is
not evidence for physical echo cancellation, enrolled-speaker authority,
voice quality, voice identity, user-onset-to-stop latency, or the production
PipeWire word-cut route. Those remain live bare-speaker requirements.

On Linux ROG, three fresh `barge_in_interrupt_stop` repetitions each passed
full-duplex coverage (0.979–0.983), produced partials while buffered user audio
was under its observer window, attested all six intended FIFO cuts, produced zero
self-interrupts, and admitted a fresh post-cut redirect final. EchoLLM repeats that
redirect; this run does not grade whether a real model tells a good joke. The
displayed stop interval begins at the barge callback and therefore measures
detector-fire-to-FIFO-cut, not acoustic user-onset-to-stop latency. The prior
1/2 and 85%-coverage artifacts remain preserved as failure evidence.
