# ADR-0121: Retain parakeet.cpp isolation after the AMI diagnostic

Date: 2026-08-02
Status: accepted

## Decision

Record the exact AMI ES2004a close/far run as a single-source, aggregate-only
development diagnostic and keep the parakeet.cpp candidate isolated. Execute
the retained schema-v8 manifest from its bound clean `d80c586` harness, using
the selected 1,280-sample chunks, 80 ms partial cadence, and 8,000-sample tail
for a three-repeat burst plus one matched real-time repeat. Do not treat this
as the four-source conversation fixture, held-out evidence, runtime/default
authority, or a substitute for capture and live validation.

## Context / why

ADR-0119 left conversation and far-field behavior open. ADR-0120 made the
locked 24-case AMI source materializable, providing twelve identical close/far
interval pairs across isolated speech and turn transitions. The first run from
the new task worktree failed closed before model load because the retained
manifest binds an absolute worker path. Running from its unchanged bound
harness preserved the exact candidate and evaluator used by ADR-0119.

Burst completed 72 evaluations with zero final disagreement and exact source
accounting. Overall WER/CER were .4684/.4113, but close WER was .2278 while far
WER was .7089. Per repeat, close produced 6/12 exact and 11/12 nonempty finals;
far produced 1/12 exact and only 3/12 nonempty finals. Eighteen of 72 runs had
an accepted endpoint and 54 exhausted the 500 ms tail. Source/model-input RTF
were .3767/.3171 and maximum reported RSS was 307.535 MiB.

The paced repeat preserved every aggregate accuracy count. Source/model-input
RTF were .4367/.3677, first-nonempty-partial p50/p95 were 622.543/2,369.120 ms,
ten cases lacked a stable partial, two deadlines missed with 14.831 ms maximum
backlog, and maximum reported RSS was 308.105 MiB. NVIDIA's model card lists
AMI in the training and evaluation datasets without identifying exact clip
membership, so training disjointness is not established. These WER values are
not comparable to NVIDIA's AMI leaderboard result because selection,
segmentation, normalization, channels, and first-EOU policy differ. Endpoint
latency covers only six accepted endpoints per repeat from declared source end
to observed feed boundary; it has no endpoint ground truth or conversational
latency authority.

## Consequences

The candidate remains fast and low-memory on one CPU but is not a stable
far-field voice-agent STT choice. The next model-quality evidence must use a
source with established disjointness, preferably the locked Common Voice SPS
v4 or Harper Valley slice, and must retain close/far or room-noise strata.
Separately test capture/VAD/endpoint ownership, AEC, open-speaker barge-in,
commands, identity/authority, tools, TTS, and live latency before adoption.
The AMI result cannot change setup, `core`, `./live.sh`, or any default.
Aggregate path-free evidence is in
`docs/evidence/2026-08-02-parakeet-cpp-ami-diagnostic.json`.
