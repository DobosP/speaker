# ADR-0122: Retain parakeet.cpp isolation after the Harper diagnostic

Date: 2026-08-02
Status: accepted

## Decision

Record the exact HarperValleyBank caller-speech run as a single-source,
aggregate-only development diagnostic and keep the parakeet.cpp candidate
isolated. Execute the retained schema-v8 manifest from its bound clean
`d80c586` harness, using 1,280-sample chunks, 80 ms partial cadence, and an
8,000-sample tail for a three-repeat burst plus one matched real-time repeat.
Do not treat this as the four-source conversation fixture, proven held-out
evidence, runtime/default authority, or a substitute for capture and live
validation.

## Context / why

The exact public repository at `0bd721e877c4a85d8c13ff837e68661ea6200a98`
materialized and independently validated the locked 24-case caller slice: three
utterances in each of eight task strata, 2,956,800 PCM bytes, and 46.2 seconds
of declared source audio. The checkout and private corpus stay outside Git;
only aggregate, path-free evidence is committed.

Burst completed 72 evaluations with zero final disagreement and exact source
accounting. WER/CER were .1562/.0677, with 45/72 exact and 69/72 nonempty
finals. Forty-eight evaluations accepted the first post-source feed EOU and 24
exhausted the 500 ms tail. Source/model-input RTF were .4524/.3712 and maximum
reported RSS was 303.895 MiB.

The paced repeat preserved every aggregate and task-stratum accuracy count.
It produced 15/24 exact and 23/24 nonempty finals, with 16 accepted endpoints
and eight tail exhaustions. Source/model-input RTF were .4736/.3886;
first-nonempty-partial p50/p95 were 615.498/788.595 ms; three cases lacked a
stable partial; four deadlines missed with 19.843 ms maximum backlog; and
maximum reported RSS was 304.012 MiB. Accepted-endpoint p50/p95 were 390/500
ms, measured only from declared source end to an observed feed boundary and
without endpoint ground truth.

The pinned NVIDIA model card does not disclose HarperValleyBank in its listed
training or evaluation datasets. Absence from that disclosure does not prove
that the exact audio or derived data were excluded, so training disjointness
remains unestablished. Its difference from the AMI diagnostic cannot be
separated from corpus, channel, segmentation, domain, and acoustic effects and
is not evidence of general conversation quality.

## Consequences

The candidate remains promising for these clean, single-channel caller excerpts
on one CPU, but it is not ready to become the stable voice-agent STT. The task
tags are transcript strata, not commands, and command/keyword attempts were
zero. Complete the
remaining Common Voice and EdAcc sources before the four-source wrapper; add
endpoint-ground-truth, owner-recorded command/negative, far-field/noise,
capture/VAD, AEC, open-speaker barge-in, identity/authority, tool, TTS, device,
and live-latency gates before adoption. The result cannot change setup, `core`,
`./live.sh`, or any default. Aggregate path-free evidence is in
`docs/evidence/2026-08-02-parakeet-cpp-harper-diagnostic.json`.
