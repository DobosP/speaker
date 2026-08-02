Valid until: main advances beyond this task's landing commit — then treat as
history.

# Task result — parakeet.cpp Harper Valley diagnostic

## Outcome

The exact ADR-0119 parakeet.cpp candidate completed the locked private
HarperValleyBank 24-case caller corpus in three-repeat burst and one-repeat
paced modes. Both reports are aggregate-only and path-free. No candidate
artifact, model, transcript, case row, PCM, source checkout, or local path is
committed.

The real-time repeat exactly matched burst accuracy: WER .1562, CER .0677,
15/24 exact, and 23/24 nonempty finals. Sixteen cases accepted the first
post-source feed EOU; eight exhausted the 500 ms tail. First-nonempty-partial
p50/p95 were 615.498/788.595 ms, three cases lacked stable partials, and four
deadlines missed with 19.843 ms maximum backlog. The worker stayed at one
thread and 304.012 MiB maximum reported RSS under its verified no-network,
no-NVIDIA hard scope.

## Receipts

The candidate manifest SHA-256 is
`906d1feb277780056b636f4c837ea7ca7985b5fb02eef9cc24a50e5e84c381c4`.
The corpus manifest and preparation-receipt SHA-256 values are
`4091a9dd163a604989db2f87f90d719347f0fb43b246b95691c829203f29d2bb`
and `ef84709b3d6fb193937ed0f9557ffa8492476f0ad01faf81a2d2779ab0bad266`.
The 40,323-byte burst report SHA-256 is
`b3eb6cb195266c79ef5f212ffdf528057427675d1f7467456d5b36829bd354c1`;
the 40,256-byte paced report SHA-256 is
`1c01daba667ed96ebef33c81413db27e79bf957fb0d859965dc12fca75d8710e`.

## Interpretation and next work

The [pinned public Harper Valley repository](https://github.com/cricketclub/gridspace-stanford-harper-valley/tree/0bd721e877c4a85d8c13ff837e68661ea6200a98)
describes simulated contact-center conversations and separate caller/agent
audio. The [pinned official NVIDIA model card](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1/blob/a7e2b4629593dce0ec19f600e00e9904353fda2d/README.md)
does not list Harper Valley in its disclosed training or evaluation datasets,
but that absence does not establish exact training disjointness. This result
therefore cannot promote the candidate. It also bypasses capture, VAD,
endpoint ground truth, AEC, devices, barge-in, identity, tools, TTS, and live
conversation.

Keep the complete pinned source checkout, corpus, receipts, scratch, and
reports private and retained. Complete Common Voice only after owner term/token
acceptance; keep EdAcc deferred until its archive and expansion have deliberate
disk budget. In parallel, use owner-labelled recordings and live A/B for the
actual target commands and room.

## Verification

The production-shaped single-source validator passed for 24 cases and
2,956,800 PCM bytes. Burst/paced scaled accuracy and source/tail accounting
matched exactly, and both reports passed the path/transcript privacy scan. The
focused fixture/candidate gate passed 65 tests; APM/DTD passed 6; strict
aggregate-evidence cross-check and `git diff --check` passed.
