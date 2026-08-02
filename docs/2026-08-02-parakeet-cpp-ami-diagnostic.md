Valid until: main advances beyond this task's landing commit — then treat as
history.

# Task result — parakeet.cpp AMI close/far diagnostic

## Outcome

The exact ADR-0119 parakeet.cpp candidate completed the private production AMI
24-case corpus in three-repeat burst and one-repeat paced modes. Both reports
are aggregate-only and path-free. No candidate artifact, model, transcript,
case row, PCM, or local path is committed.

The real-time repeat exactly matched burst accuracy: overall WER .4684, close
WER .2278, and far WER .7089. Only 3/12 far cases produced nonempty finals,
versus 11/12 close cases. First-nonempty-partial p50/p95 were 622.543/2,369.120
ms, ten cases lacked stable partials, and two deadlines missed with 14.831 ms
maximum backlog. The worker stayed at one thread and 308.105 MiB maximum
reported RSS under its verified no-network/no-NVIDIA hard scope.

## Receipts

The candidate manifest SHA-256 is
`906d1feb277780056b636f4c837ea7ca7985b5fb02eef9cc24a50e5e84c381c4`.
The corpus manifest and preparation-receipt SHA-256 values are
`5a8ba97f71c5c6c69013def3682c62bc6f4f902fc51d24a2a51ea92f46333c7f`
and `c6ca84004344c275498c787b760df0d3dfdad65876c24e84816bd8d74ba33810`.
The 30,971-byte burst report SHA-256 is
`cbf25522a43513d7443fd9cb6c15f21732bf7ccb1460beeab601b68669d1d221`;
the 30,863-byte paced report SHA-256 is
`02ce75d6f1e6382fd84c4d4bcb008596f0de24053f7c369b62d6244dc9041cbe`.

## Interpretation and next work

The [pinned official NVIDIA model card](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1/blob/a7e2b4629593dce0ec19f600e00e9904353fda2d/README.md)
lists AMI in both training and evaluation datasets but does not identify exact
clip membership. This run therefore does not establish training disjointness
and cannot promote the candidate. Its WER is not comparable to NVIDIA's AMI
leaderboard result because selection, segmentation, normalization, channels,
and first-EOU policy differ. It also bypasses capture, VAD, endpoint ground
truth, AEC, devices, barge-in, identity, tools, TTS, and live conversation.

Acquire a disjoint locked source next. Common Voice SPS v4 is the smaller
download but requires owner acceptance and an MDC token; Harper Valley is
public but its current complete-checkout contract costs several gigabytes.
Keep EdAcc deferred until there is at least 16–20 GiB of deliberate scratch
budget for its 5.92 GB archive and expansion.
