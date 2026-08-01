# ADR-0099: Add and reject exact Parakeet Realtime EOU NeMo reference

Date: 2026-08-01
Status: accepted

## Decision

Keep NVIDIA Parakeet Realtime EOU 120M v1 as an isolated English/CUDA
benchmark reference only. Do not select it in normal runtime setup, expose it as
an application entry point, or adopt its NeMo runtime for desktop or portable
devices. Run it only through the schema-v5/protocol-v3 streaming harness with
networkless Bubblewrap, read-only authenticated inputs, private scratch, and a
verified systemd-user worker scope: 200% CPU, 6 GiB memory high, 8 GiB memory
max, zero swap, 128 tasks, and OOM kill. The adapter's PyTorch 25% memory
fraction is a guard, not a hard VRAM limit.

Freeze Python 3.12.3, NeMo 2.7.3, Torch 2.7.1+cu126/CUDA 12.6, NumPy 2.2.6,
the 183-wheel lock `b58bc18f...` (77,368 bytes), runtime content
`36224564...` (49,600 files, 6,430,910,098 bytes; largest file 984,633,129
bytes), model revision `a7e2b462...`, and model `6603a22a...` (460,062,720
bytes).

Treat candidate transcripts as deltas and accumulate them before emitting a
cumulative partial or final. A marker-only terminal retains accumulated text.
Only an EOU observed after all declared source audio is consumed can authorize
a response. EOB is a non-authoritative backchannel; source-early EOU/EOB and
tail exhaustion cannot authorize a reply.

## Context / why

Parakeet's native EOU/EOB stream is relevant to natural turn-taking, but the
official NeMo execution path is a research-scale Python dependency closure, not
the portable core required by this project. The first exact burst completed all
control-plane checks yet produced invalid accuracy evidence: 309 per-call text
deltas were misread as complete snapshots and marker-only terminal chunks
erased the final. That report (`dd597e70...`) remains preserved as failed
evidence.

The corrected 14-case MInDS-14 burst (`0bf07200...`; GPU telemetry
`813fc953...`) recorded WER 0.6576, CER 0.6161, 4/14 exact, model-input RTF
0.1910, 17.745 s load, 1,906.1 MiB sampled RSS, and 924 MiB reported PyTorch
VRAM. Its first-nonempty p50/p95 of 337/802 ms and finalization p50 of 15.1 ms
are accelerated-replay measurements, not conversational latency. All 14 native
events were EOU, but 10 preceded declared source end and left 198,848 of
2,185,844 source samples (12.428 s, 9.097%) unconsumed. Only 4/14 events were
response-authoritative; saturated p50/p95 endpoint probability of 1.0 did not
separate them.

The paced run (`4fdfffa5...`; GPU telemetry `3bcddeaa...`) preserved WER,
CER, exact count, partial count, churn, and retractions. Model-input RTF rose to
0.2200, with three deadlines missed and 189.411 ms maximum backlog. First
nonempty p50/p95 was 1.697/3.456 s and stable-partial p50/p95 was
5.617/26.736 s. Ten EOU events again preceded declared source end; 193,728
samples were unconsumed, so only 4/14 endpoints were authoritative. The changed
source boundary under pacing prevents treating native endpoint position as a
deterministic ground truth.

On the identical corpus, the one-thread fp32 Zipformer control recorded WER
0.7283, CER 0.6259, 2/14 exact, RTF 0.089, 1.841 s load, and 378.8 MiB sampled
RSS. Parakeet therefore made 13 fewer word errors, but its preferred
model-input RTF was 2.146 times the control, its load was 9.641 times longer,
and its sampled RSS was 5.032 times larger. Different CPU/GPU resource profiles
make this a development comparison, not an adoption result.

## Consequences

The benchmark implementation and deterministic no-model tests may land, but
`.venv`, configuration, setup, defaults, and `./live.sh` remain unchanged. The
next runtime experiment is receipt-bound `parakeet.cpp`; recognition and turn
behavior must next be tested on Common Voice Spontaneous Speech v4, followed by
disjoint owner recordings, synchronized capture/AEC replay, multiple device
classes, and fresh physical bare-speaker A/B.

The evidence is one repeat over a small development corpus and begins after PCM
is available. It does not cover capture, VAD, AEC, device/room behavior, owner
identity, multiple speakers, semantic endpoint ground truth, or live agent
latency. RSS is sampled and GPU telemetry is device-wide. Two initial paced
power samples reported an impossible 593.51 W and are driver outliers; they are
retained rather than corrected.

The isolation also has explicit trust limits: local receipts are unsigned;
pre/post path hashes cannot exclude a same-UID transient model swap-and-restore;
stat-to-unlink cleanup is not atomic against a hostile writer; fd 1/2 suppression
is synchronous and process-global, so delayed native/background output can only
make the protocol fail closed; authenticated module cleanup assumes one isolated
importer; and pre-ready cgroup evidence does not prove post-exit emptiness.
