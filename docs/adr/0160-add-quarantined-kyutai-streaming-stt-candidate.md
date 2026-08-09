# ADR-0160: Add a quarantined Kyutai streaming STT candidate

Date: 2026-08-09
Status: accepted

## Decision

Add `kyutai-stt-1b-semantic-stream-v1` only to the isolated streaming-STT
benchmark as manifest schema v9. Keep it quarantined: it cannot be selected by
setup, `core`, `./live.sh`, the endpoint owner, the voice-agent tool/authority
plane, command authority, or any runtime/model default. Pin the CC-BY-4.0
`kyutai/stt-1b-en_fr-candle` model at revision
`095e38f6242006a93c2541149b181988397f5c7c`, Python 3.12.3, Moshi 0.2.11,
Torch 2.7.1+cu126/CUDA 12.6, Julius 0.2.8, the exact 58-wheel lock
`8d0e41563bb5e91500af42a912c5e6f825f16a67537ddc350556303e88609e25`,
and runtime content
`9edc9c42b8c718d0e4b17d917c04c05acc0b5ecca4ec9504866ad34f1f62dbf0`
(16,547 files; 5,694,993,765 bytes; largest file 984,633,129 bytes).

Provision only from an existing private runtime, offline wheelhouse, and exact
four-file model root. Generate and bind a fresh path-bearing runtime receipt at
the supplied runtime location, while requiring its wheel-install content
digest, file count, total bytes, and largest-file size to equal the frozen
tuple above. Do not freeze a receipt digest from one source path as the
portable identity. Provisioning must not resolve, install, download, import,
execute, or load candidate code or weights.

Run only through networkless Bubblewrap with the source bundle, runtime, and
direct model root read-only and private scratch writable. Before launch,
require Linux `MemAvailable` to be at least exactly 12 GiB: the 6 GiB worker
ceiling plus 6 GiB reserved for other host work. Before Ready, verify a
systemd-user cgroup-v2 scope with 5 GiB memory high, 6 GiB memory max, zero
swap, 100% CPU quota, 128 tasks, I/O weight 1 with idle scheduling, and grouped
OOM kill. Before model load, require at least 8,192 MiB free VRAM; cap the
PyTorch allocator fraction at 0.5, use one thread, and disable Torch compile
and CUDA graphs. The allocator fraction and free-VRAM preflight are guards,
not a hard VRAM limit.

Accept externally bounded complete 16 kHz PCM only. Append the declared
16,000-sample zero tail, align once, and perform one whole-buffer noncausal
resample to 24 kHz before executing the real 1,920-sample/80 ms Mimi and Moshi
LM frame loop. Match upstream initialization by passing the first Mimi code
frame through the LM twice and ignoring the priming output, then emit
diagnostic text snapshots at the fixed 160 ms cadence. Validate all four
six-dimensional semantic-head sets as finite on every step, but never use
them for endpointing, early stop, tools, or live authority. Endpoint ownership
is `none`; every source and declared-tail frame is consumed. Mark the wire
finalization placeholder and all aggregate finalization metrics not applicable,
because the last declared-tail-frame step is not speech-end latency.

## Context / why

Kyutai's delayed-stream STT is a relevant research candidate for more natural
partial text than final-only comparators, and its 1B model targets the laptop
CUDA path. It is not yet a portable or proven live recognizer. The available
Julius fractional resampler consumes the complete source-plus-tail buffer, so
the honest fidelity claim begins only at the subsequent Moshi frame loop; it
cannot be described as causal capture-to-text streaming. The duplicate first
frame is required by upstream `run_inference.py`, while the semantic heads are
not a substitute for Speaker's VAD, endpoint, trust, or action boundaries.

One real private provision completed without importing the candidate or
allocating the GPU. Its path-specific manifest SHA-256 is
`7fb4cff5702dd9de48738410447fa36bb809de45fcfb46bc806bf282932cf970` and
its artifact-set SHA-256 is
`7fa16c0f3f1310c0dac828d71b62463226b349799b0983344c3ffb9f94040138`.
At evaluation time the loaded host had only about 6.5--7 GiB available RAM,
below the exact 12 GiB launch requirement, so no worker/model execution was
appropriate. No real model load, GPU allocation, inference, corpus benchmark,
STT-quality result, latency result, device run, or live validation exists.

The deterministic implementation gate has 87 focused passes. The adjacent
manifest/runtime/wheel/sandbox/supervisor/evaluator gate has 382 passes and one
skip; APM/DTD has six passes. These are synthetic and provisioning-contract
results, not recognition evidence.

## Consequences

- Speaker can reproduce one exact Kyutai candidate boundary without adding its
  5.695 GB Python runtime to the production environment or changing the voice
  agent.
- A relocated but byte-identical runtime receives a new path-bearing receipt
  and manifest identity while retaining the frozen content tuple. The supplied
  private roots and unsigned local receipts remain a trusted same-owner
  boundary; they do not defeat a hostile same-UID swap-and-restore attack.
- When host `MemAvailable` is at least 12 GiB, run one bounded one-case smoke
  first. Only a clean smoke can authorize sequential aggregate corpus cells;
  neither outcome authorizes promotion.
- The large runtime should be reduced or replaced before multi-device work.
  Real accuracy, partial stability, resource, and after-PCM timing evidence
  must precede owner guided capture, bare-speaker barge-in, and natural live
  conversation validation. Defaults and authority remain unchanged throughout.

This ADR refines ADR-0089, ADR-0091, ADR-0099, ADR-0109, ADR-0115, ADR-0116,
ADR-0119, ADR-0128, and ADR-0159.
