# ADR-0098: Separate public voice evidence and bind the production-model baseline

Date: 2026-08-01
Status: accepted

## Decision

Require public voice evidence to use a typed, no-download catalog with distinct
STT, STOP, far-field/device, noise/reverb, AEC/double-talk, overlap/turn,
spoken-intent, and text-routing tracks. Keep raw data in a private external
cache, evaluation-only, behind explicit source terms, usage restrictions, and
content receipts. Any reduced row set must use the catalog's executable fixed,
SHA-256 top-k, or stratified round-robin selector and publish its ordered
identity digest. Do not make a corpus selectable when its license, permitted
use, immutable content, or verification route is unresolved.

Add schema-v4 Sherpa Zipformer as the comparison baseline using the exact four
installed fp32 GigaSpeech artifacts and current decode/endpoint settings. Run
it with the installed Sherpa 1.13.3/Numpy 2.4.6 metadata contract inside a
networkless Bubblewrap worker with read-only model/environment mounts. Record
that the resource-controlled benchmark uses one ASR thread while the selected
`desktop_gpu_4090` production profile uses four; report
`runtime_equivalent_to_production=false`. Do not download, install, change an
ASR default, or expose the baseline through application setup.

## Context / why

The existing 14-case MInDS development slice and isolated Moonshine/Nemotron
runs could rank recognizers after PCM was available, but they mixed neither the
target failure modes nor the metrics needed for a natural agent. WER cannot
stand in for STOP false alarms, endpoint cuts, echo removal, overlap behavior,
or tool routing. Several attractive datasets also have non-commercial,
research-only, changing, or corpus-specific privacy terms. Prose-only subset
recipes and loader-revision pins were not reproducible content evidence.

The production streaming recognizer was also missing from the isolated harness.
The older benchmark download defaults select int8 files, while this machine's
actual local session selects the fp32 `chunk-16-left-128` files. Comparing a
candidate to that stale path would answer the wrong question. Four production
threads would also compete with the owner's active CPU/RAM workload, so the
thread override must be explicit rather than passed off as production latency.

## Consequences

The catalog can fail closed before acquisition and records Mozilla
no-reidentification/no-rehosting constraints, the Southern-American
no-voice-cloning restriction, mixed AEC source terms, local SHA-256 freezes,
and deliberately deferred sources. It still downloads and prepares nothing;
corpus-specific preparers must later bind selected rows, archives, annotations,
and any LFS objects without exposing tokens, audio, or transcripts.

Zipformer becomes a reproducible production-*model* control for the same
aggregate streaming evaluator, not a new recognizer and not exact runtime or
live evidence. Its package closure is version-checked rather than content-
receipted, its one-thread latency is diagnostic, and the harness begins after
PCM is available. Candidate promotion still requires the capture-loop replay,
disjoint owner recordings, synchronized AEC/barge evidence, multi-device
resource checks, and a fresh bare-speaker `./live.sh` A/B required by ADR-0092.
