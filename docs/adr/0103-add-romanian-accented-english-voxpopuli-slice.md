# ADR-0103: Add a Romanian-accented English VoxPopuli STT slice

Date: 2026-08-01
Status: accepted

## Decision

Add an evaluation-only preparer for the VoxPopuli `en_accented` test split at
exact Hugging Face revision
`42f01879c780b4a2e90ec0b4f616c2ece526e4f1`. Require both local Parquet shards
as explicit absolute overrides, with byte counts and SHA-256 values fixed to
the official LFS objects; provide no network or download path. Record the
dataset's CC0 declaration together with the European Parliament recording
provenance and legal notice, and require both acknowledgements in the private
preparation receipt.

Read the shards only through an explicitly selected virtual-environment
interpreter with PyArrow 25.0.0. Reuse ADR-0087's environment probe and execute
a stable-read private worker snapshot with `-I -B`, sanitized one-thread
environment, inherited source descriptors, bounded batches, resource limits,
discarded standard streams, a private process group, and a timeout. The worker
must hash the two exact descriptors, validate the exact nine-column Arrow and
Hugging Face feature schema, and never open an embedded audio path. The parent
must independently hash the unchanged descriptors after the worker finishes.

Filter for language class 16, `accent=en_ro`,
`is_gold_transcript=true`, nonempty normalized text, and strict mono 16 kHz
PCM16 WAV audio from 2 through 25 seconds. Select 24 distinct speakers with one
utterance each and six cases in each duration band `[2,6)`, `[6,10)`,
`[10,15)`, and `[15,25]`. Choose the minimum deterministic row per
speaker/band, then use a deterministic bipartite assignment so a speaker
eligible in multiple bands cannot make a feasible quota fail greedily. Bind
the complete recipe, eligible-set digest, worker/schema/source identities,
selected identity hashes, transcript hashes, source-WAV hashes, and decoded
PCM hashes into a raw-path/raw-identity/transcript-free private receipt; omit
gender and exact source-row coordinates. Publish the
result only through the hardened shared schema-v2 streaming-corpus writer.

## Context / why

The existing MInDS-14 slice is scripted banking speech, Common Voice
Spontaneous Speech needs MDC credentials and has no targeted Romanian-accented
English stratum, and Speech Commands is limited to isolated keywords.
VoxPopuli's accented-English test split is openly accessible and contains
formal spontaneous European Parliament speech. Its published Romanian-accent
subset is 1.85 hours from 27 speakers, directly exercising a likely failure
mode for Paul's English speech without using his private recordings.

The complete `en_accented` test consists of 8,387 examples. The two pinned
objects are 2,471,743,754 bytes with SHA-256
`202d71bde2680f30aa7ae1050a98cdf85972458f94e6493db2aa7b8dd8ff46ec`
and 2,474,983,297 bytes with SHA-256
`7d61b4c7a2e5c565a382543236442a7a8dd081c985e52dd1084905e4becc3901`.
Together they are 4,946,727,051 bytes, too large to copy into a worker or load
eagerly. Descriptor transport, metadata-only first scans, selected-row-group
audio batches, and a worst-case 21,504,000-byte decoded corpus keep the path
bounded without weakening content identity.

Sources: the official [dataset card](https://huggingface.co/datasets/facebook/voxpopuli/blob/main/README.md),
[upstream repository](https://github.com/facebookresearch/voxpopuli),
[ACL paper](https://aclanthology.org/2021.acl-long.80), and
[European Parliament legal notice](https://www.europarl.europa.eu/legal-notice/en/).

## Consequences

The normal runtime, setup, entrypoint, STT defaults, and production environment
remain unchanged. Preparation needs roughly 4.61 GiB of already-acquired local
source data plus a separate exact PyArrow environment; it may make multiple
low-priority sequential reads but retains only bounded metadata and selected
audio batches. The worker boundary limits accidental dependency and lifecycle
exposure but is not an operating-system filesystem or network sandbox.

This slice is a formal-spontaneous L2-accent after-PCM diagnostic. It is not a
domestic conversation, command-intent, capture, device-noise, VAD, endpoint,
AEC, barge-in, agent-tool, owner-voice, live-latency, or model-adoption test.
VoxPopuli is widely used in speech-model development, so candidate training
overlap must be checked and unknown overlap reported. Synthetic preparation
tests do not establish compatibility with the real Parquet schema/audio; the
first real preparation remains a separate reviewed gate.
