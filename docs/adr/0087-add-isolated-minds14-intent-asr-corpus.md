# ADR-0087: Add an isolated MInDS-14 intent-ASR corpus

Date: 2026-07-31
Status: accepted

## Decision

Keep the public-v2 manifest, default path, suite names, preparation semantics,
eligibility, and metric behavior unchanged. Extend the shared evaluator report
provenance with the material `tools/public_voice_fixtures.py` hash. Add a
separate public-v3 `intent-asr` suite containing one deterministic en-US
MInDS-14 training example for each of its 14 intent labels. Pin revision
`40ce77cb32a384e4d50a568e1ec39ac804019d33`, the 34,196,221-byte Parquet
object, SHA-256
`37004471dc896ce20771b3fdda0ee8fb33ec7a030fb4e2fd047c561ec0a1ee30`,
and CC BY 4.0 provenance.

Select each intent's minimum rank using
`SHA256(seed UTF-8 || NUL || source SHA-256 ASCII || NUL || intent label UTF-8
|| NUL || source path UTF-8)`. Obtain the pinned Parquet file only through an
explicit, hash-verified `--source` override; do not relax the no-redirect
downloader for Hugging Face redirects.

Run Parquet preparation in a separate, explicitly selected virtual-environment
interpreter with exactly PyArrow 25.0.0, treating that interpreter and
environment as trusted local inputs. Resolve the lexical environment root,
reject production-`.venv` aliases, require the single effective `pyvenv.cfg`
to disable system site packages exactly once, and attest the isolated
interpreter's prefix, base prefix, user-site state, search path, PyArrow
version, and PyArrow location through bounded strict-JSON IPC. Recheck the
environment root, executable, and marker after reaping the probe's original
private process group.
Stable-read the regular repository worker, copy its exact bytes to a private
mode-0400 snapshot, and invoke only that snapshot with `-I -B`, a private
working directory, sanitized environment, file-only bounded IPC, no inherited
standard streams, resource limits, and a process-group timeout. The worker
reads its request and source through stable descriptors, hashes the source,
then passes that same open handle to PyArrow. It validates the exact Arrow and
Hugging Face schemas, 563 rows, six row groups, en-US language id 4, all 14
intent ids, embedded 8 kHz mono mu-law WAV bytes, and nonempty transcripts. It
never opens either Parquet path value. The parent independently validates
protocol/version provenance, ranks, transcripts, hashes, private file shape,
codec, and decoded duration before publishing 16 kHz arrays and metadata.

Treat this as aggregate whole-clip intent-domain ASR development evidence only.
It is not a held-out result, speaker-disjoint result, streaming/latency result,
intent-routing result, agent-action result, owner-voice result, or live
hardware result.

## Context / why

Public-v2 has useful command and meeting clips, but it does not provide a
small, transcript-labelled banking-intent slice for repeatable decoder
comparisons. MInDS-14 supplies embedded audio and transcripts with a clear
license and an exact canonical revision. Its upstream URL redirects, while the
existing downloader intentionally rejects redirects, so a verified local
source is safer than weakening that boundary.

PyArrow is large and is needed only to prepare this source. Adding it to the
voice agent's production environment would expand runtime dependency and
native-code risk without helping live inference. Python isolated mode and
original-process-group cleanup limit accidental dependency and lifecycle
exposure; neither is an operating-system process/filesystem/network sandbox,
and they cannot contain a descendant that creates a new session. Strict inputs,
private outputs, limits, and independent parent verification remain necessary.

Smart Speaker Commands contains phrase-level commands, but its archive has no
reliable machine-readable filename-to-phrase map; retakes and reordered files
make inferred numbering unsafe. It therefore cannot replace MInDS-14 as ASR
ground truth without circular STT labelling.

## Consequences

Normal fixture and evaluator tests do not import PyArrow. Public-v3 preparation
requires both the exact source override and an absolute isolated interpreter;
`--accept-download` cannot bypass either requirement. Prepared cases bind the
worker hash, protocol, PyArrow version, row, intent, language, source path,
transcript hash, original WAV hash, rank, and source provenance.

The 30-second per-case and 420-second aggregate decoded limits keep generated
float32 arrays below the evaluator's 32 MiB retained-corpus ceiling. Final
code-bound preparation produced 14/14 cases and 136.615 seconds of audio. Its
CUDA Faster-Whisper Small control decoded all 14 at WER 0.6685, CER 0.6412,
and whole-clip RTF 0.0121. This poor result rejects Small as an adoption
candidate; it is development accuracy only and cannot establish streaming,
held-out, live, or adoption evidence.
