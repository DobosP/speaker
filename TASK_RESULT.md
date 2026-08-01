Valid until: this branch lands or ADR-0101 is superseded — then treat as history.

# Task result — bounded Common Voice spontaneous STT slice

## Outcome

Added Common Voice Spontaneous Speech 4.0 English to the public evaluation
catalog and implemented a Linux/POSIX, no-download preparer for a deterministic
private 32-speaker P0 slice. Runtime STT defaults, setup, `python -m core
--session`, and `./live.sh` are unchanged.

The preparer requires the exact release/API identity and current terms, a
canonical owner-only archive outside every Git checkout, and a new private
output. It twice hashes the same archive descriptor, never reads the reported
TSV body, safely parses the bounded archive/metadata, and uses speaker-first
deterministic quota matching for eight clean test clips in each of four duration
bands. It decodes only selected MP3s into bounded mono 16 kHz f32le PCM.

The shared schema-v2 writer now validates cases, provenance, normalized
references, and the exact <=256 KiB manifest before output creation. It retains
the created directory descriptor through no-overwrite writes and final
identity/sidecar/hash checks. Decoder/package and local preparer-code closures
are bounded, path-free, measured before work, and required unchanged after
publication; injected decoders are explicitly test-only.

## Files changed

- Catalog/selector: `tools/public_voice_eval_matrix.py` and its tests/docs.
- Preparer: `tools/prepare_common_voice_spontaneous_v4.py` and focused tests.
- Shared publication: `tools/streaming_stt/corpus_writer.py`, the existing
  public converter, and writer regressions.
- Optional isolated decode pins: `requirements-evaluation.txt`.
- Decision/status/operator evidence: ADR-0101, `STATUS.md`, testing guide, and
  public evaluation guide.

## Verification

All commands used one-thread math, `ionice -c 3`, and `nice -n 15` limits.

- Integrated SPS/writer/catalog gate: 86 passed in 1.12 s.
- Expanded streaming-STT gate: 653 passed, 3 real-model deselected in 9.44 s.
- Complete non-model repository gate: 6,937 passed, 13 skipped, 23 deselected,
  9 pre-existing warnings in 119.17 s.
- Required APM/double-talk gate: 6 passed in 0.96 s.
- Forced clean-CI/no-PyAV focused path: 84 passed, 2 skipped.
- Scoped Ruff check and format check: passed.
- `git diff --check`: passed.

The installed optional PyAV environment exercised a synthetic MP3 encode/decode
and complete PyAV/FFmpeg/NumPy closure receipt. Clean CI can omit that optional
dependency; those two tests use explicit per-test skips rather than failing
module import. Independent security, compatibility, and writer-race audits
identified the issues above; their requested landing blockers were fixed.
Three post-fix read-only audits found no landing blocker.

## Limits and manual follow-up

No corpus was downloaded, no real SPS archive was opened, no STT model ran, and
no network, GPU, microphone, speaker, or live route was used. Therefore this
proves preparation/publication contracts, not real-release compatibility, WER,
latency, natural conversation, capture/VAD/AEC, tools, owner voice, or live
quality. The 32 clean cases are a development slice, not an official full-test
score, quality-tagged stratum, Romanian result, or training-disjoint claim.

Failed publication may intentionally retain private partial evidence; only exit
zero is readiness, and publication is not advertised as an atomic readiness
marker. Next, the owner must acquire the archive under MDC terms into a
canonical mode-0600 non-Git path, run the preparer, then execute isolated
one-thread candidate-model comparisons before any separate `./live.sh` A/B.

## Merge recommendation

Recommend fast-forwarding this green, post-audit branch. Do not promote a
recognizer or runtime default from this headless preparation milestone.
