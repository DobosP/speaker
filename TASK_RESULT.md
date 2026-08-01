Valid until: this branch lands or `main` advances beyond `0772ac1` — then treat
as history.

# Task result — licensed stratified STT evidence

## Outcome

Implemented a deterministic, receipt-bound derivative of the verified public
MInDS schema-v2 corpus using exact DEMAND kitchen, living-room, and
washing-machine noise. The private derivative contains 42 cases: the 14 source
utterances are balanced across all three environments and 20/10/0 dB SNR.
Publication requires a new absolute mode-0600 location outside every Git
worktree and binds source, archives, license, code, interpreter, and NumPy
identity. Inputs and provenance are rechecked before close.

The evaluator now preserves explicit bounded case tags and emits per-stratum
aggregates. A zero-attempt command-recall metric is `null`, not a perfect score.
A new 8-by-8 bounded suite controller runs one worker/corpus cell at a time,
binds evaluator identity across cells, rejects incomplete reports, and writes
failure-preserving private checkpoints. Its reports contain no corpus,
manifest, scratch, or transcript paths.

Rochester command speech is recorded as acquisition-only: its CC BY 4.0 source
is pinned, but no authoritative filename-to-transcript map was found. Fluent
Speech Commands and Sonos remain explicit license exclusions. The public matrix
now records 8 tracks, 15 sources, and 11 exclusions.

## GPU evidence

All cells ran sequentially on the RTX 4090 Laptop host with one math thread;
GPU candidates used the free GPU, Zipformer remained CPU-only, and no
microphone or audio device opened.

- Faster-Whisper Small clean/noisy WER: 0.6685/0.6757; RTF 0.0077/0.0071.
- Faster-Whisper Large-v3-Turbo: 0.6848/0.6830; RTF 0.0129/0.0123.
- CPU Zipformer control: 0.7283/0.9076; RTF 0.0699/0.0625.
- Parakeet clean/noisy WER: 0.6576/0.5429, but native EOU ended 85/126
  noisy evaluations before source exhaustion and one source changed final text
  across repeats. Its apparent WER advantage is invalid as adoption evidence.
- Paced clean Parakeet first/stable-partial p50 was 1.692/5.612 seconds with
  two deadline misses and 119.7 ms maximum backlog; 10/14 source utterances
  ended early.

Small is the best final-only comparator in this narrow corpus, not a runtime
promotion. The slice has only 14 unique source utterances, no command targets,
and no capture/VAD/AEC/turn-control or live conversational evidence.

## Files changed

- Added `tools/prepare_demand_noise_streaming_stt_corpus.py` and its tests.
- Added `tools/streaming_stt_suite.py` and its tests.
- Hardened public-source selection, corpus tag preservation, aggregate metrics,
  evaluator strata, and Faster-Whisper final-only semantics with adjacent tests.
- Added ADR-0109 and updated `STATUS.md`, agent routes, and testing guidance.

## Verification

- Final focused benchmark/preparer gate: 173 passed.
- Broad streaming gate: 599 passed, 1 skipped, 3 model-only deselected.
- Complete non-model split: 7,173 passed as 7,143 low-priority repository
  checks plus 30 isolated logging/path/thread-sensitive checks; 14 skipped,
  23 model-only deselected, 9 pre-existing warnings.
- Required APM/DTD: 6 passed.
- Independent review found and closed missing nested-stratum validation and
  incomplete transform provenance.
- Independent post-fix audit recomputed report, corpus, controller, evaluator,
  stratum, and runtime bindings; all private artifacts were mode 0600 and clean.
- `git diff --check`: clean.

## Limits / next

No runtime model, transcript selector, endpoint policy, entry point, tool,
enrollment policy, or default changed. No owner labels, command recall, device,
room echo, barge-in, natural-conversation, or live A/B result is claimed. Next
make one streaming-decode session own every online recognizer/stream operation,
then compare exact owner-labelled inputs and perform fresh `./live.sh` A/B.

## Merge recommendation

Land this review-hardened evidence slice. Preserve all licensed archives,
derivative audio, manifests, and aggregate reports outside Git.
