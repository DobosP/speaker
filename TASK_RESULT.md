Valid until: this branch lands or is superseded — then treat as history.

# Task result — public voice evidence and production-model control

## Summary

Added a typed, no-download public voice-evaluation catalog that keeps STT,
STOP, far-field/device, noise/reverb, AEC/double-talk, overlap/turn, spoken
intent, and text routing as separate evidence tracks. Twelve source entries
are selectable only after their terms, credentials where applicable, immutable
content receipt, privacy restrictions, and deterministic subset rule pass;
nine unsuitable or unresolved sources remain explicit exclusions.

Added schema-v4 support for the exact installed fp32 GigaSpeech Zipformer used
by the desktop configuration. Its four artifacts, installed Sherpa/Numpy
versions, decode settings, and production four-thread profile are bound, while
the benchmark deliberately runs one-thread through networkless Bubblewrap with
read-only model/runtime/source mounts. This is a production-model control, not
runtime-equivalent evidence or a recognizer/default change.

## Main files changed

- `tools/public_voice_eval_matrix.py`: terms/receipt gates, usage constraints,
  deterministic selectors, ordered row digests, and deferred exclusions.
- `tools/streaming_stt/adapters/zipformer.py`: bounded streaming adapter with
  explicit after-PCM endpointing semantics and per-case reset.
- `tools/provision_zipformer_baseline.py` and schema-v4 streaming plumbing:
  exact local-artifact provisioning and isolated worker launch.
- Focused contract tests, ADR-0098, the evaluation guide, testing guide, and
  `STATUS.md`.

## Verification

- Full non-model logic gate: 6,563 passed, 13 skipped, 23 model-only
  deselected; nine pre-existing warnings.
- Public-matrix/Zipformer focused gate: 28 passed, 1 model-only deselected.
- Streaming-STT family: 296 passed, 1 skipped, 3 model-only deselected.
- Required APM/double-talk gate: 6 passed.
- Ruff and `git diff --check`: passed.
- All checks were headless, one-threaded for math runtimes, and low priority.
  No corpus/model download or load, audio device, or GPU validation is claimed.

## Risks and manual review

The catalog does not prepare any corpus, and the Zipformer control begins after
PCM is available. It therefore does not measure capture, VAD, endpoint
decisions, AEC, room echo, bare-speaker barge-in, or owner/multi-voice behavior.
Next work must prepare only permitted receipt-bound inputs and build a separate
bounded Parakeet Realtime EOU runtime before any GPU comparison. Fresh private
recordings and physical `./live.sh` A/B remain required for adoption.

## Merge recommendation

Green for landing after independent review. The full logic, focused, streaming,
APM, lint, and whitespace gates are green; no live-audio claim is made.
