# ADR-0135: Correct and pin Smart Turn v3.2

Date: 2026-08-05
Status: accepted

## Decision

Keep the lexical turn-completion detector as the default and Smart Turn v3.2
as an explicitly configured prosody option. Require callers to supply 16 kHz
audio and reject every other sample rate before even a thin-audio neutral
return; callers own any upstream-quality resampling. Retain the last 128,000
waveform samples, LEFT-pad shorter turns to 128,000 samples, and only then
perform float32-equivalent zero-mean/unit-variance normalization and Whisper
log-mel extraction. Treat the resulting features as end-aligned. Accept an
inference result only when ONNX returns exactly one output tensor containing
exactly one finite floating-point scalar in `[0, 1]`; the scalar is already
`P(turn complete)` and must not receive another sigmoid.

Bind setup to the exact `pipecat-ai/smart-turn-v3` CPU artifact at Hugging Face
revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`: filename
`smart-turn-v3.2-cpu.onnx`, size 8,679,182 bytes, SHA-256
`2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f`,
license BSD-2-Clause. Use the immutable revision URL by default. Permit a
user-supplied mirror only when its downloaded bytes match the same exact size
and SHA-256. Verify both existing and downloaded files by descriptor against
that identity; reject symlinks, hardlinks, and non-regular files. Stage a
download in a sibling temporary file, flush and verify it, atomically replace
the destination, and verify the published path. A failed forced refresh must
preserve a valid active artifact.

When prosody is configured, construct the ONNX session and run one content-free
input through its input/output contract before capture starts. Select the
lexical detector once if loading or validation fails, or if Sherpa's configured
model sample rate is not 16 kHz. Make
`tools.turn_detect_check` call this production detector instead of maintaining
a second feature/inference implementation. Do not change endpoint thresholds,
the lexical default, model selection, barge-in, enrollment, tool authority, or
entry points.

## Context / why

The prior numpy path truncated from the beginning correctly but normalized the
short waveform before RIGHT-padding it. Smart Turn's published contract first
LEFT-pads short turns, placing the most recent speech at the end of the input,
then asks Whisper's feature extractor to normalize the full padded buffer. This
is specified by
[`audio_utils.py`](https://github.com/pipecat-ai/smart-turn/blob/4786657e242dfe77dd138699ac564ee074a2a543/audio_utils.py),
used by
[`inference.py`](https://github.com/pipecat-ai/smart-turn/blob/4786657e242dfe77dd138699ac564ee074a2a543/inference.py),
and retained by Pipecat's current
[`local_smart_turn_v3.py`](https://github.com/pipecat-ai/pipecat/blob/0db3c9a0a8d4982c997afd073a3f6372e9d44515/src/pipecat/audio/turn/smart_turn/local_smart_turn_v3.py)
plus its
[`_whisper_features.py`](https://github.com/pipecat-ai/pipecat/blob/0db3c9a0a8d4982c997afd073a3f6372e9d44515/src/pipecat/audio/turn/smart_turn/_whisper_features.py).

The layout error was large on ordinary short turns. Against the authoritative
path, deterministic 0.5-second, 2-second, and 7-second inputs had maximum
absolute feature errors of 2.3010, 2.1505, and 2.0140 respectively; the
8-second input was within float32 numerical tolerance. Therefore the retained
2026-06-01 owner-voice ranges and any reports generated through the duplicated
diagnostic path do not describe the production model contract and cannot set a
threshold, lower an endpoint floor, or promote prosody.

Setup previously downloaded from mutable `resolve/main` and trusted any
existing file with the expected name. The engine factory returned a lazy
detector before ONNX had parsed or exercised that file. A corrupt or
incompatible artifact could consequently survive setup and factory selection,
then fail repeatedly during endpoint decisions. Filename trust and per-turn
runtime fallback are weaker than validating the one small optional artifact at
acquisition and startup.

The earlier detector also accepted non-16 kHz input by applying linear numpy
interpolation. Upstream Pipecat uses HQ soxr when its pipeline rate differs from
the model rate; a deterministic 48 kHz probe materially changed both features
and probability under the linear approximation. Adding a second resampler or a
new runtime dependency inside endpointing would duplicate the caller's audio
boundary. Failing closed at 16 kHz keeps the dependency-light detector contract
honest. Sherpa may still capture natively at 48 kHz because its existing front
end converts capture audio into `config.sample_rate`; prosody is selectable only
when that model-band rate is 16 kHz.

The audit-only deterministic 2-second 48 kHz probe quantified that mismatch:
upstream-HQ `AudioResampler` features versus the removed linear interpolation
had maximum/mean absolute errors `1.5211018`/`0.0144331`, and the same pinned
model returned removed-linear `P=0.1267537` and upstream-HQ `P=0.0115352`.
These numbers justify rejecting the approximation; they are not endpoint-quality
or latency evidence.

## Consequences

Short-turn feature layout now matches the training/inference contract without
adding Transformers or Torch to the ordinary runtime. Dependency-free tests
pin full-array parity across 0.5, 2, 7, 8, and 9 seconds as well as six fixed
values generated independently by the official Whisper feature extractor.
All parity evidence is for caller-provided 16 kHz audio. Tests also cover direct
non-16 kHz rejection before the neutral return, strict startup rejection of
malformed/nonintegral/nonfinite model rates, factory fallback for a non-16 kHz
model configuration, floating output dtype/cardinality/range, eager lexical
fallback, production diagnostic delegation, immutable metadata,
symlink/hardlink/tamper rejection, interrupted acquisition, and preservation of
a valid model across an invalid forced refresh. The Slaney 0 Hz inactive log
branch no longer emits a divide warning; selected filter values are unchanged.

The exact pinned CPU file was checksum-validated from
`/tmp/speaker-smart-turn-v3.2-cpu.onnx`. Two bounded real-model tests exercised
eager loading and inference. Deterministic 1-, 4-, and 8-second synthetic
fixtures produced identical probabilities through production and Pipecat's
vendored numpy feature implementation at source commit
`0db3c9a0a8d4982c997afd073a3f6372e9d44515`, using the same pinned ONNX. This
numpy-reference observation is distinct from the fixed official-Transformers
feature probe and is artifact, preprocessing, and ONNX contract evidence only.
It is not speech-quality, endpoint-accuracy, latency, threshold, microphone,
device, GPU, owner-voice, or live evidence.

Enabling prosody now performs one additional content-free inference during
startup and falls back to lexical if it fails. The model remains unbundled and
optional. Regenerate complete/incomplete owner recordings through
`tools.turn_detect_check`, then pass causal endpoint replay and a live A/B
before considering any default, floor, or threshold change.
