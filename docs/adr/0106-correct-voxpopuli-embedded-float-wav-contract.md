# ADR-0106: Correct the VoxPopuli embedded float-WAV contract

Date: 2026-08-01
Status: accepted

## Decision

This ADR supersedes only ADR-0103's embedded-audio format and decoding
decision: its PCM16 assumption. It carries forward ADR-0103's other decisions:
the exact two local revision-pinned shards and explicit legal acknowledgements;
the isolated PyArrow 25 descriptor worker and exact schema/filter checks; the
deterministic 24-distinct-speaker, six-per-duration-band selection; private
schema-v2 publication through the shared writer with provenance-bound,
transcript-free receipts; and the non-runtime, non-adoption evidence boundary.

Require every inspected VoxPopuli `en_ro` embedded audio value to be an exact
RIFF/WAVE container with only, and in
order, an 18-byte `fmt ` chunk, a four-byte `fact` chunk, and a `data` chunk.
Require format tuple `(3, 1, 16000, 64000, 4, 32)`, zero `fmt ` extension,
little-endian IEEE binary32 data, and a `fact` sample count equal to
`data_bytes / 4`. Treat an exact zero-data container as valid but ineligible.
Require selected rows to contain data. Have the parent independently repeat
the container, size, fact, and sample-count checks; reject non-finite selected
samples and preserve all finite binary32 values without clipping,
normalization, or an invented amplitude bound. Bind this contract into the
worker schema hash, protocol v2, and preparer provenance.

## Context / why

The first real preparation attempt failed closed because ADR-0103 and its
synthetic fixtures incorrectly assumed PCM16 WAV input. A bounded aggregate-only
inspection of the pinned shards then found 485 `en_ro` embedded WAV values,
all with the exact float32 container above: 378,123,174 total bytes, matching
embedded paths, and three valid empty-data rows. An aggregate selection
simulation found 445 eligible nonempty rows from 27 speakers and a feasible
deterministic 24-row selection with six rows in each duration band, totaling
3,903,740 samples. No transcript, speaker identity, path value, or raw audio
left the private inspection boundary.

IEEE float WAV does not normatively restrict samples to `[-1, 1]`. Adding that
bound would reject otherwise valid source values and silently turn a container
contract into an unverified loudness policy. Finite-only identity conversion
instead matches the schema-v2 corpus contract while preventing NaN or infinity
from entering scoring.

## Consequences

Synthetic tests now exercise the actual source container, exact empty-row
handling, independent parent validation, non-finite rejection, and source
path/size/hash failures. The failed attempt and aggregate inspection do not
constitute a successfully published corpus, model run, WER, latency, live, or
adoption result. A new private output must complete real preparation before any
Faster-Whisper comparison. Runtime defaults, setup, `./live.sh`, and
`python -m core --session` remain unchanged.
