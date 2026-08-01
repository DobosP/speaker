# ADR-0109: Add licensed stratified STT evidence

Date: 2026-08-01
Status: accepted

## Decision

Add a license- and checksum-bound public-noise evidence path without changing
the voice runtime. Treat the University of Rochester Command Speech Dataset
(DOI `10.60593/ur.d.26417548.v1`, CC BY 4.0) as acquisition-only until its
publisher supplies an authoritative filename-to-transcript map; never infer
labels from numbering, retakes, or model output. Use the DEMAND domestic-noise
archives (DOI `10.5281/zenodo.1227121`) under the more conservative published
CC BY-SA 3.0 terms. Accept only the pinned `DKITCHEN`, `DLIVING`, and
`DWASHING` archives, read channel 01, and deterministically mix non-overlapping
noise with an already verified public schema-v2 corpus at 20, 10, and 0 dB
SNR. Require a new absolute mode-0600 output outside every Git worktree, bind
the exact corpus, archives, licenses, every executed repository dependency,
the relevant stdlib modules, interpreter, and complete NumPy package/library
trees in its receipt, and recheck them before publication.

Expose only bounded aggregate strata from the evaluator. Preserve case tags in
the corpus writer, accept an explicit finite tag allow-list, and report counts,
WER/CER, timing, endpoint, disagreement, resource, and error aggregates per
selected tag. Report command recall as `null` when no command was attempted.
Run at most an 8-worker by 8-corpus cross-product strictly one cell at a time;
bind one evaluator identity across cells, reject incomplete nested reports,
and atomically retain a private red checkpoint before and after each cell.
Keep Fluent Speech Commands and the Sonos corpus excluded because their source
terms do not authorize this product-development benchmark.

Do not promote a model from this slice. Faster-Whisper Small is the strongest
final-only comparator here: clean/noisy WER is 0.6685/0.6757, versus
0.6848/0.6830 for Large-v3-Turbo and 0.7283/0.9076 for the CPU Zipformer
control. Parakeet reports clean/noisy WER 0.6576/0.5429, but its native EOU
ended 85 of 126 noisy evaluations before source exhaustion and one source case
changed final text across repeats, so that apparent advantage is not valid
turn-control or adoption evidence.

## Context / why

The existing 14-case public MInDS slice exercises intent-like speech but not
domestic noise, owner commands, capture, or conversation. Repeating the same
small source set under controlled kitchen, living-room, and washing-machine
noise gives a reproducible robustness diagnostic while preserving the exact
licensed transcripts. It does not create new speakers or an independent test
set, and it cannot validate command recall because the source declares no
command targets.

Rochester supplies useful command audio and phrase documentation, but no
publisher-bound row map was found. Guessed labels would make WER meaningless;
using candidate transcripts as labels would make the benchmark circular.
Likewise, a low WER after a recognizer discards the difficult tail is not
evidence of natural endpointing. Both cases must fail closed instead of
producing an attractive but invalid score.

## Consequences

The public evaluation matrix now records 8 tracks, 15 sources, and 11 explicit
exclusions. Licensed archives and derivative audio stay private and outside
Git; aggregate reports contain no transcript rows or input paths. The hardened
DEMAND derivative contains 42 cases, balanced across three environments and
three SNR levels, from 14 source utterances.

The suite makes sequential GPU comparisons reproducible and preserves evidence
from completed cells when a later worker fails. It cannot prove microphone
capture, VAD, AEC, native streaming ownership, owner identity, barge-in,
agentic tool use, conversational latency, or live quality. Faster-Whisper
comparators remain final-only, and Parakeet remains rejected for endpoint
control. Models, transcript selection, runtime defaults, `./live.sh`, tools,
and enrollment policy remain unchanged. Promotion still requires disjoint
owner-labelled exact-input replay and fresh live A/B after the streaming-decode
owner is made explicit.
