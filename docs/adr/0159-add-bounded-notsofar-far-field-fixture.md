# ADR-0159: Add a bounded NOTSOFAR-1 far-field fixture

Date: 2026-08-09
Status: accepted

## Decision

Keep public matrix v5 and its canonical four-source qualification unchanged.
Add a separate, benchmark-only NOTSOFAR-1 fixture pinned to the CC-BY-4.0
`240629.1_eval_small_with_GT` source at Hugging Face revision
`bb97bae9458f477a85dfd6e8aee92e35d5887dd3`. The offline materializer accepts
only meeting `MTG_32006`, its three exact metadata files, and the two exact
single-channel far-field captures `sc_meetup_0/ch0.wav` and
`sc_rockfall_2/ch0.wav`; their combined pinned input is 24,738,302 bytes.

Publish exactly nine deterministic, mutually non-overlapping, single-speaker
windows, with three windows from each of three eligible speakers. Replay every
window through both synchronized devices to form one 18-case schema-v2 public
corpus. The two members of a pair must share the exact interval, reference,
sample count, and non-device tags. Source paths, transcript text, and raw
speaker/device identifiers stay out of the aggregate preparation receipt.

Only these isolated windows may contribute ordinary WER/CER. The meeting's
`#DebateOverlaps` label is source metadata, not selected overlap evidence. This
fixture has zero overlap cases and supplies no diarization, speaker-enrollment,
owner-identity, endpoint, capture, AEC, tool, live-latency, or default-promotion
evidence. Admit real model results only as a separate development comparison;
keep every runtime model and entry-point default unchanged.

## Context / why

The retained AMI close/far slice is useful, but it does not close the next
roadmap gap: a small, independently pinned real-room source whose speakers and
rooms are disjoint from the source's training/development partitions.
NOTSOFAR-1's open eval-small partition supplies that property, while dev-set-1
overlaps training speakers and dev-set-2 is restricted to challenge
publications. Downloading the full sixteen-hour track would be disproportionate
for this laptop and for routine candidate screening. One paired six-minute
meeting gives a bounded device-robustness gate without weakening provenance.

The current public control reinforces the need for layer-appropriate evidence.
Faster-Whisper Small decoded nine of ten spoken command clips exactly and much
faster than real time, but produced text for a whole-clip silence case. That
does not establish a live regression: production VAD rejects finals when no
speech was observed, and a verifier cannot manufacture a turn from an empty
streaming result. The new fixture therefore targets the missing far-field
accuracy dimension instead of adding a redundant live no-speech filter.

The 2026-08-09 upstream audit also does not justify a framework rewrite.
LiveKit and Pipecat validate the existing typed event, cancellation, and
transport direction; their useful next ideas are publisher-scoped sessions,
media timing/backpressure fields, and speculative response leases. Kyutai's
1B delayed-stream model is the most relevant future streaming challenger, but
its resource envelope must be measured locally before integration. Moonshine
Voice 0.1.1 improved backlog handling while retaining unresolved global VAD
state and tail-loss defects; Voxtral Realtime requires at least a 16 GB GPU.
None of those findings changes this fixture or authorizes a model promotion.

The first exact development replay used the retained manifest-bound evaluation
checkout at `52b7aef`. Faster-Whisper Small completed 18/18 cases at WER/CER
`.1538`/`.1030`, 13 exact clips, and `.0638` aggregate RTF; the one-thread
Zipformer control completed 18/18 at `.5096`/`.3467`, zero exact clips, and
`.1124` RTF. Preserve their mode-0600 aggregate reports with SHA-256
`df7fadd455f9de38639ea99fd69ec057d25dc7a22700a11385dee3fe1fb7920d` and
`46d8437310732336605443f38aec9a68fb4474235af36413d6cc18e5f22bf5c1`.
These are burst, after-PCM development results. Reported peak VRAM is null;
neither RTF nor finalization timing is capture or conversational latency.

## Consequences

- Candidate STT can be compared on paired, disjoint far-field PCM without a
  multi-gigabyte corpus acquisition or a new application entry point.
- The materializer remains offline, explicit-license, no-overwrite, bounded,
  and testable with synthetic source injection; real source bytes stay outside
  Git.
- Matrix v5 receipts and the canonical 96-case qualification remain valid.
- Actual overlapping speech needs a later separately locked diagnostic with an
  overlap-valid metric such as tcORC/tcpWER; it must not be mixed into this hard
  WER corpus.
- A green corpus preparation or model replay is development evidence only. Two
  guided owner captures, paired attestation, bare-speaker barge-in, and natural
  live conversation remain required before any stable-default decision.

This ADR refines ADR-0098, ADR-0109, ADR-0113, ADR-0117, and ADR-0151.
