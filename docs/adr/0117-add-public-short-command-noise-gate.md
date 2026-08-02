# ADR-0117: Add the public short-command/noise gate

Date: 2026-08-02
Status: accepted

## Decision

Extend the public voice matrix to v5 with the Sheffield English Consistent
Confusion Corpus v1.2 and the corrected official Google Speech Commands v0.02
test archive. Redigest the existing 96-case public-conversation lock at matrix
v5 without changing its four source recipes or slots. Bind one deterministic,
private schema-v4 corpus of 57 isolated
clips: 20 Speech Commands positives, five unknown-speech negatives, five
background-noise silence cases, six ECCC command positives, and 21 ECCC targets
whose documented majority confusion is a command. Require typed transcript,
speech-negative, or silence assertions; exact positive and forbidden command
targets; and aggregate final/partial command recall, target precision, case and
target-pair false-positive rates, and negative-audio-hour counts.
Require every positive target to occur as an exact normalized token sequence in
its transcript reference and every forbidden target to be absent from it.

Keep the corpus as a development gate and reject both evaluated recognizers as
a stable noisy-command replacement. Preserve Faster-Whisper Small and the
production GigaSpeech Zipformer as opt-in comparators, leave runtime defaults,
setup, `core/`, and `./live.sh` unchanged, and require fresh owner recordings
plus live validation before any adoption decision. Label Zipformer's
16,000-sample zero tail as a short-clip flush/context diagnostic, never endpoint
or production-latency evidence.

## Context / why

The earlier public STT slice had no command attempts, while the owner's live
transcripts still showed short-utterance errors. Generic WER could not reveal
whether a model missed a requested command, hallucinated a command on silence,
or converted a different spoken word into a dangerous command. The new schema
separates those outcomes and keeps denominators explicit. ECCC is a corpus of
consistent noisy word confusions rather than a command or paired-clean corpus,
so it contributes literal noisy WER and documented confusion targets only; it
cannot support clean/noisy degradation or training-disjoint claims.

On the 57-case corpus, Faster-Whisper Small produced 90% command recall and 0.10
WER on the clean Speech Commands stratum, but only 1/6 command positives on
ECCC and 11/21 documented command confusions on ECCC negatives. Its aggregate
command recall was 19/26, target precision 0.6333, and monitored negative-case
false-positive rate 11/31. Zipformer's legacy zero-tail default emitted no final
on any of the 20 clean commands, so that result is retained only as evidence of
insufficient short-clip context. The one-second v2 maximum tail clears its
configured 0.8-second rule-2 silence and repaired finalization: clean recall
rose to 16/20 with 0.20 WER, aggregate recall to 17/26, and monitored command
false positives stayed 0/31. It still found only 1/6 noisy ECCC positives.

## Consequences

The no-download preparer verifies exact archive hashes, licenses, member/file
counts, expanded-byte ceilings, safe archive structure, audio formats, the
complete ECCC CSV, and deterministic selections before publishing mode-0600
PCM/manifest/receipt files beneath a mode-0700 directory outside Git. It never
extracts either archive, and its committed lock and aggregate evidence contain
no selected speaker, archive-member, per-case transcript row, or local-path
data. The lock necessarily names the ten public command-vocabulary words.

The gate now exposes the real tradeoff: Faster-Whisper is stronger on clean
commands but accepts too many known noisy confusions; Zipformer is more
conservative but misses too many requested commands. The tail-padded Zipformer
RTF includes one second of synthetic context while its duration denominator is
source audio, and the adapter finalizes after declared input without calling
the native endpoint API, so latency and endpoint comparisons are invalid. This
work adds no capture, VAD, AEC, barge-in, enrollment, multi-voice, identity,
agent-tool, device, live-conversation, training-disjoint, or adoption evidence.
