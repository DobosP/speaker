# ADR-0257: Integrate development voice evaluators with legacy endpoint scope

Date: 2026-10-29
Status: accepted for headless development tooling; native measurements and live promotion remain separate

## Decision

Integrate the historical recorded-response evaluator and pinned English scoring
helper from `codex/recording-response-latency` at `befa8ea`, with their tests,
MIT vendor/provenance files and original ADR-0246/0247/0249/0250 records. Those
records retain their dated source and measurement limits. Their old STATUS and
WORKLOG snapshots are not replayed onto the current runtime authority.

The new scorer keeps the original legacy English metric alongside a separately
named normalization view. Normalization is evaluation-only: it does not change
recordings, references, model input, decoding, runtime text or action authority.
Both views retain their own denominators and formatting-sensitive limitations.
The pinned port introduces no production package installation or model download.

The response evaluator measures complete recorded PCM through selected ASR,
production local text/runtime assembly, and first nonzero post-DSP answer PCM at
a null sink. It opens no microphone or playback device. Such compute timing is
not live speech-end, endpoint, DAC or acoustic latency. Original caller-scripted
and microphone corpora stay private, separate, non-disjoint development evidence.
No new native measurement or model-quality improvement follows from this port.

## Current endpoint compatibility

The resident-model evaluator reuses recognizer objects across fresh rows.
Current FileReplay stop retires leased endpoint adapters, so reusing the new
Qwen CPU primary or SenseVoice/Parakeet CPU verifiers would reuse retired owners.
The evaluator therefore accepts only its established final backends (unset,
SenseVoice, Whisper, NeMo transducer) and verifier (unset or Faster-Whisper).
Other values fail with the existing fixed `invalid_inputs` code before asset
binding or model startup. The same guard covers direct pipeline/engine callers.
Dormant optional profiles do not fail a legacy selected configuration.

The explicit source seam now includes `_endpoint_lease.py`. Supporting newer
leased endpoint profiles requires a separate per-row lifecycle qualification;
this port does not bypass production cleanup or replace current Qwen tools.
Existing runtime/defaults, profiles, provider settings, dependencies, authority
and active voice work stay byte-identical. All runs remain explicit local-only
operations with the original safety/refusal, worker timeout and privacy rules.

## Validation and retention

The public/fake scorer, response-evaluator and existing ASR benchmark gate passes
199 tests. Ten new compatibility/source-binding cases fail against the old tool
and pass with the bounded adaptation. Existing tests cover normalizer provenance,
legacy score preservation, receipt privacy, worker ownership and fake production
runtime/TTS paths. No native models, recordings, provider calls or hardware were
used for this integration gate. Current exact-head review and any wider gate
results are recorded in WORKLOG; historical counts are not new qualification.

Recordings, original baselines, historical source refs and unique evidence stay
protected. Any scratch cleanup is an exact, separately reviewed NAS-first action;
landing this tool does not make an entire research folder disposable.
