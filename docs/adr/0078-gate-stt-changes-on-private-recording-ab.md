# ADR-0078: Gate STT changes on private recording A/B

Date: 2026-07-16
Status: accepted

## Decision

Evaluate a recognizer, streaming-ASR endpoint, language, biasing, or final-
selection candidate against the same labelled local ASR waveforms before
changing its shipped default. Run the streaming hypothesis, raw offline
hypothesis, and production-selected final through one ASR-only FileReplay seam.
Frontend, VAD/admission, and live endpoint-controller candidates first require
a separate deterministic reconstruction from aligned recordings; only their
resulting ASR PCM/segments may be compared here. Do not construct the chatbot
runtime, TTS, capabilities, network clients, or audio devices. A candidate is
promotable only when every labelled clip is
covered with a non-empty selected final, aggregate word and character errors do
not regress, target-keyword recall does not regress, lexicographic per-clip
word/character losses do not outnumber wins, and at least one aggregate accuracy
measure strictly improves.

Keep references and hypotheses in memory only and reduce each decoded clip to
aggregate counters before processing the next one. Routine stdout and optional
reports contain no transcript, clip id, or path; reports are atomically written
mode 600 and bind the corpus, effective STT config, and model artifacts by
digest. Keep the existing strict recorded runtime/barge gate
independent: it proves turn and causal interruption behavior, while the new
evaluator grades recognizer accuracy. This extends ADR-0077's evidence-first
rule and does not supersede its need for aligned pre-/post-DSP live evidence.

Do not promote a new STT default from the 2026-07-16 evidence. The labelled
owner corpus and the failed domain run rule out several guesses, but the failed
run has no exact reference transcript or pre-application-DSP track and therefore
cannot localize the remaining error.

## Context / why

The failed vault-phrase session contained six admitted, unclipped windows with
no capture, decode, finalizer, or playback-separation failure, yet the target
word was recognized in none of them. Replays found no recovery from level
normalization, English/auto language, ITN, beam width, or an alternate Whisper
model. There was also no repeated, phonetically plausible
confusion that could support a narrow deterministic repair; a broad rewrite
would already corrupt a valid word in the small negative corpus.

The existing six-clip labelled owner corpus remained healthy, but the prior
landing test used a lenient per-clip overlap threshold and exposed only the
production-selected final. The new aggregate baseline measured 25 reference
words: streaming WER 0.20, raw offline WER 0.04, and selected-final WER 0.12.
Increasing active beam paths from four to eight changed streaming CER but left
all six selected finals tied, so the candidate was correctly rejected. The
strict recorded runtime/barge gate independently remained 9/9 green.

Changing gain, denoise, language, the agreement guard, or the recognizer from a
single failed live transcript was rejected because each can increase false
finals or regress ordinary speech without fixing the unidentified capture or
domain cause. Printing per-clip hypotheses was also rejected because transcripts
are private data and aggregate statistics are sufficient for candidate selection.

## Consequences

- `python -m tools.recorded_stt_eval` produces a privacy-safe baseline from the
  hash-pinned local corpus; repeated `--set FIELD=VALUE` arguments run an exact
  baseline/candidate A/B and return nonzero when the candidate is not promotable.
- `--manifest` accepts another local labelled corpus and `--keyword` adds
  aggregate target-phrase attempts/hits. Unlabelled recordings remain useful
  diagnostics but cannot produce WER or authorize a tuning change.
- FileReplay diagnostic replay now shares production stream hotword wiring,
  resampling, endpointing, second-pass selection, and attested repair while
  suppressing runtime publication. Ordinary replay behavior is unchanged.
- Stream parity does not make plain-text hotwords effective for this BPE model;
  model-specific BPE configuration remains required and unpromoted.
- Full-runtime recorded replay disables machine-local vault, reminder, and app
  providers in its in-memory config while retaining local model paths, so an
  ASR regression cannot invoke unrelated device capabilities.
- The next physical run is still required to obtain aligned pre-/post-DSP target
  evidence. This decision and its headless tests do not claim improved live STT.
