# ADR-0147: Add an EdAcc endpoint-integrity diagnostic

Date: 2026-08-06
Status: accepted

## Decision

Add a separate, private, aggregate-only diagnostic that compares native
acoustic endpointing with the exact Smart Turn v3.2 CPU detector on the retained
24-row EdAcc corpus. Drive both cells through the synchronous production Sherpa
capture loop so the real VAD clock, streaming recognizer, endpoint policy,
final selector, acoustic lineage, and callback boundary execute. Do not weaken
the production whole-turn evaluator's one-row/one-agent-turn invariant and do
not add another voice-agent entry point.

Bind the exact schema-v2 corpus digest, 24-case/4,790,400-byte shape and 8/8/8
`clean`/`disfluent`/`overlap_adjacent` marginal, the explicit desktop and atomic
SenseVoice production configuration, the clean source revision, active audio
artifacts, runtime identities, and the pinned 8,679,182-byte Smart Turn model.
Include `core/endpointing.py` in capture-replay source identity and content-hash
an active prosody model. Recheck every binding after execution.

Adapt each source row only in memory to 1,600-sample capture frames. Append the
same 2.4-second synthetic-zero tail to both cells and disable startup input
calibration because this corpus contains neither separately labelled room tone
nor a calibration prefix. Do not invent word timings, endpoint labels, or
post-roll truth. Reset endpoint-policy state for every independent row while
reusing loaded model objects sequentially within a cell.

Fix the control to semantic endpointing disabled with the production 0.8-second
native rule 2. Fix the candidate to the exact prosody detector, the same rule 2,
0.6/0.3 completion thresholds, a 1.6-second maximum wait, and no adaptive floor.
Force early semantic commits off in the evaluator so the candidate can only
accept a native endpoint or hold it; this isolates split repair and its added
wait. Fail closed on lexical fallback, detector error, malformed model output,
configuration drift, incomplete coverage, or private report content.

Publish only aggregate final counts, zero/single/multi-final row counts, joined
selected-final WER/CER and exact counts, paired row transitions, typed endpoint
state/basis counts, semantic-HOLD counts, closed typed transcript-abort counts,
and VAD-derived timing/resource summaries. Run native work in a killable child
process with bounded receipt output and a watchdog. Preflight an absolute,
owner-private report directory outside Git, then atomically publish one
mode-0600, no-overwrite JSON report. The parent must reopen the retained bytes,
validate their canonical aggregate schema and digest, and bind the safe receipt
counters back to that report. Exclude paths, case identifiers, references,
hypotheses, PCM, per-row values, model exceptions, and assistant/tool behavior.
Fix the verdict to diagnostic-only and explicitly deny
promotion, production-runtime, microphone, device, LLM, TTS, tool, and live
latency authority.

## Context / why

The strict production whole-turn replay exposed multi-final EdAcc rows, but its
`FileReplayEngine` commits and resets directly on `recognizer.is_endpoint()`.
It never constructs or calls the VAD-backed semantic endpoint policy, so neither
Smart Turn nor the configured lexical detector could repair that failure in the
measurement path. Raising the fixed native silence threshold removed every
split only at 2.4 seconds, an unacceptable basis for a live default because it
trades fragmentation for conversational delay.

The production capture loop is the existing causal seam that actually evaluates
semantic completion. A HOLD currently leaves the Sherpa stream unreset. The
installed Sherpa-ONNX contract says to reset after `is_endpoint()` becomes true;
therefore this diagnostic must determine whether HOLD really preserves a turn
on the installed recognizer rather than assuming that native decoding continues.

## Consequences

Headless tests can prove exact input/profile/model admission, causal HOLD-only
configuration, fresh per-row state, typed detector execution, aggregate lexical
and timing reduction, binding rechecks, privacy, and no-clobber publication
without opening an audio device or invoking the agent/tool plane.

The exact run result is recorded only after a clean-revision execution. A result
that scores HOLD without reducing multi-final rows is a native-stream
compatibility failure, not evidence to tune the probability threshold or enable
Smart Turn. The next permissible implementation experiment is an opt-in
reset-and-accumulate strategy that resets the native stream at a held boundary,
retains one bounded acoustic turn, and performs final selection once. It must
remain disabled until headless evidence is green, and any default change still
requires owner-labelled physical A/B through `./live.sh`.
