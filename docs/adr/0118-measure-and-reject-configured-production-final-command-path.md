# ADR-0118: Measure and reject the configured production final-command path

Date: 2026-08-02
Status: accepted

## Decision

Add an aggregate-only schema-v2 evaluator for Speaker's configured final-STT
selection path. Run the exact 57-case public command/noise corpus through
`FileReplayEngine` in ASR-only mode, using the resolved device profile and the
same streaming, offline-final, verifier, and selection code as a local session.
Name the selected view
`file_replay_selected_without_owned_timing_recovery`: FileReplay supplies
neither VAD-owned speech duration nor the live capture path's offline-recovery
authorization, so this view is not complete live-selection evidence.
Bind the exact corpus/receipt, effective configuration, model artifacts,
evaluator sources, Python executable, installed distribution records, and
requested providers before execution and revalidate mutable inputs after it.
Reduce each complete case immediately, preserve terminal-decision boundaries
for command matching, suppress transcript/model output, and publish only a new
mode-0600 aggregate report through a failure-clean atomic no-clobber writer
outside Git.

Keep runtime defaults, setup, `core`, and `./live.sh` unchanged. Reject both the
evaluated opt-in configured selection chain and Parakeet Realtime EOU 120M as
stable noisy-command replacements. Treat every result here as after-PCM
development evidence, not promotion or live validation.

## Context / why

ADR-0117 compared isolated recognizers, but it did not execute the code that
selects a production final from streaming Zipformer, offline Parakeet Unified,
and Faster-Whisper verification. A better component score does not establish
that their selection policy improves command safety. The new evaluator ran the
current explicit `desktop_gpu_4090` configuration: four-thread CPU Zipformer,
CPU int8 Parakeet Unified English offline final, and Faster-Whisper Small CUDA
float16 verification. All 57 cases completed, yielding 58 terminal decisions,
zero zero-decision cases, and one multi-decision case. Offline outcomes were 51
decoded, six empty, and one skipped; verifier outcomes were 28 consensus, five
empty, four empty-streaming guards, one no-quorum, one skipped, and 19 ties.

The same-run raw streaming view reached 17/26 command recall, 1/31 monitored
negative-case false positives, and 0.8095 target precision. It reached 16/20
commands with 0.30 WER on clean Speech Commands and 1/6 commands with 1.3333
WER and 1/21 false activations on noisy ECCC. The FileReplay-selected view
improved clean recall to 18/20 and clean WER to 0.10, but overall recall was
only 19/26 and target precision fell to 0.7308. It did not improve noisy recall
(1/6) and raised ECCC false activations to 5/21, hence 5/31 overall. Under these
FileReplay inputs, adding verifier consensus therefore recovered two clean
commands while adding four noisy false activations; that is the wrong tradeoff
for an action-capable voice agent and still says nothing about the four
`empty_streaming_guard` cases under live recovery authorization.

The exact Parakeet Realtime EOU 120M reference from ADR-0099 was also run once
in accelerated burst mode on the same tagged corpus. It reached 21/26 overall
command recall and 0.7241 target precision: 19/20 with 0.05 WER on clean Speech
Commands and 2/6 with 0.9630 WER on ECCC. It also produced 8/21 ECCC false
activations, or 8/31 monitored negatives overall. Of 36 native endpoints, 26
preceded declared source end, only ten were response-authoritative, and 21
cases exhausted the synthetic tail. Its 1,900.398 MiB sampled RSS, 924 MiB
reported VRAM, and 21.935 s load further prevent treating this NeMo reference
as the portable, low-resource architecture sought by the project. Higher
recall does not offset its false-action risk or its invalidity as live
turn-taking evidence.

## Consequences

The FileReplay evaluator becomes the exact regression route for future changes
to FileReplay final selection. Its two views are `file_replay_streaming_raw`
and `file_replay_selected_without_owned_timing_recovery`; they share one run,
corpus, transport, and decision stream. The raw view uses 100 ms replay blocks
and a 600 ms synthetic transport tail, so it is not the one-second isolated
Zipformer context diagnostic from ADR-0117 and cannot be used as equivalent
endpoint or latency evidence.

The retained private reports are bound by SHA-256 in the committed path-free
aggregate evidence. They contain no raw audio, case identifiers, local paths,
reference or hypothesis rows, or full local-configuration bytes. The evaluator
does not build the chatbot, TTS, tools, session publisher, or audio device.

No live validation ran. This work begins after PCM is available and does not
validate microphone capture, VAD or endpoint accuracy, the native reader,
capture-mailbox behavior, AEC, open-speaker barge-in, enrollment, speaker
identity or authority, multi-voice behavior, chatbot/tool/TTS execution,
physical-device latency, natural conversation, or training disjointness. Fresh
owner target and negative recordings plus a separate `./live.sh` bare-speaker
A/B remain required. Native start, warm, decode, and stop calls also have no
in-process watchdog; run this diagnostic under an external supervisor until a
separate subprocess boundary is implemented. The next portable model
experiment should follow the receipt-bound `parakeet.cpp` route already
identified by ADR-0099. Command and control policy must prioritize precision
and explicit confirmation rather than blindly adding model votes.
