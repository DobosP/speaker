Valid until: main advances beyond this task's landing commit — then treat as
history.

# Task result — configured final-command FileReplay gate

## Outcome

Added an aggregate-only evaluator for the configured final-STT selection code.
It drives the exact public 57-case command/noise corpus through
`FileReplayEngine` in ASR-only mode, exercising streaming finals, the configured
offline recognizer, Faster-Whisper verification, and FileReplay final selection
without constructing the chatbot, TTS, tools, publisher, capture loop, or an
audio device. Its selected view explicitly lacks VAD-owned speech duration and
the live capture path's offline-recovery authorization.

The evaluator binds the exact corpus and receipt, resolved device profile,
effective configuration, model artifacts, evaluator source closure, Python
executable, installed distribution records, and CPU/CUDA provider evidence. It
revalidates mutable inputs and corpus privacy after execution, suppresses
Python/native transcript and model output, reduces each complete case
immediately, and publishes a new private mode-0600 report with failure-clean
atomic no-clobber semantics. Terminal decision boundaries remain intact for
command matching even when WER joins multiple finals.

ADR-0118 rejects both the evaluated opt-in configured selection chain and
Parakeet Realtime EOU 120M as stable noisy-command replacements. Runtime
defaults, setup, `core`, `./live.sh`, tools, enrollment, and live behavior did
not change.

## Exact configured FileReplay evidence

The authoritative seven-stratum schema-v2 report is SHA-256
`4e36cb1564934e78167f12801e213dd3ecf348844ea00d2e0bd25df2c335610a`
(23,913 bytes). Its final evaluator-source SHA-256 is
`1222c8d5cd274fef31daaa8ccf8eaa5abf9bce536931fadd5d9522e55537d2ea`.
The exact configured chain was four-thread CPU GigaSpeech
Zipformer, CPU int8 Parakeet Unified English offline final, and
Faster-Whisper Small CUDA float16 verification under the explicit
`desktop_gpu_4090` profile.

All 57 evaluations completed with 58 terminal decisions, zero zero-decision
cases, and one multi-decision case. Offline outcomes were 51 decoded, six
empty, and one skipped. Verifier outcomes were 28 consensus, five empty, four
empty-streaming guards, one no-quorum, one skipped, and 19 ties. Final sources
were 32 streaming, ten verifier consensus, and 16 none.

The same-run raw streaming view recorded:

- 17/26 overall command hits, 0.8095 target precision, and 1/31 monitored
  negative-case false activations.
- 16/20 clean Speech Commands hits, 0.30 WER, and 0/10 negative false
  activations.
- 1/6 noisy ECCC hits, 1.3333 WER, and 1/21 false activations.

The `file_replay_selected_without_owned_timing_recovery` view recorded:

- 19/26 overall command hits, 0.7308 target precision, and 5/31 monitored
  negative-case false activations.
- 18/20 clean Speech Commands hits, 0.10 WER, and 0/10 negative false
  activations.
- 1/6 noisy ECCC hits, 1.3333 WER, and 5/21 false activations.

Under FileReplay's inputs, selection recovered two clean commands but did not
improve noisy recall and added four noisy false activations. That is not safe
enough for an action-capable voice path. The four `empty_streaming_guard`
outcomes could differ under live recovery authorization, so this is not a live
selection result.

## Parakeet Realtime EOU comparator

The authoritative tagged report is SHA-256
`e973b085902cd887b51a6246756758ab382f30dc016fdf4b7231e15b798d8175`
(30,829 bytes). The receipt-bound NeMo reference ran one accelerated burst
repeat with native 80 ms chunks and up to 48,000 synthetic tail samples.

- Overall: 21/26 command hits, 0.7241 target precision, 0.5745 WER, and 8/31
  monitored negative false activations.
- Clean Speech Commands: 19/20 hits, 0.05 WER, and 0/10 negative false
  activations.
- Noisy ECCC: 2/6 hits, 0.9630 WER, and 8/21 false activations.
- Endpoint diagnostics: 36 native endpoints, 26 before declared source end,
  ten response-authoritative endpoints, and 21 tail exhaustions.
- Resource diagnostics: 1,900.398 MiB sampled RSS, five threads, 924 MiB
  reported VRAM, 21.935 s model load, 0.734 source-audio RTF, and 0.3574
  model-input RTF.

The recall improvement does not justify the false-action rate, and this NeMo
path remains too heavy to establish the project's portable architecture.

## Verification and evidence handling

The hardened evaluator/metric and adjacent FileReplay/recorded-STT gate passed
146 tests with two explicit optional skips. The public command/noise gate passed
276 with its timestamp-sensitive race test deselected, that race passed 1/1
separately, the required APM/DTD gate passed 6/6, the host-side logger/console
checks passed 2/2, and the restricted-sandbox-sensitive LiveKit seam passed
73/73 host-side.
The definitive host-side non-real split passed 7,615 broadly plus the two exact
lock/race checks, for 7,617 effective passing checks; 14 skipped, 24 real-model
tests were deselected, and 11 known warnings remained. Strict evidence JSON,
report/source hashes, file modes, STATUS line count, and `git diff --check`
also passed. The final code/privacy audit was resolved before the authoritative
GPU rerun; no live hardware validation was claimed.

Committed evidence is
`docs/evidence/2026-08-02-production-final-command-gate.json`. It is aggregate
and path-free: no audio, case identifier, selected member/speaker, local path,
reference/hypothesis row, model transcript, or local configuration bytes are
included. Private reports stay outside Git.

## Limits and next work

No live validation ran. The evidence begins after PCM and does not validate
microphone capture, VAD or endpoint accuracy, native-reader or mailbox
behavior, AEC, open-speaker barge-in, enrollment, speaker identity/authority,
multi-voice behavior, chatbot/tool/TTS execution, physical-device latency,
natural conversation, or training disjointness. The 600 ms FileReplay tail is
transport context and is not ADR-0117's isolated one-second Zipformer flush or
live endpoint evidence. Native model start, warm, decode, and stop remain
unbounded in-process; use an external supervisor until a subprocess watchdog is
implemented.

Collect fresh owner target and negative recordings, then run a separate
`./live.sh` bare-speaker A/B. Keep the next multi-device experiment on the
receipt-bound `parakeet.cpp` route, and require precision or confirmation before
command-like text can authorize an external action.
