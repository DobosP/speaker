Valid until: main advances beyond this task's landing commit — then treat as
history.

# Task result — headless conversation flow acceptance

## Outcome

The unchanged 14-scenario conversation-eval v4 contract now has an opt-in
flow-v1 extension through the existing diagnostic entry point:
`python -m tools.conversation_eval --flow-acceptance`. The combined gate
requires all 42 v4 scenario-runs plus exactly three repetitions of four
production-control-plane journeys.

Flow-v1 proves typed partial/final revision fencing, pure endpoint-policy
decisions, incremental receipt-owned scripted playback, scripted interruption
recovery, delayed read-tool ordering and follow-up, confirmation-gated
synthetic action ownership, bounded `VoiceSession` stop, fresh-session
isolation, and late-provider fencing. The action provider is in-memory and
performs no external mutation.

Reports fail closed on incomplete, duplicate, relabelled, contradictory,
nonfinite, reordered, or unbound evidence. Nonempty task-runtime
`tool_call_id` and `idempotency_key` values survive in raw diagnostic traces,
and start/finish identities are checked exactly. The default v4 CLI path stays
flow-free; the application still has one entry point, `python -m core --session`.

## Main files

- Added `tools/conversation_eval/flow.py` and `flow_schema.py`.
- Added focused runtime, schema-poison, and CLI/backward-compatibility tests.
- Extended `TraceRecorder` with nonempty opaque tool lifecycle identities.
- Added ADR-0112 and updated `STATUS.md`.

## Verification

- Exact combined CLI: base v4 42/42 and flow-v1 12/12, both `pass^3=True`,
  exit 0; report written outside Git at
  `/tmp/speaker-conversation-flow-v1-report.json`.
- Focused evaluator/flow suite: 118 passed; mandatory APM/DTD: 6 passed.
- Full non-real split: 7,300 passed (7,269 low-priority broad plus 31
  isolated logging/path/thread-sensitive checks), 14 skipped, 23 model-only
  deselected, and 9 pre-existing warnings.
- The 31 isolated checks comprise one log-capture case with debug logging, one
  writable-runlog CLI case, one private-mode wheel-lock case, and 28 fake
  LiveKit event-loop cases. LiveKit passed outside the sandbox because this
  sandbox's asyncio self-pipe did not wake; the production code was unchanged.
- Independent final review found no fail-open, evidence-overclaim, lifecycle,
  or backward-compatibility blocker. `git diff --check` is clean.

## Limits / next

Fixture transcripts are plumbing-only. No microphone/native reader, live VAD
wiring, recognizer decode, WER/CER, real TTS/PCM pacing, AEC/room/open-speaker
behavior, physical audibility, external device effect, GPU model, or
live/perceived latency was tested.

Next, implement a reproducible, hash-bound 96-case local fixture recipe from
HarperValleyBank, EdAcc, AMI, and Common Voice Spontaneous Speech v4. Keep audio
outside Git, label overlap as diagnostic-only, and do not claim the public data
is unseen or training-disjoint. Then run isolated GPU STT/configuration cells
without changing defaults; live promotion still requires Paul's recordings.
