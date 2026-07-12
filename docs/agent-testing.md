# Agent Testing Guide — speaker

## Environment
- Runtime: Python audio stack.
- Preferred interpreter on this box: `/home/dobo/work/speaker/.venv/bin/python`.
- Live hardware validation is separate from headless pytest verification.

## Commands
| Scope | Command | Expected success |
|---|---|---|
| APM/DTD regression | `/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q` | `6 passed` on current setup |
| Conversation trace (device-free) | `/home/dobo/work/speaker/.venv/bin/python -m tools.conversation_eval --runs 3` | all 14 v2 scenarios pass 3/3; aggregate 42/42 and `pass^3=True` (ADR-0051) |
| MiniCPM Q8 production-hybrid A/B | `/home/dobo/work/speaker/.venv/bin/python -m tools.conversation_eval --mode ollama --candidate-model minicpm5-1b:q8 --baseline-model gemma3:12b --runs 3` | clean revision; candidate and baseline each 42/42; verified identities, warm budgets, and no regression (ADR-0051) |
| MiniCPM all-role stress | add `--topology all-roles` to the prior command | ADR-0051 replacement-stress diagnostic; never an adoption result |
| Exact self-fact memory | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest memory` | retrieval available; prompt injection false on the controller read; exact grounded scalar; `answer_model=control` (ADR-0054) |
| Injected Sherpa barge replay | `/home/dobo/work/speaker/.venv/bin/python -m tools.live_session --scenario barge_in_interrupt_stop --repeat 3 --inject --barge-in --llm echo --no-assistant-audio` | every repetition full-duplex `ok`, intended FIFO cuts 2/2, zero self-interrupts (ADR-0052) |
| Recorded owner-voice landing gate | `SPEAKER_REQUIRE_RECORDED=1 /home/dobo/work/speaker/.venv/bin/python -m pytest tests/replay_recorded_voice_test.py -q` | reference host: exactly 9 passed/0 skipped (six utterances, one same-session multi-turn, two causal fake-stream owner talk-overs); missing private clips/models fail (ADR-0053) |
| Whitespace | `git diff --check` | no output |

## Before commit
1. Run `git diff --check`.
2. Run targeted pytest for changed audio/engine code.
3. Document live A/B validation still required when hardware behavior is affected.
4. Do not commit generated `logs/**` artifacts.

The conversation trace opens no audio device and cannot validate ASR, echo
cancellation, TTS sound, or bare-speaker barge-in. Keep that result separate
from the Sherpa duplex regression and manual live evidence.

The injected Sherpa replay also opens no physical audio device. Per ADR-0052,
as superseded by ADR-0053, its echo-free profile uses one eligible 100 ms block
and validates capture continuity, real ASR/VAD/TTS workers, and interrupt control
flow. Production retains two blocks; this does not validate physical echo,
owner enrollment, acoustic stop latency, or the live PipeWire word-cut route.
Inject timeline user times are enqueue metadata, not consumption/overlap evidence;
assistant latency and answers bind through explicit `response_to_user_idx` links.

The recorded owner talk-over replaces input/output streams and device queries,
capability checks, uses the shared inject profile, and binds actual owner-sample
consumption to one metrics token after onset grace plus a floor-only no-cut
control with a complete post-pacing floor-sample window. Its exact corpus/pairs
and one-final-per-clip contract prevent data-driven skips or trailing turns from
shrinking coverage. Its sustained clips override the synthetic one-block minimum
and run production's two-block temporal policy. Owner-to-FIFO stop must stay
within the verifier-owned 1.0 s ceiling. Without
`SPEAKER_REQUIRE_RECORDED=1`, clean clones
self-skip missing private clips/models; that diagnostic command is not a landing
gate. It covers historical waveforms, not the current room, speaker output, v5
enrollment, or live word-cut.

Per ADR-0051, production-warm real-model runs prewarm each distinct model with
the runtime system prompt. `--warm-policy cold` is a labelled red diagnostic.
Generated reports stay local under ignored `logs/conversation-eval/`; an
unverified MiniCPM override still exits 2 and can never make the gate green.
Changing `--runs` or selecting `--scenario` produces a coverage-red diagnostic;
the adoption gate is exactly all fourteen v2 scenarios repeated three times.
A real-model report is provenance-red when the revision/config changes, local
config is included, or any effective model role lacks stable identity evidence.

Per ADR-0054, the autonomous memory probe distinguishes a fact being available
from it being injected into a model prompt. Its exact live self-scalar read is a
PRIVATE controller result (`recall_available=true`, `recall_injected=false`),
not evidence that general or cross-session semantic recall is solved. Voice and
barge harnesses must retain distinct main/fast model arguments; an all-role
MiniCPM run is diagnostic only.

## Known blockers
- If `.venv` is missing, recreate/use a project venv and record the exact command.
- Do not claim live audio validation unless it was actually performed.
