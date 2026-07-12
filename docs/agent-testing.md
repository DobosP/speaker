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
| Cross-session semantic memory | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest memory` | SQLite reopen; clean native history; balanced fenced injection; `route=main`; controller false; distinct roles; grounded canary with real Ollama (Echo is incomplete diagnostic) (ADR-0060) |
| Autonomous cable diagnostic | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest voice --acoustics cable` | pipeline checks green but report says `diagnostic_pass`, `ok=false`, and barge/self/latency `not_covered`; not a landing result (ADR-0055/0058) |
| Autonomous silent voice gate | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest all --acoustics delay` | every selected non-barge labelled prompt has a matching final; remembered ordinary/self replies attest `completed`; talk-over attests same-task/generation `interrupted` after its barge marker; mean WER ≤0.50, zero errors/stuck/self-cuts, successful active cut in 0–1.0 s; aggregate PASS (ADR-0058) |
| Injected Sherpa barge replay | `/home/dobo/work/speaker/.venv/bin/python -m tools.live_session --scenario barge_in_interrupt_stop --repeat 3 --inject --barge-in --llm echo --no-assistant-audio` | every repetition full-duplex `ok`, intended FIFO cuts 2/2, zero self-interrupts (ADR-0052) |
| Recorded owner-voice landing gate | `SPEAKER_REQUIRE_RECORDED=1 /home/dobo/work/speaker/.venv/bin/python -m pytest tests/replay_recorded_voice_test.py -q` | reference host: exactly 9 passed/0 skipped (six utterances, one same-session multi-turn, two causal fake-stream owner talk-overs); missing private clips/models fail (ADR-0053) |
| Isolated enrollment prep | `/home/dobo/work/speaker/.venv/bin/python -m tools.prepare_enrollment --help` then supply the four explicit absolute paths and a unique `enrollment.v5-<id>.json` | device-free; verified no-clobber backup, empty feature candidate, regular mode-600 config with prepared marker; use its exact printed next command (ADR-0056) |
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
and run production's two-block temporal policy. The concurrent base setup uses a
400 ms floor lead and a 2.4 s acoustic tail so its full pinned window commits;
the separate per-clip replay owns production endpoint grading (ADR-0061).
Owner-to-FIFO stop must stay
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

Per ADR-0060, the autonomous memory probe uses a fresh SQLite backend and fresh
capability registry for the read. A green real-model result requires
`recall_available=true`, `recall_injected=true`, `recall_fenced=true`,
`recent_history_clean=true`, `route=main`, `controller=false`, and a grounded
canary. Echo diagnoses only the persistence/fence/routing plumbing and reports
incomplete; real-model evidence also requires distinct
shipped main/fast roles (`topology_valid=true`). Voice and barge harnesses retain distinct
main/fast model arguments; an all-role MiniCPM run is diagnostic only.

Per ADR-0058, a `speaking:` marker or quiet log interval is not sink evidence.
Voice/stress reports match each selected non-barge labelled prompt to a new final
and require remembered same-input-generation playback onset plus the scenario's
aggregate terminal outcome. Auxiliary acknowledgements cannot satisfy a reply;
talk-over requires same-task/generation `interrupted` after its barge marker.
They count only finite nonnegative first-audio values from the finalized bundle
and fail on missing labels, failed injection, late cuts, runtime errors, or stuck
hints. The asynchronous injected-onset clock is causal for the harness but is
not a physical human-onset measurement.

Per ADR-0056, run preparation before any v5 capture. Its final config publish is
already wired to the reserved candidate and its printed command includes
`--require-prepared-enrollment`; a wrong-checkout launch then refuses before the
microphone. Marker-free enrollment refuses a non-empty reference unless the
operator explicitly supplies `--replace-enrollment`.

## Known blockers
- If `.venv` is missing, recreate/use a project venv and record the exact command.
- Do not claim live audio validation unless it was actually performed.
