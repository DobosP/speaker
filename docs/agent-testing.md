# Agent Testing Guide — speaker

## Environment
- Runtime: Python audio stack.
- Preferred interpreter on this box: `/home/dobo/work/speaker/.venv/bin/python`.
- Live hardware validation is separate from headless pytest verification.

## Commands
| Scope | Command | Expected success |
|---|---|---|
| Live-launcher lifecycle (headless) | `/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_live_launcher.py tests/test_capture_integration.py tests/test_setup_doctor.py -q` | setup/reuse/failure/cleanup/private-path/doctor contracts pass without opening audio |
| APM/DTD regression | `/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q` | `6 passed` on current setup |
| Conversation trace (device-free) | `/home/dobo/work/speaker/.venv/bin/python -m tools.conversation_eval --runs 3` | all 14 v4 scenarios pass 3/3; aggregate 42/42, `pass^3=True`, exact declared answer routes, and controller-owned bounded correction (ADR-0051/0067/0068) |
| MiniCPM Q8 production-hybrid A/B | `/home/dobo/work/speaker/.venv/bin/python -m tools.conversation_eval --mode ollama --candidate-model minicpm5-1b:q8 --baseline-model gemma3:12b --runs 3` | clean revision; candidate and baseline each 42/42; verified identities, warm budgets, exact answer routes, and no regression (ADR-0051/0067/0068) |
| MiniCPM all-role stress | add `--topology all-roles` to the prior command | ADR-0051 replacement-stress diagnostic; never an adoption result |
| Cross-session semantic memory | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest memory` | SQLite reopen; clean native history; balanced fenced injection; PRIVATE first/only `route=main`; clause-grounded canary; clean/stable revision, contract, and full shipped-role identities (Echo is incomplete) (ADR-0065) |
| Autonomous cable diagnostic | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest voice --acoustics cable` | pipeline checks green but report says `diagnostic_pass`, `ok=false`, and barge/self/latency `not_covered`; not a landing result (ADR-0055/0058) |
| Autonomous silent voice gate | `/home/dobo/work/speaker/.venv/bin/python -m tools.autotest all --acoustics delay` | exact run-owned virtual EC topology and capture/playback bindings; labelled causal finals/terminals; mean WER ≤0.50; zero errors/stuck/self-cuts; synthesized exact-command cut from engine capture onset in 0–1.4 s; all six route/exit/cleanup proofs; aggregate PASS (ADR-0058/0069/0070) |
| Injected Sherpa barge replay | `/home/dobo/work/speaker/.venv/bin/python -m tools.live_session --scenario barge_in_interrupt_stop --repeat 3 --inject --barge-in --llm echo --no-assistant-audio` | exit 0 only when every repetition is full-duplex `ok`, intended FIFO cuts 2/2, zero self-interrupts (ADR-0064) |
| Recorded owner-voice landing gate | `SPEAKER_REQUIRE_RECORDED=1 /home/dobo/work/speaker/.venv/bin/python -m pytest tests/replay_recorded_voice_test.py -q` | reference host: exactly 9 passed/0 skipped (six utterances, one same-session multi-turn, two causal fake-stream owner talk-overs); missing private clips/models fail (ADR-0053) |
| Recording-driven STT accuracy | `/home/dobo/work/speaker/.venv/bin/python -m tools.recorded_stt_eval` | aggregate-only streaming/offline/selected WER+CER over every hash-pinned labelled clip; each selected offline recognizer/verifier must complete a decode with zero error outcomes; no runtime, TTS, tools, network, or audio device (ADR-0078/0080) |
| Opt-in selected final STT setup | `./install.sh --skip-system --final-asr parakeet-unified-en --final-verifier faster-whisper-small` | verify/stage the four-file Parakeet package and pinned Linux/NVIDIA verifier, publish atomically, and pass doctor; normal sessions still use `./live.sh` (ADR-0080) |
| Verifier recording A/B | `/home/dobo/work/speaker/.venv/bin/python -m tools.recorded_stt_eval --set asr_final_verifier_backend=faster_whisper --set asr_final_verifier_model=/home/dobo/work/speaker/pretrained_models/sherpa/faster_whisper_small-536b0662742c --keyword vault` | aggregate-only baseline/candidate comparison, verifier outcome counts, and artifact-bound provenance; no transcript rows or audio device (ADR-0078/0080) |
| Isolated enrollment prep | `/home/dobo/work/speaker/.venv/bin/python -m tools.prepare_enrollment --help` then supply the four explicit absolute paths and a unique `enrollment.v5-<id>.json` | device-free; verified no-clobber backup, empty feature candidate, regular mode-600 config with prepared marker; use its exact printed next command (ADR-0056) |
| Accepted v5 promotion | `/home/dobo/work/speaker/.venv/bin/python -m tools.promote_enrollment --help` then supply the exact worktree, primary config, prepared candidate/source/backup, candidate-derived adjacent accepted path, and `--accept-live-gate` | device-free and only after manual acceptance; exit 0 = active, 2 = refused, 3 = confirmed staged/inactive, 4 = ambiguous (ADR-0066) |
| Whitespace | `git diff --check` | no output |

## Before commit
1. Run `git diff --check`.
2. Run targeted pytest for changed audio/engine code.
3. Document live A/B validation still required when hardware behavior is affected.
4. Do not commit generated `logs/**` artifacts.

The conversation trace opens no audio device and cannot validate ASR, echo
cancellation, TTS sound, or bare-speaker barge-in. Keep that result separate
from the Sherpa duplex regression and manual live evidence.

The `./live.sh` tests replace PipeWire, Ollama, doctor, and normal core execution
with fakes. Minimal signal-mask cases spawn an inert real subprocess but open no
audio device. They validate resource ownership and evidence setup only; a real
run is still required for current-room audio and barge-in.

The injected Sherpa replay also opens no physical audio device. Per ADR-0064,
its clean echo-impossible profile removes echo/level/word-confirm discrimination
and denoising, uses one eligible 100 ms VAD block, and validates capture
continuity, real ASR/VAD/TTS workers, and interrupt control flow. `--denoise`
is a separate stress diagnostic, not the landing command. Production retains
its physical front end and two-block policy; this does not validate physical
echo, owner enrollment, acoustic stop latency, or the live PipeWire word-cut
route. Any machine-owned injected grade failure or scenario-coverage shortfall
makes the command nonzero.
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

Per ADR-0051/0067/0068, production-warm real-model runs prewarm each distinct model with
the runtime system prompt. `--warm-policy cold` is a labelled red diagnostic.
Generated reports stay local under ignored `logs/conversation-eval/`; an
unverified MiniCPM override still exits 2 and can never make the gate green.
Changing `--runs` or selecting `--scenario` produces a coverage-red diagnostic;
the adoption gate is exactly all fourteen v4 scenarios repeated three times.
A real-model report is provenance-red when the revision/config changes, local
config is included, or any effective model role lacks stable identity evidence.

Per ADR-0065, the autonomous memory probe reopens SQLite and creates a fresh
capability registry in one interpreter; it does not claim a fresh OS process.
A green real-model result requires
`recall_available=true`, `recall_injected=true`, `recall_fenced=true`,
`recent_history_clean=true`, PRIVATE sensitivity, first/only `route=main`,
`controller=false`, and one affirmative clause binding the canary value to its
subject. It also requires one clean stable revision, a stable effective probe
contract, an ambient-credential-isolated loopback transport, and stable full
blob plus effective-config identities for the configured MiniCPM/Gemma roles.
The persisted digest binds the evidence; it does not reconstruct undisclosed
inputs. Echo performs no Git/config/model inspection and reports incomplete. Voice and barge
harnesses retain distinct main/fast arguments; all-role MiniCPM is diagnostic.

Per ADR-0058, a `speaking:` marker or quiet log interval is not sink evidence.
Voice/stress reports match each selected non-barge labelled prompt to a new final
and require remembered same-input-generation playback onset plus the scenario's
aggregate terminal outcome. Auxiliary acknowledgements cannot satisfy a reply;
talk-over requires same-task/generation `interrupted` after its barge marker.
They count only finite nonnegative first-audio values from the finalized bundle
and fail on missing labels, failed injection, late cuts, runtime errors, or stuck
hints. The asynchronous injected-onset clock is causal for the harness but is
not a physical human-onset measurement.

For delay, require PASS on topology, capture, duplex, digest correlation, child
exit, and cleanup. Preserve any retained graph/files and the log on failure; do
not load a host EC module or change desktop defaults to make the retry green.
The synthetic delay command uses deterministic VITS `quiet`, a calibrated
three-block admission fallback, and detector-only quiet padding. Grade its exact
capture-onset clock at ≤1.4 s and show source-onset latency only as a diagnostic.
Recorded-owner, generic, stress, and physical `speaker` paths retain ≤1.0 s.
Require two fresh delay passes before calling the gate stable. See ADR-0069/0070;
keep recorded-owner and physical acceptance separate.

Per ADR-0056, run preparation before any v5 capture. Its final config publish is
already wired to the reserved candidate and its printed command includes
`--require-prepared-enrollment`; a wrong-checkout launch then refuses before the
microphone. Marker-free enrollment refuses a non-empty reference unless the
operator explicitly supplies `--replace-enrollment`.

Per ADR-0066, preparation schema v2 binds full metadata and SHA-256 lineage for
the primary config, reservation, backup, and historical source. Promotion is a
separate no-audio operation after the complete manual live gate passes. The
accepted basename is the prepared candidate basename plus `-accepted`; it stays
adjacent to historical v4 but never replaces it. Exit 3 proves an exact private
orphan plus the unchanged inactive primary pointer, so the identical command may
adopt it. Exit 4 is ambiguous and requires inspection before retry. The stable
advisory lock serializes cooperating promoters only. Never interpret preparation,
staging, or an exit-3 result as live acceptance.

## Known blockers
- If `.venv` is missing, recreate/use a project venv and record the exact command.
- Do not bypass a red delay-route contract by loading an unrelated host EC route
  or changing system audio defaults; preserve the failed evidence for diagnosis.
- Do not claim live audio validation unless it was actually performed.
