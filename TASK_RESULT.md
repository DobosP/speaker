# Task Result — enrollment-optional word-cut

Valid until: ADR-0072 or the optional-enrollment implementation changes — then
treat as history.

Branch: `fix/enrollment-optional-barge`

Status: green deterministic/headless gate; physical bare-speaker barge-in remains
red and unvalidated.

## Summary

- Single-voice word-cut now defaults to enrollment-independent lexical authority.
  Missing `barge_word_cut_require_speaker` keys resolve to false consistently in
  runtime and readiness paths.
- Identity-free generic cuts retain a non-lowerable four-novel-non-own-word floor.
  Novel exact canonical controls retain their bounded short exception; silence,
  generic own echo, empty text, and zero-to-three generic words cannot cut.
- ADR-0042 remains accepted and independent of the optional filter: a STOP-class
  control or attested repair that resembles current/recent own TTS still needs
  compatible enrolled-speaker authority and otherwise fails closed.
- `barge_word_cut_require_speaker=true` remains the explicit multi-voice filter
  for generic overrides and still requires compatible enrollment/readiness/warm
  authority. Novel exact Stop/cancel remains an open fail-safe control.
- Normal-final `speaker_gate_input`, typed owner verification, sensitive action
  gates, enrollment capture/provenance, preparation, and promotion are unchanged.
- ADR-0072 records the owner decision and preserves ADR-0071's failed physical
  evidence. ADR-0045 and ADR-0071 now point to ADR-0072 as their superseder.

## Files changed

- `core/engines/sherpa.py` — optional word-cut speaker default/fallbacks, the
  unconditional four-word generic floor, and ADR-0042's independent own-TTS
  STOP ambiguity guard across active, reply-tail, and continuation boundaries.
- `core/readiness.py` — speaker ID is advisory unless the word-cut identity filter
  is explicitly enabled.
- `core/enroll.py` and `tools/setup_models.py` — enrollment guidance distinguishes
  normal-final gating, generic multi-voice filtering, and open exact controls.
- `config.json` — committed false default and bounded multi-voice opt-in comment.
- `README.md`, `docs/audio_pipeline.md`, and `docs/unified_architecture.md` —
  operator guidance now says enrollment alone does not filter word-cut; the
  multi-voice filter requires the explicit true flag.
- `tests/test_barge_word_cut.py` — default policy, exact/four-word success,
  zero-to-three/echo/silence rejection, explicit speaker-filter coverage, and
  default-policy ADR-0042 own-TTS ambiguity regressions.
- `tests/test_setup_doctor.py` — identity-free readiness remains green while the
  existing explicit-filter failure cases remain pinned.
- `tests/test_sherpa_duplex_runtime.py` — the real worker/FIFO cancellation fixture
  exercises the enrollment-free default instead of an explicit test override.
- `STATUS.md` — current behavior, verification, preserved physical-red boundary,
  and next live diagnostic.
- `docs/adr/0072-make-word-cut-enrollment-optional.md` — new decision.
- `docs/adr/0045-require-lexical-floor-before-speaker-barge.md` and
  `docs/adr/0071-reject-current-v5-candidate-and-refocus-physical-barge-admission.md`
  — supersession status only.
- `TASK_RESULT.md` — this handoff.

## Verification

```text
/home/dobo/work/speaker/.venv/bin/python -m pytest \
  tests/test_barge_word_cut.py tests/test_setup_doctor.py \
  tests/test_sherpa_duplex_runtime.py tests/test_capture_integration.py \
  tests/test_final_trust_lineage.py tests/test_barge_in_suppression.py \
  tests/test_device_profile_invariants.py tests/test_speaker_input_gate.py -q
375 passed in 3.96s

/home/dobo/work/speaker/.venv/bin/python -m pytest \
  tests/test_apm_double_talk.py -q
6 passed in 0.87s

/home/dobo/work/speaker/.venv/bin/python -m pytest tests -q
4062 passed, 31 skipped, 9 warnings in 76.23s

git diff --check
PASS (no output)

test "$(wc -l < STATUS.md)" -le 100
PASS (STATUS.md is exactly 100 lines)
```

The first focused invocation was blocked before collection because the managed
sandbox could not create the worktree's ignored `logs/tests` directory. The same
command was rerun with the normal worktree-write permission and passed; this was
an environment permission issue, not a repository failure.

No microphone, speaker, raw audio, enrollment file, model service, network,
secret, or machine-local configuration was read or modified by these tests.

## Risks and manual validation

- This policy change does not fix the 2026-07-14 physical failure. The
  enrollment-off run already starved before identity: playback-time VAD stayed
  zero and no energy fallback/decoder trace appeared.
- Novel exact canonical controls deliberately remain short. A STOP-class control
  that resembles current/recent own TTS fails closed without a compatible,
  warmed speaker decision even when the generic multi-voice filter is off.
  Silent control and current-TTS Stop phrasing still require live no-self-cut A/B.
- Physical acceptance still requires prompt causal exact Stop on the bare laptop
  speaker, followed by one four-or-more-word override, with no self-cut. No live
  hardware validation was run or claimed here.
- Windows voice-communications word-cut remains unavailable under ADR-0019.

## Merge recommendation

The implementation and deterministic gates are green and scoped to the owner
decision. It is ready for orchestrator review and landing, with physical
bare-speaker validation retained as a separate red post-merge gate.
