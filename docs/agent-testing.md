# Agent Testing Guide — speaker

Last verified: 2026-09-07

## Environment
- Preferred interpreter: `~/work/speaker/.venv/bin/python` — it works from any worktree.
- The frozen receipts were taken under this low-priority prefix; reuse it when comparing numbers:
  `SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19 <python> -B -m pytest -p no:cacheprovider`
- Live hardware validation is separate from headless pytest verification; neither substitutes for the other.
- Flutter tests run from `mobile/`.

## Standard gates
| Gate | Command | Expected |
|---|---|---|
| Logic suite (Tier 0, CI) | `~/work/speaker/.venv/bin/python -m pytest tests -q` | green; model/live tiers self-skip. CI runs the same with `--ignore=tests/test_livekit_audio.py --ignore=tests/test_livekit_engine.py` (`.github/workflows/tests.yml:32-34`). Last full count: `WORKLOG.md`. |
| Staged runner | `python tools/run_tests.py list \| unit \| e2e \| real_model \| live \| cloud \| memory \| imports \| full` | the stages exactly as `list` prints them (`docs/testing.md`) |
| APM / double-talk | `~/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q` | `6 passed` |
| Live-launcher lifecycle (headless) | `~/work/speaker/.venv/bin/python -m pytest tests/test_live_launcher.py tests/test_capture_integration.py tests/test_setup_doctor.py -q` | setup/reuse/failure/cleanup/private-path/doctor contracts pass without opening audio |
| Bounded-read race (isolated) | `SPEAKER_TEST_LOG=0 ~/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_streaming_stt_manifest.py::test_bound_file_hash_rejects_in_place_mutation_during_streaming_read -q` | `1 passed`; keep it isolated from the broad streaming-manifest gate and report both conditions |
| Bounded SearXNG ingestion (headless) | `~/work/speaker/.venv/bin/python -m pytest tests/test_websearch.py -q` | `136 passed`; no network — the shipped transport runs through an injected stream factory (ADR-0191) |
| Capability exception sanitization (headless) | `~/work/speaker/.venv/bin/python -m pytest tests/test_capability_exception_sanitization.py tests/test_capability_context_isolation.py tests/test_failure_cascades.py -q` | `30 passed`; provider detail must never reach `error` (ADR-0192) |
| Whitespace | `git diff --check` | no output |
| Scoped lint | `ruff check --select E9,F63,F7,F82 <changed files>` | clean. Whole-file lint carries inherited E731/E402/F401 debt — never claim a full-format-clean result. |
| Readiness (no audio) | `python -m tools.doctor --defer-ollama` | `BASE READY`; this path can never issue full `READY`; run from `~/work/speaker` (models and `config.local.json` are not in task worktrees); the host must also have the ADR-0013 echo-cancel module loaded |
| Docs | `python3 ~/work/agent-ops/scripts/check_docs.py .` | `files=34 dead_links=0 stale_terms=0 retired_verbs=0 orphans=0`; dated files such as `docs/2026-07-17-performance-roadmap.md` are history and are not term-checked |
| Live (opt-in, hardware) | `./live.sh` | doctor `READY` then the microphone; state exactly what ran |

## Before commit
1. Run `git diff --check`.
2. Run targeted pytest for changed audio/engine code.
3. Document live A/B validation still required when hardware behavior is affected.
4. Do not commit generated `logs/**` artifacts.
5. Update `STATUS.md` (facts plus `Last verified`) in the same commit (`AGENTS.md`, Docs discipline).
6. Scrub run bundles for PII before `git add` (`docs/debugging.md`).
7. Keep it to one audio behavior change per branch.
8. Record the commands you ran and the results you observed — never a count you did not observe.

## Known flaky / blocked
- If `.venv` is missing, recreate/use a project venv and record the exact command.
- Do not bypass a red delay-route contract by loading an unrelated host EC route
  or changing system audio defaults; preserve the failed evidence for diagnosis.
- Do not claim live audio validation unless it was actually performed.
- `tests/test_streaming_stt_manifest.py::test_bound_file_hash_rejects_in_place_mutation_during_streaming_read` is timestamp-sensitive; run it isolated.
- The `swigvarlink` `DeprecationWarning` at interpreter exit is inherited from sherpa-onnx 1.13.3.
- `tests/test_aec_seam.py:930` skips without the DTLN ONNX model.
- `real_model` tests deselect or self-skip without downloaded models.
- Two known timeout-plugin warnings appear in the mobile evidence gates; seven inherited Pillow warnings appear in the cloud stage.

## Where the rest lives
- [`testing.md`](testing.md) — tiers, markers, staged runner, CI, and the per-feature gate matrix appendix.
- [`evaluation_runbooks.md`](evaluation_runbooks.md) — protected benchmark and diagnostic procedures, harness semantics.
- [`../WORKLOG.md`](../WORKLOG.md) — dated receipts.
