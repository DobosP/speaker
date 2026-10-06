# Linux handover — unfinished addressing shortcut repair

Valid until: this candidate is qualified and landed or explicitly replaced on Linux — then treat as history.

Date: 2026-10-06
State: WIP transport only; unqualified, not merged to main.
Owner: Paul / the existing Linux Speaker session.
Branch: `fix/addressing-shortcut-boundary-20261006`
Main base: `734713dde33b0f5bdf35430258fb066535f76985`

Paul stopped Windows loops and authorized pushing unfinished work for Linux continuation.
The earlier characterization and shortcut correction are consolidated into one task
commit. No new workspace is needed. All Windows original sources, scratch packets
and raw receipts remain preserved locally; they are not blindly exported.

## Candidate and coverage

The imperative shortcut now fullmatches only the fixed repeat request. Search,
memory and requested-speech payloads, added details, quotation and narration reach
the learned gate with the complete original utterance. Independent source review
found no issues. Proposed ADR-0227 records the decision and remaining limitations.

The three focused files contain **136 source-derived cases: 47 + 19 + 70**.
Candidate interpreter/collection/pytest execution is **NOT RUN**. No candidate
JUnit or pass count exists. Two Windows admission checks started no child/tests
because free commit was below the earlier coordinator's 16 GiB floor. The candidate
has not received a native, model, GPU, microphone, acoustic or live qualification.

Git blob identities below are portable across normal CRLF/LF checkouts:

| Path | Git blob |
| --- | --- |
| `core/addressing.py` | `48e3df78e9f6443dff670e939c81585d71683306` |
| `tests/test_addressing.py` | `19bc534b179c21b7ecfba8956068454b2a103074` |
| `tests/test_ambient_addressing_controls.py` | `dfc587a8053ddccbe87ddfff42f34500d019c948` |
| `tests/test_addressing_shortcut_boundary.py` | `7cff1c174f9cdeb4934615b10d2b58c644da0d23` |

The exact Windows physical candidate bytes are preserved. Their SHA256 values are:

- core: `530CAB961DFC007E07DCEDF78E0B31BDF50C187A5DB32DC0C1777F5B7972E705`
- addressing tests: `E23DBDB97CB26A984A28D805DA725B91C2A49EC78B6EE669513988BEBCE4E7BA`
- ambient controls: `656794418DFB1A867585306F03A9CC123B986F780F8628B8F97E35FC86788277`
- boundary tests: `0BFC8F3484A39BC98400447041368D5FAA49C3CEB7C5D1E94D3B6B8C6E3C6DE3`

## Evidence already obtained

Earlier CPython 3.10.11 baseline characterization passed 66 focused and 304 adjacent
cases, with one unavailable Go-renderer skip. These historical results do not qualify
the current source. An isolated CPython 3.11.15 / pytest 9.0.2 import-only preflight
imported all three selected files: actual pytest tests zero. A separate historical
source overlay produced exactly nine expected ordinary assertion failures, clean
setup/teardown, zero errors/skips and pytest/driver exit 1; owner/launcher exited zero.
It reproduced the old shortcut bypass, without writing the physical candidate.

Original local receipts remain in the task scratch lane. Exact SHA256 bindings:

| Receipt | SHA256 |
| --- | --- |
| Import-only result | `C8934A1AF49FEF36B76C9030B0200558A5B5D0727AEFB463EF35217C5E45AE73` |
| Historical-nine result | `54E090343CC632830FBBA4262589D9BDAEE07842A16F58E1F5B48BD677030C2D` |
| Historical-nine raw JUnit | `2259E85E998B78C0D09AC3B3838A04D8BA8E25367523887218987FC49394F9ED` |
| Immutable FILETIME erratum | `AB122546BFDF853448195D48105D3A99C2096C26AC3C22D97EEBC61822A41FBD` |
| Unexecuted candidate manifest | `C9FD7DA73C99A0EBD78EC6BDF9A80140C634AD46C89E058B39773273A8CA7826` |

The erratum preserves exact launcher FILETIME decimal string `134357628463040458`;
the original consolidated report rounded its numeric copy. Original reports are
preserved. Import success and historical failures are separate from candidate success.
Windows-specific drivers/manifests contain machine paths and are not Linux dependencies.

## Remaining Linux checks

Fetch this branch into the existing clean Linux checkout; keep current work and
all unrelated holds. Confirm the four Git blobs above and use the existing project
interpreter. Keep test logs, local config, live actors and bytecode disabled; use
one-thread BLAS and task-local temporary output. Run the focused gate first:

```bash
SPEAKER_TEST_LOG=0 SPEAKER_NO_LOCAL_CONFIG=1 SPEAKER_LIVE=0 \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
~/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider \
  tests/test_addressing.py tests/test_ambient_addressing_controls.py \
  tests/test_addressing_shortcut_boundary.py -q
```

Require actual collection of 136 unique cases and zero failures/errors/skips; record
actual outcomes and a task-local JUnit report. Any import/collection/setup failure
is an unsuccessful check, not a reproduced historical bypass or candidate pass.
Then run the adjacent files `test_core_runtime.py`, `test_final_preprocessing_cancel.py`,
`test_pretoken_cancellation.py`, `test_cleanup.py`, `test_capability_catalog.py`,
`test_history_context.py`, `test_tts_markup.py` and `test_setup_minicpm.py` under `tests/`
with the same guards. Record actual counts/skips, without inheriting the old 304 count.

Run scoped Ruff `E9,F63,F7,F82` on the four source/test files if available, the fleet
`~/work/agent-ops/scripts/check_docs.py .` gate and `git diff --check`. Windows Ruff
was unavailable; no installation or full-suite claim was made. Keep STATUS within
120 lines. After actual green qualification, update STATUS/WORKLOG and ADR-0227's
status in the same final improvement commit before landing locally on main.

Exact-ACT semantic mistakes and provider-copied speech remain open. The original
native persistence gaps and model/live/hardware gates remain held. No microphone,
doctor, model/default, enrollment, promotion or private-audio action follows this
source handover. No Windows preparation/testing resumes after transport.
