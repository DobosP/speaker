# Agent Testing Guide — speaker

## Environment
- Runtime: Python audio stack.
- Preferred interpreter on this box: `/home/dobo/work/speaker/.venv/bin/python`.
- Live hardware validation is separate from headless pytest verification.

## Commands
| Scope | Command | Expected success |
|---|---|---|
| APM/DTD regression | `/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q` | `6 passed` on current setup |
| Whitespace | `git diff --check` | no output |

## Before commit
1. Run `git diff --check`.
2. Run targeted pytest for changed audio/engine code.
3. Document live A/B validation still required when hardware behavior is affected.
4. Do not commit generated `logs/**` artifacts.

## Known blockers
- If `.venv` is missing, recreate/use a project venv and record the exact command.
- Do not claim live audio validation unless it was actually performed.
