# Status — speaker

Durable status for agents. Update this file when project direction, verification commands, or operational assumptions change.

## Role in the fleet
Local voice/audio stack with speaker, microphone, and engine behavior that may require both headless and live validation.

## Current operational focus
- Keep APM/double-talk/barge-in behavior covered by headless regression tests.
- Keep generated logs/audio captures out of commits.
- Distinguish automated test pass from live hardware validation.

## Standard verification
```bash
/home/dobo/work/speaker/.venv/bin/python -m pytest tests/test_apm_double_talk.py -q
git diff --check
```

## Agent notes
- Do not delete logs unless Paul explicitly asks.
- Do not claim live hardware validation unless it actually ran.
- Never read or print secret values.
