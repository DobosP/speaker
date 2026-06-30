# Session handoff — live A/B harness + pre-first-audio barge gate

Date: 2026-06-30  
Branch: `main`  
Machine: `dobo-ROG-Strix-G634JY-G634JY` / Linux / `.venv/bin/python`

## What landed

- Added `SherpaOnnxEngine._barge_watch_active()` and used it in the capture loop so playback-time barge-in is not armed before the first audible playback reference exists.
  - Prevents synth lead-in VAD noise from being counted/logged as a rejected/detected “during playback” barge episode.
  - Keeps the watch armed between queued sentences when the prior audio tail is still audible.
- Extended `tools/diagnose_run.py` with heartbeat underrun parsing and pass/fail verdict reporting.
- Added `tools/live_audio_ab.py`, a compact live open-speaker A/B report tool for one or two run logs.
- Added focused regression coverage in:
  - `tests/test_sherpa_playback.py`
  - `tests/test_diagnose_run.py`

## Verification

```bash
.venv/bin/python -m pytest tests/test_diagnose_run.py tests/test_sherpa_playback.py -q
```

Result: `67 passed in 1.74s`.

## Not included in the commit

- Raw run logs and WAVs under `logs/runs/` were left untracked intentionally; do not commit raw voice artifacts unless explicitly curated/scrubbed.
- The second Claude lane (`claude-audio-barge-rootfix`) did not land code in this pass.

## Next steps — pick up here

1. Run a real open-speaker A/B on the other machine:
   ```bash
   python -m tools.live_audio_ab logs/runs/run-before.txt logs/runs/run-after.txt
   ```
2. Validate the `_barge_watch_active()` gate live: no pre-first-audio false barge noise, no missed real inter-sentence talk-over.
3. Continue the deeper P1 APM/DTD residual fix from `.agents/backlog.md`: when `_apm_owns_ns`, feed DTD residual/floor from a non-NS source or reweight toward raw/coherence, then live A/B.
4. If committing future run bundles, scrub transcript/voice PII first per `docs/debugging.md`.
