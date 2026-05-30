"""Real-usage test harness: run the user's recordings through the REAL assistant
(real STT -> real LLM -> real TTS) with the REAL laptop audio OUTPUT, so the
parts the headless FileReplayEngine skips -- the ALSA output path, the blocking
playback thread, and its shutdown -- ARE exercised and the three live failures
(shutdown hang, barge-in storm, broken output) become visible + gradeable.

CLI: ``python -m tools.real_usage`` (on-machine; needs real models + audio).
The PURE grading logic lives in :mod:`tools.real_usage.report` and is unit-tested
in ``tests/test_real_usage_grading.py`` without audio/models.
"""
from . import report  # noqa: F401  (re-export the pure, unit-tested grading)

__all__ = ["report"]
