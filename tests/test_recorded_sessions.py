"""
Recorded-session regression test runner.

This module auto-discovers and executes generated test files in
``tests/recorded/``.  Each file is produced by::

    python scripts/generate_session_tests.py

Running this suite validates that the barge-in pipeline behaves consistently
with previously recorded live sessions.  Each recorded TTS turn becomes:

- A *no-barge-in* test: asserts self-interruption does NOT happen again.
- A *true-positive* test: asserts the user interrupt still fires (no regression
  in sensitivity).
- A *candidate* test (xfail): classified uncertain until a human annotates
  ``turn.json``.

Usage::

    pytest tests/test_recorded_sessions.py          # run all replay tests
    pytest tests/test_recorded_sessions.py -v       # verbose output per turn
    pytest tests/test_recorded_sessions.py -k no_self_interrupt  # only safety tests
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.recorded, pytest.mark.audio]

RECORDED_DIR = Path(__file__).parent / "recorded"
RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
VALID_TURN_ANNOTATIONS = {"", "true_positive", "false_positive", "unknown"}
OPTIONAL_LATENCY_FIELDS = {
    "speech_end_to_stt_ms",
    "stt_to_first_llm_sentence_ms",
    "tts_on_start_ms",
    "barge_in_to_stop_ms",
}


def _collect_recorded_modules():
    """Return list of (module_name, path) for all generated test files."""
    if not RECORDED_DIR.exists():
        return []
    return sorted(RECORDED_DIR.glob("test_session_*.py"))


# ── Dynamic test collection ───────────────────────────────────────────────────
# pytest discovers tests via conftest.py collect_ignore or by importing this
# file.  We import each generated module here and expose its tests.

def _import_generated(path: Path):
    spec = importlib.util.spec_from_file_location(f"recorded.{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


for _p in _collect_recorded_modules():
    try:
        _mod = _import_generated(_p)
        # Inject test functions into this module's namespace so pytest finds them
        for _name, _obj in vars(_mod).items():
            if _name.startswith("test_") and callable(_obj):
                globals()[f"test_recorded__{_p.stem}__{_name}"] = _obj
    except Exception as _exc:
        # Don't crash the whole suite if one generated file has an issue
        import warnings
        warnings.warn(f"Could not import {_p}: {_exc}")


# ── Meta-tests: recording infrastructure ─────────────────────────────────────

class TestRecordingInfrastructure:
    """
    Sanity checks that the recording infrastructure itself is working.
    These tests are always fast (no audio processing).
    """

    def test_recordings_dir_exists(self):
        assert RECORDINGS_DIR.exists(), (
            "recordings/ directory missing.  "
            "Run 'python main.py --record' at least once."
        )

    def test_aggregate_json_schema(self):
        agg_path = RECORDINGS_DIR / "aggregate.json"
        if not agg_path.exists():
            pytest.skip("No sessions recorded yet")

        with open(agg_path) as f:
            agg = json.load(f)

        assert "sessions" in agg, "aggregate.json missing 'sessions' key"
        assert "summary" in agg, "aggregate.json missing 'summary' key"
        summary = agg["summary"]
        assert "total_sessions" in summary
        assert "false_positive_rate" in summary
        assert 0.0 <= summary["false_positive_rate"] <= 1.0, (
            f"false_positive_rate out of range: {summary['false_positive_rate']}"
        )

    def test_session_metadata_schema(self):
        sessions = sorted(RECORDINGS_DIR.glob("session_*/metadata.json"))
        if not sessions:
            pytest.skip("No sessions recorded yet")

        # Check latest session
        with open(sessions[-1]) as f:
            meta = json.load(f)

        required = [
            "session_id", "start_time", "duration_sec",
            "num_tts_turns", "num_barge_ins",
        ]
        for key in required:
            assert key in meta, f"metadata.json missing '{key}'"

    def test_turn_annotations_and_optional_latency_schema(self):
        """
        Recorded turns may be human-annotated and may include realtime metrics.

        Both are optional for older sessions, but when present they must use the
        same schema that generated replay tests and dashboards expect.
        """
        turn_files = sorted(RECORDINGS_DIR.glob("session_*/turn_*/turn.json"))
        if not turn_files:
            pytest.skip("No recorded turns yet")

        for turn_path in turn_files:
            with open(turn_path) as f:
                turn = json.load(f)

            annotation = turn.get("annotation", "")
            assert annotation in VALID_TURN_ANNOTATIONS, (
                f"{turn_path} has unsupported annotation {annotation!r}; "
                f"expected one of {sorted(VALID_TURN_ANNOTATIONS)}"
            )
            if annotation and "annotation_reason" in turn:
                assert isinstance(turn["annotation_reason"], str)

            for field in OPTIONAL_LATENCY_FIELDS:
                if field in turn:
                    value = turn[field]
                    assert isinstance(value, (int, float)), (
                        f"{turn_path} field {field!r} must be numeric"
                    )
                    assert value >= 0, (
                        f"{turn_path} field {field!r} must be non-negative"
                    )

    def test_false_positive_rate_acceptable(self):
        """
        Barge-in false positive rate across all sessions must be < 50%.

        A false positive candidate is any barge-in where voiced=True but
        echo_sim > 0.20 (suggesting TTS echo rather than user speech).
        This threshold is intentionally generous — the point is to catch
        catastrophic regressions, not to enforce perfection.
        """
        agg_path = RECORDINGS_DIR / "aggregate.json"
        if not agg_path.exists():
            pytest.skip("No sessions recorded yet")

        with open(agg_path) as f:
            agg = json.load(f)

        rate = agg["summary"].get("false_positive_rate", 0.0)
        assert rate < 0.50, (
            f"False-positive candidate rate is {rate:.1%} — more than half of "
            "all recorded TTS turns had suspicious barge-ins.  Check barge-in "
            "detection configuration."
        )
