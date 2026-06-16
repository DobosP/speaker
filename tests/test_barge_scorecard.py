"""Phase-0 barge-in scorecard: regression gate + absolute owner requirement.

Pins the measured barge-in baseline (``tests.barge_scorecard.build_scorecard``)
so any future change to the barge chain that worsens self-interrupt count,
missed-barge count, or cut latency fails CI -- and asserts the absolute owner
requirement (0 self-interrupts, 0 missed talk-overs, cut <= 0.5s) independently
of the committed baseline.

Tier 0: deterministic trace replay of the REAL shipped chain, no audio/models.
After an INTENTIONAL improvement, regenerate the baseline with:
    SPEAKER_REFRESH_BARGE_BASELINE=1 python -m pytest tests/test_barge_scorecard.py
"""
from __future__ import annotations

import json
import os

import pytest

from tests.barge_scorecard import BASELINE_PATH, LATENCY_BOUND_SEC, build_scorecard

#: Deterministic replay, but allow a hair of latency slack for harmless refactors.
_LATENCY_SLACK_SEC = 0.05


@pytest.fixture(scope="module")
def scorecard():
    sc = build_scorecard()
    if os.environ.get("SPEAKER_REFRESH_BARGE_BASELINE") == "1":
        BASELINE_PATH.write_text(json.dumps(sc, indent=2) + "\n")
    return sc


def test_owner_requirement_holds_in_replay(scorecard):
    """The absolute bar (CLAUDE.md HARD REQUIREMENT), independent of the baseline."""
    s = scorecard["summary"]
    assert s["self_interrupt_cuts"] == 0, (
        f"barge chain self-interrupts in replay: {scorecard['traces']}"
    )
    assert s["missed_barges"] == 0, "a recorded talk-over was missed in replay"
    assert s["max_cut_latency_sec"] is not None, "no talk-over cut recorded at all"
    assert s["max_cut_latency_sec"] <= LATENCY_BOUND_SEC, (
        f"talk-over cut latency {s['max_cut_latency_sec']}s exceeds the "
        f"{LATENCY_BOUND_SEC}s owner bound"
    )


def test_no_regression_vs_committed_baseline(scorecard):
    """Fail if any headline metric degrades vs the committed Phase-0 baseline."""
    assert BASELINE_PATH.exists(), (
        f"missing baseline {BASELINE_PATH}; regenerate with "
        f"SPEAKER_REFRESH_BARGE_BASELINE=1 python -m pytest "
        f"tests/test_barge_scorecard.py"
    )
    base = json.loads(BASELINE_PATH.read_text())["summary"]
    cur = scorecard["summary"]

    assert cur["self_interrupt_cuts"] <= base["self_interrupt_cuts"], (
        f"self-interrupt count regressed: {cur['self_interrupt_cuts']} > "
        f"{base['self_interrupt_cuts']}"
    )
    assert cur["missed_barges"] <= base["missed_barges"], (
        f"missed-barge count regressed: {cur['missed_barges']} > {base['missed_barges']}"
    )
    assert cur["talkovers_cut"] >= base["talkovers_cut"], (
        f"fewer talk-overs cut than baseline: {cur['talkovers_cut']} < "
        f"{base['talkovers_cut']}"
    )
    if (
        base["max_cut_latency_sec"] is not None
        and cur["max_cut_latency_sec"] is not None
    ):
        assert (
            cur["max_cut_latency_sec"]
            <= base["max_cut_latency_sec"] + _LATENCY_SLACK_SEC
        ), (
            f"cut latency regressed: {cur['max_cut_latency_sec']}s vs baseline "
            f"{base['max_cut_latency_sec']}s (+{_LATENCY_SLACK_SEC}s slack)"
        )
