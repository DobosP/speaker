"""Phase-0 barge-in scorecard: a measured baseline for open-speaker barge-in.

The barge-in suite already asserts the right pass/fail properties (no
self-interrupt on echo; a normal talk-over cuts within 0.5s). This module turns
those booleans into QUANTIFIED metrics over the canonical recorded failure
traces -- self-interrupt count/rate, missed-barge count, and cut-off latency --
so the rest of the barge-in roadmap (docs/roadmap_2026-06-17.md Phase 2) is
judged against a numeric baseline instead of "the tests still pass".

It drives the SAME real shipped chain the suite uses
(``tests.barge_fixtures.run_frames_engine`` -> the real ``AdaptiveDTD`` +
residual-floor ``_looks_like_user`` gate + ``BargeSustain``), audio-free and
deterministic, so a number that moves is a real behaviour change, not noise.

SCOPE / HONESTY: this is the deterministic, CI-able proxy that pins the barge
DECISION LOGIC against the recorded traces. It does NOT replace the live
open-speaker A/B on the bare laptop speaker (that validates the ACOUSTICS with
the owner at the mic, and is the actual acceptance bar in roadmap Phase 0/2).
This scorecard is the regression net that protects the decision logic between
those live sessions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tests.barge_fixtures import (
    ACOUSTIC_EVENT_IDX,
    Frame,
    frames_with_turn_starts,
    load_self_interrupt_frames,
    load_trace_frames,
    run_frames_engine,
    talkover_frames,
)

#: Owner requirement (CLAUDE.md HARD REQUIREMENT): a real talk-over must cut
#: within ~0.5s on the bare speaker -- without a shout, without headphones.
LATENCY_BOUND_SEC = 0.5

#: Committed baseline this scorecard is diffed against (the Phase-0 reference).
BASELINE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "barge_in" / "phase0_scorecard.json"
)


def _round(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 3)


def _burst_cut(burst: Sequence[Frame], fires: List) -> Dict:
    """Did any fire land inside this talk-over burst, and at what latency?

    ``fires`` is the engine run's ``[(idx, t_sec), ...]``. Latency is measured
    from the START of the burst (its first frame) to the cut, matching
    test_barge_requirement.
    """
    idxs = {f.idx for f in burst}
    hit = next(((i, t) for i, t in fires if i in idxs), None)
    if hit is None:
        return {"cut": False, "fire_idx": None, "latency_sec": None}
    start_t = burst[0].t_sec
    return {"cut": True, "fire_idx": hit[0], "latency_sec": _round(hit[1] - start_t)}


def score_203236() -> Dict:
    """Talk-over + echo-safety over the primary live-failure trace (204 frames).

    Drives the full engine seam with per-turn re-arm (the shipped path), then
    classifies every cut by the frame it landed on:
      * a cut on a talk-over burst  -> a CORRECT barge (record latency)
      * a cut on a clean echo frame -> a SELF-INTERRUPT (must be 0)
      * a cut on an ACOUSTIC_EVENT_IDX frame -> excluded (per the suite: these
        are short acoustic transients at talk-over level, not clean echo)
    """
    frames = load_trace_frames()
    res = run_frames_engine(frames_with_turn_starts(frames))
    fires = res.fires

    tof = talkover_frames(frames)
    turn2 = _burst_cut(tof["turn2_normal"], fires)
    turn3 = _burst_cut(tof["turn3_shout"], fires)

    talkover_idxs = {f.idx for f in tof["turn2_normal"]} | {
        f.idx for f in tof["turn3_shout"]
    }
    echo_self_interrupts = [
        i
        for i, _t in fires
        if i not in talkover_idxs and i not in ACOUSTIC_EVENT_IDX
    ]
    n_echo_frames = sum(
        1
        for f in frames
        if f.idx not in talkover_idxs and f.idx not in ACOUSTIC_EVENT_IDX
    )

    bursts = {"turn2_normal": turn2, "turn3_shout": turn3}
    return {
        "frames": len(frames),
        "dtd_fire_count": res.dtd_fire_count,
        "cuts_total": len(fires),
        "talkover_bursts": bursts,
        "talkovers_total": len(bursts),
        "talkovers_cut": sum(1 for b in bursts.values() if b["cut"]),
        "missed_barges": sum(1 for b in bursts.values() if not b["cut"]),
        "self_interrupt_cuts": len(echo_self_interrupts),
        "self_interrupt_fire_idxs": echo_self_interrupts,
        "echo_frames": n_echo_frames,
        "self_interrupt_rate": _round(
            len(echo_self_interrupts) / n_echo_frames if n_echo_frames else 0.0
        ),
    }


def score_234435() -> Dict:
    """Pure self-interrupt trace (28 frames, all the assistant's own echo).

    Every cut here is a self-interrupt -- the shipped residual-floor gate must
    reject all of them (target 0).
    """
    frames = load_self_interrupt_frames()
    res = run_frames_engine(frames)
    return {
        "frames": len(frames),
        "dtd_fire_count": res.dtd_fire_count,
        "self_interrupt_cuts": len(res.fires),
        "self_interrupt_fire_idxs": [i for i, _t in res.fires],
        "echo_frames": len(frames),
        "self_interrupt_rate": _round(len(res.fires) / len(frames) if frames else 0.0),
    }


def build_scorecard() -> Dict:
    """Aggregate the per-trace metrics into a Phase-0 baseline + fleet summary."""
    t_203236 = score_203236()
    t_234435 = score_234435()

    self_interrupts = t_203236["self_interrupt_cuts"] + t_234435["self_interrupt_cuts"]
    missed = t_203236["missed_barges"]
    latencies = [
        b["latency_sec"]
        for b in t_203236["talkover_bursts"].values()
        if b["cut"] and b["latency_sec"] is not None
    ]
    max_lat = max(latencies) if latencies else None

    summary = {
        "self_interrupt_cuts": self_interrupts,
        "missed_barges": missed,
        "talkovers_total": t_203236["talkovers_total"],
        "talkovers_cut": t_203236["talkovers_cut"],
        "cut_latencies_sec": latencies,
        "max_cut_latency_sec": max_lat,
        "latency_bound_sec": LATENCY_BOUND_SEC,
        # The owner requirement, expressed as a single boolean.
        "meets_requirement": (
            self_interrupts == 0
            and missed == 0
            and (max_lat is not None and max_lat <= LATENCY_BOUND_SEC)
        ),
    }
    return {
        "_about": (
            "Phase-0 barge-in baseline (deterministic trace replay of the real "
            "shipped chain). Regenerate with SPEAKER_REFRESH_BARGE_BASELINE=1 "
            "pytest tests/test_barge_scorecard.py. Does NOT replace the live "
            "open-speaker A/B."
        ),
        "traces": {"run-20260609-203236": t_203236, "run-20260609-234435": t_234435},
        "summary": summary,
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(json.dumps(build_scorecard(), indent=2))
