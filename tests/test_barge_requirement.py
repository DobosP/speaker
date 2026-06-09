"""Owner-requirement pins for open-speaker barge-in (CLAUDE.md HARD REQUIREMENT).

These three tests encode what the owner demands and the recorded live failure
``run-20260609-203236`` does NOT yet deliver: a NORMAL-volume talk-over on the
bare ALC285 laptop speaker MUST cut the assistant's TTS promptly -- WITHOUT a
shout and WITHOUT headphones. Live, the fused-z-score ``AdaptiveDTD`` *fired* on
the turn-2 normal-volume talk-over (raw 0.0024-0.0068), but the downstream
capture-loop integrator + one-per-run latch only converted a fire into an actual
cut on the turn-3 SHOUT (raw=0.0704, ~10x normal). summary.json recorded turn-2
``barge_in_latency=null`` -- a missed barge.

All three are ``@pytest.mark.xfail(strict=True)``: they fail TODAY (so they stay
GREEN in CI as expected-fails), and the moment the integrator/latch/sustain fix
makes a normal-volume talk-over cut within the latency bound they FLIP TO FAILURE
(strict forbids xpass) -- the explicit signal to remove the marker.

Tier 0: pure replay of the recorded per-frame DTD trace through the REAL
``AdaptiveDTD.decide`` (via ``barge_fixtures.run_frames``). No audio device, no
model, no sound card. The seam under test -- ``decide()`` -- is the real code;
only the ~6-line capture-loop accumulator is mirrored in the driver (see
``barge_fixtures.run_frames`` / core/engines/sherpa.py:1404-1426).
"""
from __future__ import annotations

import pytest

from tests.barge_fixtures import (
    build_live_dtd,
    load_trace_frames,
    run_frames,
    talkover_frames,
)

#: Owner bound: a real talk-over must cut within ~0.5s -- not require a shout.
_LATENCY_BOUND_SEC = 0.5

#: 0-based indices of the turn-2 NORMAL-volume talk-over burst.
_TURN2_NORMAL_IDX = range(10, 15)  # idx 10..14

_XFAIL_REASON = (
    "run run-20260609-203236: barge does not cut on normal-volume talk-over"
)


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
def test_normal_volume_talkover_must_produce_a_stop_within_latency_bound():
    """A normal-volume talk-over MUST produce a stop within the latency bound.

    Seed the live ``AdaptiveDTD`` on the preceding echo-floor frames (idx 0..9)
    so the control charts learn the same echo baseline they had live, then feed
    the turn-2 NORMAL-volume burst (idx 10..14, raw 0.0024-0.0068) through the
    REAL detector + the live integrator.

    OWNER REQUIREMENT: a stop IS produced and it lands within ~0.5s. Fails today
    -- the intermittent DTD fires + the ``voiced_run *= 0.5`` decay never let the
    leaky integrator reach the 0.3s ``min_speech_sec`` threshold on the normal
    burst; only a shout breaks through.
    """
    frames = load_trace_frames()
    # Seed the chart baseline on the echo floor that preceded the talk-over.
    echo_baseline = [f for f in frames if f.idx <= 9]
    turn2_normal = talkover_frames(frames)["turn2_normal"]
    sequence = echo_baseline + turn2_normal

    # ONE fresh detector walks the seed frames then the burst (REAL decide()).
    result = run_frames(sequence, build_live_dtd(), reset_latch_per_turn=False)

    assert result.first_fire_index is not None, (
        "normal-volume talk-over produced NO stop (barge_in_latency=null) -- it "
        "must cut without a shout"
    )
    assert result.first_fire_latency_sec <= _LATENCY_BOUND_SEC, (
        f"normal-volume talk-over cut too late: "
        f"{result.first_fire_latency_sec:.3f}s > {_LATENCY_BOUND_SEC}s bound"
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
def test_talkover_does_not_require_a_shout_to_fire():
    """The cut must NOT wait for the 10x shout escalation to land.

    Feed only the turn-3 PRE-escalation frames (idx 185..189, raw <= 0.0144) --
    the normal-loudness portion BEFORE raw jumps ~10x to 0.0704 at idx 196. The
    cut must occur within these pre-shout frames.

    OWNER REQUIREMENT (CLAUDE.md HARD REQUIREMENT, 'no scream' half): normal
    talk-over must cut; headphones/shout are NOT an acceptable fix. Fails today
    -- live, the cut only landed at idx 196 (raw 0.0704, a 10x shout).
    """
    frames = load_trace_frames()
    pre_shout = [f for f in frames if 185 <= f.idx <= 189]
    # Sanity: this window is normal-loudness, NOT the shout.
    assert all(f.raw <= 0.0144 for f in pre_shout), (
        "pre-shout window unexpectedly contains a shout-level frame"
    )

    result = run_frames(pre_shout, build_live_dtd(), reset_latch_per_turn=False)

    assert result.first_fire_index is not None, (
        "no cut on the pre-escalation (normal-loudness) talk-over frames -- the "
        "barge must not wait for the raw=0.0704 shout at idx 196"
    )
    assert result.first_fire_index in range(185, 190), (
        f"cut landed at idx {result.first_fire_index}, outside the pre-shout "
        f"window 185..189 -- it waited for the escalation"
    )


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
def test_recorded_barge_latency_bound_holds_on_first_real_talkover():
    """End-to-end: the first real talk-over in the recorded run must cut in time.

    Replay the WHOLE recorded sequence with the latch re-armed per turn
    (``reset_latch_per_turn=True`` models the silent->speaking re-arm). The FIRST
    human talk-over is turn-2 (the normal-volume burst at idx 10..14, ~20:34:22).
    That first real talk-over must produce a cut whose latency from burst-start
    is within the owner bound.

    Today the first real talk-over (turn-2) yields ``barge_in_latency=null`` -- no
    fire converts there; the only cut lands on the turn-3 shout (idx 196). strict
    xfail flips to FAIL once the chain turns the first normal talk-over into a
    timely stop.
    """
    frames = load_trace_frames()
    result = run_frames(frames, build_live_dtd(), reset_latch_per_turn=True)

    # The first real talk-over is turn-2 (idx 10..14). Find the first cut there.
    turn2_fire = next(
        (idx, t) for idx, t in result.fires if idx in _TURN2_NORMAL_IDX
    ) if any(idx in _TURN2_NORMAL_IDX for idx, _ in result.fires) else None

    assert turn2_fire is not None, (
        f"first real talk-over (turn-2, idx 10..14) produced no cut; all cuts "
        f"landed at {result.fires} -- only the shout broke through"
    )

    # Latency from the start of the turn-2 burst to that cut.
    burst_start = next(f.t_sec for f in frames if f.idx in _TURN2_NORMAL_IDX)
    latency = turn2_fire[1] - burst_start
    assert latency <= _LATENCY_BOUND_SEC, (
        f"first real talk-over cut too late: {latency:.3f}s > "
        f"{_LATENCY_BOUND_SEC}s bound"
    )
