"""Owner-requirement guards for open-speaker barge-in (CLAUDE.md HARD REQUIREMENT).

These three tests encode what the owner demands and the recorded live failure
``run-20260609-203236`` did NOT originally deliver: a NORMAL-volume talk-over on
the bare ALC285 laptop speaker MUST cut the assistant's TTS promptly -- WITHOUT a
shout and WITHOUT headphones. Live, the fused-z-score ``AdaptiveDTD`` *fired* on
the turn-2 normal-volume talk-over (raw 0.0024-0.0068), but the downstream
capture-loop integrator + one-per-run latch only converted a fire into an actual
cut on the turn-3 SHOUT (raw=0.0704, ~10x normal). summary.json recorded turn-2
``barge_in_latency=null`` -- a missed barge.

THE FIX (2026-06-09): the leaky ``voiced_run *= 0.5`` accumulator was replaced by
``core.engines._dtd.BargeSustain`` -- a bounded windowed sustain that cuts when
enough eligible blocks land within a short trailing window, so the intermittent
DTD fires of a NORMAL-volume talk-over now convert to a cut (and the bounded
window keeps sporadic echo from ever self-interrupting -- pinned in
``test_barge_echo_must_not_fire.py``). These tests therefore now PASS, and stand
as the regression guards for the requirement: if the integrator/latch is changed
so a normal-volume talk-over stops cutting in time, they go red.

Tier 0: pure replay of the recorded per-frame DTD trace through the REAL
``AdaptiveDTD.decide`` + the REAL ``BargeSustain`` (via ``barge_fixtures.run_frames``).
No audio device, no model, no sound card.
"""
from __future__ import annotations

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


def test_normal_volume_talkover_must_produce_a_stop_within_latency_bound():
    """A normal-volume talk-over MUST produce a stop within the latency bound.

    Seed the live ``AdaptiveDTD`` on the preceding echo-floor frames (idx 0..9)
    so the control charts learn the same echo baseline they had live, then feed
    the turn-2 NORMAL-volume burst (idx 10..14, raw 0.0024-0.0068) through the
    REAL detector + the REAL ``BargeSustain``.

    OWNER REQUIREMENT: a stop IS produced, it lands within the turn-2 burst, and
    its latency *from the talk-over start* is within ~0.5s. Before the fix the
    intermittent DTD fires never let the leaky ``voiced_run *= 0.5`` accumulator
    reach the threshold; the windowed sustain now cuts on the 2nd eligible block.
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
    # The cut lands within the turn-2 normal-volume burst, not later.
    assert result.first_fire_index in _TURN2_NORMAL_IDX, (
        f"cut landed at idx {result.first_fire_index}, outside the turn-2 burst "
        f"{list(_TURN2_NORMAL_IDX)}"
    )
    # Latency is measured from the talk-over START (idx 10), not the echo-seed
    # start -- the seed frames only warm the chart, they are not the barge.
    talkover_start_t = next(f.t_sec for f in frames if f.idx == 10)
    latency = result.first_fire_t_sec - talkover_start_t
    assert 0.0 <= latency <= _LATENCY_BOUND_SEC, (
        f"normal-volume talk-over cut too late: {latency:.3f}s "
        f"(bound {_LATENCY_BOUND_SEC}s)"
    )


def test_talkover_does_not_require_a_shout_to_fire():
    """The cut must NOT wait for the 10x shout escalation to land.

    Feed only the turn-3 PRE-escalation frames (idx 185..189, raw <= 0.0144) --
    the normal-loudness portion BEFORE raw jumps ~10x to 0.0704 at idx 196 -- to a
    detector already warmed on the preceding real audio (a fresh ``AdaptiveDTD``
    needs ``warmup_frames`` blocks before it can fire at all; live it was long
    warmed by idx 185). The cut must occur within these pre-shout frames.

    OWNER REQUIREMENT (CLAUDE.md HARD REQUIREMENT, 'no scream' half): a normal
    talk-over must cut; headphones/shout are NOT an acceptable fix. Live, the cut
    only landed at idx 196 (raw 0.0704, a 10x shout); the windowed sustain now
    cuts on the 2 pre-shout eligible blocks (idx 188, 189).
    """
    frames = load_trace_frames()
    pre_shout = [f for f in frames if 185 <= f.idx <= 189]
    # Sanity: this window is normal-loudness, NOT the shout.
    assert all(f.raw <= 0.0144 for f in pre_shout), (
        "pre-shout window unexpectedly contains a shout-level frame"
    )

    # Warm the detector on the real frames preceding the burst (the chart state it
    # had live), then replay ONLY the pre-shout burst through the warmed detector.
    dtd = build_live_dtd()
    run_frames([f for f in frames if f.idx < 185], dtd, reset_latch_per_turn=True)
    result = run_frames(pre_shout, dtd, reset_latch_per_turn=False)

    assert result.first_fire_index is not None, (
        "no cut on the pre-escalation (normal-loudness) talk-over frames -- the "
        "barge must not wait for the raw=0.0704 shout at idx 196"
    )
    assert result.first_fire_index in range(185, 190), (
        f"cut landed at idx {result.first_fire_index}, outside the pre-shout "
        f"window 185..189 -- it waited for the escalation"
    )


def test_recorded_barge_latency_bound_holds_on_first_real_talkover():
    """End-to-end: the first real talk-over in the recorded run cuts in time.

    Replay the WHOLE recorded sequence with the latch re-armed per turn
    (``reset_latch_per_turn=True`` models the silent->speaking re-arm). The FIRST
    human talk-over is turn-2 (the normal-volume burst at idx 10..14, ~20:34:22).
    That first real talk-over must produce a cut whose latency from burst-start is
    within the owner bound.

    Before the fix the first real talk-over (turn-2) yielded
    ``barge_in_latency=null`` -- no fire converted there and the only cut landed on
    the turn-3 shout (idx 196). With the windowed sustain the cut now lands inside
    turn-2.
    """
    frames = load_trace_frames()
    result = run_frames(frames, build_live_dtd(), reset_latch_per_turn=True)

    # The first real talk-over is turn-2 (idx 10..14). Find the first cut there.
    turn2_fire = next(
        ((idx, t) for idx, t in result.fires if idx in _TURN2_NORMAL_IDX), None
    )

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
