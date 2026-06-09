"""Trace-replay reproduction of the recorded open-speaker barge-in failure.

Owner: agent-A-trace-replay. Tier 0 (no marker): pure stdlib + the parsed DTD
trace -- NO sound card, NO model, NO audio deps. Every assertion drives the
REAL ``core.engines._dtd.AdaptiveDTD.decide`` (and the mirrored capture-loop
integrator) over the recorded ``run-20260609-203236`` triples via the shared
``tests.barge_fixtures`` helper -- nothing is reimplemented here.

WHAT THIS FILE PINS
-------------------
The recorded live failure: the fused-z-score ``AdaptiveDTD`` *fired* on the
owner's talk-over, but the downstream integrator + one-per-run latch only
converted a fire into an actual cut on the turn-3 SHOUT (raw=0.0704); the
turn-2 NORMAL-volume talk-over (idx 10..14, raw 0.0024-0.0068) was missed
(summary.json: turn-2 ``barge_in_latency=null``).

These tests ASSERT THAT THE BUG EXISTS (they document current behavior, so they
PASS today). The sibling *requirement* file (``test_barge_requirement.py``)
carries the strict-xfail assertions that flip to FAILURE the moment a fix makes
a normal-volume talk-over cut -- that is the signal to revisit these baselines.

Per the foundation's VERIFIED findings, replaying the parsed inputs through a
fresh ``build_live_dtd()`` reproduces all 204 ``fired`` verdicts EXACTLY, but
the recomputed ``.last_D`` drifts (the live chart carried pre-trace EWMA state).
We therefore assert on the ``decide()`` boolean / ``.last_decided`` ONLY and
NEVER on ``.last_D`` -- see the comment in
``test_live_dtd_reproduces_every_recorded_fired_verdict``.
"""
from __future__ import annotations

from tests import barge_fixtures as bf


def test_trace_parses_to_204_frames_with_9_fires():
    """The fixture loader yields the ground-truth shape of the recorded failure.

    Guards the loader + pins the recorded shape so a corrupted/truncated trace
    fails loudly here, before any logic assertion downstream.
    """
    frames = bf.load_trace_frames()

    assert len(frames) == 204
    assert sum(f.exp_fired for f in frames) == 9

    parsed_fire_indices = tuple(f.idx for f in frames if f.exp_fired)
    assert parsed_fire_indices == bf.LIVE_FIRE_INDICES


def test_live_dtd_reproduces_every_recorded_fired_verdict():
    """Replaying the 204 triples through the REAL AdaptiveDTD reproduces every
    recorded ``fired`` verdict EXACTLY (204/204).

    This is the deterministic, audio-free reproduction of the decision layer and
    the regression baseline for ``AdaptiveDTD.decide``: any change to the decide
    math that perturbs a per-frame verdict turns this red.

    NOTE: we assert on the ``decide()`` boolean (== ``dtd.last_decided``) ONLY.
    We deliberately do NOT assert ``dtd.last_D`` -- the live charts carried
    pre-trace EWMA state, so the recomputed D drifts (up to ~4.3) from the logged
    D even though the *verdict* matches exactly. Pinning D would false-fail.

    The frames are replayed SEQUENTIALLY through a single detector (the charts
    are stateful / warm up over the stream), exactly as the live capture loop saw
    them -- not in isolation.
    """
    frames = bf.load_trace_frames()
    dtd = bf.build_live_dtd()

    mismatches = []
    for f in frames:
        decided = dtd.decide(f.raw, f.resid, f.incoh)  # REAL AdaptiveDTD.decide
        # decide() return is the canonical verdict; .last_decided mirrors it.
        assert dtd.last_decided == decided
        if decided != f.exp_fired:
            mismatches.append((f.idx, decided, f.exp_fired))

    assert not mismatches, f"decide() verdict drifted from recorded: {mismatches}"


def test_full_chain_replay_reproduces_the_live_miss_on_normal_talkover():
    """The full chain reproduces the EXACT live miss: the DTD fires 9x, yet the
    only converted cut lands on the turn-3 SHOUT -- the turn-2 normal-volume
    talk-over never produces an ``on_barge_in``.

    Reproduces the bug deterministically: the 9 intermittent DTD fires + the
    ``voiced_run *= 0.5`` decay between fires never let voiced_run reach the 0.3s
    threshold on the normal-volume burst, so only the escalating shout breaks
    through. PASSES today because it ASSERTS the bug exists (documents current
    behavior); its sibling requirement test xfails on the fix.

    ``reset_latch_per_turn=True`` models the silent->speaking re-arm at each turn
    boundary -- i.e. even with the latch generously re-armed every turn, the
    normal-volume talk-over still does not cut.
    """
    frames = bf.load_trace_frames()

    result = bf.run_frames(
        frames,
        bf.build_live_dtd(),
        vad_speech=True,
        reset_latch_per_turn=True,
        params=bf.LIVE_INTEGRATOR_PARAMS,
    )

    # The REAL DTD fires 9 times across the trace.
    assert result.dtd_fire_count == 9

    fire_indices = [idx for idx, _ in result.fires]

    # The turn-2 NORMAL-volume talk-over (idx 10..14) produces NO cut -- the miss.
    assert not any(10 <= idx <= 14 for idx in fire_indices), (
        f"turn-2 normal talk-over unexpectedly cut: {result.fires}"
    )

    # The only cut lands in the turn-3 SHOUT window (idx 185..196).
    assert fire_indices, "expected at least one cut on the shout"
    assert all(185 <= idx <= 196 for idx in fire_indices), (
        f"unexpected cut outside the shout window: {result.fires}"
    )

    # The shout fire's latency is computed and present.
    assert result.first_fire_index is not None
    assert result.first_fire_latency_sec is not None


def test_latch_never_reset_collapses_all_turns_to_one_fire():
    """With the latch NEVER reset, the whole 204-frame trace collapses to exactly
    ONE ``on_barge_in`` -- at idx 196, the maximum shout.

    Pins the secondary latch-collapse failure mode: once the one-per-run latch
    sets, every later talk-over (turn-2 normal AND the second rejected attempt)
    is starved. A fix that re-arms the latch per turn flips this expectation,
    which is how the requirement file's xfail then surfaces.
    """
    frames = bf.load_trace_frames()

    result = bf.run_frames(frames, bf.build_live_dtd(), reset_latch_per_turn=False)

    fire_indices = [idx for idx, _ in result.fires]
    assert fire_indices == [196], (
        f"expected exactly one fire at idx 196, got {result.fires}"
    )
    assert result.first_fire_index == 196
