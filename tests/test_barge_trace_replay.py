"""Trace-replay reproduction of the recorded open-speaker barge-in failure.

Owner: agent-A-trace-replay. Tier 0 (no marker): pure stdlib + the parsed DTD
trace -- NO sound card, NO model, NO audio deps. Every assertion drives the
REAL ``core.engines._dtd.AdaptiveDTD.decide`` (and the mirrored capture-loop
integrator) over the recorded ``run-20260609-203236`` triples via the shared
``tests.barge_fixtures`` helper -- nothing is reimplemented here.

WHAT THIS FILE PINS
-------------------
The recorded decision layer AND its post-fix conversion. Replaying the parsed
inputs through a fresh ``build_live_dtd()`` reproduces all 204 ``fired`` verdicts
EXACTLY -- the ``AdaptiveDTD.decide`` math is unchanged -- which
``test_live_dtd_reproduces_every_recorded_fired_verdict`` pins as the regression
baseline. The FULL chain (``run_frames`` -> REAL ``AdaptiveDTD.decide`` + REAL
``BargeSustain``) is then asserted to NOW cut on the normal-volume talk-over: the
windowed sustain (the 2026-06-09 fix) converts the intermittent turn-2 DTD fires
(idx 10, 11, _, _, 14) into a cut at idx 11, where the old leaky ``voiced_run *=
0.5`` accumulator needed the turn-3 shout (raw=0.0704). The owner-requirement
guards live in ``test_barge_requirement.py``; echo-safety in
``test_barge_echo_must_not_fire.py``.

Per the foundation's VERIFIED findings the recomputed ``.last_D`` drifts (the live
chart carried pre-trace EWMA state), so we assert on the ``decide()`` boolean /
``.last_decided`` ONLY and NEVER on ``.last_D`` -- see the comment in
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
    # legacy=True: the recorded session ran the PRE-2026-06-10 chart math (no
    # persistence/z-freeze/robust seed), so the fidelity pin must construct
    # that detector. The SHIPPED detector's behavior over this trace is owned
    # by the requirement/echo-safety tests, not by this parse-fidelity pin.
    dtd = bf.build_live_dtd(legacy=True)

    mismatches = []
    for f in frames:
        decided = dtd.decide(f.raw, f.resid, f.incoh)  # REAL AdaptiveDTD.decide
        # decide() return is the canonical verdict; .last_decided mirrors it.
        assert dtd.last_decided == decided
        if decided != f.exp_fired:
            mismatches.append((f.idx, decided, f.exp_fired))

    assert not mismatches, f"decide() verdict drifted from recorded: {mismatches}"


def test_full_chain_replay_now_cuts_on_the_normal_talkover():
    """The full SHIPPED chain cuts on the normal-volume talk-over -- at its onset.

    Drives the highest-fidelity audio-free seam (``run_frames_engine``: REAL
    ``_barge_in_fire_eligible`` -> ``_looks_like_user`` with the residual-floor
    gate + REAL ``BargeSustain``) over the whole recorded run, with the
    silent->speaking re-arm modeled at the verified turn boundaries. The owner
    requirement this pins: the turn-2 NORMAL-volume talk-over cuts (live it
    never did; only the turn-3 shout broke through), and turn-3 cuts at its
    ONSET, not the raw=0.0704 shout peak.

    NB this intentionally does NOT pin the per-frame decide() count: the
    2026-06-10 anti-contamination charts refuse to absorb the talk-overs, so
    the decide layer fires MORE than the recorded 9 on this pre-normalization
    trace -- the downstream floor gate + latch own the behavior, and the
    behavior is what this test asserts.
    """
    frames = bf.frames_with_turn_starts(bf.load_trace_frames())

    result = bf.run_frames_engine(frames, vad_speech=True)

    fire_indices = [idx for idx, _ in result.fires]

    # The turn-2 NORMAL-volume talk-over (idx 10..14) now produces a cut.
    assert any(10 <= idx <= 14 for idx in fire_indices), (
        f"normal-volume talk-over still did not cut: {result.fires}"
    )
    # The FIRST cut is the first real talk-over (turn-2), not the later shout.
    assert result.first_fire_index in range(10, 15), (
        f"first cut at idx {result.first_fire_index}, expected the turn-2 burst"
    )
    # The turn-3 talk-over also cuts -- at its ONSET (idx 188/189), not the
    # idx-196 raw=0.0704 shout peak.
    turn3_cuts = [idx for idx in fire_indices if 185 <= idx <= 196]
    assert turn3_cuts and min(turn3_cuts) <= 189, (
        f"turn-3 cut waited for the shout escalation: {turn3_cuts}"
    )


def test_latch_never_reset_fires_once_on_the_first_real_talkover():
    """With the latch NEVER reset, the whole 204-frame trace collapses to exactly
    ONE ``on_barge_in`` -- now at idx 11, the FIRST real (turn-2 normal-volume)
    talk-over.

    Before the fix the single cut was the idx-196 shout (the normal talk-over
    never converted, so the latch only tripped on the shout). With the windowed
    sustain the normal talk-over converts first, so the one-per-run latch trips at
    idx 11 and starves every later talk-over until the next speaking run -- pinning
    that the latch still caps a run to ONE interrupt.
    """
    frames = bf.load_trace_frames()

    result = bf.run_frames(frames, bf.build_live_dtd(), reset_latch_per_turn=False)

    fire_indices = [idx for idx, _ in result.fires]
    assert fire_indices == [11], (
        f"expected exactly one fire at idx 11 (first real talk-over), got {result.fires}"
    )
    assert result.first_fire_index == 11
