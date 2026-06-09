"""Echo-safety guardrails: the assistant must NOT self-interrupt on its own TTS.

Owner: agent-C-echo-safety. Tier 0 (no marker): pure stdlib + numpy + the parsed
DTD trace -- NO sound card, NO model, NO audio deps. Every assertion drives the
REAL ``core.engines._dtd.AdaptiveDTD.decide`` (and the mirrored capture-loop
integrator) over the recorded ``run-20260609-203236`` echo floor via the shared
``tests.barge_fixtures`` helper -- nothing is reimplemented here.

WHAT THIS FILE PINS
-------------------
The *other half* of the CLAUDE.md HARD REQUIREMENT: while a normal-volume
talk-over MUST cut (pinned by the strict-xfails in ``test_barge_requirement.py``),
the assistant must NEVER self-interrupt on its own TTS leaking into the open
laptop mic. These are the **non-regressable safety properties** -- they PASS
today and a barge-in *fix* (lowering ``K``, reweighting features, a stickier
integrator) MUST NOT break them while making normal talk-over fire. If any of
these flips red, the fix has reintroduced the self-interrupt failure mode.

REALITY THE ASSERTIONS RESPECT
------------------------------
The foundation annotates a frame ``echo_only`` iff it sits outside the two narrow
human-talk-over windows (turn-2 idx 10..14, turn-3 idx 185..196). One such
``echo_only``-annotated frame -- idx 34 (raw=0.011, resid=0.0104, ~1s after the
turn-2 burst) -- is in fact a residual acoustic event the live detector scored
``fired=True`` (it is one of the 9 ``LIVE_FIRE_INDICES``). So "no echo_only frame
ever fires" is NOT true of the raw annotation. What IS true, and what these tests
pin, is the genuinely-clean playback floor: every echo-only frame the LIVE
detector itself scored ``fired=False`` (the 186-frame TTS-playback floor) returns
False from the REAL ``decide()``, and the full chain produces NO ``on_barge_in``
cut on any echo-only frame -- the only recorded cut was the turn-3 shout.

The per-device charts calibrate from the *interleaved* echo+barge sequence the
live run actually saw, so the faithful replay feeds the FULL trace in order and
asserts the echo-floor property over it; feeding echo-only frames in isolation
would mis-calibrate the chart (it would never see the excursions that, live, broke
the confirmation run) and is therefore NOT how these tests drive the detector.

Per the foundation's VERIFIED finding the recomputed ``.last_D`` drifts from the
logged D (pre-trace EWMA state), so we assert on the ``decide()`` boolean /
``.last_decided`` -- NEVER on ``.last_D``.
"""
from __future__ import annotations

from tests import barge_fixtures as bf


def test_echo_only_frames_never_fire_dtd():
    """The clean TTS-playback floor never fires the REAL ``AdaptiveDTD.decide``.

    Drives ``build_live_dtd().decide(...)`` over the recorded run in its real
    interleaved order (the order the live per-device charts calibrated from) and
    checks that every echo-only frame the LIVE detector scored ``fired=False``
    -- the 186-frame clean playback floor -- also returns False from the REAL
    detector, with ``.last_decided`` False on that frame.

    Pins the "never self-interrupt on own TTS" half of the requirement at the
    DECISION layer over the REAL recorded echo floor. A future tuning change that
    lowers ``K`` or reweights features and starts firing on clean echo flips this
    red. Asserts on the ``decide()`` bool / ``.last_decided`` ONLY, never on
    ``.last_D`` (it drifts -- see module docstring).
    """
    frames = bf.load_trace_frames()
    dtd = bf.build_live_dtd()

    # The clean playback floor = echo-only-annotated frames the LIVE detector
    # itself scored fired=False (excludes idx 34, the borderline residual event
    # that fired both live and in replay -- see module docstring).
    clean_floor = {
        f.idx
        for f in bf.echo_only_frames(frames)
        if not f.exp_fired
    }
    assert clean_floor, "expected a clean echo floor in the recorded trace"

    violations = []
    for f in frames:  # full trace, real order -> faithful chart calibration
        decided = dtd.decide(f.raw, f.resid, f.incoh)  # REAL AdaptiveDTD.decide
        if f.idx in clean_floor and (decided or dtd.last_decided):
            violations.append((f.idx, f.raw, f.resid, f.incoh, decided))

    assert violations == [], (
        "clean echo-floor frames self-interrupted (decide()==True): "
        f"{violations}"
    )


def test_echo_only_run_produces_no_on_barge_in():
    """Full-chain echo safety: no ``on_barge_in`` cut ever lands on echo.

    Walks the REAL chain (``run_frames`` -> REAL ``AdaptiveDTD.decide`` + the
    mirrored leaky integrator/latch) over the entire recorded run, including the
    long ~85s TTS-playback stretch, and checks that NO cut lands on an echo-only
    frame -- the only recorded cut is the turn-3 shout (idx 196), exactly the
    live behavior (summary.json: the assistant never self-interrupted on echo).

    Even with the leaky integrator, echo-only never accumulates ``voiced_run`` to
    a cut. Must PASS today and stay green: this is the non-regressable safety
    property a barge fix must not break while making normal talk-over fire.
    """
    frames = bf.load_trace_frames()
    echo_idx = {f.idx for f in bf.echo_only_frames(frames)}

    # reset_latch_per_turn=True re-arms the latch at each turn boundary (the
    # silent->speaking re-arm) -- the most permissive setting for the integrator,
    # so any spurious echo cut would show up here.
    result = bf.run_frames(
        frames,
        bf.build_live_dtd(),
        vad_speech=True,
        reset_latch_per_turn=True,
        params=bf.LIVE_INTEGRATOR_PARAMS,
    )

    echo_cuts = [(idx, t) for (idx, t) in result.fires if idx in echo_idx]
    assert echo_cuts == [], (
        f"on_barge_in cut landed on echo-only frame(s): {echo_cuts} "
        f"(all fires: {result.fires})"
    )


def test_steady_echo_baseline_does_not_drift_up_to_admit_quiet_talkover():
    """Steady echo learning must not raise its own bar past a real talk-over.

    Feeds the long echo-floor stretch (the real run up to just before the turn-3
    shout) through ``build_live_dtd()``, captures the per-feature chart state,
    then checks that a normal-volume talk-over residual (the turn-2 levels,
    raw 0.0024-0.0068 / resid up to 0.0041) still elevates ``z_resid`` well above
    zero against that learned floor -- i.e. the echo-update stayed bounded and did
    NOT silently inflate the floor past a genuine talk-over.

    Pins the chart-stability safety property: an over-eager echo-update that
    raised the floor would make the requirement fix impossible (a real talk-over
    would no longer stand out). Passes today.
    """
    frames = bf.load_trace_frames()
    dtd = bf.build_live_dtd()

    # Learn the echo floor exactly as the live run did: feed the real interleaved
    # sequence up to just before the turn-3 shout (idx <= 184). This includes the
    # long playback stretch the chart calibrated against.
    for f in frames:
        if f.idx <= 184:
            dtd.decide(f.raw, f.resid, f.incoh)  # REAL AdaptiveDTD.decide

    # The learned echo floor must stay bounded (low mean, finite sigma).
    assert dtd._resid.mean < 0.001, (
        f"echo resid floor drifted up to {dtd._resid.mean:.5f} -- echo learning "
        "inflated its own bar"
    )

    # A normal-volume talk-over (turn-2 levels) must still elevate z_resid > 0
    # against the learned floor -- the floor did not swallow it.
    turn2 = bf.talkover_frames(frames)["turn2_normal"]
    talkover_resid = max(f.resid for f in turn2)  # 0.0041, the turn-2 max
    z_resid = dtd._resid.z(talkover_resid)
    assert z_resid > 0.0, (
        f"normal-volume talk-over resid {talkover_resid} did not elevate "
        f"z_resid above the learned echo floor (z_resid={z_resid:.3f}); the "
        "echo baseline drifted up and swallowed a real talk-over"
    )
    # It should stand out clearly, not marginally -- a comfortable margin.
    assert z_resid >= 1.0, (
        f"talk-over z_resid only {z_resid:.3f}; floor crept too close to a real "
        "talk-over"
    )


def test_single_feature_spike_on_echo_does_not_fire_under_live_weights():
    """A lone coherence spike cannot self-interrupt under the SHIPPED weights.

    The live weight vector is ``(w_raw, w_resid, w_coh) = (0.2, 1.0, 0.0)`` -- the
    incoherent-fraction feature is weighted ZERO. This structurally rejects the
    nonlinear-echo coherence-spike self-interrupt seen in earlier coherence-only
    designs. We warm ``build_live_dtd()`` on the recorded echo floor, then feed
    blocks where ONLY ``incoherent_fraction`` is elevated (raw/resid held at the
    learned echo floor) and confirm the REAL ``decide`` never fires.

    Pins that ``w_coh == 0.0`` makes a lone z_coh excursion contribute nothing to
    the fused statistic ``D``. Flips red if someone re-enables ``w_coh`` without
    re-validating echo safety.
    """
    frames = bf.load_trace_frames()
    dtd = bf.build_live_dtd()

    # Precondition: the live config really does zero the coherence weight.
    assert dtd.w_coh == 0.0, (
        f"live weights changed (w_coh={dtd.w_coh}); this test assumes the SHIPPED "
        "weights (0.2, 1.0, 0.0) -- re-validate echo safety before re-enabling w_coh"
    )

    # Warm the per-device charts on the real early echo floor (real order, past
    # the warmup window).
    for f in frames:
        if f.idx <= 30 and f.annotation == "echo_only":
            dtd.decide(f.raw, f.resid, f.incoh)  # REAL AdaptiveDTD.decide

    # Hold raw/resid at the learned echo-floor means; spike ONLY incoherence high.
    floor_raw = dtd._raw.mean
    floor_resid = dtd._resid.mean

    fired = []
    for _ in range(10):  # sustain the lone coherence spike for many blocks
        decided = dtd.decide(floor_raw, floor_resid, 0.99)  # REAL decide
        # z_coh is genuinely elevated, but w_coh==0 zeroes its contribution.
        assert dtd.last_z_coh >= 0.0
        if decided:
            fired.append((dtd.last_D, dtd.last_z_coh))

    assert fired == [], (
        "a lone incoherent-fraction spike self-interrupted despite w_coh==0: "
        f"{fired} -- the nonlinear-echo coherence-spike failure mode is back"
    )
