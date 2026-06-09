"""Gate-conversion seam tests for the recorded open-speaker barge-in failure.

Owner: agent-D-gate-conversion. This file owns the seam between the REAL
``AdaptiveDTD`` and the downstream capture-loop latch/integrator inside the REAL
``SherpaOnnxEngine`` -- i.e. how a DTD *fire* is (or is not) converted into an
actual cut. Sibling files own the trace-replay reproduction
(``test_barge_trace_replay.py``), the owner requirement xfails
(``test_barge_requirement.py``) and echo safety (``test_barge_echo_must_not_fire.py``);
this file does not touch them.

Everything here drives the REAL code, never a reimplementation:

* ``tests.barge_fixtures.live_engine_with_dtd()`` builds a REAL
  ``SherpaOnnxEngine`` (via ``object.__new__`` to skip ONNX model loading) wired
  for the live AEC+coherence barge path, so the REAL ``_looks_like_user``
  (sherpa.py:1929) and ``_barge_in_fire_eligible`` (sherpa.py:2120) run
  unmodified.
* The detector under each ``decide`` is the REAL ``AdaptiveDTD``
  (``build_live_dtd()``); the fake coherence only *supplies* the
  incoherent-fraction feature (the live wiring -- coherence produces the
  feature, the DTD decides), and ``make_block(rms)`` produces a real numpy block
  whose ``rms()`` is exactly a recorded frame's raw/resid level.

Recorded failure being pinned: ``run-20260609-203236`` -- the owner talked over
the assistant on the bare laptop speaker; the DTD fired on the talk-over but the
turn-2 NORMAL-volume burst (raw 0.0024-0.0068) never converted to a cut (only
the turn-3 SHOUT did). See CLAUDE.md HARD REQUIREMENT (open-speaker barge-in, no
headphones / no shout).

Tier 0: pure stdlib + numpy via ``make_block``; no sound card, no models, no
audio fixtures. ``load_trace_frames()`` reads the committed JSON parse.
"""
from __future__ import annotations

from tests.barge_fixtures import (
    LIVE_FIRE_INDICES,
    build_live_sustain,
    live_engine_with_dtd,
    load_trace_frames,
    make_block,
    talkover_frames,
)


def _drive_looks_like_user(engine, frame):
    """Run the REAL ``_looks_like_user`` over one recorded frame.

    Mirrors the live call site (sherpa.py:1971-1989): the coherence detector is
    consulted for its incoherent-fraction feature, then ``samples`` (the post-AEC
    residual block) and ``mic_raw`` (the raw pre-AEC block) are fed in. We feed
    real numpy blocks whose ``rms()`` equals this frame's ``resid``/``raw`` so the
    REAL ``rms(...)`` inside the gate measures back to the recorded levels.
    """
    engine._fake_coherence.last_incoherent_fraction = frame.incoh
    return engine._looks_like_user(make_block(frame.resid), mic_raw=make_block(frame.raw))


def _warm_engine_to(engine, frames):
    """Replay ``frames`` through the REAL ``_looks_like_user`` so the engine's
    fresh ``AdaptiveDTD`` warms its per-device charts exactly as it did live.

    A freshly built engine's DTD has ``warmup_frames=5`` and an empty echo chart;
    feeding the recorded echo-floor (and earlier) frames first reproduces the
    live chart state so the subsequent fire-frame verdict matches the recording.
    Returns the per-frame verdicts (so a caller can sanity-check warm-up).
    """
    return [_drive_looks_like_user(engine, f) for f in frames]


def test_looks_like_user_routes_through_dtd_path_when_aec_and_coherence_on():
    """PATH A (DTD) is taken and faithfully converts ``decide()`` -> bool.

    Drives the REAL ``_looks_like_user`` (sherpa.py:1929) over the recorded
    trace: on a known DTD-fire frame (the turn-3 shout, idx 195-196) it returns
    True; on an echo-floor frame it returns False. With ``_aec`` non-None and a
    coherence detector present, the gate delegates to ``AdaptiveDTD.decide`` and
    returns its verdict -- so reproducing the exact recorded fire indices proves
    PATH A is the path under test.
    """
    frames = load_trace_frames()
    engine = live_engine_with_dtd()

    # Replay the WHOLE trace through the REAL gate; the engine's DTD warms its
    # charts identically to the live run, so its fires must match the ground truth.
    verdicts = _warm_engine_to(engine, frames)
    fired_idx = tuple(f.idx for f, v in zip(frames, verdicts) if v)

    # The REAL _looks_like_user reproduced the recorded fire set EXACTLY: PATH A
    # (DTD) decided, and the bool was converted faithfully.
    assert fired_idx == LIVE_FIRE_INDICES, (
        f"_looks_like_user fires {fired_idx} != recorded {LIVE_FIRE_INDICES}"
    )

    by_idx = {f.idx: v for f, v in zip(frames, verdicts)}
    # A known DTD-fire frame (turn-3 shout) -> True.
    assert by_idx[195] is True and by_idx[196] is True
    # An echo-floor frame -> False (no self-interrupt on the assistant's own TTS).
    assert by_idx[15] is False and by_idx[197] is False


def test_fire_eligible_blocked_by_latch_then_real_barge_starves():
    """ROOT CAUSE: the one-per-run latch gates ELIGIBILITY, not just the action.

    Drives the REAL ``_barge_in_fire_eligible`` (sherpa.py:2120) with
    ``_barge_in_fired_this_run`` toggled on, feeding the turn-2 normal-volume
    talk-over blocks. The latch is checked FIRST (line 2134), so every subsequent
    block returns False regardless of how strongly the DTD would fire -- the
    integrator never even gets to accumulate. Documents current behavior (passes
    today); the requirement-file xfails own the "this must not starve a real
    barge" assertion so they flip on the fix.
    """
    frames = load_trace_frames()
    t2 = talkover_frames(frames)["turn2_normal"]  # idx 10..14, normal volume

    engine = live_engine_with_dtd()
    # Warm the engine's DTD to the live state with the echo floor preceding turn-2.
    _warm_engine_to(engine, [f for f in frames if f.idx < t2[0].idx])

    # Latch already tripped earlier this speaking run.
    engine._barge_in_fired_this_run = True

    eligible = []
    for f in t2:
        engine._fake_coherence.last_incoherent_fraction = f.incoh
        eligible.append(
            engine._barge_in_fire_eligible(make_block(f.resid), make_block(f.raw))
        )

    # With the latch set, NOTHING is eligible -- even the blocks where the DTD
    # itself fires (idx 10, 11, 14). The latch starves the integrator entry.
    assert eligible == [False] * len(t2), (
        f"latched eligibility should be all-False, got {eligible}"
    )


def test_real_barge_must_convert_to_eligible_within_one_turn():
    """REQUIREMENT at the gate seam: a fresh-turn normal-volume talk-over converts
    eligible -> cut.

    Drives the REAL ``_barge_in_fire_eligible`` (sherpa.py:2120) over the turn-2
    normal-volume burst with a FRESH latch (a new speaking run), feeding each
    block's eligibility into the REAL ``BargeSustain`` -- the SAME windowed sustain
    the capture loop runs (built here via ``build_live_sustain()``). The
    requirement: enough eligible blocks land within the window to produce a cut.

    Before the fix the DTD fired intermittently on the normal-volume burst (idx
    10, 11 fire; 12, 13 miss; 14 fires) and the leaky ``voiced_run *= 0.5``
    accumulator never reached the threshold -- only the turn-3 shout broke through.
    The windowed sustain now cuts on the 2nd eligible block; this guards that the
    gate seam converts a real normal-volume talk-over into a cut.
    """
    frames = load_trace_frames()
    t2 = talkover_frames(frames)["turn2_normal"]  # idx 10..14, normal volume

    engine = live_engine_with_dtd()
    # Warm the engine's DTD to the live echo state (frames preceding turn-2).
    _warm_engine_to(engine, [f for f in frames if f.idx < t2[0].idx])
    # FRESH latch: a new speaking run, nothing fired yet.
    engine._barge_in_fired_this_run = False

    sustain = build_live_sustain()  # REAL core.engines._dtd.BargeSustain
    cut_produced = False
    for f in t2:
        engine._fake_coherence.last_incoherent_fraction = f.incoh
        # REAL eligibility seam (latch checked first, then VAD, then _looks_like_user).
        eligible = engine._barge_in_fire_eligible(make_block(f.resid), make_block(f.raw))
        if sustain.update(eligible):  # REAL BargeSustain windowed confirmation
            cut_produced = True
            break

    assert cut_produced, (
        "normal-volume talk-over must convert eligible->cut within the burst "
        f"(sustain last_count={sustain.last_count}, need={sustain.need_frames})"
    )


def test_refractory_and_suppress_windows_do_not_swallow_a_distinct_new_talkover():
    """Debounce/refractory windows must EXPIRE so a genuinely-new talk-over a full
    turn later is evaluated, not silently dropped.

    Drives the REAL ``_in_post_speaking_refractory`` (sherpa.py:2174) plus the
    suppress-window stamp (``_barge_in_suppressed_until``), reproducing the gate
    order at the live capture-loop site (sherpa.py:1397-1402): the windows are
    checked FIRST and, if active, reset ``voiced_run`` before eligibility is even
    consulted. The recorded turn-3 talk-over arrives ~43s after turn-2 ended --
    far beyond the live refractory_sec=0.5 + suppress_sec=0.5 combined window --
    so neither window may still be active when turn-3 begins. Passes today; guards
    against a fix that over-extends suppression and regresses multi-turn barge.
    """
    frames = load_trace_frames()
    by_idx = {f.idx: f for f in frames}
    t2_end_t = by_idx[14].t_sec     # last turn-2 talk-over frame
    t3_start_t = by_idx[185].t_sec  # first turn-3 talk-over frame
    gap = t3_start_t - t2_end_t

    engine = live_engine_with_dtd()
    refractory_sec = engine.config.barge_in_refractory_sec
    suppress_sec = engine.config.barge_in_suppress_sec

    # Sanity: the recorded gap really is a full turn, well past both 0.5s windows.
    assert gap > refractory_sec + suppress_sec, (
        f"recorded turn2->turn3 gap {gap:.2f}s should exceed "
        f"refractory+suppress={refractory_sec + suppress_sec:.2f}s"
    )

    # Model turn-2's speaking end + a barge-debounce stamp on a single monotonic
    # baseline; query at turn-3's arrival (a full turn later).
    base = 1000.0  # arbitrary monotonic stamp for turn-2's _speaking->clear
    engine._last_speaking_end = base
    engine._barge_in_suppressed_until = base + suppress_sec
    now_at_t3 = base + gap

    # Immediately after turn-2 the refractory window IS active (tail suppression).
    assert engine._in_post_speaking_refractory(base + 0.1) is True
    # A full turn later, the refractory window has expired...
    assert engine._in_post_speaking_refractory(now_at_t3) is False
    # ...and the suppress window has expired too -- so the live gate's first check
    # (sherpa.py:1397-1399) does NOT swallow turn-3; eligibility may evaluate.
    assert now_at_t3 >= engine._barge_in_suppressed_until
    swallowed = (
        now_at_t3 < engine._barge_in_suppressed_until
        or engine._in_post_speaking_refractory(now_at_t3)
    )
    assert swallowed is False, "a full-turn-later talk-over must not be suppressed"
