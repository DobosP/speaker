"""Open-speaker SELF-INTERRUPT guard (run-20260609-234435, the 2026-06-10 fix).

CLAUDE.md HARD REQUIREMENT, the "never self-interrupt" half: the assistant must
NOT cut its own TTS on echo leaking into the open laptop mic, yet a real
talk-over MUST still cut promptly (no shout, no headphones). The recorded
failure ``run-20260609-234435`` cut the assistant off TWICE on reply-onset echo
(barge-in detected at 23:46:54.969 and 23:46:57.255): on a starved laptop mic
the per-feature z-scores BLOAT against a near-silent reply-onset baseline, so 2
echo transients within a 0.5s window tripped ``BargeSustain(need=2)`` even
though their absolute residual barely lifted off the echo floor.

THE FIX (2026-06-10), driven REAL (never reimplemented) here:
  1. A residual-floor gate on the DTD barge path (``_looks_like_user``): a DTD
     fire only counts when the post-AEC residual stands ``dtd_residual_floor_
     margin_db`` (default 12 dB) above the LEARNED residual echo floor
     (``_playback_floor_rms``). RELATIVE to the per-reply floor, never a fixed
     energy -- device-adaptive, off-switchable (margin 0 restores pre-fix).
  2. Re-arm per reply at the silent->speaking transition: clear the
     ``BargeSustain`` window and re-bootstrap the learned floors so the gate
     measures THIS reply's echo, not a stale prior one.

These tests drive the REAL ``SherpaOnnxEngine._barge_in_fire_eligible`` ->
``_looks_like_user`` (the gate), the REAL ``_update_playback_floor`` and the REAL
``core.engines._dtd.BargeSustain`` via ``barge_fixtures.run_frames_engine`` over
the parsed 234435 trace. Tier 0: no sound card, no model, no audio deps.

WHY THE TWO RECORDED RUNS RECONCILE (the discriminator):
  echo self-interrupt fires (234435): resid 0.0008-0.0018  (at the echo floor)
  real talk-over fires   (203236):    resid 0.0024-0.0128  (well above the floor)
The RAW mic does NOT separate them (echo 0.0026-0.0058 overlaps talk-over
0.0024-0.0068) -- only the residual does, which is why the gate keys off the
residual floor. See ``test_barge_requirement.py`` for the talk-over-MUST-cut half.
"""
from __future__ import annotations

from tests import barge_fixtures as bf


def test_self_interrupt_replies_produce_no_cut_through_real_engine():
    """Neither reply-onset echo burst cuts the assistant after the fix.

    Drives each recorded reply (B: idx 8..18, fires at 15+18; C: idx 19..27,
    fires at 26+27) through the REAL engine gate + REAL BargeSustain. Live, each
    produced a ``barge-in detected``; with the residual-floor gate the DTD still
    *fires* on the echo transients but they no longer count as eligible, so the
    windowed sustain never reaches a cut. This is the non-regressable safety
    property the fix delivers: flip it red and the self-interrupt is back.
    """
    frames = bf.load_self_interrupt_frames()
    windows = bf.self_interrupt_windows(frames)

    for name, reply in windows.items():
        result = bf.run_frames_engine(reply)
        # The DTD still fires on the echo transients (its decide math is unchanged)
        # -- the gate rejects them downstream, so NO cut lands.
        assert result.dtd_fire_count >= 1, (
            f"{name}: expected the recorded echo transients to still fire the DTD"
        )
        assert result.fires == [], (
            f"{name}: the assistant SELF-INTERRUPTED on its own reply-onset echo "
            f"(cuts={result.fires}) -- the 234435 failure is back"
        )


def test_full_self_interrupt_trace_produces_no_cut():
    """End-to-end: the WHOLE 234435 run (all replies, real order) yields NO cut.

    Walks every frame through the REAL engine chain with per-reply re-arm (the
    ``reply_start`` tags parsed from the trace's ``speaking:`` markers). Live this
    produced two cuts; after the fix it must produce zero -- the assistant lets
    its own replies finish.
    """
    frames = bf.load_self_interrupt_frames()
    result = bf.run_frames_engine(frames)
    assert result.fires == [], (
        f"self-interrupt cut(s) on echo-only frames: {result.fires}"
    )


def test_residual_floor_gate_is_off_switchable_reproduces_the_live_self_interrupt():
    """With the gate DISABLED (margin 0), the live self-interrupt RE-APPEARS.

    Pins two things at once: (a) the gate is genuinely off-switchable
    (``dtd_residual_floor_margin_db == 0`` restores pre-fix behaviour), and (b)
    the gate -- not some incidental driver detail -- is what prevents the cut. The
    same real chain, same frames, gate off: reply B's 2 echo fires (idx 15, 18)
    within the 0.5s window trip BargeSustain and cut, exactly the recorded failure.
    """
    frames = bf.load_self_interrupt_frames()
    reply_b = bf.self_interrupt_windows(frames)["reply_B"]

    # Gate ON (default 12 dB) -> no cut; gate OFF (0 dB) -> the live cut returns.
    on = bf.run_frames_engine(reply_b, residual_floor_margin_db=12.0)
    off = bf.run_frames_engine(reply_b, residual_floor_margin_db=0.0)

    assert on.fires == [], f"gate ON should suppress the self-interrupt, got {on.fires}"
    assert off.fires, (
        "gate OFF should reproduce the live self-interrupt (proving the gate is "
        "what fixes it), but no cut fired"
    )
    # The reproduced cut lands on reply B's 2nd echo fire (idx 18, the recorded
    # 23:46:54.969 barge-in), not later.
    assert off.first_fire_index == 18, (
        f"expected the reproduced self-interrupt at idx 18 (the recorded cut), got "
        f"{off.first_fire_index}"
    )


def test_residual_floor_gate_unit_rejects_at_floor_admits_above():
    """Unit-pin the gate inside the REAL ``_looks_like_user`` directly.

    Warm the engine's DTD on a steady residual echo floor, freeze the learned
    floor, then probe a single block whose residual sits AT the floor vs one that
    stands well above it. The DTD ``decide`` would fire on both (the warmup
    baseline makes the small block a z-outlier), but the residual-floor gate must
    reject the at-floor block and admit the above-floor one -- the exact
    discriminator that separates the 234435 echo from the 203236 talk-over.
    """
    eng = bf.live_engine_with_dtd()
    eng.config.dtd_residual_floor_margin_db = 12.0  # 12 dB == a 3.98x level ratio

    # Establish a residual echo floor of ~0.0008 (the 203236 playback floor level)
    # on both the chart and the engine's learned floor.
    floor = 0.0008
    for _ in range(8):
        eng._update_playback_floor(floor)
        eng._looks_like_user(bf.make_block(floor), bf.make_block(floor))
    learned = eng._playback_floor_rms
    assert learned > 0.0

    # An at-floor residual transient (234435 echo level, resid 0.0018 ~= 2.2x the
    # ~0.0008 floor = ~7 dB, UNDER the 12 dB bar): the DTD fires on the z-outlier,
    # but the gate rejects it.
    at_floor = eng._looks_like_user(bf.make_block(0.0026), bf.make_block(0.0018))
    # An above-floor residual (203236 turn-3 pre-shout level, resid 0.0105 ~= 13x
    # the floor = ~22 dB, well over the bar): a real talk-over, admitted.
    above = eng._looks_like_user(bf.make_block(0.0144), bf.make_block(0.0105))

    assert at_floor is False, (
        "an at-echo-floor residual transient passed the gate -- it would "
        "self-interrupt"
    )
    assert above is True, (
        "a well-above-floor residual (real talk-over level) was wrongly gated out "
        "-- the fix would reintroduce 'needs a shout'"
    )


def test_real_talkover_still_cuts_through_the_same_engine_gate():
    """The SAME engine gate that suppresses the 234435 echo still CUTS the 203236
    normal-volume talk-over -- the reconciliation, proven on identical real code.

    Seeds the engine on the 203236 echo floor (idx 0..9) then feeds the turn-2
    NORMAL-volume talk-over (idx 10..14, resid 0.0024-0.0041). The residual-floor
    gate admits these (they stand well above the learned floor), so the windowed
    sustain cuts -- WITHOUT a shout. This is the other side of the HARD
    REQUIREMENT: the fix must not reintroduce "needs a shout".
    """
    frames = bf.load_trace_frames()
    echo_baseline = [f for f in frames if f.idx <= 9]
    turn2 = bf.talkover_frames(frames)["turn2_normal"]
    sequence = echo_baseline + turn2

    result = bf.run_frames_engine(sequence)

    assert result.first_fire_index is not None, (
        "normal-volume talk-over produced NO cut through the gated engine -- the "
        "fix reintroduced the 'needs a shout' failure"
    )
    assert result.first_fire_index in range(10, 15), (
        f"cut landed at idx {result.first_fire_index}, outside the turn-2 burst"
    )
    talkover_start = next(f.t_sec for f in frames if f.idx == 10)
    latency = result.first_fire_t_sec - talkover_start
    assert 0.0 <= latency <= 0.5, (
        f"normal-volume talk-over cut too late: {latency:.3f}s (bound 0.5s)"
    )
