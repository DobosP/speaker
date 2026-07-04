"""Open-speaker DOUBLE-TALK regression guard under the shipped ``open_speaker``
profile (always-on WebRTC APM that OWNS noise suppression -- ``_apm_owns_ns``).

THE MISS THIS LOCKS (the 2026-06-21 APM-landing review headline P1, corroborated
by the ``barge_stress`` subset tp_rate=0.50):

  The fused-z-score ``AdaptiveDTD`` weights the post-AEC RESIDUAL at 1.0 and gates
  a fire on the residual standing ``dtd_residual_floor_margin_db`` (12 dB) above
  the learned residual echo floor. That assumes the near-end USER survives in the
  residual -- true for a LINEAR canceller (nlms/dtln, headphones), which removes
  only the far-end reference. But ``open_speaker`` runs ``apm_always_on=true`` +
  noise suppression, so the APM's ML NS runs on EVERY block and ATTENUATES the
  near-end user in that residual during double-talk. The residual barely lifts on
  a real talk-over -> the weight-1.0 feature can't push ``D`` over ``K`` AND the
  12 dB floor gate rejects it -> the documented "user had to scream / 0 fired".

THE FIX (driven REAL here, never reimplemented): under ``_apm_owns_ns`` the DTD's
residual feature + its floor gate read the RAW pre-AEC/pre-NS mic (which still
carries the user) instead of the NS-suppressed ``samples`` -- see
``SherpaOnnxEngine._dtd_residual_level``. The raw-floor + z-score self-calibration
still STRUCTURALLY reject echo-only (no new self-interrupt), so the change only
restores the detector's sight of a genuine talk-over.

Tier 0: no sound card, no model, no ``livekit`` -- drives the REAL
``_looks_like_user`` gate + REAL ``core.engines._dtd.AdaptiveDTD`` + REAL floor
updates over ``make_block`` inputs whose rms equals a modelled level.

NOTE (not a live claim): the absolute 12 dB margin over the *raw* floor is the
inherited residual-floor value; whether it is the right operating point on the
bare open speaker still needs a LIVE mic A/B (talk-over MUST cut + no
self-interrupt return). What this guard pins is the STRUCTURAL routing -- that the
detector reads a signal which still contains the user under NS, instead of one NS
has erased.
"""
from __future__ import annotations

from tests import barge_fixtures as bf

# --- Modelled open-speaker double-talk levels (rms of a 0.1s block) ----------- #
# Steady RAW echo floor (pre-AEC/pre-NS): the open laptop speaker leaking into the
# bare mic. Loud and uncancelled on the raw tee.
ECHO_RAW = 0.012
# Post-AEC + post-NS residual ECHO floor: AEC3 cancels the echo and the APM's NS
# crushes what's left -> near silence.
ECHO_RESID_NS = 0.0015
# A NORMAL-volume talk-over (no shout). On the RAW mic the user adds clearly on top
# of the echo (~+14 dB over the raw floor) -- the user is plainly present there.
TALK_RAW = 0.060
# ...but in the post-AEC residual the APM's NS has ATTENUATED the user back down to
# barely above the residual floor (~+3 dB) -- the residual is BLIND to the user.
TALK_RESID_NS = 0.0022
# An echo-only block during playback (no user): raw sits at the echo floor.
ECHO_ONLY_RAW = 0.013
ECHO_ONLY_RESID_NS = 0.0016


def _open_speaker_engine(apm_owns_ns: bool):
    """A REAL DTD-wired engine in the ``open_speaker`` shape: APM owns NS when
    ``apm_owns_ns`` (always_on + suppresses_noise). The 12 dB residual-floor gate
    is the shipped default."""
    eng = bf.live_engine_with_dtd()
    eng._apm_owns_ns = bool(apm_owns_ns)
    # _resid_blind is what the DTD residual source/floor + coh_veto now key on
    # (covers APM-NS AND the DTLN spectral-masking canceller); == _apm_owns_ns here.
    eng._resid_blind = bool(apm_owns_ns)
    eng.config.dtd_residual_floor_margin_db = 12.0  # shipped default
    eng._dtd.new_run()
    return eng


def _warm_echo_only(eng, *, raw, resid_ns, n=12):
    """Calibrate the DTD charts + learn the echo floors on echo-only playback
    blocks, exactly as the capture loop does (sherpa.py): the ``observe_echo``
    learning tap feeds the residual chart through the SAME ``_dtd_residual_level``
    the gate uses, and both floors update per block."""
    for _ in range(n):
        resid_obs = eng._dtd_residual_level(bf.make_block(resid_ns), bf.make_block(raw))
        eng._dtd.observe_echo(raw, resid_obs, eng._fake_coherence.last_incoherent_fraction)
        eng._update_playback_floor(resid_ns)
        eng._update_raw_playback_floor(raw)


def _probe(eng, *, raw, resid_ns) -> bool:
    """One block through the REAL gate. ``samples`` is the post-AEC(+NS) residual
    block, ``mic_raw`` the raw pre-AEC block -- the exact two args the capture loop
    passes ``_barge_in_fire_eligible`` -> ``_looks_like_user``."""
    return eng._looks_like_user(bf.make_block(resid_ns), bf.make_block(raw))


# --------------------------------------------------------------------------- #
# 1. The routing decision, pinned directly (the one-line structural fix).
# --------------------------------------------------------------------------- #
def test_residual_feature_source_switches_on_apm_owns_ns():
    """``_dtd_residual_level`` reads the post-AEC residual for a linear canceller
    and the RAW mic when the always-on APM owns NS -- the whole fix in one call."""
    eng = _open_speaker_engine(apm_owns_ns=False)
    samples, mic_raw = bf.make_block(TALK_RESID_NS), bf.make_block(TALK_RAW)

    # Linear AEC (dtln/nlms/headphones): the user survives in the residual -> read it.
    assert abs(eng._dtd_residual_level(samples, mic_raw) - TALK_RESID_NS) < 1e-6

    # A masking canceller (APM-NS or DTLN): the residual is blinded -> read the raw
    # pre-NS mic instead. (_resid_blind is the runtime flag; == _apm_owns_ns for APM,
    # also True for DTLN via its suppresses_nearend capability.)
    eng._resid_blind = True
    assert abs(eng._dtd_residual_level(samples, mic_raw) - TALK_RAW) < 1e-6


# --------------------------------------------------------------------------- #
# 2. The miss vs the catch -- IDENTICAL physical block, gate verdict flips on the
#    routing alone. This is the regression guard: flip the fix and (B) goes red.
# --------------------------------------------------------------------------- #
def test_ns_blinded_residual_misses_a_real_talkover():
    """PRE-FIX routing (residual feature read from the NS-suppressed ``samples``):
    a normal-volume talk-over is REJECTED even though the user is plainly present
    on the raw mic (+14 dB). This reproduces the documented open-speaker miss --
    the residual the detector trusts has had the user erased by NS."""
    eng = _open_speaker_engine(apm_owns_ns=False)  # reads the post-NS residual
    _warm_echo_only(eng, raw=ECHO_RAW, resid_ns=ECHO_RESID_NS)

    fired = _probe(eng, raw=TALK_RAW, resid_ns=TALK_RESID_NS)
    assert fired is False, (
        "NS-blinded residual let a real talk-over through -- the modelled miss is "
        "not being reproduced, so the guard below would not actually be guarding"
    )


def test_apm_owns_ns_recovers_the_talkover_through_the_real_gate():
    """POST-FIX (``_apm_owns_ns``): the SAME block now FIRES, because the residual
    feature + floor read the raw pre-NS mic where the user is plainly present
    (+14 dB over the raw floor, past the 12 dB gate). One real talk-over, one
    cut -- without a shout."""
    eng = _open_speaker_engine(apm_owns_ns=True)  # reads the raw pre-NS mic
    _warm_echo_only(eng, raw=ECHO_RAW, resid_ns=ECHO_RESID_NS)

    fired = _probe(eng, raw=TALK_RAW, resid_ns=TALK_RESID_NS)
    assert fired is True, (
        "open_speaker APM double-talk: a normal-volume talk-over was REJECTED -- "
        "the DTD is still reading the NS-suppressed residual ('user had to scream')"
    )


# --------------------------------------------------------------------------- #
# 3. Self-interrupt safety preserved under the fix (the other half of the HARD
#    REQUIREMENT): echo-only must NOT cut.
# --------------------------------------------------------------------------- #
def test_apm_owns_ns_does_not_self_interrupt_on_echo_only():
    """Under the fix, an echo-only block (raw at the echo floor) does NOT fire:
    the z-score self-calibration keeps ``D`` low and the raw-floor gate rejects a
    level that is at the floor. The fix restores sight of a talk-over WITHOUT
    re-opening the self-interrupt the open-speaker requirement forbids."""
    eng = _open_speaker_engine(apm_owns_ns=True)
    _warm_echo_only(eng, raw=ECHO_RAW, resid_ns=ECHO_RESID_NS)

    fired = _probe(eng, raw=ECHO_ONLY_RAW, resid_ns=ECHO_ONLY_RESID_NS)
    assert fired is False, (
        "self-interrupt: echo-only block fired a barge under _apm_owns_ns -- the "
        "raw-floor gate / z-calibration is not rejecting the assistant's own echo"
    )


def test_repeated_echo_only_blocks_never_self_interrupt_under_apm():
    """A whole stretch of echo-only playback blocks (not just one) stays quiet --
    the chart learned on the raw echo floor never drifts a steady echo into a fire."""
    eng = _open_speaker_engine(apm_owns_ns=True)
    _warm_echo_only(eng, raw=ECHO_RAW, resid_ns=ECHO_RESID_NS)

    fires = [
        _probe(eng, raw=ECHO_ONLY_RAW, resid_ns=ECHO_ONLY_RESID_NS)
        for _ in range(20)
    ]
    assert not any(fires), f"echo-only self-interrupt(s) under APM: {fires.count(True)}/20"


# --------------------------------------------------------------------------- #
# 4. Non-regression: the linear-canceller (dtln/headphones) path is untouched.
# --------------------------------------------------------------------------- #
def test_linear_canceller_path_unchanged_user_survives_in_residual():
    """With ``_apm_owns_ns=False`` (the default: dtln/nlms/headphones) the user
    survives in the post-AEC residual, so a talk-over whose residual stands well
    above the floor still fires through the residual feature -- exactly as before
    the fix. Guards that the APM gate did not perturb the validated linear path."""
    eng = _open_speaker_engine(apm_owns_ns=False)
    # Linear AEC: residual echo floor is small; the user SURVIVES -> residual rises.
    _warm_echo_only(eng, raw=ECHO_RAW, resid_ns=ECHO_RESID_NS)

    # dtln talk-over: the user is in the residual at ~0.030 (well over the 12 dB bar).
    fired = _probe(eng, raw=TALK_RAW, resid_ns=0.030)
    assert fired is True, (
        "the linear-canceller residual path regressed: a clear talk-over (residual "
        "well above the floor) no longer fires"
    )
