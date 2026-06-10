"""Chart-contamination requirement tests (2026-06-10 plan step 3).

Pins the fix for the recorded ``run-20260610-003800`` misses: the owner's
talk-overs were ABSORBED into the AdaptiveDTD chart baselines, so ``z_resid``
logged 0.00 against levels 7-20x above the true echo floor. Two recorded
failures, both replayed here through the REAL engine seam over the committed
trace (``tests/fixtures/barge_in/run-20260610-003800.trace.txt``):

* ``talkover_0045`` -- a normal talk-over (resid 0.0134-0.0449 vs echo floor
  ~0.0002-0.0008) that NEVER fired live (warm-up had seeded the baseline on the
  user already talking at reply onset).
* ``scream_0046`` -- the scream (resid up to 0.1553) that took ~3.7s to cut.

THE HARNESS mirrors the capture loop block-for-block, all REAL code:

* quiet playback blocks -> ``_update_playback_floor`` / ``_update_raw_playback_
  floor`` + the ``observe_echo`` learning tap (the engine feeds the charts the
  VAD-QUIET blocks live; the trace only logged VAD-speech blocks, so the tap's
  diet is modeled with this run's own recorded clean-echo levels via
  ``clean_echo_levels``);
* burst blocks -> the REAL ``_barge_in_fire_eligible`` -> ``_looks_like_user``
  (DTD + residual-floor gate) -> REAL ``BargeSustain``.

Tier 0: no sound card, no models. Sibling files own the 203236/234435 traces.
"""
from __future__ import annotations

from tests import barge_fixtures as bf


def _prewarmed_engine(clean, n: int = 40):
    """A REAL engine seam warmed exactly as the capture loop would be after a
    stretch of quiet playback: floors learned + charts fed via the tap."""
    eng = bf.live_engine_with_dtd()
    sustain = bf.build_live_sustain()
    for f in clean[:n]:
        eng._update_playback_floor(f.resid)
        eng._update_raw_playback_floor(f.raw)
        eng._dtd.observe_echo(f.raw, f.resid, f.incoh)
    return eng, sustain


def _drive_burst(eng, sustain, window):
    """Feed a recorded burst through the REAL eligibility seam + sustain;
    return the 0-based offset (within the window) of the first cut, or None."""
    for off, f in enumerate(window):
        eng._update_playback_floor(f.resid)       # capture loop: every playback block
        eng._update_raw_playback_floor(f.raw)
        eng._fake_coherence.last_incoherent_fraction = f.incoh
        eng._fake_vad.set_speech(True)
        eligible = eng._barge_in_fire_eligible(
            bf.make_block(f.resid), bf.make_block(f.raw)
        )
        if sustain.update(eligible):
            return off
    return None


def test_miss_trace_loads_with_both_recorded_windows():
    frames = bf.load_miss_frames()
    w = bf.miss_windows(frames)
    assert w["talkover_0045"], "00:45 talk-over window missing from the parse"
    assert w["scream_0046"], "00:46 scream window missing from the parse"
    # The recorded scream peak (resid 0.1553 at 00:46:19.159) must be present.
    assert any(abs(f.resid - 0.1553) < 1e-9 for f in w["scream_0046"])
    # Recorded ground truth: the 00:45 talk-over NEVER fired live.
    assert not any(f.exp_fired for f in w["talkover_0045"])
    # Clean-echo modeling data exists (the run's true quiet floor).
    assert len(bf.clean_echo_levels(frames)) >= 40


def test_scream_now_cuts_within_half_a_second():
    """The 00:46:15 scream took ~3.7s to cut live (z_resid pinned at 0 by the
    contaminated baseline). With warmed persistent charts it must convert
    within the BargeSustain window of the burst onset (<= 0.5s)."""
    frames = bf.load_miss_frames()
    clean = bf.clean_echo_levels(frames)
    window = bf.miss_windows(frames)["scream_0046"]

    eng, sustain = _prewarmed_engine(clean)
    cut_off = _drive_burst(eng, sustain, window)

    assert cut_off is not None, "the scream still does not cut after the fix"
    # need_frames eligible blocks are required by design; the cut must land as
    # soon as the sustain CAN fire (no contamination delay on top).
    assert cut_off < 5, (
        f"scream cut only at window offset {cut_off} (~{0.1 * cut_off:.1f}s) -- "
        "the baseline is still absorbing the talk-over"
    )


def test_normal_talkover_0045_now_fires():
    """The 00:45:08 normal talk-over NEVER fired live. With warmed persistent
    charts (no per-reply warm-up for the user to poison) it must cut inside
    the burst."""
    frames = bf.load_miss_frames()
    clean = bf.clean_echo_levels(frames)
    window = bf.miss_windows(frames)["talkover_0045"]

    eng, sustain = _prewarmed_engine(clean)
    cut_off = _drive_burst(eng, sustain, window)

    assert cut_off is not None, (
        "the recorded normal talk-over still never fires -- warm-up poisoning "
        "is back"
    )
    assert cut_off < 6, f"talk-over cut too late (offset {cut_off})"


def test_clean_echo_blocks_never_cut_even_with_vad_flapping():
    """Echo safety with the same warmed seam: the run's own quiet playback
    levels, fed as if the VAD (wrongly) heard speech in every one of them,
    must never convert to a cut."""
    frames = bf.load_miss_frames()
    clean = bf.clean_echo_levels(frames)

    eng, sustain = _prewarmed_engine(clean, n=20)
    cut_off = _drive_burst(eng, sustain, clean[20:])

    assert cut_off is None, (
        f"a clean echo block produced a barge-in cut at offset {cut_off} -- "
        "the persistent charts/robust seed regressed echo safety"
    )


def test_observe_echo_tap_completes_warmup_so_onset_talkover_fires():
    """The decisive property the per-reply warm-up could not give: with the
    charts warmed purely by the quiet-block tap, a talk-over arriving at the
    very FIRST evaluated block of a reply (the user already objecting at reply
    onset -- the recorded poisoning scenario) fires immediately instead of
    being seeded into the baseline."""
    frames = bf.load_miss_frames()
    clean = bf.clean_echo_levels(frames)
    scream = bf.miss_windows(frames)["scream_0046"]
    peak = max(scream, key=lambda f: f.resid)

    dtd = bf.build_live_dtd()
    for f in clean[:20]:
        dtd.observe_echo(f.raw, f.resid, f.incoh)

    # A new reply begins; the user is ALREADY talking on its first block.
    dtd.new_run()
    assert dtd.decide(peak.raw, peak.resid, peak.incoh) is True, (
        "an onset talk-over was absorbed by warm-up instead of firing -- "
        "chart persistence regressed"
    )
