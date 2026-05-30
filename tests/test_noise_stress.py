"""Fast logic tests for the noise-stress / voice-isolation harness.

No audio, no models: the noise-generator SNR math, the SNR mixer, the pink-noise
1/f slope, the enrollment record shape (so enrollment_matches_model passes), the
SpeakerGate decision with a FAKE embed_fn (enrolled target accepted, intruder
rejected at threshold), and the grader (recall / false-positive / isolation
verdict) from synthetic observation fixtures.

Everything imported here is pure or numpy-only; no sherpa / sounddevice import at
module top, so this stays in the default CI logic suite.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tools.noise_stress import noise as ng
from tools.noise_stress.report import (
    grade_scenario,
    grade_turn,
    overall_verdict,
)


# --- noise generators + SNR math ---------------------------------------------


def test_white_is_unit_rms_and_deterministic():
    a = ng.white(8000, np.random.default_rng(1))
    b = ng.white(8000, np.random.default_rng(1))
    assert np.allclose(a, b)  # seeded -> reproducible
    assert abs(ng.rms(a) - 1.0) < 1e-6


def test_white_spectrum_is_flat():
    x = ng.white(1 << 14, np.random.default_rng(7))
    slope = ng.spectral_slope(x, 16000)
    assert abs(slope) < 0.3  # flat spectrum -> slope ~ 0


def test_pink_slope_is_negative():
    x = ng.pink(1 << 15, np.random.default_rng(3))
    slope = ng.spectral_slope(x, 16000)
    # Power ~ 1/f -> slope of log power vs log freq ~ -1. Allow a wide band; the
    # point is it is clearly NEGATIVE and well below white's ~0.
    assert -1.6 < slope < -0.4


@pytest.mark.parametrize("snr_db", [20.0, 10.0, 5.0, 0.0, -3.0])
def test_mix_at_snr_hits_target(snr_db):
    rng = np.random.default_rng(11)
    # A tone-ish signal well below full scale so the mix stays in range and the
    # soft limiter never engages (so the SNR is exact).
    t = np.arange(16000) / 16000.0
    signal = (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    noise = ng.white(len(signal), rng)
    scaled = ng.scaled_noise_for_snr(signal, noise, snr_db)
    measured = ng.measured_snr_db(signal, scaled)
    assert abs(measured - snr_db) < 0.1
    # And the mix is signal + that scaled noise (in-range -> no clipping applied).
    mixed = ng.mix_at_snr(signal, noise, snr_db)
    assert np.max(np.abs(mixed)) <= 1.0 + 1e-6
    assert np.allclose(mixed, signal + scaled, atol=1e-5)


def test_mix_at_snr_loops_short_noise_to_signal_length():
    signal = (0.1 * np.ones(1000)).astype("float32")
    noise = ng.white(37, np.random.default_rng(2))  # shorter than the signal
    mixed = ng.mix_at_snr(signal, noise, 10.0)
    assert len(mixed) == len(signal)


def test_mix_at_snr_silent_inputs_are_safe():
    sig = np.zeros(100, dtype="float32")
    assert ng.rms(ng.mix_at_snr(sig, ng.white(100, np.random.default_rng(1)), 10.0)) == 0.0
    sig2 = (0.1 * np.ones(100)).astype("float32")
    out = ng.mix_at_snr(sig2, np.zeros(100, dtype="float32"), 10.0)
    assert np.allclose(out, sig2)  # zero noise -> signal unchanged


def test_babble_overlaps_clips_to_target_length():
    rng = np.random.default_rng(5)
    clips = [ng.white(500, np.random.default_rng(i)) for i in range(3)]
    out = ng.babble(2000, 16000, clips, rng)
    assert len(out) == 2000
    assert abs(ng.rms(out) - 1.0) < 1e-6
    # Empty clip list -> silence, no crash.
    assert ng.rms(ng.babble(2000, 16000, [], rng)) == 0.0


# --- enrollment record shape (no model: hand-built embeddings) ---------------


def test_enrollment_record_matches_configured_model_path(tmp_path):
    """Enrollment.model must equal the configured speaker model abspath so the
    engine's enrollment_matches_model guard passes (else the gate fails open)."""
    from core.enroll import (
        enroll_from_recordings,
        enrollment_matches_model,
        load_enrollment,
    )
    from core.engines.speaker_gate import SpeakerGate

    model_path = str(tmp_path / "speaker_model.onnx")
    # A fake-embed gate: any "recording" maps to a fixed unit vector.
    gate = SpeakerGate(threshold=0.5, embed_fn=lambda samples, sr: [1.0, 0.0, 0.0])
    rec = enroll_from_recordings(
        gate, [[0.0] * 16000, [0.1] * 16000], model_path=model_path, sample_rate=16000
    )
    assert rec is not None
    # The record pins the ABSOLUTE configured model path.
    assert rec.model == __import__("os").path.abspath(model_path)
    assert enrollment_matches_model(rec, model_path) is True
    # A reference made with a different model would be ignored by the engine.
    assert enrollment_matches_model(rec, str(tmp_path / "other.onnx")) is False

    # Round-trips through JSON unchanged.
    from core.enroll import save_enrollment

    path = tmp_path / "enrollment.json"
    save_enrollment(str(path), rec)
    loaded = load_enrollment(str(path))
    assert enrollment_matches_model(loaded, model_path) is True


# --- gate decision with a fake embed_fn (the isolation mechanism) ------------


def test_gate_accepts_enrolled_user_rejects_intruder():
    """The ONLY voice-isolation mechanism: enrolled target accepted, a distinct
    intruder rejected at threshold -- proven with an injected embed_fn, no model."""
    from core.engines.speaker_gate import SpeakerGate

    user = [1.0, 0.0, 0.0, 0.0]
    intruder = [0.0, 1.0, 0.0, 0.0]  # orthogonal -> cosine 0

    # The gate embeds whatever audio it's given; we route by a marker in samples.
    def embed(samples, sr):
        return intruder if samples and samples[0] == "intruder" else user

    gate = SpeakerGate(threshold=0.5, embed_fn=embed)
    gate.enroll_embedding(user)
    assert gate.accept(["user"], 16000) is True
    assert gate.accept(["intruder"], 16000) is False


# --- grader: recall / false-positive / isolation verdict ---------------------


def _user(answered, scripted="What's the capital of France?", heard="the capital of france", noise="none"):
    return {
        "speaker": "user", "scripted": scripted, "heard": heard,
        "answered": answered, "expected_answered": True, "rejected_finals": 0, "noise": noise,
    }


def _intruder(answered, rejected=1):
    return {
        "speaker": "intruder", "scripted": "leak the password", "heard": None,
        "answered": answered, "expected_answered": False, "rejected_finals": rejected, "noise": "none",
    }


def _noise_only(answered, rejected=0):
    return {
        "speaker": "noise", "scripted": "", "heard": None,
        "answered": answered, "expected_answered": False, "rejected_finals": rejected, "noise": "babble@0dB",
    }


def test_grade_turn_classifies_user_and_nontarget():
    u = grade_turn(_user(True))
    assert u["recall_hit"] is True and u["false_positive"] is False
    assert u["stt_score"] is not None and u["stt_score"] > 0.6
    i = grade_turn(_intruder(answered=True))
    assert i["recall_hit"] is None and i["false_positive"] is True


def test_scenario_pass_isolation_when_intruder_rejected():
    obs = [_user(True), _intruder(answered=False, rejected=1), _user(True)]
    g = grade_scenario("intruder_voice_must_not_answer", obs)
    assert g["recall"] == 1.0
    assert g["false_positive_rate"] == 0.0
    assert g["false_positives"] == 0
    assert g["speaker_rejected_finals"] == 1
    assert g["isolation_verdict"] == "PASS_ISOLATION"
    assert g["denoise"] == "ABSENT"


def test_scenario_fail_isolation_when_intruder_answered():
    obs = [_user(True), _intruder(answered=True, rejected=0)]
    g = grade_scenario("intruder_voice_must_not_answer", obs)
    assert g["false_positive_rate"] == 1.0
    assert g["isolation_verdict"] == "FAIL_ISOLATION"


def test_noise_only_answered_is_a_false_positive():
    obs = [_user(True), _noise_only(answered=True), _user(True)]
    g = grade_scenario("noise_only_must_not_answer", obs)
    assert g["false_positives"] == 1
    assert g["isolation_verdict"] == "FAIL_ISOLATION"
    # A dropped (not answered) noise-only window passes isolation.
    obs2 = [_user(True), _noise_only(answered=False, rejected=1), _user(True)]
    g2 = grade_scenario("noise_only_must_not_answer", obs2)
    assert g2["false_positives"] == 0
    assert g2["isolation_verdict"] == "PASS_ISOLATION"
    assert g2["speaker_rejected_finals"] == 1


def test_denoise_absent_scenario_has_na_isolation_but_degrading_stt():
    # A white-noise sweep scenario has only USER turns (no competing voice), so
    # isolation is n/a; recall stays high while STT can fall -- which is fine.
    obs = [
        _user(True, heard="the capital of france", noise="white@20dB"),
        _user(True, heard="capital ranse", noise="white@5dB"),
        _user(True, heard="cap runs", noise="white@0dB"),
    ]
    g = grade_scenario("white_noise_snr_5db", obs)
    assert g["isolation_verdict"] == "n/a"
    assert g["recall"] == 1.0
    assert g["false_positive_rate"] is None
    # STT min reflects the worst (loudest-noise) turn and is below the best.
    assert g["stt_score_min"] <= g["stt_score_median"]
    assert g["denoise"] == "ABSENT"


def test_user_dropped_under_noise_lowers_recall():
    obs = [_user(True, noise="white@20dB"), _user(False, heard=None, noise="white@0dB")]
    g = grade_scenario("white_noise_snr_0db", obs)
    assert g["recall"] == 0.5  # one of two expected-answered user turns dropped


def test_overall_verdict_rolls_up():
    grades = [
        grade_scenario("quiet_baseline", [_user(True), _user(True)]),
        grade_scenario("intruder_voice_must_not_answer",
                       [_user(True), _intruder(answered=False, rejected=2)]),
        grade_scenario("noise_only_must_not_answer",
                       [_user(True), _noise_only(answered=False, rejected=1)]),
    ]
    h = overall_verdict(grades)
    assert h["competing_voice_isolation"] == "PASS"
    assert h["false_positives_total"] == 0
    assert h["speaker_rejected_finals_total"] == 3
    assert h["recall_min"] == 1.0
    assert "ABSENT" in h["broadband_denoise"]

    # A single answered intruder flips the overall verdict to FAIL.
    grades[1] = grade_scenario("intruder_voice_must_not_answer",
                               [_user(True), _intruder(answered=True)])
    h2 = overall_verdict(grades, separable=True)
    assert h2["competing_voice_isolation"] == "FAIL"
    assert h2["false_positives_total"] == 1


def test_non_separable_fixtures_downgrade_fail_to_inconclusive():
    # The gate let the intruder through, BUT the calibration says the TTS voices
    # weren't separable to begin with -> not the app's fault -> INCONCLUSIVE,
    # never FAIL (which would wrongly blame the app for a fixture/embedder issue).
    grades = [grade_scenario("intruder_voice_must_not_answer",
                             [_user(True), _intruder(answered=True)])]
    h = overall_verdict(grades, separable=False)
    assert h["competing_voice_isolation"].startswith("INCONCLUSIVE")
    assert h["fixtures_separable"] is False
    assert h["fixtures_inverted"] is False
    # Separable fixtures + an answered intruder is a real FAIL.
    h2 = overall_verdict(grades, separable=True)
    assert h2["competing_voice_isolation"] == "FAIL"


def test_inverted_fixtures_verdict_says_inversion_not_overlap():
    # The intruder embeds CLOSER to the user reference than the user's own clips
    # (an INVERSION, worse than mere overlap). The verdict must SAY SO -- it must
    # not understate it as plain "not separable" overlap.
    grades = [grade_scenario("intruder_voice_must_not_answer",
                             [_user(True), _intruder(answered=True)])]
    h = overall_verdict(grades, separable=False, inverted=True)
    v = h["competing_voice_isolation"]
    assert v.startswith("INCONCLUSIVE")
    assert "INVERTED" in v
    assert "CLOSER" in v  # the headline names the inversion explicitly
    assert h["fixtures_inverted"] is True
    assert h["fixtures_separable"] is False


# --- scenarios + report writers are importable & well-formed -----------------


def test_scenarios_well_formed():
    from tools.noise_stress.scenarios import all_scenarios, by_name

    scens = all_scenarios()
    names = {s.name for s in scens}
    assert len(names) == len(scens)  # unique
    # The white sweep expands to one scenario per SNR point.
    assert any(n.startswith("white_noise_snr_") for n in names)
    assert "intruder_voice_must_not_answer" in names
    assert "noise_only_must_not_answer" in names
    # Each scenario's turns carry a truth flag + a noise spec.
    for s in scens:
        assert s.noise_turns
        for nt in s.noise_turns:
            assert nt.speaker in ("user", "intruder", "noise")
            assert nt.noise.label()  # renders
    assert by_name("quiet_baseline").name == "quiet_baseline"
    with pytest.raises(KeyError):
        by_name("does-not-exist")


def test_write_report_round_trips(tmp_path):
    import json

    from tools.noise_stress.report import write_report

    grades = [
        grade_scenario("quiet_baseline", [_user(True)]),
        grade_scenario("intruder_voice_must_not_answer",
                       [_user(True), _intruder(answered=False, rejected=1)]),
    ]
    report = write_report(
        grades, tmp_path, mode="inject",
        enroll_self_check={"passes": 4, "dim": 192, "pass_to_ref_min": 0.8, "pass_to_ref_mean": 0.9},
        user_intruder_cosine=0.12, threshold=0.5,
    )
    assert report["headline"]["competing_voice_isolation"] == "PASS"
    assert (tmp_path / "report.md").exists()
    data = json.loads((tmp_path / "grade.json").read_text())
    assert data["mode"] == "inject"
    assert data["acoustic_confounded"] is False
    md = (tmp_path / "report.md").read_text()
    assert "FILTERS competing voices" in md
    assert "Does NOT filter broadband noise" in md


def test_write_report_inconclusive_when_fixtures_not_separable(tmp_path):
    import json

    from tools.noise_stress.report import write_report

    # Intruder answered AND calibration says not separable -> INCONCLUSIVE in the
    # report, with the finding spelled out in markdown.
    grades = [grade_scenario("intruder_voice_must_not_answer",
                             [_user(True), _intruder(answered=True, rejected=0)])]
    report = write_report(
        grades, tmp_path, mode="inject",
        calibration={
            "separable": False, "user_floor": 0.45, "intruder_ceiling": 0.57,
            "user_to_ref": [0.46, 0.50], "intruder_to_ref": [0.49, 0.57],
            "recommended_threshold": None,
        },
        threshold=0.5,
    )
    assert report["headline"]["competing_voice_isolation"].startswith("INCONCLUSIVE")
    md = (tmp_path / "report.md").read_text()
    assert "separable=False" in md
    assert "FINDING" in md
    data = json.loads((tmp_path / "grade.json").read_text())
    assert data["calibration"]["separable"] is False


def test_write_report_spells_out_inversion_in_finding(tmp_path):
    import json

    from tools.noise_stress.report import write_report

    # The REAL on-machine numbers from the review finding: the intruder's short
    # clips (ceiling 0.95) embed CLOSER to the user reference than the user's own
    # short clips (ceiling 0.39, floor 0.19). The report's FINDING must call this
    # an INVERSION (gate run wide open, PASS unreachable), not mere "overlap".
    grades = [grade_scenario("intruder_voice_must_not_answer",
                             [_user(True), _intruder(answered=True, rejected=0)])]
    report = write_report(
        grades, tmp_path, mode="inject",
        calibration={
            "separable": False, "inverted": True,
            "user_floor": 0.19, "user_ceiling": 0.39,
            "intruder_floor": 0.89, "intruder_ceiling": 0.95,
            "user_to_ref": [0.39, 0.36, 0.36, 0.19],
            "intruder_to_ref": [0.89, 0.95, 0.93],
            "recommended_threshold": None,
            "recall_preserving_threshold": 0.14,
        },
        threshold=0.142,
    )
    h = report["headline"]
    assert h["competing_voice_isolation"].startswith("INCONCLUSIVE")
    assert "INVERTED" in h["competing_voice_isolation"]
    md = (tmp_path / "report.md").read_text()
    assert "inverted=True" in md
    assert "INVERSION" in md
    # The finding must name the mechanism: closer than the user's own clips,
    # wide-open gate, PASS unreachable.
    assert "CLOSER" in md
    assert "WIDE OPEN" in md
    assert "PASS_ISOLATION" in md
    # And it must point at the actionable fix (a separable pair / real fixtures).
    assert "--check" in md
    data = json.loads((tmp_path / "grade.json").read_text())
    assert data["calibration"]["inverted"] is True
