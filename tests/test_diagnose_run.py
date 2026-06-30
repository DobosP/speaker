"""Tests for tools.diagnose_run using synthetic log fixtures.

No real audio files or model weights required.  All timestamps use the
HH:MM:SS.mmm format that the sherpa engine emits.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tools.diagnose_run import (
    ParsedRun,
    _analyze_wav_segment,
    analyze_mic_ref_wav,
    analyze_ref_wav,
    _nearest_speaking_state,
    classify_barge_event,
    diagnostic_findings,
    format_report,
    parse_log,
    self_interrupt_summary,
    to_json,
)


# ---------------------------------------------------------------------------
# Synthetic log fixtures
# ---------------------------------------------------------------------------

def _write_log(tmp_path: Path, lines: list[str]) -> str:
    p = tmp_path / "run-test.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def _write_wav(path: Path, samples, sample_rate: int = 16000) -> str:
    import wave
    import numpy as np

    pcm = np.clip(samples, -1.0, 1.0)
    data = (pcm * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data)
    return str(path)


CLEAN_RUN_LINES = [
    "12:00:00.000 INFO  speaker | run 20260628-120000 started (debug=True) -> logs/runs/run-20260628-120000.txt",
    "12:00:01.000 INFO  speaker.sherpa | recording playback reference (replay) -> logs/runs/run-20260628-120000.ref.wav",
    "12:00:05.000 DEBUG speaker.sherpa | capture heartbeat: blocks=20 avg_rms=0.0085 clip=0.0% underruns=0 partials=0 finals=0 speaking=False",
    "12:00:10.000 DEBUG speaker.sherpa | speaking: 'Hello, world.' (queue depth=0)",
    "12:00:10.020 INFO  speaker.sherpa | playback opened at 24000 Hz on device default (callback)",
    "12:00:11.000 DEBUG speaker.sherpa | capture heartbeat: blocks=200 avg_rms=0.0200 clip=0.0% underruns=0 partials=1 finals=0 speaking=True",
    "12:00:13.000 DEBUG speaker.sherpa | capture heartbeat: blocks=300 avg_rms=0.0100 clip=0.0% underruns=0 partials=1 finals=1 speaking=False",
]

SELF_INTERRUPT_LINES = [
    "12:00:00.000 INFO  speaker | run 20260628-120001 started (debug=True) -> logs/runs/run-20260628-120001.txt",
    "12:00:10.000 DEBUG speaker.sherpa | speaking: 'Testing self-interrupt.' (queue depth=0)",
    "12:00:10.020 INFO  speaker.sherpa | playback opened at 24000 Hz on device default (callback)",
    "12:00:11.000 DEBUG speaker.sherpa | capture heartbeat: blocks=200 avg_rms=0.0200 clip=0.0% underruns=0 partials=0 finals=0 speaking=True",
    "12:00:11.500 DEBUG speaker.sherpa | dtd: D=9.24 K=5.0 fired=True gated=True (z_raw=6.51 z_resid=7.94 z_coh=722303.29) raw=0.0581 resid=0.0649 incoh=0.85 resid_floor=0.0091 consec=1",
    "12:00:11.600 DEBUG speaker.sherpa | dtd: D=13.09 K=5.0 fired=True gated=True (z_raw=14.88 z_resid=10.11 z_coh=914927.41) raw=0.1145 resid=0.0911 incoh=0.91 resid_floor=0.0091 consec=2",
    "12:00:11.700 INFO  speaker.sherpa | barge-in detected",
    "12:00:12.000 DEBUG speaker.sherpa | capture heartbeat: blocks=300 avg_rms=0.0100 clip=0.0% underruns=0 partials=0 finals=0 speaking=False",
]

REAL_BARGE_LINES = [
    "12:00:00.000 INFO  speaker | run 20260628-120002 started (debug=True) -> logs/runs/run-20260628-120002.txt",
    "12:00:10.000 DEBUG speaker.sherpa | speaking: 'Some assistant reply.' (queue depth=0)",
    "12:00:10.020 INFO  speaker.sherpa | playback opened at 24000 Hz on device default (callback)",
    "12:00:10.500 DEBUG speaker.sherpa | capture heartbeat: blocks=200 avg_rms=0.0200 clip=0.0% underruns=0 partials=0 finals=0 speaking=True",
    # Low incoh → user is talking, not echo
    "12:00:11.500 DEBUG speaker.sherpa | dtd: D=9.24 K=5.0 fired=True gated=True (z_raw=6.51 z_resid=7.94 z_coh=100.00) raw=0.0581 resid=0.0649 incoh=0.30 resid_floor=0.0091 consec=1",
    "12:00:11.700 INFO  speaker.sherpa | barge-in detected",
    "12:00:12.000 DEBUG speaker.sherpa | capture heartbeat: blocks=300 avg_rms=0.0100 clip=0.0% underruns=0 partials=0 finals=0 speaking=False",
]

BARGE_REJECTED_LINES = [
    "12:00:00.000 INFO  speaker | run 20260628-120003 started (debug=True) -> logs/runs/run-20260628-120003.txt",
    "12:00:10.000 DEBUG speaker.sherpa | speaking: 'Another reply.' (queue depth=0)",
    "12:00:10.020 INFO  speaker.sherpa | playback opened at 24000 Hz on device default (callback)",
    "12:00:10.500 DEBUG speaker.sherpa | capture heartbeat: blocks=200 avg_rms=0.0200 clip=0.0% underruns=0 partials=0 finals=0 speaking=True",
    "12:00:11.700 INFO  speaker.sherpa | barge-in REJECTED: 0.2s of voiced speech during playback did not trip the gate (talk-over ignored?)",
    "12:00:12.000 DEBUG speaker.sherpa | capture heartbeat: blocks=300 avg_rms=0.0100 clip=0.0% underruns=0 partials=0 finals=0 speaking=True",
]

MULTI_SENTENCE_LINES = [
    "12:00:00.000 INFO  speaker | run 20260628-120004 started (debug=True) -> logs/runs/run-20260628-120004.txt",
    "12:00:10.000 DEBUG speaker.sherpa | speaking: 'Sentence one.' (queue depth=0)",
    "12:00:10.020 INFO  speaker.sherpa | playback opened at 24000 Hz on device default (callback)",
    "12:00:11.000 DEBUG speaker.sherpa | capture heartbeat: blocks=200 avg_rms=0.0100 clip=0.0% underruns=0 partials=0 finals=0 speaking=True",
    "12:00:14.000 DEBUG speaker.sherpa | speaking: 'Sentence two.' (queue depth=0)",
    "12:00:15.000 DEBUG speaker.sherpa | capture heartbeat: blocks=400 avg_rms=0.0100 clip=0.0% underruns=0 partials=0 finals=0 speaking=True",
    "12:00:16.000 DEBUG speaker.sherpa | speaking: 'Sentence three.' (queue depth=1)",
]


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_parse_clean_run(tmp_path):
    path = _write_log(tmp_path, CLEAN_RUN_LINES)
    run = parse_log(path)

    assert run.run_id == "20260628-120000"
    assert run.run_start_t is not None
    assert len(run.sentences) == 1
    s = run.sentences[0]
    assert s.text == "Hello, world."
    assert s.queue_depth == 0
    assert s.playback_sample_rate == 24000
    assert s.playback_open_latency_ms is not None
    assert 15 < s.playback_open_latency_ms < 30  # ~20ms
    assert len(run.heartbeats) == 3
    assert len(run.barge_events) == 0
    assert run.ref_wav_path is not None


def test_parse_self_interrupt(tmp_path):
    path = _write_log(tmp_path, SELF_INTERRUPT_LINES)
    run = parse_log(path)

    assert len(run.sentences) == 1
    s = run.sentences[0]
    assert s.text == "Testing self-interrupt."
    assert len(run.barge_events) == 1
    be = run.barge_events[0]
    assert be.kind == "detected"
    assert be.speaking_at_event is True
    assert len(be.dtd_context) == 2
    assert all(f.gated for f in be.dtd_context)


def test_parse_multi_sentence(tmp_path):
    path = _write_log(tmp_path, MULTI_SENTENCE_LINES)
    run = parse_log(path)

    assert len(run.sentences) == 3
    assert run.sentences[0].text == "Sentence one."
    assert run.sentences[1].text == "Sentence two."
    assert run.sentences[2].text == "Sentence three."
    assert [s.playback_sample_rate for s in run.sentences] == [24000, 24000, 24000]
    # sentence[0] should be closed at the next sentence start
    assert run.sentences[0].t_end is not None


def test_parse_barge_rejected(tmp_path):
    path = _write_log(tmp_path, BARGE_REJECTED_LINES)
    run = parse_log(path)

    assert len(run.barge_events) == 1
    be = run.barge_events[0]
    assert be.kind == "rejected"
    assert "talk-over ignored" in be.reason
    assert be.speaking_at_event is True


def test_dtd_associated_with_sentence(tmp_path):
    path = _write_log(tmp_path, SELF_INTERRUPT_LINES)
    run = parse_log(path)

    s = run.sentences[0]
    assert len(s.dtd_frames) == 2
    assert all(f.incoh > 0.8 for f in s.dtd_frames)


def test_parse_dtd_current_suffix_and_aec_config(tmp_path):
    lines = [
        "12:00:00.000 INFO  speaker | run 20260628-120005 started -> x.txt",
        "12:00:00.100 INFO  speaker.sherpa | AEC ACTIVE on the capture path (16 kHz, backend=dtln, ref_delay=40ms, apm_always_on=False)",
        "12:00:10.000 DEBUG speaker.sherpa | speaking: 'Delay probe.' (queue depth=0)",
        "12:00:10.500 DEBUG speaker.sherpa | dtd: D=9.24 K=5.0 fired=True gated=False (z_raw=6.51 z_resid=7.94 z_coh=722303.29) raw=0.0581 resid=0.0649 incoh=0.72 resid_floor=0.0091 consec=1 coh=False coh_veto=True ref_delay=150ms",
    ]
    path = _write_log(tmp_path, lines)
    run = parse_log(path)

    assert run.aec_backend == "dtln"
    assert run.aec_config_ref_delay_ms == 40.0
    assert run.apm_always_on is False
    frame = run.dtd_frames[0]
    assert frame.coh_verdict == "False"
    assert frame.coh_veto is True
    assert frame.ref_delay_ms == 150.0


# ---------------------------------------------------------------------------
# "tts audio quality" telemetry (2026-06-30 robotic/white-noise investigation)
# ---------------------------------------------------------------------------

def _quality_log_lines(quality_json: str) -> list[str]:
    return [
        "12:00:00.000 INFO  speaker | run 20260630-120000 started -> x.txt",
        "12:00:10.000 DEBUG speaker.sherpa | speaking: 'Probe sentence.' (queue depth=0)",
        f"12:00:10.050 INFO  speaker.sherpa | tts audio quality: {quality_json}",
    ]


def test_parse_tts_quality_attaches_to_sentence(tmp_path):
    quality = {
        "mode": "whole_clip", "rms": 0.07, "peak": 0.5, "clip_pct": 0.0,
        "dc_offset": 0.0001, "hf_ratio": 0.01, "spectral_flatness": 0.02,
        "n_samples": 1000,
    }
    path = _write_log(tmp_path, _quality_log_lines(json.dumps(quality)))
    run = parse_log(path)

    assert run.sentences[0].tts_quality == quality


def test_finding_flags_noise_like_spectral_flatness(tmp_path):
    quality = {
        "mode": "whole_clip", "rms": 0.07, "peak": 0.5, "clip_pct": 0.0,
        "dc_offset": 0.0, "hf_ratio": 0.6, "spectral_flatness": 0.55,
        "n_samples": 1000,
    }
    path = _write_log(tmp_path, _quality_log_lines(json.dumps(quality)))
    run = parse_log(path)

    findings = diagnostic_findings(run)
    assert any("noise-like" in f and "sentence[0]" in f for f in findings)


def test_finding_flags_clipping_and_dc_offset(tmp_path):
    quality = {
        "mode": "whole_clip", "rms": 0.7, "peak": 1.0, "clip_pct": 12.5,
        "dc_offset": 0.05, "hf_ratio": 0.0, "spectral_flatness": 0.0,
        "n_samples": 1000,
    }
    path = _write_log(tmp_path, _quality_log_lines(json.dumps(quality)))
    run = parse_log(path)

    findings = diagnostic_findings(run)
    assert any("clipping" in f for f in findings)
    assert any("DC offset" in f for f in findings)


def test_finding_silent_on_clean_tts_quality(tmp_path):
    quality = {
        "mode": "whole_clip", "rms": 0.07, "peak": 0.3, "clip_pct": 0.0,
        "dc_offset": 0.0001, "hf_ratio": 0.01, "spectral_flatness": 0.02,
        "n_samples": 1000,
    }
    path = _write_log(tmp_path, _quality_log_lines(json.dumps(quality)))
    run = parse_log(path)

    findings = diagnostic_findings(run)
    assert not any("sentence[0]" in f for f in findings)


def test_format_report_shows_tts_quality_and_noise_flag(tmp_path):
    quality = {
        "mode": "streaming", "rms": 0.2, "peak": 0.9, "clip_pct": 0.0,
        "dc_offset": 0.0, "hf_ratio": None, "spectral_flatness": 0.6,
        "n_samples": 500,
    }
    path = _write_log(tmp_path, _quality_log_lines(json.dumps(quality)))
    run = parse_log(path)

    report = format_report(run)
    assert "tts audio quality: mode=streaming" in report
    assert "[NOISE-LIKE]" in report


def test_to_json_includes_tts_quality(tmp_path):
    quality = {
        "mode": "whole_clip", "rms": 0.07, "peak": 0.3, "clip_pct": 0.0,
        "dc_offset": 0.0, "hf_ratio": 0.01, "spectral_flatness": 0.02,
        "n_samples": 1000,
    }
    path = _write_log(tmp_path, _quality_log_lines(json.dumps(quality)))
    run = parse_log(path)

    data = to_json(run)
    assert data["sentences"][0]["tts_quality"] == quality


# ---------------------------------------------------------------------------
# Classification / suspicion tests
# ---------------------------------------------------------------------------

def test_classify_self_interrupt_high_incoh(tmp_path):
    path = _write_log(tmp_path, SELF_INTERRUPT_LINES)
    run = parse_log(path)

    be = run.barge_events[0]
    label = classify_barge_event(be)
    assert label.startswith("suspect")
    assert "incoh" in label


def test_classify_real_barge_low_incoh(tmp_path):
    path = _write_log(tmp_path, REAL_BARGE_LINES)
    run = parse_log(path)

    be = run.barge_events[0]
    label = classify_barge_event(be)
    # Low incoh (0.30) → still speaking=True but less suspicious
    assert label == "suspect:speaking"  # flagged because speaking=True but low incoh


def test_classify_barge_while_not_speaking(tmp_path):
    lines = [
        "12:00:00.000 INFO  speaker | run X started -> x.txt",
        "12:00:05.000 DEBUG speaker.sherpa | capture heartbeat: blocks=100 avg_rms=0.0100 clip=0.0% underruns=0 partials=0 finals=0 speaking=False",
        "12:00:06.000 INFO  speaker.sherpa | barge-in detected",
    ]
    path = _write_log(tmp_path, lines)
    run = parse_log(path)
    be = run.barge_events[0]
    label = classify_barge_event(be)
    assert label == "ok"


def test_self_interrupt_summary_verdict(tmp_path):
    path = _write_log(tmp_path, SELF_INTERRUPT_LINES)
    run = parse_log(path)
    si = self_interrupt_summary(run)
    assert si["verdict"] in ("self-interrupt-likely", "self-interrupt-possible")
    assert si["suspect_count"] >= 1


def test_self_interrupt_summary_clean(tmp_path):
    path = _write_log(tmp_path, CLEAN_RUN_LINES)
    run = parse_log(path)
    si = self_interrupt_summary(run)
    assert si["verdict"] == "clean"
    assert si["suspect_count"] == 0


# ---------------------------------------------------------------------------
# Nearest-speaking-state helper
# ---------------------------------------------------------------------------

def test_nearest_speaking_state_picks_before_event(tmp_path):
    from tools.diagnose_run import Heartbeat
    hbs = [
        Heartbeat(t=1.0, blocks=10, avg_rms=0.01, clip=0.0, speaking=False),
        Heartbeat(t=2.0, blocks=20, avg_rms=0.02, clip=0.0, speaking=True),
        Heartbeat(t=4.0, blocks=30, avg_rms=0.01, clip=0.0, speaking=False),
    ]
    assert _nearest_speaking_state(hbs, 2.5) is True
    assert _nearest_speaking_state(hbs, 1.5) is False
    assert _nearest_speaking_state(hbs, 5.0) is False
    assert _nearest_speaking_state(hbs, 0.5) is None


# ---------------------------------------------------------------------------
# WAV analysis (numpy required)
# ---------------------------------------------------------------------------

def test_analyze_wav_segment_sine():
    import numpy as np
    from tools.diagnose_run import _analyze_wav_segment

    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    result = _analyze_wav_segment(sine, sr, hf_cutoff_hz=4000)

    assert result["rms"] is not None
    assert 0.34 < result["rms"] < 0.37  # 0.5 * 1/sqrt(2) ≈ 0.354
    assert result["peak"] is not None
    assert 0.49 < result["peak"] < 0.51
    assert result["clip_pct"] == 0.0
    # 440 Hz sine should have very low HF ratio
    assert result["hf_ratio"] < 0.01
    assert result["duration_s"] == 1.0


def test_analyze_wav_segment_white_noise_has_high_hf():
    import numpy as np
    from tools.diagnose_run import _analyze_wav_segment

    rng = np.random.default_rng(42)
    noise = rng.standard_normal(24000).astype(np.float32) * 0.1
    result = _analyze_wav_segment(noise, 24000, hf_cutoff_hz=4000)

    # White noise distributes energy uniformly; above 4 kHz is ~2/3 of 0-12 kHz
    assert result["hf_ratio"] > 0.50, f"expected high HF for white noise, got {result['hf_ratio']}"


def test_analyze_wav_segment_empty():
    import numpy as np
    from tools.diagnose_run import _analyze_wav_segment

    result = _analyze_wav_segment(np.array([], dtype=np.float32), 24000)
    assert result["rms"] is None
    assert result["duration_s"] == 0.0


def test_ref_wav_marks_first_audio_and_pre_audio_barge(tmp_path):
    import numpy as np

    lines = [
        "12:00:00.000 INFO  speaker | run 20260628-120006 started -> x.txt",
        "12:00:00.000 INFO  speaker.sherpa | recording playback reference (replay) -> run-test.ref.wav",
        "12:00:10.000 DEBUG speaker.sherpa | speaking: 'Slow first audio.' (queue depth=0)",
        "12:00:10.200 INFO  speaker.sherpa | barge-in REJECTED: 0.2s of voiced speech during playback did not trip the gate (talk-over ignored?)",
        "12:00:10.700 DEBUG speaker.sherpa | speaking: 'Next sentence.' (queue depth=0)",
    ]
    log_path = _write_log(tmp_path, lines)
    sr = 16000
    samples = np.zeros(sr * 12, dtype=np.float32)
    t = np.arange(sr) / sr
    samples[10 * sr + sr // 2:11 * sr + sr // 2] = 0.15 * np.sin(2 * np.pi * 440 * t)
    wav_path = _write_wav(tmp_path / "run-test.ref.wav", samples, sr)

    run = parse_log(log_path)
    wav_metrics = analyze_ref_wav(wav_path, run)
    report = format_report(run, wav_metrics)
    out = to_json(run, wav_metrics)

    assert wav_metrics[0]["first_audio_offset_s"] == pytest.approx(0.5, abs=0.03)
    assert "before first playback reference audio" in report
    assert out["sentences"][0]["barge_events"][0]["phase"] == "pre-first-ref-audio"


def test_ref_wav_flags_no_reference_audio(tmp_path):
    import numpy as np

    lines = [
        "12:00:00.000 INFO  speaker | run 20260628-120007 started -> x.txt",
        "12:00:00.000 INFO  speaker.sherpa | recording playback reference (replay) -> run-test.ref.wav",
        "12:00:01.000 DEBUG speaker.sherpa | speaking: 'Silent ref.' (queue depth=0)",
    ]
    log_path = _write_log(tmp_path, lines)
    wav_path = _write_wav(tmp_path / "run-test.ref.wav", np.zeros(16000 * 3, dtype=np.float32))

    run = parse_log(log_path)
    wav_metrics = analyze_ref_wav(wav_path, run)
    report = format_report(run, wav_metrics)

    assert wav_metrics[0]["first_audio_offset_s"] is None
    assert "playback reference has no detected audio" in report
    assert "NO-REF" in report


def test_ref_wav_warns_when_not_bit_exact_output_rate(tmp_path):
    import numpy as np

    log_path = _write_log(tmp_path, CLEAN_RUN_LINES)
    samples = np.zeros(16000 * 14, dtype=np.float32)
    t = np.arange(16000) / 16000
    samples[10 * 16000:11 * 16000] = 0.1 * np.sin(2 * np.pi * 440 * t)
    wav_path = _write_wav(tmp_path / "run-test.ref.wav", samples, 16000)

    run = parse_log(log_path)
    wav_metrics = analyze_ref_wav(wav_path, run)
    report = format_report(run, wav_metrics)
    out = to_json(run, wav_metrics)

    assert wav_metrics["_sample_rate"] == 16000
    assert "not bit-exact 24000 Hz PortAudio output" in report
    assert any("not bit-exact 24000 Hz" in finding for finding in out["findings"])


def test_mic_ref_delay_estimate_flags_config_mismatch(tmp_path):
    import numpy as np

    lines = [
        "12:00:00.000 INFO  speaker | run 20260628-120008 started -> x.txt",
        "12:00:00.000 INFO  speaker.sherpa | recording playback reference (replay) -> run-test.ref.wav",
        "12:00:00.100 INFO  speaker.sherpa | AEC ACTIVE on the capture path (16 kHz, backend=dtln, ref_delay=40ms, apm_always_on=False)",
        "12:00:01.000 DEBUG speaker.sherpa | speaking: 'Delay mismatch.' (queue depth=0)",
        "12:00:03.000 DEBUG speaker.sherpa | speaking: 'Done.' (queue depth=0)",
    ]
    log_path = _write_log(tmp_path, lines)
    sr = 16000
    rng = np.random.default_rng(123)
    ref = np.zeros(sr * 4, dtype=np.float32)
    ref[sr:sr * 3] = (0.08 * rng.standard_normal(sr * 2)).astype(np.float32)
    delay = int(0.160 * sr)
    mic = np.zeros_like(ref)
    mic[delay:] = ref[:-delay] * 0.8
    mic += (0.002 * rng.standard_normal(ref.size)).astype(np.float32)
    ref_path = _write_wav(tmp_path / "run-test.ref.wav", ref, sr)
    mic_path = _write_wav(tmp_path / "run-test.wav", mic, sr)

    run = parse_log(log_path)
    wav_metrics = analyze_ref_wav(ref_path, run)
    delay_metrics = analyze_mic_ref_wav(mic_path, ref_path, run)
    report = format_report(run, wav_metrics, delay_metrics)
    out = to_json(run, wav_metrics, delay_metrics=delay_metrics)

    assert delay_metrics[0]["estimated_delay_ms"] == pytest.approx(160.0, abs=3.0)
    assert delay_metrics[0]["delay_correlation"] > 0.5
    assert "AEC delay mismatch" in report
    assert "AEC-DELAY-MISMATCH" in report
    assert any("AEC delay mismatch" in finding for finding in out["findings"])


# ---------------------------------------------------------------------------
# Format / JSON output (smoke tests)
# ---------------------------------------------------------------------------

def test_format_report_smoke(tmp_path):
    path = _write_log(tmp_path, SELF_INTERRUPT_LINES)
    run = parse_log(path)
    report = format_report(run)

    assert "Sentence Timeline" in report
    assert "Testing self-interrupt." in report
    assert "DETECTED" in report
    assert "Self-Interrupt Summary" in report
    assert "SELF-INTERRUPT" in report.upper()


def test_to_json_structure(tmp_path):
    path = _write_log(tmp_path, SELF_INTERRUPT_LINES)
    run = parse_log(path)
    out = to_json(run)
    text = json.dumps(out)
    data = json.loads(text)

    assert "run_id" in data
    assert "self_interrupt" in data
    assert "sentences" in data
    assert len(data["sentences"]) == 1
    s = data["sentences"][0]
    assert s["text"] == "Testing self-interrupt."
    assert len(s["barge_events"]) == 1
    assert s["barge_events"][0]["kind"] == "detected"
    assert s["barge_events"][0]["suspicion"].startswith("suspect")


def test_main_cli_smoke(tmp_path):
    path = _write_log(tmp_path, SELF_INTERRUPT_LINES)
    from tools.diagnose_run import main
    rc = main([str(path)])
    assert rc == 0


def test_main_cli_json(tmp_path, capsys):
    path = _write_log(tmp_path, SELF_INTERRUPT_LINES)
    from tools.diagnose_run import main
    rc = main([str(path), "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "self_interrupt" in data


def test_main_cli_missing_file(tmp_path, capsys):
    from tools.diagnose_run import main
    rc = main([str(tmp_path / "nonexistent.txt")])
    assert rc == 1


# ---------------------------------------------------------------------------
# Heartbeat underruns / partials / finals parsing
# ---------------------------------------------------------------------------

UNDERRUN_RUN_LINES = [
    "12:00:00.000 INFO  speaker | run 20260628-120010 started -> x.txt",
    "12:00:05.000 DEBUG speaker.sherpa | capture heartbeat: blocks=100 avg_rms=0.0085 clip=0.0% underruns=0 partials=1 finals=0 speaking=False",
    "12:00:07.000 DEBUG speaker.sherpa | capture heartbeat: blocks=200 avg_rms=0.0200 clip=0.1% underruns=3 partials=2 finals=1 speaking=True",
    "12:00:09.000 DEBUG speaker.sherpa | capture heartbeat: blocks=300 avg_rms=0.0100 clip=0.0% underruns=5 partials=0 finals=1 speaking=False",
]


def test_heartbeat_captures_underruns(tmp_path):
    path = _write_log(tmp_path, UNDERRUN_RUN_LINES)
    run = parse_log(path)

    assert len(run.heartbeats) == 3
    assert run.heartbeats[0].underruns == 0
    assert run.heartbeats[0].partials == 1
    assert run.heartbeats[0].finals == 0
    assert run.heartbeats[1].underruns == 3
    assert run.heartbeats[1].partials == 2
    assert run.heartbeats[1].finals == 1
    assert run.heartbeats[2].underruns == 5


def test_heartbeat_underruns_in_report(tmp_path, capsys):
    path = _write_log(tmp_path, UNDERRUN_RUN_LINES)
    from tools.diagnose_run import main
    main([str(path)])
    out = capsys.readouterr().out
    assert "playback underruns" in out
    assert "5" in out   # cumulative from last heartbeat


# ---------------------------------------------------------------------------
# pass_fail_verdict
# ---------------------------------------------------------------------------

def test_pass_fail_clean_run(tmp_path):
    from tools.diagnose_run import pass_fail_verdict
    path = _write_log(tmp_path, CLEAN_RUN_LINES)
    run = parse_log(path)
    pf = pass_fail_verdict(run)

    assert pf["self_interrupt"] == "PASS"
    assert pf["self_interrupt_suspects"] == 0
    assert pf["underruns_total"] == 0
    assert pf["underrun_verdict"] == "PASS"


def test_pass_fail_self_interrupt_fails(tmp_path):
    from tools.diagnose_run import pass_fail_verdict
    path = _write_log(tmp_path, SELF_INTERRUPT_LINES)
    run = parse_log(path)
    pf = pass_fail_verdict(run)

    assert pf["self_interrupt"] == "FAIL"
    assert pf["overall"] == "FAIL"
    assert pf["self_interrupt_suspects"] >= 1


def test_pass_fail_underrun_warn(tmp_path):
    from tools.diagnose_run import pass_fail_verdict
    path = _write_log(tmp_path, UNDERRUN_RUN_LINES)
    run = parse_log(path)
    pf = pass_fail_verdict(run)

    assert pf["underruns_total"] == 5
    assert pf["underrun_verdict"] == "WARN"
    assert pf["overall"] in ("WARN", "FAIL")


def test_pass_fail_in_json_output(tmp_path, capsys):
    path = _write_log(tmp_path, CLEAN_RUN_LINES)
    from tools.diagnose_run import main
    main([str(path), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert "pass_fail" in data
    assert data["pass_fail"]["overall"] in ("PASS", "WARN", "FAIL")


def test_exit_code_pass_returns_zero(tmp_path):
    path = _write_log(tmp_path, CLEAN_RUN_LINES)
    from tools.diagnose_run import main
    rc = main([str(path), "--exit-code"])
    assert rc == 0


def test_exit_code_fail_returns_one(tmp_path):
    path = _write_log(tmp_path, SELF_INTERRUPT_LINES)
    from tools.diagnose_run import main
    rc = main([str(path), "--exit-code"])
    assert rc == 1


def test_verdict_only_mode(tmp_path, capsys):
    path = _write_log(tmp_path, CLEAN_RUN_LINES)
    from tools.diagnose_run import main
    rc = main([str(path), "--verdict-only"])
    out = capsys.readouterr().out
    assert "OVERALL" in out
    assert "self_interrupt" in out
    assert rc == 0


# ---------------------------------------------------------------------------
# pre-first-audio barge noise detection
# ---------------------------------------------------------------------------

PRE_FIRST_AUDIO_BARGE_LINES = [
    "12:00:00.000 INFO  speaker | run 20260628-120020 started -> x.txt",
    # speaking starts (synth lead-in begins here)
    "12:00:10.000 DEBUG speaker.sherpa | speaking: 'Assistant reply.' (queue depth=0)",
    # barge REJECTED BEFORE playback is opened (pre-first-audio noise)
    "12:00:10.010 INFO  speaker.sherpa | barge-in REJECTED: 0.2s of voiced speech during playback did not trip the gate",
    # playback opens 20ms later
    "12:00:10.020 INFO  speaker.sherpa | playback opened at 24000 Hz on device default (callback)",
    "12:00:11.000 DEBUG speaker.sherpa | capture heartbeat: blocks=200 avg_rms=0.02 clip=0.0% underruns=0 partials=0 finals=0 speaking=True",
]


def test_pre_first_audio_barge_counted(tmp_path):
    from tools.diagnose_run import pass_fail_verdict
    path = _write_log(tmp_path, PRE_FIRST_AUDIO_BARGE_LINES)
    run = parse_log(path)
    pf = pass_fail_verdict(run)

    assert pf["pre_first_audio_noise"] == 1


def test_pre_first_audio_noise_in_report(tmp_path, capsys):
    path = _write_log(tmp_path, PRE_FIRST_AUDIO_BARGE_LINES)
    from tools.diagnose_run import main
    main([str(path)])
    out = capsys.readouterr().out
    assert "pre-first-audio" in out


# ---------------------------------------------------------------------------
# live_audio_ab smoke test
# ---------------------------------------------------------------------------

def test_live_audio_ab_single_run(tmp_path, capsys):
    path = _write_log(tmp_path, CLEAN_RUN_LINES)
    from tools.live_audio_ab import main as ab_main
    rc = ab_main([str(path)])
    out = capsys.readouterr().out
    assert "OVERALL" in out
    assert "first-audio latency" in out
    assert rc == 0


def test_live_audio_ab_two_runs(tmp_path, capsys):
    path_a = _write_log(tmp_path / "a" if False else tmp_path, CLEAN_RUN_LINES)
    # Need a distinct second log path - use a different name
    path_b_p = tmp_path / "run-b.txt"
    path_b_p.write_text("\n".join(SELF_INTERRUPT_LINES), encoding="utf-8")
    from tools.live_audio_ab import main as ab_main
    rc = ab_main(["--no-color", str(path_a), str(path_b_p)])
    out = capsys.readouterr().out
    assert "Side-by-Side" in out
    assert "OVERALL" in out
    assert rc == 1   # run B has a self-interrupt → FAIL


def test_live_audio_ab_missing_file(tmp_path, capsys):
    from tools.live_audio_ab import main as ab_main
    rc = ab_main([str(tmp_path / "nonexistent.txt")])
    assert rc == 1
