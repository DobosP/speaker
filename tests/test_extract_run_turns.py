"""Tests for tools.extract_run_turns using synthetic log + WAV fixtures.

No real audio, models, or git subprocesses required. Timestamps use the
HH:MM:SS.mmm format the sherpa engine emits.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

import tools.extract_run_turns as ert
from tools.extract_run_turns import (
    BundleError,
    extract_bundle,
    out_dir_refusal,
    parse_run_log,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _write_log(tmp_path: Path, lines: list[str], name: str = "run-test.txt") -> str:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def _write_wav(path: Path, samples, sample_rate: int = 16000) -> str:
    pcm = np.clip(samples, -1.0, 1.0)
    data = (pcm * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data)
    return str(path)


# Capture loop starts at 12:00:10.000 -> WAV t=0. Twenty seconds of audio.
# Turn 0: partials from 12:00:12.0, admitted final at 12:00:15.0.
# A dropped endpoint at 12:00:16.5 resets the segment; turn 1's partials start
# 12:00:17.0, admitted final at 12:00:19.0.
BUNDLE_LINES = [
    "12:00:05.000 INFO  speaker | run 20260701-120005 started (debug=True) -> logs/live/x/run-20260701-120005.txt",
    "12:00:06.000 INFO  speaker.sherpa | recording session audio -> logs/live/x/run-20260701-120005.wav",
    "12:00:10.000 INFO  speaker.sherpa | capture loop started (capture_sr=16000 -> asr_sr=16000)",
    "12:00:12.000 DEBUG speaker.sherpa | suppressing pre-VAD ASR partial: 'AND'",
    "12:00:13.000 DEBUG speaker.sherpa | asr partial: 'Hello'",
    "12:00:15.000 INFO  speaker.sherpa | asr final: 'Hello there.' (raw 'HELLO THERE')",
    "12:00:15.100 INFO  speaker.runtime | final -> brain: 'Hello there.' (mode=assistant input_generation=2)",
    "12:00:16.500 INFO  speaker.sherpa | dropping raw final 'I I' -- no sustained calibrated pre-gain speech evidence",
    "12:00:17.000 DEBUG speaker.sherpa | suppressing ASR partial without calibrated speech evidence: 'YOU' qualified=0/4 run=0/4",
    "12:00:18.000 DEBUG speaker.sherpa | asr partial: 'You there'",
    "12:00:19.000 INFO  speaker.sherpa | asr final: 'Are you there?' (raw 'YOU THERE')",
    "12:00:30.100 INFO  speaker.sherpa | recorded 20.0s of session audio -> logs/live/x/run-20260701-120005.wav",
]


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def test_parse_run_log_timeline(tmp_path: Path):
    parsed = parse_run_log(_write_log(tmp_path, BUNDLE_LINES))
    assert parsed.run_id == "run-20260701-120005"
    assert parsed.capture_t0 == pytest.approx(12 * 3600 + 10.0)
    assert parsed.close_t == pytest.approx(12 * 3600 + 30.1)
    assert parsed.recorded_sec == pytest.approx(20.0)
    assert len(parsed.turns) == 2

    first, second = parsed.turns
    assert first.admitted_text == "Hello there."
    assert first.streaming_raw == "HELLO THERE"
    # Segment starts at the capture loop; onset is the first partial activity.
    assert first.segment_start_t == pytest.approx(12 * 3600 + 10.0)
    assert first.onset_t == pytest.approx(12 * 3600 + 12.0)
    assert first.final_t == pytest.approx(12 * 3600 + 15.0)

    # The dropped final is an endpoint: it bounds the second turn's segment
    # and resets onset tracking (a suppressed partial after it counts).
    assert second.segment_start_t == pytest.approx(12 * 3600 + 16.5)
    assert second.onset_t == pytest.approx(12 * 3600 + 17.0)
    assert second.final_t == pytest.approx(12 * 3600 + 19.0)
    assert second.admitted_text == "Are you there?"


def test_parse_run_log_midnight_rollover(tmp_path: Path):
    lines = [
        "23:59:59.500 INFO  speaker.sherpa | capture loop started (capture_sr=16000 -> asr_sr=16000)",
        "00:00:01.000 DEBUG speaker.sherpa | asr partial: 'Hi'",
        "00:00:02.000 INFO  speaker.sherpa | asr final: 'Hi.' (raw 'HI')",
    ]
    parsed = parse_run_log(_write_log(tmp_path, lines))
    assert parsed.capture_t0 == pytest.approx(23 * 3600 + 59 * 60 + 59.5)
    turn = parsed.turns[0]
    assert turn.final_t == pytest.approx(86400 + 2.0)
    assert turn.final_t - parsed.capture_t0 == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _bundle(tmp_path: Path) -> tuple[str, np.ndarray]:
    log_path = _write_log(tmp_path, BUNDLE_LINES)
    # 20 s WAV whose value ramps by second so windows are content-checkable.
    sr = 16000
    samples = np.repeat(np.linspace(0.0, 0.19, 20, dtype=np.float32), sr)
    _write_wav(tmp_path / "run-test.wav", samples)
    return log_path, samples


def test_extract_bundle_end_to_end(tmp_path: Path):
    log_path, samples = _bundle(tmp_path)
    out = tmp_path / "out"
    manifest = extract_bundle(
        log_path,
        out_dir=str(out),
        lead_pad_sec=3.0,
        repo_root=str(tmp_path / "fake-repo"),
    )
    assert manifest["run_id"] == "run-20260701-120005"
    assert manifest["wav_t0_anchor"] == "capture_loop_started"
    # close-line anchor: 12:00:30.1 - 20.0s = 12:00:10.1 -> +0.1s delta.
    assert manifest["close_anchor_delta_sec"] == pytest.approx(0.1)
    clips = manifest["clips"]
    assert [c["id"] for c in clips] == ["turn-00", "turn-01"]

    # Turn 0: onset 12:00:12 - 3.0 lead pad = 12:00:09 clamps to segment start
    # (capture t0) -> wav 0.0; end = final 12:00:15 -> wav 5.0.
    assert clips[0]["start_sec"] == pytest.approx(0.0)
    assert clips[0]["end_sec"] == pytest.approx(5.0)
    assert clips[0]["admitted_text"] == "Hello there."
    assert clips[0]["streaming_raw"] == "HELLO THERE"

    # Turn 1: segment start = dropped final 12:00:16.5 -> wav 6.5 (onset 7.0
    # minus lead pad clamps to it); end = final 12:00:19 -> wav 9.0.
    assert clips[1]["start_sec"] == pytest.approx(6.5)
    assert clips[1]["end_sec"] == pytest.approx(9.0)

    # Clip files exist, carry the exact source samples, and are hash-pinned.
    from core.engines.file_replay import load_waveform
    from tools.recorded_stt_eval import _sha256_file

    for clip in clips:
        path = out / clip["file"]
        assert path.exists()
        assert clip["sha256"] == _sha256_file(path)
        audio, sr = load_waveform(str(path))
        assert sr == 16000
        i0 = int(round(clip["start_sec"] * sr))
        i1 = int(round(clip["end_sec"] * sr))
        expected = np.clip(samples[i0:i1], -1.0, 1.0)
        np.testing.assert_allclose(audio, expected, atol=2.0 / 32767.0)

    # The manifest is written mode 600 and the out dir self-ignores.
    manifest_path = out / "run-20260701-120005-turns.json"
    assert manifest_path.exists()
    assert (manifest_path.stat().st_mode & 0o777) == 0o600
    assert (out / ".gitignore").read_text(encoding="utf-8").strip().endswith("*")
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["clips"] == clips


def test_extract_bundle_close_line_anchor_fallback(tmp_path: Path):
    lines = [line for line in BUNDLE_LINES if "capture loop started" not in line]
    log_path = _write_log(tmp_path, lines)
    sr = 16000
    _write_wav(tmp_path / "run-test.wav", np.zeros(20 * sr, dtype=np.float32))
    out = tmp_path / "fallback-out"
    manifest = extract_bundle(
        log_path, out_dir=str(out), repo_root=str(tmp_path / "fake-repo")
    )
    assert manifest["wav_t0_anchor"] == "recorded_close_line"
    # t0 = 12:00:30.1 - 20.0 = 12:00:10.1; final 12:00:15 -> wav 4.9.
    assert manifest["clips"][0]["end_sec"] == pytest.approx(4.9)


def test_extract_bundle_dry_run_writes_nothing(tmp_path: Path):
    log_path, _ = _bundle(tmp_path)
    out = tmp_path / "dry-out"
    manifest = extract_bundle(
        log_path,
        out_dir=str(out),
        dry_run=True,
        repo_root=str(tmp_path / "fake-repo"),
    )
    assert len(manifest["clips"]) == 2
    assert not out.exists()


def test_extract_bundle_requires_admitted_turns(tmp_path: Path):
    lines = [line for line in BUNDLE_LINES if "asr final:" not in line]
    log_path = _write_log(tmp_path, lines)
    with pytest.raises(BundleError):
        extract_bundle(log_path, repo_root=str(tmp_path / "fake-repo"))


def test_extract_bundle_requires_wav(tmp_path: Path):
    log_path = _write_log(tmp_path, BUNDLE_LINES)
    with pytest.raises(BundleError):
        extract_bundle(log_path, repo_root=str(tmp_path / "fake-repo"))


# ---------------------------------------------------------------------------
# Output-dir privacy guard
# ---------------------------------------------------------------------------

def test_out_dir_refusal_outside_repo_is_allowed(tmp_path: Path):
    assert (
        out_dir_refusal(
            str(tmp_path / "elsewhere"),
            "m.json",
            repo_root=str(tmp_path / "repo"),
            check_ignore=lambda root, probe: False,
        )
        is None
    )


def test_out_dir_refusal_inside_repo_requires_ignore(tmp_path: Path):
    repo = tmp_path / "repo"
    inside = repo / "docs" / "clips"
    probes = []

    def fake_check(root, probe):
        probes.append((root, probe))
        return False

    reason = out_dir_refusal(
        str(inside), "m.json", repo_root=str(repo), check_ignore=fake_check
    )
    assert reason is not None and "not gitignored" in reason
    # The guard probes the manifest (the least-ignorable artifact).
    assert probes and probes[0][1].endswith("m.json")

    assert (
        out_dir_refusal(
            str(inside),
            "m.json",
            repo_root=str(repo),
            check_ignore=lambda root, probe: True,
        )
        is None
    )


def test_out_dir_refusal_fails_closed_without_git(tmp_path: Path):
    reason = out_dir_refusal(
        str(tmp_path / "repo" / "clips"),
        "m.json",
        repo_root=str(tmp_path / "repo"),
        check_ignore=lambda root, probe: None,
    )
    assert reason is not None and "fail-closed" in reason


def test_extract_bundle_refuses_unignored_out_dir(tmp_path: Path, monkeypatch):
    log_path, _ = _bundle(tmp_path)
    monkeypatch.setattr(ert, "_git_check_ignore", lambda root, probe: False)
    out = tmp_path / "clips"
    with pytest.raises(SystemExit) as excinfo:
        extract_bundle(log_path, out_dir=str(out), repo_root=str(tmp_path))
    assert excinfo.value.code == 2
    assert not out.exists()
