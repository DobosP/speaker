"""
Auto-generated regression tests for session session_20260406_170433.

DO NOT EDIT MANUALLY.  Re-generate with::

    python scripts/generate_session_tests.py --session session_20260406_170433

Each test replays the mic audio received during a TTS turn through the barge-in
pipeline and asserts the expected behaviour (no self-interrupt / fires on user
speech).
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from tests.harness import AudioHarness, make_recorder

SESSION_DIR = Path(__file__).parent.parent.parent / "recordings" / "session_20260406_170433"
GATE_SEC = 0.35     # must match production barge_in_min_delay_after_ref_sec
SR = 16_000
CHUNK_SEC = 1024 / SR

def test_turn_000_no_self_interrupt():
    """
    Turn 0: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0096, echo_sim mean: 0.276

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_000" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_000" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 0: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )

def test_turn_001_no_self_interrupt():
    """
    Turn 1: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0111, echo_sim mean: 0.242

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_001" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_001" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 1: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )

def test_turn_002_no_self_interrupt():
    """
    Turn 2: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0148, echo_sim mean: 0.183

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_002" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_002" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 2: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )

def test_turn_003_no_self_interrupt():
    """
    Turn 3: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0112, echo_sim mean: 0.219

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_003" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_003" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 3: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )

def test_turn_004_no_self_interrupt():
    """
    Turn 4: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0125, echo_sim mean: 0.216

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_004" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_004" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 4: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )

def test_turn_005_no_self_interrupt():
    """
    Turn 5: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0181, echo_sim mean: 0.231

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_005" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_005" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 5: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )

def test_turn_006_no_self_interrupt():
    """
    Turn 6: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0134, echo_sim mean: 0.242

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_006" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_006" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 6: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )

def test_turn_007_no_self_interrupt():
    """
    Turn 7: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0250, echo_sim mean: 0.215

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_007" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_007" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 7: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )

def test_turn_008_no_self_interrupt():
    """
    Turn 8: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0161, echo_sim mean: 0.195

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_008" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_008" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 8: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )

def test_turn_009_no_self_interrupt():
    """
    Turn 9: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0166, echo_sim mean: 0.238

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_009" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_009" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 9: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )

def test_turn_010_no_self_interrupt():
    """
    Turn 10: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0564, echo_sim mean: 0.225

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_010" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_010" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 10: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )

def test_turn_011_no_self_interrupt():
    """
    Turn 11: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0155, echo_sim mean: 0.191

    This turn was recorded without a barge-in (or the barge-in was annotated as
    a false positive).  The test guards against regressions that re-introduce
    self-interruption.
    """
    tts_path = SESSION_DIR / "turn_011" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_011" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found — session may have been recorded without audio capture")

    tts = np.load(str(tts_path))
    mic = np.load(str(mic_path))

    interrupts = []
    rec = make_recorder(
        on_interrupt=lambda info=None: interrupts.append(info),
        barge_in_min_delay_after_ref_sec=GATE_SEC,
        aec_enabled=False,
    )
    rec._noise_floor = 0.005

    with AudioHarness(rec) as h:
        h.set_tts_speaking(audio_ref=tts, zero_delays=False)
        h.inject(mic, inter_chunk_delay=CHUNK_SEC)
        h.drain(timeout=10.0)

    assert len(interrupts) == 0, (
        f"Turn 11: self-interruption regression detected. "
        f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
        f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
    )
