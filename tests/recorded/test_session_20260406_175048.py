"""
Auto-generated regression tests for session session_20260406_175048.

DO NOT EDIT MANUALLY.  Re-generate with::

    python scripts/generate_session_tests.py --session session_20260406_175048

Each test replays the mic audio received during a TTS turn through the barge-in
pipeline and asserts the expected behaviour (no self-interrupt / fires on user
speech).
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from tests.harness import AudioHarness, make_recorder

SESSION_DIR = Path(__file__).parent.parent.parent / "recordings" / "session_20260406_175048"
GATE_SEC = 0.35     # must match production barge_in_min_delay_after_ref_sec
SR = 16_000
CHUNK_SEC = 1024 / SR

def test_turn_000_no_self_interrupt():
    """
    Turn 0: TTS played — no barge-in must fire during replay.

    Recorded TTS text: ""
    Session mic RMS mean: 0.0108, echo_sim mean: 0.238

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
    Session mic RMS mean: 0.0088, echo_sim mean: 0.285

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
    Session mic RMS mean: 0.0082, echo_sim mean: 0.199

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
    Session mic RMS mean: 0.0086, echo_sim mean: 0.258

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

@pytest.mark.xfail(
    reason=(
        "Turn 4: barge-in classification uncertain. "
        "Annotate turn.json with annotation=\"false_positive\" or \"true_positive\" "
        "then re-run: python scripts/generate_session_tests.py"
    ),
    strict=False,
)
def test_turn_004_barge_in_candidate():
    """
    Turn 4: barge-in fired but classification is uncertain (needs annotation).

    TTS text: ""
    Barge-in: RMS=0.0401, echo_sim=0.00,
              voiced=True

    Edit recordings/session_20260406_175048/turn_004/turn.json and set
    "annotation": "false_positive" or "true_positive", then regenerate tests.
    """
    tts_path = SESSION_DIR / "turn_004" / "tts_16k.npy"
    mic_path = SESSION_DIR / "turn_004" / "mic_16k.npy"
    if not tts_path.exists() or not mic_path.exists():
        pytest.skip("Audio files not found")

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

    # This test is xfail — we do not enforce an outcome until classified.
    # Remove xfail marker and pick the right assert once annotated.
    assert False, "Awaiting human annotation in turn.json"
