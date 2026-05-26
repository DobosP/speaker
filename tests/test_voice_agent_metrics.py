"""2025 voice-agent metrics: barge-in latency SLO and false-barge-in rate.

Recent voice-agent evaluation practice goes beyond WER to interaction metrics:
barge-in detection latency (<200 ms target) and false barge-ins under noise.
These drive the REAL recorder + VAD + echo path through ConversationRunner and
aggregate over several trials, reporting p50/p95 latency and the false-positive
rate.

Requires the neural VAD stack (silero/torch); skips cleanly when it is absent.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

try:  # the barge-in path relies on the silero (torch) VAD
    import torch  # noqa: F401

    _HAS_VAD = True
except Exception:  # pragma: no cover - env guard
    _HAS_VAD = False

from tests.conversation_harness import ConversationRunner
from tests.fixtures import (
    SR,
    babble_noise,
    mix,
    pad_to,
    real_speech,
    real_tts_echo,
    tv_noise,
    voiced_speech,
)

pytestmark = [
    pytest.mark.audio,
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_VAD, reason="neural VAD (silero/torch) unavailable"),
]

_TRIALS = 5
_BARGE_IN_LATENCY_SLO_SEC = 0.20  # 200 ms


def _measure_barge_in_latency(monkeypatch, tmp_path) -> float | None:
    monkeypatch.chdir(tmp_path)
    with ConversationRunner(
        transcripts=["tell me a long story"],
        llm_responses=[["This answer keeps playing while the user interrupts."]],
        tts_playback_sec=0.65,
        recorder_kwargs={"barge_in_min_speech_sec": 0.08},
    ) as runner:
        runner.inject_user_turn(
            voiced_speech(0.4, amplitude=0.35),
            trailing_silence_sec=0.35,
            inter_chunk_delay=0.02,
        )
        if not runner.wait_for_tts_start():
            return None
        echo = runner.player.current_audio
        if echo is None:
            return None
        user = real_speech(0.45, amplitude=0.35, fallback_amplitude=0.42)
        mic = mix(pad_to(user, len(echo)), echo, snr_db=8.0)
        runner.inject_barge_in(mic, inter_chunk_delay=0.01)
        runner.wait_for_response()

    stop_at = runner.timeline.first_time("tts_stop")
    barge_at = runner.timeline.first_time("barge_in")
    if stop_at is None or barge_at is None:
        return None
    return stop_at - barge_at


def test_barge_in_latency_slo(monkeypatch, tmp_path):
    latencies = []
    for _ in range(_TRIALS):
        lat = _measure_barge_in_latency(monkeypatch, tmp_path)
        if lat is not None:
            latencies.append(lat)

    assert latencies, "no barge-in fired across trials; VAD path may be misconfigured"
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    print(json.dumps({"n": len(latencies), "p50_ms": p50 * 1e3, "p95_ms": p95 * 1e3}))
    assert p95 < _BARGE_IN_LATENCY_SLO_SEC, f"p95 barge-in latency {p95*1e3:.0f}ms exceeds SLO"


def test_false_barge_in_rate_under_noise(monkeypatch, tmp_path):
    false_positives = 0
    for _ in range(_TRIALS):
        monkeypatch.chdir(tmp_path)
        with ConversationRunner(
            transcripts=["unused"],
            llm_responses=[],
            tts_playback_sec=0.4,
            recorder_kwargs={"barge_in_min_delay_after_ref_sec": 0.20},
        ) as runner:
            echo = real_tts_echo(0.5, amplitude=0.08)
            noise = mix(
                tv_noise(0.5, amplitude=0.05),
                babble_noise(0.5, amplitude=0.03),
                snr_db=3.0,
            )
            runner.harness.set_tts_speaking(audio_ref=echo, zero_delays=False)
            runner.inject_barge_in(pad_to(noise, len(echo)), inter_chunk_delay=1024 / SR)
        if runner.interrupts:
            false_positives += 1

    rate = false_positives / _TRIALS
    print(json.dumps({"trials": _TRIALS, "false_barge_ins": false_positives, "rate": rate}))
    # Ambient TV/babble during playback must not be heard as an interruption.
    assert rate == 0.0, f"false barge-in rate {rate:.2f} (>0) under non-speech noise"
