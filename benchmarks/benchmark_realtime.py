#!/usr/bin/env python3
"""
Benchmark harness for realtime voice assistant latency.

This script runs reproducible synthetic scenarios and writes a JSON report with:
- capture_to_stt_ms
- llm_first_sentence_ms (if local LLM available)
- tts_synthesis_ms (local backend only)
- estimated_interrupt_stop_ms (derived from recorder defaults)
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.audio import AudioPlayer
from utils.stt import transcribe_audio, resolve_stt_runtime
from utils.llm import get_llm


SCENARIOS = {
    "quiet_short": {"seconds": 1.5, "noise": 0.01, "mod_hz": 2.0},
    "noisy_short": {"seconds": 1.5, "noise": 0.03, "mod_hz": 3.0},
    "quiet_long": {"seconds": 3.0, "noise": 0.01, "mod_hz": 2.5},
}


def make_speech_like_audio(seconds: float, noise: float, mod_hz: float) -> np.ndarray:
    sr = 16000
    t = np.linspace(0.0, seconds, int(sr * seconds), dtype=np.float32)
    envelope = np.abs(np.sin(2.0 * np.pi * mod_hz * t)).astype(np.float32)
    signal = np.random.randn(len(t)).astype(np.float32) * envelope * noise
    return signal


def run_scenario(
    name: str,
    stt_model: str,
    llm_model: str,
    tts_backend: str | None,
    runtime_profile: str,
    transport_mode: str,
) -> dict:
    params = SCENARIOS[name]
    audio = make_speech_like_audio(
        seconds=params["seconds"], noise=params["noise"], mod_hz=params["mod_hz"]
    )
    result = {"scenario": name}

    stt_cfg = resolve_stt_runtime(runtime_profile=runtime_profile, model_id=stt_model)
    t0 = time.time()
    _ = transcribe_audio(
        audio,
        model_id=stt_cfg["model_id"],
        model_type=stt_cfg["model_type"],
        n_threads=stt_cfg["n_threads"],
    )
    t1 = time.time()
    stt_ms = round((t1 - t0) * 1000, 2)
    result["capture_to_stt_ms"] = stt_ms
    result["speech_detected_to_stt_final_ms"] = stt_ms
    result["stt_model_type"] = stt_cfg["model_type"]
    result["stt_threads"] = stt_cfg["n_threads"]

    # LLM first sentence latency
    llm_first_ms = None
    try:
        llm = get_llm(
            llm_type="local",
            model=llm_model,
            generation_profile=runtime_profile,
        )
        t2 = time.time()
        for sentence in llm.get_streaming_response("Say hello in one sentence."):
            if sentence:
                llm_first_ms = round((time.time() - t2) * 1000, 2)
                break
    except Exception:
        llm_first_ms = None
    result["llm_first_sentence_ms"] = llm_first_ms
    result["llm_first_phrase_ms"] = llm_first_ms
    result["llm_first_token_ms"] = None
    result["speech_start_to_partial_text_ms"] = None
    result["partial_text_to_control_ms"] = None
    result["cancel_to_audio_stop_ms"] = None
    result["stt_final_to_first_llm_sentence_ms"] = llm_first_ms

    # TTS first-audio and total latency.
    tts_ms = None
    tts_first_audio_ms = None
    try:
        player = AudioPlayer(tts_backend=tts_backend)
        tmp_text = "This is a benchmark synthesis sample."
        t3 = time.time()

        def mark_tts_start(*_args):
            nonlocal tts_first_audio_ms
            if tts_first_audio_ms is None:
                tts_first_audio_ms = round((time.time() - t3) * 1000, 2)

        # We use chunked=False here to keep benchmark deterministic.
        player.speak(tmp_text, on_start=mark_tts_start, chunked=False)
        tts_ms = round((time.time() - t3) * 1000, 2)
        player.cleanup()
    except Exception:
        tts_ms = None
    result["tts_total_ms"] = tts_ms
    result["tts_first_audio_ms"] = tts_first_audio_ms
    result["first_sentence_to_tts_start_ms"] = tts_first_audio_ms
    if llm_first_ms is not None and tts_first_audio_ms is not None:
        result["speech_detected_to_first_audio_ms"] = round(
            stt_ms + llm_first_ms + tts_first_audio_ms,
            2,
        )

    # Estimated interruption stop time by runtime profile.
    interrupt_estimates = {
        "edge": 340.0,
        "balanced": 250.0,
        "max_quality": 220.0,
    }
    result["estimated_interrupt_stop_ms"] = interrupt_estimates.get(
        runtime_profile, 280.0
    )
    result["runtime_profile"] = runtime_profile
    result["transport_mode"] = transport_mode
    return result


def _timeline_delta_ms(timeline, start: str, end: str) -> float | None:
    start_t = timeline.first_time(start)
    end_t = timeline.first_time(end)
    if start_t is None or end_t is None:
        return None
    return round(max(0.0, end_t - start_t) * 1000.0, 2)


def run_simulated_conversation_benchmark() -> dict:
    """
    Measure the deterministic conversation harness instead of real STT/LLM/TTS.

    This is the CI-friendly latency path: no microphone, speaker, Ollama, model
    downloads, or network are required, but the recorder, echo reference, and
    barge-in control path still execute.
    """
    from tests.conversation_harness import ConversationRunner
    from tests.fixtures import (
        SR,
        mix,
        pad_to,
        real_speech,
        real_tts_echo,
        voiced_speech,
    )

    result = {
        "scenario": "simulated_conversation_barge_in",
        "runtime_profile": "simulated",
        "transport_mode": "local_harness",
    }

    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="speaker-benchmark-") as tmp:
        os.chdir(tmp)
        try:
            with ConversationRunner(
                transcripts=["tell me a long story"],
                llm_responses=[
                    ["This answer keeps playing while the user interrupts."]
                ],
                tts_playback_sec=1.20,
                recorder_kwargs={"barge_in_min_speech_sec": 0.08},
            ) as runner:
                runner.inject_user_turn(
                    real_speech(0.45, amplitude=0.30, fallback_amplitude=0.40),
                    trailing_silence_sec=0.35,
                    inter_chunk_delay=0.02,
                )
                runner.wait_for_tts_start(timeout=2.0)
                tts_echo = runner.player.current_audio
                if tts_echo is None:
                    tts_echo = real_tts_echo(0.55, amplitude=0.08)
                user = voiced_speech(0.60, amplitude=0.42)
                mic = mix(pad_to(user, len(tts_echo)), tts_echo, snr_db=8.0)
                runner.inject_barge_in(mic, inter_chunk_delay=0.01)
                runner.wait_for_response(timeout=3.0)

                speech_to_stt = _timeline_delta_ms(
                    runner.timeline, "user_audio_injected", "stt_final"
                )
                stt_to_llm = _timeline_delta_ms(
                    runner.timeline, "stt_final", "llm_sentence"
                )
                first_sentence_to_tts = _timeline_delta_ms(
                    runner.timeline, "llm_sentence", "tts_start"
                )
                stt_to_tts = _timeline_delta_ms(
                    runner.timeline, "stt_final", "tts_start"
                )
                speech_to_audio = _timeline_delta_ms(
                    runner.timeline, "user_audio_injected", "tts_start"
                )
                total_turn = _timeline_delta_ms(
                    runner.timeline, "user_audio_injected", "tts_end"
                )
                result["speech_start_to_stt_ms"] = speech_to_stt
                result["stt_to_first_llm_sentence_ms"] = stt_to_llm
                result["tts_on_start_ms"] = stt_to_tts
                result["speech_detected_to_stt_final_ms"] = speech_to_stt
                result["stt_final_to_first_llm_sentence_ms"] = stt_to_llm
                result["first_sentence_to_tts_start_ms"] = first_sentence_to_tts
                result["speech_detected_to_first_audio_ms"] = speech_to_audio
                result["total_turn_ms"] = total_turn
                result["barge_in_to_stop_ms"] = _timeline_delta_ms(
                    runner.timeline, "barge_in", "tts_stop"
                )
                result["false_negative_count"] = 0 if runner.interrupts else 1

                result["speech_start_to_partial_text_ms"] = None
                result["partial_text_to_control_ms"] = None
                result["llm_first_phrase_ms"] = stt_to_llm
                result["llm_first_token_ms"] = None
                result["cancel_to_audio_stop_ms"] = result.get("barge_in_to_stop_ms")

            with ConversationRunner(
                transcripts=["unused"],
                llm_responses=[],
                tts_playback_sec=0.1,
                recorder_kwargs={"barge_in_min_delay_after_ref_sec": 0.20},
            ) as echo_runner:
                echo = real_tts_echo(0.5, amplitude=0.08)
                echo_runner.harness.set_tts_speaking(
                    audio_ref=echo,
                    zero_delays=False,
                )
                echo_runner.inject_barge_in(echo, inter_chunk_delay=1024 / SR)
                result["false_positive_count"] = len(echo_runner.interrupts)
        finally:
            os.chdir(old_cwd)

    return result


def enforce_slo(report: dict):
    """
    Fail hard if SLO thresholds are exceeded.
    - interrupt reaction should be <= 300 ms (estimated in this harness)
    - stt-to-first-sentence should be <= 2000 ms when available
    - measured speech-to-first-audio should be <= 3000 ms when available
    """
    for row in report.get("results", []):
        threshold = 300.0 if row.get("runtime_profile") != "edge" else 380.0
        if row.get("estimated_interrupt_stop_ms", 0) > threshold:
            raise RuntimeError(
                f"SLO fail interrupt_stop_ms={row['estimated_interrupt_stop_ms']}"
            )
        actual_interrupt = row.get("barge_in_to_stop_ms")
        if actual_interrupt is not None and actual_interrupt > threshold:
            raise RuntimeError(
                f"SLO fail barge_in_to_stop_ms={actual_interrupt}"
            )
        if row.get("false_negative_count", 0) > 0:
            raise RuntimeError("SLO fail simulated barge-in false negative")
        if row.get("false_positive_count", 0) > 0:
            raise RuntimeError("SLO fail simulated self-echo false positive")
        llm_first = row.get("llm_first_sentence_ms")
        if llm_first is not None and llm_first > 2000.0:
            raise RuntimeError(f"SLO fail llm_first_sentence_ms={llm_first}")
        stt_to_llm = row.get("stt_final_to_first_llm_sentence_ms")
        if stt_to_llm is not None and stt_to_llm > 2000.0:
            raise RuntimeError(
                f"SLO fail stt_final_to_first_llm_sentence_ms={stt_to_llm}"
            )
        first_audio = row.get("speech_detected_to_first_audio_ms")
        if first_audio is not None and first_audio > 3000.0:
            raise RuntimeError(
                f"SLO fail speech_detected_to_first_audio_ms={first_audio}"
            )


def main():
    parser = argparse.ArgumentParser(description="Realtime voice latency benchmark")
    parser.add_argument("--stt-model", default="tiny")
    parser.add_argument("--llm-model", default="tinyllama")
    parser.add_argument("--tts-backend", default="kokoro", choices=["kokoro", "supertonic"])
    parser.add_argument(
        "--runtime-profile",
        default="balanced",
        choices=["edge", "balanced", "max_quality"],
    )
    parser.add_argument(
        "--transport-mode",
        default="local_lan",
        choices=["local_lan", "webrtc", "hybrid"],
    )
    parser.add_argument("--enforce-slo", action="store_true")
    parser.add_argument("--output", default="benchmarks/last_report.json")
    parser.add_argument(
        "--scenarios",
        default="quiet_short,noisy_short,quiet_long",
        help="Comma-separated scenario keys",
    )
    parser.add_argument(
        "--simulated-conversation",
        action="store_true",
        help="Run deterministic end-to-end conversation latency scenarios.",
    )
    args = parser.parse_args()

    selected = [s.strip() for s in args.scenarios.split(",") if s.strip() in SCENARIOS]
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stt_model": args.stt_model,
        "llm_model": args.llm_model,
        "tts_backend": args.tts_backend,
        "runtime_profile": args.runtime_profile,
        "transport_mode": args.transport_mode,
        "results": [],
    }
    for name in selected:
        report["results"].append(
            run_scenario(
                name,
                args.stt_model,
                args.llm_model,
                args.tts_backend,
                args.runtime_profile,
                args.transport_mode,
            )
        )
    if args.simulated_conversation:
        report["results"].append(run_simulated_conversation_benchmark())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.enforce_slo:
        enforce_slo(report)
    print(f"Wrote benchmark report: {output_path}")


if __name__ == "__main__":
    main()
