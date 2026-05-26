"""Tests for scripts/check_latency_slo.py bottleneck + SLO checks."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_latency_slo.py"


def test_bottleneck_llm_heavy():
    from scripts.check_latency_slo import bottleneck_label

    m = {
        "speech_detected_to_stt_final_ms": 500.0,
        "memory_ready_to_first_llm_sentence_ms": 9000.0,
    }
    assert "LLM" in bottleneck_label(m)


def test_bottleneck_stt_heavy():
    from scripts.check_latency_slo import bottleneck_label

    m = {
        "speech_detected_to_stt_final_ms": 4000.0,
        "memory_ready_to_first_llm_sentence_ms": 500.0,
    }
    assert "STT" in bottleneck_label(m)


def test_check_turn_metrics_fails_over_budget():
    from scripts.check_latency_slo import check_turn_metrics

    fails, ok = check_turn_metrics(
        {
            "stt_final_to_first_llm_sentence_ms": 3000.0,
            "speech_detected_to_first_audio_ms": 3500.0,
        }
    )
    assert any("FAIL" in x for x in fails)


def test_cli_demo_exits_nonzero():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--demo"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "Bottleneck" in r.stdout


def test_cli_map_zero():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--map"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "stt_final_to_first_llm_sentence_ms" in r.stdout
