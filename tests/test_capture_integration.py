"""Integration tests for the capture stack end to end.

Proves the pieces work *together*: setup_logging + SystemMonitor + the real
VoiceRuntime (brain -> task -> capability -> TTS) -> transcript + per-turn
metrics + system block land in run-<id>.summary.json, and the full app.main
wiring produces the same bundle. No audio hardware, no Ollama, no models.
"""
from __future__ import annotations

import io
import json
import logging
import sys

import pytest

import core.app as app
from core.engines.scripted import ScriptedEngine
from core.llm import EchoLLM
from core.runlog import setup_logging
from core.runtime import VoiceRuntime
from core.sysinfo import SystemMonitor


@pytest.fixture(autouse=True)
def _restore_speaker_logger():
    """setup_logging()/app.main() reconfigure the shared 'speaker' logger; snap
    and restore so these tests don't clobber the conftest test-log handler."""
    lg = logging.getLogger("speaker")
    saved = (list(lg.handlers), lg.level, lg.propagate)
    try:
        yield
    finally:
        lg.handlers[:] = saved[0]
        lg.setLevel(saved[1])
        lg.propagate = saved[2]


def test_runtime_turn_populates_bundle(tmp_path):
    runlog = setup_logging(log_dir=str(tmp_path), run_id="rt", console=False)
    runlog.summary.note(engine="console", llm="echo")
    monitor = SystemMonitor(runlog.summary, interval=10_000)
    monitor.start()

    engine = ScriptedEngine()
    runtime = VoiceRuntime(engine, EchoLLM(), stream_tts=True)
    runtime.start(run_bus=False)
    engine.final("what is the capital of france")
    runtime.wait_idle()
    runtime.stop()

    monitor.stop()
    runlog.finalize([r.as_dict() for r in runtime.metrics.records()])

    data = json.loads((tmp_path / "run-rt.summary.json").read_text(encoding="utf-8"))
    roles = [t["role"] for t in data["transcript"]]
    assert "user" in roles and "assistant" in roles
    assert data["transcript"][0]["text"] == "what is the capital of france"
    assert all("at_sec" in t for t in data["transcript"])
    assert data["counts"]["turns"] >= 1
    assert "system" in data and "baseline" in data["system"]
    # The .txt trace shows the brain's decision + task lifecycle.
    text = (tmp_path / "run-rt.txt").read_text(encoding="utf-8")
    assert "final -> brain" in text
    assert "task" in text and "started" in text


def test_app_main_console_writes_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEAKER_RUN_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello assistant\nstop\n"))

    rc = app.main(["--engine", "console", "--llm", "echo"])
    assert rc == 0

    summaries = list(tmp_path.glob("run-*.summary.json"))
    assert len(summaries) == 1
    data = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert data["meta"]["engine"] == "console"
    assert data["meta"]["llm"] == "echo"
    assert any(t["role"] == "user" and t["text"] == "hello assistant" for t in data["transcript"])
    assert "system" in data
    # The matching .txt log is in the same bundle.
    assert list(tmp_path.glob("run-*.txt"))


def test_app_main_record_ignored_on_console_engine(tmp_path, monkeypatch):
    # ScriptedEngine has no recorder -> --record should warn, not crash.
    monkeypatch.setenv("SPEAKER_RUN_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    rc = app.main(["--engine", "console", "--llm", "echo", "--record"])
    assert rc == 0

    data = json.loads(next(tmp_path.glob("run-*.summary.json")).read_text(encoding="utf-8"))
    assert data["counts"]["warnings"] >= 1
    assert any("record ignored" in e["message"].lower() for e in data["errors"])
    assert "recording" not in data["meta"]


def test_sherpa_without_models_fails_fast_with_fix(tmp_path, monkeypatch):
    # The shipped config has empty sherpa paths; selecting sherpa must exit with
    # the setup_models instruction rather than starting a deaf engine.
    monkeypatch.setenv("SPEAKER_RUN_LOG_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        app.main(["--engine", "sherpa", "--llm", "echo"])
    msg = str(exc.value)
    assert "tools.setup_models" in msg
    assert "sherpa models" in msg.lower()
    assert "config.local.json" in msg
    # Preflight runs before the main runtime try/finally, but must still close
    # telemetry and finalize the diagnostic bundle.
    assert len(list(tmp_path.glob("run-*.summary.json"))) == 1


def test_sherpa_readiness_failure_is_actionable_and_echo_mode_is_forwarded():
    from tools.doctor import Check

    captured = {}

    def checks(config, **kwargs):
        captured.update(kwargs)
        return [Check("word-cut VAD model", False, "vad_model missing", "setup vad")]

    with pytest.raises(SystemExit) as exc:
        app._require_sherpa_runtime_ready(
            {"sherpa": {}}, "echo", run_checks=checks
        )
    text = str(exc.value)
    assert "runtime preflight failed" in text
    assert "word-cut VAD model" in text
    assert "setup vad" in text
    assert captured["resolved"] is True
    assert captured["llm_mode"] == "echo"


def test_sherpa_readiness_forwards_virtual_binder_only_when_supplied():
    from tools.doctor import Check

    binder = object()
    captured = {}

    def checks(_config, **kwargs):
        captured.update(kwargs)
        return [Check("ready", True)]

    app._require_sherpa_runtime_ready(
        {"sherpa": {}},
        "echo",
        run_checks=checks,
        virtual_audio_binder=binder,
    )

    assert captured["virtual_audio_binder"] is binder


def test_virtual_delay_profile_is_in_memory_and_drops_owner_identity():
    original = {
        "sherpa": {
            "input_device": "real-mic",
            "output_device": "real-speaker",
            "aec_enabled": True,
            "barge_word_cut_enabled": False,
            "barge_word_cut_require_speaker": True,
            "speaker_embedding_model": "/private/model.onnx",
            "speaker_enroll_embedding": "/private/enrollment.json",
        },
        "llm": {
            "host": "https://remote-ollama.invalid",
            "live_routing": True,
            "cloud": {"enabled": True, "strategy": "hedge"},
        },
        "web_search": {"enabled": True},
        "screen_capture": {"enabled": True, "memorize": True},
        "gui_actions": {"enabled": True},
        "watch": {"enabled": True, "grants": ["private-window"]},
        "memory": {
            "backend": "postgres",
            "embeddings": True,
            "profile_enabled": True,
            "procedural_enabled": True,
            "cross_session_continuity": True,
            "persist_assistant": True,
        },
        "agent_brain": {"local_only": False, "offline": False, "os_mode": True},
        "other": {"kept": True},
    }

    isolated = app._apply_autotest_virtual_delay_profile(original)

    assert original["sherpa"]["input_device"] == "real-mic"
    assert isolated["other"] == {"kept": True}
    assert isolated["sherpa"] | {
        "input_device": "pipewire",
        "output_device": "pipewire",
    } == isolated["sherpa"]
    assert isolated["sherpa"]["aec_enabled"] is False
    assert isolated["sherpa"]["barge_word_cut_enabled"] is True
    assert isolated["sherpa"]["barge_word_cut_require_speaker"] is False
    assert isolated["sherpa"]["input_agc"] is True
    assert isolated["sherpa"]["input_calibrate"] is True
    assert isolated["sherpa"]["input_calibrate_sec"] == 1.5
    assert isolated["sherpa"]["barge_word_cut_energy_fallback_enabled"] is True
    assert isolated["sherpa"]["barge_word_cut_energy_margin_db"] == 6.0
    assert isolated["sherpa"]["barge_word_cut_energy_min_blocks"] == 3
    assert isolated["sherpa"]["barge_word_cut_min_words"] == 4
    assert isolated["sherpa"]["speaker_embedding_model"] == ""
    assert isolated["sherpa"]["speaker_enroll_embedding"] == ""
    assert original["llm"]["cloud"]["enabled"] is True
    assert isolated["llm"]["host"] == "http://127.0.0.1:11434"
    assert isolated["llm"]["cloud"]["enabled"] is False
    assert isolated["llm"]["cloud"]["strategy"] == "local_only"
    assert isolated["llm"]["live_routing"] is False
    assert isolated["web_search"]["enabled"] is False
    assert isolated["screen_capture"] == {"enabled": False, "memorize": False}
    assert isolated["gui_actions"]["enabled"] is False
    assert isolated["watch"] == {"enabled": False, "grants": []}
    assert isolated["memory"]["backend"] == "inmemory"
    assert isolated["memory"]["embeddings"] is False
    assert isolated["memory"]["profile_enabled"] is False
    assert isolated["memory"]["procedural_enabled"] is False
    assert isolated["memory"]["cross_session_continuity"] is False
    assert isolated["memory"]["persist_assistant"] is False
    assert isolated["agent_brain"] == {
        "local_only": True,
        "offline": True,
        "os_mode": False,
    }


def test_enrollment_bypasses_normal_sherpa_runtime_preflight(tmp_path, monkeypatch):
    import core.enroll as enroll

    monkeypatch.setenv("SPEAKER_RUN_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(enroll, "run_enrollment", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        app,
        "_require_sherpa_runtime_ready",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("normal runtime preflight ran during enrollment")
        ),
    )
    assert app.main(["--engine", "sherpa", "--llm", "echo", "--enroll"]) == 0


def test_load_config_merges_config_local(tmp_path, monkeypatch):
    from core.app import _load_config

    # This test exercises the merge mechanism itself, so opt out of the
    # session-wide hermetic guard (conftest sets SPEAKER_NO_LOCAL_CONFIG=1).
    monkeypatch.delenv("SPEAKER_NO_LOCAL_CONFIG", raising=False)

    (tmp_path / "config.json").write_text(
        json.dumps({"sherpa": {"asr_encoder": "", "sample_rate": 16000}, "device": "desktop"}),
        encoding="utf-8",
    )
    (tmp_path / "config.local.json").write_text(
        json.dumps({"sherpa": {"asr_encoder": "/m/asr/enc.onnx"}}), encoding="utf-8"
    )
    cfg = _load_config(str(tmp_path / "config.json"), local=str(tmp_path / "config.local.json"))
    # local override wins, untouched template fields survive, other sections kept
    assert cfg["sherpa"]["asr_encoder"] == "/m/asr/enc.onnx"
    assert cfg["sherpa"]["sample_rate"] == 16000
    assert cfg["device"] == "desktop"


def test_build_engine_requires_models_for_replay():
    import argparse

    from core.app import _build_engine

    args = argparse.Namespace(engine="replay", replay_dir="x")
    with pytest.raises(SystemExit) as exc:
        _build_engine(args, {"sherpa": {}})  # no model paths
    assert "tools.setup_models" in str(exc.value)
