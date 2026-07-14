from __future__ import annotations

import json
from pathlib import Path

from tools.autotest import __main__ as autotest_cli
from tools.autotest.voice_loop import VoiceRun


def _voice_reports(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("autotest-*-voice.json"))


def _voice_run(*, error: str = "") -> VoiceRun:
    return VoiceRun(
        ok=not error,
        mode="delay",
        run_id=None,
        summary_path=None,
        log_path=None,
        wav_path=None,
        ref_wav_path=None,
        ready=not error,
        monitor_rms=0.1,
        clip_source="synth",
        injected_refs=[],
        aec_delay_ms=None,
        prompt_score={"mean_wer": 0.0, "n": 0, "pairs": []},
        error=error,
    )


def test_voice_setup_exception_still_writes_top_level_failure_report(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(autotest_cli, "OUT", str(tmp_path))

    def fail_config():
        raise ValueError("broken test config")

    monkeypatch.setattr("core.config.load_config", fail_config)

    assert autotest_cli.main(["voice", "--llm", "echo", "--acoustics", "delay"]) == 1

    reports = _voice_reports(tmp_path)
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    report = payload["reports"][0]
    assert payload["overall"]["failed_tiers"] == ["voice"]
    assert report["outcome"] == "fail"
    assert report["complete"] is False
    assert report["runner_error"] == {
        "type": "ValueError",
        "message": "broken test config",
    }
    assert Path(report["artifact_dir"]).is_dir()
    assert report["log_path"] is None


def test_voice_child_exception_preserves_partial_engine_log_in_failure_report(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(autotest_cli, "OUT", str(tmp_path))
    monkeypatch.setattr("core.config.load_config", lambda: {"sherpa": {}})

    def fail_runner(**kwargs):
        log_path = Path(kwargs["out_dir"]) / "engine_stdout.log"
        log_path.write_text("route proof lost\n", encoding="utf-8")
        raise RuntimeError("engine child exited nonzero (1): route proof lost")

    monkeypatch.setattr("tools.autotest.voice_loop.run_voice_loop", fail_runner)

    assert autotest_cli.main(["voice", "--llm", "echo", "--acoustics", "delay"]) == 1

    reports = _voice_reports(tmp_path)
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    report = payload["reports"][0]
    assert payload["overall"]["failed_tiers"] == ["voice"]
    assert report["verdict"]["failures"] == ["runner_error"]
    assert report["runner_error"]["type"] == "RuntimeError"
    assert report["log_path"]
    assert Path(report["log_path"]).read_text(encoding="utf-8") == "route proof lost\n"


def test_voice_grading_exception_still_writes_top_level_failure_report(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(autotest_cli, "OUT", str(tmp_path))
    monkeypatch.setattr("core.config.load_config", lambda: {"sherpa": {}})
    monkeypatch.setattr(
        "tools.autotest.voice_loop.run_voice_loop", lambda **_kwargs: _voice_run()
    )

    def fail_grade(**_kwargs):
        raise RuntimeError("malformed grader evidence")

    monkeypatch.setattr(autotest_cli, "evaluate_voice", fail_grade)

    assert autotest_cli.main(["voice", "--llm", "echo", "--acoustics", "delay"]) == 1

    reports = _voice_reports(tmp_path)
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))["reports"][0]
    assert report["runner_error"] == {
        "type": "RuntimeError",
        "message": "malformed grader evidence",
    }
    assert report["verdict"]["failures"] == ["runner_error"]


def test_returned_voice_error_uses_same_failure_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(autotest_cli, "OUT", str(tmp_path))
    monkeypatch.setattr("core.config.load_config", lambda: {"sherpa": {}})
    monkeypatch.setattr(
        "tools.autotest.voice_loop.run_voice_loop",
        lambda **_kwargs: _voice_run(error="speaker mode needs make_sound=True"),
    )

    assert autotest_cli.main(["voice", "--llm", "echo", "--acoustics", "delay"]) == 1

    reports = _voice_reports(tmp_path)
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))["reports"][0]
    assert report["artifact_dir"]
    assert report["runner_error"] == {
        "type": "VoiceRunError",
        "message": "speaker mode needs make_sound=True",
    }
    assert report["verdict"]["failures"] == ["runner_error"]


def test_voice_report_composes_private_synth_command_latency_ceiling(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(autotest_cli, "OUT", str(tmp_path))
    monkeypatch.setattr("core.config.load_config", lambda: {"sherpa": {}})
    run = _voice_run()
    run.scenarios = {
        "s3_barge_in": {
            "barge_clip_role": "command",
            "barge_latency_s": 1.3,
            "pass": True,
        }
    }
    monkeypatch.setattr(
        "tools.autotest.voice_loop.run_voice_loop", lambda **_kwargs: run
    )

    # Other intentionally absent scenario/bundle evidence keeps this fixture
    # red; this assertion targets the real tier composition and persisted S3
    # policy metadata rather than duplicating the verdict helper tests.
    assert autotest_cli.main(
        ["voice", "--llm", "echo", "--acoustics", "delay"]
    ) == 1

    reports = _voice_reports(tmp_path)
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))["reports"][0]
    assert report["mode"] == "delay"
    assert report["clip_source"] == "synth"
    assert report["scenarios"]["s3_barge_in"] == {
        "barge_clip_role": "command",
        "barge_latency_s": 1.3,
        "barge_latency_ceiling_s": 1.4,
        "pass": True,
    }
