from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.app as app
import core.diagnostic_bundle as diagnostic_bundle
from core.guided_stt_plan import guided_stt_sherpa_config_sha256


_DIGEST = "1" * 64


class _Summary:
    def __init__(self) -> None:
        self.notes: dict[str, object] = {}
        self.meta = self.notes

    def note(self, **values) -> None:
        self.notes.update(values)


class _Runlog:
    def __init__(self, tmp_path) -> None:
        self.run_id = "capture-test"
        self.log_path = str(tmp_path / "run-capture-test.txt")
        self.summary_path = str(tmp_path / "run-capture-test.summary.json")
        self.summary = _Summary()
        self.logger = SimpleNamespace(
            error=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
            info=lambda *_args, **_kwargs: None,
        )
        self.finalize_calls = 0

    def finalize(self, _records) -> None:
        self.finalize_calls += 1


class _Monitor:
    def __init__(self) -> None:
        self.marks: list[str] = []
        self.stop_calls = 0

    def start(self) -> None:
        pass

    def mark(self, name: str) -> None:
        self.marks.append(name)

    def stop(self) -> None:
        self.stop_calls += 1


def _artifact_paths(tmp_path) -> dict[str, str]:
    return {
        "recording": str(tmp_path / "run.wav"),
        "pre_dsp_reference": str(tmp_path / "run.pre-dsp.wav"),
        "playback_reference": str(tmp_path / "run.ref.wav"),
        "asr_input_reference": str(tmp_path / "run.asr.wav"),
        "final_model_input": str(tmp_path / "run.final-input.f32le"),
        "diagnostic_timeline": str(tmp_path / "run.timeline.jsonl"),
        "diagnostic_manifest": str(tmp_path / "run.diagnostic.json"),
    }


def _write_private_manifest(paths: dict[str, str]) -> None:
    path = Path(paths["diagnostic_manifest"])
    path.write_bytes(b"{}\n")
    path.chmod(0o600)


def test_post_stop_evidence_gate_binds_status_count_and_exact_artifact_names(
    tmp_path,
    monkeypatch,
) -> None:
    calls = []

    def validate(path, **expectations):
        calls.append((path, expectations))
        return True

    monkeypatch.setattr(diagnostic_bundle, "validate_manifest", validate)
    engine = SimpleNamespace(
        diagnostic_status={"status": "complete", "failure_codes": []}
    )
    paths = _artifact_paths(tmp_path)
    _write_private_manifest(paths)

    complete, status, manifest_sha256 = app._capture_only_evidence_complete(
        engine,
        paths,
        expected_final_count=16,
    )

    assert complete
    assert status == {"status": "complete", "failure_codes": []}
    assert manifest_sha256 == hashlib.sha256(b"{}\n").hexdigest()
    assert calls == [
        (
            paths["diagnostic_manifest"],
            {
                "expected_final_model_input_count": 16,
                "expected_artifact_files": frozenset(
                    path.rsplit("/", 1)[-1]
                    for name, path in paths.items()
                    if name != "diagnostic_manifest"
                ),
            },
        )
    ]


@pytest.mark.parametrize(
    "status",
    [
        {"status": "incomplete", "failure_codes": []},
        {"status": "complete", "failure_codes": ["late_final"]},
        {"status": "complete"},
        None,
    ],
)
def test_post_stop_evidence_gate_rejects_incomplete_status(
    tmp_path,
    monkeypatch,
    status,
) -> None:
    monkeypatch.setattr(diagnostic_bundle, "validate_manifest", lambda *_a, **_k: True)
    engine = SimpleNamespace(diagnostic_status=status)
    paths = _artifact_paths(tmp_path)
    _write_private_manifest(paths)

    complete, _payload, _manifest_sha256 = app._capture_only_evidence_complete(
        engine,
        paths,
        expected_final_count=16,
    )

    assert not complete


def _capture_cli(*extra: str) -> list[str]:
    return [
        "--session",
        "local",
        "--capture-only",
        "--capture-only-plan",
        "/private/plan.json",
        "--capture-only-profile-sha256",
        _DIGEST,
        "--capture-only-sherpa-sha256",
        _DIGEST,
        "--capture-only-contract-sha256",
        _DIGEST,
        "--final-stt-profile",
        app.FINAL_STT_PROFILE_NAMES[0],
        "--no-speaker-enrollment",
        "--record",
        "--record-pre-dsp-reference",
        "--record-playback-reference",
        *extra,
    ]


@pytest.mark.parametrize(
    "conflict",
    [
        ("--list-devices",),
        ("--llm", "echo"),
        ("--model", "private-model"),
        ("--fast-model", "private-fast-model"),
        ("--mode", "assistant"),
        ("--stream-tts",),
        ("--agent",),
        ("--gui-actions",),
        ("--planner",),
    ],
)
def test_capture_parser_rejects_device_listing_and_assistant_options_before_setup(
    monkeypatch,
    conflict,
) -> None:
    monkeypatch.setattr(
        app,
        "setup_logging",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("capture parser reached setup")
        ),
    )

    with pytest.raises(SystemExit) as raised:
        app.main(_capture_cli(*conflict))

    assert raised.value.code == 2


def test_capture_main_binds_final_effective_config_and_bypasses_assistant_builds(
    tmp_path,
    monkeypatch,
) -> None:
    profile_digest = "2" * 64
    base = {
        "device": "safe",
        "sherpa": {
            "asr_encoder": "/models/asr.onnx",
            "asr_tokens": "/models/tokens.txt",
            "speaker_enroll_embedding": "/private/enrollment.json",
        },
    }
    runlog = _Runlog(tmp_path)
    monitor = _Monitor()
    plan = SimpleNamespace(
        mode=SimpleNamespace(value="local"),
        topology=SimpleNamespace(value="local_device"),
        audio_egress=SimpleNamespace(value="local_only"),
        engine="sherpa",
        room=None,
        require_selectable=lambda: None,
    )
    monkeypatch.setattr(app, "setup_logging", lambda _debug: runlog)
    monkeypatch.setattr(app, "SystemMonitor", lambda _summary: monitor)
    monkeypatch.setattr(app, "_load_config", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(
        app.VoiceSessionPlan,
        "resolve",
        staticmethod(lambda **_kwargs: plan),
    )
    monkeypatch.setattr(app, "resolve_device", lambda *_args: ("safe", None))
    monkeypatch.setattr(app, "_apply_device_profile", lambda config, *_a, **_k: config)

    def apply_profile(config, name):
        assert name == app.FINAL_STT_PROFILE_NAMES[0]
        return config, SimpleNamespace(
            name=name,
            sha256=profile_digest,
            schema_version=1,
        )

    monkeypatch.setattr(app, "apply_final_stt_profile", apply_profile)

    def mask_enrollment(config):
        merged = {**config, "sherpa": dict(config["sherpa"])}
        merged["sherpa"]["speaker_enroll_embedding"] = ""
        merged["sherpa"]["speaker_enroll_wav"] = ""
        return merged

    monkeypatch.setattr(app, "apply_no_speaker_enrollment", mask_enrollment)
    effective = mask_enrollment(base)
    effective["sherpa"] = dict(effective["sherpa"])
    effective["sherpa"].update(
        input_gain=2.0,
        record_pre_dsp_reference=True,
        record_playback_reference=True,
    )
    sherpa_digest = guided_stt_sherpa_config_sha256(effective["sherpa"])
    readiness_calls = []
    monkeypatch.setattr(
        app,
        "_require_sherpa_runtime_ready",
        lambda *args, **kwargs: readiness_calls.append((args, kwargs)),
    )
    capture_calls = []
    monkeypatch.setattr(
        app,
        "_run_capture_only",
        lambda args, config, **kwargs: (
            capture_calls.append((args, config, kwargs)) or 0
        ),
    )
    for name in ("_build_llms", "_build_engine", "build_router", "build_runtime"):
        monkeypatch.setattr(
            app,
            name,
            lambda *_args, _name=name, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"capture constructed {_name}")
            ),
        )

    code = app.main(
        [
            "--session",
            "local",
            "--capture-only",
            "--capture-only-plan",
            str(tmp_path / "plan.json"),
            "--capture-only-profile-sha256",
            profile_digest,
            "--capture-only-sherpa-sha256",
            sherpa_digest,
            "--capture-only-contract-sha256",
            _DIGEST,
            "--final-stt-profile",
            app.FINAL_STT_PROFILE_NAMES[0],
            "--no-speaker-enrollment",
            "--record",
            "--record-pre-dsp-reference",
            "--record-playback-reference",
            "--input-gain",
            "2.0",
        ]
    )

    assert code == 0
    assert len(readiness_calls) == 1
    readiness_args, readiness_kwargs = readiness_calls[0]
    assert readiness_args[1] == "echo"
    assert readiness_kwargs == {
        "models_needed": None,
        "virtual_audio_binder": None,
        "include_speaker": False,
        "include_tts": False,
        "include_kws": False,
        "audio_device_kinds": ("input",),
    }
    assert len(capture_calls) == 1
    assert capture_calls[0][1]["sherpa"] == effective["sherpa"]
    assert runlog.summary.notes["capture_only_contract_sha256"] == _DIGEST
    assert runlog.summary.notes["llm_constructed"] is False
    assert runlog.summary.notes["control_plane_constructed"] is False
    assert runlog.summary.notes["tts_constructed"] is False
    assert runlog.summary.notes["keyword_spotter_constructed"] is False
    assert runlog.summary.notes["speaker_gate_constructed"] is False
    assert runlog.summary.notes["playback_worker_constructed"] is False
    assert runlog.summary.notes["output_device_queried"] is False
    assert runlog.summary.notes["owner_authority"] is False


def test_capture_digest_mismatch_closes_telemetry_before_readiness_or_build(
    tmp_path,
    monkeypatch,
) -> None:
    runlog = _Runlog(tmp_path)
    monitor = _Monitor()
    plan = SimpleNamespace(
        mode=SimpleNamespace(value="local"),
        topology=SimpleNamespace(value="local_device"),
        audio_egress=SimpleNamespace(value="local_only"),
        engine="sherpa",
        room=None,
        require_selectable=lambda: None,
    )
    monkeypatch.setattr(app, "setup_logging", lambda _debug: runlog)
    monkeypatch.setattr(app, "SystemMonitor", lambda _summary: monitor)
    monkeypatch.setattr(
        app,
        "_load_config",
        lambda *_args, **_kwargs: {"device": "safe", "sherpa": {}},
    )
    monkeypatch.setattr(
        app.VoiceSessionPlan,
        "resolve",
        staticmethod(lambda **_kwargs: plan),
    )
    monkeypatch.setattr(app, "resolve_device", lambda *_args: ("safe", None))
    monkeypatch.setattr(app, "_apply_device_profile", lambda config, *_a, **_k: config)
    monkeypatch.setattr(
        app,
        "apply_final_stt_profile",
        lambda config, name: (
            config,
            SimpleNamespace(name=name, sha256="2" * 64, schema_version=1),
        ),
    )
    monkeypatch.setattr(app, "apply_no_speaker_enrollment", lambda config: config)
    for name in (
        "_require_sherpa_runtime_ready",
        "_run_capture_only",
        "_build_llms",
        "_build_engine",
    ):
        monkeypatch.setattr(
            app,
            name,
            lambda *_args, _name=name, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"digest mismatch reached {_name}")
            ),
        )

    with pytest.raises(SystemExit) as raised:
        app.main(
            [
                "--session",
                "local",
                "--capture-only",
                "--capture-only-plan",
                str(tmp_path / "plan.json"),
                "--capture-only-profile-sha256",
                "3" * 64,
                "--capture-only-sherpa-sha256",
                guided_stt_sherpa_config_sha256(
                    {
                        "record_pre_dsp_reference": True,
                        "record_playback_reference": True,
                    }
                ),
                "--capture-only-contract-sha256",
                _DIGEST,
                "--final-stt-profile",
                app.FINAL_STT_PROFILE_NAMES[0],
                "--no-speaker-enrollment",
                "--record",
                "--record-pre-dsp-reference",
                "--record-playback-reference",
            ]
        )

    assert raised.value.code == 2
    assert monitor.stop_calls == 1
    assert runlog.finalize_calls == 1


@pytest.mark.parametrize("evidence_complete", [True, False])
def test_run_capture_only_checks_evidence_only_after_session_stop(
    tmp_path,
    monkeypatch,
    evidence_complete: bool,
) -> None:
    import core.guided_stt_plan as guided_plan
    import core.stt_capture as stt_capture

    engine = SimpleNamespace(
        stopped=False,
        diagnostic_status={
            "status": "complete",
            "failure_codes": [],
        },
    )
    result = SimpleNamespace(
        completed=True,
        planned_cases=16,
        summary_facts=lambda: {"capture_completed": True},
    )

    class Session:
        accepted_finals = 16
        capture_state = "open"

        def __init__(self, selected_engine, _plan) -> None:
            assert selected_engine is engine
            self.stopped = False

        def run(self):
            self.stopped = True
            engine.stopped = True
            return result

        def stop(self) -> None:
            self.stopped = True
            engine.stopped = True

    monkeypatch.setattr(
        guided_plan, "load_private_guided_stt_plan", lambda _path: object()
    )
    monkeypatch.setattr(stt_capture, "CaptureOnlySession", Session)
    monkeypatch.setattr(app, "_build_capture_only_engine", lambda _config: engine)
    paths = _artifact_paths(tmp_path)
    monkeypatch.setattr(
        app,
        "_capture_only_recording_paths",
        lambda *_args, **_kwargs: paths,
    )

    def evidence(selected_engine, selected_paths, *, expected_final_count):
        assert selected_engine.stopped
        assert selected_paths is paths
        assert expected_final_count == 16
        return (
            evidence_complete,
            selected_engine.diagnostic_status,
            ("d" * 64 if evidence_complete else None),
        )

    monkeypatch.setattr(app, "_capture_only_evidence_complete", evidence)
    monkeypatch.setattr(
        guided_plan,
        "guided_stt_summary_contract_sha256",
        lambda _meta: "e" * 64,
    )
    runlog = _Runlog(tmp_path)
    monitor = _Monitor()
    args = SimpleNamespace(capture_only_plan=str(tmp_path / "plan.json"))

    code = app._run_capture_only(
        args,
        {"sherpa": {}},
        runlog=runlog,
        monitor=monitor,
    )

    assert code == (0 if evidence_complete else 2)
    assert engine.stopped
    assert runlog.finalize_calls == 1
    if evidence_complete:
        assert runlog.summary.notes["diagnostic_manifest_sha256"] == "d" * 64
        assert runlog.summary.notes["capture_summary_contract_sha256"] == "e" * 64


def test_run_capture_only_stop_failure_is_nonzero_and_still_finalizes(
    tmp_path,
    monkeypatch,
) -> None:
    import core.guided_stt_plan as guided_plan
    import core.stt_capture as stt_capture

    engine = SimpleNamespace(diagnostic_status=None)

    class Session:
        stopped = False
        accepted_finals = 0
        capture_state = "starting"

        def __init__(self, _engine, _plan) -> None:
            pass

        def run(self):
            raise RuntimeError("start failed")

        def stop(self) -> None:
            raise RuntimeError("stop failed")

    monkeypatch.setattr(
        guided_plan, "load_private_guided_stt_plan", lambda _path: object()
    )
    monkeypatch.setattr(stt_capture, "CaptureOnlySession", Session)
    monkeypatch.setattr(app, "_build_capture_only_engine", lambda _config: engine)
    monkeypatch.setattr(
        app,
        "_capture_only_recording_paths",
        lambda *_args, **_kwargs: _artifact_paths(tmp_path),
    )
    runlog = _Runlog(tmp_path)

    assert (
        app._run_capture_only(
            SimpleNamespace(capture_only_plan=str(tmp_path / "plan.json")),
            {"sherpa": {}},
            runlog=runlog,
            monitor=_Monitor(),
        )
        == 2
    )
    assert runlog.finalize_calls == 1
