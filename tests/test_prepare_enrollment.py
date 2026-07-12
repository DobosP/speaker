"""Device-free tests for isolated speaker-enrollment preparation."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

import tools.prepare_enrollment as prep
from core.enroll import (
    Enrollment,
    _persist_local,
    load_enrollment,
    run_enrollment,
    save_enrollment,
)
from tools.prepare_enrollment import PreparationError, main, prepare_enrollment


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _layout(tmp_path: Path):
    primary = tmp_path / "primary"
    feature = tmp_path / "feature"
    speaker_dir = primary / "pretrained_models" / "sherpa" / "speaker"
    primary.mkdir()
    feature.mkdir()
    speaker_dir.mkdir(parents=True)

    enrollment = speaker_dir / "enrollment.json"
    enrollment_bytes = b'{"embedding":[0.25,0.75],"private_marker":"VOICE-BYTES"}'
    enrollment.write_bytes(enrollment_bytes)
    os.chmod(enrollment, 0o640)

    config = primary / "config.local.json"
    config_data = {
        "api_token": "CONFIG-SECRET-MARKER",
        "unrelated": {"keep": True},
        "sherpa": {
            "speaker_embedding_model": str(speaker_dir / "model.onnx"),
            "speaker_enroll_embedding": str(enrollment),
            "input_agc": True,
        },
    }
    config_bytes = json.dumps(config_data, indent=2).encode()
    config.write_bytes(config_bytes)
    os.chmod(config, 0o640)
    (feature / "config.local.json").symlink_to(config)

    backup = speaker_dir / "enrollment.pre-v5-test.json"
    return {
        "primary": primary,
        "feature": feature,
        "speaker_dir": speaker_dir,
        "config": config,
        "config_data": config_data,
        "config_bytes": config_bytes,
        "enrollment": enrollment,
        "enrollment_bytes": enrollment_bytes,
        "backup": backup,
        "candidate_name": "enrollment.v5-test-sha.json",
    }


def _prepare(layout):
    return prepare_enrollment(
        worktree=layout["feature"],
        expected_config_target=layout["config"],
        expected_enrollment=layout["enrollment"],
        backup=layout["backup"],
        candidate_name=layout["candidate_name"],
    )


class _Gate:
    def embed(self, _samples, _sample_rate):
        return [1.0, 0.0]


def test_prepare_isolates_config_backs_up_and_reserves_candidate(tmp_path):
    layout = _layout(tmp_path)
    result = _prepare(layout)

    isolated = layout["feature"] / "config.local.json"
    assert result.config_local == isolated
    assert isolated.is_file()
    assert not isolated.is_symlink()
    assert _mode(isolated) == 0o600
    assert os.stat(isolated).st_ino != os.stat(layout["config"]).st_ino

    # The source config and historical enrollment are untouched byte-for-byte.
    assert layout["config"].read_bytes() == layout["config_bytes"]
    assert _mode(layout["config"]) == 0o640
    assert layout["enrollment"].read_bytes() == layout["enrollment_bytes"]
    assert _mode(layout["enrollment"]) == 0o640

    assert result.backup == layout["backup"]
    assert result.backup.read_bytes() == layout["enrollment_bytes"]
    assert _mode(result.backup) == 0o600
    assert os.stat(result.backup).st_ino != os.stat(layout["enrollment"]).st_ino

    expected_candidate = (
        layout["feature"]
        / "pretrained_models"
        / "sherpa"
        / "speaker"
        / layout["candidate_name"]
    )
    assert result.candidate == expected_candidate
    assert result.candidate.read_bytes() == b""
    assert _mode(result.candidate) == 0o600

    written = json.loads(isolated.read_text())
    assert written["api_token"] == layout["config_data"]["api_token"]
    assert written["unrelated"] == {"keep": True}
    assert written["sherpa"]["speaker_embedding_model"] == (
        layout["config_data"]["sherpa"]["speaker_embedding_model"]
    )
    assert written["sherpa"]["speaker_enroll_embedding"] == str(
        expected_candidate
    )
    marker = written["_speaker_enrollment_preparation"]
    assert marker["candidate"] == str(expected_candidate)
    assert marker["worktree"] == str(layout["feature"])
    assert marker["candidate_size"] == 0


def test_reserved_candidate_accepts_atomic_enrollment_save(tmp_path):
    layout = _layout(tmp_path)
    candidate = _prepare(layout).candidate

    save_enrollment(
        str(candidate),
        Enrollment(model="/models/speaker.onnx", embedding=[0.6, 0.8], passes=3),
    )

    loaded = load_enrollment(str(candidate))
    assert loaded.embedding == [0.6, 0.8]
    assert loaded.passes == 3
    assert _mode(candidate) == 0o600


def test_prepare_cli_never_prints_config_or_enrollment_contents(tmp_path, capsys):
    layout = _layout(tmp_path)
    code = main([
        "--worktree", str(layout["feature"]),
        "--expected-config-target", str(layout["config"]),
        "--expected-enrollment", str(layout["enrollment"]),
        "--backup", str(layout["backup"]),
        "--candidate-name", layout["candidate_name"],
    ])

    captured = capsys.readouterr()
    assert code == 0
    assert "CONFIG-SECRET-MARKER" not in captured.out + captured.err
    assert "VOICE-BYTES" not in captured.out + captured.err
    assert "no contents printed" in captured.out
    assert "cd " in captured.out
    assert "--require-prepared-enrollment" in captured.out


def test_prepared_candidate_runs_through_enrollment_without_touching_history(tmp_path):
    layout = _layout(tmp_path)
    result = _prepare(layout)
    config = json.loads(result.config_local.read_text())

    code = run_enrollment(
        config,
        passes=1,
        seconds=0.1,
        config_path=str(result.config_local),
        recorder=lambda _seconds: [0.1, 0.2, 0.3],
        gate=_Gate(),
        require_prepared=True,
        out=lambda _line: None,
    )

    assert code == 0
    assert load_enrollment(str(result.candidate)).embedding == [1.0, 0.0]
    assert layout["enrollment"].read_bytes() == layout["enrollment_bytes"]
    assert layout["backup"].read_bytes() == layout["enrollment_bytes"]


def test_prepared_candidate_swap_during_capture_refuses_publish(tmp_path):
    layout = _layout(tmp_path)
    result = _prepare(layout)
    config = json.loads(result.config_local.read_text())

    def recorder(_seconds):
        result.candidate.unlink()
        result.candidate.write_text("replacement must survive")
        return [0.1, 0.2, 0.3]

    code = run_enrollment(
        config,
        passes=1,
        config_path=str(result.config_local),
        recorder=recorder,
        gate=_Gate(),
        require_prepared=True,
        out=lambda _line: None,
    )

    assert code == 5
    assert result.candidate.read_text() == "replacement must survive"
    assert layout["enrollment"].read_bytes() == layout["enrollment_bytes"]


def test_prepared_candidate_ancestor_symlink_swap_refuses_publish(tmp_path):
    layout = _layout(tmp_path)
    result = _prepare(layout)
    config = json.loads(result.config_local.read_text())
    outside = tmp_path / "outside"
    outside_candidate = (
        outside / "sherpa" / "speaker" / layout["candidate_name"]
    )
    outside_candidate.parent.mkdir(parents=True)
    outside_candidate.write_text("outside must survive")

    def recorder(_seconds):
        models = layout["feature"] / "pretrained_models"
        models.rename(layout["feature"] / "pretrained_models.original")
        models.symlink_to(outside)
        return [0.1, 0.2, 0.3]

    code = run_enrollment(
        config,
        passes=1,
        config_path=str(result.config_local),
        recorder=recorder,
        gate=_Gate(),
        require_prepared=True,
        out=lambda _line: None,
    )

    assert code == 5
    assert outside_candidate.read_text() == "outside must survive"
    assert layout["enrollment"].read_bytes() == layout["enrollment_bytes"]


def test_require_prepared_refuses_marker_free_empty_target_before_capture(tmp_path):
    config_path = tmp_path / "config.local.json"
    config_path.write_text("{}")
    candidate = tmp_path / "candidate.json"
    candidate.write_bytes(b"")
    called = False

    def recorder(_seconds):
        nonlocal called
        called = True
        return [0.1, 0.2, 0.3]

    code = run_enrollment(
        {
            "sherpa": {
                "speaker_embedding_model": "/model.onnx",
                "speaker_enroll_embedding": str(candidate),
            }
        },
        config_path=str(config_path),
        recorder=recorder,
        gate=_Gate(),
        require_prepared=True,
        out=lambda _line: None,
    )

    assert code == 5
    assert called is False
    assert candidate.read_bytes() == b""


def test_marker_free_nonempty_reference_needs_explicit_replace(tmp_path):
    config_path = tmp_path / "config.local.json"
    config_path.write_text("{}")
    historical = tmp_path / "historical.json"
    historical.write_text('{"embedding":[0.5,0.5]}')
    config = {
        "sherpa": {
            "speaker_embedding_model": "/model.onnx",
            "speaker_enroll_embedding": str(historical),
        }
    }
    recorder_called = False

    def recorder(_seconds):
        nonlocal recorder_called
        recorder_called = True
        return [0.1, 0.2, 0.3]

    refused = run_enrollment(
        config,
        passes=1,
        config_path=str(config_path),
        recorder=recorder,
        gate=_Gate(),
        out=lambda _line: None,
    )

    assert refused == 5
    assert recorder_called is False
    assert historical.read_text() == '{"embedding":[0.5,0.5]}'

    replaced = run_enrollment(
        config,
        passes=1,
        config_path=str(config_path),
        recorder=recorder,
        gate=_Gate(),
        replace_existing=True,
        out=lambda _line: None,
    )
    assert replaced == 0
    assert recorder_called is True
    assert load_enrollment(str(historical)).embedding == [1.0, 0.0]


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("backup_exists", "backup path already exists"),
        ("wrong_target", "does not match expected target"),
        ("configured_mismatch", "does not match the expected path"),
        ("invalid_enrollment", "has no non-empty embedding"),
        ("candidate_bad_name", "candidate name must match"),
        ("candidate_parent_symlink", "refusing symlink for candidate directory"),
    ],
)
def test_prepare_fails_closed_before_mutating_sources(tmp_path, mutation, expected):
    layout = _layout(tmp_path)
    expected_target = layout["config"]
    candidate_name = layout["candidate_name"]

    if mutation == "backup_exists":
        layout["backup"].write_text("do not replace")
    elif mutation == "wrong_target":
        wrong = layout["primary"] / "other.local.json"
        wrong.write_text("{}")
        expected_target = wrong
    elif mutation == "configured_mismatch":
        data = json.loads(layout["config"].read_text())
        data["sherpa"]["speaker_enroll_embedding"] = str(
            layout["speaker_dir"] / "other.json"
        )
        layout["config"].write_text(json.dumps(data))
        layout["config_bytes"] = layout["config"].read_bytes()
    elif mutation == "invalid_enrollment":
        layout["enrollment"].write_text('{"embedding": []}')
    elif mutation == "candidate_bad_name":
        candidate_name = "../enrollment.v5-escape.json"
    elif mutation == "candidate_parent_symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        (layout["feature"] / "pretrained_models").symlink_to(outside)

    link = layout["feature"] / "config.local.json"
    enrollment_before = layout["enrollment"].read_bytes()
    config_before = layout["config"].read_bytes()

    with pytest.raises(PreparationError, match=expected):
        prepare_enrollment(
            worktree=layout["feature"],
            expected_config_target=expected_target,
            expected_enrollment=layout["enrollment"],
            backup=layout["backup"],
            candidate_name=candidate_name,
        )

    assert link.is_symlink()
    assert layout["config"].read_bytes() == config_before
    assert layout["enrollment"].read_bytes() == enrollment_before
    if mutation != "backup_exists":
        assert not layout["backup"].exists()
    else:
        assert layout["backup"].read_text() == "do not replace"


def test_prepare_rerun_refuses_already_isolated_state(tmp_path):
    layout = _layout(tmp_path)
    _prepare(layout)
    with pytest.raises(PreparationError, match="unprepared symlink"):
        _prepare(layout)


@pytest.mark.parametrize("failure", ("reserve", "publish"))
def test_prepare_failure_before_final_publish_keeps_config_guarded(
    tmp_path, monkeypatch, failure
):
    layout = _layout(tmp_path)

    if failure == "reserve":
        monkeypatch.setattr(
            prep,
            "_reserve_candidate",
            lambda _path: (_ for _ in ()).throw(PreparationError("reserve failed")),
        )
    else:
        monkeypatch.setattr(
            prep,
            "_publish_isolated_config",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PreparationError("publish failed")
            ),
        )

    with pytest.raises(PreparationError, match=f"{failure} failed"):
        _prepare(layout)

    config_link = layout["feature"] / "config.local.json"
    assert config_link.is_symlink()
    assert config_link.resolve() == layout["config"]
    assert layout["config"].read_bytes() == layout["config_bytes"]
    assert layout["enrollment"].read_bytes() == layout["enrollment_bytes"]
    assert layout["backup"].read_bytes() == layout["enrollment_bytes"]
    candidate = (
        layout["feature"]
        / "pretrained_models"
        / "sherpa"
        / "speaker"
        / layout["candidate_name"]
    )
    assert candidate.exists() is (failure == "publish")


def test_persist_local_refuses_symlink_and_preserves_target(tmp_path):
    target = tmp_path / "primary.local.json"
    original = b'{"secret":"unchanged","sherpa":{"old":true}}'
    target.write_bytes(original)
    link = tmp_path / "config.local.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="refusing to follow symlink"):
        _persist_local(str(link), {"speaker_enroll_embedding": "/new.json"})

    assert link.is_symlink()
    assert target.read_bytes() == original


def test_save_enrollment_refuses_symlink_and_preserves_target(tmp_path):
    target = tmp_path / "historical.json"
    original = b'{"embedding":[0.1],"marker":"historical"}'
    target.write_bytes(original)
    link = tmp_path / "candidate.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="refusing to follow symlink"):
        save_enrollment(
            str(link),
            Enrollment(model="/m.onnx", embedding=[1.0]),
        )

    assert link.is_symlink()
    assert target.read_bytes() == original


def test_run_enrollment_rejects_config_symlink_before_recorder(tmp_path):
    target = tmp_path / "primary.local.json"
    target.write_text("{}")
    link = tmp_path / "config.local.json"
    link.symlink_to(target)
    recorder_called = False
    messages: list[str] = []

    def recorder(_seconds):
        nonlocal recorder_called
        recorder_called = True
        raise AssertionError("unsafe path guard must run before audio capture")

    code = run_enrollment(
        {
            "sherpa": {
                "speaker_embedding_model": "/model.onnx",
                "speaker_enroll_embedding": str(tmp_path / "candidate.json"),
            }
        },
        config_path=str(link),
        recorder=recorder,
        out=messages.append,
    )

    assert code == 5
    assert recorder_called is False
    assert target.read_text() == "{}"
    assert "unsafe local config persistence" in "\n".join(messages)


def test_run_enrollment_rejects_enrollment_symlink_before_recorder(tmp_path):
    config = tmp_path / "config.local.json"
    config.write_text("{}")
    historical = tmp_path / "historical.json"
    historical.write_text("historical")
    candidate = tmp_path / "candidate.json"
    candidate.symlink_to(historical)
    recorder_called = False

    def recorder(_seconds):
        nonlocal recorder_called
        recorder_called = True
        return [0.1, 0.2, 0.3]

    code = run_enrollment(
        {
            "sherpa": {
                "speaker_embedding_model": "/model.onnx",
                "speaker_enroll_embedding": str(candidate),
            }
        },
        config_path=str(config),
        recorder=recorder,
        out=lambda _line: None,
    )

    assert code == 5
    assert recorder_called is False
    assert historical.read_text() == "historical"
