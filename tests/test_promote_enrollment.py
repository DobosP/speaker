"""Device-free tests for explicit accepted-v5 enrollment promotion."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

# POSIX-only: promotion locking uses fcntl, which does not exist on Windows.
# importorskip (vs a bare import) lets the suite COLLECT on a Windows dev box
# instead of erroring out before a single test runs; Linux coverage unchanged.
fcntl = pytest.importorskip("fcntl")

import tools.promote_enrollment as promo
from core.enroll import (
    ENROLLMENT_FRONTEND_VERSION,
    ENROLLMENT_PREPARATION_KEY,
    Enrollment,
    EnrollmentFrontendProvenance,
    save_enrollment,
)
from tools.prepare_enrollment import prepare_enrollment
from tools.promote_enrollment import (
    PromotionAmbiguousError,
    PromotionError,
    PromotionStagedError,
    main,
    promote_enrollment,
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _set_marker_contract(marker: dict[str, object], key: str, path: Path) -> None:
    info = path.stat()
    payload = path.read_bytes()
    for field in (
        "dev",
        "ino",
        "size",
        "mtime_ns",
        "ctime_ns",
        "mode",
        "uid",
        "gid",
        "nlink",
    ):
        marker[f"{key}_{field}"] = (
            stat.S_IMODE(info.st_mode)
            if field == "mode"
            else getattr(info, f"st_{field}")
        )
    marker[f"{key}_sha256"] = hashlib.sha256(payload).hexdigest()


def _layout(tmp_path: Path) -> dict[str, object]:
    primary = tmp_path / "primary"
    feature = tmp_path / "feature"
    speaker_dir = primary / "pretrained_models" / "sherpa" / "speaker"
    primary.mkdir(mode=0o700)
    speaker_dir.mkdir(parents=True, mode=0o700)
    os.chmod(primary / "pretrained_models", 0o700)
    os.chmod(primary / "pretrained_models" / "sherpa", 0o700)
    os.chmod(speaker_dir, 0o700)
    feature.mkdir(mode=0o700)
    os.chmod(feature, 0o700)

    model = speaker_dir / "speaker-model.onnx"
    source = speaker_dir / "enrollment.json"
    source_bytes = json.dumps(
        {
            "model": str(model),
            "dim": 2,
            "passes": 3,
            "sample_rate": 16000,
            "created": "2026-07-11T12:00:00",
            "embedding": [0.25, 0.75],
            "private_marker": "V4-VOICE-SECRET",
        }
    ).encode()
    source.write_bytes(source_bytes)
    os.chmod(source, 0o640)

    primary_config = primary / "config.local.json"
    primary_data = {
        "api_token": "CONFIG-SECRET-MARKER",
        "unrelated": {"keep": True, "nested": [1, 2, 3]},
        "sherpa": {
            "speaker_embedding_model": str(model),
            "speaker_enroll_embedding": str(source),
            "sample_rate": 16000,
            "input_agc": True,
        },
    }
    primary_config.write_text(json.dumps(primary_data, indent=2))
    os.chmod(primary_config, 0o640)
    (feature / "config.local.json").symlink_to(primary_config)

    backup = speaker_dir / "enrollment.pre-v5-test.json"
    prepared = prepare_enrollment(
        worktree=feature,
        expected_config_target=primary_config,
        expected_enrollment=source,
        backup=backup,
        candidate_name="enrollment.v5-test-sha.json",
    )
    provenance = EnrollmentFrontendProvenance(
        version=ENROLLMENT_FRONTEND_VERSION,
        fingerprint="sha256:" + "a" * 64,
        summary="pipewire-echo-cancel -> input-agc-current-signal -> gtcrn",
        raw_baseline=False,
    )
    save_enrollment(
        str(prepared.candidate),
        Enrollment(
            model=str(model),
            embedding=[0.6, 0.8],
            sample_rate=16000,
            passes=3,
            created="2026-07-12T12:00:00",
            frontend=provenance,
        ),
    )
    accepted = speaker_dir / "enrollment.v5-test-sha-accepted.json"
    return {
        "primary": primary,
        "feature": feature,
        "speaker_dir": speaker_dir,
        "source": source,
        "source_bytes": source_bytes,
        "model": model,
        "primary_config": primary_config,
        "primary_data": primary_data,
        "feature_config": prepared.config_local,
        "backup": prepared.backup,
        "candidate": prepared.candidate,
        "accepted": accepted,
    }


def _call(
    layout: dict[str, object],
    *,
    accepted: Path | None = None,
    primary_config: Path | None = None,
):
    return promote_enrollment(
        worktree=layout["feature"],
        primary_config=primary_config or layout["primary_config"],
        expected_candidate=layout["candidate"],
        expected_source_enrollment=layout["source"],
        expected_backup=layout["backup"],
        accepted_enrollment=accepted or layout["accepted"],
        accept_live_gate=True,
    )


def _cli_args(
    layout: dict[str, object], *, include_acceptance: bool = True
) -> list[str]:
    args = [
        "--worktree",
        str(layout["feature"]),
        "--primary-config",
        str(layout["primary_config"]),
        "--expected-candidate",
        str(layout["candidate"]),
        "--expected-source-enrollment",
        str(layout["source"]),
        "--expected-backup",
        str(layout["backup"]),
        "--accepted-enrollment",
        str(layout["accepted"]),
    ]
    if include_acceptance:
        args.append("--accept-live-gate")
    return args


def test_promote_publishes_independent_v5_and_changes_only_primary_pointer(tmp_path):
    layout = _layout(tmp_path)
    primary_before = Path(layout["primary_config"]).stat()
    feature_before = Path(layout["feature_config"]).read_bytes()
    candidate_before = Path(layout["candidate"]).read_bytes()
    source_before = Path(layout["source"]).read_bytes()
    backup_before = Path(layout["backup"]).read_bytes()

    result = _call(layout)

    accepted = Path(layout["accepted"])
    assert result.accepted_enrollment == accepted
    assert result.accepted_was_adopted is False
    assert accepted.read_bytes() == candidate_before
    assert _mode(accepted) == 0o600
    assert os.stat(accepted).st_ino != os.stat(layout["candidate"]).st_ino
    assert os.stat(accepted).st_ino != os.stat(layout["source"]).st_ino

    assert Path(layout["feature_config"]).read_bytes() == feature_before
    assert Path(layout["candidate"]).read_bytes() == candidate_before
    assert Path(layout["source"]).read_bytes() == source_before
    assert Path(layout["backup"]).read_bytes() == backup_before == source_before

    expected = copy.deepcopy(layout["primary_data"])
    expected["sherpa"]["speaker_enroll_embedding"] = str(accepted)
    promoted = json.loads(Path(layout["primary_config"]).read_text())
    assert promoted == expected
    assert ENROLLMENT_PREPARATION_KEY not in promoted
    assert _mode(Path(layout["primary_config"])) == 0o600
    assert Path(layout["primary_config"]).stat().st_ino != primary_before.st_ino


def test_promote_cli_exit_zero_prints_no_config_embedding_or_digest(tmp_path, capsys):
    layout = _layout(tmp_path)

    code = main(_cli_args(layout))

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert code == 0
    assert "CONFIG-SECRET-MARKER" not in output
    assert "V4-VOICE-SECRET" not in output
    assert "0.6" not in output
    assert "sha256:" not in output
    assert "no contents printed" in captured.out
    assert "historical source" in captured.out
    assert "rollback backup" in captured.out
    assert "published" in captured.out


def test_promote_requires_explicit_live_acceptance_before_any_write(tmp_path, capsys):
    layout = _layout(tmp_path)
    config_before = Path(layout["primary_config"]).read_bytes()

    code = main(_cli_args(layout, include_acceptance=False))

    captured = capsys.readouterr()
    assert code == 2
    assert "promotion refused" in captured.err
    assert not Path(layout["accepted"]).exists()
    assert Path(layout["primary_config"]).read_bytes() == config_before


def test_restored_mtime_same_size_source_and_backup_rewrite_is_refused(tmp_path):
    layout = _layout(tmp_path)
    source = Path(layout["source"])
    backup = Path(layout["backup"])
    for path in (source, backup):
        before = path.stat()
        payload = path.read_bytes().replace(
            b"V4-VOICE-SECRET", b"V4-OTHER-SECRET"
        )
        assert len(payload) == before.st_size
        path.write_bytes(payload)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.chmod(backup, 0o600)

    with pytest.raises(PromotionError, match="changed since preparation"):
        _call(layout)

    assert not Path(layout["accepted"]).exists()


def test_identical_wrong_primary_config_is_refused_by_preparation_path(tmp_path):
    layout = _layout(tmp_path)
    original = Path(layout["primary_config"])
    other = original.with_name("other.local.json")
    other.write_bytes(original.read_bytes())
    os.chmod(other, 0o640)

    with pytest.raises(PromotionError, match="primary_config does not match"):
        _call(layout, primary_config=other)

    assert not Path(layout["accepted"]).exists()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("marker_missing", "no prepared enrollment marker"),
        ("marker_candidate", "candidate does not match"),
        ("candidate_pointer", "no longer points"),
        ("candidate_v4", "not exact front-end v5"),
        ("candidate_v5_float", "not exact front-end v5"),
        ("candidate_fingerprint", "invalid front-end fingerprint"),
        ("candidate_numeric_string", "non-number"),
        ("candidate_nonfinite", "non-finite"),
        ("candidate_not_unit", "not a finite unit vector"),
        ("candidate_passes", "at least three"),
        ("candidate_sample_rate", "sample rate does not match"),
        ("candidate_raw", "not a live v5 capture"),
        ("candidate_model", "model does not match"),
        ("candidate_dimension", "historical same-model"),
        ("candidate_in_place", "not atomically populated"),
        ("source_identity", "historical enrollment changed"),
        ("backup_identity", "rollback backup changed"),
        ("backup_forged", "no longer matches"),
        ("historical_model", "different model"),
        ("historical_dimension", "invalid dimension evidence"),
        ("primary_pointer", "no longer points"),
        ("primary_model", "different speaker models"),
        ("primary_sample_rate", "different sample rates"),
        ("primary_marker", "ambiguously carries"),
        ("feature_symlink", "refusing to follow symlink"),
        ("candidate_mode", "candidate is not mode 600"),
        ("backup_mode", "backup is not mode 600"),
        ("accepted_bad_name", "must be named"),
    ],
)
def test_promotion_preflight_failures_do_not_publish_or_rewire(
    tmp_path, mutation, expected
):
    layout = _layout(tmp_path)
    accepted = Path(layout["accepted"])
    feature_config = Path(layout["feature_config"])
    primary_config = Path(layout["primary_config"])
    candidate = Path(layout["candidate"])
    source = Path(layout["source"])
    backup = Path(layout["backup"])

    if mutation in {"marker_missing", "marker_candidate", "candidate_pointer"}:
        data = json.loads(feature_config.read_text())
        if mutation == "marker_missing":
            data.pop(ENROLLMENT_PREPARATION_KEY)
        elif mutation == "marker_candidate":
            data[ENROLLMENT_PREPARATION_KEY]["candidate"] = str(
                tmp_path / "other.json"
            )
        else:
            data["sherpa"]["speaker_enroll_embedding"] = str(source)
        feature_config.write_text(json.dumps(data))
    elif mutation.startswith("candidate_") and mutation not in {
        "candidate_pointer",
        "candidate_in_place",
        "candidate_mode",
    }:
        data = json.loads(candidate.read_text())
        if mutation == "candidate_v4":
            data["frontend"]["version"] = 4
        elif mutation == "candidate_v5_float":
            data["frontend"]["version"] = 5.9
        elif mutation == "candidate_fingerprint":
            data["frontend"]["fingerprint"] = "sha256:bad"
        elif mutation == "candidate_numeric_string":
            data["embedding"][0] = "0.6"
        elif mutation == "candidate_nonfinite":
            data["embedding"][0] = float("nan")
        elif mutation == "candidate_not_unit":
            data["embedding"] = [0.6, 0.6]
        elif mutation == "candidate_passes":
            data["passes"] = 2
        elif mutation == "candidate_sample_rate":
            data["sample_rate"] = 8000
        elif mutation == "candidate_raw":
            data["frontend"]["raw_baseline"] = True
        elif mutation == "candidate_model":
            data["model"] = str(tmp_path / "other-model.onnx")
        elif mutation == "candidate_dimension":
            data["embedding"] = [1.0, 0.0, 0.0]
            data["dim"] = 3
        else:  # pragma: no cover - exhaustive mutation table
            raise AssertionError(mutation)
        candidate.write_text(json.dumps(data))
    elif mutation == "candidate_in_place":
        data = json.loads(feature_config.read_text())
        info = candidate.stat()
        data[ENROLLMENT_PREPARATION_KEY]["candidate_dev"] = info.st_dev
        data[ENROLLMENT_PREPARATION_KEY]["candidate_ino"] = info.st_ino
        feature_config.write_text(json.dumps(data))
    elif mutation == "source_identity":
        info = source.stat()
        os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
    elif mutation == "backup_identity":
        info = backup.stat()
        os.utime(backup, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
    elif mutation == "backup_forged":
        backup.write_text('{"embedding":[0.1,0.9]}')
        data = json.loads(feature_config.read_text())
        info = backup.stat()
        marker = data[ENROLLMENT_PREPARATION_KEY]
        marker["backup_dev"] = info.st_dev
        marker["backup_ino"] = info.st_ino
        marker["backup_size"] = info.st_size
        marker["backup_mtime_ns"] = info.st_mtime_ns
        feature_config.write_text(json.dumps(data))
    elif mutation in {"historical_model", "historical_dimension"}:
        historical = json.loads(source.read_text())
        if mutation == "historical_model":
            historical["model"] = str(tmp_path / "other-model.onnx")
        else:
            historical["dim"] = 3
        payload = json.dumps(historical)
        source.write_text(payload)
        backup.write_text(payload)
        os.chmod(backup, 0o600)
        data = json.loads(feature_config.read_text())
        marker = data[ENROLLMENT_PREPARATION_KEY]
        for key, path in (("source", source), ("backup", backup)):
            _set_marker_contract(marker, key, path)
        feature_config.write_text(json.dumps(data))
    elif mutation in {
        "primary_pointer",
        "primary_model",
        "primary_sample_rate",
        "primary_marker",
    }:
        data = json.loads(primary_config.read_text())
        if mutation == "primary_pointer":
            data["sherpa"]["speaker_enroll_embedding"] = str(
                Path(layout["speaker_dir"]) / "other.json"
            )
        elif mutation == "primary_model":
            data["sherpa"]["speaker_embedding_model"] = str(
                tmp_path / "other-model.onnx"
            )
        elif mutation == "primary_sample_rate":
            data["sherpa"]["sample_rate"] = 8000
        else:
            data[ENROLLMENT_PREPARATION_KEY] = {"stale": True}
        primary_config.write_text(json.dumps(data))
        isolated = json.loads(feature_config.read_text())
        _set_marker_contract(
            isolated[ENROLLMENT_PREPARATION_KEY], "primary", primary_config
        )
        feature_config.write_text(json.dumps(isolated))
    elif mutation == "feature_symlink":
        feature_config.unlink()
        feature_config.symlink_to(primary_config)
    elif mutation == "candidate_mode":
        os.chmod(candidate, 0o640)
    elif mutation == "backup_mode":
        os.chmod(backup, 0o640)
    elif mutation == "accepted_bad_name":
        accepted = accepted.with_name("enrollment.v5-other-accepted.json")
    else:  # pragma: no cover - exhaustive mutation table
        raise AssertionError(mutation)

    primary_before = primary_config.read_bytes()
    candidate_before = candidate.read_bytes()
    source_before = source.read_bytes()
    backup_before = backup.read_bytes()

    with pytest.raises(PromotionError, match=expected):
        _call(layout, accepted=accepted)

    assert primary_config.read_bytes() == primary_before
    assert candidate.read_bytes() == candidate_before
    assert source.read_bytes() == source_before
    assert backup.read_bytes() == backup_before
    assert not accepted.exists()


def test_promote_adopts_exact_mode600_orphan(tmp_path):
    layout = _layout(tmp_path)
    accepted = Path(layout["accepted"])
    accepted.write_bytes(Path(layout["candidate"]).read_bytes())
    os.chmod(accepted, 0o600)
    orphan_inode = accepted.stat().st_ino

    result = _call(layout)

    assert result.accepted_was_adopted is True
    assert accepted.stat().st_ino == orphan_inode
    promoted = json.loads(Path(layout["primary_config"]).read_text())
    assert promoted["sherpa"]["speaker_enroll_embedding"] == str(accepted)


def test_huge_integer_embedding_is_cli_refusal_without_writes(tmp_path, capsys):
    layout = _layout(tmp_path)
    candidate = Path(layout["candidate"])
    primary = Path(layout["primary_config"])
    primary_before = primary.read_bytes()
    data = json.loads(candidate.read_text())
    data["embedding"][0] = 10**1000
    candidate.write_text(json.dumps(data))
    os.chmod(candidate, 0o600)

    assert main(_cli_args(layout)) == 2

    captured = capsys.readouterr()
    assert "out-of-range number" in captured.err
    assert not Path(layout["accepted"]).exists()
    assert primary.read_bytes() == primary_before


@pytest.mark.parametrize("kind", ("mismatch", "wrong_mode", "candidate_alias"))
def test_promote_refuses_unsafe_existing_accepted_without_replacing_it(
    tmp_path, kind
):
    layout = _layout(tmp_path)
    accepted = Path(layout["accepted"])
    if kind == "mismatch":
        accepted.write_text('{"embedding":[1.0],"marker":"racer"}')
        os.chmod(accepted, 0o600)
        expected = "does not exactly match"
    elif kind == "wrong_mode":
        accepted.write_bytes(Path(layout["candidate"]).read_bytes())
        os.chmod(accepted, 0o640)
        expected = "not mode 600"
    else:
        os.link(layout["candidate"], accepted)
        expected = "exactly one link|aliases one inode"
    accepted_before = accepted.read_bytes()
    config_before = Path(layout["primary_config"]).read_bytes()

    with pytest.raises(PromotionError, match=expected):
        _call(layout)

    assert accepted.read_bytes() == accepted_before
    assert Path(layout["primary_config"]).read_bytes() == config_before


def test_exact_destination_race_is_adopted_without_replacement(tmp_path, monkeypatch):
    layout = _layout(tmp_path)
    accepted = Path(layout["accepted"])
    real_link = promo.os.link

    def racing_link(source, destination, *args, **kwargs):
        Path(destination).write_bytes(Path(source).read_bytes())
        os.chmod(destination, 0o600)
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(promo.os, "link", racing_link)

    result = _call(layout)

    assert result.accepted_was_adopted is True
    assert accepted.read_bytes() == Path(layout["candidate"]).read_bytes()
    promoted = json.loads(Path(layout["primary_config"]).read_text())
    assert promoted["sherpa"]["speaker_enroll_embedding"] == str(accepted)


def test_mismatched_destination_race_survives_and_primary_stays_v4(
    tmp_path, monkeypatch
):
    layout = _layout(tmp_path)
    accepted = Path(layout["accepted"])
    primary_before = Path(layout["primary_config"]).read_bytes()
    real_link = promo.os.link

    def racing_link(source, destination, *args, **kwargs):
        Path(destination).write_text('{"racer":"must survive"}')
        os.chmod(destination, 0o600)
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(promo.os, "link", racing_link)

    with pytest.raises(PromotionError, match="does not exactly match"):
        _call(layout)

    assert accepted.read_text() == '{"racer":"must survive"}'
    assert Path(layout["primary_config"]).read_bytes() == primary_before
    assert Path(layout["source"]).read_bytes() == layout["source_bytes"]
    assert Path(layout["backup"]).read_bytes() == layout["source_bytes"]


def test_exact_eexist_race_plus_temp_cleanup_failure_is_exit_four(
    tmp_path, monkeypatch, capsys
):
    layout = _layout(tmp_path)
    accepted = Path(layout["accepted"])
    real_link = promo.os.link
    real_unlink = promo.os.unlink

    def racing_link(source, destination, *args, **kwargs):
        Path(destination).write_bytes(Path(source).read_bytes())
        os.chmod(destination, 0o600)
        return real_link(source, destination, *args, **kwargs)

    def fail_temp_cleanup(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.name.startswith(f".{accepted.name}."):
            raise OSError("injected EEXIST temporary cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(promo.os, "link", racing_link)
    monkeypatch.setattr(promo.os, "unlink", fail_temp_cleanup)

    assert main(_cli_args(layout)) == 4
    assert "outcome ambiguous" in capsys.readouterr().err
    assert accepted.read_bytes() == Path(layout["candidate"]).read_bytes()


def test_existing_exact_accepted_file_sync_failure_is_exit_four(
    tmp_path, monkeypatch, capsys
):
    layout = _layout(tmp_path)
    accepted = Path(layout["accepted"])
    accepted.write_bytes(Path(layout["candidate"]).read_bytes())
    os.chmod(accepted, 0o600)

    def fail_file_sync(_snapshot):
        raise OSError("injected accepted-file sync failure")

    monkeypatch.setattr(promo, "_strict_fsync_bound_file", fail_file_sync)

    assert main(_cli_args(layout)) == 4
    assert "outcome ambiguous" in capsys.readouterr().err


def test_eexist_sync_ambiguity_plus_cleanup_failure_stays_exit_four(
    tmp_path, monkeypatch, capsys
):
    layout = _layout(tmp_path)
    accepted = Path(layout["accepted"])
    real_link = promo.os.link
    real_unlink = promo.os.unlink

    def racing_link(source, destination, *args, **kwargs):
        Path(destination).write_bytes(Path(source).read_bytes())
        os.chmod(destination, 0o600)
        return real_link(source, destination, *args, **kwargs)

    def fail_file_sync(_snapshot):
        raise OSError("injected accepted-file sync failure")

    def fail_temp_cleanup(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.name.startswith(f".{accepted.name}."):
            raise OSError("injected EEXIST temporary cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(promo.os, "link", racing_link)
    monkeypatch.setattr(promo, "_strict_fsync_bound_file", fail_file_sync)
    monkeypatch.setattr(promo.os, "unlink", fail_temp_cleanup)

    assert main(_cli_args(layout)) == 4
    assert "outcome ambiguous" in capsys.readouterr().err


def test_candidate_swap_during_copy_refuses_before_accepted_publish(
    tmp_path, monkeypatch
):
    layout = _layout(tmp_path)
    candidate = Path(layout["candidate"])
    original_candidate = candidate.read_bytes()
    primary_before = Path(layout["primary_config"]).read_bytes()
    real_publish = promo._publish_or_adopt_verified_copy

    def swap_then_publish(payload, destination, *, guards):
        candidate.unlink()
        candidate.write_bytes(original_candidate)
        os.chmod(candidate, 0o600)
        return real_publish(payload, destination, guards=guards)

    monkeypatch.setattr(
        promo, "_publish_or_adopt_verified_copy", swap_then_publish
    )

    with pytest.raises(PromotionError, match="populated v5 candidate changed"):
        _call(layout)

    assert candidate.read_bytes() == original_candidate
    assert not Path(layout["accepted"]).exists()
    assert Path(layout["primary_config"]).read_bytes() == primary_before
    assert Path(layout["source"]).read_bytes() == layout["source_bytes"]
    assert Path(layout["backup"]).read_bytes() == layout["source_bytes"]


def test_config_swap_after_staging_is_exit_four_and_preserves_racer(
    tmp_path, monkeypatch, capsys
):
    layout = _layout(tmp_path)
    primary = Path(layout["primary_config"])
    candidate_before = Path(layout["candidate"]).read_bytes()
    replacement = b'{"replacement":"must survive"}'
    def swap_then_write(path, payload, **kwargs):
        target = Path(path)
        target.unlink()
        target.write_bytes(replacement)
        os.chmod(target, 0o600)
        raise OSError("injected non-cooperating config race")

    monkeypatch.setattr(promo, "_atomic_replace_config", swap_then_write)

    code = main(_cli_args(layout))

    captured = capsys.readouterr()
    assert code == 4
    assert "outcome ambiguous" in captured.err
    assert Path(layout["accepted"]).read_bytes() == candidate_before
    assert primary.read_bytes() == replacement
    assert Path(layout["candidate"]).read_bytes() == candidate_before
    assert Path(layout["source"]).read_bytes() == layout["source_bytes"]
    assert Path(layout["backup"]).read_bytes() == layout["source_bytes"]


def test_config_failure_stages_retryable_orphan_that_next_run_adopts(
    tmp_path, monkeypatch
):
    layout = _layout(tmp_path)
    real_atomic_write = promo._atomic_replace_config

    def fail_write(*_args, **_kwargs):
        raise OSError("injected config failure")

    monkeypatch.setattr(promo, "_atomic_replace_config", fail_write)
    with pytest.raises(PromotionStagedError, match="exactly staged"):
        _call(layout)
    assert Path(layout["accepted"]).exists()

    monkeypatch.setattr(promo, "_atomic_replace_config", real_atomic_write)
    result = _call(layout)

    assert result.accepted_was_adopted is True
    promoted = json.loads(Path(layout["primary_config"]).read_text())
    assert promoted["sherpa"]["speaker_enroll_embedding"] == str(
        layout["accepted"]
    )


@pytest.mark.parametrize("mutated_key", ("accepted", "candidate"))
def test_exit_three_reproves_all_state_after_primary_confirmation_read(
    tmp_path, monkeypatch, mutated_key
):
    layout = _layout(tmp_path)
    primary = Path(layout["primary_config"])
    target = Path(layout[mutated_key])
    real_read = promo._read_bound_json
    primary_reads = 0

    def fail_config(*_args, **_kwargs):
        raise OSError("injected pre-replace failure")

    def mutate_during_confirmation(path, *, label):
        nonlocal primary_reads
        result = real_read(path, label=label)
        if Path(path) == primary:
            primary_reads += 1
            if primary_reads == 2:
                target.write_bytes(target.read_bytes() + b" ")
                os.chmod(target, 0o600)
        return result

    monkeypatch.setattr(promo, "_atomic_replace_config", fail_config)
    monkeypatch.setattr(promo, "_read_bound_json", mutate_during_confirmation)

    with pytest.raises(PromotionAmbiguousError, match="could not be confirmed"):
        _call(layout)

    assert primary_reads == 2


def test_success_performs_no_fallible_post_commit_read(tmp_path, monkeypatch):
    layout = _layout(tmp_path)
    committed = False
    real_atomic_write = promo._atomic_replace_config
    real_read = promo._read_bound_json

    def tracked_commit(*args, **kwargs):
        nonlocal committed
        real_atomic_write(*args, **kwargs)
        committed = True

    def refuse_post_commit_read(*args, **kwargs):
        if committed:
            raise AssertionError("no read is allowed after the config commit point")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(promo, "_atomic_replace_config", tracked_commit)
    monkeypatch.setattr(promo, "_read_bound_json", refuse_post_commit_read)

    result = _call(layout)

    assert result.accepted_enrollment == layout["accepted"]
    promoted = json.loads(Path(layout["primary_config"]).read_text())
    assert promoted["sherpa"]["speaker_enroll_embedding"] == str(
        layout["accepted"]
    )


@pytest.mark.parametrize("target_key", ("candidate", "primary_config", "accepted"))
def test_bound_read_rejects_same_bytes_path_aba(tmp_path, monkeypatch, target_key):
    layout = _layout(tmp_path)
    target = Path(layout[target_key])
    if target_key == "accepted":
        target.write_bytes(Path(layout["candidate"]).read_bytes())
        os.chmod(target, 0o600)
    original = target.read_bytes()
    mode = _mode(target)
    target_inode = target.stat().st_ino
    real_read = promo.os.read
    swapped = False

    def racing_read(fd, size):
        nonlocal swapped
        chunk = real_read(fd, size)
        if (
            not swapped
            and chunk
            and os.fstat(fd).st_ino == target_inode
        ):
            swapped = True
            target.unlink()
            target.write_bytes(original)
            os.chmod(target, mode)
        return chunk

    monkeypatch.setattr(promo.os, "read", racing_read)

    with pytest.raises(PromotionError, match="changed while reading"):
        _call(layout)

    assert swapped is True


def test_candidate_same_size_same_mtime_mutation_is_detected(tmp_path, monkeypatch):
    layout = _layout(tmp_path)
    candidate = Path(layout["candidate"])
    original = candidate.read_bytes()
    original_stat = candidate.stat()
    real_publish = promo._publish_or_adopt_verified_copy

    def mutate_then_publish(payload, destination, *, guards):
        changed = bytearray(original)
        changed[-2] = ord(" ") if changed[-2] != ord(" ") else ord("\n")
        candidate.write_bytes(changed)
        os.chmod(candidate, 0o600)
        os.utime(
            candidate,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        return real_publish(payload, destination, guards=guards)

    monkeypatch.setattr(promo, "_publish_or_adopt_verified_copy", mutate_then_publish)

    with pytest.raises(PromotionError, match="populated v5 candidate changed"):
        _call(layout)


def test_candidate_new_hardlink_is_detected_before_staging(tmp_path, monkeypatch):
    layout = _layout(tmp_path)
    candidate = Path(layout["candidate"])
    alias = candidate.with_name("candidate-alias.json")
    real_publish = promo._publish_or_adopt_verified_copy

    def alias_then_publish(payload, destination, *, guards):
        os.link(candidate, alias)
        return real_publish(payload, destination, guards=guards)

    monkeypatch.setattr(promo, "_publish_or_adopt_verified_copy", alias_then_publish)

    with pytest.raises(PromotionError, match="populated v5 candidate changed"):
        _call(layout)


def test_private_file_validator_checks_uid_mode_and_link_count(tmp_path):
    path = tmp_path / "private.json"
    path.write_text("{}")
    os.chmod(path, 0o600)
    snapshot, _ = promo._read_bound_bytes(path, label="private test")

    for field, value, expected in (
        ("uid", snapshot.uid + 1, "current user"),
        ("mode", 0o640, "mode 600"),
        ("nlink", 2, "exactly one link"),
    ):
        altered = dataclasses.replace(snapshot, **{field: value})
        with pytest.raises(PromotionError, match=expected):
            promo._require_private_current_user(altered, label="private test")


def test_group_writable_destination_directory_is_refused(tmp_path):
    layout = _layout(tmp_path)
    speaker_dir = Path(layout["speaker_dir"])
    os.chmod(speaker_dir, 0o770)

    with pytest.raises(PromotionError, match="group/world writable"):
        _call(layout)


def test_existing_config_lock_refuses_second_promoter(tmp_path):
    layout = _layout(tmp_path)
    primary = Path(layout["primary_config"])

    with promo._config_lock(primary):
        with pytest.raises(PromotionError, match="holds the config lock"):
            _call(layout)


def test_config_lock_is_stable_private_file(tmp_path):
    layout = _layout(tmp_path)
    primary = Path(layout["primary_config"])

    with promo._config_lock(primary) as (_, first_path):
        first = first_path.stat()
    with promo._config_lock(primary) as (_, second_path):
        second = second_path.stat()

    assert first_path == second_path
    assert first.st_ino == second.st_ino
    assert stat.S_IMODE(second.st_mode) == 0o600
    assert second.st_uid == os.getuid()
    assert second.st_nlink == 1


def test_config_lock_is_held_through_replace(tmp_path, monkeypatch):
    layout = _layout(tmp_path)
    primary = Path(layout["primary_config"])
    lock_path = primary.parent / f".{primary.name}.enrollment-promotion.lock"
    real_replace = promo.os.replace
    checked = False

    def checked_replace(source, destination):
        nonlocal checked
        fd = os.open(lock_path, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        checked = True
        return real_replace(source, destination)

    monkeypatch.setattr(promo.os, "replace", checked_replace)

    _call(layout)

    assert checked is True


def test_noncooperating_check_replace_race_is_outside_lock_guarantee(
    tmp_path, monkeypatch
):
    layout = _layout(tmp_path)
    primary = Path(layout["primary_config"])
    real_replace = promo.os.replace
    raced = False

    def ignored_lock_replace(source, destination):
        nonlocal raced
        target = Path(destination)
        if target == primary:
            target.unlink()
            target.write_text('{"noncooperating":"racer"}')
            os.chmod(target, 0o600)
            raced = True
        return real_replace(source, destination)

    monkeypatch.setattr(promo.os, "replace", ignored_lock_replace)

    result = _call(layout)

    assert raced is True
    assert result.primary_config == primary
    promoted = json.loads(primary.read_text())
    assert promoted["sherpa"]["speaker_enroll_embedding"] == str(layout["accepted"])


def test_accepted_directory_sync_failure_is_exit_four(tmp_path, monkeypatch, capsys):
    layout = _layout(tmp_path)
    primary_before = Path(layout["primary_config"]).read_bytes()
    real_sync = promo._strict_fsync_directory

    def fail_accepted(path, *, label):
        if label == "accepted enrollment directory":
            raise OSError("injected accepted-directory sync failure")
        return real_sync(path, label=label)

    monkeypatch.setattr(promo, "_strict_fsync_directory", fail_accepted)

    assert main(_cli_args(layout)) == 4
    assert "outcome ambiguous" in capsys.readouterr().err
    assert Path(layout["primary_config"]).read_bytes() == primary_before
    assert Path(layout["accepted"]).exists()


def test_accepted_temp_file_sync_failure_is_exit_two(tmp_path, monkeypatch, capsys):
    layout = _layout(tmp_path)
    accepted = Path(layout["accepted"])
    primary_before = Path(layout["primary_config"]).read_bytes()
    real_fsync = promo.os.fsync

    def fail_accepted_temp(fd):
        target = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if target.name.startswith(f".{accepted.name}."):
            raise OSError("injected accepted-temp file sync failure")
        return real_fsync(fd)

    monkeypatch.setattr(promo.os, "fsync", fail_accepted_temp)

    assert main(_cli_args(layout)) == 2
    assert "promotion refused" in capsys.readouterr().err
    assert not accepted.exists()
    assert Path(layout["primary_config"]).read_bytes() == primary_before


def test_config_temp_file_sync_failure_is_exit_three(tmp_path, monkeypatch, capsys):
    layout = _layout(tmp_path)
    primary = Path(layout["primary_config"])
    primary_before = primary.read_bytes()
    real_fsync = promo.os.fsync

    def fail_config_temp(fd):
        target = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if target.name.startswith(f".{primary.name}.") and target.name.endswith(".tmp"):
            raise OSError("injected config-temp file sync failure")
        return real_fsync(fd)

    monkeypatch.setattr(promo.os, "fsync", fail_config_temp)

    assert main(_cli_args(layout)) == 3
    assert "staged but not activated" in capsys.readouterr().err
    assert Path(layout["accepted"]).exists()
    assert primary.read_bytes() == primary_before


def test_persistent_post_link_cleanup_failure_is_exit_four(
    tmp_path, monkeypatch, capsys
):
    layout = _layout(tmp_path)
    accepted = Path(layout["accepted"])
    real_unlink = promo.os.unlink
    attempts = 0

    def fail_cleanup(path, *args, **kwargs):
        nonlocal attempts
        candidate = Path(path)
        if candidate.name.startswith(f".{accepted.name}."):
            attempts += 1
            raise OSError("injected temporary-link cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(promo.os, "unlink", fail_cleanup)

    assert main(_cli_args(layout)) == 4
    assert "outcome ambiguous" in capsys.readouterr().err
    assert attempts >= 2
    assert accepted.exists()


def test_lock_release_failure_does_not_false_refuse_committed_config(
    tmp_path, monkeypatch
):
    layout = _layout(tmp_path)
    real_flock = promo.fcntl.flock

    def fail_unlock(fd, operation):
        if operation == fcntl.LOCK_UN:
            raise OSError("injected unlock failure")
        return real_flock(fd, operation)

    monkeypatch.setattr(promo.fcntl, "flock", fail_unlock)

    result = _call(layout)

    assert result.accepted_enrollment == layout["accepted"]


def test_lock_close_failure_does_not_false_refuse_committed_config(
    tmp_path, monkeypatch
):
    layout = _layout(tmp_path)
    primary = Path(layout["primary_config"])
    lock_path = primary.parent / f".{primary.name}.enrollment-promotion.lock"
    real_close = promo.os.close
    failed = False

    def fail_lock_close(fd):
        nonlocal failed
        try:
            target = Path(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            target = None
        if not failed and target == lock_path:
            failed = True
            real_close(fd)
            raise OSError("injected post-commit lock close failure")
        return real_close(fd)

    monkeypatch.setattr(promo.os, "close", fail_lock_close)

    result = _call(layout)

    assert failed is True
    assert result.accepted_enrollment == layout["accepted"]


def test_config_directory_sync_failure_is_exit_four_after_replace(
    tmp_path, monkeypatch, capsys
):
    layout = _layout(tmp_path)
    real_sync = promo._strict_fsync_directory

    def fail_config(path, *, label):
        if label == "primary config directory":
            raise OSError("injected config-directory sync failure")
        return real_sync(path, label=label)

    monkeypatch.setattr(promo, "_strict_fsync_directory", fail_config)

    assert main(_cli_args(layout)) == 4
    assert "outcome ambiguous" in capsys.readouterr().err
    promoted = json.loads(Path(layout["primary_config"]).read_text())
    assert promoted["sherpa"]["speaker_enroll_embedding"] == str(layout["accepted"])


def test_exit_three_requires_independent_accepted_and_stable_guards(
    tmp_path, monkeypatch
):
    layout = _layout(tmp_path)
    accepted = Path(layout["accepted"])
    candidate = Path(layout["candidate"])

    def relink_then_fail(*_args, **_kwargs):
        accepted.unlink()
        os.link(candidate, accepted)
        raise OSError("injected post-stage relink")

    monkeypatch.setattr(promo, "_atomic_replace_config", relink_then_fail)

    with pytest.raises(PromotionAmbiguousError, match="could not be confirmed"):
        _call(layout)


_PROTECTED_KEYS = (
    "feature_config",
    "primary_config",
    "candidate",
    "source",
    "backup",
    "accepted",
)
_ALIAS_PAIRS = tuple(
    (left, right)
    for index, left in enumerate(_PROTECTED_KEYS)
    for right in _PROTECTED_KEYS[index + 1 :]
)


@pytest.mark.parametrize(("left_key", "right_key"), _ALIAS_PAIRS)
def test_all_six_protected_objects_must_have_distinct_inodes(
    tmp_path, left_key, right_key
):
    layout = _layout(tmp_path)
    left = Path(layout[left_key])
    right = Path(layout[right_key])
    if right.exists() or right.is_symlink():
        right.unlink()
    os.link(left, right)

    with pytest.raises(
        PromotionError,
        match="aliases|exactly one link|not mode 600|changed since preparation",
    ):
        _call(layout)
