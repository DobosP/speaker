from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest
import tools.guided_stt_capture as guided_capture_module

from core.guided_stt_plan import (
    GUIDED_STT_CAPTURE_PROTOCOL,
    built_in_guided_stt_plan,
    canonical_guided_stt_plan_bytes,
    guided_stt_route_availability_sha256,
)
from tools.guided_stt_capture import (
    GuidedCapturePublicationError,
    publish_guided_capture_contract,
    verify_guided_capture_publication,
)
from tools.tool_route_gate import ToolRouteGateProfile, profile_digest


_PROFILE_SHA = "a" * 64
_CAPTURE_CONFIG_SHA = "b" * 64
_SHERPA_SHA = "c" * 64


def _run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "private-run"
    path.mkdir(mode=0o700)
    return path


def _publish(tmp_path: Path):
    return publish_guided_capture_contract(
        _run_dir(tmp_path),
        final_stt_profile="sense-voice",
        final_stt_profile_sha256=_PROFILE_SHA,
        final_stt_profile_schema_version=1,
        effective_device="desktop_gpu_4090",
        effective_input_gain=1.0,
        capture_config_sha256=_CAPTURE_CONFIG_SHA,
        effective_sherpa_sha256=_SHERPA_SHA,
    )


def test_publication_binds_exact_plan_profile_policy_and_protocol(
    tmp_path: Path,
) -> None:
    publication = _publish(tmp_path)
    plan = built_in_guided_stt_plan()
    contract = json.loads(publication.contract_path.read_bytes())

    assert publication.plan_path.read_bytes() == canonical_guided_stt_plan_bytes()
    assert publication.plan_sha256 == plan.sha256
    assert publication.case_count == plan.case_count == 16
    assert contract["schema_version"] == 1
    assert contract["capture_protocol"] == GUIDED_STT_CAPTURE_PROTOCOL
    assert contract["execution_purpose"] == "stt_capture_only"
    assert contract["speaker_identity_policy"] == "off_for_session"
    assert contract["route_availability_sha256"] == (
        guided_stt_route_availability_sha256()
    )
    assert contract["tool_route_profile_sha256"] == profile_digest(
        ToolRouteGateProfile(
            vault_enabled=True,
            reminders_enabled=True,
            app_aliases=("obsidian",),
        )
    )
    assert contract["capture_environment"] == {
        "capture_config_sha256": _CAPTURE_CONFIG_SHA,
        "device_profile": "desktop_gpu_4090",
        "effective_input_gain": 1.0,
        "effective_sherpa_sha256": _SHERPA_SHA,
    }
    assert contract["plan"] == {
        "case_count": 16,
        "case_order_sha256": contract["plan"]["case_order_sha256"],
        "file": "reference-plan.json",
        "id": "agent-core-v1",
        "schema_version": 1,
        "sha256": plan.sha256,
    }
    assert len(contract["plan"]["case_order_sha256"]) == 64
    assert contract["final_stt_profile"] == {
        "name": "sense-voice",
        "schema_version": 1,
        "sha256": _PROFILE_SHA,
    }
    assert contract["side_effect_policy"] == {
        "capabilities": False,
        "control_commands": False,
        "keyword_spotter": False,
        "llm": False,
        "memory": False,
        "output_device": False,
        "owner_authority": False,
        "playback": False,
        "speaker_verification": False,
        "tts": False,
    }
    verify_guided_capture_publication(publication)


def test_publication_files_are_private_regular_single_link_and_contract_is_aggregate(
    tmp_path: Path,
) -> None:
    publication = _publish(tmp_path)

    for path in (publication.plan_path, publication.contract_path):
        metadata = path.stat()
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_uid == os.getuid()
        assert metadata.st_nlink == 1
    contract_raw = publication.contract_path.read_bytes()
    for phrase in (
        b"Search in my vault",
        b"Open Obsidian",
        b"Remind me",
        b"Bucharest",
        b"expected-tool",
    ):
        assert phrase not in contract_raw
    assert "private-run" not in repr(publication)


@pytest.mark.parametrize("existing_kind", ["file", "symlink", "hardlink"])
def test_publication_never_clobbers_an_existing_plan(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    run_dir = _run_dir(tmp_path)
    target = run_dir / "reference-plan.json"
    source = run_dir / "existing-private"
    source.write_bytes(b"preserve-private-canary")
    source.chmod(0o600)
    if existing_kind == "file":
        target.write_bytes(b"preserve-private-canary")
        target.chmod(0o600)
    elif existing_kind == "symlink":
        target.symlink_to(source.name)
    else:
        os.link(source, target)

    with pytest.raises(GuidedCapturePublicationError):
        publish_guided_capture_contract(
            run_dir,
            final_stt_profile="sense-voice",
            final_stt_profile_sha256=_PROFILE_SHA,
            final_stt_profile_schema_version=1,
            effective_device="desktop_gpu_4090",
            effective_input_gain=1.0,
            capture_config_sha256=_CAPTURE_CONFIG_SHA,
            effective_sherpa_sha256=_SHERPA_SHA,
        )

    assert target.read_bytes() == b"preserve-private-canary"
    assert not (run_dir / "capture-contract.json").exists()


def test_publication_never_clobbers_existing_contract_and_preserves_partial_plan(
    tmp_path: Path,
) -> None:
    run_dir = _run_dir(tmp_path)
    contract = run_dir / "capture-contract.json"
    contract.write_bytes(b"preserve-private-contract-canary")
    contract.chmod(0o600)

    with pytest.raises(GuidedCapturePublicationError):
        publish_guided_capture_contract(
            run_dir,
            final_stt_profile="sense-voice",
            final_stt_profile_sha256=_PROFILE_SHA,
            final_stt_profile_schema_version=1,
            effective_device="desktop_gpu_4090",
            effective_input_gain=1.0,
            capture_config_sha256=_CAPTURE_CONFIG_SHA,
            effective_sherpa_sha256=_SHERPA_SHA,
        )

    assert contract.read_bytes() == b"preserve-private-contract-canary"
    assert (run_dir / "reference-plan.json").read_bytes() == (
        canonical_guided_stt_plan_bytes()
    )


@pytest.mark.parametrize("mutation", ["bytes", "mode", "hardlink", "replacement"])
def test_close_time_verifier_rejects_plan_mutation(
    tmp_path: Path, mutation: str
) -> None:
    publication = _publish(tmp_path)
    path = publication.plan_path
    if mutation == "bytes":
        path.write_bytes(b"changed")
        path.chmod(0o600)
    elif mutation == "mode":
        path.chmod(0o640)
    elif mutation == "hardlink":
        os.link(path, path.parent / "unexpected-link")
    else:
        replacement = path.parent / "replacement"
        replacement.write_bytes(canonical_guided_stt_plan_bytes())
        replacement.chmod(0o600)
        os.replace(replacement, path)

    with pytest.raises(GuidedCapturePublicationError):
        verify_guided_capture_publication(publication)


def test_close_time_verifier_rejects_replacement_during_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    publication = _publish(tmp_path)
    replacement = publication.plan_path.parent / "replacement"
    replacement.write_bytes(canonical_guided_stt_plan_bytes())
    replacement.chmod(0o600)
    real_read = os.read
    replaced = False

    def replace_after_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, size)
        if not replaced:
            replaced = True
            os.replace(replacement, publication.plan_path)
        return chunk

    monkeypatch.setattr(guided_capture_module.os, "read", replace_after_read)

    with pytest.raises(GuidedCapturePublicationError):
        verify_guided_capture_publication(publication)

    assert replaced is True


@pytest.mark.parametrize(
    ("name", "digest", "version"),
    [
        ("../unsafe", _PROFILE_SHA, 1),
        ("sense-voice", "A" * 64, 1),
        ("sense-voice", _PROFILE_SHA, 0),
        ("sense-voice", _PROFILE_SHA, True),
    ],
)
def test_invalid_profile_binding_fails_before_publication(
    tmp_path: Path,
    name: str,
    digest: str,
    version: int,
) -> None:
    run_dir = _run_dir(tmp_path)

    with pytest.raises(GuidedCapturePublicationError):
        publish_guided_capture_contract(
            run_dir,
            final_stt_profile=name,
            final_stt_profile_sha256=digest,
            final_stt_profile_schema_version=version,
            effective_device="desktop_gpu_4090",
            effective_input_gain=1.0,
            capture_config_sha256=_CAPTURE_CONFIG_SHA,
            effective_sherpa_sha256=_SHERPA_SHA,
        )

    assert list(run_dir.iterdir()) == []
