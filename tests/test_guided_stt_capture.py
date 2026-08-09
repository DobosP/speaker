from __future__ import annotations

import hashlib
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
    guided_stt_summary_contract_sha256,
)
from tools.guided_stt_capture import (
    GuidedCapturePublicationError,
    load_verified_guided_capture_bundle,
    publish_guided_capture_bundle_receipt,
    publish_guided_capture_contract,
    verify_guided_capture_bundle,
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


def _write_terminal_bundle(publication) -> tuple[Path, Path]:
    run_dir = publication.plan_path.parent
    run_id = "capture-test"
    paths = {
        "recording": run_dir / f"run-{run_id}.wav",
        "pre_dsp_reference": run_dir / f"run-{run_id}.pre-dsp.wav",
        "playback_reference": run_dir / f"run-{run_id}.ref.wav",
        "asr_input_reference": run_dir / f"run-{run_id}.asr.wav",
        "final_model_input": run_dir / f"run-{run_id}.final-input.f32le",
        "diagnostic_timeline": run_dir / f"run-{run_id}.timeline.jsonl",
    }
    for index, path in enumerate(paths.values(), start=1):
        path.write_bytes(bytes([index]))
        path.chmod(0o600)
    manifest = run_dir / f"run-{run_id}.diagnostic.json"
    manifest.write_bytes(b'{"complete":true}\n')
    manifest.chmod(0o600)
    plan = built_in_guided_stt_plan()
    meta: dict[str, object] = {
        "session": "local",
        "topology": "local_device",
        "audio_egress": "device_only",
        "engine": "sherpa",
        "llm": "disabled",
        "device": "desktop_gpu_4090",
        "final_stt_profile": "sense-voice",
        "final_stt_profile_sha256": _PROFILE_SHA,
        "final_stt_profile_schema_version": 1,
        "capture_only_contract_sha256": publication.contract_sha256,
        "capture_only_profile_sha256": _PROFILE_SHA,
        "capture_only_sherpa_sha256": _SHERPA_SHA,
        "execution_purpose": "stt_capture_only",
        "llm_constructed": False,
        "control_plane_constructed": False,
        "tts_constructed": False,
        "keyword_spotter_constructed": False,
        "speaker_gate_constructed": False,
        "playback_worker_constructed": False,
        "output_device_queried": False,
        "owner_authority": False,
        "speaker_identity_policy": "off_for_session",
        "capture_protocol": GUIDED_STT_CAPTURE_PROTOCOL,
        "capture_plan_id": plan.plan_id,
        "capture_plan_sha256": plan.sha256,
        "capture_planned_cases": plan.case_count,
        "capture_accepted_finals": plan.case_count,
        "capture_state": "open",
        "capture_completed": True,
        "effects_enabled": False,
        "diagnostic_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "diagnostic_status": {"status": "complete", "failure_codes": []},
        "diagnostic_manifest": str(manifest),
        **{key: str(path) for key, path in paths.items()},
    }
    meta["capture_summary_contract_sha256"] = guided_stt_summary_contract_sha256(meta)
    summary = run_dir / f"run-{run_id}.summary.json"
    summary.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "log_path": str(run_dir / f"run-{run_id}.txt"),
                "meta": meta,
                "counts": {
                    "llm_requests": 0,
                    "turns": 0,
                    "transcript_entries": 0,
                    "errors": 0,
                },
                "transcript": [],
                "turns": [],
                "llm": {"requests": []},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary.chmod(0o600)
    return summary, manifest


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


def test_close_time_verifier_holds_parent_fd_across_leaf_checks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    publication = _publish(tmp_path)
    run_dir = publication.plan_path.parent
    moved = run_dir.with_name("moved-private-run")
    real_verify = guided_capture_module._verify_file_at
    replaced = False

    def replace_parent_then_verify(*args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            run_dir.rename(moved)
            run_dir.mkdir(mode=0o700)
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(
        guided_capture_module,
        "_verify_file_at",
        replace_parent_then_verify,
    )

    with pytest.raises(GuidedCapturePublicationError):
        verify_guided_capture_publication(publication)

    assert replaced is True


def test_publication_uses_anonymous_staging_without_path_unlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = _run_dir(tmp_path)
    foreign = run_dir / "foreign-private-inode"
    foreign.write_bytes(b"preserve-foreign-private-inode")
    foreign.chmod(0o600)
    unlink_calls: list[tuple[object, ...]] = []

    def reject_path_unlink(*args, **_kwargs):
        unlink_calls.append(args)
        raise AssertionError("anonymous publication must not unlink a pathname")

    with monkeypatch.context() as patch:
        patch.setattr(guided_capture_module.os, "unlink", reject_path_unlink)
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

    assert unlink_calls == []
    assert foreign.read_bytes() == b"preserve-foreign-private-inode"
    assert not any(
        path.name.startswith(".guided-capture-") for path in run_dir.iterdir()
    )


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


def test_terminal_receipt_round_trip_binds_contract_summary_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import core.diagnostic_bundle as diagnostic_bundle

    publication = _publish(tmp_path)
    _summary, manifest = _write_terminal_bundle(publication)
    monkeypatch.setattr(diagnostic_bundle, "validate_manifest", lambda *_a, **_k: True)

    receipt = publish_guided_capture_bundle_receipt(publication)
    bundle = load_verified_guided_capture_bundle(publication.plan_path.parent)

    assert receipt.verified_bundle == bundle
    assert receipt.receipt_sha256 == bundle.receipt_sha256
    assert receipt.summary_contract_sha256 == bundle.summary_contract_sha256
    assert bundle.contract_sha256 == publication.contract_sha256
    assert bundle.plan_sha256 == publication.plan_sha256
    assert (
        bundle.diagnostic_manifest_sha256
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert bundle.final_stt_profile == "sense-voice"
    assert bundle.capture_config_sha256 == _CAPTURE_CONFIG_SHA
    assert bundle.effective_sherpa_sha256 == _SHERPA_SHA
    assert bundle.case_count == 16
    assert stat.S_IMODE(receipt.receipt_path.stat().st_mode) == 0o600
    assert receipt.receipt_path.stat().st_nlink == 1
    raw = receipt.receipt_path.read_bytes()
    assert b"Search in my vault" not in raw
    assert b"expected-tool" not in raw
    verify_guided_capture_bundle(bundle)


def test_terminal_receipt_never_clobbers_and_preserves_existing_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import core.diagnostic_bundle as diagnostic_bundle

    publication = _publish(tmp_path)
    _write_terminal_bundle(publication)
    receipt = publication.plan_path.with_name("capture-bundle-receipt.json")
    receipt.write_bytes(b"preserve-terminal-canary")
    receipt.chmod(0o600)
    monkeypatch.setattr(diagnostic_bundle, "validate_manifest", lambda *_a, **_k: True)

    with pytest.raises(GuidedCapturePublicationError):
        publish_guided_capture_bundle_receipt(publication)

    assert receipt.read_bytes() == b"preserve-terminal-canary"


@pytest.mark.parametrize(
    "node",
    ["receipt", "plan", "contract", "summary", "manifest"],
)
def test_verified_bundle_recheck_rejects_same_byte_inode_replacement(
    tmp_path: Path,
    monkeypatch,
    node: str,
) -> None:
    import core.diagnostic_bundle as diagnostic_bundle

    publication = _publish(tmp_path)
    _write_terminal_bundle(publication)
    monkeypatch.setattr(diagnostic_bundle, "validate_manifest", lambda *_a, **_k: True)
    receipt = publish_guided_capture_bundle_receipt(publication)
    bundle = receipt.verified_bundle
    path = {
        "receipt": bundle.receipt_path,
        "plan": bundle.plan_path,
        "contract": bundle.contract_path,
        "summary": bundle.summary_path,
        "manifest": bundle.diagnostic_manifest_path,
    }[node]
    replacement = bundle.run_dir / "same-byte-replacement"
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, path)

    with pytest.raises(GuidedCapturePublicationError):
        verify_guided_capture_bundle(bundle)


@pytest.mark.parametrize("node", ["plan", "contract", "summary", "manifest"])
def test_terminal_publisher_rejects_same_byte_precommit_replacement(
    tmp_path: Path,
    monkeypatch,
    node: str,
) -> None:
    import core.diagnostic_bundle as diagnostic_bundle

    publication = _publish(tmp_path)
    summary, manifest = _write_terminal_bundle(publication)
    monkeypatch.setattr(diagnostic_bundle, "validate_manifest", lambda *_a, **_k: True)
    target = {
        "plan": publication.plan_path,
        "contract": publication.contract_path,
        "summary": summary,
        "manifest": manifest,
    }[node]
    real_stage = guided_capture_module._stage_terminal_file

    def stage_then_replace(*args, **kwargs):
        staged = real_stage(*args, **kwargs)
        replacement = target.parent / "same-byte-precommit-replacement"
        replacement.write_bytes(target.read_bytes())
        replacement.chmod(0o600)
        os.replace(replacement, target)
        return staged

    monkeypatch.setattr(
        guided_capture_module,
        "_stage_terminal_file",
        stage_then_replace,
    )

    with pytest.raises(GuidedCapturePublicationError):
        publish_guided_capture_bundle_receipt(publication)

    assert not publication.plan_path.with_name("capture-bundle-receipt.json").exists()


def test_terminal_publisher_rejects_run_dir_replacement_before_final_link(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import core.diagnostic_bundle as diagnostic_bundle

    publication = _publish(tmp_path)
    _write_terminal_bundle(publication)
    monkeypatch.setattr(diagnostic_bundle, "validate_manifest", lambda *_a, **_k: True)
    run_dir = publication.plan_path.parent
    moved = run_dir.with_name("moved-terminal-run")
    real_stage = guided_capture_module._stage_terminal_file

    def stage_then_replace_parent(*args, **kwargs):
        staged = real_stage(*args, **kwargs)
        run_dir.rename(moved)
        run_dir.mkdir(mode=0o700)
        return staged

    monkeypatch.setattr(
        guided_capture_module,
        "_stage_terminal_file",
        stage_then_replace_parent,
    )

    with pytest.raises(GuidedCapturePublicationError):
        publish_guided_capture_bundle_receipt(publication)

    assert not (run_dir / "capture-bundle-receipt.json").exists()
    assert not (moved / "capture-bundle-receipt.json").exists()


def test_terminal_receipt_rejects_manifest_changed_after_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import core.diagnostic_bundle as diagnostic_bundle

    publication = _publish(tmp_path)
    _summary, manifest = _write_terminal_bundle(publication)
    manifest.write_bytes(b'{"complete":false}\n')
    manifest.chmod(0o600)
    monkeypatch.setattr(diagnostic_bundle, "validate_manifest", lambda *_a, **_k: True)

    with pytest.raises(GuidedCapturePublicationError):
        publish_guided_capture_bundle_receipt(publication)

    assert not publication.plan_path.with_name("capture-bundle-receipt.json").exists()
