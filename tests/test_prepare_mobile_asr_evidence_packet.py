from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import stat
from types import SimpleNamespace

import pytest

from tools import prepare_mobile_asr_evidence_packet as packet
from tools import prepare_primock57_conversation_fixture as primock57


def _private_file(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)
    path.chmod(0o600)


def _source_arguments(root: str) -> list[str]:
    return [
        "--command-noise-root",
        f"{root}/command",
        "--demand-noise-root",
        f"{root}/demand",
        "--notsofar-far-field-root",
        f"{root}/notsofar",
        "--primock-isolated-root",
        f"{root}/isolated",
        "--primock-overlap-root",
        f"{root}/overlap",
    ]


def _synthetic_packet_contract() -> packet._SyntheticTestPacketContract:
    rows = (
        ("synthetic-manifest.json", b'{"synthetic":true}\n'),
        (
            "synthetic-component-receipt.json",
            b'{"production_inputs":false}\n',
        ),
        ("synthetic-audio.f32le", b"\x00\x00\x00\x00"),
    )
    return packet._SyntheticTestPacketContract(
        packet_id="synthetic-mobile-packet-test-v1",
        artifacts=tuple(
            packet._Artifact(
                name=name,
                payload=raw,
                sha256=hashlib.sha256(raw).hexdigest(),
            )
            for name, raw in rows
        ),
    )


def test_lock_is_exact_self_digested_and_has_fixed_totals() -> None:
    lock = packet.load_packet_lock()
    raw = packet.DEFAULT_LOCK.read_bytes()
    value = json.loads(raw)
    digest_value = dict(value)
    digest_value.pop("recipe_sha256")

    assert len(raw) == packet.LOCK_FILE_BYTES == 8481
    assert hashlib.sha256(raw).hexdigest() == packet.LOCK_FILE_SHA256
    assert packet._canonical_sha256(digest_value) == packet.LOCK_RECIPE_SHA256
    assert (
        tuple(item.component_id for item in lock.components) == packet._EXPECTED_ORDER
    )
    assert sum(item.logical_cases for item in lock.components) == 123
    assert sum(item.pcm_inputs for item in lock.components) == 129
    assert sum(item.pcm_bytes for item in lock.components) == 39_299_500
    assert sum(item.pcm_bytes // 4 for item in lock.components) == 9_824_875
    assert value["language"] == {"bcp47": "en", "english_only": True}
    assert value["publication"]["redistribution"] is False
    assert value["evidence_scope"]["qualification_authority"] is False
    assert value["evidence_scope"]["mobile_model_identity_authority"] is False
    assert value["evidence_scope"]["evaluation_result_authority"] is False
    assert value["schema_version"] == 2
    assert value["kind"] == "mobile-asr-evidence-packet-lock-v2"
    assert value["packet_id"] == "mobile-asr-english-five-component-v2"


def test_lock_pins_exact_runtime_bound_demand_receipt_and_16k_archives() -> None:
    value = json.loads(packet.DEFAULT_LOCK.read_bytes())
    demand = value["components"][1]

    assert demand["manifest"]["sha256"] == (
        "48797d031f24f19694f3c6ba81f6eca53c32f2286751612bc5661779396954a8"
    )
    assert demand["receipt"]["sha256"] == (
        "24a7005ea6b327cc89f633bcd9d9edf20a6bb17c7afe0d60c49f37753e1e2f02"
    )
    assert [row["file"] for row in demand["upstream"]["archives"]] == [
        "DKITCHEN_16k.zip",
        "DLIVING_16k.zip",
        "DWASHING_16k.zip",
    ]
    assert demand["upstream"]["receipt_runtime_binding"] == (
        "exact-current-runtime-receipt-reviewed-v2"
    )


def test_v1_lock_is_preserved_as_rejected_history() -> None:
    historical = packet.DEFAULT_LOCK.with_name(
        "mobile-asr-evidence-packet-v1.lock.json"
    )
    raw = historical.read_bytes()

    assert len(raw) == 8506
    assert hashlib.sha256(raw).hexdigest() == (
        "268377ba9ef648647112210cd2c812cf9ea8e9783f366676dee1c4fe1dc07820"
    )
    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet.load_packet_lock(historical)


def test_lock_rejects_alternate_path_and_recomputed_raw_tamper(tmp_path: Path) -> None:
    raw = packet.DEFAULT_LOCK.read_bytes()
    alternate = tmp_path / packet.DEFAULT_LOCK.name
    alternate.write_bytes(raw)

    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet.load_packet_lock(alternate)

    tampered = raw.replace(b"DKITCHEN_16k.zip", b"DKITCHEN_48k.zip")
    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet._parse_lock(
            tampered,
            expected_raw_sha256=hashlib.sha256(tampered).hexdigest(),
        )


def test_public_production_api_has_no_lock_or_digest_override() -> None:
    for function in (
        packet.preflight_packet,
        packet.publish_packet,
        packet.load_mobile_asr_evidence_packet,
    ):
        parameters = inspect.signature(function).parameters
        assert all(
            forbidden not in name
            for name in parameters
            for forbidden in ("lock", "receipt", "digest", "closure")
        )
    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet._parse_cli(
            [
                "preflight",
                *_source_arguments("/unavailable"),
                "--lock",
                "/attacker/lock.json",
            ]
        )


def test_metric_domains_are_independent_and_never_pooled() -> None:
    lock = packet.load_packet_lock()
    index = packet._packet_index(lock)

    assert index["metric_aggregation"] == {
        "component_domains_are_independent": True,
        "pooled_metric": None,
        "pooled_wer": False,
    }
    assert [row["metric_domain"] for row in index["components"]] == [
        "command-assertion-v1",
        "stratified-wer-cer-v1",
        "paired-channel-wer-cer-v1",
        "ordinary-wer-cer-v1",
        "two-utterance-min-order-wer-v1",
    ]


def test_cli_missing_current_prerequisites_is_one_detail_free_receipt(
    capfd: pytest.CaptureFixture[str],
) -> None:
    secret = "/definitely-missing/private-transcript-name"
    status = packet.main(["preflight", *_source_arguments(secret)])
    captured = capfd.readouterr()

    assert status == 2
    assert captured.err == ""
    assert json.loads(captured.out) == packet._SAFE_ERROR
    assert secret not in captured.out
    assert "command" not in captured.out


def test_publish_validates_every_input_before_reserving_output(tmp_path: Path) -> None:
    output = tmp_path / "packet"
    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet.publish_packet(
            {
                component: tmp_path / f"missing-{component}"
                for component in packet._EXPECTED_ORDER
            },
            output,
        )
    assert not output.exists()


def test_private_project_primitives_remain_bound() -> None:
    assert packet._strict_json.__module__ == "tools.streaming_stt.corpus"
    assert packet._new_private_output.__module__ == (
        "tools.streaming_stt.corpus_writer"
    )
    assert packet._write_new_private.__module__ == "tools.streaming_stt.corpus_writer"
    assert packet._verify_output_binding.__module__ == (
        "tools.streaming_stt.corpus_writer"
    )


def test_terminal_receipt_is_anonymous_until_guarded_link(tmp_path: Path) -> None:
    if not getattr(os, "O_TMPFILE", 0):
        pytest.skip("O_TMPFILE is required by the production contract")
    root = tmp_path / "packet"
    root.mkdir(mode=0o700)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    staged_fd = -1
    try:
        raw = b'{"complete":true}\n'
        staged_fd, staged = packet._stage_terminal_receipt(
            directory_fd,
            "packet-receipt.json",
            raw,
        )
        assert not (root / "packet-receipt.json").exists()
        assert os.fstat(staged_fd).st_nlink == 0
        guard_observations: list[bool] = []
        state = packet._TerminalCommitState()

        def guard() -> None:
            guard_observations.append((root / "packet-receipt.json").exists())

        packet._commit_terminal_receipt(
            directory_fd,
            staged_fd,
            staged,
            raw,
            state,
            commit_guard=guard,
        )

        metadata = (root / "packet-receipt.json").lstat()
        assert guard_observations == [False]
        assert state.committed is True
        assert (root / "packet-receipt.json").read_bytes() == raw
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1
    finally:
        if staged_fd >= 0:
            os.close(staged_fd)
        os.close(directory_fd)


def test_failed_terminal_guard_leaves_no_completion_marker(tmp_path: Path) -> None:
    if not getattr(os, "O_TMPFILE", 0):
        pytest.skip("O_TMPFILE is required by the production contract")
    root = tmp_path / "packet"
    root.mkdir(mode=0o700)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    staged_fd = -1
    try:
        raw = b'{"complete":true}\n'
        staged_fd, staged = packet._stage_terminal_receipt(
            directory_fd,
            "packet-receipt.json",
            raw,
        )
        state = packet._TerminalCommitState()

        def fail() -> None:
            raise packet.MobileAsrEvidencePacketError()

        with pytest.raises(packet.MobileAsrEvidencePacketError):
            packet._commit_terminal_receipt(
                directory_fd,
                staged_fd,
                staged,
                raw,
                state,
                commit_guard=fail,
            )
        assert state.committed is False
        assert not (root / "packet-receipt.json").exists()
    finally:
        if staged_fd >= 0:
            os.close(staged_fd)
        os.close(directory_fd)


def test_failed_committed_recovery_normalizes_detailed_ordinary_error(
    tmp_path: Path,
) -> None:
    detail = str(tmp_path / "private-transcript-name")
    output = tmp_path / "synthetic-corrupt-recovery"

    def corrupt_then_fail() -> None:
        receipt = output / "packet-receipt.json"
        receipt.write_bytes(b"corrupt\n")
        receipt.chmod(0o600)
        raise OSError(detail)

    with pytest.raises(packet.MobileAsrEvidencePacketError) as captured:
        packet._publish_synthetic_test_packet(
            _synthetic_packet_contract(),
            output,
            post_link_hook=corrupt_then_fail,
        )
    assert str(captured.value) == ""
    assert detail not in repr(captured.value)


def test_commit_signal_mask_contains_hup_int_and_term() -> None:
    expected = {int(value) for value in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
    assert packet._commit_signal_numbers() == expected


def test_private_root_rejects_recognized_git_ancestor(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    root = tmp_path / "private"
    root.mkdir(mode=0o700)

    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet._absolute_private_root(root)


def _synthetic_overlap_bundle(root: Path) -> tuple[packet.ComponentSpec, object]:
    manifest_raw = b'{"synthetic":"manifest"}\n'
    receipt_raw = b'{"synthetic":"receipt"}\n'
    _private_file(root / "overlap.json", manifest_raw)
    _private_file(root / "preparation-receipt.json", receipt_raw)
    cases = []
    total_bytes = 0
    for case_index in range(3):
        case: dict[str, object] = {}
        for field_name in ("role_a", "role_b", "mix"):
            name = f"synthetic-{case_index}-{field_name}.f32le"
            raw = bytes([case_index + 1, len(field_name), 0, 0])
            _private_file(root / name, raw)
            case[field_name] = {
                "bytes": len(raw),
                "file": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            total_bytes += len(raw)
        cases.append(case)
    spec = packet.ComponentSpec(
        component_id="synthetic-overlap",
        directory="synthetic-overlap",
        loader="synthetic-test-only",
        schema="synthetic-test-only",
        manifest_file="overlap.json",
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        receipt_file="preparation-receipt.json",
        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        licenses=("synthetic-test-only",),
        logical_cases=3,
        pcm_inputs=9,
        pcm_bytes=total_bytes,
        metric_domain="synthetic-test-only",
    )
    bundle = SimpleNamespace(
        cases=tuple(cases),
        digest=spec.manifest_sha256,
        path=root / spec.manifest_file,
        production_evidence=False,
        receipt_sha256=spec.receipt_sha256,
    )
    return spec, bundle


def _locked_primock_receipt(*, overlap: bool) -> dict[str, object]:
    source_rows = packet._load_locked_primock_source_rows()
    preparer_rows = [
        {"path": path, "sha256": sha256, "size_bytes": size_bytes}
        for path, size_bytes, sha256 in packet._PRIMOCK_PREPARER_ROWS
    ]
    selection = (
        packet._PRIMOCK_OVERLAP_SELECTION_SHA256
        if overlap
        else packet._PRIMOCK_ISOLATED_SELECTION_SHA256
    )
    value: dict[str, object] = {
        "accepted_license": packet._PRIMOCK_LICENSE_ID,
        "fixture_id": packet._PRIMOCK_FIXTURE_ID,
        "kind": (
            packet._PRIMOCK_OVERLAP_RECEIPT_KIND
            if overlap
            else packet._PRIMOCK_ISOLATED_RECEIPT_KIND
        ),
        "lock_recipe_sha256": packet._PRIMOCK_LOCK_RECIPE_SHA256,
        "preparer_files": preparer_rows,
        "privacy": packet._PRIMOCK_RECEIPT_PRIVACY,
        "production_evidence": True,
        "schema_version": 1,
        "selection_sha256": selection,
        "source_files": source_rows,
        "totals": {"cases": 3, "pcm_files": 9} if overlap else {"cases": 3},
    }
    if overlap:
        value.update(
            {
                "manifest": {
                    "bytes": 1,
                    "file": packet._PRIMOCK_OVERLAP_MANIFEST,
                    "sha256": "a" * 64,
                },
                "source_contract_sha256": (
                    packet._PRIMOCK_OVERLAP_SOURCE_CONTRACT_SHA256
                ),
            }
        )
    return value


@pytest.mark.parametrize("drift", ["add", "remove", "rename"])
def test_locked_primock_receipts_ignore_current_preparer_shape_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    current = list(primock57._PREPARER_FILES)
    if drift == "add":
        current.append("tools/streaming_stt/new-runtime.py")
    elif drift == "remove":
        current.pop()
    else:
        current[-1] = "tools/streaming_stt/protocol-next.py"
    monkeypatch.setattr(primock57, "_PREPARER_FILES", tuple(current))
    source_roles = list(primock57._SOURCE_ROLES)
    source_paths = list(primock57._SOURCE_PATHS)
    if drift == "add":
        source_roles.append("new-role")
        source_paths.append("new-source")
    elif drift == "remove":
        source_roles.pop()
        source_paths.pop()
    else:
        source_roles[-1] = "renamed-license"
        source_paths[-1] = "RENAMED-LICENSE.md"
    monkeypatch.setattr(primock57, "_SOURCE_ROLES", tuple(source_roles))
    monkeypatch.setattr(primock57, "_SOURCE_PATHS", tuple(source_paths))
    monkeypatch.setattr(
        primock57,
        "_preparer_closure",
        lambda: (_ for _ in ()).throw(AssertionError("must stay packet-local")),
    )
    for overlap in (False, True):
        value = _locked_primock_receipt(overlap=overlap)
        raw = packet._canonical_json(value, newline=True)
        parsed = packet._parse_locked_primock_receipt(raw, overlap=overlap)
        assert parsed["production_evidence"] is True


def test_locked_primock_receipts_reject_identity_and_closure_mutations() -> None:
    for overlap in (False, True):
        pristine = _locked_primock_receipt(overlap=overlap)
        canonical = packet._canonical_json(pristine, newline=True)
        with pytest.raises(packet.MobileAsrEvidencePacketError):
            packet._parse_locked_primock_receipt(canonical + b" ", overlap=overlap)
        mutations = [
            lambda value: value.update({"accepted_license": "CC0-1.0"}),
            lambda value: value.update({"production_evidence": False}),
            lambda value: value.update({"selection_sha256": "0" * 64}),
            lambda value: value["preparer_files"].append(
                {"path": "extra.py", "sha256": "0" * 64, "size_bytes": 1}
            ),
            lambda value: value["preparer_files"].pop(),
            lambda value: value["preparer_files"][0].update({"path": "renamed.py"}),
            lambda value: value["preparer_files"][0].update({"sha256": "0" * 64}),
            lambda value: value["source_files"][0].update({"role": "other"}),
            lambda value: value["source_files"][0].update({"sha256": "0" * 64}),
        ]
        if overlap:
            mutations.append(
                lambda value: value.update({"source_contract_sha256": "0" * 64})
            )
        for mutate in mutations:
            value = json.loads(packet._canonical_json(pristine))
            mutate(value)
            raw = packet._canonical_json(value, newline=True)
            with pytest.raises(packet.MobileAsrEvidencePacketError):
                packet._parse_locked_primock_receipt(raw, overlap=overlap)


def test_primock_source_lock_and_component_specs_stay_exactly_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = packet._PRIMOCK_SOURCE_LOCK.read_bytes()
    assert len(raw) == packet._PRIMOCK_SOURCE_LOCK_BYTES == 8_181
    assert hashlib.sha256(raw).hexdigest() == packet._PRIMOCK_SOURCE_LOCK_SHA256
    source_lock = json.loads(raw)
    projection = {
        name: source_lock[name]
        for name in (
            "fixture_id",
            "license_id",
            "output_artifacts",
            "recipe_sha256",
            "selection",
            "source",
        )
    }
    assert packet._canonical_sha256(projection) == (
        packet._PRIMOCK_SOURCE_LOCK_PROJECTION_SHA256
    )
    assert len(packet._load_locked_primock_source_rows()) == 5
    packet_lock = json.loads(packet.DEFAULT_LOCK.read_bytes())
    primock = {
        row["id"]: row["upstream"]
        for row in packet_lock["components"]
        if row["id"].startswith("primock-")
    }
    assert primock["primock-isolated"] == {
        "lock_file": "tools/streaming_stt/primock57-consultation01-v1.lock.json",
        "lock_recipe_sha256": packet._PRIMOCK_LOCK_RECIPE_SHA256,
        "lock_sha256": packet._PRIMOCK_SOURCE_LOCK_SHA256,
    }
    assert primock["primock-overlap"]["source_lock_recipe_sha256"] == (
        packet._PRIMOCK_LOCK_RECIPE_SHA256
    )
    assert primock["primock-overlap"]["source_lock_sha256"] == (
        packet._PRIMOCK_SOURCE_LOCK_SHA256
    )
    selected = packet.load_packet_lock()
    for component_id in ("primock-isolated", "primock-overlap"):
        spec = next(
            item for item in selected.components if item.component_id == component_id
        )
        packet._require_committed_component_spec(spec, component_id=component_id)
        with pytest.raises(packet.MobileAsrEvidencePacketError):
            packet._require_committed_component_spec(
                replace(spec, receipt_sha256="0" * 64),
                component_id=component_id,
            )

    real_read = packet.read_regular_bounded

    def tampered_read(*args: object, **kwargs: object) -> SimpleNamespace:
        loaded = real_read(*args, **kwargs)
        return SimpleNamespace(path=loaded.path, data=loaded.data + b" ")

    monkeypatch.setattr(packet, "read_regular_bounded", tampered_read)
    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet._load_locked_primock_source_rows()


def test_overlap_copy_uses_exact_loaded_leaf_map_and_rejects_extras(
    tmp_path: Path,
) -> None:
    root = tmp_path / "synthetic-overlap"
    root.mkdir(mode=0o700)
    spec, bundle = _synthetic_overlap_bundle(root)
    loaded = packet._load_declared_overlap_pcm(
        root,
        bundle.cases,
        expected_inputs=spec.pcm_inputs,
        expected_bytes=spec.pcm_bytes,
        sidecars=frozenset({spec.manifest_file, spec.receipt_file}),
    )
    assert bundle.production_evidence is False
    assert len(loaded) == 9

    _private_file(root / "attacker.f32le", b"\x00\x00\x00\x00")
    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet._load_declared_overlap_pcm(
            root,
            bundle.cases,
            expected_inputs=spec.pcm_inputs,
            expected_bytes=spec.pcm_bytes,
            sidecars=frozenset({spec.manifest_file, spec.receipt_file}),
        )


def test_overlap_copy_rejects_declared_leaf_digest_substitution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "synthetic-overlap"
    root.mkdir(mode=0o700)
    spec, bundle = _synthetic_overlap_bundle(root)
    target = root / "synthetic-0-role_a.f32le"
    target.write_bytes(b"\xff\xff\xff\xff")
    target.chmod(0o600)

    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet._load_declared_overlap_pcm(
            root,
            bundle.cases,
            expected_inputs=spec.pcm_inputs,
            expected_bytes=spec.pcm_bytes,
            sidecars=frozenset({spec.manifest_file, spec.receipt_file}),
        )


def test_packet_tree_snapshot_has_unique_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "packet"
    component_root = root / "components" / "synthetic"
    component_root.mkdir(parents=True, mode=0o700)
    (root / "components").chmod(0o700)
    root.chmod(0o700)
    _private_file(root / "packet-index.json", b"index")
    _private_file(root / "packet-receipt.json", b"receipt")
    _private_file(component_root / "one.f32le", b"\x00\x00\x00\x00")
    spec = packet.ComponentSpec(
        component_id="synthetic",
        directory="synthetic",
        loader="synthetic",
        schema="synthetic",
        manifest_file="manifest.json",
        manifest_sha256="0" * 64,
        receipt_file="receipt.json",
        receipt_sha256="1" * 64,
        licenses=("synthetic-test-only",),
        logical_cases=1,
        pcm_inputs=1,
        pcm_bytes=4,
        metric_domain="synthetic",
    )
    component = packet._ValidatedComponent(
        spec=spec,
        root=component_root,
        artifacts=(
            packet._Artifact(
                name="one.f32le",
                payload=b"\x00\x00\x00\x00",
                sha256=hashlib.sha256(b"\x00\x00\x00\x00").hexdigest(),
            ),
        ),
        inventory=(),
    )

    rows = packet._packet_tree_snapshot(root, (component,))
    names = [name for name, _snapshot in rows]
    assert len(names) == len(set(names))
    assert names.count("components/synthetic") == 1


def test_public_loader_requires_terminal_receipt_and_exact_root_layout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "incomplete-packet"
    (root / "components").mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    _private_file(root / "packet-index.json", b"{}\n")

    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet.load_mobile_asr_evidence_packet(root)


def test_synthetic_nonproduction_e2e_copy_reload_and_hostile_rejection(
    tmp_path: Path,
) -> None:
    contract = _synthetic_packet_contract()
    output = tmp_path / "synthetic-packet"
    result = packet._publish_synthetic_test_packet(contract, output)

    assert result["production_inputs"] is False
    assert str(result["kind"]).startswith("synthetic-")
    loaded = packet._load_synthetic_test_packet(output, contract)
    packet._verify_synthetic_test_packet(loaded, contract)
    assert loaded.production_inputs is False
    assert stat.S_IMODE(output.lstat().st_mode) == 0o700
    for path in output.rglob("*"):
        metadata = path.lstat()
        if path.is_dir():
            assert stat.S_IMODE(metadata.st_mode) == 0o700
        else:
            assert stat.S_IMODE(metadata.st_mode) == 0o600
            assert metadata.st_nlink == 1
    component = output / "components" / "synthetic-component"
    for artifact in contract.artifacts:
        assert (component / artifact.name).read_bytes() == artifact.payload

    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet._publish_synthetic_test_packet(contract, output)
    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet.load_mobile_asr_evidence_packet(output)

    _private_file(component / "attacker-extra", b"extra")
    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet._load_synthetic_test_packet(output, contract)
    (component / "attacker-extra").unlink()

    audio = component / "synthetic-audio.f32le"
    audio.write_bytes(b"\xff\xff\xff\xff")
    audio.chmod(0o600)
    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet._load_synthetic_test_packet(output, contract)

    hardlink_output = tmp_path / "synthetic-hardlink"
    packet._publish_synthetic_test_packet(contract, hardlink_output)
    hardlink_audio = (
        hardlink_output / "components" / "synthetic-component" / "synthetic-audio.f32le"
    )
    retained = tmp_path / "retained-hardlink-target"
    _private_file(retained, b"\x00\x00\x00\x00")
    hardlink_audio.unlink()
    os.link(retained, hardlink_audio)
    with pytest.raises(packet.MobileAsrEvidencePacketError):
        packet._load_synthetic_test_packet(hardlink_output, contract)


def test_synthetic_nonproduction_post_link_recovery_is_truthful(
    tmp_path: Path,
) -> None:
    contract = _synthetic_packet_contract()
    output = tmp_path / "synthetic-recovery"

    def post_link_failure() -> None:
        raise OSError(str(tmp_path / "private-detail"))

    result = packet._publish_synthetic_test_packet(
        contract,
        output,
        post_link_hook=post_link_failure,
    )
    assert result["production_inputs"] is False
    loaded = packet._load_synthetic_test_packet(output, contract)
    packet._verify_synthetic_test_packet(loaded, contract)
