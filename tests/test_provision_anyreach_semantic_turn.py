"""No-download provisioning tests for the inert Anyreach artifact manifest."""

from __future__ import annotations

import builtins
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from tools import provision_anyreach_semantic_turn as provision
from tools.semantic_interruption import anyreach as contract


@dataclass(frozen=True)
class _Inputs:
    model_root: Path
    benchmark: Path
    output_parent: Path
    lock: contract.AnyreachSourceLock
    model_payloads: tuple[bytes, ...]
    benchmark_payload: bytes


def _private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _private_file(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Inputs:
    production = contract.load_source_lock()
    model_payloads = (
        b"fake-q8-onnx",
        b"fake-tokenizer-json",
        b'{"fake":"config"}',
        b'{"fake":"tokenizer-config"}',
    )
    model_artifacts = tuple(
        replace(
            item,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for item, payload in zip(
            production.model_artifacts,
            model_payloads,
            strict=True,
        )
    )
    benchmark_payload = b"PAR1fake-anyreach-benchmarkPAR1"
    benchmark_artifact = replace(
        production.benchmark_artifact,
        size_bytes=len(benchmark_payload),
        sha256=hashlib.sha256(benchmark_payload).hexdigest(),
    )
    fake = replace(
        production,
        model_artifacts=model_artifacts,
        benchmark_artifact=benchmark_artifact,
    )

    original_path = production.path

    def load_fake(path: Path | str = contract.DEFAULT_LOCK):
        selected = Path(path).resolve(strict=True)
        return replace(
            fake, path=selected if path != contract.DEFAULT_LOCK else original_path
        )

    monkeypatch.setattr(contract, "load_source_lock", load_fake)

    model_root = _private_dir(tmp_path / "model")
    onnx = _private_dir(model_root / "onnx")
    for item, payload in zip(model_artifacts, model_payloads, strict=True):
        _private_file(onnx / item.filename, payload)

    source_root = _private_dir(tmp_path / "source")
    benchmark = _private_file(
        source_root / contract.BENCHMARK_COPY_FILENAME,
        benchmark_payload,
    )
    output_parent = _private_dir(tmp_path / "outputs")
    return _Inputs(
        model_root=model_root,
        benchmark=benchmark,
        output_parent=output_parent,
        lock=fake,
        model_payloads=model_payloads,
        benchmark_payload=benchmark_payload,
    )


def _provision(inputs: _Inputs, *, name: str = "provision"):
    return provision.provision_candidate(
        model_root=inputs.model_root,
        benchmark_parquet=inputs.benchmark,
        output_dir=inputs.output_parent / name,
        accepted_terms=(contract.ACCEPTANCE_ID,),
    )


def _canonical_manifest(value: dict[str, object]) -> bytes:
    unsigned = dict(value)
    unsigned.pop("manifest_self_sha256", None)
    value = dict(unsigned)
    value["manifest_self_sha256"] = contract._canonical_sha256(unsigned)
    return contract._canonical_json(value) + b"\n"


def test_provision_binds_exact_private_layout_without_candidate_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    imported: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "huggingface_hub",
            "onnx",
            "onnxruntime",
            "pyarrow",
            "tokenizers",
            "transformers",
        }:
            imported.append(name)
            raise AssertionError("candidate/runtime import attempted")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    manifest = _provision(inputs)

    assert imported == []
    assert manifest.candidate_id == contract.CANDIDATE_ID
    assert manifest.selected_rows == 24
    assert tuple(manifest.artifact_by_name) == (
        "model-q8",
        "model-tokenizer",
        "model-config",
        "model-tokenizer-config",
        "benchmark-parquet",
    )
    assert {item.path.parent for item in manifest.artifacts[:4]} == {
        inputs.model_root / "onnx"
    }
    assert [item.role for item in manifest.artifacts[:4]] == [
        "execution",
        "execution",
        "metadata-witness",
        "metadata-witness",
    ]
    assert all(
        manifest.evidence_scope[field] is False
        for field in (
            "model_executed",
            "tokenizer_executed",
            "onnx_contract_verified",
            "parquet_decoded",
            "production_evidence",
        )
    )


def test_provision_publishes_only_terminal_private_output_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    manifest = _provision(inputs)
    output = manifest.path.parent

    assert set(path.name for path in output.iterdir()) == {
        contract.LOCK_COPY_FILENAME,
        contract.BENCHMARK_COPY_FILENAME,
        contract.MANIFEST_FILENAME,
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in output.iterdir():
        metadata = path.lstat()
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1
    assert (output / contract.BENCHMARK_COPY_FILENAME).read_bytes() == (
        inputs.benchmark_payload
    )
    assert (output / contract.LOCK_COPY_FILENAME).read_bytes() == (
        contract.DEFAULT_LOCK.read_bytes()
    )


def test_manifest_is_canonical_self_digested_and_reopens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    manifest = _provision(inputs)
    raw = manifest.path.read_bytes()
    value = json.loads(raw)
    declared = value.pop("manifest_self_sha256")

    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert contract._canonical_sha256(value) == declared
    value["manifest_self_sha256"] = declared
    assert raw == contract._canonical_json(value) + b"\n"
    assert contract.load_artifact_manifest(manifest.path) == manifest


def test_manifest_serializer_and_loader_agree_for_unicode_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    manifest = _provision(inputs, name="private-π")

    assert "private-π".encode() in manifest.path.read_bytes()
    assert contract.load_artifact_manifest(manifest.path) == manifest


def test_safe_result_is_aggregate_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    result = provision._safe_result(_provision(inputs))
    serialized = json.dumps(result)

    assert result == {
        "ok": True,
        "kind": contract.CANDIDATE_ID,
        "manifest_sha256": result["manifest_sha256"],
        "artifact_set_sha256": result["artifact_set_sha256"],
        "benchmark_rows": 24,
        "model_executed": False,
        "parquet_decoded": False,
    }
    assert str(tmp_path) not in serialized
    assert "manual_sli" not in serialized
    assert "manual_cs" not in serialized


@pytest.mark.parametrize(
    "accepted", [(), ("Apache-2.0",), (contract.ACCEPTANCE_ID,) * 2]
)
def test_provision_requires_one_exact_metadata_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted: tuple[str, ...],
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    output = inputs.output_parent / "rejected"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            model_root=inputs.model_root,
            benchmark_parquet=inputs.benchmark,
            output_dir=output,
            accepted_terms=accepted,
        )

    assert not output.exists()


def test_provision_rejects_non_tuple_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            model_root=inputs.model_root,
            benchmark_parquet=inputs.benchmark,
            output_dir=inputs.output_parent / "rejected",
            accepted_terms=[contract.ACCEPTANCE_ID],  # type: ignore[arg-type]
        )


def test_flat_or_renamed_model_layout_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    flat = _private_dir(tmp_path / "flat-model")
    for item, payload in zip(
        inputs.lock.model_artifacts,
        inputs.model_payloads,
        strict=True,
    ):
        _private_file(flat / item.filename, payload)

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            model_root=flat,
            benchmark_parquet=inputs.benchmark,
            output_dir=inputs.output_parent / "flat-rejected",
            accepted_terms=(contract.ACCEPTANCE_ID,),
        )


@pytest.mark.parametrize("location", ["root", "onnx"])
def test_model_layout_rejects_every_extra_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    parent = inputs.model_root if location == "root" else inputs.model_root / "onnx"
    _private_file(parent / "extra.json", b"extra")

    with pytest.raises(provision.ProvisionError):
        _provision(inputs)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "mode", "content"])
def test_model_artifact_must_be_private_single_link_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    target = inputs.model_root / inputs.lock.model_artifacts[0].upstream_path
    if kind == "symlink":
        original = inputs.output_parent / "model-copy"
        _private_file(original, inputs.model_payloads[0])
        target.unlink()
        target.symlink_to(original)
    elif kind == "hardlink":
        os.link(target, inputs.output_parent / "model-hardlink")
    elif kind == "mode":
        target.chmod(0o640)
    else:
        target.write_bytes(b"X" + inputs.model_payloads[0][1:])
        target.chmod(0o600)

    with pytest.raises(provision.ProvisionError):
        _provision(inputs)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "mode", "content"])
def test_benchmark_source_must_be_private_single_link_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    if kind == "symlink":
        original = _private_file(
            inputs.output_parent / "benchmark-copy",
            inputs.benchmark_payload,
        )
        inputs.benchmark.unlink()
        inputs.benchmark.symlink_to(original)
    elif kind == "hardlink":
        os.link(inputs.benchmark, inputs.output_parent / "benchmark-hardlink")
    elif kind == "mode":
        inputs.benchmark.chmod(0o644)
    else:
        inputs.benchmark.write_bytes(b"X" + inputs.benchmark_payload[1:])
        inputs.benchmark.chmod(0o600)

    with pytest.raises(provision.ProvisionError):
        _provision(inputs)


def test_output_cannot_overlap_benchmark_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            model_root=inputs.model_root,
            benchmark_parquet=inputs.benchmark,
            output_dir=inputs.benchmark.parent / "nested-output",
            accepted_terms=(contract.ACCEPTANCE_ID,),
        )


def test_output_cannot_be_inside_git_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    git_root = _private_dir(tmp_path / "git-root")
    _private_file(git_root / ".git", b"gitdir: elsewhere\n")

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            model_root=inputs.model_root,
            benchmark_parquet=inputs.benchmark,
            output_dir=git_root / "output",
            accepted_terms=(contract.ACCEPTANCE_ID,),
        )


def test_loader_independently_rejects_retained_output_inside_git_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    manifest = _provision(inputs)
    _private_file(tmp_path / ".git", b"gitdir: elsewhere\n")

    with pytest.raises(contract.AnyreachContractError):
        contract.load_artifact_manifest(manifest.path)


def test_existing_output_is_no_overwrite_and_first_manifest_survives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    first = _provision(inputs)
    before = first.path.read_bytes()

    with pytest.raises(provision.ProvisionError):
        _provision(inputs)

    assert first.path.read_bytes() == before


def test_manifest_rejects_semantically_equal_noncanonical_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    manifest = _provision(inputs)
    value = json.loads(manifest.path.read_bytes())
    manifest.path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    manifest.path.chmod(0o600)

    with pytest.raises(contract.AnyreachContractError):
        contract.load_artifact_manifest(manifest.path)


def test_manifest_rejects_recomputed_unknown_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    manifest = _provision(inputs)
    value = json.loads(manifest.path.read_bytes())
    value["unexpected"] = False
    manifest.path.write_bytes(_canonical_manifest(value))
    manifest.path.chmod(0o600)

    with pytest.raises(contract.AnyreachContractError):
        contract.load_artifact_manifest(manifest.path)


def test_manifest_rejects_flat_model_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    manifest = _provision(inputs)
    value = json.loads(manifest.path.read_bytes())
    value["artifacts"][0]["path"] = str(
        inputs.model_root / inputs.lock.model_artifacts[0].filename
    )
    manifest.path.write_bytes(_canonical_manifest(value))
    manifest.path.chmod(0o600)

    with pytest.raises(contract.AnyreachContractError):
        contract.load_artifact_manifest(manifest.path)


def test_close_time_rehash_rejects_retained_benchmark_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    manifest = _provision(inputs)
    retained = manifest.path.parent / contract.BENCHMARK_COPY_FILENAME
    retained.write_bytes(b"X" + inputs.benchmark_payload[1:])
    retained.chmod(0o600)

    with pytest.raises(contract.AnyreachContractError):
        contract.load_artifact_manifest(manifest.path)


def test_close_time_rebind_rejects_late_model_leaf_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    manifest = _provision(inputs)
    model_leaf = manifest.artifacts[0].path
    original_verify = contract._verify_private_leaf_identities
    mutated = False

    def mutate_after_output_rebind(path: Path, **kwargs) -> None:
        nonlocal mutated
        original_verify(path, **kwargs)
        if path == manifest.path.parent and not mutated:
            model_leaf.write_bytes(b"X" + inputs.model_payloads[0][1:])
            model_leaf.chmod(0o600)
            mutated = True

    monkeypatch.setattr(
        contract,
        "_verify_private_leaf_identities",
        mutate_after_output_rebind,
    )

    with pytest.raises(contract.AnyreachContractError):
        contract.load_artifact_manifest(manifest.path)

    assert mutated is True


def test_loader_rejects_extra_output_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    manifest = _provision(inputs)
    _private_file(manifest.path.parent / "extra", b"extra")

    with pytest.raises(contract.AnyreachContractError):
        contract.load_artifact_manifest(manifest.path)


def test_manifest_is_published_last_and_partial_failure_never_publishes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)

    def fail_link(*_args, **_kwargs) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(provision.os, "link", fail_link)
    output = inputs.output_parent / "partial"
    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            model_root=inputs.model_root,
            benchmark_parquet=inputs.benchmark,
            output_dir=output,
            accepted_terms=(contract.ACCEPTANCE_ID,),
        )

    assert output.is_dir()
    assert not (output / contract.MANIFEST_FILENAME).exists()
    assert not (output / provision._PENDING_MANIFEST_FILENAME).exists()


def test_post_link_interrupt_removes_the_identity_bound_terminal_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    original = contract._read_private_regular

    def interrupt_manifest_reopen(path: Path, **kwargs):
        if path.name == contract.MANIFEST_FILENAME:
            raise KeyboardInterrupt("injected after-link interrupt")
        return original(path, **kwargs)

    monkeypatch.setattr(contract, "_read_private_regular", interrupt_manifest_reopen)
    output = inputs.output_parent / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        provision.provision_candidate(
            model_root=inputs.model_root,
            benchmark_parquet=inputs.benchmark,
            output_dir=output,
            accepted_terms=(contract.ACCEPTANCE_ID,),
        )

    assert output.is_dir()
    assert not (output / contract.MANIFEST_FILENAME).exists()
    assert not (output / provision._PENDING_MANIFEST_FILENAME).exists()


def test_failed_full_reopen_retracts_only_the_published_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    model_leaf = inputs.model_root / inputs.lock.model_artifacts[0].upstream_path
    original_load = contract.load_artifact_manifest

    def mutate_model_after_publication(path: Path):
        model_leaf.write_bytes(b"X" + inputs.model_payloads[0][1:])
        model_leaf.chmod(0o600)
        return original_load(path)

    monkeypatch.setattr(
        contract,
        "load_artifact_manifest",
        mutate_model_after_publication,
    )
    output = inputs.output_parent / "failed-full-reopen"
    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            model_root=inputs.model_root,
            benchmark_parquet=inputs.benchmark,
            output_dir=output,
            accepted_terms=(contract.ACCEPTANCE_ID,),
        )

    assert set(path.name for path in output.iterdir()) == {
        contract.LOCK_COPY_FILENAME,
        contract.BENCHMARK_COPY_FILENAME,
    }
    assert (output / contract.LOCK_COPY_FILENAME).read_bytes() == (
        contract.DEFAULT_LOCK.read_bytes()
    )
    assert (output / contract.BENCHMARK_COPY_FILENAME).read_bytes() == (
        inputs.benchmark_payload
    )
    assert model_leaf.read_bytes().startswith(b"X")


def test_failed_output_reopen_keeps_unrelated_leaf_while_retracting_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    original_load = contract.load_artifact_manifest
    unrelated_name = "unrelated-private-leaf"

    def add_leaf_after_publication(path: Path):
        _private_file(path.parent / unrelated_name, b"do-not-remove")
        return original_load(path)

    monkeypatch.setattr(contract, "load_artifact_manifest", add_leaf_after_publication)
    output = inputs.output_parent / "failed-output-reopen"
    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            model_root=inputs.model_root,
            benchmark_parquet=inputs.benchmark,
            output_dir=output,
            accepted_terms=(contract.ACCEPTANCE_ID,),
        )

    assert set(path.name for path in output.iterdir()) == {
        contract.LOCK_COPY_FILENAME,
        contract.BENCHMARK_COPY_FILENAME,
        unrelated_name,
    }
    assert (output / unrelated_name).read_bytes() == b"do-not-remove"


def test_cli_failure_is_detail_free_even_for_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private = tmp_path / "private-value"

    def fail(**_kwargs):
        raise KeyboardInterrupt(str(private))

    monkeypatch.setattr(provision, "provision_candidate", fail)
    code = provision.main(
        [
            "--model-root",
            str(tmp_path / "model"),
            "--benchmark-parquet",
            str(tmp_path / "benchmark"),
            "--output-dir",
            str(tmp_path / "output"),
            "--accept-license",
            contract.ACCEPTANCE_ID,
        ]
    )

    captured = capsys.readouterr().out
    assert code == 2
    assert json.loads(captured) == {
        "error": "anyreach_semantic_turn_prerequisites_unavailable",
        "ok": False,
    }
    assert str(private) not in captured


def test_cli_help_promises_no_install_execution_or_decode(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        provision.main(["--help"])

    assert raised.value.code == 0
    text = " ".join(capsys.readouterr().out.casefold().split())
    compact_text = "".join(text.split())
    for phrase in (
        "does not install",
        "download",
        "import candidate/runtime",
        "execute or load",
        "decode parquet",
    ):
        assert phrase in text
    assert contract.ACCEPTANCE_ID.casefold() in compact_text
