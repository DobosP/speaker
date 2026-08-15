from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
from types import MappingProxyType
import sys
import venv

import pytest

from tools import provision_mobile_zipformer as provision


@dataclass(frozen=True)
class _Fixture:
    private_root: Path
    model_root: Path
    output: Path
    lock: provision.SourceLock
    preflight: provision.MobileZipformerPreflight
    payloads: MappingProxyType


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _private_file(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def _fake_lock(private_root: Path) -> tuple[provision.SourceLock, dict[str, bytes]]:
    expected = provision._EXPECTED_ARTIFACTS
    payloads = {
        expected[0][1]: b"encoder-int8-test-bytes",
        expected[1][1]: b"decoder-fp32-test-bytes",
        expected[2][1]: b"joiner-fp32-test-bytes",
        expected[3][1]: b"<blk> 0\nhello 1\n",
    }
    specs = tuple(
        provision.ArtifactSpec(
            name=name,
            filename=filename,
            precision=precision,
            size_bytes=len(payloads[filename]),
            sha256=hashlib.sha256(payloads[filename]).hexdigest(),
        )
        for name, filename, precision, _size, _sha256 in expected
    )
    raw = b'{"kind":"synthetic-mobile-zipformer-lock-v1"}\n'
    lock_path = _private_file(private_root / "synthetic-source-lock.json", raw)
    config = provision.MobileZipformerConfig(**provision._EXPECTED_MOBILE_CONFIG)
    return (
        provision.SourceLock(
            path=lock_path,
            raw=raw,
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            recipe_sha256="1" * 64,
            repo_id=str(provision._EXPECTED_MOBILE_CONFIG["source_repo_id"]),
            revision=str(provision._EXPECTED_MOBILE_CONFIG["source_revision"]),
            artifacts=specs,
            runtime_distributions=MappingProxyType(
                dict(provision._EXPECTED_RUNTIME_DISTRIBUTIONS)
            ),
            mobile_config=config,
            total_size_bytes=sum(item.size_bytes for item in specs),
        ),
        payloads,
    )


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Fixture:
    tmp_path.chmod(0o700)

    def scoped_git_ancestor(path: Path) -> bool:
        current = path.resolve(strict=True)
        while current != tmp_path.parent:
            if (current / ".git").exists():
                return True
            current = current.parent
        return False

    monkeypatch.setattr(provision, "_has_git_ancestor", scoped_git_ancestor)
    private_root = _private_directory(tmp_path / "private")
    model_root = _private_directory(private_root / "source-model")
    lock, payloads = _fake_lock(private_root)
    for filename, payload in payloads.items():
        _private_file(model_root / filename, payload)
    fake_worker = private_root / "worker.py"
    fake_worker.write_text("raise SystemExit('not executed')\n", encoding="utf-8")
    fake_worker.chmod(0o644)
    monkeypatch.setattr(provision, "_FIXED_WORKER", fake_worker)
    preflight = provision._preflight_with_lock(
        python=Path(sys.executable),
        model_root=model_root,
        lock=lock,
        version_probe=lambda _python, _versions: None,
    )
    return _Fixture(
        private_root=private_root,
        model_root=model_root,
        output=private_root / "provisioned",
        lock=lock,
        preflight=preflight,
        payloads=MappingProxyType(payloads),
    )


def _publish(item: _Fixture) -> provision.MobileZipformerProvision:
    return provision._publish_preflight(
        preflight=item.preflight,
        output_dir=item.output,
    )


def test_production_lock_is_exact_hybrid_tuple() -> None:
    lock = provision._load_source_lock()

    assert lock.raw_sha256 == provision.LOCK_RAW_SHA256
    assert len(lock.raw) == provision.LOCK_SIZE_BYTES
    assert lock.recipe_sha256 == provision.LOCK_RECIPE_SHA256
    assert lock.total_size_bytes == 74_207_237
    assert [
        (
            item.name,
            item.filename,
            item.precision,
            item.size_bytes,
            item.sha256,
        )
        for item in lock.artifacts
    ] == list(provision._EXPECTED_ARTIFACTS)
    assert lock.artifacts[0].precision == "int8"
    assert lock.artifacts[1].precision == "fp32"
    assert lock.artifacts[2].precision == "fp32"
    assert lock.artifacts[3].precision == "text"


def test_production_artifact_set_recipe_matches_runner() -> None:
    lock = provision._load_source_lock()

    assert provision._artifact_set_sha256(lock.artifacts) == (
        "eebca4036773b78642c846acdc966d1bab8f377a8b8ab632f9d3aac7ab38bcc5"
    )


@pytest.mark.parametrize(
    ("artifact_index", "replacement"),
    [
        (
            0,
            {
                "filename": "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                "precision": "fp32",
                "sha256": (
                    "a423883ce5754507fd941755ab0b5bc426a84ac670cbe21cf060e9e2c66dc660"
                ),
            },
        ),
        (
            2,
            {
                "filename": "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                "precision": "int8",
                "sha256": (
                    "d944208d660d67c8d72cd2acaeac971fa5ceb8c80e76c1968148846fedd6e297"
                ),
            },
        ),
    ],
)
def test_lock_rejects_precision_drift(
    tmp_path: Path,
    artifact_index: int,
    replacement: dict[str, str],
) -> None:
    value = json.loads(provision._LOCK_PATH.read_bytes())
    value["artifacts"][artifact_index].update(replacement)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path = tmp_path / "drift.lock.json"
    path.write_bytes(raw)

    with pytest.raises(provision.MobileZipformerProvisionError):
        provision._load_source_lock(
            path,
            expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
            expected_size_bytes=len(raw),
        )


def test_runtime_probe_uses_distribution_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> object:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(provision.subprocess, "run", fake_run)
    provision._verify_runtime_versions(
        Path(sys.executable),
        provision._EXPECTED_RUNTIME_DISTRIBUTIONS,
    )

    args = observed["args"]
    assert isinstance(args, list)
    assert args[:5] == [str(Path(sys.executable)), "-I", "-B", "-S", "-c"]
    script = args[-1]
    assert script == provision._RUNTIME_METADATA_SCRIPT
    assert "importlib.metadata" not in script
    assert "METADATA" in script
    assert "os.O_NOFOLLOW" in script
    assert "sys.flags.no_site" in script
    assert "import sherpa_onnx" not in script
    assert "import numpy" not in script
    assert "sherpa-onnx-core" in script
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["stdin"] is provision.subprocess.DEVNULL
    assert kwargs["stdout"] is provision.subprocess.DEVNULL
    assert kwargs["stderr"] is provision.subprocess.DEVNULL
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"


def test_installed_metadata_probe_matches_exact_runtime() -> None:
    provision._verify_runtime_versions(
        Path(sys.executable),
        provision._EXPECTED_RUNTIME_DISTRIBUTIONS,
    )


def test_runtime_probe_cannot_execute_site_startup_hooks(tmp_path: Path) -> None:
    environment_root = tmp_path / "probe-venv"
    venv.EnvBuilder(with_pip=False).create(environment_root)
    site_packages = (
        environment_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    for name, stem, version in (
        ("numpy", "numpy", "2.4.6"),
        ("sherpa-onnx", "sherpa_onnx", "1.13.3"),
        ("sherpa-onnx-core", "sherpa_onnx_core", "1.13.3"),
    ):
        metadata_root = site_packages / f"{stem}-{version}.dist-info"
        metadata_root.mkdir()
        (metadata_root / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n",
            encoding="utf-8",
        )

    pth_marker = tmp_path / "pth-hook-executed"
    sitecustomize_marker = tmp_path / "sitecustomize-executed"
    (site_packages / "adversarial-startup.pth").write_text(
        f"import pathlib; pathlib.Path({str(pth_marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    (site_packages / "sitecustomize.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(sitecustomize_marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )

    provision._verify_runtime_versions(
        environment_root / "bin" / "python",
        provision._EXPECTED_RUNTIME_DISTRIBUTIONS,
    )

    assert not pth_marker.exists()
    assert not sitecustomize_marker.exists()


def test_commit_signal_numbers_are_exact_plain_ints() -> None:
    expected = {int(signal.SIGHUP), int(signal.SIGINT), int(signal.SIGTERM)}

    observed = provision._commit_signal_numbers()

    assert observed == expected
    assert all(type(number) is int for number in observed)


def test_synthetic_preflight_is_hash_only_and_path_free(
    synthetic: _Fixture,
) -> None:
    result = provision._safe_preflight_result(synthetic.preflight)

    assert result == {
        "artifact_set_sha256": synthetic.preflight.artifact_set_sha256,
        "metadata_only_runtime_verified": True,
        "model_id": provision.MODEL_ID,
        "ok": True,
        "ready_to_publish": True,
        "total_size_bytes": synthetic.lock.total_size_bytes,
    }
    assert str(synthetic.private_root) not in json.dumps(result)
    assert not synthetic.output.exists()


def test_preflight_rejects_fp32_encoder_name(
    synthetic: _Fixture,
) -> None:
    encoder = synthetic.lock.artifacts[0]
    source = synthetic.model_root / encoder.filename
    wrong = synthetic.model_root / encoder.filename.replace(".int8.onnx", ".onnx")
    source.rename(wrong)

    with pytest.raises(provision.MobileZipformerProvisionError):
        provision._preflight_with_lock(
            python=Path(sys.executable),
            model_root=synthetic.model_root,
            lock=synthetic.lock,
            version_probe=lambda _python, _versions: None,
        )


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink", "permissions"])
def test_preflight_rejects_unsafe_model_leaf(
    synthetic: _Fixture,
    unsafe_kind: str,
) -> None:
    spec = synthetic.lock.artifacts[0]
    path = synthetic.model_root / spec.filename
    if unsafe_kind == "symlink":
        retained = synthetic.private_root / "retained-encoder"
        path.rename(retained)
        path.symlink_to(retained)
    elif unsafe_kind == "hardlink":
        os.link(path, synthetic.private_root / "second-encoder-link")
    else:
        path.chmod(0o640)

    with pytest.raises(provision.MobileZipformerProvisionError):
        provision._preflight_with_lock(
            python=Path(sys.executable),
            model_root=synthetic.model_root,
            lock=synthetic.lock,
            version_probe=lambda _python, _versions: None,
        )


def test_preflight_rejects_nonprivate_model_root(synthetic: _Fixture) -> None:
    synthetic.model_root.chmod(0o750)

    with pytest.raises(provision.MobileZipformerProvisionError):
        provision._preflight_with_lock(
            python=Path(sys.executable),
            model_root=synthetic.model_root,
            lock=synthetic.lock,
            version_probe=lambda _python, _versions: None,
        )


def test_preflight_rejects_model_under_git_ancestor(synthetic: _Fixture) -> None:
    (synthetic.private_root / ".git").mkdir()

    with pytest.raises(provision.MobileZipformerProvisionError):
        provision._preflight_with_lock(
            python=Path(sys.executable),
            model_root=synthetic.model_root,
            lock=synthetic.lock,
            version_probe=lambda _python, _versions: None,
        )


def test_publish_creates_exact_private_terminal_tree(synthetic: _Fixture) -> None:
    loaded = _publish(synthetic)

    assert loaded.path == synthetic.output / provision.MANIFEST_FILENAME
    assert loaded.model_id == provision.MODEL_ID
    assert loaded.adapter == provision.ADAPTER
    assert loaded.artifact_set_sha256 == synthetic.preflight.artifact_set_sha256
    assert loaded.total_size_bytes == synthetic.lock.total_size_bytes
    assert {path.name for path in synthetic.output.iterdir()} == {
        provision.LOCK_COPY_FILENAME,
        provision.MANIFEST_FILENAME,
        provision.MODEL_DIRECTORY,
        provision.RECEIPT_FILENAME,
    }
    assert stat_mode(synthetic.output) == 0o700
    assert stat_mode(synthetic.output / provision.MODEL_DIRECTORY) == 0o700
    for path in (
        synthetic.output / provision.LOCK_COPY_FILENAME,
        synthetic.output / provision.MANIFEST_FILENAME,
        synthetic.output / provision.RECEIPT_FILENAME,
        *(item.path for item in loaded.artifacts),
    ):
        assert stat_mode(path) == 0o600
        assert path.stat().st_nlink == 1
    assert (
        provision._load_provision_with_lock(
            synthetic.output,
            lock=synthetic.lock,
        )
        == loaded
    )
    assert (
        provision._load_provision_with_lock(
            loaded.path,
            lock=synthetic.lock,
        )
        == loaded
    )


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_publish_refuses_existing_output_without_mutating_it(
    synthetic: _Fixture,
) -> None:
    synthetic.output.mkdir(mode=0o700)
    sentinel = _private_file(synthetic.output / "sentinel", b"keep")

    with pytest.raises(provision.MobileZipformerProvisionError):
        _publish(synthetic)

    assert sentinel.read_bytes() == b"keep"


def test_publish_rejects_nonprivate_output_parent(synthetic: _Fixture) -> None:
    synthetic.private_root.chmod(0o750)

    with pytest.raises(provision.MobileZipformerProvisionError):
        _publish(synthetic)

    assert not synthetic.output.exists()


def test_publish_rejects_output_under_git_ancestor(synthetic: _Fixture) -> None:
    (synthetic.private_root / ".git").mkdir()

    with pytest.raises(provision.MobileZipformerProvisionError):
        _publish(synthetic)

    assert not synthetic.output.exists()


def test_source_mutation_before_terminal_receipt_fails_closed(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = provision._copy_artifact
    mutated = False

    def mutate_after_copy(
        source: provision._SourceArtifact,
        destination: Path,
    ) -> provision.BoundArtifact:
        nonlocal mutated
        result = original(source, destination)
        if not mutated:
            mutated = True
            payload = bytearray(source.path.read_bytes())
            payload[0] ^= 1
            source.path.write_bytes(payload)
            source.path.chmod(0o600)
        return result

    monkeypatch.setattr(provision, "_copy_artifact", mutate_after_copy)

    with pytest.raises(provision.MobileZipformerProvisionError):
        _publish(synthetic)

    assert synthetic.output.exists()
    assert not (synthetic.output / provision.RECEIPT_FILENAME).exists()


def test_output_mutation_before_terminal_receipt_fails_closed(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = provision._stage_terminal_receipt

    def mutate_after_stage(
        root_descriptor: int,
        raw: bytes,
    ) -> tuple[int, provision._StagedReceipt]:
        result = original(root_descriptor, raw)
        spec = synthetic.lock.artifacts[0]
        path = synthetic.output / provision.MODEL_DIRECTORY / spec.filename
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        path.chmod(0o600)
        return result

    monkeypatch.setattr(provision, "_stage_terminal_receipt", mutate_after_stage)

    with pytest.raises(provision.MobileZipformerProvisionError):
        _publish(synthetic)

    assert not (synthetic.output / provision.RECEIPT_FILENAME).exists()


def test_terminal_receipt_no_clobber_race_fails_closed(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = provision._commit_terminal_receipt

    def race(
        root_descriptor: int,
        descriptor: int,
        staged: provision._StagedReceipt,
        raw: bytes,
        *,
        state: provision._TerminalCommitState,
        commit_guard: object,
    ) -> None:
        _private_file(synthetic.output / provision.RECEIPT_FILENAME, b"attacker\n")
        original(
            root_descriptor,
            descriptor,
            staged,
            raw,
            state=state,
            commit_guard=commit_guard,
        )

    monkeypatch.setattr(provision, "_commit_terminal_receipt", race)

    with pytest.raises(provision.MobileZipformerProvisionError):
        _publish(synthetic)

    assert (synthetic.output / provision.RECEIPT_FILENAME).read_bytes() == b"attacker\n"


def test_post_link_failure_recovers_exact_terminal_provision(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = provision._commit_terminal_receipt

    def fail_after_link(
        root_descriptor: int,
        descriptor: int,
        staged: provision._StagedReceipt,
        raw: bytes,
        *,
        state: provision._TerminalCommitState,
        commit_guard: object,
    ) -> None:
        original(
            root_descriptor,
            descriptor,
            staged,
            raw,
            state=state,
            commit_guard=commit_guard,
        )
        raise OSError("simulated post-link failure")

    monkeypatch.setattr(provision, "_commit_terminal_receipt", fail_after_link)

    loaded = _publish(synthetic)

    assert loaded.path == synthetic.output / provision.MANIFEST_FILENAME
    assert (
        loaded.receipt_digest
        == hashlib.sha256(
            (synthetic.output / provision.RECEIPT_FILENAME).read_bytes()
        ).hexdigest()
    )


def test_loader_requires_terminal_receipt(synthetic: _Fixture) -> None:
    loaded = _publish(synthetic)
    (synthetic.output / provision.RECEIPT_FILENAME).unlink()

    with pytest.raises(provision.MobileZipformerProvisionError):
        provision._load_provision_with_lock(loaded.path, lock=synthetic.lock)


def test_loader_rejects_manifest_mutation(synthetic: _Fixture) -> None:
    loaded = _publish(synthetic)
    with loaded.path.open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(provision.MobileZipformerProvisionError):
        provision._load_provision_with_lock(synthetic.output, lock=synthetic.lock)


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink", "permissions"])
def test_loader_rejects_unsafe_provisioned_artifact(
    synthetic: _Fixture,
    unsafe_kind: str,
) -> None:
    loaded = _publish(synthetic)
    path = loaded.artifacts[0].path
    if unsafe_kind == "symlink":
        retained = synthetic.private_root / "retained-output-encoder"
        path.rename(retained)
        path.symlink_to(retained)
    elif unsafe_kind == "hardlink":
        os.link(path, synthetic.private_root / "second-output-encoder-link")
    else:
        path.chmod(0o640)

    with pytest.raises(provision.MobileZipformerProvisionError):
        provision._load_provision_with_lock(synthetic.output, lock=synthetic.lock)


def test_loader_rejects_changed_worker(synthetic: _Fixture) -> None:
    _publish(synthetic)
    worker = provision._FIXED_WORKER
    original = worker.read_bytes()
    worker.write_bytes(b"x" * len(original))

    with pytest.raises(provision.MobileZipformerProvisionError):
        provision._load_provision_with_lock(synthetic.output, lock=synthetic.lock)


def test_manifest_rejects_mobile_config_drift(synthetic: _Fixture) -> None:
    loaded = _publish(synthetic)
    value = json.loads(loaded.path.read_bytes())
    value["mobile_config"]["decoding_method"] = "modified_beam_search"
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    with pytest.raises(provision.MobileZipformerProvisionError):
        provision._parse_manifest(raw, root=synthetic.output, lock=synthetic.lock)


def test_cli_errors_are_bounded_and_path_free(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "/private/owner/model/encoder.onnx"

    assert provision.main(["preflight", "--model-root", secret]) == 2

    output = capsys.readouterr().out
    assert output == (
        '{"error":"mobile_zipformer_prerequisites_unavailable","ok":false}\n'
    )
    assert secret not in output
    assert len(output) < 128


def test_cli_preflight_success_is_path_free(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        provision,
        "preflight_mobile_zipformer",
        lambda **_kwargs: synthetic.preflight,
    )

    assert (
        provision.main(
            [
                "preflight",
                "--python",
                str(Path(sys.executable)),
                "--model-root",
                str(synthetic.model_root),
            ]
        )
        == 0
    )

    value = json.loads(capsys.readouterr().out)
    assert value["ok"] is True
    assert value["ready_to_publish"] is True
    assert str(synthetic.private_root) not in json.dumps(value)


def test_cli_publish_success_is_path_free(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loaded = _publish(synthetic)
    monkeypatch.setattr(
        provision,
        "provision_mobile_zipformer",
        lambda **_kwargs: loaded,
    )

    assert (
        provision.main(
            [
                "publish",
                "--python",
                str(Path(sys.executable)),
                "--model-root",
                str(synthetic.model_root),
                "--output-dir",
                str(synthetic.private_root / "unused"),
            ]
        )
        == 0
    )

    value = json.loads(capsys.readouterr().out)
    assert value["ok"] is True
    assert value["manifest_sha256"] == loaded.digest
    assert value["receipt_sha256"] == loaded.receipt_digest
    assert str(synthetic.private_root) not in json.dumps(value)
