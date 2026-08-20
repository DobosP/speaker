from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace
import venv

import pytest

from tools import provision_mobile_whisper as provision


@dataclass(frozen=True)
class _Fixture:
    private_root: Path
    runtime_root: Path
    python: Path
    model_root: Path
    output: Path
    lock: provision.SourceLock
    preflight: provision.MobileWhisperPreflight


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _private_file(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _fake_lock(private_root: Path) -> tuple[provision.SourceLock, dict[str, bytes]]:
    official = provision.MOBILE_WHISPER_ARTIFACT_SPECS
    payloads = {
        official[0][1]: b"synthetic-whisper-encoder-int8",
        official[1][1]: b"synthetic-whisper-decoder-int8",
        official[2][1]: b"<|endoftext|> 50256\n",
    }
    artifacts = tuple(
        provision.ArtifactSpec(
            name=name,
            filename=filename,
            precision=precision,
            sha256=hashlib.sha256(payloads[filename]).hexdigest(),
            size_bytes=len(payloads[filename]),
        )
        for name, filename, precision, _sha256, _size_bytes in official
    )
    raw = b'{"kind":"synthetic-mobile-whisper-lock-v1"}\n'
    lock_path = _private_file(private_root / "synthetic-source-lock.json", raw)
    return (
        provision.SourceLock(
            path=lock_path,
            raw=raw,
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            recipe_sha256="1" * 64,
            artifacts=artifacts,
            runtime_distributions=MappingProxyType(
                dict(provision._EXPECTED_RUNTIME_DISTRIBUTIONS)
            ),
            mobile_config=provision.MobileWhisperConfig(),
            source=MappingProxyType(dict(provision._EXPECTED_SOURCE)),
            total_size_bytes=sum(item.size_bytes for item in artifacts),
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
    private_root = _private_directory(tmp_path / "source-authority")
    model_root = _private_directory(private_root / "source-model")
    lock, payloads = _fake_lock(private_root)
    for filename, payload in payloads.items():
        _private_file(model_root / filename, payload)
    worker_root = _private_directory(tmp_path / "worker-authority")
    worker = worker_root / "worker.py"
    worker.write_text("raise SystemExit('not executed')\n", encoding="utf-8")
    worker.chmod(0o644)
    monkeypatch.setattr(provision, "_FIXED_WORKER", worker)
    runtime_root = _private_directory(tmp_path / "runtime-authority")
    binary_root = _private_directory(runtime_root / "bin")
    python = _private_file(binary_root / "python", b"synthetic-python\n")
    library_root = _private_directory(runtime_root / "lib")
    version_root = _private_directory(library_root / "python3.12")
    _private_directory(version_root / "site-packages")
    preflight = provision._preflight_with_lock(
        python=python,
        model_root=model_root,
        lock=lock,
        version_probe=lambda _python, _versions: None,
    )
    output_root = _private_directory(tmp_path / "output-authority")
    return _Fixture(
        private_root=private_root,
        runtime_root=runtime_root,
        python=python,
        model_root=model_root,
        output=output_root / "provisioned",
        lock=lock,
        preflight=preflight,
    )


def _publish(item: _Fixture) -> provision.MobileWhisperProvision:
    return provision._publish_preflight(
        preflight=item.preflight,
        output_dir=item.output,
    )


def test_production_lock_is_exact_and_independently_recomputed() -> None:
    lock = provision._load_source_lock()
    value = json.loads(lock.raw)
    recipe_value = dict(value)
    recipe_value.pop("recipe_sha256")
    canonical = json.dumps(
        recipe_value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    recipe = hashlib.sha256(provision._LOCK_RECIPE_DOMAIN + canonical).hexdigest()

    assert len(lock.raw) == provision.LOCK_SIZE_BYTES == 3_314
    assert hashlib.sha256(lock.raw).hexdigest() == provision.LOCK_RAW_SHA256
    assert lock.recipe_sha256 == provision.LOCK_RECIPE_SHA256 == recipe
    assert (
        provision._artifact_set_sha256(lock.artifacts)
        == provision.MOBILE_WHISPER_ARTIFACT_SET_SHA256
    )
    assert provision.MOBILE_WHISPER_ARTIFACT_SET_SHA256 == (
        "3a3e907d65aced6f76a40afaabee86c12e7d281de3397bbfed33ad90f227af34"
    )
    assert lock.total_size_bytes == 160_626_066
    assert [
        (item.name, item.filename, item.precision, item.sha256, item.size_bytes)
        for item in lock.artifacts
    ] == list(provision.MOBILE_WHISPER_ARTIFACT_SPECS)


def test_lock_preserves_conversion_uncertainty_and_upstream_evidence_scope() -> None:
    source = dict(provision._load_source_lock().source)

    assert source["repo_id"] == "csukuangfj/sherpa-onnx-whisper-base.en"
    assert source["revision"] == provision.SOURCE_REVISION
    assert source["conversion_license_artifact_present"] is False
    assert source["artifact_license_status"] == "unverified"
    assert source["conversion_origin_revision_status"] == "unverified"
    assert source["exact_conversion_recipe_status"] == "unavailable"
    assert source["upstream_project"] == "openai/whisper"
    assert source["upstream_evidence_revision"] == (
        "5f86d1d86363843179951550570367b37c5d6f78"
    )
    assert source["upstream_evidence_scope"] == (
        "license-and-base.en-checkpoint-map-only"
    )


def test_mobile_config_has_only_enforced_fields() -> None:
    value = provision.MobileWhisperConfig().as_dict()
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    assert "max_active_paths" not in value
    assert value["execution_mode"] == "complete-endpoint-epoch-final-only"
    assert value["tail_paddings"] == -1
    assert value["maximum_tail_padding_samples"] == 0
    assert hashlib.sha256(canonical).hexdigest() == (
        "f3555a3ffaaa2569bb12cb71c39f31883d27d759b35ec85024ca057fbc5e34f1"
    )


def test_runtime_probe_is_metadata_only_and_blocks_native_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> object:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(provision.subprocess, "run", fake_run)
    provision._verify_runtime_versions(
        Path(sys.executable),
        provision._EXPECTED_RUNTIME_DISTRIBUTIONS,
    )

    args = observed["args"]
    assert isinstance(args, list)
    assert args[:5] == [str(Path(sys.executable)), "-I", "-B", "-S", "-c"]
    script = args[-1]
    assert isinstance(script, str)
    assert "import sherpa_onnx" not in script
    assert "import numpy" not in script
    for forbidden in (
        "sherpa_onnx",
        "sherpa_onnx_core",
        "_sherpa_onnx",
        "sherpa_onnx.lib._sherpa_onnx",
    ):
        assert repr(forbidden) in script or f'"{forbidden}"' in script
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"


def test_runtime_probe_rejects_preloaded_native_module(tmp_path: Path) -> None:
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
    injected = provision._RUNTIME_METADATA_SCRIPT.replace(
        "if not sys.flags.isolated or not sys.flags.no_site or blocked():",
        "sys.modules['sherpa_onnx_core.adversarial'] = object()\n"
        "if not sys.flags.isolated or not sys.flags.no_site or blocked():",
        1,
    )

    completed = subprocess.run(
        [str(environment_root / "bin/python"), "-I", "-B", "-S", "-c", injected],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    assert completed.returncode == 2


def test_commit_signal_numbers_are_plain_ints() -> None:
    observed = provision._commit_signal_numbers()

    assert observed == {int(signal.SIGHUP), int(signal.SIGINT), int(signal.SIGTERM)}
    assert all(type(number) is int for number in observed)


def test_publish_creates_private_terminal_tree_and_strictly_reloads(
    synthetic: _Fixture,
) -> None:
    loaded = _publish(synthetic)

    assert loaded.path == synthetic.output / provision.MANIFEST_FILENAME
    assert loaded.model_id == provision.MODEL_ID
    assert loaded.adapter == provision.ADAPTER
    assert loaded.mobile_config == provision.MobileWhisperConfig()
    assert {path.name for path in synthetic.output.iterdir()} == {
        provision.LOCK_COPY_FILENAME,
        provision.MANIFEST_FILENAME,
        provision.MODEL_DIRECTORY,
        provision.RECEIPT_FILENAME,
    }
    assert _mode(synthetic.output) == 0o700
    assert _mode(synthetic.output / provision.MODEL_DIRECTORY) == 0o700
    for path in (
        synthetic.output / provision.LOCK_COPY_FILENAME,
        synthetic.output / provision.MANIFEST_FILENAME,
        synthetic.output / provision.RECEIPT_FILENAME,
        *(item.path for item in loaded.artifacts),
    ):
        assert _mode(path) == 0o600
        assert path.stat().st_nlink == 1
    assert provision._load_provision_with_lock(
        synthetic.output,
        lock=synthetic.lock,
    ) == loaded


def test_output_creation_rechecks_parent_identity_before_dirfd_mkdir(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = provision._private_directory
    calls = 0

    def stale_identity(path: Path | str, **kwargs: object):
        nonlocal calls
        stable, identity = original(path, **kwargs)
        calls += 1
        if calls == 1:
            identity = (*identity[:-1], identity[-1] + 1)
        return stable, identity

    monkeypatch.setattr(provision, "_private_directory", stale_identity)

    with pytest.raises(provision.MobileWhisperProvisionError):
        provision._new_private_output(synthetic.output, forbidden=())

    assert not synthetic.output.exists()


def test_publish_refuses_existing_output_without_clobbering(
    synthetic: _Fixture,
) -> None:
    synthetic.output.mkdir(mode=0o700)
    sentinel = _private_file(synthetic.output / "sentinel", b"keep")

    with pytest.raises(provision.MobileWhisperProvisionError):
        _publish(synthetic)

    assert sentinel.read_bytes() == b"keep"


def test_publish_rejects_output_inside_runtime_before_creation(
    synthetic: _Fixture,
) -> None:
    site_packages = (
        synthetic.runtime_root / "lib" / "python3.12" / "site-packages"
    )
    output = site_packages / "provisioned"

    with pytest.raises(provision.MobileWhisperProvisionError):
        provision._publish_preflight(
            preflight=synthetic.preflight,
            output_dir=output,
        )

    assert not output.exists()


def test_preflight_rejects_source_model_inside_runtime(
    synthetic: _Fixture,
) -> None:
    site_packages = (
        synthetic.runtime_root / "lib" / "python3.12" / "site-packages"
    )
    nested = _private_directory(site_packages / "source-model")
    for filename in (item.filename for item in synthetic.lock.artifacts):
        _private_file(nested / filename, (synthetic.model_root / filename).read_bytes())

    with pytest.raises(provision.MobileWhisperProvisionError):
        provision._preflight_with_lock(
            python=synthetic.python,
            model_root=nested,
            lock=synthetic.lock,
            version_probe=lambda _python, _versions: None,
        )


def test_preflight_rejects_runtime_inside_source_model_membership(
    synthetic: _Fixture,
) -> None:
    nested_model = _private_directory(synthetic.private_root / "nested-model")
    for filename in (item.filename for item in synthetic.lock.artifacts):
        _private_file(
            nested_model / filename,
            (synthetic.model_root / filename).read_bytes(),
        )
    nested_runtime = _private_directory(nested_model / "runtime")
    nested_binary = _private_directory(nested_runtime / "bin")
    nested_python = _private_file(nested_binary / "python", b"python\n")

    with pytest.raises(provision.MobileWhisperProvisionError):
        provision._preflight_with_lock(
            python=nested_python,
            model_root=nested_model,
            lock=synthetic.lock,
            version_probe=lambda _python, _versions: None,
        )


def test_nonterminal_directories_are_fsynced_before_receipt_stage(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fsync = provision._fsync_directory
    original_stage = provision._stage_terminal_receipt
    ordered: list[Path] = []

    def record_fsync(path: Path) -> None:
        ordered.append(path)
        original_fsync(path)

    def stage(root_descriptor: int, raw: bytes):
        assert ordered[-2:] == [
            synthetic.output / provision.MODEL_DIRECTORY,
            synthetic.output,
        ]
        return original_stage(root_descriptor, raw)

    monkeypatch.setattr(provision, "_fsync_directory", record_fsync)
    monkeypatch.setattr(provision, "_stage_terminal_receipt", stage)

    _publish(synthetic)

    assert ordered[-1] == synthetic.output.parent


def test_terminal_receipt_no_clobber_race_fails_closed(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = provision._commit_terminal_receipt

    def race(
        root: Path,
        root_identity: tuple[int, ...],
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
            root,
            root_identity,
            root_descriptor,
            descriptor,
            staged,
            raw,
            state=state,
            commit_guard=commit_guard,
        )

    monkeypatch.setattr(provision, "_commit_terminal_receipt", race)

    with pytest.raises(provision.MobileWhisperProvisionError):
        _publish(synthetic)

    assert (synthetic.output / provision.RECEIPT_FILENAME).read_bytes() == b"attacker\n"


def test_post_link_failure_recovers_only_strict_terminal_provision(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = provision._commit_terminal_receipt

    def fail_after_link(
        root: Path,
        root_identity: tuple[int, ...],
        root_descriptor: int,
        descriptor: int,
        staged: provision._StagedReceipt,
        raw: bytes,
        *,
        state: provision._TerminalCommitState,
        commit_guard: object,
    ) -> None:
        original(
            root,
            root_identity,
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

    assert loaded.receipt_digest == hashlib.sha256(
        (synthetic.output / provision.RECEIPT_FILENAME).read_bytes()
    ).hexdigest()


def test_terminal_root_fsync_failure_never_returns_success(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = provision.os.fsync
    failed = False

    def fail_first_root_fsync_after_link(descriptor: int) -> None:
        nonlocal failed
        try:
            provision.os.stat(
                provision.RECEIPT_FILENAME,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except OSError:
            pass
        else:
            if not failed:
                failed = True
                raise OSError("simulated terminal root fsync failure")
        original(descriptor)

    monkeypatch.setattr(provision.os, "fsync", fail_first_root_fsync_after_link)

    with pytest.raises(provision.MobileWhisperProvisionError):
        _publish(synthetic)

    assert failed is True
    assert (synthetic.output / provision.RECEIPT_FILENAME).exists()


def test_terminal_commit_rejects_lexical_root_replacement(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = provision._verify_staged_receipt
    calls = 0
    displaced = synthetic.output.with_name("displaced-provision")

    def replace_after_commit_guard(
        descriptor: int,
        staged: provision._StagedReceipt,
        expected: bytes,
    ) -> None:
        nonlocal calls
        original(descriptor, staged, expected)
        calls += 1
        if calls == 2:
            synthetic.output.rename(displaced)
            shutil.copytree(displaced, synthetic.output)

    monkeypatch.setattr(
        provision,
        "_verify_staged_receipt",
        replace_after_commit_guard,
    )

    with pytest.raises(provision.MobileWhisperProvisionError):
        _publish(synthetic)

    assert calls == 2
    assert not (synthetic.output / provision.RECEIPT_FILENAME).exists()
    assert not (displaced / provision.RECEIPT_FILENAME).exists()


@pytest.mark.parametrize("row_name", ["lock", "manifest"])
def test_receipt_parser_rejects_forged_bound_rows(
    synthetic: _Fixture,
    row_name: str,
) -> None:
    loaded = _publish(synthetic)
    receipt = json.loads(
        (synthetic.output / provision.RECEIPT_FILENAME).read_bytes()
    )
    receipt[row_name]["sha256"] = "f" * 64
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    with pytest.raises(provision.MobileWhisperProvisionError):
        provision._parse_receipt(
            raw,
            lock=synthetic.lock,
            manifest_raw=loaded.path.read_bytes(),
            artifact_set_sha256=loaded.artifact_set_sha256,
        )


def test_loader_rehashes_artifacts_after_metadata_rereads(
    synthetic: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish(synthetic)
    original = provision._bind_private_artifact
    first_spec = synthetic.lock.artifacts[0]
    calls = 0

    def mutate_between_passes(path: Path, spec: provision.ArtifactSpec):
        nonlocal calls
        result = original(path, spec)
        if spec == first_spec:
            calls += 1
            if calls == 3:
                payload = bytearray(path.read_bytes())
                payload[0] ^= 1
                path.write_bytes(payload)
                path.chmod(0o600)
        return result

    monkeypatch.setattr(provision, "_bind_private_artifact", mutate_between_passes)

    with pytest.raises(provision.MobileWhisperProvisionError):
        provision._load_provision_with_lock(synthetic.output, lock=synthetic.lock)

    # The fourth pass starts and rejects the changed hash before returning.
    assert calls == 3


def test_loader_rejects_terminal_provision_nested_under_declared_runtime(
    synthetic: _Fixture,
) -> None:
    _publish(synthetic)
    runtime_root = synthetic.output.parent
    binary_root = _private_directory(runtime_root / "bin")
    python = _private_file(binary_root / "python", synthetic.python.read_bytes())
    manifest_path = synthetic.output / provision.MANIFEST_FILENAME
    value = json.loads(manifest_path.read_bytes())
    python_row = {
        "path": str(python),
        "sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        "size_bytes": python.stat().st_size,
    }
    value["python"] = python_row
    value["runtime"]["python"] = python_row
    manifest_raw = provision._canonical_json(value) + b"\n"
    manifest_path.write_bytes(manifest_raw)
    manifest_path.chmod(0o600)
    receipt_path = synthetic.output / provision.RECEIPT_FILENAME
    receipt_path.write_bytes(
        provision._canonical_json(
            provision._receipt_payload(
                synthetic.preflight,
                manifest_raw=manifest_raw,
            )
        )
        + b"\n"
    )
    receipt_path.chmod(0o600)

    with pytest.raises(provision.MobileWhisperProvisionError):
        provision._load_provision_with_lock(
            synthetic.output,
            lock=synthetic.lock,
        )


def test_source_mutation_prevents_terminal_receipt(
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
            payload[-1] ^= 1
            source.path.write_bytes(payload)
            source.path.chmod(0o600)
        return result

    monkeypatch.setattr(provision, "_copy_artifact", mutate_after_copy)

    with pytest.raises(provision.MobileWhisperProvisionError):
        _publish(synthetic)

    assert not (synthetic.output / provision.RECEIPT_FILENAME).exists()


def test_broken_stdout_is_one_write_and_returns_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenBuffer:
        def __init__(self) -> None:
            self.writes = 0

        def write(self, _payload: bytes) -> int:
            self.writes += 1
            raise BrokenPipeError()

        def flush(self) -> None:
            raise AssertionError("flush must not follow failed write")

    buffer = _BrokenBuffer()
    monkeypatch.setattr(provision.sys, "stdout", SimpleNamespace(buffer=buffer))

    assert provision.main(["--help"]) == 2
    assert buffer.writes == 1


def test_cli_error_is_bounded_and_path_free(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "/private/owner/model/encoder.onnx"

    assert provision.main(["preflight", "--model-root", secret]) == 2

    output = capsys.readouterr().out
    assert output == (
        '{"error":"mobile_whisper_prerequisites_unavailable","ok":false}\n'
    )
    assert secret not in output
    assert len(output) < 128
