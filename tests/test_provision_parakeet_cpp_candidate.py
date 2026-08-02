from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest

from tools import provision_parakeet_cpp_candidate as provision
from tools.streaming_stt import manifest as manifest_module
from tools.streaming_stt.bounded_io import FileDigest
from tools.streaming_stt.manifest import (
    PARAKEET_CPP_ADAPTER,
    PARAKEET_CPP_ARTIFACT_NAMES,
    PARAKEET_CPP_MODEL_RECEIPT,
    ParakeetCppConfig,
)


_LIBPARAKEET_NEEDED = (
    "libstdc++.so.6",
    "libm.so.6",
    "libgcc_s.so.1",
    "libc.so.6",
    "ld-linux-x86-64.so.2",
)
_BRIDGE_NEEDED = (
    "libparakeet.so",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "libc.so.6",
    "ld-linux-x86-64.so.2",
)


def _put_integer(data: bytearray, offset: int, width: int, value: int) -> None:
    data[offset : offset + width] = value.to_bytes(width, "little")


def _elf_bytes(
    dependencies: tuple[str, ...],
    *,
    runpath: str | None = None,
    hardened: bool = False,
) -> bytes:
    dynamic_dependencies = dependencies
    string_table = bytearray(b"\x00")
    string_offsets = []
    for name in dynamic_dependencies:
        string_offsets.append(len(string_table))
        string_table.extend(name.encode("ascii") + b"\x00")
    runpath_offset = None
    if runpath is not None:
        runpath_offset = len(string_table)
        string_table.extend(runpath.encode("ascii") + b"\x00")

    program_offset = 64
    program_size = 56
    program_count = 4 if hardened else 2
    dynamic_offset = program_offset + (program_size * program_count)
    dynamic_count = len(string_offsets) + 3
    if runpath_offset is not None:
        dynamic_count += 1
    if hardened:
        dynamic_count += 2
    dynamic_size = dynamic_count * 16
    string_offset = dynamic_offset + dynamic_size
    total_size = string_offset + len(string_table)
    virtual_base = 0x400000
    data = bytearray(total_size)
    data[:7] = b"\x7fELF\x02\x01\x01"
    _put_integer(data, 16, 2, 3)
    _put_integer(data, 18, 2, 62)
    _put_integer(data, 20, 4, 1)
    _put_integer(data, 32, 8, program_offset)
    _put_integer(data, 52, 2, 64)
    _put_integer(data, 54, 2, program_size)
    _put_integer(data, 56, 2, program_count)

    load = program_offset
    _put_integer(data, load, 4, 1)
    _put_integer(data, load + 8, 8, 0)
    _put_integer(data, load + 16, 8, virtual_base)
    _put_integer(data, load + 32, 8, total_size)
    _put_integer(data, load + 40, 8, total_size)

    dynamic = program_offset + program_size
    _put_integer(data, dynamic, 4, 2)
    _put_integer(data, dynamic + 8, 8, dynamic_offset)
    _put_integer(data, dynamic + 16, 8, virtual_base + dynamic_offset)
    _put_integer(data, dynamic + 32, 8, dynamic_size)
    _put_integer(data, dynamic + 40, 8, dynamic_size)

    if hardened:
        stack = program_offset + (program_size * 2)
        _put_integer(data, stack, 4, 0x6474E551)
        _put_integer(data, stack + 4, 4, 0x6)
        relro = program_offset + (program_size * 3)
        _put_integer(data, relro, 4, 0x6474E552)
        _put_integer(data, relro + 8, 8, dynamic_offset)
        _put_integer(data, relro + 16, 8, virtual_base + dynamic_offset)
        _put_integer(data, relro + 32, 8, dynamic_size)
        _put_integer(data, relro + 40, 8, dynamic_size)

    entry = dynamic_offset
    for string_index in string_offsets:
        _put_integer(data, entry, 8, 1)
        _put_integer(data, entry + 8, 8, string_index)
        entry += 16
    _put_integer(data, entry, 8, 5)
    _put_integer(data, entry + 8, 8, virtual_base + string_offset)
    entry += 16
    _put_integer(data, entry, 8, 10)
    _put_integer(data, entry + 8, 8, len(string_table))
    entry += 16
    if runpath_offset is not None:
        _put_integer(data, entry, 8, 29)
        _put_integer(data, entry + 8, 8, runpath_offset)
        entry += 16
    if hardened:
        _put_integer(data, entry, 8, 30)
        _put_integer(data, entry + 8, 8, 0x8)
        entry += 16
        _put_integer(data, entry, 8, 0x6FFFFFFB)
        _put_integer(data, entry + 8, 8, 0x1)
    data[string_offset:] = string_table
    return bytes(data)


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    path.chmod(0o600)


def _receipt_payloads(
    config: ParakeetCppConfig,
    *,
    source_sha256: str,
    libparakeet_sha256: str,
    libparakeet_size_bytes: int,
    bridge_sha256: str,
    bridge_size_bytes: int,
    bridge_source_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    source = {
        "schema_version": 1,
        "kind": "parakeet-cpp-source-receipt-v1",
        "repo_id": config.upstream_repo_id,
        "commit": config.upstream_commit,
        "tag": config.upstream_tag,
        "ggml_commit": config.ggml_commit,
        "parent_pristine_git_tree": manifest_module.PARAKEET_CPP_PARENT_GIT_TREE,
        "ggml_pristine_git_tree": manifest_module.PARAKEET_CPP_GGML_GIT_TREE,
        "ordered_patches": [
            {"order": order, "filename": filename, "sha256": sha256}
            for order, (filename, sha256) in enumerate(
                manifest_module.PARAKEET_CPP_PATCH_SPECS,
                start=1,
            )
        ],
        "patched_diff_sha256": (
            manifest_module.PARAKEET_CPP_PATCHED_DIFF_SHA256
        ),
        "license": config.upstream_license,
        "license_sha256": config.upstream_license_sha256,
    }
    build = {
        "schema_version": 1,
        "kind": "parakeet-cpp-build-receipt-v1",
        "source_receipt_sha256": source_sha256,
        "c_api_version": config.c_api_version,
        "bridge_abi_version": config.bridge_abi_version,
        "requested_device": config.requested_device,
        "actual_device": config.actual_device,
        "num_threads": config.num_threads,
        "bridge_source_sha256": bridge_source_sha256,
        "compiler_id": "GNU",
        "compiler_version": "13.3.0",
        "compiler_package": "Ubuntu 13.3.0-6ubuntu2~24.04.1",
        "cmake_version": "3.31.10",
        "ninja_version": "1.13.0.git.kitware.jobserver-pipe-1",
        "system_processor": "x86_64",
        "ggml_system_arch": "x86",
        "cpu_variant_flags": [
            "-msse4.2",
            "-mf16c",
            "-mfma",
            "-mbmi2",
            "-mavx",
            "-mavx2",
        ],
        "cmake": {
            "PARAKEET_SHARED": "ON",
            "BUILD_SHARED_LIBS": "OFF",
            "GGML_STATIC": "ON",
            "GGML_NATIVE": "OFF",
            "GGML_OPENMP": "OFF",
            "PARAKEET_GGML_CUDA": "OFF",
            "PARAKEET_GGML_METAL": "OFF",
            "PARAKEET_GGML_VULKAN": "OFF",
            "PARAKEET_GGML_HIP": "OFF",
        },
        "libparakeet_sha256": libparakeet_sha256,
        "libparakeet_size_bytes": libparakeet_size_bytes,
        "bridge_library_sha256": bridge_sha256,
        "bridge_library_size_bytes": bridge_size_bytes,
        "libparakeet_needed": list(_LIBPARAKEET_NEEDED),
        "bridge_needed": list(_BRIDGE_NEEDED),
        "bridge_runpath": "$ORIGIN",
        "bridge_relro": True,
        "bridge_bind_now": True,
        "bridge_noexecstack": True,
    }
    return source, build


def _inputs(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    config = ParakeetCppConfig()
    native_root = tmp_path / "native"
    model_root = tmp_path / "model"
    native_root.mkdir(mode=0o700)
    model_root.mkdir(mode=0o700)
    native_root.chmod(0o700)
    model_root.chmod(0o700)

    libparakeet = native_root / "libparakeet.so"
    libparakeet.write_bytes(_elf_bytes(_LIBPARAKEET_NEEDED))
    libparakeet.chmod(0o600)
    libparakeet_receipt = (
        hashlib.sha256(libparakeet.read_bytes()).hexdigest(),
        libparakeet.stat().st_size,
    )
    bridge = native_root / "libspeaker_parakeet_bridge.so"
    bridge.write_bytes(
        _elf_bytes(
            _BRIDGE_NEEDED,
            runpath="$ORIGIN",
            hardened=True,
        )
    )
    bridge.chmod(0o600)
    bridge_receipt = (
        hashlib.sha256(bridge.read_bytes()).hexdigest(),
        bridge.stat().st_size,
    )
    for module in (provision, manifest_module):
        monkeypatch.setattr(
            module,
            "PARAKEET_CPP_LIBPARAKEET_RECEIPT",
            libparakeet_receipt,
        )
        monkeypatch.setattr(
            module,
            "PARAKEET_CPP_BRIDGE_RECEIPT",
            bridge_receipt,
        )
        monkeypatch.setattr(
            module,
            "PARAKEET_CPP_LIBPARAKEET_NEEDED",
            _LIBPARAKEET_NEEDED,
        )
        monkeypatch.setattr(
            module,
            "PARAKEET_CPP_BRIDGE_NEEDED",
            _BRIDGE_NEEDED,
        )

    bridge_source = tmp_path / "parakeet_cpp_bridge.cpp"
    bridge_source.write_bytes(b"extern \"C\" int speaker_bridge(void) { return 0; }\n")
    bridge_source_sha256 = hashlib.sha256(bridge_source.read_bytes()).hexdigest()
    monkeypatch.setattr(provision, "_FIXED_BRIDGE_SOURCE", bridge_source)
    for module in (provision, manifest_module):
        monkeypatch.setattr(
            module,
            "PARAKEET_CPP_BRIDGE_SOURCE_SHA256",
            bridge_source_sha256,
        )

    source_receipt = native_root / "source-receipt.json"
    source_payload, _unused = _receipt_payloads(
        config,
        source_sha256="0" * 64,
        libparakeet_sha256=libparakeet_receipt[0],
        libparakeet_size_bytes=libparakeet_receipt[1],
        bridge_sha256=bridge_receipt[0],
        bridge_size_bytes=bridge_receipt[1],
        bridge_source_sha256=bridge_source_sha256,
    )
    _write_json(source_receipt, source_payload)
    source_sha256 = hashlib.sha256(source_receipt.read_bytes()).hexdigest()
    _unused, build_payload = _receipt_payloads(
        config,
        source_sha256=source_sha256,
        libparakeet_sha256=libparakeet_receipt[0],
        libparakeet_size_bytes=libparakeet_receipt[1],
        bridge_sha256=bridge_receipt[0],
        bridge_size_bytes=bridge_receipt[1],
        bridge_source_sha256=bridge_source_sha256,
    )
    build_receipt = native_root / "build-receipt.json"
    _write_json(build_receipt, build_payload)

    model = model_root / config.model_filename
    with model.open("wb") as handle:
        handle.truncate(config.model_size_bytes)
    model.chmod(0o600)
    model_receipt = model_root / "model-receipt.json"
    _write_json(
        model_receipt,
        {
            "schema_version": 1,
            "kind": "parakeet-cpp-model-receipt-v1",
            "model_repo_id": config.model_repo_id,
            "model_revision": config.model_revision,
            "source_model_repo_id": config.source_model_repo_id,
            "source_model_revision": config.source_model_revision,
            "filename": config.model_filename,
            "sha256": config.model_sha256,
            "size_bytes": config.model_size_bytes,
            "dtype": config.model_dtype,
            "license": config.model_license,
        },
    )

    real_provision_hash = provision.hash_regular_bounded
    real_manifest_hash = manifest_module.hash_regular_bounded
    resolved_model = model.resolve(strict=True)

    def fake_model_hash(real_hash):
        def wrapped(path, *, maximum_bytes, expected_bytes, allow_final_symlink=False):
            candidate = Path(path).resolve(strict=True)
            if candidate == resolved_model:
                assert expected_bytes == PARAKEET_CPP_MODEL_RECEIPT[1]
                assert expected_bytes <= maximum_bytes
                return FileDigest(
                    path=candidate,
                    sha256=PARAKEET_CPP_MODEL_RECEIPT[0],
                    size_bytes=expected_bytes,
                )
            return real_hash(
                path,
                maximum_bytes=maximum_bytes,
                expected_bytes=expected_bytes,
                allow_final_symlink=allow_final_symlink,
            )

        return wrapped

    monkeypatch.setattr(
        provision,
        "hash_regular_bounded",
        fake_model_hash(real_provision_hash),
    )
    monkeypatch.setattr(
        manifest_module,
        "hash_regular_bounded",
        fake_model_hash(real_manifest_hash),
    )
    return {
        "python": Path(sys.executable),
        "native_root": native_root,
        "model_root": model_root,
        "output_dir": tmp_path / "candidate",
        "source_receipt": source_receipt,
        "build_receipt": build_receipt,
        "model_receipt": model_receipt,
        "libparakeet": libparakeet,
        "bridge": bridge,
        "bridge_source": bridge_source,
        "model": model,
    }


def _provision(tmp_path: Path, monkeypatch):
    paths = _inputs(tmp_path, monkeypatch)
    manifest = provision.provision_candidate(
        python=paths["python"],
        native_root=paths["native_root"],
        model_root=paths["model_root"],
        output_dir=paths["output_dir"],
    )
    return manifest, paths


def test_bounded_elf_parser_returns_exact_dependency_order(tmp_path):
    path = tmp_path / "fixture.so"
    path.write_bytes(_elf_bytes(_LIBPARAKEET_NEEDED))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert manifest_module.elf_dynamic_dependencies(
        path,
        expected_sha256=digest,
        expected_size_bytes=path.stat().st_size,
        maximum_bytes=1024 * 1024,
    ) == _LIBPARAKEET_NEEDED

    changed = bytearray(path.read_bytes())
    changed[18:20] = (183).to_bytes(2, "little")
    path.write_bytes(changed)
    with pytest.raises(manifest_module.ManifestError):
        manifest_module.elf_dynamic_dependencies(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_size_bytes=path.stat().st_size,
            maximum_bytes=1024 * 1024,
        )


def test_provision_binds_closed_schema_v8_without_loading_candidate(
    tmp_path,
    monkeypatch,
):
    manifest, paths = _provision(tmp_path, monkeypatch)

    assert manifest.schema_version == 8
    assert manifest.adapter == PARAKEET_CPP_ADAPTER
    assert manifest.model_id == "parakeet-cpp-realtime-eou-120m-v1-f16-cpu"
    assert isinstance(manifest.adapter_config, ParakeetCppConfig)
    assert tuple(manifest.artifact_by_name) == PARAKEET_CPP_ARTIFACT_NAMES
    assert manifest.artifact_by_name["libparakeet"].path == paths["libparakeet"]
    assert manifest.artifact_by_name["bridge-library"].path == paths["bridge"]
    assert manifest.artifact_by_name["model-gguf"].path == paths["model"]
    assert manifest.limits.startup_timeout_sec == 120.0
    assert manifest.limits.case_timeout_sec == 300.0
    assert stat.S_IMODE(manifest.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest.path.stat().st_mode) == 0o600
    assert {path.name for path in manifest.path.parent.iterdir()} == {
        "worker-manifest.json"
    }
    safe = provision._safe_result(manifest)
    assert str(tmp_path) not in json.dumps(safe)
    assert safe["bridge_source_sha256"] == (
        provision.PARAKEET_CPP_BRIDGE_SOURCE_SHA256
    )


def test_provision_never_imports_or_loads_candidate_runtime(
    tmp_path,
    monkeypatch,
):
    imported = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "ctypes",
            "huggingface_hub",
            "requests",
            "subprocess",
            "torch",
            "urllib",
        }:
            imported.append(name)
            raise AssertionError("candidate loading or network import attempted")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    _manifest, _paths = _provision(tmp_path, monkeypatch)

    assert imported == []


@pytest.mark.parametrize(
    ("receipt_name", "mutate"),
    [
        (
            "source_receipt",
            lambda payload: payload["ordered_patches"][0].update(
                {"sha256": "f" * 64}
            ),
        ),
        (
            "build_receipt",
            lambda payload: payload["cmake"].update({"GGML_OPENMP": "ON"}),
        ),
        (
            "build_receipt",
            lambda payload: payload.update({"compiler_version": "13.3.1"}),
        ),
        (
            "model_receipt",
            lambda payload: payload.update({"dtype": "Q8_0"}),
        ),
    ],
)
def test_provision_rejects_changed_exact_receipt(
    tmp_path,
    monkeypatch,
    receipt_name,
    mutate,
):
    paths = _inputs(tmp_path, monkeypatch)
    receipt = paths[receipt_name]
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(receipt, payload)

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            python=paths["python"],
            native_root=paths["native_root"],
            model_root=paths["model_root"],
            output_dir=paths["output_dir"],
        )
    assert not paths["output_dir"].exists()


@pytest.mark.parametrize(
    "unsafe",
    ["mode", "root-mode", "hardlink", "extra-entry"],
)
def test_provision_rejects_nonprivate_or_open_native_input(
    tmp_path,
    monkeypatch,
    unsafe,
):
    paths = _inputs(tmp_path, monkeypatch)
    if unsafe == "mode":
        paths["bridge"].chmod(0o644)
    elif unsafe == "root-mode":
        paths["native_root"].chmod(0o750)
    elif unsafe == "hardlink":
        os.link(paths["bridge"], tmp_path / "external-bridge-link.so")
    else:
        extra = paths["native_root"] / "unreceipted.so"
        extra.write_bytes(b"not closed")
        extra.chmod(0o600)

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            python=paths["python"],
            native_root=paths["native_root"],
            model_root=paths["model_root"],
            output_dir=paths["output_dir"],
        )
    assert not paths["output_dir"].exists()


def test_provision_rejects_existing_output_without_overwrite(
    tmp_path,
    monkeypatch,
):
    paths = _inputs(tmp_path, monkeypatch)
    paths["output_dir"].mkdir(mode=0o700)
    sentinel = paths["output_dir"] / "keep"
    sentinel.write_bytes(b"preserve")

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            python=paths["python"],
            native_root=paths["native_root"],
            model_root=paths["model_root"],
            output_dir=paths["output_dir"],
        )
    assert sentinel.read_bytes() == b"preserve"


def test_provision_revalidates_inputs_after_canonical_link(
    tmp_path,
    monkeypatch,
):
    paths = _inputs(tmp_path, monkeypatch)

    def mutate_input() -> None:
        paths["bridge"].chmod(0o640)

    monkeypatch.setattr(provision, "_post_validation_fault_point", mutate_input)
    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            python=paths["python"],
            native_root=paths["native_root"],
            model_root=paths["model_root"],
            output_dir=paths["output_dir"],
        )
    assert not (paths["output_dir"] / "worker-manifest.json").exists()
    assert not (
        paths["output_dir"] / provision._PENDING_MANIFEST_FILENAME
    ).exists()


def test_provision_hash_binds_committed_bridge_source(tmp_path, monkeypatch):
    paths = _inputs(tmp_path, monkeypatch)
    paths["bridge_source"].write_bytes(b"changed bridge source\n")

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            python=paths["python"],
            native_root=paths["native_root"],
            model_root=paths["model_root"],
            output_dir=paths["output_dir"],
        )
    assert not paths["output_dir"].exists()


def test_cli_failure_is_path_and_detail_free(tmp_path, capsys):
    secret = tmp_path / "private-name-never-print"

    code = provision.main(
        [
            "--python",
            str(secret / "python"),
            "--native-root",
            str(secret / "native"),
            "--model-root",
            str(secret / "model"),
            "--output-dir",
            str(secret / "output"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.err == ""
    assert str(secret) not in captured.out
    assert json.loads(captured.out) == provision._SAFE_ERROR


def test_cli_help_preserves_offline_no_execution_boundary(capsys):
    with pytest.raises(SystemExit) as raised:
        provision.main(["--help"])

    captured = capsys.readouterr()
    assert raised.value.code == 0
    assert captured.err == ""
    help_text = " ".join(captured.out.split())
    for promise in (
        "without building",
        "downloading",
        "importing",
        "executing",
        "loading the candidate",
    ):
        assert promise in help_text


def test_failed_publication_preserves_foreign_replacement(
    tmp_path,
    monkeypatch,
):
    paths = _inputs(tmp_path, monkeypatch)
    canonical = paths["output_dir"] / "worker-manifest.json"

    def replace_canonical() -> None:
        canonical.unlink()
        canonical.write_bytes(b"foreign replacement")
        canonical.chmod(0o600)

    monkeypatch.setattr(
        provision,
        "_post_validation_fault_point",
        replace_canonical,
    )
    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            python=paths["python"],
            native_root=paths["native_root"],
            model_root=paths["model_root"],
            output_dir=paths["output_dir"],
        )

    assert canonical.read_bytes() == b"foreign replacement"
    assert not (
        paths["output_dir"] / provision._PENDING_MANIFEST_FILENAME
    ).exists()
