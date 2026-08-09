from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from tools.semantic_interruption import anyreach_runtime as runtime
from tools.streaming_stt.wheel_lock import load_wheel_lock


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _write_rebound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    *,
    raw: bytes | None = None,
) -> Path:
    payload = _canonical(value) if raw is None else raw
    path = tmp_path / "runtime.lock.json"
    path.write_bytes(payload)
    monkeypatch.setattr(runtime, "RUNTIME_LOCK_SIZE_BYTES", len(payload))
    monkeypatch.setattr(
        runtime,
        "RUNTIME_LOCK_RAW_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    return path


def test_runtime_source_lock_has_exact_settled_bytes() -> None:
    raw = runtime.DEFAULT_RUNTIME_LOCK.read_bytes()

    assert len(raw) == runtime.RUNTIME_LOCK_SIZE_BYTES == 8_185
    assert hashlib.sha256(raw).hexdigest() == runtime.RUNTIME_LOCK_RAW_SHA256
    assert raw == _canonical(json.loads(raw))


def test_runtime_source_lock_closes_exact_target_and_roots() -> None:
    lock = runtime.load_runtime_source_lock()

    assert lock.runtime_source_id == runtime.RUNTIME_SOURCE_ID
    assert lock.digest == runtime.RUNTIME_LOCK_RAW_SHA256
    assert lock.target == {
        "implementation": "cpython",
        "python_version": "3.12",
        "abi": "cp312",
        "operating_system": "linux",
        "architecture": "x86_64",
        "minimum_glibc": "2.28",
        "provider": "CPUExecutionProvider",
    }
    assert lock.root_versions == {
        "numpy": "2.4.6",
        "onnxruntime": "1.28.0",
        "tokenizers": "0.23.1",
    }
    assert isinstance(lock.target, MappingProxyType)
    assert isinstance(lock.root_versions, MappingProxyType)


def test_runtime_source_lock_has_exact_complete_package_selection() -> None:
    lock = runtime.load_runtime_source_lock()

    assert len(lock.wheels) == runtime.RUNTIME_WHEEL_COUNT == 20
    assert lock.total_size_bytes == runtime.RUNTIME_WHEEL_BYTES == 46_744_727
    assert tuple(lock.package_versions.items()) == (
        ("anyio", "4.14.2"),
        ("certifi", "2026.7.22"),
        ("click", "8.4.2"),
        ("filelock", "3.32.2"),
        ("flatbuffers", "25.12.19"),
        ("fsspec", "2026.7.0"),
        ("h11", "0.16.0"),
        ("hf-xet", "1.5.2"),
        ("httpcore", "1.0.9"),
        ("httpx", "0.28.1"),
        ("huggingface-hub", "1.26.0"),
        ("idna", "3.18"),
        ("numpy", "2.4.6"),
        ("onnxruntime", "1.28.0"),
        ("packaging", "26.2"),
        ("protobuf", "7.35.1"),
        ("pyyaml", "6.0.3"),
        ("tokenizers", "0.23.1"),
        ("tqdm", "4.70.0"),
        ("typing-extensions", "4.16.0"),
    )
    assert isinstance(lock.package_versions, MappingProxyType)


def test_runtime_source_lock_distinguishes_source_from_execution_evidence() -> None:
    scope = runtime.load_runtime_source_lock().evidence_scope

    assert scope == {
        "package_sources_bound": True,
        "dependency_resolution_declared": True,
        "wheel_bytes_acquired": False,
        "wheelhouse_verified": False,
        "installed_runtime_verified": False,
        "package_imports_executed": False,
        "tokenizer_executed": False,
        "model_executed": False,
        "graph_contract_verified": False,
        "runtime_compatible_with_model": False,
        "production_evidence": False,
        "redistribution_authority": False,
    }
    assert isinstance(scope, MappingProxyType)


def test_runtime_source_lock_binds_official_https_wheel_locations() -> None:
    lock = runtime.load_runtime_source_lock()

    assert all(
        item.source_url.startswith("https://files.pythonhosted.org/packages/")
        and item.source_url.endswith("/" + item.filename)
        for item in lock.wheels
    )
    assert all(len(item.sha256) == 64 for item in lock.wheels)
    assert all(item.size_bytes > 0 for item in lock.wheels)


def test_byte_identical_relocated_runtime_lock_loads(tmp_path: Path) -> None:
    candidate = tmp_path / "relocated.json"
    candidate.write_bytes(runtime.DEFAULT_RUNTIME_LOCK.read_bytes())

    loaded = runtime.load_runtime_source_lock(candidate)

    assert loaded.digest == runtime.RUNTIME_LOCK_RAW_SHA256
    assert loaded.path == candidate.resolve()


def test_runtime_source_lock_is_compatible_with_exact_private_wheel_loader(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "measured-wheel-source.lock.json"
    candidate.write_bytes(runtime.DEFAULT_RUNTIME_LOCK.read_bytes())
    candidate.chmod(0o600)

    loaded = load_wheel_lock(
        candidate,
        expected_digest=runtime.RUNTIME_LOCK_RAW_SHA256,
    )

    assert len(loaded.wheels) == runtime.RUNTIME_WHEEL_COUNT
    assert loaded.total_size_bytes == runtime.RUNTIME_WHEEL_BYTES


def test_changed_runtime_lock_fails_even_when_json_is_valid(tmp_path: Path) -> None:
    value = json.loads(runtime.DEFAULT_RUNTIME_LOCK.read_bytes())
    value["wheels"][0]["size_bytes"] += 1
    candidate = tmp_path / "changed.json"
    candidate.write_bytes(_canonical(value))

    with pytest.raises(runtime.AnyreachRuntimeContractError) as caught:
        runtime.load_runtime_source_lock(candidate)

    assert str(caught.value) == ""


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("unknown-root", True),
        ("schema", 2),
        ("unsorted", None),
        ("duplicate-distribution", None),
        ("wrong-version", "0.0.0"),
        ("wrong-tag", "py3-none-any"),
        ("wrong-host", "https://example.com/anyio-4.14.2-py3-none-any.whl"),
        ("query", "?download=1"),
        ("zero-size", 0),
        ("uppercase-sha", "A" * 64),
    ),
)
def test_semantic_runtime_lock_validation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    value: object,
) -> None:
    payload = json.loads(runtime.DEFAULT_RUNTIME_LOCK.read_bytes())
    if mutation == "unknown-root":
        payload["unexpected"] = value
    elif mutation == "schema":
        payload["schema_version"] = value
    elif mutation == "unsorted":
        payload["wheels"][0], payload["wheels"][1] = (
            payload["wheels"][1],
            payload["wheels"][0],
        )
    elif mutation == "duplicate-distribution":
        payload["wheels"][1] = copy.deepcopy(payload["wheels"][0])
    elif mutation == "wrong-version":
        payload["wheels"][0]["version"] = value
    elif mutation == "wrong-tag":
        payload["wheels"][12]["wheel_tag"] = value
    elif mutation == "wrong-host":
        payload["wheels"][0]["source_url"] = value
    elif mutation == "query":
        payload["wheels"][0]["source_url"] += value
    elif mutation == "zero-size":
        payload["wheels"][0]["size_bytes"] = value
    else:
        payload["wheels"][0]["sha256"] = value
    candidate = _write_rebound(tmp_path, monkeypatch, payload)

    with pytest.raises(runtime.AnyreachRuntimeContractError):
        runtime.load_runtime_source_lock(candidate)


@pytest.mark.parametrize(
    "raw",
    (
        b'{"schema_version":1,"schema_version":1,"wheels":[]}',
        b'{"schema_version":NaN,"wheels":[]}',
        b"\xff",
    ),
)
def test_runtime_lock_strict_json_rejects_ambiguous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
) -> None:
    candidate = _write_rebound(tmp_path, monkeypatch, {}, raw=raw)

    with pytest.raises(runtime.AnyreachRuntimeContractError):
        runtime.load_runtime_source_lock(candidate)


def test_runtime_contract_module_has_no_candidate_or_network_imports() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert not imports & {
        "huggingface_hub",
        "numpy",
        "onnx",
        "onnxruntime",
        "pyarrow",
        "requests",
        "tokenizers",
        "torch",
        "transformers",
        "urllib3",
    }
    assert "subprocess" not in imports


def test_runtime_source_dataclasses_do_not_expose_paths_or_urls_in_repr() -> None:
    lock = runtime.load_runtime_source_lock()
    private_path = str(lock.path)
    source_url = lock.wheels[0].source_url

    assert private_path not in repr(lock)
    assert source_url not in repr(lock.wheels[0])
