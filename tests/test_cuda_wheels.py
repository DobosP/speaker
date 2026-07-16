from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.engines import _cuda_wheels


def _wheel_directories(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for package, soname in _cuda_wheels._WHEEL_LIBRARIES:
        directory = result.setdefault(package, tmp_path / package / "lib")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / soname).touch()
    return result


def test_bootstrap_loads_exact_wheel_sonames_once_in_dependency_order(tmp_path):
    directories = _wheel_directories(tmp_path)
    loads = []
    environment_before = dict(os.environ)

    def load_library(path, *, mode):
        handle = object()
        loads.append((path, mode, handle))
        return handle

    bootstrap = _cuda_wheels._CudaWheelBootstrap(
        lib_directory=directories.__getitem__,
        load_library=load_library,
        system=lambda: "linux",
        machine=lambda: "x86_64",
    )

    bootstrap.preload()
    bootstrap.preload()

    assert [Path(path).name for path, _, _ in loads] == [
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libnvrtc.so.12",
        "libcudnn.so.9",
    ]
    assert all(Path(path).is_absolute() for path, _, _ in loads)
    assert all(mode == _cuda_wheels._LOAD_MODE for _, mode, _ in loads)
    assert bootstrap._handles == [handle for _, _, handle in loads]
    assert os.environ == environment_before


def test_bootstrap_fails_closed_without_using_an_ambient_library(tmp_path):
    directories = _wheel_directories(tmp_path)
    missing = directories["nvidia.cudnn"] / "libcudnn.so.9"
    missing.unlink()
    loads = []

    bootstrap = _cuda_wheels._CudaWheelBootstrap(
        lib_directory=directories.__getitem__,
        load_library=lambda path, *, mode: loads.append((path, mode)),
        system=lambda: "linux",
        machine=lambda: "x86_64",
    )

    with pytest.raises(RuntimeError, match="pinned CUDA wheel runtime") as raised:
        bootstrap.preload()

    assert "cudnn" not in str(raised.value).lower()
    assert not bootstrap._complete
    assert all(str(tmp_path.resolve()) in path for path, _ in loads)


def test_bootstrap_can_retry_after_a_transient_loader_failure(tmp_path):
    directories = _wheel_directories(tmp_path)
    calls = 0

    def load_library(path, *, mode):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("SENTINEL_PRIVATE_LOADER_DETAIL")
        return object()

    bootstrap = _cuda_wheels._CudaWheelBootstrap(
        lib_directory=directories.__getitem__,
        load_library=load_library,
        system=lambda: "linux",
        machine=lambda: "x86_64",
    )

    with pytest.raises(RuntimeError) as raised:
        bootstrap.preload()
    assert "SENTINEL_PRIVATE_LOADER_DETAIL" not in str(raised.value)

    bootstrap.preload()

    assert bootstrap._complete
    assert calls == len(_cuda_wheels._WHEEL_LIBRARIES) + 1


@pytest.mark.parametrize(
    ("system", "machine"),
    [("darwin", "x86_64"), ("linux", "aarch64"), ("win32", "AMD64")],
)
def test_bootstrap_rejects_unsupported_platform_before_resolution(system, machine):
    bootstrap = _cuda_wheels._CudaWheelBootstrap(
        lib_directory=lambda package: pytest.fail("must not resolve wheels"),
        load_library=lambda path, *, mode: pytest.fail("must not load libraries"),
        system=lambda: system,
        machine=lambda: machine,
    )

    with pytest.raises(RuntimeError, match="requires Linux x86_64"):
        bootstrap.preload()


def test_wheel_directory_accepts_only_one_namespace_below_active_prefix(
    tmp_path,
    monkeypatch,
):
    prefix = tmp_path / "venv"
    package = prefix / "lib/python3.12/site-packages/nvidia/cublas"
    expected = package / "lib"
    expected.mkdir(parents=True)
    monkeypatch.setattr(_cuda_wheels.sys, "prefix", str(prefix))
    monkeypatch.setattr(
        _cuda_wheels,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(package)]),
    )

    assert _cuda_wheels._wheel_lib_directory("nvidia.cublas") == expected.resolve()


def test_wheel_directory_rejects_ambient_and_ambiguous_namespaces(
    tmp_path,
    monkeypatch,
):
    prefix = tmp_path / "venv"
    prefix.mkdir()
    ambient = tmp_path / "usr/local/cuda/nvidia/cublas"
    (ambient / "lib").mkdir(parents=True)
    first = prefix / "site-packages-a/nvidia/cublas"
    second = prefix / "site-packages-b/nvidia/cublas"
    (first / "lib").mkdir(parents=True)
    (second / "lib").mkdir(parents=True)
    monkeypatch.setattr(_cuda_wheels.sys, "prefix", str(prefix))

    monkeypatch.setattr(
        _cuda_wheels,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(ambient)]),
    )
    with pytest.raises(RuntimeError, match="missing or ambiguous"):
        _cuda_wheels._wheel_lib_directory("nvidia.cublas")

    monkeypatch.setattr(
        _cuda_wheels,
        "find_spec",
        lambda name: SimpleNamespace(
            submodule_search_locations=[str(first), str(second)]
        ),
    )
    with pytest.raises(RuntimeError, match="missing or ambiguous"):
        _cuda_wheels._wheel_lib_directory("nvidia.cublas")
