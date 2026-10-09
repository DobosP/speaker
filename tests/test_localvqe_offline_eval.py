"""Synthetic/foreign-runtime controls only; no library, model, mic or network."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import localvqe_offline_eval as evaluate


@pytest.fixture
def assets(tmp_path, monkeypatch):
    native = tmp_path / "native"
    native.mkdir()
    files = []
    for name in ("liblocalvqe.so.0.1.0", "libggml.so.0", "libggml-base.so.0"):
        raw = name.encode()
        (native / name).write_bytes(raw)
        files.append(
            {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    models, weights = {}, {}
    for model_id in evaluate.WEIGHTS:
        raw = model_id.encode()
        path = tmp_path / (model_id + ".gguf")
        path.write_bytes(raw)
        models[model_id] = str(path)
        weights[model_id] = (len(raw), hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(evaluate, "WEIGHTS", weights)
    value = {
        "schema_version": 1,
        "localvqe_commit": evaluate.CODE_COMMIT,
        "ggml_commit": evaluate.GGML_COMMIT,
        "weight_revision": evaluate.WEIGHT_REVISION,
        "native_dir": str(native),
        "library": "liblocalvqe.so.0.1.0",
        "files": files,
        "models": models,
    }
    path = tmp_path / "assets.json"
    path.write_text(json.dumps(value))
    return path, value


def test_exact_asset_tuple_loads_without_native_import(assets):
    path, value = assets
    assert evaluate.load_assets(path) == value


@pytest.mark.parametrize(
    "change",
    [
        "schema_bool",
        "code",
        "ggml",
        "revision",
        "digest",
        "size",
        "traversal",
        "duplicate",
        "library",
        "unknown",
    ],
)
def test_bad_or_unbound_assets_refuse(assets, change):
    path, value = assets
    if change == "schema_bool":
        value["schema_version"] = True
    elif change == "code":
        value["localvqe_commit"] = "0" * 40
    elif change == "ggml":
        value["ggml_commit"] = "0" * 40
    elif change == "revision":
        value["weight_revision"] = "0" * 40
    elif change == "digest":
        value["files"][0]["sha256"] = "0" * 64
    elif change == "size":
        value["files"][0]["bytes"] += 1
    elif change == "traversal":
        value["files"][0]["name"] = "../outside.so"
    elif change == "duplicate":
        value["files"].append(value["files"][0])
    elif change == "library":
        value["library"] = "arbitrary.so"
    elif change == "unknown":
        value["unexpected"] = "private marker"
    path.write_text(json.dumps(value))
    with pytest.raises(evaluate.EvalError):
        evaluate.load_assets(path)


def test_extra_native_file_and_model_mutation_refuse(assets):
    path, value = assets
    extra = Path(value["native_dir"]) / "unbound.dll"
    extra.write_bytes(b"public fake")
    with pytest.raises(evaluate.EvalError):
        evaluate.load_assets(path)
    extra.unlink()
    Path(value["models"][next(iter(value["models"]))]).write_bytes(b"changed")
    with pytest.raises(evaluate.EvalError):
        evaluate.load_assets(path)


def test_symlink_asset_refuses_before_loading(assets):
    path, value = assets
    original = Path(value["models"][next(iter(value["models"]))])
    alias = original.with_name("alias.gguf")
    try:
        alias.symlink_to(original)
    except OSError:
        pytest.skip("host cannot create test symlink")
    value["models"][next(iter(value["models"]))] = str(alias)
    path.write_text(json.dumps(value))
    with pytest.raises(evaluate.EvalError):
        evaluate.load_assets(path)


def test_json_boundary_rejects_duplicate_and_nonfinite():
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b"\xff"):
        with pytest.raises((evaluate.EvalError, UnicodeError)):
            evaluate.io._json(raw)


def test_read_and_digest_reject_hardlink(tmp_path):
    source, linked = tmp_path / "source", tmp_path / "linked"
    source.write_bytes(b"public synthetic")
    os.link(source, linked)
    with pytest.raises(evaluate.EvalError):
        evaluate.io._read(linked, 1024)
    with pytest.raises(evaluate.EvalError):
        evaluate.io._file_digest(linked, 1024)


def test_only_explicit_runtime_hash_may_bind_empty_python_file(tmp_path):
    path = tmp_path / "empty.py"
    path.write_bytes(b"")
    with pytest.raises(evaluate.EvalError):
        evaluate._hash(path, 1024)
    assert evaluate._hash(path, 1024, allow_empty=True) == (
        hashlib.sha256(b"").hexdigest(),
        0,
    )


def test_generated_tracks_are_deterministic_aligned_and_bounded():
    first, second = evaluate.signal_cases(), evaluate.signal_cases()
    assert tuple(case.id for case in first) == evaluate.CASES
    assert evaluate.signal_binding(first) == evaluate.signal_binding(second)
    for case in first:
        for array in (case.microphone, case.reference, case.near):
            assert array.shape == (evaluate.SAMPLES,)
            assert array.dtype == np.float32
            assert np.all(np.isfinite(array)) and np.max(np.abs(array)) <= 1
    assert not first[3].reference.any()
    assert not first[4].near[:32000].any()
    assert first[4].near[32000:].any()


class Passthrough:
    def __init__(self):
        self.closed = False

    def process(self, near, far):
        assert near.dtype == far.dtype == np.float32
        return near.copy()

    def close(self):
        self.closed = True


def test_feed_preserves_complete_geometry_and_observes_every_callback():
    case = evaluate.signal_cases()[0]
    observations = []
    output, wall, cpu = evaluate.feed_case(
        Passthrough(), case, observe=lambda: observations.append(1)
    )
    np.testing.assert_array_equal(output, case.microphone)
    assert len(observations) == evaluate.SAMPLES // evaluate.CALLBACK_SAMPLES
    assert wall >= 0 and cpu >= 0


@pytest.mark.parametrize("kind", ["int", "nan", "shape", "oversize"])
def test_invalid_adapter_output_refuses(kind):
    class Broken(Passthrough):
        def process(self, near, far):
            if kind == "int":
                return near.astype(np.int16)
            if kind == "nan":
                return near * np.nan
            if kind == "shape":
                return near.reshape(2, -1)
            return np.zeros(evaluate.CALLBACK_SAMPLES + 513, dtype=np.float32)

    with pytest.raises(evaluate.EvalError):
        evaluate.feed_case(Broken(), evaluate.signal_cases()[0])


def test_muting_near_end_is_explicit_instead_of_a_false_quality_win():
    case = evaluate.signal_cases()[3]
    measured = evaluate.metrics(case, np.zeros_like(case.microphone))
    assert measured["near_output_zero"] is True
    assert measured["near_projection_gain"] == 0
    assert measured["near_si_sdr_db"] is None
    assert measured["far_only_attenuation_db"] is None


def test_fresh_adapter_per_case_and_close_on_failure():
    adapters = []

    def create(_assets, _model):
        result = Passthrough()
        adapters.append(result)
        return result

    result = evaluate.benchmark_cell({}, "nlms-default", adapter_factory=create)
    assert len(adapters) == len(evaluate.CASES)
    assert all(adapter.closed for adapter in adapters)
    assert result["cells"][0]["metrics"]["far_only_attenuation_db"] == 0
    phase = []

    class Broken(Passthrough):
        def process(self, near, far):
            raise RuntimeError("private fake exception")

    broken = Broken()
    with pytest.raises(RuntimeError):
        evaluate.benchmark_cell(
            {}, "nlms-default", adapter_factory=lambda *_: broken, phase=phase.append
        )
    assert broken.closed
    assert phase[-1] == "frames"


class Function:
    def __init__(self, name, owner):
        self.name, self.owner = name, owner

    def __call__(self, *args):
        self.owner.calls.append((self.name, args))
        defaults = {
            "localvqe_options_new": 1,
            "localvqe_new_with_options": 2,
            "localvqe_sample_rate": 16000,
            "localvqe_hop_length": 256,
            "localvqe_fft_size": 512,
        }
        return self.owner.results.get(self.name, defaults.get(self.name, 0))


class FakeLibrary:
    def __init__(self, results=None):
        self.calls, self.results = [], results or {}

    def __getattr__(self, name):
        result = Function(name, self)
        setattr(self, name, result)
        return result


def test_c_api_enforces_cpu_one_thread_reset_and_noise_gate_off(monkeypatch):
    library = FakeLibrary()
    monkeypatch.setattr(evaluate.ctypes, "CDLL", lambda _path: library)
    adapter = evaluate.LocalVqe(
        {
            "native_dir": "/public",
            "library": "liblocalvqe.dylib",
            "models": {"model": "/public/model.gguf"},
        },
        "model",
    )
    adapter.close()
    adapter.close()
    assert ("localvqe_options_set_backend", (1, b"CPU")) in library.calls
    assert ("localvqe_options_set_threads", (1, 1)) in library.calls
    assert ("localvqe_set_noise_gate", (2, 0, -45.0)) in library.calls
    assert sum(name == "localvqe_reset" for name, _args in library.calls) == 1
    assert sum(name == "localvqe_free" for name, _args in library.calls) == 1
    assert sum(name == "localvqe_options_free" for name, _args in library.calls) == 1


@pytest.mark.parametrize(
    "failure",
    ["localvqe_options_set_threads", "localvqe_hop_length", "localvqe_set_noise_gate"],
)
def test_constructor_failure_frees_entered_resources(monkeypatch, failure):
    library = FakeLibrary({failure: -1})
    monkeypatch.setattr(evaluate.ctypes, "CDLL", lambda _path: library)
    with pytest.raises(evaluate.EvalError):
        evaluate.LocalVqe(
            {
                "native_dir": "/public",
                "library": "localvqe.dll",
                "models": {"model": "/public/model.gguf"},
            },
            "model",
        )
    assert sum(name == "localvqe_options_free" for name, _args in library.calls) == 1
    assert sum(name == "localvqe_free" for name, _args in library.calls) == (
        0 if failure == "localvqe_options_set_threads" else 1
    )


def test_windows_dll_directory_is_exact_and_closed(monkeypatch):
    closed = []
    scopes = []

    def directory(path):
        scopes.append(path)
        return SimpleNamespace(close=lambda: closed.append(1))

    monkeypatch.setattr(
        evaluate,
        "os",
        SimpleNamespace(name="nt", add_dll_directory=directory, fsencode=os.fsencode),
    )
    monkeypatch.setattr(evaluate.ctypes, "CDLL", lambda _path: FakeLibrary())
    adapter = evaluate.LocalVqe(
        {
            "native_dir": "/public",
            "library": "localvqe.dll",
            "models": {"model": "/public/model.gguf"},
        },
        "model",
    )
    adapter.close()
    assert scopes == ["/public"] and closed == [1]


def test_missing_affinity_api_is_honest_and_does_not_import_native(monkeypatch):
    monkeypatch.setattr(evaluate, "os", SimpleNamespace())
    assert evaluate._affinity() is False


def _complete_result(monkeypatch):
    monkeypatch.setattr(evaluate.sys, "platform", "darwin")
    result = evaluate.benchmark_cell(
        {}, "nlms-default", adapter_factory=lambda *_: Passthrough()
    )
    result.update(
        {
            "bindings_sha256": "a" * 64,
            "peak_rss_kib": None,
            "cpu_affinity_applied": False,
            "posix_resource_limits_applied": False,
            "thread_affinity": None,
        }
    )
    return result


def test_portable_scalar_report_validation(monkeypatch):
    result = _complete_result(monkeypatch)
    evaluate.validate_result(result, "nlms-default", "a" * 64)


@pytest.mark.parametrize(
    "change",
    [
        "text",
        "bool_time",
        "nan",
        "missing_case",
        "wrong_digest",
        "bad_geometry",
        "array",
        "wrong_null",
        "negative_peak",
    ],
)
def test_worker_result_cannot_publish_unbound_data(monkeypatch, change):
    result = _complete_result(monkeypatch)
    cell = result["cells"][0]
    if change == "text":
        result["hypothesis"] = "private fake"
    elif change == "bool_time":
        cell["process_wall_seconds"] = True
    elif change == "nan":
        cell["metrics"]["output_peak"] = float("nan")
    elif change == "missing_case":
        result["cells"].pop()
    elif change == "wrong_digest":
        result["signal_digest"] = "b" * 64
    elif change == "bad_geometry":
        cell["output_samples"] -= 1
    elif change == "array":
        cell["metrics"]["output_peak"] = [1, 2]
    elif change == "wrong_null":
        cell["metrics"]["near_projection_gain"] = 1
    elif change == "negative_peak":
        cell["metrics"]["output_peak"] = -1
    with pytest.raises(evaluate.EvalError):
        evaluate.validate_result(result, "nlms-default", "a" * 64)


def test_fixed_failure_protocol_contains_only_id_and_phase():
    evaluate.validate_result(
        {"model_id": "nlms-default", "status": "failed", "phase": "timeout"},
        "nlms-default",
        "a" * 64,
    )
    with pytest.raises(evaluate.EvalError):
        evaluate.validate_result(
            {
                "model_id": "nlms-default",
                "status": "failed",
                "phase": "timeout",
                "error": "private fake",
            },
            "nlms-default",
            "a" * 64,
        )


def test_cli_errors_never_echo_paths_or_exception_details(capsys):
    assert evaluate.main(["--unknown-private-path", "/private/fake"]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "ok": False,
        "error": "localvqe_offline_evaluation_failed",
    }


def test_parent_timeout_terminates_child_and_keeps_honest_failure_phase(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "bindings", lambda _path: {"public_binding": 1})
    monkeypatch.setattr(
        evaluate, "load_assets", lambda _path: {"native_dir": str(tmp_path)}
    )
    killed = []

    class Hung:
        def wait(self, timeout):
            raise evaluate.subprocess.TimeoutExpired("public fake", timeout)

    def launch(_args, **kwargs):
        assert (
            kwargs["stdin"]
            == kwargs["stdout"]
            == kwargs["stderr"]
            == evaluate.subprocess.DEVNULL
        )
        assert kwargs["env"]["LOCALVQE_ALLOW_UNHASHED"] == "0"
        assert kwargs["env"]["GGML_NTHREADS"] == "1"
        return Hung()

    monkeypatch.setattr(evaluate.subprocess, "Popen", launch)
    monkeypatch.setattr(evaluate.io, "_stop_group", lambda child: killed.append(child))
    result = evaluate.run_cell(
        tmp_path / "assets.json", "nlms-default", tmp_path, timeout=10
    )
    assert len(killed) == 1
    assert result == {
        "model_id": "nlms-default",
        "status": "failed",
        "phase": "timeout",
    }


def test_parent_refuses_mixed_source_binding_before_publication(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "report.json"
    snapshots = iter([{"source": 1}, {"source": 2}])
    monkeypatch.setattr(evaluate, "bindings", lambda _path: next(snapshots))
    monkeypatch.setattr(
        evaluate,
        "run_cell",
        lambda _path, model_id, _scratch: {
            "model_id": model_id,
            "status": "failed",
            "phase": "load",
        },
    )
    assert (
        evaluate.main(
            [
                "--assets",
                str(tmp_path / "assets.json"),
                "--scratch",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()
    assert (
        json.loads(capsys.readouterr().out)["error"]
        == "localvqe_offline_evaluation_failed"
    )
