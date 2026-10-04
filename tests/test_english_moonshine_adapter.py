"""Native-free contracts for the isolated English Moonshine adapter."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import english_moonshine_adapter as adapter


class FakeTranscriber:
    def __init__(self):
        self.calls = []
        self.closed = 0
        self.result = SimpleNamespace(
            lines=[
                SimpleNamespace(text=" first line "),
                SimpleNamespace(text=""),
                SimpleNamespace(text="second line"),
            ]
        )
        self.failure = None
        self.close_failure = None

    def transcribe_without_streaming(self, samples, *, sample_rate, flags):
        self.calls.append((samples, sample_rate, flags))
        if self.failure:
            raise self.failure
        return self.result

    def close(self):
        self.closed += 1
        if self.close_failure:
            raise self.close_failure


@pytest.fixture
def native(tmp_path, monkeypatch):
    root = tmp_path / "tiny"
    root.mkdir()
    # Sparse dummy files test only layout admission; they are never loaded.
    for name, (size, _sha, _md5) in adapter.PUBLIC_ARTIFACTS["tiny-streaming"].items():
        with (root / name).open("wb") as handle:
            handle.truncate(size)
    instance = FakeTranscriber()
    loads = []
    monkeypatch.setattr(adapter.os, "sched_getaffinity", lambda _: {2, 5})

    def load(model_dir, arch):
        loads.append((model_dir, arch))
        return instance

    monkeypatch.setattr(adapter, "_load_native_transcriber", load)
    return root, instance, loads


def test_uses_exact_local_tuple_and_normalized_complete_clip(native, monkeypatch):
    monkeypatch.delenv("MOONSHINE_ORT_SINGLE_THREAD", raising=False)
    root, instance, loads = native
    decoder = adapter.build_decoder(root, "tiny-streaming", 2)
    samples = np.array([-1.0, 0.0, 32767 / 32768], dtype=np.float32)
    assert decoder(samples) == "first line second line"
    assert loads == [(root, "tiny-streaming")]
    assert instance.calls == [(samples.tolist(), 16000, 0)]
    assert decoder.cpu_budget_kind == "process_cpu_affinity"
    assert decoder.native_thread_count_configured is None
    assert decoder.metadata()["native_thread_budget_supported"] is False
    assert decoder.metadata()["single_thread_env"] is False
    decoder.close()
    decoder.close()
    assert instance.closed == 1
    with pytest.raises(adapter.MoonshineAdapterError, match="unavailable"):
        decoder(samples)


@pytest.mark.parametrize(
    "value,enabled", [("", False), ("0", False), ("1", True), ("true", True)]
)
def test_thread_escape_hatch_is_reported_as_loaded_not_reread(
    native, monkeypatch, value, enabled
):
    root, _instance, _loads = native
    monkeypatch.setenv("MOONSHINE_ORT_SINGLE_THREAD", value)
    decoder = adapter.build_decoder(root, "tiny-streaming", 2)
    monkeypatch.setenv("MOONSHINE_ORT_SINGLE_THREAD", "1" if not enabled else "0")
    assert decoder.metadata()["single_thread_env"] is enabled
    decoder.close()


@pytest.mark.parametrize(
    "samples",
    [
        np.array([], dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.array([0.0], dtype=np.float64),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([1.1], dtype=np.float32),
        np.zeros(adapter.MAX_SAMPLES + 1, dtype=np.float32),
        [0.0],
    ],
)
def test_invalid_pcm_never_enters_native(native, samples):
    root, instance, _loads = native
    decoder = adapter.build_decoder(root, "tiny-streaming", 2)
    with pytest.raises(adapter.MoonshineAdapterError, match="pcm_invalid"):
        decoder(samples)
    assert instance.calls == []
    decoder.close()


@pytest.mark.parametrize("change", ["missing", "size", "extra", "symlink", "hardlink"])
def test_bad_layout_never_enters_native(native, change):
    root, _instance, loads = native
    target = root / "adapter.ort"
    if change == "missing":
        target.unlink()
    elif change == "size":
        target.write_bytes(b"bad")
    elif change == "extra":
        (root / "frontend.ort").write_bytes(b"bad")
    elif change == "symlink":
        target.rename(root.parent / "outside")
        target.symlink_to(root.parent / "outside")
    else:
        (root.parent / "outside").hardlink_to(target)
    with pytest.raises(adapter.MoonshineAdapterError, match="layout_invalid"):
        adapter.build_decoder(root, "tiny-streaming", 2)
    assert loads == []


def test_symlink_parent_and_relative_path_rejected_before_native(native):
    root, _instance, loads = native
    alias = root.parent / "alias"
    alias.symlink_to(root, target_is_directory=True)
    for value in (alias, Path("relative")):
        with pytest.raises(adapter.MoonshineAdapterError, match="layout_invalid"):
            adapter.build_decoder(value, "tiny-streaming", 2)
    assert loads == []


def test_cpu_budget_must_remain_enforced(native, monkeypatch):
    root, instance, loads = native
    with pytest.raises(adapter.MoonshineAdapterError, match="unverified"):
        adapter.build_decoder(root, "tiny-streaming", 1)
    assert loads == []
    decoder = adapter.build_decoder(root, "tiny-streaming", 2)
    monkeypatch.setattr(adapter.os, "sched_getaffinity", lambda _: {2, 5, 8})
    with pytest.raises(adapter.MoonshineAdapterError, match="unverified"):
        decoder(np.zeros(16, dtype=np.float32))
    assert instance.calls == []
    decoder.close()


def test_native_exception_is_content_free_and_does_not_retry(native):
    root, instance, _loads = native
    decoder = adapter.build_decoder(root, "tiny-streaming", 2)
    instance.failure = RuntimeError("PRIVATE TRANSCRIPT")
    with pytest.raises(adapter.MoonshineAdapterError) as failed:
        decoder(np.zeros(16, dtype=np.float32))
    assert str(failed.value) == "moonshine_native_decode_failed"
    with pytest.raises(adapter.MoonshineAdapterError, match="unavailable"):
        decoder(np.zeros(16, dtype=np.float32))
    assert len(instance.calls) == 1
    decoder.close()


@pytest.mark.parametrize(
    "lines",
    [
        (),
        [SimpleNamespace(text=3)],
        [SimpleNamespace(text="x" * 4097)],
        [SimpleNamespace(text="") for _ in range(257)],
    ],
)
def test_invalid_native_results_fail_closed(native, lines):
    root, instance, _loads = native
    decoder = adapter.build_decoder(root, "tiny-streaming", 2)
    instance.result = SimpleNamespace(lines=lines)
    with pytest.raises(adapter.MoonshineAdapterError, match="native_decode_failed"):
        decoder(np.zeros(16, dtype=np.float32))
    decoder.close()


def test_uncertain_close_is_retained_without_a_second_native_call(native):
    root, instance, _loads = native
    decoder = adapter.build_decoder(root, "tiny-streaming", 2)
    instance.close_failure = RuntimeError("PRIVATE DETAIL")
    for _ in range(2):
        with pytest.raises(adapter.MoonshineAdapterError, match="native_close_failed"):
            decoder.close()
    assert instance.closed == 1


@pytest.mark.parametrize(
    "arch,threads",
    [("tiny", 2), ("small", 2), ("tiny-streaming", True), ("tiny-streaming", 0)],
)
def test_bad_architecture_or_budget_never_loads_native(native, arch, threads):
    root, _instance, loads = native
    with pytest.raises(adapter.MoonshineAdapterError):
        adapter.build_decoder(root, arch, threads)
    assert loads == []
