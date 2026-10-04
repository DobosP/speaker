"""Native-free VAD framing, state ownership, privacy and receipt tests."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import sys
import wave

import numpy as np
import pytest

from tools import english_vad_benchmark as subject


class FakeVad:
    def __init__(self, length=600):
        self.length = length
        self.reset_count = 0
        self.flush_count = 0
        self.inputs = []
        self.queue = []
        self.counter = 0
        self.mutate = False
        self.segment = True

    def reset(self):
        self.reset_count += 1
        self.counter = 0
        self.queue.clear()

    def accept_waveform(self, frame):
        self.inputs.append(frame.copy())
        self.counter += 1
        if self.mutate:
            frame[:] = -1

    def is_speech_detected(self):
        return self.counter % 2 == 0

    def flush(self):
        self.flush_count += 1
        if self.segment:
            self.queue.append(SimpleNamespace(start=0, samples=np.zeros(self.length)))

    def empty(self):
        return not self.queue

    @property
    def front(self):
        return self.queue[0]

    def pop(self):
        self.queue.pop(0)


def inputs(tmp_path, frames=600):
    root = tmp_path / "private-recordings"
    root.mkdir()
    clip = root / "private-owner.wav"
    with wave.open(str(clip), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(frames * 2))
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "file": clip.name,
                        "text": "private owner reference",
                        "role": "command",
                    }
                ]
            }
        )
    )
    model_path = tmp_path / "private-model.onnx"
    model_path.write_bytes(b"fake public model")
    return subject._helper().load_corpus(manifest), subject.Model(
        subject.MODEL_IDS[0], model_path
    )


def complete(tmp_path, monkeypatch):
    corpus, model = inputs(tmp_path)
    monkeypatch.setattr(subject.metadata, "version", lambda _name: "1.13.3")
    instance = FakeVad()
    result = subject.benchmark_cell(
        model, corpus, 3, 2, factory=lambda _model, _threads: instance
    )
    result["cpu_affinity"] = {"applied": True, "logical_cpus": 2}
    return corpus, model, instance, result


def test_frames_copy_only_final_partial_padding_and_reset_flush():
    vad = FakeVad(length=600)
    vad.mutate = True
    samples = np.arange(600, dtype=np.float32) / 600
    original = samples.copy()
    result = subject.feed_clip(vad, samples)
    assert vad.reset_count == 1 and vad.flush_count == 1
    assert len(vad.inputs) == 2
    np.testing.assert_array_equal(vad.inputs[0], original[:512])
    np.testing.assert_array_equal(vad.inputs[1][:88], original[512:])
    np.testing.assert_array_equal(vad.inputs[1][88:], np.zeros(424, dtype=np.float32))
    np.testing.assert_array_equal(samples, original)
    assert result["frames"] == 2 and result["active_frames"] == 1
    assert result["padding_samples"] == 424
    assert result["segment_samples"] == 600 and result["segments"] == 1
    assert result["frame_cpu_seconds"] >= 0
    assert len(result["frame_ms"]) == 2
    assert vad.empty()


@pytest.mark.parametrize("frames,padding", [(1, 511), (512, 0), (513, 511), (1024, 0)])
def test_exact_window_and_one_sample_tail(frames, padding):
    vad = FakeVad(length=frames)
    result = subject.feed_clip(vad, np.zeros(frames, dtype=np.float32))
    assert result["frames"] == (frames + 511) // 512
    assert result["padding_samples"] == padding
    assert all(len(frame) == 512 for frame in vad.inputs)


def test_each_clip_owns_fresh_state_and_drains_flush_segment():
    vad = FakeVad()
    for _ in range(3):
        vad.queue.append(SimpleNamespace(start=100000, samples=[]))
        result = subject.feed_clip(vad, np.zeros(600, dtype=np.float32))
        assert result["segments"] == 1 and result["active_frames"] == 1
        assert vad.empty()
    assert vad.reset_count == 3 and vad.flush_count == 3


def test_segment_tail_clamped_to_real_clip_and_completed_segments_drained():
    vad = FakeVad(length=1024)
    result = subject.feed_clip(vad, np.zeros(600, dtype=np.float32))
    assert result["segment_samples"] == 600
    vad.queue = [SimpleNamespace(start=400, samples=np.zeros(500))]
    assert subject._segment_lengths(vad, 600) == (200, 1)
    assert vad.empty()


@pytest.mark.parametrize("start,size", [(-1, 1), (1113, 1), (0, 1113)])
def test_unbounded_or_invalid_segment_is_rejected(start, size):
    vad = FakeVad()
    vad.queue = [SimpleNamespace(start=start, samples=np.zeros(size))]
    with pytest.raises(subject.VadBenchmarkError):
        subject._segment_lengths(vad, 600)


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(0, dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros(10, dtype=np.float64),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([1.01], dtype=np.float32),
        np.zeros(480001, dtype=np.float32),
        [0.0],
    ],
)
def test_invalid_pcm_never_enters_vad(samples):
    vad = FakeVad()
    with pytest.raises(subject.VadBenchmarkError):
        subject.feed_clip(vad, samples)
    assert not vad.inputs and vad.reset_count == 0


def test_full_cell_metrics_state_and_private_text_never_render(tmp_path, monkeypatch):
    corpus, model, vad, result = complete(tmp_path, monkeypatch)
    subject._validate_result(result, model.id, corpus, 3, 2)
    rendered = json.dumps(result)
    for private in ("private", str(tmp_path), "owner reference", ".wav"):
        assert private not in rendered
    assert "private" not in repr(model)
    assert vad.reset_count == 3 and vad.flush_count == 3
    assert result["frames"] == 6 and result["segments"] == 3
    assert result["padding_samples"] == 3 * 424
    assert result["active_frame_fraction"] == 0.5
    assert result["segment_seconds"] == pytest.approx(corpus.seconds * 3)
    assert result["frame_rtf"] == pytest.approx(
        result["frame_seconds"] / (corpus.seconds * 3)
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("frame_seconds",), "private text"),
        (("clips",), True),
        (("repeats",), 99),
        (("active_frame_fraction",), 2),
        (("segments",), "private filename"),
        (("config", "window_size"), 1024),
        (("config", "threshold"), False),
        (("cpu_affinity", "applied"), 1),
        (("versions", "python"), "/private/path"),
        (("code_binding", "benchmark_sha256"), "PRIVATE"),
        (("model_binding", "bytes"), 0),
    ],
)
def test_strict_parent_validator_rejects_text_boolean_and_forged_counters(
    tmp_path, monkeypatch, path, value
):
    corpus, model, _vad, result = complete(tmp_path, monkeypatch)
    forged = copy.deepcopy(result)
    leaf = forged
    for key in path[:-1]:
        leaf = leaf[key]
    leaf[path[-1]] = value
    with pytest.raises(subject.VadBenchmarkError):
        subject._validate_result(forged, model.id, corpus, 3, 2)


def test_unknown_private_fields_and_failure_details_rejected(tmp_path, monkeypatch):
    corpus, model, _vad, result = complete(tmp_path, monkeypatch)
    result["private-reference"] = "private words"
    with pytest.raises(subject.VadBenchmarkError):
        subject._validate_result(result, model.id, corpus, 3, 2)
    failed = {
        "model_id": model.id,
        "status": "worker_failed",
        "error_count": 1,
        "phase": "frames",
    }
    subject._validate_result(failed, model.id, corpus, 3, 2)
    failed["exception"] = "private filename"
    with pytest.raises(subject.VadBenchmarkError):
        subject._validate_result(failed, model.id, corpus, 3, 2)


@pytest.mark.parametrize(
    "kind", ["mutate_model", "replace_same_bytes", "mutate_corpus"]
)
def test_post_model_identity_and_corpus_binding_rechecked(tmp_path, monkeypatch, kind):
    corpus, model = inputs(tmp_path)
    monkeypatch.setattr(subject.metadata, "version", lambda _name: "1.13.3")

    def factory(_model, _threads):
        if kind == "mutate_model":
            model.path.write_bytes(b"different model")
        elif kind == "replace_same_bytes":
            old = model.path.read_bytes()
            replacement = model.path.with_suffix(".replacement")
            replacement.write_bytes(old)
            replacement.replace(model.path)
        else:
            corpus.manifest.write_text(corpus.manifest.read_text() + " ")
        return FakeVad()

    with pytest.raises((subject.VadBenchmarkError, subject._helper().BenchmarkError)):
        subject.benchmark_cell(model, corpus, 1, 2, factory=factory)


def test_cpu_only_native_options_match_explicit_config_without_real_constructor(
    tmp_path, monkeypatch
):
    _corpus, model = inputs(tmp_path)
    settings = SimpleNamespace(silero_vad=SimpleNamespace())
    captured = {}

    def construct(config, **kwargs):
        captured.update(config=config, kwargs=kwargs)
        return FakeVad()

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(
            VadModelConfig=lambda: settings, VoiceActivityDetector=construct
        ),
    )
    monkeypatch.setattr(subject.metadata, "version", lambda _name: "1.13.3")
    subject._native(model, 2)
    assert settings.provider == "cpu" and settings.num_threads == 2
    assert settings.debug is False and settings.sample_rate == 16000
    assert captured["kwargs"] == {"buffer_size_in_seconds": 31.0}
    for key in (
        "window_size",
        "threshold",
        "min_speech_duration",
        "min_silence_duration",
        "max_speech_duration",
    ):
        assert getattr(settings.silero_vad, key) == subject.CONFIG[key]
    assert not hasattr(settings.silero_vad, "neg_threshold")
    assert subject.NEG_THRESHOLD_BINDING == "native-sdk-default-unsettable"


@pytest.mark.parametrize(
    "repeats,threads,timeout",
    [
        (True, 2, 30),
        (0, 2, 30),
        (1, True, 30),
        (1, 0, 30),
        (1, 2, float("inf")),
        (1, 2, 1801),
    ],
)
def test_limits_are_bounded_and_not_boolean(repeats, threads, timeout):
    with pytest.raises(subject.VadBenchmarkError):
        subject._limits(repeats, threads, timeout)


def test_public_model_size_symlink_and_name_limits(tmp_path):
    corpus, model = inputs(tmp_path)
    with pytest.raises(subject.VadBenchmarkError):
        subject._model_binding(subject.Model("private-id", model.path))
    alias = tmp_path / "alias.onnx"
    alias.symlink_to(model.path)
    with pytest.raises(subject._helper().BenchmarkError):
        subject._model_binding(subject.Model(model.id, alias))
    model.path.write_bytes(b"")
    with pytest.raises(subject._helper().BenchmarkError):
        subject._model_binding(model)


def test_fixed_argument_failure_never_echoes_supplied_private_arguments(capsys):
    with pytest.raises(SystemExit):
        subject.main(["--private-owner-name", "private words"])
    assert "private" not in capsys.readouterr().err


def test_nondefault_unsettable_negative_threshold_is_not_silently_ignored(
    tmp_path, monkeypatch
):
    _corpus, model = inputs(tmp_path)
    settings = SimpleNamespace(silero_vad=SimpleNamespace())
    called = []
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(
            VadModelConfig=lambda: settings,
            VoiceActivityDetector=lambda *a, **k: called.append(True),
        ),
    )
    monkeypatch.setattr(subject.metadata, "version", lambda _name: "1.13.3")
    monkeypatch.setitem(subject.CONFIG, "neg_threshold", 0.2)
    with pytest.raises(subject.VadBenchmarkError):
        subject._native(model, 2)
    assert not called


def test_unsettable_binding_and_unattested_exit_threshold_are_strict(
    tmp_path, monkeypatch
):
    corpus, model, _vad, result = complete(tmp_path, monkeypatch)
    assert result["neg_threshold_binding"] == "native-sdk-default-unsettable"
    assert result["native_exit_threshold_attested"] is False
    for field, value in (
        ("neg_threshold_binding", "explicit"),
        ("native_exit_threshold_attested", True),
    ):
        forged = copy.deepcopy(result)
        forged[field] = value
        with pytest.raises(subject.VadBenchmarkError):
            subject._validate_result(forged, model.id, corpus, 3, 2)
