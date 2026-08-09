from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from tools.streaming_stt.adapters.kyutai import (
    KyutaiAdapter,
    KyutaiAdapterError,
    _bounded_proc_ascii,
)
from tools.streaming_stt.manifest import KyutaiConfig
from tools.streaming_stt.protocol import (
    PcmInput,
    ResourceUsage,
    StreamConfig,
    TranscribeRequest,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 1.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        self.value += 0.001
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _Scalar:
    def __init__(self, value) -> None:
        self.value = value

    def item(self):
        return self.value


class _AudioTensor:
    def __init__(self, shape, *, values=None) -> None:
        self.shape = tuple(shape)
        self.values = values

    def view(self, *shape):
        resolved = list(shape)
        if resolved.count(-1) == 1:
            known = math.prod(value for value in resolved if value != -1)
            resolved[resolved.index(-1)] = math.prod(self.shape) // known
        assert math.prod(resolved) == math.prod(self.shape)
        return _AudioTensor(resolved, values=self.values)

    def __getitem__(self, key):
        batch, samples = key
        assert batch == slice(None)
        start, stop, step = samples.indices(self.shape[-1])
        assert step == 1
        return _AudioTensor((self.shape[0], stop - start))


class _TokenTensor:
    shape = (1, 1, 1)

    def __init__(self, token_id: int) -> None:
        self.token_id = token_id

    def __getitem__(self, key):
        assert key == (0, 0, 0)
        return _Scalar(self.token_id)


class _Head:
    def __init__(self, *, shape=(1, 1, 6), finite=True, value=1.0) -> None:
        self.shape = shape
        self.finite = finite
        self.value = value


class _Finite:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self):
        return _Scalar(self.value)


class _Device:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class _Context:
    def __init__(self, state, kind: str) -> None:
        self.state = state
        self.kind = kind
        self.identity = len(state["contexts"])
        state["contexts"].append((kind, self.identity))

    def __enter__(self):
        self.state["context_events"].append(("enter", self.kind, self.identity))
        return self

    def __exit__(self, _type, _value, _traceback):
        self.state["context_events"].append(("exit", self.kind, self.identity))


def _fake_api(*, head_fault: str | None = None, free_vram_mb: int = 12_000):
    state = {
        "events": [],
        "loader_calls": [],
        "tensor_frames": [],
        "resample_calls": [],
        "resampler_instances": [],
        "mimi_shapes": [],
        "mimi_tokens": [],
        "lm_audio_tokens": [],
        "lm_instances": [],
        "contexts": [],
        "context_events": [],
        "resample_hook": None,
    }
    _bfloat16 = object()
    _float32 = object()

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported():
            return True

        @staticmethod
        def mem_get_info(device):
            assert str(device) == "cuda:0"
            return free_vram_mb * 1024**2, 24_000 * 1024**2

        @staticmethod
        def set_per_process_memory_fraction(fraction, device):
            state["events"].append(("memory_fraction", fraction, str(device)))

        @staticmethod
        def synchronize(device):
            state["events"].append(("synchronize", str(device)))

        @staticmethod
        def empty_cache():
            state["events"].append(("empty_cache",))

    class Torch:
        __version__ = "2.7.1+cu126"
        version = SimpleNamespace(cuda="12.6")
        cuda = Cuda()
        bfloat16 = _bfloat16
        float32 = _float32
        threads = 4
        interop_threads = 4

        @staticmethod
        def device(value):
            return _Device(value)

        @classmethod
        def set_num_threads(cls, value):
            state["events"].append(("set_num_threads", value))
            cls.threads = value

        @classmethod
        def set_num_interop_threads(cls, value):
            state["events"].append(("set_num_interop_threads", value))
            cls.interop_threads = value

        @classmethod
        def get_num_threads(cls):
            return cls.threads

        @classmethod
        def get_num_interop_threads(cls):
            return cls.interop_threads

        @staticmethod
        def tensor(values, *, dtype, device):
            assert dtype is _float32
            assert str(device) == "cuda:0"
            captured = list(values)
            state["tensor_frames"].append(captured)
            return _AudioTensor((len(captured),), values=captured)

        @staticmethod
        def isfinite(head):
            return _Finite(head.finite)

        @staticmethod
        def inference_mode():
            return nullcontext()

    class _ResampleFrac:
        def __init__(self, old_sr, new_sr):
            self.old_sr = old_sr
            self.new_sr = new_sr
            state["resampler_instances"].append(self)

        def to(self, *, device, dtype):
            assert str(device) == "cuda:0"
            assert dtype is _float32
            return self

        def __call__(self, audio):
            assert audio.shape[0] == 1
            assert audio.shape[1] % 1_280 == 0
            if state["resample_hook"] is not None:
                state["resample_hook"]()
            state["resample_calls"].append((self.old_sr, self.new_sr, audio.shape))
            return _AudioTensor((1, audio.shape[1] * 3 // 2))

    class Julius:
        ResampleFrac = _ResampleFrac

    class Mimi:
        sample_rate = 24_000
        frame_size = 1_920
        frame_rate = 12.5

        def streaming(self, batch_size):
            assert batch_size == 1
            return _Context(state, "mimi")

        def encode(self, audio):
            state["mimi_shapes"].append(audio.shape)
            token = _AudioTensor((1, 32, 1))
            state["mimi_tokens"].append(token)
            return token

    class Tokenizer:
        @staticmethod
        def id_to_piece(token_id):
            assert token_id == 10
            return "▁hello"

    class Lm:
        @staticmethod
        def parameters():
            return iter(
                (
                    SimpleNamespace(
                        dtype=_bfloat16,
                        device="cuda:0",
                    ),
                )
            )

    class Checkpoint:
        raw_config = {
            "model_type": "stt",
            "n_q": 32,
            "dep_q": 0,
            "delays": [0] * 33,
            "extra_heads_num_heads": 4,
            "extra_heads_dim": 6,
            "existing_text_padding_id": 3,
            "lm_gen_config": {"temp": 0.0, "temp_text": 0.0},
        }
        stt_config = {
            "audio_delay_seconds": 0.5,
            "audio_silence_prefix_seconds": 0.0,
        }

        @staticmethod
        def get_mimi(*, device):
            assert device == "cuda:0"
            return Mimi()

        @staticmethod
        def get_text_tokenizer():
            return Tokenizer()

        @staticmethod
        def get_moshi(*, device, dtype):
            assert device == "cuda:0"
            assert dtype is _bfloat16
            return Lm()

    class CheckpointInfo:
        @staticmethod
        def from_hf_repo(repo_id, **kwargs):
            state["loader_calls"].append((repo_id, dict(kwargs)))
            return Checkpoint()

    class LMGen:
        def __init__(self, lm, **kwargs):
            assert isinstance(lm, Lm)
            self.kwargs = dict(kwargs)
            self.steps = 0
            state["lm_instances"].append(self)

        def streaming(self, batch_size):
            assert batch_size == 1
            return _Context(state, "lm")

        def step_with_extra_heads(self, audio_tokens):
            assert audio_tokens.shape == (1, 32, 1)
            state["lm_audio_tokens"].append(audio_tokens)
            self.steps += 1
            heads = [_Head() for _ in range(4)]
            if head_fault == "count":
                heads.pop()
            elif head_fault == "shape":
                heads[0] = _Head(shape=(1, 1, 5))
            elif head_fault == "finite":
                heads[2] = _Head(finite=False)
            return _TokenTensor(11 if self.steps == 1 else 10), heads

    api = SimpleNamespace(
        torch=Torch,
        julius=Julius,
        moshi_models=SimpleNamespace(
            loaders=SimpleNamespace(CheckpointInfo=CheckpointInfo),
            LMGen=LMGen,
        ),
        moshi_version="0.2.11",
        julius_version="0.2.8",
    )
    return api, state


def _paths(tmp_path: Path):
    paths = (
        tmp_path / "config.json",
        tmp_path / "model.safetensors",
        tmp_path / "mimi-pytorch-e351c8d8@125.safetensors",
        tmp_path / "tokenizer_en_fr_audio_8000.model",
    )
    for index, path in enumerate(paths, start=1):
        path.write_bytes(bytes([index]))
    return paths


def _request(
    tmp_path: Path,
    *,
    request_id: str = "case-r0",
    source_samples: int = 1_280,
    tail_samples: int = 16_000,
    chunk_samples: int = 1_280,
    partial_interval_ms: int = 160,
    pace: str = "burst",
) -> TranscribeRequest:
    payload = b"".join(struct.pack("<f", 0.25) for _ in range(source_samples))
    path = tmp_path / f"{request_id}.f32le"
    path.write_bytes(payload)
    return TranscribeRequest(
        request_id=request_id,
        pcm=PcmInput(
            path=path.resolve(),
            sha256=hashlib.sha256(payload).hexdigest(),
            samples=source_samples,
        ),
        stream=StreamConfig(
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=tail_samples,
        ),
    )


def _adapter(
    tmp_path,
    monkeypatch,
    *,
    head_fault=None,
    free_vram_mb=12_000,
    resample_cost=0.0,
):
    monkeypatch.setenv("NO_TORCH_COMPILE", "1")
    monkeypatch.setenv("NO_CUDA_GRAPH", "1")
    api, state = _fake_api(
        head_fault=head_fault,
        free_vram_mb=free_vram_mb,
    )
    clock = _Clock()
    state["resample_hook"] = lambda: clock.advance(resample_cost)
    adapter = KyutaiAdapter(
        *_paths(tmp_path),
        config=KyutaiConfig(),
        api=api,
        clock=clock,
        sleeper=clock.sleep,
        host_available_reader=lambda: 16 * 1024**3,
        resource_reader=lambda _torch, _device: ResourceUsage(
            rss_mb=512.0,
            threads=1,
            vram_mb=3_072.0,
        ),
    )
    return adapter, state


def test_true_moshi_stream_consumes_exact_tail_without_semantic_early_stop(
    tmp_path,
    monkeypatch,
):
    adapter, state = _adapter(tmp_path, monkeypatch)
    emitted = []

    first = adapter.transcribe(
        _request(tmp_path, request_id="case-r0"),
        emit_partial=lambda seq, partial: emitted.append((seq, partial)),
    )
    second = adapter.transcribe(
        _request(tmp_path, request_id="case-r1"),
        emit_partial=lambda _seq, _partial: None,
    )

    assert first.source_samples == first.source_samples_consumed == 1_280
    assert first.declared_tail_samples == first.tail_samples_consumed == 16_000
    assert first.chunks_yielded == 14
    assert first.semantic_head_steps == 15
    assert first.model_padding_samples == 640
    assert first.final == " ".join(["hello"] * 14)
    assert second.chunks_yielded == 14
    assert second.semantic_head_steps == 15
    assert len(state["resample_calls"]) == 2
    assert len(state["resampler_instances"]) == 1
    assert set(state["resample_calls"]) == {(16_000, 24_000, (1, 17_920))}
    assert set(state["mimi_shapes"]) == {(1, 1, 1_920)}
    assert len(state["mimi_tokens"]) == 28
    assert [instance.steps for instance in state["lm_instances"]] == [15, 15]
    assert len(state["lm_audio_tokens"]) == 30
    assert state["lm_audio_tokens"][0] is state["lm_audio_tokens"][1]
    assert state["lm_audio_tokens"][15] is state["lm_audio_tokens"][16]
    assert state["lm_audio_tokens"][1] is not state["lm_audio_tokens"][2]
    assert all(
        instance.kwargs == {"temp": 0.0, "temp_text": 0.0, "use_sampling": False}
        for instance in state["lm_instances"]
    )
    assert [kind for kind, _identity in state["contexts"]].count("mimi") == 2
    assert [kind for kind, _identity in state["contexts"]].count("lm") == 2
    assert len({identity for _kind, identity in state["contexts"]}) == 4
    assert emitted

    assert len(state["tensor_frames"]) == 2
    first_buffer = state["tensor_frames"][0]
    assert first_buffer[:1_280] == [0.25] * 1_280
    assert first_buffer[1_280:] == [0.0] * (16_000 + 640)
    assert first.compute_ms == pytest.approx(15.0)


def test_loader_receives_only_resolved_local_paths_and_resource_guards(
    tmp_path,
    monkeypatch,
):
    adapter, state = _adapter(tmp_path, monkeypatch)

    assert len(state["loader_calls"]) == 1
    repo_id, kwargs = state["loader_calls"][0]
    assert repo_id == "kyutai/stt-1b-en_fr-candle"
    assert set(kwargs) == {
        "moshi_weights",
        "mimi_weights",
        "tokenizer",
        "config_path",
    }
    assert all(
        isinstance(value, Path) and value.is_absolute() for value in kwargs.values()
    )
    assert ("set_num_threads", 1) in state["events"]
    assert ("set_num_interop_threads", 1) in state["events"]
    assert ("memory_fraction", 0.5, "cuda:0") in state["events"]
    assert adapter.ready.stream_geometry.input_chunk_samples == 1_280
    assert adapter.ready.stream_geometry.mimi_frame_samples == 1_920
    assert adapter.ready.stream_geometry.resampling_mode == "whole-buffer-noncausal"
    assert adapter.ready.resources.threads == 1


@pytest.mark.parametrize("fault", ["count", "shape", "finite"])
def test_semantic_heads_must_be_four_by_six_and_finite(
    tmp_path,
    monkeypatch,
    fault,
):
    adapter, _state = _adapter(tmp_path, monkeypatch, head_fault=fault)

    with pytest.raises(KyutaiAdapterError):
        adapter.transcribe(
            _request(tmp_path),
            emit_partial=lambda _seq, _partial: None,
        )


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"tail_samples": 15_999},
        {"chunk_samples": 640},
        {"partial_interval_ms": 500},
    ],
)
def test_stream_contract_rejects_nonexact_tail_geometry_or_partial_interval(
    tmp_path,
    monkeypatch,
    request_kwargs,
):
    adapter, state = _adapter(tmp_path, monkeypatch)

    with pytest.raises(KyutaiAdapterError):
        adapter.transcribe(
            _request(tmp_path, **request_kwargs),
            emit_partial=lambda _seq, _partial: None,
        )
    assert state["lm_instances"] == []


def test_load_fails_without_compile_prohibition_or_resource_floor(
    tmp_path,
    monkeypatch,
):
    api, _state = _fake_api()
    monkeypatch.delenv("NO_TORCH_COMPILE", raising=False)
    monkeypatch.setenv("NO_CUDA_GRAPH", "1")
    with pytest.raises(KyutaiAdapterError):
        KyutaiAdapter(
            *_paths(tmp_path),
            config=KyutaiConfig(),
            api=api,
            host_available_reader=lambda: 16 * 1024**3,
        )

    monkeypatch.setenv("NO_TORCH_COMPILE", "1")
    monkeypatch.setenv("NO_CUDA_GRAPH", "1")
    low_vram_api, _state = _fake_api(free_vram_mb=8_191)
    with pytest.raises(KyutaiAdapterError):
        KyutaiAdapter(
            *_paths(tmp_path),
            config=KyutaiConfig(),
            api=low_vram_api,
            host_available_reader=lambda: 16 * 1024**3,
        )


def test_proc_reader_is_bounded_ascii_regular_and_nofollow(tmp_path):
    status = tmp_path / "status"
    status.write_text("Threads:\t1\n", encoding="ascii")
    assert _bounded_proc_ascii(str(status), maximum_bytes=64) == "Threads:\t1\n"

    status.write_bytes(b"x" * 65)
    with pytest.raises(KyutaiAdapterError):
        _bounded_proc_ascii(str(status), maximum_bytes=64)

    status.write_text("Threads:\t1\n", encoding="ascii")
    alias = tmp_path / "status-link"
    alias.symlink_to(status)
    with pytest.raises(KyutaiAdapterError):
        _bounded_proc_ascii(str(alias), maximum_bytes=64)


def test_noncausal_resample_cost_counts_toward_realtime_backlog(
    tmp_path,
    monkeypatch,
):
    adapter, _state = _adapter(
        tmp_path,
        monkeypatch,
        resample_cost=0.2,
    )

    result = adapter.transcribe(
        _request(tmp_path, pace="realtime"),
        emit_partial=lambda _seq, _partial: None,
    )

    assert result.compute_ms == pytest.approx(215.0)
    assert result.deadline_misses >= 1
    assert result.max_backlog_ms >= 100.0
