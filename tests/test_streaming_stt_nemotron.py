from __future__ import annotations

from array import array
from contextlib import nullcontext
import hashlib
import importlib
from importlib.abc import MetaPathFinder
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tools.streaming_stt.manifest import ManifestError, NemotronConfig
from tools.streaming_stt.protocol import (
    PcmInput,
    StreamConfig,
    TranscribeRequest,
)


EXPECTED_PARAMETERS = 637_997_088


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


class FakeNumpy:
    float32 = "numpy-float32"

    def __init__(self) -> None:
        self.arrays: list[list[float]] = []

    def asarray(self, values, *, dtype):
        assert dtype == self.float32
        result = list(values)
        self.arrays.append(result)
        return result


class FakeDevice:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        device_count: int = 1,
        current_device: int = 0,
        capability: tuple[int, int] = (8, 9),
        allocated_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.available = available
        self.count = device_count
        self.current = current_device
        self.capability = capability
        self.allocated_bytes = allocated_bytes
        self.synchronizations: list[str] = []
        self.peak_resets: list[str] = []
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def current_device(self) -> int:
        return self.current

    def get_device_capability(self, device: FakeDevice) -> tuple[int, int]:
        assert str(device) == "cuda:0"
        return self.capability

    def synchronize(self, device: FakeDevice) -> None:
        self.synchronizations.append(str(device))

    def reset_peak_memory_stats(self, device: FakeDevice) -> None:
        self.peak_resets.append(str(device))

    def max_memory_allocated(self, device: FakeDevice) -> int:
        assert str(device) == "cuda:0"
        return self.allocated_bytes

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorch:
    __version__ = "2.12.1+cu126"
    float32 = "torch-float32"
    version = SimpleNamespace(cuda="12.6")

    def __init__(self, cuda: FakeCuda | None = None) -> None:
        self.cuda = cuda if cuda is not None else FakeCuda()

    @staticmethod
    def device(value: str) -> FakeDevice:
        return FakeDevice(value)

    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeFeatures:
    def __init__(self, frames: int, bins: int = 8) -> None:
        self.shape = (1, frames, bins)

    def __getitem__(self, key):
        batch, frames, bins = key
        assert batch == slice(None)
        assert bins == slice(None)
        start, stop, step = frames.indices(self.shape[1])
        return FakeFeatures(len(range(start, stop, step)), self.shape[2])


class FakeBatch(dict):
    def __init__(self, frames: int) -> None:
        super().__init__(
            input_features=FakeFeatures(frames),
            attention_mask="attention",
            prompt_ids="prompt",
            num_lookahead_tokens=3,
        )
        self.to_calls: list[dict[str, object]] = []

    @property
    def input_features(self) -> FakeFeatures:
        return self["input_features"]

    def to(self, **kwargs):
        self.to_calls.append(dict(kwargs))
        return self


class FakeFeatureExtractor:
    sampling_rate = 16_000
    hop_length = 160
    n_fft = 512
    win_length = 400


class FakeProcessor:
    supported_num_lookahead_tokens = [3, 0, 6, 13]
    prompt_dictionary = {"en-US": 0, "en-GB": 1, "ro-RO": 20}

    def __init__(
        self,
        clock: FakeClock,
        *,
        processor_cost: float = 0.0,
        provisional_texts: list[str] | None = None,
        final_text: str = "authoritative final",
        provisional_decode_cost: float = 0.0,
        final_decode_cost: float = 0.0,
    ) -> None:
        self.default_num_lookahead_tokens = 3
        self.feature_extractor = FakeFeatureExtractor()
        self.clock = clock
        self.processor_cost = processor_cost
        self.provisional_texts = list(provisional_texts or [])
        self.final_text = final_text
        self.provisional_decode_cost = provisional_decode_cost
        self.final_decode_cost = final_decode_cost
        self.calls: list[tuple[list[float], dict[str, object], FakeBatch]] = []
        self.lookaheads: list[int] = []
        self.decode_tokens: list[list[int]] = []
        self.batch_decode_sequences: list[object] = []

    @property
    def num_mel_frames_first_audio_chunk(self) -> int:
        return 1 + 8 * self.default_num_lookahead_tokens

    @property
    def num_mel_frames_per_audio_chunk(self) -> int:
        return 8 * (self.default_num_lookahead_tokens + 1)

    @property
    def num_samples_first_audio_chunk(self) -> int:
        return (
            self.num_mel_frames_first_audio_chunk - 1
        ) * self.feature_extractor.hop_length + self.feature_extractor.win_length // 2

    @property
    def num_samples_per_audio_chunk(self) -> int:
        return (
            self.num_mel_frames_per_audio_chunk * self.feature_extractor.hop_length
            + self.feature_extractor.win_length
        )

    @property
    def streaming_latency_ms(self) -> int:
        return 80 * (self.default_num_lookahead_tokens + 1)

    def set_num_lookahead_tokens(self, value: int) -> None:
        self.lookaheads.append(value)
        self.default_num_lookahead_tokens = value

    def __call__(self, audio, **kwargs) -> FakeBatch:
        self.clock.advance(self.processor_cost)
        frames = (
            self.num_mel_frames_first_audio_chunk + 1
            if kwargs["is_first_audio_chunk"]
            else self.num_mel_frames_per_audio_chunk
        )
        batch = FakeBatch(frames)
        self.calls.append((list(audio), dict(kwargs), batch))
        return batch

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.clock.advance(self.provisional_decode_cost)
        self.decode_tokens.append(list(token_ids))
        if self.provisional_texts:
            return self.provisional_texts.pop(0)
        return ""

    def batch_decode(self, sequences, *, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens is True
        self.clock.advance(self.final_decode_cost)
        self.batch_decode_sequences.append(sequences)
        return [self.final_text]


class FakeTensor:
    def __init__(self, count: int, *, device: FakeDevice | None = None) -> None:
        self.count = count
        self.device = device if device is not None else FakeDevice("cpu")

    def numel(self) -> int:
        return self.count


class FakeModel:
    def __init__(
        self,
        clock: FakeClock,
        *,
        token_batches: list[list[list[int]]] | None = None,
        per_chunk_cost: float = 0.0,
        final_cost: float = 0.0,
        fail_after_chunks: int | None = None,
        consume_all: bool = True,
        end_streamer: bool = True,
        parameter_count: int = EXPECTED_PARAMETERS,
    ) -> None:
        self.clock = clock
        self.token_batches = token_batches
        self.per_chunk_cost = per_chunk_cost
        self.final_cost = final_cost
        self.fail_after_chunks = fail_after_chunks
        self.consume_all = consume_all
        self.end_streamer = end_streamer
        self.training = True
        self.device = FakeDevice("cpu")
        self.dtype = None
        self.to_calls: list[dict[str, object]] = []
        self.eval_calls = 0
        self.generate_calls: list[dict[str, object]] = []
        self._parameters = [FakeTensor(parameter_count)]
        self._buffers = [FakeTensor(0)]
        self.config = SimpleNamespace(
            encoder_config=SimpleNamespace(supported_num_lookahead_tokens=[3, 0, 6, 13])
        )

    def to(self, **kwargs):
        self.to_calls.append(dict(kwargs))
        self.device = kwargs["device"]
        self.dtype = kwargs["dtype"]
        for tensor in (*self._parameters, *self._buffers):
            tensor.device = self.device
        return self

    def eval(self):
        self.eval_calls += 1
        self.training = False
        return self

    def parameters(self):
        return iter(self._parameters)

    def buffers(self):
        return iter(self._buffers)

    def generate(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        generator = kwargs["input_features"]
        streamer = kwargs["streamer"]
        batches = self.token_batches
        index = 0
        for _features in generator:
            self.clock.advance(self.per_chunk_cost)
            current = (
                batches[index]
                if batches is not None and index < len(batches)
                else [[index + 1]]
            )
            for tokens in current:
                streamer.put(tokens)
            index += 1
            if self.fail_after_chunks == index:
                raise RuntimeError("candidate detail must not escape")
            if not self.consume_all:
                break
        if self.end_streamer:
            streamer.end()
        self.clock.advance(self.final_cost)
        return SimpleNamespace(sequences=[[99, 100]])


class FakeLoader:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, root: str, **kwargs):
        self.calls.append((root, dict(kwargs)))
        return self.value


class FakeApi:
    def __init__(
        self,
        clock: FakeClock,
        *,
        processor: FakeProcessor | None = None,
        model: FakeModel | None = None,
        torch: FakeTorch | None = None,
    ) -> None:
        self.numpy = FakeNumpy()
        self.torch = torch if torch is not None else FakeTorch()
        self.processor = processor if processor is not None else FakeProcessor(clock)
        self.model = model if model is not None else FakeModel(clock)
        self.processor_loader = FakeLoader(self.processor)
        self.model_loader = FakeLoader(self.model)
        self.transformers = SimpleNamespace(
            __version__="5.13.1",
            Nemotron3_5AsrProcessor=self.processor_loader,
            Nemotron3_5AsrForRNNT=self.model_loader,
        )


class RejectCandidateImport(MetaPathFinder):
    def find_spec(self, fullname, _path, _target=None):
        roots = ("numpy", "torch", "transformers")
        if any(fullname == root or fullname.startswith(f"{root}.") for root in roots):
            raise AssertionError("candidate import during adapter module import")
        return None


def _config(
    *,
    lookahead_tokens: int = 3,
    language: str = "en-US",
) -> NemotronConfig:
    return NemotronConfig(
        python_version="3.12.3",
        transformers_version="5.13.1",
        torch_version="2.12.1+cu126",
        cuda_version="12.6",
        model_revision="f3d333391852ba876df169dcc9ba902d25b6ab0b",
        lookahead_tokens=lookahead_tokens,
        language=language,
        device="cuda:0",
        dtype="float32",
    )


def _request(
    tmp_path: Path,
    samples: list[float],
    *,
    chunk_samples: int = 5_120,
    pace: str = "burst",
    partial_interval_ms: int = 20,
    tail_padding_samples: int = 0,
) -> TranscribeRequest:
    raw = array("f", samples).tobytes()
    path = tmp_path / f"case-{len(samples)}-{tail_padding_samples}.f32le"
    path.write_bytes(raw)
    return TranscribeRequest(
        request_id="case-0000-r0",
        pcm=PcmInput(
            path=path.resolve(),
            sha256=hashlib.sha256(raw).hexdigest(),
            samples=len(samples),
        ),
        stream=StreamConfig(
            chunk_samples=chunk_samples,
            pace=pace,
            partial_interval_ms=partial_interval_ms,
            tail_padding_samples=tail_padding_samples,
        ),
    )


def _adapter(
    tmp_path: Path,
    api: FakeApi,
    clock: FakeClock,
    *,
    config: NemotronConfig | None = None,
):
    from tools.streaming_stt.adapters.nemotron import NemotronAdapter

    model_root = tmp_path / "model"
    model_root.mkdir(exist_ok=True)
    return NemotronAdapter(
        model_root,
        config=config if config is not None else _config(),
        api=api,
        clock=clock,
        sleeper=clock.sleep,
    )


@pytest.mark.parametrize(
    (
        "lookahead_tokens",
        "first_frames",
        "chunk_frames",
        "first_window",
        "window",
        "stride",
        "latency_ms",
    ),
    [
        (0, 1, 8, 200, 1_680, 1_280, 80),
        (3, 25, 32, 4_040, 5_520, 5_120, 320),
        (6, 49, 56, 7_880, 9_360, 8_960, 560),
        (13, 105, 112, 16_840, 18_320, 17_920, 1_120),
    ],
)
def test_config_binds_exact_native_geometry_for_each_supported_lookahead(
    lookahead_tokens,
    first_frames,
    chunk_frames,
    first_window,
    window,
    stride,
    latency_ms,
):
    config = _config(lookahead_tokens=lookahead_tokens)

    assert config.python_version == "3.12.3"
    assert config.transformers_version == "5.13.1"
    assert config.librosa_version == "0.11.0"
    assert config.native_sample_rate_hz == 16_000
    assert config.native_hop_length_samples == 160
    assert config.native_n_fft_samples == 512
    assert config.native_win_length_samples == 400
    assert config.native_first_chunk_frames == first_frames
    assert config.native_chunk_frames == chunk_frames
    assert config.native_first_window_samples == first_window
    assert config.native_window_samples == window
    assert config.native_stride_samples == stride
    assert config.streaming_latency_ms == latency_ms


def test_config_rejects_geometry_that_disagrees_with_lookahead():
    with pytest.raises(ManifestError):
        NemotronConfig(lookahead_tokens=3, native_stride_samples=5_119)


def test_import_is_inert_and_does_not_import_candidate_packages():
    before = set(sys.modules)
    reject = RejectCandidateImport()
    sys.meta_path.insert(0, reject)
    try:
        module = importlib.import_module("tools.streaming_stt.adapters.nemotron")
        importlib.reload(module)
    finally:
        sys.meta_path.remove(reject)

    added = set(sys.modules) - before
    assert not any(
        name.split(".", 1)[0] in {"librosa", "numpy", "torch", "transformers"}
        for name in added
    )


def test_candidate_api_requires_exact_distribution_versions_before_import(
    monkeypatch,
):
    from tools.streaming_stt.adapters import nemotron

    versions = {"transformers": "5.13.1", "librosa": "0.11.0"}
    modules = {
        "numpy": object(),
        "torch": object(),
        "transformers": SimpleNamespace(__version__="5.13.1"),
    }
    imports: list[str] = []

    monkeypatch.setattr(
        nemotron.metadata,
        "version",
        lambda distribution: versions[distribution],
    )

    def import_module(name):
        imports.append(name)
        return modules[name]

    monkeypatch.setattr(nemotron.importlib, "import_module", import_module)

    api = nemotron._candidate_api("5.13.1", "0.11.0")

    assert api.numpy is modules["numpy"]
    assert api.torch is modules["torch"]
    assert api.transformers is modules["transformers"]
    assert imports == ["numpy", "torch", "transformers"]


@pytest.mark.parametrize(
    "versions",
    [
        {"transformers": "5.13.0", "librosa": "0.11.0"},
        {"transformers": "5.13.1", "librosa": "0.10.2"},
    ],
)
def test_candidate_api_rejects_distribution_mismatch_before_import(
    monkeypatch,
    versions,
):
    from tools.streaming_stt.adapters import nemotron

    imports: list[str] = []
    monkeypatch.setattr(
        nemotron.metadata,
        "version",
        lambda distribution: versions[distribution],
    )
    monkeypatch.setattr(
        nemotron.importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    with pytest.raises(nemotron.NemotronAdapterError):
        nemotron._candidate_api("5.13.1", "0.11.0")

    assert imports == []


def test_candidate_api_rejects_missing_distribution_before_import(monkeypatch):
    from tools.streaming_stt.adapters import nemotron

    imports: list[str] = []

    def missing(_distribution):
        raise nemotron.metadata.PackageNotFoundError

    monkeypatch.setattr(nemotron.metadata, "version", missing)
    monkeypatch.setattr(
        nemotron.importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    with pytest.raises(nemotron.NemotronAdapterError):
        nemotron._candidate_api("5.13.1", "0.11.0")

    assert imports == []


@pytest.mark.parametrize("language", ["en-US", "en-GB", "ro-RO"])
def test_load_is_exact_local_native_cuda_and_language_prompted(
    tmp_path,
    language,
):
    clock = FakeClock()
    api = FakeApi(clock)
    adapter = _adapter(
        tmp_path,
        api,
        clock,
        config=_config(lookahead_tokens=6, language=language),
    )
    model_root = str((tmp_path / "model").resolve())

    assert api.processor_loader.calls == [
        (
            model_root,
            {
                "revision": "f3d333391852ba876df169dcc9ba902d25b6ab0b",
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]
    assert api.model_loader.calls == [
        (
            model_root,
            {
                "revision": "f3d333391852ba876df169dcc9ba902d25b6ab0b",
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
                "weights_only": True,
                "device_map": None,
                "dtype": FakeTorch.float32,
            },
        )
    ]
    assert api.model.to_calls == [
        {"device": api.model.device, "dtype": FakeTorch.float32}
    ]
    assert str(api.model.device) == "cuda:0"
    assert api.model.eval_calls == 1
    assert api.processor.lookaheads == [6]
    assert adapter.config.language == language
    assert adapter.ready.resources.vram_mb == 512.0
    assert adapter.ready.stream_geometry.as_dict() == {
        "sample_rate_hz": 16_000,
        "hop_length_samples": 160,
        "n_fft_samples": 512,
        "win_length_samples": 400,
        "first_chunk_frames": 49,
        "chunk_frames": 56,
        "first_window_samples": 7_880,
        "window_samples": 9_360,
        "stride_samples": 8_960,
        "streaming_latency_ms": 560,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "transformers-version",
        "torch-version",
        "cuda-version",
        "device-count",
        "capability",
        "parameter-count",
    ],
)
def test_load_fails_closed_on_runtime_or_model_identity_mismatch(
    tmp_path,
    mutation,
):
    from tools.streaming_stt.adapters.nemotron import (
        NemotronAdapter,
        NemotronAdapterError,
    )

    clock = FakeClock()
    cuda = FakeCuda()
    torch = FakeTorch(cuda)
    model = FakeModel(clock)
    api = FakeApi(clock, model=model, torch=torch)
    if mutation == "transformers-version":
        api.transformers.__version__ = "5.13.0"
    elif mutation == "torch-version":
        api.torch.__version__ = "2.12.0+cu126"
    elif mutation == "cuda-version":
        api.torch.version = SimpleNamespace(cuda="12.8")
    elif mutation == "device-count":
        cuda.count = 2
    elif mutation == "capability":
        cuda.capability = (9, 0)
    elif mutation == "parameter-count":
        model._parameters[0].count -= 1
    model_root = tmp_path / "model"
    model_root.mkdir()

    with pytest.raises(NemotronAdapterError) as caught:
        NemotronAdapter(
            model_root,
            config=_config(),
            api=api,
            clock=clock,
            sleeper=clock.sleep,
        )

    assert str(caught.value) == ""


def test_load_fails_closed_on_exact_python_version_mismatch(
    tmp_path,
    monkeypatch,
):
    from tools.streaming_stt.adapters import nemotron

    clock = FakeClock()
    api = FakeApi(clock)
    model_root = tmp_path / "model"
    model_root.mkdir()
    monkeypatch.setattr(nemotron.platform, "python_version", lambda: "3.12.4")

    with pytest.raises(nemotron.NemotronAdapterError):
        nemotron.NemotronAdapter(
            model_root,
            config=_config(),
            api=api,
            clock=clock,
            sleeper=clock.sleep,
        )


@pytest.mark.parametrize(
    "field",
    [
        "sampling_rate",
        "hop_length",
        "n_fft",
        "win_length",
        "num_mel_frames_first_audio_chunk",
        "num_mel_frames_per_audio_chunk",
        "num_samples_first_audio_chunk",
        "num_samples_per_audio_chunk",
        "streaming_latency_ms",
    ],
)
def test_load_fails_closed_when_processor_geometry_is_not_pinned(
    tmp_path,
    monkeypatch,
    field,
):
    from tools.streaming_stt.adapters.nemotron import (
        NemotronAdapter,
        NemotronAdapterError,
    )

    clock = FakeClock()
    processor = FakeProcessor(clock)
    api = FakeApi(clock, processor=processor)
    if hasattr(processor.feature_extractor, field):
        current = getattr(processor.feature_extractor, field)
        setattr(processor.feature_extractor, field, current + 1)
    else:
        current = getattr(processor, field)
        monkeypatch.setattr(
            FakeProcessor,
            field,
            property(lambda _self, value=current: value + 1),
        )
    model_root = tmp_path / "model"
    model_root.mkdir()

    with pytest.raises(NemotronAdapterError):
        NemotronAdapter(
            model_root,
            config=_config(),
            api=api,
            clock=clock,
            sleeper=clock.sleep,
        )


def test_request_chunk_size_must_equal_native_unique_stride(tmp_path):
    from tools.streaming_stt.adapters.nemotron import NemotronAdapterError

    clock = FakeClock()
    api = FakeApi(clock)
    adapter = _adapter(tmp_path, api, clock)

    with pytest.raises(NemotronAdapterError):
        adapter.transcribe(
            _request(tmp_path, [0.0] * 4_040, chunk_samples=5_119),
            emit_partial=lambda _seq, _partial: None,
        )

    assert api.model.generate_calls == []


@pytest.mark.parametrize(
    ("sample_count", "expected_lengths", "expected_padding"),
    [
        (4_000, [4_040], 40),
        (4_040, [4_040], 0),
        # The native subsequent chunk ending exactly at the source boundary
        # must be included (the upstream example's strict '<' drops it).
        (9_264, [4_040, 5_520], 0),
        (9_265, [4_040, 5_520, 5_520], 5_119),
    ],
)
def test_final_native_chunk_is_always_included_and_zero_padded(
    tmp_path,
    sample_count,
    expected_lengths,
    expected_padding,
):
    clock = FakeClock()
    api = FakeApi(clock)
    adapter = _adapter(tmp_path, api, clock)
    source = [float(index) for index in range(sample_count)]

    case = adapter.transcribe(
        _request(tmp_path, source),
        emit_partial=lambda _seq, _partial: None,
    )

    windows = [audio for audio, _kwargs, _batch in api.processor.calls]
    assert [len(window) for window in windows] == expected_lengths
    assert case.model_padding_samples == expected_padding
    assert windows[-1][-expected_padding:] == (
        [0.0] * expected_padding if expected_padding else windows[-1]
    )
    assert case.chunks_yielded == len(expected_lengths)
    if sample_count == 9_264:
        assert windows[1] == source[3_744:9_264]
    if sample_count == 9_265:
        assert windows[2][0] == source[8_864]
        assert windows[2][400] == source[9_264]


def test_declared_tail_is_source_progress_not_model_padding(tmp_path):
    clock = FakeClock()
    api = FakeApi(clock)
    adapter = _adapter(tmp_path, api, clock)

    case = adapter.transcribe(
        _request(
            tmp_path,
            [1.0] * 4_020,
            tail_padding_samples=20,
        ),
        emit_partial=lambda _seq, _partial: None,
    )

    assert len(api.processor.calls) == 1
    assert api.processor.calls[0][0][-20:] == [0.0] * 20
    assert case.model_padding_samples == 0


def test_zero_lookahead_left_pads_native_context_without_wrapping_source(tmp_path):
    clock = FakeClock()
    processor = FakeProcessor(clock)
    api = FakeApi(clock, processor=processor)
    adapter = _adapter(
        tmp_path,
        api,
        clock,
        config=_config(lookahead_tokens=0),
    )
    source = [float(index + 1) for index in range(201)]

    case = adapter.transcribe(
        _request(tmp_path, source, chunk_samples=1280),
        emit_partial=lambda _seq, _partial: None,
    )

    second = processor.calls[1][0]
    assert second[:96] == [0.0] * 96
    assert second[96] == source[0]
    assert second[296] == source[200]
    assert case.model_padding_samples == 1_383


def test_streamer_snapshots_are_provisional_and_final_sequence_is_authoritative(
    tmp_path,
):
    clock = FakeClock()
    processor = FakeProcessor(
        clock,
        provisional_texts=["streamer glitch", "streamer glitch revised"],
        final_text="goodness survives",
    )
    model = FakeModel(
        clock,
        token_batches=[[[1]], [[2]]],
    )
    api = FakeApi(clock, processor=processor, model=model)
    adapter = _adapter(tmp_path, api, clock)
    emitted = []

    case = adapter.transcribe(
        _request(tmp_path, [0.0] * 9_264),
        emit_partial=lambda seq, partial: emitted.append((seq, partial.text)),
    )

    assert emitted == [
        (0, "streamer glitch"),
        (1, "streamer glitch revised"),
    ]
    assert tuple(partial.text for partial in case.partials) == (
        "streamer glitch",
        "streamer glitch revised",
    )
    assert case.final == "goodness survives"
    assert processor.batch_decode_sequences == [[[99, 100]]]


def test_partial_interval_throttles_multiple_snapshots_from_same_chunk(tmp_path):
    clock = FakeClock()
    processor = FakeProcessor(
        clock,
        provisional_texts=["one", "one two", "one two three"],
    )
    model = FakeModel(
        clock,
        token_batches=[[[1], [2]], [[3]]],
    )
    api = FakeApi(clock, processor=processor, model=model)
    adapter = _adapter(tmp_path, api, clock)
    emitted = []

    case = adapter.transcribe(
        _request(tmp_path, [0.0] * 9_264, partial_interval_ms=20),
        emit_partial=lambda _seq, partial: emitted.append(
            (partial.after_samples, partial.text)
        ),
    )

    assert emitted == [(4_040, "one"), (9_264, "one two")]
    assert len(case.partials) == 2


def test_initial_blank_does_not_consume_the_chunks_partial_interval(tmp_path):
    clock = FakeClock()
    processor = FakeProcessor(
        clock,
        provisional_texts=["", "first words"],
    )
    model = FakeModel(
        clock,
        token_batches=[[[0], [1]]],
    )
    api = FakeApi(clock, processor=processor, model=model)
    adapter = _adapter(tmp_path, api, clock)
    emitted = []

    adapter.transcribe(
        _request(tmp_path, [0.0] * 4_040),
        emit_partial=lambda _seq, partial: emitted.append(partial.text),
    )

    assert emitted == ["first words"]
    assert processor.decode_tokens == [[0], [0, 1]]


def test_candidate_failure_is_detail_free_and_poisons_adapter(tmp_path):
    from tools.streaming_stt.adapters.nemotron import NemotronAdapterError

    clock = FakeClock()
    api = FakeApi(
        clock,
        model=FakeModel(clock, fail_after_chunks=1),
    )
    adapter = _adapter(tmp_path, api, clock)
    request = _request(tmp_path, [0.0] * 9_264)

    with pytest.raises(NemotronAdapterError) as first:
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)
    with pytest.raises(NemotronAdapterError) as second:
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)

    assert str(first.value) == ""
    assert str(second.value) == ""


def test_generate_must_consume_every_final_tail_chunk(tmp_path):
    from tools.streaming_stt.adapters.nemotron import NemotronAdapterError

    clock = FakeClock()
    api = FakeApi(
        clock,
        model=FakeModel(clock, consume_all=False),
    )
    adapter = _adapter(tmp_path, api, clock)

    with pytest.raises(NemotronAdapterError):
        adapter.transcribe(
            _request(tmp_path, [0.0] * 9_264),
            emit_partial=lambda _seq, _partial: None,
        )


def test_generate_must_close_the_synchronous_streamer(tmp_path):
    from tools.streaming_stt.adapters.nemotron import NemotronAdapterError

    clock = FakeClock()
    api = FakeApi(
        clock,
        model=FakeModel(clock, end_streamer=False),
    )
    adapter = _adapter(tmp_path, api, clock)

    with pytest.raises(NemotronAdapterError):
        adapter.transcribe(
            _request(tmp_path, [0.0] * 4_040),
            emit_partial=lambda _seq, _partial: None,
        )


def test_realtime_pacing_compute_finalization_and_vram_are_truthful(tmp_path):
    clock = FakeClock()
    processor = FakeProcessor(
        clock,
        processor_cost=0.010,
        provisional_texts=["one", "two", "three"],
        provisional_decode_cost=0.003,
        final_decode_cost=0.004,
    )
    model = FakeModel(
        clock,
        per_chunk_cost=0.005,
        final_cost=0.007,
    )
    api = FakeApi(clock, processor=processor, model=model)
    adapter = _adapter(tmp_path, api, clock)

    case = adapter.transcribe(
        _request(tmp_path, [0.0] * 14_384, pace="realtime"),
        emit_partial=lambda _seq, _partial: None,
    )

    assert clock.sleeps == pytest.approx([0.2345, 0.3085])
    assert case.compute_ms == pytest.approx(65.0)
    assert case.elapsed_ms == pytest.approx(608.0)
    assert case.finalization_ms == pytest.approx(11.0)
    assert case.deadline_misses == 0
    assert case.max_backlog_ms == 0.0
    assert case.model_padding_samples == 0
    assert case.chunks_yielded == 3
    assert case.resources.vram_mb == 512.0
    assert api.torch.cuda.peak_resets == ["cuda:0"]


def test_realtime_post_consumption_accounts_every_chunk_including_final(tmp_path):
    clock = FakeClock()
    processor = FakeProcessor(
        clock,
        processor_cost=0.010,
        provisional_decode_cost=0.003,
    )
    model = FakeModel(clock, per_chunk_cost=0.400)
    api = FakeApi(clock, processor=processor, model=model)
    adapter = _adapter(tmp_path, api, clock)

    case = adapter.transcribe(
        _request(tmp_path, [0.0] * 14_384, pace="realtime"),
        emit_partial=lambda _seq, _partial: None,
    )

    assert clock.sleeps == []
    assert case.deadline_misses == 3
    assert case.max_backlog_ms == pytest.approx(340.0)
    assert case.chunks_yielded == 3


def test_synchronous_streamer_rejects_batch_mismatch_without_thread_or_queue():
    from tools.streaming_stt.adapters.nemotron import (
        NemotronAdapterError,
        _SynchronousSnapshotStreamer,
    )

    streamer = _SynchronousSnapshotStreamer(
        decode=lambda _tokens: "",
        should_decode=lambda: True,
        publish=lambda _text, _decode_ms: None,
        clock=lambda: 0.0,
    )

    with pytest.raises(NemotronAdapterError):
        streamer.put([1, 2])


def test_close_is_idempotent_and_releases_cuda_allocator(tmp_path):
    from tools.streaming_stt.adapters.nemotron import NemotronAdapterError

    clock = FakeClock()
    api = FakeApi(clock)
    adapter = _adapter(tmp_path, api, clock)
    adapter.close()
    adapter.close()

    assert api.torch.cuda.empty_cache_calls == 1
    with pytest.raises(NemotronAdapterError):
        adapter.transcribe(
            _request(tmp_path, [0.0] * 4_040),
            emit_partial=lambda _seq, _partial: None,
        )
