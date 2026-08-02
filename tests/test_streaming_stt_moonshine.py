from __future__ import annotations

from array import array
from dataclasses import dataclass, field
import hashlib
import importlib
from importlib.abc import MetaPathFinder
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tools.streaming_stt.manifest import MoonshineConfig
from tools.streaming_stt.protocol import (
    MAX_PARTIALS_PER_CASE,
    PcmInput,
    StreamConfig,
    TranscribeRequest,
)


@dataclass(frozen=True)
class Line:
    line_id: int
    start_time: float
    text: str


@dataclass(frozen=True)
class Transcript:
    lines: list[Line]


@dataclass(frozen=True)
class Error:
    error: Exception


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


@dataclass
class StreamPlan:
    updates: list[Transcript] = field(default_factory=list)
    stop_transcript: Transcript | None = field(default_factory=lambda: Transcript([]))
    create_cost: float = 0.0
    start_cost: float = 0.0
    add_cost: float = 0.0
    update_cost: float = 0.0
    stop_cost: float = 0.0
    stop_error_event: bool = False
    close_error: bool = False
    native_cache_without_force: bool = False


class FakeStream:
    def __init__(self, clock: FakeClock, plan: StreamPlan) -> None:
        self.clock = clock
        self.plan = plan
        self.listeners: list[object] = []
        self.added: list[tuple[list[float], int]] = []
        self.update_flags: list[int] = []
        self.actions: list[str] = []
        self.closed = False
        self.cached_transcript = Transcript([])

    def add_listener(self, listener) -> None:
        self.listeners.append(listener)

    def start(self) -> None:
        self.actions.append("start")
        self.clock.advance(self.plan.start_cost)

    def add_audio(self, audio: list[float], sample_rate: int) -> None:
        self.actions.append("add")
        self.added.append((list(audio), sample_rate))
        self.clock.advance(self.plan.add_cost)

    def update_transcription(self, flags: int = 0) -> Transcript:
        self.actions.append(f"update:{flags}")
        self.update_flags.append(flags)
        self.clock.advance(self.plan.update_cost)
        if self.plan.native_cache_without_force and flags == 0:
            return self.cached_transcript
        if self.plan.updates:
            self.cached_transcript = self.plan.updates.pop(0)
        return self.cached_transcript

    def stop(self) -> Transcript | None:
        self.actions.append("stop")
        self.clock.advance(self.plan.stop_cost)
        if self.plan.stop_error_event:
            event = Error(RuntimeError("swallowed"))
            for listener in self.listeners:
                listener(event)
            return None
        return self.plan.stop_transcript

    def close(self) -> None:
        self.actions.append("close")
        self.closed = True
        if self.plan.close_error:
            raise RuntimeError("close")


class FakeTranscriber:
    def __init__(
        self,
        api: FakeApi,
        *,
        model_path: str,
        model_arch: int,
        update_interval: float,
        options: dict[str, str],
    ) -> None:
        self.api = api
        self.model_path = model_path
        self.model_arch = model_arch
        self.update_interval = update_interval
        self.options = options
        self.streams: list[FakeStream] = []
        self.closed = False
        api.clock.advance(api.model_load_cost)

    def get_version(self) -> int:
        return self.api.library_version

    def create_stream(self, *, update_interval: float) -> FakeStream:
        plan = self.api.plans.pop(0)
        self.api.clock.advance(plan.create_cost)
        stream = FakeStream(self.api.clock, plan)
        self.streams.append(stream)
        return stream

    def close(self) -> None:
        self.closed = True


class FakeTranscriberFactory:
    MOONSHINE_HEADER_VERSION = 30000
    MOONSHINE_FLAG_FORCE_UPDATE = 1

    def __init__(self, api: FakeApi) -> None:
        self.api = api
        self.instances: list[FakeTranscriber] = []

    def __call__(self, **kwargs) -> FakeTranscriber:
        instance = FakeTranscriber(self.api, **kwargs)
        self.instances.append(instance)
        return instance


class FakeApi:
    __version__ = "0.1.0"
    ModelArch = SimpleNamespace(
        TINY_STREAMING=2,
        SMALL_STREAMING=4,
        MEDIUM_STREAMING=5,
    )

    def __init__(
        self,
        clock: FakeClock,
        plans: list[StreamPlan],
        *,
        model_load_cost: float = 0.0,
        library_version: int = 30000,
    ) -> None:
        self.clock = clock
        self.plans = list(plans)
        self.model_load_cost = model_load_cost
        self.library_version = library_version
        self.Transcriber = FakeTranscriberFactory(self)


class RejectCandidateImport(MetaPathFinder):
    def find_spec(self, fullname, _path, _target=None):
        if fullname == "moonshine_voice" or fullname.startswith("moonshine_voice."):
            raise AssertionError("candidate import during adapter module import")
        return None


def _config(*, model_arch: str = "tiny-streaming") -> MoonshineConfig:
    return MoonshineConfig(
        package_version="0.1.0",
        api_version=30000,
        model_arch=model_arch,
        provider="cpu",
        language="en",
    )


def _request(
    tmp_path: Path,
    samples: list[float],
    *,
    chunk_samples: int,
    partial_interval_ms: int = 5000,
    tail_padding_samples: int = 0,
    pace: str = "burst",
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
    model_arch: str = "tiny-streaming",
):
    from tools.streaming_stt.adapters.moonshine import MoonshineAdapter

    model_root = tmp_path / "model"
    model_root.mkdir(exist_ok=True)
    return MoonshineAdapter(
        model_root,
        config=_config(model_arch=model_arch),
        api=api,
        clock=clock,
        sleeper=clock.sleep,
    )


def test_import_is_inert_and_does_not_import_candidate_package(monkeypatch):
    for name in tuple(sys.modules):
        if name == "moonshine_voice" or name.startswith("moonshine_voice."):
            monkeypatch.delitem(sys.modules, name)
    reject = RejectCandidateImport()
    sys.meta_path.insert(0, reject)

    try:
        module = importlib.import_module("tools.streaming_stt.adapters.moonshine")
        importlib.reload(module)
    finally:
        sys.meta_path.remove(reject)

    assert not any(
        name == "moonshine_voice" or name.startswith("moonshine_voice.")
        for name in sys.modules
    )


def test_exact_chunks_append_only_declared_zero_tail(tmp_path):
    clock = FakeClock()
    plan = StreamPlan(
        updates=[Transcript([Line(1, 0.0, "tail preserved")])],
        stop_transcript=Transcript([Line(1, 0.0, "tail preserved")]),
    )
    api = FakeApi(clock, [plan])
    adapter = _adapter(tmp_path, api, clock)
    request = _request(
        tmp_path,
        [1.0, 2.0, 3.0, 4.0, 5.0],
        chunk_samples=4,
        tail_padding_samples=3,
    )
    emitted = []

    case = adapter.transcribe(
        request, emit_partial=lambda seq, part: emitted.append((seq, part))
    )

    stream = api.Transcriber.instances[0].streams[0]
    assert stream.added == [
        ([1.0, 2.0, 3.0, 4.0], 16_000),
        ([5.0, 0.0, 0.0, 0.0], 16_000),
    ]
    assert case.final == "tail preserved"
    assert case.partials == ()
    assert emitted == []


@pytest.mark.parametrize(
    ("model_arch", "expected_arch"),
    [
        ("tiny-streaming", 2),
        ("small-streaming", 4),
        ("medium-streaming", 5),
    ],
)
def test_transcriber_uses_only_closed_cpu_low_metadata_options(
    tmp_path,
    model_arch,
    expected_arch,
):
    clock = FakeClock()
    api = FakeApi(clock, [StreamPlan(updates=[Transcript([])])])
    adapter = _adapter(
        tmp_path,
        api,
        clock,
        model_arch=model_arch,
    )

    transcriber = api.Transcriber.instances[0]
    assert transcriber.options == {
        "return_audio_data": "false",
        "identify_speakers": "false",
        "word_timestamps": "false",
        "ort_providers": "CPU",
    }
    assert transcriber.model_arch == expected_arch
    assert adapter.config.provider == "cpu"


def test_exact_package_and_library_api_versions_are_enforced(tmp_path):
    from tools.streaming_stt.adapters.moonshine import (
        MoonshineAdapter,
        MoonshineAdapterError,
    )

    model_root = tmp_path / "model"
    model_root.mkdir()
    clock = FakeClock()
    wrong_package = FakeApi(clock, [])
    wrong_package.__version__ = "0.0.69"

    with pytest.raises(MoonshineAdapterError):
        MoonshineAdapter(
            model_root,
            config=_config(),
            api=wrong_package,
            clock=clock,
            sleeper=clock.sleep,
        )

    wrong_library = FakeApi(clock, [], library_version=29999)
    with pytest.raises(MoonshineAdapterError):
        MoonshineAdapter(
            model_root,
            config=_config(),
            api=wrong_library,
            clock=clock,
            sleeper=clock.sleep,
        )
    assert wrong_library.Transcriber.instances[0].closed


def test_multiline_snapshots_are_ordered_revised_and_deduplicated(tmp_path):
    clock = FakeClock()
    first = Transcript(
        [
            Line(20, 1.0, "second"),
            Line(10, 0.0, "first"),
        ]
    )
    revised = Transcript([Line(20, 1.0, "second revised")])
    plan = StreamPlan(
        updates=[first, first, revised, revised, revised],
        stop_transcript=revised,
    )
    api = FakeApi(clock, [plan])
    adapter = _adapter(tmp_path, api, clock)
    request = _request(
        tmp_path,
        [0.1] * 64,
        chunk_samples=16,
        partial_interval_ms=1,
    )
    emitted = []

    case = adapter.transcribe(
        request,
        emit_partial=lambda _seq, partial: emitted.append(partial.text),
    )

    assert emitted == [
        "first\nsecond",
        "first\nsecond revised",
    ]
    assert case.final == "first\nsecond revised"
    assert len(case.partials) == 2


def test_scheduled_updates_force_past_the_native_transcription_cache(tmp_path):
    clock = FakeClock()
    first = Transcript([Line(1, 0.0, "first")])
    second = Transcript([Line(1, 0.0, "second")])
    plan = StreamPlan(
        updates=[first, second],
        stop_transcript=second,
        native_cache_without_force=True,
    )
    api = FakeApi(clock, [plan])
    adapter = _adapter(tmp_path, api, clock)
    request = _request(
        tmp_path,
        [0.1] * 3200,
        chunk_samples=1600,
        partial_interval_ms=100,
    )
    emitted = []

    case = adapter.transcribe(
        request,
        emit_partial=lambda _seq, partial: emitted.append(partial.text),
    )

    stream = api.Transcriber.instances[0].streams[0]
    assert stream.update_flags == [1, 1, 1]
    assert emitted == ["first", "second"]
    assert tuple(partial.text for partial in case.partials) == ("first", "second")
    assert case.final == "second"


def test_finalization_snapshots_never_become_online_partials(tmp_path):
    clock = FakeClock()
    forced = Transcript([Line(1, 0.0, "forced draft")])
    stopped = Transcript([Line(1, 0.0, "stop revision")])
    plan = StreamPlan(
        updates=[forced],
        stop_transcript=stopped,
    )
    api = FakeApi(clock, [plan])
    adapter = _adapter(tmp_path, api, clock)
    request = _request(tmp_path, [0.1] * 32, chunk_samples=16)
    emitted = []

    case = adapter.transcribe(
        request,
        emit_partial=lambda seq, partial: emitted.append((seq, partial.text)),
    )

    assert emitted == []
    assert case.partials == ()
    assert case.final == "stop revision"


def test_subinterval_tail_is_force_updated_while_active_before_stop(tmp_path):
    clock = FakeClock()
    forced = Transcript([Line(1, 0.0, "last tail")])
    plan = StreamPlan(
        updates=[forced],
        stop_transcript=forced,
    )
    api = FakeApi(clock, [plan])
    adapter = _adapter(tmp_path, api, clock)
    request = _request(
        tmp_path,
        [0.25] * 10,
        chunk_samples=8,
        tail_padding_samples=2,
    )

    adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)

    stream = api.Transcriber.instances[0].streams[0]
    assert stream.actions == [
        "start",
        "add",
        "add",
        "update:1",
        "stop",
        "close",
    ]
    assert stream.update_flags == [1]
    assert stream.added[-1][0] == [0.25, 0.25, 0.0, 0.0]


@pytest.mark.parametrize("emit_error", [False, True])
def test_none_or_swallowed_stop_error_fails_and_closes_stream(
    tmp_path,
    emit_error,
):
    from tools.streaming_stt.adapters.moonshine import MoonshineAdapterError

    clock = FakeClock()
    plan = StreamPlan(
        updates=[Transcript([])],
        stop_transcript=None,
        stop_error_event=emit_error,
    )
    api = FakeApi(clock, [plan])
    adapter = _adapter(tmp_path, api, clock)
    request = _request(tmp_path, [0.0] * 32, chunk_samples=16)

    with pytest.raises(MoonshineAdapterError):
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)

    assert api.Transcriber.instances[0].streams[0].closed


def test_one_transcriber_owns_fresh_streams_and_closes_idempotently(tmp_path):
    clock = FakeClock()
    plans = [
        StreamPlan(updates=[Transcript([])]),
        StreamPlan(updates=[Transcript([])]),
    ]
    api = FakeApi(clock, plans)
    adapter = _adapter(tmp_path, api, clock)
    request = _request(tmp_path, [0.0] * 32, chunk_samples=16)

    adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)
    adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)

    assert len(api.Transcriber.instances) == 1
    transcriber = api.Transcriber.instances[0]
    assert len(transcriber.streams) == 2
    assert all(stream.closed for stream in transcriber.streams)
    assert not transcriber.closed
    adapter.close()
    adapter.close()
    assert transcriber.closed


def test_realtime_pacing_and_timing_metrics_exclude_sleep_from_compute(tmp_path):
    clock = FakeClock()
    transcript = Transcript([Line(1, 0.0, "done")])
    plan = StreamPlan(
        updates=[transcript],
        stop_transcript=transcript,
        create_cost=0.01,
        start_cost=0.01,
        add_cost=0.02,
        update_cost=0.03,
        stop_cost=0.04,
    )
    api = FakeApi(clock, [plan])
    adapter = _adapter(tmp_path, api, clock)
    request = _request(
        tmp_path,
        [0.0] * 3200,
        chunk_samples=1600,
        pace="realtime",
    )

    case = adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)

    assert clock.sleeps == pytest.approx([0.08])
    assert case.compute_ms == pytest.approx(130.0)
    assert case.elapsed_ms == pytest.approx(210.0)
    assert case.finalization_ms == pytest.approx(70.0)
    assert case.deadline_misses == 0
    assert case.max_backlog_ms == 0.0
    assert case.resources.vram_mb is None


def test_realtime_slow_decode_reports_deadline_misses_and_peak_backlog(tmp_path):
    clock = FakeClock()
    plan = StreamPlan(
        updates=[Transcript([])],
        add_cost=0.15,
    )
    api = FakeApi(clock, [plan])
    adapter = _adapter(tmp_path, api, clock)
    request = _request(
        tmp_path,
        [0.0] * 3200,
        chunk_samples=1600,
        pace="realtime",
    )

    case = adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)

    assert clock.sleeps == []
    assert case.deadline_misses == 2
    assert case.max_backlog_ms == pytest.approx(100.0)


def test_distinct_partial_output_is_bounded_but_final_keeps_advancing(tmp_path):
    clock = FakeClock()
    count = MAX_PARTIALS_PER_CASE + 1
    updates = [Transcript([Line(1, 0.0, f"value {index}")]) for index in range(count)]
    final = updates[-1]
    plan = StreamPlan(
        updates=[*updates, final],
        stop_transcript=final,
    )
    api = FakeApi(clock, [plan])
    adapter = _adapter(tmp_path, api, clock)
    request = _request(
        tmp_path,
        [0.0] * (count * 16),
        chunk_samples=16,
        partial_interval_ms=1,
    )
    emitted = []

    case = adapter.transcribe(
        request,
        emit_partial=lambda seq, partial: emitted.append((seq, partial.text)),
    )

    assert len(emitted) == MAX_PARTIALS_PER_CASE
    assert [seq for seq, _text in emitted] == list(range(MAX_PARTIALS_PER_CASE))
    assert len(case.partials) == MAX_PARTIALS_PER_CASE
    assert case.final == f"value {count - 1}"


def test_stream_close_failure_is_not_silently_accepted(tmp_path):
    from tools.streaming_stt.adapters.moonshine import MoonshineAdapterError

    clock = FakeClock()
    plan = StreamPlan(
        updates=[Transcript([])],
        close_error=True,
    )
    api = FakeApi(clock, [plan])
    adapter = _adapter(tmp_path, api, clock)
    request = _request(tmp_path, [0.0] * 32, chunk_samples=16)

    with pytest.raises(MoonshineAdapterError):
        adapter.transcribe(request, emit_partial=lambda _seq, _partial: None)

    assert api.Transcriber.instances[0].streams[0].closed
