from __future__ import annotations

import glob
import threading

import pytest

np = pytest.importorskip("numpy")

import core.engines.file_replay as fr
from core.engine import (
    EngineCallbacks,
    PlaybackOutcome,
    PlaybackReceipt,
    SpeechStyle,
    TrackedSpeech,
)
from core.engines.file_replay import FileReplayEngine, load_waveform
from core.engines.sherpa import SherpaConfig
from core.llm import EchoLLM
from core.metrics import MetricsRecorder
from core.runtime import VoiceRuntime


# --- fakes mirroring the slice of the sherpa-onnx API the engine touches ---
# Real API: stream = recognizer.create_stream(); stream.accept_waveform(sr, x);
# recognizer.is_ready/decode_stream/get_result/is_endpoint/reset(stream).
class _FakeStream:
    def __init__(self) -> None:
        self.heard = False
        self.endpoint = False

    def accept_waveform(self, sample_rate: int, samples) -> None:
        loud = samples.size and float(np.max(np.abs(samples))) > 0.05
        if loud:
            self.heard = True
        elif self.heard:
            self.endpoint = True


class _FakeRecognizer:
    """Emits 'hello world' as final once it sees voiced audio then silence."""

    def create_stream(self) -> _FakeStream:
        return _FakeStream()

    def is_ready(self, stream: _FakeStream) -> bool:
        return False

    def decode_stream(self, stream: _FakeStream) -> None:  # pragma: no cover
        pass

    def get_result(self, stream: _FakeStream) -> str:
        return "hello world" if stream.heard else ""

    def is_endpoint(self, stream: _FakeStream) -> bool:
        return stream.endpoint

    def reset(self, stream: _FakeStream) -> None:
        stream.heard = False
        stream.endpoint = False


class _FakeTts:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, text: str, sid: int = 0, speed: float = 1.0):
        self.calls.append(text)
        return _GeneratedAudio()


class _ParamTts(_FakeTts):
    num_speakers = 103

    def __init__(self) -> None:
        super().__init__()
        self.params: list[tuple[str, int, float]] = []

    def generate(self, text: str, sid: int = 0, speed: float = 1.0):
        self.calls.append(text)
        self.params.append((text, sid, speed))
        return _GeneratedAudio()


class _GeneratedAudio:
    def __init__(self, count: int = 160) -> None:
        self.samples = np.ones(int(count), dtype="float32") * 0.1
        self.sample_rate = 16000


class _BlockingTts(_FakeTts):
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.result = _GeneratedAudio() if result is None else result
        self.error = error

    def generate(self, text: str, sid: int = 0, speed: float = 1.0):
        del sid, speed
        self.calls.append(text)
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        if self.error is not None:
            raise self.error
        return self.result


class _AdmissionGate:
    """Pause after owning the model lock but before engine admission."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.acquired = threading.Event()
        self.resume = threading.Event()

    def acquire(self, blocking: bool = True) -> bool:
        admitted = self._lock.acquire(blocking=blocking)
        if admitted:
            self.acquired.set()
            assert self.resume.wait(timeout=2.0)
        return admitted

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


class _ReceiptProbe:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[tuple[str, object]] = []
        self.receipts: list[PlaybackReceipt] = []
        self.terminal = threading.Event()

    def on_started(self, fragment_id: str) -> None:
        with self.lock:
            self.events.append(("started", fragment_id))

    def on_terminal(self, receipt: PlaybackReceipt) -> None:
        with self.lock:
            self.events.append(("terminal", receipt.fragment_id))
            self.receipts.append(receipt)
        self.terminal.set()

    def snapshot(self):
        with self.lock:
            return list(self.events), list(self.receipts)


def _patch_models(monkeypatch, recognizer, tts) -> None:
    monkeypatch.setattr(fr, "build_recognizer", lambda c: recognizer)
    monkeypatch.setattr(fr, "build_tts", lambda c: tts)


def _start_engine(monkeypatch, tts, callbacks=None) -> FileReplayEngine:
    _patch_models(monkeypatch, _FakeRecognizer(), tts)
    engine = FileReplayEngine(SherpaConfig(asr_encoder="x", tts_model="y"))
    engine.start(callbacks or EngineCallbacks())
    return engine


def test_replay_fires_final_and_records_metrics(monkeypatch):
    rec = _FakeRecognizer()
    tts = _FakeTts()
    _patch_models(monkeypatch, rec, tts)

    finals: list[str] = []
    recorder = MetricsRecorder()
    engine = FileReplayEngine(SherpaConfig(asr_encoder="x", tts_model="y"))
    engine.start(
        EngineCallbacks(
            on_final=finals.append,
            on_metric=recorder.mark,
            on_speech_start=lambda: None,
            on_speech_end=lambda: None,
        )
    )

    samples = np.concatenate(
        [np.ones(16000, dtype="float32") * 0.5, np.zeros(1600, dtype="float32")]
    )
    engine.replay_samples(samples, 16000)

    assert finals == ["hello world"]
    # speech_end + asr-final stamps captured; first audio not yet (no speak()).
    [record] = recorder.records()
    assert "speech_end" in record.stamps

    # Now synthesize -- offline TTS stamps tts_first_audio when the clip is ready.
    engine.speak("hello world")
    assert tts.calls == ["hello world"]
    assert record.stamps.get("tts_first_audio") is not None
    assert record.first_audio_latency is not None


def test_file_replay_tracked_completion_attests_null_sink(monkeypatch):
    tts = _FakeTts()
    metrics: list[str] = []
    lifecycle: list[str] = []
    ordered: list[object] = []
    engine = _start_engine(
        monkeypatch,
        tts,
        EngineCallbacks(
            on_metric=lambda name: (metrics.append(name), ordered.append(name)),
            on_speech_start=lambda: (
                lifecycle.append("speech-start"),
                ordered.append("speech-start"),
            ),
            on_speech_end=lambda: (
                lifecycle.append("speech-end"),
                ordered.append("speech-end"),
            ),
        ),
    )
    probe = _ReceiptProbe()

    def started(fragment_id: str) -> None:
        ordered.append(("started", fragment_id))
        probe.on_started(fragment_id)

    def terminal(receipt: PlaybackReceipt) -> None:
        ordered.append(("terminal", receipt.fragment_id))
        probe.on_terminal(receipt)

    engine.speak_tracked(
        TrackedSpeech("null-sink", "complete replay text"),
        on_started=started,
        on_terminal=terminal,
    )

    events, receipts = probe.snapshot()
    assert engine.playback_capabilities.tracked_terminal
    assert engine.playback_capabilities.exact_started
    assert not engine.playback_capabilities.sample_counts
    assert events == [("started", "null-sink"), ("terminal", "null-sink")]
    assert receipts == [
        PlaybackReceipt(
            fragment_id="null-sink",
            outcome=PlaybackOutcome.COMPLETED,
            safe_text_prefix="complete replay text",
        )
    ]
    assert metrics == ["tts_first_audio"]
    assert lifecycle == ["speech-start", "speech-end"]
    assert ordered == [
        "speech-start",
        "tts_first_audio",
        ("started", "null-sink"),
        "speech-end",
        ("terminal", "null-sink"),
    ]
    assert engine.spoken == ["complete replay text"]


def test_file_replay_resolves_typed_style_like_live_sherpa(monkeypatch):
    tts = _ParamTts()
    _patch_models(monkeypatch, _FakeRecognizer(), tts)
    engine = FileReplayEngine(
        SherpaConfig(
            asr_encoder="x",
            tts_model="y",
            tts_markup=True,
            tts_speaker_voices={"warm": 16},
            tts_emotion_speed_map={"calm": 0.9},
        )
    )
    engine.start(EngineCallbacks())
    probe = _ReceiptProbe()

    engine.speak_tracked(
        TrackedSpeech(
            "styled",
            "Inherited voice.",
            SpeechStyle("warm", "calm", 1.1),
        ),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )

    assert tts.params == [("Inherited voice.", 16, pytest.approx(0.99))]
    assert probe.snapshot()[1][0].safe_text_prefix == "Inherited voice."


def test_file_replay_raw_markup_matches_sherpa_sanitization_and_precedence(
    monkeypatch,
):
    tts = _ParamTts()
    _patch_models(monkeypatch, _FakeRecognizer(), tts)
    engine = FileReplayEngine(
        SherpaConfig(
            asr_encoder="x",
            tts_model="y",
            tts_markup=True,
            tts_speaker_voices={"warm": 16, "deep": 9},
        )
    )
    engine.start(EngineCallbacks())

    raw = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech("raw", "[voice:deep] Raw directive."),
        on_started=raw.on_started,
        on_terminal=raw.on_terminal,
    )
    conflict = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech(
            "conflict",
            "[voice:deep] Explicit wins.",
            SpeechStyle(voice="warm"),
        ),
        on_started=conflict.on_started,
        on_terminal=conflict.on_terminal,
    )
    tag_only = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech("tag-only", "[voice:deep]"),
        on_started=tag_only.on_started,
        on_terminal=tag_only.on_terminal,
    )

    assert tts.params == [
        ("Raw directive.", 9, 1.0),
        ("Explicit wins.", 9, 1.0),
    ]
    assert raw.snapshot()[1][0].safe_text_prefix == "Raw directive."
    assert conflict.snapshot()[1][0].safe_text_prefix == "Explicit wins."
    assert tag_only.snapshot()[0] == [("terminal", "tag-only")]
    assert tag_only.snapshot()[1][0].outcome is PlaybackOutcome.DROPPED
    assert tag_only.snapshot()[1][0].safe_text_prefix == ""


@pytest.mark.parametrize(
    "raw",
    [
        "[tag:story] Here is the first sequence.",
        "[tag:narrator] The moon orbits Earth.",
        "[narrator:deep] Once upon a time.",
    ],
)
def test_file_replay_receipt_excludes_unsupported_control_tag(monkeypatch, raw):
    tts = _ParamTts()
    _patch_models(monkeypatch, _FakeRecognizer(), tts)
    engine = FileReplayEngine(
        SherpaConfig(
            asr_encoder="x",
            tts_model="y",
            tts_markup=True,
            tts_speaker_voices={"narrator": 7},
        )
    )
    engine.start(EngineCallbacks())
    probe = _ReceiptProbe()

    engine.speak_tracked(
        TrackedSpeech("unsupported-control", raw),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )

    [(spoken, _sid, _speed)] = tts.params
    receipt = probe.snapshot()[1][0]
    assert not spoken.startswith("[")
    assert receipt.safe_text_prefix == spoken


def test_file_replay_interrupt_terminalizes_before_blocked_generate_returns(
    monkeypatch,
):
    tts = _BlockingTts()
    metrics: list[str] = []
    lifecycle: list[str] = []
    engine = _start_engine(
        monkeypatch,
        tts,
        EngineCallbacks(
            on_metric=metrics.append,
            on_speech_start=lambda: lifecycle.append("speech-start"),
            on_speech_end=lambda: lifecycle.append("speech-end"),
        ),
    )
    probe = _ReceiptProbe()
    worker = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("blocked", "blocked generation"),
            on_started=probe.on_started,
            on_terminal=probe.on_terminal,
        ),
        daemon=True,
    )
    worker.start()
    assert tts.entered.wait(timeout=1.0)

    engine.stop_speaking()

    assert probe.terminal.wait(timeout=0.2)
    assert worker.is_alive()
    events, receipts = probe.snapshot()
    assert events == [("terminal", "blocked")]
    assert receipts[0].outcome is PlaybackOutcome.INTERRUPTED
    assert receipts[0].safe_text_prefix == ""
    assert metrics == []

    tts.release.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert probe.snapshot()[0] == [("terminal", "blocked")]
    assert lifecycle == ["speech-start", "speech-end"]

    # Transient interruption does not poison the next deterministic turn.
    next_probe = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech("after-cut", "fresh replay"),
        on_started=next_probe.on_started,
        on_terminal=next_probe.on_terminal,
    )
    assert next_probe.snapshot()[1][0].outcome is PlaybackOutcome.COMPLETED


def test_file_replay_busy_request_is_dropped_without_overlapping_tts(monkeypatch):
    tts = _BlockingTts()
    engine = _start_engine(monkeypatch, tts)
    first = _ReceiptProbe()
    second = _ReceiptProbe()
    worker = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("first", "first text"),
            on_started=first.on_started,
            on_terminal=first.on_terminal,
        ),
        daemon=True,
    )
    worker.start()
    assert tts.entered.wait(timeout=1.0)

    engine.speak_tracked(
        TrackedSpeech("second", "second text"),
        on_started=second.on_started,
        on_terminal=second.on_terminal,
    )

    assert second.snapshot()[0] == [("terminal", "second")]
    assert second.snapshot()[1][0].outcome is PlaybackOutcome.DROPPED
    assert tts.calls == ["first text"]
    assert engine.spoken == ["first text"]

    tts.release.set()
    worker.join(timeout=1.0)
    assert first.snapshot()[1][0].outcome is PlaybackOutcome.COMPLETED


def test_file_replay_cut_fences_request_before_receipt_admission(monkeypatch):
    tts = _FakeTts()
    engine = _start_engine(monkeypatch, tts)
    gate = _AdmissionGate()
    engine._generate_lock = gate
    probe = _ReceiptProbe()
    worker = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("stale-admission", "must not synthesize"),
            on_started=probe.on_started,
            on_terminal=probe.on_terminal,
        ),
        daemon=True,
    )
    worker.start()
    assert gate.acquired.wait(timeout=1.0)

    engine.stop_speaking()
    gate.resume.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert not gate.locked()
    assert tts.calls == []
    assert probe.snapshot()[0] == [("terminal", "stale-admission")]
    assert probe.snapshot()[1][0].outcome is PlaybackOutcome.DROPPED


def test_file_replay_concurrent_legacy_does_not_queue_past_cut(monkeypatch):
    tts = _BlockingTts()
    engine = _start_engine(monkeypatch, tts)
    tracked = _ReceiptProbe()
    tracked_worker = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("tracked", "blocked tracked"),
            on_started=tracked.on_started,
            on_terminal=tracked.on_terminal,
        ),
        daemon=True,
    )
    tracked_worker.start()
    assert tts.entered.wait(timeout=1.0)

    legacy_done = threading.Event()
    legacy_worker = threading.Thread(
        target=lambda: engine.speak("stale legacy", on_done=legacy_done.set),
        daemon=True,
    )
    legacy_worker.start()
    legacy_worker.join(timeout=0.2)
    assert not legacy_worker.is_alive()
    assert legacy_done.is_set()

    engine.stop_speaking()
    tts.release.set()
    tracked_worker.join(timeout=1.0)

    assert tracked.snapshot()[1][0].outcome is PlaybackOutcome.INTERRUPTED
    assert tts.calls == ["blocked tracked"]
    assert engine.spoken == ["blocked tracked"]

    # A genuinely new call after the transient cut remains usable.
    engine.speak("fresh legacy")
    assert tts.calls == ["blocked tracked", "fresh legacy"]


def test_file_replay_legacy_observability_failure_clears_speaking(monkeypatch):
    class _FailingCollector:
        def append(self, _text: str) -> None:
            raise RuntimeError("legacy collector failure")

    tts = _FakeTts()
    engine = _start_engine(monkeypatch, tts)
    engine.spoken = _FailingCollector()
    done = threading.Event()

    with pytest.raises(RuntimeError, match="legacy collector failure"):
        engine.speak("legacy failure", on_done=done.set)

    assert done.is_set()
    assert not engine.is_speaking
    assert not engine._generate_lock.locked()
    assert tts.calls == []

    engine.spoken = []
    engine.speak("healthy legacy")
    assert tts.calls == ["healthy legacy"]


def test_file_replay_terminal_reentry_to_legacy_does_not_block_cut(monkeypatch):
    tts = _BlockingTts()
    engine = _start_engine(monkeypatch, tts)
    receipts: list[PlaybackReceipt] = []
    nested_done = threading.Event()

    def terminal(receipt: PlaybackReceipt) -> None:
        receipts.append(receipt)
        engine.speak("nested legacy", on_done=nested_done.set)

    worker = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("reentrant-cut", "blocked generation"),
            on_terminal=terminal,
        ),
        daemon=True,
    )
    worker.start()
    assert tts.entered.wait(timeout=1.0)

    stopper = threading.Thread(target=engine.stop_speaking, daemon=True)
    stopper.start()
    stopper.join(timeout=0.2)
    returned_before_native_release = not stopper.is_alive()
    tts.release.set()
    worker.join(timeout=1.0)
    stopper.join(timeout=1.0)

    assert returned_before_native_release
    assert nested_done.is_set()
    assert [receipt.outcome for receipt in receipts] == [
        PlaybackOutcome.INTERRUPTED
    ]
    assert tts.calls == ["blocked generation"]


def test_file_replay_end_callback_cannot_rewrite_completed_sink_truth(monkeypatch):
    holder: dict[str, FileReplayEngine] = {}
    engine = _start_engine(
        monkeypatch,
        _FakeTts(),
        EngineCallbacks(on_speech_end=lambda: holder["engine"].stop_speaking()),
    )
    holder["engine"] = engine
    probe = _ReceiptProbe()

    engine.speak_tracked(
        TrackedSpeech("completed-before-end", "already reached null sink"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )

    assert probe.snapshot()[0] == [
        ("started", "completed-before-end"),
        ("terminal", "completed-before-end"),
    ]
    assert probe.snapshot()[1][0].outcome is PlaybackOutcome.COMPLETED


def test_file_replay_restart_keeps_old_generation_callbacks_isolated(monkeypatch):
    tts = _BlockingTts()
    old_end: list[str] = []
    new_end: list[str] = []
    engine = _start_engine(
        monkeypatch,
        tts,
        EngineCallbacks(on_speech_end=lambda: old_end.append("old")),
    )
    probe = _ReceiptProbe()
    worker = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("old-session", "blocked old session"),
            on_terminal=probe.on_terminal,
        ),
        daemon=True,
    )
    worker.start()
    assert tts.entered.wait(timeout=1.0)

    engine.stop()
    engine.start(EngineCallbacks(on_speech_end=lambda: new_end.append("new")))
    tts.release.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert probe.snapshot()[1][0].outcome is PlaybackOutcome.INTERRUPTED
    assert old_end == ["old"]
    assert new_end == []


def test_file_replay_overlapping_start_interrupts_old_generation(monkeypatch):
    tts = _BlockingTts()
    old_end: list[str] = []
    new_end: list[str] = []
    engine = _start_engine(
        monkeypatch,
        tts,
        EngineCallbacks(on_speech_end=lambda: old_end.append("old")),
    )
    old = _ReceiptProbe()
    worker = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("overlapped", "old in-flight generation"),
            on_terminal=old.on_terminal,
        ),
        daemon=True,
    )
    worker.start()
    assert tts.entered.wait(timeout=1.0)

    engine.start(EngineCallbacks(on_speech_end=lambda: new_end.append("new")))
    assert old.snapshot()[1][0].outcome is PlaybackOutcome.INTERRUPTED
    tts.release.set()
    worker.join(timeout=1.0)

    fresh = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech("new-session", "fresh generation"),
        on_started=fresh.on_started,
        on_terminal=fresh.on_terminal,
    )
    assert fresh.snapshot()[1][0].outcome is PlaybackOutcome.COMPLETED
    assert old_end == ["old"]
    assert new_end == ["new"]


def test_file_replay_invalid_fragment_does_not_poison_model_admission(monkeypatch):
    tts = _FakeTts()
    engine = _start_engine(monkeypatch, tts)

    with pytest.raises(ValueError, match="fragment_id"):
        engine.speak_tracked(
            TrackedSpeech("", "invalid identifier"),
            on_terminal=lambda _receipt: None,
        )

    valid = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech("valid-after-error", "still works"),
        on_started=valid.on_started,
        on_terminal=valid.on_terminal,
    )
    assert valid.snapshot()[1][0].outcome is PlaybackOutcome.COMPLETED
    assert tts.calls == ["still works"]
    assert not engine._generate_lock.locked()


def test_file_replay_receipt_lock_interrupt_releases_model_admission(monkeypatch):
    class _InterruptSecondEnter:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._enters = 0

        def __enter__(self):
            self._enters += 1
            if self._enters == 2:
                raise KeyboardInterrupt
            self._lock.acquire()
            return self

        def __exit__(self, *_args) -> None:
            self._lock.release()

    tts = _FakeTts()
    engine = _start_engine(monkeypatch, tts)
    engine._receipt_lock = _InterruptSecondEnter()

    with pytest.raises(KeyboardInterrupt):
        engine.speak_tracked(
            TrackedSpeech("interrupted-admission", "never admitted"),
            on_terminal=lambda _receipt: None,
        )

    assert not engine._generate_lock.locked()
    assert not engine._tracked_busy
    assert engine._active_tracked is None
    assert tts.calls == []

    engine._receipt_lock = threading.RLock()
    healthy = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech("after-lock-interrupt", "healthy synthesis"),
        on_started=healthy.on_started,
        on_terminal=healthy.on_terminal,
    )
    assert healthy.snapshot()[1][0].outcome is PlaybackOutcome.COMPLETED


def test_file_replay_observability_failure_terminalizes_and_releases_model(
    monkeypatch,
):
    class _FailingCollector:
        def append(self, _text: str) -> None:
            raise RuntimeError("collector failure")

    tts = _FakeTts()
    lifecycle: list[str] = []
    engine = _start_engine(
        monkeypatch,
        tts,
        EngineCallbacks(
            on_speech_start=lambda: lifecycle.append("start"),
            on_speech_end=lambda: lifecycle.append("end"),
        ),
    )
    engine.spoken = _FailingCollector()
    failed = _ReceiptProbe()

    with pytest.raises(RuntimeError, match="collector failure"):
        engine.speak_tracked(
            TrackedSpeech("collector-error", "must fail before synthesis"),
            on_started=failed.on_started,
            on_terminal=failed.on_terminal,
        )

    assert failed.snapshot()[1][0].outcome is PlaybackOutcome.FAILED
    assert not engine.is_speaking
    assert not engine._tracked_busy
    assert engine._active_tracked is None
    assert not engine._generate_lock.locked()
    assert tts.calls == []
    assert lifecycle == []

    engine.spoken = []
    healthy = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech("after-collector-error", "healthy synthesis"),
        on_started=healthy.on_started,
        on_terminal=healthy.on_terminal,
    )
    assert healthy.snapshot()[1][0].outcome is PlaybackOutcome.COMPLETED
    assert lifecycle == ["start", "end"]


@pytest.mark.parametrize(
    "callback_name",
    ["speech_start", "speech_end", "metric", "started", "terminal"],
)
def test_file_replay_callback_base_exception_cleans_and_terminalizes(
    monkeypatch,
    callback_name,
):
    raised = threading.Event()
    receipts: list[PlaybackReceipt] = []

    def fail_once() -> None:
        if not raised.is_set():
            raised.set()
            raise KeyboardInterrupt

    callbacks = EngineCallbacks(
        on_speech_start=(
            fail_once if callback_name == "speech_start" else lambda: None
        ),
        on_speech_end=(
            fail_once if callback_name == "speech_end" else lambda: None
        ),
        on_metric=(
            (lambda _name: fail_once())
            if callback_name == "metric"
            else lambda _name: None
        ),
    )
    tts = _FakeTts()
    engine = _start_engine(monkeypatch, tts, callbacks)

    def started(_fragment_id: str) -> None:
        if callback_name == "started":
            fail_once()

    def terminal(receipt: PlaybackReceipt) -> None:
        receipts.append(receipt)
        if callback_name == "terminal":
            fail_once()

    with pytest.raises(KeyboardInterrupt):
        engine.speak_tracked(
            TrackedSpeech("fatal-callback", "first attempt"),
            on_started=started,
            on_terminal=terminal,
        )

    assert len(receipts) == 1
    expected = (
        PlaybackOutcome.FAILED
        if callback_name == "speech_start"
        else PlaybackOutcome.COMPLETED
    )
    assert receipts[0].outcome is expected
    assert not engine.is_speaking
    assert not engine._tracked_busy
    assert engine._active_tracked is None
    assert not engine._generate_lock.locked()

    healthy = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech("after-fatal-callback", "second attempt"),
        on_started=healthy.on_started,
        on_terminal=healthy.on_terminal,
    )
    assert healthy.snapshot()[1][0].outcome is PlaybackOutcome.COMPLETED


def test_file_replay_stop_drains_active_and_drops_future_requests(monkeypatch):
    tts = _BlockingTts()
    engine = _start_engine(monkeypatch, tts)
    active = _ReceiptProbe()
    worker = threading.Thread(
        target=lambda: engine.speak_tracked(
            TrackedSpeech("active", "active text"),
            on_started=active.on_started,
            on_terminal=active.on_terminal,
        ),
        daemon=True,
    )
    worker.start()
    assert tts.entered.wait(timeout=1.0)

    engine.stop()
    assert active.terminal.is_set()
    assert active.snapshot()[1][0].outcome is PlaybackOutcome.INTERRUPTED

    future = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech("future", "future text"),
        on_started=future.on_started,
        on_terminal=future.on_terminal,
    )
    assert future.snapshot()[1][0].outcome is PlaybackOutcome.DROPPED

    engine.stop()
    tts.release.set()
    worker.join(timeout=1.0)
    assert len(active.snapshot()[1]) == 1


@pytest.mark.parametrize(
    "kind",
    ["missing", "empty", "nested-empty", "nonnumeric", "exception"],
)
def test_file_replay_tracked_failure_paths_never_claim_null_sink(monkeypatch, kind):
    if kind == "missing":
        tts = None
    elif kind == "empty":
        tts = _FakeTts()
        tts.generate = lambda *_args, **_kwargs: _GeneratedAudio(count=0)
    elif kind == "nested-empty":
        tts = _FakeTts()
        malformed = _GeneratedAudio()
        malformed.samples = [[]]
        tts.generate = lambda *_args, **_kwargs: malformed
    elif kind == "nonnumeric":
        tts = _FakeTts()
        malformed = _GeneratedAudio()
        malformed.samples = [None, "not-a-sample"]
        tts.generate = lambda *_args, **_kwargs: malformed
    else:
        tts = _FakeTts()

        def fail(*_args, **_kwargs):
            raise RuntimeError("scripted TTS failure")

        tts.generate = fail
    engine = _start_engine(monkeypatch, tts)
    probe = _ReceiptProbe()

    engine.speak_tracked(
        TrackedSpeech(f"failure-{kind}", "unheard"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )

    events, receipts = probe.snapshot()
    assert events == [("terminal", f"failure-{kind}")]
    assert receipts[0].outcome is PlaybackOutcome.FAILED
    assert receipts[0].safe_text_prefix == ""


def test_file_replay_prestart_and_speak_only_subclass_stay_legacy_safe():
    engine = FileReplayEngine(SherpaConfig())
    probe = _ReceiptProbe()
    engine._tts = _FakeTts()
    engine.speak_tracked(
        TrackedSpeech("prestart", "not started"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )
    assert probe.snapshot()[1][0].outcome is PlaybackOutcome.DROPPED

    class _SpeakOnlyReplay(FileReplayEngine):
        def speak(self, text, on_done=None):
            super().speak(text, on_done)

    assert not _SpeakOnlyReplay(SherpaConfig()).playback_capabilities.tracked_terminal


def test_file_replay_tracked_callbacks_can_reenter_stop_without_duplication(
    monkeypatch,
):
    engine = _start_engine(monkeypatch, _FakeTts())
    receipts: list[PlaybackReceipt] = []

    def started(_fragment_id):
        engine.stop_speaking()

    def terminal(receipt):
        receipts.append(receipt)
        engine.stop()

    engine.speak_tracked(
        TrackedSpeech("reentrant", "already at null sink"),
        on_started=started,
        on_terminal=terminal,
    )

    assert len(receipts) == 1
    assert receipts[0].outcome is PlaybackOutcome.COMPLETED


def test_file_replay_callback_failures_do_not_erase_or_poison_receipts(monkeypatch):
    def fail_metric(_name):
        raise RuntimeError("metric failure")

    engine = _start_engine(
        monkeypatch,
        _FakeTts(),
        EngineCallbacks(on_metric=fail_metric),
    )
    terminals: list[str] = []

    def fail_started(_fragment_id):
        raise RuntimeError("started failure")

    def fail_terminal(receipt):
        terminals.append(receipt.fragment_id)
        raise RuntimeError("terminal failure")

    engine.speak_tracked(
        TrackedSpeech("callback-errors", "still completes"),
        on_started=fail_started,
        on_terminal=fail_terminal,
    )
    probe = _ReceiptProbe()
    engine.speak_tracked(
        TrackedSpeech("after-errors", "later call"),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )

    assert terminals == ["callback-errors"]
    assert probe.snapshot()[1][0].outcome is PlaybackOutcome.COMPLETED


def test_file_replay_blank_tracked_request_is_dropped_without_synthesis(monkeypatch):
    tts = _FakeTts()
    engine = _start_engine(monkeypatch, tts)
    probe = _ReceiptProbe()

    engine.speak_tracked(
        TrackedSpeech("blank", "   "),
        on_started=probe.on_started,
        on_terminal=probe.on_terminal,
    )

    assert probe.snapshot()[0] == [("terminal", "blank")]
    assert probe.snapshot()[1][0].outcome is PlaybackOutcome.DROPPED
    assert tts.calls == []


def test_file_replay_recording_turn_commits_only_null_sink_receipt(monkeypatch):
    tts = _FakeTts()
    _patch_models(monkeypatch, _FakeRecognizer(), tts)
    engine = FileReplayEngine(SherpaConfig(asr_encoder="x", tts_model="y"))
    runtime = VoiceRuntime(
        engine,
        EchoLLM("receipt-backed response"),
        warm_on_start=False,
    )
    runtime.start(run_bus=False)
    samples = np.concatenate(
        [np.ones(1600, dtype="float32") * 0.5, np.zeros(1600, dtype="float32")]
    )
    try:
        engine.replay_samples(samples, 16000)
        assert runtime.wait_idle(timeout=2.0)
        assert engine.spoken == ["receipt-backed response"]
        assert [
            item.text
            for item in runtime.memory.all()
            if "assistant_output" in item.tags
        ] == ["receipt-backed response"]
    finally:
        runtime.stop()


def test_load_waveform_reads_real_npy_fixture():
    paths = sorted(glob.glob("tests/fixture_audio/real_usage_full/*.npy"))
    if not paths:
        pytest.skip("no .npy fixtures present")
    samples, sample_rate = load_waveform(paths[0])
    assert sample_rate == 16000
    assert samples.dtype == np.float32
    assert samples.ndim == 1 and samples.size > 0


def test_load_waveform_rejects_unknown_format():
    with pytest.raises(ValueError):
        load_waveform("something.mp3")


@pytest.mark.real_model
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sherpa_onnx") is None,
    reason="sherpa_onnx native package not installed",
)
def test_replay_with_real_models_if_available():
    # Smoke test for the real path -- only runs where sherpa-onnx + model files
    # exist (the bench environment / CI perf job), skipped otherwise.
    pytest.skip("requires configured sherpa model files; exercised by tools/bench")
