"""Retired synthesis spends no successor native/DSP work; no models or devices."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.engines.sherpa import SherpaConfig, SherpaOnnxEngine


SAMPLES = np.asarray([0.1, -0.2, 0.3, -0.1] * 16, dtype=np.float32)


def engine(tts, *, whole=False, **config):
    result = SherpaOnnxEngine(SherpaConfig(
        tts_target_rms=0, tts_output_leveler=whole, tts_declick=False,
        tts_dc_block=False, **config))
    result._tts = tts
    return result


class Native:
    sample_rate = 16000
    num_speakers = 4
    def __init__(self):
        self.calls = []
    def generate(self, text, sid=0, speed=1.0, callback=None):
        self.calls.append((text, sid, speed))
        if callback is not None:
            callback(SAMPLES.copy(), 1.0)
        return SimpleNamespace(samples=SAMPLES.copy(), sample_rate=self.sample_rate)


class ObservedLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.attempted = threading.Event()
    def __enter__(self):
        self.attempted.set()
        self.lock.acquire()
        return self
    def __exit__(self, *args):
        self.lock.release()


def run_thread(function, errors):
    def run():
        try:
            function()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    return thread


@pytest.mark.parametrize("whole", [False, True])
@pytest.mark.parametrize("retirement", ["stop", "generation"])
def test_already_retired_request_never_enters_native_or_dsp(whole, retirement):
    native = Native()
    subject = engine(native, whole=whole)
    generation = subject._speak_gen
    if retirement == "stop":
        subject.stop_speaking()
    else:
        subject._speak_gen += 1  # cleared stop from a fresh generation is insufficient
    subject._dc_block = lambda *_: pytest.fail("retired DSP")
    subject._synthesize("public retired phrase", lambda *_: pytest.fail("retired sink"), gen=generation)
    assert native.calls == []


@pytest.mark.parametrize("whole", [False, True])
def test_stop_while_waiting_for_model_lock_skips_native_after_release(whole):
    native = Native()
    subject = engine(native, whole=whole)
    lock = ObservedLock()
    subject._tts_lock = lock
    generation = subject._speak_gen
    errors, written = [], []
    lock.lock.acquire()
    held = True
    thread = run_thread(lambda: subject._synthesize("public retired phrase", written.append, gen=generation), errors)
    try:
        assert lock.attempted.wait(1)
        subject.stop_speaking()
        assert subject._claim_utterance(subject._speak_gen) is not None  # clears stop for successor
        lock.lock.release()
        held = False
        thread.join(1)
        assert not thread.is_alive()
        assert errors == [] and native.calls == [] and written == []
    finally:
        if held:
            lock.lock.release()
        thread.join(2)


def test_successor_progress_does_not_wait_for_retired_render_to_finish():
    class BlockingRetired(Native):
        def __init__(self):
            super().__init__()
            self.release_retired = threading.Event()
            self.successor_entered = threading.Event()
        def generate(self, text, sid=0, speed=1.0, callback=None):
            if text == "retired":
                assert self.release_retired.wait(3)
            else:
                self.successor_entered.set()
            return super().generate(text, sid=sid, speed=speed, callback=callback)
    native = BlockingRetired()
    subject = engine(native)
    lock = ObservedLock()
    subject._tts_lock = lock
    prior_generation = subject._speak_gen
    errors = []
    lock.lock.acquire()
    held = True
    old = run_thread(lambda: subject._synthesize("retired", lambda _: None, gen=prior_generation), errors)
    successor = None
    try:
        assert lock.attempted.wait(1)
        subject.stop_speaking()
        next_generation = subject._speak_gen
        subject._claim_utterance(next_generation)
        successor = run_thread(lambda: subject._synthesize("successor", lambda _: None, gen=next_generation), errors)
        lock.lock.release()
        held = False
        assert native.successor_entered.wait(1)
        assert not native.release_retired.is_set()  # obsolete work was never needed
        old.join(1)
        successor.join(1)
        assert not old.is_alive() and not successor.is_alive()
        assert errors == [] and [call[0] for call in native.calls] == ["successor"]
    finally:
        native.release_retired.set()
        if held:
            lock.lock.release()
        old.join(2)
        if successor is not None:
            successor.join(2)


def test_callback_after_native_stop_returns_zero_without_converting_or_processing():
    class InvalidRetiredChunk(Native):
        def generate(self, text, sid=0, speed=1.0, callback=None):
            self.calls.append(text)
            subject.stop_speaking()
            assert callback(object(), 1.0) == 0  # stale native payload is never read
    native = InvalidRetiredChunk()
    subject = engine(native)
    subject._dc_block = lambda *_: pytest.fail("retired callback DSP")
    subject._synthesize("public phrase", lambda _: pytest.fail("retired write"), gen=subject._speak_gen)
    assert native.calls == ["public phrase"]


def test_stop_during_callback_dsp_skips_successor_dsp_and_sink(monkeypatch):
    import core.engines.sherpa as module
    native = Native()
    subject = engine(native)
    subject.config.tts_declick = True
    def dc(samples, sr):
        subject.stop_speaking()
        return samples
    subject._dc_block = dc
    monkeypatch.setattr(module, "declick", lambda *a, **k: pytest.fail("DSP after retirement"))
    subject._synthesize("public phrase", lambda _: pytest.fail("retired write"), gen=subject._speak_gen)
    assert len(native.calls) == 1


def test_cancelled_callback_keeps_native_lock_owned_until_true_generate_return():
    callback_done, cleanup_release = threading.Event(), threading.Event()
    class CleanupNative(Native):
        def generate(self, text, sid=0, speed=1.0, callback=None):
            subject.stop_speaking()
            assert callback(SAMPLES, 1.0) == 0
            callback_done.set()
            assert cleanup_release.wait(3)
    native = CleanupNative()
    subject = engine(native)
    errors = []
    generation = subject._speak_gen
    thread = run_thread(lambda: subject._synthesize("public phrase", lambda _: None, gen=generation), errors)
    try:
        assert callback_done.wait(1)
        assert thread.is_alive()
        acquired = subject._tts_lock.acquire(blocking=False)
        if acquired:
            subject._tts_lock.release()
        assert not acquired  # a stop request is not native cleanup completion
    finally:
        cleanup_release.set()
        thread.join(2)
    assert not thread.is_alive() and errors == []


def test_wholeclip_return_after_retirement_does_not_materialize_samples_or_mutate_carry():
    class ReturnedAudio:
        @property
        def samples(self):
            pytest.fail("retired native waveform must not be materialized")
    class BatchNative:
        sample_rate = 16000
        def generate(self, text, sid=0, speed=1.0):
            subject.stop_speaking()
            return ReturnedAudio()
    subject = engine(BatchNative(), whole=True)
    subject._tts_normalize_gain = 0.75
    subject._tts_level_gain_db = -2.0
    subject._dc_block = lambda *_: pytest.fail("retired whole-clip DSP")
    subject._synthesize("public phrase", lambda _: pytest.fail("retired write"), gen=subject._speak_gen)
    assert subject._tts_normalize_gain == 0.75 and subject._tts_level_gain_db == -2.0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("locked", [False, True])
def test_healthy_samples_and_expressive_native_parameters_are_unchanged(streaming, locked):
    class BatchNative(Native):
        def generate(self, text, sid=0, speed=1.0):
            return super().generate(text, sid=sid, speed=speed)
    native = Native() if streaming else BatchNative()
    subject = engine(native, tts_markup=True, tts_speaker_voices={"soft": 2},
                     tts_emotion_speed_map={"calm": 0.9}, tts_lock_speaker_id=locked)
    written = []
    subject._synthesize("[voice:soft emotion:calm] Public phrase.", written.append, gen=subject._speak_gen)
    np.testing.assert_array_equal(np.concatenate(written), SAMPLES)
    assert len(native.calls) == 1
    assert native.calls[0] == ("Public phrase.", 0 if locked else 2, pytest.approx(0.9))


def test_entered_native_failure_is_not_swallowed_or_retried_after_stop():
    class Failed(Native):
        def generate(self, text, sid=0, speed=1.0, callback=None):
            self.calls.append(text)
            subject.stop_speaking()
            raise RuntimeError("public entered native failure")
    native = Failed()
    subject = engine(native)
    with pytest.raises(RuntimeError, match="public entered native failure"):
        subject._synthesize("public phrase", lambda _: None, gen=subject._speak_gen)
    assert native.calls == ["public phrase"]


def test_retirement_during_wholeclip_leveler_does_not_publish_new_gain(monkeypatch):
    import core.engines.sherpa as module
    native = Native()
    subject = engine(native, whole=True)
    subject._tts_level_gain_db = -2.0
    def leveler(samples, **kwargs):
        subject.stop_speaking()
        return samples, 99.0
    monkeypatch.setattr(module, "output_leveler", leveler)
    subject._synthesize("public phrase", lambda _: pytest.fail("retired write"), gen=subject._speak_gen)
    assert subject._tts_level_gain_db == -2.0


def test_retirement_during_wholeclip_normalize_does_not_seed_successor_gain(monkeypatch):
    import core.engines.sherpa as module
    native = Native()
    subject = SherpaOnnxEngine(SherpaConfig(tts_target_rms=0.12, tts_output_leveler=False,
                                         tts_declick=False, tts_dc_block=False))
    subject._tts = native
    def normalize(samples, target):
        subject.stop_speaking()
        return samples
    monkeypatch.setattr(module, "normalize_rms", normalize)
    subject._synthesize("public phrase", lambda _: pytest.fail("retired write"), gen=subject._speak_gen)
    assert subject._tts_normalize_gain is None
