"""Tests for ``core.engines._recovering_input._RecoveringInputStream``.

The wrapper protects ``core/engines/sherpa.py``'s capture loop from
PortAudio errors that would otherwise silently kill the audio thread.
These tests inject scripted errors via the ``opener``/``stream`` hooks
so we don't need real hardware -- the wrapper's error-classification,
backoff, fallback chain, and state-listener contracts are all
exercised end-to-end."""
from __future__ import annotations

import threading
import time
from typing import Iterator

import numpy as np
import pytest

# sounddevice is imported by the wrapper module's docstring example but
# the module itself doesn't import it -- so we can synthesize a
# ``PortAudioError`` exception class shape without the real dep.
try:
    import sounddevice as sd  # type: ignore
    _HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    _HAS_SOUNDDEVICE = False


from core.engines._recovering_input import (
    DEFAULT_BACKOFFS,
    OpenAttempt,
    PA_BAD_STREAM_PTR,
    PA_DEVICE_UNAVAILABLE,
    PA_INPUT_OVERFLOWED,
    PA_OUTPUT_UNDERFLOWED,
    PA_STREAM_IS_STOPPED,
    PA_TIMED_OUT,
    PA_UNANTICIPATED_HOST_ERROR,
    REOPEN_CODES,
    StreamState,
    TRANSIENT_CODES,
    _RecoveringInputStream,
    _err_code,
)


# --- fake exception + scripted stream -------------------------------------


class _FakePortAudioError(Exception):
    """Stand-in for sd.PortAudioError when sounddevice isn't available.

    Mirrors the real shape: ``args = (message, code, host_triple)`` so
    ``_err_code`` reads the code from ``args[1]``."""


def _make_err(code: int, message: str = "simulated"):
    """Build either a real sd.PortAudioError or our local stand-in,
    whichever is importable."""
    if _HAS_SOUNDDEVICE:
        return sd.PortAudioError(message, code, (0, 0, ""))
    return _FakePortAudioError(message, code, (0, 0, ""))


class _ScriptedStream:
    """A fake ``sd.InputStream`` whose ``read`` walks a plan.

    Each plan entry is either an Exception (raised) or a tuple
    ``(np.ndarray, bool)`` (returned as the read result). The stream
    counts ``start``/``stop``/``close`` calls so tests can assert
    proper lifecycle handling."""

    def __init__(self, plan):
        self._plan = list(plan)
        self._i = 0
        self.read_frames: list[int] = []
        self.start_calls = 0
        self.stop_calls = 0
        self.abort_calls = 0
        self.close_calls = 0
        self.started = False

    def start(self):
        self.start_calls += 1
        self.started = True

    def stop(self):
        self.stop_calls += 1
        self.started = False

    def abort(self):
        self.abort_calls += 1
        self.started = False

    def close(self):
        self.close_calls += 1

    def read(self, frames):
        self.read_frames.append(frames)
        if self._i >= len(self._plan):
            # Default to returning silence if we run out of script -- lets
            # tests assert specific results without exhaustively scripting.
            return (np.zeros((frames, 1), dtype="float32"), False)
        item = self._plan[self._i]
        self._i += 1
        if isinstance(item, Exception):
            raise item
        return item


# --- error-code extraction -------------------------------------------------


def test_err_code_reads_args1():
    exc = _make_err(PA_DEVICE_UNAVAILABLE)
    assert _err_code(exc) == PA_DEVICE_UNAVAILABLE


def test_err_code_handles_malformed_exception():
    """An exception that doesn't follow the PortAudioError shape -- e.g.
    a generic RuntimeError -- must yield None so the caller treats it
    as unrecognized rather than crashing on the read."""
    assert _err_code(RuntimeError("nope")) is None
    assert _err_code(ValueError("a", "not-int", "c")) is None


def test_transient_and_reopen_sets_are_disjoint():
    assert TRANSIENT_CODES.isdisjoint(REOPEN_CODES)


# --- open + fallback chain -------------------------------------------------


def test_open_uses_first_attempt_that_succeeds():
    attempts = [
        OpenAttempt(device="primary", samplerate=16000),
        OpenAttempt(device="primary", samplerate=48000),
    ]
    opens = []

    def opener(device, samplerate):
        opens.append((device, samplerate))
        return _ScriptedStream([])

    rs = _RecoveringInputStream(attempts, opener=opener)
    rs.open()
    assert opens == [("primary", 16000)]
    assert rs.actual_samplerate == 16000
    assert rs.actual_device == "primary"
    assert rs.generation == 1
    assert rs.state == StreamState.OPEN


def test_open_falls_through_to_next_attempt_on_failure():
    """When the preferred device rejects the preferred rate, the wrapper
    tries the next attempt in the chain."""
    attempts = [
        OpenAttempt(device="primary", samplerate=16000),
        OpenAttempt(device="primary", samplerate=48000),
        OpenAttempt(device=None, samplerate=16000),
    ]
    err = _make_err(PA_DEVICE_UNAVAILABLE)

    seen: list[tuple] = []
    def opener(device, samplerate):
        seen.append((device, samplerate))
        if (device, samplerate) == ("primary", 16000):
            raise err
        return _ScriptedStream([])

    rs = _RecoveringInputStream(attempts, opener=opener)
    rs.open()
    assert seen[0] == ("primary", 16000)
    assert seen[1] == ("primary", 48000)
    assert rs.actual_samplerate == 48000


def test_open_closes_candidate_when_start_fails_before_fallback():
    class _StartFails(_ScriptedStream):
        def start(self):
            super().start()
            raise _make_err(PA_DEVICE_UNAVAILABLE)

    bad = _StartFails([])
    good = _ScriptedStream([])
    streams = iter([bad, good])
    rs = _RecoveringInputStream(
        [
            OpenAttempt(device="primary", samplerate=16000),
            OpenAttempt(device=None, samplerate=16000),
        ],
        opener=lambda _device, _samplerate: next(streams),
    )

    rs.open()

    assert bad.close_calls == 1
    assert rs.actual_device is None


def test_initial_open_validates_candidate_before_start():
    events = []

    class _OrderedStream(_ScriptedStream):
        def start(self):
            events.append("start")
            super().start()

    stream = _OrderedStream([])
    attempt = OpenAttempt(device="virtual-capture", samplerate=16000)

    def opener(device, samplerate):
        events.append("construct")
        assert (device, samplerate) == (attempt.device, attempt.samplerate)
        return stream

    def validate(candidate, selected):
        events.append("validate")
        assert candidate is stream
        assert selected is attempt
        assert stream.start_calls == 0

    rs = _RecoveringInputStream(
        [attempt],
        opener=opener,
        pre_start_validate=validate,
    )

    rs.open()

    assert events == ["construct", "validate", "start"]
    assert stream.start_calls == 1


def test_initial_validation_failure_closes_candidate_before_fallback():
    rejected = _ScriptedStream([])
    accepted = _ScriptedStream([])
    streams = iter([rejected, accepted])
    validations = []

    def validate(candidate, attempt):
        validations.append((candidate, attempt.device))
        if candidate is rejected:
            raise RuntimeError("route proof failed")

    rs = _RecoveringInputStream(
        [
            OpenAttempt(device="preferred", samplerate=16000),
            OpenAttempt(device="fallback", samplerate=16000),
        ],
        opener=lambda _device, _samplerate: next(streams),
        pre_start_validate=validate,
    )

    rs.open()

    assert validations == [(rejected, "preferred"), (accepted, "fallback")]
    assert rejected.start_calls == 0
    assert rejected.close_calls == 1
    assert accepted.start_calls == 1
    assert rs.actual_device == "fallback"


def test_open_raises_when_all_attempts_fail():
    """Initial open() raises on full chain exhaustion; state does NOT
    transition to FATAL (that's reserved for runtime recovery failure --
    a startup failure should crash startup, not put the wrapper into a
    weird half-state that the engine has to inspect)."""
    attempts = [OpenAttempt(device=d, samplerate=r) for d, r in [("x", 16000), (None, 16000)]]
    err = _make_err(PA_DEVICE_UNAVAILABLE)

    def opener(device, samplerate):
        raise err

    states: list[StreamState] = []
    rs = _RecoveringInputStream(
        attempts, opener=opener, on_state=lambda s, _m: states.append(s),
    )
    with pytest.raises(Exception):
        rs.open()
    # FATAL is for runtime recovery exhaustion, not startup failure.
    assert rs.state == StreamState.OPEN  # default; no transition emitted
    assert StreamState.FATAL not in states


def test_open_with_no_attempts_raises_value_error():
    with pytest.raises(ValueError):
        _RecoveringInputStream(attempts=[], opener=lambda d, sr: None)


# --- read: transient errors ----------------------------------------------


def test_read_drops_frame_on_input_overflow():
    """Overflow code -> silent block returned with overflowed=True;
    no reopen; stream still healthy."""
    attempts = [OpenAttempt(device=None, samplerate=16000)]
    stream = _ScriptedStream([
        _make_err(PA_INPUT_OVERFLOWED),
        (np.ones((1600, 1), dtype="float32"), False),
    ])

    def opener(device, samplerate):
        return stream

    rs = _RecoveringInputStream(attempts, opener=opener)
    rs.open()
    samples, overflowed = rs.read(1600)
    assert samples.shape == (1600, 1)
    assert overflowed is True
    assert (samples == 0).all()
    # Stream not reopened.
    assert stream.close_calls == 0
    assert rs.state == StreamState.OPEN

    # Next read returns the scripted ones.
    samples, _ = rs.read(1600)
    assert (samples == 1.0).all()


def test_read_drops_frame_on_output_underflow():
    """OutputUnderflowed shows up on the input stream sometimes (shared
    PortAudio loop); should also be treated transient."""
    attempts = [OpenAttempt(device=None, samplerate=16000)]
    stream = _ScriptedStream([_make_err(PA_OUTPUT_UNDERFLOWED)])

    rs = _RecoveringInputStream(attempts, opener=lambda d, sr: stream)
    rs.open()
    _, overflowed = rs.read(1600)
    assert overflowed is True
    assert rs.state == StreamState.OPEN


def test_read_handles_stream_is_stopped_transient():
    attempts = [OpenAttempt(device=None, samplerate=16000)]
    stream = _ScriptedStream([_make_err(PA_STREAM_IS_STOPPED)])

    rs = _RecoveringInputStream(attempts, opener=lambda d, sr: stream)
    rs.open()
    _, overflowed = rs.read(100)
    assert overflowed is True


# --- read: reopen flow ----------------------------------------------------


def test_read_triggers_reopen_on_device_unavailable_then_succeeds():
    """USB unplug: first read raises -9985; the wrapper closes the bad
    stream, walks the fallback chain, and the second open call returns
    a fresh fake."""
    attempts = [OpenAttempt(device=None, samplerate=16000)]

    bad_stream = _ScriptedStream([_make_err(PA_DEVICE_UNAVAILABLE)])
    good_stream = _ScriptedStream([(np.ones((100, 1), dtype="float32"), False)])

    streams = iter([bad_stream, good_stream])
    def opener(device, samplerate):
        return next(streams)

    states: list[tuple[StreamState, str]] = []
    sleeps: list[float] = []
    rs = _RecoveringInputStream(
        attempts, opener=opener,
        on_state=lambda s, m: states.append((s, m)),
        backoffs=(0.0,),  # zero-backoff for fast tests
        sleep_fn=sleeps.append,
    )
    rs.open()
    samples, _ = rs.read(100)
    # Recovered stream returned its scripted samples on the retry.
    assert (samples == 1.0).all()
    # State went OPEN -> RECOVERING -> OPEN.
    state_seq = [s for s, _ in states]
    assert StreamState.RECOVERING in state_seq
    assert state_seq[-1] == StreamState.OPEN
    # The bad stream was closed when we retired it.
    assert bad_stream.close_calls >= 1
    assert good_stream.start_calls == 1


def test_unanticipated_host_error_minus_9999_reopens_then_succeeds():
    """sounddevice/PortAudio's current host-error value is -9999."""
    assert PA_UNANTICIPATED_HOST_ERROR == -9999
    failed = _ScriptedStream([_make_err(PA_UNANTICIPATED_HOST_ERROR)])
    recovered = _ScriptedStream(
        [(np.ones((100, 1), dtype="float32"), False)]
    )
    streams = iter([failed, recovered])
    states: list[StreamState] = []
    rs = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=lambda _device, _samplerate: next(streams),
        on_state=lambda state, _message: states.append(state),
        backoffs=(0.0,),
        sleep_fn=lambda _delay: None,
    )
    rs.open()

    samples, overflowed = rs.read(100)

    assert not overflowed
    assert (samples == 1.0).all()
    assert failed.close_calls == 1
    assert StreamState.RECOVERING in states
    assert states[-1] is StreamState.OPEN


@pytest.mark.parametrize("code", sorted(REOPEN_CODES))
def test_reopen_error_fails_before_replacement_when_recovery_disabled(code):
    error = _make_err(code)
    failed = _ScriptedStream([error])
    replacement = _ScriptedStream(
        [(np.ones((100, 1), dtype="float32"), False)]
    )
    streams = iter([failed, replacement])
    opens = []
    states = []

    def opener(device, samplerate):
        opens.append((device, samplerate))
        return next(streams)

    rs = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=opener,
        on_state=lambda state, message: states.append((state, message)),
        recovery_enabled=False,
    )
    rs.open()

    with pytest.raises(Exception) as raised:
        rs.read(100)

    assert raised.value is error
    assert opens == [(None, 16000)]
    assert failed.close_calls == 0
    assert failed.stop_calls == 0
    assert failed.abort_calls == 0
    assert replacement.start_calls == 0
    assert replacement.read_frames == []
    assert rs.generation == 1
    assert rs.state is StreamState.FATAL
    assert StreamState.RECOVERING not in [state for state, _ in states]
    assert states[-1][0] is StreamState.FATAL
    assert "recovery disabled" in states[-1][1]


def test_reopen_retry_uses_recovered_rate_block_duration():
    """A rate-changing fallback returns one correctly timed recovered block."""
    attempts = [
        OpenAttempt(device="usb", samplerate=48000),
        OpenAttempt(device=None, samplerate=16000),
    ]
    bad = _ScriptedStream([_make_err(PA_DEVICE_UNAVAILABLE)])
    recovered = _ScriptedStream(
        [(np.ones((1600, 1), dtype="float32"), False)]
    )
    opens = iter([bad, _make_err(PA_DEVICE_UNAVAILABLE), recovered])

    def opener(device, samplerate):
        result = next(opens)
        if isinstance(result, Exception):
            raise result
        return result

    rs = _RecoveringInputStream(
        attempts,
        opener=opener,
        backoffs=(0.0,),
        block_seconds=0.1,
        sleep_fn=lambda _t: None,
    )
    rs.open()
    samples, overflowed = rs.read(4800)

    assert not overflowed
    assert samples.shape == (1600, 1)
    assert bad.read_frames == [4800]
    assert recovered.read_frames == [1600]
    assert rs.actual_samplerate == 16000
    assert rs.generation == 2


def test_recovered_stream_first_read_failure_is_bounded_and_fatal():
    attempts = [OpenAttempt(device=None, samplerate=16000)]
    first = _ScriptedStream([_make_err(PA_DEVICE_UNAVAILABLE)])
    recovered = _ScriptedStream([_make_err(PA_DEVICE_UNAVAILABLE)])
    streams = iter([first, recovered])
    rs = _RecoveringInputStream(
        attempts,
        opener=lambda _device, _samplerate: next(streams),
        backoffs=(0.0,),
        sleep_fn=lambda _delay: None,
    )
    rs.open()

    with pytest.raises(Exception):
        rs.read(1600)

    assert rs.generation == 1
    assert rs.state == StreamState.FATAL
    assert recovered.read_frames == [1600]
    assert recovered.close_calls == 1


def test_recovery_skips_startable_but_unreadable_preferred_for_default():
    attempts = [
        OpenAttempt(device="preferred", samplerate=48000),
        OpenAttempt(device=None, samplerate=16000),
    ]
    initial = _ScriptedStream([_make_err(PA_DEVICE_UNAVAILABLE)])
    unreadable = _ScriptedStream([_make_err(PA_DEVICE_UNAVAILABLE)])
    fallback = _ScriptedStream(
        [(np.ones((1600, 1), dtype="float32"), False)]
    )
    streams = iter([initial, unreadable, fallback])
    rs = _RecoveringInputStream(
        attempts,
        opener=lambda _device, _samplerate: next(streams),
        backoffs=(0.0,),
        sleep_fn=lambda _delay: None,
    )
    rs.open()

    samples, overflowed = rs.read(4800)

    assert not overflowed
    assert samples.shape == (1600, 1)
    assert (samples == 1.0).all()
    assert unreadable.close_calls == 1
    assert rs.actual_device is None
    assert rs.actual_samplerate == 16000
    assert rs.generation == 2
    assert rs.state == StreamState.OPEN


def test_close_during_recovery_backoff_prevents_late_reopen():
    attempts = [OpenAttempt(device=None, samplerate=16000)]
    failed = _ScriptedStream([_make_err(PA_DEVICE_UNAVAILABLE)])
    sleep_started = threading.Event()
    release_sleep = threading.Event()
    open_calls = 0

    def opener(_device, _samplerate):
        nonlocal open_calls
        open_calls += 1
        return failed if open_calls == 1 else _ScriptedStream([])

    def sleeper(_delay):
        sleep_started.set()
        assert release_sleep.wait(timeout=2.0)

    rs = _RecoveringInputStream(
        attempts,
        opener=opener,
        backoffs=(3.2,),
        sleep_fn=sleeper,
    )
    rs.open()
    errors = []
    reader = threading.Thread(
        target=lambda: _read_error_into(rs, errors), daemon=True
    )
    reader.start()
    assert sleep_started.wait(timeout=2.0)

    rs.close()
    release_sleep.set()
    reader.join(timeout=2.0)
    assert rs.close(wait_timeout=0.0)

    assert not reader.is_alive()
    assert len(errors) == 1
    assert "closed during recovery" in str(errors[0])
    assert open_calls == 1


def test_shutdown_during_recovery_candidate_start_closes_unpublished_candidate():
    """A start-in-flight candidate is retained until start returns, then released."""
    initial = _ScriptedStream([_make_err(PA_DEVICE_UNAVAILABLE)])
    start_entered = threading.Event()
    release_start = threading.Event()

    class _StartingCandidate(_ScriptedStream):
        def start(self):
            self.start_calls += 1
            start_entered.set()
            assert release_start.wait(timeout=2.0)
            self.started = True

    candidate = _StartingCandidate([])
    streams = iter([initial, candidate])
    rs = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=lambda _device, _samplerate: next(streams),
        backoffs=(0.0,),
        sleep_fn=lambda _delay: None,
    )
    rs.open()
    errors = []
    reader = threading.Thread(
        target=lambda: _read_error_into(rs, errors), daemon=True
    )
    reader.start()
    assert start_entered.wait(timeout=1.0)

    rs.request_close()
    release_start.set()
    reader.join(timeout=1.0)

    assert not reader.is_alive()
    assert len(errors) == 1
    assert "closed during recovery" in str(errors[0])
    assert candidate.close_calls == 1
    assert rs.close(wait_timeout=0.0)


def _read_error_into(stream, errors):
    try:
        stream.read(1600)
    except Exception as exc:  # noqa: BLE001 - asserted by the caller
        errors.append(exc)


def test_read_raises_fatal_after_budget_exhausted():
    """Every reopen attempt fails -> FATAL state + the last exception is raised."""
    attempts = [OpenAttempt(device=None, samplerate=16000)]
    bad = _ScriptedStream([_make_err(PA_DEVICE_UNAVAILABLE)])

    open_calls = [0]
    def opener(device, samplerate):
        open_calls[0] += 1
        if open_calls[0] == 1:
            return bad
        # Subsequent opens fail.
        raise _make_err(PA_DEVICE_UNAVAILABLE)

    states: list[tuple[StreamState, str]] = []
    sleeps: list[float] = []
    rs = _RecoveringInputStream(
        attempts, opener=opener,
        on_state=lambda s, m: states.append((s, m)),
        # Positive backoffs so the sleep_fn actually gets called and we
        # can assert the budget was walked end-to-end.
        backoffs=(0.001, 0.001, 0.001),
        sleep_fn=sleeps.append,
    )
    rs.open()
    with pytest.raises(Exception):
        rs.read(100)
    assert rs.state == StreamState.FATAL
    # Walked the full backoff list.
    assert sleeps == [0.001, 0.001, 0.001]
    # Listener saw RECOVERING then FATAL.
    state_seq = [s for s, _ in states]
    assert StreamState.RECOVERING in state_seq
    assert state_seq[-1] == StreamState.FATAL


def test_recover_walks_full_fallback_chain_on_each_attempt():
    """When the preferred device disappears entirely, the wrapper must
    try the system-default fallback during recovery, not just the
    preferred one."""
    attempts = [
        OpenAttempt(device="hifi-usb", samplerate=48000),
        OpenAttempt(device=None, samplerate=16000),
    ]

    bad = _ScriptedStream([_make_err(PA_DEVICE_UNAVAILABLE)])
    good = _ScriptedStream([(np.zeros((100, 1), dtype="float32"), False)])
    open_log: list[tuple] = []
    open_calls = iter([
        bad,                                                  # initial open succeeds (preferred)
        _make_err(PA_DEVICE_UNAVAILABLE),                     # preferred gone during reopen
        good,                                                 # fallback works
    ])

    def opener(device, samplerate):
        open_log.append((device, samplerate))
        out = next(open_calls)
        if isinstance(out, Exception):
            raise out
        return out

    rs = _RecoveringInputStream(
        attempts, opener=opener, backoffs=(0.0,),
        sleep_fn=lambda _t: None,
    )
    rs.open()
    rs.read(100)  # triggers reopen
    assert rs.generation == 2
    devices = [d for d, _ in open_log]
    assert "hifi-usb" in devices
    assert None in devices  # fell back to default during recovery


def test_unrecognized_error_bubbles_up():
    """A PortAudio error code we don't classify should NOT be silently
    treated as transient. Let the engine's outer except surface it."""
    attempts = [OpenAttempt(device=None, samplerate=16000)]
    # paBufferTooSmall is not a read-recovery condition in this wrapper.
    other = _make_err(-9990)
    stream = _ScriptedStream([other])

    rs = _RecoveringInputStream(attempts, opener=lambda d, sr: stream)
    rs.open()
    with pytest.raises(Exception):
        rs.read(100)


def test_close_releases_stream_handles():
    stream = _ScriptedStream([])
    rs = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=lambda d, sr: stream,
    )
    rs.open()
    assert stream.start_calls == 1
    rs.close()
    assert stream.close_calls == 1


def test_close_waits_for_active_read_before_native_teardown():
    """The normal phase-two close never overlaps a one-block native read."""
    read_started = threading.Event()
    release_read = threading.Event()
    close_called = threading.Event()
    overlap = []

    class _BlockingStream(_ScriptedStream):
        def __init__(self):
            super().__init__([])
            self.read_active = False

        def read(self, frames):
            self.read_active = True
            read_started.set()
            assert release_read.wait(timeout=2.0)
            self.read_active = False
            return np.zeros((frames, 1), dtype="float32"), False

        def close(self):
            overlap.append(self.read_active)
            super().close()
            close_called.set()

    native = _BlockingStream()
    rs = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=lambda _device, _samplerate: native,
        block_seconds=1.0,
    )
    rs.open()
    reader = threading.Thread(target=lambda: rs.read(1600), daemon=True)
    reader.start()
    assert read_started.wait(timeout=1.0)

    outcomes = []
    closer = threading.Thread(target=lambda: outcomes.append(rs.close()), daemon=True)
    closer.start()
    assert rs._closed.wait(timeout=1.0)  # noqa: SLF001 - synchronization seam
    assert not close_called.is_set()
    release_read.set()
    reader.join(timeout=1.0)
    closer.join(timeout=1.0)

    assert not reader.is_alive()
    assert not closer.is_alive()
    assert outcomes == [True]
    assert overlap == [False]


def test_close_aborts_stuck_read_then_closes_only_after_quiescence():
    """Abort may overlap a stuck read; physical close never does."""
    read_started = threading.Event()
    native_aborted = threading.Event()

    class _StuckStream(_ScriptedStream):
        def __init__(self):
            super().__init__([])
            self.read_active = False
            self.close_overlapped = False

        def read(self, frames):
            self.read_active = True
            read_started.set()
            assert native_aborted.wait(timeout=2.0)
            self.read_active = False
            return np.zeros((frames, 1), dtype="float32"), False

        def abort(self):
            super().abort()
            native_aborted.set()

        def close(self):
            self.close_overlapped = self.read_active
            super().close()

    native = _StuckStream()
    rs = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=lambda _device, _samplerate: native,
        block_seconds=0.02,
    )
    rs.open()
    reader = threading.Thread(target=lambda: rs.read(1600), daemon=True)
    reader.start()
    assert read_started.wait(timeout=1.0)

    started = time.monotonic()
    closed = rs.close()
    elapsed = time.monotonic() - started
    reader.join(timeout=1.0)

    assert closed is True
    assert elapsed < 0.5
    assert native.abort_calls == 1
    assert native.close_calls == 1
    assert native.close_overlapped is False
    assert not reader.is_alive()


def test_uninterruptible_abort_is_bounded_and_retains_native_until_idle():
    """A wedged abort cannot trigger concurrent close or an unbounded wait."""
    read_started = threading.Event()
    abort_started = threading.Event()
    allow_abort = threading.Event()
    release_read = threading.Event()

    class _UninterruptibleStream(_ScriptedStream):
        def __init__(self):
            super().__init__([])
            self.read_active = False
            self.close_overlapped = False

        def read(self, frames):
            self.read_active = True
            read_started.set()
            assert release_read.wait(timeout=2.0)
            self.read_active = False
            return np.zeros((frames, 1), dtype="float32"), False

        def abort(self):
            self.abort_calls += 1
            abort_started.set()
            assert allow_abort.wait(timeout=2.0)
            release_read.set()

        def close(self):
            self.close_overlapped = self.read_active
            super().close()

    native = _UninterruptibleStream()
    rs = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=lambda _device, _samplerate: native,
        block_seconds=0.02,
    )
    rs.open()
    reader = threading.Thread(target=lambda: rs.read(1600), daemon=True)
    reader.start()
    assert read_started.wait(timeout=1.0)

    started = time.monotonic()
    closed = rs.close()
    elapsed = time.monotonic() - started

    assert closed is False
    assert abort_started.is_set()
    assert elapsed < 0.5
    assert native.close_calls == 0
    assert reader.is_alive()

    allow_abort.set()
    reader.join(timeout=1.0)
    assert not reader.is_alive()
    assert rs.close(wait_timeout=0.0)
    assert native.close_calls == 1
    assert native.close_overlapped is False


def test_close_retains_when_read_returns_but_native_abort_is_still_active():
    """Read-idle alone is insufficient: close must also wait for abort-idle."""
    read_started = threading.Event()
    abort_started = threading.Event()
    allow_abort = threading.Event()
    abort_done = threading.Event()
    release_read = threading.Event()

    class _AbortStallsAfterRead(_ScriptedStream):
        def __init__(self):
            super().__init__([])
            self.read_active = False
            self.abort_active = False
            self.close_overlapped = False

        def read(self, frames):
            self.read_active = True
            read_started.set()
            assert release_read.wait(timeout=2.0)
            self.read_active = False
            return np.zeros((frames, 1), dtype="float32"), False

        def abort(self):
            self.abort_calls += 1
            self.abort_active = True
            abort_started.set()
            assert allow_abort.wait(timeout=2.0)
            self.abort_active = False
            abort_done.set()

        def close(self):
            self.close_overlapped = self.read_active or self.abort_active
            super().close()

    native = _AbortStallsAfterRead()
    rs = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=lambda _device, _samplerate: native,
        block_seconds=0.02,
    )
    rs.open()
    reader = threading.Thread(target=lambda: rs.read(1600), daemon=True)
    reader.start()
    assert read_started.wait(timeout=1.0)

    def _release_read_during_abort():
        assert abort_started.wait(timeout=1.0)
        release_read.set()

    releaser = threading.Thread(target=_release_read_during_abort, daemon=True)
    releaser.start()
    started = time.monotonic()
    closed = rs.close()
    elapsed = time.monotonic() - started
    reader.join(timeout=1.0)

    assert closed is False
    assert elapsed < 0.5
    assert not reader.is_alive()
    assert native.close_calls == 0

    allow_abort.set()
    assert abort_done.wait(timeout=1.0)
    assert rs.close(wait_timeout=0.0)
    assert native.close_calls == 1
    assert native.close_overlapped is False


@pytest.mark.parametrize("failing_method", ["stop", "close"])
def test_native_teardown_exception_retains_stream_for_safe_retry(failing_method):
    """Neither stop() nor close() failure may detach/report success."""
    class _FailsOnce(_ScriptedStream):
        def __init__(self):
            super().__init__([])
            self.failed = False

        def stop(self):
            super().stop()
            if failing_method == "stop" and not self.failed:
                self.failed = True
                raise RuntimeError("stop failed")

        def close(self):
            self.close_calls += 1
            if failing_method == "close" and not self.failed:
                self.failed = True
                raise RuntimeError("close failed")

    native = _FailsOnce()
    rs = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=lambda _device, _samplerate: native,
        block_seconds=0.02,
    )
    rs.open()

    assert rs.close(teardown_timeout=0.2) is False
    assert rs._stream is native  # noqa: SLF001 - retained ownership contract
    assert rs.actual_samplerate == 16000

    assert rs.close(teardown_timeout=0.2) is True
    assert rs._stream is None  # noqa: SLF001
    assert native.stop_calls == 2
    assert native.close_calls == (1 if failing_method == "stop" else 2)


def test_abort_reservation_precedes_stream_selection_and_blocks_close_toctou():
    """Forced interleaving cannot detach between abort decision/reservation."""
    read_started = threading.Event()
    release_read = threading.Event()
    lifecycle_entered = threading.Event()
    allow_lifecycle = threading.Event()
    events = []

    class _Native(_ScriptedStream):
        def read(self, frames):
            read_started.set()
            assert release_read.wait(timeout=2.0)
            return np.zeros((frames, 1), dtype="float32"), False

        def abort(self):
            events.append("abort")
            super().abort()
            release_read.set()

        def stop(self):
            events.append("stop")
            super().stop()

        def close(self):
            events.append("close")
            super().close()

    class _GatedLifecycleLock:
        def __init__(self, delegate):
            self.delegate = delegate
            self.gated = False

        def __enter__(self):
            if threading.current_thread().name == "abort-reservation" and not self.gated:
                self.gated = True
                lifecycle_entered.set()
                assert allow_lifecycle.wait(timeout=2.0)
            self.delegate.acquire()
            return self

        def __exit__(self, *_args):
            self.delegate.release()

    native = _Native([])
    rs = _RecoveringInputStream(
        [OpenAttempt(device=None, samplerate=16000)],
        opener=lambda _device, _samplerate: native,
        block_seconds=0.02,
    )
    rs.open()
    rs._lifecycle_lock = _GatedLifecycleLock(rs._lifecycle_lock)  # noqa: SLF001
    reader = threading.Thread(target=lambda: rs.read(1600), daemon=True)
    reader.start()
    assert read_started.wait(timeout=1.0)

    abort_results = []
    aborter = threading.Thread(
        target=lambda: abort_results.append(rs.abort_read(timeout=0.5)),
        name="abort-reservation",
        daemon=True,
    )
    aborter.start()
    assert lifecycle_entered.wait(timeout=1.0)
    assert rs._active_aborts == 1  # noqa: SLF001 - reserved before selection

    close_results = []
    closer = threading.Thread(
        target=lambda: close_results.append(
            rs.close(wait_timeout=0.0, teardown_timeout=0.2)
        ),
        daemon=True,
    )
    closer.start()
    assert native.close_calls == 0
    allow_lifecycle.set()
    aborter.join(timeout=1.0)
    closer.join(timeout=1.0)
    reader.join(timeout=1.0)

    assert abort_results == [True]
    assert not aborter.is_alive() and not reader.is_alive()
    assert events[0] == "abort"
    if close_results == [False]:
        assert rs.close(teardown_timeout=0.2)
    else:
        assert close_results == [True]
    assert events.index("abort") < events.index("close")


# --- state-listener contract ---------------------------------------------


def test_state_listener_exceptions_do_not_crash_capture():
    """A buggy listener must not propagate -- the capture loop has to
    keep running even if the brain crashes processing the notification."""
    attempts = [OpenAttempt(device=None, samplerate=16000)]
    stream = _ScriptedStream([
        (np.zeros((100, 1), dtype="float32"), False)
    ])

    def boom(state, message):
        raise RuntimeError("listener broke")

    rs = _RecoveringInputStream(attempts, opener=lambda d, sr: stream, on_state=boom)
    rs.open()  # listener raises here; must not propagate
    samples, _ = rs.read(100)
    assert samples.shape == (100, 1)


def test_state_listener_sees_every_transition():
    states: list[tuple[StreamState, str]] = []
    attempts = [OpenAttempt(device=None, samplerate=16000)]
    bad_stream = _ScriptedStream([_make_err(PA_DEVICE_UNAVAILABLE)])
    good_stream = _ScriptedStream([])
    streams = iter([bad_stream, good_stream])

    rs = _RecoveringInputStream(
        attempts,
        opener=lambda d, sr: next(streams),
        on_state=lambda s, m: states.append((s, m)),
        backoffs=(0.0,),
        sleep_fn=lambda _t: None,
    )
    rs.open()
    rs.read(100)  # triggers reopen
    state_seq = [s for s, _ in states]
    # We expect at least OPEN, RECOVERING, OPEN; the implementation may
    # also re-emit OPEN at the initial open call -- both are fine.
    assert state_seq.count(StreamState.OPEN) >= 2
    assert state_seq.count(StreamState.RECOVERING) >= 1


def test_default_backoffs_total_within_design_budget():
    """Sanity: the default backoff list cumulates to ~6.4 s, matching
    the design doc (the wrapper survives a typical USB stumble but
    declares FATAL before the user thinks the assistant is broken)."""
    total = sum(DEFAULT_BACKOFFS)
    assert 5.0 < total < 8.0  # 6.2 currently, leave headroom for tuning
    # And the curve is monotonically increasing.
    assert list(DEFAULT_BACKOFFS) == sorted(DEFAULT_BACKOFFS)
