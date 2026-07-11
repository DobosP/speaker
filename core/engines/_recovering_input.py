"""Recovering ``sd.InputStream`` wrapper.

The audited bug: ``core/engines/sherpa.py`` calls ``self._stream_in.read(...)``
unguarded. A USB unplug / sample-rate change / xrun raises
``sd.PortAudioError`` and kills the capture thread silently -- the
assistant just stops listening.

This wrapper sits behind the raw stream and classifies every
``PortAudioError`` it sees:

- **TRANSIENT** (overflow/underflow/start-stop race): drop the frame,
  return a silent block, continue.
- **REOPEN** (device unavailable / bad-stream-ptr / host error): retire
  the current stream and try to reopen with a configurable backoff
  + fallback chain over (device, sample_rate).

State transitions are published via a callable so the runtime can:

- raise an :class:`always_on_agent.events.AgentEvent` so the brain knows
  not to flag the silent gap as "user fell asleep"
- tell the watchdog to skip its "audio thread stalled" warning during
  legitimate reopens

PortAudio error codes are read from ``exc.args[1]`` (NOT ``.errno`` --
that's PyAudio's convention; sounddevice puts the code in the args
tuple). Verified against
https://raw.githubusercontent.com/spatialaudio/python-sounddevice/0.5.1/sounddevice.py
and https://files.portaudio.com/docs/v19-doxydocs-dev/portaudio_8h.html.

The wrapper is platform-agnostic by design -- the same code path covers
ALSA dmix shutdown (-9985), WASAPI exclusive-mode eviction (-9999 with
host_err_code = AUDCLNT_E_DEVICE_INVALIDATED), and CoreAudio aggregate-
device vanish (-9999 with kAudioHardwareBadDeviceError). Reopen is
always by enumeration, never by stale device index.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Sequence

log = logging.getLogger("speaker.engine.audio_recovery")


class StreamState(str, Enum):
    """Lifecycle of the input stream as seen from outside."""

    OPEN = "open"            # capture is healthy
    RECOVERING = "recovering"  # PortAudio error; trying to reopen
    FATAL = "fatal"          # recovery budget exhausted; capture is dead


# PortAudio error codes (PaErrorCode enum). Sourced from
# files.portaudio.com/docs/v19-doxydocs-dev/portaudio_8h.html
# and cross-checked against the integer values in portaudio-sharp/PaErrorCode.cs.
PA_INPUT_OVERFLOWED   = -9981  # ASR fell behind; drop the frame
PA_OUTPUT_UNDERFLOWED = -9980  # TTS thread starved the speaker
PA_STREAM_IS_STOPPED  = -9983  # benign: start() while already running
PA_STREAM_IS_NOT_STOPPED = -9982
PA_DEVICE_UNAVAILABLE = -9985  # USB unplug, BT disconnect, sleep/resume
PA_TIMED_OUT          = -9987  # CoreAudio aggregate vanish, WASAPI eviction
PA_BAD_STREAM_PTR     = -9988  # read-after-close
PA_INTERNAL_ERROR     = -9986  # host driver crash
PA_UNANTICIPATED_HOST_ERROR = -9999  # real code in host_err_msg

# Drop the frame; return silence; continue. The next read is expected to
# succeed.
TRANSIENT_CODES = frozenset({
    PA_INPUT_OVERFLOWED,
    PA_OUTPUT_UNDERFLOWED,
    PA_STREAM_IS_STOPPED,
    PA_STREAM_IS_NOT_STOPPED,
})
# Retire the stream and reopen it.
REOPEN_CODES = frozenset({
    PA_DEVICE_UNAVAILABLE,
    PA_TIMED_OUT,
    PA_BAD_STREAM_PTR,
    PA_INTERNAL_ERROR,
    PA_UNANTICIPATED_HOST_ERROR,
})

# Exponential-ish backoff before each reopen attempt. The cumulative time
# (~6.4 s) is intentionally longer than a typical USB-unplug-replug so
# the assistant survives a stumble; FATAL kicks in only when the device
# is genuinely gone.
DEFAULT_BACKOFFS: tuple[float, ...] = (0.2, 0.4, 0.8, 1.6, 3.2)


@dataclass(frozen=True)
class OpenAttempt:
    """One entry in the open-fallback chain: device + sample_rate."""

    device: Any            # None means "system default"; otherwise a device index/name
    samplerate: int


# Callback shape published to the runtime.
StateListener = Callable[[StreamState, str], None]


def _noop_listener(state: StreamState, message: str) -> None:
    pass


def _err_code(exc: Exception) -> Optional[int]:
    """Extract the PortAudio numeric code from an ``sd.PortAudioError``.

    sounddevice stores the code as the SECOND positional arg (the first
    is the message, the third is the host error triple). Returns None if
    the exception doesn't follow that shape -- callers should treat
    unknown shapes as fatal."""
    args = getattr(exc, "args", None)
    if not args or len(args) < 2:
        return None
    try:
        return int(args[1])
    except (TypeError, ValueError):
        return None


class _RecoveringInputStream:
    """Drop-in facade over ``sd.InputStream`` that recovers from PortAudio
    errors.

    Construction does NOT open the device -- call :meth:`open` (matches
    sounddevice's two-step pattern of construct-then-``start()``). The
    returned object exposes ``read(n) -> (ndarray, overflowed_bool)`` so
    callers don't change shape.

    ``opener`` is injectable so tests can produce scripted faults without
    touching PortAudio. Production passes the lazy ``sd.InputStream``
    factory at construction.

    ``attempts`` is the ordered fallback chain. The first attempt that
    opens cleanly wins; on a later REOPEN we walk the chain again from
    the top, on the theory that a USB device that briefly disappeared
    may now be available at the user's preferred rate.
    """

    def __init__(
        self,
        attempts: Sequence[OpenAttempt],
        *,
        opener: Callable[[Any, int], Any],
        on_state: StateListener = _noop_listener,
        backoffs: Sequence[float] = DEFAULT_BACKOFFS,
        channels: int = 1,
        block_seconds: float = 0.1,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if not attempts:
            raise ValueError("_RecoveringInputStream needs at least one open attempt")
        self._attempts = tuple(attempts)
        self._opener = opener
        self._on_state = on_state
        self._backoffs = tuple(backoffs)
        self.channels = channels
        self.block_seconds = block_seconds
        self._sleep = sleep_fn
        self._closed = threading.Event()
        # Never hold this condition across a native read.  It only linearizes
        # "a read is now active" against request_close(), then lets close() wait
        # a bounded interval without racing PortAudio's blocking read buffer.
        self._read_condition = threading.Condition()
        self._active_reads = 0
        self._active_aborts = 0
        self._active_closes = 0
        self._lifecycle_lock = threading.Lock()
        self._stream: Any = None
        self._current: Optional[OpenAttempt] = None
        self._state: StreamState = StreamState.OPEN
        self._generation = 0

    # --- public surface ---------------------------------------------------

    @property
    def actual_samplerate(self) -> int:
        """Sample rate of the currently open stream. Raises if not open."""
        if self._current is None:
            raise RuntimeError("input stream is not open")
        return self._current.samplerate

    @property
    def actual_device(self):
        """Selector of the attempt that is currently open (``None`` = default)."""
        if self._current is None:
            raise RuntimeError("input stream is not open")
        return self._current.device

    @property
    def generation(self) -> int:
        """Incremented after every successful initial open or reopen."""
        return self._generation

    @property
    def state(self) -> StreamState:
        return self._state

    def open(self) -> None:
        """Open the first attempt that works. Raises on full chain exhaust.

        Only ``OPEN`` state is published from here -- the FATAL transition
        belongs to ``_recover`` (which decides whether a chain failure
        is final after walking all backoffs) and to the initial caller
        in ``start()`` (whose first failed open should bubble as a
        startup error, not as a runtime FATAL)."""
        if self._closed.is_set():
            raise RuntimeError("input stream is closed")
        last_exc: Optional[Exception] = None
        for attempt in self._attempts:
            if self._closed.is_set():
                raise RuntimeError("input stream is closed")
            candidate = None
            try:
                candidate = self._opener(attempt.device, attempt.samplerate)
                candidate.start()
                with self._lifecycle_lock:
                    if self._closed.is_set():
                        cancelled = True
                    else:
                        cancelled = False
                        self._stream = candidate
                        self._current = attempt
                        self._generation += 1
                if cancelled:
                    raise RuntimeError("input stream is closed")
                if self._closed.is_set():
                    self._retire_stream()
                    raise RuntimeError("input stream is closed")
                self._set_state(StreamState.OPEN, f"opened at {attempt.samplerate} Hz")
                return
            except Exception as exc:  # noqa: BLE001
                if candidate is not None and candidate is not self._stream:
                    try:
                        candidate.close()
                    except Exception:
                        pass
                if self._closed.is_set():
                    raise RuntimeError("input stream is closed") from exc
                last_exc = exc
                log.warning(
                    "open failed for device=%r sr=%d: %s",
                    attempt.device, attempt.samplerate, exc,
                )
        # All attempts failed -- raise; caller (initial start() or
        # _recover()) decides what state transition to emit.
        raise last_exc if last_exc is not None else RuntimeError("no opens attempted")

    def read(self, frames: int) -> tuple[Any, bool]:
        """Read ``frames`` samples; transparently recover from PortAudio
        errors. Returns ``(samples, overflowed)`` like ``sd.InputStream.read``.

        On TRANSIENT errors: returns a zero-filled block with
        ``overflowed=True``. On REOPEN errors: invokes the bounded recovery path
        which may raise FATAL after the backoff budget is exhausted."""
        with self._read_condition:
            if self._closed.is_set():
                raise RuntimeError("read() on closed input stream")
            if self._stream is None:
                raise RuntimeError("read() before open()")
            stream = self._stream
            self._active_reads += 1
        try:
            return stream.read(frames)
        except Exception as exc:  # narrowed to PortAudioError below
            return self._handle_read_error(exc, frames)
        finally:
            with self._read_condition:
                self._active_reads -= 1
                if self._active_reads == 0:
                    self._read_condition.notify_all()

    def request_close(self) -> None:
        """Stop admission and recovery without touching the native stream.

        This is phase one of shutdown.  A capture owner can clear its running
        flag, call this method, and join the reader before :meth:`close`
        physically hands the device back.  Holding ``_read_condition`` makes
        the close request linear with admission of a new blocking read.
        """
        with self._read_condition:
            self._closed.set()

    def close(
        self,
        *,
        wait_timeout: Optional[float] = None,
        abort_timeout: Optional[float] = None,
        teardown_timeout: Optional[float] = None,
    ) -> bool:
        """Close the native stream after a bounded in-flight-read grace.

        Returns ``True`` only when every wrapper read quiesced and native
        stop/close completed successfully. The default read grace is one
        configured audio block. On
        timeout this method asks PortAudio to abort the read, waits once more,
        and closes only after the read owner quiesces.  If abort itself or the
        read remains stuck, the native handle is deliberately retained: a
        bounded leak is safer than racing ``close()`` against native code.
        """
        self.request_close()
        timeout = self.block_seconds if wait_timeout is None else wait_timeout
        quiesced = self._wait_for_reads(max(0.0, float(timeout)))
        if not quiesced:
            log.warning(
                "input read did not quiesce within %.3fs; requesting native abort",
                max(0.0, float(timeout)),
            )
            abort_wait = timeout if abort_timeout is None else abort_timeout
            self.abort_read(timeout=max(0.0, float(abort_wait)))
            quiesced = self._wait_for_reads(max(0.0, float(timeout)))
        if not quiesced:
            log.error(
                "input read remained active after bounded abort; retaining "
                "native stream instead of racing close()"
            )
            return False
        return self._close_native_stream(timeout=teardown_timeout)

    def abort_read(self, *, timeout: Optional[float] = None) -> bool:
        """Ask PortAudio to abort an active read within a bounded call budget.

        The native abort call runs on a disposable daemon because a broken host
        API can wedge even there.  This method never calls ``close()`` and never
        detaches the handle; the capture owner must quiesce before phase two.
        """
        self.request_close()
        budget = self.block_seconds if timeout is None else timeout
        budget = max(0.0, float(budget))
        wait_for_existing_abort = False
        with self._read_condition:
            active = self._active_reads > 0
            abort_active = self._active_aborts > 0
            close_active = self._active_closes > 0
            if not active:
                return not abort_active and not close_active
            if close_active:
                log.error("native input close is already active; not racing abort()")
                return False
            if abort_active:
                wait_for_existing_abort = True
                stream = None
            else:
                # Reserve abort ownership before selecting the native stream,
                # under the same condition->lifecycle lock order used by close.
                # Close therefore cannot observe zero aborts and detach between
                # our decision and native-handle selection.
                self._active_aborts += 1
                with self._lifecycle_lock:
                    stream = self._stream
        if wait_for_existing_abort:
            return self._wait_for_aborts(budget)
        abort = getattr(stream, "abort", None) if stream is not None else None
        if not callable(abort):
            with self._read_condition:
                self._active_aborts -= 1
                self._read_condition.notify_all()
            log.error("native input stream has no abort(); retaining active handle")
            return False

        completed = threading.Event()
        succeeded = []

        def _abort() -> None:
            try:
                abort()
                succeeded.append(True)
            except Exception:  # noqa: BLE001 - host may already be gone
                log.exception("native input abort failed")
            finally:
                with self._read_condition:
                    self._active_aborts -= 1
                    self._read_condition.notify_all()
                completed.set()

        abort_thread = threading.Thread(
            target=_abort,
            name="speaker-input-abort",
            daemon=True,
        )
        try:
            abort_thread.start()
        except Exception:
            with self._read_condition:
                self._active_aborts -= 1
                self._read_condition.notify_all()
            log.exception("could not start native input abort helper")
            return False
        if not completed.wait(budget):
            log.error(
                "native input abort did not return within %.3fs",
                budget,
            )
            return False
        return bool(succeeded)

    def _wait_for_aborts(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._read_condition:
            while self._active_aborts:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._read_condition.wait(remaining)
            return True

    def _wait_for_reads(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._read_condition:
            while self._active_reads:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._read_condition.wait(remaining)
            return True

    def _wait_for_closes(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._read_condition:
            while self._active_closes:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._read_condition.wait(remaining)
            return True

    def _close_native_stream(self, *, timeout: Optional[float] = None) -> bool:
        budget = self.block_seconds if timeout is None else timeout
        budget = max(0.0, float(budget))
        deadline = time.monotonic() + budget

        # A prior timed-out helper may still own native stop/close. Wait only the
        # caller's bounded budget. If it completed successfully, _stream is None;
        # if it failed, ownership remains and this call can safely retry below.
        with self._read_condition:
            close_active = self._active_closes > 0
        if close_active and not self._wait_for_closes(budget):
            log.error("native input teardown is still active; retaining stream")
            return False

        with self._read_condition:
            if self._active_reads or self._active_aborts or self._active_closes:
                log.error(
                    "retaining native input while lifecycle call is active "
                    "(reads=%d aborts=%d closes=%d)",
                    self._active_reads,
                    self._active_aborts,
                    self._active_closes,
                )
                return False
            # Reserve teardown before selecting the stream. abort_read() uses the
            # same condition->lifecycle order, so neither can slip between a zero
            # counter observation and native-handle selection.
            self._active_closes += 1
            with self._lifecycle_lock:
                stream = self._stream
            if stream is None:
                self._active_closes -= 1
                self._read_condition.notify_all()
                return True

        completed = threading.Event()
        succeeded = []

        def _teardown() -> None:
            ok = False
            try:
                stream.stop()
                stream.close()
                ok = True
            except Exception:  # noqa: BLE001 - retain for a safe later retry
                log.exception("native input stop/close failed; retaining stream")
            finally:
                with self._read_condition:
                    if ok:
                        with self._lifecycle_lock:
                            if self._stream is stream:
                                self._stream = None
                                self._current = None
                        succeeded.append(True)
                    self._active_closes -= 1
                    self._read_condition.notify_all()
                completed.set()

        teardown_thread = threading.Thread(
            target=_teardown,
            name="speaker-input-close",
            daemon=True,
        )
        try:
            teardown_thread.start()
        except Exception:
            with self._read_condition:
                self._active_closes -= 1
                self._read_condition.notify_all()
            log.exception("could not start native input teardown helper")
            return False
        remaining = max(0.0, deadline - time.monotonic())
        if not completed.wait(remaining):
            log.error(
                "native input stop/close did not return within %.3fs; "
                "retaining stream",
                budget,
            )
            return False
        return bool(succeeded)

    # --- error handling ---------------------------------------------------

    def _handle_read_error(self, exc: Exception, frames: int) -> tuple[Any, bool]:
        code = _err_code(exc)
        if code in TRANSIENT_CODES:
            log.warning("transient PortAudio error %d: %s -- dropping frame", code, exc)
            return self._silent_block(frames), True
        if code in REOPEN_CODES:
            return self._recover_and_read(reason=f"PortAudio error {code}: {exc}")
        # Unknown shape -> bubble; the engine's outer except will surface it.
        log.error("unrecognized PortAudio error (code=%s): %s", code, exc)
        raise exc

    def _recover_and_read(self, *, reason: str) -> tuple[Any, bool]:
        """Reopen and validate candidates within one bounded recovery budget.

        A stream is not considered recovered merely because ``start()`` worked:
        some stale USB/host handles fail on their first read. Probe one correctly
        timed block from each candidate, continue through the fallback chain on
        failure, and publish OPEN/generation only after a usable read. This avoids
        both premature fatal and recursive recovery-budget resets.
        """
        if self._closed.is_set():
            raise RuntimeError("input stream closed during recovery")
        self._set_state(StreamState.RECOVERING, reason)
        if not self._retire_stream():
            raise RuntimeError("input stream closed during recovery")

        last_exc: Optional[Exception] = None
        for delay in self._backoffs:
            if self._closed.is_set():
                raise RuntimeError("input stream closed during recovery")
            if delay > 0:
                if self._sleep is time.sleep:
                    self._closed.wait(delay)
                else:
                    self._sleep(delay)
            if self._closed.is_set():
                raise RuntimeError("input stream closed during recovery")
            for attempt in self._attempts:
                if self._closed.is_set():
                    raise RuntimeError("input stream closed during recovery")
                candidate = None
                try:
                    candidate = self._opener(attempt.device, attempt.samplerate)
                    candidate.start()
                    with self._lifecycle_lock:
                        if self._closed.is_set():
                            cancelled = True
                        else:
                            cancelled = False
                            self._stream = candidate
                            self._current = attempt
                    if cancelled:
                        raise RuntimeError("input stream closed during recovery")
                    recovered_frames = max(
                        1, int(attempt.samplerate * self.block_seconds)
                    )
                    try:
                        result = candidate.read(recovered_frames)
                    except Exception as read_exc:  # noqa: BLE001
                        if _err_code(read_exc) in TRANSIENT_CODES:
                            result = self._silent_block(recovered_frames), True
                        else:
                            raise
                    with self._lifecycle_lock:
                        if self._closed.is_set() or self._stream is not candidate:
                            committed = False
                        else:
                            self._generation += 1
                            committed = True
                    if not committed:
                        raise RuntimeError("input stream closed during recovery")
                    self._set_state(
                        StreamState.OPEN,
                        f"opened at {attempt.samplerate} Hz",
                    )
                    log.info(
                        "input stream recovered (device=%r sr=%d)",
                        attempt.device,
                        attempt.samplerate,
                    )
                    return result
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if self._closed.is_set():
                        if candidate is not None and candidate is not self._stream:
                            # Shutdown landed while candidate.start() was in
                            # flight. It never became the readable owner, so it
                            # is safe (and necessary) to release it here.
                            try:
                                candidate.close()
                            except Exception:
                                pass
                        # Phase two owns the physical stream handback.  In the
                        # assigned/readable path it may already own this candidate;
                        # never issue a second native close from the read owner.
                        raise RuntimeError(
                            "input stream closed during recovery"
                        ) from exc
                    if candidate is self._stream:
                        self._retire_stream()
                    elif candidate is not None:
                        try:
                            candidate.close()
                        except Exception:
                            pass
                    log.warning(
                        "reopen candidate failed after %.1fs device=%r sr=%d: %s",
                        delay,
                        attempt.device,
                        attempt.samplerate,
                        exc,
                    )

        # All retries failed.
        self._set_state(StreamState.FATAL, f"recovery exhausted: {last_exc}")
        raise last_exc if last_exc is not None else RuntimeError("recovery failed")

    # --- helpers ----------------------------------------------------------

    def _silent_block(self, frames: int) -> Any:
        """Return a zero-filled (frames, channels) ndarray. numpy is lazy-
        imported because this module is otherwise dependency-free."""
        import numpy as np
        return np.zeros((frames, self.channels), dtype="float32")

    def _retire_stream(self) -> bool:
        """Detach and close the current stream without closing the wrapper."""
        with self._read_condition:
            if (
                self._closed.is_set()
                or self._active_aborts
                or self._active_closes
            ):
                return False
            self._active_closes += 1
            with self._lifecycle_lock:
                stream = self._stream
        try:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    log.exception("recovery could not retire native input stream")
                    return False
                with self._lifecycle_lock:
                    if self._stream is stream:
                        self._stream = None
                        self._current = None
            return True
        finally:
            with self._read_condition:
                self._active_closes -= 1
                self._read_condition.notify_all()

    def _set_state(self, new_state: StreamState, message: str) -> None:
        if new_state != self._state:
            log.info("capture state: %s -> %s (%s)", self._state.value, new_state.value, message)
        self._state = new_state
        try:
            self._on_state(new_state, message)
        except Exception:  # noqa: BLE001 - listener errors must not crash capture
            log.exception("on_state listener raised; ignoring")
