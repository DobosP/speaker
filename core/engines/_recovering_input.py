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
ALSA dmix shutdown (-9985), WASAPI exclusive-mode eviction (-9989 with
host_err_code = AUDCLNT_E_DEVICE_INVALIDATED), and CoreAudio aggregate-
device vanish (-9989 with kAudioHardwareBadDeviceError). Reopen is
always by enumeration, never by stale device index.
"""
from __future__ import annotations

import logging
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
PA_UNANTICIPATED_HOST_ERROR = -9989  # real code in host_err_msg

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
        self._stream: Any = None
        self._current: Optional[OpenAttempt] = None
        self._state: StreamState = StreamState.OPEN

    # --- public surface ---------------------------------------------------

    @property
    def actual_samplerate(self) -> int:
        """Sample rate of the currently open stream. Raises if not open."""
        if self._current is None:
            raise RuntimeError("input stream is not open")
        return self._current.samplerate

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
        last_exc: Optional[Exception] = None
        for attempt in self._attempts:
            try:
                self._stream = self._opener(attempt.device, attempt.samplerate)
                self._stream.start()
                self._current = attempt
                self._set_state(StreamState.OPEN, f"opened at {attempt.samplerate} Hz")
                return
            except Exception as exc:  # noqa: BLE001
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
        ``overflowed=True``. On REOPEN errors: invokes :meth:`_recover`
        which may raise FATAL after the backoff budget is exhausted."""
        if self._stream is None:
            raise RuntimeError("read() before open()")
        try:
            return self._stream.read(frames)
        except Exception as exc:  # narrowed to PortAudioError below
            return self._handle_read_error(exc, frames)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._current = None

    # --- error handling ---------------------------------------------------

    def _handle_read_error(self, exc: Exception, frames: int) -> tuple[Any, bool]:
        code = _err_code(exc)
        if code in TRANSIENT_CODES:
            log.warning("transient PortAudio error %d: %s -- dropping frame", code, exc)
            return self._silent_block(frames), True
        if code in REOPEN_CODES:
            self._recover(reason=f"PortAudio error {code}: {exc}")
            # After recovery, retry the read once.
            return self.read(frames)
        # Unknown shape -> bubble; the engine's outer except will surface it.
        log.error("unrecognized PortAudio error (code=%s): %s", code, exc)
        raise exc

    def _recover(self, *, reason: str) -> None:
        """Cycle through the backoff list, attempting to reopen on each tick.

        On success: state goes back to OPEN.
        On full exhaust: state becomes FATAL and the original error is raised.
        """
        self._set_state(StreamState.RECOVERING, reason)
        # Close the current stream so the device can be re-acquired.
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            pass
        self._stream = None

        last_exc: Optional[Exception] = None
        for delay in self._backoffs:
            if delay > 0:
                self._sleep(delay)
            try:
                self.open()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("reopen attempt failed after %.1fs: %s", delay, exc)
                continue
            # open() already set state to OPEN.
            log.info("input stream recovered (sr=%d)", self.actual_samplerate)
            return

        # All retries failed.
        self._set_state(StreamState.FATAL, f"recovery exhausted: {last_exc}")
        raise last_exc if last_exc is not None else RuntimeError("recovery failed")

    # --- helpers ----------------------------------------------------------

    def _silent_block(self, frames: int) -> Any:
        """Return a zero-filled (frames, channels) ndarray. numpy is lazy-
        imported because this module is otherwise dependency-free."""
        import numpy as np
        return np.zeros((frames, self.channels), dtype="float32")

    def _set_state(self, new_state: StreamState, message: str) -> None:
        if new_state != self._state:
            log.info("capture state: %s -> %s (%s)", self._state.value, new_state.value, message)
        self._state = new_state
        try:
            self._on_state(new_state, message)
        except Exception:  # noqa: BLE001 - listener errors must not crash capture
            log.exception("on_state listener raised; ignoring")
