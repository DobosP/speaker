"""Remote :class:`AudioEngine` that swaps the local mic/speaker for a LiveKit
(WebRTC) room, so a browser/phone becomes a thin client of one running brain.

It is the remote analogue of :class:`core.engines.sherpa.SherpaOnnxEngine`: the
same on-device sherpa-onnx streaming ASR + VITS TTS, but audio arrives from a
room track instead of ``sounddevice`` and TTS is published back as a room track.
Because it is just another ``AudioEngine``, the whole control-plane brain
(``VoiceRuntime`` + ``always_on_agent``: modes, router, agent, barge-in,
cancellation) is reused unchanged.

Fully-local stays intact: STT/LLM/TTS run on-device; LiveKit is only transport
and can be self-hosted (``livekit-server --dev``). ``livekit`` (rtc) and
``numpy`` are imported lazily so this module imports without them installed --
install ``requirements-remote.txt`` to actually run it.

Threading model (mirrors the local engine): a background thread owns an asyncio
loop that does only room I/O; incoming frames are resampled to 16 kHz and handed
to a dedicated ASR worker thread that drives the streaming recognizer (so model
decode never blocks the event loop). ``speak`` schedules TTS onto the loop via
``run_coroutine_threadsafe``.

The live room/audio path needs a real LiveKit server + a client and so cannot be
verified headless; the pure audio helpers and the ASR/barge-in stepping are
unit-tested with fakes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from always_on_agent.acoustic import AcousticSource, EndpointReason

from ..engine import (
    AudioEngine,
    EngineCallbacks,
    FinalTranscript,
    NO_PLAYBACK_CAPABILITIES,
    PartialTranscript,
    PlaybackCapabilities,
    PlaybackOutcome,
    PlaybackReceipt,
    TrackedSpeech,
    TranscriptAbort,
    TranscriptAbortReason,
)
from ._acoustic_turn import AcousticTurnTracker
from ._sherpa_models import build_recognizer, build_tts, build_vad
from .sherpa import SherpaConfig

log = logging.getLogger("speaker.livekit")

OUT_SR = 48000  # LiveKit-friendly output sample rate for published TTS
STT_SR = 16000  # sherpa streaming-ASR input rate
FRAME_MS = 20   # published audio frame size


_LIVEKIT_PLAYBACK_CAPABILITIES = PlaybackCapabilities(
    tracked_terminal=True,
    exact_started=True,
    sample_counts=False,
    speech_style_hints=False,
)


@dataclass(eq=False)
class _LiveKitPlaybackCall:
    """One accepted output fragment, keyed by admission identity.

    ``fragment_id`` is deliberately not the registry key: callers may reuse an
    opaque ID, and a late coroutine from the older call must not claim the new
    call's ownership. Mutable fields are guarded by ``_playback_lock``.
    """

    text: str
    generation: int = -1
    tracked: Optional[TrackedSpeech] = None
    on_terminal: Optional[Callable[[PlaybackReceipt], None]] = None
    on_started: Optional[Callable[[str], None]] = None
    on_done: Optional[Callable[[], None]] = None
    phase: str = "queued"
    source: object = None
    future: object = None
    started: bool = False
    terminal: bool = False


# -- pure audio helpers (unit-tested) ---------------------------------------
def pcm_int16_to_float32(pcm):
    import numpy as np

    return np.asarray(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm_int16(samples):
    import numpy as np

    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def resample_linear(samples, src_sr: int, dst_sr: int):
    """Linear-interpolate float32 mono audio from ``src_sr`` to ``dst_sr``."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32)
    if src_sr == dst_sr or x.size == 0:
        return x
    n_out = int(round(x.shape[0] * float(dst_sr) / float(src_sr)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0.0, x.shape[0] - 1, num=n_out)
    return np.interp(src_idx, np.arange(x.shape[0]), x).astype(np.float32)


class LiveKitEngine(AudioEngine):
    """``AudioEngine`` bridging a LiveKit room to sherpa-onnx STT/TTS.

    ``url``/``token`` authenticate the agent into a room (the token encodes the
    room grant; ``room`` is kept for logging). Build models from the same
    ``SherpaConfig`` the local engine uses.
    """

    def __init__(
        self,
        sherpa_config: SherpaConfig,
        *,
        url: str,
        token: str,
        room: str = "assistant",
        out_sample_rate: int = OUT_SR,
        queue_max: int = 200,
    ):
        self._cfg = sherpa_config
        self._url = url
        self._token = token
        self._room_name = room
        self._out_sr = out_sample_rate
        self._barge_in_min_speech_sec = float(
            getattr(sherpa_config, "barge_in_min_speech_sec", 0.2)
        )

        self._cb = EngineCallbacks()
        self._recognizer = None
        self._vad = None
        self._tts = None
        self._asr_stream = None
        self._last_partial = ""
        self._voiced_run = 0.0
        self._acoustic_turn = AcousticTurnTracker(
            "livekit-remote",
            source=AcousticSource.REMOTE_TRACK,
        )

        self._in_q: "queue.Queue" = queue.Queue(maxsize=queue_max)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._room = None
        self._source = None
        self._asr_thread: Optional[threading.Thread] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._session_task: Optional[asyncio.Task] = None
        self._session_cancel_requested = threading.Event()
        self._session_teardown = threading.Event()
        self._track_tasks: set[asyncio.Task] = set()
        self._running = threading.Event()
        self._speaking = threading.Event()
        self._output_ready = threading.Event()
        # Ordered async locks pipeline the next synthesis while the current
        # sentence drains, then serialize its frames on the room track. This
        # avoids a synthesis-sized silence between streamed sentences. Two
        # slots bound memory to the active sentence plus one prepared successor.
        self._playback_slots: Optional[asyncio.Semaphore] = None
        self._synth_order_lock: Optional[asyncio.Lock] = None
        self._speak_lock: Optional[asyncio.Lock] = None
        # Sync guard so the startup warm pass and the to_thread synthesis never
        # call ``tts.generate`` on the single model concurrently.
        self._tts_lock = threading.Lock()
        # Playback admission and terminal ownership cross the runtime thread,
        # LiveKit loop, and TTS worker. A generation fences every call admitted
        # before STOP; a clear barrier prevents a delayed old clear from erasing
        # fresh-generation audio.
        self._playback_lock = threading.RLock()
        self._playback_generation = 0
        self._clear_requested_generation = -1
        self._clear_applied_generation = -1
        self._pending_playback: set[_LiveKitPlaybackCall] = set()
        self._playback_futures: set[object] = set()
        self._failed_playback_sources: dict[int, object] = {}

    # --- AudioEngine ---
    def start(self, callbacks: EngineCallbacks) -> None:
        self._cb = callbacks
        self._output_ready.clear()
        self._session_cancel_requested.clear()
        self._session_teardown.clear()
        self._recognizer = build_recognizer(self._cfg)
        self._vad = build_vad(self._cfg)
        self._tts = build_tts(self._cfg)
        self._running.set()
        self._asr_thread = threading.Thread(target=self._asr_loop, daemon=True)
        self._asr_thread.start()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._output_ready.clear()
        self._fence_playback()
        loop = self._loop
        session_task = self._session_task
        with self._playback_lock:
            cancel_session = (
                not self._session_cancel_requested.is_set()
                and not self._session_teardown.is_set()
            )
            if cancel_session:
                self._session_cancel_requested.set()
        if (
            cancel_session
            and loop is not None
            and session_task is not None
            and not session_task.done()
        ):
            def cancel_before_teardown() -> None:
                if not self._session_teardown.is_set():
                    session_task.cancel()

            try:
                on_owning_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                on_owning_loop = False
            if on_owning_loop:
                cancel_before_teardown()
            elif not loop.is_closed() and loop.is_running():
                loop.call_soon_threadsafe(cancel_before_teardown)
        if (
            self._asr_thread is not None
            and self._asr_thread is not threading.current_thread()
        ):
            self._asr_thread.join(timeout=1.0)
        if (
            self._loop_thread is not None
            and self._loop_thread is not threading.current_thread()
        ):
            self._loop_thread.join(timeout=2.0)
            if self._loop_thread.is_alive():
                log.warning(
                    "LiveKit loop retained while native playback work unwinds"
                )

    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        self._submit_playback(
            _LiveKitPlaybackCall(
                text=text,
                on_done=on_done,
            )
        )

    def stop_speaking(self) -> None:
        self._fence_playback()

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def playback_capabilities(self) -> PlaybackCapabilities:
        # Preserve subclasses that historically customized only legacy speak().
        # Silently bypassing that override through inherited tracked playback
        # would change their instrumentation and behavior.
        if (
            type(self).speak is not LiveKitEngine.speak
            and type(self).speak_tracked is LiveKitEngine.speak_tracked
        ):
            return NO_PLAYBACK_CAPABILITIES
        return _LIVEKIT_PLAYBACK_CAPABILITIES

    def speak_tracked(
        self,
        speech: TrackedSpeech,
        *,
        on_terminal: Callable[[PlaybackReceipt], None],
        on_started: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._submit_playback(
            _LiveKitPlaybackCall(
                text=speech.text,
                tracked=speech,
                on_terminal=on_terminal,
                on_started=on_started,
            )
        )

    def _submit_playback(self, call: _LiveKitPlaybackCall) -> None:
        """Admit one call and arrange an exactly-once terminal path."""
        with self._playback_lock:
            call.generation = self._playback_generation
            self._pending_playback.add(call)
            running = self._running.is_set()
            loop = self._loop
            tts = self._tts

        if not running:
            self._terminalize(call, PlaybackOutcome.DROPPED)
            return
        if (
            tts is None
            or loop is None
            or loop.is_closed()
            or not loop.is_running()
        ):
            self._terminalize(call, PlaybackOutcome.FAILED)
            return
        with self._playback_lock:
            if call.terminal:
                return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._playback_worker(call), loop
            )
        except Exception:  # noqa: BLE001 - event-loop admission may race teardown
            self._terminalize(call, PlaybackOutcome.FAILED)
            return
        with self._playback_lock:
            self._playback_futures.add(future)
            call.future = future
        future.add_done_callback(
            lambda completed, ticket=call: self._observe_playback_future(
                ticket, completed
            )
        )

    def _observe_playback_future(self, call, future) -> None:
        """Consume future errors and close any path the worker missed."""
        try:
            if future.cancelled():
                with self._playback_lock:
                    native_running = (
                        not call.terminal and call.phase == "synthesizing"
                    )
                if not native_running:
                    self._terminalize(
                        call,
                        self._noncompletion_outcome(call),
                    )
                return
            error = future.exception()
        except Exception:  # noqa: BLE001 - foreign Future implementations
            error = True
        finally:
            with self._playback_lock:
                self._playback_futures.discard(future)
        if error is not None:
            self._terminalize(call, self._noncompletion_outcome(call))

    def _fenced_outcome(self, call: _LiveKitPlaybackCall) -> PlaybackOutcome:
        with self._playback_lock:
            return (
                PlaybackOutcome.INTERRUPTED
                if call.started
                else PlaybackOutcome.DROPPED
            )

    def _noncompletion_outcome(
        self,
        call: _LiveKitPlaybackCall,
    ) -> PlaybackOutcome:
        """Distinguish a control fence from an adapter/source failure."""
        with self._playback_lock:
            fenced = (
                call.generation != self._playback_generation
                or not self._running.is_set()
            )
            if fenced:
                return (
                    PlaybackOutcome.INTERRUPTED
                    if call.started
                    else PlaybackOutcome.DROPPED
                )
            return PlaybackOutcome.FAILED

    @staticmethod
    def _source_is_open(source) -> bool:
        # LiveKit 1.1.10 capture_frame() silently returns for a disposed source.
        # The handle is private, so unknown/fake sources remain usable; the
        # pinned adapter receives an explicit fail-closed disposed check.
        handle = getattr(source, "_ffi_handle", None)
        return not bool(getattr(handle, "disposed", False))

    async def _await_output_source(self, call: _LiveKitPlaybackCall):
        """Wait for connect+publish without synthesizing an unrouteable reply."""
        while True:
            with self._playback_lock:
                if call.terminal:
                    return None
                fenced = (
                    call.generation != self._playback_generation
                    or not self._running.is_set()
                )
                ready = self._output_ready.is_set()
                source = self._source
            if fenced:
                self._terminalize(call, self._fenced_outcome(call))
                return None
            if ready:
                if source is None or not self._source_is_open(source):
                    self._terminalize(call, PlaybackOutcome.FAILED)
                    return None
                return source
            await asyncio.sleep(0.01)

    def _is_current(
        self,
        call: _LiveKitPlaybackCall,
        source=None,
    ) -> bool:
        with self._playback_lock:
            return (
                not call.terminal
                and call.generation == self._playback_generation
                and self._running.is_set()
                and self._output_ready.is_set()
                and (source is None or source is self._source)
                and (source is None or self._source_is_open(source))
            )

    def _mark_started(self, call: _LiveKitPlaybackCall) -> bool:
        """Claim start after a successful frame admission, then call outward."""
        with self._playback_lock:
            if call.terminal or call.started:
                return False
            call.started = True
            call.phase = "capturing"
            self._speaking.set()
            tracked = call.tracked
            on_started = call.on_started
        try:
            self._cb.on_speech_start()
        except Exception:  # noqa: BLE001 - isolate runtime callbacks
            log.exception("LiveKit speech-start callback raised")
        if tracked is not None and on_started is not None:
            try:
                on_started(tracked.fragment_id)
            except Exception:  # noqa: BLE001 - isolate receipt consumers
                log.exception("LiveKit tracked on_started callback raised")
        return True

    def _terminalize(
        self,
        call: _LiveKitPlaybackCall,
        outcome: PlaybackOutcome,
        *,
        require_current: bool = False,
        source=None,
    ) -> bool:
        """Claim one terminal result and invoke callbacks outside the lock."""
        with self._playback_lock:
            if call.terminal:
                return False
            if require_current and (
                call.generation != self._playback_generation
                or not self._running.is_set()
                or not self._output_ready.is_set()
                or source is not self._source
                or not self._source_is_open(source)
            ):
                return False
            call.terminal = True
            call.phase = "terminal"
            self._pending_playback.discard(call)
            started = call.started
            if started:
                self._speaking.clear()
            tracked = call.tracked
            on_terminal = call.on_terminal
            on_done = call.on_done

        if started:
            try:
                self._cb.on_speech_end()
            except Exception:  # noqa: BLE001 - isolate runtime callbacks
                log.exception("LiveKit speech-end callback raised")
        if tracked is not None and on_terminal is not None:
            try:
                on_terminal(
                    PlaybackReceipt(
                        fragment_id=tracked.fragment_id,
                        outcome=outcome,
                        safe_text_prefix=(
                            call.text
                            if outcome is PlaybackOutcome.COMPLETED
                            else ""
                        ),
                    )
                )
            except Exception:  # noqa: BLE001 - one consumer cannot poison output
                log.exception("LiveKit tracked terminal callback raised")
        if on_done is not None:
            try:
                on_done()
            except Exception:  # noqa: BLE001 - legacy callback is isolated too
                log.exception("LiveKit legacy on_done callback raised")
        return True

    def _fence_playback(self) -> None:
        """Fence old admissions and request an ordered source-queue clear."""
        with self._playback_lock:
            old_generation = self._playback_generation
            self._playback_generation += 1
            self._clear_requested_generation = max(
                self._clear_requested_generation,
                old_generation,
            )
            queued = tuple(
                call
                for call in self._pending_playback
                if call.generation <= old_generation
                and call.phase in {"queued", "ready"}
            )
            source = self._source
            loop = self._loop

        clear_scheduled = False
        if loop is None or loop.is_closed() or not loop.is_running():
            future = None
        else:
            try:
                on_owning_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                on_owning_loop = False
            try:
                if on_owning_loop:
                    self._clear_source_through_now(old_generation, source)
                    clear_scheduled = True
                    future = None
                else:
                    future = asyncio.run_coroutine_threadsafe(
                        self._clear_source_through(old_generation, source),
                        loop,
                    )
                    clear_scheduled = True
            except Exception:  # noqa: BLE001 - loop/source may race teardown
                future = None

        # Queue the physical cut before invoking potentially slow/re-entrant
        # consumer callbacks. Synthesizing jobs remain owned until their native
        # to_thread call really returns.
        for call in queued:
            self._terminalize(call, PlaybackOutcome.DROPPED)

        if not clear_scheduled:
            self._schedule_source_failure(source)
            return
        if future is not None:
            future.add_done_callback(
                lambda completed, stopped_source=source: (
                    self._observe_clear_future(stopped_source, completed)
                )
            )

    def _observe_clear_future(self, source, future) -> None:
        try:
            error = None if future.cancelled() else future.exception()
        except Exception:  # noqa: BLE001 - foreign Future implementations
            error = True
        if future.cancelled() or error is not None:
            self._schedule_source_failure(source)

    def _begin_source_failure(self, source) -> bool:
        with self._playback_lock:
            key = id(source)
            if (
                key in self._failed_playback_sources
                and self._failed_playback_sources[key] is source
            ):
                return False
            self._failed_playback_sources[key] = source
            old_generation = self._playback_generation
            self._playback_generation += 1
            self._clear_requested_generation = max(
                self._clear_requested_generation,
                old_generation,
            )
            self._running.clear()
            self._output_ready.clear()
        return True

    def _finish_source_failure(
        self,
        source,
        *,
        origin: Optional[_LiveKitPlaybackCall] = None,
    ) -> None:
        with self._playback_lock:
            if self._source is source:
                self._source = None
            calls = tuple(
                call
                for call in self._pending_playback
                if call.phase != "synthesizing"
            )
        for call in calls:
            with self._playback_lock:
                phase = call.phase
                future = call.future
                started = call.started
            outcome = (
                PlaybackOutcome.FAILED
                if started or phase in {"capturing", "waiting"}
                else PlaybackOutcome.DROPPED
            )
            self._terminalize(call, outcome)
            if call is not origin and future is not None and not future.done():
                future.cancel()

    @staticmethod
    def _dispose_source_handle(source) -> None:
        handle = getattr(source, "_ffi_handle", None)
        dispose = getattr(handle, "dispose", None)
        if callable(dispose) and not bool(getattr(handle, "disposed", False)):
            dispose()

    async def _fail_playback_source(
        self,
        source,
        *,
        origin: Optional[_LiveKitPlaybackCall] = None,
    ) -> None:
        """Make a failed clear transport-fatal before issuing receipts."""
        if not self._begin_source_failure(source):
            return
        aclose = getattr(source, "aclose", None)
        try:
            if callable(aclose):
                await aclose()
        except Exception:
            log.exception("LiveKit AudioSource close after clear failure raised")
        try:
            self._dispose_source_handle(source)
        except Exception:
            log.exception("LiveKit AudioSource dispose fallback raised")
        self._finish_source_failure(source, origin=origin)

    def _schedule_source_failure(self, source) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed() and loop.is_running():
            try:
                owning_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                owning_loop = False
            try:
                if owning_loop:
                    asyncio.create_task(self._fail_playback_source(source))
                else:
                    asyncio.run_coroutine_threadsafe(
                        self._fail_playback_source(source),
                        loop,
                    )
                return
            except Exception:
                log.exception("failed to schedule LiveKit source disposal")

        # The loop is unavailable, so use the pinned SDK handle's synchronous
        # disposal as the last fail-closed path before callbacks.
        if not self._begin_source_failure(source):
            return
        try:
            self._dispose_source_handle(source)
        except Exception:
            log.exception("LiveKit AudioSource synchronous dispose raised")
        self._finish_source_failure(source)

    def _clear_source_through_now(self, generation: int, source) -> None:
        """Apply one loop-owned stop barrier; late copies are harmless."""
        with self._playback_lock:
            if generation <= self._clear_applied_generation:
                return
            current_source = self._source
        if source is not None and source is current_source:
            clear_queue = getattr(source, "clear_queue", None)
            if not callable(clear_queue):
                raise RuntimeError("LiveKit AudioSource lacks clear_queue()")
            clear_queue()
        with self._playback_lock:
            self._clear_applied_generation = max(
                self._clear_applied_generation,
                generation,
            )

    async def _clear_source_through(self, generation: int, source) -> None:
        self._clear_source_through_now(generation, source)

    async def _ensure_clear_barrier(
        self,
        call: _LiveKitPlaybackCall,
        source,
    ) -> None:
        with self._playback_lock:
            target = min(
                self._clear_requested_generation,
                call.generation - 1,
            )
        if target >= 0:
            await self._clear_source_through(target, source)

    async def _finish_noncompletion(
        self,
        call: _LiveKitPlaybackCall,
        source,
    ) -> None:
        """Clear any possibly admitted audio before releasing ownership."""
        outcome = self._noncompletion_outcome(call)
        with self._playback_lock:
            needs_clear = call.started or call.phase in {"capturing", "waiting"}
            failure_is_current = (
                outcome is PlaybackOutcome.FAILED
                and call.generation == self._playback_generation
                and self._running.is_set()
            )
        if needs_clear:
            # A current adapter failure must also fence siblings; otherwise its
            # clear could erase a later same-generation fragment.
            if failure_is_current:
                self._fence_playback()
            try:
                await self._clear_source_through(call.generation, source)
            except Exception:
                await self._fail_playback_source(source, origin=call)
                return
        self._terminalize(call, outcome)

    def _synth_locked(self, text: str):
        """Synthesize under the TTS lock (the only thing that touches the model)."""
        with self._tts_lock:
            return self._tts.generate(
                text, sid=self._cfg.tts_speaker_id, speed=self._cfg.tts_speed
            )

    def warm(self) -> None:
        """Pay the TTS model's cold-start cost before the first reply, off the
        hot path. Mirrors the sherpa engine: synthesize a throwaway phrase and
        discard it, under the TTS lock so it can't race a live synthesis. The
        recognizer/VAD self-warm via the capture path. Best-effort."""
        if self._tts is None or self._speaking.is_set():
            return
        try:
            self._synth_locked("ok")
            log.info("livekit TTS warm-up complete")
        except Exception:  # noqa: BLE001 - warm-up is best-effort, never fatal
            log.debug("livekit TTS warm-up failed", exc_info=True)

    # --- ASR worker thread (keeps model decode off the event loop) ---
    def _asr_loop(self) -> None:
        rec = self._recognizer
        self._asr_stream = rec.create_stream() if rec is not None else None
        while self._running.is_set():
            try:
                mono = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._speaking.is_set():
                self._watch_barge_in(mono)
            else:
                self._feed_asr(mono)

    def _feed_asr(self, mono) -> None:
        """Feed one 16 kHz mono float32 frame into the streaming recognizer and
        raise ``on_partial``/``on_final``. Pure logic; unit-tested with a fake."""
        rec = self._recognizer
        stream = self._asr_stream
        if rec is None or stream is None:
            return
        stream.accept_waveform(STT_SR, mono)
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        text = rec.get_result(stream)
        if text and text != self._last_partial:
            self._last_partial = text
            emitted_at = time.perf_counter()
            acoustic, revision = self._acoustic_turn.partial(
                emitted_at=emitted_at,
            )
            if self._cb.on_partial_result is not None:
                self._cb.on_partial_result(
                    PartialTranscript(
                        text=text,
                        acoustic=acoustic,
                        revision=revision,
                    )
                )
            else:
                self._cb.on_partial(text)
        if rec.is_endpoint(stream):
            final_text = rec.get_result(stream)
            rec.reset(stream)
            self._last_partial = ""
            if final_text.strip():
                emitted_at = time.perf_counter()
                acoustic, revision = self._acoustic_turn.close(
                    speech_end_at=emitted_at,
                    endpoint_committed_at=emitted_at,
                    emitted_at=emitted_at,
                    endpoint_reason=EndpointReason.ASR,
                )
                if self._cb.on_final_result is not None:
                    self._cb.on_final_result(
                        FinalTranscript(
                            text=final_text,
                            acoustic=acoustic,
                            revision=revision,
                        )
                    )
                else:
                    self._cb.on_final(final_text)
                self._publish("user_transcript", {"text": final_text})
            else:
                aborted = self._acoustic_turn.abort(
                    emitted_at=time.perf_counter(),
                )
                if (
                    aborted is not None
                    and self._cb.on_transcript_abort is not None
                ):
                    acoustic, revision = aborted
                    self._cb.on_transcript_abort(
                        TranscriptAbort(
                            acoustic=acoustic,
                            revision=revision,
                            reason=TranscriptAbortReason.EMPTY_FINAL,
                        )
                    )

    def _watch_barge_in(self, mono) -> None:
        """While the assistant speaks, treat sustained user voice as barge-in."""
        vad = self._vad
        if vad is None:
            return
        vad.accept_waveform(mono)
        if vad.is_speech_detected():
            self._voiced_run += getattr(mono, "size", len(mono)) / STT_SR
            if self._voiced_run >= self._barge_in_min_speech_sec:
                self._voiced_run = 0.0
                self._cb.on_barge_in()
        else:
            self._voiced_run = 0.0

    # --- asyncio room loop ---
    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._playback_slots = asyncio.Semaphore(2)
        self._synth_order_lock = asyncio.Lock()
        self._speak_lock = asyncio.Lock()
        self._session_task = self._loop.create_task(self._session())
        if self._session_cancel_requested.is_set():
            self._session_task.cancel()
        try:
            self._loop.run_until_complete(self._session_task)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - needs a live server
            print(f"[livekit] session error: {exc}")
        finally:
            self._session_task = None
            self._loop.close()
            self._loop = None

    def _track_task_done(self, task: asyncio.Task) -> None:
        self._track_tasks.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except Exception:
            error = None
        if error is not None:
            log.warning("LiveKit subscribed-track reader failed: %s", error)

    async def _session(self) -> None:  # pragma: no cover - needs a live server
        room = None
        source = None
        try:
            from livekit import rtc

            room = rtc.Room()
            self._room = room
            source = rtc.AudioSource(self._out_sr, 1)
            self._source = source
            track = rtc.LocalAudioTrack.create_audio_track(
                "assistant-voice",
                source,
            )

            @room.on("track_subscribed")
            def _on_track(track_, publication, participant):
                if (
                    self._running.is_set()
                    and track_.kind == rtc.TrackKind.KIND_AUDIO
                ):
                    task = asyncio.create_task(self._read_track(track_))
                    self._track_tasks.add(task)
                    task.add_done_callback(self._track_task_done)

            @room.on("reconnecting")
            def _on_reconnecting():
                self._output_ready.clear()
                self._fence_playback()

            @room.on("reconnected")
            def _on_reconnected():
                if self._running.is_set():
                    self._output_ready.set()

            @room.on("disconnected")
            def _on_disconnected(reason):
                self._output_ready.clear()
                if self._running.is_set():
                    self._running.clear()
                    self._fence_playback()

            await room.connect(self._url, self._token)
            await room.local_participant.publish_track(track)
            self._output_ready.set()
            while self._running.is_set():
                await asyncio.sleep(0.1)
        finally:
            self._session_teardown.set()
            # A transport failure is also a playback fence. Keep this loop and
            # source alive while native synthesis unwinds; only then close the
            # source and let _run_loop exit.
            self._output_ready.clear()
            if self._running.is_set():
                self._running.clear()
                self._fence_playback()
            if source is not None:
                with self._playback_lock:
                    clear_through = self._clear_requested_generation
                if clear_through >= 0:
                    try:
                        await self._clear_source_through(clear_through, source)
                    except Exception:
                        await self._fail_playback_source(source)
            while True:
                with self._playback_lock:
                    pending = bool(self._pending_playback)
                    workers = bool(self._playback_futures)
                if not pending and not workers:
                    break
                await asyncio.sleep(0.01)
            track_tasks = tuple(self._track_tasks)
            for task in track_tasks:
                task.cancel()
            if track_tasks:
                await asyncio.gather(*track_tasks, return_exceptions=True)
            if room is not None:
                try:
                    await room.disconnect()
                except Exception:
                    pass
            if source is not None and self._source_is_open(source):
                aclose = getattr(source, "aclose", None)
                if callable(aclose):
                    try:
                        await aclose()
                    except Exception:
                        pass
            with self._playback_lock:
                if source is not None and self._source is source:
                    self._source = None
                if room is not None and self._room is room:
                    self._room = None

    async def _read_track(self, audio_track) -> None:  # pragma: no cover - live
        import numpy as np
        from livekit import rtc

        stream = rtc.AudioStream(audio_track)
        try:
            async for event in stream:
                if not self._running.is_set():
                    break
                frame = event.frame
                data = np.frombuffer(frame.data, dtype=np.int16)
                if frame.num_channels > 1:
                    data = data.reshape(-1, frame.num_channels)[:, 0]
                mono = resample_linear(
                    pcm_int16_to_float32(data), frame.sample_rate, STT_SR
                )
                try:
                    self._in_q.put_nowait(mono)
                except queue.Full:
                    pass  # drop under backpressure rather than lag the room
        finally:
            aclose = getattr(stream, "aclose", None)
            if callable(aclose):
                await aclose()

    async def _playback_worker(
        self,
        call: _LiveKitPlaybackCall,
    ) -> None:
        slots = self._playback_slots
        if slots is None:
            self._terminalize(call, PlaybackOutcome.FAILED)
            return
        async with slots:
            if call.terminal:
                return
            await self._playback_worker_in_slot(call)

    async def _playback_worker_in_slot(
        self,
        call: _LiveKitPlaybackCall,
    ) -> None:  # pragma: no cover - optional adapters are faked in tests
        import numpy as np
        from livekit import rtc

        synth_lock = self._synth_order_lock
        output_lock = self._speak_lock
        if synth_lock is None or output_lock is None:
            self._terminalize(call, PlaybackOutcome.FAILED)
            return

        source = None
        try:
            # asyncio.Lock acquisition is FIFO. Synthesis therefore preserves
            # admission order, while releasing this lock before source drain
            # lets the following sentence pre-generate without interleaving its
            # frames on the output lock.
            async with synth_lock:
                source = await self._await_output_source(call)
                if source is None:
                    return
                with self._playback_lock:
                    if call.terminal:
                        return
                    call.source = source
                    call.phase = "synthesizing"

                # Shield the native worker. asyncio cancellation cannot stop a
                # C/C++ TTS call, so terminal ownership stays pending until the
                # real generate() returns and its stale output can be fenced.
                synth_task = asyncio.create_task(
                    asyncio.to_thread(self._synth_locked, call.text)
                )
                try:
                    audio = await asyncio.shield(synth_task)
                except asyncio.CancelledError:
                    try:
                        await synth_task
                    except Exception:  # noqa: BLE001 - terminal handled below
                        await self._finish_noncompletion(call, source)
                    else:
                        await self._finish_noncompletion(call, source)
                    raise

                if not self._is_current(call, source):
                    await self._finish_noncompletion(call, source)
                    return
                samples = np.asarray(audio.samples, dtype=np.float32)
                out = float32_to_pcm_int16(
                    resample_linear(samples, audio.sample_rate, self._out_sr)
                )
                chunk = int(self._out_sr * FRAME_MS / 1000)
                if chunk <= 0 or len(out) == 0:
                    raise RuntimeError("LiveKit TTS produced no output frames")
                with self._playback_lock:
                    if not call.terminal:
                        call.phase = "ready"

            # Only one fragment may own the source queue. Holding this lock
            # through wait_for_playout makes that fragment's terminal receipt
            # exact, while its successor is already synthesized above.
            async with output_lock:
                if not self._is_current(call, source):
                    await self._finish_noncompletion(call, source)
                    return
                await self._ensure_clear_barrier(call, source)
                if not self._is_current(call, source):
                    await self._finish_noncompletion(call, source)
                    return
                await self._publish_data(
                    "assistant_sentence",
                    {"text": call.text},
                )
                if not self._is_current(call, source):
                    await self._finish_noncompletion(call, source)
                    return

                capture_frame = getattr(source, "capture_frame", None)
                wait_for_playout = getattr(source, "wait_for_playout", None)
                if not callable(capture_frame) or not callable(wait_for_playout):
                    raise RuntimeError(
                        "LiveKit AudioSource lacks tracked playout methods"
                    )
                with self._playback_lock:
                    if not call.terminal:
                        call.phase = "capturing"
                for i in range(0, len(out), chunk):
                    if not self._is_current(call, source):
                        await self._finish_noncompletion(call, source)
                        return
                    seg = out[i : i + chunk]
                    await capture_frame(
                        rtc.AudioFrame(
                            data=seg.tobytes(),
                            sample_rate=self._out_sr,
                            num_channels=1,
                            samples_per_channel=len(seg),
                        )
                    )
                    if not self._source_is_open(source):
                        await self._finish_noncompletion(call, source)
                        return
                    # A STOP may race the capture await. The frame still reached
                    # the local source, so start must be reported before the
                    # resulting interruption terminal.
                    self._mark_started(call)
                    if not self._is_current(call, source):
                        await self._finish_noncompletion(call, source)
                        return
                with self._playback_lock:
                    if not call.terminal:
                        call.phase = "waiting"
                await wait_for_playout()
                if self._terminalize(
                    call,
                    PlaybackOutcome.COMPLETED,
                    require_current=True,
                    source=source,
                ):
                    return
                await self._finish_noncompletion(call, source)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - adapter/model failures are receipts
            await self._finish_noncompletion(call, source)

    # --- data channel (transcripts for the web/mobile UI) ---
    def _publish(self, event_type: str, payload: dict) -> None:
        """Schedule a data-channel publish from a worker thread onto the loop."""
        loop = self._loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._publish_data(event_type, payload), loop)

    async def _publish_data(self, event_type: str, payload: dict) -> None:
        room = self._room
        if room is None:
            return
        try:  # pragma: no cover - needs a live room
            await room.local_participant.publish_data(
                json.dumps({"event_type": event_type, "payload": payload}).encode("utf-8"),
                reliable=True,
            )
        except Exception:
            pass
