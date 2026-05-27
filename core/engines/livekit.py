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
import contextlib
import json
import queue
import threading
from typing import Callable, Optional

from ..engine import AudioEngine, EngineCallbacks
from ._sherpa_models import build_recognizer, build_tts, build_vad
from .sherpa import SherpaConfig

OUT_SR = 48000  # LiveKit-friendly output sample rate for published TTS
STT_SR = 16000  # sherpa streaming-ASR input rate
FRAME_MS = 20   # published audio frame size


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

        self._in_q: "queue.Queue" = queue.Queue(maxsize=queue_max)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._room = None
        self._source = None
        self._asr_thread: Optional[threading.Thread] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._speaking = threading.Event()
        self._stop_speaking = threading.Event()
        # Serialize TTS so streamed sentences publish in order on the room track
        # instead of interleaving frames. Created on the loop in _run_loop.
        self._speak_lock: Optional[asyncio.Lock] = None

    # --- AudioEngine ---
    def start(self, callbacks: EngineCallbacks) -> None:
        self._cb = callbacks
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
        self._stop_speaking.set()
        if self._asr_thread is not None:
            self._asr_thread.join(timeout=1.0)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)

    def speak(self, text: str, on_done: Optional[Callable[[], None]] = None) -> None:
        loop = self._loop
        if self._tts is None or loop is None:
            if on_done:
                on_done()
            return
        fut = asyncio.run_coroutine_threadsafe(self._speak(text), loop)
        if on_done:
            fut.add_done_callback(lambda _f: on_done())

    def stop_speaking(self) -> None:
        self._stop_speaking.set()

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

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
            self._cb.on_partial(text)
        if rec.is_endpoint(stream):
            final_text = rec.get_result(stream)
            rec.reset(stream)
            self._last_partial = ""
            if final_text.strip():
                self._cb.on_final(final_text)
                self._publish("user_transcript", {"text": final_text})

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
        self._speak_lock = asyncio.Lock()
        try:
            self._loop.run_until_complete(self._session())
        except Exception as exc:  # pragma: no cover - needs a live server
            print(f"[livekit] session error: {exc}")
        finally:
            self._loop.close()

    async def _session(self) -> None:  # pragma: no cover - needs a live server
        from livekit import rtc

        room = rtc.Room()
        self._room = room
        source = rtc.AudioSource(self._out_sr, 1)
        self._source = source
        track = rtc.LocalAudioTrack.create_audio_track("assistant-voice", source)

        @room.on("track_subscribed")
        def _on_track(track_, publication, participant):
            if track_.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.create_task(self._read_track(track_))

        await room.connect(self._url, self._token)
        await room.local_participant.publish_track(track)
        while self._running.is_set():
            await asyncio.sleep(0.1)
        try:
            await room.disconnect()
        except Exception:
            pass

    async def _read_track(self, audio_track) -> None:  # pragma: no cover - live
        import numpy as np
        from livekit import rtc

        stream = rtc.AudioStream(audio_track)
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

    async def _speak(self, text: str) -> None:  # pragma: no cover - needs models
        import numpy as np
        from livekit import rtc

        # One sentence on the track at a time: streamed sentences arrive as
        # separate _speak coroutines, and interleaving their frames would garble
        # the audio. The lock makes them publish back-to-back, in order.
        lock = self._speak_lock
        async with lock if lock is not None else contextlib.nullcontext():
            self._stop_speaking.clear()
            self._speaking.set()
            self._cb.on_speech_start()
            try:
                await self._publish_data("assistant_sentence", {"text": text})
                audio = await asyncio.to_thread(
                    lambda: self._tts.generate(
                        text, sid=self._cfg.tts_speaker_id, speed=self._cfg.tts_speed
                    )
                )
                samples = np.asarray(audio.samples, dtype=np.float32)
                out = float32_to_pcm_int16(
                    resample_linear(samples, audio.sample_rate, self._out_sr)
                )
                chunk = int(self._out_sr * FRAME_MS / 1000)
                for i in range(0, len(out), chunk):
                    if self._stop_speaking.is_set():
                        break
                    seg = out[i : i + chunk]
                    await self._source.capture_frame(
                        rtc.AudioFrame(
                            data=seg.tobytes(),
                            sample_rate=self._out_sr,
                            num_channels=1,
                            samples_per_channel=len(seg),
                        )
                    )
            finally:
                self._speaking.clear()
                self._cb.on_speech_end()

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
