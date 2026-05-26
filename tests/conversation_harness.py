"""
Deterministic conversation-level harness for voice-agent tests.

The harness keeps the real recorder, endpointing, echo reference, barge-in,
and VoiceAssistant orchestration in the loop while replacing STT, LLM, and
TTS with scripted in-process doubles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Callable, Iterable, Optional

import numpy as np

from main import VoiceAssistant
from tests.fixtures import SR, real_tts_echo, silence
from tests.harness import AudioHarness, make_recorder


@dataclass
class TimelineEvent:
    name: str
    timestamp: float
    payload: dict = field(default_factory=dict)


class Timeline:
    def __init__(self):
        self._t0 = time.time()
        self._events: list[TimelineEvent] = []
        self._lock = threading.Lock()

    def mark(self, name: str, **payload):
        with self._lock:
            self._events.append(
                TimelineEvent(
                    name=name,
                    timestamp=time.time() - self._t0,
                    payload=payload,
                )
            )

    def names(self) -> list[str]:
        with self._lock:
            return [event.name for event in self._events]

    def events(self, name: Optional[str] = None) -> list[TimelineEvent]:
        with self._lock:
            if name is None:
                return list(self._events)
            return [event for event in self._events if event.name == name]

    def first_time(self, name: str) -> Optional[float]:
        events = self.events(name)
        return events[0].timestamp if events else None


class ScriptedSTT:
    """Callable transcriber that returns one transcript per captured utterance."""

    def __init__(self, transcripts: Iterable[str], timeline: Timeline | None = None):
        self._transcripts = list(transcripts)
        self.timeline = timeline or Timeline()
        self.calls: list[np.ndarray] = []

    def __call__(self, audio_data, **_kwargs) -> str:
        self.calls.append(np.asarray(audio_data, dtype=np.float32))
        transcript = self._transcripts.pop(0) if self._transcripts else ""
        self.timeline.mark(
            "stt_final",
            transcript=transcript,
            samples=len(self.calls[-1]),
        )
        return transcript


class ScriptedStreamingLLM:
    """
    Scripted LLM that yields sentence chunks and honors cancellation callbacks.
    """

    def __init__(
        self,
        responses: Iterable[Iterable[str] | str],
        timeline: Timeline | None = None,
        sentence_delay_sec: float = 0.0,
    ):
        self._responses = list(responses)
        self.timeline = timeline or Timeline()
        self.sentence_delay_sec = sentence_delay_sec
        self.prompts: list[str] = []
        self.cancelled = False

    def _next_response(self) -> list[str]:
        if not self._responses:
            return []
        response = self._responses.pop(0)
        if isinstance(response, str):
            return [response]
        return list(response)

    def get_response(self, text: str, context=None, history=None) -> str:
        self.prompts.append(text)
        parts = self._next_response()
        response = " ".join(parts)
        self.timeline.mark("llm_response", prompt=text, text=response)
        return response

    def get_streaming_response(
        self,
        text: str,
        context=None,
        history=None,
        should_cancel: Optional[Callable[[], bool]] = None,
        **kwargs,
    ):
        self.prompts.append(text)
        self.timeline.mark("llm_start", prompt=text)
        for sentence in self._next_response():
            if should_cancel and should_cancel():
                self.cancelled = True
                self.timeline.mark("llm_cancelled", prompt=text)
                break
            if self.sentence_delay_sec > 0:
                time.sleep(self.sentence_delay_sec)
            if should_cancel and should_cancel():
                self.cancelled = True
                self.timeline.mark("llm_cancelled", prompt=text)
                break
            self.timeline.mark("llm_sentence", text=sentence)
            yield sentence


class DeterministicTTSPlayer:
    """
    TTS double that produces deterministic echo-reference audio and blocks long
    enough for tests to inject barge-in audio while playback is active.
    """

    def __init__(
        self,
        timeline: Timeline | None = None,
        sample_rate: int = SR,
        playback_sec: float = 0.45,
        audio_factory: Optional[Callable[[str, float], np.ndarray]] = None,
    ):
        self.timeline = timeline or Timeline()
        self.sample_rate = sample_rate
        self.playback_sec = playback_sec
        self.audio_factory = audio_factory or self._default_audio_factory
        self.spoken_texts: list[str] = []
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.finished = threading.Event()
        self.current_audio: Optional[np.ndarray] = None
        self.stop_count = 0

    def _default_audio_factory(self, _text: str, duration_sec: float) -> np.ndarray:
        return real_tts_echo(duration_sec, sr=self.sample_rate, amplitude=0.08)

    def speak(self, text: str, on_start=None, chunked: bool = False):
        self.spoken_texts.append(text)
        self.stopped.clear()
        self.finished.clear()
        self.current_audio = self.audio_factory(text, self.playback_sec)
        self.timeline.mark("tts_start", text=text, chunked=chunked)
        self.started.set()
        if on_start:
            on_start(self.current_audio, self.sample_rate)

        deadline = time.time() + self.playback_sec
        while time.time() < deadline and not self.stopped.is_set():
            time.sleep(0.005)

        self.finished.set()
        self.timeline.mark("tts_end", text=text, stopped=self.stopped.is_set())

    def stop(self):
        self.stop_count += 1
        self.stopped.set()
        self.timeline.mark("tts_stop", count=self.stop_count)

    def cleanup(self):
        self.stop()


class ConversationRunner:
    """
    Runs scripted conversations through VoiceAssistant plus the real AudioRecorder.
    """

    def __init__(
        self,
        transcripts: Iterable[str],
        llm_responses: Iterable[Iterable[str] | str],
        *,
        wakeword_enabled: bool = False,
        wakeword_policy: str = "strict_required",
        recorder_kwargs: Optional[dict] = None,
        tts_playback_sec: float = 0.45,
    ):
        self.timeline = Timeline()
        self.stt = ScriptedSTT(transcripts, timeline=self.timeline)
        self.llm = ScriptedStreamingLLM(llm_responses, timeline=self.timeline)
        self.player = DeterministicTTSPlayer(
            timeline=self.timeline,
            playback_sec=tts_playback_sec,
        )
        self.interrupts: list[dict] = []

        self.assistant = VoiceAssistant(
            enable_memory=False,
            stt_transcriber=self.stt,
            llm=self.llm,
            audio_player=self.player,
            streaming_llm=True,
            chunked_tts=False,
            live_partial_log=False,
            wakeword_enabled=wakeword_enabled,
            wakeword_policy=wakeword_policy,
        )

        kwargs = {
            "callback": self.assistant._on_speech_detected,
            "on_interrupt": self._on_interrupt,
            "wakeword_enabled": wakeword_enabled,
            "wakeword_policy": wakeword_policy,
            "silence_duration": 0.08,
            "barge_in_min_speech_sec": 0.12,
            "barge_in_min_delay_after_ref_sec": 0.0,
            "barge_in_min_delay_sec": 0.0,
            "barge_in_cooldown_sec": 0.0,
            "aec_enabled": False,
        }
        if recorder_kwargs:
            kwargs.update(recorder_kwargs)
        self.recorder = make_recorder(**kwargs)
        self.assistant._recorder = self.recorder
        self.assistant._player = self.player
        self.assistant._llm = self.llm
        self.harness = AudioHarness(self.recorder)

    def _on_interrupt(self, info=None):
        info = info or {}
        self.interrupts.append(info)
        self.timeline.mark("barge_in", **info)
        return self.assistant._on_barge_in(info)

    def __enter__(self) -> "ConversationRunner":
        self.harness.start()
        return self

    def __exit__(self, *_):
        self.harness.stop()
        if self.assistant._response_thread:
            self.assistant._response_thread.join(timeout=2.0)

    def inject_user_turn(
        self,
        audio: np.ndarray,
        *,
        trailing_silence_sec: float = 0.25,
        inter_chunk_delay: float = 0.0,
    ):
        self.timeline.mark("user_audio_injected", samples=len(audio))
        self.harness.inject(audio, inter_chunk_delay=inter_chunk_delay)
        if trailing_silence_sec > 0:
            self.harness.inject(
                silence(trailing_silence_sec),
                inter_chunk_delay=inter_chunk_delay,
            )
        self.harness.drain(timeout=5.0)
        self.timeline.mark("user_audio_drained")

    def inject_barge_in(
        self,
        audio: np.ndarray,
        *,
        inter_chunk_delay: float = 0.0,
    ):
        self.timeline.mark("barge_audio_injected", samples=len(audio))
        self.harness.inject(audio, inter_chunk_delay=inter_chunk_delay)
        self.harness.drain(timeout=5.0)
        self.timeline.mark("barge_audio_drained")

    def wait_for_response(self, timeout: float = 3.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            thread = self.assistant._response_thread
            if thread is not None:
                thread.join(timeout=0.02)
                if not thread.is_alive():
                    return True
            time.sleep(0.01)
        return False

    def wait_for_tts_start(self, timeout: float = 2.0) -> bool:
        return self.player.started.wait(timeout=timeout)

    def assert_event_order(self, *names: str):
        seen = self.timeline.names()
        positions = []
        for name in names:
            assert name in seen, f"Missing timeline event {name!r}; saw {seen!r}"
            positions.append(seen.index(name))
        assert positions == sorted(positions), (
            f"Expected event order {names!r}; saw {seen!r}"
        )
