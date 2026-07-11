"""Runtime integration for engine-attested playback history."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from always_on_agent.events import AgentEvent, EventKind, Mode
from always_on_agent.followups import FollowupConfig
from always_on_agent.memory import SessionMemory
from always_on_agent.models import IntentDecision, IntentKind
from core.engine import (
    AudioEngine,
    EngineCallbacks,
    PlaybackCapabilities,
    PlaybackOutcome,
    PlaybackReceipt,
    SpeechStyle,
    TrackedSpeech,
)
from core.engines.sherpa import SherpaConfig
from core.llm import EchoLLM
from core.resume import ResumeConfig
from core.runtime import VoiceRuntime


@dataclass
class _HeldFragment:
    speech: TrackedSpeech
    on_started: object
    on_terminal: object
    started: bool = False


class _ControlledReceiptEngine(AudioEngine):
    """Manual sink: tests decide exactly which fragments start and terminal."""

    _CAPS = PlaybackCapabilities(
        tracked_terminal=True,
        exact_started=True,
        sample_counts=True,
        speech_style_hints=True,
    )

    def __init__(self, *, stop_prefix: str = "") -> None:
        self._cb = EngineCallbacks()
        self._lock = threading.Lock()
        self.fragments: list[_HeldFragment] = []
        self.stop_prefix = stop_prefix

    @property
    def playback_capabilities(self) -> PlaybackCapabilities:
        return self._CAPS

    def start(self, callbacks: EngineCallbacks) -> None:
        self._cb = callbacks

    def stop(self) -> None:
        self.stop_speaking()

    def speak(self, text: str, on_done=None) -> None:  # pragma: no cover - guard
        raise AssertionError("receipt runtime must not call legacy speak")

    def speak_tracked(
        self,
        speech: TrackedSpeech,
        *,
        on_terminal,
        on_started=None,
    ) -> None:
        with self._lock:
            self.fragments.append(_HeldFragment(speech, on_started, on_terminal))

    def start_fragment(self, index: int) -> None:
        with self._lock:
            fragment = self.fragments[index]
            assert not fragment.started
            fragment.started = True
        self._cb.on_speech_start()
        if fragment.on_started is not None:
            fragment.on_started(fragment.speech.fragment_id)

    def terminal(
        self,
        index: int,
        outcome: PlaybackOutcome,
        *,
        safe_text_prefix: str = "",
        played_samples: int = 0,
        total_samples: int = 100,
    ) -> None:
        with self._lock:
            fragment = self.fragments[index]
            callback = fragment.on_terminal
            fragment.on_terminal = None
            started = fragment.started
        assert callback is not None
        if started:
            self._cb.on_speech_end()
        callback(
            PlaybackReceipt(
                fragment_id=fragment.speech.fragment_id,
                outcome=outcome,
                safe_text_prefix=safe_text_prefix,
                played_samples=played_samples,
                total_samples=total_samples,
                output_sample_rate=16000,
            )
        )

    def stop_speaking(self) -> None:
        with self._lock:
            pending = [
                (i, fragment.started)
                for i, fragment in enumerate(self.fragments)
                if fragment.on_terminal is not None
            ]
        for index, started in pending:
            self.terminal(
                index,
                PlaybackOutcome.INTERRUPTED if started else PlaybackOutcome.DROPPED,
                safe_text_prefix=self.stop_prefix if started else "",
                played_samples=1 if started else 0,
            )

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return any(
                fragment.started and fragment.on_terminal is not None
                for fragment in self.fragments
            )


def _task(runtime: VoiceRuntime, text: str = "question"):
    task = runtime.supervisor.tasks.create_task(
        IntentDecision(
            kind=IntentKind.ASSISTANT,
            confidence=1.0,
            text=text,
            reason="receipt-test",
            mode=Mode.ASSISTANT,
        )
    )
    runtime.supervisor.state.active_tasks[task.task_id] = task
    return task


def _assistant_memory(runtime: VoiceRuntime) -> list[str]:
    return [
        item.text
        for item in runtime.memory.all()
        if "assistant_output" in item.tags
    ]


class _JoiningCallbackEngine(_ControlledReceiptEngine):
    """Adapter that synchronously waits for its callback worker to finish."""

    def __init__(self) -> None:
        super().__init__()
        self.callback_joined = False
        self.callback_errors: list[BaseException] = []

    def speak_tracked(
        self,
        speech: TrackedSpeech,
        *,
        on_terminal,
        on_started=None,
    ) -> None:
        def dispatch_callbacks() -> None:
            try:
                if on_started is not None:
                    on_started(speech.fragment_id)
                on_terminal(
                    PlaybackReceipt(
                        fragment_id=speech.fragment_id,
                        outcome=PlaybackOutcome.COMPLETED,
                        safe_text_prefix=speech.text,
                        played_samples=100,
                        total_samples=100,
                        output_sample_rate=16000,
                    )
                )
            except BaseException as exc:  # pragma: no cover - surfaced below
                self.callback_errors.append(exc)

        callback_thread = threading.Thread(target=dispatch_callbacks, daemon=True)
        callback_thread.start()
        callback_thread.join(timeout=1.0)
        self.callback_joined = not callback_thread.is_alive()


def test_speak_tracked_can_join_callback_thread_without_deadlock():
    engine = _JoiningCallbackEngine()
    runtime = VoiceRuntime(engine, EchoLLM())
    runtime.start(run_bus=False)
    task = _task(runtime)
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "Callback-thread playback.",
                "speak": True,
                "followup": False,
                "data": {},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    try:
        runtime.bus.drain()

        assert engine.callback_joined
        assert engine.callback_errors == []
        assert runtime.wait_idle()
        assert _assistant_memory(runtime) == ["Callback-thread playback."]
    finally:
        runtime.stop()


def test_nonstream_history_and_followup_wait_for_terminal_playout():
    engine = _ControlledReceiptEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        followup_config=FollowupConfig(enabled=True, delay_sec=60.0),
    )
    runtime.start(run_bus=False)
    task = _task(runtime)
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "A complete answer.",
                "speak": True,
                "followup": False,
                "data": {},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    try:
        runtime.bus.drain()
        assert len(engine.fragments) == 1
        assert _assistant_memory(runtime) == []
        assert runtime.supervisor._followup_timer is None
        assert not runtime.wait_idle(timeout=0.02)

        engine.start_fragment(0)
        assert _assistant_memory(runtime) == []
        assert runtime.supervisor._followup_timer is None

        engine.terminal(
            0,
            PlaybackOutcome.COMPLETED,
            safe_text_prefix="A complete answer.",
            played_samples=16000,
            total_samples=16000,
        )
        assert runtime.wait_idle()
        assert _assistant_memory(runtime) == ["A complete answer."]
        assert runtime.supervisor.state.spoken_outputs[-1] == "A complete answer."
        assert runtime.supervisor._followup_timer is not None
    finally:
        runtime.stop()


def test_barge_before_queued_commit_cannot_rearm_followup():
    engine = _ControlledReceiptEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        followup_config=FollowupConfig(enabled=True, delay_sec=60.0),
    )
    runtime.start(run_bus=False)
    task = _task(runtime)
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "Already played.",
                "speak": True,
                "followup": False,
                "data": {},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    try:
        assert runtime.bus.drain_once()  # completion -> queues TTS_REQUEST
        assert runtime.bus.drain_once()  # admission; engine now owns fragment
        engine.start_fragment(0)
        engine.terminal(
            0,
            PlaybackOutcome.COMPLETED,
            safe_text_prefix="Already played.",
            played_samples=100,
            total_samples=100,
        )
        # MEMORY_COMMIT is queued but not handled.  The cut advances the epoch
        # and resets cadence before that old commit reaches the supervisor.
        runtime._on_barge_in()
        runtime.bus.drain()

        assert _assistant_memory(runtime) == ["Already played."]
        assert runtime.supervisor._followup_timer is None
    finally:
        runtime.stop()


def test_stream_cut_commits_only_completed_sentence_and_not_queued_echo():
    engine = _ControlledReceiptEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        resume_config=ResumeConfig(enabled=True, echo_guard_enabled=True),
    )
    runtime.start(run_bus=False)
    runtime._resume.note_query("Tell me a story")
    task = _task(runtime, "Tell me a story")
    parts = (
        "The first sentence reached the speaker.",
        "The second sentence was cut midway.",
        "The third sentence stayed queued.",
    )
    for part in parts:
        runtime.bus.publish(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {
                    "task_id": task.task_id,
                    "text": part,
                    "epoch": task.speech_epoch,
                },
            )
        )
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": " ".join(parts),
                "speak": True,
                "followup": False,
                "data": {"streamed": True},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_STREAM_END,
            {"task_id": task.task_id, "epoch": task.speech_epoch},
            priority=110,
        )
    )
    try:
        runtime.bus.drain()
        assert len(engine.fragments) == 3
        assert _assistant_memory(runtime) == []

        engine.start_fragment(0)
        engine.terminal(
            0,
            PlaybackOutcome.COMPLETED,
            safe_text_prefix=parts[0],
            played_samples=100,
            total_samples=100,
        )
        engine.start_fragment(1)
        runtime._on_barge_in()
        runtime.bus.drain()

        assert _assistant_memory(runtime) == [parts[0]]
        prompt = runtime._resume.resume_prompt("continue")
        assert prompt is not None and "first sentence" in prompt
        assert "second sentence" not in prompt
        assert "third sentence" not in prompt
        assert not runtime._resume.is_self_echo(parts[2])
        assert runtime.supervisor._followup_timer is None
    finally:
        runtime.stop()


def test_stream_voice_survives_priority_completion_until_stream_end():
    engine = _ControlledReceiptEngine()
    engine.config = SherpaConfig(
        tts_markup=True,
        tts_speaker_voices={"warm": 16, "deep": 9},
    )
    runtime = VoiceRuntime(engine, EchoLLM())
    runtime.start(run_bus=False)
    task = _task(runtime, "story")
    first = AgentEvent(
        EventKind.TTS_REQUEST,
        {
            "task_id": task.task_id,
            "text": "[voice:warm] First sentence.",
            "epoch": task.speech_epoch,
            "streaming": True,
        },
    )
    second = AgentEvent(
        EventKind.TTS_REQUEST,
        {
            "task_id": task.task_id,
            "text": "Second sentence.",
            "epoch": task.speech_epoch,
            "streaming": True,
        },
    )
    runtime.bus.publish(first)
    runtime.bus.drain()
    # TASK_COMPLETED priority 60 overtakes the already-queued second sentence
    # (priority 100). It must not close voice state; only STREAM_END does that.
    runtime.bus.publish(second)
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "[voice:warm] First sentence. Second sentence.",
                "speak": True,
                "followup": False,
                "data": {"streamed": True},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_STREAM_END,
            {"task_id": task.task_id, "epoch": task.speech_epoch},
            priority=110,
        )
    )
    try:
        runtime.bus.drain()
        assert [fragment.speech.text for fragment in engine.fragments] == [
            "First sentence.",
            "Second sentence.",
        ]
        assert [fragment.speech.style for fragment in engine.fragments] == [
            SpeechStyle(voice="warm"),
            SpeechStyle(voice="warm"),
        ]

        # END tombstones the producer after every legitimate priority-100
        # fragment, so a buggy late emitter cannot reopen an unclosable group.
        runtime.bus.publish(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {
                    "task_id": task.task_id,
                    "text": "Late fragment.",
                    "epoch": task.speech_epoch,
                    "streaming": True,
                },
            )
        )
        runtime.bus.drain()
        assert len(engine.fragments) == 2
        assert not runtime.supervisor.tts_request_allowed(
            task.task_id,
            task.speech_epoch,
        )
    finally:
        runtime.stop()


def test_nonstream_reply_applies_one_leading_voice_to_whole_fragment():
    engine = _ControlledReceiptEngine()
    engine.config = SherpaConfig(
        tts_markup=True,
        tts_speaker_voices={"warm": 16},
    )
    runtime = VoiceRuntime(engine, EchoLLM())
    runtime.start(run_bus=False)
    task = _task(runtime, "short answer")
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": (
                    "[voice:warm] First sentence. "
                    "[voice:deep] Second sentence. "
                    "[emotion:calm] Third sentence."
                ),
                "speak": True,
                "followup": False,
                "data": {},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    try:
        runtime.bus.drain()
        assert len(engine.fragments) == 1
        assert engine.fragments[0].speech.text == (
            "First sentence. Second sentence. Third sentence."
        )
        assert engine.fragments[0].speech.style == SpeechStyle(voice="warm")
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    "raw",
    [
        "[tag:story] Here is the first sequence.",
        "[tag:narrator] The moon orbits Earth.",
        "[narrator:deep] Once upon a time.",
    ],
)
def test_runtime_excludes_unsupported_control_tag_from_tracked_text(raw):
    engine = _ControlledReceiptEngine()
    engine.config = SherpaConfig(
        tts_markup=True,
        tts_speaker_voices={"narrator": 7},
    )
    runtime = VoiceRuntime(engine, EchoLLM())
    runtime.start(run_bus=False)
    task = _task(runtime, "story")
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_REQUEST,
            {
                "task_id": task.task_id,
                "text": raw,
                "epoch": task.speech_epoch,
                "streaming": True,
            },
        )
    )
    try:
        runtime.bus.drain()

        assert len(engine.fragments) == 1
        assert not engine.fragments[0].speech.text.startswith("[")
        assert engine.fragments[0].speech.style is None
    finally:
        runtime.stop()


def test_runtime_preserves_listener_visible_bracket_text():
    engine = _ControlledReceiptEngine()
    engine.config = SherpaConfig(
        tts_markup=True,
        tts_speaker_voices={"narrator": 7},
    )
    runtime = VoiceRuntime(engine, EchoLLM())
    runtime.start(run_bus=False)
    task = _task(runtime, "citation")
    text = "[citation needed] This claim needs a source."
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_REQUEST,
            {
                "task_id": task.task_id,
                "text": text,
                "epoch": task.speech_epoch,
                "streaming": True,
            },
        )
    )
    try:
        runtime.bus.drain()

        assert engine.fragments[0].speech.text == text
    finally:
        runtime.stop()


def test_interleaved_replies_and_auxiliary_tts_keep_voice_scopes_isolated():
    engine = _ControlledReceiptEngine()
    engine.config = SherpaConfig(
        tts_markup=True,
        tts_speaker_voices={"warm": 16, "deep": 9},
    )
    runtime = VoiceRuntime(engine, EchoLLM())
    runtime.start(run_bus=False)
    task_a = _task(runtime, "reply a")
    task_b = _task(runtime, "reply b")
    aux_id = runtime.supervisor.register_aux_tts(
        task_a.task_id,
        speech_epoch=task_a.speech_epoch,
    )

    def stream(task, text):
        runtime.bus.publish(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {
                    "task_id": task.task_id,
                    "text": text,
                    "epoch": task.speech_epoch,
                    "streaming": True,
                },
            )
        )

    stream(task_a, "[voice:warm] A one.")
    stream(task_b, "[voice:deep] B one.")
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_REQUEST,
            {
                "task_id": task_a.task_id,
                "text": "Please wait.",
                "epoch": task_a.speech_epoch,
                "auxiliary_tts": True,
                "aux_tts_id": aux_id,
            },
        )
    )
    stream(task_a, "A two.")
    stream(task_b, "B two.")
    try:
        runtime.bus.drain()
        assert [fragment.speech.text for fragment in engine.fragments] == [
            "A one.",
            "B one.",
            "Please wait.",
            "A two.",
            "B two.",
        ]
        assert [fragment.speech.style for fragment in engine.fragments] == [
            SpeechStyle(voice="warm"),
            SpeechStyle(voice="deep"),
            None,
            SpeechStyle(voice="warm"),
            SpeechStyle(voice="deep"),
        ]
    finally:
        runtime.stop()


def test_barge_clears_voice_and_stale_fragment_cannot_reseed_it():
    engine = _ControlledReceiptEngine()
    engine.config = SherpaConfig(
        tts_markup=True,
        tts_speaker_voices={"warm": 16, "deep": 9},
    )
    runtime = VoiceRuntime(engine, EchoLLM())
    runtime.start(run_bus=False)
    old = _task(runtime, "old reply")
    old_epoch = old.speech_epoch
    runtime.bus.publish(
        AgentEvent(
            EventKind.TTS_REQUEST,
            {
                "task_id": old.task_id,
                "text": "[voice:warm] Old reply.",
                "epoch": old_epoch,
                "streaming": True,
            },
        )
    )
    runtime.bus.drain()
    assert engine.fragments[-1].speech.style == SpeechStyle(voice="warm")

    try:
        runtime._on_barge_in()
        runtime.bus.drain()
        runtime.bus.publish(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {
                    "task_id": old.task_id,
                    "text": "[voice:deep] Stale reply.",
                    "epoch": old_epoch,
                    "streaming": True,
                },
            )
        )
        fresh = _task(runtime, "fresh reply")
        fresh.speech_epoch = runtime.supervisor.speech_epoch
        runtime.bus.publish(
            AgentEvent(
                EventKind.TTS_REQUEST,
                {
                    "task_id": fresh.task_id,
                    "text": "Fresh reply.",
                    "epoch": fresh.speech_epoch,
                    "streaming": True,
                },
            )
        )
        runtime.bus.drain()

        assert [fragment.speech.text for fragment in engine.fragments] == [
            "Old reply.",
            "Fresh reply.",
        ]
        assert engine.fragments[-1].speech.style is None
    finally:
        runtime.stop()


def test_new_user_memory_waits_for_prior_assistant_receipt():
    engine = _ControlledReceiptEngine()
    runtime = VoiceRuntime(engine, EchoLLM())
    runtime.start(run_bus=False)
    task = _task(runtime, "old question")
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "Old heard answer.",
                "speak": True,
                "data": {},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    runtime.bus.drain()
    assert len(engine.fragments) == 1

    result = {}

    def answer_new_turn() -> None:
        result["value"] = runtime.supervisor.capabilities.invoke(
            "assistant.answer",
            "new question",
            {"mode": Mode.ASSISTANT.value},
        )

    provider = threading.Thread(target=answer_new_turn)
    provider.start()
    try:
        # The provider is held before recent-context read / user ingest.
        provider.join(timeout=0.03)
        assert provider.is_alive()
        assert runtime.memory.all() == []

        engine.start_fragment(0)
        engine.terminal(
            0,
            PlaybackOutcome.COMPLETED,
            safe_text_prefix="Old heard answer.",
            played_samples=100,
            total_samples=100,
        )
        runtime.bus.drain()
        provider.join(timeout=1.0)

        assert not provider.is_alive()
        assert result["value"].ok
        conversation = [
            (
                "assistant" if "assistant_output" in item.tags else "user",
                item.text,
            )
            for item in runtime.memory.all()
            if "assistant_output" in item.tags or "user" in item.tags
        ]
        assert conversation == [
            ("assistant", "Old heard answer."),
            ("user", "new question"),
        ]
    finally:
        if provider.is_alive():
            engine.stop_speaking()
            runtime.bus.drain()
            provider.join(timeout=1.0)
        runtime.stop()


def test_continuation_user_memory_is_staged_behind_prior_playback():
    engine = _ControlledReceiptEngine()
    runtime = VoiceRuntime(engine, EchoLLM())
    runtime.start(run_bus=False)
    task = _task(runtime, "old question")
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "Old answer.",
                "speak": True,
                "data": {},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    try:
        runtime.bus.drain()
        runtime._record_user_memory_ordered("user add-on")
        assert runtime.memory.all() == []

        engine.start_fragment(0)
        engine.terminal(
            0,
            PlaybackOutcome.COMPLETED,
            safe_text_prefix="Old answer.",
            played_samples=100,
            total_samples=100,
        )
        runtime.bus.drain()

        assert [item.text for item in runtime.memory.all()] == [
            "Old answer.",
            "user add-on",
        ]
    finally:
        runtime.stop()


def test_zero_play_interruption_does_not_arm_resume():
    engine = _ControlledReceiptEngine()
    runtime = VoiceRuntime(
        engine,
        EchoLLM(),
        resume_config=ResumeConfig(enabled=True),
    )
    runtime.start(run_bus=False)
    runtime._resume.note_query("Explain something")
    task = _task(runtime, "Explain something")
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "Never started.",
                "speak": True,
                "data": {},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    try:
        runtime.bus.drain()
        runtime._on_barge_in()
        runtime.bus.drain()
        assert _assistant_memory(runtime) == []
        assert runtime._resume.resume_prompt("continue") is None
    finally:
        runtime.stop()


class _ClosingMemory(SessionMemory):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def add(self, text: str, tags=()) -> None:
        assert not self.closed, "playback history arrived after memory.close()"
        super().add(text, tags)

    def close(self) -> None:
        self.closed = True


def test_shutdown_drains_receipt_commit_before_memory_close():
    memory = _ClosingMemory()
    engine = _ControlledReceiptEngine(stop_prefix="Safely heard prefix.")
    runtime = VoiceRuntime(engine, EchoLLM(), memory=memory)
    runtime.start(run_bus=False)
    task = _task(runtime)
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "Safely heard prefix. Unheard suffix.",
                "speak": True,
                "data": {},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    runtime.bus.drain()
    engine.start_fragment(0)

    runtime.stop()

    assert memory.closed
    assert [
        item.text for item in memory.all() if "assistant_output" in item.tags
    ] == ["Safely heard prefix."]


def test_shutdown_waits_through_terminal_resolve_to_commit_handoff():
    memory = _ClosingMemory()
    engine = _ControlledReceiptEngine()
    runtime = VoiceRuntime(engine, EchoLLM(), memory=memory)
    runtime.start(run_bus=False)
    task = _task(runtime)
    runtime.bus.publish(
        AgentEvent(
            EventKind.TASK_COMPLETED,
            {
                "task_id": task.task_id,
                "text": "Race-safe history.",
                "speak": True,
                "data": {},
                "epoch": task.speech_epoch,
            },
            priority=60,
        )
    )
    runtime.bus.drain()
    engine.start_fragment(0)

    publish_entered = threading.Event()
    release_publish = threading.Event()
    original_publish = runtime._publish_playback_commits

    def blocked_publish(commits):
        publish_entered.set()
        assert release_publish.wait(1.0)
        original_publish(commits)

    runtime._publish_playback_commits = blocked_publish
    terminal_thread = threading.Thread(
        target=lambda: engine.terminal(
            0,
            PlaybackOutcome.COMPLETED,
            safe_text_prefix="Race-safe history.",
            played_samples=100,
            total_samples=100,
        )
    )
    terminal_thread.start()
    assert publish_entered.wait(1.0)

    stop_thread = threading.Thread(target=runtime.stop)
    stop_thread.start()
    assert stop_thread.is_alive()
    assert memory.closed is False

    release_publish.set()
    terminal_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    assert not terminal_thread.is_alive() and not stop_thread.is_alive()
    assert memory.closed
    assert [
        item.text for item in memory.all() if "assistant_output" in item.tags
    ] == ["Race-safe history."]
