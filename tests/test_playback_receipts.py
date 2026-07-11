"""Runtime integration for engine-attested playback history."""
from __future__ import annotations

import threading
from dataclasses import dataclass

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
    TrackedSpeech,
)
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
