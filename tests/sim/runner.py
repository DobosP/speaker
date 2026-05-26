"""SimulationRunner: drives multi-turn conversations through the real pipeline.

It reuses the proven backbone from ``tests.conversation_harness`` (the real
VoiceAssistant + AudioRecorder + endpointing + DeterministicTTSPlayer) but feeds
user turns *interactively* one at a time, so the simulated user can react to the
assistant's previous reply. The assistant LLM is pluggable: a scripted double in
the mock tier, a real ``LocalLLM`` in the opt-in tier.

Success is evaluated by deterministic ``GoalCheck`` predicates; ``pass_hat_k``
reports the tau-bench reliability metric (all k trials must pass).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import queue
import time
from typing import Callable

from main import VoiceAssistant
from tests.conversation_harness import (
    DeterministicTTSPlayer,
    ScriptedSTT,
    Timeline,
)
from tests.fixtures import silence, voiced_speech
from tests.harness import AudioHarness, make_recorder
from tests.sim.persona import Goal, Persona
from tests.sim.user_agent import BaseUser


@dataclass
class SimTranscript:
    """Everything observable about one finished conversation."""

    turns: list[tuple[str, str]] = field(default_factory=list)  # (role, text)
    assistant_spoken: list[str] = field(default_factory=list)
    llm_prompts: list[str] = field(default_factory=list)
    route_actions: list[str] = field(default_factory=list)  # one per user turn
    shutdown: bool = False
    wall_time_sec: float = 0.0


class InteractiveSTT(ScriptedSTT):
    """ScriptedSTT whose transcripts are supplied one at a time at run time."""

    def __init__(self, timeline: Timeline | None = None):
        super().__init__([], timeline=timeline)
        self._queue: "queue.Queue[str]" = queue.Queue()

    def push(self, text: str) -> None:
        self._queue.put(text)

    def __call__(self, audio_data, **_kwargs) -> str:  # noqa: ANN001
        import numpy as np

        self.calls.append(np.asarray(audio_data, dtype=np.float32))
        try:
            transcript = self._queue.get(timeout=5.0)
        except queue.Empty:
            transcript = ""
        self.timeline.mark("stt_final", transcript=transcript, samples=len(self.calls[-1]))
        return transcript


class SimulationRunner:
    """Builds a fresh real pipeline per ``run_once`` and drives it turn-by-turn."""

    def __init__(
        self,
        persona: Persona,
        goal: Goal,
        *,
        assistant_llm_factory: Callable[[], object],
        user_factory: Callable[[], BaseUser],
        tts_playback_sec: float = 0.03,
    ):
        self.persona = persona
        self.goal = goal
        self._assistant_llm_factory = assistant_llm_factory
        self._user_factory = user_factory
        self._tts_playback_sec = tts_playback_sec

    def run_once(self) -> SimTranscript:
        timeline = Timeline()
        stt = InteractiveSTT(timeline=timeline)
        llm = self._assistant_llm_factory()
        player = DeterministicTTSPlayer(timeline=timeline, playback_sec=self._tts_playback_sec)
        user = self._user_factory()

        assistant = VoiceAssistant(
            enable_memory=False,
            stt_transcriber=stt,
            llm=llm,
            audio_player=player,
            streaming_llm=True,
            chunked_tts=False,
            live_partial_log=False,
            wakeword_enabled=False,
        )
        recorder = make_recorder(
            callback=assistant._on_speech_detected,
            on_interrupt=lambda info=None: assistant._on_barge_in(info or {}),
            wakeword_enabled=False,
            silence_duration=0.08,
            barge_in_min_speech_sec=0.12,
            barge_in_min_delay_after_ref_sec=0.0,
            barge_in_min_delay_sec=0.0,
            barge_in_cooldown_sec=0.0,
            aec_enabled=False,
        )
        assistant._recorder = recorder
        assistant._player = player
        assistant._llm = llm
        harness = AudioHarness(recorder)

        transcript = SimTranscript()
        started = time.time()
        harness.start()
        try:
            for _ in range(self.goal.max_turns):
                history = list(transcript.turns)
                user_text = user.next_turn(history)
                if not user_text:
                    break

                spoken_before = len(player.spoken_texts)
                prompts_before = len(llm.prompts) if hasattr(llm, "prompts") else 0

                stt.push(user_text)
                harness.inject(voiced_speech(0.4, amplitude=0.35), inter_chunk_delay=0.02)
                harness.inject(silence(0.35), inter_chunk_delay=0.02)
                harness.drain(timeout=5.0)
                self._wait_for_response(assistant)

                reply_chunks = player.spoken_texts[spoken_before:]
                reply = " ".join(reply_chunks).strip()
                prompts_after = len(llm.prompts) if hasattr(llm, "prompts") else 0
                shutdown = assistant._shutdown_event.is_set()

                transcript.turns.append(("user", user_text))
                if reply:
                    transcript.turns.append(("assistant", reply))
                transcript.route_actions.append(
                    self._classify_turn(shutdown, prompts_after > prompts_before, bool(reply))
                )
                if shutdown:
                    transcript.shutdown = True
                    break
        finally:
            harness.stop()
            if assistant._response_thread:
                assistant._response_thread.join(timeout=2.0)

        transcript.assistant_spoken = list(player.spoken_texts)
        transcript.llm_prompts = list(getattr(llm, "prompts", []))
        transcript.wall_time_sec = time.time() - started
        return transcript

    @staticmethod
    def _classify_turn(shutdown: bool, llm_called: bool, spoke: bool) -> str:
        if shutdown:
            return "shutdown"
        if llm_called:
            return "llm"
        if spoke:
            return "capability"
        return "control"

    @staticmethod
    def _wait_for_response(assistant: VoiceAssistant, timeout: float = 3.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            thread = assistant._response_thread
            if thread is not None:
                thread.join(timeout=0.02)
                if not thread.is_alive():
                    return True
            time.sleep(0.01)
        return False


def state_success(goal: Goal, transcript: SimTranscript) -> bool:
    """Primary success gate: every deterministic GoalCheck must pass."""
    return all(check.passed(transcript) for check in goal.checks)


def failed_checks(goal: Goal, transcript: SimTranscript) -> list[str]:
    return [check.name for check in goal.checks if not check.passed(transcript)]


def pass_hat_k(runner: SimulationRunner, k: int = 3) -> dict:
    """Run the scenario k times; report pass@1 and pass^k (all-k-pass reliability)."""
    transcripts = [runner.run_once() for _ in range(k)]
    successes = [t for t in transcripts if state_success(runner.goal, t)]
    return {
        "k": k,
        "pass_at_1": len(successes) / k,
        "pass_hat_k": 1.0 if len(successes) == k else 0.0,
        "transcripts": transcripts,
    }
