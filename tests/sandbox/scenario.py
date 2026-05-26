from __future__ import annotations

import time
from typing import Callable, Optional

from always_on_agent.events import Mode

from core.runtime import VoiceRuntime

from .profiles import DeviceProfile
from .sim_engine import SimulatedEngine
from .sim_llm import SimulatedLLM


class Sandbox:
    """Drives the real threaded runtime through a simulated device.

    The runtime runs fully asynchronously (``run_bus=True``), exactly like
    production, while the test thread drives user speech and observes outcomes.
    Assertions are on *outcomes* (what was spoken, current mode, whether
    playback stopped) with polling + timeouts, not exact millisecond timing.
    """

    def __init__(
        self,
        profile: DeviceProfile,
        *,
        start_mode: Mode = Mode.ASSISTANT,
        reply_fn: Optional[Callable[[str], str]] = None,
    ):
        self.profile = profile
        self.engine = SimulatedEngine(profile)
        self.llm = SimulatedLLM(profile, reply_fn)
        self.runtime = VoiceRuntime(self.engine, self.llm, start_mode=start_mode)
        self.runtime.start(run_bus=True)

    # --- drive ---
    def user_says(self, text: str, *, incremental: bool = True) -> None:
        """Speak to the assistant and block until the brain has *ingested* the
        final (so subsequent ``wait_task_active``/idle checks aren't racing the
        bus thread that processes the utterance)."""
        before = len(self.runtime.supervisor.state.transcript_log)
        self.engine.say(text, incremental=incremental)
        self._poll(
            lambda: len(self.runtime.supervisor.state.transcript_log) > before,
            timeout=2.0,
        )

    def barge_in(self) -> None:
        self.engine.user_barge_in()

    def settle(self, seconds: Optional[float] = None) -> None:
        """Wait out the window in which a reply *would* have begun playing, so a
        subsequent ``spoken == []`` assertion is meaningful."""
        time.sleep(seconds if seconds is not None else self._reply_window())

    def _reply_window(self) -> float:
        p = self.profile
        return p.llm_ttft_sec + 30 * p.llm_per_token_sec + p.tts_ttfa_sec + 0.3

    # --- observe ---
    @property
    def spoken(self) -> list[str]:
        with self.engine._lock:  # noqa: SLF001 - test helper
            return list(self.engine.spoken)

    @property
    def mode(self) -> Mode:
        return self.runtime.mode

    def wait_task_active(self, timeout: float = 5.0) -> bool:
        return self._poll(lambda: bool(self.runtime.supervisor.state.active_tasks), timeout)

    def wait_speaking(self, timeout: float = 5.0) -> bool:
        return self._poll(lambda: self.engine.is_speaking, timeout)

    def wait_not_speaking(self, timeout: float = 5.0) -> bool:
        return self._poll(lambda: not self.engine.is_speaking, timeout)

    def wait_spoke_count(self, count: int, timeout: float = 5.0) -> bool:
        return self._poll(lambda: len(self.spoken) >= count, timeout)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        def idle() -> bool:
            state = self.runtime.supervisor.state
            return (
                not state.active_tasks
                and not state.queued_tasks
                and not self.engine.is_speaking
            )

        return self._poll(idle, timeout)

    def close(self) -> None:
        self.runtime.stop()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def _poll(predicate: Callable[[], bool], timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()
