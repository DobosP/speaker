from __future__ import annotations

import time
from typing import Callable, Iterator, Optional

from .profiles import DeviceProfile


class SimulatedLLM:
    """LLM whose latency models a real local model's weight + hardware.

    ``generate`` blocks for ``llm_ttft_sec`` then ``llm_per_token_sec`` per
    output word, so a slow phone model genuinely stalls the task thread — which
    is exactly the window a user might barge in. The reply text itself is
    deterministic (via ``reply_fn``) so assertions stay stable.
    """

    def __init__(
        self,
        profile: DeviceProfile,
        reply_fn: Optional[Callable[[str], str]] = None,
    ):
        self.profile = profile
        self._reply_fn = reply_fn or (lambda prompt: f"You said: {prompt}")
        self.calls: list[str] = []
        self.tokens_yielded = 0

    def _reply(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._reply_fn(prompt)

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        reply = self._reply(prompt)
        time.sleep(self.profile.llm_ttft_sec)
        n_tokens = max(1, len(reply.split()))
        time.sleep(self.profile.llm_per_token_sec * n_tokens)
        return reply

    def stream(self, prompt: str, *, system: Optional[str] = None) -> Iterator[str]:
        reply = self._reply(prompt)
        time.sleep(self.profile.llm_ttft_sec)
        for token in reply.split():
            time.sleep(self.profile.llm_per_token_sec)
            self.tokens_yielded += 1
            yield token + " "
