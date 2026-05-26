"""
Turn detector interface for explicit speech turn boundaries.
"""

from __future__ import annotations

import time
import json
import os
import threading
from dataclasses import dataclass


@dataclass
class TurnDecision:
    should_end_turn: bool
    reason: str


class TurnDetector:
    """
    Lightweight turn detector:
    - uses recent partial transcript activity as continuation evidence
    - falls back to endpoint timeout when no new partial text arrives
    """

    def __init__(
        self,
        min_endpointing_delay: float = 0.5,
        max_endpointing_delay: float = 3.0,
        diagnostics_log_path: str | None = None,
    ):
        self.min_endpointing_delay = min_endpointing_delay
        self.max_endpointing_delay = max_endpointing_delay
        self._last_partial_ts = 0.0
        self._last_partial_text = ""
        self.diagnostics_log_path = diagnostics_log_path
        self._diag_lock = threading.Lock()

    def _diag_log(self, event: str, **payload):
        if not self.diagnostics_log_path:
            return
        record = {"ts": time.time(), "event": event, "component": "turn_detector"}
        record.update(payload)
        try:
            with self._diag_lock:
                parent = os.path.dirname(self.diagnostics_log_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self.diagnostics_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            pass

    def on_partial_text(self, text: str):
        if text and text.strip():
            self._last_partial_text = text.strip()
            self._last_partial_ts = time.time()
            self._diag_log("turn_partial", partial_len=len(self._last_partial_text))

    def on_final_text(self, text: str):
        if text and text.strip():
            self._last_partial_text = text.strip()
            self._last_partial_ts = time.time()
            self._diag_log("turn_final", final_len=len(self._last_partial_text))

    def evaluate(self, silence_sec: float) -> TurnDecision:
        now = time.time()
        partial_age = now - self._last_partial_ts if self._last_partial_ts else 999.0
        if silence_sec < self.min_endpointing_delay:
            decision = TurnDecision(False, "silence_too_short")
            self._diag_log("turn_eval", silence_sec=silence_sec, reason=decision.reason)
            return decision
        if partial_age <= self.min_endpointing_delay:
            decision = TurnDecision(False, "recent_partial")
            self._diag_log("turn_eval", silence_sec=silence_sec, reason=decision.reason)
            return decision
        if silence_sec >= self.max_endpointing_delay:
            decision = TurnDecision(True, "max_endpoint_timeout")
            self._diag_log("turn_eval", silence_sec=silence_sec, reason=decision.reason)
            return decision
        decision = TurnDecision(True, "silence_endpoint")
        self._diag_log("turn_eval", silence_sec=silence_sec, reason=decision.reason)
        return decision
