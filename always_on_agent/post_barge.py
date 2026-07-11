"""Bounded provenance token for the first final after a real barge-in.

The acoustic engine's ``on_barge_in`` callback is stronger evidence than an
addressing model's guess: it proves that a live capture episode interrupted an
active reply.  It is *not* speaker identity and must never mint action trust.
This small state machine lets the runtime remember that conversational context
across held/merged ASR finals while keeping admission one-shot, epoch-bound,
and time-bounded.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Lock
import time

from .origin import Origin


@dataclass(frozen=True)
class PostBargeFinalObservation:
    """Snapshot of an armed grant observed by one terminal final candidate."""

    token: int
    input_epoch: int
    expires_at: float
    response_eligible: bool


class PostBargeResponseGate:
    """One-shot, monotonic post-barge response admission.

    ``inspect`` is deliberately non-destructive: a final-preprocessing lease may
    still be cancelled by a newer partial/final.  The winning runtime terminal
    seam calls ``consume``.  Token equality prevents an older lease from
    consuming a grant re-armed by a newer barge-in.
    """

    def __init__(self, window_sec: float = 8.0) -> None:
        try:
            configured = float(window_sec)
        except (TypeError, ValueError):
            configured = 0.0
        self.window_sec = (
            configured if math.isfinite(configured) and configured > 0.0 else 0.0
        )
        self._lock = Lock()
        self._next_token = 0
        self._armed: tuple[int, int, float] | None = None

    def arm(self, input_epoch: int, *, now: float | None = None) -> None:
        at = time.monotonic() if now is None else float(now)
        with self._lock:
            self._next_token += 1
            self._armed = (
                self._next_token,
                int(input_epoch),
                at + self.window_sec,
            )

    def inspect(
        self,
        input_epoch: int,
        origin: str,
        *,
        now: float | None = None,
    ) -> PostBargeFinalObservation | None:
        """Return the current grant snapshot, expiring stale/foreign epochs.

        A non-live origin is still returned as an ineligible observation so the
        winning final can consume the one-shot grant.  It can never be followed
        by a second, more convenient final that inherits the old interruption.
        """

        at = time.monotonic() if now is None else float(now)
        with self._lock:
            armed = self._armed
            if armed is None:
                return None
            token, armed_epoch, expires_at = armed
            if (
                self.window_sec <= 0.0
                or at >= expires_at
                or int(input_epoch) != armed_epoch
            ):
                self._armed = None
                return None
            return PostBargeFinalObservation(
                token=token,
                input_epoch=armed_epoch,
                expires_at=expires_at,
                response_eligible=(str(origin) == Origin.LIVE_AUDIO.value),
            )

    def consume(self, observation: PostBargeFinalObservation) -> bool:
        """Consume exactly the grant ``observation`` inspected.

        Expiry is evaluated when the final is observed, not after potentially
        slow addressing/cleanup.  Once a timely final is in preprocessing, a
        slow local model cannot turn it into "late ambient".
        """

        with self._lock:
            armed = self._armed
            if armed is None:
                return False
            token, input_epoch, _expires_at = armed
            if (
                token != observation.token
                or input_epoch != observation.input_epoch
            ):
                return False
            self._armed = None
            return True

    def invalidate(self) -> None:
        with self._lock:
            self._armed = None

    def is_armed(self, input_epoch: int, *, now: float | None = None) -> bool:
        """Whether the current epoch still owns an unexpired grant."""

        # An unknown origin cannot become eligible, but inspect still performs
        # the shared expiry/epoch cleanup without exposing internal state.
        return self.inspect(input_epoch, Origin.UNKNOWN.value, now=now) is not None


__all__ = ["PostBargeFinalObservation", "PostBargeResponseGate"]
