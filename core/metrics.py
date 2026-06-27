from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

# Per-turn latency instrumentation shared by the real engine, the file-replay
# engine, and the simulated sandbox engine, so measured and simulated numbers
# land in the same shape and can be compared against the specsim budgets.
#
# A "turn" is one user utterance and the assistant's response to it. We record
# monotonic ``time.perf_counter()`` stamps at named stage boundaries; the
# deltas between them are the user-facing latencies (first audio, barge-in
# stop). Stamps are fed from three threads -- the capture loop, the bus/worker
# thread that plays TTS, and the task thread running the LLM -- so the recorder
# is lock-guarded. It computes nothing on the hot path beyond a dict write.

SPEECH_END = "speech_end"          # user stopped speaking (last voiced audio)
ASR_FINAL = "asr_final"            # recognizer emitted the final transcript
LLM_FIRST_TOKEN = "llm_first_token"  # first token streamed from the model
TTS_FIRST_AUDIO = "tts_first_audio"  # assistant's first audio sample played
BARGE_IN = "barge_in"              # user spoke over playback
BARGE_IN_STOP = "barge_in_stop"    # playback actually halted
HANDLED_LOCAL = "handled_local"    # turn resolved with NO LLM (intent fast-path) -- never reaches first token
HELD = "held"                      # final is intentionally held for turn-merge; not yet dispatched
MERGED = "merged"                  # turn was folded into a newer merged continuation
SUPERSEDED = "superseded"          # turn preempted by a newer final (newest-input-wins) -- cancelled pre-answer

# A new utterance begins at whichever of these we see first (speech_end leads
# asr_final, but the real streaming engine only knows the latter).
_TURN_START = (SPEECH_END, ASR_FINAL)


def _delta(stamps: dict[str, float], a: str, b: str) -> Optional[float]:
    if a in stamps and b in stamps:
        return stamps[b] - stamps[a]
    return None


@dataclass
class TurnRecord:
    """Stage stamps for one turn (seconds, ``perf_counter`` epoch) + deltas."""

    stamps: dict[str, float] = field(default_factory=dict)

    @property
    def _anchor(self) -> Optional[str]:
        # First-audio latency is measured from when the user stopped speaking.
        # Prefer a true speech_end; fall back to the ASR final (the streaming
        # endpointer fires it at end-of-speech, so they nearly coincide).
        if SPEECH_END in self.stamps:
            return SPEECH_END
        if ASR_FINAL in self.stamps:
            return ASR_FINAL
        return None

    @property
    def first_audio_latency(self) -> Optional[float]:
        anchor = self._anchor
        if anchor is None or TTS_FIRST_AUDIO not in self.stamps:
            return None
        return self.stamps[TTS_FIRST_AUDIO] - self.stamps[anchor]

    @property
    def endpoint_latency(self) -> Optional[float]:
        return _delta(self.stamps, SPEECH_END, ASR_FINAL)

    @property
    def final_to_first_token(self) -> Optional[float]:
        return _delta(self.stamps, ASR_FINAL, LLM_FIRST_TOKEN)

    @property
    def first_token_to_audio(self) -> Optional[float]:
        return _delta(self.stamps, LLM_FIRST_TOKEN, TTS_FIRST_AUDIO)

    @property
    def barge_in_latency(self) -> Optional[float]:
        return _delta(self.stamps, BARGE_IN, BARGE_IN_STOP)

    def as_dict(self) -> dict[str, Optional[float]]:
        return {
            "first_audio_latency": _round(self.first_audio_latency),
            "endpoint_latency": _round(self.endpoint_latency),
            "final_to_first_token": _round(self.final_to_first_token),
            "first_token_to_audio": _round(self.first_token_to_audio),
            "barge_in_latency": _round(self.barge_in_latency),
        }


def _round(value: Optional[float]) -> Optional[float]:
    return round(value, 4) if value is not None else None


# EWMA smoothing factor for the rolling local time-to-first-token estimate.
# Higher = more reactive to the latest turn (a freshly loaded model surfaces
# fast); lower = steadier. 0.3 trusts recent turns while damping single-turn
# spikes. Updated cheaply on the hot path -- one multiply-add per turn.
_TTFT_EWMA_ALPHA = 0.3


class MetricsRecorder:
    """Collects :class:`TurnRecord`s as stage stamps arrive from any thread."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._lock = threading.Lock()
        self._current: Optional[TurnRecord] = None
        self._completed: list[TurnRecord] = []
        # Rolling local time-to-first-token estimate (milliseconds), updated as
        # ``llm_first_token`` is stamped against the open turn's ASR_FINAL
        # anchor. ``None`` until the first measurable turn -- the router treats
        # an unknown signal as "no nudge" so a cold start can never bias it.
        self._ttft_ewma_ms: Optional[float] = None
        # Rolling local TTS first-audio EWMA (ms): first_token -> tts_first_audio,
        # i.e. the synth latency of the turn's first sentence. Feeds the watchdog's
        # adaptive "tts stuck" deadline (control-plane-3). None until first sample.
        self._tts_ewma_ms: Optional[float] = None

    def mark(self, stage: str, *, fold_local_ttft: bool = True, at: Optional[float] = None) -> None:
        """Stamp ``stage`` on the open turn.

        ``fold_local_ttft`` (default True) controls only the LOCAL TTFT EWMA
        fold on the ``llm_first_token`` stamp -- the stamp itself (and every
        recorded latency) is always set regardless. Pass ``False`` so a turn
        whose first token came from a CLOUD hedge winner is still *recorded*
        but is NOT folded into the LOCAL headroom EWMA, which would otherwise
        mislabel a fast cloud answer as a fast local tier (P4 low). The route
        is known only at the capability call site, so the fold gate is decided
        there (see :mod:`core.capabilities`) and threaded through here.

        ``at`` (default ``None`` -> stamp ``now``) lets a caller record the
        stage at a *known earlier* instant in this recorder's clock epoch
        (``perf_counter``). The engine uses it to stamp ``SPEECH_END`` at the
        true silence onset rather than when the endpointer fires ~0.8s later,
        so ``endpoint_latency`` stops reading 0 and the fixed trailing-silence
        cost becomes visible (lat-1). Must be a ``perf_counter`` value; a value
        in any other clock epoch would corrupt the deltas."""
        now = self._clock() if at is None else float(at)
        with self._lock:
            if stage in _TURN_START:
                # A repeat of the same start stage signals the next utterance:
                # bank the open turn before opening a fresh one.
                if self._current is not None and stage in self._current.stamps:
                    self._completed.append(self._current)
                    self._current = None
                if self._current is None:
                    self._current = TurnRecord()
                self._current.stamps.setdefault(stage, now)
            else:
                # Mid-turn stamps only count once and only inside an open turn;
                # a stray late event (e.g. trailing speech_end) is ignored.
                if self._current is not None:
                    if stage == LLM_FIRST_TOKEN and stage not in self._current.stamps:
                        # First token of this turn just landed: fold the local
                        # ASR_FINAL -> first-token delta into the rolling EWMA --
                        # but ONLY when the answering tier was local
                        # (``fold_local_ttft``). A cloud-hedge win is recorded
                        # (stamp set below) yet skipped here so the LOCAL
                        # headroom signal isn't mislabeled (P4 low). Cheap (one
                        # branch + one float op), no change to recording.
                        anchor = self._current.stamps.get(ASR_FINAL)
                        if fold_local_ttft and anchor is not None:
                            self._observe_ttft_ms((now - anchor) * 1000.0)
                    elif stage == TTS_FIRST_AUDIO and stage not in self._current.stamps:
                        # First audio just landed: fold the first-token -> audio
                        # delta (the TTS synth latency of sentence 1) into its own
                        # rolling EWMA. TTS is always local, so no cloud gate. The
                        # watchdog scales its "tts stuck" deadline off this so a
                        # slow device isn't false-flagged (control-plane-3).
                        anchor = self._current.stamps.get(LLM_FIRST_TOKEN)
                        if anchor is not None:
                            self._observe_tts_ms((now - anchor) * 1000.0)
                    self._current.stamps.setdefault(stage, now)

    def _observe_ttft_ms(self, ttft_ms: float) -> None:
        """Fold one observed local TTFT (ms) into the rolling EWMA.

        Must be called under ``self._lock``. Ignores non-finite/negative
        samples so a clock glitch can never poison the estimate."""
        if not math.isfinite(ttft_ms) or ttft_ms < 0.0:  # rejects NaN and +inf
            return
        if self._ttft_ewma_ms is None:
            self._ttft_ewma_ms = ttft_ms
        else:
            self._ttft_ewma_ms = (
                _TTFT_EWMA_ALPHA * ttft_ms
                + (1.0 - _TTFT_EWMA_ALPHA) * self._ttft_ewma_ms
            )

    def recent_ttft_ms(self) -> Optional[float]:
        """Cheap read of the rolling local time-to-first-token EWMA (ms).

        Returns ``None`` until at least one turn has produced a measurable
        ASR_FINAL -> first-token delta. The router consumes this as a *live
        signal*: a high value (local model slow/loaded) nudges borderline
        turns toward the main/cloud tier, while ``None`` leaves the static
        per-profile decision untouched (it can never starve the local tier)."""
        with self._lock:
            return self._ttft_ewma_ms

    def _observe_tts_ms(self, tts_ms: float) -> None:
        """Fold one observed local first-token -> first-audio delta (ms) into the
        rolling TTS EWMA. Must be called under ``self._lock``; ignores
        non-finite/negative samples (mirrors :meth:`_observe_ttft_ms`)."""
        if not math.isfinite(tts_ms) or tts_ms < 0.0:  # rejects NaN and +inf
            return
        if self._tts_ewma_ms is None:
            self._tts_ewma_ms = tts_ms
        else:
            self._tts_ewma_ms = (
                _TTFT_EWMA_ALPHA * tts_ms
                + (1.0 - _TTFT_EWMA_ALPHA) * self._tts_ewma_ms
            )

    def recent_tts_ms(self) -> Optional[float]:
        """Cheap read of the rolling local TTS first-audio EWMA (ms). ``None``
        until a turn has produced a measurable first-token -> first-audio delta.
        The watchdog scales its "tts stuck" deadline off this (control-plane-3)."""
        with self._lock:
            return self._tts_ewma_ms

    def mark_superseded_turn(self) -> None:
        """Stamp ``SUPERSEDED`` on the most-recently-banked turn (rc-5).

        Newest-input-wins (``core/runtime.py``) processes the NEW final, whose
        turn-start mark (``SPEECH_END`` on the sherpa engine, else ``ASR_FINAL``
        in tests) banks the in-flight turn into ``_completed``, and only THEN
        cancels it. So the turn being preempted is ``_completed[-1]`` (not the
        open ``_current``) by the time this runs. Marking it lets the watchdog
        skip it instead of mis-reading its missing ``llm_first_token`` as a
        stalled LLM. No-op if nothing has been banked yet (the supersede guard
        implies an in-flight turn, so this is belt-and-braces)."""
        with self._lock:
            if self._completed:
                self._completed[-1].stamps.setdefault(SUPERSEDED, self._clock())

    def mark_merged_turn(self) -> None:
        """Stamp the most-recently-banked turn as folded into a merged follow-up.

        Continuation merge has the same metrics shape as newest-input-wins: the
        follow-up's ASR_FINAL banks the earlier turn, then the supervisor cancels
        that earlier task and starts one synthetic merged task. Marking the
        banked turn keeps the watchdog from mistaking the cancelled pre-audio
        turn for an LLM stall while still letting the merged replacement turn be
        checked normally.
        """
        with self._lock:
            if self._completed:
                now = self._clock()
                self._completed[-1].stamps.setdefault(MERGED, now)
                self._completed[-1].stamps.setdefault(SUPERSEDED, now)

    def close_turn(self) -> None:
        """Bank the open turn (call once a replayed utterance has settled)."""
        with self._lock:
            if self._current is not None:
                self._completed.append(self._current)
                self._current = None

    def records(self) -> list[TurnRecord]:
        with self._lock:
            out = list(self._completed)
            if self._current is not None:
                out.append(self._current)
            return out

    def reset(self) -> None:
        with self._lock:
            self._current = None
            self._completed = []
            self._ttft_ewma_ms = None
            self._tts_ewma_ms = None


def mark_first_token(
    tokens: Iterator[str],
    recorder: Optional[MetricsRecorder],
    *,
    fold_local_ttft: bool = True,
) -> Iterator[str]:
    """Wrap an LLM token stream to stamp ``llm_first_token`` on the first token.

    ``fold_local_ttft`` (default True) is forwarded to :meth:`MetricsRecorder.mark`
    so a caller that answered from a non-local source (a cloud hedge winner) can
    still stamp the turn while keeping the sample out of the LOCAL TTFT EWMA
    (P4 low). The default preserves the historical fold-always behaviour."""
    if recorder is None:
        yield from tokens
        return
    first = True
    for token in tokens:
        if first:
            recorder.mark(LLM_FIRST_TOKEN, fold_local_ttft=fold_local_ttft)
            first = False
        yield token
