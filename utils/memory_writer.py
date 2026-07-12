"""
Smart, debounced Postgres persistence for user speech only.

Filters junk/boilerplate, dedupes near-duplicates, optionally cleans text via
a small local LLM, and flushes on a timer or session end.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Callable, List, Optional, Protocol

from utils.memory_config import MemoryWriterConfig, config_from_dict

# Junk markers aligned with VoiceAssistant._is_junk_stt_text
_JUNK_MARKERS = (
    "[blank_audio]",
    "[blank audio]",
    "(blank audio)",
    "blank_audio",
    "[birds chirping]",
    "(birds chirping)",
    "birds chirping",
    "[music]",
    "(music playing)",
)


def normalize_for_dedupe(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_junk_stt_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if len(t) <= 2 and t in (".", "?", "!", "…"):
        return True
    return any(m in t for m in _JUNK_MARKERS)


def texts_near_duplicate(a: str, b: str, threshold: float) -> bool:
    na, nb = normalize_for_dedupe(a), normalize_for_dedupe(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def looks_like_assistant_echo(user_text: str, assistant_text: str, threshold: float) -> bool:
    if not assistant_text.strip():
        return False
    return texts_near_duplicate(user_text, assistant_text, threshold)


@dataclass
class UtteranceCandidate:
    raw_text: str
    source: str = "user_final"
    confidence: float = 1.0
    session_id: str = ""
    queued_at: float = field(default_factory=time.time)
    last_assistant_text: str = ""


@dataclass
class CleanupResult:
    worth_saving: bool
    cleaned_text: str
    reason: str = ""


class LLMCleanupClient(Protocol):
    def cleanup(self, text: str, *, gate: bool) -> CleanupResult: ...


class TextGenerationClient(Protocol):
    def generate(self, prompt: str, *, system: Optional[str] = None) -> str: ...


def _cleanup_prompt(text: str, *, gate: bool) -> tuple[str, str]:
    system = (
        "You normalize voice transcripts for long-term memory. "
        "Fix obvious STT typos, punctuation, and casing. "
        "Do not invent facts. Keep the user's meaning. "
        "If the line is only filler, noise, or a control phrase (stop/quit), "
        "set worth_saving false."
    )
    if gate:
        system += " Only save substantive user content worth recalling later."
    prompt = (
        f"Transcript:\n{text}\n\n"
        "Reply with JSON only: "
        '{"worth_saving": bool, "cleaned_text": str, "reason": str}'
    )
    return system, prompt


def _decode_cleanup(raw: str, fallback: str) -> CleanupResult:
    """Decode one JSON object, tolerating a model's surrounding prose/fence."""

    decoder = json.JSONDecoder()
    data = None
    for index, char in enumerate(raw or ""):
        if char != "{":
            continue
        try:
            candidate, _end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            data = candidate
            break
    if data is None:
        raise ValueError("cleanup response contained no JSON object")

    raw_worth = data.get("worth_saving", True)
    worth = raw_worth if isinstance(raw_worth, bool) else True
    raw_cleaned = data.get("cleaned_text", fallback)
    cleaned = raw_cleaned.strip() if isinstance(raw_cleaned, str) else fallback.strip()
    cleaned = cleaned or fallback.strip()
    raw_reason = data.get("reason", "")
    reason = raw_reason if isinstance(raw_reason, str) else ""
    if not cleaned:
        worth = False
    return CleanupResult(worth, cleaned, reason)


class LLMClientMemoryCleanup:
    """Cleanup through the runtime's already-configured fast LLM client."""

    def __init__(self, client: TextGenerationClient):
        self.client = client

    def cleanup(self, text: str, *, gate: bool) -> CleanupResult:
        system, prompt = _cleanup_prompt(text, gate=gate)
        try:
            generate_json = getattr(self.client, "generate_json", None)
            raw = (
                generate_json(prompt, system=system)
                if callable(generate_json)
                else self.client.generate(prompt, system=system)
            )
            return _decode_cleanup(raw, text)
        except Exception as exc:
            return CleanupResult(True, text.strip(), f"llm_error:{exc}")


class OllamaMemoryCleanup:
    """Structured cleanup / gate via Ollama (mockable in tests)."""

    def __init__(self, model: str, *, enabled: bool = True):
        self.model = model
        self.enabled = enabled

    def cleanup(self, text: str, *, gate: bool) -> CleanupResult:
        if not self.enabled:
            return CleanupResult(True, text.strip(), "llm_disabled")

        system, prompt = _cleanup_prompt(text, gate=gate)
        try:
            import ollama

            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={"temperature": 0.1, "num_predict": 256},
            )
            raw = resp.get("message", {}).get("content", "{}")
            return _decode_cleanup(raw, text)
        except Exception as exc:
            return CleanupResult(True, text.strip(), f"llm_error:{exc}")


class UtteranceBuffer:
    def __init__(self, config: MemoryWriterConfig):
        self.config = config
        self._items: List[UtteranceCandidate] = []
        self._lock = threading.Lock()

    def add(self, item: UtteranceCandidate) -> bool:
        with self._lock:
            for existing in self._items:
                if texts_near_duplicate(
                    existing.raw_text,
                    item.raw_text,
                    self.config.dedupe_similarity,
                ):
                    return False
            self._items.append(item)
            while len(self._items) > self.config.max_buffer_items:
                self._items.pop(0)
            return True

    def drain(self) -> List[UtteranceCandidate]:
        with self._lock:
            items = list(self._items)
            self._items.clear()
            return items

    def pending_count(self) -> int:
        with self._lock:
            return len(self._items)


def should_persist(
    candidate: UtteranceCandidate,
    config: MemoryWriterConfig,
    *,
    recent_saved: Optional[List[str]] = None,
) -> tuple[bool, str]:
    text = (candidate.raw_text or "").strip()
    if not text or len(text) < config.min_chars:
        return False, "too_short"
    if candidate.confidence < config.min_confidence:
        return False, "low_confidence"
    if candidate.source == "user_partial":
        return False, "partial_not_allowed"
    if is_junk_stt_text(text):
        return False, "junk_stt"
    if looks_like_assistant_echo(
        text, candidate.last_assistant_text, config.dedupe_similarity
    ):
        return False, "assistant_echo"
    norm = normalize_for_dedupe(text)
    if not config.save_control_phrases and norm in config.boilerplate_phrases:
        return False, "boilerplate"
    if recent_saved:
        for prev in recent_saved:
            if texts_near_duplicate(prev, text, config.dedupe_similarity):
                return False, "duplicate"
    return True, "ok"


class MemoryWriter:
    """Buffers user utterances and persists on interval or flush."""

    def __init__(
        self,
        *,
        config: MemoryWriterConfig,
        persist_fn: Callable[..., None],
        llm_client: Optional[LLMCleanupClient] = None,
        text_cleaner: Optional[Callable[[str, str], Optional[str]]] = None,
    ):
        self.config = config
        self._persist_fn = persist_fn
        self._text_cleaner = text_cleaner
        self._buffer = UtteranceBuffer(config)
        self._llm = llm_client
        if config.llm_cleanup or config.llm_gate:
            self._llm = llm_client or OllamaMemoryCleanup(
                config.cleanup_model,
                enabled=config.llm_cleanup or config.llm_gate,
            )
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._recent_saved: List[str] = []
        self._timer: Optional[threading.Timer] = None
        self._closed = False
        # A flush removes candidates from ``_buffer`` before optional model
        # cleanup.  Keep those drained candidates lifecycle-visible until their
        # persistence decision is committed.  ``close()`` can then revoke a
        # blocked cleaner's ownership and persist the candidate raw without
        # waiting for inference; the old worker must re-check the token before
        # it is allowed to touch the persistence backend.
        self._inflight: dict[int, UtteranceCandidate] = {}
        self._next_inflight_id = 0

    def set_text_cleaner(
        self, cleaner: Optional[Callable[[str, str], Optional[str]]]
    ) -> None:
        with self._lock:
            self._text_cleaner = cleaner

    def enqueue(
        self,
        raw_text: str,
        *,
        source: str = "user_final",
        confidence: float = 1.0,
        last_assistant_text: str = "",
    ) -> bool:
        if not self.config.enabled:
            return False
        candidate = UtteranceCandidate(
            raw_text=raw_text,
            source=source,
            confidence=confidence,
            last_assistant_text=last_assistant_text or "",
        )
        should_flush = False
        with self._lock:
            if self._closed:
                return False
            ok, _reason = should_persist(
                candidate, self.config, recent_saved=self._recent_saved
            )
            if not ok:
                return False
            added = self._buffer.add(candidate)
            if added:
                self._schedule_flush_timer_locked()
                should_flush = (
                    self._buffer.pending_count() >= self.config.max_buffer_items
                )
        if should_flush:
            self.flush(force=True)
        return added

    def _schedule_flush_timer_locked(self) -> None:
        if self._closed or self._timer is not None:
            return
        delay = max(0.5, self.config.save_interval_sec)
        self._timer = threading.Timer(delay, self._timer_flush)
        self._timer.daemon = True
        self._timer.start()

    def schedule_job(self, fn: Callable[[], None]) -> bool:
        """Run ``fn`` on a background daemon thread, off the caller's thread.

        The one off-the-hot-path scheduling primitive (R2): the rolling-summary
        LLM call and any profile LLM extraction MUST NOT run on the single bus
        thread that drives ASR/LLM/TTS. ``MemoryManager`` hands those jobs here
        so the calling ``add_message``/``queue_user_utterance`` returns promptly.
        Returns ``False`` (job dropped) if the writer is already closed."""
        if self._closed:
            return False
        worker = threading.Thread(target=self._run_job, args=(fn,), daemon=True)
        worker.start()
        return True

    @staticmethod
    def _run_job(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - background jobs must not crash the thread
            print(f"⚠️  Memory background job failed: {exc}")

    def _timer_flush(self) -> None:
        with self._lock:
            self._timer = None
        self.flush()

    def flush(self, *, force: bool = False, use_llm: bool = True) -> int:
        with self._lock:
            # ``force`` bypasses the debounce interval, not lifecycle closure.
            # Only ``close()`` may persist after marking the writer closed.
            if self._closed:
                return 0
            elapsed = time.monotonic() - self._last_flush
            if not force and elapsed < self.config.save_interval_sec:
                if self._buffer.pending_count() < self.config.max_buffer_items:
                    return 0

            work: list[tuple[int, UtteranceCandidate]] = []
            for item in self._buffer.drain():
                token = self._next_inflight_id
                self._next_inflight_id += 1
                self._inflight[token] = item
                work.append((token, item))

        saved = 0
        for token, item in work:
            with self._lock:
                if self._closed or self._inflight.get(token) is not item:
                    continue
                ok, _reason = should_persist(
                    item, self.config, recent_saved=self._recent_saved
                )
                if not ok:
                    self._inflight.pop(token, None)
                    continue
                text_cleaner = self._text_cleaner
                llm = self._llm

            cleaned = item.raw_text.strip()
            worth = True
            if use_llm and text_cleaner is not None:
                try:
                    maybe = text_cleaner("user", cleaned)
                    if maybe is None:
                        with self._lock:
                            if self._inflight.get(token) is item:
                                self._inflight.pop(token, None)
                        continue
                    cleaned = maybe.strip()
                    if not cleaned:
                        with self._lock:
                            if self._inflight.get(token) is item:
                                self._inflight.pop(token, None)
                        continue
                except Exception:
                    pass
            elif (
                use_llm
                and llm
                and (self.config.llm_cleanup or self.config.llm_gate)
            ):
                result = llm.cleanup(cleaned, gate=self.config.llm_gate)
                worth = result.worth_saving
                if self.config.llm_cleanup and result.cleaned_text:
                    cleaned = result.cleaned_text
            if not worth or not cleaned:
                with self._lock:
                    if self._inflight.get(token) is item:
                        self._inflight.pop(token, None)
                continue

            # Persistence is the commit point.  Holding the lifecycle lock here
            # makes it impossible for ``close()`` to close the pool underneath
            # a call that has already committed to using it.  Slow model work is
            # deliberately outside this lock.
            with self._lock:
                if self._closed or self._inflight.get(token) is not item:
                    continue
                ok, _reason = should_persist(
                    item, self.config, recent_saved=self._recent_saved
                )
                if not ok:
                    self._inflight.pop(token, None)
                    continue
                captured_at = datetime.now()
                self._persist_fn(
                    raw_text=item.raw_text,
                    cleaned_text=cleaned,
                    source=item.source,
                    confidence=item.confidence,
                    captured_at=captured_at,
                    reason="",
                )
                self._recent_saved.append(item.raw_text)
                if len(self._recent_saved) > 50:
                    self._recent_saved = self._recent_saved[-50:]
                self._inflight.pop(token, None)
                saved += 1

        with self._lock:
            if not self._closed and (saved or force):
                self._last_flush = time.monotonic()
        return saved

    def close(self) -> int:
        with self._lock:
            if self._closed:
                return 0
            self._closed = True
            if self._timer:
                self._timer.cancel()
                self._timer = None

            # Revoke all flush ownership before persisting.  A worker blocked in
            # optional cleanup will later find its token absent and return
            # without touching a pool that the caller may already have closed.
            items = list(self._inflight.values())
            self._inflight.clear()
            items.extend(self._buffer.drain())

            # Shutdown is a bounded persistence path, not a second inference
            # batch: up to 32 pending items times the fast client's 60s timeout
            # could otherwise hold teardown for many minutes.  Re-run the
            # deterministic admission because another concurrent candidate may
            # have committed since enqueue, then persist the original text.
            saved = 0
            for item in items:
                ok, _reason = should_persist(
                    item, self.config, recent_saved=self._recent_saved
                )
                cleaned = item.raw_text.strip()
                if not ok or not cleaned:
                    continue
                self._persist_fn(
                    raw_text=item.raw_text,
                    cleaned_text=cleaned,
                    source=item.source,
                    confidence=item.confidence,
                    captured_at=datetime.now(),
                    reason="",
                )
                self._recent_saved.append(item.raw_text)
                if len(self._recent_saved) > 50:
                    self._recent_saved = self._recent_saved[-50:]
                saved += 1
            self._last_flush = time.monotonic()
            return saved

    @property
    def pending_count(self) -> int:
        with self._lock:
            return self._buffer.pending_count() + len(self._inflight)
