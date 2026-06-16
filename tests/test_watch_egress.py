"""INV-4: egress redaction + spotlight-fencing for the watch/monitor capability.

The watch feature's ONLY egress is a spoken/published alert. Before any captured
screen text reaches that alert, ``WatchManager._fire`` MUST:

  1. redact PII (force=True) -- so an API key / email / card / SSN that was visible
     on the watched window never leaves the device verbatim, and
  2. spotlight-fence the (now-redacted) evidence -- so on-screen text that smuggled
     instructions is delimited as UNTRUSTED reference DATA, never obeyed.

These tests drive a real fire through ``WatchManager.tick`` with a FakeSource (no
real screen / mss / Xlib / tesseract) and assert on the published alert text. Each
test is self-contained with its own tiny fakes (Tier-0).

GrantStore is always pointed at a tmp ``config.local.json`` so the real machine-local
config is never touched.
"""
from __future__ import annotations

import pytest

from always_on_agent.events import AgentEvent, EventKind
from always_on_agent.origin import Origin
from always_on_agent.untrusted import (
    SPOTLIGHT_DIRECTIVE,
    _BEGIN,
    _END,
    redact_pii,
)
from core.watch import GrantStore, TextMatchEvaluator, WatchManager
from core.watch_source import Observation


# --- tiny, self-contained fakes --------------------------------------------

class FakeSource:
    """A WatchSource that yields one fixed Observation and counts calls."""

    def __init__(self, obs):
        self._obs = obs
        self.calls = 0

    def observe(self, app):
        self.calls += 1
        return self._obs


class FakeEvaluator:
    """An evaluator that always matches and returns a fixed raw evidence string --
    lets a test inject the exact PII payload to be redacted, without depending on
    the (snippet-windowed) real evaluator."""

    def __init__(self, evidence: str, met: bool = True):
        self._evidence = evidence
        self._met = met

    def evaluate(self, obs, condition):
        return self._met, self._evidence


# Raw secrets reused across tests -- assembled so each PII type is bounded by a
# non-digit label (the realistic "Key: ... Card: ..." OCR row), so adjacent
# digit-runs do not bleed into one another.
_KEY = "sk-ABCDEFGHIJKLMNOPQRSTUVWX"
_EMAIL = "alice@example.com"
_CARD = "4111 1111 1111 1111"  # Luhn-valid test card
_SSN = "123-45-6789"
_PII_EVIDENCE = f"Login Key: {_KEY} Email: {_EMAIL} Card: {_CARD} SSN: {_SSN}"


def _make_grant_dict():
    return {"id": "g1", "label": "Mail", "app": {"wm_class": "thunderbird"}}


def _fire_once(tmp_path, *, evidence_obs=None, evaluator=None, condition="alert"):
    """Build a manager over a tmp local config, arm a one-shot watch, run ONE tick
    (autostart=False -> no poller thread), and return the list of published events."""
    local = str(tmp_path / "config.local.json")
    store = GrantStore([_make_grant_dict()], local_path=local)
    published: list[AgentEvent] = []
    source = FakeSource(evidence_obs if evidence_obs is not None else Observation(text="x"))
    mgr = WatchManager(
        store,
        source=source,
        publish=published.append,
        current_epoch=lambda: 7,
        evaluator=evaluator,
        autostart=False,
    )
    mgr.start_watch("g1", condition, interval_sec=5.0, one_shot=True)
    mgr.tick()
    return published, source, mgr


# --- INV-4: redaction -------------------------------------------------------

def test_fire_redacts_all_pii_types_in_published_text(tmp_path):
    """The published alert shows [REDACTED_*] placeholders and contains NONE of the
    raw secrets (API key, email, card, SSN)."""
    published, _src, _mgr = _fire_once(
        tmp_path, evaluator=FakeEvaluator(_PII_EVIDENCE)
    )
    assert len(published) == 1
    ev = published[0]
    assert ev.kind == EventKind.TTS_REQUEST
    text = ev.payload["text"]

    # Every raw secret is gone.
    for raw in (_KEY, _EMAIL, _CARD, _SSN):
        assert raw not in text, f"raw secret {raw!r} leaked into the spoken alert"

    # Each PII type left its placeholder behind (so the alert is still informative).
    for placeholder in ("[REDACTED_KEY]", "[REDACTED_EMAIL]", "[REDACTED_CARD]", "[REDACTED_SSN]"):
        assert placeholder in text, f"missing {placeholder} -- redaction did not run"


def test_redaction_runs_even_with_local_record_redact_disabled(tmp_path, monkeypatch):
    """``_fire`` redacts with force=True, so the operator opt-out for *durable-record*
    scrubbing (SPEAKER_DISABLE_REDACT) can NEVER let raw PII reach the egress alert."""
    monkeypatch.setenv("SPEAKER_DISABLE_REDACT", "1")
    published, _src, _mgr = _fire_once(
        tmp_path, evaluator=FakeEvaluator(_PII_EVIDENCE)
    )
    text = published[0].payload["text"]
    for raw in (_KEY, _EMAIL, _CARD, _SSN):
        assert raw not in text
    assert "[REDACTED_KEY]" in text and "[REDACTED_SSN]" in text


# --- INV-4: spotlight fencing ----------------------------------------------

def test_fire_spotlight_fences_evidence(tmp_path):
    """The evidence portion is wrapped with wrap_untrusted(source="screen"): the
    published text carries the SPOTLIGHT directive and BOTH untrusted-data fence
    markers, with the redacted body between them."""
    published, _src, _mgr = _fire_once(
        tmp_path, evaluator=FakeEvaluator(_PII_EVIDENCE)
    )
    text = published[0].payload["text"]

    # The never-obey directive prefixes the fenced block.
    assert SPOTLIGHT_DIRECTIVE in text
    # Both fence markers (the per-process nonce'd <<<UNTRUSTED ... >>> pair) are present.
    assert _BEGIN in text
    assert _END in text
    # It is tagged as screen-sourced untrusted data.
    assert "[untrusted screen]" in text
    # The redacted evidence sits strictly between the fences (i.e. inside the envelope).
    begin_at = text.index(_BEGIN)
    end_at = text.index(_END)
    assert begin_at < end_at
    body = text[begin_at:end_at]
    assert "[REDACTED_CARD]" in body and "[REDACTED_KEY]" in body
    # Each fence marker appears exactly once -- a forged fence in the content would
    # have been stripped; here there is none, so the boundary is unambiguous.
    assert text.count(_BEGIN) == 1
    assert text.count(_END) == 1


def test_published_alert_shape(tmp_path):
    """The egress event is a system-initiated SCREEN-origin TTS request, epoch-stamped
    and keyed to the watch id (INV-6: it never carries the granting turn's owner
    trust, so watched text can shape a heads-up but never originate an action)."""
    published, _src, _mgr = _fire_once(
        tmp_path, evaluator=FakeEvaluator(_PII_EVIDENCE)
    )
    payload = published[0].payload
    assert published[0].kind == EventKind.TTS_REQUEST
    assert payload["origin"] == Origin.SCREEN.value
    assert payload["epoch"] == 7
    assert payload["task_id"] == "w1"
    # The human-readable lead-in names the granted app label (not raw screen text).
    assert payload["text"].startswith("Heads up: Mail")


# --- the source path stamps SCREEN origin ----------------------------------

def test_observation_origin_is_screen_on_source_path(tmp_path):
    """An Observation produced via the watch source path carries Origin.SCREEN.value
    (capture is attacker-controllable DATA, never an action-trusted channel). Driving
    the real TextMatchEvaluator over such an Observation still fires + redacts."""
    obs = Observation(text=f"BREACH {_KEY} {_EMAIL}")
    assert obs.origin == Origin.SCREEN.value  # default on the source path

    # Real evaluator (substring match) -> fire -> redacted egress.
    published, source, _mgr = _fire_once(
        tmp_path,
        evidence_obs=obs,
        evaluator=TextMatchEvaluator(),
        condition="BREACH",
    )
    assert source.calls == 1
    assert len(published) == 1
    text = published[0].payload["text"]
    assert _KEY not in text and _EMAIL not in text
    assert "[REDACTED_KEY]" in text and "[REDACTED_EMAIL]" in text
    # And the egress event is still stamped SCREEN-origin.
    assert published[0].payload["origin"] == Origin.SCREEN.value


# --- regression: a card immediately abutting an SSN must STILL be redacted ----
# (Found via this suite: the greedy _CARD_RE over-consumed the SSN's leading digits
# into the card match, the combined run failed Luhn, and the raw card leaked. Fixed
# in always_on_agent.untrusted._redact_card_span -- recover the Luhn-valid card
# window inside an over-greedy match. See tests/test_untrusted.py for the direct net.)

def test_card_abutting_ssn_is_still_redacted(tmp_path):
    # Card directly followed (single space) by an SSN -- a plausible OCR row.
    evidence = f"Payment {_CARD} {_SSN}"
    published, _src, _mgr = _fire_once(
        tmp_path, evaluator=FakeEvaluator(evidence)
    )
    text = published[0].payload["text"]
    # The raw card never appears in the spoken alert; both PII types are scrubbed.
    assert _CARD not in text
    assert _SSN not in text
    assert "[REDACTED_CARD]" in text and "[REDACTED_SSN]" in text
