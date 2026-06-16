"""Regression tests for the watch-capability adversarial security review fixes.

Each test pins a finding the review confirmed and we then fixed, so a regression
fails loudly:
- HIGH  : a fired alert leaked the owner-supplied label/condition VERBATIM (no
          redaction) into the spoken text + the git-committed run bundle (§9.7).
- MEDIUM: _X11Resolver matched wm_class as a SUBSTRING, letting an attacker window
          (e.g. class "Signal-Phisher") satisfy a grant for "signal".
- MEDIUM: an owner /regex/ condition + title_pattern ran unbounded on the shared
          poller thread -> catastrophic-backtracking DoS.
- LOW   : GrantStore._persist reset to {} on an unreadable local file, destroying
          every other machine-local key.
"""
from __future__ import annotations

import time

import pytest

from core.watch import GrantStore, TextMatchEvaluator, WatchGrant, WatchManager
from core.watch_source import Observation, _is_redos, safe_search, wm_class_matches


class _FixedSource:
    def __init__(self, obs):
        self._obs = obs

    def observe(self, app):
        return self._obs


def _fire_once(grant_dict, condition, obs_text, tmp_path):
    published = []
    store = GrantStore([grant_dict], local_path=str(tmp_path / "config.local.json"))
    mgr = WatchManager(
        store, source=_FixedSource(Observation(text=obs_text)),
        publish=lambda e: published.append(e), current_epoch=lambda: 3,
        autostart=False, min_poll_sec=1.0,
    )
    mgr.start_watch(grant_dict["id"], condition, interval_sec=1, one_shot=True)
    mgr.tick()
    return published


# --- HIGH: label + condition must be redacted in the spoken/committed alert -------

def test_alert_redacts_pii_in_label_and_condition(tmp_path):
    # PII in the LABEL (card) and the CONDITION (ssn); the condition matches the
    # observation so the watch fires and both must be scrubbed from the alert text.
    grant = {"id": "bank", "label": "Mom bank 4111 1111 1111 1111", "app": {"wm_class": "x"}}
    published = _fire_once(grant, "123-45-6789", "alert 123-45-6789 now", tmp_path)
    assert published, "watch should have fired on the matching observation"
    text = published[0].payload["text"]
    assert "4111" not in text and "123-45-6789" not in text
    assert "[REDACTED_CARD]" in text and "[REDACTED_SSN]" in text


# --- MEDIUM: wm_class is matched EXACTLY, never as a substring ---------------------

def test_wm_class_match_is_exact_not_substring():
    assert wm_class_matches("signal", "signal", "Signal")          # instance/class hit
    assert wm_class_matches("Firefox", "Navigator", "firefox")     # case-insensitive class hit
    # the proven spoof: an attacker window classed "Signal-Phisher" must NOT match.
    assert not wm_class_matches("signal", "signal-phisher", "Signal-Phisher")
    assert not wm_class_matches("ff", "ffmpeg", "FFmpeg")           # no substring bleed
    assert not wm_class_matches("", "anything", "Anything")         # empty grant matches nothing


# --- MEDIUM: ReDoS guard on owner-supplied regex over display text ----------------

def test_safe_search_rejects_catastrophic_pattern_fast():
    start = time.monotonic()
    assert safe_search("(a+)+$", "a" * 60 + "!") is None           # rejected, not run
    assert time.monotonic() - start < 1.0                          # would be many seconds if run
    assert _is_redos("(a+)+") and _is_redos(r"(\d+){2,}") and _is_redos("(a*)*")
    assert not _is_redos("Inbox.*Firefox") and not _is_redos(r"\d{3}-\d{2}")
    # a legitimate pattern still matches normally
    assert safe_search("Inbox.*Firefox", "My Inbox - Mozilla Firefox") is not None


def test_evaluator_regex_condition_is_redos_guarded():
    ev = TextMatchEvaluator()
    start = time.monotonic()
    met, _ = ev.evaluate(Observation(text="a" * 60 + "!"), "/(a+)+$/")
    assert met is False and time.monotonic() - start < 1.0
    # a normal regex condition still fires
    met, _ = ev.evaluate(Observation(text="Build SUCCESS"), "/succ\\w+/")
    assert met is True


# --- LOW: a corrupt local file is backed up + aborted, never clobbered -------------

def test_persist_does_not_clobber_unreadable_local_file(tmp_path):
    local = tmp_path / "config.local.json"
    local.write_text("{ this is not valid json")                  # pre-existing corrupt file
    store = GrantStore([], local_path=str(local))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        store.add(WatchGrant.from_dict({"id": "g", "label": "G", "app": {"wm_class": "x"}}))
    # the corrupt content is preserved (backed up), NOT replaced with a fresh {}
    assert (tmp_path / "config.local.json.corrupt").exists()
