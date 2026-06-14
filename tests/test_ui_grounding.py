"""Read-only on-screen element identification (core/ui_grounding.py) + the
screen.identify capability. Tier-0: pure matching over hand-built word boxes +
fake capture/OCR -- no display, no tesseract, no mss.
"""
from __future__ import annotations

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.untrusted import SPOTLIGHT_DIRECTIVE
from core.ui_grounding import (
    Element,
    attach_computer_use_capability,
    find_targets,
    render_elements,
)

_WORDS = [
    {"text": "Save", "left": 10, "top": 20, "width": 40, "height": 12, "conf": 95},
    {"text": "Cancel", "left": 60, "top": 20, "width": 50, "height": 12, "conf": 90},
    {"text": "Delete", "left": 120, "top": 20, "width": 50, "height": 12, "conf": 88},
    {"text": "lowconfnoise", "left": 0, "top": 0, "width": 1, "height": 1, "conf": 5},
]


def test_find_targets_matches_query_and_centers():
    els = find_targets(_WORDS, "save")
    assert els and els[0].label == "Save"
    assert els[0].center == (30, 26)  # left+w/2, top+h/2
    # low-confidence noise is dropped
    assert all(e.label != "lowconfnoise" for e in els)


def test_find_targets_requires_overlap_when_query_given():
    # a named target that matches nothing returns nothing actionable. "button" is a
    # stopword-free decorative word here; with stopword filtering the real content
    # token ("nonexistent") drives the (empty) match.
    assert find_targets(_WORDS, "nonexistent thingy") == []


def test_find_targets_stopwords_do_not_outrank_target():
    # "where is the save" -> stopwords dropped -> "save" wins, not a decorative box.
    els = find_targets(_WORDS, "where is the save")
    assert els and els[0].label == "Save"


def test_find_targets_empty_query_returns_top_by_confidence():
    els = find_targets(_WORDS, "")
    labels = [e.label for e in els]
    assert "Save" in labels and "Cancel" in labels
    assert "lowconfnoise" not in labels  # still drops noise


def test_render_elements_is_fenced_and_pii_redacted():
    els = [Element("card 4111 1111 1111 1111", 0, 0, 10, 10, 1.0)]
    block = render_elements(els)
    assert SPOTLIGHT_DIRECTIVE in block          # untrusted-fenced (screen text)
    assert "4111" not in block                   # PII redacted
    assert "[REDACTED_CARD]" in block


def test_render_elements_empty_has_placeholder():
    block = render_elements([])
    assert "no matching" in block.lower()
    assert SPOTLIGHT_DIRECTIVE in block  # still fenced (definite "nothing", not raw)


def test_attach_disabled_registers_nothing():
    reg = CapabilityRegistry()
    attach_computer_use_capability(reg, enabled=False)
    assert "screen.identify" not in reg.names()


def test_screen_identify_is_read_only_private_and_fenced():
    reg = CapabilityRegistry()
    attach_computer_use_capability(
        reg, enabled=True,
        capture_fn=lambda: b"fake-image-bytes",
        ocr_fn=lambda img: _WORDS,
    )
    assert "screen.identify" in reg.names()
    spec = reg.spec("screen.identify")
    assert spec.side_effecting is False and spec.egress == "local"  # read-only + local
    # §9.7: kept OFF the (cloud-routable) ReAct planner until the float is enforced.
    assert spec.planner_tool is False
    res = reg.invoke("screen.identify", "where is the save button", {})
    assert res.ok
    assert SPOTLIGHT_DIRECTIVE in res.text          # screen labels fenced as untrusted
    assert "Save" in res.text
    assert res.data["sensitivity"] == "private"     # §9.7: pins on-device
    assert res.data["origin"] == "screen"
    assert res.data["count"] >= 1


def test_screen_identify_degrades_when_capture_unavailable():
    reg = CapabilityRegistry()
    attach_computer_use_capability(reg, enabled=True, capture_fn=lambda: None, ocr_fn=lambda img: _WORDS)
    res = reg.invoke("screen.identify", "save", {})
    assert res.ok and res.data["count"] == 0
    assert "unavailable" in res.text.lower()
