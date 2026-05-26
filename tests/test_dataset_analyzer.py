"""LiveSpeechAnalyzer tested at scale over real bilingual corpora (MASSIVE + CLINC).

Per-record tests assert robustness invariants (valid IntentKind, determinism,
crash-free observe, punctuation invariance) across two modes; the aggregate test
measures language-detection accuracy on real Romanian vs English utterances --
which also surfaces how ASCII-only normalization handles Romanian diacritics.

Default runs use the committed sample slices; SPEAKER_DATASET_DOWNLOAD=1 expands
to the full datasets.
"""
from __future__ import annotations

import pytest

from always_on_agent.events import Mode
from always_on_agent.models import IntentDecision, IntentKind
from always_on_agent.speech_analyzer import LiveSpeechAnalyzer
from always_on_agent.text import detect_language
from tests.datasets.loaders import analyzer_corpus, load_massive
from tests.router_relations import analyze

pytestmark = [pytest.mark.dev, pytest.mark.audio]

_CORPUS = analyzer_corpus()
_IDS = [f"{r.locale.lower()}-{i}" for i, r in enumerate(_CORPUS)]


@pytest.mark.parametrize("rec", _CORPUS, ids=_IDS)
def test_analyzer_returns_valid_kind_passive(rec):
    decision = analyze(rec.text, mode=Mode.PASSIVE)
    assert isinstance(decision, IntentDecision)
    assert isinstance(decision.kind, IntentKind)
    assert 0.0 <= decision.confidence <= 1.0


@pytest.mark.parametrize("rec", _CORPUS, ids=_IDS)
def test_analyzer_returns_valid_kind_assistant(rec):
    decision = analyze(rec.text, mode=Mode.ASSISTANT)
    assert isinstance(decision.kind, IntentKind)


@pytest.mark.parametrize("rec", _CORPUS, ids=_IDS)
def test_analyzer_is_deterministic(rec):
    assert analyze(rec.text).kind == analyze(rec.text).kind


@pytest.mark.parametrize("rec", _CORPUS, ids=_IDS)
def test_observe_is_crash_free_and_populated(rec):
    obs = LiveSpeechAnalyzer().observe(rec.text, is_final=True)
    assert obs.normalized is not None
    assert 0.0 <= obs.activation_score <= 1.0
    assert obs.language in {"en", "ro", "unknown"}


@pytest.mark.parametrize("rec", _CORPUS, ids=_IDS)
def test_analyzer_punctuation_invariant(rec):
    base = analyze(rec.text).kind
    assert analyze(f"{rec.text}.").kind == base
    assert analyze(f"{rec.text}!").kind == base


def test_language_detection_directional_accuracy():
    """Romanian utterances should be classified 'ro' more often than 'en'."""
    ro = load_massive("ro-RO")
    en = load_massive("en-US")
    if not ro or not en:
        pytest.skip("MASSIVE locales unavailable")
    ro_as_ro = sum(1 for r in ro if detect_language(r.text) == "ro")
    ro_as_en = sum(1 for r in ro if detect_language(r.text) == "en")
    en_as_en = sum(1 for r in en if detect_language(r.text) == "en")
    en_as_ro = sum(1 for r in en if detect_language(r.text) == "ro")
    print(
        f"\nRO -> ro:{ro_as_ro} en:{ro_as_en} (n={len(ro)});  "
        f"EN -> en:{en_as_en} ro:{en_as_ro} (n={len(en)})"
    )
    # detect_language is a marker-based heuristic (most generic text is "unknown"),
    # so the meaningful invariant is that neither language is mislabeled the other.
    assert ro_as_ro >= ro_as_en, "Romanian text mislabeled English"
    assert en_as_en >= en_as_ro, "English text mislabeled Romanian"
    assert ro_as_ro > 0, "Romanian marker detection never fired"


def test_corpus_is_bilingual_and_nontrivial():
    locales = {r.locale for r in _CORPUS}
    assert "ro-RO" in locales and "en-US" in locales
    assert len(_CORPUS) >= 150
