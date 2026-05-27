"""Cross-language behavior contract — the Python half.

Both the Python core (``core/contract.py``) and the Dart mobile shell
(``mobile/lib/contract.dart``) must produce identical results for the shared
fixtures in ``tests/golden/``. This pins the streaming-TTS sentence splitter and
the stop-command recognizer so the two runtimes can't silently drift (see
``docs/target_architecture.md`` §9). The Dart half lives in
``mobile/test/golden_contract_test.dart`` and runs in the ``mobile-tests`` CI
workflow over these same JSON files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.contract import is_stop_command, normalize_command, stream_sentences

GOLDEN = Path(__file__).parent / "golden"


def _load(name: str) -> dict:
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "case", _load("sentence_split.json")["cases"], ids=lambda c: c["name"]
)
def test_sentence_split_contract(case):
    assert stream_sentences(case["tokens"]) == case["expect"]


@pytest.mark.parametrize(
    "case", _load("commands.json")["normalize"], ids=lambda c: c["in"] or "<empty>"
)
def test_command_normalization_contract(case):
    assert normalize_command(case["in"]) == case["out"]


@pytest.mark.parametrize(
    "case", _load("commands.json")["is_stop"], ids=lambda c: c["in"] or "<empty>"
)
def test_stop_command_contract(case):
    assert is_stop_command(case["in"]) == case["expect"]
