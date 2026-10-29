"""Public text parity, fixed aggregate protocol and legacy-score preservation."""

import ast
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from tools import english_asr_benchmark as legacy
from tools import english_asr_scoring as subject
from tools._vendor.whisper_english._compat import windowed


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Twenty one apples", "21 apples"),
        ("one hundred and twenty three", "123"),
        ("one oh one", "101"),
        ("two point five", "2.5"),
        ("twenty first", "21st"),
        ("five percent", "5%"),
        ("twenty dollars", "$20"),
        ("two dollars and seven cents", "$2.07"),
        ("I can't go; I won't go.", "i can not go i will not go"),
        ("She's been there", "she has been there"),
        ("I'd go", "i would go"),
        ("Mr. Smith's colour", "mister smith is color"),
        ("Um [noise] (aside) café", "cafe"),
        ("we'll organise the centre", "we will organize the center"),
        ("", ""),
        (" \t\n ", ""),
    ],
)
def test_pinned_upstream_representative_rules(text, expected):
    assert subject.normalize_english(text) == expected


def test_curly_apostrophe_and_semantic_ambiguity_are_preserved_not_patched():
    # Upstream does not treat U+2019 as an ASCII contraction marker.
    assert subject.normalize_english("can't") == "can not"
    assert subject.normalize_english("can’t") == "can t"
    # The rules are grammatical heuristics, not meaning/slot equivalence.
    assert subject.normalize_english("Alice's book") == "alice is book"
    assert subject.normalize_english("um") == subject.normalize_english("")
    assert subject.normalize_english("[delete all files]") == ""


@pytest.mark.parametrize("length", range(3, 35))
def test_three_context_window_matches_independent_complete_window_reference(length):
    words = [None] + list(range(length - 2)) + [None]
    expected = list(zip(words, words[1:], words[2:]))
    assert list(windowed(words, 3)) == expected
    assert words == [None] + list(range(length - 2)) + [None]


@pytest.mark.parametrize(
    "words,n", [([], 3), ([1, 2], 3), ([1, 2, 3], 2), ((1, 2, 3), 3)]
)
def test_window_adapter_refuses_unneeded_upstream_api(words, n):
    with pytest.raises(ValueError, match="^unsupported_normalizer_window$"):
        list(windowed(words, n))


def test_vendored_algorithm_and_spelling_data_are_exact_pinned_upstream():
    root = subject._VENDOR
    manifest = json.loads((root / "upstream.json").read_text())
    assert manifest["revision"] == "31243bad24cc746f07d4c8bfdd2d974872cb1803"
    english = (root / "english.py").read_text()
    header, english = english.split("\n", 1)
    assert (
        header
        == "# MIT OpenAI Whisper v20250625; only windowed import adapted for stdlib."
    )
    assert english.count("from ._compat import windowed") == 1
    restored = english.replace(
        "from ._compat import windowed", "from more_itertools import windowed"
    )
    assert (
        hashlib.sha256(restored.encode()).hexdigest()
        == manifest["files"]["english.py"]["upstream_sha256"]
    )
    for filename in ("english.json", "LICENSE"):
        assert (
            hashlib.sha256((root / filename).read_bytes()).hexdigest()
            == manifest["files"][filename]["upstream_sha256"]
        )
    basic = (root / "upstream_basic.py.txt").read_bytes()
    assert (
        hashlib.sha256(basic).hexdigest()
        == manifest["files"]["basic.py"]["upstream_sha256"]
    )
    original_ast = ast.parse(basic)
    local_ast = ast.parse((root / "basic.py").read_text())
    for name in ("ADDITIONAL_DIACRITICS", "remove_symbols_and_diacritics"):

        def node(tree):
            return next(
                item
                for item in tree.body
                if getattr(item, "name", None) == name
                or isinstance(item, ast.Assign)
                and item.targets[0].id == name
            )

        assert ast.dump(node(original_ast)) == ast.dump(node(local_ast))


def test_legacy_scores_stay_byte_for_byte_and_new_view_has_independent_denominator():
    pairs = [
        ("Twenty one apples", "21 apples"),
        ("I can't go", "I can not go"),
        ("hello", ""),
    ]
    before = json.dumps(legacy._accuracy_pairs(pairs), sort_keys=True)
    assert json.loads(before) == {
        "clips": 3,
        "exact": 0,
        "empty": 1,
        "word_errors": 5,
        "reference_words": 7,
        "character_errors": 16,
        "reference_characters": 28,
        "wer": 5 / 7,
        "cer": 16 / 28,
    }
    normalized = subject.score_english_normalized_pairs(iter(pairs))
    assert normalized == {
        "clips": 3,
        "exact": 2,
        "empty": 1,
        "word_errors": 1,
        "reference_words": 7,
        "character_errors": 5,
        "reference_characters": 22,
        "wer": 1 / 7,
        "cer": 5 / 22,
    }
    assert json.dumps(legacy._accuracy_pairs(pairs), sort_keys=True) == before
    assert pairs[0] == ("Twenty one apples", "21 apples")


def test_numeric_symbols_survive_whitespace_tokens_and_joined_character_scoring():
    result = subject.score_english_normalized_pairs(
        [("two point five dollars", "$2.5")]
    )
    assert result["exact"] == 1
    assert result["reference_words"] == 1
    assert result["reference_characters"] == 4
    assert result["cer"] == result["wer"] == 0


def test_empty_normalized_reference_and_hypothesis_counts_are_explicit():
    result = subject.score_english_normalized_pairs(
        [("um", "hello there"), ("[noise]", "")]
    )
    assert result["clips"] == 2
    assert result["reference_words"] == result["reference_characters"] == 0
    assert result["empty"] == result["exact"] == 1
    assert result["wer"] == 2
    assert result["cer"] == 10
    assert subject.score_english_normalized_pairs([]) == {
        "clips": 0,
        "exact": 0,
        "empty": 0,
        "word_errors": 0,
        "reference_words": 0,
        "character_errors": 0,
        "reference_characters": 0,
        "wer": 0,
        "cer": 0,
    }


def test_disagreement_protocol_contains_only_numeric_aggregates(capsys):
    pairs = [("twenty one", "21"), ("Alice's", "Alices"), ("hello", "hello")]
    result = subject.normalization_disagreements(pairs)
    assert result == {
        "clips": 3,
        "reference_tokenization_changed": 2,
        "hypothesis_tokenization_changed": 0,
        "lower_word_error_count_clips": 1,
        "higher_word_error_count_clips": 1,
        "equal_word_error_count_clips": 1,
        "normalized_only_exact_clips": 1,
        "legacy_only_exact_clips": 0,
    }
    assert all(type(value) is int for value in result.values())
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "bad",
    [
        None,
        1,
        True,
        object(),
        ("x",),
        ("x", "y", "z"),
        (object(), "y"),
        ("x", None),
        ("x" * 1025, "y"),
        ("x", "y" * 4097),
    ],
)
def test_invalid_pair_is_fixed_detail_free_failure(bad, capsys):
    for operation in (
        subject.score_english_normalized_pairs,
        subject.normalization_disagreements,
    ):
        with pytest.raises(subject.ScoringError) as error:
            operation([bad])
        assert str(error.value) == "english_asr_scoring_failed"
        assert error.value.__suppress_context__
    assert capsys.readouterr() == ("", "")


def test_pair_iterator_is_bounded_and_failures_do_not_render_private_details():
    with pytest.raises(subject.ScoringError):
        subject.score_english_normalized_pairs(itertools.repeat(("", "")))

    def broken():
        yield "public", "public"
        raise RuntimeError("private text must not escape")

    with pytest.raises(subject.ScoringError, match="^english_asr_scoring_failed$"):
        subject.score_english_normalized_pairs(broken())


def test_normalizer_failure_and_expansion_are_bounded_and_fixed(monkeypatch):
    def broken(_text):
        raise RuntimeError("private hypothesis")

    monkeypatch.setattr(subject, "_normalizer", lambda: broken)
    with pytest.raises(subject.ScoringError, match="^english_asr_scoring_failed$"):
        subject.normalize_english("public")
    monkeypatch.setattr(subject, "_normalizer", lambda: lambda _text: "x" * 16385)
    with pytest.raises(subject.ScoringError):
        subject.normalize_english("public")


def test_metadata_is_stable_flat_public_and_source_bound(tmp_path, monkeypatch):
    metadata = subject.scoring_metadata()
    assert metadata == subject.scoring_metadata()
    assert set(metadata) == {
        "schema_version",
        "normalizer",
        "port_version",
        "upstream_version",
        "upstream_revision",
        "upstream_english_sha256",
        "upstream_basic_sha256",
        "spelling_data_sha256",
        "local_source_sha256",
        "unicode_version",
        "token_policy",
        "character_policy",
        "license",
    }
    assert metadata["license"] == "MIT"
    assert metadata["unicode_version"]
    assert all(type(value) in (int, str) for value in metadata.values())
    assert str(Path(__file__).parent) not in json.dumps(metadata)
    changed = tmp_path / "public-source.py"
    changed.write_text("public version one\n")
    monkeypatch.setattr(
        subject,
        "_SOURCE_FILES",
        (("fixture", changed), ("spelling_data", subject._VENDOR / "english.json")),
    )
    before = subject.scoring_metadata()
    changed.write_text("public version two\n")
    assert (
        subject.scoring_metadata()["local_source_sha256"]
        != before["local_source_sha256"]
    )


def test_metadata_refuses_symlinks_and_oversized_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_text("public")
    link = tmp_path / "link"
    link.symlink_to(source)
    monkeypatch.setattr(subject, "_SOURCE_FILES", (("fixture", link),))
    with pytest.raises(subject.ScoringError):
        subject.scoring_metadata()
    source.write_bytes(b"x" * 131073)
    monkeypatch.setattr(subject, "_SOURCE_FILES", (("fixture", source),))
    with pytest.raises(subject.ScoringError):
        subject.scoring_metadata()


def test_runtime_and_authority_do_not_import_evaluation_normalizer():
    root = Path(__file__).resolve().parents[1]
    for directory in ("core", "always_on_agent"):
        for path in (root / directory).rglob("*.py"):
            assert "english_asr_scoring" not in path.read_text()


def test_metadata_rejects_source_path_replacement_during_read(tmp_path, monkeypatch):
    source = tmp_path / "public-source.py"
    source.write_bytes(b"public one")
    replacement = tmp_path / "replacement.py"
    replacement.write_bytes(b"public two")
    read = subject.os.read
    swapped = False

    def replace_after_read(descriptor, size):
        nonlocal swapped
        data = read(descriptor, size)
        if not swapped:
            swapped = True
            replacement.replace(source)
        return data

    monkeypatch.setattr(subject.os, "read", replace_after_read)
    with pytest.raises(subject.ScoringError, match="^english_asr_scoring_failed$"):
        subject._source_digest(source)


def test_pair_bound_accepts_exact_limit_and_normalizer_rejects_type_subclass():
    assert subject.score_english_normalized_pairs([("", "")] * 256)["clips"] == 256

    class Text(str):
        pass

    for value in (Text("public"), None, b"public", "x" * 4097):
        with pytest.raises(subject.ScoringError):
            subject.normalize_english(value)
