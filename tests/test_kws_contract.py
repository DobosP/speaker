from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import core.kws_contract as kws_contract
from core.kws_contract import (
    KWS_STOP_PHRASES,
    KwsPhraseBinding,
    collapsed_label_surfaces,
    load_kws_phrase_bindings,
    resolve_kws_stop_binding,
    resolve_kws_stop_phrase,
)


_TOKENS = (
    ("S", "T", "AA1", "P"),
    ("S", "T", "AA1", "P", "T", "AO1", "K", "IH0", "NG"),
    ("S", "T", "AA1", "P", "S", "P", "IY1", "K", "IH0", "NG"),
    ("B", "IY1", "K", "W", "AY1", "AH0", "T"),
    ("W", "EY1", "T"),
    ("HH", "OW1", "L", "D", "AA1", "N"),
)


def _keyword_text(*, tokens=_TOKENS) -> str:
    return (
        "\n".join(
            " ".join(
                (
                    *row,
                    ":2.0",
                    f"#{phrase.threshold:g}",
                    f"@{phrase.result_label}",
                )
            )
            for row, phrase in zip(tokens, KWS_STOP_PHRASES, strict=True)
        )
        + "\n"
    )


def test_load_and_resolve_exact_shipped_phrase_rows(tmp_path) -> None:
    keywords = tmp_path / "keywords_barge.txt"
    keywords.write_text(_keyword_text(), encoding="utf-8")

    bindings = load_kws_phrase_bindings(str(keywords))

    assert bindings == tuple(
        KwsPhraseBinding(
            tokens=tokens,
            word_token_counts=phrase.word_token_counts,
            surface=phrase.surface,
            result_label=phrase.result_label,
        )
        for tokens, phrase in zip(_TOKENS, KWS_STOP_PHRASES, strict=True)
    )
    for binding in bindings:
        assert (
            resolve_kws_stop_phrase(
                bindings,
                result_label=binding.result_label,
                tokens=list(binding.tokens),
            )
            == binding.surface
        )


def test_shipped_word_token_counts_are_exact_and_resolve_with_binding() -> None:
    assert tuple(phrase.word_token_counts for phrase in KWS_STOP_PHRASES) == (
        (4,),
        (4, 5),
        (4, 6),
        (2, 5),
        (3,),
        (4, 2),
    )

    bindings = tuple(
        KwsPhraseBinding(
            tokens=phrase.tokens,
            word_token_counts=phrase.word_token_counts,
            surface=phrase.surface,
            result_label=phrase.result_label,
        )
        for phrase in KWS_STOP_PHRASES
    )
    for expected in bindings:
        assert (
            resolve_kws_stop_binding(
                bindings,
                result_label=expected.result_label,
                tokens=list(expected.tokens),
            )
            is expected
        )


@pytest.mark.parametrize(
    "contents",
    [
        "",
        _keyword_text().rsplit("\n", 2)[0] + "\n",
        _keyword_text().replace(":2.0", ":nan", 1),
        _keyword_text().replace("#0.25", "#1.01", 1),
        _keyword_text().replace("@stop", "@wait", 1),
        _keyword_text(tokens=(_TOKENS[0], _TOKENS[0], *_TOKENS[2:])),
        _keyword_text(tokens=(_TOKENS[1], _TOKENS[0], *_TOKENS[2:])),
    ],
    ids=(
        "empty",
        "missing-row",
        "nonfinite-boost",
        "threshold-outside-unit-interval",
        "wrong-collapsed-label",
        "duplicate-token-row",
        "same-label-token-rows-reordered",
    ),
)
def test_loader_rejects_rows_outside_closed_contract(tmp_path, contents) -> None:
    keywords = tmp_path / "keywords_barge.txt"
    keywords.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        load_kws_phrase_bindings(str(keywords))


def test_loader_rejects_non_utf8_and_oversized_files(tmp_path) -> None:
    keywords = tmp_path / "keywords_barge.txt"
    keywords.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        load_kws_phrase_bindings(str(keywords))

    keywords.write_bytes(b"X" * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="bounded contract"):
        load_kws_phrase_bindings(str(keywords))


def test_loader_rejects_file_identity_change_while_read(
    monkeypatch,
    tmp_path,
) -> None:
    keywords = tmp_path / "keywords_barge.txt"
    keywords.write_text(_keyword_text(), encoding="utf-8")
    real_fstat = kws_contract.os.fstat
    calls = 0

    def changing_fstat(fd):
        nonlocal calls
        stat_result = real_fstat(fd)
        calls += 1
        if calls == 1:
            return stat_result
        return SimpleNamespace(
            st_dev=stat_result.st_dev,
            st_ino=stat_result.st_ino,
            st_size=stat_result.st_size,
            st_mtime_ns=stat_result.st_mtime_ns + 1,
        )

    monkeypatch.setattr(kws_contract.os, "fstat", changing_fstat)

    with pytest.raises(ValueError, match="changed while it was read"):
        load_kws_phrase_bindings(str(keywords))
    assert calls == 2


@pytest.mark.parametrize(
    "label,tokens,mutate",
    [
        ("wait", _TOKENS[0], None),
        ("stop", ("UNKNOWN",), None),
        ("stop", _TOKENS[0], "surface"),
        ("stop", _TOKENS[0], "label"),
        ("stop", _TOKENS[0], "duplicate"),
        ("stop", _TOKENS[0], "reorder"),
        ("stop", _TOKENS[0], "word-count-shape"),
        ("stop", _TOKENS[0], "word-count-sum"),
        ("stop", _TOKENS[0], "word-count-bool"),
        ("stop", ("S", None), None),
    ],
    ids=(
        "label-token-disagree",
        "unknown-token-row",
        "binding-surface-mutated",
        "binding-label-mutated",
        "ambiguous-duplicate-tokens",
        "same-label-binding-rows-reordered",
        "binding-word-count-shape-mutated",
        "binding-word-count-sum-mutated",
        "binding-word-count-bool-mutated",
        "non-string-token",
    ),
)
def test_resolver_fails_closed_for_mismatched_or_ambiguous_identity(
    tmp_path,
    label,
    tokens,
    mutate,
) -> None:
    keywords = tmp_path / "keywords_barge.txt"
    keywords.write_text(_keyword_text(), encoding="utf-8")
    bindings = list(load_kws_phrase_bindings(str(keywords)))
    if mutate == "surface":
        bindings[0] = replace(bindings[0], surface="different")
    elif mutate == "label":
        bindings[0] = replace(bindings[0], result_label="wait")
    elif mutate == "duplicate":
        bindings[1] = replace(bindings[1], tokens=bindings[0].tokens)
    elif mutate == "reorder":
        first_tokens, second_tokens = bindings[0].tokens, bindings[1].tokens
        bindings[0] = replace(bindings[0], tokens=second_tokens)
        bindings[1] = replace(bindings[1], tokens=first_tokens)
    elif mutate == "word-count-shape":
        bindings[0] = replace(bindings[0], word_token_counts=(2, 2))
    elif mutate == "word-count-sum":
        bindings[0] = replace(bindings[0], word_token_counts=(3,))
    elif mutate == "word-count-bool":
        bindings[0] = replace(bindings[0], word_token_counts=(True,))

    assert (
        resolve_kws_stop_binding(
            tuple(bindings),
            result_label=label,
            tokens=tokens,
        )
        is None
    )
    assert (
        resolve_kws_stop_phrase(
            tuple(bindings),
            result_label=label,
            tokens=tokens,
        )
        is None
    )


def test_collapsed_labels_expand_to_every_own_echo_alias() -> None:
    assert collapsed_label_surfaces("stop") == (
        "stop",
        "stop talking",
        "stop speaking",
        "be quiet",
    )
    assert collapsed_label_surfaces("wait") == ("wait", "hold on")
    assert collapsed_label_surfaces("other") == ()
    assert collapsed_label_surfaces(True) == ()
