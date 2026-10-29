"""Bounded, text-only ASR evaluation; never an inference or trust transform.

Keep legacy benchmark ASCII scores alongside this separately named view. The
MIT Whisper English rules can erase meaningful distinctions (including filler,
bracketed text, spelling and ambiguous contractions); neither score establishes
intent, numeric-slot accuracy, speaker authority or successful conversation.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import unicodedata

from tools._vendor.whisper_english.english import EnglishTextNormalizer

MAX_PAIRS = 256
MAX_REFERENCE_CHARACTERS = 1024
MAX_HYPOTHESIS_CHARACTERS = 4096
MAX_NORMALIZED_CHARACTERS = 16384
_LEGACY_WORD = re.compile(r"[a-z0-9']+")
_VENDOR = Path(__file__).parent / "_vendor" / "whisper_english"
_SOURCE_FILES = (
    ("scorer", Path(__file__)),
    ("normalizer", _VENDOR / "english.py"),
    ("unicode_helper", _VENDOR / "basic.py"),
    ("window_helper", _VENDOR / "_compat.py"),
    ("spelling_data", _VENDOR / "english.json"),
    ("package", _VENDOR / "__init__.py"),
    ("upstream_manifest", _VENDOR / "upstream.json"),
    ("upstream_unicode_source", _VENDOR / "upstream_basic.py.txt"),
    ("license", _VENDOR / "LICENSE"),
)


class ScoringError(ValueError):
    """Fixed detail-free failure; input text must never appear in errors."""

    def __init__(self):
        super().__init__("english_asr_scoring_failed")


def _distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for index, ref in enumerate(reference, 1):
        current = [index]
        for other, hyp in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[other] + 1,
                    current[-1] + 1,
                    previous[other - 1] + (ref != hyp),
                )
            )
        previous = current
    return previous[-1]


@lru_cache(maxsize=1)
def _normalizer():
    return EnglishTextNormalizer()


def normalize_english(text: str) -> str:
    """Apply only pinned upstream English rules, then trim outer whitespace."""
    if type(text) is not str or len(text) > MAX_HYPOTHESIS_CHARACTERS:
        raise ScoringError()
    try:
        result = _normalizer()(text).strip()
    except Exception:
        raise ScoringError() from None
    if type(result) is not str or len(result) > MAX_NORMALIZED_CHARACTERS:
        raise ScoringError()
    return result


def _pairs(pairs):
    try:
        iterator = iter(pairs)
        for index, pair in enumerate(iterator):
            if (
                index >= MAX_PAIRS
                or type(pair) not in (list, tuple)
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not str
                or len(pair[0]) > MAX_REFERENCE_CHARACTERS
                or len(pair[1]) > MAX_HYPOTHESIS_CHARACTERS
            ):
                raise ScoringError()
            yield pair[0], pair[1]
    except Exception:
        raise ScoringError() from None


def score_english_normalized_pairs(pairs) -> dict:
    """Return aggregates only; tokens are normalized whitespace-separated words.

    CER joins these tokens without spaces, preserving the normalizer's numeric
    symbols. Empty references use denominator max(1, total_reference_units),
    as in the legacy view; insertion errors can make WER/CER exceed one.
    """
    totals = Counter(
        clips=0,
        exact=0,
        empty=0,
        word_errors=0,
        reference_words=0,
        character_errors=0,
        reference_characters=0,
    )
    for reference, hypothesis in _pairs(pairs):
        ref = normalize_english(reference).split()
        hyp = normalize_english(hypothesis).split()
        ref_chars, hyp_chars = "".join(ref), "".join(hyp)
        totals.update(
            clips=1,
            exact=int(ref == hyp),
            empty=int(not hyp),
            word_errors=_distance(ref, hyp),
            reference_words=len(ref),
            character_errors=_distance(ref_chars, hyp_chars),
            reference_characters=len(ref_chars),
        )
    return {
        **dict(totals),
        "wer": totals["word_errors"] / max(1, totals["reference_words"]),
        "cer": totals["character_errors"] / max(1, totals["reference_characters"]),
    }


def normalization_disagreements(pairs) -> dict:
    """Compare per-clip error counts, never return normalized/private text.

    Lower counts are formatting-sensitive diagnostic differences, not a model
    improvement. WER denominators may change and both views must be retained.
    """
    counts = Counter(
        clips=0,
        reference_tokenization_changed=0,
        hypothesis_tokenization_changed=0,
        lower_word_error_count_clips=0,
        higher_word_error_count_clips=0,
        equal_word_error_count_clips=0,
        normalized_only_exact_clips=0,
        legacy_only_exact_clips=0,
    )
    for reference, hypothesis in _pairs(pairs):
        old_ref = _LEGACY_WORD.findall(reference.lower())
        old_hyp = _LEGACY_WORD.findall(hypothesis.lower())
        ref = normalize_english(reference).split()
        hyp = normalize_english(hypothesis).split()
        old_errors, errors = _distance(old_ref, old_hyp), _distance(ref, hyp)
        counts.update(
            clips=1,
            reference_tokenization_changed=int(old_ref != ref),
            hypothesis_tokenization_changed=int(old_hyp != hyp),
            lower_word_error_count_clips=int(errors < old_errors),
            higher_word_error_count_clips=int(errors > old_errors),
            equal_word_error_count_clips=int(errors == old_errors),
            normalized_only_exact_clips=int(ref == hyp and old_ref != old_hyp),
            legacy_only_exact_clips=int(old_ref == old_hyp and ref != hyp),
        )
    return dict(counts)


def _source_digest(path):
    """Hash only bounded shipped source; refuse named/descriptor replacement."""
    descriptor = None
    try:
        named_before = path.lstat()
        if (
            not stat.S_ISREG(named_before.st_mode)
            or not 0 < named_before.st_size <= 131072
        ):
            raise ScoringError()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
        )
        before = os.fstat(descriptor)

        def identity(item):
            return (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        if identity(named_before) != identity(before):
            raise ScoringError()
        chunks = []
        total = 0
        while total <= 131072:
            chunk = os.read(descriptor, min(65536, 131073 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            identity(before) != identity(after)
            or identity(after) != identity(path.lstat())
            or len(data) != after.st_size
        ):
            raise ScoringError()
        return hashlib.sha256(data).hexdigest(), len(data)
    except Exception:
        raise ScoringError() from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except Exception:
                raise ScoringError() from None


def scoring_metadata() -> dict:
    """Return only fixed public provenance; compare snapshots before/after use.

    This is source/data identity, not executed-bytecode or import attestation.
    It adds no audio path, corpus label, reference or model-specific rule.
    """
    files = []
    for role, path in _SOURCE_FILES:
        digest, size = _source_digest(path)
        files.append({"role": role, "sha256": digest, "bytes": size})
    source_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 1,
        "normalizer": "openai-whisper-english-stdlib-port",
        "port_version": "1",
        "upstream_version": "v20250625",
        "upstream_revision": "31243bad24cc746f07d4c8bfdd2d974872cb1803",
        "upstream_english_sha256": "2898e4672aeefc9f44cb4de38c115cdf286f5b35ae321433467a7674ce8841d0",
        "upstream_basic_sha256": "4742eaa040e0657fa1247a1361e0d856c62317a43326ea59a40c2e9edd8d2c38",
        "spelling_data_sha256": next(
            item["sha256"] for item in files if item["role"] == "spelling_data"
        ),
        "local_source_sha256": source_digest,
        "unicode_version": unicodedata.unidata_version,
        "token_policy": "normalized_whitespace",
        "character_policy": "joined_normalized_tokens",
        "license": "MIT",
    }
