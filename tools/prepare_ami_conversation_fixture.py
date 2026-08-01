"""Materialize the private AMI close/far conversation fixture source.

The command is deliberately offline.  It accepts the official AMI manual
annotation archive and synchronized ES2004a Mix-Headset/Array1-01 WAV files,
recomputes their content identities, derives twelve unambiguous reference
windows, and publishes each interval once per channel.  Literal references,
source paths, and source speaker labels stay out of the sibling receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from core.wer import normalize
from tools import prepare_ami_capture_replay as _ami_source
from tools import public_conversation_fixture as fixture
from tools.public_voice_eval_matrix import (
    ChecksumAlgorithm,
    IntegrityKind,
    dataset_by_id,
    selection_policy_sha256,
)
from tools.streaming_stt.bounded_io import BoundedReadError, read_regular_bounded
from tools.streaming_stt.corpus import CorpusProvenance, LoadedCorpus
from tools.streaming_stt.corpus_writer import CorpusWriteCase, publish_private_corpus
from tools.streaming_stt.protocol import MAX_CORPUS_BYTES, MAX_PCM_BYTES


DATASET_ID = "ami_es2004a_close_far_v1"
SOURCE_ID = "ami-es2004a-close-far"
MEETING_ID = "ES2004a"
SAMPLE_RATE_HZ = 16_000

ANNOTATIONS_BYTES = 22_887_865
ANNOTATIONS_SHA256 = (
    "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
)
FAR_BYTES = 33_579_394
FAR_SHA256 = "6936edac5d0904fc5c4ab175546c5cc5366601fdc1b1e5183a6ea2c10f05d150"
REQUIRED_TERMS = frozenset({"CC-BY-4.0"})

DECODER_CONTRACT = "ami-riff-pcm16-mono16k-window-to-f32le-v1"
PREPARER_CONTRACT = "public-conversation-ami-v1"
SELECTION_ALGORITHM = "paired_window_projection_v1"
SOURCE_RECIPE_ELIGIBILITY = (
    "manual_word_timings",
    "synchronized_channels",
    "no_actual_overlap",
    "identical_pair_interval",
    "identical_pair_reference",
)
SOURCE_RECIPE_STRATA = (
    ("isolated_close", 8),
    ("isolated_far", 8),
    ("turn_transition_close", 4),
    ("turn_transition_far", 4),
)
REQUIRED_PREPARER_FILES = frozenset(
    {
        "tools/prepare_ami_capture_replay.py",
        "tools/prepare_ami_conversation_fixture.py",
    }
)

ISOLATED_WINDOWS = 8
TRANSITION_WINDOWS = 4
TOTAL_WINDOWS = ISOLATED_WINDOWS + TRANSITION_WINDOWS
TOTAL_CASES = TOTAL_WINDOWS * 2

_ANNOTATION_MAX_BYTES = 32 * 1024 * 1024
_WAV_MAX_BYTES = 64 * 1024 * 1024
_RECEIPT_MAX_BYTES = 256 * 1024
_REPO_FILE_MAX_BYTES = 4 * 1024 * 1024
_CONTEXT_SAMPLES = SAMPLE_RATE_HZ // 10
_MAX_INTRA_TURN_GAP = 3 * SAMPLE_RATE_HZ // 4
_MAX_TRANSITION_GAP = 3 * SAMPLE_RATE_HZ // 2
_MAX_TURN_SPAN = 8 * SAMPLE_RATE_HZ
_MAX_TRANSITION_SPAN = 12 * SAMPLE_RATE_HZ
_MIN_ISOLATED_SPEECH = SAMPLE_RATE_HZ // 2
_MAX_REFERENCE_CHARS = 4096
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_TEST_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")

_PRIVATE_RECEIPT_PRIVACY = {
    "contains_audio": False,
    "contains_transcripts": False,
    "contains_local_paths": False,
    "contains_raw_identifiers": False,
}
_RECEIPT_EVIDENCE_SCOPE = {
    "development_only": True,
    "hard_wer_requires_unambiguous_nonoverlap": True,
    "ambiguous_overlap": "diagnostic_only",
    "capture": False,
    "vad": False,
    "aec": False,
    "agent_tools": False,
    "live_latency": False,
    "default_promotion": False,
    "pcm_preparation_complete": True,
    "stt_model_run": False,
}
_SAFE_ERROR: Mapping[str, object] = {
    "ok": False,
    "error": "ami_conversation_fixture_prerequisites_unavailable",
}


class AmiConversationPreparationError(RuntimeError):
    """A detail-free source, selection, decode, or publication failure."""


@dataclass(frozen=True, slots=True)
class TestSourceIdentityInjection:
    """Unmistakably test-only replacement for the two public content pins."""

    annotations_sha256: str
    annotations_bytes: int
    far_sha256: str
    far_bytes: int
    fixture_id: str


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    data: bytes = field(repr=False)
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _Turn:
    words: tuple[object, ...] = field(repr=False)

    @property
    def speaker(self) -> str:
        return str(self.words[0].speaker)

    @property
    def start_sample(self) -> int:
        return int(self.words[0].start_sample)

    @property
    def end_sample(self) -> int:
        return max(int(word.end_sample) for word in self.words)


@dataclass(frozen=True, slots=True)
class _Window:
    kind: str
    start_sample: int
    end_sample: int
    words: tuple[object, ...] = field(repr=False)
    reference: str = field(repr=False)
    identity_sha256: str
    window_sha256: str
    speaker_sha256: str | None


@dataclass(frozen=True, slots=True)
class PreparedAmiConversationCorpus:
    corpus: LoadedCorpus
    receipt_sha256: str
    metadata_sha256: str
    selected_windows_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
    except (MemoryError, TypeError, UnicodeError, ValueError, OverflowError):
        raise AmiConversationPreparationError() from None


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _salted_digest(seed: str, kind: str, value: str) -> str:
    try:
        payload = (
            seed.encode("ascii")
            + b"\0"
            + kind.encode("ascii")
            + b"\0"
            + value.encode("utf-8")
        )
    except (UnicodeError, ValueError):
        raise AmiConversationPreparationError() from None
    return hashlib.sha256(payload).hexdigest()


def _identity(value: object, *, maximum: int) -> tuple[str, int]:
    try:
        digest, size = value
    except (TypeError, ValueError):
        raise AmiConversationPreparationError() from None
    if (
        not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or type(size) is not int
        or not 0 < size <= maximum
    ):
        raise AmiConversationPreparationError()
    return digest, size


def _expected_source_identities(
    injection: TestSourceIdentityInjection | None,
) -> tuple[tuple[str, int], tuple[str, int], bool]:
    if injection is None:
        return (
            (ANNOTATIONS_SHA256, ANNOTATIONS_BYTES),
            (FAR_SHA256, FAR_BYTES),
            True,
        )
    if (
        type(injection) is not TestSourceIdentityInjection
        or _SAFE_TEST_ID_RE.fullmatch(injection.fixture_id) is None
    ):
        raise AmiConversationPreparationError()
    annotations = _identity(
        (injection.annotations_sha256, injection.annotations_bytes),
        maximum=_ANNOTATION_MAX_BYTES,
    )
    far = _identity(
        (injection.far_sha256, injection.far_bytes),
        maximum=_WAV_MAX_BYTES,
    )
    return annotations, far, False


def _verified_source(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_bytes: int,
    maximum_bytes: int,
) -> _SourceSnapshot:
    expected_sha256, expected_bytes = _identity(
        (expected_sha256, expected_bytes),
        maximum=maximum_bytes,
    )
    try:
        snapshot = read_regular_bounded(
            Path(path).expanduser(),
            maximum_bytes=maximum_bytes,
            expected_bytes=expected_bytes,
        )
    except BoundedReadError:
        raise AmiConversationPreparationError() from None
    digest = hashlib.sha256(snapshot.data).hexdigest()
    if digest != expected_sha256:
        raise AmiConversationPreparationError()
    return _SourceSnapshot(snapshot.data, digest, len(snapshot.data))


def _catalog_contract() -> None:
    try:
        dataset = dataset_by_id(DATASET_ID)
        annotations, far, close = dataset.artifacts
        if (
            dataset.license.acceptance_ids != ("CC-BY-4.0",)
            or annotations.artifact_id != "annotations"
            or annotations.size_bytes != ANNOTATIONS_BYTES
            or annotations.integrity is not IntegrityKind.PINNED
            or annotations.checksum is None
            or annotations.checksum.algorithm is not ChecksumAlgorithm.SHA256
            or annotations.checksum.digest != ANNOTATIONS_SHA256
            or far.artifact_id != "es2004a-array1-01"
            or far.size_bytes != FAR_BYTES
            or far.integrity is not IntegrityKind.PINNED
            or far.checksum is None
            or far.checksum.algorithm is not ChecksumAlgorithm.SHA256
            or far.checksum.digest != FAR_SHA256
            or close.artifact_id != "es2004a-mix-headset"
            or close.size_bytes is not None
            or close.integrity is not IntegrityKind.LOCAL_SHA256_RECEIPT
            or close.checksum is not None
        ):
            raise AmiConversationPreparationError()
    except AmiConversationPreparationError:
        raise
    except Exception:
        raise AmiConversationPreparationError() from None


def _expected_source_recipe(*, seed: str) -> dict[str, object]:
    dataset = dataset_by_id(DATASET_ID)
    return {
        "algorithm": SELECTION_ALGORITHM,
        "matrix_policy_sha256": selection_policy_sha256(dataset.selection),
        "split": "es2004a",
        "seed": seed,
        "eligibility": list(SOURCE_RECIPE_ELIGIBILITY),
        "strata": [
            {"id": stratum, "cases": cases}
            for stratum, cases in SOURCE_RECIPE_STRATA
        ],
        "speaker_requirement": "not_required_pair_bound",
        "scoreability": "hard_wer_nonoverlap_only",
    }


def _validate_fixture_source_contract(
    source: fixture.FixtureSource,
    *,
    seed: str,
) -> None:
    if (
        source.dataset_id != DATASET_ID
        or source.expected_cases != TOTAL_CASES
        or source.decoder_contract != DECODER_CONTRACT
        or dict(source.selection_recipe) != _expected_source_recipe(seed=seed)
    ):
        raise AmiConversationPreparationError()


def _word_key(word: object) -> tuple[int, int, str, int, str]:
    return (
        int(word.start_sample),
        int(word.end_sample),
        str(word.speaker),
        int(word.ordinal),
        str(word.text),
    )


def _ordered(words: Sequence[object]) -> tuple[object, ...]:
    return tuple(sorted(words, key=_word_key))


def _turns(words: Sequence[object]) -> tuple[_Turn, ...]:
    ordered = _ordered(words)
    result: list[_Turn] = []
    active: list[object] = []
    for word in ordered:
        if not active:
            active = [word]
            continue
        previous = active[-1]
        gap = int(word.start_sample) - int(previous.end_sample)
        span = int(word.end_sample) - int(active[0].start_sample)
        if (
            str(word.speaker) == str(previous.speaker)
            and 0 <= gap <= _MAX_INTRA_TURN_GAP
            and span <= _MAX_TURN_SPAN
        ):
            active.append(word)
        else:
            result.append(_Turn(tuple(active)))
            active = [word]
    if active:
        result.append(_Turn(tuple(active)))
    return tuple(result)


def _reference(words: Sequence[object]) -> str:
    value = " ".join(str(word.text) for word in _ordered(words))
    if (
        not value.strip()
        or len(value) > _MAX_REFERENCE_CHARS
        or not normalize(value)
        or "\x00" in value
    ):
        raise AmiConversationPreparationError()
    return value


def _different_speaker_overlap(words: Sequence[object]) -> int:
    try:
        return int(_ami_source._different_speaker_overlap(words))
    except Exception:
        raise AmiConversationPreparationError() from None


def _activity_intersects(
    activity: object,
    *,
    start_sample: int,
    end_sample: int,
) -> bool:
    try:
        start = int(activity.start_sample)
        end = int(activity.end_sample)
    except Exception:
        raise AmiConversationPreparationError() from None
    if start < 0 or end <= start:
        raise AmiConversationPreparationError()
    return start < end_sample and end > start_sample


def _candidate_window(
    *,
    kind: str,
    selected_words: Sequence[object],
    all_words: Sequence[object],
    activities: Sequence[object],
    audio_samples: int,
    seed: str,
) -> _Window | None:
    chosen = _ordered(selected_words)
    if not chosen or kind not in {"isolated", "turn_transition"}:
        return None
    speech_start = min(int(word.start_sample) for word in chosen)
    speech_end = max(int(word.end_sample) for word in chosen)
    if speech_start < 0 or speech_end <= speech_start or speech_end > audio_samples:
        return None
    chosen_keys = {_word_key(word) for word in chosen}
    if len(chosen_keys) != len(chosen):
        return None
    external = [word for word in all_words if _word_key(word) not in chosen_keys]
    if any(
        int(word.start_sample) < speech_end and int(word.end_sample) > speech_start
        for word in external
    ):
        return None
    previous_end = max(
        (int(word.end_sample) for word in external if int(word.end_sample) <= speech_start),
        default=0,
    )
    next_start = min(
        (int(word.start_sample) for word in external if int(word.start_sample) >= speech_end),
        default=audio_samples,
    )
    start = max(0, previous_end, speech_start - _CONTEXT_SAMPLES)
    end = min(audio_samples, next_start, speech_end + _CONTEXT_SAMPLES)
    if start >= speech_start or end <= speech_end or end <= start:
        return None
    if any(
        _activity_intersects(
            activity,
            start_sample=start,
            end_sample=end,
        )
        for activity in activities
    ):
        return None
    enclosed = tuple(
        word
        for word in all_words
        if int(word.start_sample) < end and int(word.end_sample) > start
    )
    if {_word_key(word) for word in enclosed} != chosen_keys:
        return None
    speakers = {str(word.speaker) for word in chosen}
    if _different_speaker_overlap(chosen) != 0:
        return None
    if kind == "isolated":
        if (
            len(speakers) != 1
            or speech_end - speech_start < _MIN_ISOLATED_SPEECH
            or end - start > _MAX_TURN_SPAN + 2 * _CONTEXT_SAMPLES
        ):
            return None
        raw_speaker = next(iter(speakers))
        speaker_sha256 = _salted_digest(seed, "speaker", raw_speaker)
    else:
        if len(speakers) < 2 or end - start > _MAX_TRANSITION_SPAN:
            return None
        speaker_sha256 = None
    reference = _reference(chosen)
    identity_payload = "\0".join(
        [
            MEETING_ID,
            kind,
            str(start),
            str(end),
            *(
                f"{word.speaker}:{word.ordinal}:{word.start_sample}:{word.end_sample}"
                for word in chosen
            ),
        ]
    )
    identity_sha256 = _salted_digest(seed, "window", identity_payload)
    window_sha256 = _canonical_sha256(
        {
            "identity_sha256": identity_sha256,
            "kind": kind,
            "start_sample": start,
            "end_sample": end,
            "reference_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
        }
    )
    return _Window(
        kind=kind,
        start_sample=start,
        end_sample=end,
        words=chosen,
        reference=reference,
        identity_sha256=identity_sha256,
        window_sha256=window_sha256,
        speaker_sha256=speaker_sha256,
    )


def _derive_candidates(
    words: Sequence[object],
    activities: Sequence[object],
    *,
    audio_samples: int,
    seed: str,
) -> tuple[_Window, ...]:
    ordered = _ordered(words)
    turns = _turns(ordered)
    candidates: list[_Window] = []
    for turn in turns:
        candidate = _candidate_window(
            kind="isolated",
            selected_words=turn.words,
            all_words=ordered,
            activities=activities,
            audio_samples=audio_samples,
            seed=seed,
        )
        if candidate is not None:
            candidates.append(candidate)
    for left, right in zip(turns, turns[1:]):
        gap = right.start_sample - left.end_sample
        if (
            left.speaker == right.speaker
            or gap < 0
            or gap > _MAX_TRANSITION_GAP
            or right.end_sample - left.start_sample > _MAX_TRANSITION_SPAN
        ):
            continue
        candidate = _candidate_window(
            kind="turn_transition",
            selected_words=(*left.words, *right.words),
            all_words=ordered,
            activities=activities,
            audio_samples=audio_samples,
            seed=seed,
        )
        if candidate is not None:
            candidates.append(candidate)
    unique: dict[tuple[str, int, int, str], _Window] = {}
    for candidate in candidates:
        key = (
            candidate.kind,
            candidate.start_sample,
            candidate.end_sample,
            candidate.identity_sha256,
        )
        unique[key] = candidate
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.kind,
                item.identity_sha256,
                item.start_sample,
                item.end_sample,
            ),
        )
    )


def _overlaps(left: _Window, right: _Window) -> bool:
    return left.start_sample < right.end_sample and right.start_sample < left.end_sample


def _select_windows(
    candidates: Sequence[_Window],
    *,
    seed: str,
) -> tuple[_Window, ...]:
    # Reserve the scarcer transition evidence before choosing isolated turns.
    # The returned order still follows the committed slots (isolated first).
    by_kind: dict[str, list[_Window]] = {}
    selected: list[_Window] = []
    for kind, quota in (
        ("turn_transition", TRANSITION_WINDOWS),
        ("isolated", ISOLATED_WINDOWS),
    ):
        ranked = sorted(
            (item for item in candidates if item.kind == kind),
            key=lambda item: (
                _salted_digest(seed, "rank", item.identity_sha256),
                item.identity_sha256,
            ),
        )
        chosen: list[_Window] = []
        for candidate in ranked:
            if any(_overlaps(candidate, previous) for previous in (*selected, *chosen)):
                continue
            chosen.append(candidate)
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise AmiConversationPreparationError()
        by_kind[kind] = chosen
        selected.extend(chosen)
    if len(selected) != TOTAL_WINDOWS:
        raise AmiConversationPreparationError()
    return tuple((*by_kind["isolated"], *by_kind["turn_transition"]))


def _metadata_sha256(annotations: Mapping[str, bytes], *, seed: str) -> str:
    rows = [
        {
            "speaker_sha256": _salted_digest(seed, "speaker", speaker),
            "word_xml_sha256": hashlib.sha256(annotations[speaker]).hexdigest(),
            "word_xml_bytes": len(annotations[speaker]),
        }
        for speaker in sorted(annotations)
    ]
    return _canonical_sha256(rows)


def _selection_row(window: _Window, *, channel: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "identity_sha256": window.identity_sha256,
        "window_sha256": window.window_sha256,
        "window_kind": window.kind,
        "start_sample": window.start_sample,
        "end_sample": window.end_sample,
        "reference_sha256": hashlib.sha256(window.reference.encode("utf-8")).hexdigest(),
    }
    if channel is not None:
        row["channel"] = channel
    return row


def _repo_file_sha256(relative: str) -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        candidate = (root / relative).resolve(strict=True)
        if root not in candidate.parents:
            raise AmiConversationPreparationError()
        snapshot = read_regular_bounded(candidate, maximum_bytes=_REPO_FILE_MAX_BYTES)
        return hashlib.sha256(snapshot.data).hexdigest()
    except Exception:
        raise AmiConversationPreparationError() from None


def _validated_output(path: Path | str, *, production_evidence: bool) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise AmiConversationPreparationError()
    candidate = Path(os.path.abspath(supplied))
    repo_root = Path(__file__).resolve().parents[1]
    try:
        parent = candidate.parent.resolve(strict=True)
        inside_git = production_evidence and any(
            os.path.lexists(ancestor / ".git") for ancestor in (parent, *parent.parents)
        )
    except (OSError, RuntimeError, ValueError):
        raise AmiConversationPreparationError() from None
    if (
        candidate.name in {"", ".", ".."}
        or parent != candidate.parent
        or candidate == repo_root
        or repo_root in candidate.parents
        or inside_git
        or os.path.lexists(candidate)
    ):
        raise AmiConversationPreparationError()
    return candidate


def _preparer_binding(source: fixture.FixtureSource) -> dict[str, object]:
    if (
        source.preparer_contract != PREPARER_CONTRACT
        or not REQUIRED_PREPARER_FILES <= set(source.preparer_files)
    ):
        raise AmiConversationPreparationError()
    return {
        "contract": PREPARER_CONTRACT,
        "files": {
            relative: _repo_file_sha256(relative) for relative in source.preparer_files
        },
    }


def _decode_window(source: np.ndarray, window: _Window) -> bytes:
    if (
        source.dtype != np.dtype("<i2")
        or window.start_sample < 0
        or window.end_sample > int(source.size)
        or window.end_sample <= window.start_sample
    ):
        raise AmiConversationPreparationError()
    values = source[window.start_sample : window.end_sample].astype(np.float32)
    values *= np.float32(1.0 / 32768.0)
    if (
        values.size != window.end_sample - window.start_sample
        or values.nbytes > MAX_PCM_BYTES
        or not bool(np.all(np.isfinite(values)))
        or bool(np.any(np.abs(values) > 1.0))
    ):
        raise AmiConversationPreparationError()
    return values.astype("<f4", copy=False).tobytes(order="C")


def _artifact_receipts(
    annotations: _SourceSnapshot,
    far: _SourceSnapshot,
    close: _SourceSnapshot,
) -> list[dict[str, object]]:
    return [
        {
            "artifact_id": "annotations",
            "upstream_algorithm": "sha256",
            "upstream_digest": annotations.sha256,
            "size_bytes": annotations.size_bytes,
            "local_sha256": annotations.sha256,
        },
        {
            "artifact_id": "es2004a-array1-01",
            "upstream_algorithm": "sha256",
            "upstream_digest": far.sha256,
            "size_bytes": far.size_bytes,
            "local_sha256": far.sha256,
        },
        {
            "artifact_id": "es2004a-mix-headset",
            "upstream_algorithm": "sha256",
            "upstream_digest": close.sha256,
            "size_bytes": close.size_bytes,
            "local_sha256": close.sha256,
        },
    ]


def prepare_ami_conversation_fixture(
    *,
    annotations_zip: Path | str,
    far_wav: Path | str,
    close_wav: Path | str,
    close_sha256: str,
    close_bytes: int,
    output_dir: Path | str,
    accepted_terms: frozenset[str],
    lock_path: Path | str = fixture.DEFAULT_LOCK,
    test_source_identity_injection: TestSourceIdentityInjection | None = None,
) -> PreparedAmiConversationCorpus:
    """Verify local AMI inputs and publish one immutable schema-v2 corpus."""

    if not REQUIRED_TERMS <= accepted_terms:
        raise AmiConversationPreparationError()
    annotations_identity, far_identity, production_evidence = _expected_source_identities(
        test_source_identity_injection
    )
    if production_evidence:
        _catalog_contract()
    close_identity = _identity((close_sha256, close_bytes), maximum=_WAV_MAX_BYTES)
    lock = fixture.load_fixture_lock(lock_path)
    source = lock.source_by_id(SOURCE_ID)
    _validate_fixture_source_contract(source, seed=lock.selection_seed)
    preparer = _preparer_binding(source)
    destination = _validated_output(
        output_dir,
        production_evidence=production_evidence,
    )

    annotations = _verified_source(
        annotations_zip,
        expected_sha256=annotations_identity[0],
        expected_bytes=annotations_identity[1],
        maximum_bytes=_ANNOTATION_MAX_BYTES,
    )
    far = _verified_source(
        far_wav,
        expected_sha256=far_identity[0],
        expected_bytes=far_identity[1],
        maximum_bytes=_WAV_MAX_BYTES,
    )
    close = _verified_source(
        close_wav,
        expected_sha256=close_identity[0],
        expected_bytes=close_identity[1],
        maximum_bytes=_WAV_MAX_BYTES,
    )
    if close.sha256 == far.sha256:
        raise AmiConversationPreparationError()
    try:
        far_pcm = _ami_source._riff_pcm16_mono(far.data)
        close_pcm = _ami_source._riff_pcm16_mono(close.data)
        if int(far_pcm.size) != int(close_pcm.size):
            raise AmiConversationPreparationError()
        annotation_members = _ami_source._annotation_members(annotations.data)
        words, activities = _ami_source._ordered_annotations(
            annotation_members,
            audio_samples=int(far_pcm.size),
        )
    except AmiConversationPreparationError:
        raise
    except Exception:
        raise AmiConversationPreparationError() from None

    seed = lock.selection_seed
    candidates = _derive_candidates(
        words,
        activities,
        audio_samples=int(far_pcm.size),
        seed=seed,
    )
    selected = _select_windows(candidates, seed=seed)
    metadata_sha256 = _metadata_sha256(annotation_members, seed=seed)

    slots = lock.slots_for(SOURCE_ID)
    write_cases: list[CorpusWriteCase] = []
    receipt_cases: list[dict[str, object]] = []
    total_samples = 0
    channels = {"close": close_pcm, "far": far_pcm}
    projected_windows = tuple(
        item for window in selected for item in (window, window)
    )
    for slot, window in zip(slots, projected_windows, strict=True):
        if (
            slot.ordinal // 2 >= len(selected)
            or slot.stratum != window.kind
            or slot.channel not in channels
            or slot.pair_id != f"ami-pair-{slot.ordinal // 2:02d}"
        ):
            raise AmiConversationPreparationError()
        channel = str(slot.channel)
        pcm_source = channels[channel]
        raw_pcm = _decode_window(pcm_source, window)
        sample_count = len(raw_pcm) // 4
        total_samples += sample_count
        if total_samples * 4 > MAX_CORPUS_BYTES:
            raise AmiConversationPreparationError()
        case_id = f"ami-es2004a-{slot.ordinal // 2:02d}-{channel}"
        tags = tuple(dict.fromkeys((*source.required_tags, window.kind, channel)))
        write_cases.append(
            CorpusWriteCase(
                case_id=case_id,
                audio_bytes=raw_pcm,
                reference=window.reference,
                tags=tags,
            )
        )
        # The hardened RIFF parser accepts arbitrary chunk order/padding, so the
        # raw per-window evidence is derived from decoded int16 samples instead
        # of assuming a 44-byte header.
        source_window = pcm_source[
            window.start_sample : window.end_sample
        ].astype("<i2", copy=False).tobytes(order="C")
        case_identity = _salted_digest(
            seed,
            "projected-case",
            f"{window.identity_sha256}\0{channel}",
        )
        receipt_cases.append(
            {
                "slot_id": slot.slot_id,
                "case_id": case_id,
                "identity_sha256": case_identity,
                "speaker_sha256": window.speaker_sha256,
                "reference_sha256": hashlib.sha256(
                    window.reference.encode("utf-8")
                ).hexdigest(),
                "source_audio_sha256": hashlib.sha256(source_window).hexdigest(),
                "source_audio_bytes": len(source_window),
                "pcm_sha256": hashlib.sha256(raw_pcm).hexdigest(),
                "pcm_samples": sample_count,
                "tags": list(tags),
                "attributes": {
                    "window_index": slot.ordinal // 2,
                    "window_sha256": window.window_sha256,
                    "window_kind": window.kind,
                    "channel": channel,
                    "start_sample": window.start_sample,
                    "end_sample": window.end_sample,
                    "actual_overlap": False,
                    "scoreability": "hard_wer",
                },
            }
        )

    eligible_rows = [
        _selection_row(window, channel=channel)
        for window in candidates
        for channel in ("close", "far")
    ]
    selected_rows = [
        _selection_row(window, channel=channel)
        for window in selected
        for channel in ("close", "far")
    ]
    preparer_files = preparer["files"]
    if not isinstance(preparer_files, dict):
        raise AmiConversationPreparationError()
    receipt = {
        "schema_version": 1,
        "kind": fixture.RECEIPT_KIND,
        "fixture_id": fixture.FIXTURE_ID,
        "lock_recipe_sha256": lock.recipe_sha256,
        "source_id": SOURCE_ID,
        "source_recipe_sha256": source.selection_recipe_sha256,
        "dataset_contract_sha256": source.dataset_contract_sha256,
        "accepted_terms": sorted(REQUIRED_TERMS),
        "artifacts": _artifact_receipts(annotations, far, close),
        "metadata_sha256": metadata_sha256,
        "selection": {
            "policy_sha256": selection_policy_sha256(dataset_by_id(DATASET_ID).selection),
            "eligible_rows": len(eligible_rows),
            "eligible_set_sha256": _canonical_sha256(eligible_rows),
            "selected_rows_sha256": _canonical_sha256(selected_rows),
            "selected_cases": TOTAL_CASES,
            "selected_case_set_sha256": fixture._canonical_sha256(receipt_cases),
            "parent_receipt_sha256": None,
            "parent_corpus_manifest_sha256": None,
        },
        "preparer": preparer,
        "decoder": {
            "contract": DECODER_CONTRACT,
            "closure_sha256": fixture.decoder_closure_sha256(
                DECODER_CONTRACT,
                preparer_files,
            ),
            "files": 2,
            "bytes": far.size_bytes + close.size_bytes,
            "production_evidence": production_evidence,
        },
        "cases": receipt_cases,
        "totals": {
            "cases": TOTAL_CASES,
            "pcm_samples": total_samples,
            "pcm_bytes": total_samples * 4,
        },
        "privacy": dict(_PRIVATE_RECEIPT_PRIVACY),
        "evidence_scope": dict(_RECEIPT_EVIDENCE_SCOPE),
    }
    receipt_raw = _canonical_json_bytes(receipt)
    if len(receipt_raw) > _RECEIPT_MAX_BYTES:
        raise AmiConversationPreparationError()
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    provenance = CorpusProvenance(
        kind="public-voice-v1",
        suite=source.corpus_suite,
        manifest_sha256=source.selection_recipe_sha256,
        metadata_sha256=metadata_sha256,
        source_set_sha256=receipt_sha256,
    )
    try:
        corpus = publish_private_corpus(
            cases=write_cases,
            provenance=provenance,
            output_dir=destination,
            purpose="AMI ES2004a synchronized close/far conversation fixture v1",
            sidecars={"preparation-receipt.json": receipt_raw},
        )
    except Exception:
        raise AmiConversationPreparationError() from None
    if (
        corpus.provenance != provenance
        or len(corpus.cases) != TOTAL_CASES
        or corpus.audio_bytes != total_samples * 4
        or _preparer_binding(source) != preparer
    ):
        raise AmiConversationPreparationError()
    return PreparedAmiConversationCorpus(
        corpus=corpus,
        receipt_sha256=receipt_sha256,
        metadata_sha256=metadata_sha256,
        selected_windows_sha256=_canonical_sha256(selected_rows),
    )


def _safe_result(value: PreparedAmiConversationCorpus) -> dict[str, object]:
    seconds = value.corpus.audio_bytes / 4 / SAMPLE_RATE_HZ
    if not math.isfinite(seconds):
        raise AmiConversationPreparationError()
    return {
        "ok": True,
        "dataset_id": DATASET_ID,
        "cases": len(value.corpus.cases),
        "audio_bytes": value.corpus.audio_bytes,
        "total_seconds": round(seconds, 3),
        "corpus_sha256": value.corpus.digest,
        "receipt_sha256": value.receipt_sha256,
        "metadata_sha256": value.metadata_sha256,
        "selected_windows_sha256": value.selected_windows_sha256,
    }


def _positive_int(value: str) -> int:
    try:
        result = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive byte count") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive byte count")
    return result


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise AmiConversationPreparationError()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description=(
            "Prepare a private AMI ES2004a synchronized close/far fixture; "
            "no download, model, network, or audio-device access."
        )
    )
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--far-wav", type=Path, required=True)
    parser.add_argument("--close-wav", type=Path, required=True)
    parser.add_argument("--close-sha256", required=True)
    parser.add_argument("--close-bytes", type=_positive_int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-term", action="append", default=[])
    parser.add_argument("--lock", type=Path, default=fixture.DEFAULT_LOCK)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = prepare_ami_conversation_fixture(
            annotations_zip=args.annotations_zip,
            far_wav=args.far_wav,
            close_wav=args.close_wav,
            close_sha256=args.close_sha256,
            close_bytes=args.close_bytes,
            output_dir=args.output_dir,
            accepted_terms=frozenset(args.accept_term),
            lock_path=args.lock,
        )
        payload = _safe_result(result)
        code = 0
    except Exception:  # noqa: BLE001 - never expose private paths or references
        payload = _SAFE_ERROR
        code = 2
    print(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AmiConversationPreparationError",
    "PreparedAmiConversationCorpus",
    "TestSourceIdentityInjection",
    "prepare_ami_conversation_fixture",
]
