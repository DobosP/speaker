from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from tools.streaming_stt.corpus import CorpusCase, LoadedCorpus
from tools.streaming_stt.final_metrics import (
    FinalDecisionRecord,
    FinalMetricsAccumulator,
    aggregate_final_metrics,
)


def _case(
    case_id: str,
    *,
    expected_text: str,
    assertion: str,
    commands: tuple[str, ...] = (),
    forbidden_commands: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    samples: int = 16_000,
) -> CorpusCase:
    return CorpusCase(
        case_id=case_id,
        source_path=Path(f"/private/{case_id}.f32le"),
        sha256="a" * 64,
        samples=samples,
        expected_text=expected_text,
        assertion=assertion,
        commands=commands,
        forbidden_commands=forbidden_commands,
        tags=tags,
        audio_bytes=b"",
    )


def _corpus(*cases: CorpusCase, schema_version: int = 4) -> LoadedCorpus:
    return LoadedCorpus(
        path=Path("/private/corpus.json"),
        digest="b" * 64,
        schema_version=schema_version,
        purpose="private final metric fixture",
        provenance=None,
        cases=tuple(cases),
        audio_bytes=sum(case.samples * 4 for case in cases),
    )


def _contains_private_decision(value: object, sentinel: str) -> bool:
    """Inspect retained accumulator data without following arbitrary objects."""

    if isinstance(value, FinalDecisionRecord):
        return True
    if isinstance(value, str):
        return sentinel in value
    if isinstance(value, dict):
        return any(
            _contains_private_decision(item, sentinel)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_private_decision(item, sentinel) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_private_decision(getattr(value, descriptor.name), sentinel)
            for descriptor in fields(value)
        )
    return False


def test_command_matching_preserves_decision_boundaries_but_wer_joins_them():
    corpus = _corpus(
        _case(
            "positive",
            expected_text="turn on",
            assertion="transcript",
            commands=("turn on",),
            forbidden_commands=("turn off",),
        )
    )

    result = aggregate_final_metrics(
        corpus,
        [FinalDecisionRecord(0, 0, ("turn", "on"))],
        repeats=1,
    )

    accuracy = result["accuracy"]
    assert accuracy["transcript"]["exact"] == 1
    assert accuracy["transcript"]["wer"] == 0.0
    assert accuracy["command_attempts"] == 1
    assert accuracy["command_hits"] == 0
    assert accuracy["positive_command_false_negatives"] == 1
    assert accuracy["forbidden_final_false_positives"] == 0
    assert result["decisions"] == {
        "total": 2,
        "zero_decision_evaluations": 0,
        "multi_decision_evaluations": 1,
    }


def test_schema_v4_final_metrics_cover_positive_negative_and_empty_outputs():
    corpus = _corpus(
        _case(
            "positive",
            expected_text="turn on",
            assertion="transcript",
            commands=("turn on",),
            forbidden_commands=("turn off",),
            tags=("clean",),
        ),
        _case(
            "transcript-negative",
            expected_text="dark",
            assertion="transcript",
            forbidden_commands=("turn on", "turn off"),
            tags=("negative", "noisy"),
        ),
        _case(
            "speech-negative",
            expected_text="",
            assertion="speech_negative",
            forbidden_commands=("turn on",),
            tags=("negative",),
        ),
        _case(
            "monitored-silence",
            expected_text="",
            assertion="silence",
            forbidden_commands=("turn on",),
            tags=("negative",),
        ),
        _case(
            "unmonitored-silence",
            expected_text="",
            assertion="silence",
            tags=("silence",),
        ),
    )
    records = (
        FinalDecisionRecord(0, 0, ("turn on", "turn off")),
        FinalDecisionRecord(1, 0, ("turn on",)),
        FinalDecisionRecord(2, 0, ("ordinary words",)),
        FinalDecisionRecord(3, 0, ("turn on",)),
        FinalDecisionRecord(4, 0, ()),
    )

    result = aggregate_final_metrics(corpus, records, repeats=1)

    assert result["evaluations"] == 5
    assert result["coverage_complete"] is True
    accuracy = result["accuracy"]
    assert accuracy["transcript"]["clips"] == 2
    assert accuracy["command_attempts"] == 1
    assert accuracy["command_hits"] == 1
    assert accuracy["command_recall"] == 1.0
    assert accuracy["positive_command_true_positives"] == 1
    assert accuracy["positive_command_false_negatives"] == 0
    assert accuracy["positive_command_recall"] == 1.0
    assert accuracy["positive_command_confusion_hits"] == 1
    assert accuracy["forbidden_final_false_positives"] == 3
    assert accuracy["forbidden_final_true_negatives"] == 2
    assert accuracy["final_command_target_precision"] == 0.25
    assert accuracy["negative_pair_false_positives"] == 2
    assert accuracy["negative_pair_true_negatives"] == 2
    assert accuracy["negative_pair_false_positive_rate"] == 0.5
    assert accuracy["negative_final_case_false_positives"] == 2
    assert accuracy["negative_final_case_true_negatives"] == 1
    assert accuracy["negative_final_case_false_positive_rate"] == 0.6667
    assert accuracy["monitored_negative_evaluations_including_silence"] == 3
    assert accuracy["monitored_negative_audio_seconds_including_silence"] == 3.0
    assert (
        accuracy["isolated_clip_lexical_false_positives_per_negative_audio_hour"]
        == 2400.0
    )
    assert accuracy["silence_nonempty_finals"] == 1
    assert accuracy["speech_negative_nonempty_finals"] == 1
    assert result["decisions"] == {
        "total": 5,
        "zero_decision_evaluations": 1,
        "multi_decision_evaluations": 1,
    }
    assert result["determinism"] == {
        "repeats": 1,
        "cases_with_final_disagreement": 0,
    }


def test_repeat_disagreement_is_boundary_sensitive_and_output_is_text_free():
    private_reference = "SENTINEL_PRIVATE_REFERENCE turn on"
    private_hypothesis = "SENTINEL_PRIVATE_HYPOTHESIS"
    corpus = _corpus(
        _case(
            "sentinel-private-case-id",
            expected_text=private_reference,
            assertion="transcript",
            commands=("turn on",),
        )
    )
    records = (
        FinalDecisionRecord(0, 0, (private_hypothesis, "turn on")),
        FinalDecisionRecord(0, 1, (private_hypothesis, "turn", "on")),
    )

    result = aggregate_final_metrics(corpus, records, repeats=2)
    encoded = json.dumps(result, sort_keys=True)

    assert result["accuracy"]["command_attempts"] == 2
    assert result["accuracy"]["command_hits"] == 1
    assert result["accuracy"]["command_recall"] == 0.5
    assert result["determinism"]["cases_with_final_disagreement"] == 1
    assert private_reference not in encoded
    assert private_hypothesis not in encoded
    assert "sentinel-private-case-id" not in encoded
    assert private_hypothesis not in repr(records[0])


def test_explicit_strata_are_sorted_deduplicated_and_aggregate_only():
    corpus = _corpus(
        _case(
            "clean-positive",
            expected_text="go",
            assertion="transcript",
            commands=("go",),
            tags=("clean", "all"),
        ),
        _case(
            "noisy-negative",
            expected_text="snow",
            assertion="transcript",
            forbidden_commands=("go",),
            tags=("noisy", "all"),
        ),
    )
    records = (
        FinalDecisionRecord(0, 0, ("go",)),
        FinalDecisionRecord(1, 0, ("go",)),
    )

    result = aggregate_final_metrics(
        corpus,
        records,
        repeats=1,
        stratum_tags=("noisy", "clean", "noisy"),
    )

    assert [row["tag"] for row in result["strata"]] == ["clean", "noisy"]
    assert [row["cases"] for row in result["strata"]] == [1, 1]
    assert all(row["metrics"]["coverage_complete"] for row in result["strata"])
    assert result["strata"][0]["metrics"]["accuracy"]["command_recall"] == 1.0
    assert (
        result["strata"][1]["metrics"]["accuracy"]
        ["negative_final_case_false_positive_rate"]
        == 1.0
    )


def test_incremental_accumulator_matches_batch_for_out_of_order_cases_and_repeats():
    corpus = _corpus(
        _case(
            "positive",
            expected_text="turn on lights",
            assertion="transcript",
            commands=("turn on",),
            forbidden_commands=("turn off",),
            tags=("all", "clean"),
            samples=8_000,
        ),
        _case(
            "negative",
            expected_text="ordinary phrase",
            assertion="transcript",
            forbidden_commands=("turn on", "turn off"),
            tags=("all", "noisy"),
            samples=32_000,
        ),
        _case(
            "silence",
            expected_text="",
            assertion="silence",
            forbidden_commands=("turn on",),
            tags=("all", "noisy"),
            samples=24_000,
        ),
    )
    by_case = {
        0: (
            FinalDecisionRecord(0, 0, ("turn on", "lights")),
            FinalDecisionRecord(0, 1, ("turn", "on lights")),
        ),
        1: (
            FinalDecisionRecord(1, 0, ("ordinary phrase",)),
            FinalDecisionRecord(1, 1, ("turn off",)),
        ),
        2: (
            FinalDecisionRecord(2, 0, ()),
            FinalDecisionRecord(2, 1, ("turn on",)),
        ),
    }
    batch_records = tuple(
        record
        for case_index in (1, 0, 2)
        for record in reversed(by_case[case_index])
    )
    tags = ("noisy", "clean", "all", "noisy")

    batch = aggregate_final_metrics(
        corpus,
        batch_records,
        repeats=2,
        stratum_tags=tags,
    )
    accumulator = FinalMetricsAccumulator(corpus, 2, stratum_tags=tags)
    for case_index in (2, 0, 1):
        accumulator.add_case(case_index, tuple(reversed(by_case[case_index])))

    assert accumulator.result() == batch


def test_incremental_accumulator_rejects_missing_duplicate_and_readded_cases():
    corpus = _corpus(
        _case(
            "positive",
            expected_text="go",
            assertion="transcript",
            commands=("go",),
        )
    )
    repeat_zero = FinalDecisionRecord(0, 0, ("go",))
    repeat_one = FinalDecisionRecord(0, 1, ("go",))
    accumulator = FinalMetricsAccumulator(corpus, 2)

    with pytest.raises(ValueError, match="incomplete final decision repeats"):
        accumulator.add_case(0, (repeat_zero,))
    assert accumulator.result()["evaluations"] == 0

    with pytest.raises(ValueError, match="invalid final decision record"):
        accumulator.add_case(0, (repeat_zero, repeat_zero))
    assert accumulator.result()["evaluations"] == 0

    accumulator.add_case(0, (repeat_one, repeat_zero))
    assert accumulator.result()["coverage_complete"] is True
    with pytest.raises(ValueError, match="duplicate metric case"):
        accumulator.add_case(0, (repeat_zero, repeat_one))


def test_incremental_accumulator_retains_no_private_final_decisions_or_repr_text():
    private_hypothesis = "SENTINEL_PRIVATE_INCREMENTAL_HYPOTHESIS"
    corpus = _corpus(
        _case(
            "public-case-id",
            expected_text="public reference",
            assertion="transcript",
            tags=("clean",),
        )
    )
    accumulator = FinalMetricsAccumulator(
        corpus,
        1,
        stratum_tags=("clean",),
    )
    accumulator.add_case(
        0,
        (FinalDecisionRecord(0, 0, (private_hypothesis,)),),
    )

    retained_state = tuple(
        getattr(accumulator, slot) for slot in FinalMetricsAccumulator.__slots__
    )
    result = accumulator.result()

    assert not _contains_private_decision(retained_state, private_hypothesis)
    assert private_hypothesis not in repr(accumulator)
    assert private_hypothesis not in repr(result)
    assert private_hypothesis not in json.dumps(result, sort_keys=True)


def test_incomplete_coverage_is_reported_but_duplicate_or_malformed_runs_fail():
    corpus = _corpus(
        _case(
            "first",
            expected_text="go",
            assertion="transcript",
            commands=("go",),
        ),
        _case(
            "second",
            expected_text="stop",
            assertion="transcript",
            commands=("stop",),
        ),
    )
    first = FinalDecisionRecord(0, 0, ("go",))

    result = aggregate_final_metrics(corpus, [first], repeats=1)
    assert result["coverage_complete"] is False

    with pytest.raises(ValueError):
        aggregate_final_metrics(corpus, [first, first], repeats=1)
    with pytest.raises(ValueError, match="invalid final decision record"):
        aggregate_final_metrics(
            corpus,
            [FinalDecisionRecord(0, 0, ["go"])],  # type: ignore[arg-type]
            repeats=1,
        )
    with pytest.raises(ValueError, match="invalid final decision record"):
        aggregate_final_metrics(
            corpus,
            [FinalDecisionRecord(2, 0, ("private",))],
            repeats=1,
        )


def test_only_schema_v4_and_valid_repeat_or_stratum_contracts_are_accepted():
    case = _case(
        "positive",
        expected_text="go",
        assertion="transcript",
        commands=("go",),
        tags=("clean",),
    )
    record = FinalDecisionRecord(0, 0, ("go",))

    with pytest.raises(ValueError, match="schema-v4 corpus required"):
        aggregate_final_metrics(_corpus(case, schema_version=3), [record], repeats=1)
    for repeats in (True, 0, 9):
        with pytest.raises(ValueError, match="invalid repeats"):
            aggregate_final_metrics(_corpus(case), [record], repeats=repeats)
    with pytest.raises(ValueError, match="invalid stratum tags"):
        aggregate_final_metrics(
            _corpus(case),
            [record],
            repeats=1,
            stratum_tags=("missing",),
        )
