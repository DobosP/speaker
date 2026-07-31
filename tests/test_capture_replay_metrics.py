from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from core.engine import TranscriptAbortReason
from tools.capture_replay.corpus import (
    FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
    LoadedReplayCorpus,
    ReplayAssertion,
    ReplayCase,
    ReplayTrack,
)
from tools.capture_replay.metrics import (
    ReplayAcousticEvent,
    ReplayRunRecord,
    TimedFinal,
    TimedHypothesis,
    aggregate_metrics,
)


def _case(
    tmp_path: Path,
    index: int,
    *,
    assertion: ReplayAssertion,
    expected_text: str,
    commands: tuple[str, ...] = (),
    tags: tuple[str, ...] = ("private-tag",),
) -> ReplayCase:
    samples = FRAME_SAMPLES * 20
    track = ReplayTrack(
        role="mic",
        path=tmp_path / f"private-{index}.f32le",
        sha256=f"{index + 1:064x}",
        samples=samples,
        pcm_bytes=b"\0" * (samples * 4),
    )
    return ReplayCase(
        case_id=f"private-case-{index}",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        tracks=MappingProxyType({"mic": track}),
        speech_intervals=(),
        speaker_intervals=(),
        word_intervals=(),
        assertion=assertion,
        expected_text=expected_text,
        commands=commands,
        tags=tags,
        aec_delay_samples=0,
    )


def _corpus(tmp_path: Path) -> LoadedReplayCorpus:
    cases = (
        _case(
            tmp_path,
            0,
            assertion=ReplayAssertion.TRANSCRIPT,
            expected_text="find my vault",
            commands=("find the vault",),
        ),
        _case(
            tmp_path,
            1,
            assertion=ReplayAssertion.SILENCE,
            expected_text="",
        ),
        _case(
            tmp_path,
            2,
            assertion=ReplayAssertion.DOUBLE_TALK,
            expected_text="remember the appointment",
        ),
    )
    return LoadedReplayCorpus(
        path=tmp_path / "private-corpus.json",
        digest="a" * 64,
        schema_version=1,
        purpose="private purpose",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=cases,
        audio_bytes=sum(case.samples * 4 for case in cases),
    )


def test_metrics_reduce_text_and_typed_endpoint_lineage_to_aggregates(tmp_path):
    corpus = _corpus(tmp_path)
    records = (
        ReplayRunRecord(
            case_index=0,
            repeat=0,
            partials=(
                TimedHypothesis(
                    "find",
                    emitted_at=10.4,
                    speech_start_at=10.0,
                    utterance_id="u1",
                ),
                TimedHypothesis(
                    "find the vault",
                    emitted_at=10.8,
                    speech_start_at=10.0,
                    utterance_id="u1",
                ),
            ),
            finals=(
                TimedFinal(
                    "find the vault",
                    emitted_at=11.25,
                    speech_start_at=10.0,
                    speech_end_at=10.7,
                    endpoint_committed_at=11.0,
                    utterance_id="u1",
                ),
            ),
            commands=("find the vault",),
            wall_seconds=2.2,
            peak_rss_mb=128.0,
        ),
        ReplayRunRecord(
            case_index=1,
            repeat=0,
            partials=(
                TimedHypothesis(
                    "private hallucination",
                    emitted_at=20.1,
                    utterance_id="u-silence",
                ),
            ),
            abort_reasons=(TranscriptAbortReason.INPUT_REJECTED,),
            wall_seconds=2.0,
            peak_rss_mb=130.0,
        ),
        ReplayRunRecord(
            case_index=2,
            repeat=0,
            partials=(
                TimedHypothesis(
                    "remember",
                    emitted_at=30.3,
                    speech_start_at=30.0,
                    utterance_id="u2",
                ),
                TimedHypothesis(
                    "remember the appointment",
                    emitted_at=30.7,
                    speech_start_at=30.0,
                    utterance_id="u2",
                ),
            ),
            finals=(
                TimedFinal(
                    "remember the appointment",
                    emitted_at=31.2,
                    speech_start_at=30.0,
                    speech_end_at=30.6,
                    endpoint_committed_at=31.0,
                    utterance_id="u2",
                ),
            ),
            acoustic_events=(ReplayAcousticEvent.BARGE_IN,),
            wall_seconds=2.1,
            peak_vram_mb=512.0,
        ),
    )

    report = aggregate_metrics(corpus, records, repeats=1)

    assert report["coverage"]["complete"] is True
    assert report["accuracy"]["transcript_runs"] == 2
    assert report["accuracy"]["word_errors"] == 1
    assert report["accuracy"]["reference_words"] == 6
    assert report["accuracy"]["wer"] == pytest.approx(1 / 6, abs=0.0001)
    assert report["accuracy"]["command_recall"] == 1.0
    assert report["accuracy"]["lexical_command_hits"] == 1
    assert report["accuracy"]["silence_nonempty_partials"] == 1
    assert report["accuracy"]["silence_nonempty_finals"] == 0
    assert report["latency"]["speech_end_to_endpoint_p50_ms"] == 300.0
    assert report["latency"]["endpoint_to_final_p95_ms"] == 250.0
    assert report["latency"]["first_partial_observations"] == 2
    assert report["latency"]["stable_partial_observations"] == 2
    assert report["latency"]["speech_end_to_final_observations"] == 2
    assert report["streaming"]["abort_reasons"] == {"input_rejected": 1}
    assert report["acoustic"]["double_talk_with_event"] == 1
    assert report["resources"] == {
        "peak_rss_mb": 130.0,
        "peak_vram_mb": 512.0,
    }

    encoded = json.dumps(report, sort_keys=True)
    for private in (
        "private-corpus",
        "private-case",
        "private purpose",
        "find my vault",
        "find the vault",
        "remember the appointment",
        "private hallucination",
        "private-tag",
        str(tmp_path),
    ):
        assert private not in encoded


def test_stable_partial_and_churn_include_final_transition(tmp_path):
    corpus = LoadedReplayCorpus(
        path=tmp_path / "private.json",
        digest="b" * 64,
        schema_version=1,
        purpose="private",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=(
            _case(
                tmp_path,
                0,
                assertion=ReplayAssertion.TRANSCRIPT,
                expected_text="alpha beta",
            ),
        ),
        audio_bytes=FRAME_SAMPLES * 20 * 4,
    )
    record = ReplayRunRecord(
        case_index=0,
        repeat=0,
        partials=(
            TimedHypothesis(
                "alpha wrong",
                1.2,
                speech_start_at=1.0,
                utterance_id="u1",
            ),
            TimedHypothesis(
                "alpha beta",
                1.5,
                speech_start_at=1.0,
                utterance_id="u1",
            ),
        ),
        finals=(
            TimedFinal(
                "alpha beta",
                emitted_at=2.0,
                speech_start_at=1.0,
                speech_end_at=1.6,
                endpoint_committed_at=1.9,
                utterance_id="u1",
            ),
        ),
        wall_seconds=2.0,
    )

    report = aggregate_metrics(corpus, (record,), repeats=1)

    assert report["streaming"]["churn_token_edits"] == 1
    assert report["streaming"]["retracted_words"] == 1
    assert report["streaming"]["stable_partial_missing"] == 0
    assert (
        report["latency"]["stable_partial_from_speech_start_p50_ms"]
        == 500.0
    )
    assert report["accuracy"]["command_recall"] is None


def test_accuracy_breakdown_uses_only_closed_nonprivate_conditions(tmp_path):
    corpus = LoadedReplayCorpus(
        path=tmp_path / "private.json",
        digest="1" * 64,
        schema_version=1,
        purpose="private",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=(
            _case(
                tmp_path,
                0,
                assertion=ReplayAssertion.TRANSCRIPT,
                expected_text="alpha beta",
                tags=("private-one", "single-speaker"),
            ),
            _case(
                tmp_path,
                1,
                assertion=ReplayAssertion.TRANSCRIPT,
                expected_text="gamma delta",
                tags=("private-two", "overlapping-speakers"),
            ),
        ),
        audio_bytes=2 * FRAME_SAMPLES * 20 * 4,
    )
    records = (
        ReplayRunRecord(
            case_index=0,
            repeat=0,
            finals=(
                TimedFinal(
                    "alpha beta",
                    emitted_at=2.0,
                    speech_start_at=1.0,
                    speech_end_at=1.5,
                    endpoint_committed_at=1.8,
                    utterance_id="u1",
                ),
            ),
            wall_seconds=1.0,
        ),
        ReplayRunRecord(
            case_index=1,
            repeat=0,
            finals=(
                TimedFinal(
                    "gamma",
                    emitted_at=4.0,
                    speech_start_at=3.0,
                    speech_end_at=3.5,
                    endpoint_committed_at=3.8,
                    utterance_id="u2",
                ),
            ),
            wall_seconds=1.0,
        ),
    )

    report = aggregate_metrics(corpus, records, repeats=1)
    conditions = report["accuracy"]["by_condition"]

    assert conditions["single_speaker"]["wer"] == 0.0
    assert conditions["single_speaker"]["reference_words"] == 2
    assert conditions["human_overlap"]["wer"] is None
    assert conditions["human_overlap"]["linearized_wer"] == 0.5
    assert conditions["human_overlap"]["reference_order_comparable"] is False
    assert conditions["human_overlap"]["reference_words"] == 2
    assert conditions["turn_transition"]["transcript_runs"] == 0
    assert conditions["turn_transition"]["wer"] is None
    assert conditions["turn_transition"]["cer"] is None
    encoded = json.dumps(report, sort_keys=True)
    assert "private-one" not in encoded
    assert "private-two" not in encoded


def test_duplicate_record_and_negative_lineage_time_are_rejected(tmp_path):
    corpus = LoadedReplayCorpus(
        path=tmp_path / "private.json",
        digest="c" * 64,
        schema_version=1,
        purpose="private",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=(
            _case(
                tmp_path,
                0,
                assertion=ReplayAssertion.TRANSCRIPT,
                expected_text="hello",
            ),
        ),
        audio_bytes=FRAME_SAMPLES * 20 * 4,
    )
    good = ReplayRunRecord(case_index=0, repeat=0, wall_seconds=1.0)

    with pytest.raises(ValueError):
        aggregate_metrics(corpus, (good, good), repeats=1)

    backwards = ReplayRunRecord(
        case_index=0,
        repeat=0,
        finals=(
            TimedFinal(
                "hello",
                emitted_at=2.0,
                speech_start_at=1.0,
                speech_end_at=1.8,
                endpoint_committed_at=1.7,
                utterance_id="u1",
            ),
        ),
        wall_seconds=1.0,
    )
    with pytest.raises(ValueError):
        aggregate_metrics(corpus, (backwards,), repeats=1)


def test_lexical_command_does_not_mask_missing_typed_command_callback(tmp_path):
    corpus = LoadedReplayCorpus(
        path=tmp_path / "private.json",
        digest="d" * 64,
        schema_version=1,
        purpose="private",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=(
            _case(
                tmp_path,
                0,
                assertion=ReplayAssertion.TRANSCRIPT,
                expected_text="stop speaking",
                commands=("stop speaking",),
            ),
        ),
        audio_bytes=FRAME_SAMPLES * 20 * 4,
    )
    record = ReplayRunRecord(
        case_index=0,
        repeat=0,
        finals=(
            TimedFinal(
                "stop speaking",
                emitted_at=3.0,
                speech_start_at=1.0,
                speech_end_at=2.0,
                endpoint_committed_at=2.5,
                utterance_id="u1",
            ),
        ),
        commands=(),
        wall_seconds=2.0,
    )

    report = aggregate_metrics(corpus, (record,), repeats=1)

    assert report["accuracy"]["lexical_command_hits"] == 1
    assert report["accuracy"]["command_hits"] == 0
    assert report["accuracy"]["command_recall"] == 0.0


def test_untyped_private_metadata_and_unbounded_observations_are_rejected(
    tmp_path,
):
    corpus = LoadedReplayCorpus(
        path=tmp_path / "private.json",
        digest="e" * 64,
        schema_version=1,
        purpose="private",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=(
            _case(
                tmp_path,
                0,
                assertion=ReplayAssertion.TRANSCRIPT,
                expected_text="hello",
            ),
        ),
        audio_bytes=FRAME_SAMPLES * 20 * 4,
    )
    private = str(tmp_path / "transcript-private-path")
    bad_event = ReplayRunRecord(
        case_index=0,
        repeat=0,
        acoustic_events=(private,),  # type: ignore[arg-type]
        wall_seconds=1.0,
    )
    bad_reason = ReplayRunRecord(
        case_index=0,
        repeat=0,
        abort_reasons=(private,),  # type: ignore[arg-type]
        wall_seconds=1.0,
    )
    too_long = ReplayRunRecord(
        case_index=0,
        repeat=0,
        partials=(
            TimedHypothesis(
                "x" * 4_097,
                emitted_at=1.0,
                utterance_id="u1",
            ),
        ),
        wall_seconds=1.0,
    )

    for record in (bad_event, bad_reason, too_long):
        with pytest.raises(ValueError) as raised:
            aggregate_metrics(corpus, (record,), repeats=1)
        assert private not in str(raised.value)


def test_missing_final_and_lineage_are_explicit_coverage_failures(tmp_path):
    corpus = LoadedReplayCorpus(
        path=tmp_path / "private.json",
        digest="f" * 64,
        schema_version=1,
        purpose="private",
        sample_rate_hz=SAMPLE_RATE_HZ,
        frame_samples=FRAME_SAMPLES,
        cases=(
            _case(
                tmp_path,
                0,
                assertion=ReplayAssertion.TRANSCRIPT,
                expected_text="hello",
            ),
        ),
        audio_bytes=FRAME_SAMPLES * 20 * 4,
    )

    missing_final = aggregate_metrics(
        corpus,
        (ReplayRunRecord(case_index=0, repeat=0, wall_seconds=1.0),),
        repeats=1,
    )
    assert missing_final["streaming"]["text_runs_without_final"] == 1
    assert missing_final["streaming"]["stable_partial_missing"] == 1

    missing_timing = aggregate_metrics(
        corpus,
        (
            ReplayRunRecord(
                case_index=0,
                repeat=0,
                finals=(
                    TimedFinal(
                        "hello",
                        emitted_at=2.0,
                        speech_start_at=None,
                        speech_end_at=None,
                        endpoint_committed_at=None,
                        utterance_id="u1",
                    ),
                ),
                wall_seconds=1.0,
            ),
        ),
        repeats=1,
    )
    lineage = missing_timing["acoustic"]["lineage_coverage"]
    assert lineage["finals"] == 1
    assert lineage["finals_complete"] == 0
    assert lineage["finals_missing_speech_start"] == 1
    assert lineage["speech_end_to_endpoint_observations"] == 0
