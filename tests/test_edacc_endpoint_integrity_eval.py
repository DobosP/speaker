from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest

from always_on_agent.acoustic import EndpointReason
from always_on_agent.logical_turn import LogicalTurnBoundary
from always_on_agent.logical_turn_shadow import (
    LogicalTurnCompositionShadow,
    aggregate_logical_turn_shadows,
    validate_logical_turn_shadow_report,
)
from core.endpointing import (
    CompletionScoreState,
    LexicalTurnCompletionDetector,
    TurnDecision,
    TurnDecisionBasis,
    TurnObservation,
)
from core.engine import TranscriptAbortReason
from core.engines.sherpa import SherpaConfig
from tools.bench.production import PreparedProduction, ProductionInputError
from tools.capture_replay.metrics import (
    ReplayAcousticEvent,
    ReplayRunRecord,
    TimedFinal,
)
from tools.streaming_stt.corpus import CorpusCase, LoadedCorpus
from tools import edacc_endpoint_integrity_eval as endpoint_eval


_PRIVATE_REFERENCE = "SENTINEL private reference"
_PRIVATE_HYPOTHESIS = "SENTINEL private hypothesis"


def _corpus() -> LoadedCorpus:
    samples = endpoint_eval.EXPECTED_SOURCE_BYTES // 4 // endpoint_eval.EXPECTED_CASES
    assert samples * endpoint_eval.EXPECTED_CASES * 4 == endpoint_eval.EXPECTED_SOURCE_BYTES
    cases = []
    for ordinal in range(endpoint_eval.EXPECTED_CASES):
        tag = endpoint_eval.STRATA[ordinal // 8]
        raw = b"\0" * (samples * 4)
        cases.append(
            CorpusCase(
                case_id=f"private-{ordinal}",
                source_path=Path(f"/private/source-{ordinal}.f32le"),
                sha256=hashlib.sha256(raw).hexdigest(),
                samples=samples,
                expected_text=f"{_PRIVATE_REFERENCE} {ordinal}",
                assertion="transcript",
                commands=(),
                forbidden_commands=(),
                tags=("private", tag),
                audio_bytes=raw,
            )
        )
    return LoadedCorpus(
        path=Path("/private/corpus.json"),
        digest=endpoint_eval.EXPECTED_CORPUS_SHA256,
        schema_version=2,
        purpose="private fixture",
        provenance=None,
        cases=tuple(cases),
        audio_bytes=endpoint_eval.EXPECTED_SOURCE_BYTES,
    )


def _base_config() -> SherpaConfig:
    return SherpaConfig(
        sample_rate=endpoint_eval.SAMPLE_RATE_HZ,
        block_sec=0.1,
        asr_rule2_min_trailing_silence=endpoint_eval.ACOUSTIC_RULE2_SEC,
        asr_encoder="/private/encoder.onnx",
        asr_tokens="/private/tokens.txt",
        vad_model="/private/vad.onnx",
        asr_final_backend="sense_voice",
        asr_final_model="/private/final.onnx",
        asr_final_tokens="/private/final-tokens.txt",
    )


def _prepared() -> PreparedProduction:
    return PreparedProduction(
        config={"sherpa": asdict(_base_config())},
        device_profile=endpoint_eval.DEVICE_PROFILE,
        final_stt_profile=endpoint_eval.FINAL_STT_PROFILE,
        final_stt_profile_sha256="1" * 64,
        final_stt_profile_schema_version=2,
        production_config_sha256="2" * 64,
        execution_config_sha256="3" * 64,
        replay_safety_schema_version=2,
        replay_safety_sha256="4" * 64,
        active_artifacts_sha256="5" * 64,
        active_artifact_roots=5,
        active_artifact_files=9,
        active_artifact_bytes=1234,
        source_revision="9" * 40,
        runtime_identities={"runtime": "private-but-hashed"},
    )


def _model(path: Path = Path("/private/smart-turn.onnx")) -> endpoint_eval._ModelBinding:
    return endpoint_eval._ModelBinding(
        path=path,
        sha256=endpoint_eval.SMART_TURN_MODEL_SHA256,
        size_bytes=endpoint_eval.SMART_TURN_MODEL_BYTES,
    )


def _finals(count: int, ordinal: int) -> tuple[TimedFinal, ...]:
    return tuple(
        TimedFinal(
            f"{_PRIVATE_HYPOTHESIS} {ordinal} {index}",
            emitted_at=2.0 + index,
            speech_start_at=0.1,
            speech_end_at=1.0 + index,
            endpoint_committed_at=1.8 + index,
            utterance_id=f"private-utterance-{ordinal}-{index}",
        )
        for index in range(count)
    )


def _cell_result(
    corpus: LoadedCorpus,
    *,
    candidate: bool,
    reset_accumulate: bool = False,
    buckets: tuple[str, ...],
    logical_turn_shadow: bool = False,
) -> endpoint_eval._CellResult:
    aggregate = endpoint_eval._CellAccumulator()
    trace = endpoint_eval._TraceAccumulator(candidate=candidate)
    for ordinal, (case, bucket) in enumerate(zip(corpus.cases, buckets, strict=True)):
        count = {"zero": 0, "single": 1, "multi": 2}[bucket]
        trace.begin_row()
        if candidate and ordinal == 0:
            trace.observe(
                TurnObservation(
                    acoustic_endpoint=True,
                    vad_active=False,
                    early_endpoint_allowed=False,
                    trailing_silence_sec=0.8,
                    completion_state=CompletionScoreState.SCORED,
                    completion_score=0.1,
                ),
                TurnDecision(
                    commit=False,
                    endpoint_reason=EndpointReason.UNKNOWN,
                    basis=TurnDecisionBasis.SEMANTIC_HOLD,
                ),
            )
        trace.observe(
            TurnObservation(
                acoustic_endpoint=True,
                vad_active=False,
                early_endpoint_allowed=False,
                trailing_silence_sec=1.6,
                completion_state=(
                    CompletionScoreState.SCORED
                    if candidate
                    else CompletionScoreState.UNAVAILABLE
                ),
                completion_score=0.5 if candidate else None,
            ),
            TurnDecision(
                commit=True,
                endpoint_reason=EndpointReason.ASR,
                basis=TurnDecisionBasis.ACOUSTIC,
            ),
        )
        trace.finish_row()
        aggregate.add(
            case,
            ReplayRunRecord(
                case_index=ordinal,
                repeat=0,
                finals=_finals(count, ordinal),
                abort_reasons=(
                    (TranscriptAbortReason.ABANDONED,)
                    if candidate and ordinal == 0
                    else ()
                ),
                wall_seconds=0.01,
                peak_rss_mb=64.0,
            ),
            ordinal=ordinal,
        )
    report = aggregate.report()
    report["config_sha256"] = (
        "9" if reset_accumulate else "7" if candidate else "6"
    ) * 64
    report["provider"] = endpoint_eval.ASR_PROVIDER
    report["asr_threads"] = endpoint_eval.ASR_THREADS
    report["semantic_hold_reset_enabled"] = reset_accumulate
    report["model_load_ms"] = 1.0
    report["endpoint_trace"] = trace.report()
    if logical_turn_shadow:
        report["logical_turn_shadow"] = _logical_turn_shadow_report(
            buckets,
            multi_epoch=reset_accumulate,
        )
    return endpoint_eval._CellResult(
        report=report,
        row_final_buckets=tuple(aggregate.row_final_buckets),
    )


def _logical_turn_shadow_report(
    buckets: tuple[str, ...],
    *,
    multi_epoch: bool,
) -> dict[str, object]:
    snapshots = []
    for ordinal, bucket in enumerate(buckets):
        shadow = LogicalTurnCompositionShadow()
        finals = {"zero": 0, "single": 1, "multi": 2}[bucket]
        for final_index in range(finals):
            if multi_epoch and ordinal == 0 and final_index == 0:
                assert shadow.observe_native_final(
                    native_epoch=1,
                    native_sequence=0,
                    text="first native final",
                    held=True,
                )
                assert shadow.observe_native_final(
                    native_epoch=2,
                    native_sequence=0,
                    text="second native final",
                    held=False,
                )
                legacy_text = "first native final second native final"
            else:
                legacy_text = f"native final {ordinal} {final_index}"
                assert shadow.observe_native_final(
                    native_epoch=1,
                    native_sequence=0,
                    text=legacy_text,
                    held=False,
                )
            assert shadow.compare_legacy_composition(
                boundary=LogicalTurnBoundary.NATIVE_FINAL,
                legacy_text=legacy_text,
            )
        shadow.close()
        snapshots.append(shadow.snapshot())
    return aggregate_logical_turn_shadows(snapshots)


def _valid_report(*, logical_turn_shadow: bool = False) -> dict[str, object]:
    corpus = _corpus()
    control = ("multi", "multi") + ("single",) * 22
    hold_only = ("multi",) + ("single",) * 23
    reset_accumulate = ("single",) * 24
    return endpoint_eval._report(
        prepared=_prepared(),
        model=_model(),
        source_sha256="8" * 64,
        cells={
            "acoustic": _cell_result(
                corpus,
                candidate=False,
                buckets=control,
                logical_turn_shadow=logical_turn_shadow,
            ),
            "prosody_hold_only": _cell_result(
                corpus,
                candidate=True,
                buckets=hold_only,
                logical_turn_shadow=logical_turn_shadow,
            ),
            "prosody_reset_accumulate": _cell_result(
                corpus,
                candidate=True,
                reset_accumulate=True,
                buckets=reset_accumulate,
                logical_turn_shadow=logical_turn_shadow,
            ),
        },
        logical_turn_shadow=logical_turn_shadow,
    )


def test_exact_private_source_and_smart_turn_pins_are_fixed() -> None:
    assert endpoint_eval.EXPECTED_CORPUS_SHA256 == (
        "4da392c39a0b6bd18057f63c96b4f67c0dfdaf4c14e1d2d76176cdbee0920772"
    )
    assert endpoint_eval.EXPECTED_CASES == 24
    assert endpoint_eval.EXPECTED_SOURCE_BYTES == 4_790_400
    assert endpoint_eval.EXPECTED_STRATUM_COUNTS == (8, 8, 8)
    assert endpoint_eval.LOGICAL_RULE3_SEC == 20.0
    assert endpoint_eval.SMART_TURN_MODEL_BYTES == 8_679_182
    assert endpoint_eval.SMART_TURN_MODEL_SHA256 == (
        "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
    )


def test_shadow_source_binding_is_strictly_conditional() -> None:
    default_files = endpoint_eval._source_files()
    shadow_files = endpoint_eval._source_files(logical_turn_shadow=True)
    assert default_files == endpoint_eval._SOURCE_FILES
    assert shadow_files == (
        *default_files,
        "always_on_agent/logical_turn.py",
        "always_on_agent/logical_turn_shadow.py",
        "core/contract.py",
    )
    assert endpoint_eval._source_binding() != endpoint_eval._source_binding(
        logical_turn_shadow=True
    )
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._source_files(logical_turn_shadow=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    ("schema", "digest", "cases", "bytes", "strata"),
)
def test_exact_corpus_contract_rejects_every_pin_drift(mutation: str) -> None:
    corpus = _corpus()
    if mutation == "schema":
        corpus = replace(corpus, schema_version=3)
    elif mutation == "digest":
        corpus = replace(corpus, digest="0" * 64)
    elif mutation == "cases":
        corpus = replace(corpus, cases=corpus.cases[:-1])
    elif mutation == "bytes":
        corpus = replace(corpus, audio_bytes=corpus.audio_bytes - 4)
    else:
        changed = replace(corpus.cases[0], tags=("private", "disfluent"))
        corpus = replace(corpus, cases=(changed, *corpus.cases[1:]))
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._validate_exact_corpus(corpus)


def test_capture_adapter_adds_exact_tail_and_no_invented_timing() -> None:
    case = _corpus().cases[0]
    adapted = endpoint_eval._adapt_case(case)
    alignment = (-case.samples) % endpoint_eval.FRAME_SAMPLES
    assert adapted.samples == (
        case.samples + alignment + endpoint_eval.ZERO_TAIL_SAMPLES
    )
    assert adapted.samples % endpoint_eval.FRAME_SAMPLES == 0
    assert adapted.mic.pcm_bytes[: len(case.audio_bytes)] == case.audio_bytes
    assert set(adapted.mic.pcm_bytes[len(case.audio_bytes) :]) <= {0}
    assert adapted.speech_intervals == ()
    assert adapted.speaker_intervals == ()
    assert adapted.word_intervals == ()


def test_three_cells_are_exact_and_calibration_is_disabled() -> None:
    model = _model()
    acoustic = endpoint_eval._cell_config(
        _base_config(), cell="acoustic", model=model
    )
    hold_only = endpoint_eval._cell_config(
        _base_config(), cell="prosody_hold_only", model=model
    )
    reset_accumulate = endpoint_eval._cell_config(
        _base_config(), cell="prosody_reset_accumulate", model=model
    )
    assert acoustic.endpoint_enabled is False
    assert acoustic.endpoint_prosody_model == ""
    assert acoustic.endpoint_reset_on_semantic_hold is False
    assert hold_only.endpoint_enabled is True
    assert hold_only.endpoint_detector == "prosody"
    assert hold_only.endpoint_prosody_model == str(model.path)
    assert hold_only.endpoint_reset_on_semantic_hold is False
    assert reset_accumulate.endpoint_enabled is True
    assert reset_accumulate.endpoint_detector == "prosody"
    assert reset_accumulate.endpoint_prosody_model == str(model.path)
    assert reset_accumulate.endpoint_reset_on_semantic_hold is True
    for value in (acoustic, hold_only, reset_accumulate):
        assert value.provider == "cpu"
        assert value.asr_num_threads == 1
        assert value.input_calibrate is False
        assert value.input_calibrate_sec == 0.0
        assert value.asr_rule2_min_trailing_silence == 0.8
        assert value.endpoint_max_silence_sec == 1.6
        assert value.asr_rule3_min_utterance_length == 20.0
        assert value.endpoint_complete_threshold == 0.6
        assert value.endpoint_incomplete_threshold == 0.3
        assert value.endpoint_adaptive_floor is False


@pytest.mark.parametrize(
    ("cell", "reset_accumulate"),
    (
        ("prosody_hold_only", False),
        ("prosody_reset_accumulate", True),
    ),
)
def test_run_semantic_cells_force_hold_only_and_install_fresh_policy_per_row(
    cell: str,
    reset_accumulate: bool,
    monkeypatch,
) -> None:
    policies = []
    received_allow_early = []

    class FakeEngine:
        def __init__(self, config):
            self.config = config
            self._endpoint_policy = object()
            self._streaming_decode_session = None

        def _build(self):
            return None

        def warm(self):
            return None

        def _turn_evaluation(self, **kwargs):
            received_allow_early.append(kwargs["allow_early"])
            return (
                TurnObservation(
                    acoustic_endpoint=True,
                    vad_active=False,
                    early_endpoint_allowed=kwargs["allow_early"],
                    trailing_silence_sec=0.8,
                    completion_state=CompletionScoreState.SCORED,
                    completion_score=0.7,
                ),
                TurnDecision(
                    commit=True,
                    endpoint_reason=EndpointReason.ASR,
                    basis=TurnDecisionBasis.ACOUSTIC,
                ),
            )

    def execute(engine, case, *, case_index, repeat):
        del case
        policies.append(engine._endpoint_policy)
        engine._turn_evaluation(
            acoustic_endpoint=True,
            partial=_PRIVATE_HYPOTHESIS,
            silence_sec=0.8,
            samples=None,
            allow_early=True,
            vad_active=False,
        )
        return ReplayRunRecord(
            case_index=case_index,
            repeat=repeat,
            finals=_finals(1, case_index),
            wall_seconds=0.01,
            peak_rss_mb=32.0,
        )

    monkeypatch.setattr(endpoint_eval, "_replay_config", lambda config, **_: config)
    monkeypatch.setattr(endpoint_eval, "_quiet_model_output", nullcontext)
    monkeypatch.setattr(endpoint_eval, "SherpaOnnxEngine", FakeEngine)
    monkeypatch.setattr(endpoint_eval, "_attest_engine", lambda *_, **__: None)
    monkeypatch.setattr(endpoint_eval, "_adapt_case", lambda case: case)
    monkeypatch.setattr(endpoint_eval, "_execute_case", execute)
    monkeypatch.setattr(endpoint_eval.gc, "collect", lambda: 0)

    result = endpoint_eval._run_cell(
        _corpus(),
        endpoint_eval._cell_config(
            _base_config(), cell=cell, model=_model()
        ),
        cell=cell,
    )
    assert result.report["coverage"] == {"rows": 24, "complete": True}
    assert result.report["semantic_hold_reset_enabled"] is reset_accumulate
    assert received_allow_early == [False] * 24
    assert len(policies) == 24
    assert len({id(policy) for policy in policies}) == 24


def test_run_cell_shadow_uses_one_closed_observer_per_row(monkeypatch) -> None:
    observers = []

    class FakeEngine:
        def __init__(self, config):
            self.config = config
            self._endpoint_policy = None
            self._streaming_decode_session = None

        def _build(self):
            return None

        def warm(self):
            return None

        def _turn_evaluation(self, **kwargs):
            return (
                TurnObservation(
                    acoustic_endpoint=True,
                    vad_active=False,
                    early_endpoint_allowed=kwargs["allow_early"],
                    trailing_silence_sec=0.8,
                    completion_state=CompletionScoreState.UNAVAILABLE,
                ),
                TurnDecision(
                    commit=True,
                    endpoint_reason=EndpointReason.ASR,
                    basis=TurnDecisionBasis.ACOUSTIC,
                ),
            )

    def execute(
        engine,
        case,
        *,
        case_index,
        repeat,
        logical_turn_shadow,
    ):
        del case
        observers.append(logical_turn_shadow)
        engine._turn_evaluation(
            acoustic_endpoint=True,
            partial="native final",
            silence_sec=0.8,
            samples=None,
            allow_early=True,
            vad_active=False,
        )
        assert logical_turn_shadow.observe_native_final(
            native_epoch=1,
            native_sequence=0,
            text="native final",
            held=False,
        )
        assert logical_turn_shadow.compare_legacy_composition(
            boundary=LogicalTurnBoundary.NATIVE_FINAL,
            legacy_text="native final",
        )
        return ReplayRunRecord(
            case_index=case_index,
            repeat=repeat,
            finals=_finals(1, case_index),
            wall_seconds=0.01,
            peak_rss_mb=32.0,
        )

    monkeypatch.setattr(endpoint_eval, "_replay_config", lambda config, **_: config)
    monkeypatch.setattr(endpoint_eval, "_quiet_model_output", nullcontext)
    monkeypatch.setattr(endpoint_eval, "SherpaOnnxEngine", FakeEngine)
    monkeypatch.setattr(endpoint_eval, "_attest_engine", lambda *_, **__: None)
    monkeypatch.setattr(endpoint_eval, "_adapt_case", lambda case: case)
    monkeypatch.setattr(endpoint_eval, "_execute_case", execute)
    monkeypatch.setattr(endpoint_eval.gc, "collect", lambda: 0)

    result = endpoint_eval._run_cell(
        _corpus(),
        endpoint_eval._cell_config(
            _base_config(), cell="acoustic", model=_model()
        ),
        cell="acoustic",
        logical_turn_shadow=True,
    )
    evidence = result.report["logical_turn_shadow"]
    assert len(observers) == endpoint_eval.EXPECTED_CASES
    assert len({id(observer) for observer in observers}) == endpoint_eval.EXPECTED_CASES
    assert all(observer.snapshot().closed for observer in observers)
    assert evidence["coverage"]["closed_runs"] == endpoint_eval.EXPECTED_CASES
    assert evidence["parity"]["composition_matches"] == endpoint_eval.EXPECTED_CASES
    assert evidence["parity"]["evidence_sufficient"] is False
    assert evidence["parity"]["exact"] is False


def test_trace_rejects_fallback_errors_and_exposes_hard_boundary_counts() -> None:
    trace = endpoint_eval._TraceAccumulator(candidate=True)
    trace.begin_row()
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        trace.observe(
            TurnObservation(
                acoustic_endpoint=True,
                vad_active=False,
                early_endpoint_allowed=False,
                trailing_silence_sec=0.8,
                completion_state=CompletionScoreState.DETECTOR_ERROR,
            ),
            TurnDecision(True, EndpointReason.ASR, TurnDecisionBasis.ACOUSTIC),
        )

    trace = endpoint_eval._TraceAccumulator(candidate=True)
    trace.begin_row()
    trace.observe(
        TurnObservation(
            acoustic_endpoint=True,
            vad_active=True,
            early_endpoint_allowed=False,
            trailing_silence_sec=1.7,
            completion_state=CompletionScoreState.HARD_BOUNDARY,
        ),
        TurnDecision(True, EndpointReason.MAX_WAIT, TurnDecisionBasis.MAX_WAIT),
    )
    trace.finish_row()
    # Add one scored row so this is valid candidate evidence.
    trace.begin_row()
    trace.observe(
        TurnObservation(
            acoustic_endpoint=True,
            vad_active=False,
            early_endpoint_allowed=False,
            trailing_silence_sec=1.6,
            completion_state=CompletionScoreState.SCORED,
            completion_score=0.5,
        ),
        TurnDecision(True, EndpointReason.ASR, TurnDecisionBasis.ACOUSTIC),
    )
    trace.finish_row()
    report = trace.report()
    assert report["hard_boundary_count"] == 1
    assert report["max_wait_count"] == 1


def test_engine_attestation_rejects_lexical_fallback() -> None:
    class Session:
        @staticmethod
        def get_providers():
            return ["CPUExecutionProvider"]

    detector = endpoint_eval.ProsodyTurnCompletionDetector.__new__(
        endpoint_eval.ProsodyTurnCompletionDetector
    )
    detector._session = Session()
    config = SimpleNamespace(
        asr_final_backend="sense_voice",
        provider="cpu",
        asr_num_threads=1,
        endpoint_enabled=True,
        endpoint_detector="prosody",
        endpoint_prosody_threads=1,
        endpoint_adaptive_floor=False,
        endpoint_reset_on_semantic_hold=False,
    )
    engine = SimpleNamespace(
        _recognizer=object(),
        _vad=object(),
        _final_recognizer=object(),
        _turn_detector=detector,
        _endpoint_policy=object(),
        _endpoint_wants_audio=True,
        config=config,
    )
    endpoint_eval._attest_engine(engine, cell="prosody_hold_only")
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._attest_engine(
            engine, cell="prosody_reset_accumulate"
        )
    config.endpoint_reset_on_semantic_hold = True
    endpoint_eval._attest_engine(engine, cell="prosody_reset_accumulate")
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._attest_engine(engine, cell="prosody_hold_only")
    config.endpoint_reset_on_semantic_hold = False
    engine._turn_detector = LexicalTurnCompletionDetector()
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._attest_engine(engine, cell="prosody_hold_only")


@pytest.mark.parametrize(
    ("field", "value"),
    (("provider", "cuda"), ("asr_num_threads", 2)),
)
def test_engine_attestation_rejects_cpu_execution_drift(
    field: str,
    value: object,
) -> None:
    config = SimpleNamespace(
        asr_final_backend="sense_voice",
        provider="cpu",
        asr_num_threads=1,
        endpoint_enabled=False,
        endpoint_reset_on_semantic_hold=False,
    )
    setattr(config, field, value)
    engine = SimpleNamespace(
        _recognizer=object(),
        _vad=object(),
        _final_recognizer=object(),
        _endpoint_policy=None,
        config=config,
    )
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._attest_engine(engine, cell="acoustic")


def test_aggregate_report_has_split_transitions_without_private_values() -> None:
    report = _valid_report()
    encoded = json.dumps(report, sort_keys=True)
    assert _PRIVATE_REFERENCE not in encoded
    assert _PRIVATE_HYPOTHESIS not in encoded
    assert "/private/" not in encoded
    comparisons = report["comparison"]
    assert comparisons["prosody_hold_only_vs_acoustic"][
        "row_final_count_transitions"
    ]["multi->single"] == 1
    assert comparisons["prosody_reset_accumulate_vs_acoustic"][
        "row_final_count_transitions"
    ]["multi->single"] == 2
    assert comparisons[
        "prosody_reset_accumulate_vs_prosody_hold_only"
    ]["row_final_count_transitions"]["multi->single"] == 1
    assert comparisons["prosody_reset_accumulate_vs_acoustic"][
        "candidate_reduced_multi_rows"
    ] is True
    assert report["cells"]["prosody_hold_only"]["transcript_aborts"] == {
        "total": 1,
        "terminal_events": 26,
        "reason_counts": {
            reason.value: int(reason is TranscriptAbortReason.ABANDONED)
            for reason in TranscriptAbortReason
        },
    }
    assert report["cells"]["acoustic"]["semantic_hold_reset_enabled"] is False
    assert report["cells"]["prosody_hold_only"][
        "semantic_hold_reset_enabled"
    ] is False
    assert report["cells"]["prosody_reset_accumulate"][
        "semantic_hold_reset_enabled"
    ] is True
    assert report["limitations"]["native_reset_accumulate_evaluated"] is True
    assert report["limitations"]["model_quality_promotion"] is False


def test_opt_in_shadow_uses_schema_three_without_changing_default_shape() -> None:
    default = _valid_report()
    shadowed = _valid_report(logical_turn_shadow=True)

    assert default["schema_version"] == endpoint_eval.SCHEMA_VERSION
    assert default["kind"] == endpoint_eval.KIND
    assert default["bindings"]["source"]["files"] == len(
        endpoint_eval._SOURCE_FILES
    )
    assert "raw_composition_shadow_enabled" not in default["protocol"]
    assert "raw_composition_shadow_evaluated" not in default["limitations"]
    assert all(
        "logical_turn_shadow" not in cell for cell in default["cells"].values()
    )

    assert shadowed["schema_version"] == endpoint_eval.SHADOW_SCHEMA_VERSION
    assert shadowed["kind"] == endpoint_eval.SHADOW_KIND
    assert shadowed["bindings"]["source"]["files"] == len(
        endpoint_eval._SOURCE_FILES
    ) + len(endpoint_eval._SHADOW_SOURCE_FILES)
    assert shadowed["protocol"]["raw_composition_shadow_enabled"] is True
    assert shadowed["limitations"]["raw_composition_shadow_evaluated"] is True
    for name, cell in shadowed["cells"].items():
        evidence = cell["logical_turn_shadow"]
        validate_logical_turn_shadow_report(evidence)
        assert evidence["coverage"] == {
            "runs": 24,
            "closed_runs": 24,
            "active_at_report": 0,
            "complete": True,
        }
        assert evidence["health"]["ok"] is True
        assert evidence["parity"]["composition_mismatches"] == 0
        assert evidence["parity"]["extra_legacy_terminals"] == 0
        assert evidence["parity"]["missing_legacy_terminals"] == 0
        if name == "prosody_reset_accumulate":
            assert evidence["parity"]["multi_epoch_commits"] > 0
            assert evidence["parity"]["evidence_sufficient"] is True
            assert evidence["parity"]["exact"] is True
        else:
            assert evidence["parity"]["multi_epoch_commits"] == 0
            assert evidence["parity"]["evidence_sufficient"] is False
            assert evidence["parity"]["exact"] is False
    encoded = json.dumps(shadowed, sort_keys=True)
    assert _PRIVATE_REFERENCE not in encoded
    assert _PRIVATE_HYPOTHESIS not in encoded
    assert "/private/" not in encoded


@pytest.mark.parametrize(
    "mutation",
    (
        "identity",
        "missing_cell_shadow",
        "protocol_mode",
        "source_count",
        "coverage",
        "health",
        "composition",
        "reset_evidence",
        "vacuous_control",
        "sparse_control",
        "capability",
        "forbidden_key",
    ),
)
def test_shadow_report_mutations_fail_closed(mutation: str) -> None:
    report = _valid_report(logical_turn_shadow=True)
    reset = report["cells"]["prosody_reset_accumulate"]["logical_turn_shadow"]
    if mutation == "identity":
        report["kind"] = endpoint_eval.KIND
    elif mutation == "missing_cell_shadow":
        report["cells"]["acoustic"].pop("logical_turn_shadow")
    elif mutation == "protocol_mode":
        report["protocol"]["raw_composition_shadow_enabled"] = False
    elif mutation == "source_count":
        report["bindings"]["source"]["files"] -= 1
    elif mutation == "coverage":
        reset["coverage"]["closed_runs"] -= 1
    elif mutation == "health":
        reset["health"]["internal_errors"] += 1
    elif mutation == "composition":
        reset["parity"]["composition_mismatches"] += 1
    elif mutation == "reset_evidence":
        report["cells"]["prosody_reset_accumulate"][
            "logical_turn_shadow"
        ] = report["cells"]["acoustic"]["logical_turn_shadow"]
    elif mutation == "vacuous_control":
        report["cells"]["acoustic"][
            "logical_turn_shadow"
        ] = _logical_turn_shadow_report(("zero",) * 24, multi_epoch=False)
    elif mutation == "sparse_control":
        report["cells"]["prosody_hold_only"][
            "logical_turn_shadow"
        ] = _logical_turn_shadow_report(("single",) * 24, multi_epoch=False)
    elif mutation == "capability":
        reset["capabilities"]["selected_final_parity"] = True
    else:
        reset["parity"]["text"] = _PRIVATE_HYPOTHESIS
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._validate_report(report)


def test_untyped_abort_reason_is_rejected_before_aggregation() -> None:
    aggregate = endpoint_eval._CellAccumulator()
    record = ReplayRunRecord(
        case_index=0,
        repeat=0,
        finals=_finals(1, 0),
        abort_reasons=("SENTINEL private abort",),  # type: ignore[arg-type]
        wall_seconds=0.01,
    )
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        aggregate.add(_corpus().cases[0], record, ordinal=0)


@pytest.mark.parametrize("surface", ("command", "acoustic_event"))
def test_impossible_command_and_acoustic_surfaces_are_rejected(surface: str) -> None:
    aggregate = endpoint_eval._CellAccumulator()
    record = ReplayRunRecord(
        case_index=0,
        repeat=0,
        finals=_finals(1, 0),
        commands=("SENTINEL private command",) if surface == "command" else (),
        acoustic_events=(
            (ReplayAcousticEvent.BARGE_IN,)
            if surface == "acoustic_event"
            else ()
        ),
        wall_seconds=0.01,
    )
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        aggregate.add(_corpus().cases[0], record, ordinal=0)


@pytest.mark.parametrize("forbidden", tuple(sorted(endpoint_eval._FORBIDDEN_REPORT_KEYS)))
def test_report_schema_rejects_private_surface_keys(forbidden: str) -> None:
    report = _valid_report()
    report["protocol"][forbidden] = _PRIVATE_REFERENCE
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._validate_report(report)


def test_private_report_is_mode_0600_and_no_overwrite(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    destination = tmp_path / "aggregate.json"
    digest = endpoint_eval._publish_private_report(destination, b"{}\n")
    assert digest == hashlib.sha256(b"{}\n").hexdigest()
    metadata = destination.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._publish_private_report(destination, b"{}\n")


@pytest.mark.parametrize("failure", ("writer_close", "published_reread"))
def test_private_report_failure_cleans_every_owned_file(
    tmp_path: Path,
    monkeypatch,
    failure: str,
) -> None:
    tmp_path.chmod(0o700)
    destination = tmp_path / "aggregate.json"
    if failure == "writer_close":
        real_close = endpoint_eval._close_checked
        calls = 0

        def fail_once(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("SENTINEL private close failure")
            real_close(descriptor)

        monkeypatch.setattr(endpoint_eval, "_close_checked", fail_once)
    else:
        real_read = endpoint_eval._read_descriptor_exact
        calls = 0

        def corrupt_second(descriptor: int, expected: int) -> bytes:
            nonlocal calls
            calls += 1
            observed = real_read(descriptor, expected)
            return observed if calls == 1 else b"x" * expected

        monkeypatch.setattr(
            endpoint_eval,
            "_read_descriptor_exact",
            corrupt_second,
        )
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError) as error:
        endpoint_eval._publish_private_report(destination, b'{"safe":true}\n')
    assert str(error.value) == ""
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_private_report_requires_absolute_path_and_private_parent(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o755)
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._publish_private_report(tmp_path / "report.json", b"{}\n")
    tmp_path.chmod(0o700)
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._publish_private_report(Path("report.json"), b"{}\n")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "mutation",
    (
        "wer_formula",
        "cer_negative",
        "exact_rows",
        "exact_rate",
        "hold_rows",
        "hold_ticks",
        "state_transition_total",
        "model_load",
        "wall_order",
        "delay_order",
        "abort_total",
        "terminal_total",
        "final_minimum",
        "provider",
        "threads",
        "artifact_count",
        "source_revision",
        "runtime_digest",
        "logical_rule3",
        "transition_marginal",
        "reset_transition_marginal",
        "comparison_missing",
        "reset_flag",
        "reset_zero_hold",
        "reset_reference_totals",
        "word_geometry",
        "exact_with_errors",
        "character_geometry",
        "reference_totals",
        "hold_zero_equivalence",
        "control_semantic_domain",
        "candidate_semantic_wait",
        "impossible_state_transition",
    ),
)
def test_report_numeric_and_binding_mutations_fail_closed(mutation: str) -> None:
    report = _valid_report()
    cell = report["cells"]["prosody_hold_only"]
    accuracy = cell["joined_selected_final_accuracy"]
    trace = cell["endpoint_trace"]
    if mutation == "wer_formula":
        accuracy["wer"] = float(accuracy["wer"]) + 0.1
    elif mutation == "cer_negative":
        accuracy["cer"] = -0.1
    elif mutation == "exact_rows":
        accuracy["exact_rows"] = 25
    elif mutation == "exact_rate":
        accuracy["exact_rate"] = 0.5
    elif mutation == "hold_rows":
        trace["hold_rows"] = 25
    elif mutation == "hold_ticks":
        trace["hold_ticks"] = trace["evaluations"] + 1
    elif mutation == "state_transition_total":
        trace["state_transitions"]["scored->scored"] += 1
    elif mutation == "model_load":
        cell["model_load_ms"] = -1.0
    elif mutation == "wall_order":
        cell["resources"]["row_wall_ms_p95"] = 0.0
        cell["resources"]["row_wall_ms_p50"] = 1.0
    elif mutation == "delay_order":
        cell["vad_speech_end_to_endpoint_ms"]["p95"] = 2_000.0
        cell["vad_speech_end_to_endpoint_ms"]["max"] = 1_000.0
    elif mutation == "abort_total":
        cell["transcript_aborts"]["total"] += 1
    elif mutation == "terminal_total":
        cell["transcript_aborts"]["terminal_events"] += 1
    elif mutation == "final_minimum":
        cell["turn_integrity"]["finals"] = 1
        cell["vad_speech_end_to_endpoint_ms"]["samples"] = 1
    elif mutation == "provider":
        cell["provider"] = "cuda"
    elif mutation == "threads":
        cell["asr_threads"] = 2
    elif mutation == "artifact_count":
        report["bindings"]["production"]["active_artifact_files"] = 0
    elif mutation == "source_revision":
        report["bindings"]["production"]["source_revision"] = "not-a-revision"
    elif mutation == "runtime_digest":
        report["bindings"]["production"]["runtime_identities_sha256"] = "0"
    elif mutation == "logical_rule3":
        report["protocol"]["logical_rule3_ms"] = 19_000
    elif mutation == "transition_marginal":
        transitions = report["comparison"]["prosody_hold_only_vs_acoustic"][
            "row_final_count_transitions"
        ]
        transitions["single->single"] -= 1
        transitions["zero->single"] += 1
    elif mutation == "reset_transition_marginal":
        transitions = report["comparison"][
            "prosody_reset_accumulate_vs_prosody_hold_only"
        ]["row_final_count_transitions"]
        transitions["single->single"] -= 1
        transitions["zero->single"] += 1
    elif mutation == "comparison_missing":
        report["comparison"].pop("prosody_reset_accumulate_vs_acoustic")
    elif mutation == "reset_flag":
        report["cells"]["prosody_reset_accumulate"][
            "semantic_hold_reset_enabled"
        ] = False
    elif mutation == "reset_zero_hold":
        reset_trace = report["cells"]["prosody_reset_accumulate"][
            "endpoint_trace"
        ]
        reset_trace["evaluations"] -= 1
        reset_trace["hold_rows"] = 0
        reset_trace["hold_ticks"] = 0
        reset_trace["state_counts"]["scored"] -= 1
        reset_trace["basis_counts"]["semantic_hold"] = 0
        reset_trace["state_transitions"]["scored->scored"] = 0
        reset_trace["basis_transitions"]["semantic_hold->acoustic"] = 0
    elif mutation == "reset_reference_totals":
        reset_accuracy = report["cells"]["prosody_reset_accumulate"][
            "joined_selected_final_accuracy"
        ]
        reset_accuracy["reference_words"] += 1
        reset_accuracy["hypothesis_words"] += 1
        reset_accuracy["wer"] = round(
            reset_accuracy["word_errors"] / reset_accuracy["reference_words"],
            4,
        )
    elif mutation == "word_geometry":
        accuracy["hypothesis_words"] += 1
    elif mutation == "exact_with_errors":
        accuracy["exact_rows"] = endpoint_eval.EXPECTED_CASES
        accuracy["exact_rate"] = 1.0
    elif mutation == "character_geometry":
        accuracy["hypothesis_characters"] = (
            accuracy["reference_characters"] + accuracy["character_errors"] + 1
        )
    elif mutation == "reference_totals":
        accuracy["reference_words"] += 1
        accuracy["hypothesis_words"] += 1
        accuracy["wer"] = round(
            accuracy["word_errors"] / accuracy["reference_words"],
            4,
        )
    elif mutation == "hold_zero_equivalence":
        trace["hold_rows"] = 0
    elif mutation == "control_semantic_domain":
        control_trace = report["cells"]["acoustic"]["endpoint_trace"]
        control_trace["basis_counts"]["acoustic"] -= 1
        control_trace["basis_counts"]["semantic_hold"] += 1
        control_trace["hold_ticks"] = 1
        control_trace["hold_rows"] = 1
    elif mutation == "candidate_semantic_wait":
        trace["basis_counts"]["acoustic"] -= 1
        trace["basis_counts"]["semantic_wait"] += 1
    else:
        state_transitions = trace["state_transitions"]
        positive = next(
            name for name, count in state_transitions.items() if count > 0
        )
        state_transitions[positive] -= 1
        state_transitions["detector_error->scored"] += 1
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._validate_report(report)


def _guard_args(tmp_path: Path) -> list[str]:
    return [
        "--corpus",
        "/private/corpus.json",
        "--config",
        "/private/config.json",
        "--local-config",
        "/private/config.local.json",
        "--smart-turn-model",
        "/private/model.onnx",
        "--report",
        str(tmp_path / "report.json"),
        "--watchdog-seconds",
        "30",
    ]


def _worker_success_bytes(
    report_sha256: str = "a" * 64,
    *,
    hold_only_hold_rows: int = 1,
    reset_accumulate_hold_rows: int = 1,
    reset_multi_row_repairs_vs_acoustic: int = 2,
    reset_multi_row_repairs_vs_hold_only: int = 1,
) -> bytes:
    return (
        json.dumps(
            {
                "ok": True,
                "quality_verdict": "diagnostic_only",
                "report_sha256": report_sha256,
                "rows": 24,
                "hold_only_hold_rows": hold_only_hold_rows,
                "reset_accumulate_hold_rows": reset_accumulate_hold_rows,
                "reset_multi_row_repairs_vs_acoustic": (
                    reset_multi_row_repairs_vs_acoustic
                ),
                "reset_multi_row_repairs_vs_hold_only": (
                    reset_multi_row_repairs_vs_hold_only
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def test_parser_requires_explicit_absolute_config_pair(tmp_path: Path) -> None:
    args = _guard_args(tmp_path)
    parsed = endpoint_eval._parser().parse_args(args)
    assert parsed.corpus.is_absolute()
    assert parsed.config.is_absolute()
    assert parsed.local_config.is_absolute()
    assert parsed.smart_turn_model.is_absolute()
    assert parsed.report.is_absolute()
    assert parsed.logical_turn_shadow is False
    assert endpoint_eval._parser().parse_args(
        [*args, "--logical-turn-shadow"]
    ).logical_turn_shadow is True
    for flag in (
        "--corpus",
        "--config",
        "--local-config",
        "--smart-turn-model",
        "--report",
    ):
        relative = list(args)
        relative[relative.index(flag) + 1] = "relative-input"
        with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
            endpoint_eval._parser().parse_args(relative)
    missing = args[:]
    index = missing.index("--local-config")
    del missing[index : index + 2]
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError):
        endpoint_eval._parser().parse_args(missing)


def test_worker_main_preflights_report_before_native_evaluation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    tmp_path.chmod(0o700)
    report = tmp_path / "report.json"
    report.write_text("owner evidence\n", encoding="utf-8")
    report.chmod(0o600)
    called = []
    monkeypatch.setattr(
        endpoint_eval,
        "evaluate_edacc_endpoint_integrity",
        lambda **_: called.append(True),
    )
    code = endpoint_eval.main(_guard_args(tmp_path))
    output = capsys.readouterr()
    assert code == 2
    assert called == []
    assert report.read_text(encoding="utf-8") == "owner evidence\n"
    assert json.loads(output.out) == dict(endpoint_eval._SAFE_ERROR)


def test_guarded_parent_success_is_bounded_and_validated(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.delenv(endpoint_eval._WORKER_ENV, raising=False)

    def run_worker(_argv, *, timeout_seconds):
        del timeout_seconds
        payload = endpoint_eval._canonical_bytes(_valid_report()) + b"\n"
        digest = endpoint_eval._publish_private_report(
            tmp_path / "report.json",
            payload,
        )
        return 0, _worker_success_bytes(digest)

    monkeypatch.setattr(
        endpoint_eval,
        "_run_guarded_worker",
        run_worker,
    )
    code = endpoint_eval.guarded_main(_guard_args(tmp_path))
    output = capsys.readouterr()
    assert code == 0
    receipt = json.loads(output.out)
    assert receipt["report_sha256"] == hashlib.sha256(
        (tmp_path / "report.json").read_bytes()
    ).hexdigest()
    assert output.err == ""


def test_guarded_parent_accepts_requested_shadow_report(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.delenv(endpoint_eval._WORKER_ENV, raising=False)

    def run_worker(raw_argv, *, timeout_seconds):
        del timeout_seconds
        assert "--logical-turn-shadow" in raw_argv
        payload = endpoint_eval._canonical_bytes(
            _valid_report(logical_turn_shadow=True)
        ) + b"\n"
        digest = endpoint_eval._publish_private_report(
            tmp_path / "report.json",
            payload,
        )
        return 0, _worker_success_bytes(digest)

    monkeypatch.setattr(endpoint_eval, "_run_guarded_worker", run_worker)
    code = endpoint_eval.guarded_main(
        [*_guard_args(tmp_path), "--logical-turn-shadow"]
    )
    output = capsys.readouterr()
    assert code == 0
    assert json.loads(output.out)["quality_verdict"] == "diagnostic_only"
    assert output.err == ""


@pytest.mark.parametrize(
    "failure",
    (
        "missing",
        "wrong_sha",
        "invalid_report",
        "wrong_hold_only_summary",
        "wrong_reset_hold_summary",
        "wrong_acoustic_repair_summary",
        "wrong_hold_only_repair_summary",
        "worker_report_mode_mismatch",
    ),
)
def test_guarded_parent_rejects_unverified_retained_report(
    failure: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.delenv(endpoint_eval._WORKER_ENV, raising=False)

    def run_worker(_argv, *, timeout_seconds):
        del timeout_seconds
        if failure == "missing":
            return 0, _worker_success_bytes()
        if failure == "invalid_report":
            payload = b'{"schema_version":1}\n'
        else:
            payload = endpoint_eval._canonical_bytes(
                _valid_report(
                    logical_turn_shadow=(
                        failure == "worker_report_mode_mismatch"
                    )
                )
            ) + b"\n"
        digest = endpoint_eval._publish_private_report(
            tmp_path / "report.json",
            payload,
        )
        return 0, _worker_success_bytes(
            "a" * 64 if failure == "wrong_sha" else digest,
            hold_only_hold_rows=(
                0 if failure == "wrong_hold_only_summary" else 1
            ),
            reset_accumulate_hold_rows=(
                0 if failure == "wrong_reset_hold_summary" else 1
            ),
            reset_multi_row_repairs_vs_acoustic=(
                0 if failure == "wrong_acoustic_repair_summary" else 2
            ),
            reset_multi_row_repairs_vs_hold_only=(
                0 if failure == "wrong_hold_only_repair_summary" else 1
            ),
        )

    monkeypatch.setattr(endpoint_eval, "_run_guarded_worker", run_worker)
    code = endpoint_eval.guarded_main(_guard_args(tmp_path))
    output = capsys.readouterr()
    assert code == 2
    assert json.loads(output.out) == dict(endpoint_eval._SAFE_ERROR)
    assert output.err == ""


def test_guarded_parent_timeout_is_detail_free_and_cleans_worker(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv(endpoint_eval._WORKER_ENV, raising=False)
    terminated = []

    class TimeoutProcess:
        pid = 12345
        stdout = io.BytesIO()

        @staticmethod
        def wait(timeout):
            raise subprocess.TimeoutExpired("SENTINEL-private-command", timeout)

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(endpoint_eval.subprocess, "Popen", lambda *_, **__: TimeoutProcess())
    monkeypatch.setattr(
        endpoint_eval,
        "_terminate_worker_process",
        lambda process: terminated.append(process),
    )
    code = endpoint_eval.guarded_main(_guard_args(tmp_path))
    output = capsys.readouterr()
    assert code == 2
    assert json.loads(output.out) == dict(endpoint_eval._SAFE_ERROR)
    assert "SENTINEL" not in output.out + output.err
    assert terminated


def test_guarded_worker_rejects_overflow_without_retaining_it(monkeypatch) -> None:
    class CompleteProcess:
        pid = 12345

        def __init__(self):
            self.stdout = io.BytesIO(b"S" * (endpoint_eval._MAX_RECEIPT_BYTES + 100))
            self.done = False

        def wait(self, timeout):
            del timeout
            self.done = True
            return 0

        def poll(self):
            return 0 if self.done else None

    monkeypatch.setattr(endpoint_eval.subprocess, "Popen", lambda *_, **__: CompleteProcess())
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError) as error:
        endpoint_eval._run_guarded_worker([], timeout_seconds=1)
    assert str(error.value) == ""


def test_evaluator_propagates_shadow_mode_through_all_cells_and_recheck(
    monkeypatch,
) -> None:
    corpus = _corpus()
    prepared = _prepared()
    cells = {
        "acoustic": _cell_result(
            corpus,
            candidate=False,
            buckets=("multi", "multi", *("single",) * 22),
            logical_turn_shadow=True,
        ),
        "prosody_hold_only": _cell_result(
            corpus,
            candidate=True,
            buckets=("multi", *("single",) * 23),
            logical_turn_shadow=True,
        ),
        "prosody_reset_accumulate": _cell_result(
            corpus,
            candidate=True,
            reset_accumulate=True,
            buckets=("single",) * 24,
            logical_turn_shadow=True,
        ),
    }
    cell_modes = []
    source_modes = []

    def run_cell(_corpus, _config, *, cell, logical_turn_shadow=False):
        cell_modes.append((cell, logical_turn_shadow))
        return cells[cell]

    def source_binding(*, logical_turn_shadow=False):
        source_modes.append(logical_turn_shadow)
        return "a" * 64

    monkeypatch.setattr(endpoint_eval, "prepare_production", lambda **_: prepared)
    monkeypatch.setattr(endpoint_eval, "load_production_corpus", lambda _: corpus)
    monkeypatch.setattr(endpoint_eval, "_snapshot_model", lambda path: _model(Path(path)))
    monkeypatch.setattr(endpoint_eval, "_base_config", lambda _: _base_config())
    monkeypatch.setattr(endpoint_eval, "_run_cell", run_cell)
    monkeypatch.setattr(endpoint_eval, "_source_binding", source_binding)
    monkeypatch.setattr(endpoint_eval, "verify_production_inputs", lambda *_, **__: None)

    report = endpoint_eval.evaluate_edacc_endpoint_integrity(
        corpus_path=Path("/private/corpus"),
        config_path=Path("/private/config"),
        local_config_path=Path("/private/local"),
        smart_turn_model=Path("/private/model"),
        logical_turn_shadow=True,
    )

    assert report["schema_version"] == endpoint_eval.SHADOW_SCHEMA_VERSION
    assert cell_modes == [(name, True) for name in endpoint_eval._CELL_NAMES]
    assert source_modes == [True, True]


@pytest.mark.parametrize("drift", ("model", "source", "config"))
def test_pre_post_binding_drift_fails_closed(monkeypatch, drift: str) -> None:
    corpus = _corpus()
    prepared = _prepared()
    control = _cell_result(
        corpus, candidate=False, buckets=("single",) * 24
    )
    candidate = _cell_result(
        corpus, candidate=True, buckets=("single",) * 24
    )
    reset = _cell_result(
        corpus,
        candidate=True,
        reset_accumulate=True,
        buckets=("single",) * 24,
    )
    monkeypatch.setattr(endpoint_eval, "prepare_production", lambda **_: prepared)
    monkeypatch.setattr(endpoint_eval, "load_production_corpus", lambda _: corpus)
    monkeypatch.setattr(endpoint_eval, "_base_config", lambda _: _base_config())
    monkeypatch.setattr(
        endpoint_eval,
        "_run_cell",
        lambda _corpus, _config, *, cell: {
            "acoustic": control,
            "prosody_hold_only": candidate,
            "prosody_reset_accumulate": reset,
        }[cell],
    )
    model_values = iter(
        (
            _model(Path("/private/model-a")),
            _model(
                Path("/private/model-b")
                if drift == "model"
                else Path("/private/model-a")
            ),
        )
    )
    monkeypatch.setattr(endpoint_eval, "_snapshot_model", lambda _: next(model_values))
    source_values = iter(("a" * 64, "b" * 64 if drift == "source" else "a" * 64))
    monkeypatch.setattr(endpoint_eval, "_source_binding", lambda: next(source_values))

    def verify(*_, **__):
        if drift == "config":
            raise ProductionInputError()

    monkeypatch.setattr(endpoint_eval, "verify_production_inputs", verify)
    with pytest.raises(endpoint_eval.EdaccEndpointIntegrityError) as error:
        endpoint_eval.evaluate_edacc_endpoint_integrity(
            corpus_path=Path("/private/corpus"),
            config_path=Path("/private/config"),
            local_config_path=Path("/private/local"),
            smart_turn_model=Path("/private/model"),
        )
    assert str(error.value) == ""


def test_cli_failure_is_detail_free(monkeypatch, capsys, tmp_path: Path) -> None:
    def fail(**_):
        raise RuntimeError(_PRIVATE_REFERENCE)

    monkeypatch.setattr(endpoint_eval, "evaluate_edacc_endpoint_integrity", fail)
    code = endpoint_eval.main(
        [
            "--corpus",
            "/private/SENTINEL-corpus.json",
            "--smart-turn-model",
            "/private/SENTINEL-model.onnx",
            "--config",
            "/private/config.json",
            "--local-config",
            "/private/config.local.json",
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    output = capsys.readouterr()
    assert code == 2
    assert json.loads(output.out) == dict(endpoint_eval._SAFE_ERROR)
    assert "SENTINEL" not in output.out + output.err
