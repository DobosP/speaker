from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import tools.stt_consensus_v2_eval as consensus_v2


def _hyp(source: str, family: str, text: str):
    return consensus_v2.AcousticHypothesis(source, family, text)


def test_private_text_is_hidden_from_diagnostic_reprs():
    sentinel = "SENTINEL_PRIVATE_TRANSCRIPT"
    hypothesis = _hyp("small", "whisper", sentinel)
    decision = consensus_v2.RepairDecision(
        sentinel,
        consensus_v2.RepairReason.EXACT_VOTE,
        True,
        True,
    )

    assert sentinel not in repr(hypothesis)
    assert sentinel not in repr(decision)


def test_exact_vote_requires_cross_family_agreement_and_returns_existing_rendering():
    hypotheses = (
        _hyp("small", "whisper", "Find, my vault!"),
        _hyp("small_en", "whisper", "find my vault"),
        _hyp("parakeet", "parakeet", "find my vault"),
    )

    accepted = consensus_v2.exact_acoustic_vote(
        baseline="find my bolt",
        hypotheses=hypotheses,
        source_ids=("small", "small_en", "parakeet"),
    )
    correlated_only = consensus_v2.exact_acoustic_vote(
        baseline="find my bolt",
        hypotheses=hypotheses[:2],
        source_ids=("small", "small_en"),
    )

    assert accepted.accepted is True
    assert accepted.changed is True
    assert accepted.chosen in {hypothesis.text for hypothesis in hypotheses}
    assert accepted.source_support == 3
    assert accepted.family_support == 2
    assert correlated_only.accepted is False
    assert correlated_only.reason is consensus_v2.RepairReason.NO_SUPPORT


def test_exact_vote_abstains_on_equal_strength_tie():
    hypotheses = (
        _hyp("small", "whisper", "alpha one"),
        _hyp("parakeet", "parakeet", "alpha one"),
        _hyp("offline", "sensevoice", "beta two"),
        _hyp("streaming", "zipformer", "beta two"),
    )

    decision = consensus_v2.exact_acoustic_vote(
        baseline="gamma three",
        hypotheses=hypotheses,
        source_ids=("small", "parakeet", "offline", "streaming"),
    )

    assert decision.accepted is False
    assert decision.reason is consensus_v2.RepairReason.TIE
    assert decision.chosen == "gamma three"


def test_exact_vote_cannot_manufacture_a_control():
    hypotheses = (
        _hyp("small", "whisper", "stop"),
        _hyp("parakeet", "parakeet", "Stop!"),
    )

    decision = consensus_v2.exact_acoustic_vote(
        baseline="stock",
        hypotheses=hypotheses,
        source_ids=("small", "parakeet"),
    )

    assert decision.reason is consensus_v2.RepairReason.CONTROL_GUARD
    assert decision.chosen == "stock"
    assert decision.changed is False


def test_span_repair_copies_only_independently_supported_nonconflicting_spans():
    hypotheses = (
        _hyp("small", "whisper", "find my vault tomorrow"),
        _hyp("parakeet", "parakeet", "find my vault tomorrow"),
        _hyp("base", "whisper", "find my bolt today"),
    )

    decision = consensus_v2.conservative_span_repair(
        baseline="find my bolt today",
        hypotheses=hypotheses,
        source_ids=("small", "parakeet", "base"),
    )

    assert decision.accepted is True
    assert decision.chosen == "find my vault tomorrow"
    assert decision.repaired_spans == 1
    assert decision.source_support == 2
    assert decision.family_support == 2
    assert decision.constructed is False


def test_span_repair_can_construct_from_supported_spans_without_new_tokens():
    hypotheses = (
        _hyp("parakeet", "parakeet", "search my vault on friday please now"),
        _hyp("small", "whisper", "open my vault on monday please"),
        _hyp("base", "whisper", "open my bolt on friday please"),
    )

    decision = consensus_v2.conservative_span_repair(
        baseline="open my bolt on monday please",
        hypotheses=hypotheses,
        source_ids=("parakeet", "small", "base"),
    )

    # Parakeet anchors both copied edits, each confirmed by a Whisper model.
    # Its unsupported search/open and now/please edits remain on the baseline.
    assert decision.accepted is True
    assert decision.chosen == "open my vault on friday please"
    assert decision.constructed is True
    source_tokens = {
        token
        for hypothesis in hypotheses
        for token in consensus_v2._exact_tokens(hypothesis.text)
    }
    assert set(consensus_v2._exact_tokens(decision.chosen)) <= source_tokens


def test_span_repair_rejects_conflicting_supported_spans():
    hypotheses = (
        _hyp("small", "whisper", "alpha delta"),
        _hyp("parakeet", "parakeet", "alpha delta"),
        _hyp("offline", "sensevoice", "alpha epsilon"),
        _hyp("streaming", "zipformer", "alpha epsilon"),
    )

    decision = consensus_v2.conservative_span_repair(
        baseline="alpha beta",
        hypotheses=hypotheses,
        source_ids=("small", "parakeet", "offline", "streaming"),
    )

    assert decision.accepted is False
    assert decision.reason is consensus_v2.RepairReason.CONFLICT
    assert decision.chosen == "alpha beta"


def test_span_repair_cannot_change_control_semantics():
    hypotheses = (
        _hyp("small", "whisper", "stop"),
        _hyp("parakeet", "parakeet", "stop"),
    )

    decision = consensus_v2.conservative_span_repair(
        baseline="stock",
        hypotheses=hypotheses,
        source_ids=("small", "parakeet"),
    )

    assert decision.reason is consensus_v2.RepairReason.CONTROL_GUARD
    assert decision.chosen == "stock"


def test_span_repair_rejects_support_from_only_one_model_family():
    hypotheses = (
        _hyp("small", "whisper", "find my vault"),
        _hyp("small_en", "whisper", "find my vault"),
    )

    decision = consensus_v2.conservative_span_repair(
        baseline="find my bolt",
        hypotheses=hypotheses,
        source_ids=("small", "small_en"),
    )

    assert decision.accepted is False
    assert decision.reason is consensus_v2.RepairReason.NO_SUPPORT


def test_parakeet_decode_uses_exact_local_nemo_contract_and_aggregate_batch(tmp_path):
    for name in consensus_v2._PARAKEET_FILES:
        (tmp_path / name).write_bytes(b"model")
    outputs = iter(("private first", "private first", "private second", "private second"))
    factory_calls = []

    class _Stream:
        def __init__(self):
            self.result = SimpleNamespace(text="")

        def accept_waveform(self, rate, samples):
            assert rate == 16_000
            assert samples.dtype == np.float32

    class _Recognizer:
        def create_stream(self):
            return _Stream()

        def decode_stream(self, stream):
            stream.result.text = next(outputs)

    def factory(model_dir, *, threads, provider):
        factory_calls.append((model_dir, threads, provider))
        return _Recognizer()

    corpus = (
        SimpleNamespace(samples=np.zeros(80, np.float32), sample_rate=16_000),
        SimpleNamespace(samples=np.zeros(80, np.float32), sample_rate=16_000),
    )
    batch = consensus_v2._decode_parakeet(
        tmp_path,
        corpus,
        repeats=2,
        threads=3,
        recognizer_factory=factory,
    )

    assert factory_calls == [(tmp_path.resolve(), 3, "cpu")]
    assert batch.repeat_disagreements == 0
    assert batch.empty_outputs == 0
    assert "private" not in json.dumps(batch.as_dict())


def test_parakeet_factory_wires_nemo_tdt_feature_contract(monkeypatch, tmp_path):
    for name in consensus_v2._PARAKEET_FILES:
        (tmp_path / name).write_bytes(b"model")
    calls = []

    class _OfflineRecognizer:
        @staticmethod
        def from_transducer(**kwargs):
            calls.append(kwargs)
            return object()

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=_OfflineRecognizer),
    )

    consensus_v2._default_parakeet_factory(
        tmp_path,
        threads=5,
        provider="cpu",
    )

    assert len(calls) == 1
    assert calls[0]["model_type"] == "nemo_transducer"
    assert calls[0]["feature_dim"] == 80
    assert calls[0]["decoding_method"] == "greedy_search"
    assert calls[0]["num_threads"] == 5
    assert calls[0]["provider"] == "cpu"
    assert calls[0]["encoder"] == str(tmp_path / "encoder.int8.onnx")


def test_aggregate_evaluation_beats_current_only_with_zero_losses_and_no_controls():
    cases = (
        consensus_v2._PrivateCase(
            "find my vault",
            "find my bolt",
            (
                _hyp("small", "whisper", "find my vault"),
                _hyp("parakeet", "parakeet", "find my vault"),
            ),
        ),
        consensus_v2._PrivateCase(
            "ordinary sentence",
            "ordinary sentence",
            (
                _hyp("small", "whisper", "ordinary sentence"),
                _hyp("parakeet", "parakeet", "ordinary sentence"),
            ),
        ),
    )
    variants = {
        "independent": lambda case: consensus_v2.exact_acoustic_vote(
            baseline=case.baseline,
            hypotheses=case.hypotheses,
            source_ids=("small", "parakeet"),
        )
    }

    report = consensus_v2._evaluate_variants(cases, variants)
    result = report["variants"]["independent"]

    assert report["current_exact_small_consensus"]["accuracy"]["word_errors"] == 1
    assert result["accuracy"]["word_errors"] == 0
    assert result["comparison"]["losses"] == 0
    assert result["comparison"]["strictly_safe"] is True
    serialized = json.dumps(report)
    assert "find my" not in serialized
    assert "ordinary sentence" not in serialized


def test_aggregate_evaluation_rejects_an_improvement_with_any_clip_loss():
    cases = (
        consensus_v2._PrivateCase(
            "alpha correct",
            "alpha wrong",
            (
                _hyp("small", "whisper", "alpha correct"),
                _hyp("parakeet", "parakeet", "alpha correct"),
            ),
        ),
        consensus_v2._PrivateCase(
            "beta correct",
            "beta correct",
            (
                _hyp("small", "whisper", "beta badly wrong"),
                _hyp("parakeet", "parakeet", "beta badly wrong"),
            ),
        ),
    )
    variants = {
        "mixed": lambda case: consensus_v2.exact_acoustic_vote(
            baseline=case.baseline,
            hypotheses=case.hypotheses,
            source_ids=("small", "parakeet"),
        )
    }

    result = consensus_v2._evaluate_variants(cases, variants)["variants"]["mixed"]

    assert result["comparison"]["losses"] == 1
    assert result["comparison"]["strictly_safe"] is False


def test_cli_failure_never_prints_private_exception(monkeypatch, capsys):
    monkeypatch.setattr(
        consensus_v2,
        "_load_corpus",
        lambda _manifest: (_ for _ in ()).throw(
            RuntimeError("SENTINEL_PRIVATE_TRANSCRIPT_AND_PATH")
        ),
    )

    assert consensus_v2.main(["--decode-repeats", "2"]) == 2
    output = capsys.readouterr().out
    assert "SENTINEL" not in output
    assert json.loads(output) == consensus_v2._SAFE_ERROR


def test_contract_digest_binds_local_acoustic_and_control_rules():
    first = consensus_v2._contract_digest(3, "cpu")
    second = consensus_v2._contract_digest(3, "cpu")
    changed = consensus_v2._contract_digest(2, "cpu")
    changed_threads = consensus_v2._contract_digest(3, "cpu", 2)

    assert first == second
    assert first != changed
    assert first != changed_threads
    assert len(first) == 64
