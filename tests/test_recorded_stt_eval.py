from __future__ import annotations

import json
import stat
import wave
from dataclasses import dataclass

import pytest

import tools.recorded_stt_eval as stt_eval


def _totals(pairs, *, keywords=(), verifier_outcomes=None):
    measured = stt_eval._measure(pairs, keywords=keywords)
    return stt_eval.EvaluationTotals(
        streaming=measured,
        offline=measured,
        selected=measured,
        clips=len(pairs),
        decisions=len(pairs),
        offline_outcomes={"decoded": len(pairs)},
        verifier_outcomes=verifier_outcomes or {},
    )


def test_aggregate_accuracy_counts_micro_wer_cer_and_keyword_recall():
    totals = stt_eval._measure(
        [
            ("alpha beta", "alpha gamma"),
            ("find vault now", "find vault now"),
        ],
        keywords=("vault",),
    )

    assert totals.clips == 2
    assert totals.nonempty == 2
    assert totals.exact == 1
    assert totals.substitutions == 1
    assert totals.insertions == 0
    assert totals.deletions == 0
    assert totals.ref_words == 5
    assert totals.wer == 0.2
    assert 0.0 < totals.cer < 1.0
    assert totals.keyword_attempts == 1
    assert totals.keyword_hits == 1


def test_candidate_requires_measured_improvement_without_regression():
    references = ["alpha beta", "gamma delta"]
    baseline_rows = list(zip(references, ["alpha wrong", "bad delta"]))
    candidate_rows = list(zip(references, ["alpha beta", "bad delta"]))
    comparison = stt_eval.compare_candidate(
        _totals(baseline_rows),
        _totals(candidate_rows),
        [stt_eval._pair_errors(*pair) for pair in baseline_rows],
        [stt_eval._pair_errors(*pair) for pair in candidate_rows],
    )

    assert comparison.promotable is True
    assert (comparison.wins, comparison.ties, comparison.losses) == (1, 1, 0)

    regressed_rows = list(zip(references, ["alpha beta", "totally wrong"]))
    regressed = stt_eval.compare_candidate(
        _totals(baseline_rows),
        _totals(regressed_rows),
        [stt_eval._pair_errors(*pair) for pair in baseline_rows],
        [stt_eval._pair_errors(*pair) for pair in regressed_rows],
    )
    assert regressed.promotable is False
    assert regressed.losses >= 1


def test_candidate_counts_per_clip_cer_losses_when_wer_ties():
    references = ["abcdefghij", "cat", "dog"]
    baseline_rows = list(zip(references, ["xxxxxxxxxx", "bat", "fog"]))
    candidate_rows = list(zip(references, ["abcdefghik", "zzzz", "zzzz"]))

    comparison = stt_eval.compare_candidate(
        _totals(baseline_rows),
        _totals(candidate_rows),
        [stt_eval._pair_errors(*pair) for pair in baseline_rows],
        [stt_eval._pair_errors(*pair) for pair in candidate_rows],
    )

    assert candidate_rows and _totals(candidate_rows).selected.char_edits < _totals(
        baseline_rows
    ).selected.char_edits
    assert (comparison.wins, comparison.losses) == (1, 2)
    assert comparison.promotable is False


def test_private_report_is_mode_600_and_aggregate_only(tmp_path):
    report = tmp_path / "report.json"
    payload = {"ok": True, "selected": {"wer": 0.0}}

    stt_eval._write_report(report, payload)

    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert json.loads(report.read_text(encoding="utf-8")) == payload


def test_private_report_replaces_symlink_without_touching_target(tmp_path):
    target = tmp_path / "target.txt"
    report = tmp_path / "report.json"
    target.write_text("keep", encoding="utf-8")
    report.symlink_to(target)

    stt_eval._write_report(report, {"ok": True})

    assert target.read_text(encoding="utf-8") == "keep"
    assert report.is_file() and not report.is_symlink()
    assert stat.S_IMODE(report.stat().st_mode) == 0o600


def test_artifact_digest_binds_bias_dictionary_contents(tmp_path):
    @dataclass(frozen=True)
    class _Config:
        asr_encoder: str
        asr_final_hr_dict_dir: str

    model = tmp_path / "model.onnx"
    dictionary = tmp_path / "bias"
    dictionary.mkdir()
    lexicon = dictionary / "lexicon.txt"
    model.write_bytes(b"model")
    lexicon.write_text("first", encoding="utf-8")
    config = _Config(str(model), str(dictionary))

    first = stt_eval._model_digest(config)
    lexicon.write_text("second", encoding="utf-8")
    second = stt_eval._model_digest(config)

    assert first != second


def test_artifact_digest_binds_final_verifier_snapshot_contents(tmp_path):
    @dataclass(frozen=True)
    class _Config:
        asr_encoder: str
        asr_final_verifier_model: str

    model = tmp_path / "streaming.onnx"
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    weights = verifier / "model.bin"
    model.write_bytes(b"streaming")
    weights.write_bytes(b"first")
    config = _Config(str(model), str(verifier))

    first = stt_eval._model_digest(config)
    weights.write_bytes(b"second")
    second = stt_eval._model_digest(config)

    assert first != second


def test_artifact_digest_binds_nemo_joiner_contents(tmp_path):
    @dataclass(frozen=True)
    class _Config:
        asr_encoder: str
        asr_final_joiner: str

    model = tmp_path / "streaming.onnx"
    joiner = tmp_path / "joiner.onnx"
    model.write_bytes(b"streaming")
    joiner.write_bytes(b"first")
    config = _Config(str(model), str(joiner))

    first = stt_eval._model_digest(config)
    joiner.write_bytes(b"second")
    second = stt_eval._model_digest(config)

    assert first != second


def test_artifact_digest_fails_when_configured_resource_is_missing(tmp_path):
    @dataclass(frozen=True)
    class _Config:
        asr_encoder: str
        asr_final_hr_lexicon: str

    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    config = _Config(str(model), str(tmp_path / "missing.txt"))

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._model_digest(config)


def test_artifact_digest_ignores_stale_disabled_verifier_path(tmp_path):
    @dataclass(frozen=True)
    class _Config:
        asr_encoder: str
        asr_final_verifier_backend: str
        asr_final_verifier_model: str

    model = tmp_path / "streaming.onnx"
    model.write_bytes(b"model")
    stale = str(tmp_path / "missing-verifier")

    disabled = _Config(str(model), "", stale)
    clean = _Config(str(model), "", "")

    assert stt_eval._model_digest(disabled) == stt_eval._model_digest(clean)


@pytest.mark.parametrize(
    "outcome",
    [
        "consensus",
        "empty",
        "tie",
        "no_quorum",
        "control_guard",
        "attested_control",
        "empty_veto",
        "empty_streaming_guard",
    ],
)
def test_enabled_verifier_accepts_completed_safe_fallback_outcomes(outcome):
    config = type(
        "Config",
        (),
        {"asr_final_verifier_backend": "faster_whisper"},
    )()
    totals = _totals([("alpha", "alpha")], verifier_outcomes={outcome: 1})

    assert stt_eval._enabled_verifier_evaluation_ok(config, totals)


@pytest.mark.parametrize(
    "outcomes",
    [
        {},
        {"skipped": 1},
        {"unavailable": 1},
        {"error": 1},
        {"consensus": 1, "error": 1},
    ],
)
def test_enabled_verifier_rejects_errors_or_zero_completed_decodes(outcomes):
    config = type(
        "Config",
        (),
        {"asr_final_verifier_backend": "faster_whisper"},
    )()
    totals = _totals([("alpha", "alpha")], verifier_outcomes=outcomes)

    assert not stt_eval._enabled_verifier_evaluation_ok(config, totals)


def test_disabled_verifier_does_not_require_verifier_outcomes():
    config = type("Config", (), {"asr_final_verifier_backend": ""})()
    totals = _totals([("alpha", "alpha")])

    assert stt_eval._enabled_verifier_evaluation_ok(config, totals)


@pytest.mark.parametrize("speech_sec", [True, float("inf"), 1.1])
def test_corpus_rejects_impossible_attested_speech_duration(tmp_path, speech_sec):
    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 16000)
    digest = stt_eval._sha256_file(clip)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clip_dir": str(tmp_path),
                "clips": [
                    {
                        "id": "clip",
                        "expected_text": "private reference",
                        "sha256": digest,
                        "speech_sec": speech_sec,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(stt_eval.EvaluationPrerequisiteError):
        stt_eval._load_corpus(manifest)


def test_cli_never_prints_private_rows_or_config(capsys, monkeypatch):
    private_reference = "SENTINEL_PRIVATE_REFERENCE"
    private_hypothesis = "SENTINEL_PRIVATE_HYPOTHESIS"

    @dataclass(frozen=True)
    class _Config:
        value: int = 1

    item = stt_eval._CorpusItem(private_reference, object(), 16000, None)
    measured = stt_eval._measure([(private_reference, private_hypothesis)])
    totals = stt_eval.EvaluationTotals(
        streaming=measured,
        offline=measured,
        selected=measured,
        clips=1,
        decisions=1,
        offline_outcomes={"decoded": 1},
    )
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: ([item], "c" * 64))
    monkeypatch.setattr(stt_eval, "_load_config", lambda: _Config())
    monkeypatch.setattr(
        stt_eval,
        "_evaluate",
        lambda _config, _corpus, _keywords: (
            totals,
            [stt_eval._pair_errors(private_reference, private_hypothesis)],
        ),
    )
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main([]) == 0
    output = capsys.readouterr().out
    assert private_reference not in output
    assert private_hypothesis not in output
    payload = json.loads(output)
    assert payload["baseline"]["selected"]["word_errors"] > 0


def test_cli_fails_closed_when_enabled_baseline_verifier_errors(
    capsys,
    monkeypatch,
):
    @dataclass(frozen=True)
    class _Config:
        asr_final_verifier_backend: str = "faster_whisper"

    item = stt_eval._CorpusItem("reference", object(), 16000, None)
    totals = _totals(
        [("reference", "reference")],
        verifier_outcomes={"error": 1},
    )
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: ([item], "c" * 64))
    monkeypatch.setattr(stt_eval, "_load_config", _Config)
    monkeypatch.setattr(
        stt_eval,
        "_evaluate",
        lambda _config, _corpus, _keywords: (totals, [(0, 0)]),
    )
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main([]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["baseline"]["verifier_outcomes"] == {"error": 1}


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ({"decoded": 1}, True),
        ({"empty": 1}, True),
        ({"error": 1, "decoded": 1}, False),
        ({"unavailable": 1}, False),
        ({"skipped": 1}, False),
    ],
)
def test_selected_offline_evaluation_requires_completed_error_free_decode(
    outcomes,
    expected,
):
    @dataclass(frozen=True)
    class _Config:
        asr_final_backend: str = "nemo_transducer"

    totals = _totals([("reference", "reference")])
    totals = stt_eval.EvaluationTotals(
        streaming=totals.streaming,
        offline=totals.offline,
        selected=totals.selected,
        clips=totals.clips,
        decisions=totals.decisions,
        offline_outcomes=outcomes,
        verifier_outcomes=totals.verifier_outcomes,
    )
    assert stt_eval._enabled_offline_evaluation_ok(_Config(), totals) is expected


def test_cli_fails_closed_when_selected_offline_model_never_decodes(
    capsys,
    monkeypatch,
):
    @dataclass(frozen=True)
    class _Config:
        asr_final_backend: str = "nemo_transducer"

    item = stt_eval._CorpusItem("reference", object(), 16000, None)
    measured = stt_eval._measure([("reference", "reference")])
    totals = stt_eval.EvaluationTotals(
        streaming=measured,
        offline=measured,
        selected=measured,
        clips=1,
        decisions=1,
        offline_outcomes={"unavailable": 1},
    )
    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: ([item], "c" * 64))
    monkeypatch.setattr(stt_eval, "_load_config", _Config)
    monkeypatch.setattr(
        stt_eval,
        "_evaluate",
        lambda _config, _corpus, _keywords: (totals, [(0, 0)]),
    )
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main([]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["baseline"]["offline_outcomes"] == {"unavailable": 1}


def test_cli_rejects_improved_candidate_when_enabled_verifier_errors(
    capsys,
    monkeypatch,
):
    @dataclass(frozen=True)
    class _Config:
        asr_final_verifier_backend: str = ""

    item = stt_eval._CorpusItem("alpha beta", object(), 16000, None)
    baseline = _totals([("alpha beta", "alpha wrong")])
    candidate = _totals(
        [("alpha beta", "alpha beta")],
        verifier_outcomes={"error": 1},
    )

    def evaluate(config, _corpus, _keywords):
        if config.asr_final_verifier_backend:
            return candidate, [(0, 0)]
        return baseline, [(1, 5)]

    monkeypatch.setattr(stt_eval, "_load_corpus", lambda _path: ([item], "c" * 64))
    monkeypatch.setattr(stt_eval, "_load_config", _Config)
    monkeypatch.setattr(stt_eval, "_evaluate", evaluate)
    monkeypatch.setattr(stt_eval, "_config_digest", lambda _config: "d" * 64)
    monkeypatch.setattr(stt_eval, "_model_digest", lambda _config: "e" * 64)

    assert stt_eval.main(
        ["--set", "asr_final_verifier_backend=faster_whisper"]
    ) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate"]["selected"]["word_errors"] == 0
    assert payload["comparison"]["wins"] == 1
    assert payload["comparison"]["promotable"] is False
    assert payload["ok"] is False


def test_cli_prerequisite_failure_is_detail_free(capsys, monkeypatch):
    monkeypatch.setattr(
        stt_eval,
        "_load_corpus",
        lambda _path: (_ for _ in ()).throw(
            stt_eval.EvaluationPrerequisiteError("SENTINEL_PRIVATE_PATH")
        ),
    )

    assert stt_eval.main([]) == 2
    output = capsys.readouterr().out
    assert "SENTINEL_PRIVATE_PATH" not in output
    assert json.loads(output) == stt_eval._SAFE_ERROR


def test_cli_unexpected_failure_is_also_detail_free(capsys, monkeypatch):
    monkeypatch.setattr(
        stt_eval,
        "_load_corpus",
        lambda _path: (_ for _ in ()).throw(
            RuntimeError("SENTINEL_PRIVATE_TRANSCRIPT")
        ),
    )

    assert stt_eval.main([]) == 2
    output = capsys.readouterr().out
    assert "SENTINEL_PRIVATE_TRANSCRIPT" not in output
    assert json.loads(output) == stt_eval._SAFE_ERROR
