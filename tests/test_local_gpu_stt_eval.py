from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import tools.local_gpu_stt_eval as gpu_eval


def _result(text: str, confidence: float = 0.8):
    return SimpleNamespace(
        text=text,
        confidence=confidence,
        avg_logprob=-0.25,
        no_speech_probability=0.05,
    )


def test_decode_batch_repeats_identical_pcm_and_reports_only_aggregate_metrics(
    tmp_path,
):
    private_first = "SENTINEL_PRIVATE_FIRST"
    private_second = "SENTINEL_PRIVATE_SECOND"
    outputs = iter(
        (
            _result(private_first),
            _result(private_first),
            _result(private_second),
            _result("different private words"),
        )
    )

    class _Recognizer:
        def transcribe(self, _samples, _sample_rate):
            return next(outputs)

    factory_calls = []

    def factory(model_path):
        factory_calls.append(model_path)
        return _Recognizer()

    corpus = (
        SimpleNamespace(samples=np.zeros(10, dtype=np.float32), sample_rate=16000),
        SimpleNamespace(samples=np.zeros(20, dtype=np.float32), sample_rate=16000),
    )
    batch = gpu_eval._decode_model(
        tmp_path,
        corpus,
        repeats=2,
        recognizer_factory=factory,
    )

    assert factory_calls == [tmp_path]
    assert batch.texts == (private_first, private_second)
    assert batch.repeat_disagreements == 1
    assert batch.empty_outputs == 0
    assert batch.confidence_min == 0.8
    encoded = json.dumps(batch.as_dict())
    assert private_first not in encoded
    assert private_second not in encoded
    assert "different private words" not in encoded


def test_cases_keep_private_rows_in_memory_and_bind_all_existing_sources():
    private_reference = "SENTINEL_PRIVATE_REFERENCE"
    row = gpu_eval._PrivateSourceRow(
        private_reference,
        "production words",
        "streaming words",
        "offline words",
    )

    cases = gpu_eval._cases([row], ["small words"], ["turbo words"])

    assert len(cases) == 1
    assert [source.source_id for source in cases[0].sources] == [
        "production",
        "streaming",
        "offline",
        "gpu_small",
        "gpu_turbo",
    ]
    assert private_reference not in repr(cases)
    assert "production words" not in repr(cases)


def test_local_snapshot_accepts_existing_directory_without_hub_access(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("hub should not be consulted")
            )
        ),
    )

    assert gpu_eval._local_snapshot(str(tmp_path)) == tmp_path.resolve()


def test_local_snapshot_resolves_exact_cached_revision(tmp_path, monkeypatch):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=download),
    )

    resolved = gpu_eval._local_snapshot(
        "org/model",
        revision="a" * 40,
    )

    assert resolved == tmp_path.resolve()
    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "a" * 40,
            "local_files_only": True,
        }
    ]


def test_local_snapshot_rejects_mutable_repo_revision():
    with pytest.raises(gpu_eval.EvaluationPrerequisiteError):
        gpu_eval._local_snapshot("org/model", revision="main")


def test_machine_bakeoff_baseline_always_disables_existing_verifier(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "device": "desktop",
                "device_profiles": {},
                "sherpa": {
                    "asr_encoder": "/private/streaming.onnx",
                    "asr_final_verifier_backend": "faster_whisper",
                    "asr_final_verifier_model": "/private/verifier",
                },
            }
        ),
        encoding="utf-8",
    )

    baseline, effective, root = gpu_eval._load_machine_config(tmp_path)

    assert root == tmp_path.resolve()
    assert baseline.asr_final_verifier_backend == ""
    assert baseline.asr_final_verifier_model == ""
    assert effective.asr_final_verifier_backend == "faster_whisper"
    assert effective.asr_final_verifier_model == "/private/verifier"


def test_cli_failure_never_prints_private_exception_detail(monkeypatch, capsys):
    monkeypatch.setattr(
        gpu_eval,
        "_load_corpus",
        lambda _path: (_ for _ in ()).throw(
            RuntimeError("SENTINEL_PRIVATE_TRANSCRIPT_AND_PATH")
        ),
    )

    assert gpu_eval.main(["--decode-repeats", "2"]) == 2
    output = capsys.readouterr().out
    assert "SENTINEL" not in output
    assert json.loads(output) == gpu_eval._SAFE_ERROR


def test_selector_variants_make_llm_choice_only_and_consensus_explicit():
    variants = gpu_eval._selector_variants((), repeats=2)

    assert set(variants) == {
        "gpu_small_direct",
        "gpu_turbo_direct",
        "exact_consensus_small",
        "exact_consensus_turbo",
    }


def test_decode_and_choice_contracts_are_digest_bound_without_transcripts():
    digests = gpu_eval._contract_digests(decode_repeats=3, llm_repeats=2)

    assert set(digests) == {"gpu_decode", "llm_choice"}
    assert all(len(value) == 64 for value in digests.values())
    assert "transcript" not in json.dumps(digests)
