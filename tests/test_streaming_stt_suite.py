from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from tools.streaming_stt.protocol import StreamConfig
from tools import streaming_stt_suite as suite


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _aggregate_report(
    worker_manifest: Path,
    corpus: Path,
    *,
    ok: bool = True,
    coverage_complete: bool | None = None,
    worker_digest: str | None = None,
    corpus_digest: str | None = None,
    evaluator_file_digest: str = "e" * 64,
    stratum_tags: tuple[str, ...] = (),
) -> dict[str, object]:
    complete = ok if coverage_complete is None else coverage_complete
    config: dict[str, object] = {}
    evidence: dict[str, object] = {"kind": "aggregate-test"}
    metrics: dict[str, object] = {
        "coverage_complete": complete,
        "evaluations": 1,
    }
    if stratum_tags:
        config["stratum_tags"] = list(stratum_tags)
        selection_sha256 = suite._canonical_digest(list(stratum_tags))
        evidence["stratification"] = {
            "kind": "explicit_corpus_case_tags",
            "aggregate_only": True,
            "overlapping": True,
            "tags": len(stratum_tags),
            "selection_sha256": selection_sha256,
        }
        metrics["strata"] = [
            {
                "tag": tag,
                "cases": 1,
                "metrics": {"coverage_complete": True, "evaluations": 1},
            }
            for tag in stratum_tags
        ]
    config["contract_sha256"] = suite._canonical_digest(config)
    return {
        "ok": ok,
        "evidence": evidence,
        "worker": {
            "manifest_sha256": worker_digest or _digest(worker_manifest.name),
        },
        "corpus": {
            "manifest_sha256": corpus_digest or _digest(corpus.name),
        },
        "config": config,
        "metrics": metrics,
        "worker_stderr": {"bytes": 0},
        "evaluator": {
            "schema_version": 1,
            "kind": "injected-test",
            "files": {"tools/streaming_stt_eval.py": evaluator_file_digest},
        },
    }


def test_cross_product_is_sequential_deduplicated_path_free_and_forwarded(
    tmp_path: Path,
) -> None:
    workers = [tmp_path / "w1.json", tmp_path / "w1.json", tmp_path / "w2.json"]
    corpora = [tmp_path / "c1.json", tmp_path / "c2.json", tmp_path / "c1.json"]
    calls: list[tuple[str, str, dict[str, object]]] = []
    active = 0
    maximum_active = 0

    def run(worker: Path, corpus: Path, **kwargs: object) -> dict[str, object]:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        calls.append((worker.name, corpus.name, kwargs))
        result = _aggregate_report(
            worker,
            corpus,
            stratum_tags=kwargs["stratum_tags"],
        )
        active -= 1
        return result

    report = suite.run_suite(
        workers,
        corpora,
        scratch_parent=tmp_path / "private-scratch",
        repeats=2,
        chunk_samples=1280,
        pace="realtime",
        partial_interval_ms=80,
        tail_padding_samples=16000,
        stratum_tags=("noisy", "clean", "clean"),
        run_benchmark_fn=run,
    )

    assert maximum_active == 1
    assert [(worker, corpus) for worker, corpus, _kwargs in calls] == [
        ("w1.json", "c1.json"),
        ("w1.json", "c2.json"),
        ("w2.json", "c1.json"),
        ("w2.json", "c2.json"),
    ]
    for _worker, _corpus, kwargs in calls:
        assert kwargs == {
            "scratch_parent": (tmp_path / "private-scratch").resolve(),
            "repeats": 2,
            "stream": None,
            "chunk_samples": 1280,
            "pace": "realtime",
            "partial_interval_ms": 80,
            "tail_padding_samples": 16000,
            "stratum_tags": ("clean", "noisy"),
        }
    assert report["ok"] is True
    assert report["complete"] is True
    assert report["controller"]["execution"] == "strictly_sequential_cross_product"
    assert report["controller"]["worker_manifests"] == 2
    assert report["controller"]["corpora"] == 2
    assert report["controller"]["runs"] == 4
    assert report["config"]["stratum_tags"] == ["clean", "noisy"]
    assert len(report["runs"]) == 4
    implementation = report["controller"]["evaluator_implementation_sha256"]
    for item in report["runs"]:
        assert set(item) == {"binding", "report"}
        assert len(item["binding"]["binding_sha256"]) == 64
        assert len(item["binding"]["report_sha256"]) == 64
        assert item["binding"]["evaluator_implementation_sha256"] == implementation
    encoded = json.dumps(report)
    for private_path in (*workers, *corpora, tmp_path / "private-scratch"):
        assert str(private_path.resolve()) not in encoded


@pytest.mark.parametrize(
    "failure",
    (
        "missing_metrics",
        "missing_evidence",
        "wrong_order",
        "zero_cases",
        "incomplete_coverage",
        "wrong_selection_hash",
        "wrong_config_hash",
    ),
)
def test_requested_strata_require_exact_bound_complete_nested_evidence(
    tmp_path: Path,
    failure: str,
) -> None:
    tags = ("clean", "noisy")
    report = _aggregate_report(
        tmp_path / "worker.json",
        tmp_path / "corpus.json",
        stratum_tags=tags,
    )
    if failure == "missing_metrics":
        del report["metrics"]["strata"]
    elif failure == "missing_evidence":
        del report["evidence"]["stratification"]
    elif failure == "wrong_order":
        report["metrics"]["strata"].reverse()
    elif failure == "zero_cases":
        report["metrics"]["strata"][0]["cases"] = 0
    elif failure == "incomplete_coverage":
        report["metrics"]["strata"][0]["metrics"]["coverage_complete"] = False
    elif failure == "wrong_selection_hash":
        report["evidence"]["stratification"]["selection_sha256"] = "f" * 64
    elif failure == "wrong_config_hash":
        report["config"]["contract_sha256"] = "f" * 64
    else:  # pragma: no cover - the parameter list is closed above
        raise AssertionError

    with pytest.raises(suite.SuiteError):
        suite._snapshot_nested_report(report, expected_stratum_tags=tags)


@pytest.mark.parametrize(
    "unexpected",
    ("config", "metrics", "evidence"),
)
def test_unrequested_strata_are_rejected_at_nested_report_boundary(
    tmp_path: Path,
    unexpected: str,
) -> None:
    report = _aggregate_report(
        tmp_path / "worker.json",
        tmp_path / "corpus.json",
    )
    if unexpected == "config":
        report["config"]["stratum_tags"] = []
        contract = dict(report["config"])
        del contract["contract_sha256"]
        report["config"]["contract_sha256"] = suite._canonical_digest(contract)
    elif unexpected == "metrics":
        report["metrics"]["strata"] = []
    elif unexpected == "evidence":
        report["evidence"]["stratification"] = {
            "kind": "explicit_corpus_case_tags",
            "aggregate_only": True,
            "overlapping": True,
            "tags": 0,
            "selection_sha256": suite._canonical_digest([]),
        }
    else:  # pragma: no cover - the parameter list is closed above
        raise AssertionError

    with pytest.raises(suite.SuiteError):
        suite._snapshot_nested_report(report, expected_stratum_tags=())


def test_per_corpus_strata_allow_clean_and_demand_without_missing_tags(
    tmp_path: Path,
) -> None:
    clean = tmp_path / "clean-corpus.json"
    demand = tmp_path / "demand-corpus.json"
    seen: list[tuple[str, tuple[str, ...]]] = []

    def run(
        worker: Path,
        corpus: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        tags = kwargs["stratum_tags"]
        assert isinstance(tags, tuple)
        seen.append((corpus.name, tags))
        return _aggregate_report(worker, corpus, stratum_tags=tags)

    report = suite.run_suite(
        [tmp_path / "worker.json"],
        [clean, demand],
        scratch_parent=tmp_path / "scratch",
        corpus_stratum_tags={
            clean: ("clean",),
            demand: ("demand-kitchen", "snr-10db"),
        },
        run_benchmark_fn=run,
    )

    assert seen == [
        ("clean-corpus.json", ("clean",)),
        ("demand-corpus.json", ("demand-kitchen", "snr-10db")),
    ]
    assert report["config"]["strata"]["mode"] == "per_corpus"
    selections = report["config"]["strata"]["corpora"]
    assert selections == [
        {
            "corpus_index": 0,
            "tags": ["clean"],
            "selection_sha256": suite._canonical_digest(["clean"]),
        },
        {
            "corpus_index": 1,
            "tags": ["demand-kitchen", "snr-10db"],
            "selection_sha256": suite._canonical_digest(
                ["demand-kitchen", "snr-10db"]
            ),
        },
    ]
    assert [
        item["binding"]["stratum_selection_sha256"]
        for item in report["runs"]
    ] == [item["selection_sha256"] for item in selections]
    encoded = json.dumps(report)
    assert str(clean) not in encoded
    assert str(demand) not in encoded


@pytest.mark.parametrize(
    "corpus_strata",
    [
        {"unknown-corpus": ("clean",)},
        {"corpus": ()},
    ],
)
def test_per_corpus_strata_reject_unknown_or_empty_selection_before_run(
    tmp_path: Path,
    corpus_strata: dict[str, tuple[str, ...]],
) -> None:
    called = False

    def run(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(suite.SuiteError):
        suite.run_suite(
            [tmp_path / "worker"],
            [tmp_path / "corpus"],
            scratch_parent=tmp_path / "scratch",
            corpus_stratum_tags=corpus_strata,
            run_benchmark_fn=run,
        )
    assert called is False


def test_global_and_per_corpus_strata_are_unambiguously_mutually_exclusive(
    tmp_path: Path,
) -> None:
    with pytest.raises(suite.SuiteError):
        suite.run_suite(
            [tmp_path / "worker"],
            [tmp_path / "corpus"],
            scratch_parent=tmp_path / "scratch",
            stratum_tags=("clean",),
            corpus_stratum_tags={tmp_path / "corpus": ("clean",)},
            run_benchmark_fn=lambda *_args, **_kwargs: pytest.fail(
                "runner must not start"
            ),
        )


def test_explicit_stream_is_forwarded_and_cannot_mix_with_overrides(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    selected = StreamConfig(1600, "burst", 200, 0)

    def run(worker: Path, corpus: Path, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return _aggregate_report(worker, corpus)

    report = suite.run_suite(
        [tmp_path / "worker.json"],
        [tmp_path / "corpus.json"],
        scratch_parent=tmp_path / "scratch",
        stream=selected,
        run_benchmark_fn=run,
    )

    assert report["config"]["stream"] == selected.as_dict()
    assert calls[0]["stream"] is selected
    assert calls[0]["chunk_samples"] is None

    with pytest.raises(suite.SuiteError):
        suite.run_suite(
            [tmp_path / "worker.json"],
            [tmp_path / "corpus.json"],
            scratch_parent=tmp_path / "scratch",
            stream=selected,
            pace="realtime",
            run_benchmark_fn=run,
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("workers", "corpora", "kwargs"),
    [
        ([], ["corpus"], {}),
        (["worker"], [], {}),
        ([f"worker-{index}" for index in range(9)], ["corpus"], {}),
        (["worker"], [f"corpus-{index}" for index in range(9)], {}),
        (["worker"], ["corpus"], {"repeats": True}),
        (["worker"], ["corpus"], {"repeats": 9}),
        (["worker"], ["corpus"], {"chunk_samples": 0}),
        (["worker"], ["corpus"], {"partial_interval_ms": 5001}),
        (["worker"], ["corpus"], {"tail_padding_samples": 48001}),
        (["worker"], ["corpus"], {"stratum_tags": ("Private/Path",)}),
    ],
)
def test_invalid_or_over_cap_input_fails_before_a_run(
    tmp_path: Path,
    workers: list[str],
    corpora: list[str],
    kwargs: dict[str, object],
) -> None:
    called = False

    def run(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(suite.SuiteError):
        suite.run_suite(
            workers,
            corpora,
            scratch_parent=tmp_path / "scratch",
            run_benchmark_fn=run,
            **kwargs,
        )
    assert called is False


def test_runner_failure_and_malformed_report_hide_private_details(
    tmp_path: Path,
) -> None:
    secret = str(tmp_path / "private-model-with-owner-name")

    def explode(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError(secret)

    with pytest.raises(suite.SuiteError) as caught:
        suite.run_suite(
            [tmp_path / "worker.json"],
            [tmp_path / "corpus.json"],
            scratch_parent=tmp_path / "scratch",
            run_benchmark_fn=explode,
        )
    assert secret not in str(caught.value)

    def leaks_path(worker: Path, corpus: Path, **_kwargs: object) -> dict[str, object]:
        report = _aggregate_report(worker, corpus)
        report["evidence"] = {"private_path": secret}
        return report

    with pytest.raises(suite.SuiteError) as caught:
        suite.run_suite(
            [tmp_path / "worker.json"],
            [tmp_path / "corpus.json"],
            scratch_parent=tmp_path / "scratch",
            run_benchmark_fn=leaks_path,
        )
    assert secret not in str(caught.value)


def test_green_nested_report_requires_complete_coverage(tmp_path: Path) -> None:
    def inconsistent(worker: Path, corpus: Path, **_kwargs: object) -> dict[str, object]:
        return _aggregate_report(
            worker,
            corpus,
            ok=True,
            coverage_complete=False,
        )

    with pytest.raises(suite.SuiteError):
        suite.run_suite(
            [tmp_path / "worker.json"],
            [tmp_path / "corpus.json"],
            scratch_parent=tmp_path / "scratch",
            run_benchmark_fn=inconsistent,
        )


def test_evaluator_implementation_is_bound_and_must_match_across_cells(
    tmp_path: Path,
) -> None:
    calls = 0
    checkpoint = tmp_path / "checkpoint.json"

    def drift(worker: Path, corpus: Path, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _aggregate_report(
            worker,
            corpus,
            evaluator_file_digest=("e" if calls == 1 else "f") * 64,
        )

    with pytest.raises(suite.SuiteError):
        suite.run_suite(
            [tmp_path / "worker.json"],
            [tmp_path / "c1.json", tmp_path / "c2.json"],
            scratch_parent=tmp_path / "scratch",
            checkpoint_path=checkpoint,
            run_benchmark_fn=drift,
        )

    assert calls == 2
    saved = json.loads(checkpoint.read_text())
    assert saved["ok"] is False
    assert saved["complete"] is False
    assert saved["controller"]["state"] == "failed"
    assert saved["controller"]["completed_runs"] == 1
    implementation = saved["controller"]["evaluator_implementation_sha256"]
    assert implementation == saved["runs"][0]["binding"][
        "evaluator_implementation_sha256"
    ]
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600


def test_later_cell_failure_preserves_private_atomic_red_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "private" / "suite.json"
    secret = "raw owner transcript that must not escape"
    calls = 0

    def fail_second(worker: Path, corpus: Path, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError(secret)
        return _aggregate_report(worker, corpus)

    with pytest.raises(suite.SuiteError) as caught:
        suite.run_suite(
            [tmp_path / "worker.json"],
            [tmp_path / "first.json", tmp_path / "second.json"],
            scratch_parent=tmp_path / "scratch",
            checkpoint_path=checkpoint,
            run_benchmark_fn=fail_second,
        )

    assert secret not in str(caught.value)
    saved_text = checkpoint.read_text()
    saved = json.loads(saved_text)
    assert saved["ok"] is False
    assert saved["complete"] is False
    assert saved["controller"]["state"] == "failed"
    assert saved["controller"]["planned_runs"] == 2
    assert saved["controller"]["completed_runs"] == 1
    assert len(saved["runs"]) == 1
    assert secret not in saved_text
    for private in (
        tmp_path / "worker.json",
        tmp_path / "first.json",
        tmp_path / "second.json",
        tmp_path / "scratch",
    ):
        assert str(private) not in saved_text
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600
    assert not list(checkpoint.parent.glob(".streaming-stt-suite-*"))


def test_first_cell_failure_replaces_stale_output_with_zero_run_red_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "suite.json"
    checkpoint.write_text('{"ok":true,"complete":true}')
    checkpoint.chmod(0o644)
    secret = "first-cell-private-error"

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError(secret)

    with pytest.raises(suite.SuiteError):
        suite.run_suite(
            [tmp_path / "worker.json"],
            [tmp_path / "corpus.json"],
            scratch_parent=tmp_path / "scratch",
            checkpoint_path=checkpoint,
            run_benchmark_fn=fail,
        )

    saved_text = checkpoint.read_text()
    saved = json.loads(saved_text)
    assert saved["ok"] is False
    assert saved["complete"] is False
    assert saved["controller"]["state"] == "failed"
    assert saved["controller"]["completed_runs"] == 0
    assert saved["runs"] == []
    assert secret not in saved_text
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600


def test_successful_checkpoint_is_replaced_by_complete_final_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "suite.json"

    report = suite.run_suite(
        [tmp_path / "worker.json"],
        [tmp_path / "corpus.json"],
        scratch_parent=tmp_path / "scratch",
        checkpoint_path=output,
        run_benchmark_fn=lambda worker, corpus, **_kwargs: _aggregate_report(
            worker, corpus
        ),
    )

    saved = json.loads(output.read_text())
    assert saved == report
    assert saved["ok"] is True
    assert saved["complete"] is True
    assert saved["controller"]["state"] == "complete"
    assert saved["controller"]["completed_runs"] == 1
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_digest_drift_across_cells_fails_closed(tmp_path: Path) -> None:
    calls = 0

    def run(worker: Path, corpus: Path, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _aggregate_report(
            worker,
            corpus,
            worker_digest=f"{calls:064x}",
        )

    with pytest.raises(suite.SuiteError):
        suite.run_suite(
            [tmp_path / "worker.json"],
            [tmp_path / "c1.json", tmp_path / "c2.json"],
            scratch_parent=tmp_path / "scratch",
            run_benchmark_fn=run,
        )
    assert calls == 2


def test_red_nested_report_keeps_running_but_makes_suite_red(tmp_path: Path) -> None:
    calls = 0

    def run(worker: Path, corpus: Path, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _aggregate_report(worker, corpus, ok=calls != 1)

    report = suite.run_suite(
        [tmp_path / "w1.json", tmp_path / "w2.json"],
        [tmp_path / "corpus.json"],
        scratch_parent=tmp_path / "scratch",
        run_benchmark_fn=run,
    )

    assert calls == 2
    assert report["ok"] is False
    assert [item["report"]["ok"] for item in report["runs"]] == [False, True]


def test_write_report_is_strict_json_and_mode_600(tmp_path: Path) -> None:
    output = tmp_path / "private" / "suite.json"
    output.parent.mkdir()
    output.write_text("old")
    output.chmod(0o644)
    payload = {"ok": True, "controller_sha256": "a" * 64}

    suite.write_report(output, payload)

    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text()) == payload
    assert not list(output.parent.glob(".streaming-stt-suite-*"))


def test_write_report_replaces_final_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "unrelated.json"
    target.write_text("unchanged")
    output = tmp_path / "suite.json"
    output.symlink_to(target)

    suite.write_report(output, {"ok": True})

    assert not output.is_symlink()
    assert json.loads(output.read_text()) == {"ok": True}
    assert target.read_text() == "unchanged"


def test_cli_per_corpus_strata_are_two_argument_pairs(tmp_path: Path) -> None:
    corpus = str(tmp_path / "demand corpus.json")
    args = suite._parser().parse_args(
        [
            "--worker-manifest",
            str(tmp_path / "worker.json"),
            "--corpus",
            corpus,
            "--corpus-stratum-tag",
            corpus,
            "demand-kitchen",
            "--corpus-stratum-tag",
            corpus,
            "snr-10db",
        ]
    )

    assert suite._cli_corpus_stratum_tags(args.corpus_stratum_tag) == {
        corpus: ["demand-kitchen", "snr-10db"]
    }


def test_cli_hides_exception_and_does_not_write_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = str(tmp_path / "secret-worker")
    output = tmp_path / "report.json"

    def explode(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError(secret)

    monkeypatch.setattr(suite, "run_suite", explode)
    result = suite.main(
        [
            "--worker-manifest",
            secret,
            "--corpus",
            str(tmp_path / "private-corpus"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out) == {
        "ok": False,
        "error": "streaming_stt_suite_failed",
    }
    assert secret not in captured.out
    assert secret not in captured.err
    assert not output.exists()
