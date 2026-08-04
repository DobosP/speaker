from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_public_conversation_fixture import _publish_set
from tests.test_streaming_stt_suite import _aggregate_report
from tools import public_conversation_fixture as fixture
from tools import public_conversation_qualification as qualification
from tools import streaming_stt_eval
from tools import streaming_stt_suite
from tools.streaming_stt.corpus import load_corpus

_FAKE_STRATA = (
    (tuple(f"task_{index:02d}" for index in range(8)), (3,) * 8),
    (("clean", "disfluent", "overlap_adjacent"), (8, 8, 8)),
    (("close", "far", "isolated", "turn_transition"), (12, 12, 16, 8)),
    (
        ("d0_2_6s", "d1_6_10s", "d2_10_15s", "d3_15_25s"),
        (6, 6, 6, 6),
    ),
)


def _worker_digest(path: Path | str) -> str:
    selected = Path(path).resolve(strict=False)
    return hashlib.sha256(str(selected).encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _fake_worker_manifest_loader(monkeypatch):
    monkeypatch.setattr(
        qualification,
        "load_worker_manifest",
        lambda path: SimpleNamespace(digest=_worker_digest(path)),
    )


def _fake_result(
    workers: tuple[Path, ...],
    corpora: tuple[Path, ...],
) -> dict[str, object]:
    fixture_set = fixture.validate_fixture_set(corpora)
    digests = [load_corpus(path).digest for path in corpora]
    worker_digests = [_worker_digest(path) for path in workers]
    config_rows = [
        {
            "corpus_index": index,
            "tags": list(tags),
            "selection_sha256": qualification._canonical_sha256(list(tags)),
        }
        for index, (tags, _counts) in enumerate(_FAKE_STRATA)
    ]
    strata_config = {
        "mode": "per_corpus",
        "corpora": config_rows,
        "selection_sha256": qualification._canonical_sha256(config_rows),
    }
    config: dict[str, object] = {
        "repeats": 3,
        "strata": strata_config,
        "overrides": {
            "chunk_samples": None,
            "pace": None,
            "partial_interval_ms": None,
            "tail_padding_samples": None,
        },
    }
    config["contract_sha256"] = qualification._canonical_sha256(config)
    evaluator_digest: str | None = None
    runs: list[dict[str, object]] = []
    for worker_index in range(2):
        for corpus_index, digest in enumerate(digests):
            report = _aggregate_report(
                workers[worker_index],
                corpora[corpus_index],
                worker_digest=worker_digests[worker_index],
                corpus_digest=digest,
                stratum_tags=_FAKE_STRATA[corpus_index][0],
            )
            metrics = report["metrics"]
            for row, count in zip(
                metrics["strata"],
                _FAKE_STRATA[corpus_index][1],
                strict=True,
            ):
                row["cases"] = count
                row["metrics"]["evaluations"] = count
            metrics["evaluations"] = 24
            current_evaluator = streaming_stt_suite._evaluator_implementation_sha256(
                report["evaluator"]
            )
            if evaluator_digest is None:
                evaluator_digest = current_evaluator
            else:
                assert evaluator_digest == current_evaluator
            binding: dict[str, object] = {
                "worker_index": worker_index,
                "corpus_index": corpus_index,
                "worker_manifest_sha256": worker_digests[worker_index],
                "corpus_manifest_sha256": digest,
                "stratum_selection_sha256": qualification._canonical_sha256(
                    list(_FAKE_STRATA[corpus_index][0])
                ),
                "evaluator_implementation_sha256": evaluator_digest,
                "report_sha256": qualification._canonical_sha256(report),
            }
            binding["binding_sha256"] = qualification._canonical_sha256(binding)
            runs.append({"binding": binding, "report": report})
    assert evaluator_digest is not None
    suite = streaming_stt_suite._suite_report(
        state="complete",
        planned_workers=2,
        planned_corpora=4,
        worker_digests=worker_digests,
        corpus_digests=digests,
        evaluator_implementation_sha256=evaluator_digest,
        controller_sha256=streaming_stt_suite._controller_source_sha256(),
        config=config,
        runs=runs,
    )
    return {
        "ok": True,
        "fixture": {
            "fixture_id": fixture.FIXTURE_ID,
            "lock_recipe_sha256": fixture_set.lock.recipe_sha256,
            "fixture_set_sha256": fixture_set.fixture_set_sha256,
            "sources": len(fixture_set.sources),
            "cases": fixture_set.cases,
            "pcm_bytes": fixture_set.pcm_bytes,
            "validation": "before_and_after_suite",
        },
        "suite": suite,
    }


def _rebind_fake_run(result: dict[str, object], run_index: int = 0) -> None:
    suite = result["suite"]
    assert isinstance(suite, dict)
    runs = suite["runs"]
    assert isinstance(runs, list)
    run = runs[run_index]
    report = run["report"]
    binding = run["binding"]
    binding.pop("binding_sha256")
    binding["report_sha256"] = qualification._canonical_sha256(report)
    binding["binding_sha256"] = qualification._canonical_sha256(binding)
    matrix = qualification._canonical_sha256([item["binding"] for item in runs])
    suite["controller"]["matrix_sha256"] = matrix
    suite["controller"]["completed_matrix_sha256"] = matrix


def test_run_supplies_exact_canonical_strata_and_nonpromotional_result(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    observed: dict[str, object] = {}

    def run_suite(workers, corpora, **kwargs):
        observed["workers"] = tuple(workers)
        observed["corpora"] = tuple(corpora)
        observed["stratum_tags"] = kwargs["stratum_tags"]
        observed["corpus_stratum_tags"] = kwargs["corpus_stratum_tags"]
        observed["scratch"] = kwargs["scratch_parent"]
        observed["checkpoint"] = kwargs["checkpoint_path"]
        return _fake_result(observed["workers"], observed["corpora"])

    result = qualification.run_qualification(
        baseline_worker_manifest=tmp_path / "baseline.json",
        candidate_worker_manifest=tmp_path / "candidate.json",
        corpus_paths=reversed(paths),
        scratch_parent=tmp_path / "scratch",
        output_path=tmp_path / "report.json",
        run_validated_suite_fn=run_suite,
    )

    assert tuple(path.parent.name for path in observed["corpora"]) == fixture.SOURCE_IDS
    assert observed["stratum_tags"] == ()
    assert observed["checkpoint"] != tmp_path / "report.json"
    assert Path(observed["checkpoint"]).parent == observed["scratch"]
    assert Path(observed["checkpoint"]).name.startswith(".canonical-strata-inner-")
    selected = observed["corpus_stratum_tags"]
    assert isinstance(selected, dict)
    assert list(selected.values()) == [
        tuple(f"task_{index:02d}" for index in range(8)),
        ("clean", "disfluent", "overlap_adjacent"),
        ("close", "far", "isolated", "turn_transition"),
        ("d0_2_6s", "d1_6_10s", "d2_10_15s", "d3_15_25s"),
    ]
    assert result["ok"] is True
    assert result["execution_semantics"] == "coverage_only"
    assert result["quality_decision"] == "not_evaluated"
    assert result["development_only"] is True
    assert result["promotional"] is False
    assert result["worker_roles"] == [
        {
            "role": "baseline",
            "worker_index": 0,
            "manifest_sha256": _worker_digest(tmp_path / "baseline.json"),
        },
        {
            "role": "candidate",
            "worker_index": 1,
            "manifest_sha256": _worker_digest(tmp_path / "candidate.json"),
        },
    ]
    encoded = json.dumps(result, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "expected_text" not in encoded


def test_complete_internally_consistent_red_result_stays_red(tmp_path):
    _lock, paths = _publish_set(tmp_path)

    def run_suite(workers, corpora, **_kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        result["ok"] = False
        suite = result["suite"]
        suite["ok"] = False
        suite["runs"][0]["report"]["ok"] = False
        _rebind_fake_run(result)
        return result

    result = qualification.run_qualification(
        baseline_worker_manifest=tmp_path / "baseline.json",
        candidate_worker_manifest=tmp_path / "candidate.json",
        corpus_paths=paths,
        scratch_parent=tmp_path / "scratch",
        output_path=tmp_path / "report.json",
        run_validated_suite_fn=run_suite,
    )

    assert result["ok"] is False
    assert result["execution_complete"] is True
    assert result["quality_decision"] == "not_evaluated"


def test_default_fixture_and_suite_controllers_run_exact_two_by_four(
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)
    calls: list[tuple[int, int]] = []
    path_index = {path.resolve(): index for index, path in enumerate(paths)}

    def benchmark(worker, corpus, **kwargs):
        corpus_index = path_index[Path(corpus).resolve()]
        worker_index = 0 if Path(worker).name == "baseline.json" else 1
        calls.append((worker_index, corpus_index))
        report = _aggregate_report(
            Path(worker),
            Path(corpus),
            worker_digest=_worker_digest(worker),
            corpus_digest=load_corpus(corpus).digest,
            stratum_tags=tuple(kwargs["stratum_tags"]),
        )
        metrics = report["metrics"]
        assert isinstance(metrics, dict)
        rows = metrics["strata"]
        assert isinstance(rows, list)
        for row, count in zip(rows, _FAKE_STRATA[corpus_index][1], strict=True):
            row["cases"] = count
            row["metrics"]["evaluations"] = count
        metrics["evaluations"] = 24
        return report

    monkeypatch.setattr(streaming_stt_eval, "run_benchmark", benchmark)
    result = qualification.run_qualification(
        baseline_worker_manifest=tmp_path / "baseline.json",
        candidate_worker_manifest=tmp_path / "candidate.json",
        corpus_paths=reversed(paths),
        scratch_parent=tmp_path / "scratch",
        output_path=tmp_path / "report.json",
        repeats=1,
    )

    assert calls == [
        (worker_index, corpus_index)
        for worker_index in range(2)
        for corpus_index in range(4)
    ]
    assert result["ok"] is True
    assert result["suite"]["controller"]["runs"] == 8


@pytest.mark.parametrize(
    "mutation",
    ("missing", "unknown", "wrong_count", "reordered"),
)
def test_lock_selection_strata_fail_closed(mutation, tmp_path):
    _lock, paths = _publish_set(tmp_path)
    fixture_set = fixture.validate_fixture_set(paths)
    source = fixture_set.lock.sources[0]
    rows = list(source.selection_recipe["strata"])
    if mutation == "missing":
        rows = rows[:-1]
    elif mutation == "unknown":
        rows[-1] = {"id": "unknown", "cases": rows[-1]["cases"]}
    elif mutation == "wrong_count":
        rows[-1] = {"id": rows[-1]["id"], "cases": rows[-1]["cases"] + 1}
    else:
        rows = list(reversed(rows))
    changed_source = replace(
        source,
        selection_recipe={**source.selection_recipe, "strata": rows},
    )
    changed_lock = replace(
        fixture_set.lock,
        sources=(changed_source, *fixture_set.lock.sources[1:]),
    )

    with pytest.raises(qualification.QualificationError):
        qualification.canonical_stratum_plan(
            replace(fixture_set, lock=changed_lock)
        )


def test_reordered_slots_fail_closed(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    fixture_set = fixture.validate_fixture_set(paths)
    slots = list(fixture_set.lock.slots)
    slots[0], slots[1] = slots[1], slots[0]

    with pytest.raises(qualification.QualificationError):
        qualification.canonical_stratum_plan(
            replace(fixture_set, lock=replace(fixture_set.lock, slots=tuple(slots)))
        )


def test_changed_slot_distribution_fails_closed(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    fixture_set = fixture.validate_fixture_set(paths)
    slots = list(fixture_set.lock.slots)
    slots[2] = replace(slots[2], stratum="task_01")

    with pytest.raises(qualification.QualificationError):
        qualification.canonical_stratum_plan(
            replace(fixture_set, lock=replace(fixture_set.lock, slots=tuple(slots)))
        )


def test_missing_case_stratum_fails_before_runner(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    real_load = qualification.load_corpus
    called = False

    def load_without_first_task(path):
        corpus = real_load(path)
        if corpus.path == paths[0]:
            case = corpus.cases[0]
            changed = replace(
                case,
                tags=tuple(tag for tag in case.tags if tag != "task_00"),
            )
            return replace(corpus, cases=(changed, *corpus.cases[1:]))
        return corpus

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(qualification, "load_corpus", load_without_first_task)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
            run_validated_suite_fn=runner,
        )
    assert called is False


def test_extra_sibling_stratum_fails_before_runner(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    real_load = qualification.load_corpus
    called = False

    def load_with_sibling_task(path):
        corpus = real_load(path)
        if corpus.path == paths[0]:
            case = corpus.cases[0]
            changed = replace(case, tags=(*case.tags, "task_01"))
            return replace(corpus, cases=(changed, *corpus.cases[1:]))
        return corpus

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(qualification, "load_corpus", load_with_sibling_task)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
            run_validated_suite_fn=runner,
        )
    assert called is False


def test_same_worker_role_path_fails_before_runner(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    worker = tmp_path / "same.json"
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=worker,
            candidate_worker_manifest=worker,
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
            run_validated_suite_fn=runner,
        )
    assert called is False


def test_worker_role_symlink_alias_fails_before_runner(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    worker = tmp_path / "worker.json"
    worker.write_text("{}", encoding="utf-8")
    alias = tmp_path / "worker-alias.json"
    alias.symlink_to(worker)
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=worker,
            candidate_worker_manifest=alias,
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
            run_validated_suite_fn=runner,
        )
    assert called is False


def test_same_worker_manifest_digest_fails_before_runner(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    monkeypatch.setattr(
        qualification,
        "load_worker_manifest",
        lambda _path: SimpleNamespace(digest="3" * 64),
    )
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
            run_validated_suite_fn=runner,
        )
    assert called is False


def test_implementation_change_during_run_fails_closed(tmp_path, monkeypatch):
    _lock, paths = _publish_set(tmp_path)
    digests = iter(("1" * 64, "2" * 64))
    monkeypatch.setattr(
        qualification,
        "_implementation_sha256",
        lambda: next(digests),
    )

    output = tmp_path / "report.json"

    def runner(workers, corpora, **kwargs):
        inner = Path(kwargs["checkpoint_path"])
        inner.write_text('{"complete":true,"ok":true}\n', encoding="utf-8")
        inner.chmod(0o600)
        return _fake_result(tuple(workers), tuple(corpora))

    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=output,
            run_validated_suite_fn=runner,
        )
    assert output.exists() is False
    inner = tmp_path / "scratch" / ".canonical-strata-inner-suite-checkpoint.json"
    assert json.loads(inner.read_text(encoding="utf-8"))["ok"] is True


def test_existing_output_fails_before_runner(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    output = tmp_path / "report.json"
    output.write_text('{"ok":true}\n', encoding="utf-8")
    output.chmod(0o600)
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=output,
            run_validated_suite_fn=runner,
        )
    assert called is False


def test_bound_corpus_cannot_be_output(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=paths[0],
            run_validated_suite_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError
            ),
        )


def test_scratch_inside_corpus_evidence_fails_without_creation(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    scratch = paths[0].parent / "qualification-scratch"
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=scratch,
            output_path=tmp_path / "report.json",
            run_validated_suite_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError
            ),
        )
    assert scratch.exists() is False


@pytest.mark.parametrize("nested", (False, True))
def test_scratch_cannot_replace_or_descend_from_future_output(tmp_path, nested):
    _lock, paths = _publish_set(tmp_path)
    output = tmp_path / "future-report"
    scratch = output / "scratch" if nested else output
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=scratch,
            output_path=output,
            run_validated_suite_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError
            ),
        )
    assert output.exists() is False


def test_paths_under_any_git_worktree_fail_before_runner(tmp_path):
    _lock, paths = _publish_set(tmp_path)
    other_checkout = tmp_path / "other-checkout"
    other_checkout.mkdir(mode=0o700)
    (other_checkout / ".git").write_text("gitdir: elsewhere\n", encoding="ascii")
    private = other_checkout / "private"
    private.mkdir(mode=0o700)
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=private / "report.json",
            run_validated_suite_fn=runner,
        )
    assert called is False
    assert (tmp_path / "scratch").exists() is False


@pytest.mark.parametrize("placement", ("scratch_runtime", "output_provision"))
def test_worker_evidence_trees_cannot_receive_outputs(
    placement,
    tmp_path,
    monkeypatch,
):
    _lock, paths = _publish_set(tmp_path)
    provision = tmp_path / "worker-provision"
    provision.mkdir(mode=0o700)
    runtime = tmp_path / "worker-runtime"
    runtime.mkdir(mode=0o700)
    receipt = provision / "runtime-receipt.json"
    receipt.write_text(
        json.dumps({"schema_version": 1, "root": str(runtime), "files": []}),
        encoding="ascii",
    )
    receipt.chmod(0o600)
    receipt_bytes = receipt.read_bytes()

    def manifest(path):
        selected = Path(path)
        return SimpleNamespace(
            digest=_worker_digest(selected),
            path=provision / f"{selected.stem}-manifest.json",
            worker=SimpleNamespace(path=provision / "worker.py"),
            python=SimpleNamespace(path=provision / "venv" / "bin" / "python"),
            artifacts=(
                SimpleNamespace(
                    name="runtime-receipt",
                    path=receipt,
                    sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                    size_bytes=len(receipt_bytes),
                ),
            ),
        )

    (provision / "venv").mkdir(mode=0o700)
    monkeypatch.setattr(qualification, "load_worker_manifest", manifest)
    scratch = (
        runtime / "qualification-scratch"
        if placement == "scratch_runtime"
        else tmp_path / "scratch"
    )
    output = (
        tmp_path / "report.json"
        if placement == "scratch_runtime"
        else provision / "report.json"
    )
    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=scratch,
            output_path=output,
            run_validated_suite_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError
            ),
        )
    assert tuple(runtime.iterdir()) == ()
    assert scratch.exists() is False
    assert output.exists() is False


def test_final_report_is_private_and_no_clobber(tmp_path):
    output = tmp_path / "report.json"
    payload = {"ok": True, "development_only": True, "promotional": False}
    qualification.write_new_report(output, payload)

    assert output.stat().st_mode & 0o777 == 0o600
    assert json.loads(output.read_text(encoding="ascii")) == payload
    with pytest.raises(qualification.QualificationError):
        qualification.write_new_report(output, payload)


@pytest.mark.parametrize("failure", ("partial_write", "file_fsync"))
def test_final_report_publication_failures_leave_no_artifact(
    failure,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "report.json"
    payload = {"ok": True, "development_only": True, "promotional": False}
    if failure == "partial_write":
        real_write = qualification.os.write
        writes = 0

        def fail_after_partial(descriptor, data):
            nonlocal writes
            writes += 1
            if writes == 1:
                partial = max(1, len(data) // 2)
                return real_write(descriptor, data[:partial])
            raise OSError("injected write failure")

        monkeypatch.setattr(qualification.os, "write", fail_after_partial)
    else:

        def fail_fsync(_descriptor):
            raise OSError("injected fsync failure")

        monkeypatch.setattr(qualification.os, "fsync", fail_fsync)

    with pytest.raises(qualification.QualificationError):
        qualification.write_new_report(output, payload)

    assert output.exists() is False
    assert not any(
        entry.name.startswith(qualification._REPORT_TEMP_PREFIX)
        for entry in tmp_path.iterdir()
    )


def test_first_temporary_fstat_failure_is_cleanup_safe(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    real_open = qualification.os.open
    real_fstat = qualification.os.fstat
    temporary_descriptor: int | None = None
    injected = False

    def record_open(path, *args, **kwargs):
        nonlocal temporary_descriptor
        descriptor = real_open(path, *args, **kwargs)
        if isinstance(path, str) and path.startswith(qualification._REPORT_TEMP_PREFIX):
            temporary_descriptor = descriptor
        return descriptor

    def fail_first_temporary_fstat(descriptor):
        nonlocal injected
        if descriptor == temporary_descriptor and not injected:
            injected = True
            raise OSError("injected first temp fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(qualification.os, "open", record_open)
    monkeypatch.setattr(qualification.os, "fstat", fail_first_temporary_fstat)
    with pytest.raises(qualification.QualificationError):
        qualification.write_new_report(
            output,
            {"ok": True, "development_only": True, "promotional": False},
        )

    assert injected is True
    assert output.exists() is False
    assert not any(
        entry.name.startswith(qualification._REPORT_TEMP_PREFIX)
        for entry in tmp_path.iterdir()
    )


@pytest.mark.parametrize(
    "failure",
    (
        "reordered",
        "worker_digest",
        "corpus_digest",
        "stratum_config",
        "stratum_count",
        "fixture",
        "top_ok",
        "suite_ok",
        "run_ok",
        "path_leak",
    ),
)
def test_returned_matrix_and_privacy_are_revalidated(failure, tmp_path):
    _lock, paths = _publish_set(tmp_path)

    def runner(workers, corpora, **_kwargs):
        result = _fake_result(tuple(workers), tuple(corpora))
        suite = result["suite"]
        assert isinstance(suite, dict)
        if failure == "reordered":
            suite["runs"] = list(reversed(suite["runs"]))
        elif failure == "worker_digest":
            suite["runs"][0]["report"]["worker"]["manifest_sha256"] = "0" * 64
            suite["runs"][0]["binding"]["worker_manifest_sha256"] = "0" * 64
            _rebind_fake_run(result)
        elif failure == "corpus_digest":
            suite["runs"][0]["report"]["corpus"]["manifest_sha256"] = "0" * 64
            suite["runs"][0]["binding"]["corpus_manifest_sha256"] = "0" * 64
            _rebind_fake_run(result)
        elif failure == "stratum_config":
            suite["config"]["strata"]["corpora"][0]["tags"] = ["task_00"]
        elif failure == "stratum_count":
            suite["runs"][0]["report"]["metrics"]["strata"][0]["cases"] = 4
            _rebind_fake_run(result)
        elif failure == "fixture":
            result["fixture"]["cases"] = 95
        elif failure == "top_ok":
            result["ok"] = False
        elif failure == "suite_ok":
            suite["ok"] = False
        elif failure == "run_ok":
            suite["runs"][0]["report"]["ok"] = False
            _rebind_fake_run(result)
        else:
            suite["runs"][0]["report"]["metrics"]["private_path"] = str(
                tmp_path / "canary"
            )
            _rebind_fake_run(result)
        return result

    with pytest.raises(qualification.QualificationError):
        qualification.run_qualification(
            baseline_worker_manifest=tmp_path / "baseline.json",
            candidate_worker_manifest=tmp_path / "candidate.json",
            corpus_paths=paths,
            scratch_parent=tmp_path / "scratch",
            output_path=tmp_path / "report.json",
            run_validated_suite_fn=runner,
        )


def test_cli_requires_exact_two_roles_and_hides_paths(tmp_path, capsys):
    canary = tmp_path / "private-baseline.json"
    assert (
        qualification.main(
            [
                "--baseline-worker-manifest",
                str(canary),
                "--corpus",
                str(tmp_path / "corpus.json"),
                "--scratch-root",
                str(tmp_path / "scratch"),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "public conversation qualification failed",
        "ok": False,
    }
    assert str(canary) not in captured.err
