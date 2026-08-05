from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import struct
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from core import endpointing
from tools import livekit_causal_endpoint_eval as evaluator
from tools import livekit_causal_endpoint_model_worker as model_worker
from tools import livekit_causal_endpoint_worker as materializer_worker
from tools import livekit_eot_parquet_worker as inventory_worker


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _close_registry_and_assert_descriptors_closed(
    registry: evaluator.ScratchRegistry,
    descriptors: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    retained = (
        tuple(entry.descriptor for entry in registry.entries.values())
        if descriptors is None
        else descriptors
    )
    registry.close()
    assert registry.entries == {}
    for descriptor in set(retained):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    registry.close()
    return retained


def _pcm_wav(pcm: bytes, *, odd_ancillary_chunk: bool = False) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if odd_ancillary_chunk:
        chunks += b"JUNK" + struct.pack("<I", 1) + b"x\x00"
    chunks += b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks


def _patterned_row() -> tuple[dict[str, object], bytes]:
    pcm = b"".join(struct.pack("<h", (index % 2001) - 1000) for index in range(16_000))
    return (
        {
            "audio": {
                "bytes": _pcm_wav(pcm, odd_ancillary_chunk=True),
                "path": "synthetic.wav",
            },
            "language": "en",
            "duration": 1.0,
            "silence_spans": [
                {"start": 0.25, "end": 0.40},
                {"start": 0.75, "end": 1.0},
            ],
        },
        pcm,
    )


def _policy_contract(**overrides: object) -> model_worker.PolicyContract:
    config = endpointing.EndpointConfig(
        enabled=True,
        min_silence_sec=0.2,
        max_silence_sec=1.6,
        complete_threshold=0.6,
        incomplete_threshold=0.3,
        high_confidence_floor=0.0,
        adaptive_floor=False,
    )
    values: dict[str, object] = {
        "endpoint_config": config,
        "acoustic_rule2_samples": 12_800,
        "prosody_min_samples": 2_400,
        "max_wait_samples": 25_600,
        "rule3_samples": 320_000,
    }
    values.update(overrides)
    return model_worker.PolicyContract(**values)


def _model_row(
    spans: tuple[tuple[int, int], ...],
    *,
    ordinal: int = 0,
    pcm_samples: int | None = None,
    suffix_poison_at: int | None = None,
) -> tuple[model_worker.MaterializedRow, bytes]:
    final_end = spans[-1][1]
    sample_count = final_end if pcm_samples is None else pcm_samples
    samples = [0] * sample_count
    if suffix_poison_at is not None:
        samples[suffix_poison_at:] = [32_000] * (sample_count - suffix_poison_at)
    pcm = struct.pack(f"<{sample_count}h", *samples)
    return (
        model_worker.MaterializedRow(
            ordinal=ordinal,
            pcm_filename=f"row-{ordinal:04d}.pcm",
            pcm_bytes=len(pcm),
            pcm_samples=sample_count,
            pcm_sha256=hashlib.sha256(pcm).hexdigest(),
            silence_spans=tuple(
                model_worker.CausalSpan(start, end) for start, end in spans
            ),
        ),
        pcm,
    )


def _summary(values: list[int]) -> dict[str, object]:
    return dict(model_worker._sample_summary(values))


def _profile(
    *,
    partial_state: str,
    state: str,
    basis: str,
    holds: int = 0,
    eot: int = 1,
    early: int = 0,
    committed: list[int] | None = None,
    censored: list[int] | None = None,
    decisions: int = 1,
) -> dict[str, object]:
    committed = [3_200] if committed is None and eot else list(committed or [])
    censored = list(censored or [])
    conservative = [*committed, *([25_600] * len(censored))]
    return {
        "partial_state": partial_state,
        "hold": {
            "denominator": holds,
            "early_cut_count": early,
            "early_cut_rate": early / holds if holds else None,
            "rows_with_any_early_cut": min(early, 1),
        },
        "eot": {
            "denominator": eot,
            "publisher_labelled_commit_count": len(committed),
            "right_censored_at_publisher_labelled_silence_end_count": len(censored),
            "observed_publisher_labelled_silence_delay": _summary(committed),
            "publisher_labelled_censored_delay_lower_bound": _summary(censored),
            "conservative_all_row_delay_with_censored_at_max_wait": _summary(
                conservative
            ),
        },
        "recorded_prefix_decision_count": decisions,
        "completion_state_counts": ({state: decisions} if decisions else {}),
        "decision_basis_counts": ({basis: decisions} if decisions else {}),
    }


def test_materialize_row_accepts_off_grid_span_and_writes_exact_wav_pcm(
    tmp_path: Path,
):
    row, expected_pcm = _patterned_row()
    outputs = []
    for index in range(2):
        directory = _private_directory(tmp_path / f"materialized-{index}")
        descriptor = _open_directory(directory)
        try:
            leaf_descriptor = os.open(
                "row-0007.pcm",
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=descriptor,
            )
            try:
                expected_identity = materializer_worker._stable_file_identity(
                    os.fstat(leaf_descriptor)
                )
            finally:
                os.close(leaf_descriptor)
            record, summary, packed_spans = materializer_worker._materialize_row(
                row,
                ordinal=7,
                output_descriptor=descriptor,
                inventory_contract=inventory_worker,
                output_leaf_identities={"row-0007.pcm": expected_identity},
            )
        finally:
            os.close(descriptor)
        pcm_path = directory / "row-0007.pcm"
        info = pcm_path.lstat()
        assert pcm_path.read_bytes() == expected_pcm
        assert stat.S_IMODE(info.st_mode) == 0o600
        assert info.st_nlink == 1
        assert record == {
            "ordinal": 7,
            "pcm_filename": "row-0007.pcm",
            "pcm_bytes": len(expected_pcm),
            "pcm_samples": 16_000,
            "pcm_sha256": hashlib.sha256(expected_pcm).hexdigest(),
            "silence_spans": [[4_000, 6_400], [12_000, 16_000]],
        }
        assert summary["audio_samples"] == 16_000
        assert packed_spans == struct.pack(">QQQQ", 4_000, 6_400, 12_000, 16_000)
        assert (record["silence_spans"][0][1] - record["silence_spans"][0][0]) % 1_600
        outputs.append((record, summary, packed_spans))
    assert outputs[0] == outputs[1]


def test_materializer_requests_only_four_admitted_unthreaded_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    class FakeParquet:
        def iter_batches(self, *, batch_size, columns, use_threads):
            captured.update(
                batch_size=batch_size,
                columns=columns,
                use_threads=use_threads,
            )
            if set(columns) & {"id", "words", "messages"}:
                raise AssertionError("forbidden value column was materialized")
            return iter(())

    class Contract:
        PYARROW_VERSION = "25.0.0"
        _SCAN_COLUMNS = ("audio", "language", "duration", "silence_spans")
        SOURCE_REVISION = "revision"
        SOURCE_SIZE_BYTES = 1
        SOURCE_SHA256 = "a" * 64
        SOURCE_ROWS = 400

        @staticmethod
        @contextmanager
        def _opened_parquet(_pq, descriptor):
            assert descriptor == 41
            yield FakeParquet(), 2

    pyarrow = ModuleType("pyarrow")
    pyarrow.__path__ = []  # type: ignore[attr-defined]
    pyarrow.__version__ = "25.0.0"
    parquet = ModuleType("pyarrow.parquet")
    pyarrow.parquet = parquet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    directory = _private_directory(tmp_path / "output")
    descriptor = _open_directory(directory)
    try:
        with pytest.raises(materializer_worker.WorkerError):
            materializer_worker._materialize(
                {
                    "source_descriptor": 41,
                    "source_revision": "revision",
                    "source_size_bytes": 1,
                    "source_sha256": "a" * 64,
                    "output_leaf_identities": {
                        **{
                            f"row-{ordinal:04d}.pcm": [1, ordinal + 1, 0, 0, 0, 1]
                            for ordinal in range(400)
                        },
                        "manifest.json": [1, 401, 0, 0, 0, 1],
                    },
                },
                descriptor,
                Contract,
            )
    finally:
        os.close(descriptor)

    assert captured == {
        "batch_size": 4,
        "columns": ["audio", "language", "duration", "silence_spans"],
        "use_threads": False,
    }


def test_materializer_private_writer_rejects_replacement_without_unlinking_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    directory = _private_directory(tmp_path / "worker-output")
    descriptor = _open_directory(directory)
    real_stat = os.stat
    replacement = b"replacement"
    raced = False
    leaf_descriptor = os.open(
        "leaf.bin",
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=descriptor,
    )
    try:
        expected_identity = materializer_worker._stable_file_identity(
            os.fstat(leaf_descriptor)
        )
    finally:
        os.close(leaf_descriptor)

    def racing_stat(path, *args, **kwargs):
        nonlocal raced
        if path == "leaf.bin" and kwargs.get("dir_fd") == descriptor and not raced:
            raced = True
            os.unlink("leaf.bin", dir_fd=descriptor)
            replacement_fd = os.open(
                "leaf.bin",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=descriptor,
            )
            try:
                os.write(replacement_fd, replacement)
                os.fsync(replacement_fd)
            finally:
                os.close(replacement_fd)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(materializer_worker.os, "stat", racing_stat)
    try:
        with pytest.raises(materializer_worker.WorkerError):
            materializer_worker._write_private_file(
                descriptor,
                "leaf.bin",
                b"original-data",
                maximum=1024,
                expected_identity=expected_identity,
            )
    finally:
        os.close(descriptor)

    assert raced is True
    assert (directory / "leaf.bin").read_bytes() == replacement


def test_materializer_resource_limits_are_exact_requested_ceilings(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_resource = ModuleType("resource")
    fake_resource.RLIM_INFINITY = -1
    names = (
        "RLIMIT_CORE",
        "RLIMIT_CPU",
        "RLIMIT_FSIZE",
        "RLIMIT_NOFILE",
        "RLIMIT_AS",
    )
    for index, name in enumerate(names, start=1):
        setattr(fake_resource, name, index)
    requested = {
        fake_resource.RLIMIT_CORE: 0,
        fake_resource.RLIMIT_CPU: 600,
        fake_resource.RLIMIT_FSIZE: 32 * 1024 * 1024,
        fake_resource.RLIMIT_NOFILE: 64,
        fake_resource.RLIMIT_AS: 2 * 1024 * 1024 * 1024,
    }
    calls: list[tuple[int, tuple[int, int]]] = []
    fake_resource.getrlimit = lambda _resource_id: (999, -1)
    fake_resource.setrlimit = lambda resource_id, limits: calls.append(
        (resource_id, limits)
    )
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    materializer_worker._set_resource_limits()

    assert calls == [
        (resource_id, (ceiling, -1))
        for resource_id, ceiling in requested.items()
    ]


def test_materializer_resource_limits_fail_closed_when_constant_is_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_resource = ModuleType("resource")
    fake_resource.RLIM_INFINITY = -1
    for index, name in enumerate(
        ("RLIMIT_CORE", "RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE"),
        start=1,
    ):
        setattr(fake_resource, name, index)
    fake_resource.getrlimit = lambda _resource_id: (999, -1)
    fake_resource.setrlimit = lambda _resource_id, _limits: None
    monkeypatch.setitem(sys.modules, "resource", fake_resource)

    with pytest.raises(materializer_worker.WorkerError):
        materializer_worker._set_resource_limits()


def test_leaf_identity_tuple_order_matches_across_all_process_boundaries(
    tmp_path: Path,
):
    leaf = tmp_path / "leaf.bin"
    leaf.write_bytes(b"identity")
    leaf.chmod(0o600)
    info = leaf.lstat()
    expected = (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_uid,
        info.st_nlink,
    )

    assert evaluator._stable_file_identity(info) == expected
    assert materializer_worker._stable_file_identity(info) == expected
    assert model_worker._stable_file_identity(info) == expected


def test_execution_closure_reader_admits_current_sherpa_above_public_limit():
    sherpa = evaluator._REPO_ROOT / "core/engines/sherpa.py"

    payload = evaluator._read_stable_execution_closure_member(sherpa)
    closure = evaluator._snapshot_execution_closure()

    assert 512 * 1024 < len(payload) <= evaluator._MAX_EXECUTION_CLOSURE_MEMBER_BYTES
    assert closure["core/engines/sherpa.py"] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(evaluator.public_fixtures.PublicFixtureError):
        evaluator.public_fixtures._read_stable_repo_worker(sherpa)


def test_execution_closure_reader_rejects_unallowlisted_repo_member():
    with pytest.raises(evaluator.LiveKitCausalEndpointError):
        evaluator._read_stable_execution_closure_member(Path(__file__).resolve())


def test_supervisor_private_writer_rejects_replacement_without_unlinking_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    directory = _private_directory(tmp_path / "supervisor-output")
    descriptor = _open_directory(directory)
    real_stat = os.stat
    replacement = b"replacement"
    raced = False

    def racing_stat(path, *args, **kwargs):
        nonlocal raced
        if path == "leaf.bin" and kwargs.get("dir_fd") == descriptor and not raced:
            raced = True
            os.unlink("leaf.bin", dir_fd=descriptor)
            replacement_fd = os.open(
                "leaf.bin",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=descriptor,
            )
            try:
                os.write(replacement_fd, replacement)
                os.fsync(replacement_fd)
            finally:
                os.close(replacement_fd)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(evaluator.os, "stat", racing_stat)
    try:
        with pytest.raises(evaluator.LiveKitCausalEndpointError):
            evaluator._write_new_private_file_at(
                descriptor,
                "leaf.bin",
                b"original-data",
                maximum=1024,
            )
    finally:
        os.close(descriptor)

    assert raced is True
    assert (directory / "leaf.bin").read_bytes() == replacement


def test_parent_created_private_leaf_ownership_comes_from_retained_descriptor(
    tmp_path: Path,
):
    directory = _private_directory(tmp_path / "supervisor-output")
    directory_descriptor = _open_directory(directory)
    descriptor = -1
    try:
        descriptor, identity = evaluator._create_filled_private_file_at(
            directory_descriptor,
            "leaf.bin",
            b"descriptor-bound",
            maximum=1024,
            mode=0o400,
        )
        opened = os.fstat(descriptor)
        current = os.stat(
            "leaf.bin",
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        assert evaluator._inode_identity(opened) == identity
        assert evaluator._identity(opened) == evaluator._identity(current)
        assert os.pread(descriptor, opened.st_size, 0) == b"descriptor-bound"
        assert stat.S_IMODE(opened.st_mode) == 0o400
        assert opened.st_nlink == 1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)


@pytest.mark.parametrize(
    ("span_samples", "is_eot", "expected"),
    [
        (1_600, False, ()),
        (3_200, False, (1_600,)),
        (2_400, True, (1_600, 2_400)),
        (3_201, True, (1_600, 3_200, 3_201)),
        (30_000, True, tuple(range(1_600, 25_601, 1_600))),
        (30_000, False, tuple(range(1_600, 25_601, 1_600))),
    ],
)
def test_model_decision_ticks_pin_hold_equality_eot_end_and_max_wait(
    span_samples: int,
    is_eot: bool,
    expected: tuple[int, ...],
):
    assert model_worker._decision_ticks(
        span_samples,
        is_eot=is_eot,
        max_wait_samples=25_600,
    ) == expected


def test_model_profile_owns_exact_causal_prefix_and_stops_after_first_commit():
    np = pytest.importorskip("numpy")
    monkeypatch_target = model_worker
    monkeypatch_target.np = np
    row, pcm = _model_row(
        ((8_000, 14_401),),
        suffix_poison_at=11_200,
    )
    calls = []

    class Detector:
        needs_audio = True

        def completion_score(self, text, *, samples=None, sample_rate=16_000):
            assert text == model_worker._OPAQUE_PARTIAL_SENTINEL
            assert sample_rate == 16_000
            assert samples is not None
            assert samples.shape == (11_200,)
            assert samples.flags.owndata is True
            assert samples.flags.writeable is False
            assert samples.base is None
            assert float(np.max(np.abs(samples))) == 0.0
            calls.append(samples)
            return 0.9

    result = model_worker._evaluate_profile(
        rows=(row,),
        pcm_loader=lambda requested: pcm if requested is row else b"",
        detector=Detector(),
        endpointing=endpointing,
        contract=_policy_contract(),
        assumed_partial=True,
        expected_hold_denominator=0,
        expected_eot_denominator=1,
    )

    assert len(calls) == 1
    full = result["semantic_counterfactual_full_source"]
    assert full["eot"]["publisher_labelled_commit_count"] == 1
    assert full["eot"]["observed_publisher_labelled_silence_delay"]["samples"] == {
        "count": 1,
        "sum": 3_200,
        "min": 3_200,
        "p50_nearest_rank": 3_200,
        "p95_nearest_rank": 3_200,
        "max": 3_200,
    }


def test_model_profile_censors_at_publisher_labelled_final_silence_end_without_zero_tail():
    np = pytest.importorskip("numpy")
    model_worker.np = np
    row, pcm = _model_row(((8_000, 14_401),), suffix_poison_at=14_400)
    prefix_lengths = []

    class Detector:
        needs_audio = True

        def completion_score(self, text, *, samples=None, sample_rate=16_000):
            assert samples is not None
            prefix_lengths.append(int(samples.size))
            return 0.0

    result = model_worker._evaluate_profile(
        rows=(row,),
        pcm_loader=lambda _row: pcm,
        detector=Detector(),
        endpointing=endpointing,
        contract=_policy_contract(),
        assumed_partial=True,
        expected_hold_denominator=0,
        expected_eot_denominator=1,
    )

    assert prefix_lengths == [11_200, 12_800, 14_400, 14_401]
    assert max(prefix_lengths) == row.silence_spans[-1].end_sample
    eot = result["semantic_counterfactual_full_source"]["eot"]
    assert eot["publisher_labelled_commit_count"] == 0
    assert eot["right_censored_at_publisher_labelled_silence_end_count"] == 1
    assert eot["publisher_labelled_censored_delay_lower_bound"]["samples"] == {
        "count": 1,
        "sum": 6_401,
        "min": 6_401,
        "p50_nearest_rank": 6_401,
        "p95_nearest_rank": 6_401,
        "max": 6_401,
    }
    assert eot["conservative_all_row_delay_with_censored_at_max_wait"]["samples"][
        "max"
    ] == 25_600


def test_model_detector_exception_fails_closed_instead_of_reporting_fallback():
    np = pytest.importorskip("numpy")
    model_worker.np = np
    row, pcm = _model_row(((8_000, 14_400),))

    class Detector:
        needs_audio = True

        def completion_score(self, text, *, samples=None, sample_rate=16_000):
            raise RuntimeError("synthetic detector failure")

    with pytest.raises(model_worker.ModelWorkerError):
        model_worker._evaluate_profile(
            rows=(row,),
            pcm_loader=lambda _row: pcm,
            detector=Detector(),
            endpointing=endpointing,
            contract=_policy_contract(),
            assumed_partial=True,
            expected_hold_denominator=0,
            expected_eot_denominator=1,
        )


def test_metric_accumulator_requires_strict_hold_cut_and_dedupes_cut_rows():
    accumulator = model_worker._MetricAccumulator.empty()
    first_hold = model_worker.CausalSpan(0, 3_200)
    second_hold = model_worker.CausalSpan(4_000, 8_000)
    for span in (first_hold, second_hold):
        accumulator.begin_span(is_eot=False)
        accumulator.finish_span(
            is_eot=False,
            committed_delay=1_600,
            span=span,
            row_ordinal=7,
            max_wait_samples=25_600,
        )
    accumulator.begin_span(is_eot=True)
    accumulator.finish_span(
        is_eot=True,
        committed_delay=None,
        span=model_worker.CausalSpan(9_000, 15_401),
        row_ordinal=7,
        max_wait_samples=25_600,
    )
    report = accumulator.report(
        assumed_partial=True,
        expected_holds=2,
        expected_eot=1,
    )

    assert report["hold"] == {
        "denominator": 2,
        "early_cut_count": 2,
        "early_cut_rate": 1.0,
        "rows_with_any_early_cut": 1,
    }
    assert report["eot"][
        "publisher_labelled_censored_delay_lower_bound"
    ]["samples"]["min"] == 6_401

    invalid = model_worker._MetricAccumulator.empty()
    invalid.begin_span(is_eot=False)
    with pytest.raises(model_worker.ModelWorkerError):
        invalid.finish_span(
            is_eot=False,
            committed_delay=first_hold.duration_samples,
            span=first_hold,
            row_ordinal=7,
            max_wait_samples=25_600,
        )


def test_no_partial_profile_never_loads_pcm_or_calls_model_and_is_separate():
    np = pytest.importorskip("numpy")
    model_worker.np = np
    row, _pcm = _model_row(((8_000, 20_800),))

    class Detector:
        needs_audio = True

        def completion_score(self, *args, **kwargs):
            raise AssertionError("no-partial profile called the detector")

    result = model_worker._evaluate_profile(
        rows=(row,),
        pcm_loader=lambda _row: (_ for _ in ()).throw(
            AssertionError("no-partial profile loaded PCM")
        ),
        detector=Detector(),
        endpointing=endpointing,
        contract=_policy_contract(),
        assumed_partial=False,
        expected_hold_denominator=0,
        expected_eot_denominator=1,
    )

    full = result["semantic_counterfactual_full_source"]
    assert full["partial_state"] == model_worker.NO_PARTIAL_PROFILE
    assert full["completion_state_counts"] == {"no_partial": 8}
    assert full["decision_basis_counts"] == {"acoustic": 8}
    assert full["eot"]["publisher_labelled_commit_count"] == 1
    assert full["eot"]["observed_publisher_labelled_silence_delay"]["samples"][
        "min"
    ] == 12_800


def test_model_policy_is_fresh_and_receives_only_prior_publisher_holds(
    monkeypatch: pytest.MonkeyPatch,
):
    np = pytest.importorskip("numpy")
    model_worker.np = np
    row, pcm = _model_row(
        (
            (1_000, 2_600),
            (3_000, 6_200),
            (7_000, 10_200),
        )
    )
    observed_prior: list[tuple[tuple[int, int], ...]] = []
    original = model_worker._policy_with_prior_holds

    def recording_policy(endpointing_module, config, prior_holds):
        observed_prior.append(
            tuple((span.start_sample, span.end_sample) for span in prior_holds)
        )
        return original(endpointing_module, config, prior_holds)

    monkeypatch.setattr(model_worker, "_policy_with_prior_holds", recording_policy)

    class Detector:
        needs_audio = True

        def completion_score(self, text, *, samples=None, sample_rate=16_000):
            return 0.9

    model_worker._evaluate_profile(
        rows=(row,),
        pcm_loader=lambda _row: pcm,
        detector=Detector(),
        endpointing=endpointing,
        contract=_policy_contract(),
        assumed_partial=True,
        expected_hold_denominator=2,
        expected_eot_denominator=1,
    )

    assert observed_prior == [
        (),
        ((1_000, 2_600),),
        ((1_000, 2_600), (3_000, 6_200)),
    ]


def test_rule3_subset_is_strict_whole_label_scope_and_reports_crossing_labels():
    rows = (
        evaluator.MaterializedRow(
            ordinal=0,
            pcm_filename="row-0000.pcm",
            pcm_bytes=640_002,
            pcm_samples=320_001,
            pcm_sha256="a" * 64,
            silence_spans=(
                evaluator.CausalSpan(318_000, 319_999),
                evaluator.CausalSpan(319_000, 320_000),
            ),
        ),
        evaluator.MaterializedRow(
            ordinal=1,
            pcm_filename="row-0001.pcm",
            pcm_bytes=646_400,
            pcm_samples=323_200,
            pcm_sha256="b" * 64,
            silence_spans=(evaluator.CausalSpan(320_000, 323_200),),
        ),
    )

    assert evaluator._fully_before_rule3_denominators(rows, 320_000) == (1, 0)
    scope = evaluator._publisher_rule3_label_scope(rows, 320_000)
    assert scope["publisher_labels_fully_before_rule3"] == {
        "hold_denominator": 1,
        "eot_denominator": 0,
    }
    assert scope["crossing_rule3"] == {
        "row_count": 1,
        "hold_label_count": 0,
        "eot_label_count": 1,
    }
    assert scope["starting_at_or_beyond_rule3"] == {
        "row_count": 1,
        "hold_label_count": 0,
        "eot_label_count": 1,
    }
    assert scope["crossing_labels_may_contain_earlier_decision_ticks"] is True


def test_model_result_validator_rejects_censor_counter_and_summary_tampering():
    base = _profile(
        partial_state=evaluator.PARTIAL_ASSUMPTION,
        state="scored",
        basis="semantic_early",
    )
    evaluator._validate_profile_result(
        base,
        expected_partial_state=evaluator.PARTIAL_ASSUMPTION,
        expected_holds=0,
        expected_eot=1,
    )

    mutations = []
    wrong_total = copy.deepcopy(base)
    wrong_total["eot"]["publisher_labelled_commit_count"] = 0
    mutations.append(wrong_total)
    wrong_ms = copy.deepcopy(base)
    wrong_ms["eot"]["observed_publisher_labelled_silence_delay"]["milliseconds"][
        "max"
    ] = 0.0
    mutations.append(wrong_ms)
    unknown_state = copy.deepcopy(base)
    unknown_state["completion_state_counts"] = {"detector_error": 1}
    mutations.append(unknown_state)
    unknown_basis = copy.deepcopy(base)
    unknown_basis["decision_basis_counts"] = {"max_wait": 1}
    mutations.append(unknown_basis)
    wrong_decisions = copy.deepcopy(base)
    wrong_decisions["recorded_prefix_decision_count"] = 2
    mutations.append(wrong_decisions)
    nonfinite_rate = copy.deepcopy(base)
    nonfinite_rate["hold"]["early_cut_rate"] = float("nan")
    mutations.append(nonfinite_rate)

    censored_at_cap = _profile(
        partial_state=evaluator.PARTIAL_ASSUMPTION,
        state="scored",
        basis="semantic_wait",
        committed=[],
        censored=[25_600],
    )
    mutations.append(censored_at_cap)

    for mutated in mutations:
        with pytest.raises(evaluator.LiveKitCausalEndpointError):
            evaluator._validate_profile_result(
                mutated,
                expected_partial_state=evaluator.PARTIAL_ASSUMPTION,
                expected_holds=0,
                expected_eot=1,
            )


def test_model_result_validator_rejects_cross_summary_and_quantile_tampering():
    censored = _profile(
        partial_state=evaluator.PARTIAL_ASSUMPTION,
        state="scored",
        basis="semantic_wait",
        eot=2,
        committed=[3_200],
        censored=[15_000],
    )
    evaluator._validate_profile_result(
        censored,
        expected_partial_state=evaluator.PARTIAL_ASSUMPTION,
        expected_holds=0,
        expected_eot=2,
    )

    wrong_sum = copy.deepcopy(censored)
    conservative = wrong_sum["eot"][
        "conservative_all_row_delay_with_censored_at_max_wait"
    ]
    conservative["samples"]["sum"] = 30_400
    conservative["milliseconds"]["sum"] = 1_900.0

    wrong_max = copy.deepcopy(censored)
    conservative = wrong_max["eot"][
        "conservative_all_row_delay_with_censored_at_max_wait"
    ]
    conservative["samples"]["max"] = 24_000
    conservative["samples"]["p95_nearest_rank"] = 24_000
    conservative["milliseconds"]["max"] = 1_500.0
    conservative["milliseconds"]["p95_nearest_rank"] = 1_500.0

    uncensored = _profile(
        partial_state=evaluator.PARTIAL_ASSUMPTION,
        state="scored",
        basis="semantic_early",
    )
    wrong_uncensored = copy.deepcopy(uncensored)
    conservative = wrong_uncensored["eot"][
        "conservative_all_row_delay_with_censored_at_max_wait"
    ]
    for name in ("sum", "min", "p50_nearest_rank", "p95_nearest_rank", "max"):
        conservative["samples"][name] = 4_800
        conservative["milliseconds"][name] = 300.0

    wrong_quantile_order = _profile(
        partial_state=evaluator.PARTIAL_ASSUMPTION,
        state="scored",
        basis="semantic_wait",
        eot=2,
        committed=[],
        censored=[3_200, 6_400],
    )
    lower_bound = wrong_quantile_order["eot"][
        "publisher_labelled_censored_delay_lower_bound"
    ]
    lower_bound["samples"]["p50_nearest_rank"] = 6_400
    lower_bound["samples"]["p95_nearest_rank"] = 3_200
    lower_bound["milliseconds"]["p50_nearest_rank"] = 400.0
    lower_bound["milliseconds"]["p95_nearest_rank"] = 200.0

    for mutated in (wrong_sum, wrong_max, wrong_uncensored, wrong_quantile_order):
        with pytest.raises(evaluator.LiveKitCausalEndpointError):
            evaluator._validate_profile_result(
                mutated,
                expected_partial_state=evaluator.PARTIAL_ASSUMPTION,
                expected_holds=0,
                expected_eot=mutated["eot"]["denominator"],
            )


def test_supervisor_rejects_less_than_512_spare_fds_before_scratch_or_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    report_parent = _private_directory(tmp_path / "reports")
    destination = report_parent / "aggregate.json"
    opened = len(os.listdir("/proc/self/fd"))
    required = evaluator._SCRATCH_REGISTRY_MINIMUM_SPARE_FDS
    assert required == 512
    monkeypatch.setattr(
        resource,
        "getrlimit",
        lambda which: (opened + required - 1, opened + required - 1),
    )
    monkeypatch.setattr(evaluator, "_snapshot_execution_closure", lambda: {})
    monkeypatch.setattr(evaluator, "_validate_inventory_receipt", lambda _path: {})
    monkeypatch.setattr(evaluator, "_load_policy_contract", lambda _path: object())
    monkeypatch.setattr(
        evaluator.inventory_source,
        "_output_path",
        lambda _path: (destination, report_parent.lstat()),
    )
    scratch_calls: list[Path | str] = []
    spawn_calls: list[object] = []

    def forbidden_scratch(path):
        scratch_calls.append(path)
        raise AssertionError("scratch creation ran after failed descriptor preflight")

    def forbidden_spawn(*args, **kwargs):
        spawn_calls.append((args, kwargs))
        raise AssertionError("worker spawned after failed descriptor preflight")

    monkeypatch.setattr(evaluator, "_create_private_scratch", forbidden_scratch)
    monkeypatch.setattr(evaluator.subprocess, "Popen", forbidden_spawn)

    with pytest.raises(
        evaluator.LiveKitCausalEndpointError,
        match="scratch descriptor capacity failed",
    ):
        evaluator.evaluate_livekit_causal_endpoint(
            source_parquet=tmp_path / "source.parquet",
            inventory_report=tmp_path / "inventory.json",
            parquet_python=sys.executable,
            model=tmp_path / "model.onnx",
            config=tmp_path / "config.json",
            scratch_root=tmp_path / "scratch",
            output=destination,
            accepted_terms=frozenset({evaluator.LICENSE}),
            accept_partial_assumption=True,
        )

    assert scratch_calls == []
    assert spawn_calls == []


def test_registered_scratch_cleanup_removes_only_bound_inodes(tmp_path: Path):
    parent = _private_directory(tmp_path / "private-parent")
    scratch_path = parent / "scratch"
    scratch, descriptor, identity, registry = evaluator._create_private_scratch(
        scratch_path
    )
    evaluator._write_new_private_file_at(
        descriptor,
        "model-result.json",
        b"{}\n",
        maximum=128,
    )
    evaluator._register_scratch_tree(descriptor, "model-result.json", registry)

    retained = tuple(entry.descriptor for entry in registry.entries.values())
    try:
        evaluator._cleanup_private_scratch(scratch, descriptor, identity, registry)
    finally:
        _close_registry_and_assert_descriptors_closed(registry, retained)
        os.close(descriptor)
    assert not scratch.exists()


def test_registered_scratch_cleanup_preserves_replaced_inode_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent = _private_directory(tmp_path / "private-parent")
    scratch, descriptor, identity, registry = evaluator._create_private_scratch(
        parent / "scratch"
    )
    evaluator._write_new_private_file_at(
        descriptor,
        "model-result.json",
        b"original\n",
        maximum=128,
    )
    evaluator._register_scratch_tree(descriptor, "model-result.json", registry)
    original_validate = evaluator._validate_registered_scratch_tree
    raced = False

    def validate_then_replace(*args, **kwargs):
        nonlocal raced
        original_validate(*args, **kwargs)
        if kwargs.get("prefix") == () and not raced:
            raced = True
            os.unlink("model-result.json", dir_fd=descriptor)
            replacement_fd = os.open(
                "model-result.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=descriptor,
            )
            try:
                os.write(replacement_fd, b"replacement\n")
            finally:
                os.close(replacement_fd)

    monkeypatch.setattr(
        evaluator,
        "_validate_registered_scratch_tree",
        validate_then_replace,
    )

    try:
        with pytest.raises(evaluator.LiveKitCausalEndpointError, match="cleanup"):
            evaluator._cleanup_private_scratch(scratch, descriptor, identity, registry)

        assert raced is True
        assert (scratch / "model-result.json").read_bytes() == b"replacement\n"
    finally:
        _close_registry_and_assert_descriptors_closed(registry)
        os.close(descriptor)


def test_report_publication_is_private_no_clobber_and_cleans_temporary_files(
    tmp_path: Path,
):
    parent = _private_directory(tmp_path / "reports")
    destination = parent / "aggregate.json"
    payload = b'{"aggregate":true}\n'
    parent_info = parent.lstat()
    rebind_calls = []

    digest = evaluator._publish_report(
        destination,
        parent_info,
        payload,
        rebind=lambda: rebind_calls.append("rebind"),
    )
    info = destination.lstat()
    assert destination.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    assert stat.S_IMODE(info.st_mode) == 0o600
    assert info.st_nlink == 1
    assert rebind_calls == ["rebind", "rebind"]
    assert set(parent.iterdir()) == {destination}

    with pytest.raises(evaluator.LiveKitCausalEndpointError):
        evaluator._publish_report(
            destination,
            parent.lstat(),
            b"replacement\n",
            rebind=lambda: None,
        )
    assert destination.read_bytes() == payload
    assert set(parent.iterdir()) == {destination}


def test_report_destination_created_during_rebind_is_preserved(tmp_path: Path):
    parent = _private_directory(tmp_path / "reports")
    destination = parent / "aggregate.json"
    sentinel = b"racing-owner-output\n"

    def race() -> None:
        destination.write_bytes(sentinel)
        destination.chmod(0o600)

    with pytest.raises(evaluator.LiveKitCausalEndpointError):
        evaluator._publish_report(
            destination,
            parent.lstat(),
            b'{"aggregate":true}\n',
            rebind=race,
        )

    assert destination.read_bytes() == sentinel
    assert set(parent.iterdir()) == {destination}


def test_report_temporary_replacement_is_preserved_and_never_published(
    tmp_path: Path,
):
    parent = _private_directory(tmp_path / "reports")
    destination = parent / "aggregate.json"
    sentinel = b"foreign-temporary-inode\n"
    raced_name: Path | None = None

    def replace_temporary() -> None:
        nonlocal raced_name
        temporary = [path for path in parent.iterdir() if path.name.endswith(".part")]
        assert len(temporary) == 1
        raced_name = temporary[0]
        raced_name.unlink()
        raced_name.write_bytes(sentinel)
        raced_name.chmod(0o600)

    with pytest.raises(evaluator.LiveKitCausalEndpointError):
        evaluator._publish_report(
            destination,
            parent.lstat(),
            b'{"aggregate":true}\n',
            rebind=replace_temporary,
        )

    assert not destination.exists()
    assert raced_name is not None
    assert raced_name.read_bytes() == sentinel


def test_report_bound_link_never_publishes_temporary_name_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent = _private_directory(tmp_path / "reports")
    destination = parent / "aggregate.json"
    payload = b'{"aggregate":"retained-owner"}\n'
    foreign = b"foreign-temporary-inode\n"
    linked_payloads: list[bytes] = []
    link_attempted = False
    real_link = os.link

    def swap_temporary_then_link(source, target, *args, **kwargs):
        nonlocal link_attempted
        assert source.startswith("/proc/self/fd/")
        assert target == destination.name
        temporary = [path for path in parent.iterdir() if path.name.endswith(".part")]
        assert len(temporary) == 1
        temporary[0].unlink()
        temporary[0].write_bytes(foreign)
        temporary[0].chmod(0o600)
        link_attempted = True
        try:
            return real_link(source, target, *args, **kwargs)
        finally:
            if destination.exists():
                linked_payloads.append(destination.read_bytes())

    monkeypatch.setattr(evaluator.os, "link", swap_temporary_then_link)

    with pytest.raises(evaluator.LiveKitCausalEndpointError):
        evaluator._publish_report(
            destination,
            parent.lstat(),
            payload,
            rebind=lambda: None,
        )

    assert link_attempted is True
    assert foreign not in linked_payloads
    assert not destination.exists()
    temporary = [path for path in parent.iterdir() if path.name.endswith(".part")]
    assert len(temporary) == 1
    assert temporary[0].read_bytes() == foreign


def test_report_destination_swap_immediately_after_bound_link_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent = _private_directory(tmp_path / "reports")
    destination = parent / "aggregate.json"
    payload = b'{"aggregate":"retained-owner"}\n'
    foreign = b"foreign-destination-after-link\n"
    linked_payloads: list[bytes] = []
    real_link = os.link

    def link_then_swap_destination(source, target, *args, **kwargs):
        assert source.startswith("/proc/self/fd/")
        assert target == destination.name
        result = real_link(source, target, *args, **kwargs)
        linked_payloads.append(destination.read_bytes())
        destination.unlink()
        destination.write_bytes(foreign)
        destination.chmod(0o600)
        return result

    monkeypatch.setattr(evaluator.os, "link", link_then_swap_destination)

    with pytest.raises(evaluator.LiveKitCausalEndpointError):
        evaluator._publish_report(
            destination,
            parent.lstat(),
            payload,
            rebind=lambda: None,
        )

    assert linked_payloads == [payload]
    assert destination.read_bytes() == foreign
    assert set(parent.iterdir()) == {destination}


def test_report_post_link_rollback_never_unlinks_replaced_destination(
    tmp_path: Path,
):
    parent = _private_directory(tmp_path / "reports")
    destination = parent / "aggregate.json"
    sentinel = b"foreign-published-inode\n"
    calls = 0

    def replace_after_link() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.unlink()
            destination.write_bytes(sentinel)
            destination.chmod(0o600)
            raise evaluator.LiveKitCausalEndpointError("synthetic post-link failure")

    with pytest.raises(evaluator.LiveKitCausalEndpointError):
        evaluator._publish_report(
            destination,
            parent.lstat(),
            b'{"aggregate":true}\n',
            rebind=replace_after_link,
        )

    assert calls == 2
    assert destination.read_bytes() == sentinel
    assert set(parent.iterdir()) == {destination}


def test_private_subdirectory_rejects_replacement_after_descriptor_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent = _private_directory(tmp_path / "private-parent")
    parent_descriptor = _open_directory(parent)
    registry = evaluator.ScratchRegistry()
    target = parent / "materialized"
    real_stat = os.stat
    raced = False

    def replace_after_capture(path, *args, **kwargs):
        nonlocal raced
        if (
            path == target.name
            and kwargs.get("dir_fd") == parent_descriptor
            and not raced
        ):
            raced = True
            os.rmdir(target.name, dir_fd=parent_descriptor)
            os.mkdir(target.name, 0o700, dir_fd=parent_descriptor)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(evaluator.os, "stat", replace_after_capture)
    try:
        with pytest.raises(evaluator.LiveKitCausalEndpointError, match="scratch"):
            evaluator._make_private_subdirectory(
                parent,
                target.name,
                parent_descriptor=parent_descriptor,
                registry=registry,
            )
        assert raced is True
        assert target.is_dir()
        assert registry.entries == {}
    finally:
        _close_registry_and_assert_descriptors_closed(registry)
        os.close(parent_descriptor)


def test_materializer_supervisor_uses_bound_fds_sanitized_env_and_reaps_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent = _private_directory(tmp_path / "private-parent")
    scratch, scratch_fd, scratch_identity, registry = evaluator._create_private_scratch(
        parent / "scratch"
    )
    source = tmp_path / "source.parquet"
    source.write_bytes(b"descriptor-bound source")
    source.chmod(0o600)
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    interpreter = Path(sys.executable)
    execution_closure = {
        "tools/livekit_causal_endpoint_worker.py": hashlib.sha256(
            evaluator._CAUSAL_WORKER_PATH.read_bytes()
        ).hexdigest(),
        "tools/livekit_eot_parquet_worker.py": hashlib.sha256(
            evaluator._INVENTORY_WORKER_PATH.read_bytes()
        ).hexdigest(),
    }
    validated_interpreters: list[Path | str] = []

    def validate_interpreter(path):
        validated_interpreters.append(path)
        return interpreter

    monkeypatch.setattr(
        evaluator.public_fixtures,
        "_validate_parquet_python",
        validate_interpreter,
    )
    events: list[object] = []
    captured: dict[str, object] = {}
    original_rebind_materializer_venv = evaluator._rebind_materializer_venv_launch
    materializer_venv_rebinds: list[evaluator.VenvLaunchSnapshot] = []

    def rebind_materializer_venv(snapshot):
        materializer_venv_rebinds.append(snapshot)
        events.append("venv-post" if "reap" in events else "venv-pre")
        original_rebind_materializer_venv(snapshot)

    class FakeProcess:
        pid = 987_601

        def __init__(self, argv, **kwargs):
            self.returncode = None
            captured["argv"] = list(argv)
            captured["kwargs"] = kwargs
            request_fd = int(argv[argv.index("--request-fd") + 1])
            request_info = os.fstat(request_fd)
            captured["request"] = json.loads(
                os.pread(request_fd, request_info.st_size, 0)
            )

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            self.returncode = 0
            return 0

    def reap(process):
        assert process.pid == FakeProcess.pid
        events.append("reap")

    def rebind(_closure):
        events.append("validate")

    monkeypatch.setattr(evaluator.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        evaluator,
        "_rebind_materializer_venv_launch",
        rebind_materializer_venv,
    )
    monkeypatch.setattr(
        evaluator.public_fixtures,
        "_terminate_worker_process_group",
        reap,
    )
    monkeypatch.setattr(evaluator, "_rebind_execution_closure", rebind)
    monkeypatch.setenv("SPEAKER_SECRET_MUST_NOT_REACH_WORKER", "secret")

    try:
        result = evaluator._materialize_source(
            SimpleNamespace(descriptor=source_fd),
            scratch=scratch,
            scratch_descriptor=scratch_fd,
            scratch_identity=scratch_identity,
            parquet_python=interpreter,
            execution_closure=execution_closure,
            scratch_registry=registry,
        )
        argv = captured["argv"]
        kwargs = captured["kwargs"]
        pass_fds = kwargs["pass_fds"]
        assert argv[0] == str(interpreter)
        assert argv[1:3] == ["-I", "-B"]
        assert argv[3].startswith("/proc/self/fd/")
        assert kwargs["executable"].startswith("/proc/self/fd/")
        assert kwargs["cwd"] == Path(f"/proc/self/fd/{pass_fds[-1]}")
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert kwargs["close_fds"] is True
        assert kwargs["start_new_session"] is True
        assert "shell" not in kwargs
        assert len(pass_fds) == 7
        assert pass_fds[0] == source_fd
        assert set(
            captured["request"]
        ) == materializer_worker._REQUEST_FIELDS
        assert captured["request"]["source_descriptor"] == source_fd
        leaf_identities = captured["request"]["output_leaf_identities"]
        expected_leaf_names = {
            *(f"row-{ordinal:04d}.pcm" for ordinal in range(400)),
            "manifest.json",
        }
        assert set(leaf_identities) == expected_leaf_names
        assert len(leaf_identities) == 401
        assert all(
            isinstance(identity, list)
            and len(identity) == 6
            and all(isinstance(item, int) and not isinstance(item, bool) for item in identity)
            for identity in leaf_identities.values()
        )
        for name, expected_identity in leaf_identities.items():
            current = os.fstat(
                registry.entries[("materialized", name)].descriptor
            )
            assert list(evaluator._stable_file_identity(current)) == expected_identity
        assert len(evaluator._canonical_json_bytes(captured["request"])) <= (
            evaluator._MAX_MATERIALIZER_REQUEST_BYTES
        )
        environment = kwargs["env"]
        assert "SPEAKER_SECRET_MUST_NOT_REACH_WORKER" not in environment
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "ARROW_NUM_THREADS",
        ):
            assert environment[name] == "1"
        assert events == [
            "venv-pre",
            "venv-pre",
            ("wait", evaluator._WORKER_TIMEOUT_SECONDS),
            "reap",
            "venv-post",
            "validate",
        ]
        assert validated_interpreters == [interpreter, interpreter]
        assert len(materializer_venv_rebinds) == 3
        assert materializer_venv_rebinds[0] == materializer_venv_rebinds[1]
        assert materializer_venv_rebinds[1] == materializer_venv_rebinds[2]
        assert result.output == scratch / "materialized"
        assert set(result.execution_receipt) == {
            "python_executable_sha256",
            "lexical_venv_argv0_preserved",
            "venv_marker_sha256",
            "pyarrow_version",
            "wall_timeout_seconds",
            "requested_cpu_soft_limit_ceiling_seconds",
            "requested_address_space_soft_limit_ceiling_bytes",
            "requested_file_size_soft_limit_ceiling_bytes",
            "requested_file_descriptor_soft_limit_ceiling",
            "inherited_hard_limits_may_reduce_ceilings",
            "worker_threads",
            "offline_environment",
            "network_namespace_isolation",
            "cgroup_scope",
        }
        assert result.execution_receipt["lexical_venv_argv0_preserved"] is True
        assert result.execution_receipt[
            "inherited_hard_limits_may_reduce_ceilings"
        ] is True
        for descriptor in pass_fds[1:]:
            with pytest.raises(OSError):
                os.fstat(descriptor)
        assert os.fstat(source_fd).st_size == len(b"descriptor-bound source")
    finally:
        os.close(source_fd)
        try:
            evaluator._cleanup_private_scratch(
                scratch,
                scratch_fd,
                scratch_identity,
                registry,
            )
        finally:
            _close_registry_and_assert_descriptors_closed(registry)
            os.close(scratch_fd)
    assert not scratch.exists()


def test_materializer_revalidates_snapshot_after_validation_to_snapshot_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent = _private_directory(tmp_path / "private-parent")
    scratch, scratch_fd, scratch_identity, registry = evaluator._create_private_scratch(
        parent / "scratch"
    )
    interpreter = Path(sys.executable)
    validation_calls: list[Path] = []
    snapshots: list[evaluator.VenvLaunchSnapshot] = []
    spawn_calls: list[object] = []
    original_snapshot = evaluator._snapshot_materializer_venv_launch

    def validate(path):
        validation_calls.append(Path(path))
        if len(validation_calls) == 1:
            return interpreter
        raise RuntimeError("replacement interpreter failed the import-root probe")

    def snapshot_after_replacement(path):
        snapshot = original_snapshot(path)
        snapshots.append(snapshot)
        return snapshot

    def forbidden_spawn(*args, **kwargs):
        spawn_calls.append((args, kwargs))
        raise AssertionError("replacement interpreter reached spawn")

    monkeypatch.setattr(
        evaluator.public_fixtures,
        "_validate_parquet_python",
        validate,
    )
    monkeypatch.setattr(
        evaluator,
        "_snapshot_materializer_venv_launch",
        snapshot_after_replacement,
    )
    monkeypatch.setattr(evaluator.subprocess, "Popen", forbidden_spawn)

    try:
        with pytest.raises(
            evaluator.LiveKitCausalEndpointError,
            match="materializer prerequisite failed",
        ):
            evaluator._materialize_source(
                SimpleNamespace(descriptor=-1),
                scratch=scratch,
                scratch_descriptor=scratch_fd,
                scratch_identity=scratch_identity,
                parquet_python=interpreter,
                execution_closure={},
                scratch_registry=registry,
            )
        assert len(snapshots) == 1
        assert validation_calls == [interpreter, snapshots[0].argv0]
        assert spawn_calls == []
    finally:
        try:
            evaluator._cleanup_private_scratch(
                scratch,
                scratch_fd,
                scratch_identity,
                registry,
            )
        finally:
            _close_registry_and_assert_descriptors_closed(registry)
            os.close(scratch_fd)
    assert not scratch.exists()


def test_materializer_precreation_failure_at_n_cleans_every_owned_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent = _private_directory(tmp_path / "private-parent")
    scratch, scratch_fd, scratch_identity, registry = evaluator._create_private_scratch(
        parent / "scratch"
    )
    source = tmp_path / "source.parquet"
    source.write_bytes(b"descriptor-bound source")
    source.chmod(0o600)
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    interpreter = Path(sys.executable)
    execution_closure = {
        "tools/livekit_causal_endpoint_worker.py": hashlib.sha256(
            evaluator._CAUSAL_WORKER_PATH.read_bytes()
        ).hexdigest(),
        "tools/livekit_eot_parquet_worker.py": hashlib.sha256(
            evaluator._INVENTORY_WORKER_PATH.read_bytes()
        ).hexdigest(),
    }
    monkeypatch.setattr(
        evaluator.public_fixtures,
        "_validate_parquet_python",
        lambda _path: interpreter,
    )
    real_create = evaluator._create_empty_private_file_at
    created_names: list[str] = []
    spawn_calls: list[object] = []

    def fail_at_seventh(directory_descriptor, name):
        if len(created_names) == 6:
            raise evaluator.LiveKitCausalEndpointError(
                "synthetic materialization precreation failure"
            )
        result = real_create(directory_descriptor, name)
        created_names.append(name)
        return result

    def forbidden_spawn(*args, **kwargs):
        spawn_calls.append((args, kwargs))
        raise AssertionError("worker spawned after precreation failure")

    monkeypatch.setattr(
        evaluator,
        "_create_empty_private_file_at",
        fail_at_seventh,
    )
    monkeypatch.setattr(evaluator.subprocess, "Popen", forbidden_spawn)

    try:
        with pytest.raises(evaluator.LiveKitCausalEndpointError):
            evaluator._materialize_source(
                SimpleNamespace(descriptor=source_fd),
                scratch=scratch,
                scratch_descriptor=scratch_fd,
                scratch_identity=scratch_identity,
                parquet_python=interpreter,
                execution_closure=execution_closure,
                scratch_registry=registry,
            )
        assert created_names == [f"row-{ordinal:04d}.pcm" for ordinal in range(6)]
        assert {path.name for path in (scratch / "materialized").iterdir()} == set(
            created_names
        )
        assert spawn_calls == []
    finally:
        os.close(source_fd)
        try:
            evaluator._cleanup_private_scratch(
                scratch,
                scratch_fd,
                scratch_identity,
                registry,
            )
        finally:
            _close_registry_and_assert_descriptors_closed(registry)
            os.close(scratch_fd)
    assert not scratch.exists()


def test_materializer_venv_snapshot_preserves_lexical_python_symlink(tmp_path: Path):
    target = tmp_path / "python3.12"
    target.write_bytes(b"interpreter")
    target.chmod(0o700)
    venv = tmp_path / "venv"
    binary_directory = venv / "bin"
    binary_directory.mkdir(parents=True)
    interpreter = binary_directory / "python"
    interpreter.symlink_to(target)
    (venv / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )

    snapshot = evaluator._snapshot_materializer_venv_launch(interpreter)

    assert snapshot.argv0 == interpreter
    assert snapshot.argv0.is_symlink()
    assert snapshot.executable_snapshot[1] == target
    assert snapshot.argv0 != snapshot.executable_snapshot[1]


@pytest.mark.parametrize(
    "failure",
    ["timeout", "nonzero", "venv-root", "venv-executable", "venv-marker"],
)
def test_materializer_supervisor_failure_reaps_closes_and_remains_cleanup_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
):
    parent = _private_directory(tmp_path / "private-parent")
    scratch, scratch_fd, scratch_identity, registry = evaluator._create_private_scratch(
        parent / "scratch"
    )
    source = tmp_path / "source.parquet"
    source.write_bytes(b"descriptor-bound source")
    source.chmod(0o600)
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    interpreter = Path(sys.executable)
    execution_closure = {
        "tools/livekit_causal_endpoint_worker.py": hashlib.sha256(
            evaluator._CAUSAL_WORKER_PATH.read_bytes()
        ).hexdigest(),
        "tools/livekit_eot_parquet_worker.py": hashlib.sha256(
            evaluator._INVENTORY_WORKER_PATH.read_bytes()
        ).hexdigest(),
    }
    monkeypatch.setattr(
        evaluator.public_fixtures,
        "_validate_parquet_python",
        lambda _path: interpreter,
    )
    captured_fds: tuple[int, ...] = ()
    partial_payload = b"partial-pcm-before-child-failure"
    reaped = []
    venv_changed = False
    lexical_interpreter = Path(os.path.abspath(interpreter))
    venv_root = lexical_interpreter.parent.parent
    marker_path = evaluator.public_fixtures._effective_venv_marker(
        lexical_interpreter,
        venv_root,
    )
    original_snapshot_venv_path = evaluator.public_fixtures._snapshot_venv_path
    original_read_venv_marker = evaluator.public_fixtures._read_valid_venv_marker

    def snapshot_venv_path(path, *, directory):
        value = original_snapshot_venv_path(path, directory=directory)
        if not venv_changed:
            return value
        if failure == "venv-root" and directory and Path(path) == venv_root:
            return ("replacement-root",)
        if (
            failure == "venv-executable"
            and not directory
            and Path(path) == lexical_interpreter
        ):
            return ("replacement-executable",)
        return value

    def read_venv_marker(path):
        payload, identity = original_read_venv_marker(path)
        if venv_changed and failure == "venv-marker" and Path(path) == marker_path:
            return payload + b"\nreplacement", identity
        return payload, identity

    monkeypatch.setattr(
        evaluator.public_fixtures,
        "_snapshot_venv_path",
        snapshot_venv_path,
    )
    monkeypatch.setattr(
        evaluator.public_fixtures,
        "_read_valid_venv_marker",
        read_venv_marker,
    )

    class FakeProcess:
        pid = 987_602

        def __init__(self, argv, **kwargs):
            nonlocal captured_fds
            self.returncode = None
            captured_fds = kwargs["pass_fds"]
            output_descriptor = int(argv[argv.index("--output-dir-fd") + 1])
            before = os.stat(
                "row-0000.pcm",
                dir_fd=output_descriptor,
                follow_symlinks=False,
            )
            assert before.st_size == 0
            partial_descriptor = os.open(
                "row-0000.pcm",
                os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=output_descriptor,
            )
            try:
                assert os.write(partial_descriptor, partial_payload) == len(partial_payload)
                os.fsync(partial_descriptor)
            finally:
                os.close(partial_descriptor)

        def wait(self, timeout=None):
            nonlocal venv_changed
            if failure == "timeout":
                raise subprocess.TimeoutExpired("worker", timeout)
            if failure.startswith("venv-"):
                venv_changed = True
                self.returncode = 0
                return 0
            self.returncode = 2
            return 2

    monkeypatch.setattr(evaluator.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        evaluator.public_fixtures,
        "_terminate_worker_process_group",
        lambda process: reaped.append(process.pid),
    )

    try:
        with pytest.raises(evaluator.LiveKitCausalEndpointError):
            evaluator._materialize_source(
                SimpleNamespace(descriptor=source_fd),
                scratch=scratch,
                scratch_descriptor=scratch_fd,
                scratch_identity=scratch_identity,
                parquet_python=interpreter,
                execution_closure=execution_closure,
                scratch_registry=registry,
            )
        assert reaped == [FakeProcess.pid]
        for descriptor in captured_fds[1:]:
            with pytest.raises(OSError):
                os.fstat(descriptor)
        assert os.fstat(source_fd).st_size > 0
        materialized = scratch / "materialized"
        assert (materialized / "row-0000.pcm").read_bytes() == partial_payload
        assert (materialized / "row-0001.pcm").stat().st_size == 0
        assert (materialized / "row-0399.pcm").stat().st_size == 0
        assert (materialized / "manifest.json").stat().st_size == 0
        assert {path.name for path in materialized.iterdir()} == {
            *(f"row-{ordinal:04d}.pcm" for ordinal in range(400)),
            "manifest.json",
        }
    finally:
        os.close(source_fd)
        try:
            evaluator._cleanup_private_scratch(
                scratch,
                scratch_fd,
                scratch_identity,
                registry,
            )
        finally:
            _close_registry_and_assert_descriptors_closed(registry)
            os.close(scratch_fd)
    assert not scratch.exists()


@pytest.mark.parametrize("raced_name", ["manifest.json", "row-0000.pcm"])
def test_model_worker_rejects_post_load_leaf_swap_without_scoring_or_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raced_name: str,
):
    directory = _private_directory(tmp_path / "materialized")
    pcm_payload = b"\x00\x00" * 6_400
    rows = [
        {
            "ordinal": 0,
            "pcm_filename": "row-0000.pcm",
            "pcm_bytes": len(pcm_payload),
            "pcm_samples": len(pcm_payload) // 2,
            "pcm_sha256": hashlib.sha256(pcm_payload).hexdigest(),
            "silence_spans": [[0, 3_200], [3_200, 6_400]],
        }
    ]
    manifest = {
        "rows": rows,
        "materialization_sha256": "a" * 64,
        "pcm_set_sha256": "b" * 64,
        "row_count": 1,
        "hold_label_count": 1,
    }
    manifest_payload = evaluator._canonical_json_bytes(manifest)
    owner_payloads = {
        "manifest.json": manifest_payload,
        "row-0000.pcm": pcm_payload,
    }
    for name, payload in owner_payloads.items():
        path = directory / name
        path.write_bytes(payload)
        path.chmod(0o600)
    directory_descriptor = _open_directory(directory)
    leaf_identities = {
        name: model_worker._stable_file_identity(path.lstat())
        for name, path in (
            (name, directory / name) for name in owner_payloads
        )
    }
    backup = tmp_path / f"{raced_name}.owner-backup"
    os.rename(directory / raced_name, backup)
    foreign = directory / raced_name
    foreign.write_bytes(owner_payloads[raced_name])
    foreign.chmod(0o600)
    detector_score_calls: list[object] = []
    result_writes: list[object] = []

    class FakeEndpointConfig:
        def __init__(self, **values):
            self.values = values

    class FakeSession:
        @staticmethod
        def get_providers():
            return ["CPUExecutionProvider"]

    class FakeDetector:
        needs_audio = True

        def __init__(self, *_args, **_kwargs):
            self._session = FakeSession()

        def load(self):
            return None

    endpointing_module = SimpleNamespace(
        EndpointConfig=FakeEndpointConfig,
        ProsodyTurnCompletionDetector=FakeDetector,
        evaluate_turn_completion=lambda **kwargs: detector_score_calls.append(kwargs),
    )
    monkeypatch.setattr(
        model_worker,
        "_exec_project_modules",
        lambda _request: (endpointing_module, {}),
    )
    monkeypatch.setattr(
        model_worker,
        "_bound_payload",
        lambda *_args, **_kwargs: (b"model", ("bound-model",)),
    )
    monkeypatch.setattr(
        model_worker,
        "_assert_max_wait_backstop",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_worker,
        "_write_result",
        lambda *args, **kwargs: result_writes.append((args, kwargs)),
    )
    request = {
        "schema_version": 1,
        "worker_descriptor": 101,
        "endpointing_descriptor": 102,
        "acoustic_descriptor": 103,
        "text_descriptor": 104,
        "model_descriptor": 105,
        "materialization_descriptor": directory_descriptor,
        "result_descriptor": 106,
        "code_sha256": {
            "worker": "c" * 64,
            "endpointing": "d" * 64,
            "acoustic": "e" * 64,
            "text": "f" * 64,
        },
        "config_sha256": "1" * 64,
        "model_size_bytes": 5,
        "model_sha256": "2" * 64,
        "materialization_manifest_sha256": hashlib.sha256(
            manifest_payload
        ).hexdigest(),
        "materialization_sha256": manifest["materialization_sha256"],
        "pcm_set_sha256": manifest["pcm_set_sha256"],
        "materialization_directory_identity": list(
            model_worker._stable_directory_identity(os.fstat(directory_descriptor))
        ),
        "materialization_leaf_identities": {
            name: list(identity) for name, identity in leaf_identities.items()
        },
        "endpoint_config": {
            "enabled": True,
            "min_silence_sec": 0.5,
            "max_silence_sec": 1.6,
            "complete_threshold": 0.6,
            "incomplete_threshold": 0.3,
            "high_confidence_floor": 0.6,
            "high_confidence_score": 0.75,
            "adaptive_floor": True,
            "pause_window": 64,
            "pause_quantile": 0.85,
            "pause_margin": 0.15,
            "pause_min_samples": 8,
        },
        "acoustic_rule2_samples": 12_800,
        "prosody_min_samples": 2_400,
        "max_wait_samples": 25_600,
        "rule3_samples": 320_000,
        "expected_hold_denominator": 1,
        "expected_eot_denominator": 1,
        "rows": rows,
    }
    assert set(request) == model_worker._REQUEST_FIELDS

    try:
        with pytest.raises(model_worker.ModelWorkerError):
            model_worker._run(request, request_sha256="3" * 64)
    finally:
        foreign.unlink()
        os.rename(backup, directory / raced_name)
        os.close(directory_descriptor)

    assert detector_score_calls == []
    assert result_writes == []
    assert not hasattr(model_worker, "_publish_report")
    assert {
        name: model_worker._stable_file_identity((directory / name).lstat())
        for name in owner_payloads
    } == leaf_identities


def test_model_supervisor_uses_descriptor_bundle_cpu_env_and_validates_after_reap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parent = _private_directory(tmp_path / "private-parent")
    scratch, scratch_fd, scratch_identity, registry = evaluator._create_private_scratch(
        parent / "scratch"
    )
    materialized = evaluator._make_private_subdirectory(
        scratch,
        "materialized",
        parent_descriptor=scratch_fd,
        registry=registry,
    )
    materialized_fd = _open_directory(materialized)
    try:
        manifest_payload = b'{"private":"manifest"}\n'
        pcm_payload = b"\x00\x00" * 3_200
        evaluator._write_new_private_file_at(
            materialized_fd,
            "manifest.json",
            manifest_payload,
            maximum=1024,
        )
        evaluator._write_new_private_file_at(
            materialized_fd,
            "row-0000.pcm",
            pcm_payload,
            maximum=len(pcm_payload),
        )
    finally:
        os.close(materialized_fd)
    evaluator._register_scratch_tree(scratch_fd, "materialized", registry)
    row = evaluator.MaterializedRow(
        ordinal=0,
        pcm_filename="row-0000.pcm",
        pcm_bytes=6_400,
        pcm_samples=3_200,
        pcm_sha256=hashlib.sha256(pcm_payload).hexdigest(),
        silence_spans=(evaluator.CausalSpan(0, 3_200),),
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    materialization = evaluator.Materialization(
        directory=materialized,
        directory_identity=evaluator._identity(materialized.lstat()),
        manifest_sha256=manifest_sha256,
        manifest={
            "materialization_sha256": "b" * 64,
            "pcm_set_sha256": "c" * 64,
        },
        rows=(row,),
        leaf_identities={
            name: evaluator._stable_file_identity(
                os.fstat(registry.entries[("materialized", name)].descriptor)
            )
            for name in ("manifest.json", "row-0000.pcm")
        },
        retained_directory_descriptor=registry.entries[("materialized",)].descriptor,
    )
    model_payload = b"synthetic exact model"
    model_path = scratch / evaluator._SMART_TURN_MODEL_FILENAME
    evaluator._write_new_private_file_at(
        scratch_fd,
        model_path.name,
        model_payload,
        maximum=len(model_payload),
        mode=0o400,
    )
    evaluator._register_scratch_tree(scratch_fd, model_path.name, registry)
    monkeypatch.setattr(evaluator, "_SMART_TURN_MODEL_BYTES", len(model_payload))
    monkeypatch.setattr(
        evaluator,
        "_SMART_TURN_MODEL_SHA256",
        hashlib.sha256(model_payload).hexdigest(),
    )
    monkeypatch.setattr(evaluator, "_EXPECTED_ROWS", 1)
    monkeypatch.setattr(evaluator, "_EXPECTED_HOLDS", 0)
    source_paths = {
        "tools/livekit_causal_endpoint_model_worker.py": evaluator._MODEL_WORKER_PATH,
        "core/endpointing.py": evaluator._ENDPOINTING_PATH,
        "always_on_agent/acoustic.py": evaluator._ACOUSTIC_PATH,
        "always_on_agent/text.py": evaluator._TEXT_PATH,
    }
    execution_closure = {
        relative: hashlib.sha256(path.read_bytes()).hexdigest()
        for relative, path in source_paths.items()
    }
    policy = evaluator.EndpointPolicyContract(
        endpoint_config={
            "enabled": True,
            "min_silence_sec": 0.5,
            "max_silence_sec": 1.6,
            "complete_threshold": 0.6,
            "incomplete_threshold": 0.3,
            "high_confidence_floor": 0.6,
            "high_confidence_score": 0.75,
            "adaptive_floor": True,
            "pause_window": 64,
            "pause_quantile": 0.85,
            "pause_margin": 0.15,
            "pause_min_samples": 8,
        },
        config_sha256="d" * 64,
        sample_rate_hz=16_000,
        grid_samples=1_600,
        acoustic_rule2_samples=12_800,
        prosody_min_samples=2_400,
        max_wait_samples=25_600,
        rule3_samples=320_000,
        runtime_default_detector="lexical",
    )
    captured: dict[str, object] = {}
    events: list[object] = []

    class FakeProcess:
        pid = 987_603

        def __init__(self, argv, **kwargs):
            self.returncode = None
            captured["argv"] = list(argv)
            captured["kwargs"] = kwargs
            request_fd = int(argv[argv.index("--request-fd") + 1])
            request_info = os.fstat(request_fd)
            request_payload = os.pread(request_fd, request_info.st_size, 0)
            request = json.loads(request_payload)
            captured["request"] = request
            candidate = _profile(
                partial_state=evaluator.PARTIAL_ASSUMPTION,
                state="scored",
                basis="semantic_early",
            )
            no_partial = _profile(
                partial_state=evaluator.NO_PARTIAL_PROFILE,
                state="no_partial",
                basis="acoustic",
            )
            result = {
                "schema_version": 1,
                "evaluation_contract": {
                    "request_sha256": hashlib.sha256(request_payload).hexdigest(),
                    "worker_sha256": request["code_sha256"]["worker"],
                    "endpointing_sha256": request["code_sha256"]["endpointing"],
                    "acoustic_sha256": request["code_sha256"]["acoustic"],
                    "text_sha256": request["code_sha256"]["text"],
                    "model_sha256": request["model_sha256"],
                    "config_sha256": request["config_sha256"],
                    "materialization_manifest_sha256": request[
                        "materialization_manifest_sha256"
                    ],
                    "materialization_sha256": request["materialization_sha256"],
                    "pcm_set_sha256": request["pcm_set_sha256"],
                },
                "candidate": {
                    "publisher_labels_fully_before_rule3": copy.deepcopy(candidate),
                    "semantic_counterfactual_full_source": candidate,
                },
                "acoustic_no_partial_fallback": {
                    "publisher_labels_fully_before_rule3": copy.deepcopy(no_partial),
                    "semantic_counterfactual_full_source": no_partial,
                },
                "runtime_receipt": {
                    "python_version": sys.version.split()[0],
                    "numpy_version": "2.0.0",
                    "onnxruntime_version": "1.0.0",
                    "provider": "CPUExecutionProvider",
                },
            }
            result_payload = evaluator._canonical_json_bytes(result)
            result_fd = request["result_descriptor"]
            assert os.pwrite(result_fd, result_payload, 0) == len(result_payload)
            os.fsync(result_fd)

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            self.returncode = 0
            return 0

    def reap(process):
        assert process.pid == FakeProcess.pid
        events.append("reap")

    original_validate = evaluator._validate_profile_result

    def validate_after_reap(*args, **kwargs):
        assert "reap" in events
        events.append("validate")
        return original_validate(*args, **kwargs)

    original_rebind_venv = evaluator._rebind_model_venv_launch
    venv_rebinds: list[evaluator.VenvLaunchSnapshot] = []

    def rebind_venv(snapshot):
        venv_rebinds.append(snapshot)
        original_rebind_venv(snapshot)

    monkeypatch.setattr(evaluator.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        evaluator.public_fixtures,
        "_terminate_worker_process_group",
        reap,
    )
    monkeypatch.setattr(evaluator, "_validate_profile_result", validate_after_reap)
    monkeypatch.setattr(evaluator, "_rebind_model_venv_launch", rebind_venv)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "owner-gpu-must-not-leak")
    monkeypatch.setenv("SPEAKER_SECRET_MUST_NOT_REACH_WORKER", "secret")

    try:
        result = evaluator._run_model_stage(
            materialization=materialization,
            model_snapshot=model_path,
            policy=policy,
            execution_closure=execution_closure,
            scratch=scratch,
            scratch_descriptor=scratch_fd,
            scratch_identity=scratch_identity,
            scratch_registry=registry,
        )
        argv = captured["argv"]
        kwargs = captured["kwargs"]
        pass_fds = kwargs["pass_fds"]
        assert argv[0] == sys.executable
        assert argv[1:3] == ["-I", "-B"]
        assert argv[3].startswith("/proc/self/fd/")
        assert kwargs["executable"].startswith("/proc/self/fd/")
        assert kwargs["cwd"] == Path(f"/proc/self/fd/{scratch_fd}")
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert kwargs["close_fds"] is True
        assert kwargs["start_new_session"] is True
        assert "shell" not in kwargs
        assert len(pass_fds) == 10
        assert pass_fds[-1] == scratch_fd
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert set(captured["request"]) == model_worker._REQUEST_FIELDS
        assert captured["request"]["materialization_leaf_identities"] == {
            name: list(identity)
            for name, identity in sorted(materialization.leaf_identities.items())
        }
        assert "SPEAKER_SECRET_MUST_NOT_REACH_WORKER" not in kwargs["env"]
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert kwargs["env"][name] == "1"
        assert events[0:2] == [
            ("wait", evaluator._MODEL_WORKER_TIMEOUT_SECONDS),
            "reap",
        ]
        assert events.count("validate") == 4
        assert len(venv_rebinds) == 2
        assert venv_rebinds[0] == venv_rebinds[1]
        assert set(result.execution_receipt) == {
            "source_bundle_tree_sha256",
            "python_executable_sha256",
            "lexical_venv_argv0_preserved",
            "venv_marker_sha256",
            "wall_timeout_seconds",
            "requested_cpu_soft_limit_ceiling_seconds",
            "requested_address_space_soft_limit_ceiling_bytes",
            "requested_file_size_soft_limit_ceiling_bytes",
            "requested_file_descriptor_soft_limit_ceiling",
            "inherited_hard_limits_may_reduce_ceilings",
            "worker_threads",
            "network_namespace_isolation",
            "offline_environment",
            "cgroup_scope",
            "runtime_receipt",
        }
        assert result.execution_receipt["lexical_venv_argv0_preserved"] is True
        assert result.execution_receipt[
            "inherited_hard_limits_may_reduce_ceilings"
        ] is True
        assert result.execution_receipt["runtime_receipt"]["provider"] == (
            "CPUExecutionProvider"
        )
        for descriptor in pass_fds[:-1]:
            with pytest.raises(OSError):
                os.fstat(descriptor)
        assert os.fstat(scratch_fd).st_mode
    finally:
        try:
            evaluator._cleanup_private_scratch(
                scratch,
                scratch_fd,
                scratch_identity,
                registry,
            )
        finally:
            _close_registry_and_assert_descriptors_closed(registry)
            os.close(scratch_fd)
    assert not scratch.exists()


@pytest.mark.parametrize("raced_name", ["manifest.json", "row-0000.pcm"])
def test_evaluator_rejects_self_consistent_foreign_materialization_inode_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raced_name: str,
):
    private_parent = _private_directory(tmp_path / "private-parent")
    scratch_tuple = evaluator._create_private_scratch(private_parent / "scratch")
    scratch, scratch_fd, _scratch_identity, registry = scratch_tuple
    materialized = evaluator._make_private_subdirectory(
        scratch,
        "materialized",
        parent_descriptor=scratch_fd,
        registry=registry,
    )
    materialized_fd = os.dup(registry.entries[("materialized",)].descriptor)
    owner_payloads = {
        "manifest.json": b'{"self_consistent":"owner"}\n',
        "row-0000.pcm": b"\x01\x00\x02\x00",
    }
    try:
        for name, payload in owner_payloads.items():
            descriptor, _identity_value = evaluator._create_empty_private_file_at(
                materialized_fd,
                name,
            )
            assert os.pwrite(descriptor, payload, 0) == len(payload)
            os.fsync(descriptor)
            info = os.stat(name, dir_fd=materialized_fd, follow_symlinks=False)
            evaluator._register_open_scratch_descriptor(
                registry,
                ("materialized", name),
                descriptor,
                is_directory=False,
                path_info=info,
            )
    finally:
        os.close(materialized_fd)
    retained_descriptors = tuple(
        entry.descriptor for entry in registry.entries.values()
    )
    report_parent = _private_directory(tmp_path / "reports")
    destination = report_parent / "aggregate.json"
    backup_path = private_parent / f"{raced_name}.owner-backup"
    parser_calls: list[object] = []
    publish_calls: list[object] = []
    real_cleanup = evaluator._cleanup_private_scratch

    @contextmanager
    def opened_source(_path):
        yield SimpleNamespace(descriptor=91)

    def materialize_then_swap(*_args, **_kwargs):
        directory_descriptor = registry.entries[("materialized",)].descriptor
        os.rename(
            raced_name,
            backup_path,
            src_dir_fd=directory_descriptor,
        )
        foreign_descriptor = os.open(
            raced_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        try:
            payload = owner_payloads[raced_name]
            assert os.write(foreign_descriptor, payload) == len(payload)
            os.fsync(foreign_descriptor)
        finally:
            os.close(foreign_descriptor)
        return evaluator.MaterializerStageResult(
            output=materialized,
            execution_receipt={},
        )

    def forbidden_parse(value):
        parser_calls.append(value)
        raise AssertionError("foreign materialization reached manifest parsing")

    def restore_then_cleanup(path, descriptor, identity, scratch_registry):
        directory_descriptor = scratch_registry.entries[("materialized",)].descriptor
        assert evaluator._read_private_regular_at(
            directory_descriptor,
            raced_name,
            maximum=1024,
        )[0] == owner_payloads[raced_name]
        os.unlink(raced_name, dir_fd=directory_descriptor)
        os.rename(
            backup_path,
            raced_name,
            dst_dir_fd=directory_descriptor,
        )
        real_cleanup(path, descriptor, identity, scratch_registry)

    def forbidden_publish(*args, **kwargs):
        publish_calls.append((args, kwargs))
        raise AssertionError("foreign materialization reached publication")

    monkeypatch.setattr(evaluator, "_snapshot_execution_closure", lambda: {})
    monkeypatch.setattr(evaluator, "_validate_inventory_receipt", lambda _path: {})
    monkeypatch.setattr(evaluator, "_load_policy_contract", lambda _path: object())
    monkeypatch.setattr(
        evaluator.inventory_source,
        "_output_path",
        lambda _path: (destination, report_parent.lstat()),
    )
    monkeypatch.setattr(evaluator, "_require_scratch_registry_capacity", lambda: None)
    monkeypatch.setattr(evaluator, "_create_private_scratch", lambda _path: scratch_tuple)
    monkeypatch.setattr(
        evaluator,
        "_snapshot_exact_model",
        lambda *_args, **_kwargs: (scratch / "model.onnx", {}),
    )
    monkeypatch.setattr(evaluator.inventory_source, "_opened_source", opened_source)
    monkeypatch.setattr(
        evaluator.inventory_source,
        "_verify_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(evaluator, "_materialize_source", materialize_then_swap)
    monkeypatch.setattr(evaluator, "_strict_json", forbidden_parse)
    monkeypatch.setattr(evaluator, "_cleanup_private_scratch", restore_then_cleanup)
    monkeypatch.setattr(evaluator, "_publish_report", forbidden_publish)

    with pytest.raises(evaluator.LiveKitCausalEndpointError):
        evaluator.evaluate_livekit_causal_endpoint(
            source_parquet=tmp_path / "source.parquet",
            inventory_report=tmp_path / "inventory.json",
            parquet_python=sys.executable,
            model=tmp_path / "model.onnx",
            config=tmp_path / "config.json",
            scratch_root=scratch,
            output=destination,
            accepted_terms=frozenset({evaluator.LICENSE}),
            accept_partial_assumption=True,
        )

    assert parser_calls == []
    assert publish_calls == []
    assert not destination.exists()
    assert not scratch.exists()
    for descriptor in retained_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_final_report_is_aggregate_only_and_names_publisher_label_censoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    private_parent = _private_directory(tmp_path / "private-parent")
    scratch_tuple = evaluator._create_private_scratch(private_parent / "scratch")
    scratch, scratch_fd, scratch_identity, registry = scratch_tuple
    retained_materialized = evaluator._make_private_subdirectory(
        scratch,
        "materialized",
        parent_descriptor=scratch_fd,
        registry=registry,
    )
    report_parent = _private_directory(tmp_path / "reports")
    destination = report_parent / "aggregate.json"
    source = tmp_path / "private-source-path-SOURCE_SENTINEL.parquet"
    inventory = tmp_path / "private-inventory-path-INVENTORY_SENTINEL.json"
    model = tmp_path / "private-model-path-MODEL_SENTINEL.onnx"
    config = tmp_path / "private-config-path-CONFIG_SENTINEL.json"
    policy = evaluator.EndpointPolicyContract(
        endpoint_config={
            "enabled": True,
            "min_silence_sec": 0.5,
            "max_silence_sec": 1.6,
            "complete_threshold": 0.6,
            "incomplete_threshold": 0.3,
            "high_confidence_floor": 0.6,
            "high_confidence_score": 0.75,
            "adaptive_floor": True,
            "pause_window": 64,
            "pause_quantile": 0.85,
            "pause_margin": 0.15,
            "pause_min_samples": 8,
        },
        config_sha256="a" * 64,
        sample_rate_hz=16_000,
        grid_samples=1_600,
        acoustic_rule2_samples=12_800,
        prosody_min_samples=2_400,
        max_wait_samples=25_600,
        rule3_samples=320_000,
        runtime_default_detector="lexical",
    )
    candidate_profile = _profile(
        partial_state=evaluator.PARTIAL_ASSUMPTION,
        state="scored",
        basis="semantic_early",
        holds=0,
        eot=1,
    )
    no_partial_profile = _profile(
        partial_state=evaluator.NO_PARTIAL_PROFILE,
        state="no_partial",
        basis="acoustic",
        holds=0,
        eot=1,
    )
    candidate = {
        "publisher_labels_fully_before_rule3": copy.deepcopy(candidate_profile),
        "semantic_counterfactual_full_source": candidate_profile,
    }
    no_partial = {
        "publisher_labels_fully_before_rule3": copy.deepcopy(no_partial_profile),
        "semantic_counterfactual_full_source": no_partial_profile,
    }
    row = evaluator.MaterializedRow(
        ordinal=0,
        pcm_filename="row-0000.pcm",
        pcm_bytes=6_400,
        pcm_samples=3_200,
        pcm_sha256="b" * 64,
        silence_spans=(evaluator.CausalSpan(0, 3_200),),
    )
    materialization = evaluator.Materialization(
        directory=retained_materialized,
        directory_identity=(1,),
        manifest_sha256="c" * 64,
        manifest={
            "materialization_sha256": "d" * 64,
            "pcm_set_sha256": "e" * 64,
            "row_count": 1,
            "silence_span_count": 1,
            "hold_label_count": 0,
            "eot_label_count": 1,
            "off_grid_span_count": 0,
            "final_gap_zero_sample_count": 1,
            "final_gap_one_sample_count": 0,
        },
        rows=(row,),
    )
    model_stage = evaluator.ModelStageResult(
        candidate=candidate,
        acoustic_no_partial_fallback=no_partial,
        execution_receipt={
            "source_bundle_tree_sha256": "f" * 64,
            "python_executable_sha256": "1" * 64,
            "lexical_venv_argv0_preserved": True,
            "requested_cpu_soft_limit_ceiling_seconds": 1_800,
            "inherited_hard_limits_may_reduce_ceilings": True,
            "worker_threads": 1,
            "runtime_receipt": {
                "python_version": "3.12.0",
                "numpy_version": "2.0.0",
                "onnxruntime_version": "1.0.0",
                "provider": "CPUExecutionProvider",
            },
        },
    )
    materializer_stage = evaluator.MaterializerStageResult(
        output=materialization.directory,
        execution_receipt={
            "python_executable_sha256": "2" * 64,
            "lexical_venv_argv0_preserved": True,
            "requested_cpu_soft_limit_ceiling_seconds": 600,
            "inherited_hard_limits_may_reduce_ceilings": True,
            "worker_threads": 1,
        },
    )
    captured: dict[str, object] = {}

    @contextmanager
    def opened_source(_path):
        yield SimpleNamespace(descriptor=91)

    def publish(path, parent_info, payload, *, rebind):
        assert path == destination
        assert evaluator._directory_entry_identity(parent_info) == (
            evaluator._directory_entry_identity(report_parent.lstat())
        )
        captured["payload"] = payload
        captured["rebind"] = rebind
        return hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(evaluator, "_snapshot_execution_closure", lambda: {"safe.py": "3" * 64})
    monkeypatch.setattr(evaluator, "_validate_inventory_receipt", lambda _path: {})
    monkeypatch.setattr(evaluator, "_load_policy_contract", lambda _path: policy)
    monkeypatch.setattr(
        evaluator.inventory_source,
        "_output_path",
        lambda _path: (destination, report_parent.lstat()),
    )
    monkeypatch.setattr(evaluator, "_create_private_scratch", lambda _path: scratch_tuple)
    monkeypatch.setattr(
        evaluator,
        "_snapshot_exact_model",
        lambda *_args, **_kwargs: (
            scratch / "private-model-snapshot-SNAPSHOT_SENTINEL.onnx",
            {
                "model_id": "safe-model",
                "revision": "safe-revision",
                "filename": "safe-model.onnx",
                "size_bytes": 1,
                "sha256": "4" * 64,
                "license": "BSD-2-Clause",
                "provider": "CPUExecutionProvider",
                "threads": 1,
            },
        ),
    )
    monkeypatch.setattr(evaluator, "_register_scratch_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluator.inventory_source, "_opened_source", opened_source)
    monkeypatch.setattr(evaluator.inventory_source, "_verify_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        evaluator,
        "_materialize_source",
        lambda *_args, **_kwargs: materializer_stage,
    )
    monkeypatch.setattr(evaluator, "_load_materialization", lambda *_args, **_kwargs: materialization)
    monkeypatch.setattr(evaluator, "_run_model_stage", lambda **_kwargs: model_stage)
    monkeypatch.setattr(evaluator, "_rebind_materialization", lambda _value: None)
    monkeypatch.setattr(evaluator, "_rebind_execution_closure", lambda _value: None)
    monkeypatch.setattr(evaluator, "_cleanup_private_scratch", lambda *_args: None)
    monkeypatch.setattr(evaluator, "_publish_report", publish)

    result = evaluator.evaluate_livekit_causal_endpoint(
        source_parquet=source,
        inventory_report=inventory,
        parquet_python=tmp_path / "private-python-PYTHON_SENTINEL",
        model=model,
        config=config,
        scratch_root=scratch,
        output=destination,
        accepted_terms=frozenset({"CC-BY-4.0"}),
        accept_partial_assumption=True,
    )

    report = json.loads(captured["payload"])
    serialized = captured["payload"].decode("ascii").casefold()
    assert result.hold_denominator == 0
    assert result.eot_denominator == 1
    assert report["protocol"]["eot_censoring"].startswith(
        "stop_model_scoring_at_publisher_labelled_final_silence_end"
    )
    assert "recorded pcm end" not in serialized
    assert "materialization_leaf_identities" not in serialized
    for sentinel in (
        "source_sentinel",
        "inventory_sentinel",
        "model_sentinel",
        "config_sentinel",
        "materialized_sentinel",
        "snapshot_sentinel",
        "row-0000.pcm",
    ):
        assert sentinel.casefold() not in serialized

    forbidden_keys = {
        "rows",
        "ordinal",
        "pcm_filename",
        "silence_spans",
        "materialization_leaf_identities",
    }

    def walk(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(report)


@pytest.mark.slow
@pytest.mark.backend
def test_pinned_livekit_source_materializes_with_aggregate_contract_only(
    tmp_path: Path,
):
    configured = {
        "source": os.environ.get("SPEAKER_LIVEKIT_SOURCE_PARQUET"),
        "inventory": os.environ.get("SPEAKER_LIVEKIT_INVENTORY_REPORT"),
        "python": os.environ.get("SPEAKER_LIVEKIT_PARQUET_PYTHON"),
    }
    if any(not value for value in configured.values()):
        pytest.skip(
            "set SPEAKER_LIVEKIT_SOURCE_PARQUET, SPEAKER_LIVEKIT_INVENTORY_REPORT, "
            "and SPEAKER_LIVEKIT_PARQUET_PYTHON for the pinned-source materializer gate"
        )
    source_path = Path(str(configured["source"])).expanduser().resolve(strict=True)
    inventory_path = Path(str(configured["inventory"])).expanduser().resolve(
        strict=True
    )
    parquet_python = Path(
        os.path.abspath(Path(str(configured["python"])).expanduser())
    )
    evaluator._validate_inventory_receipt(inventory_path)
    evaluator._require_scratch_registry_capacity()
    execution_closure = evaluator._snapshot_execution_closure()
    private_parent = _private_directory(tmp_path / "private-parent")
    scratch, scratch_fd, scratch_identity, registry = evaluator._create_private_scratch(
        private_parent / "scratch"
    )
    retained: tuple[int, ...] = ()
    try:
        with evaluator.inventory_source._opened_source(source_path) as source:
            evaluator.inventory_source._verify_source(source, hash_content=True)
            stage = evaluator._materialize_source(
                source,
                scratch=scratch,
                scratch_descriptor=scratch_fd,
                scratch_identity=scratch_identity,
                parquet_python=parquet_python,
                execution_closure=execution_closure,
                scratch_registry=registry,
            )
            evaluator.inventory_source._verify_source(source, hash_content=True)
        materialization = evaluator._load_materialization(
            stage.output,
            execution_closure,
            retained_directory_descriptor=registry.entries[("materialized",)].descriptor,
            retained_leaf_entries={
                relative[-1]: entry
                for relative, entry in registry.entries.items()
                if len(relative) == 2 and relative[0] == "materialized"
            },
        )
        assert len(materialization.rows) == evaluator._EXPECTED_ROWS == 400
        assert materialization.manifest["row_count"] == 400
        assert materialization.manifest["pcm_samples_total"] == (
            evaluator._EXPECTED_PCM_SAMPLES
        )
        assert materialization.manifest["silence_span_count"] == (
            evaluator._EXPECTED_SPANS
        )
        assert materialization.manifest["hold_label_count"] == (
            evaluator._EXPECTED_HOLDS
        )
        assert materialization.manifest["eot_label_count"] == 400
        assert stage.execution_receipt["worker_threads"] == 1
        assert stage.execution_receipt["offline_environment"] is True
    finally:
        retained = tuple(entry.descriptor for entry in registry.entries.values())
        try:
            evaluator._cleanup_private_scratch(
                scratch,
                scratch_fd,
                scratch_identity,
                registry,
            )
        finally:
            _close_registry_and_assert_descriptors_closed(registry, retained)
            os.close(scratch_fd)
    assert not scratch.exists()


@pytest.mark.real_model
def test_exact_smart_turn_model_loads_from_proc_fd_on_one_cpu_thread():
    candidates = []
    configured = os.environ.get("SPEAKER_SMART_TURN_MODEL")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            Path("/tmp/speaker-smart-turn-v3.2-cpu.onnx"),
            Path("pretrained_models/sherpa/turn/smart-turn-v3.2-cpu.onnx"),
        )
    )
    model = next((path for path in candidates if path.exists()), None)
    if model is None:
        pytest.skip("exact Smart Turn v3.2 model is not present")
    model = model.resolve(strict=True)
    info = model.lstat()
    assert stat.S_ISREG(info.st_mode)
    assert info.st_nlink == 1
    assert info.st_size == evaluator._SMART_TURN_MODEL_BYTES
    digest = hashlib.sha256()
    with model.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == evaluator._SMART_TURN_MODEL_SHA256

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(model, flags)
    try:
        detector = endpointing.ProsodyTurnCompletionDetector(
            f"/proc/self/fd/{descriptor}",
            num_threads=1,
        )
        detector.load()
        assert detector._session.get_providers() == ["CPUExecutionProvider"]
        options = detector._session.get_session_options()
        assert options.intra_op_num_threads == 1
        assert options.inter_op_num_threads == 1
        assert str(options.execution_mode).endswith("ORT_SEQUENTIAL")
    finally:
        os.close(descriptor)
