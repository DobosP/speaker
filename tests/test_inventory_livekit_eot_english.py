from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

from tools import inventory_livekit_eot_english as inventory


def _patch_tiny_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    source = tmp_path / "source.parquet"
    payload = b"synthetic-parquet-shaped-fixture"
    source.write_bytes(payload)
    source.chmod(0o600)
    monkeypatch.setattr(inventory, "SOURCE_SIZE_BYTES", len(payload))
    monkeypatch.setattr(inventory, "SOURCE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(inventory, "SOURCE_ROWS", 3)
    monkeypatch.setattr(inventory, "DURATION_EXACT_SAMPLE_COUNT", 2)
    monkeypatch.setattr(inventory, "DURATION_MICROSECOND_QUANTIZED_COUNT", 1)
    monkeypatch.setattr(inventory, "FINAL_GAP_ZERO_SAMPLE_COUNT", 2)
    monkeypatch.setattr(inventory, "FINAL_GAP_ONE_SAMPLE_COUNT", 1)
    return source


def _worker_result() -> dict[str, object]:
    return {
        "schema_version": inventory.PROTOCOL_VERSION,
        "pyarrow_version": inventory.PYARROW_VERSION,
        "source_revision": inventory.SOURCE_REVISION,
        "source_size_bytes": inventory.SOURCE_SIZE_BYTES,
        "source_sha256": inventory.SOURCE_SHA256,
        "schema_contract_sha256": inventory.SCHEMA_CONTRACT_SHA256,
        "row_count": inventory.SOURCE_ROWS,
        "row_group_count": 1,
        "audio_file_count": inventory.SOURCE_ROWS,
        "audio_bytes_total": 300,
        "audio_samples_total": 48_000,
        "silence_span_count": 5,
        "hold_label_count": 2,
        "eot_label_count": inventory.SOURCE_ROWS,
        "minimum_eot_silence_samples": 3_200,
        "maximum_silence_span_samples": 4_000,
        "duration_exact_sample_count": inventory.DURATION_EXACT_SAMPLE_COUNT,
        "duration_microsecond_quantized_count": (
            inventory.DURATION_MICROSECOND_QUANTIZED_COUNT
        ),
        "final_gap_zero_sample_count": inventory.FINAL_GAP_ZERO_SAMPLE_COUNT,
        "final_gap_one_sample_count": inventory.FINAL_GAP_ONE_SAMPLE_COUNT,
        "label_convention": inventory.LABEL_CONVENTION,
        "audio_set_sha256": "a" * 64,
        "silence_set_sha256": "b" * 64,
        "inventory_sha256": "c" * 64,
    }


def _injection(
    mutate: object | None = None,
) -> inventory.TestWorkerInjection:
    def run(_descriptor: int, output: Path) -> None:
        if callable(mutate):
            mutate()
        result = output / "result.json"
        result.write_text(
            json.dumps(_worker_result(), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        result.chmod(0o400)

    return inventory.TestWorkerInjection(run=run, fixture="synthetic-parquet-v1")


def _private_output_parent(tmp_path: Path) -> Path:
    output_parent = tmp_path / "private-output"
    output_parent.mkdir(mode=0o700)
    output_parent.chmod(0o700)
    return output_parent


def _mark_git_repository(root: Path, *, linked_worktree: bool) -> None:
    marker = root / ".git"
    if linked_worktree:
        marker.write_text("gitdir: /private/git/worktrees/synthetic\n", encoding="utf-8")
        marker.chmod(0o600)
        return
    marker.mkdir(mode=0o700)
    head = marker / "HEAD"
    head.write_text("ref: refs/heads/main\n", encoding="utf-8")
    head.chmod(0o600)


def _make_inert_git_sentinel(root: Path, sentinel_kind: str) -> Path:
    marker = root / ".git"
    if sentinel_kind == "empty-file":
        marker.touch(mode=0o600)
        marker.chmod(0o600)
    elif sentinel_kind == "empty-directory":
        marker.mkdir(mode=0o700)
    else:  # pragma: no cover - helper contract
        raise AssertionError(sentinel_kind)
    return marker


def _promote_inert_git_sentinel(root: Path, mutation: str) -> None:
    marker = root / ".git"
    if mutation == "grow-empty-file":
        marker.write_text("gitdir: /private/git/worktrees/synthetic\n", encoding="utf-8")
        marker.chmod(0o600)
    elif mutation == "grow-empty-directory":
        head = marker / "HEAD"
        head.write_text("ref: refs/heads/main\n", encoding="utf-8")
        head.chmod(0o600)
    elif mutation == "replace-empty-file":
        marker.unlink()
        marker.mkdir(mode=0o700)
        head = marker / "HEAD"
        head.write_text("ref: refs/heads/main\n", encoding="utf-8")
        head.chmod(0o600)
    elif mutation == "replace-empty-directory":
        marker.rmdir()
        marker.write_text(
            "gitdir: /private/git/worktrees/synthetic\n",
            encoding="utf-8",
        )
        marker.chmod(0o600)
    else:  # pragma: no cover - helper contract
        raise AssertionError(mutation)


def _drifted_execution_closure(value: dict[str, str]) -> dict[str, str]:
    changed = dict(value)
    first = inventory._EXECUTION_CLOSURE_RELATIVE_PATHS[0]
    changed[first] = "f" * 64 if changed[first] != "f" * 64 else "e" * 64
    return changed


def test_synthetic_inventory_is_private_aggregate_only_and_nonproduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    output = _private_output_parent(tmp_path) / "inventory.json"

    result = inventory.inventory_livekit_eot_english(
        source_parquet=source,
        parquet_python=None,
        output=output,
        accepted_terms=frozenset({"CC-BY-4.0"}),
        test_worker_injection=_injection(),
    )

    assert result.row_count == 3
    assert result.silence_span_count == 5
    assert result.production_evidence is False
    assert output.stat().st_mode & 0o777 == 0o600
    payload = output.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == result.report_sha256
    report = json.loads(payload)
    assert report["evaluation_scope"] == "endpoint-turn-taking-only-no-wer"
    assert report["attribution"] == {
        "creator": "LiveKit",
        "source_url": "https://huggingface.co/datasets/livekit/eot-bench-data",
        "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
        "use_note": (
            "Aggregate endpoint validation only; source bytes are not rehosted or "
            "modified."
        ),
    }
    assert report["worker"]["production_evidence"] is False
    expected_closure = {
        relative_path: hashlib.sha256(
            (inventory._REPO_ROOT / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in inventory._EXECUTION_CLOSURE_RELATIVE_PATHS
    }
    assert report["execution_closure"] == expected_closure
    assert report["inventory"]["hold_label_count"] == 2
    assert report["inventory"]["eot_label_count"] == 3
    assert report["inventory"]["duration_exact_sample_count"] == 2
    assert report["inventory"]["duration_microsecond_quantized_count"] == 1
    assert report["inventory"]["final_gap_zero_sample_count"] == 2
    assert report["inventory"]["final_gap_one_sample_count"] == 1
    serialized = payload.decode("ascii")
    for private_value in (
        str(source),
        str(output),
        str(inventory._REPO_ROOT),
        "message",
        "transcript",
        "words",
    ):
        assert private_value not in serialized


def test_execution_closure_reads_exact_files_before_and_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    output = _private_output_parent(tmp_path) / "inventory.json"
    original_reader = inventory.public_fixtures._read_stable_repo_worker
    reads: list[Path] = []

    def recording_reader(path: Path) -> bytes:
        reads.append(path)
        return original_reader(path)

    monkeypatch.setattr(
        inventory.public_fixtures,
        "_read_stable_repo_worker",
        recording_reader,
    )

    inventory.inventory_livekit_eot_english(
        source_parquet=source,
        parquet_python=None,
        output=output,
        accepted_terms=frozenset({"CC-BY-4.0"}),
        test_worker_injection=_injection(),
    )

    expected_pass = [
        inventory._REPO_ROOT / relative_path
        for relative_path in inventory._EXECUTION_CLOSURE_RELATIVE_PATHS
    ]
    assert reads == expected_pass * 3
    report = json.loads(output.read_bytes())
    assert tuple(report["execution_closure"]) == (
        inventory._EXECUTION_CLOSURE_RELATIVE_PATHS
    )


def test_production_worker_hash_must_equal_execution_closure_worker_hash():
    closure = inventory._snapshot_execution_closure()
    worker_hash = closure["tools/livekit_eot_parquet_worker.py"]

    assert (
        inventory._validate_execution_worker_binding(
            {"production_evidence": True, "worker_sha256": worker_hash},
            closure,
        )
        is True
    )
    assert (
        inventory._validate_execution_worker_binding(
            {"production_evidence": False},
            closure,
        )
        is False
    )
    with pytest.raises(inventory.LiveKitEotInventoryError, match="execution closure"):
        inventory._validate_execution_worker_binding(
            {"production_evidence": True, "worker_sha256": "0" * 64},
            closure,
        )


def test_execution_closure_drift_before_publication_fails_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    output = _private_output_parent(tmp_path) / "inventory.json"
    initial = inventory._snapshot_execution_closure()
    changed = _drifted_execution_closure(initial)
    calls = 0

    def drifting_snapshot() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return dict(initial if calls == 1 else changed)

    monkeypatch.setattr(inventory, "_snapshot_execution_closure", drifting_snapshot)

    with pytest.raises(inventory.LiveKitEotInventoryError, match="execution closure"):
        inventory.inventory_livekit_eot_english(
            source_parquet=source,
            parquet_python=None,
            output=output,
            accepted_terms=frozenset({"CC-BY-4.0"}),
            test_worker_injection=_injection(),
        )
    assert calls == 2
    assert not output.exists()


def test_execution_closure_drift_after_publication_fails_without_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    output = _private_output_parent(tmp_path) / "inventory.json"
    initial = inventory._snapshot_execution_closure()
    changed = _drifted_execution_closure(initial)
    calls = 0

    def drifting_snapshot() -> dict[str, str]:
        nonlocal calls
        calls += 1
        if calls < 3:
            assert not output.exists()
            return dict(initial)
        assert output.is_file()
        return dict(changed)

    monkeypatch.setattr(inventory, "_snapshot_execution_closure", drifting_snapshot)

    with pytest.raises(inventory.LiveKitEotInventoryError, match="execution closure"):
        inventory.inventory_livekit_eot_english(
            source_parquet=source,
            parquet_python=None,
            output=output,
            accepted_terms=frozenset({"CC-BY-4.0"}),
            test_worker_injection=_injection(),
        )
    assert calls == 3
    assert output.is_file()
    report = json.loads(output.read_bytes())
    assert report["execution_closure"] == initial


def test_source_and_output_fail_closed_on_license_permissions_links_and_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    output_parent = _private_output_parent(tmp_path)
    output = output_parent / "inventory.json"
    kwargs = {
        "source_parquet": source,
        "parquet_python": None,
        "output": output,
        "accepted_terms": frozenset({"CC-BY-4.0"}),
        "test_worker_injection": _injection(),
    }

    with pytest.raises(inventory.LiveKitEotInventoryError, match="license"):
        inventory.inventory_livekit_eot_english(
            **{**kwargs, "accepted_terms": frozenset()}
        )

    source.chmod(0o644)
    with pytest.raises(inventory.LiveKitEotInventoryError, match="source"):
        inventory.inventory_livekit_eot_english(**kwargs)
    source.chmod(0o600)

    hardlink = tmp_path / "source-hard.parquet"
    os.link(source, hardlink)
    with pytest.raises(inventory.LiveKitEotInventoryError, match="source"):
        inventory.inventory_livekit_eot_english(**kwargs)
    hardlink.unlink()

    symlink = tmp_path / "source-link.parquet"
    symlink.symlink_to(source)
    with pytest.raises(inventory.LiveKitEotInventoryError, match="path"):
        inventory.inventory_livekit_eot_english(
            **{**kwargs, "source_parquet": symlink}
        )

    output.write_text("existing", encoding="utf-8")
    with pytest.raises(inventory.LiveKitEotInventoryError, match="output"):
        inventory.inventory_livekit_eot_english(**kwargs)
    assert output.read_text(encoding="utf-8") == "existing"


def test_output_created_during_worker_is_preserved_and_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    output = _private_output_parent(tmp_path) / "inventory.json"

    def race() -> None:
        output.write_text("racing-owner", encoding="utf-8")
        output.chmod(0o600)

    with pytest.raises(inventory.LiveKitEotInventoryError, match="publication"):
        inventory.inventory_livekit_eot_english(
            source_parquet=source,
            parquet_python=None,
            output=output,
            accepted_terms=frozenset({"CC-BY-4.0"}),
            test_worker_injection=_injection(race),
        )
    assert output.read_text(encoding="utf-8") == "racing-owner"


def test_output_destination_cannot_create_a_git_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    output = _private_output_parent(tmp_path) / ".git"
    with pytest.raises(inventory.LiveKitEotInventoryError, match="output"):
        inventory.inventory_livekit_eot_english(
            source_parquet=source,
            parquet_python=None,
            output=output,
            accepted_terms=frozenset({"CC-BY-4.0"}),
            test_worker_injection=_injection(),
        )
    assert not output.exists()


@pytest.mark.parametrize("sentinel_kind", ["empty-file", "empty-directory"])
def test_source_and_output_accept_stable_inert_git_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sentinel_kind: str,
):
    sentinel_root = tmp_path / sentinel_kind
    sentinel_root.mkdir(mode=0o700)
    _make_inert_git_sentinel(sentinel_root, sentinel_kind)
    source = _patch_tiny_source(sentinel_root, monkeypatch)
    output = _private_output_parent(sentinel_root) / "inventory.json"

    result = inventory.inventory_livekit_eot_english(
        source_parquet=source,
        parquet_python=None,
        output=output,
        accepted_terms=frozenset({"CC-BY-4.0"}),
        test_worker_injection=_injection(),
    )

    assert result.production_evidence is False
    assert output.is_file()


def test_git_probe_rejects_gitfile_growth_in_same_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cache = tmp_path / "growing-gitfile"
    cache.mkdir(mode=0o700)
    marker = _make_inert_git_sentinel(cache, "empty-file")
    original_stat = inventory.os.stat
    mutated = False

    def growing_stat(path, *args, **kwargs):
        nonlocal mutated
        result = original_stat(path, *args, **kwargs)
        if path == ".git" and kwargs.get("dir_fd") is not None and not mutated:
            marker.write_text(
                "gitdir: /private/git/worktrees/synthetic\n",
                encoding="utf-8",
            )
            marker.chmod(0o600)
            mutated = True
        return result

    monkeypatch.setattr(inventory.os, "stat", growing_stat)

    with inventory.opened_directory_nofollow(cache) as (_bound, descriptor):
        with pytest.raises(
            inventory.LiveKitEotInventoryError,
            match="Git boundary verification",
        ):
            inventory._git_marker_present(descriptor)
    assert mutated is True


def test_git_probe_rejects_marker_created_after_missing_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cache = tmp_path / "appearing-marker"
    cache.mkdir(mode=0o700)
    marker = cache / ".git"
    original_stat = inventory.os.stat
    mutated = False

    def creating_stat(path, *args, **kwargs):
        nonlocal mutated
        try:
            return original_stat(path, *args, **kwargs)
        except FileNotFoundError:
            if path == ".git" and kwargs.get("dir_fd") is not None and not mutated:
                marker.write_text(
                    "gitdir: /private/git/worktrees/synthetic\n",
                    encoding="utf-8",
                )
                marker.chmod(0o600)
                mutated = True
            raise

    monkeypatch.setattr(inventory.os, "stat", creating_stat)

    with pytest.raises(
        inventory.LiveKitEotInventoryError,
        match="Git boundary verification",
    ):
        inventory._has_git_ancestor(cache)
    assert mutated is True


def test_git_probe_rejects_head_added_to_inert_directory_in_same_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cache = tmp_path / "growing-git-directory"
    cache.mkdir(mode=0o700)
    marker = _make_inert_git_sentinel(cache, "empty-directory")
    original_stat = inventory.os.stat
    mutated = False

    def adding_head_stat(path, *args, **kwargs):
        nonlocal mutated
        try:
            return original_stat(path, *args, **kwargs)
        except FileNotFoundError:
            if path == "HEAD" and kwargs.get("dir_fd") is not None and not mutated:
                head = marker / "HEAD"
                head.write_text("ref: refs/heads/main\n", encoding="utf-8")
                head.chmod(0o600)
                mutated = True
            raise

    monkeypatch.setattr(inventory.os, "stat", adding_head_stat)

    with inventory.opened_directory_nofollow(cache) as (_bound, descriptor):
        with pytest.raises(
            inventory.LiveKitEotInventoryError,
            match="Git boundary verification",
        ):
            inventory._git_marker_present(descriptor)
    assert mutated is True


@pytest.mark.parametrize("target", ["source", "output"])
@pytest.mark.parametrize(
    "mutation",
    [
        "grow-empty-file",
        "grow-empty-directory",
        "replace-empty-file",
        "replace-empty-directory",
    ],
)
def test_inert_git_sentinel_becoming_real_during_worker_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    mutation: str,
):
    source_root = tmp_path / "source-root"
    source_root.mkdir(mode=0o700)
    source = _patch_tiny_source(source_root, monkeypatch)
    output = _private_output_parent(tmp_path) / "inventory.json"
    sentinel_root = source_root if target == "source" else output.parent
    sentinel_kind = (
        "empty-file" if mutation.endswith("empty-file") else "empty-directory"
    )
    _make_inert_git_sentinel(sentinel_root, sentinel_kind)

    def race() -> None:
        _promote_inert_git_sentinel(sentinel_root, mutation)

    failure = "source verification" if target == "source" else "output publication"
    with pytest.raises(inventory.LiveKitEotInventoryError, match=failure):
        inventory.inventory_livekit_eot_english(
            source_parquet=source,
            parquet_python=None,
            output=output,
            accepted_terms=frozenset({"CC-BY-4.0"}),
            test_worker_injection=_injection(race),
        )
    assert not output.exists()


@pytest.mark.parametrize("linked_worktree", [False, True])
def test_source_and_output_reject_any_git_repository_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    linked_worktree: bool,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    safe_output = _private_output_parent(tmp_path) / "safe-inventory.json"
    repository = tmp_path / (
        "linked-worktree" if linked_worktree else "alternate-repository"
    )
    repository.mkdir(mode=0o700)
    repository.chmod(0o700)
    _mark_git_repository(repository, linked_worktree=linked_worktree)

    repository_source = repository / "source.parquet"
    repository_source.write_bytes(source.read_bytes())
    repository_source.chmod(0o600)
    with pytest.raises(inventory.LiveKitEotInventoryError, match="path prerequisite"):
        inventory.inventory_livekit_eot_english(
            source_parquet=repository_source,
            parquet_python=None,
            output=safe_output,
            accepted_terms=frozenset({"CC-BY-4.0"}),
            test_worker_injection=_injection(),
        )

    repository_output_parent = repository / "private-output"
    repository_output_parent.mkdir(mode=0o700)
    repository_output_parent.chmod(0o700)
    with pytest.raises(inventory.LiveKitEotInventoryError, match="output"):
        inventory.inventory_livekit_eot_english(
            source_parquet=source,
            parquet_python=None,
            output=repository_output_parent / "inventory.json",
            accepted_terms=frozenset({"CC-BY-4.0"}),
            test_worker_injection=_injection(),
        )


def test_output_git_marker_created_during_worker_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    output = _private_output_parent(tmp_path) / "inventory.json"

    def race() -> None:
        _mark_git_repository(output.parent, linked_worktree=True)

    with pytest.raises(inventory.LiveKitEotInventoryError, match="output publication"):
        inventory.inventory_livekit_eot_english(
            source_parquet=source,
            parquet_python=None,
            output=output,
            accepted_terms=frozenset({"CC-BY-4.0"}),
            test_worker_injection=_injection(race),
        )
    assert not output.exists()


def test_source_change_during_worker_is_rejected_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    payload = source.read_bytes()
    output = _private_output_parent(tmp_path) / "inventory.json"

    def mutate() -> None:
        source.write_bytes(payload[::-1])
        source.chmod(0o600)

    with pytest.raises(inventory.LiveKitEotInventoryError, match="source verification"):
        inventory.inventory_livekit_eot_english(
            source_parquet=source,
            parquet_python=None,
            output=output,
            accepted_terms=frozenset({"CC-BY-4.0"}),
            test_worker_injection=_injection(mutate),
        )
    assert not output.exists()


def test_worker_result_tampering_is_rejected_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    output = _private_output_parent(tmp_path) / "inventory.json"

    def run(_descriptor: int, worker_output: Path) -> None:
        value = _worker_result()
        value["hold_label_count"] = 1
        result = worker_output / "result.json"
        result.write_text(json.dumps(value), encoding="utf-8")
        result.chmod(0o400)

    with pytest.raises(inventory.LiveKitEotInventoryError, match="worker result"):
        inventory.inventory_livekit_eot_english(
            source_parquet=source,
            parquet_python=None,
            output=output,
            accepted_terms=frozenset({"CC-BY-4.0"}),
            test_worker_injection=inventory.TestWorkerInjection(
                run=run,
                fixture="synthetic-bad-count-v1",
            ),
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "field_name",
    [
        "duration_exact_sample_count",
        "duration_microsecond_quantized_count",
        "final_gap_zero_sample_count",
        "final_gap_one_sample_count",
    ],
)
def test_source_profile_count_tampering_is_rejected_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
):
    source = _patch_tiny_source(tmp_path, monkeypatch)
    output = _private_output_parent(tmp_path) / "inventory.json"

    def run(_descriptor: int, worker_output: Path) -> None:
        value = _worker_result()
        value[field_name] = int(value[field_name]) + 1
        result = worker_output / "result.json"
        result.write_text(json.dumps(value), encoding="utf-8")
        result.chmod(0o400)

    with pytest.raises(inventory.LiveKitEotInventoryError, match="worker provenance"):
        inventory.inventory_livekit_eot_english(
            source_parquet=source,
            parquet_python=None,
            output=output,
            accepted_terms=frozenset({"CC-BY-4.0"}),
            test_worker_injection=inventory.TestWorkerInjection(
                run=run,
                fixture="synthetic-bad-profile-v1",
            ),
        )
    assert not output.exists()


def test_cli_has_no_downloader_runtime_dependency_or_injection_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        inventory._parser().parse_args(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--source-parquet" in help_text
    assert "--parquet-python" in help_text
    assert "--accept-license" in help_text
    assert "inject" not in help_text.casefold()
    source = inspect.getsource(inventory)
    assert "urllib" not in source
    assert "requests" not in source
    assert "import pyarrow" not in source
