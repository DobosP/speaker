"""Admission/refusal tests only: no enrollment transaction or audio/model IO."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import _enrollment_persistence as admission
from core import enroll
from tools import prepare_enrollment as prep
from tools import promote_enrollment as promo


class UnusablePath:
    def __fspath__(self):
        pytest.fail("unsupported persistence inspected a supplied path")


class UnusableConfig(dict):
    def get(self, *args, **kwargs):
        pytest.fail("unsupported enrollment inspected its config")


def forbidden(*args, **kwargs):
    pytest.fail("unsupported persistence reached IO, capture or model construction")


@pytest.fixture
def unsupported_windows(monkeypatch):
    # Replace the helper's binding only; never mutate global os.name/sys.platform.
    monkeypatch.setattr(admission, "sys", SimpleNamespace(platform="win32"))


def test_windows_refusal_explains_unqualified_backend(unsupported_windows):
    with pytest.raises(admission.EnrollmentPersistenceUnavailable, match="Windows") as exc:
        admission.require_enrollment_persistence()
    assert "ACL, handle, durability and locking" in str(exc.value)


def test_posix_capabilities_admit_without_path_io(monkeypatch):
    monkeypatch.setattr(admission, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        admission, "os", SimpleNamespace(name="posix", fchmod=forbidden, getuid=forbidden)
    )
    admission.require_enrollment_persistence()


@pytest.mark.parametrize("missing", ["fchmod", "getuid", "name"])
def test_unavailable_posix_capability_refuses(monkeypatch, missing):
    capabilities = dict(name="posix", fchmod=forbidden, getuid=forbidden)
    capabilities[missing] = "unknown" if missing == "name" else None
    monkeypatch.setattr(admission, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(admission, "os", SimpleNamespace(**capabilities))
    with pytest.raises(admission.EnrollmentPersistenceUnavailable, match="POSIX"):
        admission.require_enrollment_persistence()


def test_run_refuses_before_config_paths_capture_and_models(unsupported_windows, monkeypatch):
    monkeypatch.setattr(enroll, "verify_required_os_echo_route", forbidden)
    monkeypatch.setattr(enroll, "build_enrollment_frontend", forbidden)
    monkeypatch.setattr(enroll, "sherpa_speaker_gate", forbidden)
    output = []
    assert enroll.run_enrollment(
        UnusableConfig(), config_path=UnusablePath(), recorder=forbidden, out=output.append
    ) == 5
    assert len(output) == 1 and "Windows" in output[0]


@pytest.mark.parametrize("entry", ["save", "atomic", "config"])
def test_mutation_refuses_before_path_io(unsupported_windows, entry):
    path = UnusablePath()
    with pytest.raises(admission.EnrollmentPersistenceUnavailable, match="Windows"):
        if entry == "save":
            enroll.save_enrollment(path, enroll.Enrollment(model="synthetic", embedding=[1.0]))
        elif entry == "atomic":
            enroll._atomic_write_json(path, {}, label="synthetic")
        else:
            enroll._persist_local(path, {})


def test_prepare_refuses_before_path_resolution(unsupported_windows):
    path = UnusablePath()
    with pytest.raises(prep.PreparationError, match="Windows"):
        prep.prepare_enrollment(
            worktree=path, expected_config_target=path, expected_enrollment=path,
            backup=path, candidate_name="enrollment.v5-synthetic.json",
        )


def test_promotion_refuses_before_dispatch_or_paths(unsupported_windows, monkeypatch):
    monkeypatch.setattr(promo, "_promote_enrollment", forbidden)
    path = UnusablePath()
    with pytest.raises(promo.PromotionError, match="Windows"):
        promo.promote_enrollment(
            worktree=path, primary_config=path, expected_candidate=path,
            expected_source_enrollment=path, expected_backup=path,
            accepted_enrollment=path, accept_live_gate=True,
        )


def test_private_promotion_and_lock_also_refuse(unsupported_windows):
    path = UnusablePath()
    with pytest.raises(promo.PromotionError, match="Windows"):
        promo._promote_enrollment(
            worktree=path, primary_config=path, expected_candidate=path,
            expected_source_enrollment=path, expected_backup=path,
            accepted_enrollment=path, accept_live_gate=True,
        )
    with pytest.raises(promo.PromotionError, match="Windows"):
        with promo._config_lock(path):
            forbidden()


def test_missing_fcntl_refuses_before_lock_path(monkeypatch):
    monkeypatch.setattr(promo, "require_enrollment_persistence", lambda: None)
    monkeypatch.setattr(promo, "fcntl", None)
    with pytest.raises(promo.PromotionError, match="fcntl"):
        with promo._config_lock(UnusablePath()):
            forbidden()


def test_refused_mutations_preserve_synthetic_files(unsupported_windows, tmp_path):
    existing = tmp_path / "synthetic.json"
    existing.write_bytes(b'{"synthetic": true}')
    before = existing.stat()
    with pytest.raises(admission.EnrollmentPersistenceUnavailable):
        enroll._persist_local(str(existing), {"speaker_enroll_embedding": "synthetic"})
    with pytest.raises(admission.EnrollmentPersistenceUnavailable):
        enroll.save_enrollment(str(existing), enroll.Enrollment(model="synthetic", embedding=[1.0]))
    assert existing.read_bytes() == b'{"synthetic": true}'
    assert (existing.stat().st_ino, existing.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert list(tmp_path.iterdir()) == [existing]


def test_cli_refusals_are_exit_two_without_traceback(unsupported_windows, capsys):
    assert prep.main([
        "--worktree", "synthetic", "--expected-config-target", "synthetic",
        "--expected-enrollment", "synthetic", "--backup", "synthetic",
        "--candidate-name", "enrollment.v5-synthetic.json",
    ]) == 2
    assert promo.main([
        "--worktree", "synthetic", "--primary-config", "synthetic",
        "--expected-candidate", "synthetic", "--expected-source-enrollment", "synthetic",
        "--expected-backup", "synthetic", "--accepted-enrollment", "synthetic",
        "--accept-live-gate",
    ]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "preparation refused" in output.err and "promotion refused" in output.err
    assert "Traceback" not in output.err


def test_import_and_help_work_without_fcntl():
    root = Path(__file__).resolve().parents[1]
    # Fresh process proves that import/help require neither fcntl nor a backend.
    code = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def without_fcntl(name, *args, **kwargs):
    if name == 'fcntl':
        raise ModuleNotFoundError('synthetic absence of fcntl')
    return original(name, *args, **kwargs)
builtins.__import__ = without_fcntl
from tools import promote_enrollment as promo
assert promo.fcntl is None
try:
    promo.main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('help did not exit')
"""
    child = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(root)],
        text=True, capture_output=True, timeout=30,
    )
    assert child.returncode == 0, child.stderr
    assert "--accept-live-gate" in child.stdout