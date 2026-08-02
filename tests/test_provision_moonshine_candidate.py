from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import stat
import sys
import zipfile

import pytest

from tools import provision_moonshine_candidate as provision
from tools.streaming_stt import manifest as manifest_module
from tools.streaming_stt.manifest import MOONSHINE_ARTIFACT_NAMES


_DIST_INFO = "moonshine_voice-0.1.0.dist-info"


def _fake_wheel(
    wheel: Path,
    site_packages: Path,
    *,
    install_moonshine: bool,
) -> None:
    files = {
        "moonshine_voice/__init__.py": b'__version__ = "0.1.0"\n',
        "moonshine_voice/libmoonshine.so": b"native-moonshine",
        "moonshine_voice/transcriber.py": b"class Transcriber: pass\n",
        "moonshine_voice.libs/libonnxruntime-13ab8084.so.1": b"native-ort",
        f"{_DIST_INFO}/METADATA": (
            b"Metadata-Version: 2.4\nName: moonshine-voice\nVersion: 0.1.0\n"
        ),
        f"{_DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n",
        f"{_DIST_INFO}/entry_points.txt": b"",
        f"{_DIST_INFO}/top_level.txt": b"moonshine_voice\n",
        f"{_DIST_INFO}/licenses/LICENSE": b"test license\n",
    }
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for relative, payload in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        writer.writerow([relative, f"sha256={digest.decode('ascii')}", len(payload)])
    writer.writerow([f"{_DIST_INFO}/RECORD", "", ""])
    record = record_buffer.getvalue().encode("utf-8")
    with zipfile.ZipFile(wheel, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for relative, payload in files.items():
            archive.writestr(relative, payload)
        archive.writestr(f"{_DIST_INFO}/RECORD", record)
    if not install_moonshine:
        (site_packages / "candidate.py").write_bytes(b"candidate-runtime")
        return
    for relative, payload in {**files, f"{_DIST_INFO}/RECORD": record}.items():
        destination = site_packages / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def _inputs(tmp_path: Path, monkeypatch, *, install_moonshine: bool = True):
    venv = tmp_path / "candidate-venv"
    venv.mkdir(mode=0o700)
    python = venv / "bin" / "python"
    python.parent.mkdir(mode=0o700)
    python.symlink_to(Path(sys.executable).resolve(strict=True))
    (venv / "pyvenv.cfg").write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    site_packages = (
        venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True)
    site_packages.chmod(0o700)

    wheel = tmp_path / provision._WHEEL_NAME
    _fake_wheel(
        wheel,
        site_packages,
        install_moonshine=install_moonshine,
    )
    model = tmp_path / "model"
    model.mkdir()
    model_paths = {
        name: model / basename for name, basename in provision._MODEL_FILES.items()
    }
    for index, path in enumerate(model_paths.values(), start=1):
        path.write_bytes(bytes([index]))

    monkeypatch.setattr(
        manifest_module,
        "_MOONSHINE_RELEASE_RECEIPT",
        (hashlib.sha256(wheel.read_bytes()).hexdigest(), wheel.stat().st_size),
    )
    monkeypatch.setattr(
        manifest_module,
        "_MOONSHINE_MODEL_RECEIPTS",
        {
            arch: {
                name: (
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    path.stat().st_size,
                )
                for name, path in model_paths.items()
            }
            for arch in (
                "tiny-streaming",
                "small-streaming",
                "medium-streaming",
            )
        },
    )
    return venv, site_packages, wheel, model


def test_provision_binds_existing_disposable_runtime_without_importing_it(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, wheel, model = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "receipt"

    manifest = provision.provision_candidate(
        venv_root=venv,
        wheel=wheel,
        model_root=model,
        output_dir=output,
    )

    assert manifest.adapter == "moonshine-voice-stream-v1"
    assert manifest.python.path == venv / "bin" / "python"
    assert tuple(manifest.artifact_by_name) == MOONSHINE_ARTIFACT_NAMES
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest.path.stat().st_mode) == 0o600
    assert (
        stat.S_IMODE(manifest.artifact_by_name["runtime-receipt"].path.stat().st_mode)
        == 0o600
    )
    assert "moonshine_voice" not in sys.modules
    safe = provision._safe_result(manifest)
    assert set(safe) == {
        "ok",
        "adapter",
        "model_id",
        "manifest_sha256",
        "runtime_receipt_sha256",
        "artifact_set_sha256",
    }
    assert str(tmp_path) not in str(safe)


def test_provision_is_no_overwrite_and_preserves_first_receipt(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, wheel, model = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "receipt"
    manifest = provision.provision_candidate(
        venv_root=venv,
        wheel=wheel,
        model_root=model,
        output_dir=output,
    )
    before = manifest.path.read_bytes()

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            wheel=wheel,
            model_root=model,
            output_dir=output,
        )

    assert manifest.path.read_bytes() == before


@pytest.mark.parametrize("model_arch", ["small-streaming", "medium-streaming"])
def test_provision_selects_closed_nondefault_streaming_receipt(
    tmp_path,
    monkeypatch,
    model_arch,
):
    venv, _site_packages, wheel, model = _inputs(tmp_path, monkeypatch)

    manifest = provision.provision_candidate(
        venv_root=venv,
        wheel=wheel,
        model_root=model,
        output_dir=tmp_path / f"{model_arch}-receipt",
        model_arch=model_arch,
    )

    assert manifest.model_id == f"moonshine-voice-0.1.0-{model_arch}-en-cpu"
    assert manifest.adapter_config is not None
    assert manifest.adapter_config.model_arch == model_arch


def test_provision_rejects_hardlinked_runtime_before_manifest_publish(
    tmp_path,
    monkeypatch,
):
    venv, site_packages, wheel, model = _inputs(tmp_path, monkeypatch)
    os.link(
        site_packages / "moonshine_voice" / "__init__.py",
        site_packages / "moonshine_voice" / "candidate-hardlink.py",
    )
    output = tmp_path / "rejected"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            wheel=wheel,
            model_root=model,
            output_dir=output,
        )

    assert output.is_dir()
    assert not (output / "worker-manifest.json").exists()


def test_provision_rejects_runtime_without_installed_moonshine_distribution(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, wheel, model = _inputs(
        tmp_path,
        monkeypatch,
        install_moonshine=False,
    )
    output = tmp_path / "candidate-only"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            wheel=wheel,
            model_root=model,
            output_dir=output,
        )

    assert output.is_dir()
    assert not (output / "worker-manifest.json").exists()


@pytest.mark.parametrize(
    "mutation",
    ["native-library", "installed-record", "shadow-module"],
)
def test_provision_rejects_runtime_not_resolved_from_exact_wheel(
    tmp_path,
    monkeypatch,
    mutation,
):
    venv, site_packages, wheel, model = _inputs(tmp_path, monkeypatch)
    if mutation == "native-library":
        target = site_packages / "moonshine_voice" / "libmoonshine.so"
        target.write_bytes(b"changed-native")
    elif mutation == "installed-record":
        target = site_packages / _DIST_INFO / "RECORD"
        target.write_bytes(target.read_bytes() + b"\n")
    else:
        (site_packages / "moonshine_voice.py").write_text(
            'raise RuntimeError("shadow")\n',
            encoding="utf-8",
        )
    output = tmp_path / mutation

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            wheel=wheel,
            model_root=model,
            output_dir=output,
        )

    assert output.is_dir()
    assert not (output / "worker-manifest.json").exists()


def test_provision_rejects_system_site_packages_before_creating_output(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, wheel, model = _inputs(tmp_path, monkeypatch)
    (venv / "pyvenv.cfg").write_text(
        "home = /usr/bin\ninclude-system-site-packages = true\n",
        encoding="utf-8",
    )
    output = tmp_path / "non-isolated"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            wheel=wheel,
            model_root=model,
            output_dir=output,
        )

    assert not output.exists()


def test_provision_rejects_wrong_release_name_without_creating_output(
    tmp_path,
    monkeypatch,
):
    venv, _site_packages, wheel, model = _inputs(tmp_path, monkeypatch)
    wrong = wheel.with_name("moonshine-unpinned.whl")
    wrong.write_bytes(wheel.read_bytes())
    output = tmp_path / "not-created"

    with pytest.raises(provision.ProvisionError):
        provision.provision_candidate(
            venv_root=venv,
            wheel=wrong,
            model_root=model,
            output_dir=output,
        )

    assert not output.exists()


def test_provision_help_states_no_install_or_download():
    help_text = " ".join(provision._parser().format_help().split())

    assert "does not install or download" in help_text
