"""Public synthetic asset/setup tests: no real models, credentials or network."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from core.engines import _qwen_assets as assets
from tools import setup_portable_asr as setup


@pytest.fixture
def tiny_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "supported_platform", lambda: True)
    model, library = tmp_path / "model.gguf", tmp_path / "lib.so"
    model.write_bytes(b"public model")
    library.write_bytes(b"public CPU library")
    for prefix, path in (("MODEL", model), ("LIBRARY", library)):
        monkeypatch.setattr(assets, prefix + "_SIZE", path.stat().st_size)
        monkeypatch.setattr(assets, prefix + "_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = bundle / assets.BUNDLE_MANIFEST
    manifest.write_text(json.dumps({"schema_version": 1, "model": str(model), "library": str(library)}))
    return bundle, model, library


def test_validator_returns_exact_pinned_paths(tiny_assets):
    bundle, model, library = tiny_assets
    assert assets.validate_qwen_bundle(bundle) == (model, library)
    model.write_bytes(b"changed data")
    with pytest.raises(assets.QwenAssetError, match="qwen_asset_invalid"):
        assets.validate_qwen_bundle(bundle)


@pytest.mark.parametrize("mutation", ["bool", "extra", "relative", "symlink", "oversize"])
def test_manifest_refuses_unsafe_or_unbound_shapes(tiny_assets, mutation):
    bundle, model, library = tiny_assets
    path = bundle / assets.BUNDLE_MANIFEST
    data = json.loads(path.read_text())
    if mutation == "bool": data["schema_version"] = True
    elif mutation == "extra": data["device"] = "cuda"
    elif mutation == "relative": data["model"] = model.name
    elif mutation == "symlink":
        alias = model.parent / "alias"
        alias.symlink_to(model)
        data["model"] = str(alias)
    else:
        path.write_text(" " * 4097)
    if mutation != "oversize": path.write_text(json.dumps(data))
    with pytest.raises(assets.QwenAssetError): assets.validate_qwen_bundle(bundle)


def test_platform_refused_before_opening_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "supported_platform", lambda: False)
    with pytest.raises(assets.QwenAssetError, match="qwen_cpu_platform_unsupported"):
        assets.validate_qwen_bundle(tmp_path)


def test_verified_file_refuses_directory_and_size(tmp_path):
    with pytest.raises(assets.QwenAssetError): assets.verified_file(tmp_path, 1, "0" * 64)
    file = tmp_path / "public"
    file.write_bytes(b"payload")
    with pytest.raises(assets.QwenAssetError): assets.verified_file(file, 1, hashlib.sha256(b"p").hexdigest())


def test_atomic_write_preserves_existing_and_detects_conflict(tmp_path):
    path = tmp_path / "config.local.json"
    setup._write_atomic(path, b"original")
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError): setup._write_atomic(path, b"replacement")
    with pytest.raises(ValueError): setup._write_atomic(path, b"replacement", previous=b"stale")
    assert path.read_bytes() == b"original"
    setup._write_atomic(path, b"next", previous=b"original")
    assert path.read_bytes() == b"next"
    assert not list(tmp_path.glob(".portable-asr-*"))


def test_profile_install_changes_only_optional_map(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "validate_qwen_bundle", lambda path: None)
    monkeypatch.setattr(assets, "validate_sense_voice", lambda path: None)
    base = {"final_stt_profiles": {"legacy": {"value": 1}}, "selection": "legacy"}
    local = {"enrollment": "PUBLIC_FIXTURE", "llm": {"local_only": True}}
    (tmp_path / "config.json").write_text(json.dumps(base))
    path = tmp_path / "config.local.json"
    path.write_text(json.dumps(local))
    def add(config, **paths):
        assert paths == {"qwen_model_dir": str(tmp_path / "bundle"), "sense_voice_dir": str(tmp_path / "sense")}
        value = deepcopy(config)
        value["final_stt_profiles"]["qwen-sensevoice-cpu"] = {"value": 2}
        return value
    result = setup.install_profile(tmp_path, tmp_path / "bundle", tmp_path / "sense", add_profile=add)
    saved = json.loads(path.read_text())
    assert {k: v for k, v in saved.items() if k != "final_stt_profiles"} == local
    assert saved["final_stt_profiles"] == {"legacy": {"value": 1}, "qwen-sensevoice-cpu": {"value": 2}}
    assert result["selection_changed"] is False
    before = path.read_bytes()
    setup.install_profile(tmp_path, tmp_path / "bundle", tmp_path / "sense", add_profile=add)
    assert before == path.read_bytes()


def test_profile_helper_cannot_change_other_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "validate_qwen_bundle", lambda path: None)
    monkeypatch.setattr(assets, "validate_sense_voice", lambda path: None)
    (tmp_path / "config.json").write_text('{"selection":"legacy","final_stt_profiles":{}}')
    def bad(config, **paths): return {**config, "selection": "new"}
    with pytest.raises(ValueError, match="portable_asr_profile_changed_other_settings"):
        setup.install_profile(tmp_path, tmp_path / "bundle", tmp_path / "sense", add_profile=bad)
    assert not (tmp_path / "config.local.json").exists()


def test_cache_reuse_has_no_weight_copies_or_download(tiny_assets, tmp_path, monkeypatch):
    _, model, library = tiny_assets
    cache = tmp_path / "cache"
    (cache / "qwen").mkdir(parents=True)
    (cache / "speech_cpp").mkdir()
    model.rename(cache / "qwen" / assets.MODEL_FILE)
    library.rename(cache / "speech_cpp" / assets.LIBRARY_FILE)
    rights_root = cache / "rights"
    rights_root.mkdir()
    rights = []
    for index in range(2):
        name = f"right-{index}"
        raw = b"public rights"
        (rights_root / name).write_bytes(raw)
        rights.append((name, len(raw), hashlib.sha256(raw).hexdigest(), "https://example.invalid/public"))
    monkeypatch.setattr(setup, "RIGHTS", rights)
    monkeypatch.setattr(setup, "_download", lambda *a: pytest.fail("network must not be used"))
    bundle = tmp_path / "new-bundle"
    assert setup.prepare_bundle(bundle, cache_root=cache) == (cache / "qwen" / assets.MODEL_FILE, cache / "speech_cpp" / assets.LIBRARY_FILE)
    assert set(p.name for p in bundle.iterdir()) == {assets.BUNDLE_MANIFEST, "rights.json"}
    setup.prepare_bundle(bundle, cache_root=cache)


def test_missing_local_cache_never_fetches_implicitly(tmp_path):
    with pytest.raises(ValueError, match="portable_asr_cache_or_download_required"):
        setup.prepare_bundle(tmp_path / "bundle")


def test_download_is_bounded_verified_and_not_implicit_on_reuse(tmp_path, monkeypatch):
    import io
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"public payload"))
    path = tmp_path / "model"
    checksum = hashlib.sha256(b"public payload").hexdigest()
    assert setup._download(path, 14, checksum, "https://example.invalid/model") == path
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: pytest.fail("existing verified file must be reused"))
    assert setup._download(path, 14, checksum, "https://example.invalid/model") == path
    path.write_bytes(b"bad")
    with pytest.raises(assets.QwenAssetError): setup._download(path, 14, checksum, "https://example.invalid/model")
    assert path.read_bytes() == b"bad"


def test_failed_download_leaves_no_unverified_model(tmp_path, monkeypatch):
    import io
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"too much public data"))
    path = tmp_path / "model"
    with pytest.raises(ValueError): setup._download(path, 3, "0" * 64, "https://example.invalid/model")
    assert not path.exists() and not list(tmp_path.glob(".download-*"))


def test_sdk_reads_only_exact_member_never_extracts_archive_paths(tmp_path, monkeypatch):
    import zipfile
    monkeypatch.setattr(assets, "supported_platform", lambda: True)
    model_raw, library_raw = b"public model", b"public CPU library"
    for prefix, raw in (("MODEL", model_raw), ("LIBRARY", library_raw)):
        monkeypatch.setattr(assets, prefix + "_SIZE", len(raw))
        monkeypatch.setattr(assets, prefix + "_SHA256", hashlib.sha256(raw).hexdigest())
    archive = tmp_path / "public.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(assets.LIBRARY_FILE, library_raw)
        package.writestr("../escape", b"never extracted")
    def fake_download(path, size, checksum, url):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(model_raw if path.name == assets.MODEL_FILE else archive.read_bytes())
        return path
    monkeypatch.setattr(setup, "_download", fake_download)
    monkeypatch.setattr(setup, "RIGHTS", ())
    bundle = tmp_path / "installed"
    assert setup.prepare_bundle(bundle, download=True) == (bundle / assets.MODEL_FILE, bundle / assets.LIBRARY_FILE)
    assert (bundle / assets.LIBRARY_FILE).read_bytes() == library_raw
    assert not (tmp_path / "escape").exists()


def test_small_fd_reader_rejects_fifo_and_symlink_without_opening(tmp_path):
    import os
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(assets.QwenAssetError): assets.read_regular(fifo, 4096)
    regular = tmp_path / "regular"
    regular.write_bytes(b"public")
    link = tmp_path / "link"
    link.symlink_to(regular)
    with pytest.raises(assets.QwenAssetError): assets.read_regular(link, 4096)
    assert assets.read_regular(regular, 6) == b"public"
    with pytest.raises(assets.QwenAssetError): assets.read_regular(regular, 5)


def test_existing_bundle_reuse_checks_retained_rights(tiny_assets, monkeypatch):
    bundle, _, _ = tiny_assets
    rights = bundle / "public-LICENSE"
    rights.write_bytes(b"public rights")
    checksum = hashlib.sha256(rights.read_bytes()).hexdigest()
    monkeypatch.setattr(setup, "RIGHTS", ((rights.name, 13, checksum, "https://example.invalid/license"),))
    manifest = {"schema_version": 1, "artifacts": [{"file": str(rights), "bytes": 13, "sha256": checksum, "url": "https://example.invalid/license"}]}
    (bundle / "rights.json").write_text(json.dumps(manifest))
    setup.prepare_bundle(bundle)
    rights.write_bytes(b"changed")
    with pytest.raises(assets.QwenAssetError): setup.prepare_bundle(bundle)


def test_download_deadline_removes_unverified_file(tmp_path, monkeypatch):
    import io
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"public"))
    ticks = iter([0., 901.])
    monkeypatch.setattr(setup.time, "monotonic", lambda: next(ticks))
    path = tmp_path / "model"
    with pytest.raises(ValueError, match="portable_asr_download_deadline"):
        setup._download(path, 6, hashlib.sha256(b"public").hexdigest(), "https://example.invalid/model")
    assert not path.exists() and not list(tmp_path.glob(".download-*"))


def test_real_profile_helper_adds_optional_entry_to_legacy_opaque_local_map(tmp_path, monkeypatch):
    import copy
    from core.config import load_config, apply_final_stt_profile

    repo = Path(__file__).resolve().parents[1]
    base = json.loads((repo / "config.json").read_bytes())
    (tmp_path / "config.json").write_text(json.dumps(base))
    legacy = copy.deepcopy(base["final_stt_profiles"])
    local = {"final_stt_profiles": legacy, "private_fixture": {"preserved": "public test"},
             "sherpa": {"speaker_enroll_embedding": "existing enrollment reference"}}
    (tmp_path / "config.local.json").write_text(json.dumps(local))
    monkeypatch.setattr(assets, "validate_qwen_bundle", lambda path: (tmp_path / "model.gguf", tmp_path / "libspeech.so"))
    monkeypatch.setattr(assets, "validate_sense_voice", lambda path: path)
    setup.install_profile(tmp_path, tmp_path / "bundle", tmp_path / "sense")
    saved = json.loads((tmp_path / "config.local.json").read_bytes())
    assert saved["private_fixture"] == local["private_fixture"]
    assert saved["sherpa"] == local["sherpa"]
    assert "final_stt_profile" not in saved
    assert all(saved["final_stt_profiles"][name] == value for name, value in legacy.items())
    # Use the real config merger: its opaque map must retain the new entry.
    # The headless global test prefix normally disables every local file.
    monkeypatch.delenv("SPEAKER_NO_LOCAL_CONFIG", raising=False)
    merged = load_config(str(tmp_path / "config.json"), local=str(tmp_path / "config.local.json"))
    chosen, metadata = apply_final_stt_profile(merged, "qwen-sensevoice-cpu")
    assert metadata.name == "qwen-sensevoice-cpu"
    assert chosen["sherpa"]["asr_final_model"] == str(tmp_path / "bundle")
    assert chosen["sherpa"]["asr_final_verifier_model"] == str(tmp_path / "sense")
    assert chosen["sherpa"]["asr_num_threads"] == 2
    assert chosen["sherpa"]["speaker_enroll_embedding"] == local["sherpa"]["speaker_enroll_embedding"]
