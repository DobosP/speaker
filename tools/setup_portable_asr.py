"""Install an explicit Qwen/SenseVoice CPU option without changing its selection.

No inference, microphone, playback, or automatic model activation occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request
import zipfile

from core.engines import _qwen_assets as assets

RIGHTS = (
    ("speech-code-LICENSE", 1067, "688f8aca0abe167966d277ed00bd594a5a51b028a4fbc1516ff9de7ef05c521a", "https://raw.githubusercontent.com/nyosegawa/speech.cpp/49f022a48bc1c7e797aa64dd08d2d9655baf299e/LICENSE"),
    ("Qwen-official-README.md", 57456, "5058416891bc47a2051557765997e8c42f8eb78a0e33c3e775bd17d4b0ba4d50", "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md"),
)


def _json(raw):
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("portable_asr_setup_refused")
    return value


def _read(path, maximum=2 * 1024**2):
    return assets.read_regular(Path(path).absolute(), maximum)


def _write_atomic(path, raw, *, previous=None):
    """Same-directory replace, rejecting an intervening configuration change."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".portable-asr-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if previous is None:
            # Link creates exclusively, unlike replace; an existing file is preserved.
            os.link(temporary, path)
        else:
            if _read(path) != previous:
                raise ValueError("portable_asr_config_changed")
            os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _download(path, size, sha256, url):
    if path.exists():
        return assets.verified_file(path, size, sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=path.parent)
    try:
        checksum = hashlib.sha256()
        count = 0
        deadline = time.monotonic() + 900
        with os.fdopen(fd, "wb") as stream, urllib.request.urlopen(url, timeout=60) as response:
            read = getattr(response, "read1", response.read)
            while True:
                if time.monotonic() >= deadline:
                    raise ValueError("portable_asr_download_deadline")
                chunk = read(min(1024**2, size + 1 - count))
                if time.monotonic() >= deadline:
                    raise ValueError("portable_asr_download_deadline")
                if not chunk:
                    break
                count += len(chunk)
                if count > size:
                    raise ValueError("portable_asr_download_invalid")
                checksum.update(chunk)
                stream.write(chunk)
        if count != size or checksum.hexdigest() != sha256:
            raise ValueError("portable_asr_download_invalid")
        os.link(temporary, path)
        return assets.verified_file(path, size, sha256)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare_bundle(bundle_dir, *, cache_root=None, download=False):
    """Reuse verified owned cache bytes, or download into one owned bundle."""
    bundle_dir = Path(bundle_dir).absolute()
    if not assets.supported_platform():
        raise ValueError("qwen_cpu_platform_unsupported")
    manifest_path = bundle_dir / assets.BUNDLE_MANIFEST
    if manifest_path.exists():
        result = assets.validate_qwen_bundle(bundle_dir)
        retained = _json(_read(bundle_dir / "rights.json", 65536))
        if set(retained) != {"schema_version", "artifacts"} or type(retained["schema_version"]) is not int or retained["schema_version"] != 1:
            raise ValueError("portable_asr_rights_changed")
        rows = retained["artifacts"]
        if type(rows) is not list or len(rows) != len(RIGHTS):
            raise ValueError("portable_asr_rights_changed")
        for row, (name, size, checksum, url) in zip(rows, RIGHTS, strict=True):
            if type(row) is not dict or set(row) != {"file", "bytes", "sha256", "url"} or type(row["file"]) is not str:
                raise ValueError("portable_asr_rights_changed")
            path = Path(row["file"])
            if path.name != name or (row["bytes"], row["sha256"], row["url"]) != (size, checksum, url):
                raise ValueError("portable_asr_rights_changed")
            assets.verified_file(path, size, checksum)
        return result
    if cache_root is not None:
        cache_root = Path(cache_root).absolute()
        model = assets.verified_file(cache_root / "qwen" / assets.MODEL_FILE, assets.MODEL_SIZE, assets.MODEL_SHA256)
        library = assets.verified_file(cache_root / "speech_cpp" / assets.LIBRARY_FILE, assets.LIBRARY_SIZE, assets.LIBRARY_SHA256)
        rights_root = cache_root / "rights"
    elif download:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(bundle_dir).free < assets.MODEL_SIZE + assets.SDK_SIZE + assets.LIBRARY_SIZE + 64 * 1024**2:
            raise ValueError("portable_asr_disk_budget")
        model = _download(bundle_dir / assets.MODEL_FILE, assets.MODEL_SIZE, assets.MODEL_SHA256, assets.MODEL_URL)
        archive = _download(bundle_dir / assets.SDK_FILE, assets.SDK_SIZE, assets.SDK_SHA256, assets.SDK_URL)
        with zipfile.ZipFile(archive) as package:
            # Read only one known regular member; no archive paths/symlinks extracted.
            member = package.getinfo(assets.LIBRARY_FILE)
            if member.file_size != assets.LIBRARY_SIZE or len(package.infolist()) > 16:
                raise ValueError("portable_asr_archive_invalid")
            raw = package.read(member)
        if hashlib.sha256(raw).hexdigest() != assets.LIBRARY_SHA256:
            raise ValueError("portable_asr_archive_invalid")
        library = bundle_dir / assets.LIBRARY_FILE
        _write_atomic(library, raw)
        assets.verified_file(library, assets.LIBRARY_SIZE, assets.LIBRARY_SHA256)
        rights_root = bundle_dir / "rights"
    else:
        raise ValueError("portable_asr_cache_or_download_required")
    # Rights are part of setup provenance; runtime constructor opens only pinned bytes.
    rights = []
    for name, size, checksum, url in RIGHTS:
        path = rights_root / name
        if download and cache_root is None:
            _download(path, size, checksum, url)
        assets.verified_file(path, size, checksum)
        rights.append({"file": str(path), "bytes": size, "sha256": checksum, "url": url})
    manifest = {"schema_version": 1, "model": str(model), "library": str(library)}
    rights_path = bundle_dir / "rights.json"
    rights_raw = (json.dumps({"schema_version": 1, "artifacts": rights}, sort_keys=True, indent=2) + "\n").encode()
    if rights_path.exists():
        if _read(rights_path) != rights_raw:
            raise ValueError("portable_asr_rights_changed")
    else:
        _write_atomic(rights_path, rights_raw)
    _write_atomic(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    return assets.validate_qwen_bundle(bundle_dir)


def install_profile(config_root, bundle_dir, sense_voice_dir=None, *, verifier="sensevoice", parakeet_dir=None, add_profile=None):
    """Add one optional profile to the local overlay; keep every other key."""
    from core.config import deep_merge
    if verifier not in {"sensevoice", "parakeet"}:
        raise ValueError("portable_asr_verifier_invalid")
    if add_profile is None:
        from core import config as profile_config
        add_profile = profile_config.add_qwen_sensevoice_profile if verifier == "sensevoice" else profile_config.add_qwen_parakeet_profile
    config_root = Path(config_root).absolute()
    assets.validate_qwen_bundle(bundle_dir)
    voter_dir = sense_voice_dir if verifier == "sensevoice" else parakeet_dir
    if voter_dir is None:
        raise ValueError("portable_asr_voter_missing")
    if verifier == "sensevoice":
        assets.validate_sense_voice(voter_dir)
    else:
        assets.validate_parakeet_tdt(voter_dir)
    base = _json(_read(config_root / "config.json"))
    local_path = config_root / "config.local.json"
    previous = _read(local_path) if local_path.exists() else None
    local = _json(previous) if previous is not None else {}
    effective = deep_merge(base, local)
    voter_key = "sense_voice_dir" if verifier == "sensevoice" else "parakeet_model_dir"
    updated = add_profile(effective, qwen_model_dir=str(Path(bundle_dir).absolute()), **{voter_key: str(Path(voter_dir).absolute())})
    if {k: v for k, v in updated.items() if k != "final_stt_profiles"} != {k: v for k, v in effective.items() if k != "final_stt_profiles"}:
        raise ValueError("portable_asr_profile_changed_other_settings")
    local["final_stt_profiles"] = updated["final_stt_profiles"]
    raw = (json.dumps(local, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if raw != previous:
        _write_atomic(local_path, raw, previous=previous)
    name = "qwen-sensevoice-cpu" if verifier == "sensevoice" else "qwen-parakeet-cpu"
    return {"profile": name, "profile_installed": True, "selection_changed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=Path.cwd())
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--sense-voice-dir", type=Path)
    parser.add_argument("--verifier", choices=("sensevoice", "parakeet"), default="sensevoice")
    parser.add_argument("--parakeet-dir", type=Path)
    parser.add_argument("--download", action="store_true", help="Download missing pinned Qwen assets, rather than reuse --cache-root")
    parser.add_argument("--install-profile", action="store_true", help="Add the optional CPU profile to config.local.json; never selects it")
    args = parser.parse_args(argv)
    try:
        bundle = args.bundle_dir or args.config_root / "pretrained_models/sherpa/qwen_cpu"
        sense = args.sense_voice_dir or args.config_root / "pretrained_models/sherpa/sense_voice"
        prepare_bundle(bundle, cache_root=args.cache_root, download=args.download)
        result = {"bundle_verified": True, "profile_installed": False, "selection_changed": False}
        if args.install_profile:
            parakeet = args.parakeet_dir or args.config_root / "pretrained_models/sherpa/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
            result.update(install_profile(args.config_root, bundle, sense, verifier=args.verifier, parakeet_dir=parakeet))
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        print('{"status":"portable_asr_setup_refused"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
