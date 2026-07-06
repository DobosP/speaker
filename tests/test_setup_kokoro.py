"""``fetch_kokoro_package`` unpack + wiring contract (no network).

The runtime's ``build_tts`` warning and ``tools/doctor`` both point users at
``python -m tools.setup_models --kokoro``; until 2026-07-06 that flag did not
exist (recon finding: the referenced command was fictional, so the ADR-0010
voice could never be fetched the documented way). These pin the helper with a
locally-built archive: whole-package unpack past the top-level dir, the
path-traversal guard, resolved config paths, and idempotent re-runs. The
pre-placed archive doubles as the "already downloaded" fixture -- the URLs are
``.invalid`` so any accidental network attempt fails loudly.
"""
from __future__ import annotations

import io
import os
import tarfile

import pytest

from tools.setup_models import fetch_kokoro_package

_URL = "https://example.invalid/kokoro-int8-multi-lang-v1_1.tar.bz2"


def _add_file(tar: tarfile.TarFile, name: str, data: bytes = b"x") -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _make_archive(dest_dir, members: dict[str, bytes]):
    """Pre-place the release archive where fetch_speaker_model expects it, so
    the existing-file check short-circuits the download (no network)."""
    path = os.path.join(dest_dir, _URL.rsplit("/", 1)[-1])
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(path, "w:bz2") as tar:
        for name, data in members.items():
            _add_file(tar, name, data)
    return path


_PACKAGE = {
    "kokoro-int8-multi-lang-v1_1/model.int8.onnx": b"model",
    "kokoro-int8-multi-lang-v1_1/voices.bin": b"voices",
    "kokoro-int8-multi-lang-v1_1/tokens.txt": b"tokens",
    "kokoro-int8-multi-lang-v1_1/lexicon-us-en.txt": b"lex",
    "kokoro-int8-multi-lang-v1_1/espeak-ng-data/en_dict": b"dict",
}


def test_unpacks_package_and_resolves_config_paths(tmp_path):
    dest = str(tmp_path / "tts_kokoro")
    archive = _make_archive(dest, _PACKAGE)
    got = fetch_kokoro_package(dest, _URL)
    assert got["tts_model"] == os.path.join(dest, "model.int8.onnx")
    assert got["tts_voices"] == os.path.join(dest, "voices.bin")
    assert got["tts_tokens"] == os.path.join(dest, "tokens.txt")
    assert got["tts_data_dir"] == os.path.join(dest, "espeak-ng-data")
    assert got["tts_lexicon"] == os.path.join(dest, "lexicon-us-en.txt")
    for key in ("tts_model", "tts_voices", "tts_tokens", "tts_lexicon"):
        assert os.path.exists(got[key]), key
    assert os.path.isdir(got["tts_data_dir"])
    assert not os.path.exists(archive)  # archive cleaned up after unpack


def test_second_run_is_idempotent_without_redownload(tmp_path):
    dest = str(tmp_path / "tts_kokoro")
    _make_archive(dest, _PACKAGE)
    first = fetch_kokoro_package(dest, _URL)
    # The archive is gone and the URL is .invalid -- a second call only works
    # if it resolves the already-extracted files instead of re-fetching.
    second = fetch_kokoro_package(dest, _URL)
    assert second == first


def test_partial_prior_unpack_refetches(tmp_path):
    # A prior unpack missing tokens.txt must NOT idempotent-skip -- returning
    # partial paths would leave a stale Piper tokens path wired beside Kokoro
    # voices (codex-review 2026-07-06). With the archive re-placed, the helper
    # must unpack again and restore the missing file.
    dest = str(tmp_path / "tts_kokoro")
    _make_archive(dest, _PACKAGE)
    first = fetch_kokoro_package(dest, _URL)
    os.remove(first["tts_tokens"])
    _make_archive(dest, _PACKAGE)  # re-placed "download"
    again = fetch_kokoro_package(dest, _URL)
    assert os.path.exists(again["tts_tokens"])
    assert again == first


def test_traversal_member_is_rejected(tmp_path):
    dest = str(tmp_path / "tts_kokoro")
    _make_archive(dest, {"kokoro/../evil.txt": b"evil", **_PACKAGE})
    with pytest.raises(ValueError, match="unsafe member path"):
        fetch_kokoro_package(dest, _URL)
    assert not os.path.exists(tmp_path / "evil.txt")


def test_incomplete_package_raises(tmp_path):
    dest = str(tmp_path / "tts_kokoro")
    members = {k: v for k, v in _PACKAGE.items() if not k.endswith("voices.bin")}
    _make_archive(dest, members)
    with pytest.raises(FileNotFoundError, match="voices.bin"):
        fetch_kokoro_package(dest, _URL)
