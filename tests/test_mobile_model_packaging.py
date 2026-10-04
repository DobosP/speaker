"""Packaging controls use synthetic asset files and fake curl/tar; no network/models."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "mobile" / "tool"
ASR = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
TTS = "vits-piper-en_US-amy-low"
WHISPER = "sherpa-onnx-whisper-base.en"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "mobile_asset_list", TOOL / "generate-asset-list.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _asset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic-asset")


def _populate_model_assets(tmp_path: Path) -> None:
    generator = _generator()
    for filename in generator.ASR_FILES:
        _asset(tmp_path / "assets" / ASR / filename)
    for filename in generator.WHISPER_FILES:
        _asset(tmp_path / "assets" / WHISPER / filename)
    for source in (
        f"{ASR}/encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        f"{ASR}/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        f"{ASR}/test_wavs/example.wav",
        f"{ASR}/LICENSE",
        f"{ASR}/README.md",
        f"{ASR}/NOTICE.txt",
        f"{TTS}/model.onnx",
        f"{TTS}/espeak-ng-data/lang/gmw/en",
        f"{WHISPER}/base.en-encoder.onnx",
        f"{WHISPER}/base.en-decoder.onnx",
        f"{WHISPER}/test_wavs/example.wav",
        f"{WHISPER}/LICENSE",
        f"{WHISPER}/README.md",
    ):
        _asset(tmp_path / "assets" / source)


@pytest.mark.parametrize("with_whisper", [False, True])
def test_asset_manifest_keeps_exact_tuples_and_rights_without_unused_files(
    tmp_path, monkeypatch, with_whisper
):
    monkeypatch.chdir(tmp_path)
    _populate_model_assets(tmp_path)
    generator = _generator()
    entries = generator.asset_entries(with_whisper=with_whisper)
    asr_entries = {entry for entry in entries if f"/{ASR}/" in entry}
    assert asr_entries == {
        f"    - assets/{ASR}/{filename}"
        for filename in (*generator.ASR_FILES, "LICENSE", "README.md", "NOTICE.txt")
    }
    assert f"    - assets/{ASR}/" not in entries
    assert f"    - assets/{TTS}/" in entries
    assert f"    - assets/{TTS}/espeak-ng-data/lang/gmw/" in entries
    whisper_entries = {entry for entry in entries if f"/{WHISPER}/" in entry}
    assert whisper_entries == (
        {
            f"    - assets/{WHISPER}/{filename}"
            for filename in (*generator.WHISPER_FILES, "LICENSE", "README.md")
        }
        if with_whisper
        else set()
    )
    # Neither filtering path destroys cached precisions or examples.
    for family, filename in (
        (ASR, "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
        (ASR, "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        (ASR, "test_wavs/example.wav"),
        (WHISPER, "base.en-encoder.onnx"),
        (WHISPER, "test_wavs/example.wav"),
    ):
        assert (
            tmp_path / "assets" / family / filename
        ).read_bytes() == b"synthetic-asset"


@pytest.mark.parametrize("with_whisper", [False, True])
def test_generator_cli_replaces_prior_broad_optional_bundle(tmp_path, with_whisper):
    _populate_model_assets(tmp_path)
    pubspec = tmp_path / "pubspec.yaml"
    pubspec.write_text(
        f"name: fake\nflutter:\n  assets:\n    - assets/{ASR}/\n    - assets/{WHISPER}/\n"
    )
    command = [sys.executable, str(TOOL / "generate-asset-list.py")]
    if with_whisper:
        command.append("--with-whisper")
    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0
    text = pubspec.read_text()
    assert text.startswith("name: fake\nflutter:\n  assets:\n")
    assert f"    - assets/{ASR}/\n" not in text
    assert f"    - assets/{WHISPER}/\n" not in text
    assert f"assets/{ASR}/tokens.txt" in text
    assert f"assets/{TTS}/" in text
    assert (WHISPER in text) is with_whisper
    assert "example.wav" not in text
    assert f"assets/{ASR}/encoder-epoch-99-avg-1-chunk-16-left-128.onnx" not in text
    assert f"assets/{ASR}/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx" not in text
    assert f"assets/{ASR}/LICENSE" in text


@pytest.mark.parametrize(
    "family,filename,empty",
    [
        (ASR, "tokens.txt", False),
        (ASR, "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx", True),
        (WHISPER, "base.en-decoder.int8.onnx", False),
    ],
)
def test_incomplete_selected_tuple_fails_before_pubspec_mutation(
    tmp_path, family, filename, empty
):
    _populate_model_assets(tmp_path)
    missing = tmp_path / "assets" / family / filename
    if empty:
        missing.write_bytes(b"")
    else:
        missing.unlink()
    pubspec = tmp_path / "pubspec.yaml"
    original = "name: unchanged\nflutter:\n  assets:\n    - old/path/\n"
    pubspec.write_text(original)
    args = [sys.executable, str(TOOL / "generate-asset-list.py")]
    if family == WHISPER:
        args.append("--with-whisper")
    result = subprocess.run(
        args, cwd=tmp_path, capture_output=True, text=True, timeout=5
    )
    assert result.returncode != 0
    assert "Required mobile model tuple is incomplete" in result.stderr
    assert pubspec.read_text() == original


def test_bundle_filenames_match_current_pinned_model_locks():
    generator = _generator()
    for filenames, lockname in (
        (generator.ASR_FILES, "mobile-zipformer-en-2023-06-26-v1.lock.json"),
        (generator.WHISPER_FILES, "mobile-whisper-base-en-v1.lock.json"),
    ):
        lock = json.loads((ROOT / "tools" / "streaming_stt" / lockname).read_text())
        assert set(filenames) == {
            artifact["filename"] for artifact in lock["artifacts"]
        }


def _fake_download_tools(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    mobile = tmp_path / "mobile"
    tool = mobile / "tool"
    tool.mkdir(parents=True)
    script = tool / "download-models.sh"
    shutil.copyfile(TOOL / script.name, script)
    binary = tmp_path / "fake-bin"
    binary.mkdir()
    request_log = tmp_path / "requests.jsonl"
    curl = binary / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "args=sys.argv[1:]\n"
        'pathlib.Path(args[args.index("-o")+1]).write_bytes(b"fake-archive")\n'
        'with open(os.environ["MODEL_TEST_REQUESTS"], "a") as output:\n'
        '    output.write(json.dumps(args)+"\\n")\n'
    )
    curl.chmod(0o755)
    tar = binary / "tar"
    tar.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        'directory=pathlib.Path(sys.argv[-1].removesuffix(".tar.bz2"))\n'
        "directory.mkdir()\n"
        '(directory/"model.int8.onnx").write_bytes(b"synthetic")\n'
        'if "whisper" in directory.name:\n'
        '    (directory/"base.en-encoder.onnx").write_bytes(b"synthetic-fp32")\n'
        '    (directory/"base.en-decoder.onnx").write_bytes(b"synthetic-fp32")\n'
        '    (directory/"test_wavs").mkdir()\n'
        '    (directory/"test_wavs"/"fake.wav").write_bytes(b"synthetic-not-audio")\n'
    )
    tar.chmod(0o755)
    environment = {
        "PATH": str(binary) + os.pathsep + "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "MODEL_TEST_REQUESTS": str(request_log),
    }
    return script, request_log, environment


@pytest.mark.parametrize("with_whisper", [False, True])
def test_download_script_keeps_default_two_models_and_explicit_whisper_option(
    tmp_path, with_whisper
):
    script, requests, environment = _fake_download_tools(tmp_path)
    command = ["bash", str(script)]
    if with_whisper:
        command.append("--with-whisper")
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0
    urls = [
        next(arg for arg in json.loads(row) if arg.startswith("https://"))
        for row in requests.read_text().splitlines()
    ]
    families = [Path(url).name.removesuffix(".tar.bz2") for url in urls]
    assert set(families) == ({ASR, TTS, WHISPER} if with_whisper else {ASR, TTS})
    assets = script.parent.parent / "assets"
    assert (assets / ASR).is_dir() and (assets / TTS).is_dir()
    assert (assets / WHISPER).is_dir() is with_whisper
    if with_whisper:
        assert (assets / WHISPER / "model.int8.onnx").is_file()
        assert (
            assets / WHISPER / "base.en-encoder.onnx"
        ).read_bytes() == b"synthetic-fp32"
        assert (
            assets / WHISPER / "base.en-decoder.onnx"
        ).read_bytes() == b"synthetic-fp32"
        assert (
            assets / WHISPER / "test_wavs" / "fake.wav"
        ).read_bytes() == b"synthetic-not-audio"


def test_default_download_preserves_existing_optional_cache(tmp_path):
    script, requests, environment = _fake_download_tools(tmp_path)
    cached = script.parent.parent / "assets" / WHISPER / "test_wavs" / "kept.wav"
    _asset(cached)
    result = subprocess.run(
        ["bash", str(script)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert WHISPER not in requests.read_text()
    assert cached.read_bytes() == b"synthetic-asset"


@pytest.mark.parametrize("arguments", [["--unknown"], ["--with-whisper", "extra"]])
def test_invalid_download_arguments_fail_before_network_or_asset_creation(
    tmp_path, arguments
):
    script, requests, environment = _fake_download_tools(tmp_path)
    result = subprocess.run(
        ["bash", str(script), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert not requests.exists()
    assert not (script.parent.parent / "assets").exists()
