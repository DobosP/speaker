"""``fetch_kws_package`` + ``generate_barge_keywords`` contract (no network).

Pins the streaming keyword-spotter setup for the playback-time stop-word
interrupt floor (roadmap Phase 2 item 14): selective chunk-16 (int8-preferred)
member extraction, the path-traversal guard, the lexicon-driven keyword file
generation, resolved config keys, and idempotent re-runs. A locally-built
archive with the model's real member-name shapes stands in for the release
asset; the URL is ``.invalid`` so any accidental network attempt fails loudly.
"""
from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from tools.setup_models import (
    KWS_BARGE_PHRASES,
    fetch_kws_package,
    generate_barge_keywords,
    wire_sherpa_paths,
)


def _write_paths(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        paths[name] = str(path)
    return paths


def test_unrelated_kws_setup_preserves_complete_kokoro_family(tmp_path):
    old = _write_paths(
        tmp_path / "old",
        ("tts_model", "tts_tokens", "tts_voices", "tts_lexicon"),
    )
    old_data = tmp_path / "old" / "tts_data_dir"
    old_data.mkdir()
    old["tts_data_dir"] = str(old_data)
    piper = _write_paths(tmp_path / "piper", ("tts_model", "tts_tokens"))
    piper_data = tmp_path / "piper" / "tts_data_dir"
    piper_data.mkdir()
    piper["tts_data_dir"] = str(piper_data)
    kws = _write_paths(
        tmp_path / "kws",
        (
            "kws_tokens",
            "kws_encoder",
            "kws_decoder",
            "kws_joiner",
            "kws_keywords_file",
        ),
    )
    config = {"sherpa": dict(old)}

    wire_sherpa_paths(
        config,
        {**piper, **kws},
        requested_families={"kws"},
    )

    assert {key: config["sherpa"][key] for key in old} == old
    assert {key: config["sherpa"][key] for key in kws} == kws


def test_invalid_kokoro_family_repairs_to_coherent_piper(tmp_path):
    old = _write_paths(
        tmp_path / "old",
        ("tts_model", "tts_voices", "tts_lexicon"),
    )
    old["tts_tokens"] = str(tmp_path / "old" / "missing-tokens")
    piper = _write_paths(tmp_path / "piper", ("tts_model", "tts_tokens"))
    piper_data = tmp_path / "piper" / "tts_data_dir"
    piper_data.mkdir()
    piper["tts_data_dir"] = str(piper_data)
    config = {"sherpa": dict(old)}

    wire_sherpa_paths(config, piper)

    assert {key: config["sherpa"][key] for key in piper} == piper
    assert "tts_voices" not in config["sherpa"]
    assert "tts_lexicon" not in config["sherpa"]


_URL = "https://example.invalid/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2"
_ROOT = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/"

# A minimal CMU-dict lexicon covering every word in KWS_BARGE_PHRASES.
_LEXICON = "\n".join(
    [
        "STOP S T AA1 P",
        "TALKING T AO1 K IH0 NG",
        "SPEAKING S P IY1 K IH0 NG",
        "BE B IY1",
        "QUIET K W AY1 AH0 T",
        "WAIT W EY1 T",
        "HOLD HH OW1 L D",
        "ON AA1 N",
        "STOP S T OW1 P",  # second pronunciation -- first one must win
    ]
).encode("utf-8")

# Both chunk sizes + int8/fp32 encoder & joiner, matching the real package, so
# the selector's chunk-16 + int8-preference is genuinely exercised.
_PACKAGE = {
    _ROOT + "tokens.txt": b"tokens",
    _ROOT + "en.phone": _LEXICON,
    _ROOT + "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx": b"enc16-int8",
    _ROOT + "encoder-epoch-13-avg-2-chunk-16-left-64.onnx": b"enc16-fp32",
    _ROOT + "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx": b"enc8",
    _ROOT + "decoder-epoch-13-avg-2-chunk-16-left-64.onnx": b"dec16",
    _ROOT + "decoder-epoch-13-avg-2-chunk-8-left-64.onnx": b"dec8",
    _ROOT + "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx": b"join16-int8",
    _ROOT + "joiner-epoch-13-avg-2-chunk-16-left-64.onnx": b"join16-fp32",
    _ROOT + "test_wavs/en_0.wav": b"wav",
}


def _add_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _make_archive(dest_dir, members: dict[str, bytes]):
    """Pre-place the archive where fetch_speaker_model expects it, so the
    existing-file check short-circuits the download (no network)."""
    path = os.path.join(dest_dir, _URL.rsplit("/", 1)[-1])
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(path, "w:bz2") as tar:
        for name, data in members.items():
            _add_file(tar, name, data)
    return path


def test_generate_barge_keywords_uses_lexicon_phonemes(tmp_path):
    lexicon = tmp_path / "en.phone"
    lexicon.write_bytes(_LEXICON)
    out = tmp_path / "keywords_barge.txt"
    generate_barge_keywords(str(lexicon), str(out))

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(KWS_BARGE_PHRASES)
    # Thresholds come from the KWS_BARGE_PHRASES source table, not pinned
    # literals: tuning them is a live-A/B decision (ADR-0082 measured phantom
    # cuts at the original recall-biased values) and must not break this test.
    stop_thr = KWS_BARGE_PHRASES[0][2]
    wait_thr = KWS_BARGE_PHRASES[-1][2]
    # STOP: first pronunciation (AA1) wins, boost 2.0, and the single-word
    # label the runtime command seam recognizes.
    assert lines[0] == f"S T AA1 P :2.0 #{stop_thr:g} @stop"
    # HOLD ON: two-word phrase, phonemes concatenated, soft-stop label.
    assert lines[-1] == f"HH OW1 L D AA1 N :2.0 #{wait_thr:g} @wait"
    # Every emitted label normalizes to a control the runtime resolves to stop.
    from core.contract import STOP_COMMANDS, normalize_command

    for line in lines:
        label = line.split("@", 1)[1]
        norm = normalize_command(label)
        assert norm in STOP_COMMANDS or norm in {"wait", "hold on"}


def test_generate_barge_keywords_rejects_missing_word(tmp_path):
    lexicon = tmp_path / "en.phone"
    lexicon.write_text("STOP S T AA1 P\n", encoding="utf-8")  # missing TALKING, ...
    with pytest.raises(KeyError):
        generate_barge_keywords(str(lexicon), str(tmp_path / "kw.txt"))


def test_unpacks_selected_members_and_resolves_config_paths(tmp_path):
    dest = str(tmp_path / "kws")
    _make_archive(dest, _PACKAGE)
    got = fetch_kws_package(dest, _URL)

    assert got["kws_encoder"] == os.path.join(
        dest, "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
    )
    assert got["kws_joiner"] == os.path.join(
        dest, "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
    )
    assert got["kws_decoder"] == os.path.join(
        dest, "decoder-epoch-13-avg-2-chunk-16-left-64.onnx"
    )
    assert got["kws_tokens"] == os.path.join(dest, "tokens.txt")
    assert got["kws_keywords_file"] == os.path.join(dest, "keywords_barge.txt")
    for key in got:
        assert os.path.exists(got[key]), key

    # chunk-8 members and the test wav are not kept; the archive + lexicon are
    # cleaned up after generation.
    remaining = set(os.listdir(dest))
    assert not any("chunk-8" in name for name in remaining)
    assert "en.phone" not in remaining
    assert not any(name.endswith(".tar.bz2") for name in remaining)


def test_second_run_is_idempotent_without_redownload(tmp_path):
    dest = str(tmp_path / "kws")
    _make_archive(dest, _PACKAGE)
    first = fetch_kws_package(dest, _URL)
    # The archive is gone and the URL is .invalid -- a second call only works if
    # it resolves the already-extracted files instead of re-fetching.
    second = fetch_kws_package(dest, _URL)
    assert second == first


def test_traversal_member_is_rejected(tmp_path):
    # The evil member carries a WANTED basename (tokens.txt) so it is actually
    # processed and hits the traversal guard -- a non-wanted basename would be
    # skipped by the selective extraction and never written, which is also safe.
    dest = str(tmp_path / "kws")
    _make_archive(dest, {_ROOT + "../tokens.txt": b"evil", **_PACKAGE})
    with pytest.raises(ValueError, match="unsafe member path"):
        fetch_kws_package(dest, _URL)
    assert not os.path.exists(tmp_path / "tokens.txt")


def test_incomplete_archive_raises(tmp_path):
    dest = str(tmp_path / "kws")
    members = {
        k: v for k, v in _PACKAGE.items() if "chunk-16" not in k or "joiner" not in k
    }
    _make_archive(dest, members)
    with pytest.raises(FileNotFoundError):
        fetch_kws_package(dest, _URL)
