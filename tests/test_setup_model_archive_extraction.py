"""Adversarial contracts for setup-model tar extraction.

Every archive in this module is made locally.  The package URLs use the
reserved ``.invalid`` domain, so a missing pre-positioned archive fails loudly
instead of turning a headless test into a network request.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import stat
import tarfile
from typing import Iterable

import pytest

import tools.setup_models as setup_models


_KOKORO_URL = "https://example.invalid/kokoro-contract.tar.bz2"
_KWS_URL = "https://example.invalid/kws-contract.tar.bz2"
_KOKORO_ROOT = "kokoro-contract"
_KWS_ROOT = "kws-contract"
_LEXICON = (
    b"\n".join(
        (
            b"STOP S T AA1 P",
            b"TALKING T AO1 K IH0 NG",
            b"SPEAKING S P IY1 K IH0 NG",
            b"BE B IY1",
            b"QUIET K W AY1 AH0 T",
            b"WAIT W EY1 T",
            b"HOLD HH OW1 L D",
            b"ON AA1 N",
        )
    )
    + b"\n"
)


def _regular(name: str, payload: bytes = b"x") -> tuple[str, bytes, bytes, str]:
    return name, tarfile.REGTYPE, payload, ""


def _archive_entry(
    name: str,
    *,
    kind: bytes,
    payload: bytes = b"",
    linkname: str = "",
) -> tuple[str, bytes, bytes, str]:
    return name, kind, payload, linkname


def _write_archive(
    path: Path,
    entries: Iterable[tuple[str, bytes, bytes, str]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, kind, payload, linkname in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = linkname
            if kind in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            else:
                archive.addfile(member)
    return path


def _prepositioned_archive(
    destination: Path,
    url: str,
    entries: Iterable[tuple[str, bytes, bytes, str]],
) -> Path:
    return _write_archive(destination / url.rsplit("/", 1)[-1], entries)


def _kokoro_entries(
    *,
    extras: Iterable[tuple[str, bytes, bytes, str]] = (),
) -> list[tuple[str, bytes, bytes, str]]:
    return [
        _regular(f"{_KOKORO_ROOT}/model.int8.onnx", b"model"),
        _regular(f"{_KOKORO_ROOT}/voices.bin", b"voices"),
        _regular(f"{_KOKORO_ROOT}/tokens.txt", b"tokens"),
        _regular(f"{_KOKORO_ROOT}/lexicon-us-en.txt", b"lexicon"),
        _regular(f"{_KOKORO_ROOT}/espeak-ng-data/en_dict", b"dictionary"),
        *extras,
    ]


def _kws_entries(
    *,
    extras: Iterable[tuple[str, bytes, bytes, str]] = (),
) -> list[tuple[str, bytes, bytes, str]]:
    return [
        _regular(f"{_KWS_ROOT}/tokens.txt", b"tokens"),
        _regular(f"{_KWS_ROOT}/en.phone", _LEXICON),
        _regular(f"{_KWS_ROOT}/encoder-contract-chunk-16.int8.onnx", b"encoder"),
        _regular(f"{_KWS_ROOT}/decoder-contract-chunk-16.onnx", b"decoder"),
        _regular(f"{_KWS_ROOT}/joiner-contract-chunk-16.int8.onnx", b"joiner"),
        *extras,
    ]


def _assert_materialized_single_link(path: Path, payload: bytes) -> None:
    metadata = path.lstat()
    assert stat.S_ISREG(metadata.st_mode)
    assert metadata.st_nlink == 1
    assert path.read_bytes() == payload


@pytest.mark.parametrize(
    ("path_mode", "member_name", "expected_relative"),
    (
        ("flatten", "release/nested/model.onnx", Path("model.onnx")),
        (
            "strip_top_level",
            "release/nested/model.onnx",
            Path("nested/model.onnx"),
        ),
    ),
)
def test_private_extractor_preserves_normal_path_modes(
    tmp_path: Path,
    path_mode: str,
    member_name: str,
    expected_relative: Path,
) -> None:
    archive = _write_archive(tmp_path / "model.tar", [_regular(member_name, b"model")])
    destination = tmp_path / path_mode

    extracted = setup_models._extract_tar_members(
        str(archive),
        str(destination),
        select=lambda members: members,
        path_mode=path_mode,
        reject_parent_parts=True,
        unreadable_is_error=True,
    )

    expected = destination / expected_relative
    assert extracted == [str(expected)]
    _assert_materialized_single_link(expected, b"model")


def test_private_strip_mode_normalizes_repeated_and_dot_components(
    tmp_path: Path,
) -> None:
    archive = _write_archive(
        tmp_path / "model.tar",
        [_regular("release//./nested///model.onnx", b"model")],
    )
    destination = tmp_path / "destination"

    extracted = setup_models._extract_tar_members(
        str(archive),
        str(destination),
        select=lambda members: members,
        path_mode="strip_top_level",
        reject_parent_parts=True,
        unreadable_is_error=True,
    )

    expected = destination / "nested" / "model.onnx"
    assert extracted == [str(expected)]
    _assert_materialized_single_link(expected, b"model")


@pytest.mark.parametrize(
    ("path_mode", "expected_relative"),
    (
        ("flatten", Path("model.onnx")),
        ("strip_top_level", Path("nested/model.onnx")),
    ),
)
def test_private_extractor_reroots_absolute_member_names(
    tmp_path: Path,
    path_mode: str,
    expected_relative: Path,
) -> None:
    archive = _write_archive(
        tmp_path / "model.tar", [_regular("/release/nested/model.onnx", b"model")]
    )
    destination = tmp_path / path_mode

    extracted = setup_models._extract_tar_members(
        str(archive),
        str(destination),
        select=lambda members: members,
        path_mode=path_mode,
        reject_parent_parts=True,
        unreadable_is_error=True,
    )

    expected = destination / expected_relative
    assert extracted == [str(expected)]
    _assert_materialized_single_link(expected, b"model")
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize(
    "component",
    (
        "D:",
        "file:stream",
        ".. ",
        "...",
        "trailing.",
        "trailing ",
        "NUL.txt",
        "COM1.log",
    ),
)
def test_private_extractor_rejects_windows_ambiguous_components(
    tmp_path: Path,
    component: str,
) -> None:
    archive = _write_archive(
        tmp_path / "model.tar",
        [_regular(f"release/{component}/model.onnx", b"model")],
    )
    destination = tmp_path / "destination"

    with pytest.raises(ValueError, match="unsafe member path"):
        setup_models._extract_tar_members(
            str(archive),
            str(destination),
            select=lambda members: members,
            path_mode="strip_top_level",
            reject_parent_parts=True,
            unreadable_is_error=True,
        )

    assert not (destination / "model.onnx").exists()


def test_private_extractor_preserves_unreadable_skip_and_error_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    archive = _write_archive(
        tmp_path / "model.tar", [_regular("release/model.onnx", b"model")]
    )
    real_extractfile = tarfile.TarFile.extractfile

    def unreadable(self, member):
        if member.name.endswith("model.onnx"):
            return None
        return real_extractfile(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", unreadable)

    skipped = setup_models._extract_tar_members(
        str(archive),
        str(tmp_path / "skip"),
        select=lambda members: members,
        path_mode="flatten",
        reject_parent_parts=True,
        unreadable_is_error=False,
    )
    assert skipped == []
    assert not (tmp_path / "skip" / "model.onnx").exists()

    with pytest.raises(FileNotFoundError, match="could not read"):
        setup_models._extract_tar_members(
            str(archive),
            str(tmp_path / "error"),
            select=lambda members: members,
            path_mode="flatten",
            reject_parent_parts=True,
            unreadable_is_error=True,
        )
    assert not (tmp_path / "error" / "model.onnx").exists()


@pytest.mark.parametrize("package", ("extract", "kokoro", "kws"))
def test_public_extractors_reject_a_symlinked_destination_root(
    tmp_path: Path,
    package: str,
) -> None:
    real_destination = tmp_path / "real-destination"
    real_destination.mkdir()
    linked_destination = tmp_path / "linked-destination"
    linked_destination.symlink_to(real_destination, target_is_directory=True)

    if package == "extract":
        archive = _write_archive(
            tmp_path / "model.tar", [_regular("release/model.onnx", b"model")]
        )
        invoke = lambda: setup_models.extract_member(  # noqa: E731
            str(archive), "model.onnx", str(linked_destination)
        )
    elif package == "kokoro":
        _prepositioned_archive(real_destination, _KOKORO_URL, _kokoro_entries())
        invoke = lambda: setup_models.fetch_kokoro_package(  # noqa: E731
            str(linked_destination), _KOKORO_URL
        )
    else:
        _prepositioned_archive(real_destination, _KWS_URL, _kws_entries())
        invoke = lambda: setup_models.fetch_kws_package(  # noqa: E731
            str(linked_destination), _KWS_URL
        )

    with pytest.raises(
        ValueError, match="(?i)destination.*symlink|symlink.*destination"
    ):
        invoke()

    assert not (real_destination / "model.onnx").exists()
    assert not (real_destination / "model.int8.onnx").exists()
    assert not (real_destination / "tokens.txt").exists()


@pytest.mark.parametrize("package", ("punctuation", "parakeet"))
def test_archive_fetch_rejects_a_linked_root_before_download(
    monkeypatch,
    tmp_path: Path,
    package: str,
) -> None:
    real_destination = tmp_path / "real-destination"
    real_destination.mkdir()
    linked_destination = tmp_path / "linked-destination"
    linked_destination.symlink_to(real_destination, target_is_directory=True)
    calls: list[tuple[object, ...]] = []

    def unexpected_fetch(*args, **kwargs):
        calls.append((*args, kwargs))
        raise AssertionError("download must follow destination admission")

    monkeypatch.setattr(setup_models, "fetch_speaker_model", unexpected_fetch)
    if package == "punctuation":
        invoke = lambda: setup_models.fetch_punct_model(  # noqa: E731
            str(linked_destination),
            "https://example.invalid/punctuation.tar.bz2",
        )
    else:
        invoke = lambda: setup_models.fetch_parakeet_final(  # noqa: E731
            str(linked_destination),
            url="https://example.invalid/parakeet.tar.bz2",
            expected_sha256="0" * 64,
        )

    with pytest.raises(ValueError, match="unsafe symlink"):
        invoke()

    assert calls == []


def test_kokoro_rejects_a_nested_destination_symlink_without_writing_through_it(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "kokoro"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    nested = destination / "espeak-ng-data"
    nested.symlink_to(outside, target_is_directory=True)
    sentinel = outside / "en_dict"
    sentinel.write_bytes(b"outside-sentinel")
    entries = [
        _regular(f"{_KOKORO_ROOT}/espeak-ng-data/en_dict", b"replacement"),
        *_kokoro_entries()[:-1],
    ]
    _prepositioned_archive(destination, _KOKORO_URL, entries)

    with pytest.raises(ValueError, match="(?i)symlink"):
        setup_models.fetch_kokoro_package(str(destination), _KOKORO_URL)

    assert sentinel.read_bytes() == b"outside-sentinel"
    assert nested.is_symlink()


@pytest.mark.parametrize("package", ("extract", "kokoro", "kws"))
@pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
def test_existing_leaf_link_is_replaced_without_touching_its_external_target(
    tmp_path: Path,
    package: str,
    link_kind: str,
) -> None:
    destination = tmp_path / package
    destination.mkdir()
    external = tmp_path / f"{package}-external"
    external.write_bytes(b"external-sentinel")

    if package == "extract":
        leaf = destination / "model.onnx"
        replacement = b"new-model"
        archive = _write_archive(
            tmp_path / "model.tar", [_regular("release/model.onnx", replacement)]
        )
        invoke = lambda: setup_models.extract_member(  # noqa: E731
            str(archive), "model.onnx", str(destination)
        )
    elif package == "kokoro":
        leaf = destination / "voices.bin"
        replacement = b"voices"
        _prepositioned_archive(destination, _KOKORO_URL, _kokoro_entries())
        invoke = lambda: setup_models.fetch_kokoro_package(  # noqa: E731
            str(destination), _KOKORO_URL
        )
    else:
        leaf = destination / "tokens.txt"
        replacement = b"tokens"
        _prepositioned_archive(destination, _KWS_URL, _kws_entries())
        invoke = lambda: setup_models.fetch_kws_package(  # noqa: E731
            str(destination), _KWS_URL
        )

    if link_kind == "symlink":
        leaf.symlink_to(external)
    else:
        os.link(external, leaf)

    invoke()

    assert external.read_bytes() == b"external-sentinel"
    _assert_materialized_single_link(leaf, replacement)


def test_cross_platform_path_fallback_replaces_leaf_but_rejects_root_link(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        setup_models,
        "_descriptor_extract_available",
        lambda: False,
    )
    destination = tmp_path / "models"
    destination.mkdir()
    external = tmp_path / "external"
    external.write_bytes(b"external-sentinel")
    leaf = destination / "model.onnx"
    leaf.symlink_to(external)
    archive = _write_archive(
        tmp_path / "model.tar",
        [_regular("release/model.onnx", b"replacement")],
    )

    setup_models.extract_member(str(archive), "model.onnx", str(destination))

    assert external.read_bytes() == b"external-sentinel"
    _assert_materialized_single_link(leaf, b"replacement")

    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe symlink"):
        setup_models.extract_member(str(archive), "model.onnx", str(linked_root))


def test_cross_platform_fallback_rejects_a_directory_junction_marker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "junction-destination"
    destination.mkdir()
    archive = _write_archive(
        tmp_path / "model.tar",
        [_regular("release/model.onnx", b"replacement")],
    )
    monkeypatch.setattr(
        setup_models,
        "_descriptor_extract_available",
        lambda: False,
    )
    original_isjunction = getattr(
        setup_models.os.path, "isjunction", lambda _path: False
    )
    monkeypatch.setattr(
        setup_models.os.path,
        "isjunction",
        lambda path: (
            os.path.abspath(path) == str(destination) or original_isjunction(path)
        ),
        raising=False,
    )

    with pytest.raises(ValueError, match="unsafe symlink"):
        setup_models.extract_member(str(archive), "model.onnx", str(destination))

    assert not (destination / "model.onnx").exists()


def test_python311_fallback_rejects_any_directory_reparse_tag(monkeypatch) -> None:
    metadata = type(
        "SyntheticWindowsStat",
        (),
        {"st_mode": stat.S_IFDIR | 0o755, "st_reparse_tag": 0xA0000003},
    )()
    monkeypatch.delattr(setup_models.os.path, "isjunction", raising=False)

    assert setup_models._is_linklike_directory("C:\\models", metadata)


def test_copy_failure_preserves_the_previous_leaf(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "models"
    destination.mkdir()
    leaf = destination / "model.onnx"
    leaf.write_bytes(b"working-model")
    archive = _write_archive(
        tmp_path / "model.tar", [_regular("release/model.onnx", b"replacement")]
    )

    def fail_after_one_chunk(source, target, _declared_size):
        target.write(source.read(3))
        raise OSError("injected copy failure")

    monkeypatch.setattr(setup_models, "_copy_tar_source_exact", fail_after_one_chunk)

    with pytest.raises(OSError, match="injected copy failure"):
        setup_models.extract_member(str(archive), "model.onnx", str(destination))

    _assert_materialized_single_link(leaf, b"working-model")
    assert not list(destination.glob(".*.extract-*.part"))


def test_extract_member_flattens_parent_components_without_escape(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "safe" / "models"
    archive = _write_archive(
        tmp_path / "model.tar", [_regular("../../model.onnx", b"safe-model")]
    )

    extracted = Path(
        setup_models.extract_member(str(archive), "model.onnx", str(destination))
    )

    assert extracted == destination / "model.onnx"
    assert extracted.read_bytes() == b"safe-model"
    assert not (tmp_path / "model.onnx").exists()


@pytest.mark.parametrize("separator", ("/", "\\"))
@pytest.mark.parametrize("package", ("kokoro", "kws"))
def test_selected_parent_component_is_rejected_for_package_extractors(
    tmp_path: Path,
    package: str,
    separator: str,
) -> None:
    destination = tmp_path / package
    if package == "kokoro":
        unsafe = _regular(separator.join((_KOKORO_ROOT, "..", "voices.bin")), b"unsafe")
        entries = [unsafe, *_kokoro_entries()]
        url = _KOKORO_URL
        invoke = lambda: setup_models.fetch_kokoro_package(  # noqa: E731
            str(destination), url
        )
    else:
        # A trailing forward slash keeps the wanted basename visible to
        # ``os.path.basename`` on POSIX while the preceding separator still
        # exercises slash/backslash normalization in the common extractor.
        unsafe_name = f"{_KWS_ROOT}{separator}..{separator}nested/tokens.txt"
        unsafe = _regular(unsafe_name, b"unsafe")
        entries = [unsafe, *_kws_entries()]
        url = _KWS_URL
        invoke = lambda: setup_models.fetch_kws_package(  # noqa: E731
            str(destination), url
        )
    _prepositioned_archive(destination, url, entries)

    with pytest.raises(ValueError, match="unsafe member path"):
        invoke()

    assert not (tmp_path / "voices.bin").exists()
    assert not (tmp_path / "tokens.txt").exists()


def test_extract_member_keeps_first_matching_member(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path / "model.tar",
        [
            _regular("first/model.onnx", b"first"),
            _regular("second/model.onnx", b"second"),
        ],
    )

    extracted = setup_models.extract_member(
        str(archive), "model.onnx", str(tmp_path / "models")
    )

    assert Path(extracted).read_bytes() == b"first"


def test_sensevoice_two_member_call_shape_publishes_both_regular_files(
    tmp_path: Path,
) -> None:
    archive = _write_archive(
        tmp_path / "sense-voice.tar",
        [
            _regular("sense-voice/model.int8.onnx", b"sense-model"),
            _regular("sense-voice/tokens.txt", b"sense-tokens"),
        ],
    )
    destination = tmp_path / "sense-voice"

    model = Path(
        setup_models.extract_member(
            str(archive),
            "model.int8.onnx",
            str(destination),
        )
    )
    tokens = Path(
        setup_models.extract_member(
            str(archive),
            "tokens.txt",
            str(destination),
        )
    )

    _assert_materialized_single_link(model, b"sense-model")
    _assert_materialized_single_link(tokens, b"sense-tokens")


@pytest.mark.parametrize("package", ("kokoro", "kws"))
def test_package_duplicate_output_preserves_archive_order_last_write(
    tmp_path: Path,
    package: str,
) -> None:
    destination = tmp_path / package
    if package == "kokoro":
        entries = _kokoro_entries(
            extras=[_regular(f"{_KOKORO_ROOT}/tokens.txt", b"last-tokens")]
        )
        _prepositioned_archive(destination, _KOKORO_URL, entries)
        result = setup_models.fetch_kokoro_package(str(destination), _KOKORO_URL)
        leaf = Path(result["tts_tokens"])
    else:
        entries = _kws_entries(
            extras=[_regular("alternate/tokens.txt", b"last-tokens")]
        )
        _prepositioned_archive(destination, _KWS_URL, entries)
        result = setup_models.fetch_kws_package(str(destination), _KWS_URL)
        leaf = Path(result["kws_tokens"])

    assert leaf.read_bytes() == b"last-tokens"


@pytest.mark.parametrize(
    "kind",
    (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE),
)
def test_kokoro_skips_archive_links_and_special_members(
    tmp_path: Path,
    kind: bytes,
) -> None:
    destination = tmp_path / "kokoro"
    suspicious = _archive_entry(
        f"{_KOKORO_ROOT}/suspicious",
        kind=kind,
        linkname="../../outside" if kind in {tarfile.SYMTYPE, tarfile.LNKTYPE} else "",
    )
    _prepositioned_archive(
        destination,
        _KOKORO_URL,
        [suspicious, *_kokoro_entries()],
    )

    setup_models.fetch_kokoro_package(str(destination), _KOKORO_URL)

    assert not (destination / "suspicious").exists()
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize(
    "kind",
    (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE),
)
def test_extract_member_skips_nonregular_suffix_match(
    tmp_path: Path,
    kind: bytes,
) -> None:
    archive = _write_archive(
        tmp_path / "model.tar",
        [
            _archive_entry(
                "first/model.onnx",
                kind=kind,
                linkname="../../outside"
                if kind in {tarfile.SYMTYPE, tarfile.LNKTYPE}
                else "",
            ),
            _regular("second/model.onnx", b"regular-model"),
        ],
    )

    extracted = Path(
        setup_models.extract_member(
            str(archive), "model.onnx", str(tmp_path / "destination")
        )
    )

    _assert_materialized_single_link(extracted, b"regular-model")
    assert not (tmp_path / "outside").exists()


def test_kws_ignores_unselected_traversal_member(tmp_path: Path) -> None:
    destination = tmp_path / "kws"
    _prepositioned_archive(
        destination,
        _KWS_URL,
        [_regular(f"{_KWS_ROOT}/../../ignored.wav", b"ignored"), *_kws_entries()],
    )

    result = setup_models.fetch_kws_package(str(destination), _KWS_URL)

    assert all(Path(value).is_file() for value in result.values())
    assert not (tmp_path / "ignored.wav").exists()


def test_declared_member_size_limit_fails_before_writing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "models"
    archive = _write_archive(
        tmp_path / "model.tar", [_regular("release/model.onnx", b"12345")]
    )
    monkeypatch.setattr(setup_models, "_MAX_TAR_MEMBER_BYTES", 4)

    with pytest.raises(ValueError, match="(?i)member.*(size|limit|large)"):
        setup_models.extract_member(str(archive), "model.onnx", str(destination))

    assert not (destination / "model.onnx").exists()


def test_declared_total_size_limit_fails_before_writing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "kokoro"
    entries = _kokoro_entries()
    _prepositioned_archive(destination, _KOKORO_URL, entries)
    monkeypatch.setattr(setup_models, "_MAX_TAR_MEMBER_BYTES", 1_000)
    monkeypatch.setattr(
        setup_models,
        "_MAX_TAR_TOTAL_BYTES",
        sum(len(payload) for _, _, payload, _ in entries) - 1,
    )

    with pytest.raises(ValueError, match="(?i)(total|archive).*(size|limit|large)"):
        setup_models.fetch_kokoro_package(str(destination), _KOKORO_URL)

    assert not (destination / "model.int8.onnx").exists()
    assert not (destination / "voices.bin").exists()


def test_declared_member_count_limit_fails_before_writing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "kokoro"
    entries = _kokoro_entries()
    _prepositioned_archive(destination, _KOKORO_URL, entries)
    monkeypatch.setattr(setup_models, "_MAX_TAR_MEMBERS", len(entries) - 1)

    with pytest.raises(ValueError, match="(?i)(member|archive).*(count|many|limit)"):
        setup_models.fetch_kokoro_package(str(destination), _KOKORO_URL)

    assert not (destination / "model.int8.onnx").exists()
    assert not (destination / "voices.bin").exists()
