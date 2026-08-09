from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import signal
import stat
import struct
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from tools.capture_replay import ami_natural_turn as fixture
from tools import prepare_ami_natural_turn_capture_replay as prepare


@dataclass(frozen=True)
class _Synthetic:
    annotations: Path
    close: Path
    far: Path
    lock: Path
    injection: fixture.AmiNaturalTurnTestInjection
    references: tuple[str, ...]


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True, mode=0o700)
    path.chmod(0o700)
    return path


def _private_file(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    path.chmod(0o600)
    return path


def _words_xml(
    rows: list[tuple[str, float, float, str, bool]],
    *,
    nonlexical_ids: frozenset[str] = frozenset(),
) -> bytes:
    rendered = [
        '<?xml version="1.0" encoding="ISO-8859-1"?>',
        '<nite:root xmlns:nite="http://nite.sourceforge.net/">',
    ]
    for identity, start, end, text, punctuation in rows:
        tag = "disfmarker" if identity in nonlexical_ids else "w"
        punc = ' punc="true"' if punctuation else ""
        rendered.append(
            f'<{tag} nite:id="{identity}" starttime="{start:g}" '
            f'endtime="{end:g}"{punc}>{html.escape(text)}</{tag}>'
        )
    rendered.append("</nite:root>")
    return "".join(rendered).encode("iso-8859-1")


def _root_xml(identity: str, rows: list[str]) -> bytes:
    return (
        '<?xml version="1.0" encoding="ISO-8859-1"?>'
        '<nite:root xmlns:nite="http://nite.sourceforge.net/" '
        f'nite:id="{identity}">{"".join(rows)}</nite:root>'
    ).encode("iso-8859-1")


def _dact(identity: str, type_href: str, words_href: str) -> str:
    return (
        f'<dact nite:id="{identity}">'
        f'<nite:pointer role="da-aspect" href="{type_href}"/>'
        f'<nite:child href="{words_href}"/>'
        "</dact>"
    )


def _segment(identity: str, start: float, end: float, words_href: str) -> str:
    return (
        f'<segment nite:id="{identity}" channel="0" '
        f'transcriber_start="{start:g}" transcriber_end="{end:g}">'
        f'<nite:child href="{words_href}"/>'
        "</segment>"
    )


def _wav(values: np.ndarray) -> bytes:
    payload = values.astype("<i2", copy=False).tobytes(order="C")
    fmt = struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


def _reference_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _annotation_payloads(*, semantic_drift: str | None = None) -> dict[str, bytes]:
    word_rows = {
        "A": [("ES2004a.A.words1", 1.0, 1.1, "alpha", False)],
        "B": [
            ("ES2004a.B.words488", 4.3, 4.6, "yes", False),
            ("ES2004a.B.words489", 4.6, 4.6, ".", True),
            ("ES2004a.B.words615", 8.8, 9.0, "reply", False),
            ("ES2004a.B.words616", 9.0, 9.4, "now", False),
            ("ES2004a.B.words617", 9.4, 9.4, ".", True),
            ("ES2004a.B.words626", 9.7, 9.8, "outside", False),
        ],
        "C": [("ES2004a.C.words1", 2.0, 2.1, "charlie", False)],
        "D": [
            ("ES2004a.D.words329", 3.7, 3.9, "before", False),
            ("ES2004a.D.words338", 4.0, 4.2, "host", False),
            ("ES2004a.D.words339", 4.2, 5.0, "continues", False),
            ("ES2004a.D.words340", 5.0, 5.0, ".", True),
            ("ES2004a.D.words348", 5.2, 5.3, "after", False),
            ("ES2004a.D.words428", 8.0, 8.2, "first", False),
            ("ES2004a.D.words429", 8.2, 8.3, "side", False),
            ("ES2004a.D.words430", 8.3, 8.5, "says", False),
            ("ES2004a.D.words431", 8.5, 8.7, "this", False),
            ("ES2004a.D.words432", 8.7, 8.7, ".", True),
        ],
    }
    if semantic_drift == "external-word":
        word_rows["A"].append(("ES2004a.A.words2", 4.4, 4.5, "intruder", False))
    payloads = {
        f"words/ES2004a.{speaker}.words.xml": _words_xml(rows)
        for speaker, rows in word_rows.items()
    }
    payloads["words/ES2004a.B.words.xml"] = _words_xml(
        word_rows["B"],
        nonlexical_ids=frozenset({"ES2004a.B.words626"}),
    )
    back_d = "ES2004a.D.words.xml#id(ES2004a.D.words338)..id(ES2004a.D.words340)"
    back_b = "ES2004a.B.words.xml#id(ES2004a.B.words488)..id(ES2004a.B.words489)"
    adjacent_d = "ES2004a.D.words.xml#id(ES2004a.D.words428)..id(ES2004a.D.words432)"
    adjacent_b = "ES2004a.B.words.xml#id(ES2004a.B.words615)..id(ES2004a.B.words617)"
    drifted_back_d = adjacent_d if semantic_drift == "da-href" else back_d
    payloads.update(
        {
            "dialogueActs/ES2004a.A.dialog-act.xml": _root_xml(
                "ES2004a.A.dialog-act", []
            ),
            "dialogueActs/ES2004a.B.dialog-act.xml": _root_xml(
                "ES2004a.B.dialog-act",
                [
                    _dact(
                        "ES2004a.B.dialog-act.s9553330.83",
                        "da-types.xml#id(ami_da_1)",
                        back_b,
                    ),
                    _dact(
                        "ES2004a.B.dialog-act.s9553330.107",
                        "da-types.xml#id(ami_da_4)",
                        adjacent_b,
                    ),
                ],
            ),
            "dialogueActs/ES2004a.C.dialog-act.xml": _root_xml(
                "ES2004a.C.dialog-act", []
            ),
            "dialogueActs/ES2004a.D.dialog-act.xml": _root_xml(
                "ES2004a.D.dialog-act",
                [
                    _dact(
                        "ES2004a.D.dialog-act.s9553330.59",
                        "da-types.xml#id(ami_da_16)",
                        drifted_back_d,
                    ),
                    _dact(
                        "ES2004a.D.dialog-act.s9553330.77",
                        "da-types.xml#id(ami_da_5)",
                        adjacent_d,
                    ),
                ],
            ),
            "segments/ES2004a.A.segments.xml": _root_xml("ES2004a.A.segs", []),
            "segments/ES2004a.B.segments.xml": _root_xml(
                "ES2004a.B.segs",
                [
                    _segment("ES2004a.sync.185", 4.3, 4.6, back_b),
                    _segment(
                        "ES2004a.sync.211",
                        8.8,
                        9.8,
                        "ES2004a.B.words.xml#id(ES2004a.B.words615)..id(ES2004a.B.words626)",
                    ),
                ],
            ),
            "segments/ES2004a.C.segments.xml": _root_xml("ES2004a.C.segs", []),
            "segments/ES2004a.D.segments.xml": _root_xml(
                "ES2004a.D.segs",
                [
                    _segment(
                        "ES2004a.sync.459",
                        3.7,
                        5.3 if semantic_drift != "segment" else 5.4,
                        "ES2004a.D.words.xml#id(ES2004a.D.words329)..id(ES2004a.D.words348)",
                    ),
                    _segment("ES2004a.sync.483", 8.0, 8.7, adjacent_d),
                ],
            ),
            "dialogueActs/ES2004a.adjacency-pairs.xml": _root_xml(
                "ES2004a.adjacency-pairs",
                [
                    '<adjacency-pair nite:id="ES2004a.adjacency-pairs.s9553330.23">'
                    '<nite:pointer role="type" href="ap-types.xml#id(apt_1)"/>'
                    '<nite:pointer role="source" '
                    'href="ES2004a.D.dialog-act.xml#id(ES2004a.D.dialog-act.s9553330.77)"/>'
                    '<nite:pointer role="target" '
                    f'href="ES2004a.B.dialog-act.xml#id(ES2004a.B.dialog-act.s9553330.{108 if semantic_drift == "adjacency" else 107})"/>'
                    "</adjacency-pair>"
                ],
            ),
            "ontologies/da-types.xml": (
                '<da-type xmlns:nite="http://nite.sourceforge.net/" nite:id="root">'
                '<da-type nite:id="ami_da_1"/><da-type nite:id="ami_da_4"/>'
                '<da-type nite:id="ami_da_5"/><da-type nite:id="ami_da_16"/>'
                "</da-type>"
            ).encode(),
            "ontologies/ap-types.xml": (
                '<ap-type xmlns:nite="http://nite.sourceforge.net/" nite:id="root">'
                '<ap-type nite:id="apt_1"/></ap-type>'
            ).encode(),
            "00README_MANUAL.txt": b"synthetic AMI-shape annotations\n",
            "LICENCE.txt": b"synthetic test material; CC-BY-4.0 gate shape only\n",
            "MANIFEST_MANUAL.txt": b"synthetic exact member manifest\n",
        }
    )
    if semantic_drift == "dtd":
        payloads["words/ES2004a.A.words.xml"] = (
            b'<!DOCTYPE root [<!ENTITY x "bad">]>'
            + payloads["words/ES2004a.A.words.xml"]
        )
    return payloads


def _zip_bytes(path: Path, payloads: dict[str, bytes]) -> bytes:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(payloads):
            archive.writestr(name, payloads[name])
    return path.read_bytes()


def _f32_case(
    values: np.ndarray, start: int, end: int, room_start: int, room_end: int
) -> tuple[str, str, str]:
    source_i16 = values[start:end]
    room_i16 = values[room_start:room_end]
    source_f32 = source_i16.astype(np.float32) * np.float32(1 / 32768)
    room_f32 = room_i16.astype(np.float32) * np.float32(1 / 32768)
    output = np.empty(128_000, dtype="<f4")
    output[:32_000] = room_f32
    source_end = 32_000 + source_f32.size
    output[32_000:source_end] = source_f32
    output[source_end:] = np.resize(room_f32, 128_000 - source_end)
    return (
        hashlib.sha256(source_i16.tobytes()).hexdigest(),
        hashlib.sha256(source_f32.tobytes()).hexdigest(),
        hashlib.sha256(output.tobytes()).hexdigest(),
    )


def _synthetic(
    tmp_path: Path,
    *,
    semantic_drift: str | None = None,
) -> _Synthetic:
    source_root = _private_dir(tmp_path / "private-sources")
    payloads = _annotation_payloads(semantic_drift=semantic_drift)
    annotation_path = source_root / "annotations.zip"
    annotation_raw = _zip_bytes(annotation_path, payloads)
    annotation_path.chmod(0o600)

    samples = 240_000
    base = ((np.arange(samples, dtype=np.int64) % 20_003) - 10_001).astype("<i2")
    close_values = base
    far_values = np.roll(base, 137).astype("<i2")
    close_raw = _wav(close_values)
    far_raw = _wav(far_values)
    close_path = _private_file(source_root / "close.wav", close_raw)
    far_path = _private_file(source_root / "far.wav", far_raw)

    production = json.loads(fixture.DEFAULT_LOCK.read_text(encoding="utf-8"))
    value = copy.deepcopy(production)
    value["fixture_id"] = "synthetic-ami-natural-turn-v1"
    value["production_evidence"] = False
    annotation = value["sources"]["annotation_archive"]
    annotation.update(
        {
            "filename": annotation_path.name,
            "url": "https://invalid.example/synthetic-annotations",
            "size_bytes": len(annotation_raw),
            "sha256": hashlib.sha256(annotation_raw).hexdigest(),
            "integrity_provenance": "synthetic-test-injection",
            "artifact_label": "synthetic-full-shape",
            "embedded_readme_release": "synthetic",
        }
    )
    room_start, room_end = 182_400, 214_400
    for item, channel, raw, values, path in zip(
        value["sources"]["audio"],
        ("close", "far"),
        (close_raw, far_raw),
        (close_values, far_values),
        (close_path, far_path),
    ):
        room_i16 = values[room_start:room_end]
        room_f32 = room_i16.astype(np.float32) * np.float32(1 / 32768)
        item.update(
            {
                "channel": channel,
                "filename": path.name,
                "url": f"https://invalid.example/{channel}",
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "integrity_provenance": "synthetic-test-injection",
                "samples": samples,
                "roomtone_pcm16_sha256": hashlib.sha256(room_i16.tobytes()).hexdigest(),
                "roomtone_f32_sha256": hashlib.sha256(room_f32.tobytes()).hexdigest(),
            }
        )

    roles = {
        item["path"]: item["role"]
        for item in production["archive_contract"]["required_members"]
    }
    value["archive_contract"]["required_members"] = [
        {
            "path": name,
            "role": roles[name],
            "size_bytes": len(payloads[name]),
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }
        for name in sorted(payloads)
    ]
    with zipfile.ZipFile(annotation_path) as archive:
        infos = archive.infolist()
        value["archive_contract"].update(
            {
                "member_count": len(infos),
                "expanded_bytes": sum(item.file_size for item in infos),
                "compressed_payload_bytes": sum(item.compress_size for item in infos),
                "archive_comment_bytes": len(archive.comment),
                "encrypted_members": sum(bool(item.flag_bits & 1) for item in infos),
                "zip64_sized_members": 0,
            }
        )

    back_d = "host continues."
    back_b = "yes."
    back_reference = "host continues. yes."
    adjacent_d = "first side says this."
    adjacent_b = "reply now."
    adjacent_reference = "first side says this. reply now."
    value["selection"]["windows"] = [
        {
            "window_id": "nested_backchannel",
            "start_sample": 64_000,
            "end_sample": 80_000,
            "linearized_reference_sha256": _reference_hash(back_reference),
            "dialogue_acts": [
                {
                    "id": "ES2004a.D.dialog-act.s9553330.59",
                    "speaker": "D",
                    "type_href": "da-types.xml#id(ami_da_16)",
                    "words_href": "ES2004a.D.words.xml#id(ES2004a.D.words338)..id(ES2004a.D.words340)",
                    "start_sample": 64_000,
                    "end_sample": 80_000,
                    "reference_sha256": _reference_hash(back_d),
                },
                {
                    "id": "ES2004a.B.dialog-act.s9553330.83",
                    "speaker": "B",
                    "type_href": "da-types.xml#id(ami_da_1)",
                    "words_href": "ES2004a.B.words.xml#id(ES2004a.B.words488)..id(ES2004a.B.words489)",
                    "start_sample": 68_800,
                    "end_sample": 73_600,
                    "reference_sha256": _reference_hash(back_b),
                },
            ],
            "segments": [
                {
                    "id": "ES2004a.sync.459",
                    "speaker": "D",
                    "words_href": "ES2004a.D.words.xml#id(ES2004a.D.words329)..id(ES2004a.D.words348)",
                    "start_sample": 59_200,
                    "end_sample": 84_800,
                },
                {
                    "id": "ES2004a.sync.185",
                    "speaker": "B",
                    "words_href": "ES2004a.B.words.xml#id(ES2004a.B.words488)..id(ES2004a.B.words489)",
                    "start_sample": 68_800,
                    "end_sample": 73_600,
                },
            ],
            "adjacency_pair": None,
            "relation": {
                "kind": "nested-overlap",
                "start_sample": 68_800,
                "end_sample": 73_600,
            },
            "timed_words": [
                {
                    "id": "ES2004a.D.words338",
                    "speaker": "D",
                    "start_sample": 64_000,
                    "end_sample": 67_200,
                    "punctuation": False,
                },
                {
                    "id": "ES2004a.D.words339",
                    "speaker": "D",
                    "start_sample": 67_200,
                    "end_sample": 80_000,
                    "punctuation": False,
                },
                {
                    "id": "ES2004a.D.words340",
                    "speaker": "D",
                    "start_sample": 80_000,
                    "end_sample": 80_000,
                    "punctuation": True,
                },
                {
                    "id": "ES2004a.B.words488",
                    "speaker": "B",
                    "start_sample": 68_800,
                    "end_sample": 73_600,
                    "punctuation": False,
                },
                {
                    "id": "ES2004a.B.words489",
                    "speaker": "B",
                    "start_sample": 73_600,
                    "end_sample": 73_600,
                    "punctuation": True,
                },
            ],
            "external_positive_duration_annotations": 0,
            "edge_complete_turn": False,
        },
        {
            "window_id": "adjacent_exchange",
            "start_sample": 128_000,
            "end_sample": 150_400,
            "linearized_reference_sha256": _reference_hash(adjacent_reference),
            "dialogue_acts": [
                {
                    "id": "ES2004a.D.dialog-act.s9553330.77",
                    "speaker": "D",
                    "type_href": "da-types.xml#id(ami_da_5)",
                    "words_href": "ES2004a.D.words.xml#id(ES2004a.D.words428)..id(ES2004a.D.words432)",
                    "start_sample": 128_000,
                    "end_sample": 139_200,
                    "reference_sha256": _reference_hash(adjacent_d),
                },
                {
                    "id": "ES2004a.B.dialog-act.s9553330.107",
                    "speaker": "B",
                    "type_href": "da-types.xml#id(ami_da_4)",
                    "words_href": "ES2004a.B.words.xml#id(ES2004a.B.words615)..id(ES2004a.B.words617)",
                    "start_sample": 140_800,
                    "end_sample": 150_400,
                    "reference_sha256": _reference_hash(adjacent_b),
                },
            ],
            "segments": [
                {
                    "id": "ES2004a.sync.483",
                    "speaker": "D",
                    "words_href": "ES2004a.D.words.xml#id(ES2004a.D.words428)..id(ES2004a.D.words432)",
                    "start_sample": 128_000,
                    "end_sample": 139_200,
                },
                {
                    "id": "ES2004a.sync.211",
                    "speaker": "B",
                    "words_href": "ES2004a.B.words.xml#id(ES2004a.B.words615)..id(ES2004a.B.words626)",
                    "start_sample": 140_800,
                    "end_sample": 156_800,
                },
            ],
            "adjacency_pair": {
                "id": "ES2004a.adjacency-pairs.s9553330.23",
                "type_href": "ap-types.xml#id(apt_1)",
                "source_href": "ES2004a.D.dialog-act.xml#id(ES2004a.D.dialog-act.s9553330.77)",
                "target_href": "ES2004a.B.dialog-act.xml#id(ES2004a.B.dialog-act.s9553330.107)",
            },
            "relation": {
                "kind": "adjacent-gap",
                "start_sample": 139_200,
                "end_sample": 140_800,
            },
            "timed_words": [
                {
                    "id": "ES2004a.D.words428",
                    "speaker": "D",
                    "start_sample": 128_000,
                    "end_sample": 131_200,
                    "punctuation": False,
                },
                {
                    "id": "ES2004a.D.words429",
                    "speaker": "D",
                    "start_sample": 131_200,
                    "end_sample": 132_800,
                    "punctuation": False,
                },
                {
                    "id": "ES2004a.D.words430",
                    "speaker": "D",
                    "start_sample": 132_800,
                    "end_sample": 136_000,
                    "punctuation": False,
                },
                {
                    "id": "ES2004a.D.words431",
                    "speaker": "D",
                    "start_sample": 136_000,
                    "end_sample": 139_200,
                    "punctuation": False,
                },
                {
                    "id": "ES2004a.D.words432",
                    "speaker": "D",
                    "start_sample": 139_200,
                    "end_sample": 139_200,
                    "punctuation": True,
                },
                {
                    "id": "ES2004a.B.words615",
                    "speaker": "B",
                    "start_sample": 140_800,
                    "end_sample": 144_000,
                    "punctuation": False,
                },
                {
                    "id": "ES2004a.B.words616",
                    "speaker": "B",
                    "start_sample": 144_000,
                    "end_sample": 150_400,
                    "punctuation": False,
                },
                {
                    "id": "ES2004a.B.words617",
                    "speaker": "B",
                    "start_sample": 150_400,
                    "end_sample": 150_400,
                    "punctuation": True,
                },
            ],
            "external_positive_duration_annotations": 0,
            "edge_complete_turn": False,
        },
    ]
    value["selection"]["selected_windows_sha256"] = fixture._canonical_sha256(
        fixture._selected_windows_binding(value["selection"])
    )
    value["transform"].update(
        {
            "silence_window_start_sample": 166_400,
            "silence_window_end_sample": 230_400,
            "roomtone_start_sample": room_start,
            "roomtone_end_sample": room_end,
        }
    )
    case_specs = [
        (
            "synthetic-back-close",
            "nested_backchannel",
            "close",
            64_000,
            80_000,
            close_values,
        ),
        ("synthetic-back-far", "nested_backchannel", "far", 64_000, 80_000, far_values),
        (
            "synthetic-adjacent-close",
            "adjacent_exchange",
            "close",
            128_000,
            150_400,
            close_values,
        ),
        (
            "synthetic-adjacent-far",
            "adjacent_exchange",
            "far",
            128_000,
            150_400,
            far_values,
        ),
    ]
    outputs = []
    for case_id, window_id, channel, start, end, values in case_specs:
        i16_hash, f32_hash, output_hash = _f32_case(
            values, start, end, room_start, room_end
        )
        outputs.append(
            {
                "case_id": case_id,
                "window_id": window_id,
                "channel": channel,
                "file": f"{case_id}.f32le",
                "samples": 128_000,
                "size_bytes": 512_000,
                "source_pcm16_sha256": i16_hash,
                "source_f32_sha256": f32_hash,
                "sha256": output_hash,
            }
        )
    value["outputs"]["cases"] = outputs
    value["layout"]["case_order"] = [item["case_id"] for item in outputs]
    value["self_digest"]["value"] = "0" * 64
    value["self_digest"]["value"] = fixture._canonical_self_digest(value)
    lock_raw = fixture._canonical_json(value, newline=True)
    lock_path = _private_file(source_root / "synthetic.lock.json", lock_raw)
    injection = fixture.AmiNaturalTurnTestInjection(
        fixture_id=value["fixture_id"],
        lock_sha256=hashlib.sha256(lock_raw).hexdigest(),
        lock_size_bytes=len(lock_raw),
    )
    return _Synthetic(
        annotation_path,
        close_path,
        far_path,
        lock_path,
        injection,
        (back_reference, back_d, back_b, adjacent_reference, adjacent_d, adjacent_b),
    )


def _prepare(tmp_path: Path, *, semantic_drift: str | None = None):
    source = _synthetic(tmp_path, semantic_drift=semantic_drift)
    output_parent = _private_dir(tmp_path / "private-output-parent")
    output = output_parent / "bundle"
    prepared = prepare.prepare_ami_natural_turn_capture_replay(
        annotations_zip=source.annotations,
        close_wav=source.close,
        far_wav=source.far,
        output_dir=output,
        accepted_terms=("CC-BY-4.0",),
        lock_path=source.lock,
        _test_injection=source.injection,
    )
    loaded = fixture.load_ami_natural_turn_fixture(
        output,
        lock_path=source.lock,
        _test_injection=source.injection,
    )
    return prepared, loaded, source


def test_production_lock_is_exact_self_bound_and_transcript_free() -> None:
    raw = fixture.DEFAULT_LOCK.read_bytes()
    value = json.loads(raw)
    loaded = fixture._load_fixture_lock()

    assert len(raw) == fixture.LOCK_FILE_BYTES == 14_424
    assert hashlib.sha256(raw).hexdigest() == fixture.LOCK_FILE_SHA256
    assert loaded.recipe_sha256 == value["self_digest"]["value"]
    assert fixture._canonical_self_digest(value) == loaded.recipe_sha256
    assert len(loaded.members) == 18
    assert value["sources"]["audio"][0]["sha256"] == (
        "3e2560b19bee6952c7c7ce041b0f1ea8a7ea9468044c4eea79d2a2c67e24ab0f"
    )
    assert value["selection"]["selected_windows_sha256"] == (
        "67ed6b3966f790531480ecf09e8eb1bec4c9030a8c050d7e8c0f8a59f8541ffe"
    )
    assert b"host continues" not in raw
    assert b"reply now" not in raw


def test_synthetic_full_shape_prepares_terminal_private_bundle(tmp_path: Path) -> None:
    prepared, loaded, source = _prepare(tmp_path)

    with zipfile.ZipFile(source.annotations) as archive:
        assert b'<disfmarker nite:id="ES2004a.B.words626"' in archive.read(
            "words/ES2004a.B.words.xml"
        )

    assert prepared.production_evidence is loaded.production_evidence is False
    assert prepared.fixture_id == loaded.fixture_id == source.injection.fixture_id
    assert prepared.lock_recipe_sha256 == loaded.lock_recipe_sha256
    assert prepared.labels_sha256 == loaded.labels_sha256
    assert prepared.receipt_sha256 == loaded.receipt_sha256
    assert prepared.case_count == len(loaded.cases) == 4
    assert [case.case_id for case in loaded.cases] == [
        "synthetic-back-close",
        "synthetic-back-far",
        "synthetic-adjacent-close",
        "synthetic-adjacent-far",
    ]
    assert [case.overlap_interval is not None for case in loaded.cases] == [
        True,
        True,
        False,
        False,
    ]
    assert [case.gap_interval is not None for case in loaded.cases] == [
        False,
        False,
        True,
        True,
    ]
    assert loaded.cases[0].overlap_interval == fixture.SampleInterval(36_800, 41_600)
    assert loaded.cases[2].gap_interval == fixture.SampleInterval(43_200, 44_800)
    assert loaded.cases[0].source_complete_sample == 128_000
    assert loaded.cases[0].overlap_references == source.references[1:3]
    assert "host continues" not in repr(loaded)
    assert "host continues" not in repr(loaded.cases[0])
    assert "host continues" not in repr(loaded.cases[0].dialogue_acts[0])
    with pytest.raises(FrozenInstanceError):
        loaded.cases[0].channel = "far"  # type: ignore[misc]

    expected_leaves = {
        "capture-replay.json",
        "ami-natural-turn.json",
        "preparation-receipt.json",
        "synthetic-back-close.f32le",
        "synthetic-back-far.f32le",
        "synthetic-adjacent-close.f32le",
        "synthetic-adjacent-far.f32le",
    }
    assert {item.name for item in loaded.root.iterdir()} == expected_leaves
    assert stat.S_IMODE(loaded.root.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE((loaded.root / name).stat().st_mode) == 0o600
        for name in expected_leaves
    )
    receipt_raw = (loaded.root / "preparation-receipt.json").read_bytes()
    assert all(text.encode() not in receipt_raw for text in source.references)
    assert str(source.annotations).encode() not in receipt_raw
    fixture.verify_ami_natural_turn_fixture_snapshot(loaded)


def test_generic_manifest_keeps_overlap_and_transition_semantics(
    tmp_path: Path,
) -> None:
    _prepared, loaded, _source = _prepare(tmp_path)
    back_close, back_far, adjacent_close, adjacent_far = loaded.replay_corpus.cases

    assert "overlapping-speakers" in back_close.tags
    assert "overlapping-speakers" in back_far.tags
    assert "turn-transition" not in back_close.tags
    assert "turn-transition" in adjacent_close.tags
    assert "turn-transition" in adjacent_far.tags
    assert "overlapping-speakers" not in adjacent_close.tags
    assert all(
        case.assertion.value == "transcript" for case in loaded.replay_corpus.cases
    )
    assert [case.expected_text for case in loaded.replay_corpus.cases] == [
        loaded.cases[index].reference for index in range(4)
    ]


def _call_prepare(
    tmp_path: Path,
    source: _Synthetic,
    *,
    output: Path | None = None,
    accepted_terms=("CC-BY-4.0",),
    **kwargs,
):
    if output is None:
        output = _private_dir(tmp_path / "output-parent") / "bundle"
    return prepare.prepare_ami_natural_turn_capture_replay(
        annotations_zip=source.annotations,
        close_wav=source.close,
        far_wav=source.far,
        output_dir=output,
        accepted_terms=accepted_terms,
        lock_path=source.lock,
        _test_injection=source.injection,
        **kwargs,
    )


@pytest.mark.parametrize(
    "semantic_drift",
    ["dtd", "da-href", "segment", "adjacency", "external-word"],
)
def test_self_consistent_source_semantic_drift_fails_before_publication(
    tmp_path: Path, semantic_drift: str
) -> None:
    source = _synthetic(tmp_path, semantic_drift=semantic_drift)
    output = _private_dir(tmp_path / "output-parent") / "bundle"

    with pytest.raises(prepare.AmiNaturalTurnPreparationError):
        _call_prepare(tmp_path, source, output=output)

    assert not output.exists()


@pytest.mark.parametrize(
    "accepted_terms",
    [(), ("CC-BY-SA-4.0",), ("CC-BY-4.0", "CC-BY-4.0"), "CC-BY-4.0"],
)
def test_license_gate_is_exact_and_prepublication(
    tmp_path: Path, accepted_terms
) -> None:
    source = _synthetic(tmp_path)
    output = _private_dir(tmp_path / "output-parent") / "bundle"

    with pytest.raises(prepare.AmiNaturalTurnPreparationError):
        _call_prepare(tmp_path, source, output=output, accepted_terms=accepted_terms)

    assert not output.exists()


def test_private_sources_reject_mode_symlink_and_hardlink(tmp_path: Path) -> None:
    source = _synthetic(tmp_path)
    output_parent = _private_dir(tmp_path / "output-parent")

    source.close.chmod(0o644)
    with pytest.raises(prepare.AmiNaturalTurnPreparationError):
        _call_prepare(tmp_path, source, output=output_parent / "mode")
    source.close.chmod(0o600)

    symlink = source.close.parent / "close-link.wav"
    symlink.symlink_to(source.close)
    with pytest.raises(prepare.AmiNaturalTurnPreparationError):
        prepare.prepare_ami_natural_turn_capture_replay(
            annotations_zip=source.annotations,
            close_wav=symlink,
            far_wav=source.far,
            output_dir=output_parent / "symlink",
            accepted_terms=("CC-BY-4.0",),
            lock_path=source.lock,
            _test_injection=source.injection,
        )
    symlink.unlink()

    hardlink = source.close.parent / "close-hard.wav"
    os.link(source.close, hardlink)
    with pytest.raises(prepare.AmiNaturalTurnPreparationError):
        _call_prepare(tmp_path, source, output=output_parent / "hardlink")


def test_output_is_absolute_absent_outside_source_and_git_trees(tmp_path: Path) -> None:
    source = _synthetic(tmp_path)
    with pytest.raises(prepare.AmiNaturalTurnPreparationError):
        _call_prepare(tmp_path, source, output=Path("relative-bundle"))
    with pytest.raises(prepare.AmiNaturalTurnPreparationError):
        _call_prepare(tmp_path, source, output=source.close.parent / "nested-output")

    git_parent = _private_dir(tmp_path / "git-parent")
    _private_dir(git_parent / ".git")
    with pytest.raises(prepare.AmiNaturalTurnPreparationError):
        _call_prepare(tmp_path, source, output=git_parent / "bundle")

    output = _private_dir(tmp_path / "output-parent") / "bundle"
    output.mkdir(mode=0o700)
    with pytest.raises(prepare.AmiNaturalTurnPreparationError):
        _call_prepare(tmp_path, source, output=output)


def test_prelink_source_leaf_and_extra_entry_attacks_publish_no_receipt(
    tmp_path: Path,
) -> None:
    for attack in ("source", "leaf", "extra"):
        root = _private_dir(tmp_path / attack)
        source = _synthetic(root)
        output = _private_dir(root / "output-parent") / "bundle"

        def guard() -> None:
            if attack == "source":
                raw = bytearray(source.close.read_bytes())
                raw[-1] ^= 1
                source.close.write_bytes(raw)
                source.close.chmod(0o600)
            elif attack == "leaf":
                leaf = output / "synthetic-back-close.f32le"
                raw = bytearray(leaf.read_bytes())
                raw[0] ^= 1
                leaf.write_bytes(raw)
                leaf.chmod(0o600)
            else:
                _private_file(output / "unexpected.bin", b"unexpected")

        with pytest.raises(prepare.AmiNaturalTurnPreparationError):
            _call_prepare(root, source, output=output, _commit_guard=guard)
        assert output.exists()
        assert not (output / "preparation-receipt.json").exists()


def test_prelink_staged_receipt_mutation_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _synthetic(tmp_path)
    output = _private_dir(tmp_path / "output-parent") / "bundle"
    original_read = prepare.os.read
    changed = False

    def corrupt_staged(descriptor: int, size: int) -> bytes:
        nonlocal changed
        raw = original_read(descriptor, size)
        if raw and not changed and os.fstat(descriptor).st_nlink == 0:
            changed = True
            return bytes([raw[0] ^ 1]) + raw[1:]
        return raw

    monkeypatch.setattr(prepare.os, "read", corrupt_staged)
    with pytest.raises(prepare.AmiNaturalTurnPreparationError):
        _call_prepare(tmp_path, source, output=output)
    assert changed is True
    assert not (output / "preparation-receipt.json").exists()


def test_prelink_baseexception_is_preserved(tmp_path: Path) -> None:
    class StopNow(BaseException):
        pass

    source = _synthetic(tmp_path)
    output = _private_dir(tmp_path / "output-parent") / "bundle"

    def stop() -> None:
        raise StopNow()

    with pytest.raises(StopNow):
        _call_prepare(tmp_path, source, output=output, _commit_guard=stop)
    assert not (output / "preparation-receipt.json").exists()


def test_link_created_then_exception_is_committed_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _synthetic(tmp_path)
    output = _private_dir(tmp_path / "output-parent") / "bundle"
    original_link = prepare.os.link

    def linked_then_raise(*args, **kwargs):
        original_link(*args, **kwargs)
        raise OSError("synthetic post-link failure")

    monkeypatch.setattr(prepare.os, "link", linked_then_raise)
    prepared = _call_prepare(tmp_path, source, output=output)

    assert (
        prepared.receipt_sha256
        == hashlib.sha256(
            (output / "preparation-receipt.json").read_bytes()
        ).hexdigest()
    )
    loaded = fixture.load_ami_natural_turn_fixture(
        output,
        lock_path=source.lock,
        _test_injection=source.injection,
    )
    assert loaded.receipt_sha256 == prepared.receipt_sha256


def test_loader_and_reverification_reject_leaf_mode_link_and_extra_drift(
    tmp_path: Path,
) -> None:
    for attack in ("bytes", "mode", "hardlink", "extra"):
        root = _private_dir(tmp_path / attack)
        _prepared, loaded, source = _prepare(root)
        leaf = loaded.root / "synthetic-back-close.f32le"
        if attack == "bytes":
            raw = bytearray(leaf.read_bytes())
            raw[-1] ^= 1
            leaf.write_bytes(raw)
            leaf.chmod(0o600)
        elif attack == "mode":
            leaf.chmod(0o640)
        elif attack == "hardlink":
            os.link(leaf, loaded.root.parent / "foreign-hardlink")
        else:
            _private_file(loaded.root / "extra.bin", b"extra")

        with pytest.raises(fixture.AmiNaturalTurnFixtureError):
            fixture.verify_ami_natural_turn_fixture_snapshot(loaded)
        with pytest.raises(fixture.AmiNaturalTurnFixtureError):
            fixture.load_ami_natural_turn_fixture(
                loaded.root,
                lock_path=source.lock,
                _test_injection=source.injection,
            )


def test_loader_rejects_self_consistent_reference_manifest_receipt_rewrite(
    tmp_path: Path,
) -> None:
    _prepared, loaded, source = _prepare(tmp_path)
    metadata_path = loaded.root / "ami-natural-turn.json"
    manifest_path = loaded.root / "capture-replay.json"
    receipt_path = loaded.root / "preparation-receipt.json"
    metadata = json.loads(metadata_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())

    forged_act = "forged continues."
    forged_window = "forged continues. yes."
    metadata["cases"][0]["dialogue_acts"][0]["reference"] = forged_act
    metadata["cases"][0]["dialogue_acts"][0]["reference_sha256"] = _reference_hash(
        forged_act
    )
    metadata["cases"][0]["reference"] = forged_window
    metadata["cases"][0]["reference_sha256"] = _reference_hash(forged_window)
    parsed = tuple(
        fixture._parse_metadata_case(item, index=index)
        for index, item in enumerate(metadata["cases"])
    )
    metadata["labels_sha256"] = fixture._canonical_sha256(fixture._label_rows(parsed))
    metadata_raw = fixture._canonical_json(metadata, newline=True)
    _private_file(metadata_path, metadata_raw)

    manifest["cases"][0]["word_intervals"][0]["text"] = "forged"
    manifest["cases"][0]["expected_text"] = forged_window
    manifest_raw = fixture._canonical_json(manifest, newline=True)
    _private_file(manifest_path, manifest_raw)
    receipt["labels_sha256"] = metadata["labels_sha256"]
    receipt["metadata"].update(
        {
            "size_bytes": len(metadata_raw),
            "sha256": hashlib.sha256(metadata_raw).hexdigest(),
        }
    )
    receipt["manifest"].update(
        {
            "size_bytes": len(manifest_raw),
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        }
    )
    receipt["binding_sha256"] = fixture._receipt_binding(receipt)
    _private_file(receipt_path, fixture._canonical_json(receipt, newline=True))

    with pytest.raises(fixture.AmiNaturalTurnFixtureError):
        fixture.load_ami_natural_turn_fixture(
            loaded.root,
            lock_path=source.lock,
            _test_injection=source.injection,
        )


def test_loader_recomputes_source_contract_instead_of_trusting_sidecars(
    tmp_path: Path,
) -> None:
    _prepared, loaded, source = _prepare(tmp_path)
    metadata_path = loaded.root / "ami-natural-turn.json"
    receipt_path = loaded.root / "preparation-receipt.json"
    metadata = json.loads(metadata_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    forged = hashlib.sha256(b"forged-source-contract").hexdigest()
    metadata["source_contract_sha256"] = forged
    metadata_raw = fixture._canonical_json(metadata, newline=True)
    _private_file(metadata_path, metadata_raw)
    receipt["source_contract_sha256"] = forged
    receipt["metadata"].update(
        {
            "size_bytes": len(metadata_raw),
            "sha256": hashlib.sha256(metadata_raw).hexdigest(),
        }
    )
    receipt["binding_sha256"] = fixture._receipt_binding(receipt)
    _private_file(receipt_path, fixture._canonical_json(receipt, newline=True))

    with pytest.raises(fixture.AmiNaturalTurnFixtureError):
        fixture.load_ami_natural_turn_fixture(
            loaded.root,
            lock_path=source.lock,
            _test_injection=source.injection,
        )


def test_test_injection_cannot_claim_production_identity_or_default_alias(
    tmp_path: Path,
) -> None:
    source = _synthetic(tmp_path)
    forbidden = fixture.AmiNaturalTurnTestInjection(
        fixture_id=fixture.FIXTURE_ID,
        lock_sha256=source.injection.lock_sha256,
        lock_size_bytes=source.injection.lock_size_bytes,
    )
    with pytest.raises(fixture.AmiNaturalTurnFixtureError):
        fixture._load_fixture_lock(source.lock, test_injection=forbidden)

    alias = _private_file(
        tmp_path / "production-alias.json", fixture.DEFAULT_LOCK.read_bytes()
    )
    with pytest.raises(fixture.AmiNaturalTurnFixtureError):
        fixture._load_fixture_lock(alias)


def test_clean_process_local_closure_is_exact() -> None:
    script = """
import json
import pathlib
import sys
from tools import prepare_ami_natural_turn_capture_replay
root = pathlib.Path.cwd().resolve()
rows = []
for module in sys.modules.values():
    path = getattr(module, '__file__', None)
    if not path:
        continue
    try:
        relative = pathlib.Path(path).resolve().relative_to(root)
    except (OSError, ValueError):
        continue
    if (
        relative.suffix == '.py'
        and relative.parts
        and relative.parts[0] != '.venv'
    ):
        rows.append(relative.as_posix())
print(json.dumps(sorted(set(rows))))
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    observed = tuple(json.loads(completed.stdout))
    assert observed == fixture._PREPARER_FILES


def test_prelink_lifecycle_signal_is_preserved_without_receipt(tmp_path: Path) -> None:
    source = _synthetic(tmp_path)
    output = _private_dir(tmp_path / "output-parent") / "bundle"

    def stop() -> None:
        raise prepare._LifecycleSignal(15)

    with pytest.raises(prepare._LifecycleSignal) as raised:
        _call_prepare(tmp_path, source, output=output, _commit_guard=stop)
    assert raised.value.signum == 15
    assert not (output / "preparation-receipt.json").exists()


def test_main_call_to_store_interrupt_after_real_commit_is_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _synthetic(tmp_path)
    output = _private_dir(tmp_path / "output-parent") / "bundle"
    original = prepare.prepare_ami_natural_turn_capture_replay
    captured: dict[str, prepare.PreparedAmiNaturalTurnFixture] = {}

    def committed_then_interrupt(**kwargs):
        captured["prepared"] = original(
            annotations_zip=source.annotations,
            close_wav=source.close,
            far_wav=source.far,
            output_dir=output,
            accepted_terms=("CC-BY-4.0",),
            lock_path=source.lock,
            _test_injection=source.injection,
            _commit_state=kwargs["_commit_state"],
        )
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        prepare,
        "prepare_ami_natural_turn_capture_replay",
        committed_then_interrupt,
    )
    code = prepare.main(
        [
            "--annotations-zip",
            str(source.annotations),
            "--close-wav",
            str(source.close),
            "--far-wav",
            str(source.far),
            "--output-dir",
            str(output),
            "--accept-license",
            "CC-BY-4.0",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == prepare._success_result(
        captured["prepared"]
    )
    loaded = fixture.load_ami_natural_turn_fixture(
        output,
        lock_path=source.lock,
        _test_injection=source.injection,
    )
    assert loaded.receipt_sha256 == captured["prepared"].receipt_sha256
    fixture.verify_ami_natural_turn_fixture_snapshot(loaded)


@pytest.mark.parametrize(
    ("signum", "expected"),
    [
        (signal.SIGHUP, 129),
        (signal.SIGTERM, 143),
        (signal.SIGINT, 130),
    ],
)
def test_main_real_signal_prelink_returns_signal_status_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    signum: int,
    expected: int,
) -> None:
    source = _synthetic(tmp_path)
    output = _private_dir(tmp_path / "output-parent") / "bundle"
    original = prepare.prepare_ami_natural_turn_capture_replay
    previous = signal.getsignal(signum)

    def interrupted(**kwargs):
        def send_signal() -> None:
            if signum != signal.SIGINT:
                assert signal.getsignal(signum) is prepare._lifecycle_handler
            os.kill(os.getpid(), signum)
            raise AssertionError("signal handler did not interrupt")

        return original(
            annotations_zip=source.annotations,
            close_wav=source.close,
            far_wav=source.far,
            output_dir=output,
            accepted_terms=("CC-BY-4.0",),
            lock_path=source.lock,
            _test_injection=source.injection,
            _commit_state=kwargs["_commit_state"],
            _commit_guard=send_signal,
        )

    monkeypatch.setattr(
        prepare,
        "prepare_ami_natural_turn_capture_replay",
        interrupted,
    )
    code = prepare.main(
        [
            "--annotations-zip",
            str(source.annotations),
            "--close-wav",
            str(source.close),
            "--far-wav",
            str(source.far),
            "--output-dir",
            str(output),
            "--accept-license",
            "CC-BY-4.0",
        ]
    )

    assert code == expected
    assert signal.getsignal(signum) == previous
    assert json.loads(capsys.readouterr().out) == prepare._SAFE_ERROR
    assert not (output / "preparation-receipt.json").exists()


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGTERM, signal.SIGINT])
def test_main_real_signal_postlink_recovers_full_success_and_valid_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    signum: int,
) -> None:
    source = _synthetic(tmp_path)
    output = _private_dir(tmp_path / "output-parent") / "bundle"
    original = prepare.prepare_ami_natural_turn_capture_replay
    previous = signal.getsignal(signum)
    captured: dict[str, prepare.PreparedAmiNaturalTurnFixture] = {}

    def committed_then_signal(**kwargs):
        captured["prepared"] = original(
            annotations_zip=source.annotations,
            close_wav=source.close,
            far_wav=source.far,
            output_dir=output,
            accepted_terms=("CC-BY-4.0",),
            lock_path=source.lock,
            _test_injection=source.injection,
            _commit_state=kwargs["_commit_state"],
        )
        if signum != signal.SIGINT:
            assert signal.getsignal(signum) is prepare._lifecycle_handler
        os.kill(os.getpid(), signum)
        raise AssertionError("signal handler did not interrupt")

    monkeypatch.setattr(
        prepare,
        "prepare_ami_natural_turn_capture_replay",
        committed_then_signal,
    )
    code = prepare.main(
        [
            "--annotations-zip",
            str(source.annotations),
            "--close-wav",
            str(source.close),
            "--far-wav",
            str(source.far),
            "--output-dir",
            str(output),
            "--accept-license",
            "CC-BY-4.0",
        ]
    )

    assert code == 0
    assert signal.getsignal(signum) == previous
    assert json.loads(capsys.readouterr().out) == prepare._success_result(
        captured["prepared"]
    )
    loaded = fixture.load_ami_natural_turn_fixture(
        output,
        lock_path=source.lock,
        _test_injection=source.injection,
    )
    assert loaded.receipt_sha256 == captured["prepared"].receipt_sha256
    fixture.verify_ami_natural_turn_fixture_snapshot(loaded)


@pytest.mark.parametrize(
    ("error", "expected"),
    [(KeyboardInterrupt(), 130), (prepare._LifecycleSignal(15), 143)],
)
def test_main_preserves_precommit_interrupt_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
    expected: int,
) -> None:
    def fail(**_kwargs):
        raise error

    monkeypatch.setattr(prepare, "prepare_ami_natural_turn_capture_replay", fail)
    code = prepare.main(
        [
            "--annotations-zip",
            "/private/a",
            "--close-wav",
            "/private/b",
            "--far-wav",
            "/private/c",
            "--output-dir",
            "/private/d",
            "--accept-license",
            "CC-BY-4.0",
        ]
    )
    assert code == expected
    assert json.loads(capsys.readouterr().out) == prepare._SAFE_ERROR


def test_main_stdout_failure_after_commit_stays_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[dict[str, object]] = []
    prepared = prepare.PreparedAmiNaturalTurnFixture(
        Path("/private/bundle"),
        "synthetic",
        False,
        "0" * 64,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
        "6" * 64,
        "7" * 64,
        4,
        128_000,
    )

    def committed(**kwargs):
        kwargs["_commit_state"].pending_prepared = prepared
        kwargs["_commit_state"].committed = True
        return prepared

    def broken_stdout(value: str, **_kwargs):
        printed.append(json.loads(value))
        raise OSError("synthetic stdout failure")

    monkeypatch.setattr(prepare, "prepare_ami_natural_turn_capture_replay", committed)
    monkeypatch.setattr(prepare, "print", broken_stdout, raising=False)
    assert (
        prepare.main(
            [
                "--annotations-zip",
                "/private/a",
                "--close-wav",
                "/private/b",
                "--far-wav",
                "/private/c",
                "--output-dir",
                "/private/d",
                "--accept-license",
                "CC-BY-4.0",
            ]
        )
        == 0
    )
    assert printed == [prepare._success_result(prepared)]
