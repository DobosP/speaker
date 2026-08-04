from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from core.diagnostic_bundle import FinalModelInputReceipt, FinalModelInputRole
from tools.streaming_stt.private_diagnostic_receipt import (
    PrivateDiagnosticCaseSurface,
    PrivateDiagnosticReceiptError,
    create_private_diagnostic_receipt,
    verify_private_diagnostic_receipt,
)


def _fixture():
    cases = (
        PrivateDiagnosticCaseSurface(
            case_id="owner-vault-001",
            filename="owner-vault-001.f32le",
            sha256=hashlib.sha256(b"first").hexdigest(),
            samples=4,
            expected_text="private transcript canary one",
            assertion="transcript",
            commands=(),
            tags=(
                "private-diagnostic",
                "selected_asr_segment",
                "expected-tool.vault.search",
            ),
        ),
        PrivateDiagnosticCaseSurface(
            case_id="owner-negative-002",
            filename="owner-negative-002.f32le",
            sha256=hashlib.sha256(b"second").hexdigest(),
            samples=5,
            expected_text="private transcript canary two",
            assertion="transcript",
            commands=(),
            tags=(
                "private-diagnostic",
                "selected_asr_segment",
                "expected-tool.none",
            ),
        ),
    )
    receipts = tuple(
        FinalModelInputReceipt(
            input_index=7 + index,
            monotonic_ns=10 + index,
            stream_id="sherpa-11111111111111111111111111111111",
            utterance_id=f"u{index + 1}",
            capture_epoch=2,
            capture_generation=3,
            revision=1,
            role=FinalModelInputRole.SELECTED_ASR_SEGMENT,
            sample_rate_hz=16_000,
            sample_start=0,
            sample_end=case.samples,
            sha256=case.sha256,
        )
        for index, case in enumerate(cases)
    )
    return cases, receipts


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def test_v2_receipt_round_trip_binds_cases_without_disclosing_labels():
    cases, receipts = _fixture()

    raw = create_private_diagnostic_receipt(
        diagnostic_manifest_sha256="1" * 64,
        labels_sha256="2" * 64,
        cases=cases,
        selected_inputs=receipts,
    )

    payload = json.loads(raw)
    assert set(payload) == {
        "schema_version",
        "kind",
        "diagnostic_manifest_sha256",
        "labels_sha256",
        "case_binding_sha256",
        "selected_inputs",
    }
    assert payload["schema_version"] == 2
    assert payload["kind"] == "private-diagnostic-selection-v2"
    assert verify_private_diagnostic_receipt(
        raw,
        diagnostic_manifest_sha256="1" * 64,
        labels_sha256="2" * 64,
        cases=cases,
    ) == receipts
    assert b"private transcript canary" not in raw
    assert b"expected-tool" not in raw
    assert b"owner-vault" not in raw


@pytest.mark.parametrize(
    "mutation",
    ["expected-text", "route-tag", "tag-order", "case-id", "case-order"],
)
def test_v2_receipt_rejects_changed_case_surface(mutation):
    cases, receipts = _fixture()
    raw = create_private_diagnostic_receipt(
        diagnostic_manifest_sha256="1" * 64,
        labels_sha256="2" * 64,
        cases=cases,
        selected_inputs=receipts,
    )
    changed = list(cases)
    if mutation == "expected-text":
        changed[0] = replace(changed[0], expected_text="changed transcript")
    elif mutation == "route-tag":
        changed[0] = replace(
            changed[0],
            tags=(*changed[0].tags[:-1], "expected-tool.web.search"),
        )
    elif mutation == "tag-order":
        changed[0] = replace(changed[0], tags=tuple(reversed(changed[0].tags)))
    elif mutation == "case-id":
        changed[0] = replace(
            changed[0],
            case_id="owner-vault-renamed",
            filename="owner-vault-renamed.f32le",
        )
    else:
        changed.reverse()

    with pytest.raises(PrivateDiagnosticReceiptError):
        verify_private_diagnostic_receipt(
            raw,
            diagnostic_manifest_sha256="1" * 64,
            labels_sha256="2" * 64,
            cases=tuple(changed),
        )


@pytest.mark.parametrize("mutation", ["labels", "case-binding", "legacy-v1"])
def test_v2_receipt_rejects_changed_or_legacy_wrapper(mutation):
    cases, receipts = _fixture()
    raw = create_private_diagnostic_receipt(
        diagnostic_manifest_sha256="1" * 64,
        labels_sha256="2" * 64,
        cases=cases,
        selected_inputs=receipts,
    )
    payload = json.loads(raw)
    if mutation == "labels":
        payload["labels_sha256"] = "0" * 64
    elif mutation == "case-binding":
        payload["case_binding_sha256"] = "0" * 64
    else:
        payload = {
            "schema_version": 1,
            "kind": "private-diagnostic-selection-v1",
            "diagnostic_manifest_sha256": payload[
                "diagnostic_manifest_sha256"
            ],
            "selected_inputs": payload["selected_inputs"],
        }

    with pytest.raises(PrivateDiagnosticReceiptError):
        verify_private_diagnostic_receipt(
            _canonical(payload),
            diagnostic_manifest_sha256="1" * 64,
            labels_sha256="2" * 64,
            cases=cases,
        )
