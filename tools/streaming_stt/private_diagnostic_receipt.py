"""Closed receipt binding for private diagnostic streaming-STT corpora."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import re
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from core.diagnostic_bundle import FinalModelInputReceipt


_RECEIPT_SCHEMA_VERSION = 2
_RECEIPT_KIND = "private-diagnostic-selection-v2"
_CASE_BINDING_SCHEMA_VERSION = 1
_CASE_BINDING_DOMAIN = b"speaker-private-diagnostic-case-binding-v1\0"
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RECEIPT_FIELDS = {
    "schema_version",
    "kind",
    "diagnostic_manifest_sha256",
    "labels_sha256",
    "case_binding_sha256",
    "selected_inputs",
}


class PrivateDiagnosticReceiptError(RuntimeError):
    """A detail-free malformed or inconsistent private receipt."""


@dataclass(frozen=True, slots=True)
class PrivateDiagnosticCaseSurface:
    """The exact schema-v3 case surface covered by the private receipt."""

    case_id: str = field(repr=False)
    filename: str = field(repr=False)
    sha256: str
    samples: int
    expected_text: str = field(repr=False)
    assertion: str
    commands: tuple[str, ...] = field(repr=False)
    tags: tuple[str, ...] = field(repr=False)


def _bad() -> object:
    raise PrivateDiagnosticReceiptError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise PrivateDiagnosticReceiptError()
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: _bad())
    except (
        UnicodeError,
        ValueError,
        OverflowError,
        PrivateDiagnosticReceiptError,
    ):
        raise PrivateDiagnosticReceiptError() from None


def _canonical(value: object) -> bytes:
    try:
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
    except (MemoryError, TypeError, UnicodeError, ValueError, OverflowError):
        raise PrivateDiagnosticReceiptError() from None


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PrivateDiagnosticReceiptError()
    return value


def _validate_surfaces(
    cases: Sequence[PrivateDiagnosticCaseSurface],
) -> tuple[PrivateDiagnosticCaseSurface, ...]:
    if not isinstance(cases, (tuple, list)) or not cases:
        raise PrivateDiagnosticReceiptError()
    prepared = tuple(cases)
    identifiers: set[str] = set()
    filenames: set[str] = set()
    for case in prepared:
        if (
            not isinstance(case, PrivateDiagnosticCaseSurface)
            or _SAFE_ID_RE.fullmatch(case.case_id) is None
            or case.case_id in identifiers
            or not isinstance(case.filename, str)
            or case.filename != f"{case.case_id}.f32le"
            or case.filename.casefold() in filenames
            or _SHA256_RE.fullmatch(case.sha256) is None
            or type(case.samples) is not int
            or case.samples <= 0
            or not isinstance(case.expected_text, str)
            or not isinstance(case.assertion, str)
            or not isinstance(case.commands, tuple)
            or any(not isinstance(value, str) for value in case.commands)
            or not isinstance(case.tags, tuple)
            or any(not isinstance(value, str) for value in case.tags)
        ):
            raise PrivateDiagnosticReceiptError()
        identifiers.add(case.case_id)
        filenames.add(case.filename.casefold())
    return prepared


def _validate_receipts(
    selected_inputs: Sequence[FinalModelInputReceipt],
    cases: tuple[PrivateDiagnosticCaseSurface, ...],
) -> tuple[FinalModelInputReceipt, ...]:
    from core.diagnostic_bundle import FinalModelInputReceipt

    if not isinstance(selected_inputs, (tuple, list)):
        raise PrivateDiagnosticReceiptError()
    receipts = tuple(selected_inputs)
    if len(receipts) != len(cases):
        raise PrivateDiagnosticReceiptError()
    indexes: list[int] = []
    for case, receipt in zip(cases, receipts, strict=True):
        if (
            not isinstance(receipt, FinalModelInputReceipt)
            or receipt.sha256 != case.sha256
            or receipt.samples != case.samples
            or receipt.sample_rate_hz != 16_000
        ):
            raise PrivateDiagnosticReceiptError()
        indexes.append(receipt.input_index)
    if tuple(indexes) != tuple(sorted(set(indexes))):
        raise PrivateDiagnosticReceiptError()
    return receipts


def _case_binding_sha256(
    cases: tuple[PrivateDiagnosticCaseSurface, ...],
    receipts: tuple[FinalModelInputReceipt, ...],
) -> str:
    rows = []
    for case, receipt in zip(cases, receipts, strict=True):
        rows.append(
            {
                "input_index": receipt.input_index,
                "input_sha256": receipt.sha256,
                "id": case.case_id,
                "file": case.filename,
                "sha256": case.sha256,
                "samples": case.samples,
                "expected_text": case.expected_text,
                "assertion": case.assertion,
                "commands": list(case.commands),
                "tags": list(case.tags),
            }
        )
    payload = {
        "schema_version": _CASE_BINDING_SCHEMA_VERSION,
        "cases": rows,
    }
    return hashlib.sha256(_CASE_BINDING_DOMAIN + _canonical(payload)).hexdigest()


def create_private_diagnostic_receipt(
    *,
    diagnostic_manifest_sha256: str,
    labels_sha256: str,
    cases: Sequence[PrivateDiagnosticCaseSurface],
    selected_inputs: Sequence[FinalModelInputReceipt],
) -> bytes:
    """Create the canonical transcript-free schema-v2 selection receipt."""

    try:
        manifest_digest = _sha256(diagnostic_manifest_sha256)
        labels_digest = _sha256(labels_sha256)
        prepared_cases = _validate_surfaces(cases)
        receipts = _validate_receipts(selected_inputs, prepared_cases)
        return _canonical(
            {
                "schema_version": _RECEIPT_SCHEMA_VERSION,
                "kind": _RECEIPT_KIND,
                "diagnostic_manifest_sha256": manifest_digest,
                "labels_sha256": labels_digest,
                "case_binding_sha256": _case_binding_sha256(
                    prepared_cases,
                    receipts,
                ),
                "selected_inputs": [
                    receipt.to_payload() for receipt in receipts
                ],
            }
        )
    except PrivateDiagnosticReceiptError:
        raise
    except Exception:
        raise PrivateDiagnosticReceiptError() from None


def verify_private_diagnostic_receipt(
    raw: bytes,
    *,
    diagnostic_manifest_sha256: str,
    labels_sha256: str,
    cases: Sequence[PrivateDiagnosticCaseSurface],
) -> tuple[FinalModelInputReceipt, ...]:
    """Verify one receipt against its exact ordered published case surface."""

    try:
        from core.diagnostic_bundle import FinalModelInputReceipt

        manifest_digest = _sha256(diagnostic_manifest_sha256)
        labels_digest = _sha256(labels_sha256)
        prepared_cases = _validate_surfaces(cases)
        payload = _strict_json(raw)
        if not isinstance(payload, dict) or set(payload) != _RECEIPT_FIELDS:
            raise PrivateDiagnosticReceiptError()
        selected_inputs = payload.get("selected_inputs")
        if (
            type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != _RECEIPT_SCHEMA_VERSION
            or payload.get("kind") != _RECEIPT_KIND
            or payload.get("diagnostic_manifest_sha256") != manifest_digest
            or payload.get("labels_sha256") != labels_digest
            or not isinstance(selected_inputs, list)
        ):
            raise PrivateDiagnosticReceiptError()
        receipts = _validate_receipts(
            tuple(
                FinalModelInputReceipt.from_payload(value)
                for value in selected_inputs
            ),
            prepared_cases,
        )
        if not hmac.compare_digest(
            _sha256(payload.get("case_binding_sha256")),
            _case_binding_sha256(prepared_cases, receipts),
        ):
            raise PrivateDiagnosticReceiptError()
        return receipts
    except PrivateDiagnosticReceiptError:
        raise
    except Exception:
        raise PrivateDiagnosticReceiptError() from None


__all__ = [
    "PrivateDiagnosticCaseSurface",
    "PrivateDiagnosticReceiptError",
    "create_private_diagnostic_receipt",
    "verify_private_diagnostic_receipt",
]
