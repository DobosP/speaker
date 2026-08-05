"""Typed no-download catalog for public endpoint/turn-taking evaluation data.

This catalog is deliberately independent of the legacy public voice matrix.
Changing that matrix would invalidate the retained four-source conversation
receipts.  Sources admitted here may be used only by endpoint/turn-taking
diagnostics and acquire no STT, WER, overlap-WER, or canonical-conversation
meaning through their presence in this module.
"""

from __future__ import annotations

from tools.public_voice_eval_matrix import (
    Artifact,
    Checksum,
    ChecksumAlgorithm,
    Dataset,
    EvaluationTrack,
    IntegrityKind,
    LicenseGate,
    Metric,
    SelectionAlgorithm,
    SelectionPolicy,
    TaskCategory,
    UsageConstraint,
)


SOURCE_CATALOG_VERSION = 1
LIVEKIT_DATASET_ID = "livekit_eot_bench_en_validation"
LIVEKIT_REVISION = "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"
LIVEKIT_SHARD_PATH = "data/en/validation-00000-of-00001.parquet"
LIVEKIT_SHARD_URL = (
    "https://huggingface.co/datasets/livekit/eot-bench-data/resolve/"
    f"{LIVEKIT_REVISION}/{LIVEKIT_SHARD_PATH}"
)
LIVEKIT_SHARD_BYTES = 162_406_142
LIVEKIT_SHARD_SHA256 = (
    "e475d435c8e693912243e8a810bf7e34a59e01e208551ce5b362c9c778b0fdc4"
)
LIVEKIT_ROWS = 400

CC_BY_4 = LicenseGate(
    "CC-BY-4.0",
    "https://creativecommons.org/licenses/by/4.0/legalcode",
    ("CC-BY-4.0",),
)

SOURCES: tuple[Dataset, ...] = (
    Dataset(
        dataset_id=LIVEKIT_DATASET_ID,
        name="LiveKit EoT Benchmark English validation",
        upstream_id="huggingface:livekit/eot-bench-data:en:validation",
        revision=LIVEKIT_REVISION,
        canonical_url="https://huggingface.co/datasets/livekit/eot-bench-data",
        license=CC_BY_4,
        artifacts=(
            Artifact(
                artifact_id="english-validation-parquet",
                locator=LIVEKIT_SHARD_URL,
                filename="livekit-eot-en-validation.parquet",
                size_bytes=LIVEKIT_SHARD_BYTES,
                integrity=IntegrityKind.PINNED,
                checksum=Checksum(
                    ChecksumAlgorithm.SHA256,
                    LIVEKIT_SHARD_SHA256,
                    "upstream-lfs",
                ),
                integrity_note=(
                    "Pin the exact English validation LFS object; production "
                    "inventory accepts only an explicit local private file."
                ),
            ),
        ),
        selection=SelectionPolicy(
            (
                "Inventory the complete 400-row English validation split. At "
                "each row, earlier ordered silence spans are HOLD labels and "
                "the final span is the publisher-convention EOT label."
            ),
            algorithm=SelectionAlgorithm.COMPLETE_SPLIT,
            split="validation",
            identity_fields=("id",),
            expected_examples=LIVEKIT_ROWS,
        ),
        evidence_note=(
            "Endpoint/turn-taking labels only. Publisher word timings and "
            "conversation context are non-authoritative for WER and are not "
            "admitted to the legacy STT or canonical conversation suites."
        ),
        usage_constraints=(
            UsageConstraint.EVALUATION_ONLY,
            UsageConstraint.PRIVATE_CACHE_ONLY,
            UsageConstraint.NO_REIDENTIFICATION,
            UsageConstraint.NO_REHOSTING,
        ),
    ),
)

TRACKS: tuple[EvaluationTrack, ...] = (
    EvaluationTrack(
        track_id="endpoint-turn-taking",
        category=TaskCategory.OVERLAP_TURN_TAKING,
        dataset_ids=(LIVEKIT_DATASET_ID,),
        metrics=(Metric.ENDPOINT_EARLY_CUT_RATE, Metric.ENDPOINT_DELAY_MS),
        protocol=(
            "Score causally at publisher silence spans: earlier spans are HOLD "
            "and the final span is EOT. Never score WER, CER, overlap-WER, or "
            "backchannel retention from this source."
        ),
    ),
)

FORBIDDEN_METRICS = frozenset(
    {
        Metric.LITERAL_WER,
        Metric.CANONICAL_WER,
        Metric.LITERAL_CER,
        Metric.CANONICAL_CER,
        Metric.OVERLAP_WER,
        Metric.BACKCHANNEL_RETENTION,
    }
)


class EndpointCatalogError(RuntimeError):
    """The endpoint-only source contract failed closed."""


def source_by_id(dataset_id: str) -> Dataset:
    try:
        return next(source for source in SOURCES if source.dataset_id == dataset_id)
    except StopIteration as exc:
        raise EndpointCatalogError(f"unknown endpoint source {dataset_id!r}") from exc


def validate_catalog() -> None:
    """Validate the complete v1 endpoint-only catalog contract."""

    if SOURCE_CATALOG_VERSION != 1 or len(SOURCES) != 1 or len(TRACKS) != 1:
        raise EndpointCatalogError("invalid endpoint source catalog shape")
    source = SOURCES[0]
    track = TRACKS[0]
    if (
        source.dataset_id != LIVEKIT_DATASET_ID
        or source.revision != LIVEKIT_REVISION
        or source.license.license_id != "CC-BY-4.0"
        or source.selection.expected_examples != LIVEKIT_ROWS
        or source.selection.algorithm is not SelectionAlgorithm.COMPLETE_SPLIT
        or source.selection.split != "validation"
        or source.evaluation_only is not True
        or source.usage_constraints
        != (
            UsageConstraint.EVALUATION_ONLY,
            UsageConstraint.PRIVATE_CACHE_ONLY,
            UsageConstraint.NO_REIDENTIFICATION,
            UsageConstraint.NO_REHOSTING,
        )
        or len(source.artifacts) != 1
    ):
        raise EndpointCatalogError("invalid LiveKit endpoint source contract")
    artifact = source.artifacts[0]
    if (
        artifact.size_bytes != LIVEKIT_SHARD_BYTES
        or artifact.integrity is not IntegrityKind.PINNED
        or artifact.checksum
        != Checksum(
            ChecksumAlgorithm.SHA256,
            LIVEKIT_SHARD_SHA256,
            "upstream-lfs",
        )
        or artifact.locator != LIVEKIT_SHARD_URL
    ):
        raise EndpointCatalogError("invalid LiveKit endpoint artifact contract")
    if (
        track.category is not TaskCategory.OVERLAP_TURN_TAKING
        or track.dataset_ids != (LIVEKIT_DATASET_ID,)
        or track.metrics
        != (Metric.ENDPOINT_EARLY_CUT_RATE, Metric.ENDPOINT_DELAY_MS)
        or FORBIDDEN_METRICS.intersection(track.metrics)
    ):
        raise EndpointCatalogError("endpoint source escaped its metric scope")


__all__ = [
    "EndpointCatalogError",
    "FORBIDDEN_METRICS",
    "LIVEKIT_DATASET_ID",
    "LIVEKIT_REVISION",
    "LIVEKIT_ROWS",
    "LIVEKIT_SHARD_BYTES",
    "LIVEKIT_SHARD_PATH",
    "LIVEKIT_SHARD_SHA256",
    "LIVEKIT_SHARD_URL",
    "SOURCE_CATALOG_VERSION",
    "SOURCES",
    "TRACKS",
    "source_by_id",
    "validate_catalog",
]
