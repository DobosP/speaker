from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools import public_endpoint_eval_catalog as catalog
from tools import public_voice_eval_matrix as legacy


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_SHA256 = {
    "tools/public_voice_eval_matrix.py": (
        "52da7296b266404e1900111d25b6beec9d8e1caa1f9ed9c6330154c165bdb284"
    ),
    "tools/public_conversation_fixture.py": (
        "8b1932bf08bc6f37fb5a2c69fd026af6c0616bd0aa713c091cfcd46dbb758134"
    ),
    "tools/streaming_stt/public-conversation-96.lock.json": (
        "68ac2c85b8825fabda10b578646780fc16a37a1b2ab0ca3285fa9f8c93a3168b"
    ),
}


def test_endpoint_catalog_is_exact_and_legacy_closure_is_byte_identical():
    catalog.validate_catalog()

    assert catalog.SOURCE_CATALOG_VERSION == 1
    for relative, expected in _LEGACY_SHA256.items():
        assert hashlib.sha256((_REPO_ROOT / relative).read_bytes()).hexdigest() == expected

    legacy_ids = {dataset.dataset_id for dataset in legacy.DATASETS}
    legacy_track_ids = {
        dataset_id for track in legacy.TRACKS for dataset_id in track.dataset_ids
    }
    assert catalog.LIVEKIT_DATASET_ID not in legacy_ids
    assert catalog.LIVEKIT_DATASET_ID not in legacy_track_ids


def test_livekit_source_pin_license_scope_and_constraints_are_exact():
    source = catalog.source_by_id(catalog.LIVEKIT_DATASET_ID)
    assert source.revision == "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"
    assert source.license.license_id == "CC-BY-4.0"
    assert source.license.acceptance_ids == ("CC-BY-4.0",)
    assert source.evaluation_only is True
    assert source.usage_constraints == (
        legacy.UsageConstraint.EVALUATION_ONLY,
        legacy.UsageConstraint.PRIVATE_CACHE_ONLY,
        legacy.UsageConstraint.NO_REIDENTIFICATION,
        legacy.UsageConstraint.NO_REHOSTING,
    )
    assert source.selection.algorithm is legacy.SelectionAlgorithm.COMPLETE_SPLIT
    assert source.selection.split == "validation"
    assert source.selection.expected_examples == 400
    assert "non-authoritative for WER" in source.evidence_note

    artifact = source.artifacts[0]
    assert artifact.locator == (
        "https://huggingface.co/datasets/livekit/eot-bench-data/resolve/"
        "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd/"
        "data/en/validation-00000-of-00001.parquet"
    )
    assert artifact.size_bytes == 162_406_142
    assert artifact.checksum == legacy.Checksum(
        legacy.ChecksumAlgorithm.SHA256,
        "e475d435c8e693912243e8a810bf7e34a59e01e208551ce5b362c9c778b0fdc4",
        "upstream-lfs",
    )


def test_livekit_track_has_only_endpoint_metrics_and_unknown_source_fails():
    assert len(catalog.TRACKS) == 1
    track = catalog.TRACKS[0]
    assert track.category is legacy.TaskCategory.OVERLAP_TURN_TAKING
    assert track.dataset_ids == (catalog.LIVEKIT_DATASET_ID,)
    assert track.metrics == (
        legacy.Metric.ENDPOINT_EARLY_CUT_RATE,
        legacy.Metric.ENDPOINT_DELAY_MS,
    )
    assert not catalog.FORBIDDEN_METRICS.intersection(track.metrics)
    for forbidden_text in ("wer", "cer", "backchannel"):
        assert forbidden_text in track.protocol.casefold()

    with pytest.raises(catalog.EndpointCatalogError, match="unknown endpoint"):
        catalog.source_by_id("not-a-source")
