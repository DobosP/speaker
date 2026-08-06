from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from core.guided_stt_plan import (
    GUIDED_STT_CAPTURE_PROTOCOL,
    GUIDED_STT_PLAN_ID,
    GUIDED_STT_ROUTE_PROFILE,
    GuidedSttPlanError,
    built_in_guided_stt_plan,
    canonical_guided_stt_plan_bytes,
    guided_stt_capture_config_sha256,
    guided_stt_route_availability_sha256,
    guided_stt_sherpa_config_sha256,
    load_private_guided_stt_plan,
)
from tools.prepare_live_stt_corpus import _reference_plan
from tools.tool_route_gate import (
    ToolRouteClassifier,
    ToolRouteGateProfile,
    expected_route,
)


def _private_plan(path: Path) -> Path:
    path.parent.mkdir(mode=0o700)
    path.write_bytes(canonical_guided_stt_plan_bytes())
    path.chmod(0o600)
    return path


def test_fixed_plan_has_exact_canonical_schema_and_digest() -> None:
    raw = canonical_guided_stt_plan_bytes()
    payload = json.loads(raw)
    plan = built_in_guided_stt_plan()

    assert raw.endswith(b"\n")
    assert raw == (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    assert set(payload) == {"schema_version", "cases"}
    assert payload["schema_version"] == 1
    assert len(payload["cases"]) == plan.case_count == 16
    assert tuple(case["id"] for case in payload["cases"]) == tuple(
        case.case_id for case in plan.cases
    )
    assert len({case.case_id for case in plan.cases}) == 16
    assert plan.plan_id == GUIDED_STT_PLAN_ID == "agent-core-v1"
    assert plan.sha256 == hashlib.sha256(raw).hexdigest()
    assert GUIDED_STT_CAPTURE_PROTOCOL == "guided-stt-capture-v1"
    assert len(_reference_plan(raw)) == 16


def test_fixed_plan_routes_exactly_without_invocation() -> None:
    plan = built_in_guided_stt_plan()
    profile = ToolRouteGateProfile(
        vault_enabled=GUIDED_STT_ROUTE_PROFILE["vault_enabled"],
        reminders_enabled=GUIDED_STT_ROUTE_PROFILE["reminders_enabled"],
        app_aliases=tuple(GUIDED_STT_ROUTE_PROFILE["app_aliases"]),
    )
    classifier = ToolRouteClassifier(profile)

    actual = tuple(classifier.classify(case.expected_text) for case in plan.cases)
    expected = tuple(expected_route(case.tags) for case in plan.cases)

    assert actual == expected == tuple(case.expected_route for case in plan.cases)
    assert actual.count("none") == 5
    assert actual.count("vault.search") == 5
    assert actual.count("web.search") == 2
    assert actual.count("reminder.create") == 1
    assert actual.count("reminder.list") == 1
    assert actual.count("reminder.cancel") == 1
    assert actual.count("app.open") == 1
    assert actual[2] == "vault.search"  # Find in my Obsidian...
    assert actual[5] == "app.open"  # Open Obsidian.


def test_route_profile_digest_is_canonical_and_stable() -> None:
    raw = (
        json.dumps(
            GUIDED_STT_ROUTE_PROFILE,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    assert (
        guided_stt_route_availability_sha256()
        == hashlib.sha256(b"speaker-guided-stt-route-profile-v1\0" + raw).hexdigest()
    )


def test_effective_sherpa_digest_is_canonical_sensitive_and_path_free() -> None:
    first = {
        "asr_encoder": "/private/model-a.onnx",
        "input_gain": 1.0,
        "record_pre_dsp_reference": True,
    }
    reordered = dict(reversed(tuple(first.items())))
    changed = dict(first, input_gain=2.0)

    digest = guided_stt_sherpa_config_sha256(first)

    assert digest == guided_stt_sherpa_config_sha256(reordered)
    assert digest != guided_stt_sherpa_config_sha256(changed)
    assert "/private" not in digest
    assert len(digest) == 64


def test_capture_config_digest_excludes_only_final_stt_ab_fields() -> None:
    base = {
        "asr_encoder": "/private/streaming.onnx",
        "asr_final_model": "/private/control.onnx",
        "asr_final_backend": "sense_voice",
        "input_gain": 1.0,
        "vad_model": "/private/vad.onnx",
    }
    other_final = dict(
        base,
        asr_final_model="/private/candidate.bin",
        asr_final_backend="whisper",
    )

    digest = guided_stt_capture_config_sha256(base)

    assert digest == guided_stt_capture_config_sha256(other_final)
    assert digest != guided_stt_capture_config_sha256(dict(base, input_gain=2.0))
    assert digest != guided_stt_capture_config_sha256(
        dict(base, vad_model="/private/other-vad.onnx")
    )


def test_private_plan_loader_accepts_only_exact_private_single_inode(
    tmp_path: Path,
) -> None:
    plan_path = _private_plan(tmp_path / "private" / "reference-plan.json")

    loaded = load_private_guided_stt_plan(plan_path)

    assert loaded.sha256 == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert stat.S_IMODE(plan_path.stat().st_mode) == 0o600
    assert plan_path.stat().st_nlink == 1


@pytest.mark.parametrize("failure", ["mode", "symlink", "hardlink", "changed"])
def test_private_plan_loader_fails_closed_without_exposing_content(
    tmp_path: Path,
    failure: str,
) -> None:
    original = _private_plan(tmp_path / "private" / "reference-plan.json")
    requested = original
    if failure == "mode":
        original.chmod(0o640)
    elif failure == "symlink":
        requested = original.parent / "reference-plan-link.json"
        requested.symlink_to(original.name)
    elif failure == "hardlink":
        requested = original.parent / "reference-plan-hardlink.json"
        os.link(original, requested)
    else:
        payload = json.loads(original.read_bytes())
        payload["cases"][0]["expected_text"] = "private mutation canary"
        original.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        original.chmod(0o600)

    with pytest.raises(GuidedSttPlanError) as captured:
        load_private_guided_stt_plan(requested)

    assert str(captured.value) == "guided STT plan is unavailable"
    assert "mutation" not in repr(captured.value)


def test_plan_repr_omits_phrases_and_tags() -> None:
    plan = built_in_guided_stt_plan()
    rendered = repr(plan)

    assert "Search in my vault" not in rendered
    assert "expected-tool" not in rendered
    assert "cases=" not in rendered


def test_private_plan_loader_rejects_relative_public_or_symlink_parent(
    tmp_path: Path,
) -> None:
    private = _private_plan(tmp_path / "private" / "reference-plan.json")
    with pytest.raises(GuidedSttPlanError):
        load_private_guided_stt_plan(Path("relative-plan.json"))

    private.parent.chmod(0o750)
    with pytest.raises(GuidedSttPlanError):
        load_private_guided_stt_plan(private)
    private.parent.chmod(0o700)

    linked_parent = tmp_path / "linked-private"
    linked_parent.symlink_to(private.parent, target_is_directory=True)
    with pytest.raises(GuidedSttPlanError):
        load_private_guided_stt_plan(linked_parent / private.name)
