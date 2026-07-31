from __future__ import annotations

import hashlib
import io
import json
import tarfile
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from always_on_agent.capabilities import CapabilityRegistry
from always_on_agent.events import EventKind, Mode
from always_on_agent.models import IntentKind
from always_on_agent.planner import TaskPlanner
from always_on_agent.speech_analyzer import (
    LiveSpeechAnalyzer,
    ModePolicy,
    exact_control_class,
)
from core.intents import IntentGrammar, parse_duration
from core.obsidian import ObsidianVault
from core.reminders import ReminderParser
from core.trusted_apps import DeviceToolCommandDispatcher
from tools import public_voice_fixtures as public


_STOP_RUNTIME_IDS = [
    "always_on_agent.models.IntentKind.STOP",
    "always_on_agent.speech_analyzer.exact_control_class:stop",
    "always_on_agent.events.EventKind.CONTROL_STOP",
]


def _wav_bytes(
    *,
    sample_rate: int = 16000,
    duration_sec: float = 0.05,
    value: int = 1000,
) -> bytes:
    frames = int(round(sample_rate * duration_sec))
    payload = np.full(frames, value, dtype="<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(payload)
    return output.getvalue()


def _source(
    source_id: str,
    filename: str,
    payload: bytes,
    *,
    redistribution: str = "cache-only",
) -> dict[str, object]:
    return {
        "id": source_id,
        "upstream_dataset": "Synthetic unit-test source",
        "upstream_version": "unit-test-v1",
        "source_kind": "canonical",
        "canonical_url": "https://example.invalid/dataset",
        "download_url": f"https://example.invalid/{filename}",
        "filename": filename,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "license_id": "CC0-1.0",
        "license_url": "https://example.invalid/license",
        "attribution": "Synthetic unit-test source.",
        "citation": "Synthetic unit-test source.",
        "redistribution": redistribution,
        "human_voice": False,
        "personal_data": "none",
    }


def _stop_scenario() -> dict[str, object]:
    return {
        "id": "unit-stop",
        "phrase": "stop",
        "evidence": "text-contract-only",
        "contract_status": "exact",
        "role": "target",
        "expected_action": "stop",
        "expected_slots": {},
        "runtime_identifiers": list(_STOP_RUNTIME_IDS),
        "tags": [],
    }


def _write_manifest(
    path: Path,
    *,
    sources: list[dict[str, object]],
    fixtures: list[dict[str, object]],
    groups: list[dict[str, object]] | None = None,
    product_scenarios: list[dict[str, object]] | None = None,
) -> public.PublicVoiceManifest:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "purpose": (
                    "Development and regression fixtures only. This corpus is "
                    "never a final held-out set."
                ),
                "selection_seed": "unit-test-seed",
                "output_sample_rate": 16000,
                "evidence_scope": public.EVIDENCE_SCOPE,
                "sources": sources,
                "fixtures": fixtures,
                "groups": groups or [],
                "product_scenarios": product_scenarios or [_stop_scenario()],
            }
        ),
        encoding="utf-8",
    )
    return public.load_manifest(path)


def _fixture(
    fixture_id: str,
    source_id: str,
    extract: dict[str, object],
    *,
    suite: str = "echo",
    assertion: str = "nearend_only_mic",
    expected_text: str | None = None,
    required_source_ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "id": fixture_id,
        "suite": suite,
        "source_id": source_id,
        "extract": extract,
        "assertion": assertion,
    }
    if expected_text is not None:
        item["expected_text"] = expected_text
    if required_source_ids is not None:
        item["required_source_ids"] = required_source_ids
    if tags is not None:
        item["tags"] = tags
    return item


def _add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, final_url: str):
        super().__init__(payload)
        self._final_url = final_url

    def geturl(self) -> str:
        return self._final_url


def _decision(phrase: str, policy: ModePolicy | None = None):
    analyzer = LiveSpeechAnalyzer(policy)
    return analyzer.decide(
        analyzer.observe(phrase, is_final=True),
        Mode.ASSISTANT,
    )


def test_committed_manifest_is_pinned_licensed_and_development_only():
    manifest = public.load_manifest()

    assert len(manifest.sources) == 3
    assert len(manifest.fixtures) == 15
    assert len(manifest.product_scenarios) == 12
    assert {
        suite: sum(1 for item in manifest.fixtures if item["suite"] == suite)
        for suite in public.SUITES
    } == {"commands": 12, "conversation": 3, "echo": 0}
    assert all(
        len(str(source["sha256"])) == 64
        and int(source["size_bytes"]) > 0
        and source["redistribution"] == "cache-only"
        and source["license_id"] == "CC-BY-4.0"
        and str(source["download_url"]).startswith("https://")
        for source in manifest.sources
    )
    assert "never a final held-out" in str(manifest.data["purpose"])
    assert manifest.data["evidence_scope"] == public.EVIDENCE_SCOPE
    serialized = json.dumps(manifest.data)
    assert "cmu" not in serialized.casefold()
    assert "microsoft" not in serialized.casefold()
    assert "review-required" not in serialized


def test_ami_laughter_is_event_only_and_preserves_annotation_tokens(tmp_path):
    manifest = public.load_manifest()
    laughter = next(
        fixture
        for fixture in manifest.fixtures
        if fixture["id"] == "ami-es2002a-laughter"
    )
    assert laughter["assertion"] == "event_only"
    assert "expected_text" not in laughter

    archive_path = tmp_path / "annotations.zip"
    mixed = (
        '<root><w starttime="1.0" endtime="1.2">well</w>'
        '<vocalsound starttime="1.1" endtime="1.4" type="laugh"/></root>'
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("words/MEET.A.words.xml", mixed)
    evidence = public._annotation_evidence(
        archive_path,
        {
            "members": ["words/MEET.A.words.xml"],
            "min_active_speakers": 1,
            "max_active_speakers": 1,
            "min_concurrent_speakers": 1,
            "required_vocal_type": "laugh",
        },
        start_sec=0.9,
        end_sec=1.5,
    )

    assert evidence["vocal_types"] == ["laugh"]
    assert evidence["annotation_tokens"] == ["well"]
    assert evidence["annotation_token_count"] == 1
    assert evidence["annotation_transcript"] == "well"


def test_ami_annotation_rejects_renamed_alias_for_the_same_speaker(tmp_path):
    archive_path = tmp_path / "annotations.zip"
    active = '<root><w starttime="1.0" endtime="1.4">yes</w></root>'
    members = [
        "words/MEET.A.words.xml",
        "aliases/MEET.A-copy.words.xml",
    ]
    with zipfile.ZipFile(archive_path, "w") as archive:
        for member in members:
            archive.writestr(member, active)

    with pytest.raises(public.PublicFixtureError, match="speaker identity"):
        public._annotation_evidence(
            archive_path,
            {
                "members": members,
                "min_active_speakers": 2,
                "max_active_speakers": 2,
                "min_concurrent_speakers": 2,
            },
            start_sec=0.9,
            end_sec=1.5,
        )


def test_ami_expected_text_uses_token_subsequence_not_string_substring(tmp_path):
    archive_path = tmp_path / "annotations.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "words/MEET.A.words.xml",
            '<root><w starttime="1.0" endtime="1.4">candy</w></root>',
        )

    with pytest.raises(public.PublicFixtureError, match="transcript"):
        public._annotation_evidence(
            archive_path,
            {
                "members": ["words/MEET.A.words.xml"],
                "expected_text": "and",
                "min_active_speakers": 1,
                "max_active_speakers": 1,
                "min_concurrent_speakers": 1,
            },
            start_sec=0.9,
            end_sec=1.5,
        )


def test_product_contracts_match_current_runtime_identifiers_and_semantics():
    manifest = public.load_manifest()
    scenarios = {str(item["id"]): item for item in manifest.product_scenarios}

    for scenario_id in ("control-stop", "control-cancel"):
        scenario = scenarios[scenario_id]
        phrase = str(scenario["phrase"])
        decision = _decision(phrase)
        assert exact_control_class(phrase) == ("stop", "")
        assert decision.kind == IntentKind.STOP
        assert decision.reason == "stop_phrase"
        assert scenario["expected_action"] == "stop"
        assert scenario["runtime_identifiers"] == _STOP_RUNTIME_IDS
        assert EventKind.CONTROL_STOP.value == "control.stop"

    vault_analyzer = LiveSpeechAnalyzer(ModePolicy(vault_search_enabled=True))
    for scenario_id in ("vault-search-in", "vault-go-in", "vault-find-in"):
        scenario = scenarios[scenario_id]
        phrase = str(scenario["phrase"])
        decision = vault_analyzer.decide(
            vault_analyzer.observe(phrase, is_final=True), Mode.ASSISTANT
        )
        plan = TaskPlanner().plan(decision)
        assert decision.kind == IntentKind.SEARCH
        assert decision.metadata == {"search_scope": "vault"}
        assert [step.capability for step in plan.steps] == [
            "vault.search",
            "research.local",
        ]
        assert list(ObsidianVault._terms(phrase)) == scenario["expected_slots"]["terms"]

    now = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
    parser = ReminderParser(
        clock=lambda: now,
        local_timezone=timezone.utc,
        min_delay_sec=1,
    )
    dispatcher = DeviceToolCommandDispatcher(
        CapabilityRegistry(),
        reminder_manager=SimpleNamespace(parser=parser),
    )
    reminder = scenarios["reminder-numeral"]
    phrase = str(reminder["phrase"])
    parsed = parser.parse(phrase)
    matched = dispatcher.match(phrase)
    decision = _decision(
        phrase,
        ModePolicy(
            device_tools_enabled=True,
            device_tool_matcher=dispatcher.match,
        ),
    )
    plan = TaskPlanner().plan(decision)
    assert parsed.message == reminder["expected_slots"]["message"]
    assert int((parsed.due_at - now).total_seconds()) == reminder["expected_slots"][
        "delay_seconds"
    ]
    assert matched is not None and matched.capability_name == "reminder.create"
    assert decision.kind == IntentKind.COMMAND
    assert decision.metadata["device_tool"] == "reminder.create"
    assert decision.requires_confirmation is True
    assert [step.capability for step in plan.steps] == ["device.command"]

    timer = scenarios["timer-numeral"]
    match = IntentGrammar.default().match(str(timer["phrase"]))
    assert match is not None and match.name == "timer"
    assert match.slots == {"value": "five", "unit": "minutes"}
    assert parse_duration(**match.slots) == timer["expected_slots"]["duration_seconds"]

    for scenario_id in ("near-control-dont-stop", "near-control-dont-cancel"):
        scenario = scenarios[scenario_id]
        phrase = str(scenario["phrase"])
        assert exact_control_class(phrase) is None
        assert _decision(phrase).kind == IntentKind.ASSISTANT
        assert scenario["expected_action"] is None

    for scenario_id in (
        "control-wait",
        "control-hold-on",
        "bystander-vault-command",
    ):
        scenario = scenarios[scenario_id]
        assert scenario["contract_status"] == "future-taxonomy"
        assert scenario["expected_action"] is None
        assert scenario["runtime_identifiers"] == []
        assert "future-taxonomy" in scenario["tags"]


def test_manifest_rejects_missing_development_only_boundary(tmp_path):
    payload = _wav_bytes()
    source = _source("audio", "audio.wav", payload)
    fixture = _fixture("clip", "audio", {"type": "direct_wav"})
    path = tmp_path / "manifest.json"
    document = {
        "schema_version": 2,
        "purpose": "benchmark corpus",
        "selection_seed": "seed",
        "output_sample_rate": 16000,
        "evidence_scope": public.EVIDENCE_SCOPE,
        "sources": [source],
        "fixtures": [fixture],
        "groups": [],
        "product_scenarios": [_stop_scenario()],
    }
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(public.PublicFixtureError, match="development-only"):
        public.load_manifest(path)


def test_duplicate_json_keys_are_rejected(tmp_path):
    payload = _wav_bytes()
    source = _source("audio", "audio.wav", payload)
    fixture = _fixture("clip", "audio", {"type": "direct_wav"})
    path = tmp_path / "manifest.json"
    document = {
        "schema_version": 2,
        "purpose": "Development only; never a final held-out corpus.",
        "selection_seed": "seed",
        "output_sample_rate": 16000,
        "evidence_scope": public.EVIDENCE_SCOPE,
        "sources": [source],
        "fixtures": [fixture],
        "groups": [],
        "product_scenarios": [_stop_scenario()],
    }
    raw = json.dumps(document).replace(
        '"schema_version": 2', '"schema_version": 2, "schema_version": 2'
    )
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(public.PublicFixtureError, match="unable to load"):
        public.load_manifest(path)


@pytest.mark.parametrize(("field", "value"), [("id", 7), ("filename", 7)])
def test_manifest_rejects_string_fields_with_numeric_values(tmp_path, field, value):
    payload = _wav_bytes()
    source = _source("audio", "audio.wav", payload)
    source[field] = value

    with pytest.raises(public.PublicFixtureError):
        _write_manifest(
            tmp_path / "manifest.json",
            sources=[source],
            fixtures=[_fixture("clip", "audio", {"type": "direct_wav"})],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_kind", "unknown", "source kind"),
        ("redistribution", "review-required", "redistribution"),
        ("license_id", "CC-BY-4.0 OR MIT", "license id"),
        ("download_url", "http://example.invalid/audio.wav", "download URL"),
    ],
)
def test_manifest_rejects_unresolved_or_invalid_provenance(
    tmp_path, field, value, message
):
    payload = _wav_bytes()
    source = _source("audio", "audio.wav", payload)
    source[field] = value

    with pytest.raises(public.PublicFixtureError, match=message):
        _write_manifest(
            tmp_path / "manifest.json",
            sources=[source],
            fixtures=[_fixture("clip", "audio", {"type": "direct_wav"})],
        )


@pytest.mark.parametrize("filename", [".audio.wav", "..", "folder/audio.wav"])
def test_manifest_rejects_hidden_or_non_flat_source_filenames(tmp_path, filename):
    payload = _wav_bytes()
    source = _source("audio", filename, payload)

    with pytest.raises(public.PublicFixtureError, match="filename"):
        _write_manifest(
            tmp_path / "manifest.json",
            sources=[source],
            fixtures=[_fixture("clip", "audio", {"type": "direct_wav"})],
        )


def test_manifest_rejects_casefold_duplicate_source_filenames(tmp_path):
    payload = _wav_bytes()
    first = _source("audio-a", "Clip.wav", payload)
    second = _source("audio-b", "clip.WAV", payload)

    with pytest.raises(public.PublicFixtureError, match="filename"):
        _write_manifest(
            tmp_path / "manifest.json",
            sources=[first, second],
            fixtures=[_fixture("clip", "audio-a", {"type": "direct_wav"})],
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda fixture: fixture["extract"].update({"unexpected": True}),
        lambda fixture: fixture.update({"tags": ["not-allowlisted"]}),
    ],
)
def test_manifest_rejects_unknown_extractor_fields_and_tags(tmp_path, mutate):
    payload = _wav_bytes()
    fixture = _fixture("clip", "audio", {"type": "direct_wav"})
    mutate(fixture)

    with pytest.raises(public.PublicFixtureError):
        _write_manifest(
            tmp_path / "manifest.json",
            sources=[_source("audio", "audio.wav", payload)],
            fixtures=[fixture],
        )


@pytest.mark.parametrize(
    "annotation_change",
    [
        {"source_id": "missing"},
        {
            "members": [
                "words/MEET.A.words.xml",
                "words/MEET.A.words.xml",
            ]
        },
        {"start_sec": float("nan")},
    ],
)
def test_manifest_rejects_malicious_annotation_recipes(
    tmp_path, annotation_change
):
    wav = _wav_bytes()
    annotation_bytes = b"zip"
    annotation = {
        "source_id": "annotations",
        "members": ["words/MEET.A.words.xml"],
        "min_active_speakers": 1,
        "max_active_speakers": 1,
        "min_concurrent_speakers": 1,
    }
    extract = {
        "type": "wav_segment",
        "start_sec": 1.0,
        "end_sec": 2.0,
        "annotation": annotation,
    }
    if "start_sec" in annotation_change:
        extract["start_sec"] = annotation_change["start_sec"]
    else:
        annotation.update(annotation_change)
    fixture = _fixture(
        "clip",
        "audio",
        extract,
        required_source_ids=["annotations"],
    )

    with pytest.raises(public.PublicFixtureError):
        _write_manifest(
            tmp_path / "manifest.json",
            sources=[
                _source("audio", "audio.wav", wav),
                _source("annotations", "annotations.zip", annotation_bytes),
            ],
            fixtures=[fixture],
        )


@pytest.mark.parametrize(
    "group",
    [
        {
            "id": "bad",
            "suite": "echo",
            "assertion": "farend_single_talk",
            "roles": {"farend_reference": "far"},
        },
        {
            "id": "bad",
            "suite": "conversation",
            "assertion": "farend_single_talk",
            "roles": {"farend_reference": "far", "microphone": "mic"},
        },
    ],
)
def test_groups_require_exact_roles_assertions_and_same_suite(tmp_path, group):
    payload = _wav_bytes()
    fixtures = [
        _fixture(
            "far",
            "audio",
            {"type": "direct_wav"},
            assertion="farend_reference",
        ),
        _fixture(
            "mic",
            "audio",
            {"type": "direct_wav"},
            assertion="echo_only_mic",
        ),
    ]
    with pytest.raises(public.PublicFixtureError, match="group"):
        _write_manifest(
            tmp_path / "manifest.json",
            sources=[_source("audio", "audio.wav", payload)],
            fixtures=fixtures,
            groups=[group],
        )


def test_product_scenario_schema_rejects_invented_runtime_support(tmp_path):
    payload = _wav_bytes()
    bad = {
        "id": "future",
        "phrase": "wait",
        "evidence": "text-contract-only",
        "contract_status": "future-taxonomy",
        "role": "target",
        "expected_action": "stop",
        "expected_slots": {},
        "runtime_identifiers": list(_STOP_RUNTIME_IDS),
        "tags": ["future-taxonomy"],
    }
    with pytest.raises(public.PublicFixtureError, match="future taxonomy"):
        _write_manifest(
            tmp_path / "manifest.json",
            sources=[_source("audio", "audio.wav", payload)],
            fixtures=[_fixture("clip", "audio", {"type": "direct_wav"})],
            product_scenarios=[bad],
        )


def test_download_is_streamed_hash_checked_exact_url_and_atomic(tmp_path):
    payload = b"pinned public bytes"
    source = _source("source", "source.bin", payload)
    destination = tmp_path / "source.bin"
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return _Response(payload, request.full_url)

    result = public._download_source(source, destination, opener=opener)

    assert result.read_bytes() == payload
    assert calls == [("https://example.invalid/source.bin", 30.0)]
    assert not list(tmp_path.glob("*.part"))


def test_redirect_or_bad_download_never_publishes_destination(tmp_path):
    payload = b"expected"
    source = _source("source", "source.bin", payload)
    destination = tmp_path / "source.bin"

    with pytest.raises(public.PublicFixtureError, match="redirected"):
        public._download_source(
            source,
            destination,
            opener=lambda request, _timeout: _Response(
                payload, request.full_url + "?redirected=1"
            ),
        )
    assert not destination.exists()

    with pytest.raises(public.PublicFixtureError, match="pinned source"):
        public._download_source(
            source,
            destination,
            opener=lambda request, _timeout: _Response(
                b"tampered", request.full_url
            ),
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_missing_source_requires_opt_in_before_network(tmp_path):
    payload = _wav_bytes()
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        sources=[_source("audio", "audio.wav", payload)],
        fixtures=[_fixture("clip", "audio", {"type": "direct_wav"})],
    )
    called = False

    def opener(request, _timeout):
        nonlocal called
        called = True
        return _Response(payload, request.full_url)

    with pytest.raises(public.DownloadApprovalRequired) as error:
        public.prepare_suite(
            manifest,
            "echo",
            tmp_path / "cache",
            opener=opener,
        )

    assert error.value.size_bytes == len(payload)
    assert called is False


def test_resolved_source_is_reverified_immediately_before_extraction(
    tmp_path, monkeypatch
):
    expected = _wav_bytes()
    changed = tmp_path / "audio.wav"
    changed.write_bytes(expected + b"changed")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        sources=[_source("audio", "audio.wav", expected)],
        fixtures=[_fixture("clip", "audio", {"type": "direct_wav"})],
    )
    monkeypatch.setattr(
        public,
        "_resolve_sources",
        lambda *_args, **_kwargs: {"audio": changed},
    )

    with pytest.raises(public.PublicFixtureError, match="size"):
        public.prepare_suite(manifest, "echo", tmp_path / "cache", offline=True)
    assert not (tmp_path / "cache" / "v2" / "echo").exists()


def test_speech_command_selection_is_stable_and_uses_distinct_speakers():
    members = [
        public._SpeechMember("yes/a_nohash_0.wav", "yes", "a", "00", 10, 0),
        public._SpeechMember("yes/b_nohash_0.wav", "yes", "b", "10", 10, 1),
        public._SpeechMember("no/a_nohash_0.wav", "no", "a", "00", 10, 2),
        public._SpeechMember("no/c_nohash_0.wav", "no", "c", "10", 10, 3),
        public._SpeechMember("tree/d_nohash_0.wav", "tree", "d", "00", 10, 4),
        public._SpeechMember("_silence_/noise.wav", "_silence_", None, "00", 10, 5),
    ]

    selected = public._select_speech_command_members(
        members, ["yes", "no", "_unknown_", "_silence_"]
    )

    assert selected["yes"].speaker_id == "a"
    assert selected["no"].speaker_id == "c"
    assert selected["_unknown_"].label == "tree"
    assert selected["_silence_"].speaker_id is None
    assert selected == public._select_speech_command_members(
        reversed(members), ["yes", "no", "_unknown_", "_silence_"]
    )


def test_speech_commands_tar_extracts_only_deterministic_targets(tmp_path):
    archive_path = tmp_path / "commands.tar.gz"
    wav = _wav_bytes(duration_sec=0.08)
    with tarfile.open(archive_path, "w:gz") as archive:
        _add_tar_bytes(archive, "yes/speaker-a_nohash_0.wav", wav)
        _add_tar_bytes(archive, "yes/speaker-b_nohash_0.wav", wav)
        _add_tar_bytes(archive, "no/speaker-a_nohash_0.wav", wav)
        _add_tar_bytes(archive, "no/speaker-c_nohash_0.wav", wav)
        _add_tar_bytes(archive, "tree/speaker-d_nohash_0.wav", wav)
        _add_tar_bytes(archive, "_silence_/noise.wav", wav)
        _add_tar_bytes(archive, "../escape.wav", wav)
    fixtures = [
        _fixture(
            "yes-clip",
            "commands",
            {"type": "speech_commands_tar", "target": "yes"},
            suite="commands",
            assertion="transcript",
            expected_text="yes",
        ),
        _fixture(
            "no-clip",
            "commands",
            {"type": "speech_commands_tar", "target": "no"},
            suite="commands",
            assertion="transcript",
            expected_text="no",
        ),
        {
            **_fixture(
                "unknown-clip",
                "commands",
                {"type": "speech_commands_tar", "target": "_unknown_"},
                suite="commands",
                assertion="unknown_word",
            ),
            "expected_text_from_archive_label": True,
        },
        _fixture(
            "silence-clip",
            "commands",
            {
                "type": "speech_commands_tar",
                "target": "_silence_",
                "max_duration_sec": 0.04,
            },
            suite="commands",
            assertion="empty_reference",
            expected_text="",
        ),
    ]

    prepared = public._extract_speech_commands(
        archive_path, fixtures, seed="seed", output_sample_rate=16000
    )

    assert set(prepared) == {
        "yes-clip",
        "no-clip",
        "unknown-clip",
        "silence-clip",
    }
    assert prepared["unknown-clip"].expected_text == "tree"
    speakers = {
        prepared[name].selection["speaker_id"]
        for name in ("yes-clip", "no-clip", "unknown-clip")
    }
    assert len(speakers) == 3
    assert len(prepared["silence-clip"].samples) == 640


def test_prepare_suite_offline_emits_exact_replay_metadata(tmp_path):
    wav = _wav_bytes(sample_rate=8000, duration_sec=0.1)
    source_path = tmp_path / "input.wav"
    source_path.write_bytes(wav)
    source = _source("audio", "audio.wav", wav)
    fixture = _fixture(
        "public-clip",
        "audio",
        {"type": "direct_wav"},
        assertion="nearend_only_mic",
    )
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        sources=[source],
        fixtures=[fixture],
    )

    output_dir = public.prepare_suite(
        manifest,
        "echo",
        tmp_path / "cache",
        source_overrides={"audio": source_path},
        offline=True,
    )

    samples = np.load(output_dir / "public-clip.npy")
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert samples.dtype == np.float32
    assert len(samples) == 1600
    assert metadata["schema_version"] == 2
    assert metadata["manifest_sha256"] == manifest.digest
    assert metadata["evidence_scope"] == public.EVIDENCE_SCOPE
    assert metadata["audio_committed"] is False
    assert metadata["sources"] == list(public.source_records_for_suite(manifest, "echo"))
    assert metadata["cases"][0]["source_sha256"] == {
        "audio": hashlib.sha256(wav).hexdigest()
    }
    assert metadata["cases"][0]["sha256"] == public._sha256_file(
        output_dir / "public-clip.npy"
    )


def test_prepare_uses_verified_source_snapshot_during_change_restore(
    tmp_path, monkeypatch
):
    original_bytes = _wav_bytes(value=1000)
    replacement_bytes = _wav_bytes(value=-1000)
    assert len(original_bytes) == len(replacement_bytes)
    source_path = tmp_path / "audio.wav"
    source_path.write_bytes(original_bytes)
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        sources=[_source("audio", "audio.wav", original_bytes)],
        fixtures=[
            _fixture(
                "clip",
                "audio",
                {"type": "direct_wav"},
                assertion="nearend_only_mic",
            )
        ],
    )
    real_extract = public._extract_direct_wav
    extraction_paths: list[Path] = []

    def swap_original_during_extract(snapshot_path, fixture, *, output_sample_rate):
        extraction_paths.append(snapshot_path)
        source_path.write_bytes(replacement_bytes)
        try:
            return real_extract(
                snapshot_path,
                fixture,
                output_sample_rate=output_sample_rate,
            )
        finally:
            source_path.write_bytes(original_bytes)

    monkeypatch.setattr(public, "_extract_direct_wav", swap_original_during_extract)
    output_dir = public.prepare_suite(
        manifest,
        "echo",
        tmp_path / "cache",
        source_overrides={"audio": source_path},
        offline=True,
    )

    samples = np.load(output_dir / "clip.npy")
    assert float(samples.mean()) > 0.0
    assert extraction_paths and extraction_paths[0] != source_path
    assert not extraction_paths[0].exists()
    assert source_path.read_bytes() == original_bytes
    assert not list((tmp_path / "cache" / ".source-snapshots").iterdir())


def test_fsdd_adapted_arrays_have_complete_sharealike_notice():
    root = Path(__file__).resolve().parent / "fixture_audio"
    creator_names = (
        "Zohar Jackson",
        "César Souza",
        "Jason Flaks",
        "Yuxin Pan",
        "Hereman Nicolas",
        "Adhish Thite",
    )
    for relative in (
        "real_usage_full/metadata.json",
        "virtual_real_world/metadata.json",
    ):
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        provenance = payload["dataset_provenance"]
        assert provenance["license_id"] == "CC-BY-SA-4.0"
        assert provenance["adapted_material_license_id"] == "CC-BY-SA-4.0"
        assert provenance["license_url"] == (
            "https://creativecommons.org/licenses/by-sa/4.0/"
        )
        assert provenance["provenance_complete"] is False
        assert provenance["upstream_revision"] is None
        assert provenance["changes"]
        assert provenance["fixture_notice"] == "../FSDD-NOTICE.md"
        for name in creator_names:
            assert name in provenance["creator_attribution"]
    notice = (root / "FSDD-NOTICE.md").read_text(encoding="utf-8")
    for name in creator_names:
        assert name in notice
    assert "CC BY-SA 4.0" in notice
    assert "not covered by the repository's MIT software license" in notice
    assert "revision is therefore reported as unknown" in notice
