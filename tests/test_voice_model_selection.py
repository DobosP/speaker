"""Shared runtime/readiness selection; no host, model or audio access."""

from copy import deepcopy
import json

import pytest

from core.voice_model_profile import PROFILE_ALIAS, apply_voice_model_profile
from core.voice_model_selection import select_voice_model


def base():
    return {
        "device": "safe",
        "device_profiles": {"safe": {"sherpa": {"aec_enabled": False}}},
        "llm": {
            "backend": "ollama",
            "main_model": "vision",
            "fast_model": "previous",
            "options": {"num_ctx": 2048, "num_predict": 256},
        },
        "sherpa": {"aec_enabled": False},
        "assistant": {"name": "Iris"},
        "privacy": {"local_only": True},
    }


def test_shared_selection_is_atomic_and_main_override_preserves_quality_policy():
    original = base()
    saved = deepcopy(original)
    selected, metadata = select_voice_model(original, "qwen2.5-1.5b", model="new-main")
    assert original == saved
    assert metadata.name == "qwen2.5-1.5b"
    assert selected["llm"]["main_model"] == "new-main"
    assert selected["llm"]["fast_model"] == PROFILE_ALIAS
    assert selected["llm"]["options"] == original["llm"]["options"]
    assert selected["privacy"] == original["privacy"]


def test_raw_profile_can_roll_back_but_resolved_selection_cannot_hide_qwen():
    original = {**base(), "voice_model_profile": "qwen2.5-1.5b"}
    rolled, metadata = select_voice_model(original, "current", fast_model="custom")
    assert metadata is None and rolled["voice_model_profile"] == "current"
    assert rolled["llm"]["fast_model"] == "custom"
    selected, _ = select_voice_model(original)
    with pytest.raises(ValueError, match="original device config"):
        apply_voice_model_profile(selected, "current")


@pytest.mark.parametrize("bad", ["", " ", False, [], "x" * 513])
@pytest.mark.parametrize("key", ["model", "fast_model"])
def test_invalid_override_refuses_without_mutation(key, bad):
    original = base()
    saved = deepcopy(original)
    with pytest.raises(ValueError):
        select_voice_model(original, **{key: bad})
    assert original == saved


def test_conflicting_fast_override_refuses_instead_of_mislabeling_profile():
    with pytest.raises(ValueError, match="conflicts"):
        select_voice_model(base(), "qwen2.5-1.5b", fast_model="custom")
    selected, _ = select_voice_model(base(), "qwen2.5-1.5b", fast_model=PROFILE_ALIAS)
    assert selected["llm"]["fast_model"] == PROFILE_ALIAS


def test_doctor_launcher_and_core_share_effective_selection(tmp_path, monkeypatch):
    from tools import doctor, live_launcher

    config = base()
    (tmp_path / "config.json").write_text(json.dumps(config))
    seen = []
    monkeypatch.setattr(
        doctor, "run_runtime_checks", lambda config, **_: seen.append(config) or []
    )
    checks = doctor.run_all(
        config,
        device="safe",
        voice_model="qwen2.5-1.5b",
        model="custom-main",
        config_root=tmp_path,
    )
    selected = live_launcher._selected_live_config(
        tmp_path, "safe", voice_model="qwen2.5-1.5b", model="custom-main"
    )
    effective, metadata = select_voice_model(
        config, "qwen2.5-1.5b", model="custom-main"
    )
    assert seen[0]["llm"] == effective["llm"]
    assert seen[0]["assistant"] == effective["assistant"]
    assert selected.voice_model_profile == metadata.name
    assert selected.voice_model_profile_sha256 == metadata.sha256
    assert any(
        metadata.sha256 in row.detail for row in checks if row.name == "voice model"
    )


def test_conflict_precedes_readiness_side_effects(monkeypatch):
    from tools import doctor

    monkeypatch.setattr(
        doctor, "run_runtime_checks", lambda *a, **kw: pytest.fail("side effect")
    )
    with pytest.raises(ValueError, match="conflicts"):
        doctor.run_all(
            base(), device="safe", voice_model="qwen2.5-1.5b", fast_model="custom"
        )


def test_direct_readiness_lists_selected_voice_alias():
    from core.readiness import profile_ollama_models
    config = {**base(), "voice_model_profile": "qwen2.5-1.5b"}
    assert profile_ollama_models(config, "safe") == ("vision", PROFILE_ALIAS)
    assert config["llm"]["fast_model"] == "previous"
