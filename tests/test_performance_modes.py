from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from core.config import deep_merge
from core.performance import PerformanceModeError, apply_performance_mode, asset_sha256


def mode_config(tmp_path: Path) -> dict:
    profiles = {}
    for mode in ("responsive", "compact"):
        paths = {}
        for key in ("asr_encoder", "asr_decoder", "asr_joiner", "asr_tokens", "tts_model", "tts_tokens"):
            path = tmp_path / mode / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((mode + key).encode())
            paths[key] = str(path)
        data = tmp_path / mode / "espeak"
        data.mkdir()
        (data / "nested").mkdir()
        (data / "nested" / "voice").write_bytes(b"public fake asset")
        paths["tts_data_dir"] = str(data)
        paths.update(tts_voices="", tts_lexicon="")
        if mode == "compact":
            voice = tmp_path / mode / "voices"
            voice.write_bytes(b"public fake voice table")
            paths["tts_voices"] = str(voice)
        hashes = {key: asset_sha256(Path(value)) for key, value in paths.items() if value}
        profiles[mode] = {
            "schema_version": 1,
            "sherpa": {**paths, "tts_backend": "kitten" if mode == "compact" else "",
                       "tts_speaker_id": 0,
                       "asr_num_threads": 2, "tts_num_threads": 2},
            "llm": {"main_keep_alive": "60s", "fast_keep_alive": -1},
            "warm_start_policy": "fast", "asset_sha256": hashes,
        }
    return {
        "performance_profiles": profiles,
        "sherpa": {"provider": "cpu", "aec_enabled": False,
                   "asr_final_backend": "nemo_transducer", "asr_final_verifier_backend": "faster_whisper",
                   "asr_final_required": True, "asr_decoding_method": "modified_beam_search",
                   "asr_hotwords": "", "asr_rule2_min_trailing_silence": 0.7,
                   "barge_word_cut_require_speaker": True, "tts_output_leveler": True,
                   "tts_speed": 1.1},
        "llm": {"backend": "ollama", "main_model": "vision", "fast_model": "text",
                "options": {"num_ctx": 8192, "num_predict": 512}, "cloud": {"enabled": False}},
        "memory": {"enabled": True}, "capabilities": {"tools": True},
        "local_only": True, "device": "safe", "device_profiles": {"safe": {"sherpa": {"aec_enabled": False}}},
    }


def test_current_is_exact_noop_without_reading_assets(monkeypatch):
    config = {"arbitrary_existing_settings": {"x": 1}}
    monkeypatch.setattr("core.performance.asset_sha256", lambda _: pytest.fail("asset read"))
    result, metadata = apply_performance_mode(config)
    assert result is config and metadata is None


@pytest.mark.parametrize("mode", ["responsive", "compact"])
def test_models_change_but_capabilities_and_quality_controls_survive(tmp_path, mode):
    config = mode_config(tmp_path)
    original = copy.deepcopy(config)
    result, metadata = apply_performance_mode(config, mode)
    assert config == original
    assert metadata.name == mode and len(metadata.sha256) == 64
    assert result["warm_start_policy"] == "fast"
    for key, value in original["sherpa"].items():
        assert result["sherpa"][key] == value
    for key, value in original["llm"].items():
        assert result["llm"][key] == value
    for key in ("memory", "capabilities", "local_only"):
        assert result[key] == original[key]
    assert result["sherpa"]["tts_backend"] == ("kitten" if mode == "compact" else "")
    assert bool(result["sherpa"]["tts_voices"]) == (mode == "compact")


def test_cli_current_overrides_configured_mode(tmp_path):
    config = mode_config(tmp_path)
    config["performance_mode"] = "compact"
    result, metadata = apply_performance_mode(config, "current")
    assert result is config and metadata is None
    _, metadata = apply_performance_mode(config)
    assert metadata.name == "compact"


@pytest.mark.parametrize("target", ["asr_encoder", "tts_tokens", "tts_data_dir"])
def test_changed_or_missing_asset_refuses_atomically(tmp_path, target):
    config = mode_config(tmp_path)
    original = copy.deepcopy(config)
    path = Path(config["performance_profiles"]["compact"]["sherpa"][target])
    if path.is_dir():
        (path / "extra").write_bytes(b"changed inventory")
    else:
        path.unlink()
    with pytest.raises(PerformanceModeError):
        apply_performance_mode(config, "compact")
    assert config == original


@pytest.mark.parametrize("key,value", [("asr_num_threads", True), ("tts_num_threads", 0),
                                        ("asr_final_backend", ""), ("provider", "cuda"),
                                        ("tts_lock_speaker_id", False)])
def test_incomplete_or_authority_changing_profile_refuses(tmp_path, key, value):
    config = mode_config(tmp_path)
    config["performance_profiles"]["compact"]["sherpa"][key] = value
    with pytest.raises(PerformanceModeError):
        apply_performance_mode(config, "compact")


def test_symlinks_and_tree_additions_are_not_silently_accepted(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"model")
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(PerformanceModeError):
        asset_sha256(link)
    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "link").symlink_to(source)
    with pytest.raises(PerformanceModeError):
        asset_sha256(directory)


def test_file_hash_matches_standard_sha256(tmp_path):
    path = tmp_path / "model"
    path.write_bytes(b"public deterministic model bytes")
    assert asset_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_performance_bag_is_atomic_in_local_config_merge():
    result = deep_merge({"performance_profiles": {"compact": {"old": 1}, "responsive": {}}},
                        {"performance_profiles": {"compact": {"new": 2}}})
    assert result["performance_profiles"] == {"compact": {"new": 2}}


def test_doctor_and_launcher_resolve_same_mode_digest(tmp_path, monkeypatch):
    from tools import doctor, live_launcher
    config = mode_config(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(config))
    seen = []
    monkeypatch.setattr(doctor, "run_runtime_checks", lambda config, **_: seen.append(config) or [])
    checks = doctor.run_all(config, device="safe", performance="compact", config_root=tmp_path)
    selected = live_launcher._selected_live_config(tmp_path, "safe", performance="compact")
    metadata = apply_performance_mode(config, "compact", root=tmp_path)[1]
    assert selected.performance_mode == metadata.name
    assert selected.performance_mode_sha256 == metadata.sha256
    assert any(metadata.sha256 in check.detail for check in checks if check.name == "performance mode")
    assert seen[0]["sherpa"]["tts_backend"] == "kitten"
    assert seen[0]["sherpa"]["asr_final_verifier_backend"] == "faster_whisper"


def test_duplicate_or_unknown_mode_rejected_before_route_preparation():
    from tools.live_launcher import _parse_live_arguments
    with pytest.raises(SystemExit):
        _parse_live_arguments(["--performance", "compact", "--performance", "responsive"])
    with pytest.raises(SystemExit):
        _parse_live_arguments(["--performance", "unknown"])


def test_relative_paths_bind_to_explicit_config_root(tmp_path):
    config = mode_config(tmp_path)
    profile = config["performance_profiles"]["responsive"]
    for key, value in profile["sherpa"].items():
        if isinstance(value, str) and value.startswith(str(tmp_path)):
            profile["sherpa"][key] = str(Path(value).relative_to(tmp_path))
    result, _ = apply_performance_mode(config, "responsive", root=tmp_path)
    assert Path(result["sherpa"]["asr_encoder"]).is_absolute()


def test_profile_mutation_during_asset_hash_cannot_change_policy_or_binding(tmp_path, monkeypatch):
    import core.performance as policy
    config = mode_config(tmp_path)
    expected = apply_performance_mode(config, "compact")[1]
    real = policy.asset_sha256
    mutated = False

    def mutate(path):
        nonlocal mutated
        if not mutated:
            mutated = True
            config["performance_profiles"]["compact"]["llm"]["cloud"] = {"enabled": True}
        return real(path)

    monkeypatch.setattr(policy, "asset_sha256", mutate)
    result, metadata = apply_performance_mode(config, "compact")
    assert result["llm"]["cloud"]["enabled"] is False
    assert metadata == expected


def test_incompatible_active_hotwords_are_rejected_instead_of_losing_biasing(tmp_path):
    config = mode_config(tmp_path)
    config["sherpa"].update(asr_hotwords="OWNER NAME", asr_modeling_unit="cjkchar")
    with pytest.raises(PerformanceModeError, match="incompatible active"):
        apply_performance_mode(config, "compact")


def test_path_replacement_during_read_is_detected(tmp_path, monkeypatch):
    import core.performance as policy
    path = tmp_path / "model"
    path.write_bytes(b"original")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"different")
    original = policy.os.fstat
    calls = 0

    def replace_after_open(fd):
        nonlocal calls
        calls += 1
        info = original(fd)
        if calls == 1:
            replacement.replace(path)
        return info

    monkeypatch.setattr(policy.os, "fstat", replace_after_open)
    with pytest.raises(PerformanceModeError, match="changed while verifying"):
        asset_sha256(path)


def test_unreadable_tree_walk_fails_closed(tmp_path, monkeypatch):
    import core.performance as policy
    directory = tmp_path / "data"
    directory.mkdir()

    def denied(*args, **kwargs):
        raise PermissionError("private detail")

    monkeypatch.setattr(policy.os, "scandir", denied)
    with pytest.raises(PerformanceModeError, match="unavailable or changed") as caught:
        asset_sha256(directory)
    assert "private detail" not in str(caught.value)


def test_nested_inventory_addition_before_descent_is_detected(tmp_path, monkeypatch):
    import core.performance as policy
    directory = tmp_path / "data"
    child = directory / "nested"
    child.mkdir(parents=True)
    (child / "asset").write_bytes(b"model")
    expected = asset_sha256(directory)
    original = policy.os.scandir
    changed = False

    def add_before_enumeration(path):
        nonlocal changed
        if Path(path) == child and not changed:
            changed = True
            (child / "addition").write_bytes(b"new")
        return original(path)

    monkeypatch.setattr(policy.os, "scandir", add_before_enumeration)
    try:
        actual = asset_sha256(directory)
    except PerformanceModeError:
        pass
    else:
        assert actual != expected


def test_expressive_voice_permission_is_preserved(tmp_path):
    config = mode_config(tmp_path)
    config["sherpa"]["tts_lock_speaker_id"] = False
    config["sherpa"]["tts_speaker_voices"] = {"soft": 1}
    result, _ = apply_performance_mode(config, "compact")
    assert result["sherpa"]["tts_lock_speaker_id"] is False
    assert result["sherpa"]["tts_speaker_voices"] == {"soft": 1}
