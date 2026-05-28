"""Tests for the ``docker/entrypoint.py`` config-overlay merge.

The overlay is the mechanism the docker-compose cloud-streaming flow
uses to flip ``cloud.enabled=true`` on the ``desktop_gpu_4090`` device
profile without editing the committed ``config.json``. The shape needs
to match what ``core.app._load_config`` does:

- Top-level sections merge shallow (overlay wins).
- ``device_profiles`` is two-level deep: per-profile, per-section.
- Non-dict values replace.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_entrypoint():
    """Import docker/entrypoint.py as a module without executing main()."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "docker" / "entrypoint.py"
    spec = importlib.util.spec_from_file_location("docker_entrypoint", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shallow_merge_overlays_top_level_section():
    """A new top-level key in the overlay shows up in the merged config."""
    ep = _load_entrypoint()
    merged = ep._shallow_merge({"a": 1}, {"b": 2})
    assert merged == {"a": 1, "b": 2}


def test_shallow_merge_replaces_non_dict_values():
    """If both sides have a scalar/list under the same key, overlay wins."""
    ep = _load_entrypoint()
    merged = ep._shallow_merge({"x": 1, "y": [1, 2]}, {"x": 9, "y": [9]})
    assert merged == {"x": 9, "y": [9]}


def test_shallow_merge_merges_dicts_one_level_deep():
    """Per-section dicts merge key-by-key (matches core.app._load_config)."""
    ep = _load_entrypoint()
    merged = ep._shallow_merge(
        {"llm": {"backend": "ollama", "main_model": "gemma3:12b"}},
        {"llm": {"main_model": "gemma3:4b"}},
    )
    assert merged == {"llm": {"backend": "ollama", "main_model": "gemma3:4b"}}


def test_shallow_merge_recurses_into_device_profiles():
    """``device_profiles`` is the one place the merge needs to go two
    levels deep -- otherwise a one-line overlay would have to restate
    the entire profile."""
    ep = _load_entrypoint()
    base = {
        "device_profiles": {
            "desktop_gpu_4090": {
                "llm": {
                    "backend": "ollama",
                    "main_model": "gemma3:12b",
                    "cloud": {"enabled": False, "strategy": "hedge"},
                },
                "sherpa": {"asr_num_threads": 4},
            },
        },
    }
    overlay = {
        "device_profiles": {
            "desktop_gpu_4090": {
                "llm": {
                    "cloud": {"enabled": True, "hedge_delay_ms": 50},
                },
            },
        },
    }
    merged = ep._shallow_merge(base, overlay)
    profile = merged["device_profiles"]["desktop_gpu_4090"]
    # The deep section was merged, not replaced wholesale:
    assert profile["llm"]["main_model"] == "gemma3:12b"
    assert profile["llm"]["cloud"]["enabled"] is True
    assert profile["llm"]["cloud"]["strategy"] == "hedge"          # base survived
    assert profile["llm"]["cloud"]["hedge_delay_ms"] == 50         # overlay added
    assert profile["sherpa"]["asr_num_threads"] == 4               # untouched


def test_committed_overlay_enables_cloud_on_4090_profile():
    """The shipped ``docker/config.overlay.json`` must actually flip
    ``cloud.enabled`` true on the 4090 profile -- the whole point of
    the docker-compose flow."""
    ep = _load_entrypoint()
    repo_root = Path(__file__).resolve().parent.parent
    base = json.load(open(repo_root / "config.json"))
    overlay = json.load(open(repo_root / "docker" / "config.overlay.json"))
    merged = ep._shallow_merge(base, overlay)
    cloud = merged["device_profiles"]["desktop_gpu_4090"]["llm"]["cloud"]
    assert cloud["enabled"] is True
    # And it didn't break anything else on the profile.
    profile_llm = merged["device_profiles"]["desktop_gpu_4090"]["llm"]
    assert profile_llm["main_model"] == "gemma3:12b"
    assert profile_llm["backend"] == "ollama"


def test_overlay_path_default_can_be_overridden_via_env(monkeypatch):
    """``SPEAKER_CONFIG_OVERLAY`` env var overrides the default
    /app/config.overlay.json path -- lets users mount their own
    overlay at a different location."""
    monkeypatch.setenv("SPEAKER_CONFIG_OVERLAY", "/tmp/my-overlay.json")
    ep = _load_entrypoint()  # _load reads module fresh; env applied
    assert ep.OVERLAY_PATH == "/tmp/my-overlay.json"


def test_deep_merge_preserves_siblings_at_every_level():
    """Counter-test for the bug the deep-merge replaces: an overlay
    targeting one leaf must not nuke unrelated leaves on the way down."""
    ep = _load_entrypoint()
    base = {
        "llm": {
            "backend": "ollama",
            "options": {"num_ctx": 4096, "num_predict": 384},
            "cloud": {"enabled": False, "strategy": "hedge", "model": "x"},
        },
    }
    overlay = {"llm": {"cloud": {"enabled": True}}}
    merged = ep._shallow_merge(base, overlay)
    # The leaf changed.
    assert merged["llm"]["cloud"]["enabled"] is True
    # Sibling keys in llm.cloud preserved.
    assert merged["llm"]["cloud"]["strategy"] == "hedge"
    assert merged["llm"]["cloud"]["model"] == "x"
    # Sibling subsections in llm preserved.
    assert merged["llm"]["backend"] == "ollama"
    assert merged["llm"]["options"]["num_ctx"] == 4096
