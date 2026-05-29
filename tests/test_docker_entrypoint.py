"""Tests for the ``docker/entrypoint.py`` config-overlay merge.

The overlay is the mechanism the docker-compose cloud-streaming flow
uses to flip ``cloud.enabled=true`` on the ``desktop_gpu_4090`` device
profile without editing the committed ``config.json``. The entrypoint
reuses the runtime's single recursive merge (``core.config.deep_merge``,
re-exported as ``entrypoint.deep_merge``) -- so the docker overlay shape
matches what ``core.config.load_config`` / ``apply_device_profile`` do:

- Nested dicts recurse (an overlay leaf keeps the base's siblings).
- ``device_profiles`` therefore goes as deep as needed (per-profile,
  per-section, per-sub-section) without restating the whole profile.
- The ``options`` bag is OPAQUE: replaced wholesale, never merged.
- Non-dict values replace.

The entrypoint imports ``core.config.deep_merge`` (PYTHONPATH=/app makes
``core`` importable at runtime) with a tiny inline fallback so it still
runs if ``core`` can't be imported; either path behaves identically.
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


def test_deep_merge_overlays_top_level_section():
    """A new top-level key in the overlay shows up in the merged config."""
    ep = _load_entrypoint()
    merged = ep.deep_merge({"a": 1}, {"b": 2})
    assert merged == {"a": 1, "b": 2}


def test_deep_merge_replaces_non_dict_values():
    """If both sides have a scalar/list under the same key, overlay wins."""
    ep = _load_entrypoint()
    merged = ep.deep_merge({"x": 1, "y": [1, 2]}, {"x": 9, "y": [9]})
    assert merged == {"x": 9, "y": [9]}


def test_deep_merge_merges_dicts_one_level_deep():
    """Per-section dicts merge key-by-key (matches core.config.deep_merge)."""
    ep = _load_entrypoint()
    merged = ep.deep_merge(
        {"llm": {"backend": "ollama", "main_model": "gemma3:12b"}},
        {"llm": {"main_model": "gemma3:4b"}},
    )
    assert merged == {"llm": {"backend": "ollama", "main_model": "gemma3:4b"}}


def test_deep_merge_recurses_into_device_profiles():
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
    merged = ep.deep_merge(base, overlay)
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
    merged = ep.deep_merge(base, overlay)
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
    merged = ep.deep_merge(base, overlay)
    # The leaf changed.
    assert merged["llm"]["cloud"]["enabled"] is True
    # Sibling keys in llm.cloud preserved.
    assert merged["llm"]["cloud"]["strategy"] == "hedge"
    assert merged["llm"]["cloud"]["model"] == "x"
    # Sibling subsections in llm preserved.
    assert merged["llm"]["backend"] == "ollama"
    # The untouched ``options`` bag survives intact (the overlay never names it).
    assert merged["llm"]["options"]["num_ctx"] == 4096


def test_deep_merge_treats_options_bag_as_opaque():
    """The ``options`` bag is OPAQUE (core.config._OPAQUE_KEYS): when an
    overlay supplies its own ``options`` it replaces the base bag wholesale
    rather than merging key-by-key, so a backend switch never inherits the
    old backend's generation params. Mirrors core.config.deep_merge."""
    ep = _load_entrypoint()
    base = {"llm": {"backend": "ollama", "options": {"num_ctx": 4096, "num_predict": 384}}}
    overlay = {"llm": {"backend": "llamacpp", "options": {"n_ctx": 2048}}}
    merged = ep.deep_merge(base, overlay)
    assert merged["llm"]["backend"] == "llamacpp"
    # Replaced wholesale -- the base's num_ctx/num_predict are gone.
    assert merged["llm"]["options"] == {"n_ctx": 2048}


def test_entrypoint_uses_core_config_deep_merge():
    """The entrypoint reuses the single recursive merge from core.config
    (P4 dedup) -- not a private duplicate. PYTHONPATH=/app makes ``core``
    importable at runtime; this test runs from the repo root where ``core``
    is on the path, so the real import (not the inline fallback) is used."""
    import core.config as cc

    ep = _load_entrypoint()
    assert ep.deep_merge is cc.deep_merge


def test_build_config_allow_prc_env_maps_to_cloud_flag(tmp_path, monkeypatch):
    """ALLOW_PRC opt-in (Locked Decision 2): a clearly-truthy value maps to
    ``llm.cloud.allow_prc=True``; anything else (incl. unset/garbage) leaves the
    config default untouched -- no surprise enable of the PRC-hosted presets."""
    ep = _load_entrypoint()
    src = tmp_path / "config.json"
    run_dir = tmp_path / "run"
    monkeypatch.setattr(ep, "SRC_CONFIG", str(src))
    monkeypatch.setattr(ep, "RUN_DIR", str(run_dir))
    monkeypatch.setattr(ep, "RUN_CONFIG", str(run_dir / "config.json"))
    monkeypatch.setattr(ep, "MODEL_PATHS", str(tmp_path / "absent-models.json"))
    monkeypatch.setattr(ep, "OVERLAY_PATH", str(tmp_path / "absent-overlay.json"))

    def _allow_prc_after(value):
        # Fresh base each run with the shipped default (allow_prc False).
        src.write_text(json.dumps({"llm": {"cloud": {"allow_prc": False}}}), encoding="utf-8")
        if value is None:
            monkeypatch.delenv("ALLOW_PRC", raising=False)
        else:
            monkeypatch.setenv("ALLOW_PRC", value)
        ep.build_config()
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        return cfg["llm"]["cloud"]["allow_prc"]

    assert _allow_prc_after("1") is True
    assert _allow_prc_after("True") is True
    assert _allow_prc_after("on") is True
    # Non-truthy / garbage / unset must NOT flip the default.
    assert _allow_prc_after("0") is False
    assert _allow_prc_after("banana") is False
    assert _allow_prc_after(None) is False
