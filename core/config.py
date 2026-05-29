"""Config loading + device-profile layering for the runtime.

This module owns the two pure config transforms the CLI (``core/app.py``) and
the remote worker (``remote/worker.py``) both need, so neither has to reach into
``core.app`` internals:

- :func:`load_config` -- read ``config.json`` and overlay a machine-local
  ``config.local.json``.
- :func:`apply_device_profile` -- layer ``device_profiles[device]`` over the
  base config.

Both compose via :func:`deep_merge`, a **recursive** merge: an override that
touches a *nested* key (e.g. ``llm.cloud.enabled``) updates that leaf and keeps
its siblings, instead of replacing the whole sub-dict and stranding them. The
old per-section shallow merge could silently disable the cloud tier this way
(cross-platform-2): a profile that set only ``llm.cloud.enabled`` would drop the
sibling ``cloud_providers`` / ``cloud_chains`` and quietly degrade to local.

``_load_config`` / ``_apply_device_profile`` remain as aliases for the
historical private names some tools/tests still import.
"""
from __future__ import annotations

import json
import os

__all__ = [
    "deep_merge",
    "load_config",
    "apply_device_profile",
    "_load_config",
    "_apply_device_profile",
]


# Keys whose dict value is an OPAQUE, backend-specific bag that must be replaced
# WHOLESALE rather than recursively merged. ``llm.options`` carries
# backend-specific generation params -- Ollama's ``num_ctx`` vs llama.cpp's
# (which has no such generation kwarg and would TypeError). A device_profile that
# switches ``llm.backend`` and sets its own ``options`` must NOT inherit the base
# backend's option keys; the old shallow merge replaced these wholesale, so we
# preserve that here while still deep-merging every other nested key.
_OPAQUE_KEYS = frozenset({"options"})


def deep_merge(base: dict, overrides: dict, *, opaque_keys: frozenset = _OPAQUE_KEYS) -> dict:
    """Recursively merge ``overrides`` onto ``base``, returning a new dict.

    Where both sides hold a dict under the same key, recurse so an override
    that touches one nested leaf keeps the base's sibling leaves (the fix for
    the shallow-merge bug that could strand ``cloud_providers`` when a profile
    set only ``llm.cloud.enabled``). Any non-dict value -- or a dict replacing
    a scalar (or vice versa) -- replaces wholesale, matching the previous
    behaviour for configs without nested overrides.

    Keys in ``opaque_keys`` (default ``{"options"}``) are ALWAYS replaced
    wholesale even when both sides are dicts: they are backend-specific bags
    where merging the base backend's keys into a profile that switched backends
    would inject invalid params. Inputs are not mutated.
    """
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if key not in opaque_keys and isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = deep_merge(existing, value, opaque_keys=opaque_keys)
        else:
            merged[key] = value
    return merged


def load_config(path: str = "config.json", *, local: str = "config.local.json") -> dict:
    """Load ``config.json`` (the committed template) and deep-merge a
    machine-local ``config.local.json`` over it. Keeping machine-specific values
    (e.g. the sherpa model paths written by ``tools.setup_models``) in the
    gitignored local file keeps the template portable and out of git.

    The merge is recursive (:func:`deep_merge`): a local override of a nested
    key (e.g. ``llm.cloud.enabled``) preserves the base's siblings instead of
    replacing the whole ``llm.cloud`` sub-dict."""
    config: dict = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    # Hermetic-test guard: when SPEAKER_NO_LOCAL_CONFIG is truthy, skip the
    # machine-local overlay entirely. Without this, a machine that has real
    # model paths in config.local.json makes `--engine sherpa` start the live
    # capture loop instead of failing fast, hanging the test suite. Production
    # and default behaviour are unchanged (the var is unset by default).
    _skip_local = os.environ.get("SPEAKER_NO_LOCAL_CONFIG", "").strip().lower() not in ("", "0", "false", "no")
    if os.path.exists(local) and not _skip_local:
        with open(local, "r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        config = deep_merge(config, overrides)
    return config


def apply_device_profile(config: dict, device: str) -> dict:
    """Layer ``device_profiles[device]`` over the base config.

    A profile holds per-section overrides (``llm``, ``sherpa``); each is deep-
    merged onto the base so a phone profile can swap the LLM backend/models and
    retune CPU threads without restating every field -- and so a profile that
    overrides only a nested key (e.g. ``llm.cloud.enabled``) keeps the base's
    siblings (``cloud_providers`` / ``cloud_chains``) instead of stranding them
    (cross-platform-2). Unknown device -> no-op (returns the input unchanged)."""
    profile = config.get("device_profiles", {}).get(device)
    if not profile:
        return config
    return deep_merge(config, profile)


# Back-compat aliases for the historical private names.
_load_config = load_config
_apply_device_profile = apply_device_profile
