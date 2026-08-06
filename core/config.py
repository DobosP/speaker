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

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "deep_merge",
    "load_config",
    "resolve_device",
    "apply_device_profile",
    "FINAL_STT_PROFILE_NAMES",
    "FinalSttProfileMetadata",
    "SpeakerIdentityPolicyError",
    "apply_final_stt_profile",
    "apply_no_speaker_enrollment",
    "_load_config",
    "_apply_device_profile",
]


# Keys whose dict value is an OPAQUE bag that must be replaced WHOLESALE rather
# than recursively merged. ``llm.options`` carries backend-specific generation
# params -- Ollama's ``num_ctx`` vs llama.cpp's (which has no such generation
# kwarg and would TypeError). ``final_stt_profiles`` is a complete, validated
# profile collection: merging only model or verifier leaves would destroy its
# atomic evidence identity. Preserve wholesale behavior for both bags.
_OPAQUE_KEYS = frozenset({"options", "final_stt_profiles"})


FINAL_STT_PROFILE_NAMES = ("sense-voice", "parakeet-faster-whisper")
_FINAL_STT_PROFILE_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_FINAL_STT_STRING_FIELDS = frozenset(
    {
        "asr_final_backend",
        "asr_final_model",
        "asr_final_tokens",
        "asr_final_decoder",
        "asr_final_joiner",
        "asr_final_language",
        "asr_final_verifier_backend",
        "asr_final_verifier_model",
        "asr_final_hr_dict_dir",
        "asr_final_hr_lexicon",
        "asr_final_hr_rule_fsts",
        "asr_final_rule_fsts",
    }
)
_FINAL_STT_BOOL_FIELDS = frozenset(
    {"asr_final_use_itn", "asr_final_async", "asr_final_required"}
)
_FINAL_STT_NUMBER_FIELDS = frozenset(
    {"asr_final_min_sec", "asr_final_preroll_sec"}
)
_FINAL_STT_INTEGER_FIELDS = frozenset({"asr_final_verifier_cpu_threads"})
_FINAL_STT_SHERPA_FIELDS = (
    _FINAL_STT_STRING_FIELDS
    | _FINAL_STT_BOOL_FIELDS
    | _FINAL_STT_NUMBER_FIELDS
    | _FINAL_STT_INTEGER_FIELDS
)


@dataclass(frozen=True)
class FinalSttProfileMetadata:
    """Path- and transcript-free identity for one applied final-STT profile."""

    name: str
    sha256: str
    schema_version: int


class SpeakerIdentityPolicyError(ValueError):
    """A process-local speaker-identity selection conflicts with policy."""


def _canonical_profile_sha256(profile: dict) -> str:
    try:
        payload = json.dumps(
            profile,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"final STT profile is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _validate_final_stt_profile(name: str, profile: object) -> dict:
    if not isinstance(profile, dict):
        raise ValueError(f"final STT profile {name!r} must be an object")
    if set(profile) != {"schema_version", "sherpa"}:
        raise ValueError(
            f"final STT profile {name!r} must contain exactly "
            "'schema_version' and 'sherpa'"
        )
    schema_version = profile.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ValueError(f"final STT profile {name!r} schema_version must be 1")
    sherpa = profile.get("sherpa")
    if not isinstance(sherpa, dict):
        raise ValueError(f"final STT profile {name!r} sherpa must be an object")
    missing = sorted(_FINAL_STT_SHERPA_FIELDS - set(sherpa))
    extra = sorted(set(sherpa) - _FINAL_STT_SHERPA_FIELDS)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unknown {', '.join(extra)}")
        raise ValueError(
            f"final STT profile {name!r} has an incomplete sherpa object: "
            + "; ".join(details)
        )
    for key in _FINAL_STT_STRING_FIELDS:
        value = sherpa[key]
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"final STT profile {name!r} {key} must be a string")
    for key in _FINAL_STT_BOOL_FIELDS:
        if not isinstance(sherpa[key], bool):
            raise ValueError(f"final STT profile {name!r} {key} must be boolean")
    for key in _FINAL_STT_NUMBER_FIELDS:
        value = sherpa[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"final STT profile {name!r} {key} must be a finite non-negative number"
            )
    threads = sherpa["asr_final_verifier_cpu_threads"]
    if (
        isinstance(threads, bool)
        or not isinstance(threads, int)
        or not 0 <= threads <= 256
    ):
        raise ValueError(
            f"final STT profile {name!r} asr_final_verifier_cpu_threads "
            "must be an integer from 0 to 256"
        )

    backend = sherpa["asr_final_backend"]
    verifier = sherpa["asr_final_verifier_backend"]
    expected_pair = {
        "sense-voice": ("sense_voice", ""),
        "parakeet-faster-whisper": ("nemo_transducer", "faster_whisper"),
    }[name]
    if (backend, verifier) != expected_pair:
        raise ValueError(
            f"final STT profile {name!r} must select "
            f"backend={expected_pair[0]!r}, verifier={expected_pair[1]!r}"
        )
    if not sherpa["asr_final_model"] or not sherpa["asr_final_tokens"]:
        raise ValueError(f"final STT profile {name!r} needs final model and tokens")
    if backend == "sense_voice":
        if sherpa["asr_final_decoder"] or sherpa["asr_final_joiner"]:
            raise ValueError(
                f"final STT profile {name!r} cannot set decoder or joiner"
            )
    else:
        if not sherpa["asr_final_decoder"] or not sherpa["asr_final_joiner"]:
            raise ValueError(
                f"final STT profile {name!r} needs decoder and joiner"
            )
    if verifier:
        if not sherpa["asr_final_verifier_model"] or threads < 1:
            raise ValueError(
                f"final STT profile {name!r} needs a verifier model and bounded CPU threads"
            )
    elif sherpa["asr_final_verifier_model"] or threads:
        raise ValueError(
            f"final STT profile {name!r} has asr_final_verifier_model or "
            "asr_final_verifier_cpu_threads settings while disabled"
        )
    fixed_policy = {
        "asr_final_use_itn": True,
        "asr_final_language": "en",
        "asr_final_hr_dict_dir": "",
        "asr_final_hr_lexicon": "",
        "asr_final_hr_rule_fsts": "",
        "asr_final_rule_fsts": "",
        "asr_final_min_sec": 0.5,
        "asr_final_async": True,
        "asr_final_required": True,
        "asr_final_preroll_sec": 0.8,
        "asr_final_verifier_cpu_threads": (
            0 if name == "sense-voice" else 1
        ),
    }
    for key, expected in fixed_policy.items():
        if sherpa[key] != expected:
            raise ValueError(
                f"final STT profile {name!r} {key} must remain {expected!r}"
            )
    return profile


def apply_final_stt_profile(
    config: dict, name: str
) -> tuple[dict, FinalSttProfileMetadata]:
    """Atomically apply one complete, validated final-STT profile.

    The profile is applied after device selection by callers and replaces every
    final-ASR model, verifier, biasing, timing, and scheduling field as one unit.
    Inputs are never mutated.  The returned metadata deliberately contains only
    a CLI-safe name and the canonical selected-profile digest; model paths never
    need to enter a run summary.
    """

    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    if (
        not isinstance(name, str)
        or not _FINAL_STT_PROFILE_NAME_RE.fullmatch(name)
        or name not in FINAL_STT_PROFILE_NAMES
    ):
        valid = ", ".join(FINAL_STT_PROFILE_NAMES)
        raise ValueError(f"unknown final STT profile {name!r}; valid profiles: {valid}")
    profiles = config.get("final_stt_profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(
        FINAL_STT_PROFILE_NAMES
    ):
        raise ValueError(
            "final_stt_profiles must be one complete opaque map containing "
            + ", ".join(FINAL_STT_PROFILE_NAMES)
        )
    validated = {
        profile_name: _validate_final_stt_profile(profile_name, value)
        for profile_name, value in profiles.items()
    }
    selected = validated[name]
    effective = deep_merge(config, {"sherpa": dict(selected["sherpa"])})
    metadata = FinalSttProfileMetadata(
        name=name,
        sha256=_canonical_profile_sha256(selected),
        schema_version=1,
    )
    return effective, metadata


def apply_no_speaker_enrollment(config: dict) -> dict:
    """Ignore persisted speaker enrollment for one effective configuration.

    Only the two enrollment references are masked, and the input mapping is
    never mutated.  The configured embedding model and all policy switches are
    preserved: in particular, an explicit identity-required word-cut policy is
    rejected when this transform makes enrollment unavailable.
    Callers apply this after device and final-STT profiles so neither can
    reintroduce a reference during the selected process.
    """

    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    effective = deep_merge(
        config,
        {
            "sherpa": {
                "speaker_enroll_embedding": "",
                "speaker_enroll_wav": "",
            }
        },
    )
    from .engines.speaker_gate import resolve_speaker_identity_activation

    sherpa = effective.get("sherpa", {}) or {}
    activation = resolve_speaker_identity_activation(
        speaker_enroll_embedding=sherpa.get("speaker_enroll_embedding", ""),
        speaker_enroll_wav=sherpa.get("speaker_enroll_wav", ""),
        barge_in_enabled=sherpa.get("barge_in_enabled", True),
        barge_word_cut_enabled=sherpa.get("barge_word_cut_enabled", False),
        aec_enabled=sherpa.get("aec_enabled", False),
        barge_word_cut_require_speaker=sherpa.get(
            "barge_word_cut_require_speaker", False
        ),
        # Both references were masked above.  Avoid probing any configured
        # enrollment path while resolving this process-local policy.
        exists=lambda _path: False,
    )
    if activation.word_cut_requires_speaker:
        raise SpeakerIdentityPolicyError(
            "speaker identity off conflicts with the active "
            "identity-required multi-voice word-cut policy"
        )
    return effective


def deep_merge(base: dict, overrides: dict, *, opaque_keys: frozenset = _OPAQUE_KEYS) -> dict:
    """Recursively merge ``overrides`` onto ``base``, returning a new dict.

    Where both sides hold a dict under the same key, recurse so an override
    that touches one nested leaf keeps the base's sibling leaves (the fix for
    the shallow-merge bug that could strand ``cloud_providers`` when a profile
    set only ``llm.cloud.enabled``). Any non-dict value -- or a dict replacing
    a scalar (or vice versa) -- replaces wholesale, matching the previous
    behaviour for configs without nested overrides.

    Keys in ``opaque_keys`` (by default ``options`` and
    ``final_stt_profiles``) are ALWAYS replaced wholesale even when both sides
    are dicts: they are indivisible backend/profile bags where recursive merging
    could inject invalid params or break an evidence identity. Inputs are not
    mutated.
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
    # model paths in config.local.json makes `--session local` start the live
    # capture loop instead of failing fast, hanging the test suite. Production
    # and default behaviour are unchanged (the var is unset by default).
    _skip_local = os.environ.get("SPEAKER_NO_LOCAL_CONFIG", "").strip().lower() not in ("", "0", "false", "no")
    if os.path.exists(local) and not _skip_local:
        with open(local, "r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        config = deep_merge(config, overrides)
    return config


def resolve_device(config: dict, device) -> Tuple[str, Optional[str]]:
    """Resolve a device selector to a concrete ``device_profiles`` name.

    ``device`` may be a concrete profile name, or one of ``None`` / ``""`` /
    ``"auto"`` to PROBE the host hardware (cores / RAM / GPU / mobile) and pick
    the matching profile (device-adapt-1). Returns ``(profile_name,
    rationale_or_None)`` -- ``rationale`` is non-``None`` only when auto-detection
    actually ran, so the caller can surface "auto-selected X because Y".

    The probe (``tools.recommend_profile``) is stdlib-only and reads ONLY local
    hardware counters: no audio, no transcripts, no network, no cloud. Auto
    selection can never enable cloud or relax an owner gate -- every shipped
    profile keeps ``local_only``/cloud-off and the actuator/speaker-ID gates
    closed (enforced by ``tests/test_device_profile_invariants.py``)."""
    if device in (None, "", "auto"):
        # Lazy import: keeps the hardware probe (and the core->tools edge) out of
        # the hot path; only paid when a caller actually asks to auto-detect.
        from tools.recommend_profile import probe, recommend

        name, rationale = recommend(probe())
        return name, rationale
    return device, None


def apply_device_profile(config: dict, device, *, strict: bool = False) -> dict:
    """Layer ``device_profiles[device]`` over the base config.

    A profile holds per-section overrides (``llm``, ``sherpa``); each is deep-
    merged onto the base so a phone profile can swap the LLM backend/models and
    retune CPU threads without restating every field -- and so a profile that
    overrides only a nested key (e.g. ``llm.cloud.enabled``) keeps the base's
    siblings (``cloud_providers`` / ``cloud_chains``) instead of stranding them
    (cross-platform-2).

    ``device`` may be ``"auto"`` (or ``None`` / ``""``) to auto-detect the
    profile from the host hardware via :func:`resolve_device`.

    Unknown device -> no-op (returns the input unchanged) UNLESS ``strict``, in
    which case it raises ``ValueError`` listing the valid names. The CLI and the
    remote worker pass ``strict=True`` so a mistyped ``--device`` fails fast
    instead of silently running the heavy base config on exactly the low-spec
    box that needed a profile (cross-platform-8)."""
    device, _ = resolve_device(config, device)
    profiles = config.get("device_profiles", {})
    profile = profiles.get(device)
    if not profile:
        if strict:
            valid = ", ".join(sorted(profiles)) or "(none defined)"
            raise ValueError(
                f"unknown --device {device!r}; valid profiles: {valid} (or 'auto')"
            )
        return config
    return deep_merge(config, profile)


# Back-compat aliases for the historical private names.
_load_config = load_config
_apply_device_profile = apply_device_profile
