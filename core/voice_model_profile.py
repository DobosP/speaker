"""Pure desktop voice model selection and transport-free pinned identity.

The project alias retains ``candidate`` to bind the measured local import. It is
an explicit software profile, not an acoustic/live or universal quality verdict.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable

from .minicpm_identity import (
    _blob_ref, _capabilities, _field, _modelfile_parameter_values,
    _mapping, _parameter_values, _template, _top_level_directives,
)

VOICE_MODEL_PROFILE_NAMES = ("current", "qwen2.5-1.5b")
PROFILE_ALIAS = "speaker-qwen2.5-1.5b:q4km-candidate"
PROFILE_REVISION = "dd26da440ef0330c47919d1ecae0966d24022222"
PROFILE_SOURCE_SHA256 = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
PROFILE_RUNTIME_SHA256 = "098cb604ff3cc846891b7e8c00abe4f52f5c6fdc936e21e7e41f2eaf22c1c7cb"
PROFILE_SOURCE_BYTES = 1_117_320_736
PROFILE_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
PROFILE_ASSET_RELATIVE = "models/candidates/" + PROFILE_FILENAME
PROFILE_SOURCE_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/"
    + PROFILE_REVISION + "/" + PROFILE_FILENAME
)
# Exact native/Go templates, parameter multisets and capabilities; independent
# of cache paths and Ollama's nondeterministic Modelfile parameter ordering.
PROFILE_CONFIG_SHA256 = "0a824a712723416708ee894c82f8e4a6823dcd608c8f0e281435c04acd24620f"
PROFILE_FAST_OPTIONS = {"temperature": 0.0, "top_p": 0.95, "seed": 0, "num_thread": 2}


@dataclass(frozen=True)
class VoiceModelProfileMetadata:
    name: str
    sha256: str
    schema_version: int = 1


def apply_voice_model_profile(
    config: dict, requested: str | None = None,
) -> tuple[dict, VoiceModelProfileMetadata | None]:
    """Return an idempotent selection, without I/O or config mutations.

    Explicit ``current`` writes a rollback marker when overriding a selected
    profile; normal unselected current remains the original object unchanged.
    The answer caps, main/vision model and all tool/privacy settings survive.
    """
    if type(config) is not dict:
        raise ValueError("voice model config must be a dictionary")
    name = config.get("voice_model_profile", "current") if requested is None else requested
    if type(name) is not str or name not in VOICE_MODEL_PROFILE_NAMES:
        raise ValueError("unsupported voice_model_profile")
    if name == "current":
        if requested is None or config.get("voice_model_profile", "current") == "current":
            return config, None
        if (config.get("llm") or {}).get("fast_model") == PROFILE_ALIAS:
            raise ValueError("current requires the original device config before voice-model selection")
        result = deepcopy(config)
        result["voice_model_profile"] = "current"
        return result, None
    result = deepcopy(config)
    llm = result.setdefault("llm", {})
    assistant = result.setdefault("assistant", {})
    if type(llm) is not dict or type(assistant) is not dict:
        raise ValueError("voice profile requires llm and assistant dictionaries")
    if llm.get("backend", "ollama") != "ollama":
        raise ValueError("qwen voice profile requires the ollama backend")
    result["voice_model_profile"] = name
    llm["fast_model"] = PROFILE_ALIAS
    # These keys are deliberately sampling/thread policy only: caps stay owned
    # by the user's existing llm.options and device profile.
    fast_options = llm.get("fast_options", {})
    if type(fast_options) is not dict:
        raise ValueError("llm.fast_options must be a dictionary")
    llm["fast_options"] = {**fast_options, **PROFILE_FAST_OPTIONS}
    assistant["prompt_profile"] = "spoken"
    payload = {"name": name, "alias": PROFILE_ALIAS,
               "source_sha256": PROFILE_SOURCE_SHA256,
               "runtime_sha256": PROFILE_RUNTIME_SHA256,
               "config_sha256": PROFILE_CONFIG_SHA256,
               "fast_options": PROFILE_FAST_OPTIONS, "prompt_profile": "spoken"}
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result, VoiceModelProfileMetadata(name, digest)


def portable_config_digest(shown: object) -> str:
    """Canonical effective behavior, with exact templates and unordered maps."""
    modelfile = str(_field(shown, "modelfile", "") or "")
    directives = _top_level_directives(modelfile)
    if (sum(key == "FROM" for key, _ in directives) != 1
            or sum(key == "TEMPLATE" for key, _ in directives) != 1):
        return ""
    if any(key not in {"FROM", "TEMPLATE", "PARAMETER", "LICENSE"} for key, _ in directives):
        return ""
    # Ollama renders its parameter map in nondeterministic Modelfile order.
    # Comments and FROM cache paths are not inference behavior; bind both exact
    # templates and both parsed parameter views instead. Preserve duplicate
    # values so a second declaration cannot disappear during normalization.
    def parameters(values):
        return {key: sorted(entries) for key, entries in values.items()}
    payload = {
        "go_template": _template(shown),
        "native_template": str(_field(shown, "template", "") or "").replace("\r\n", "\n"),
        "parameters": parameters(_parameter_values(shown)),
        "modelfile_parameters": parameters(_modelfile_parameter_values(modelfile)),
        "capabilities": sorted(_capabilities(shown)),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class VoiceModelIdentity:
    ok: bool
    error: str
    blob_sha256: str = ""
    config_sha256: str = ""


def verify_voice_model_identity(*, show: Callable[[str], object]) -> VoiceModelIdentity:
    """Check the selected native alias, without model loading or source I/O."""
    try:
        shown = show(PROFILE_ALIAS)
        blob = _blob_ref(shown)
        config_digest = portable_config_digest(shown)
        details = _mapping(_field(shown, "details", {}))
        ok = (
            blob == PROFILE_RUNTIME_SHA256
            and config_digest == PROFILE_CONFIG_SHA256
            and details.get("format") == "gguf"
            and details.get("family") == "qwen2"
            and details.get("quantization_level") == "Q4_K_M"
            and sorted(_capabilities(shown)) == ["completion", "tools"]
        )
        return VoiceModelIdentity(ok, "" if ok else "voice_model_identity_mismatch", blob, config_digest)
    except Exception:  # no transport/backend details in readiness receipts
        return VoiceModelIdentity(False, "voice_model_identity_unavailable")
