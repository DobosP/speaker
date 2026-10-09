"""Shared CLI selection for runtime, readiness and the physical launcher."""

from __future__ import annotations

from .voice_model_profile import (
    PROFILE_ALIAS,
    VoiceModelProfileMetadata,
    apply_voice_model_profile,
)


def select_voice_model(
    config: dict,
    requested: str | None = None,
    *,
    model: str | None = None,
    fast_model: str | None = None,
) -> tuple[dict, VoiceModelProfileMetadata | None]:
    selected, metadata = apply_voice_model_profile(config, requested)
    for value in (model, fast_model):
        if value is not None and (
            type(value) is not str or not value.strip() or len(value) > 512
        ):
            raise ValueError("model overrides must be nonempty bounded names")
    if metadata is not None and fast_model is not None and fast_model != PROFILE_ALIAS:
        raise ValueError(
            "voice-model profile conflicts with --fast-model; select --voice-model current for a custom fast model"
        )
    if model is not None or fast_model is not None:
        selected = dict(selected)
        llm = dict(selected.get("llm", {}) or {})
        if model is not None:
            llm["main_model"] = model
        if fast_model is not None:
            llm["fast_model"] = fast_model
        selected["llm"] = llm
    return selected, metadata
