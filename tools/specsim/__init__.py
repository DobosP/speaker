"""Spec-simulation harness: model an ASR->LLM->TTS turn across machine specs
and render a self-contained HTML capability report."""

from .specs import CATALOG, MachineSpec

__all__ = ["CATALOG", "MachineSpec"]
