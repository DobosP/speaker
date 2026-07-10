"""Catalog of target machine specs and local-model fit logic.

A ``MachineSpec`` is a *model* of a real device: hardware summary plus the
latency numbers that make a simulation "feel" like that machine (in the spirit
of ``tests/sandbox/profiles.py``, extended with RAM/VRAM and platform so we can
also answer "does the chosen model even fit?"). These are estimates for a
comparative view, not measurements of real silicon.
"""
from __future__ import annotations

from dataclasses import dataclass

# Approx on-disk/VRAM weight footprint for the supported local artifacts, in GB.
MODEL_FOOTPRINTS_GB: dict[str, float] = {
    "minicpm5-1b:q4": 0.688,
    "minicpm5-1b:q8": 1.15,
    "gemma3:1b": 0.9,
    "gemma3:4b": 3.3,
    "gemma3:12b": 8.1,
    "gemma3:27b": 17.0,
}


@dataclass(frozen=True)
class MachineSpec:
    name: str
    platform: str  # Linux / Windows / macOS / Android / Web
    accelerator: str  # human-readable: what runs the LLM
    cores: int
    ram_gb: float
    # Memory actually available to the model (VRAM for GPU specs; a realistic
    # slice of RAM for CPU/unified/phone/web after OS + app + sherpa).
    model_budget_gb: float
    configured_model: str  # the model this device's profile would run
    # --- latency model (seconds) ---
    stt_partial_interval_sec: float
    stt_endpoint_delay_sec: float
    llm_ttft_sec: float
    llm_per_token_sec: float
    tts_ttfa_sec: float
    tts_realtime_factor: float  # playback seconds per spoken word
    barge_in_stop_sec: float  # mic-voice -> playback halts

    @property
    def tokens_per_sec(self) -> float:
        return round(1.0 / self.llm_per_token_sec, 1) if self.llm_per_token_sec else 0.0

    def fits(self, model: str) -> bool:
        return MODEL_FOOTPRINTS_GB.get(model, 1e9) <= self.model_budget_gb

    def fit_status(self, model: str) -> str:
        """``good`` (>=20% headroom), ``tight`` (fits, <20% headroom), ``fail``."""
        footprint = MODEL_FOOTPRINTS_GB.get(model, 1e9)
        if footprint > self.model_budget_gb:
            return "fail"
        if footprint > self.model_budget_gb * 0.8:
            return "tight"
        return "good"

    def largest_fitting_model(self) -> str | None:
        ordered = sorted(MODEL_FOOTPRINTS_GB, key=MODEL_FOOTPRINTS_GB.get)  # type: ignore[arg-type]
        best = None
        for model in ordered:
            if self.fits(model):
                best = model
        return best


# A spread from a CUDA laptop down to a browser, so the report shows a gradient.
CATALOG: tuple[MachineSpec, ...] = (
    MachineSpec(
        name="RTX 4090 Laptop",
        platform="Windows / Linux",
        accelerator="RTX 4090 Laptop GPU (16 GB)",
        cores=16,
        ram_gb=32,
        model_budget_gb=16.0,
        configured_model="gemma3:12b",
        stt_partial_interval_sec=0.10,
        stt_endpoint_delay_sec=0.50,
        llm_ttft_sec=0.15,
        llm_per_token_sec=0.009,
        tts_ttfa_sec=0.10,
        tts_realtime_factor=0.28,
        barge_in_stop_sec=0.30,
    ),
    MachineSpec(
        name="MacBook (M2)",
        platform="macOS",
        accelerator="Apple M2 GPU (16 GB unified)",
        cores=10,
        ram_gb=16,
        model_budget_gb=11.0,
        configured_model="gemma3:4b",
        stt_partial_interval_sec=0.12,
        stt_endpoint_delay_sec=0.55,
        llm_ttft_sec=0.30,
        llm_per_token_sec=0.020,
        tts_ttfa_sec=0.15,
        tts_realtime_factor=0.30,
        barge_in_stop_sec=0.30,
    ),
    MachineSpec(
        name="Windows laptop (CPU/iGPU)",
        platform="Windows",
        accelerator="CPU / integrated GPU",
        cores=8,
        ram_gb=16,
        model_budget_gb=9.0,
        configured_model="gemma3:4b",
        stt_partial_interval_sec=0.15,
        stt_endpoint_delay_sec=0.60,
        llm_ttft_sec=0.80,
        llm_per_token_sec=0.080,
        tts_ttfa_sec=0.30,
        tts_realtime_factor=0.32,
        barge_in_stop_sec=0.35,
    ),
    MachineSpec(
        name="Android phone (12 GB)",
        platform="Android",
        accelerator="mobile CPU (llama.cpp)",
        cores=8,
        ram_gb=12,
        model_budget_gb=5.0,
        configured_model="minicpm5-1b:q4",
        stt_partial_interval_sec=0.25,
        stt_endpoint_delay_sec=0.80,
        llm_ttft_sec=1.20,
        llm_per_token_sec=0.120,
        tts_ttfa_sec=0.50,
        tts_realtime_factor=0.35,
        barge_in_stop_sec=0.40,
    ),
    MachineSpec(
        name="Low-end phone (6 GB)",
        platform="Android",
        accelerator="mobile CPU (llama.cpp)",
        cores=6,
        ram_gb=6,
        model_budget_gb=2.2,
        configured_model="minicpm5-1b:q4",
        stt_partial_interval_sec=0.30,
        stt_endpoint_delay_sec=0.90,
        llm_ttft_sec=1.50,
        llm_per_token_sec=0.090,
        tts_ttfa_sec=0.60,
        tts_realtime_factor=0.40,
        barge_in_stop_sec=0.45,
    ),
    MachineSpec(
        name="Web (WASM)",
        platform="Web",
        accelerator="browser WASM (CPU)",
        cores=4,
        ram_gb=4,
        model_budget_gb=1.5,
        configured_model="minicpm5-1b:q4",
        stt_partial_interval_sec=0.35,
        stt_endpoint_delay_sec=0.90,
        llm_ttft_sec=3.00,
        llm_per_token_sec=0.300,
        tts_ttfa_sec=1.00,
        tts_realtime_factor=0.60,
        barge_in_stop_sec=0.60,
    ),
)
