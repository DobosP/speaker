from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DeviceProfile:
    """Latency model for a device + model-stack combination.

    These numbers are what make a simulation "feel" like a real device: they
    set how fast STT emits partials, how long the endpoint silence is, how slow
    the LLM is (a function of model *weight* and hardware), and how long TTS
    playback lasts. Scenarios run against a profile so we can ask "does the
    decision agent behave on a slow phone the way it does on a fast desktop?"
    """

    name: str
    # STT
    stt_partial_interval_sec: float  # gap between incremental partial hypotheses
    stt_endpoint_delay_sec: float  # trailing silence before a final fires
    # LLM (dominated by model weight: bigger model => higher ttft + per-token)
    llm_ttft_sec: float  # time to first token
    llm_per_token_sec: float  # streaming gap per token
    # TTS
    tts_ttfa_sec: float  # time to first audio
    tts_realtime_factor: float  # playback seconds per spoken word

    def scaled(self, factor: float) -> "DeviceProfile":
        """Return a profile with all latencies multiplied by ``factor``.

        Preserves *relative* timing (so concurrency behavior is unchanged) while
        letting tests run fast (e.g. ``factor=0.1`` => 10x quicker)."""
        return replace(
            self,
            name=f"{self.name}_x{factor}",
            stt_partial_interval_sec=self.stt_partial_interval_sec * factor,
            stt_endpoint_delay_sec=self.stt_endpoint_delay_sec * factor,
            llm_ttft_sec=self.llm_ttft_sec * factor,
            llm_per_token_sec=self.llm_per_token_sec * factor,
            tts_ttfa_sec=self.tts_ttfa_sec * factor,
            tts_realtime_factor=self.tts_realtime_factor * factor,
        )


# Slow phone running a tiny quantized LLM: high TTFT, slow tokens, long endpoint.
PHONE_LOW = DeviceProfile(
    name="phone_low",
    stt_partial_interval_sec=0.25,
    stt_endpoint_delay_sec=0.8,
    llm_ttft_sec=1.2,
    llm_per_token_sec=0.06,
    tts_ttfa_sec=0.5,
    tts_realtime_factor=0.35,
)

# Mid desktop with a small/medium local model.
DESKTOP_MID = DeviceProfile(
    name="desktop_mid",
    stt_partial_interval_sec=0.15,
    stt_endpoint_delay_sec=0.6,
    llm_ttft_sec=0.4,
    llm_per_token_sec=0.02,
    tts_ttfa_sec=0.2,
    tts_realtime_factor=0.3,
)

# Fast desktop / GPU.
DESKTOP_HIGH = DeviceProfile(
    name="desktop_high",
    stt_partial_interval_sec=0.1,
    stt_endpoint_delay_sec=0.5,
    llm_ttft_sec=0.15,
    llm_per_token_sec=0.008,
    tts_ttfa_sec=0.1,
    tts_realtime_factor=0.28,
)
