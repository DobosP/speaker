"""Analytic latency model: project a conversation turn onto a machine spec.

Given a scenario (how many words the user speaks and the assistant replies) and
a :class:`MachineSpec`, compute the timeline of an ordinary fast-tier
``ASR -> LLM -> TTS`` turn and the user-perceived latencies. This mirrors how
``core``'s capabilities collect the *full* LLM stream before emitting one
``TTS_REQUEST`` (so TTS starts after generation completes). Main-tier latency
is deliberately not projected because the catalog has no measured estimates
for that role.

It's deterministic and instant -- no threads, no sleeps -- which is what makes
it usable as a cross-spec comparison rather than a timing-sensitive test.
"""
from __future__ import annotations

from dataclasses import dataclass

from .specs import MachineSpec

# Average user speaking rate (seconds per word) and token/word ratio. Constants
# because they describe the human + tokenizer, not the device.
SECONDS_PER_SPOKEN_WORD = 0.38
TOKENS_PER_WORD = 1.3

# Budget thresholds (seconds): (good_max, ok_max). Above ok_max => "fail".
FIRST_AUDIO_BUDGET = (1.2, 2.5)
BARGE_IN_BUDGET = (0.30, 0.50)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    user_words: int
    reply_words: int
    barge_in: bool = False


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("quick", "Short question, short spoken reply", user_words=6, reply_words=18),
    Scenario(
        "research",
        "Legacy long-reply workload projected on the fast/ordinary path",
        user_words=9,
        reply_words=75,
    ),
    Scenario("barge_in", "User interrupts mid-reply", user_words=5, reply_words=40, barge_in=True),
)


@dataclass(frozen=True)
class Segment:
    label: str
    kind: str  # css class: speech/endpoint/ttft/gen/ttfa/play
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class TurnResult:
    spec: str
    scenario: str
    segments: tuple[Segment, ...]
    first_audio_latency: float  # speech end -> first assistant audio
    response_complete: float  # speech end -> reply finished playing
    barge_in_stop: float | None  # None unless the scenario barges in
    total: float  # full timeline length (for drawing)


def simulate_turn(spec: MachineSpec, scenario: Scenario) -> TurnResult:
    reply_tokens = max(1, round(scenario.reply_words * TOKENS_PER_WORD))

    speech_dur = scenario.user_words * SECONDS_PER_SPOKEN_WORD
    endpoint = spec.stt_endpoint_delay_sec
    ttft = spec.llm_ttft_sec
    gen = reply_tokens * spec.llm_per_token_sec
    ttfa = spec.tts_ttfa_sec
    play = scenario.reply_words * spec.tts_realtime_factor

    t = 0.0
    segs: list[Segment] = []

    def add(label: str, kind: str, dur: float) -> None:
        nonlocal t
        segs.append(Segment(label, kind, t, dur))
        t += dur

    add("user speech", "speech", speech_dur)
    speech_end = speech_dur
    add("endpoint", "endpoint", endpoint)
    add("fast LLM ttft", "ttft", ttft)
    add("fast LLM generate", "gen", gen)
    add("TTS ttfa", "ttfa", ttfa)

    barge_stop = None
    if scenario.barge_in:
        # User speaks over playback partway through; barge-in halts TTS after the
        # gate's stop latency, so the reply never finishes.
        played = play * 0.4
        add("TTS (interrupted)", "play", played)
        first_audio = t - played
        barge_stop = spec.barge_in_stop_sec
        add("barge-in stop", "endpoint", barge_stop)
        complete = t - speech_end
    else:
        first_audio = t
        add("TTS playback", "play", play)
        complete = t - speech_end

    return TurnResult(
        spec=spec.name,
        scenario=scenario.name,
        segments=tuple(segs),
        first_audio_latency=round(first_audio - speech_end, 3),
        response_complete=round(complete, 3),
        barge_in_stop=round(barge_stop, 3) if barge_stop is not None else None,
        total=round(t, 3),
    )


def classify(value: float, budget: tuple[float, float]) -> str:
    good_max, ok_max = budget
    if value <= good_max:
        return "good"
    if value <= ok_max:
        return "ok"
    return "fail"


def simulate_all(specs, scenarios=SCENARIOS) -> list[TurnResult]:
    return [simulate_turn(spec, sc) for spec in specs for sc in scenarios]
