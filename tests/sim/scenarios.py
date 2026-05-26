"""Catalog of (Persona, Goal) scenarios for both simulator tiers.

MOCK_SCENARIOS: scripted user + scripted assistant -> fully deterministic; they
exercise the real recorder/endpointing/router path and the runner/GoalCheck/
pass^k machinery in CI without any network.

LLM_SCENARIOS: persona prompts + goals for the opt-in real-Ollama tier; checks
are intentionally robust (model-agnostic) so a small local model does not red-bar
the build -- reliability is reported via pass^k, not asserted hard.
"""
from __future__ import annotations

from tests.sim.persona import GoalCheck, Goal, Persona, Scenario
from tests.sim.quality import is_voice_format_ok


def _assistant_responded(transcript) -> bool:  # noqa: ANN001
    return any(chunk.strip() for chunk in transcript.assistant_spoken)


def _replies_stay_short(transcript) -> bool:  # noqa: ANN001
    # Every assistant reply must obey the voice-format contract, even under an
    # adversarial user demanding a long answer.
    replies = [t for role, t in transcript.turns if role == "assistant"]
    return bool(replies) and all(is_voice_format_ok(r) for r in replies)


# ── Mock tier (deterministic) ───────────────────────────────────────────────
MOCK_SCENARIOS: list[Scenario] = [
    Scenario(
        persona=Persona(name="terse-asker", style="terse", scripted_turns=("what is the weather like today",)),
        goal=Goal(
            description="Find out today's weather.",
            checks=(
                GoalCheck("got_weather", "tts_spoken", target="sunny"),
                GoalCheck("used_llm", "route_action", target="llm"),
                GoalCheck("converged", "turn_count_max", target="2"),
            ),
        ),
        assistant_script=("It is sunny today.",),
    ),
    Scenario(
        persona=Persona(name="clock-watcher", style="neutral", scripted_turns=("what time is it",)),
        goal=Goal(
            description="Ask for the current time.",
            checks=(
                GoalCheck("hit_capability", "route_action", target="capability"),
                GoalCheck("spoke_time", "tts_spoken", target="current time"),
            ),
        ),
    ),
    Scenario(
        persona=Persona(name="polite-leaver", style="polite", scripted_turns=("goodbye",)),
        goal=Goal(
            description="End the session.",
            checks=(GoalCheck("shut_down", "route_action", target="shutdown"),),
        ),
    ),
    Scenario(
        persona=Persona(
            name="introducer",
            style="chatty",
            scripted_turns=("my name is sam", "what is my name"),
        ),
        goal=Goal(
            description="Introduce yourself and confirm the assistant has your name.",
            checks=(
                GoalCheck("name_echoed", "tts_spoken", target="sam"),
                GoalCheck("no_early_exit", "no_shutdown"),
                GoalCheck("converged", "turn_count_max", target="4"),
            ),
        ),
        assistant_script=("Nice to meet you Sam.", "Your name is Sam."),
    ),
]


# ── Real-Ollama tier (opt-in) ───────────────────────────────────────────────
LLM_SCENARIOS: list[Scenario] = [
    Scenario(
        persona=Persona(
            name="weather-curious",
            style="casual",
            system_prompt="You are curious about the weather today.",
            scripted_turns=("what is the weather like today",),  # fallback only
        ),
        goal=Goal(
            description="Ask the assistant about the weather and get a spoken answer.",
            checks=(
                GoalCheck("assistant_responded", "custom", predicate=_assistant_responded),
                GoalCheck("no_crash_shutdown", "no_shutdown"),
            ),
            max_turns=2,
        ),
    ),
    Scenario(
        persona=Persona(
            name="time-asker",
            style="direct",
            system_prompt='You want to know the current time. Ask plainly, e.g. "what time is it".',
            scripted_turns=("what time is it",),
        ),
        goal=Goal(
            description="Find out the current time.",
            checks=(GoalCheck("assistant_responded", "custom", predicate=_assistant_responded),),
            max_turns=2,
        ),
    ),
    # Adversarial: pressure the model for a long answer; the voice-format contract
    # must still hold on every reply (metamorphic format-robustness under a real user).
    Scenario(
        persona=Persona(
            name="format-attacker",
            style="demanding",
            system_prompt=(
                "You keep demanding very long, detailed, multi-paragraph answers and "
                "complain that replies are too short. Push hard for a long response."
            ),
            scripted_turns=("give me a long detailed essay about the solar system",),
        ),
        goal=Goal(
            description="Try to make the assistant produce a long, multi-paragraph answer.",
            checks=(
                GoalCheck("replies_stay_short", "custom", predicate=_replies_stay_short),
                GoalCheck("no_crash_shutdown", "no_shutdown"),
            ),
            max_turns=3,
        ),
    ),
    # Diverse: a Romanian-speaking user (router is English-only, so this exercises
    # the LLM fallback path with non-English input).
    Scenario(
        persona=Persona(
            name="romanian-speaker",
            style="casual",
            language="ro",
            system_prompt="Vorbesti numai in romana. Intreaba ceva simplu despre vreme.",
            scripted_turns=("cum este vremea astazi",),
        ),
        goal=Goal(
            description="Have a short exchange in Romanian.",
            checks=(
                GoalCheck("assistant_responded", "custom", predicate=_assistant_responded),
                GoalCheck("no_crash_shutdown", "no_shutdown"),
            ),
            max_turns=2,
        ),
    ),
    # Multi-turn topic switch: tests context handling across turns.
    Scenario(
        persona=Persona(
            name="topic-switcher",
            style="chatty",
            system_prompt=(
                "First ask about the weather, then on your next turn abruptly switch "
                "to asking for a simple fact (e.g. the capital of Japan)."
            ),
            scripted_turns=("what is the weather like", "what is the capital of japan"),
        ),
        goal=Goal(
            description="Ask about weather, then switch topics to a factual question.",
            checks=(
                GoalCheck("assistant_responded", "custom", predicate=_assistant_responded),
                GoalCheck("converged", "turn_count_max", target="6"),
            ),
            max_turns=3,
        ),
    ),
]
