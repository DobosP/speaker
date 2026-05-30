"""The live test scenarios: scripted spoken interactions that exercise each of
the assistant's current capabilities. Designed to be graded by a human reading
the attributed timeline + latency report the harness produces.

Turn timing (how the driver schedules a user line relative to the assistant):
- ``wait_for_response``: speak, then block until the assistant has finished.
- ``immediately``: speak now without waiting -- used to tack an ADD-ON onto the
  line just spoken, BEFORE the assistant answers (the before-audio merge case).
- ``barge_in``: speak now, OVER the assistant while it is still answering.
- ``pause:<N>``: wait N seconds of silence first, then wait_for_response.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    text: str
    timing: str = "wait_for_response"
    note: str = ""


@dataclass(frozen=True)
class Scenario:
    name: str
    capability: str
    goal: str
    turns: tuple[Turn, ...]
    validates: str = ""
    expected_behavior: str = ""
    pass_signals: tuple[str, ...] = field(default=())
    failure_modes: tuple[str, ...] = field(default=())


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="baseline_latency_single_turn_qa",
        capability="Single-turn Q&A baseline",
        goal="Round-trip latency floor (SPEECH_END -> first assistant audio) + clean attribution on the easy path.",
        turns=(
            Turn("What's the capital of France?", "wait_for_response",
                 "Short decisive question; canonical latency-floor turn. Expect 'Paris'."),
            Turn("How many days are in a week?", "pause:1.5",
                 "1.5s gap so the turns are cleanly separated. Warm-floor turn. Expect 'seven'."),
        ),
        validates="Each question answered exactly once, sequentially, correctly attributed; first_audio_latency recorded per turn.",
        expected_behavior="Two separate, correct, in-order answers (Paris; seven). No merge, no barge-in, no timeout apology, no duplicate answers.",
        pass_signals=(
            "Timeline strictly alternates user(Q1)->assistant(A1)->[~1.5s]->user(Q2)->assistant(A2), zero overlap.",
            "Exactly 2 user + 2 assistant segments, each attributed correctly.",
            "Both turns have a populated first_audio_latency; turn 2 is the warm floor.",
        ),
        failure_modes=(
            "first_audio_latency null (a stage stamp never fired).",
            "Segments overlap or are mis-attributed (assistant TTS captured as a user turn).",
            "Q2 merged into Q1 as a continuation instead of a fresh turn.",
            "Wrong answer or a 'Sorry, that took too long' timeout instead of a real answer.",
        ),
    ),
    Scenario(
        name="context_aggregation_its_population",
        capability="Short-term memory / context aggregation",
        goal="A bare-pronoun follow-up resolves only against the recent-conversation block held in short-term memory.",
        turns=(
            Turn("What's the capital of France?", "wait_for_response",
                 "Establishes the referent (Paris enters the recent-conversation block)."),
            Turn("And what's its population?", "wait_for_response",
                 "'its' has NO in-utterance noun -> resolvable only from the prior turns. Not a continuation marker, so it's a standalone follow-up isolating memory."),
            Turn("Which of those two cities is older?", "wait_for_response",
                 "'those two cities' needs BOTH earlier turns present -> proves multi-turn aggregation, not just last-turn carry. Droppable if runtime is tight."),
        ),
        validates="core/conversation.py recent-context injection lets the model resolve coreference itself with no noun repeated.",
        expected_behavior="T2 answers Paris's population (treats 'its' as Paris/France), does NOT ask 'whose?'. T3 compares the cities discussed and names a concrete older one.",
        pass_signals=(
            "T2 answer is about Paris/France's population, not a clarifying question.",
            "T3 names/compares concrete cities from the prior turns.",
        ),
        failure_modes=(
            "Assistant asks 'population of what?' / answers a generic 'I need more context'.",
            "Answers about a wrong place; T3 cannot resolve 'those two cities'.",
        ),
    ),
    Scenario(
        name="addon_continuation_merge_and_queue",
        capability="ADD-ON / continuation",
        goal="An add-on tacked onto a still-running turn yields ONE coherent answer (merge before audio, queue-behind after audio), never two racing answers.",
        turns=(
            Turn("What's the weather like in Paris today?", "immediately",
                 "Ask, then DON'T wait -- the add-on lands while this is still being processed."),
            Turn("And also tomorrow.", "immediately",
                 "Before-audio add-on: should MERGE into one answer covering today AND tomorrow."),
            Turn("Tell me about the Eiffel Tower.", "pause:3",
                 "Fresh longer turn after a clean gap; sets up the after-audio case."),
            Turn("Oh wait, make it shorter.", "barge_in",
                 "After-audio add-on while it's speaking: answered as a continuation/adjustment, not a competing answer."),
        ),
        validates="supervisor._maybe_continue: before-audio merge into one fresh turn; after-audio context-carrying continuation queued behind the speaking turn.",
        expected_behavior="The 'and also tomorrow' add-on produces a single answer covering both days (not two overlapping weather answers). 'Make it shorter' is handled as one adjustment, not a second racing reply.",
        pass_signals=(
            "Exactly ONE assistant answer for the weather pair, mentioning today AND tomorrow.",
            "No two assistant audio segments overlapping for a single logical request.",
        ),
        failure_modes=(
            "Two overlapping/sequential weather answers (the racing-cold-task bug).",
            "The add-on is dropped entirely (only today answered).",
        ),
    ),
    Scenario(
        name="self_awareness_enumerate_do_decline",
        capability="Capability catalog + self-aware model",
        goal="The assistant truthfully enumerates its real, deliverable skills and does NOT confabulate silent/mode-gated abilities.",
        turns=(
            Turn("Hey, what are you, and what can you actually do for me?", "wait_for_response",
                 "Self-description: should list real skills (answer from knowledge, research), not claim it can take notes/run commands in a plain turn."),
            Turn("Okay, do that research thing then: which is better for sleep, magnesium glycinate or melatonin?", "wait_for_response",
                 "Exercise a skill it HAS (research/answer)."),
            Turn("Great. Now take a note for me and remember it: buy oat milk tomorrow.", "wait_for_response",
                 "A silent/mode-gated 'skill' -- it must NOT falsely claim it noted/stored it."),
            Turn("And can you look that up online for me to double check the price?", "wait_for_response",
                 "Web is off by default -> it should say it can't search the web (no false 'I searched')."),
        ),
        validates="The capability-aware system prompt enumerates user-facing skills; silent/mode-gated capabilities are user_facing=False; web gated on availability.",
        expected_behavior="Lists answer/research skills; answers the research question; does NOT claim 'I've saved that note'; says it can't look things up online (web disabled).",
        pass_signals=(
            "Self-description names real skills and omits dictation/notes/commands as plain-turn abilities.",
            "Does not say it stored the note; does not claim it searched the web.",
        ),
        failure_modes=(
            "Confabulates: 'Got it, I've noted that down' (nothing stored).",
            "Claims it searched online / found a price (no web).",
        ),
    ),
    Scenario(
        name="smart_endpoint_hold_vs_crisp",
        capability="Smart endpoint (semantic turn-completion, EXPERIMENTAL/default-OFF)",
        goal="The assistant does not cut off a mid-thought pause, yet endpoints crisply on a complete utterance. Measures endpoint latency.",
        turns=(
            Turn("What's two plus two?", "wait_for_response",
                 "Crisp complete utterance -> fast endpoint. Endpoint-latency baseline."),
            Turn("What is the ...", "pause:1",
                 "Deliberately trailing -> a 1s pause follows; the assistant must NOT commit here (with smart endpoint on, the mid-phrase '...the' holds)."),
            Turn("capital of France?", "immediately",
                 "Completes the held utterance; the full 'what is the capital of France' should be captured as ONE final."),
            Turn("Thanks, that's all.", "wait_for_response",
                 "Polite closer; confirms it returns to idle."),
        ),
        validates="core/endpointing.py adaptive policy (only effective when sherpa.endpoint_enabled; otherwise this measures the fixed-timer endpoint latency).",
        expected_behavior="With endpoint on: the paused 'what is the ... capital of France' is captured whole and answered once (Paris). With endpoint off: it may split into two finals -- note which. Measure endpoint_latency on the crisp turn.",
        pass_signals=(
            "The completed 'what is the capital of France' yields a single correct answer (Paris).",
            "endpoint_latency recorded for the crisp '2+2' turn.",
        ),
        failure_modes=(
            "The pause splits the utterance: 'what is the' answered as its own (failed) turn then 'capital of France' separately.",
        ),
    ),
    Scenario(
        name="barge_in_interrupt_stop",
        capability="Barge-in / interrupt",
        goal="The assistant halts mid-answer promptly on barge-in and never talks over the user or speaks a stale sentence after the stop.",
        turns=(
            Turn("Tell me the full story of the three little pigs, with lots of detail.", "wait_for_response",
                 "A long answer to interrupt. NB wait_for_response here waits until idle; the next line barges in -- see driver: a barge_in line speaks over whatever is playing."),
            Turn("Stop.", "barge_in",
                 "Interrupt: playback should halt within a tight budget; no stale sentence after."),
            Turn("What's the capital of France?", "wait_for_response",
                 "Confirms the pipeline recovered and answers normally after the interrupt."),
            Turn("Actually, give me a long explanation of how rainbows form.", "wait_for_response",
                 "Another long answer to interrupt with a redirect."),
            Turn("Never mind, just tell me a joke.", "barge_in",
                 "Redirect mid-answer: should stop the rainbow answer and tell a joke instead."),
        ),
        validates="runtime._on_barge_in / the engine barge-in gate + epoch staleness suppression.",
        expected_behavior="On 'Stop' the assistant stops talking quickly; on 'just tell me a joke' it abandons the rainbow explanation and tells a joke. No assistant audio continues after a barge-in.",
        pass_signals=(
            "Assistant audio stops shortly after each barge-in (measure barge_in -> stop).",
            "No stale rainbow/story sentence plays after the interrupt; the redirect is honored.",
        ),
        failure_modes=(
            "Assistant keeps talking over the user / finishes the long answer anyway.",
            "A stale sentence plays after the stop.",
        ),
    ),
    Scenario(
        name="never_stuck_heavy_then_recover",
        capability="Never-stuck controller",
        goal="A heavy multi-step turn completes within its deadline or recovers with the spoken apology; the controller never wedges across a long run.",
        turns=(
            Turn("Research the main causes of the 1929 stock market crash and give me three of them.", "wait_for_response",
                 "Heavy/multi-step turn. Should either answer within the deadline or speak 'Sorry, that took too long'."),
            Turn("Okay, and now just tell me what two plus two is.", "wait_for_response",
                 "Proves the controller returned to idle and answers a trivial turn right after."),
            Turn("Compare the economies of Japan and Germany in two short points.", "wait_for_response",
                 "Another moderately heavy turn; keep the run going."),
            Turn("Thanks, what time of day works best to take a short walk?", "wait_for_response",
                 "Final light turn; the assistant should still be responsive, never permanently silent."),
        ),
        validates="supervisor.reap_overdue_tasks (per-mode wall-clock deadline) + the spoken timeout apology + return-to-idle.",
        expected_behavior="No turn leaves dead air: each either answers or apologizes. Every turn after a heavy one is answered promptly (the controller is never wedged).",
        pass_signals=(
            "Every user turn gets either a real answer or the 'Sorry, that took too long' apology -- never silence.",
            "The trivial follow-ups (2+2, the walk question) are answered promptly after the heavy turns.",
        ),
        failure_modes=(
            "Dead air on a turn (no assistant audio at all).",
            "The pipeline wedges: a later turn never gets answered.",
        ),
    ),
)


def by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)
