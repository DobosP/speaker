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


# A reused LONG-answer prompt. Every barge-in target uses it so there is always
# speech in flight to cut -- this is the fix for the short-response flakiness: a
# short answer's task finishes and every sentence drains from the play queue
# before the VAD accumulates its required barge_in_min_speech_sec (0.2s) of voiced
# audio, by which point _speaking has cleared and the barge watch (gated on
# self._speaking) is no longer armed, so nothing interrupts. The rainbow prompt
# reliably produced a long, interruptible answer in the live run. Barge-in
# grading is INJECT MODE ONLY (the acoustic two-stream path is impossible on the
# reference box's exclusive ALSA hardware).
LONG_ANSWER = "Give me a long, detailed explanation of how rainbows form, step by step."


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
            Turn(LONG_ANSWER, "wait_for_response",
                 "The shared LONG_ANSWER prompt -- always enough speech in flight to cut. NB wait_for_response here waits until idle; the next line barges in -- see driver: a barge_in line speaks over whatever is playing."),
            Turn("Stop.", "barge_in",
                 "Interrupt: playback should halt within a tight budget; no stale sentence after."),
            Turn("What's the capital of France?", "wait_for_response",
                 "Confirms the pipeline recovered and answers normally after the interrupt."),
            Turn(LONG_ANSWER, "wait_for_response",
                 "Another LONG_ANSWER to interrupt with a redirect (re-asked so there is speech to cut)."),
            Turn("Never mind, just tell me a joke.", "barge_in",
                 "Redirect mid-answer: should stop the long answer and tell a joke instead."),
        ),
        validates="runtime._on_barge_in / the engine barge-in gate + epoch staleness suppression. Graded INJECT-MODE ONLY; barge target is the shared LONG_ANSWER.",
        expected_behavior="On 'Stop' the assistant stops talking quickly; on 'just tell me a joke' it abandons the long explanation and tells a joke. No assistant audio continues after a barge-in. Barge-in grade: both barge turns stopped=yes, zero self-interrupts.",
        pass_signals=(
            "Assistant audio stops shortly after each barge-in (measure barge_in -> stop).",
            "No stale rainbow/story sentence plays after the interrupt; the redirect is honored.",
            "Barge-in grade: stops-when-barged 2/2, self_interrupt_count == 0.",
        ),
        failure_modes=(
            "Assistant keeps talking over the user / finishes the long answer anyway.",
            "A stale sentence plays after the stop.",
            "A self-interrupt fires (a barge-in stamp lands on a turn the scenario did not barge).",
        ),
    ),
    Scenario(
        name="barge_in_early_vs_late",
        capability="Barge-in / interrupt (timing depth)",
        goal="A long answer is interrupted EARLY (short 'Stop.') and another is interrupted LATE (deeper into playback, via a longer interrupter); both must halt. Compares stop latency at two depths.",
        turns=(
            Turn(LONG_ANSWER, "wait_for_response",
                 "LONG_ANSWER #1 -- interrupted EARLY with a one-word 'Stop.' that the VAD latches almost immediately."),
            Turn("Stop.", "barge_in",
                 "EARLY interrupt: a single short word; the barge lands soon after the answer starts."),
            Turn(LONG_ANSWER, "wait_for_response",
                 "LONG_ANSWER #2 (re-asked) -- interrupted LATE. 'Late' here means deeper into a long answer, not a precisely scheduled offset: the driver fires a barge the instant the assistant starts speaking, so depth is approximated by a LONGER interrupter line that takes longer to detect+inject."),
            Turn("Hold on a moment, I actually want you to stop talking now please.", "barge_in",
                 "LATE interrupt: a long interrupter line lands deeper into playback than the one-word 'Stop.'. Both should stop; compare stop_ms (the EARLY one is expected sooner, but both must be non-null and the answer must halt)."),
        ),
        validates="runtime._on_barge_in halts at two playback depths; the one-barge-per-run latch fires for each fresh speaking run. INJECT-MODE ONLY; both barge targets are the shared LONG_ANSWER.",
        expected_behavior="Both long answers stop on their barge. The barge-in grade shows stops-when-barged 2/2 and zero self-interrupts; stop_ms is populated for both (the EARLY one typically smaller, but the harness 'late' is approximate -- read it as depth, not a scheduled offset).",
        pass_signals=(
            "Both LONG_ANSWER turns stopped=yes; barge-in grade rate 2/2.",
            "stop_ms populated (non-null) for both barge turns; self_interrupt_count == 0.",
        ),
        failure_modes=(
            "Either long answer finishes anyway (a barge missed its in-flight window).",
            "stop_ms null on a turn that DID stop (BARGE_IN fired but no live stream to abort -- the short-answer race; should not happen with LONG_ANSWER).",
        ),
    ),
    Scenario(
        name="barge_stop_command_vs_new_topic",
        capability="Barge-in / interrupt (stop-word vs redirect)",
        goal="A long answer is halted by a pure STOP-class barge (no follow-on), and a second long answer is halted by a NEW-TOPIC barge that must stop the long answer AND answer the new question.",
        turns=(
            Turn(LONG_ANSWER, "wait_for_response",
                 "LONG_ANSWER #1 -- the target of a stop-class barge."),
            Turn("Stop talking.", "barge_in",
                 "STOP-class barge: should HALT and produce no follow-on answer (the assistant just goes quiet)."),
            Turn(LONG_ANSWER, "wait_for_response",
                 "LONG_ANSWER #2 (re-asked) -- the target of a new-topic barge."),
            Turn("Actually, what's the capital of France?", "barge_in",
                 "NEW-TOPIC barge: should halt the long answer AND answer the fresh question (Paris)."),
        ),
        validates="runtime._on_barge_in stops the in-flight answer; the redirect path then answers the new turn. INJECT-MODE ONLY; both barge targets are the shared LONG_ANSWER.",
        expected_behavior="Both long answers stop on their barge (stops-when-barged 2/2, zero self-interrupts). The stop-word barge yields no further answer; the new-topic barge yields a fresh correct answer (Paris).",
        pass_signals=(
            "Both LONG_ANSWER turns stopped=yes; self_interrupt_count == 0.",
            "After 'Stop talking.' no stale long-answer sentence plays; after 'what's the capital of France?' a Paris answer is produced.",
        ),
        failure_modes=(
            "The long answer continues after either barge.",
            "The new-topic barge stops the answer but never answers the new question (or answers the wrong place).",
        ),
    ),
    Scenario(
        name="barge_multiple_in_one_turn",
        capability="Barge-in / interrupt (latch + re-arm)",
        goal="ONE long answer is hit by TWO consecutive barges. The first must stop it; the second hits an already-idle/short assistant and must be benign -- exercising the one-barge-per-run latch + re-arm.",
        turns=(
            Turn(LONG_ANSWER, "wait_for_response",
                 "ONE LONG_ANSWER, about to be barged twice in a row."),
            Turn("Wait.", "barge_in",
                 "First barge: should STOP the long answer (one barge per speaking run; latch fires here)."),
            Turn("No, stop.", "barge_in",
                 "Second barge in the same turn: the assistant is already idle / on a short reply, so this is expected to be a benign no-op. The grade must NOT penalize a benign second barge, and self_interrupt_count must stay 0."),
        ),
        validates="The engine one-barge-per-run latch (_barge_in_fired_this_run) + its re-arm on the silent->speaking edge; a second barge against an idle assistant is a no-op. INJECT-MODE ONLY; barge target is the shared LONG_ANSWER.",
        expected_behavior="At least the FIRST barge stops the long answer. The second barge does not crash and does not register as a self-interrupt. Barge-in grade: at least 1 of the intended barges stopped, self_interrupt_count == 0.",
        pass_signals=(
            "The first barge stopped the long answer (stopped=yes on its turn).",
            "No crash; self_interrupt_count == 0 (the benign second barge is not a self-interrupt).",
        ),
        failure_modes=(
            "The long answer continues past the first barge.",
            "The second barge triggers a self-interrupt or wedges the pipeline.",
        ),
    ),
    Scenario(
        name="barge_no_barge_control",
        capability="Barge-in / interrupt (NO-BARGE control)",
        goal="The CONTROL: a long answer runs to completion with NO barge at all. The barge-in grade must show ZERO self-interrupts -- this is the regression catch for the self-interruption storm (the assistant barging ITSELF).",
        turns=(
            Turn(LONG_ANSWER, "wait_for_response",
                 "The full LONG_ANSWER, spoken to completion. NO barge is scripted -- if a barge-in stamp lands on this turn it is the assistant interrupting itself."),
            Turn("Thanks, that's all.", "wait_for_response",
                 "A trivial closer; confirms the assistant returned to idle cleanly with no self-interrupt anywhere in the run."),
        ),
        validates="No self-interruption storm: with barge_in_enabled on and the input gate active, the assistant must NOT fire a barge against its own playback. INJECT-MODE ONLY; the answer is the shared LONG_ANSWER.",
        expected_behavior="The long answer plays to completion (not interrupted). The closer is answered. Barge-in grade: n_intended_barges == 0, self_interrupt_count == 0, verdict 'ok' (rate is n/a -- no barge was intended).",
        pass_signals=(
            "self_interrupt_count == 0 across the whole run (no turn is interrupted without an intended barge).",
            "The long answer is NOT marked interrupted; the closer is answered normally.",
        ),
        failure_modes=(
            "The assistant interrupts its own long answer (a self-interrupt -- the storm regression).",
            "Any turn shows interrupted=yes with no preceding barge_in line.",
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
