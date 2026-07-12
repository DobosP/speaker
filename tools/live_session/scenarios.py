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
    # Response-quality grading (graded by report.response_score against the
    # assistant's ANSWER, not the STT of the user line). ``expect`` lists the
    # CONCEPTS the answer should contain; each item may offer alternatives with a
    # "|" (e.g. "seven|7") and matches if ANY alternative appears. The turn's
    # response score is the fraction of expect items satisfied (1.0 when expect is
    # empty -- nothing to check). ``forbid`` lists substrings the answer must NOT
    # contain -- the honesty probes (e.g. a note/reminder/web claim the assistant
    # cannot actually fulfil); any hit flags the turn. Both default empty, so every
    # existing Turn is unchanged and ungraded for response quality.
    expect: tuple[str, ...] = field(default=())
    forbid: tuple[str, ...] = field(default=())


# Reusable LONG-answer prompts. Every barge-in scenario uses at least one so
# there is always speech in flight to cut. This fixes short-response flakiness: a
# short answer's task finishes and every sentence drains from the play queue
# before the VAD accumulates its required voiced audio (production: 0.2s;
# explicit echo-free inject profile: 0.1s), by which point _speaking has cleared
# and the barge watch (gated on
# self._speaking) is no longer armed, so nothing interrupts. The rainbow prompt
# reliably produced a long, interruptible answer in the live run. Barge-in
# grading is INJECT MODE ONLY and deliberately opens no physical audio hardware.
LONG_ANSWER = "Give me a long, detailed explanation of how rainbows form, step by step."
# Scenarios with two independent interrupts deliberately use a distinct second
# target. EchoLLM speaks the prompt itself; repeating LONG_ANSWER shortly after
# that playback correctly trips the runtime's own-echo guard and leaves no second
# answer to interrupt. The alternate isolates barge behavior from that unrelated
# guard while remaining long enough for a real in-flight cut.
LONG_ANSWER_ALT = (
    "Describe the complete lifecycle of a volcano, including magma buildup, "
    "eruption, and recovery, with plenty of detail."
)


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
                 "Short decisive question; canonical latency-floor turn. Expect 'Paris'.",
                 expect=("paris",)),
            Turn("How many days are in a week?", "pause:1.5",
                 "1.5s gap so the turns are cleanly separated. Warm-floor turn. Expect 'seven'.",
                 expect=("seven|7",)),
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
                 "Establishes the referent (Paris enters the recent-conversation block).",
                 expect=("paris",)),
            Turn("And what's its population?", "wait_for_response",
                 "'its' has NO in-utterance noun -> resolvable only from the prior turns. Not a continuation marker, so it's a standalone follow-up isolating memory.",
                 expect=("million",), forbid=("population of what|whose population|need more context|which city|what do you mean",)),
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
                 "A silent/mode-gated 'skill' -- it must NOT falsely claim it noted/stored it.",
                 forbid=("i've saved|i saved|i've noted|noted that down|added to your|i'll remember|saved that note|got it i've|written that down",)),
            Turn("And can you look that up online for me to double check the price?", "wait_for_response",
                 "Web is off by default -> it should say it can't search the web (no false 'I searched').",
                 expect=("can't|cannot|can not|unable|don't have|do not have|not able|no access",),
                 forbid=("i searched|i looked it up|i found the price|according to my search|the price is|i checked online",)),
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
                 "Crisp complete utterance -> fast endpoint. Endpoint-latency baseline.",
                 expect=("four|4",)),
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
            Turn(LONG_ANSWER_ALT, "wait_for_response",
                 "A distinct long answer to interrupt with a redirect (distinct so EchoLLM's own-echo guard cannot consume a repeated prompt)."),
            Turn("Never mind, just tell me a joke.", "barge_in",
                 "Redirect mid-answer: should stop the long answer and tell a joke instead."),
        ),
        validates="runtime._on_barge_in / the engine barge-in gate + epoch staleness suppression. Graded INJECT-MODE ONLY; barge target is the shared LONG_ANSWER.",
        expected_behavior="On 'Stop' the assistant stops talking quickly; on 'just tell me a joke' it abandons the long explanation and tells a joke. No assistant audio continues after a barge-in. Barge-in grade: both barge turns stopped=yes, zero self-interrupts.",
        pass_signals=(
            "The playback FIFO is cut after each detector fire (stop_ms); physical onset-to-stop remains a live measurement.",
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
        goal="A long answer is interrupted EARLY (short 'Stop.') and another is interrupted LATE (deeper into playback, via a longer interrupter); both must halt.",
        turns=(
            Turn(LONG_ANSWER, "wait_for_response",
                 "LONG_ANSWER #1 -- interrupted EARLY with a one-word 'Stop.' that the VAD latches almost immediately."),
            Turn("Stop.", "barge_in",
                 "EARLY interrupt: a single short word; the barge lands soon after the answer starts."),
            Turn(LONG_ANSWER_ALT, "wait_for_response",
                 "Distinct long-answer target #2 -- interrupted LATE. 'Late' here means deeper into a long answer, not a precisely scheduled offset: the driver fires a barge the instant the assistant starts speaking, so depth is approximated by a LONGER interrupter line that takes longer to detect+inject."),
            Turn("Hold on a moment, I actually want you to stop talking now please.", "barge_in",
                 "LATE interrupt: a long interrupter line lands deeper into playback than the one-word 'Stop.'. Both should stop; stop_ms must be non-null for each and measures only detector fire to FIFO cut."),
        ),
        validates="runtime._on_barge_in halts at two playback depths; the one-barge-per-run latch fires for each fresh speaking run. INJECT-MODE ONLY; both targets are long, distinct prompts.",
        expected_behavior="Both long answers stop on their barge. The barge-in grade shows stops-when-barged 2/2 and zero self-interrupts; stop_ms is populated for both. The harness 'late' is approximate playback depth, not a scheduled offset or an onset-to-stop measurement.",
        pass_signals=(
            "Both long-answer targets stopped=yes; barge-in grade rate 2/2.",
            "stop_ms populated (non-null) for both barge turns; self_interrupt_count == 0.",
        ),
        failure_modes=(
            "Either long answer finishes anyway (a barge missed its in-flight window).",
            "stop_ms null on a turn that DID stop (BARGE_IN fired but no live FIFO to cut -- the short-answer race; should not happen with a long target).",
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
            Turn(LONG_ANSWER_ALT, "wait_for_response",
                 "Distinct long-answer target #2 -- the target of a new-topic barge."),
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
                 "Proves the controller returned to idle and answers a trivial turn right after.",
                 expect=("four|4",)),
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
    Scenario(
        name='realistic_morning_planning',
        capability='Multi-turn daily-life planning: live-data deflection, factual recall, arithmetic, coreference, and capability honesty',
        goal='Exercise a realistic morning-planning conversation end-to-end: over-the-air STT on natural spoken lines, per-turn latency, and answer quality across a weather/time deflection, a quick general-knowledge fact, a follow-up that depends on coreference, a simple spoken-number arithmetic turn, and one honesty probe (set a reminder) that the assistant must NOT falsely claim to perform.',
        turns=(
            Turn("Good morning, what's the weather looking like outside today?", 'wait_for_response',
                 'Opener. Assistant has no live weather feed, so it should deflect or caveat rather than state a real forecast. Open-ended -> graded on STT + latency only, expect empty. Sets the morning-planning frame.'),
            Turn('Got it. And what time is it right now?', 'wait_for_response',
                 "Topic stays on planning; 'right now' is a live-clock ask the assistant can't truly answer. No reliably-checkable concept -> expect empty; graded on STT + latency. ('Got it' coreferences the prior weather reply.)"),
            Turn("While I get ready, what's the largest planet in our solar system?", 'pause:1.5',
                 'Quick curiosity fact, famous and unambiguous. 1.5s gap separates it cleanly from the time turn. The MiniCPM fast tier should say Jupiter.', expect=('jupiter',)),
            Turn('Interesting. How many moons does it have, roughly?', 'wait_for_response',
                 "Follow-up: 'it' must resolve to Jupiter from the prior turn -> tests coreference / short-term context. Exact moon count is disputed, so grade only that the answer echoes 'moon(s)'; the bound is fair for a 4b model.", expect=('moon',)),
            Turn("Okay back to planning. If I leave in twenty minutes, what time will I get there if it's a half hour drive?", 'pause:1',
                 "Simple spoken-number arithmetic: 20 + 30 = 50 minutes from now. Spoken as words. The answer should surface 'fifty' minutes (or roughly 50 minutes from now). Robust for 4b.", expect=('fifty|50',)),
            Turn('Can you set a reminder to leave the house in twenty minutes?', 'wait_for_response',
                 "HONESTY PROBE. The assistant cannot set reminders. It must decline / say it can't, NOT fabricate a confirmation. forbid lists the false-claim phrasings; expect empty (a correct decline has no required keyword).", forbid=("i've set|i set|reminder is set|i'll remind you|i will remind you|i've added|added a reminder|scheduled a reminder|you'll be reminded",)),
            Turn("No worries. What's a good quick breakfast I could make in five minutes?", 'wait_for_response',
                 "Topic shift back to planning, open-ended/opinion -> expect empty, graded on STT + latency. 'No worries' coreferences the prior decline, keeping it conversational."),
            Turn("Thanks, that's everything for now.", 'wait_for_response',
                 'Polite closer; confirms the pipeline returns to idle cleanly after the multi-turn run. expect empty.'),
        ),
        validates="Over-the-air STT accuracy on natural conversational lines; per-turn latency (SPEECH_END -> first assistant audio); core/conversation.py short-term context for the 'it' coreference (Jupiter -> its moons); spoken-number arithmetic in the LLM tier; and the capability-honesty path (no live weather/clock fabrication; truthful decline of the unsupported reminder skill).",
        expected_behavior="Eight in-order, correctly attributed turns with no merges or self-interrupts. T1/T2 deflect or caveat the live weather/time asks instead of inventing data. T3 answers Jupiter. T4 resolves 'it' to Jupiter and talks about its moons. T5 computes ~50 minutes (20 + 30). T6 truthfully says it cannot set a reminder and does NOT claim to have set one. T7 offers a quick breakfast idea. T8 closes cleanly to idle. Every turn has a populated first_audio_latency and a clean STT transcript.",
        pass_signals=(
            'Timeline strictly alternates user->assistant for all 8 turns, zero overlap, each segment attributed correctly.',
            "T3 answer names Jupiter; T4 answer is about Jupiter's moons (coreference resolved, not a 'which planet?' clarifying question).",
            'T5 answer surfaces fifty / ~50 minutes (correct 20+30 arithmetic).',
            'T6 declines the reminder and contains none of the forbidden false-confirmation phrasings.',
            'T1/T2 do not state a fabricated specific temperature or wall-clock time as fact.',
            'Every turn has a non-null first_audio_latency; STT transcript closely matches each spoken line.',
        ),
        failure_modes=(
            'Fabricates a specific live forecast or exact current clock time as if it had real data (T1/T2).',
            "T4 fails to resolve 'it' and asks 'which planet/object do you mean?' instead of answering about Jupiter's moons.",
            'T5 gives wrong arithmetic (not ~50 minutes) or refuses to compute.',
            "T6 falsely confirms the reminder ('I've set a reminder', 'I'll remind you') despite having no such capability.",
            'Turns merge, overlap, or are mis-attributed (assistant TTS captured as a user turn); a turn yields dead air or a timeout apology.',
            'STT mangles a spoken line badly enough that the wrong question is answered.',
        ),
    ),
    Scenario(
        name='realistic_cooking_help',
        capability='Hands-free multi-turn kitchen Q&A (substitution, conversion, timing, coreference follow-up, temperature)',
        goal="A person cooking hands-free asks a natural chain of kitchen questions; the assistant recognizes the speech over the air, answers each well-known cooking fact correctly and quickly, and resolves a 'that'/'it' follow-up against the prior turn.",
        turns=(
            Turn("I'm out of buttermilk for these pancakes, what can I use instead?", 'wait_for_response',
                 'Opening substitution question, hands busy at the stove. Canonical answer is milk plus lemon juice or vinegar to sour it. Accept either acid. Latency floor for the run.', expect=('milk', 'lemon|vinegar|acid')),
            Turn('Okay good. How many teaspoons are in one tablespoon?', 'wait_for_response',
                 'Unambiguous kitchen conversion, exactly three teaspoons per tablespoon. No ingredient dependence, so the answer is deterministic and safe to grade.', expect=('three|3',)),
            Turn('And how many grams are in one kilogram of flour?', 'pause:1.5',
                 'Clean metric conversion spoken in words; 1.5s gap separates it from the previous conversion. A thousand grams in a kilogram regardless of the flour. Deterministic.', expect=('thousand|1000|one thousand',)),
            Turn("I'm also boiling some eggs, how long for a hard boiled one?", 'wait_for_response',
                 'Classic timing question. Hard-boiled is about nine to twelve minutes; accept any of those numbers plus the word minute. Topic shift from baking to eggs.', expect=('minute', 'ten|twelve|nine|10|12|9')),
            Turn('Got it. And what about a soft boiled one instead?', 'wait_for_response',
                 'Follow-up that relies on the prior egg context to know we still mean boiling an egg. Soft-boiled is roughly four to seven minutes; accept that range plus minute.', expect=('minute', 'six|five|four|seven|6|5|4|7')),
            Turn('Is it safe to eat that if the yolk is still runny?', 'wait_for_response',
                 "Bare-pronoun coreference: 'that' = the soft-boiled egg from the previous turn. Open-ended safety opinion, so expect is empty; graded for STT accuracy, latency, and whether it resolves 'that' to the egg rather than asking what."),
            Turn('Last thing, what temperature does water boil at in Celsius?', 'pause:1',
                 'Well-known physical fact closing the run; water boils at one hundred degrees Celsius at sea level. A 1s pause sets it apart as a fresh closing turn.', expect=('hundred|100',)),
            Turn("Perfect, thanks, that's everything.", 'wait_for_response',
                 'Polite closer; confirms the assistant returns to idle cleanly. No checkable fact, graded for STT and return-to-idle only.'),
        ),
        validates="Over-the-air STT on conversational kitchen phrasing; per-turn first-audio latency under a busy hands-free chain; MiniCPM factual accuracy on common cooking knowledge (substitution, unit conversion, egg timing, boiling point); core/conversation.py recent-context injection resolving the bare pronoun 'that'/'it' against the immediately prior egg turns without the noun being repeated.",
        expected_behavior="Eight clean, in-order turns each answered exactly once. Buttermilk substitution names milk soured with lemon juice or vinegar. Conversions are exact: three teaspoons per tablespoon, one thousand grams per kilogram. Hard-boiled egg is given as roughly nine to twelve minutes, soft-boiled as roughly four to seven minutes. The 'is it safe to eat that' turn treats 'that' as the soft-boiled egg (a runny-yolk safety answer) and does not ask 'eat what?'. Water boils at one hundred degrees Celsius. The closer returns the assistant to idle. No merged turns, no barge-in, no timeout apology, no duplicate answers.",
        pass_signals=(
            'Timeline strictly alternates user->assistant for all eight turns, zero overlap, each correctly attributed.',
            'Substitution turn mentions milk plus an acid (lemon juice or vinegar).',
            "Conversions are exact: 'three' teaspoons and 'a thousand'/'1000' grams.",
            "Hard-boiled answer contains 'minute' and a number in the nine-to-twelve range; soft-boiled contains 'minute' and a number in the four-to-seven range.",
            "The 'is it safe to eat that' turn resolves 'that' to the egg and gives a runny-yolk safety answer rather than a clarifying question.",
            "Water-boiling turn contains 'one hundred' / '100' (Celsius).",
            'Every turn has a populated first_audio_latency.',
        ),
        failure_modes=(
            "STT mangles a spoken number or unit (e.g. hears 'three teaspoons' as 'tree' or drops 'kilogram'), causing a wrong-fact grade that is really a recognition miss.",
            "Coreference fails: the 'eat that' turn asks 'eat what?' or answers about an unrelated food instead of the soft-boiled egg.",
            'Wrong conversion (not three teaspoons, not a thousand grams) or an egg time far outside the accepted range.',
            'Two turns merge into one answer, or a turn is answered twice / overlaps.',
            "A 'Sorry, that took too long' timeout or dead air on any turn.",
            'first_audio_latency null on a turn (a stage stamp never fired).',
        ),
    ),
    Scenario(
        name='realistic_curiosity_chat',
        capability='General-knowledge Q&A with short-term-memory coreference (a curious learner)',
        goal="Exercise multi-turn coreference resolution over the recent-conversation block: establish a country, then resolve bare pronouns ('its capital', 'the language they speak there') against it, shift to a second country, and compare the two — all while grading over-the-air STT, per-turn latency, and whether each answer contains the unambiguous fact.",
        turns=(
            Turn('Tell me one interesting thing about France.', 'wait_for_response',
                 "Establishes France as the referent entering the recent-conversation block. Open-ended (no single checkable fact) but the answer must echo 'France'; graded for STT + latency.", expect=('france',)),
            Turn('What is its capital city?', 'wait_for_response',
                 "'its' has no in-utterance noun -> resolvable only from the prior France turn. Canonical coreference probe; answer must be Paris.", expect=('paris',)),
            Turn('And what language do they speak there?', 'wait_for_response',
                 "'they'/'there' both corefer to France/Paris from short-term memory; unambiguous fact (French).", expect=('french',)),
            Turn('Roughly how many people live in that city?', 'pause:1.5',
                 "'that city' corefers to Paris. Population of Paris is ~2 million (city proper) / ~10-12 million metro; accept 'million' plus a number-word. 1.5s gap separates this as a fresh follow-up, not a continuation.", expect=('million|2|two',)),
            Turn('Now tell me the capital of Japan instead.', 'wait_for_response',
                 'Topic shift to a second country; unambiguous fact (Tokyo). Sets up the cross-turn comparison.', expect=('tokyo',)),
            Turn('Which of those two capitals has more people?', 'wait_for_response',
                 "'those two capitals' needs BOTH Paris and Tokyo present -> proves multi-turn aggregation, not just last-turn carry. Tokyo metro (~37M) is far larger; answer should name Tokyo.", expect=('tokyo',)),
            Turn('Can you remember Tokyo for me so I can ask later?', 'wait_for_response',
                 'Honesty probe: the assistant has no persistent note/memory store across sessions. It must NOT falsely claim it saved/will-remember Tokyo. Open answer, so expect is empty; graded on forbid + STT + latency.', forbid=("i've saved|i saved|i'll remember|noted that|added to your|stored it|i will remember",)),
            Turn('Okay, what is the currency used in Japan?', 'wait_for_response',
                 'Returns to Japan via short-term memory after the honesty probe; unambiguous fact (yen). Confirms the pipeline recovered and context survived the off-topic turn.', expect=('yen',)),
        ),
        validates="core/conversation.py recent-context injection (bare-pronoun + 'those two' coreference resolved by the model with no noun repeated); the MiniCPM answering tier on unambiguous general-knowledge facts; over-the-air sherpa STT accuracy on natural conversational phrasing; per-turn first-audio latency; and the capability-honesty guard (no confabulated note/memory store).",
        expected_behavior="A natural learning chat resolves cleanly turn by turn: France is described; 'its capital' answers Paris; 'they speak there' answers French; 'that city' population is a Paris figure in the millions; the topic shifts to Tokyo; 'those two capitals' is correctly compared (Tokyo is larger); the 'remember Tokyo' turn is answered honestly without claiming to have stored anything; and the final currency turn answers yen. No turn asks 'capital of what?' / 'remember what?'; no dead air; each answer is correctly attributed and arrives within a normal first-audio latency.",
        pass_signals=(
            'Timeline strictly alternates user->assistant across all 8 turns, zero overlap, each segment correctly attributed.',
            'Coreference resolves: T2 answers Paris (not a clarifying question), T3 answers French, T4 gives a Paris population in the millions, T6 compares Paris vs Tokyo and names Tokyo as larger.',
            'Unambiguous facts present: Paris (T2), French (T3), Tokyo (T5), Tokyo-larger (T6), yen (T8).',
            'T7 (remember Tokyo) does NOT claim it saved/will-remember anything (forbid clean).',
            'Every turn has a populated first_audio_latency; STT transcripts of the spoken lines match the intended text closely (low WER over-the-air).',
        ),
        failure_modes=(
            "A coreference turn asks for clarification ('capital of what?', 'which city?', 'remember what?') instead of resolving from short-term memory.",
            "T6 cannot resolve 'those two capitals' or names the wrong/ smaller city as larger.",
            "T7 confabulates persistent storage ('I've saved Tokyo' / 'I'll remember that') — a false capability claim.",
            "Wrong famous fact (capital not Paris/Tokyo, language not French, currency not yen) or a 'took too long' timeout apology in place of an answer.",
            'Segments overlap or assistant TTS is mis-attributed as a user turn; any turn yields dead air (null first_audio_latency).',
            'High over-the-air WER garbles a spoken line so the wrong question is answered (an STT failure, surfaced separately from the answer grade).',
        ),
    ),
    Scenario(
        name='realistic_quickfire_assist',
        capability='Fast turn-taking under terse commands and quick questions (latency floor)',
        goal='Stress the round-trip latency floor with short, snappy turns: rapid-fire factual one-liners, a tiny math question, a spelling/unit question, and one open-ended quick ask, while staying correctly attributed turn-to-turn. A multitasking person fires terse asks and expects near-instant, correct replies.',
        turns=(
            Turn("What's the capital of Japan?", 'wait_for_response',
                 'Cold-start latency-floor turn. Canonical fact, single salient answer the reply must echo: Tokyo.', expect=('tokyo',)),
            Turn("What's seven times eight?", 'wait_for_response',
                 'Tiny math, terse. Warm-floor turn. Answer must contain 56 in digits or words.', expect=('56|fifty-six|fifty six',)),
            Turn('How do you spell restaurant?', 'wait_for_response',
                 'Spelling question; the model spells it out letter-by-letter or restates the word. Fair: accept either the spelled form or the word echoed.', expect=('r-e-s-t-a-u-r-a-n-t|restaurant',)),
            Turn('How many grams are in a kilogram?', 'pause:1',
                 'Unit conversion, terse. 1s gap to separate cleanly from the spelling turn. Answer must contain a thousand.', expect=('1000|one thousand|thousand',)),
            Turn('Who painted the Mona Lisa?', 'wait_for_response',
                 'Snappy factual one-liner; salient answer Leonardo da Vinci. Topic shift away from units.', expect=('da vinci|leonardo|vinci',)),
            Turn('Where is it on display right now?', 'wait_for_response',
                 "Coreference: 'it' = the Mona Lisa from the prior turn, no noun repeated. Tests short-term context resolution. Answer should name the Louvre or Paris.", expect=('louvre|paris',)),
            Turn('Give me a quick fun fact about octopuses.', 'pause:1',
                 'Open-ended quick ask; no single checkable fact, so expect is empty (still graded on STT + latency). 1s gap keeps the quickfire rhythm natural.'),
            Turn("Nice. What's the boiling point of water in Celsius?", 'wait_for_response',
                 'Final terse factual closer; answer must contain 100. Confirms the pipeline is still snappy after the open-ended turn.', expect=('100|hundred',)),
        ),
        validates='core round-trip latency path (speech_end -> asr_final -> llm_first_token -> tts_first_audio) under short terse turns; per-turn first_audio_latency / warm-floor measurement; over-the-air STT on short utterances; correct sequential attribution with no merge; one coreference turn exercises core/conversation.py recent-context injection.',
        expected_behavior="Each short question is answered exactly once, in order, correctly attributed, with a populated per-turn first_audio_latency. Tokyo; 56; restaurant spelled or restated; a thousand grams; Leonardo da Vinci; Louvre/Paris for the 'it' follow-up; a plausible octopus fact; 100 degrees Celsius. No turns merge, no timeout apologies, no duplicate or overlapping answers. Warm turns should sit at or near the latency floor.",
        pass_signals=(
            'Timeline strictly alternates user(Qn) -> assistant(An) with zero overlap for all eight turns; each correctly attributed.',
            'Every turn has a populated first_audio_latency; turns after the first sit near the warm floor (terse asks should not inflate latency).',
            'Factual turns hit their expect concepts: Tokyo, 56, the restaurant spelling/word, a thousand, Leonardo da Vinci, Louvre/Paris, 100.',
            "The 'Where is it on display' turn resolves 'it' to the Mona Lisa without asking 'what?' (coreference from the prior turn).",
            'The open-ended octopus turn is answered with a plausible fact (graded only on clean STT + latency, no expect).',
        ),
        failure_modes=(
            'first_audio_latency null on a turn (a stage stamp never fired).',
            'Short adjacent turns merge (e.g. the math and spelling turns captured as one final) instead of separate turns.',
            "Over-the-air STT mangles a terse utterance (e.g. 'seven times eight' misheard) so the answer is off-topic.",
            "Coreference fails: 'Where is it on display' is answered with 'display what?' or about a wrong subject.",
            "A 'Sorry, that took too long' timeout or a duplicate/overlapping answer appears on a turn that should be at the latency floor.",
            'Wrong fact returned (not Tokyo / not 56 / not 100 / wrong painter).',
        ),
    ),
    Scenario(
        name='latency_profile_mixed',
        capability='Latency profile (mixed turns)',
        goal='Build a per-turn LATENCY DISTRIBUTION across deliberately varied input lengths (1-word factual through multi-clause), while also grading STT accuracy and answer correctness. Each turn is INDEPENDENT (no coreference), so the only variable driving latency is the input/output shape, not conversational state.',
        turns=(
            Turn('What is the capital of Japan?', 'wait_for_response',
                 "Very short factual, 1-clause. Canonical warm latency floor. Almost certainly 'Tokyo'.", expect=('tokyo',)),
            Turn('What is two plus two?', 'pause:1',
                 "Trivial math, short input + short output. Floor-class latency. Answer 'four'.", expect=('four|4',)),
            Turn('What is the square root of sixty four?', 'wait_for_response',
                 "Simple math, slightly longer spoken input. Answer 'eight'.", expect=('eight|8',)),
            Turn('What happens to water when it gets very cold?', 'pause:1',
                 'Short definition/science. Well-known: water freezes into ice at zero Celsius.', expect=('freeze|freezes|ice|solid',)),
            Turn('Name three primary colors.', 'wait_for_response',
                 'Short-list request, longer output. Standard art primaries red, blue, yellow.', expect=('red', 'blue', 'yellow')),
            Turn('Which planet is known as the red planet?', 'wait_for_response',
                 'Short factual with a salient keyword the answer must echo: Mars.', expect=('mars',)),
            Turn('Is the sun a planet, yes or no?', 'pause:1',
                 "Explicit yes/no. The sun is a star, so the answer is 'no'.", expect=('no',)),
            Turn('Can you spell the word Mars for me?', 'wait_for_response',
                 'Spelling turn: answer must echo the letters M A R S. Alternatives cover spacing/hyphen phrasing.', expect=('m-a-r-s|m a r s|m, a, r, s',)),
            Turn('What is the largest ocean on Earth?', 'wait_for_response',
                 'Short factual, salient keyword. Answer: the Pacific Ocean.', expect=('pacific',)),
            Turn('In one sentence, how do plants make their own food using sunlight, water, and air?', 'wait_for_response',
                 'Multi-clause longer input (more ASR + more generation) - the high end of the latency distribution. Answer names photosynthesis.', expect=('photosynthesis',)),
            Turn('How many players are on a baseball team on the field?', 'pause:1',
                 'Short factual counting. A baseball team fields nine players. Closes the distribution with another floor-class turn.', expect=('nine|9',)),
        ),
        validates='core/metrics.py per-turn stage timings (speech_end -> asr_final -> llm_first_token -> tts_first_audio) across varied input lengths; sherpa over-the-air ASR accuracy per turn; MiniCPM answer correctness on famous facts. Independent turns isolate latency-vs-input-length from conversation/context cost.',
        expected_behavior='Each turn is answered exactly once, correctly, with a populated first_audio_latency. Short factual turns (capital, yes/no) should hit the warm latency floor; the multi-clause and short-list turns cost more ASR + more generation and sit higher in the distribution. The spelling turn must echo the requested letters. No merges, no barge-ins, no timeouts, no clarifying questions.',
        pass_signals=(
            'Every one of the 11 turns has a non-null first_audio_latency, yielding a full per-turn distribution.',
            "Each answer contains its 'expect' concept (Tokyo; four; eight; freezes/zero; red/blue/yellow; Mars; no; M-A-R-S; Pacific; photosynthesis; nine).",
            'Timeline is 11 cleanly separated user->assistant pairs, zero overlap, each correctly attributed.',
            'Latency visibly tracks input length: 1-word/short factual turns at the floor, multi-clause/list turns higher.',
        ),
        failure_modes=(
            'first_audio_latency null on any turn (a stage stamp never fired).',
            'An answer omits its expect concept or answers a different question (e.g. names a non-Pacific ocean).',
            'Over-the-air STT mangles a turn so the assistant answers the wrong question (track which turn).',
            "Two turns merge or an answer is duplicated; a 'Sorry, that took too long' timeout replaces a real answer.",
            'The spelling turn does not spell out M-A-R-S letter by letter.',
        ),
    ),
    Scenario(
        name="capability_latency_profile",
        capability="Per-capability / per-tier latency (fast vs main vs research)",
        goal="Measure first-audio latency across the shipped two-model setup (fast=MiniCPM5-1B Q8, main=gemma3:12b): short factual -> FAST; reasoning/compare/long-form -> MAIN; research -> MAIN + ReAct planner. Shows where each capability's latency lands and which tier dominates the tail.",
        turns=(
            Turn("What's the capital of Japan?", "wait_for_response",
                 "FAST tier: 5-word literal factual, no complexity/generation markers -> router score 0 < 0.3 -> MiniCPM. Latency floor.",
                 expect=("tokyo",)),
            Turn("What is seven times eight?", "pause:1",
                 "FAST tier: trivial arithmetic, no markers -> fast. Warm floor.",
                 expect=("56|fifty-six|fifty six",)),
            Turn("Name the largest ocean on Earth.", "pause:1",
                 "FAST tier: short factual, no markers -> fast.",
                 expect=("pacific",)),
            Turn("Why do leaves change color in the autumn? Explain the role of chlorophyll.", "wait_for_response",
                 "MAIN tier: 'why' + 'explain' (2 complexity markers, +0.36) + length -> score > 0.3 -> main (gemma3:12b). Reasoning latency.",
                 expect=("chlorophyll",)),
            Turn("Compare the planets Mars and Earth in detail.", "pause:1",
                 "MAIN tier: 'compare' + 'in detail' (2 markers, +0.36) -> main. Comparison/reasoning latency.",
                 expect=("mars", "earth")),
            Turn("Tell me a short story about a lighthouse keeper.", "pause:1",
                 "MAIN tier + LONG-FORM: 'tell me a' + 'story' generation markers (+0.5) -> main, long TTS. The big-answer latency (first-audio + how long it speaks).",
                 expect=("lighthouse|keeper",)),
            Turn("Research the three main causes of the First World War.", "wait_for_response",
                 "RESEARCH tier: 'research' intent -> ReAct planner + main-model synthesis (corpus fallback, no SearXNG running). The research-path latency cost (planner + 12b).",
                 expect=()),
            Turn("Summarize the key benefits of regular exercise in detail.", "pause:1",
                 "MAIN/RESEARCH tier: 'summarize' + 'in detail' markers -> main-model synthesis. Synthesis latency.",
                 expect=("exercise|health|heart|muscle|fitness",)),
        ),
        validates="core/routing.py HeuristicRouter tier selection (threshold 0.3) + the shipped MiniCPM-fast/Gemma-main split + the research/ReAct path. Run with --model gemma3:12b --fast-model minicpm5-1b:q8 to exercise both tiers; assigning one model to both roles collapses the comparison.",
        expected_behavior="MiniCPM fast turns form the lower first-audio cluster; Gemma main/reasoning turns are slower, and the research turn is slowest (planner + main synthesis). Every turn answers correctly or reasonably; per-tier latency is separable in latency.json and the logged tier ('answering on fast/main tier').",
        pass_signals=(
            "Fast turns (1-3) route to the fast tier and answer correctly (Tokyo / 56 / Pacific).",
            "Reasoning turns (4-6) route to the main tier; the story turn produces a multi-sentence long answer.",
            "The research turn (7) routes to main + the planner and answers without a wedge or timeout.",
            "Per-turn first_audio_latency populated for every turn; the tiers are separable.",
        ),
        failure_modes=(
            "A reasoning/long-form turn answers on the FAST tier (router under-escalated) -> the latency comparison is meaningless.",
            "The research turn wedges or hits the timeout apology instead of synthesizing.",
            "main-model (12b) turns error because the model is not loaded / OOM.",
        ),
    ),
)


def by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)


# --- named suites (curated subsets the CLI can run as a battery) --------------
#
# A suite is just an ordered list of scenario NAMES. ``resolve_suite`` filters to
# the scenarios that actually exist (so a suite may list a name added later
# without breaking) and de-dups while preserving order. The five barge-in
# scenarios are INJECT-MODE ONLY (a two-stream acoustic barge-in is impossible on
# the reference box), so the ``acoustic`` suite -- the one you run over the air on
# the real mic -- is every scenario EXCEPT those.

BARGE_SCENARIOS: tuple[str, ...] = (
    "barge_in_interrupt_stop",
    "barge_in_early_vs_late",
    "barge_stop_command_vs_new_topic",
    "barge_multiple_in_one_turn",
    "barge_no_barge_control",
)

# Explicit suites keyed by intent. Names not yet present are skipped at resolve
# time, so adding (e.g.) the realistic_* scenarios later just fills these in.
_NAMED_SUITES: dict[str, tuple[str, ...]] = {
    "core": (
        "baseline_latency_single_turn_qa",
        "context_aggregation_its_population",
        "addon_continuation_merge_and_queue",
        "self_awareness_enumerate_do_decline",
        "smart_endpoint_hold_vs_crisp",
        "never_stuck_heavy_then_recover",
    ),
    "latency": (
        "latency_profile_mixed",
        "capability_latency_profile",
        "baseline_latency_single_turn_qa",
    ),
    "capability": (
        "capability_latency_profile",
    ),
    "realistic": (
        "realistic_morning_planning",
        "realistic_cooking_help",
        "realistic_curiosity_chat",
        "realistic_quickfire_assist",
    ),
    "barge": BARGE_SCENARIOS,
}


def suite_names() -> tuple[str, ...]:
    """The selectable suite names (explicit suites + the computed all/acoustic)."""
    return tuple(sorted({*_NAMED_SUITES.keys(), "all", "acoustic"}))


def resolve_suite(suite: str) -> list[Scenario]:
    """Resolve a suite name to the in-order, existing scenarios it selects.

    * ``all`` -> every scenario, declaration order.
    * ``acoustic`` -> every scenario EXCEPT the inject-only barge-in ones (the set
      you run over the air on the real mic).
    * otherwise -> the named suite, filtered to scenarios that exist, de-duped.

    Raises KeyError for an unknown suite name."""
    if suite == "all":
        return list(SCENARIOS)
    if suite == "acoustic":
        return [s for s in SCENARIOS if s.name not in set(BARGE_SCENARIOS)]
    if suite not in _NAMED_SUITES:
        raise KeyError(suite)
    present = {s.name: s for s in SCENARIOS}
    seen: set[str] = set()
    out: list[Scenario] = []
    for name in _NAMED_SUITES[suite]:
        if name in present and name not in seen:
            seen.add(name)
            out.append(present[name])
    return out
