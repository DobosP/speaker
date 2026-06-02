# Live on-hardware validation harness (2026-05)

A self-driving spoken conversation test: a **synthetic user** (a TTS voice
distinct from the assistant's) speaks scripted lines aloud through your real
speakers; the **real assistant** hears them over the air via your real mic,
thinks, and answers aloud. The harness records an **attributed timeline** (every
turn labelled user vs assistant, with the exact audio file + timestamps) and
**per-turn latency**, so you can validate the capabilities end-to-end on your own
machine.

It runs **only on request** — it needs real ASR/TTS/LLM models and audio
hardware, so it is **not** part of the pytest logic suite. `tools/live_session`.

## Run it

```bash
python -m tools.live_session --check        # preflight: are models + audio ready?
python -m tools.live_session --list          # list scenarios
python -m tools.live_session --list-devices  # your audio devices (pick input/output)

# one scenario, all of them, or a named suite:
python -m tools.live_session --scenario baseline_latency_single_turn_qa
python -m tools.live_session --all
python -m tools.live_session --list-suites             # the named suites
python -m tools.live_session --suite latency           # the latency profile
python -m tools.live_session --suite realistic         # the realistic conversations
python -m tools.live_session --suite acoustic --inject # everything safe over the air
python -m tools.live_session --suite latency --repeat 3 # 3x the turns -> tighter p90/p99
```

Useful flags: `--device <profile>` (config.json profile), `--llm ollama --model
... --fast-model ...`, `--input-device/--output-device <id>`, `--user-speaker-id
/ --user-speed` (the synthetic user's voice), `--no-assistant-audio`,
`--response-timeout <s>`, `--inject`, `--no-input-gate`, `--smart-endpoint`,
`--input-gain`, `--user-volume`, `--barge-in`.

**Suites** (`--suite <name>`, see `--list-suites`) run a curated battery and emit
a **consolidated `SUITE.md` / `SUITE.json`** on top of the per-scenario artifacts:

| Suite | What |
|---|---|
| `all` | every scenario |
| `acoustic` | every scenario EXCEPT the inject-only barge-in ones — the set you run over the air on the real mic |
| `latency` | the `latency_profile_mixed` distribution + the baseline floor |
| `realistic` | the four natural multi-turn conversations |
| `core` | the original capability scenarios (context, add-on, self-awareness, …) |
| `barge` | the five barge-in scenarios (inject-mode only) |

`--repeat N` runs each chosen scenario N times and pools all the turns into the
suite's latency distribution — the cheap way to turn ~10 turns into a tight
p90/p99 sample without writing more scenarios.

**Setup:** the assistant captures the real mic, so put the speakers and mic
**near each other** at a **sane volume** so the synthetic user is heard clearly.
The assistant gates ASR during its own playback, so it won't transcribe itself.
A barge-in scenario deliberately speaks over the assistant — keep the room quiet
otherwise.

### `--inject` (clean pipeline test, no acoustic loop)

Over-the-air needs a **well-coupled** mic+speaker. On a laptop's built-in
speaker→mic the STT garbles badly and the mic re-hears the assistant's own TTS
(feedback) — so on such hardware use **`--inject`**: the synthetic-user audio is
fed **straight into the recognizer** (the real mic is never opened) and the
assistant's TTS goes to a **null sink**. You still exercise the **real**
STT→LLM→TTS pipeline + the brain on **clean** audio — latency comes from metrics,
the assistant track is re-synthesized for the artifact — but with no acoustic
degradation, no feedback, and no flaky-device crashes. `--inject` does **not**
test the literal mic/speaker or barge-in (interruption needs live audio during
playback + AEC). See `docs/archive/live_validation_run_2026-05-30.md` for the first run's
analysis (and why each path was needed). `--no-input-gate` disables the
ACT/INGEST addressing gate, useful when garbled over-the-air STT gets INGEST'd.

## What it produces (per scenario, under `logs/live/<run-id>/<scenario>/`)

- `timeline.json` — the attributed timeline. Every event has `speaker`
  (`user`/`assistant`), the **exact `audio` file** that produced/played it,
  timestamps, and — for assistant turns — the latency breakdown. A user event's
  `asr_final` is **what the assistant heard** (the transcript the user's audio
  produced), closing the loop played-audio → recognition → answer.
- `latency.json` — per-turn `first_audio_latency` (SPEECH_END → first assistant
  audio) + the stage breakdown (endpoint / final→token / token→audio), the
  aggregate floor, **and a distribution**: `aggregate_first_audio` carries
  `p50/p90/p99/mean` alongside the original `median/min/max`, and a `stages` block
  gives the same percentiles per stage — so you can see *where the time goes*
  (the endpoint trailing-silence wait usually dominates).
- `grade.json` — the honest auto-grades: over-the-air **STT accuracy** (per turn +
  aggregate), the **full-duplex** verdict, the **barge-in** grade, and (new) the
  **response-quality** grade (`response.per_turn` + `response.aggregate`): did the
  assistant's *answer* contain the expected concepts and avoid the forbidden ones.
- `summary.md` — a human-readable, **gradeable** report: the attributed
  conversation, the latency table + distribution, the STT grade, the
  **response-quality table**, and the scenario's expected behavior / pass signals
  / failure modes.
- `user/NN.wav` — the **exact** synthesized user audio that was played.
- `assistant/NN.wav` — a clean re-synthesis of what the assistant said (the
  separable assistant track; `--no-assistant-audio` to skip).
- `heard_over_air.wav` — the **ground-truth** mic recording (what the assistant
  actually heard over the air — the user + the assistant's own voice mixed).

## Differentiating the two speakers

Attribution never relies on the acoustic difference: the harness *controls* the
user audio (it synthesizes + plays it), so it always knows which side is which.
The distinct user voice (different speaker id / speed) is so a *human* listening
to `heard_over_air.wav` can tell them apart. The `timeline.json` is the source of
truth: each event is labelled and points at its exact audio file.

## Response-quality grading (did the *answer* address the question)

STT grading asks "did the mic hear the user right?"; **response grading** asks the
complementary "did the assistant *answer* right?" — the thing the user actually
cares about. A scenario `Turn` can carry two optional fields:

- `expect=(...)` — CONCEPTS the answer should contain. Each item may list
  alternatives with a `|` (e.g. `"seven|7"`, `"freeze|freezes|ice"`) and is
  satisfied if **any** matches. A digit (`"4"`) matches a whole token (so it won't
  hit inside `"40"`); an alphabetic concept matches as a substring (so `"moon"`
  hits `"moons"`). The turn's response score is the **fraction of expect items
  satisfied** (1.0 when `expect` is empty — nothing checkable, e.g. an open-ended
  or live-data turn; those are still graded on STT + latency).
- `forbid=(...)` — substrings the answer must NOT contain: the **honesty probes**
  (a note/reminder/web-search claim the assistant can't actually fulfil). Any hit
  flags the turn regardless of score — this is how the realistic scenarios catch
  the model *fabricating* a confirmation ("I've set a reminder").

`grade.json → response` and the `summary.md` "Response quality" table carry the raw
`matched` / `missing` / `forbidden_hit` per turn, so the bar
(`RESPONSE_OK_THRESHOLD`, default 0.6) is recalibratable without a re-run. All of
this is pure + unit-tested (`tests/test_live_session.py`); no audio/models needed
to test the grader itself.

## The consolidated suite report (`SUITE.md` / `SUITE.json`)

Running a suite (`--suite`, `--all`, several `--scenario`, or `--repeat`) writes a
**pooled dashboard** at the run root: one **latency distribution** over *every turn
across every scenario* (p50/p90/p99 + the per-stage `endpoint`/`LLM`/`TTS`
breakdown), pooled **STT** and **response-quality** numbers, and a **per-scenario
table** (turns, first-audio p50/p90, STT median, response median + ok-count +
forbidden hits, full-duplex, barge verdict). It is the "test everything + see
latency + see how it responds" view; the per-scenario `summary.md` files stay the
drill-down. `build_suite_report` recomputes purely from each run's events, so the
suite view can never drift from the per-scenario artifacts.

## The scenarios (what each validates)

| Scenario | Capability |
|---|---|
| `baseline_latency_single_turn_qa` | latency floor + clean attribution on the easy path |
| `context_aggregation_its_population` | short-term memory ("what's *its* population?") |
| `addon_continuation_merge_and_queue` | ADD-ON / continuation (one answer, not two racing) |
| `self_awareness_enumerate_do_decline` | knows its real skills, doesn't confabulate |
| `smart_endpoint_hold_vs_crisp` | endpoint hold-on-pause vs crisp (enable `sherpa.endpoint_enabled`) |
| `barge_in_*` (5 scenarios) | interrupt promptly, no stale audio, no self-interrupt (inject-only) |
| `never_stuck_heavy_then_recover` | heavy turns recover; controller never wedges |
| `latency_profile_mixed` | 11 mixed single-turns → a real first-audio **distribution** + response correctness |
| `realistic_morning_planning` | natural multi-turn: live-data deflection, arithmetic, coreference, reminder honesty probe |
| `realistic_cooking_help` | hands-free kitchen Q&A: substitution, conversions, egg-timing, "that"-coreference |
| `realistic_curiosity_chat` | general-knowledge chain with short-term-memory coreference (its capital → its language → compare) |
| `realistic_quickfire_assist` | terse fast turn-taking to probe the latency floor |

The `realistic_*` + `latency_profile_mixed` scenarios are **response-graded** (they
carry `expect`/`forbid`); `--list` tags each scenario with its graded-turn count.

Timing of each user line (how it's scheduled relative to the assistant):
`wait_for_response` (speak, wait until idle) · `immediately` (tack on an add-on
before the answer) · `barge_in` (speak over the assistant mid-answer) ·
`pause:<N>` (silent gap, then wait).

## Caveats

- Over-the-air acoustics vary by room/hardware; treat absolute latency numbers as
  *your* machine's, and use them as a baseline to track regressions, not a
  universal spec.
- `smart_endpoint` only changes behavior when `sherpa.endpoint_enabled` is true
  (default off / experimental); otherwise the scenario just measures the
  fixed-timer endpoint latency.
- A true capability *hang* is hard to force live; `never_stuck` mainly confirms
  the controller stays responsive across a long run.
- The assistant track is a faithful **re-synthesis** of the spoken text (same
  voice), not the exact played PCM; `heard_over_air.wav` is the ground truth.
- The synthetic user opens its own output stream while the assistant's playback
  stream is also open. Most backends mix concurrent output streams fine; on an
  exclusive-mode backend, point them at the same `--output-device` (a per-scenario
  failure is caught and reported, not fatal to the run).
- Assistant turns are captured from what the engine actually `speak()`s, so a
  **barged/interrupted** answer IS recorded (flagged `interrupted: true` in the
  timeline) — the start of the story you cut off shows up.
```
