> **SUPERSEDED (2026-07-02):** describes the pre-refactor stack —
> `scripts/run_tests.py`, `python main.py --record`, `benchmarks/`, and
> `VoiceAssistant` were all deleted with the `main.py` monolith
> (2026-05-26, `docs/adr/0002`). The current testing guide is
> [`docs/testing.md`](../testing.md) (staged runner: `python tools/run_tests.py`).
> Historical record.

# Audio TDD Workflow

This project uses three layers of voice testing so failures can be reproduced
without a microphone, speaker, local LLM, or network.

## Fast Deterministic Tests

Preferred staged command:

```bash
python scripts/run_tests.py dev
```

This writes raw output, JUnit XML, `summary.json`, `failures.json`, duplicate
analysis, and an LLM-readable summary under `test-reports/latest/dev/`.

You can also run the conversation simulator directly:

```bash
python -m pytest tests/test_conversation_simulation.py -q
```

These tests keep the real `AudioRecorder` and `VoiceAssistant` control flow in
the loop, but replace STT, LLM, and TTS with scripted doubles from
`tests/conversation_harness.py`.

Use this layer for TDD when adding a new real-world failure:

1. Add a scenario to `tests/test_conversation_simulation.py`.
2. Script the transcript with `ScriptedSTT`.
3. Script streamed assistant sentences with `ScriptedStreamingLLM`.
4. Use realistic fixture audio from `tests/fixtures.py`.
5. Assert the observable behavior: callback delivery, TTS start, cancellation,
   no self-interrupt, wakeword gating, or latency budget.

The LLM is intentionally simulated in this layer. Real model output is slow,
nondeterministic, and depends on Ollama availability; the scripted LLM captures
the behaviors that matter for voice systems: streaming chunks, delay, empty
answers, and cancellation.

## Staged Test Runner

Use `scripts/run_tests.py` for repeatable local and CI runs:

```bash
python scripts/run_tests.py list
python scripts/run_tests.py dev
python scripts/run_tests.py full
python scripts/run_tests.py discovery --allow-failures
python scripts/run_tests.py analyze
```

Stages:

- `smoke`: fastest import/config/schema checks.
- `dev`: critical fast TDD tests.
- `audio`: deterministic audio and replay subset.
- `replay`: recorded session replay.
- `discovery`: known failure-discovery suites; useful for bug hunting.
- `backend`: optional STT/TTS/LLM backend checks.
- `full`: normal regression tests excluding discovery/backend/hardware/network/LLM.
- `all`: everything.

Every run creates a timestamped folder under `test-reports/` and refreshes
`test-reports/latest`. For quick analysis, start with:

```bash
test-reports/latest/llm-summary.md
```

Per-stage details live in files such as:

- `stdout.txt`
- `junit.xml`
- `summary.json`
- `failures.json`
- `duplicates.json`
- `llm-summary.md`

## Recorded Session Replay

Capture real sessions while using the assistant, then generate replay tests:

```bash
python main.py --record
python scripts/generate_session_tests.py --all
python -m pytest tests/test_recorded_sessions.py -q
```

Each recorded TTS turn stores `tts_16k.npy`, `mic_16k.npy`, and `turn.json` under
`recordings/session_*/turn_*`. Generated tests replay those mic samples through
the barge-in pipeline.

When a recorded turn is ambiguous, annotate `turn.json`:

```json
{
  "annotation": "false_positive",
  "annotation_reason": "assistant echo only"
}
```

Allowed annotations are `true_positive`, `false_positive`, and `unknown`.
Regenerate tests after annotating.

## Acoustic Fixtures

`tests/fixtures.py` provides deterministic audio primitives and realistic
fallbacks:

- `real_speech`, `human_voice`, `human_voice_concat`
- `real_tts_echo`, `reverberant_echo`, `apply_room_delay`
- `tv_noise`, `babble_noise`, `music_noise`
- `mix`, `near_far_mix`, `apply_gain_db`, `hard_clip`,
  `sample_rate_roundtrip`

Prefer real samples when available, but keep synthetic fallbacks so CI remains
offline-capable.

## Virtual Real-World Scenes

`tests/virtual_scenes.py` builds metadata-rich acoustic scenes for virtual
real-world testing. It combines open-source FSDD human speech with deterministic
room profiles, distance/SNR changes, device artifacts, background media/noise,
and assistant echo paths.

Run the virtual scene suite directly:

```bash
python -m pytest tests/test_virtual_real_world_scenes.py -q
```

The suite writes samples and metadata under
`tests/fixture_audio/virtual_real_world/`. Each scene records the source
dataset, speaker, distance, SNR, room profile, device artifact, expectation,
cause category, solvability, and mitigation.

These tests intentionally separate two ideas:

- Ungated raw-audio limitation: background human speech can be acoustically
  indistinguishable from a user speaking to the assistant, so these cases are
  categorized as `unsolvable_without_intent_gate`.
- Gated mitigation: the same background scenes must be blocked when strict
  wakeword gating or an intent gate supplies the missing intent signal.

This is the useful virtual-world contract: the tests do not pretend raw mono
audio can infer intent perfectly; they quantify that limit and prove which
gates make the product behavior safe.

Structured reports include failure category counts, solvability counts, and
false-accept / false-reject style rates in `summary.json` and
`llm-summary.md`.

## Latency Benchmark

Run the existing backend benchmark:

```bash
python benchmarks/benchmark_realtime.py --enforce-slo
```

Run the deterministic conversation benchmark as well:

```bash
python benchmarks/benchmark_realtime.py \
  --scenarios "" \
  --simulated-conversation \
  --enforce-slo
```

The simulated benchmark records measured `barge_in_to_stop_ms`,
`stt_to_first_llm_sentence_ms`, `tts_on_start_ms`, and false-positive /
false-negative counts for the scripted conversation path.
