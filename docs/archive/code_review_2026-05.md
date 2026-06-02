# Code review — 2026-05

> ⚠️ **Superseded — durable content merged into [`docs/unified_architecture.md`](unified_architecture.md).** Kept for revision history; do not treat as current. (2026-06-02 consolidation.)

Scope: an honest read on (a) whether the architecture makes sense, (b) what can
be simplified for performance, (c) the state of tests/CI, and (d) overall
direction. Method: three exploration passes over `core/`, `always_on_agent/`,
`mobile/`, `remote/`, `tests/`, `tools/`, and the workflows, plus reading the
hot-path files directly. This is a backlog to pick from, not a set of decisions.

## 1. Architecture — verdict: sound

The split is right: a portable runtime (`core/`) wired to a small control-plane
brain (`always_on_agent/`) across the `AgentEvent`/`Mode` seam
(`always_on_agent/events.py`), with thin per-platform shells. `core/runtime.py`
(~190 lines) is a clean replacement for the deleted monolith; the `AudioEngine`
(`core/engine.py`) and `LLMClient` (`core/llm.py`) protocols are the load-bearing
seams and they hold.

**The thing most reviews miss:** the heavy, performance-critical parts are
*already unified*. Both the Python and Dart runtimes are thin bindings over the
**same native C++ engines** — `sherpa-onnx` (ASR/TTS) and `llama.cpp`/Gemma. You
don't have two engines, you have one set of engines and two thin controllers.

**Only real risk — Python↔Dart brain drift.** `mobile/lib/assistant.dart`
re-derives the two hottest loops instead of sharing the brain:
- sentence-boundary streaming TTS — `core/capabilities.py:43` (`_SENTENCE_END`)
  vs the Dart `_flushSentences` / `_firstSentenceBoundary`;
- the command fast-path — `core/intents.py` + the runtime command map vs the
  Dart `_commands` dict.
This is the documented risk in `docs/target_architecture.md` §9.

**Decision (with the user):** keep §9 — *share the contract + tests, not a binary
core*. The brain is small and the engines are already shared, so a cross-language
golden test suite (see §6) buys ~95% of "one source of truth" for near-zero cost.
Rejected for now: rewriting the core in Rust/FFI (roadmap Phase 3 — revisit only
if drift hurts or iOS-background forces it) and collapsing everything into Flutter
(weaker desktop LLM story, throws away the Python core + its 200+ tests).

## 2. Simplification / performance — honest take: little fat to cut

This is a lean, post-refactor codebase (0 TODO/FIXME markers, the 1,800-line
monolith is gone). The dominant latency cost is **model inference** (ASR/LLM/TTS),
which dwarfs the orchestration code by orders of magnitude — and the levers that
matter are already in place: per-device model/thread tuning (`device_profiles`),
the two-model split (`core/capabilities.py:96-103`), the KWS command fast-path,
and streaming TTS.

Two micro-optimizations flagged during exploration **do not pay off** on
inspection — recording them so they don't get re-proposed:
- `_stream_and_speak` (`core/capabilities.py:46-74`) already uses the efficient
  `list.append` + single `"".join`; only `buffer += token` is extra work, and
  it's negligible next to token-generation time. **Leave it.**
- The speaker-gate (`core/engines/speaker_gate.py:75-106`) already caches the
  sherpa extractor (`extractor_holder`, lines 93-99); each barge-in necessarily
  runs one ONNX forward on *new* audio, so there's nothing to cache away.
  **Leave it.**

Lower-priority, genuinely optional:
- `always_on_agent/tasks.py` spawns a worker thread per task. Fine for a voice
  assistant (one turn at a time); only worth a pool/asyncio if task churn ever
  grows. Not now.

**The one win worth doing is maintainability, not speed:** collapse the
Python↔Dart duplication behind the golden suite (§6). Correctness > a micro-opt
the model latency would swallow anyway.

## 3. Tests / CI — healthy, with a few nits

Suite is solid: **203 passed / 1 skipped in ~13s**, 24 files, no flakiness, no
collection errors. CI (`.github/workflows/tests.yml`, Python 3.11) runs the logic
suite on every push to `main` and every PR. Nits to consider (not urgent):
- LiveKit tests are excluded by `--ignore` *path* (`tests.yml:32-34`) rather than
  by the `audio`/`network` markers that `pytest.ini` already declares — those
  markers are defined but unused. Marker-based exclusion would be self-documenting
  and wouldn't silently drop a renamed file.
- The staged runner's `core` stage (`tools/testing/`) omits
  `test_core_routing.py` (it only runs in `sandbox`/`full`) — likely an oversight.
- Untested modules: `always_on_agent/{bridge,adapters,diagnostics,snapshots}.py`
  and capabilities composition in `always_on_agent/capabilities.py`.

## 4. Direction

- **Keystone (decided):** build the cross-language golden suite (§6) and converge
  `mobile/lib/assistant.dart` onto the `AgentEvent` contract. This is what keeps
  "one core, many shells" from drifting into "many apps."
- **Then:** define the FFI/IPC core boundary (roadmap Phase 3) only if/when needed.
- **Open product question:** `docs/PROJECT_KICKOFF.md` §1 ("what does v1 do") is
  still unanswered — worth nailing down, since it scopes everything else.

## 5. Open-source adoption — flag, don't pull (yet)

Per the "only if it clearly wins" steer:
- ASR/TTS/VAD/speaker-ID: `sherpa-onnx` already *is* the best-in-class local OSS
  here. **Do not replace.**
- Sentence segmentation: a library (`pysbd`, `wtpsplit`) could back a shared
  splitter, but the current regex is adequate and a new dependency on every
  platform likely isn't worth it. Revisit only if the golden suite surfaces real
  edge-case failures.

Net: nothing compelling to import right now.

## 6. Cross-language golden test suite — BUILT

Landed on branch `claude/golden-contract-tests`. The known duplication is now a
single tested contract, reconciled to one canonical spec (the safer behavior):
- **Shared pure modules:** `core/contract.py` and `mobile/lib/contract.dart`,
  used by production on each side (`core/capabilities.py` + `core/runtime.py`;
  `mobile/lib/assistant.dart`). Same algorithm, two languages.
- **Shared fixtures:** `tests/golden/sentence_split.json` + `commands.json`.
- **Two runners over the same files:** `tests/test_golden_contract.py` (pytest,
  in `tests.yml`) and `mobile/test/golden_contract_test.dart` (`flutter test`,
  via the new `.github/workflows/mobile-tests.yml`).
- **Canonical reconciliation:** sentence boundary = newline OR `.!?`+whitespace
  (mobile no longer splits `"3.14"`; desktop now honors bare newlines); commands
  share one normalization + the `{stop,cancel,quiet,stop talking,be quiet}` stop
  set (mode/confirm/deny stay desktop-only). Any future change must update the
  fixtures, so the shells can't silently diverge (§9 mitigation, realized).

## Prioritized backlog (pick from these)

1. ~~Golden test suite (§6)~~ — **DONE** (branch `claude/golden-contract-tests`).
2. CI marker hygiene + add `test_core_routing.py` to the `core` stage (§3) — cheap.
3. Tests for `bridge`/`adapters`/`diagnostics`/`snapshots` (§3) — fills real gaps.
4. Answer `PROJECT_KICKOFF.md` §1 (§4) — unblocks scoping.
5. (Defer) Rust/FFI core, thread pooling, OSS segmenter — only if a concrete need
   appears.
