# Project Kickoff — Clarifying Questions

A running list of decisions to pin down before/while building. Answer inline
(replace the `> _answer:_` lines). Unanswered items are open. This file is the
source of truth for *intent*; `docs/target_architecture.md` is the source of
truth for the *technical plan*. Claude reads both each session.

Legend: ⭐ = blocks real work until answered. 💡 = my current recommendation.

**Round resolved 2026-05-28:** v1 product shape, modes, listening model,
local/cloud boundary, hardware target, v1 scope. The boundaries those answers
draw are mirrored in `docs/target_architecture.md` §9 (decisions 7–10).

---

## 1. What "done" looks like

- ⭐ In one sentence, what should the v1 you actually *use daily* do?
  > _answer:_ One flexible assistant that blends three roles — hands-free
  > work companion (talks to me while I code/write), ambient researcher
  > (listens, captures, summarizes, pulls in relevant material), and
  > voice-driven automation (does things for me on confirmation). It
  > decides itself when to act vs. just ingest.
- ⭐ Which single platform must work first? 💡 desktop Linux, then Android.
  > _answer:_ **Desktop Linux.** Mobile (Android) is explicitly out of v1
  > scope (§7); the existing `mobile/` Flutter app keeps shipping as a
  > parallel track but does not gate v1.
- What does success feel like — faster replies, never mis-hearing you,
  hands-free, or running tasks in the background?
  > _answer:_
- What's the one thing the current version does that most annoys you?
  > _answer:_ (you've said: latency, missing the exact question, bad stop-talking)

## 2. Modes & behavior

- ⭐ List the modes you want and what each one *does* differently.
  (You mentioned: research, quiet, interact/ask-questions, background tasks.)
  > _answer:_ Four modes:
  > * **quiet / passive** — transcribes silently for memory; never speaks.
  > * **assistant** — default conversational; short replies when addressed.
  > * **research** — spawns background gathering/summarizing tasks; uses
  >   the cloud thinking tier + cloud web search (§4).
  > * **command / action** — voice-driven action with spoken confirmation
  >   for destructive ops (§3).
  > Dictation (pure speech→text, no LLM) is not in v1 — add later if useful.
- How does it switch modes — a wake phrase, a hotkey, automatic by context?
  > _answer:_
- In "quiet" mode, does it still listen and remember silently, or fully sleep?
  > _answer:_
- When should it speak *unprompted* (proactive), vs only when addressed?
  > _answer:_ Configurable per task — see §3 result delivery.
- ⭐ "Listens all the time": does that mean always transcribing, or only acting
  after a wake word? (Always-transcribing has battery/privacy cost.)
  > _answer:_ **Always transcribing; acts via implicit addressing.** The
  > brain judges from conversational context whether you're talking to it;
  > no wake word required. Combined with speaker-ID (§4) so only your
  > voice is even considered an "address." This is the central design
  > problem — the "smart input gate" — and is the next PR after kickoff.

## 3. The "background tasks" you want

- ⭐ Give 3 concrete examples of background tasks (e.g. "research X while I keep
  talking", "summarize this meeting", "remind me later").
  > _answer:_ All four families in v1:
  > 1. Background research while you keep working (returns with a summary).
  > 2. Summarize what you're looking at (current screen / open file /
  >    link / meeting) — uses multimodal + cloud thinking tier.
  > 3. Reminders / timers (time-based; persistent across restarts).
  > 4. Watch / monitor for events ("tell me when X happens") — open-ended,
  >    needs integrations per source.
- Should results interrupt you when ready, or wait until you ask / go idle?
  > _answer:_ **Configurable per task.** Each task tags its preferred
  > delivery when started: speak-immediately / wait-until-idle /
  > non-audio notification. Sensible default per family TBD.
- How many tasks at once is realistic for you? 💡 cap parallel research at 2–3.
  > _answer:_ **2–3 concurrent.** Matches the recommendation; also the
  > realistic upper bound given working memory + cloud rate limits.
- Do any tasks need to *act* (send, delete, change files), or only read/report?
  Action tasks will require a spoken confirmation step.
  > _answer:_ **Yes — destructive-only confirmation.** Read-only tasks
  > execute automatically; anything that sends/deletes/modifies asks
  > spoken yes/no before acting. Wired through the existing action brain.

## 4. Privacy & locality

- ⭐ Fully local, no exceptions? (You said yes.) Confirm this includes *web
  search* — i.e. "research" mode searches local docs only, not the internet?
  > _answer:_ **No longer strictly local.** The boundary (now also in
  > target_architecture.md §9.7):
  > * **Stays local:** STT (sherpa-onnx), TTS, VAD, speaker-ID, the
  >   always-on capture loop, the fast/answering LLM tier
  >   (gemma3:4b-class), conversation memory.
  > * **May use cloud:** the *thinking* tier (main planner, research,
  >   multimodal summarize) and **web search** for research mode.
  > * **Raw audio never leaves the device** — only post-ASR text /
  >   screen captures / files you give it can cross to the cloud tier
  >   when it's invoked.
- Is the always-on transcript ever written to disk? For how long? Encrypted?
  > _answer:_
- Multiple speakers in the room — should it only respond to *you* (speaker ID)?
  > _answer:_ **Yes — speaker-ID gates acting.** Only your voice triggers
  > responses; other speakers are still transcribed into memory (so you
  > can ask "what did Alice say earlier?"). sherpa-onnx speaker-ID is
  > already wired in `core/engines/speaker_gate.py`; v1 enrolls your
  > voice once at setup.

## 5. Hardware reality

- ⭐ What machine(s) will you run this on? (CPU, RAM, GPU, OS versions.)
  This decides which LLM/STT models are realistic.
  > _answer:_ **~16 GB VRAM GPU + 33 GB RAM workstation** (Linux). Already
  > runs `gemma3:12b` (main) + `gemma3:4b` (fast) on Ollama; this is the
  > `desktop` device profile in `config.json`. Cloud thinking tier (§4)
  > extends this when local headroom isn't enough.
- Which phone(s)? (Model + OS version.) On-device LLM viability depends on it.
  > _answer:_ Out of v1 scope (§7).
- Headset/AirPods, laptop mic, or a far-field mic? (Affects echo/barge-in.)
  > _answer:_ **Varies — both headset and laptop-mic-plus-speakers in
  > daily use.** Speaker-ID gating (§4) is therefore *essential*: with
  > speakers, mic picks up our own TTS, which without the gate produces
  > the barge-in storms visible in `logs/runs/run-20260528-004726.txt`
  > lines 68–75. Calibrate at session start per device.

## 6. The assistant's "personality" & output

- How long should spoken answers be by default? 💡 short, 1–3 sentences.
  > _answer:_
- Formal, casual, terse? Any persona?
  > _answer:_
- Language(s)? English only, or multilingual?
  > _answer:_ **English only for v1.** Multilingual is explicitly out of
  > scope (§7).

## 7. Scope guardrails (so we don't boil the ocean)

- ⭐ For the *first working rebuild*, which of these can we drop initially?
  (speaker verification / wakeword / mobile / multilingual / web research /
  vector memory)
  > _answer:_ **Drop: mobile, multilingual, cross-device sync.**
  > **Keep:** speaker-ID (it's essential per §4/§5), web research (cloud
  > tier; §4), vector memory (existing Postgres+pgvector on desktop).
  > No wakeword (implicit addressing instead; §2).
- What's explicitly OUT of scope for v1?
  > _answer:_ Mobile/Android shell, multilingual, cross-device sync of
  > memory or tasks. Single-machine desktop-Linux v1.
- 💡 Recommended v1: desktop, English, push-to-talk or single wake word,
  3 modes (quiet/assistant/research-local), short replies, SQLite memory,
  built on sherpa-onnx + the existing brain. Agree / adjust?
  > _answer:_ **Adjust** — desktop ✓, English ✓, sherpa-onnx + existing
  > brain ✓. But: **4 modes** (quiet/assistant/research/command); **no
  > wake word** (implicit addressing via "smart input gate"); **cloud
  > thinking tier + cloud web search allowed** (relaxes the original
  > fully-local stance); **Postgres+pgvector memory** stays (desktop;
  > SQLite stays the mobile choice when mobile comes back).

## 8. Technical decisions — RESOLVED (mirrors target_architecture.md §9)

- UI shell: Flutter vs native? 💡 Flutter.
  > _answer:_ **Flutter** — chosen and built (`mobile/`).
- Mobile LLM runtime: llama.cpp vs MLC-LLM vs ExecuTorch? 💡 decide after a phone spike.
  > _answer:_ **MediaPipe/LiteRT via `flutter_gemma`** (Gemma 3 1B) in the shipped
  > app; `llama.cpp`/Ollama stay the runtimes for the Python core's desktop/phone
  > profiles. (The shipped choice supersedes the three-way comparison.)
- Core language for the eventual mobile port: Python-embedded vs Rust/C++? 💡 Rust/C++ core, Python desktop binding.
  > _answer:_ **Neither** — share the `AgentEvent`/`Mode` **contract + tests** and
  > reimplement the small brain per runtime (Python desktop/server, Dart mobile).
  > A Rust/C++ FFI core is the most work for the least gain now; revisit only if
  > the Python↔Dart brains drift despite the shared golden tests.
- iOS always-on is OS-restricted — accept push-to-talk/wakeword on iPhone? 💡 yes.
  > _answer:_ **Yes** — push-to-talk / wakeword-gated on iOS; the `remote/` host
  > path covers continuous-listening needs where required.

**Topology (one app or many?):** one portable **core** + thin per-platform
**shells**, **hybrid** deployment (on-device first, host+thin-client fallback).
Not a monolith, not independent apps. See target_architecture.md §0/§9.

---

## How to use this with Claude

- Fill in the ⭐ items first — those unblock real work.
- Tell Claude "read PROJECT_KICKOFF.md and target_architecture.md, then start
  Phase 1" and it will work from your answers instead of guessing.
- Update this file as decisions change; it's the living brief.
