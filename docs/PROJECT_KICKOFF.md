# Project Kickoff — Clarifying Questions

A running list of decisions to pin down before/while building. Answer inline
(replace the `> _answer:_` lines). Unanswered items are open. This file is the
source of truth for *intent*; `docs/target_architecture.md` is the source of
truth for the *technical plan*. Claude reads both each session.

Legend: ⭐ = blocks real work until answered. 💡 = my current recommendation.

---

## 1. What "done" looks like

- ⭐ In one sentence, what should the v1 you actually *use daily* do?
  > _answer:_
- ⭐ Which single platform must work first? 💡 desktop Linux, then Android.
  > _answer:_
- What does success feel like — faster replies, never mis-hearing you,
  hands-free, or running tasks in the background?
  > _answer:_
- What's the one thing the current version does that most annoys you?
  > _answer:_ (you've said: latency, missing the exact question, bad stop-talking)

## 2. Modes & behavior

- ⭐ List the modes you want and what each one *does* differently.
  (You mentioned: research, quiet, interact/ask-questions, background tasks.)
  > _answer:_
- How does it switch modes — a wake phrase, a hotkey, automatic by context?
  > _answer:_
- In "quiet" mode, does it still listen and remember silently, or fully sleep?
  > _answer:_
- When should it speak *unprompted* (proactive), vs only when addressed?
  > _answer:_
- ⭐ "Listens all the time": does that mean always transcribing, or only acting
  after a wake word? (Always-transcribing has battery/privacy cost.)
  > _answer:_

## 3. The "background tasks" you want

- ⭐ Give 3 concrete examples of background tasks (e.g. "research X while I keep
  talking", "summarize this meeting", "remind me later").
  > _answer:_
- Should results interrupt you when ready, or wait until you ask / go idle?
  > _answer:_
- How many tasks at once is realistic for you? 💡 cap parallel research at 2–3.
  > _answer:_
- Do any tasks need to *act* (send, delete, change files), or only read/report?
  Action tasks will require a spoken confirmation step.
  > _answer:_

## 4. Privacy & locality

- ⭐ Fully local, no exceptions? (You said yes.) Confirm this includes *web
  search* — i.e. "research" mode searches local docs only, not the internet?
  > _answer:_
- Is the always-on transcript ever written to disk? For how long? Encrypted?
  > _answer:_
- Multiple speakers in the room — should it only respond to *you* (speaker ID)?
  > _answer:_

## 5. Hardware reality

- ⭐ What machine(s) will you run this on? (CPU, RAM, GPU, OS versions.)
  This decides which LLM/STT models are realistic.
  > _answer:_
- Which phone(s)? (Model + OS version.) On-device LLM viability depends on it.
  > _answer:_
- Headset/AirPods, laptop mic, or a far-field mic? (Affects echo/barge-in.)
  > _answer:_

## 6. The assistant's "personality" & output

- How long should spoken answers be by default? 💡 short, 1–3 sentences.
  > _answer:_
- Formal, casual, terse? Any persona?
  > _answer:_
- Language(s)? English only, or multilingual?
  > _answer:_

## 7. Scope guardrails (so we don't boil the ocean)

- ⭐ For the *first working rebuild*, which of these can we drop initially?
  (speaker verification / wakeword / mobile / multilingual / web research /
  vector memory)
  > _answer:_
- What's explicitly OUT of scope for v1?
  > _answer:_
- 💡 Recommended v1: desktop, English, push-to-talk or single wake word,
  3 modes (quiet/assistant/research-local), short replies, SQLite memory,
  built on sherpa-onnx + the existing brain. Agree / adjust?
  > _answer:_

## 8. Open technical decisions (mirrors target_architecture.md §9)

- UI shell: Flutter vs native? 💡 Flutter.
  > _answer:_
- Mobile LLM runtime: llama.cpp vs MLC-LLM vs ExecuTorch? 💡 decide after a phone spike.
  > _answer:_
- Core language for the eventual mobile port: Python-embedded vs Rust/C++? 💡 Rust/C++ core, Python desktop binding.
  > _answer:_
- iOS always-on is OS-restricted — accept push-to-talk/wakeword on iPhone? 💡 yes.
  > _answer:_

---

## How to use this with Claude

- Fill in the ⭐ items first — those unblock real work.
- Tell Claude "read PROJECT_KICKOFF.md and target_architecture.md, then start
  Phase 1" and it will work from your answers instead of guessing.
- Update this file as decisions change; it's the living brief.
