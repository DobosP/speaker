# Task Result — enrollment-optional barge + read-only Obsidian vault

Valid until: ADR-0072 or ADR-0073, or either implementation, changes — then treat as history.

Branch: `feat/obsidian-vault-tool` rebased onto local `main` at `0820773`.

Status: combined deterministic/headless gates green; physical bare-speaker
barge-in remains red and unvalidated.

## Outcome

### Enrollment-optional barge-in (ADR-0072)

- Single-voice word-cut defaults to enrollment-independent lexical authority;
  missing `barge_word_cut_require_speaker` keys resolve to false consistently.
- Identity-free generic cuts retain a non-lowerable four-novel-non-own-word
  floor. Exact novel controls retain their bounded short exception.
- Own-TTS-ambiguous STOP still requires compatible warmed speaker authority;
  silence, generic own echo, empty text, and zero-to-three generic words fail
  closed. `barge_word_cut_require_speaker=true` remains the multi-voice opt-in.
- Normal-final `speaker_gate_input`, typed owner verification, sensitive action
  gates, enrollment capture/provenance, preparation, and promotion are unchanged.
- Physical enrollment-on `192151` and enrollment-off `193713` remain failed;
  this headless change does not claim live barge validation.

### Read-only Obsidian vault (ADR-0073)

- A default-off `vault.search` capability reads bounded Markdown excerpts from
  `~/work/dobo-brain/paul-brain` when enabled in machine-local configuration.
- Explicit and conversational private-vault requests stay local, synthesize
  before speech, and never fall through to web search. Web-source grammar is
  shared by deterministic speech routing and the ReAct controller allowlist.
- The POSIX reader retains a root descriptor opened component-by-component
  without following links; descendants use verified relative handles. It skips
  `.git`/`.obsidian`, rejects unsafe names, and caps traversal, reads, results,
  excerpts, and output. It never mutates the vault or returns absolute paths.
- Note text is PRIVATE file-origin data, independently output-bounded and fenced
  as untrusted before model use. Missing/unreadable/unsupported roots are not
  registered or advertised. The committed template remains disabled.
- MiniCPM's verified native first-four schema is unchanged. Deterministic SEARCH
  handles supported vault phrases; vault-scoped native ReAct fails closed.

## Main files

- Barge lane: `core/engines/sherpa.py`, `core/readiness.py`, `core/enroll.py`,
  `tools/setup_models.py`, barge/readiness/duplex tests, ADR-0072, and related
  operator documentation.
- Vault lane: `core/obsidian.py`, runtime/app/config/persona/capability wiring,
  `always_on_agent` analyzer/planner/ReAct policy, `tests/test_obsidian.py`,
  ADR-0073, and profile/capture invariants.
- Shared truth: `STATUS.md`, `TASK_RESULT.md`, `config.json`, and
  `docs/unified_architecture.md` retain both lanes.

## Combined-tree verification

```text
PYTHONDONTWRITEBYTECODE=1 /home/dobo/work/speaker/.venv/bin/python -m pytest \
  -p no:cacheprovider tests/test_obsidian.py -q
29 passed in 0.36s

/home/dobo/work/speaker/.venv/bin/python -m pytest \
  tests/test_barge_word_cut.py tests/test_setup_doctor.py \
  tests/test_sherpa_duplex_runtime.py tests/test_capture_integration.py \
  tests/test_final_trust_lineage.py tests/test_barge_in_suppression.py \
  tests/test_device_profile_invariants.py tests/test_speaker_input_gate.py -q
376 passed in 4.87s

/home/dobo/work/speaker/.venv/bin/python -m pytest \
  tests/test_apm_double_talk.py -q
6 passed in 0.97s

/home/dobo/work/speaker/.venv/bin/python -m pytest tests -q
4093 passed, 31 skipped, 9 warnings in 81.31s

git diff --check
PASS (no output)

test "$(wc -l < STATUS.md)" -le 100
PASS (STATUS.md is exactly 100 lines)
```

The nine full-suite warnings are the existing two endpointing divide-by-zero
warnings and seven Pillow `getdata()` deprecation warnings.

Prior strict recorded-owner evidence is 9 passed. It is historical/root-session
evidence and is not claimed as a worktree gate here unless separately rerun and
reported by the orchestrator.

## Risks and manual validation

- Enrollment optionality does not fix the 2026-07-14 physical failure. The
  enrollment-off run starved before identity: playback-time VAD stayed zero and
  no energy fallback/decoder trace appeared.
- Novel exact controls deliberately remain short; current/recent-own-TTS STOP
  ambiguity still fails closed without compatible warmed speaker authority.
- Physical acceptance still requires prompt causal exact Stop followed by a
  four-or-more-word override, with no self-cut. No live hardware validation is
  run or claimed by this combined headless gate.
- Vault access is POSIX-only for this version and remains default off. Enable it
  only in gitignored machine-local configuration after integration.

## Merge recommendation

The two lanes are headless-green and compatible for orchestrator review;
physical barge acceptance remains a separate red post-merge gate, and no push
is performed by this worktree.
