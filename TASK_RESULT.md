# Task Result — explicit personal-vault command phrasing

Valid until: ADR-0074 or its phrase-routing implementation changes — then treat as history.

## Outcome

- The voice agent now treats natural personal-source forms such as `search in
  my vault`, `go in my vault`, and `find in my vault` as the same deterministic
  local read. Notes, Obsidian, second-brain, Dobo-brain, and Paul-brain aliases
  use the existing SEARCH intent with `search_scope=vault`.
- Topicless commands return the existing bounded listing. Topical commands
  retain their actual ranking terms; routing words and ordered source-correction
  glue are removed without globally deleting words such as `web`, `online`, or
  action verbs when those words are the topic.
- Source choice is ordered and polarity-aware. Explicit vault exclusions remain
  public, explicit web/online choices cannot see the vault, and unambiguous later
  corrections can switch back to the private source. Literal topic clauses are
  not mistaken for corrections or exclusions.
- Status/capability questions and requests containing a separate action or
  mutation clause receive no private read tool. Topic phrases such as `tools to
  edit files` remain searchable, while requests to edit a found note/result fail
  closed.
- Text and native ReAct allowlists enforce the same classification. Local vault
  reads cannot acquire web tools, public searches cannot acquire `vault.search`,
  and non-lookups receive no read tool. This does not rely on model obedience.
- ADR-0073's default-off, no-follow, bounded, read-only, PRIVATE, no-egress
  reader is unchanged. No write authority, general filesystem access, new
  intent/mode, phone-native schema, or barge-in behavior was added.

## Verification

- Vault and phrase matrix: `1066 passed`.
- Vault + speech contract + ReAct + cloud-PII gate: `1108 passed`.
- Adjacent cloud/provider integration gate: `1152 passed`.
- Full deterministic suite: `5130 passed, 31 skipped, 9 known warnings`.
- Required APM/double-talk regression: `6 passed`.
- Python compilation, changed-module imports, and `git diff --check`: passed.
- Independent read-only review matrices passed: 393 established classifier
  cases, 80 mutation topologies, 26 topical controls, 24 common action-word
  topics, and 12 additional mutation-safety controls.
- A headless real-root smoke against `/home/dobo/work/dobo-brain/paul-brain`
  ran all three requested forms with `ScriptedEngine` + `EchoLLM`. Each invoked
  only `vault.search` then `research.local`; none invoked web or spoke a raw
  untrusted-content fence.

## Documentation

- ADR-0074 supersedes ADR-0073's narrow phrase-routing rule while retaining the
  reader's containment, privacy, cancellation, and output-bound decisions.
- `docs/unified_architecture.md` and `STATUS.md` link the current decision and
  record the final 2026-07-16 verification evidence.

## Manual validation and limits

- This is deterministic headless control-plane behavior. No live microphone,
  ASR, speaker hardware, or physical audio validation ran or is claimed.
- The smoke printed only phrases, invoked tool names, and privacy booleans. It
  did not print note bodies, credentials, or raw transcripts, and it did not
  modify the vault.

## Merge recommendation

The implementation and documentation gates are green and may be landed on
`main`. Preserve the existing live-audio red status: this change does not resolve
or validate the separate physical barge-in failure tracked by ADR-0072.
