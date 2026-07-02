# ADR-0008: Owner decisions D-A (open-speaker barge-in is a MUST) and D-B (WAV/PII purge deferred)

Date: 2026-06-17
Status: accepted

## Decision
Record two owner decisions from `docs/roadmap_2026-06-17.md`:
**D-A** — open-speaker barge-in on the bare laptop speaker is a HARD
requirement; headphones are never an acceptable fix and must not be suggested.
**D-B** — the git-tracked `logs/runs/` run bundles (raw-voice WAVs +
transcript-bearing summaries) stay committed during active development; the
purge + transcript redaction is deferred to a pre-release hygiene gate,
coupled to the D1 public-history purge, and is OWNER-ONLY.

## Context / why
D-A (first stated 2026-06-05): the "hardware limit → use headphones"
conclusion from early sessions was REJECTED by the owner; the product is an
open-speaker assistant. D-B: the tracked bundles are the barge-in
golden-regression corpus and were in active use — purging mid-development
would destroy the evidence base the open P1 work replays. Why not purge now:
filter-repo/force-push on the public repo is destructive and owner-only;
doing it piecemeal would have to be redone at release anyway. Caveat that
stands: the repo is PUBLIC, so the WAVs are already exposed until the gate
runs — a private remote remains the mitigation option.

## Consequences
- Agents never suggest headphones, and never delete/redact tracked run
  bundles on their own.
- New `logs/runs/` files are gitignored (2026-07-02); the tracked set is
  frozen, not growing.
- The pre-release gate must run: history purge (filter-repo), WAV un-ignore
  flip, writer-level redaction, gitleaks/PII CI. Trigger: before any
  public/non-owner release. Tracked in STATUS.md (owner-decided, deferred).
- Owner reaffirmation (2026-07-02, at ADR writing): the current public
  exposure is accepted for now; the exposed Gemini key was rotated same day
  (old key dead). The release path is republication from the organization's
  GitHub account — D1 + D-B both execute at that gate, not before.
