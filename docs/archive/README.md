# docs/archive — superseded historical docs

These docs are **historical record**, not current truth. Each was absorbed into
[`../unified_architecture.md`](../unified_architecture.md) (the single current-truth
overview, §0–§13) during the 2026-06-02 doc-unification pass and carries a
"superseded" banner at its top.

**They are not consulted for current project status.** The live sources are:

- `STATUS.md` — current truth (top of the precedence order)
- `docs/unified_architecture.md` — current-truth architecture (start here)
- `docs/target_architecture.md` (north-star) and `docs/adr/` (dated decisions)
- `.agents/status.json` + `.agents/backlog.md` — current state + open work
- the newest `docs/session_*.md` handoff (read by `python -m tools.session_bootstrap`)

(2026-07-02: the old as-built snapshot `architecture.md` and the dated
barge-in/ASR/endpoint decision docs were moved into this archive with
SUPERSEDED banners pointing at their successor ADRs.)

They are kept (rather than deleted) because a handful of code/test comments cite
them as the origin of a decision, and because they preserve the rationale trail.
Read them for *why* something was done historically — not for *what is true now*.
