# ADR-0073: Add bounded read-only Obsidian access

Date: 2026-07-14
Status: accepted

## Decision

Add one default-off `vault.search` capability for bounded search/read access to
a configured Markdown vault. Use `~/work/dobo-brain/paul-brain` as the portable
Linux path default, but register and advertise the capability only when the
operator enables it, that directory is readable, and safe POSIX openat/no-follow
primitives are available. Read regular
`.md` files only; never follow symlinks or traverse outside the resolved root;
open every root component without following links and retain that root handle;
skip `.git` and `.obsidian`; reject non-printable path components; and hard-cap
query size, traversed entries and
directories, files, per-file bytes,
aggregate bytes, matches, excerpts, and output while polling task cancellation.
Return only vault-relative paths and never mutate the vault.

Treat every excerpt as PRIVATE, local file-origin data: stamp `egress=false`,
spotlight-fence the text as untrusted before any model sees it, and float PRIVATE
onto later ReAct or explicit-plan synthesis calls. Route explicit phrases such
as `search my vault ...`, `read my notes ...`, and `check my vault ...` through
the existing SEARCH intent using a narrow `search_scope=vault` metadata flag,
then use the existing `research.local` synthesis step so raw fenced excerpts are
not spoken. Explicit SEARCH/RESEARCH vault requests remain locally scoped when
the tool is unavailable and never fall through to web search. Add the planner
tool only after a real vault registers, appending it
after the four ADR-0033 phone-native tools; do not expand that verified phone
schema in this change.

## Context / why

The voice agent needs direct access to Paul's new `dobo-brain/paul-brain`
knowledge layer without copying vault notes into speaker memory or granting an
LLM arbitrary filesystem access. A generic file-open tool, an unrestricted
recursive scan, or a write-capable Obsidian integration would create needless
path, privacy, prompt-injection, and sync-conflict risk. Sending scoped SEARCH
turns through `web.search` is also wrong: it bypasses the vault and its one-step
result path would speak raw tool output instead of producing a short answer.

The vault is private data but is still content, not trusted instruction. Notes
can contain copied web text or prompt-like prose, so file origin must never be
promoted into action authority. The existing §9.7 boundary permits an
explicitly invoked thinking tier to process files; PRIVATE routing keeps such
synthesis on the configured private chain and out of a public chain. Cloud
thinking remains separately opt-in, and this capability does not itself egress.

## Consequences

Linux can enable the feature with only a machine-local config override; other
POSIX devices with the same safe descriptor primitives can supply their own
vault path. Unsupported platforms, or a missing or unreadable clone, produce
no tool and no capability claim rather than falling back to race-prone path
opens. Completed scans use deterministic lexical ordering; once an entry cap
is hit, the response is explicitly partial and makes no ordering guarantee for
the unseen remainder. Returned content is excerpted rather than a full-note dump.
MiniCPM's accepted four-tool phone grammar remains unchanged, so native phone
planning cannot call the appended vault tool until a separate re-audit. This is
headless file/control-plane behavior; it requires no microphone validation and
does not prove live ASR phrasing quality.
