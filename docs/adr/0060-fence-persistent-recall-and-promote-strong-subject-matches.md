# ADR-0060: Fence persistent recall and promote strong subject matches

Date: 2026-07-12
Status: accepted

## Decision

Refine ADR-0054 without widening its controller grammar. Keep SQLite `all()` as
a bounded process-local working window, matching the Postgres adapter; retain
durable SQLite rows for `search()` and `context_for_llm()` only, so prior-session
text reaches a fresh process solely through the untrusted-memory fence. Keep
`meeting`-tagged notes in that process ring only and never write them to SQLite.

For fenced recall and last-session blocks, compute one backend-neutral routing
scalar before wrapping: a question is a strong match only when one recalled line
covers every substantive subject term after bounded scaffold removal and small
inflection folding. A nonempty block, stopword hit, or terms spread across
unrelated lines is not a match. Publish only the boolean, never raw memory, to
the tier router. Promote strong matches to the main tier at the default/desktop
threshold; retain the phone profile's shared-MiniCPM low-power threshold. Keep
sensitivity floating, spotlighting, planner/tool boundaries, and the exact
self-scalar controller unchanged.

Make the autonomous memory probe close and reopen a temporary SQLite backend
and create a fresh capability registry before asking for a non-PII canary. A
green model result requires durable availability, fenced prompt injection,
main routing, no native prior-session history, no controller authority, distinct
shipped model roles, and a grounded answer. Verify that the canary lies between
one balanced nonce-fence pair. Echo reports only an incomplete diagnostic for
the deterministic persistence/fence/routing plumbing, never a semantic PASS.

## Context / why

MiniCPM's historical preference failure was not selector failure: the fact was
present in its prompt, but it returned persona/privacy disclaimers five of five
times. ADR-0054 safely made that exact live self-scalar controller-owned, but its
probe then bypassed both models and could not prove general or restart recall.
Ordinary remembered subjects had worked, and Gemma answered the troublesome
preference case, so a conservative hybrid promotion is smaller and safer than
parsing arbitrary recalled text into trusted controller state.

Blanket promotion remains rejected. Keyword recall can render a favorite-color
row for an unrelated capital question through stopword overlap, profile content
can appear broadly, and a first-turn recap is not necessarily relevant. Requiring
all subject terms in one line keeps those cases on MiniCPM.

The audit also found that SQLite reread durable rows through `all()` after reopen.
Recent-context assembly then converted prior-session `user` rows into native chat
messages even with recall disabled, bypassing the spotlight fence. A process-local
ring restores the documented recent-versus-past authority boundary while leaving
durable retrieval intact.

## Consequences

- Fresh-process SQLite rows cannot inherit current-session user/tool authority;
  meeting notes remain process-local, while persistent recall stays opt-in,
  bounded, sensitivity-floated, and fenced.
- Strong remembered subjects use Gemma on desktop; weak/non-question/no-subject
  matches retain MiniCPM. The exact private self-scalar controller still works.
- Pure and capability tests cover strong, weak, cross-line, non-question, phone,
  stale-context, SQLite reopen, native-history absence, and untrusted fencing.
- Echo can diagnose plumbing without claiming a complete pass. The final real Ollama
  memory probe and production-hybrid conversation A/B remain release evidence.
