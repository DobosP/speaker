# ADR-0074: Broaden explicit personal-vault phrasing

Date: 2026-07-16
Status: accepted

## Decision

Retain ADR-0073's default-off, bounded, read-only `vault.search` capability and
all of its path-containment, no-follow, PRIVATE-data, untrusted-content,
cancellation, output-cap, and no-mutation constraints. Replace its narrow
example-led speech rule with one deterministic personal-vault source grammar.
Clear lookup or navigation requests that name the configured personal source,
including `search in my vault`, `go in my vault`, `find in my vault`, and
equivalent notes/Obsidian/dobo-brain forms, use the existing SEARCH intent with
`search_scope=vault`. Topicless forms produce the existing bounded vault
listing; topical forms rank on the requested topic rather than routing words.

Resolve source grammar in textual order. An explicitly excluded personal
source stays excluded, an explicit public source selects public search, and a
later unambiguous correction can select the private source or public source.
Treat correction-like or exclusion-like words inside an explicit topic slot as
topic text, not control syntax. Remove only recognized source-selection glue
from the vault query: corrected and topicless local requests become bounded
listings, while real topic words such as `online`, `internet`, or action verbs
remain available for ranking.

Keep explicit SEARCH/RESEARCH source choice independent of provider
availability: those turns fail closed locally when the tool is disabled or
unavailable. Other natural lookup wording uses the vault when registered and
otherwise retains the existing PRIVATE assistant route; neither path may fall
through to web search. An explicit web/online source override remains
non-vault. Do not reinterpret declarative mentions of the vault, generic
Obsidian/vault questions, confirmed `open`/`run` commands, or dictation as vault
lookups. Status/capability questions and requests containing a separate action
or mutation clause receive no private read tool. Topic infinitives such as
searching for tools to edit files remain reads, while a found note/result to be
edited remains an action request and fails closed. Enforce the same source and
lookup classification in deterministic plans and text/native ReAct allowlists;
do not rely on model obedience to hide `vault.search`. Keep the portable
intent/mode contract and MiniCPM's verified native first-four tool schema
unchanged.

## Context / why

ADR-0073 made the data path safe but documented only a few leading forms such
as `search my vault`, `read my notes`, and `check my vault`. Natural voice turns
also put the source after a preposition or use a navigation verb. In particular,
`search in my vault`, `go in my vault`, and `find in my vault` must describe one
local operation rather than depend on an LLM or accidentally select a public
search path. Treating every occurrence of `my vault` as a lookup is too broad,
however: statements about the vault and explicit web requests need their
ordinary conversational or public-search behavior.

A new intent enum or a general filesystem tool would expand the cross-platform
contract and authority surface without adding useful behavior. A lexical,
source-scoped refinement keeps routing deterministic and preserves the bounded
reader adopted by ADR-0073.

## Consequences

Users can phrase the same private lookup naturally without memorizing one
leading verb. The analyzer, assistant-mode final preview, deterministic planner,
query cleanup, and ReAct source guard share one local/public/non-lookup result.
Explicit local requests cannot acquire web tools, public requests cannot
acquire the vault tool, and status or action requests receive neither private
read path. Topicless navigation must not leave verbs such as `go` as ranking
terms. The refinement remains headless control-plane behavior; it does not
validate microphone transcription or physical audio behavior.
