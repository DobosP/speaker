# ADR-0189: Fence retained context from model and web egress

Date: 2026-08-12
Status: accepted
Refines: ADR-0003, ADR-0060, ADR-0073, ADR-0074, ADR-0076, ADR-0187
Supersedes: none

## Decision

Keep sensitivity selection and retained-context egress admission as separate
decisions. A current post-ASR user turn, including one classified PRIVATE,
remains eligible for the configured sensitivity-selected US cloud-model chain.
The existing web raw-query and sensitivity gates remain stricter for web
search. Sensitivity chooses a permitted model chain; it never grants permission
to send enumerated dynamic retained context to a provider.

For every assembled Assistant, research, and ReAct main-tier call, carry the
assembly-owned `CloudEgressScope` shared from `always_on_agent.models` in
`CLOUD_EGRESS_SCOPE_CONTEXT_KEY = "cloud_egress_scope"`. Its exact enum members
have values `current_turn_only` and `local_only`; `core.llm` re-exports the
shared symbols. Only an exact typed enum member is a valid marker; the
corresponding raw strings, coercible objects, missing values, wrong types, and
unknown values are not provenance. A factory-managed `HedgeLLM` must treat
every missing or malformed scope as `local_only`.

At the assembled capability boundary, mint `current_turn_only` only when the
scope key is absent. Capability entry must never overwrite or upgrade a present
`local_only` or malformed scope. Canonical production task contexts are fresh;
a direct caller that reuses a context may therefore remain conservatively
local. The scope is monotonic for the whole call lineage: it may move only to
`local_only`, never back. Before any model stream is created, downgrade it when
the actual prompt gains any locally retained recent role history, composed
recent text, planner recent block, or nonempty prepublished legacy
`recent_conversation`; recall, profile, last-session, or recalled screen-memory
text; a nonempty procedural-rule block; or a PRIVATE local-tool finding. ReAct
must downgrade the same turn before its next planning or final call after a
PRIVATE result, and deterministic research synthesis must do so before folding
a PRIVATE local or vault finding into its prompt. Procedural rules remain
trusted instructions for model behavior, but that trust does not authorize
disclosure to a cloud provider.

The same shared module owns
`SYNTHETIC_RESUME_TAIL_METADATA_KEY = "synthetic_resume_tail"`. An accepted
`ResumeTracker` resume is retained-context synthesis, not a current-turn-only
prompt: it embeds the prior user query and the assistant tail that was actually
spoken. `core.runtime` stamps the synthetic final with exact built-in `True`;
the Supervisor propagates that marker into the exact created task and then
clears it from controller state; capability entry downgrades that task before
its first model call. Generic post-barge response-only text and ordinary current
user text carry no marker and remain eligible unless another listed source
downgrades them.

It also owns
`RETAINED_PROMPT_CONTEXT_METADATA_KEY = "retained_prompt_context"`. The runtime
mints exact built-in `True` only when an accepted transcript-cleaner rewrite is
substantive and the cleaner received nonempty bounded prior-user context. The
bound is the newest four exact canonical user-tagged utterances. Substantive is
defined by a strict Unicode-preserving equivalence: NFC plus case-folding may
ignore spacing and ASCII punctuation, but any other content difference is
retained-prompt provenance. A case/spacing/ASCII-punctuation-only rewrite is
deliberately eligible, as is a token-changing rewrite made with no prior-user
context. This is a token/content provenance claim, not literal byte
noninterference.

The Supervisor treats presence of the reserved retained-prompt key as
restrictive, monotonically carries exact `True` through published-unheard,
reservation, extension/coalescing, merge, queue/replacement, and folded
continuation lineage, copies it into the exact created task, then clears ambient
turn state. Assistant and research capability entry treat present retained
provenance, a non-exact metadata container, or non-string metadata keys as
`local_only` before model selection; unrelated exact-string metadata is
ignored.

`web.search` uses the same top-level scope as a restriction before its
classifier or backend. An absent scope preserves canonical deterministic
SEARCH/RESEARCH and direct-call compatibility. If scope is present, only exact
typed `current_turn_only` may proceed to the existing sensitivity and raw-query
gates; `local_only` or any present malformed scope vetoes locally. A non-exact
metadata container, non-string metadata key, or presence of the reserved
retained-prompt key also vetoes; unrelated exact-string metadata is ignored.
Thus a ReAct-generated public-looking query cannot egress after recent, recall,
procedural, cleaner-derived, or PRIVATE local-tool context has downgraded the
turn. Exact `current_turn_only`, when present, is restrictive-only and never
sufficient for web egress.

An explicit file/image attachment and ambient/current image retain their
existing thinking-tier policy. A public finding already returned by the
web-search egress path likewise does not by itself downgrade model scope. These
inputs still obey their existing sensitivity, tool, and provider rules; this
decision does not create new authorization for them.

Every factory-created multi-chain and single-cloud `HedgeLLM` must snapshot the
exact scope eagerly for each call and choose the local leg before constructing
the shared queue, any worker thread, any member's provider stream, or other
cloud work when it is `local_only` or invalid. This applies to hedge and
fallback strategies and every member of a provider chain. The monotonic scope
remains `local_only` across cross-tier retry and the complete nested
ReAct/research call lineage. The high-confidence outbound redactor remains a
last-line scrub, not the admission decision.

This contract covers the canonical `build_llms` plus `build_runtime` assembly,
its factory-managed Hedges, and its registered `web.search`. Direct custom
clients, a directly constructed unguarded router or hedge, raw
`OpenAICompatLLM` or backend calls, and arbitrary same-process code remain
outside the claim. Python object identity is not a process sandbox: another
holder of the context, shallow alias, or provider can still mutate or bypass
it. Static configured system/persona/capability-manifest text is not one of the
source-derived downgrades and remains cloud-eligible under existing policy.
The wider compositor is not globally hook-free; only the documented exact
snapshot/admission seams make that narrower claim. `HedgeLLM.last_source` is a
shared diagnostic and can race across concurrent streams. An unexpected
private-tool exception string that lacks source tagging is not covered as a
PRIVATE local finding.

ADR-0097's media topology is unchanged. This decision changes no audio, STT,
TTS, device, or trusted-LAN authority.

## Context / why

ADR-0187 made floated turn sensitivity a monotonic web-search veto before a
generated query could hide the PRIVATE material that caused classification. It
left source-derived retained prompt context and the cloud-model path open. A
ReAct turn can acquire recent/recall/procedural/private-local context, generate
a public-looking web query, and still need a veto even when its sensitivity is
not enough to express that provenance. On the model path,
`SensitivityRouterLLM` selects a jurisdictional chain, then `HedgeLLM` may start
its cloud members. PRIVATE currently selects a US chain rather than the bare
local model.

The Assistant composes more than the current utterance. Default-on recent
history, optional episodic recall and profile facts, cross-session summaries,
recalled screen text, procedural rules, and local tool findings can all enter a
main-tier prompt. ReAct can discover PRIVATE vault text only after an earlier
model step. Classifying the current query therefore cannot prove that the final
assembled prompt contains only current-turn material.

The transcript cleaner is another compositor. Its bounded prior-user context
can supply words that appear in an accepted rewritten `final_text`, after which
the Assistant otherwise sees a fresh current turn. Synthetic resume prompts
similarly embed the prior query and assistant tail actually spoken. Both need
explicit provenance that survives task and continuation boundaries.

Why not inspect the final system prompt with `may_leave_device`: that predicate
is a heuristic raw-query gate, while the static assistant prompt itself uses
ordinary possessive language. Whole-prompt classification would both overblock
normal cloud calls and retain heuristic false negatives. The compositor knows
the source of each injected block deterministically, so provenance is the
smaller and stronger boundary.

Why not make every PRIVATE turn local: ADR-0003 and ADR-0187 deliberately retain
the configured US chain for current post-ASR PRIVATE text. Reversing that product
choice would be a broader decision and would make the private cloud chain's
role ambiguous. This ADR closes the injected-local-context gap without silently
changing current-turn routing.

Why not exempt procedural memory: instruction trust answers whether the model
may follow a rule, not whether a third party may receive it. A standing rule can
contain a preferred name, relationship, location, or other personal detail, so
its presence is a deterministic local-only restriction.

Why not mark every cleaner byte change: case, spacing, and ASCII-punctuation
cleanup does not import retained semantic content, and a context-free rewrite
has no prior-user source to disclose. The Unicode-preserving equivalence keeps
that narrow compatibility without allowing non-equivalent CJK, emoji,
combining, bidi/control, or other non-ASCII content changes to disappear under
an English-only token normalizer. Deliberately ignoring case, spacing, and
ASCII punctuation leaves a byte/covert channel outside this token/content
provenance claim.

## Consequences

- A current turn without an enumerated dynamic retained source can retain the
  existing fast/local or sensitivity-selected main/cloud behavior. Once one of
  those sources downgrades the scope, main-tier reasoning uses the bare local
  leg for the rest of that lineage.
- Default-on recent history therefore makes contextual follow-up reasoning
  local. Composed and prepublished legacy recent context, recall, profile,
  last-session, recalled screen memory, procedural memory, substantive
  contextful cleaner rewrites, synthetic resume tails, and PRIVATE vault/local
  findings cannot start a factory-managed cloud worker even when their text
  evades PII heuristics.
- Retained-prompt metadata and any present non-current top-level scope also keep
  canonical web search on the local corpus before classifier/backend work.
  Absent scope retains canonical deterministic SEARCH/RESEARCH and direct-web
  compatibility.
- Public already-egressed web findings and explicit current-turn files/images
  keep their existing policy, as do ambient/current images and static configured
  system/persona/capability-manifest text. Changing them requires a later ADR.
- Cloud-disabled configurations, the fast tier, local router/classifier roles,
  startup warm-up, local visual captioning, and remote local `/chat` remain
  unchanged. Both multi-provider and single-cloud compatibility paths need the
  same fail-local scope check.
- Direct-call compatibility is not evidence for assembled production. A reused
  direct-call context is never reset to cloud-eligible, and the typed marker is
  not authenticated against malicious same-process code.
- The case/spacing/ASCII-punctuation channel, static configured prompts/current
  images, non-hook-free composer surfaces, shared `last_source` diagnostic race,
  and untagged unexpected private-tool exception strings remain explicit
  residuals/nonclaims.
- Deterministic fake-only tests must prove exact-type admission, missing and
  malformed fail-local behavior, monotonic downgrade across retries and ReAct,
  every listed dynamic source, no cloud-worker preallocation/start, cleaner and
  continuation marker lifecycle, pre-classifier/backend web veto, context
  isolation, and unchanged eligible-current-turn behavior. Executable current
  image coverage is positive; no file-path test is claimed. The frozen receipts
  and audits below establish these headless contracts without exercising a
  network, cloud provider, model, GPU, audio, microphone, device, or live path.

## Verification

The frozen low-priority/no-cache command prefix was
`/usr/bin/time -p env SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19
/home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider`.
With only the paths named in `docs/agent-testing.md` appended, the final
receipts are:

- focused changed surface: `262 passed in 6.09s` (`real 6.24s`, `user 1.31s`,
  `sys 0.20s`);
- exact registered cloud stage: `275 passed in 9.91s` (`real 10.08s`,
  `user 1.14s`, `sys 0.13s`);
- declared adjacent: `101 passed in 1.57s` (`real 1.73s`, `user 0.74s`,
  `sys 0.06s`);
- broad policy/context/ReAct/vault: `1196 passed in 1.91s` (`real 2.09s`,
  `user 1.62s`, `sys 0.20s`);
- final-preprocessing plus post-barge resume integration: `103 passed in
  7.77s` (`real 7.92s`, `user 1.09s`, `sys 0.18s`);
- import smoke: `314 passed in 1.88s` (`real 2.15s`, `user 1.94s`,
  `sys 0.13s`); and
- APM/DTD: `6 passed in 0.72s` (`real 0.90s`, `user 0.78s`, `sys 0.11s`).

All 13 changed Python files parse as ASTs. Scoped Ruff
`E9,F63,F7,F82`, `git diff --check`, the exact 100-line `STATUS.md` check,
pre/post SHA-256 comparison, and the exact 19-path inventory are green.
Unrestricted whole-file Ruff retains 11 inherited `E402` findings in
`always_on_agent/supervisor.py` and one inherited `F401` in `core/runtime.py`.
The formatter was not clean: managed policy rejected its three task-owned
mechanical suggestions at `always_on_agent/supervisor.py:461-463` and
`core/runtime.py:2220-2223,2299-2301`, so it changed no bytes. No
full-format-clean claim follows.

The exact 19-path inventory is the 13 Python paths hashed below plus
`.agents/backlog.md`, `STATUS.md`, `docs/agent-map.md`,
`docs/agent-testing.md`, `docs/unified_architecture.md`, and this ADR. Frozen
Python SHA-256 values are:

- `always_on_agent/models.py`:
  `a2df6d403d4ee744349cff729b928cbc7237d78bb1b03facd20a9f8b338eb4ec`
- `always_on_agent/react.py`:
  `0da1993cefbe789b12f8ebc700462ca34c3d2943242eafe889e40fde8cc3eb34`
- `always_on_agent/supervisor.py`:
  `5f03c18fe7433d59cd1842ebdb6ca59edd9a0e8c784b59183a8c4a899df950cb`
- `core/capabilities.py`:
  `fc0af01990044b051768fd863b178e615b0015f06a32155cc9487b14d64e3193`
- `core/llm.py`:
  `79532989eb5d99f552c515c62b0c63476ea64f9cce766c1e0969d95e293e8e00`
- `core/llm_factory.py`:
  `b95ea2ee067319014e92cbd9beefe6fbfdc68b08740ee7e27bbaaffb335af016`
- `core/runtime.py`:
  `f7dd4103ef213d8cd2a34ed82a59bc99c0e4b9a7ed945efb8d88b656ad891b47`
- `core/websearch.py`:
  `3cef3b7fca4da26d12a8d5eb35592d189e71cdfcca0d4983b4c8bdd127334c0b`
- `tools/testing/stages.py`:
  `46fc79bbda3a2751e475455ef17c1fa6ba389e23030d29accedcc42f5c70c63e`
- `tests/test_cleanup.py`:
  `3fcec37d3ebae0716feb4e45b329fdbd3dc014f7e790ddb5379514b7e0e6f677`
- `tests/test_final_preprocessing_cancel.py`:
  `f844160730e6c9dc17c4c6b5f39db187d048cc2429e899fd400a4ec111c447ef`
- `tests/test_websearch.py`:
  `4139e2d7f273669480db94a261ea966b02a2012fa6f6b7166c7974b5c05d55c1`
- `tests/test_llm_egress_policy.py`:
  `0b4e13f68cbfcc35df8677973e51a7fac0247f3f0901c256f0b0575ba264ead9`

Independent security and architecture/adversarial reviews are GO on those
frozen bytes and this claim/residual boundary. The verification was entirely
headless and fake/local: no network, SearXNG, cloud-provider, model, GPU, audio,
microphone, device, or live path ran.
