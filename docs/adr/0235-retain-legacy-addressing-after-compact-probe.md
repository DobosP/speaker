# ADR-0235: Retain legacy addressing after rejecting a compact prompt

Date: 2026-10-10
Status: accepted

## Decision

Retain the exact legacy addressing system and user-prompt format for current
and explicit Qwen voice profiles. Reject the tested 492-character prompt plus
JSON-input candidate: it produced false ACT on all 16 standalone negative
cases, both through the complete classifier and forced native helper. Its CPU
mixed check also timed out once, and GPU mixed check produced two false ACT.
The unchanged legacy system is 1546 characters. Do not ship the candidate,
increase the three-second per-turn decision budget or add a startup warm API.

Expose the classifier's exact selected system prompt for reproducible diagnostics;
its strict `prompt_profile=current|qwen2.5-1.5b` selector currently retains the
same legacy policy. Preserve native enum choices/aliases, the 16-token cap,
cancellation, unavailable→INGEST and genuine semantic UNSURE. Do not infer
semantic competence from exact-format output or deterministic shortcuts.

Keep the Qwen answer profile explicit. CPU startup/context-switch latency and
broader model classification are unqualified. The current CPU device profile's
existing input_gate=false remains unchanged. GPU measurements qualify only the
recorded finite headless cases, not model semantics generally, default adoption,
full application READY, owner speech, or physical voice behavior.

## Context / why

ADR-0232 established substantially better finite public answer behavior with a
small model. Grouped resident decisions initially passed, but the first CPU
mixed-prefix check had an unavailable result. Reducing prompt size therefore
needed a quality and context check rather than a timing-only adoption decision.
The rejected candidate changed both system text and user serialization, so its
failure does not isolate their individual causes.

The restored production classifier was measured with its actual existing warm
sequence: answer generate("hi") then classify("hi", recent=()) at three seconds.
With four unpruned public recent lines in later cycles, CPU mixed answers and
decisions were 4/4, but maximum answer latency was 11.226 s; another CPU semantic
cell reached 14.410 s. Startup classification completed at about 3.02–3.05 s
with fail-closed INGEST, without attesting successful native prefill. Historical
aggregate reports do not identify the slow answer's ordinal.

The broader legacy CPU component matrix had 7/8 strict positives (one genuine
semantic UNSURE), 10/16 negatives and six false ACT; no negative used a shortcut.
Its contextual short-answer full classifier timed out, while a subsequent forced
helper succeeded. The context quote forced helper was unavailable. Four ambiguous
cases had no fabricated semantic reference; actual UNSURE/availability are counted
separately. These are model-component failures: the newly separate conversation
admission gate rejects all 16 idle negative fixtures before this model, so six
component false ACT do not establish six activated replies in the full runtime.

The actual-warm GPU legacy mixed control passed 4/4 answers and 4/4 decisions,
including recent context, with maximum decision latency 629 ms. A single final
benchmark-only direct prefill used the exact legacy prompt/enum/16-token settings
and a separate 20-second startup cancellation budget. It expired at 20.012 s.
Its subsequent CPU mixed cycles passed, with answers 1.31–1.37 s and decisions
1.65–2.70 s by ordinal. This does not justify a larger runtime warm allowance;
no successful bounded prefill was established. All temporary owned daemons were
stopped; unrelated user processes were not modified.

## Consequences

- The public benchmark selects the factory's actual classifier. Full-path labels,
  shortcut counts, forced-helper availability, genuine UNSURE, typed-scope behavior
  and per-ordinal mixed timings remain separate. It accepts no caller text/audio.
- Probe source, hashes, aggregate receipts and public diagnostic source are
  preserved outside task scratch. No transcript dumps enter Git; no source/model
  identity is relabeled to hide a failed cell.
- CPU cold/startup latency and model semantic gates remain open. No new model,
  prompt campaign, silent context pruning, cloud warm-up or per-turn deadline
  increase follows. Native main/vision, Windows/macOS, microphone, echo and
  owner live validation remain independent gates.
