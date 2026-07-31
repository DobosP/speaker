# ADR-0085: Bind public voice evidence to exact licensed inputs

Date: 2026-07-31
Status: accepted

## Decision

Version the fetch-on-demand public voice recipe as strict schema v2. Accept
only source records with an allowlisted SPDX license identifier, exact HTTPS
canonical/download/license URLs, version, size, SHA-256, attribution, citation,
voice-data class, and cache-only or redistributable policy. Reject duplicate
JSON keys, non-finite numbers, unknown fields/tags, hidden or case-fold
duplicate filenames, unsafe extractor members, malformed annotation/group
relationships, redirects, and any source changed after resolution.
Extract only from private, hash-verified on-disk source snapshots kept for the
entire suite; use reflink-first/file-copy fallback and clean them afterward.
Derive AMI speaker identity from member names, require unique identities, and
prove expected text as a contiguous normalized token sequence.

The accepted v2 data is Google Speech Commands v0.01 test data plus AMI ES2002a
and AMI manual annotations 1.6.2. CMU/Festvox and Microsoft AEC Challenge
inputs are excluded until an exact release, download hash, and applicable
terms are resolved. Non-commercial or mixed-term audio such as SLURP and
Audio2Tool is schema inspiration only.

Bind every `tools.public_stt_eval` report to the exact manifest bytes, generated
metadata bytes, complete source records, every generated case hash, logical
model-directory contents, decoder implementation hash, dependency versions,
fixed production decode configuration, and a versioned evaluator identity.
The evaluator binding contains exact hashes for the public evaluator,
aggregate metric implementation, and WER implementation and records whether
the result is a production-decoder report. Decode eligible audio only from
in-memory bytes read and hash-checked together, capped at 8 MiB per case and
32 MiB cumulatively. Load the recognizer
only from a private read-only on-disk model snapshot whose complete fingerprint
matches the source, and keep it through recognizer destruction. Reject
duplicate, missing, extra, or malformed metadata cases. Decode only `transcript` and
`empty_reference`; still validate/hash all excluded cases and report excluded
assertion counts. Call nonempty decoder text for an empty reference
`nonempty_transcripts`, not an agent false activation. Raw stripped text owns
empty/nonempty counts; normalized text owns WER/CER.

Keep this evaluator classified as aggregate whole-clip decoder smoke. It is not
frame replay, streaming, AEC, session/action, speaker-role, held-out,
owner-voice, live-hardware, or conversational-latency evidence. Its production
recognizer contract is Linux, NVIDIA CUDA device 0, and float16.
Reject recognizer injection by default and at the CLI boundary. An explicit
test-only override binds its actual callable source as non-production and does
not inherit production runtime or decoder-configuration claims.

Represent product phrases as text-only contracts using actual current runtime
identifiers. `stop` and `cancel` both map to STOP/control.stop; the three vault
forms map to scoped vault search and its planner steps; the reminder maps to
the typed reminder dispatcher; and the timer maps to the local intent grammar.
Negated controls assert only that semantic STOP is absent. Mark `wait`, `hold
on`, and bystander-role behavior as future taxonomy with no runtime claim.

Distribute the existing adapted FSDD fixture arrays under CC BY-SA 4.0 with a
fixture-local notice containing project and creator attribution, license link,
and changes. Record the exact historical source revision as unknown rather
than inferring a later release. Provenance remains incomplete, but the known
ShareAlike terms are explicit.

## Context

The draft recipe mixed resolved and unresolved sources, allowed free-form
license expressions, followed redirects, and did not bind decoder results to
one exact manifest, metadata file, complete case set, or model tree. The
evaluator silently skipped malformed and unsupported rows and labelled raw
silence transcripts as false activations. Its product scenario action names
also did not match the implementation.

AMI laughter annotations can contain both a vocal event and words. Treating
that window as a blank/no-action reference would convert valid speech into a
false error. Event-only grading with retained annotation tokens preserves what
the source actually proves.

Historical FSDD-derived arrays were incorrectly described as MIT. Their known
upstream and adapted-material license is CC BY-SA 4.0; the missing source
revision must not erase known attribution and ShareAlike obligations or be
filled with an invented tag.

## Consequences

Preparation and evaluation remain deterministic and headless, but accepted
acoustic coverage is intentionally smaller. There is no accepted AEC suite in
v2. Report hashes make two decoder-smoke results comparable only when corpus,
model tree, decoder binding, and evaluator binding match exactly.

Private snapshots avoid verification-to-use races. They prefer filesystem
reflinks but can temporarily consume the selected sources' total logical size
during preparation and the model tree's full logical size during evaluation.
Operators may select a model snapshot disk; snapshots are cleaned after use.

The product scenarios now catch drift in current deterministic text routing
without overstating audio behavior. A later branch still needs timestamped
frame replay, licensed AEC pairs, target/bystander audio, a disjoint owner set,
and fresh `./live.sh` validation on the physical route.
