# ADR-0102: Add a receipt-bound Faster-Whisper endpoint benchmark

Date: 2026-08-01
Status: accepted

## Decision

Add Faster-Whisper 1.2.1 with CTranslate2 4.8.1 as a schema-v6, isolated,
final-only endpoint comparator in the ADR-0089 benchmark harness. Require one
private manifest to bind the exact Python 3.12 package tree and the complete
local CTranslate2 model tree through separate bounded content receipts. Launch
the worker with `-I -S -B` inside a Bubblewrap namespace with no network, a
read-only runtime/model/source closure, receipt rechecks before ready and after
close, CUDA device mounts, one CTranslate2 worker, one CPU thread, and
one-thread numerical-library environment settings. Buffer the bounded replay
case, call the candidate exactly once only after complete PCM is available, and
emit a final with no partial hypotheses. Treat declared tail padding as replay
pacing only, not candidate decode input. Require owner-private receipt roots
and materialized, single-link regular-file closures; reject symlinks and hard
links. Before creating output, require the exact Python 3.12 directory and
3.12.3 marker, reject overlap among output/runtime/model roots, and bind fixed
inputs. Create the manifest through its already-opened parent descriptor, with
no-follow exclusive creation, file and parent synchronization, and final
descriptor/name identity checks. Independently reject partial events and a
nonzero final sequence at both worker and supervisor boundaries. Keep the
adapter unavailable to setup, application entry points, and runtime
configuration. Retain numeric zero deadline/backlog placeholders only at the
existing worker protocol boundary, then publish those aggregates as null with
an explicit not-applicable marker; only complete-PCM-to-final decode latency is
meaningful for this comparator.

## Context / why

Faster-Whisper Small is already an opt-in final verifier, but its recording
evaluator does not share the isolated public-corpus/provenance boundary used to
compare newer recognizers. Pretending its repeated offline decodes are online
partials would overstate natural-conversation and streaming performance. Using
the production environment or a model-directory path without a tree receipt
would also leave dependency/model changes invisible between runs. The separate
runtime and model receipts allow multiple already-downloaded CTranslate2 trees
to be compared without downloader access or candidate imports during
provisioning.

## Consequences

Future private GPU runs can compare final transcript accuracy, endpoint-to-final
decode time, determinism, and aggregate process resources against the same
public corpus and evaluator provenance as other candidates. The report labels
the recognizer as final-only, requires an external complete-PCM endpoint, and
records that streaming deadlines are not applicable and VRAM and host-memory
hard limits are not independently enforced.
Provisioning hashes the full local package tree, so it can be slow and must run
low-priority on owner-critical storage/CPU. Package managers that install with
hard links require a separate materialized copy before provisioning.

This change executes no real model and supplies no WER, latency, GPU, live,
streaming, capture, VAD, endpoint-detection, AEC, owner-voice, multilingual,
thermal, or adoption evidence. SenseVoice, the opt-in Parakeet/Faster-Whisper
final pair, `./live.sh`, setup, and the single application entry point remain
unchanged. Promotion still requires disjoint recordings, synchronized
capture/AEC evidence, multi-device resource checks, and fresh bare-speaker live
A/B evidence under ADR-0092 and ADR-0100.
