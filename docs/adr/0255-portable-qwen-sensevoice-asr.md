# ADR-0255: Add an explicit Qwen/SenseVoice CPU final-ASR option

Date: 2026-10-28
Status: accepted for explicit software integration; composed recording qualification pending

## Decision

Add `--final-stt-profile qwen-sensevoice-cpu` to the existing desktop core CLI,
launcher and doctor selection contract. It is optional, installed explicitly
and never selected by setup. Keep legacy default profiles, performance modes,
streaming model weights/decoder, endpoint/control policy, capability/egress,
identity/enrollment and existing authority fences unchanged. The new profile
requests two streaming/native ASR threads and two SenseVoice CPU threads; that
resource delta is explicit and must appear in qualification context.

Use Qwen3-ASR0.6B Q8 as a new complete-utterance **primary offline** recognizer,
with a distinct SenseVoice INT8 CPU voter. Qwen sees the full endpoint-owned PCM,
forces native CPU/thread2 and English without reference/context hints. It never
owns partials, barge-in, STOP, endpointing, speaker admission or tool authority.
SenseVoice forces CPU/English/ITN/thread2, with a fresh offline stream per input.
Do not count the same SenseVoice primary and voter as two independent models.

The component screen favoured Qwen's recorded English accuracy over the other
newly screened CPU models; it is not a production-selected result. Existing
SenseVoice remains a much faster/smaller baseline, and the new October Granite
multilingual export remains a research alternative. The feature adds one SDK,
not every benchmark runtime. Qwen's original trained weights are January2026;
this pins the measured converted weights and October SDK, without claiming new
October Qwen training. Full component evidence belongs to ADR-0254, not a
replacement runtime accuracy claim here.

## Assets and operational selection

`core/engines/_qwen_assets.py` owns a fixed Linuxx86_64 CPU artifact contract:
Qwen GGUF 841,502,336 bytes, SHA
`416e10c15b4a3d9002bd337d18fc450233fdf68502b6e10d1379d2789838afd0`,
and speech.cpp0.9.0 CPU `libspeech.so.3`, 6,001,680 bytes, SHA
`935cf967511af4d4baf8e80a865edbcfa70ec4b305aa00695ea9cc191aa03b93`.
The bounded `qwen-cpu.json` manifest references exact regular local files;
constructors only verify them and never download. SenseVoice model/token bytes
are independently pinned before native construction. Qwen weights retain
Apache2.0 attribution and speech.cpp code MIT; setup retains the exact upstream
model-card/license text. No HF gate, login or cloud inference is required.

`python -m tools.setup_portable_asr --config-root <root> --cache-root <verified-noncuda-cache> --bundle-dir <bundle> --sense-voice-dir <sense-dir> --install-profile`
reuses already verified model/library bytes by manifest reference, without
copying weights or replacing the canonical environment. `--download` is an
explicit alternative with bounded public downloads. Setup uses the pure
`core.config.add_qwen_sensevoice_profile` helper to add one complete optional
entry to an existing opaque two-profile local map, preserving all legacy
entries, selected/default fields, enrollment and unrelated configuration. File
publication is bounded/atomic with intervening-change checks; no output prints
configuration values. Select the option separately with the existing
`--final-stt-profile qwen-sensevoice-cpu` CLI flag. Existing two-profile maps
remain valid without provisioning this optional model, and their profile digests
remain unchanged. Unsupported OS/architecture refuses this new binary tuple;
no Windows/macOS native qualification is claimed. Existing choices remain.

## Selection and trust

Keep `_resolve_final_transcript` and `verify_asr_consensus` as the selection
plane. Qwen is explicitly quorum-required, like the existing NeMo primary:
without an independent exact normalized quorum it cannot replace the streaming
rendering. The baseline is not an extra vote. The unchanged empty-streaming,
control-change and decoded-empty-veto guards apply. Retain the historical
attested control repair only for the exact existing NeMo/Faster-Whisper family
pair. New Qwen/SenseVoice and other CPU voters do not inherit that exception.
No new STOP repair list or consensus relaxation follows from component scores.

Speaker/echo gates still see their original PCM domain. Any changed final text
retains the existing loss of direct-live/owner action authority. Full original
PCM is copied into per-request ownership; no research-only480000-sample/30s
limit or silent crop enters production. speech.cpp's complete-audio API owns
long-input chunking. Capture/endpoint bounds and callbacks remain unchanged.

## Native lifetime and failures

`_endpoint_lease.py` is a narrow owner for these new warm/decode/close adapters,
not a provider framework. One operation lock serializes warm/decode, and a
short state lock makes retirement reject every later entry. A queued operation
rechecks retirement after acquiring its operation slot. An entered native call
retains its exact handle until it returns. Result text is copied to a repr-safe
Python object before request/stream cleanup; no native view escapes the lease.

An active owner frees after its native work unwinds. Idle retirement launches
one exact cleanup owner so stop does not wait on a native destructor, and no
native free occurs under engine capture/receipt/final-verifier locks. Successful
free and confirmed cleanup-thread launch are both necessary for closed; failed
or ambiguous free/launch retains the exact incarnation process-wide and blocks
replacement. No entered request/free/model load is replayed after TypeError.
Temporary readiness probes obey the same retirement/replacement fence.

The engine/replay hold retired incarnations until true cleanup. Existing final
worker and capture epoch/effect fences remain; inline decoding and background
verifier warm-up are covered by the model's own lease. Start failure retires
only this attempt's newly published leased models; duplicate-start/early guard
rejection must not retire healthy prior models. Required voter warm failure and
partial Qwen load retain their exact cleanup owner. Old captured warm errors
cannot disable a replacement verifier. Unknown legacy model `.close` methods
are not invoked by these hooks. Native calls remain nonpreemptive; no new
native deadline or guarantee of immediate cleanup is claimed.

## Verification and open qualification

Source helper `27dbc5e5ba76fce45e4a21e8f5d59b8d6817bdad` was integrated as
`1ec813f`; its nineteen public/fake checks and independent review are retained.
The new Qwen/lease/Sense/profile/final-selection gate passed153 in4.66seconds;
independent review separately passed153 in3.87seconds. Covered full32s PCM,
caller mutation isolation, one request/free, partial load, incomplete results,
CPU settings, uncertainty, queued retirement, blocked warm/inline decode,
borrowed-reference/destructor ordering, partial start rollback, duplicate start,
replay restart, exact control-family fencing and legacy profile digests.

The larger targeted runtime/setup/source-inventory gate passed1058 with4skips
and identified three mechanical closure count/order mismatches. Prospective
inventories now bind the unconditional lease import; production final-factory
qualification additionally binds the lazy Qwen/assets/Sense sources. Historical
locks/receipts are not rewritten. Corrected source-count/import tests and the
actual installer/opaque-map regression are checked separately before commit.
No full-suite count from this stage replaces the historical full gate.

The next required evidence is actual `FileReplayEngine` production final
selection with the installed optional profile on Mic6 and Owner37 separately,
using the same two-CPU budget and source/model/config bindings. Report raw
streaming/Qwen/Sense and selected scores/control/drop cohorts honestly; component
wins cannot override a guarded selected result. It needs clean source,
independent review and the next serialized native grant. No microphone, speaker,
doctor execution, live/platform/default promotion, model experiment, merge or
push was performed by this implementation lane.

Affected source-inventory rerun passed172; its additional installer regression
first observed the global headless local-config-disable flag and was corrected
to enable only its owned fake local file. The assets/setup file then passed20
in0.18seconds. Thus the three inventory failures and real opaque-map installer
path are covered by successful focused reruns; the earlier17-file invocation
remains recorded as1058passed/4skipped/3failed, not retrospectively all green.

## 2026-10-29: composed compact result and one stronger CPU voter option

Actual production FileReplay on clean feature05e536c4 completed both original
corpora, each two repeats, with source/config/asset/original-PCM checks and exact
lease cleanup. Compact Qwen/SenseVoice selected errors were Mic6 3/25 legacy,
4/25 normalized, and Owner37 22/212 legacy,21/213 normalized in each repeat.
Raw offline models were better than the selected rendering, but exact quorum
and control guards retained disagreements and one STOP control guard per
cohort. This is not a selected-quality win over the older selected development
control (Mic6 3/25,4/25; Owner37 24/212,20/213). No repair list, thresholds,
authority rule or labels were changed to improve these numbers.

Frozen canonical compact reports are
`logs/runs/local-model-comparison-20261028/noncuda-asr/production-selected/mic6.json`
SHA `375d5ddedb60560708db2301371b68c7c8327e4c5e819e4e4b2ad349b3c8911e`
and `owner37.json` SHA
`f00e28dca5f29816a741e6a07523341d09b14ad6fecb8d343eb6199bdd09b864`.
Their combined source identity is
`6c9ad8792c6108805e15f5beb397966569885797fc999b55d633162c84897d21`;
original receipts and raw-versus-selected cohorts remain unchanged history.
Component-only and historical cross-load timings are not causal response
comparisons or physical/live qualification.

Add one explicit alternative, `qwen-parakeet-cpu`: the same Qwen CPU primary,
streaming-ASRthreads2 and independent CPU2 Parakeet TDT v3 INT8 voter instead
of SenseVoice. Keep the compact profile and all legacy/default selections.
Parakeet's existing four verified assets total670,478,772bytes; the companion
asset helper pins every file before construction. No new model download,
SDK/runtime, CUDA path, authority rule or extra concurrent acoustic model is
introduced. This trades a larger voter for a prospective exact-quorum quality
improvement; no selected win is claimed before its own composed replay.

The32-line voter subclass reuses the existing fresh-stream/full-PCM/resampler/
lease/cleanup implementation and changes only its fixed verified constructor:
`OfflineRecognizer.from_transducer`, CPU,2threads,16kHz,feature80,greedy,
maxpaths4,model_type `nemo_transducer`. It is not an invented
`from_nemo_transducer` Python API. A NeMo-family primary plus this voter is
refused to prevent double-counting the same model family. New Qwen/Parakeet
still receives no legacy NeMo/Faster-Whisper attested control exception.

`add_qwen_parakeet_profile` adds only its complete optional entry. Setup's
`--verifier parakeet --parakeet-dir <verified-dir>` validates and installs it
without selecting it, preserving any existing compact entry, enrollment and
legacy configuration. Legacy two-profile and compact three-profile maps remain
accepted. No unselected model must be downloaded. The lazy voter source is
added only to the prospective production-factory inventory, not eager-import
inventories or historical locks.

Companion helper74dea8c was integrated as4785c03, preserving both conflict-side
tests/evidence. New-pair focused/public-fake gate:237passed,1skipped in2.71s;
this covers new CPU constructor/fullPCM/retirement, fixed profile compatibility,
control fences, actual installer choices and affected factory-source inventory.
Independent source review is GO. Parent-owned full CI gate and then one clean
source-bound Mic6/Owner37 composed pair are still required. No further voter
or consensus experiment is authorized by this decision. Existing defaults,
physical/live/other-OS gates and compact05 evidence remain unchanged.
