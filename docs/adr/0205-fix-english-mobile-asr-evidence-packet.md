# ADR-0205: Fix the five-component English mobile ASR evidence packet

Date: 2026-08-14
Status: accepted
Refines: ADR-0098, ADR-0109, ADR-0117, ADR-0159, ADR-0161, ADR-0162,
ADR-0203, ADR-0204
Supersedes: none

## Decision

Fix one English-only, offline, no-download mobile ASR evidence-packet contract
as an exact private composition of five already defined component families.
Do not flatten the components into one generic streaming-STT corpus, resample
their PCM, rewrite their references, or merge their evaluation domains. The
accepted component envelope is:

| Component | Existing domain | License(s) | Logical groups | PCM leaves | f32le bytes |
|---|---|---|---:|---:|---:|
| Google Speech Commands v0.02 test + ECCC v1.2 | schema-v4 `command-assertion-v1` | CC-BY-4.0 | 57 | 57 | 3,714,904 |
| MInDS-derived speech + three-environment DEMAND domestic noise | schema-v2 `stratified-wer-cer-v1` | CC-BY-4.0 + CC-BY-SA-3.0 | 42 | 42 | 26,230,128 |
| NOTSOFAR-1 MTG_32006 far-field fixture | schema-v2 `paired-channel-wer-cer-v1` | CC-BY-4.0 | 18 | 18 | 2,309,120 |
| PriMock57 consultation-01 isolated fixture | schema-v2 `ordinary-wer-cer-v1` | CC-BY-4.0 | 3 | 3 | 284,668 |
| PriMock57 consultation-01 two-role overlap fixture | custom `two-utterance-min-order-wer-v1` | CC-BY-4.0 | 3 | 9 | 6,760,680 |
| **Fixed total** | **five separate domains** | **preserved; no redistribution** | **123** | **129** | **39,299,500** |

All leaves are mono 16 kHz float32 little-endian PCM. Their byte total is
9,824,875 samples, or 614.0546875 aggregate stored-leaf seconds. That duration
counts every retained overlap stem and mix leaf; it is a storage bound, not
unique speech time, a model-input denominator, or evidence for pooled RTF,
latency, or audio hours.

The NOTSOFAR component is only ADR-0159's 18-case far-field schema-v2
fixture. ADR-0194's 14-leaf natural-overlap bundle is a different private
artifact with a different metric surface and is not interchangeable with this
packet component.

Pin the exact five component identities, existing lock/manifest/receipt
bindings, schema or grouped-bundle kind, case/group order, leaf order, sizes,
hashes, sample format, license identifiers, and fixed totals in one committed
packet lock. GSC/ECCC, NOTSOFAR-1, and PriMock57 remain CC-BY-4.0 components;
the DEMAND derivative combines MInDS CC-BY-4.0 speech with CC-BY-SA-3.0 DEMAND
noise and retains its cache-only boundary. Packet storage remains private-cache
and redistribution stays false. The packet does not relax any component's
acceptance, privacy, redistribution, or evaluator rules.

The offline assembler grants success only when it accepts all five exact
retained inputs. It copies their admitted bytes into a self-contained private
root outside every Git worktree, under fixed component names and deterministic
order. It publishes one exact packet index and then a no-clobber terminal
receipt; the strict loader must rederive the packet/component/leaf totals and
all committed hashes before returning authority. A failed invocation may leave
its newly reserved output root or copied bytes visible, but without the exact
terminal receipt that root is incomplete, invalid, and must never be called a
packet or success. No-clobber recovery uses a fresh output path rather than
repairing or overwriting that root. The packet index, receipt, and command
output are aggregate and hash only: no transcript, local path, source member,
speaker, role, device identifier, or per-case hypothesis crosses that boundary.
The committed lock states `mobile_model_identity_authority: false` and
`evaluation_result_authority: false`; the index and receipt bind that exact
lock digest. Packet-level pooled WER is explicitly absent; this does not erase
the ordinary-WER authority retained inside eligible component domains, and no
global `ordinary_wer_authority` key is introduced.

Treat only the committed component lock, manifest, and receipt digests as input
authority. Recomputed component hashes are compare-only and can never be
supplied or learned through the production CLI. In particular, a newly
rematerialized DEMAND receipt requires review and a new packet lock rather than
runtime acceptance. The packet receipt is a runtime result, not component-
selection authority. Any future evaluator must pin both the exact packet-index
and packet-receipt SHA-256 values before consuming the private copy.

Keep evaluation outside this preparation contract. GSC/ECCC retains its
command-positive, monitored-negative, and command/noise metrics. DEMAND,
NOTSOFAR far-field, and PriMock isolated retain their own ordinary isolated-
speech WER domains. PriMock overlap retains only its separately locked grouped
diagnostic, including the custom two-utterance minimum-order surface; it is not
ordinary WER, ORC-WER, tcpWER, or tcORC. Never pool WER, error counts, exact-
match counts, latency, or RTF across the five components. A later runner must
publish one bounded result cell per component and state explicitly when a
metric is not comparable.

Accept the deterministic packet tooling and synthetic headless contract while
recording that no production packet exists in this slice. Host inspection
found the exact retained DEMAND derivative and all three pinned DEMAND 16 kHz
archives absent, plus the PriMock production source and both prepared PriMock
outputs absent. The exact production preflight therefore could not pass all-
five admission or reach output reservation/publication. Do not substitute
reports for corpora, regenerate a component from unlocked inputs, download
missing data, or publish a partial packet.

Leave the existing English mobile Zipformer asset tuple and application
default unchanged. The desktop fp32 GigaSpeech Zipformer and any CUDA
Faster-Whisper proxy are different comparators and cannot be labelled mobile-
model evidence. This slice authorizes no model run. The next separate slice may
add a hash-only provisioner for the exact mobile hybrid asset tuple and an
endpoint-faithful runner, but it must not gain quality, promotion, or default
authority from packet preparation alone.

## Context / why

ADR-0203 and ADR-0204 closed mobile ASR/listening and app-session ownership at
the Dart boundary, but deliberately added no acoustic corpus or quality
evidence. The repo already has useful retained English components, yet they
answer different questions: short command safety, domestic additive noise,
paired far-field recognition, clean conversational intervals, and synthetic
two-role overlap. A generic corpus union would discard schema-v4 command
assertions, erase PriMock grouping, or invite a statistically meaningless
pooled WER.

The fixed packet is therefore an evidence index and exact private copy, not a
new benchmark metric. Fixed counts and bytes prevent silent corpus growth;
component locks preserve the decisions already reviewed in ADR-0117,
ADR-0109, ADR-0159, ADR-0161, and ADR-0162. Requiring all five inputs prevents a
convenient retained subset from being reported as the accepted mobile packet.

The mobile app declares a different Zipformer bundle from the retained desktop
fp32 worker, and the exact mobile assets are not present locally. An idle GPU
does not change model identity, endpoint semantics, or evidence authority.
Running the available CUDA reference would produce a useful comparator only
after a separately accepted runner exists; it would not validate the shipped
mobile recognizer. The honest result here is a green deterministic contract
and an unavailable production materialization, not an invented quality cell.

External English and Romanian sources were researched but are not packet
inputs. Romanian remains opt-in and blocked on a separate decision that pins a
multilingual model/export, language label, tokenizer and Unicode
normalization, metrics, compatibility strata, private owner recordings, and a
physical-phone CPU/RSS/thermal/live gate.

## Consequences

- The next retained-data session has one exact, bounded English target: five
  components, 123 logical groups, 129 PCM leaves, 9,824,875 samples,
  39,299,500 bytes, and 614.0546875 stored-leaf seconds.
- Exact success is terminal-receipt-bound and private. A missing, mismatched,
  mutable, or partially retained component yields no valid packet and no
  success receipt; an incomplete reserved root may remain and is never resumed.
- Each component remains independently interpretable; no aggregate packet WER
  or cross-domain promotion score exists. Packet preparation has neither
  mobile-model identity nor evaluation-result authority.
- Headless synthetic tests can prove lock parsing, strict admission, exact-copy
  publication and reload through the shared authority-neutral lifecycle,
  0700/0600/single-link metadata, terminal receipt closure, no-clobber,
  production-loader isolation, extra/tamper/hardlink rejection, truthful post-
  link recovery, privacy, and fail-closed behavior. Their tiny explicit test
  authority remains unmistakably nonproduction.
  They do not prove the production packet, model quality, endpoint behavior,
  capture, latency, held-out or training-disjoint status, qualification,
  device behavior, or live speech.
- The currently retained GSC/ECCC and NOTSOFAR inputs do not authorize a
  partial result. Exact DEMAND and PriMock production bundles must be recovered
  or rematerialized from their already pinned sources before the production
  packet can exist.
- No download, source-data or packet materialization, decode/evaluation, model,
  GPU, Flutter plugin, playback/capture, microphone, audio device, physical
  phone, or live validation ran in this slice. Retained GSC/ECCC and NOTSOFAR
  bytes were opened only for bounded loader/integrity validation. The current
  English Zipformer/default and every runtime entry point remain unchanged.
- A later owner-recorded holdout must stay private, hash-bound, and disjoint
  between tuning and verdict clips. It remains separate from this public-data
  packet and from the later Romanian decision.

## Verification

- Exact task-worktree focused gate with cacheprovider disabled passed `19/19`
  (`0.56s` pytest, `0.80s` wall, `50,128 KiB` maximum RSS) with only two known
  unknown-timeout-config warnings.
- The seven-file adjacent command covering the shared corpus writer plus public
  command/noise, DEMAND, NOTSOFAR far-field, PriMock preparation, and both
  PriMock evaluators passed `264/264` (`10.98s` pytest, `11.46s` wall,
  `70,896 KiB` maximum RSS).
- APM/DTD synthetic PCM passed `6/6` (`1.53s` pytest, `1.98s` wall,
  `119,580 KiB` maximum RSS).
- Scoped Ruff check reported `All checks passed!`; Ruff format-check reported
  `2 files already formatted`.
- Independent semantic reconstruction printed `semantic-contract-ok` after
  checking the exact lock bytes/digests, component order, totals, production-
  API surface, metric separation, and false model-identity/evaluation authority.
- The exact ordered production preflight read and validated retained GSC/ECCC,
  then failed at the absent DEMAND input before reaching NOTSOFAR. It exited `2`
  in `1.415s`, emitted only
  `{"error":"mobile_asr_evidence_packet_prerequisites_unavailable","ok":false}`,
  and created no output root or packet.
- Independent final security inspection separately close-validated both
  retained GSC/ECCC (`57`) and NOTSOFAR far-field (`18`) roots against the lock.
  This was bounded read-only integrity/loader validation, not packet
  materialization or decode/evaluation.
- Three new-file no-index whitespace checks were clean; the unresolved-work and
  conflict-marker scan was clean; post-gate hashes were stable. Final independent
  security audit is GO on the exact hashes: its focused gate passed `19` in
  `3.79s`, Ruff was clean, and retained-source close-validation covered command-
  noise `57 cases/59 files` plus NOTSOFAR far-field `18 cases/20 files` with no
  transcript or PCM content crossing stdout/logs. It found no semantic,
  security, or maintainability blocker; `O_TMPFILE` is a deliberate Linux
  fail-closed requirement. The native documentation replay kept STATUS at 100
  lines, changed exactly nine documentation/handoff targets, and passed its
  whitespace, stale-language, unresolved-marker, link, and inventory checks.
- Frozen implementation patch: `101,958` bytes, SHA-256
  `4dc6a07a4bbdf30582fcc9d0c820617be8f966e644fd0625327019dad11b721a`,
  exactly three added paths. Lock: `8,506` bytes, raw SHA-256
  `268377ba9ef648647112210cd2c812cf9ea8e9783f366676dee1c4fe1dc07820`,
  recipe SHA-256
  `30b5b460a3a9fb7ff95f2c7f79f27e5f2c26a35b429effcb494315831d7225bf`.
  Implementation/test SHA-256:
  `7229d8df434eb6447d03da1047e9c0570ad40d6853f983d09e521724c8fb9002` /
  `0cddedf638e16e2747caf8c990796238b9060d0cd43077eb140847b7927dc673`.
- Exact production materialization: **not run** because the DEMAND derivative
  and PriMock isolated/overlap production bundles are absent. Consequently
  there is no production packet index, terminal receipt, model result, or quality
  claim.
