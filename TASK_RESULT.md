Valid until: main advances beyond this task's landing commit — then treat as
history.

# Task result — public conversation/STT fixture

## Outcome

Implemented matrix v4 and one reproducible, transcript/path/identity-free
logical lock for 96 private conversation/STT cases. The lock has four ordered
24-case sources: HarperValleyBank caller turns, EdAcc official test segments,
paired AMI ES2004a close/far windows, and a Common Voice SPS v4 32-to-24
projection. Audio, literal references, selected source identities, and
preparation receipts remain outside Git.

Each source now has an executable materializer that recomputes evidence from
the raw checkout/archive/audio. Receipts bind the catalog/license contract,
selection and eligible sets, exact preparation/decoder code, reference and
source-audio hashes, and output f32le PCM. Test-only raw-source injections emit
`production_evidence=false`; the production validator rejects them. Separate
production-shaped validator grammar fixtures exercise cross-layer consistency
but are not preparation evidence or proof of receipt authorship.

`tools.public_conversation_fixture` is both the validation-only command and the
model-suite entry point. With worker manifests it validates all four corpora,
passes them to the existing strictly sequential suite in lock order, and
revalidates the lock, receipts, manifests, references, and PCM after success or
failure before wrapping the aggregate report. The eager 32 MiB per-corpus cap
is unchanged.

## Main files

- Added the canonical lock and strict validator/suite envelope.
- Added HarperValleyBank, EdAcc, AMI, and Common Voice projection materializers
  with deterministic synthetic raw-source tests.
- Extended the Common Voice SPS v4 parent receipt with salted prompt digests;
  its accepted 32-case selection is unchanged.
- Extended the typed catalog to 18 sources and added the explicit
  CC-BY-SA-4.0 gate.
- Added ADR-0113 and updated status, agent routing/testing, and the public-data
  guide.

## Verification

- Documented focused gate: 195 passed in 13.77 seconds at low CPU/I/O priority.
- Mandatory APM/DTD regression: 6 passed in 0.69 seconds.
- Ruff passed on every changed Python module and focused test.
- All five relevant module `--help` paths exit successfully.
- `git diff --check` is clean.

The focused gate covers source parsers with synthetic raw inputs, exact lock
and catalog contracts, source quotas/pairing, Common Voice parent binding,
private publication, single-layer tampering, a fully rebound Common Voice
parent PCM/receipt/manifest rejected by archive re-decode, and the integrated
pre/post suite envelope. It performs no download and loads no speech model or
audio device. The same-owner private receipts detect inconsistent changes;
they are not signatures and coordinated reauthoring of every layer is outside
their trust model.

## Actual-source checks and limits

- Harper's pinned Git objects validated all 1,446 metadata/transcript pairs and
  8,000 lexical caller turns after marker-only exclusions. Deterministic
  selection produced 24 callers and three cases per task; all 24 selected WAVs
  decoded as mono PCM16 at 8 kHz (739,200 projected samples). A clean private
  checkout was unavailable, so the production corpus was not published.
- Current EdAcc test metadata exposed 9,289 rows and 1,845 eligible segments
  after key-matching the official filtered STM hard-WER reference;
  deterministic selection produced 8/8/8 cases, 24 speakers, and 23 accent
  digests. Only an initial current WAV header and metadata proxy were inspected.
- Exact pinned AMI annotation and far-WAV hashes were checked. The parser found
  2,614 words, then 22 isolated and four transition candidates after timed
  activity exclusion; the exact 8/4 selection had zero activity/window or
  selected-window overlap and projected 3,802,880 paired far-channel PCM
  bytes. Mix-Headset was unavailable, so no paired corpus was published.
- The complete 5.9 GB EdAcc archive and Common Voice archive/parent corpus were
  not materialized end to end.
- AMI Mix-Headset has no accepted shared digest in the catalog; a verified
  local SHA-256/size receipt is mandatory, so the close-channel raw identity is
  not yet predeclared cross-machine.
- No STT model, GPU benchmark, microphone/native reader, VAD, endpoint, AEC,
  barge-in, speaker authority, agent/tool, device latency, or live conversation
  evidence was produced. This fixture cannot change runtime defaults.

## Next

Acquire and prepare all four private corpora under their terms, then run
receipt-bound GPU candidates one worker/corpus cell at a time through the
validated fixture entry point. Compare aggregate accuracy and latency without
changing defaults. Promotion still requires Paul's fresh private recordings
and live open-speaker A/B.
