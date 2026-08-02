Valid until: main advances beyond this task's landing commit — then treat as
history.

# Task result — public short-command/noise gate

## Outcome

Added the first deterministic public gate that scores Speaker's short-command
failure mode directly. Public matrix v5 now has 8 tracks, 19 selectable sources,
and 11 exclusions; it uses the official Google Speech Commands v0.02 test
archive and adds the Sheffield English Consistent Confusion Corpus v1.2. The
existing 96-case conversation lock was redigested at matrix v5 without changing
its four source recipes or slots.

The new no-download preparer publishes one private schema-v4 corpus with 57
isolated clips: 26 command positives, 26 non-silence command negatives, and five
silence cases. Positive and forbidden command targets are normalized,
duplicate-free, disjoint, and reference-consistent at both writer and loader
boundaries. Aggregate metrics distinguish final and partial recall, target
precision, target-pair and case false positives, documented confusions, silence,
and false-positive clips per monitored negative audio hour.

No runtime model, setup selector, `core/` path, `./live.sh`, enrollment rule,
agent tool, or default changed. ADR-0117 retains both evaluated recognizers as
development comparators and rejects default promotion.

## Exact corpus evidence

- Corpus manifest: 57 cases, 928,726 samples/58.045375 seconds, SHA-256
  `50ec2c1ec7ad7ef2ca58d56d3330ca1a496db3784b78d41be9c8804413c552b9`.
- Preparation receipt: SHA-256
  `bf6cf442e4fb02e489dc7ceef763df3b39c1a4ef2178a5f8f2d2dab664517ac2`;
  PCM set `b0eeb32c314b5d9b4078fa640331378f8d9e31372436aca178dfdef4c7fbadd4`.
- Recipe lock: SHA-256
  `1ae5a5845a70b69aa7e1d32e6f799566e2c715f8202b2d86760a6cef73d51564`;
  source set `3f74cabd5b827ef2c4ef7afc628b3ce0eb5e237fe672d1555b1357940566d246`.
- The real preparation completed twice deterministically; the fresh audited run
  published all 57 cases in 6.4 seconds with identical hashes.
- Aggregate-only committed evidence is
  `docs/evidence/2026-08-02-public-command-noise.json`. It contains no selected
  speaker/member identity, local path, per-case transcript, or model output row.

## Real model findings

Faster-Whisper Small ran on CUDA float16 with beam 5 and final-only decoding.
Its current-source report is
`4aed945464d51718512ce799bf76f2428f058cfa2fd0c75b5e314d17fdd095e3`:

- Google Speech Commands: WER 0.10, 18/20 command recall, zero monitored command
  false positives; four of five silence clips had non-command text.
- ECCC: literal WER 1.4074, 1/6 requested commands, and 11/21 documented command
  confusions on negative targets.
- Overall: WER 0.8511, 19/26 command recall, target precision 0.6333, and 11/31
  monitored negative-case false positives. RTF was 0.0254 after PCM; there were
  no partials, so this is not streaming or endpoint evidence.

The production GigaSpeech Zipformer CPU control first ran with its legacy zero
tail and emitted no final for any of the 20 clean commands. The retained
current-source diagnostic report is
`f35eafac6ecb7e857f847cc3ef854153608303728f2a2fa83f6eaacbb68d73ce`.
One legal 16,000-sample zero tail then supplied enough context to clear its
configured 0.8-second rule-2 silence. The current-source report is
`10848d74f10379512ca7c2e7342c2ea450b79bac811f2b54ed3ad0a309a69b49`:

- Google Speech Commands: WER 0.20, 16/20 command recall, zero monitored command
  false positives, and empty finals on all five silence clips.
- ECCC: literal WER 1.2593, 1/6 requested commands, and 0/21 documented command
  confusions on negative targets.
- Overall: WER 0.8085, 17/26 command recall, target precision 1.0, and 0/31
  monitored negative-case false positives. Source-only-denominator RTF was
  0.1038 with 76 partial events.

The Zipformer adapter never calls the native endpoint API; it finalizes after
declared input. Tail compute is divided by source-only duration. Its padded RTF,
partials, and finalization therefore describe a short-clip context diagnostic,
not live endpoint latency or a fair latency comparison with Faster-Whisper.

## Verification

- Public command/noise focused split: 235 passed together; the pre-existing
  timestamp-sensitive bounded-read mutation test passed 1/1 in isolation.
- Matrix-v5 lock/materializer focused rerun (six files): 93 passed; the full
  documented fixture suite is covered by the repository-wide pass below.
- Repository-wide non-model split: 7,566 low-priority broad passes plus three
  exact-condition passes (logging enabled, repository lock mode 0600, and the
  bounded-read race isolated); 13 skipped, 24 model-only deselected, 11 known
  warnings. Effective total: 7,569 passing checks.
- Required APM/double-talk regression: 6 passed.
- Ruff, `git diff --check`, JSON parsing, mode checks, report/receipt hash and
  size checks, and STATUS's 100-line limit passed.
- Two independent final audits found no remaining code, metric, privacy,
  documentation, or evidence mismatch. All 63 evaluator-file bindings across
  the three final reports match current source files.

## Limits and next work

This is after-PCM isolated-word development evidence. It does not validate
microphone capture, VAD, native endpointing, AEC, barge-in, open-speaker STOP,
enrollment or multi-voice identity, agent tools, device latency, natural
conversation, or training disjointness. ECCC has no paired clean decode and is
not a command corpus; its literal noisy WER and documented confusions must stay
separate from clean/noisy degradation claims.

The preparer checks that output is outside Git before its bounded archive pass;
a concurrent external `git init` can change that classification before
publication, although descriptor binding, exclusive creation, and private modes
still prevent redirection or overwrite. Systemic writer-level hardening can
close that shared residual later.

Next, collect fresh owner target and negative phrases, then use this gate to
evaluate noise-robust model/front-end changes before a separate `./live.sh`
open-speaker validation. Do not promote a model from this corpus alone.
