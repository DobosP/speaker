# Exact private voice evidence

`./live.sh` produces a synchronized diagnostic bundle when recording is active
([ADR-0108](adr/0108-bind-exact-final-model-inputs-to-private-stt-replay.md)).
The manifest ending in `.diagnostic.json` is the authority: use the bundle only
when `core.diagnostic_bundle.validate_manifest()` returns `True`.

The four WAVs are continuous parallel views of capture and playback. They are
useful for DSP/AEC diagnosis, but they are PCM16 and are not turn fixtures. The
`.final-input.f32le` artifact is different: every packed slice is an exact,
unclipped, unquantized copy of the endpoint-owned float32 segment offered to
final transcript selection. Its transcript-free timeline receipt includes the
acoustic identity, revision, selected segment role, sample/byte ranges, and
SHA-256 digest.

This evidence still does not contain physical raw-microphone PCM, exact online
recognizer reset/accept operations, or proof of audible playback. It does not
replace a live room/device/AEC/barge-in test, and it establishes no model, GPU,
WER, latency, natural-conversation, or default-change result.

Exact-input admission has bounded synchronous cost on the finalizer thread: it
checks finite values, canonicalizes and copies the segment, and computes its
digest before a non-waiting queue admission. The writer thread owns disk I/O.
Each input is limited to 8 MiB, pending exact-input bytes to 32 MiB, and the
session spool to 256 MiB/4,096 receipts. A limit or write failure invalidates
the evidence bundle but does not stop final transcript processing.
Validation also caps the timeline at 64 MiB, 131,072 records, and 64 KiB per
line; a session that exceeds those evidence limits fails closed.

## Validate and inspect receipts

Keep every artifact in its original private directory. Do not rename, edit,
copy into the repository, or upload it.

```bash
python -c 'from core.diagnostic_bundle import validate_manifest; import sys; raise SystemExit(0 if validate_manifest(sys.argv[1]) else 2)' \
  logs/live/<run>/run-<id>.diagnostic.json

jq -c 'select(.kind == "final_model_input")' \
  logs/live/<run>/run-<id>.timeline.jsonl
```

`validate_manifest()` retains strict read-only validation for diagnostic schema
v1. Those legacy bundles lack the exact f32le slices and cannot be exported by
the tool below. A newly recorded export source must be diagnostic schema v2.

Receipts contain no transcript. The separate label file is private and must be
mode `0600` inside an owner-private directory. Bind both the diagnostic manifest
and each chosen input digest; the label file itself uses label schema v1:

```json
{
  "schema_version": 1,
  "diagnostic_manifest_sha256": "<sha256 of .diagnostic.json>",
  "cases": [
    {
      "id": "owner-command-001",
      "input_index": 0,
      "input_sha256": "<receipt sha256>",
      "expected_text": "search in my vault",
      "tags": ["owner-voice", "quiet-room"]
    }
  ]
}
```

Case IDs and tags use lowercase letters, digits, `.`, `_`, or `-`. The selected
inputs must total at most 32 MiB; each input is at most 8 MiB. Labels may select
up to 512 distinct receipt indexes.

## Export a benchmark corpus

The destination must not exist. Publication is no-overwrite and private:

```bash
chmod 600 /private/path/labels.json
python -m tools.prepare_diagnostic_streaming_stt_corpus \
  --diagnostic-manifest logs/live/<run>/run-<id>.diagnostic.json \
  --labels /private/path/labels.json \
  --output-dir /private/path/exact-final-input-corpus
```

Only a complete diagnostic schema-v2 bundle is eligible. The result uses
streaming-corpus schema v3 and provenance kind
`private-diagnostic-v1`. Public voice corpora remain schema v2 with
`public-voice-v1`; the loader rejects cross-kind version changes. Output audio
is copied bit-for-bit from the exact f32le spool. The preparation receipt has
identities and hashes but no paths or transcripts, and command output is
aggregate-only.

## Replay the configured final selector

Use the exported `corpus.json` directly; do not convert its f32le cases to WAV
or substitute a continuous diagnostic track:

```bash
python -m tools.recorded_stt_eval \
  --manifest /private/path/exact-final-input-corpus/corpus.json \
  --keyword vault \
  --output /private/path/recorded-final-selector-report.json
```

Add one or more `--set FIELD=VALUE` options to compare a candidate Sherpa
configuration with the current machine baseline. The evaluator passes a new
owned float32 copy of each exact 16 kHz input to FileReplay with no inferred
speech duration. It reports aggregate streaming/offline/selected accuracy,
the closed selected-source counts, and whether those counts attest every
terminal decision. It retains empty terminal boundaries and case tags only in
private memory; neither transcript rows nor tags enter stdout or the report.

Schema-v3 replay rechecks the source-set-bound selection receipt, private file
metadata, manifest, and PCM after copying and again after evaluation before
publication. The manifest must be a regular single-link file no larger than
8 MiB. Choose a new report path: an output that names or aliases the manifest,
receipt, PCM, or a legacy input is rejected without overwriting it. A changed
or malformed input returns only the detail-free prerequisite error and does
not publish the requested report. This is
after-endpoint model-input replay: it starts no chatbot, TTS, tool, recorder,
or audio device and does not establish capture, VAD/endpoint, AEC, barge-in,
latency, natural-conversation, or live behavior
([ADR-0124](adr/0124-replay-private-diagnostic-inputs-through-final-selector.md)).

Generic `python -m core --session replay --replay-dir ...` refuses any directory
containing a diagnostic manifest. Its input contract is a directory of
independent utterance fixtures; the bundle WAVs are parallel stage views, not
turns. Use the exact-input exporter above for a synchronized bundle.
