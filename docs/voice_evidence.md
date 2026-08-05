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

## Bind a planned live script automatically

Prepare the reference plan before recording and keep it in an owner-private
directory. The plan deliberately contains no receipt indexes or hashes:

```json
{
  "schema_version": 1,
  "cases": [
    {
      "id": "owner-vault-search-001",
      "expected_text": "search in my vault for the speaker roadmap",
      "tags": ["owner-voice", "expected-tool.vault.search"]
    },
    {
      "id": "owner-vault-find-002",
      "expected_text": "find the speaker roadmap in my vault",
      "tags": ["owner-voice", "expected-tool.vault.search"]
    },
    {
      "id": "owner-reminder-003",
      "expected_text": "remind me tomorrow to run the voice test",
      "tags": ["owner-voice", "expected-tool.reminder.create"]
    }
  ]
}
```

Save it mode `0600` under a mode-`0700` parent. Then make one lightweight live
recording, speaking each case exactly once in file order and waiting for each
turn to finish before starting the next:

```bash
./live.sh --run-label owner-stt-commands --llm echo
```

After one Ctrl-C and completed cleanup, use the `.diagnostic.json` path printed
by the launcher. Both destinations below must be absent and their parents must
already be private:

```bash
python -m tools.prepare_live_stt_corpus \
  --diagnostic-manifest logs/live/<run>/run-<id>.diagnostic.json \
  --reference-plan /private/path/reference-plan.json \
  --labels-output /private/path/generated-labels.json \
  --output-dir /private/path/exact-final-input-corpus
```

The binder pairs every plan case with every exact final-input receipt in order
and fails on any count mismatch. It never reads recognized text as truth. The
generated label file is the ordinary label schema v1 described below, and the
corpus is the same schema-v3 output consumed by `tools.recorded_stt_eval`.
Output stays aggregate-only.

If downstream corpus publication fails after the complete generated label is
published, that mode-`0600` label remains reusable. Keep any failed corpus
directory for diagnosis and choose a new destination when retrying the existing
exporter:

```bash
python -m tools.prepare_diagnostic_streaming_stt_corpus \
  --diagnostic-manifest logs/live/<run>/run-<id>.diagnostic.json \
  --labels /private/path/generated-labels.json \
  --output-dir /private/path/new-exact-final-input-corpus
```

An equal count establishes only ordered binding. It cannot prove that each
phrase was spoken correctly or that one missed turn and one accidental extra
turn did not cancel out. This remains owner-reviewed, after-endpoint evidence
([ADR-0138](adr/0138-bind-ordered-live-stt-references.md)).

## Label selected receipts manually

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
      "tags": ["owner-voice", "quiet-room", "expected-tool.vault.search"]
    }
  ]
}
```

Case IDs and tags use lowercase letters, digits, `.`, `_`, or `-`. The selected
inputs must total at most 32 MiB; each input is at most 8 MiB. Labels may select
up to 512 distinct receipt indexes. Put any `expected-tool.*` annotation in
this source label file before export. Never add or change a reference or tag in
the published `corpus.json`.

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
identities and hashes but no paths, transcripts, case IDs, or tags. Receipt v2
cross-binds the raw label-file digest and a domain-separated digest of every
ordered emitted case field. Receipt v1 corpora are rejected: re-export from the
original diagnostic bundle and original mode-0600 label file rather than
editing or upgrading a published corpus. Command output is aggregate-only.

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

Schema-v3 replay rechecks the source-set-bound selection receipt, exact case
binding, private file metadata, manifest, and PCM after copying and again after
evaluation before publication. The manifest must be a regular single-link file
no larger than 8 MiB. Choose a new report path: an output that names or aliases
the manifest, receipt, PCM, or a legacy input is rejected without overwriting
it. A changed or malformed input returns only the detail-free prerequisite
error and does not publish the requested report. The local unkeyed receipt
detects inconsistent edits; malicious coordinated re-authoring still requires
an externally retained digest or signature to detect. This is
after-endpoint model-input replay: it starts no chatbot, TTS, tool, recorder,
or audio device and does not establish capture, VAD/endpoint, AEC, barge-in,
latency, natural-conversation, or live behavior
([ADR-0124](adr/0124-replay-private-diagnostic-inputs-through-final-selector.md),
[ADR-0126](adr/0126-bind-private-diagnostic-case-labels.md)).

## Compare complete final-STT profiles

For a physical control/candidate check, use the same private reference plan in
two separate runs. Speak every planned case exactly once and in the same order
in each run:

```bash
./live.sh --run-label owner-stt-sense-control --llm echo \
  --final-stt-profile sense-voice

./live.sh --run-label owner-stt-parakeet-fws --llm echo \
  --final-stt-profile parakeet-faster-whisper
```

Each run still uses the single supported physical entry and creates its own
synchronized private bundle. The profile name and canonical selection digest
are recorded in its private summary. Readiness consumes the same complete
profile as the core child and stops before capture if any selected artifact or
runtime is unavailable. Do not use the backend-only `--asr-final` option for
this comparison.

After exporting one labelled exact-input corpus as above, compare both complete
selectors on the *same* PCM without relying on the machine's ambient final-ASR
configuration:

```bash
python -m tools.recorded_stt_eval \
  --manifest /private/path/exact-final-input-corpus/corpus.json \
  --baseline-final-stt-profile sense-voice \
  --candidate-final-stt-profile parakeet-faster-whisper \
  --keyword vault \
  --output /private/path/new-final-profile-report.json
```

The two profile arguments are a pair and cannot be mixed with `--set`. Keep the
report in a private directory and use a new output name. This replay measures
the production final-selection result after the endpoint on identical audio;
the two physical bundles separately expose capture, room, interruption, and
latency behavior. Neither result alone authorizes a default change
([ADR-0144](adr/0144-add-atomic-live-final-stt-profiles.md)).

## Dry-run recognized-text tool routing

Route-labelled cases may additionally carry exactly one of these private tags:
`expected-tool.none`, `expected-tool.vault.search`,
`expected-tool.web.search`, `expected-tool.reminder.create`,
`expected-tool.reminder.list`, `expected-tool.reminder.cancel`, or
`expected-tool.app.open`. The `expected-tool.` prefix is reserved; the opt-in
gate rejects missing, unknown, or multiple annotations before starting a
model. Include at least one positive route and one `none` case. Add the
annotation to the source `labels.json` as shown above, without adding command
fields, and export a new corpus.

Pass only the inert availability profile represented by the labelled cases:

```bash
python -m tools.recorded_stt_eval \
  --manifest /private/path/exact-final-input-corpus/corpus.json \
  --tool-route-gate \
  --tool-route-vault-enabled \
  --tool-route-reminders-enabled \
  --tool-route-app-alias your_alias \
  --output /private/path/recorded-tool-route-report.json
```

Omit availability flags for providers intentionally absent from the profile;
repeat `--tool-route-app-alias` for each tested allowlisted alias. The
evaluator does not read machine tool configuration. It emits only a
domain-separated profile digest, never alias values.

The gate sends exactly one nonempty selected final through the production
deterministic speech analyzer and task planner, with strict reminder/app
matching only. It never dispatches or invokes a capability, opens an app,
reads the vault, creates reminder state, calls a provider or LLM, or constructs
the voice runtime. Aggregate output counts closed expected-route attempts/hits
and coarse safety outcomes; text, tags, aliases, paths, reminder identities,
and per-case rows do not enter stdout or reports. Zero, empty, or multiple
terminal decisions fail the route gate instead of being joined.

This remains after-endpoint dry evidence. It does not validate capture/VAD,
AEC, barge-in, enrollment or owner authority, confirmation, provider
readiness, reminder identity/state, vault results, cleaning/addressing,
continuation, external effects, devices, or live latency
([ADR-0125](adr/0125-add-dry-recorded-tool-route-gate.md)).

Generic `python -m core --session replay --replay-dir ...` refuses any directory
containing a diagnostic manifest. Its input contract is a directory of
independent utterance fixtures; the bundle WAVs are parallel stage views, not
turns. Use the exact-input exporter above for a synchronized bundle.
