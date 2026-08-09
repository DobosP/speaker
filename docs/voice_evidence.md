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

## Bind an already-retained diagnostic bundle

This legacy binder is only for an already-retained bundle whose collection
path was separately established as safe for the labelled phrases. Do not use
the ordinary assistant runtime to collect a new command plan: deterministic
routes can execute even when its LLM is `echo`. For a new physical control/
candidate recording, use the effect-free guided capture procedure below.

For an eligible retained bundle, keep its reference plan in an owner-private
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

Save it mode `0600` under a mode-`0700` parent. Use the retained bundle's
`.diagnostic.json`; both destinations below must be absent and their parents
must already be private:

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

For a physical control/candidate check, run the immutable guided plan once per
complete profile. Follow the terminal geometry and case prompts; speak each
armed phrase exactly once. Run both commands from the same revision without
changing configuration or the audio route between them. If you intentionally
pass `--device` or `--input-gain`, pass the exact same value to both runs:

```bash
./live.sh --guided-stt-capture --run-label owner-stt-sense-control \
  --final-stt-profile sense-voice

./live.sh --guided-stt-capture --run-label owner-stt-parakeet-candidate \
  --final-stt-profile parakeet-faster-whisper
```

Each run uses the single physical entry and creates its own synchronized
private bundle. The capture path has no assistant or effect plane: it does not
construct tools, control, memory, TTS, KWS, speaker identity, playback, or an
output device. The private contract binds the fixed plan, profile, device,
input gain, and effective capture/configuration digests before microphone
startup. Exit zero additionally requires 16/16 armed live finals and a valid
post-stop diagnostic manifest. Keep both bundles unchanged. Do not substitute
`--llm echo`, the backend-only `--asr-final` option, or a low-level core command
(ADR-0157).

Qualify only those two completed bundles through the fixed device-free paired
attestor. The scratch root and report must be new absent absolute paths under
owner-private parents:

```bash
python -B -m tools.guided_stt_pair_attestor \
  --control-bundle /absolute/private/owner-stt-sense-control-bundle \
  --candidate-bundle /absolute/private/owner-stt-parakeet-candidate-bundle \
  --scratch-root /absolute/private/new-guided-stt-pair-scratch \
  --output /absolute/private/new-guided-stt-pair-attestation.json
```

The attestor reopens both canonical plans, capture contracts, and diagnostic
manifests. It requires the fixed profile roles and matching non-final capture
protocol, plan/order, configuration, device, input gain, and route-policy
bindings. It compiles each bundle's 16 receipt-bound final inputs separately,
then runs the fixed crossover in this order: both profiles on the control
capture, followed by both profiles on the candidate capture. The selectors see
identical PCM within each physical-take comparison; the two independently
spoken bundles are not identical PCM.

The same evaluation runs the inert deterministic tool-route gate with
fail-closed no-invoke/no-open capability and launcher sentinels. It constructs
no execution-capable provider, voice runtime, capability, reminder store, app
launcher, audio device, or effect path. Selected hypotheses exist only in
private process memory and are reduced to aggregate counters; stdout and the
report contain no transcript, case row, path, alias, or identity. Exit zero
means the four cells, dry route checks, closing input revalidation, and
no-clobber report publication completed. Before that terminal link, Ctrl-C
returns 130, SIGHUP or SIGTERM returns 128 plus the signal number, and ordinary
failure returns 2, all without a final report. After the link, every outcome
returns 0 with the report retained;
it does not mean that either profile won. The tool cannot prove that each
reference was spoken correctly or that the prompted physical geometry was
followed, so physical owner review remains required (ADR-0158).

For exploratory diagnosis of one already-retained labelled corpus, both
complete selectors can still be compared on that corpus's same PCM without
relying on the machine's ambient final-ASR configuration:

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
the production final-selection result after the endpoint on identical audio,
but it does not enforce or replace the two-bundle attestation contract. The
guided bundles contain different spoken takes and no assistant playback path;
they do not establish comparative live latency, interruption, AEC, barge-in,
or natural-conversation behavior. Neither result authorizes a default change
([ADR-0144](adr/0144-add-atomic-live-final-stt-profiles.md),
[ADR-0158](adr/0158-add-paired-guided-stt-bundle-attestation.md)).

The locked public command/noise corpus has a separate production-selector pair
route. It fixes the same ordered profiles and requires both model closures
before the first cell starts:

```bash
python -m tools.production_final_stt_eval \
  --corpus /private/path/command-noise/corpus.json \
  --config /home/dobo/work/speaker/config.json \
  --local-config /home/dobo/work/speaker/config.local.json \
  --device desktop_gpu_4090 \
  --baseline-final-stt-profile sense-voice \
  --candidate-final-stt-profile parakeet-faster-whisper \
  --repeats 1 \
  --stratum-tag command-negative --stratum-tag command-positive \
  --stratum-tag eccc --stratum-tag gsc --stratum-tag noisy \
  --stratum-tag silence --stratum-tag speech-negative \
  --report /private/path/new-command-profile-pair-report.json
```

The schema-v3 report contains both aggregate cells and a separate comparison
verdict. A raw-streaming mismatch is retained as
`inconclusive_streaming_control_mismatch`; operational exit 0 means both cells
and publication completed, not that a profile won. The retained exact run and
its non-promotional command-safety result are recorded in
[ADR-0145](adr/0145-add-paired-final-stt-command-noise-diagnostic.md). Sequential
execution supplies no latency comparison, and this route opens no microphone.

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
