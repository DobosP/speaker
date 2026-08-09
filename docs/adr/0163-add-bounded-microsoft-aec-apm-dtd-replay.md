# ADR-0163: Add a bounded Microsoft AEC APM/DTD replay

Date: 2026-08-09
Status: accepted

## Decision

Keep public matrix v5, every retained private fixture receipt, the shipped
voice runtime, `open_speaker` profile, voice-agent entry points, callable
capability/tool authority, and all live/default authority unchanged. Add a
separate self-digested lock and offline preparer for 32 deterministic rows from
the Microsoft AEC Challenge synthetic source at
commit `6c633d0a9d2a143a0e364899b91b06f127315b18`. Bind each row's echo, far-end,
near-end microphone, and clean near-end WAV: 128 exact Git-LFS objects plus the
exact 10,000-row metadata file, totalling 44,209,144 pinned source bytes.

The fixture lock owns its selector rather than changing the shared catalog:
rank all metadata rows by SHA-256 of UTF-8 seed, one NUL byte, and compact
canonical JSON `[fileid]`; sort by rank and identity; retain the first 32. The
shared catalog file is already part of older fixture execution closures. Its
dormant selector identity field is the upstream-incompatible `file_id`, and its
prose omits the NUL/canonical-JSON ranking details; correcting either now would invalidate
retained evidence. A later compatibility migration may correct that entry only
together with deliberate rematerialization and republication of every affected
retained receipt/report whose closure binds the matrix; it is not this fixture's
authority.

Require explicit acceptance of the LibriVox public-domain condition, Edinburgh
VCTK data terms, AudioSet CC-BY-4.0, selected Freesound CC0-1.0, and the
conservative DEMAND CC-BY-SA-3.0 condition. The preparer downloads nothing. It
accepts only a detached, owner-private materialized source snapshot; verifies
the exact consumed metadata and selected payload bytes, synchronized RIFF
geometry, modes, links, owner, and stable private file identities; and publishes
opaque copied WAV names into a new private directory. Commit/tree/pointer values
are lock-pinned and independently audited against upstream; supplied-checkout
Git-object provenance is not recertified. A staged anonymous receipt is linked
only after close-time source, output, lock, and execution-closure validation.
The strict loader binds reopening to the committed lock, the re-derived
source-contract digest, and copied-output hashes; it does not reopen the source
checkout.

Add one receipt-bound, CPU-only component evaluator. Production mode must load
the committed `open_speaker.sherpa` values, construct the real WebRTC APM through
`build_aec`, and reuse the shipped `AdaptiveDTD`, `BargeSustain`,
`EchoCoherenceDetector`, and Sherpa gate/floor methods. It must verify the exact
installed LiveKit distribution members and require the committed exact LiveKit
version before constructing APM. A mismatched or unbound runtime fails before
processing; a test double always marks runtime and APM-component production
evidence false without changing the fixture's own provenance classification.
The evaluator binds only that exact RTC requirement. It does not claim the full
optional `requirements-remote.txt` set is jointly resolvable: official
[`livekit-agents==1.6.7` metadata](https://pypi.org/pypi/livekit-agents/1.6.7/json)
requires `livekit==1.1.13`, which conflicts with the file's exact
`livekit==1.1.14`. Correcting the older ADR-0104 pair requires a separate
compatibility review and superseding decision.

For each case, use one fresh APM for the full near-only phase and another fresh
APM plus one continuous detector state for the full echo-only calibration phase
followed immediately by the full double-talk phase. Decode PCM16 as float32 by
dividing by 32,768. Convert the decimal `nearend_scale` once to float32 and form
the clean target with float32 multiplication. Preserve every source sample. A
159,999-sample track receives exactly one terminal float32 zero so the APM and
100 ms detector frames reach 160,000 samples. Sample-domain audio metrics
exclude that pad; the detector still observes the completed terminal frame.
Within each phase no source sample may be truncated, duplicated, or dropped.
The far-end reference is deliberately replayed once in the echo-only phase and
again in the immediately following double-talk phase.

Use an energy oracle only to identify double-talk frames: one 1,600-sample frame
is active when its scaled-clean RMS is at least the larger of `1e-5` and ten
percent of that case's nearest-rank p95 frame RMS. The oracle is not VAD, live
speech admission, or endpoint authority. Aggregate diagnostic echo attenuation
after a fixed one-second convergence exclusion; near-only projection gain,
absolute cosine, and SI-SDR; active-frame double-talk retention and interference
reduction; and detector false-cut, recall, first-cut, and D-score summaries. Do
not add AECMOS without a separately pinned compatible scorer, and do not set a
quality or promotion threshold in this first replay.

Publish only a closed, aggregate-only, finite private report with exact fixture,
protocol, evaluator, runtime, and coverage bindings. Reopen the fixture and
runtime before execution, after execution, and immediately before the terminal
no-clobber report link. Exclude paths, source/public IDs, speaker names, case
rows, PCM, and transcripts. Mark the report's actual `live_capture`, `device`,
`vad`, `stt`, `endpoint`, `tool`, `identity`, `conversation_latency`,
`promotion`, and `default_change` evidence fields false. The replay also cannot
establish room behavior or any qualification verdict.

## Context / why

The PriMock57 overlap slice measures natural two-role STT loss but does not
contain synchronized echo, clean near-end, and microphone component truth. The
voice stack's remaining bare-speaker risk is orthogonal: APM may erase near-end
speech or the DTD may self-cut on echo. The
[Microsoft AEC Challenge source](https://github.com/microsoft/AEC-Challenge/tree/6c633d0a9d2a143a0e364899b91b06f127315b18/datasets/synthetic)
and its [dataset paper](https://arxiv.org/abs/2309.12553) publish exact
clean/far/echo and microphone tracks that can exercise those components without
a microphone, room, GPU, or another large corpus.

The selected audio is synthetic overlapping read speech. It has no assistant
intent, command transcript, natural turn timing, physical laptop route, or
device generalization. Its microphone signal may include near-end noise, and
the clean target requires the published scale; it is incorrect to assume
`mic = near + echo`. ERLE-like attenuation here is therefore a component
diagnostic, not AECMOS or product quality.

## Consequences

- A future explicitly term-accepted materialization can test the production APM
  and oracle-gated DTD with exact source/tail accounting and no GPU allocation.
- This branch may land its lock, preparer, evaluator, and synthetic tests without
  downloading source audio or claiming that the currently installed mismatched
  LiveKit runtime ran production evidence.
- A green replay cannot change barge thresholds or establish natural
  conversation, latency, endpoint, STT, identity, tool, entry-point, or default
  behavior. Owner bare-speaker live A/B remains mandatory.
- The mixed upstream terms make the fixture private/cache-only. Redistribution
  requires a separate legal/provenance review, attribution, share-alike handling,
  and jurisdiction-specific LibriVox public-domain confirmation.

This ADR refines ADR-0004, ADR-0006, ADR-0008, ADR-0012, ADR-0013, ADR-0098,
ADR-0159, ADR-0161, and ADR-0162.

## Verification

- Lock recipe SHA-256:
  `a1c6c65f8d40dd147bb7ba13fc0e389bfe33adecf411b3200031215ba65cf198`;
  selected-row SHA-256:
  `542d6d102cad3f47c7c5aad96bbf04c7796654ba13573068bc930232b38ba331`;
  exact selected source bytes: 44,209,144.
- Settled headless gates pass 49 fixture/lock tests, 83 evaluator/metrics tests,
  138 combined fixture/evaluator/APM tests, and 116 adjacent AEC/DTD/coherence
  tests with two expected skips. The evaluator execution closure binds 53 files,
  1,783,933 bytes, and SHA-256
  `3c359032e69a0333221b7818ec2024f0a1d38a33666b4d5ddb95efef4afa9740`.
  Scoped Ruff lint/format and whitespace checks are green, and independent
  settled-byte correctness review is GO; `STATUS.md` records the same evidence.
- No source WAV download, official fixture materialization, exact-1.1.14
  production evaluator construction or replay, model, GPU, microphone,
  audio-device, quality, latency, or live run occurred. One all-zero-frame APM
  construction smoke on installed LiveKit 1.1.10 was explicitly non-production
  and remains rejected by the exact 1.1.14 production preflight.
