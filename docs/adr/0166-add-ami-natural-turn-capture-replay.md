# ADR-0166: Add an exact AMI natural-turn capture replay

Date: 2026-08-09
Status: accepted

## Decision

Add a new, self-digested AMI ES2004a natural-turn fixture and a dedicated
aggregate-only capture-replay wrapper. Keep the historical AMI capture and
close/far preparers, their locks and retained evidence, the generic
capture-replay schema/report/CLI, and every voice/runtime/default path
unchanged. Admit exactly two fixed manual-annotation envelopes projected once
to Mix-Headset close audio and once to Array1-01 far audio:

- `[8,719,200,8,736,800)` contains manual B backchannel DA `.83` strictly
  inside D DA `.59`; the corresponding B manual segment is also strictly
  inside D's longer segment.
- `[10,085,920,10,121,760)` is the exact positive adjacency pair from D DA
  `.77` to B DA `.107`, with a 1,440-sample forced-alignment gap and no
  cross-speaker word overlap.

For each channel, prepend the exact annotation-clean roomtone
`[3,625,520,3,657,520)`, retain the selected source envelope byte-for-byte
after PCM16-to-f32 conversion, and repeat the same channel's roomtone as
postroll to exactly 128,000 samples/80 frames. The four private PCM leaves are
therefore each 512,000 bytes. Bind the exact annotation archive, its consumed
word/dialogue-act/segment/ontology members, both WAVs, selection geometry,
references, transform, intermediate hashes, final PCM hashes, license, and
evidence limits in the new lock. Require explicit `CC-BY-4.0` acceptance and
offline local sources; the preparer never downloads.

Keep the two references and manual-dialogue-act metadata in a private sidecar.
Validate segment word ranges against every ordered source word-XML node ID,
including nonlexical activity markers at a segment edge; only timed word and
punctuation nodes may contribute reference text or scoring. The public report
may retain only aggregate counts and digests. On the two
backchannel cases, set ordinary WER to null and add only the existing custom
`two-utterance-min-order-wer-v1` integer micro-aggregate. The nested unchanged
generic capture report may retain its explicitly named order-ambiguous
linearized diagnostic; it never becomes ordinary overlap WER. The custom
metric is not ORC-WER, tcpWER, or tcORC. Count selected manual-DA exposures,
typed final callbacks, and typed abort callbacks descriptively; never turn
their equality, ratio, or difference into correctness, recall, a threshold,
or a verdict.

## Context / why

The retained 24-case synchronized AMI fixture intentionally excludes actual
overlap so its one-reference hard WER is unambiguous. The earlier four-case
capture replay does contain real overlap and a transition, but linearizes
simultaneous words and does not grade terminal count or manual conversational
units. A generic streaming-worker input can also stop at its first EOU, so it
cannot faithfully expose a later adjacent turn. Those contracts are valid for
their original purposes and must not be rewritten or rematerialized.

AMI's exact manual dialogue acts, segments, adjacency pairs, synchronized
close/far recordings, and forced-aligned words allow a much smaller diagnostic
to preserve natural backchannel and exchange geometry. The official filename
and catalog call the annotation archive v1.6.2, while its embedded README calls
the exact bytes release 1.7 dated 2014-06-16; the lock records both labels and
makes the exact archive hash authoritative. The Mix-Headset digest remains a
local content receipt, not an upstream-published checksum.

The selected envelopes touch neighboring manual-DA boundaries. Their word
spans and padding are exact, but they are not complete acoustic turns or
endpoint ground truth. The close/far pair isolates channel acoustics only on
this tiny, development-only slice; it does not create speaker attribution,
room behavior, device, or live evidence.

## Consequences

- One private loader binds the terminal receipt, private two-reference sidecar,
  unchanged schema-v1 replay manifest, four PCM leaves, committed lock, and
  full private file/directory identities before, after, and immediately before
  evaluation publication.
- `capture_replay_eval` gains only a private typed outcome seam. Its public
  evaluator signature, report, parser, CLI, and default behavior remain
  unchanged; ephemeral `ReplayRunRecord` text never crosses the wrapper's
  aggregate boundary.
- The wrapper report records an exact clean-import execution-closure file count
  and digest. Its ordinary canonical report binding includes that object while
  the nested generic evaluator keeps its own existing source digest.
- The wrapper runs the existing production capture seam in a bounded child,
  requires all 128,000 samples of every selected case to be consumed, and
  publishes a closed report through a no-clobber terminal link. A final may
  occur before source end; source completion describes the padded selected
  envelope only.
- No ordinary overlap WER, reference order, endpoint/VAD boundary, complete
  turn, quality, qualification, promotion, latency, AEC, identity, tool,
  capture-device, or live claim follows. No application entry point, model
  choice, GPU policy, or runtime default changes.

## Verification

The final gate must cover the exact production lock and real-source
materialization; synthetic source/parser/publication failures; private
load/reopen identity races; public capture-replay compatibility; custom metric
integer equations and privacy; complete-source/terminal-count semantics;
watchdog/lifecycle handling; and terminal report no-clobber/ambiguous-link
recovery. Record exact settled-byte counts and artifact digests in `STATUS.md`
only after independent audits are GO. No model, GPU, audio-device, or live run
is required to land this quarantined diagnostic path.

On the settled bytes, the focused preparer/evaluator/public-seam/custom-metric
gate passed 193 tests, the adjacent capture-replay gate passed 251, and the
required APM/DTD gate passed 6. Scoped Ruff lint and format checks plus
`git diff --check` are green; independent source and evaluator audits are GO.
The lock raw/recipe SHA-256s are
`02ebfbbbf0871464586f2d62a181cf3dbd2b2699eaec0d72650a335601ea9614`
and `9fe99f0b4483f99755ddc5a16d96cf567d852e9f5cfb9fe4dd539d6e1370fa81`.
The wrapper binds exactly 56 clean-import files under execution-closure digest
`706b237d27bbca5b0fba70c501a2c58eb2de28bffa15ccf19fb91a80cb237472`.

The exact retained sources produced and reopened one private production
fixture with four cases / 2,048,000 PCM bytes. Its manifest, private metadata,
terminal receipt, label-set, source-contract, and preparer-closure SHA-256s are
`378ebdcdf914d378a2e280ccb89ee0d1f24ad817972c4fe0e329a12a80f924dd`,
`7d69f72e6d31c561ae04f4f44ef81e50aff8756d3250b9542c1e1144c5ee77fe`,
`e11ab9cfbad43b03a16d1dff41a132cc075df36d16cb45cdb18494fa9e439725`,
`b79f4a361c32c03c9f3e68ce22bc314bae30dd72dab9724e1980f9caf7cd18d7`,
`e6d78daa9a1fdd5fb99aff75a3d024efabb29d129def8be36f9b949069aa7ef8`,
and `336abd9260b4eb9fac972e11858cca946ea5d36a21aea6e80ee2cae70b10ac85`.
The root remained mode 0700 and all seven leaves mode 0600, current-owner,
single-link. No model, GPU, audio device, service, or live path ran.
