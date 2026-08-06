# ADR-0151: Retain the Faster-Whisper Small AMI GPU diagnostic

Date: 2026-08-06
Status: accepted

Refines: ADR-0102, ADR-0120, ADR-0121, ADR-0143

Related: ADR-0150

## Decision

Record the exact 24-case AMI ES2004a close/far Faster-Whisper Small run as a
receipt-bound, aggregate-only GPU development diagnostic. Keep Faster-Whisper
in its existing final-only comparator and opt-in verifier roles; do not change
setup, the application entry point, runtime selection, transcript authority,
tools, thresholds, or defaults from this result.

Retain the original private evidence tree at
`/var/tmp/speaker-fws-ami-20260806-01` and its byte-identical durable copy at
`~/.local/state/speaker/evidence-fws-ami-20260806-01`. The 16,062-byte
mode-0600, single-link report has SHA-256
`06e51d89536e19d7c2e232d7c11e9857c3bb95eab535cec79b2361ac535dfc47`.
The report binds corpus SHA-256
`0b5e4d78847f886785a814f68055eab1f3007319c028ce92d42388b822e25772`,
preparation-receipt SHA-256
`b7ef76fc528add6bf8017a3532d649102cdedd4b70001141a1f8c835dd609585`,
worker-manifest SHA-256
`0a4615f69366015978dc7b632470db2537c29a9044d159db246c5805643c6b75`,
and the unchanged ADR-0143 runtime/model receipts.

## Context / why

ADR-0102 made this candidate measurable without pretending that its
complete-PCM decode is streaming recognition. ADR-0143 found a substantial
accuracy lead on close conversational EdAcc audio, while ADR-0121 showed that
far-field AMI audio can expose a much larger quality gap than close audio. A
fresh current-checkout run was therefore needed before prioritizing the next
owner test.

At clean revision `a46942c`, the CUDA/FP16 candidate used beam size 5, one
CTranslate2 worker, one CPU thread, no VAD, no previous-text conditioning,
1,600-sample replay chunks, 200 ms partial cadence, zero tail padding, burst
pacing, and three repeats. All 72 evaluations completed with zero final-text
disagreement. Those are three observations of each of 24 windows from one
meeting, not 72 independent samples, and `ok=true` establishes complete
coverage rather than a quality verdict. Overall WER/CER were
`0.3797`/`0.2905`; close-channel WER/CER
were `0.2025`/`0.1376`, while far-channel WER/CER were
`0.5570`/`0.4434`. The aggregate had 33 exact and 66 nonempty finals. The
after-complete-PCM finalization p50/p95/max were
`28.562`/`80.935`/`199.715` ms, model load was `529.402` ms, and
candidate-call-only compute RTF was `0.0153`. There was no same-run control;
close/far nonempty counts were 36/36 and 30/36 across repeats.

The final-only adapter emitted zero partials, so all 72 stable-partial values
were missing and streaming deadline metrics were not applicable. Reported
maximum RSS was 1,053.957 MiB only across ready/final samples, not a process
peak; peak VRAM was unavailable. The worker report independently attests
neither a hard host-memory nor a hard VRAM limit. Training disjointness was not
established.

The retained report was reparsed with duplicate-key and non-finite rejection,
matched its canonical bytes, and revalidated against the current corpus,
preparation receipt, PCM snapshots, manifest, runtime/model receipts,
reconstructed source bundle, worker binding, and evaluator closure. The empty
mode-0700 scratch parent remains retained. The durable copy matches the
original recursively and retains four mode-0700 directories and 31 mode-0600,
single-link regular files. No transcript row was printed or committed.

## Consequences

This run records repeatable final recognition and adapter decode time after
complete PCM; close/far WER differed by `0.3545` on this slice. It does not
measure microphone capture, native streaming, partial quality, VAD or endpoint
detection, capture-to-text or conversational latency, AEC, room echo,
barge-in, owner identity, tool routing, TTS, or live conversation. It cannot
promote a model or transfer action authority. The generic final-only harness
also did not exercise ADR-0150's logical-turn shadow.

Use the result only to prioritize matched owner exact-input replay and
`./live.sh` close/far, command, multi-voice, and open-speaker A/B. Preserve the
corpus, receipts, manifest, empty scratch parent, and report. A later decision
must evaluate those owner/live gates before changing any production default.
