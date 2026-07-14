# ADR-0045: Require the lexical floor before speaker-authorized barge-in

Date: 2026-07-11
Status: superseded-by ADR-0072

## Decision

Keep the verified OS/PipeWire echo-cancel route, isolated playback recognizer,
bounded candidate PCM, and enrolled-speaker comparison, but restore lexical
content as mandatory generic cut evidence. In production speaker-required mode,
require at least `max(4, barge_word_cut_min_words,
barge_word_cut_speaker_min_words)` novel words before consulting speaker-ID or
promoting PCM. Values from zero through three cannot lower this invariant. Apply
the same floor to an active playback cut, a natural reply-tail handoff, and fresh
post-playback tail probation. Drop empty tails; one-to-three-word tails may remain
bounded only while waiting to reach the full floor. They cannot acquire speaker,
replay, offline-recovery, or post-barge authority.

Keep typed STOP-class controls as the deliberately short exception. Canonical
controls, including exact STOP/cancel, and ADR-0042's exact attested `OF HE STOP`
repair may cut without the generic floor. When current/recent TTS makes a control
read like own speech, evaluate the existing `0.10`-second short speaker gate
regardless of word count. Do not add a global own-speech text veto: an enrolled
owner may legitimately repeat or correct assistant words. Keep the purpose-
specific speaker threshold and final trust separation; a barge score never
creates owner verification.

This supersedes ADR-0036's zero-word audio-first authority while retaining its
PCM bounding, retry, replay, route-readiness, and trust-lineage safeguards. It
restores ADR-0013's evidence-backed four-word generic floor and adds speaker-ID as
a second required signal rather than a replacement for lexical evidence.

## Context / why

In live run `20260711-193818`, the owner confirmed silence at `19:39:34`. The
post-EC window reported RMS average `0.0009`, peak `0.0028`, VAD fraction `0.00`,
and calibrated floor `0.0052`, yet a later 412 ms empty candidate scored `0.32`
against the `0.30` barge threshold and stopped playback. The same run had already
made a 396 ms zero-word cut at `0.31`. The second cut promoted empty PCM as
offline-recovery-authorized; its raw-empty final became `I.`, and the bounded
post-barge response path correctly but confusingly started another answer.

The score cannot be repaired by threshold tuning. Silent residuals scored
`0.31-0.32`, while the prior real owner talk-over in run `20260711-170840` scored
only `0.19-0.23`. Raising the threshold would preserve these false cuts only by
making real interruptions still less available; lowering it is less safe.
ADR-0036's earlier corpus (`0.15-0.179` own TTS, `0.387-0.537` owner slices) did
not cover speaker embeddings on near-silence. ADR-0013 did record novel-looking
garbled echo at two to three words, so a one-word prerequisite would not restore
the known echo-safety boundary. Lexical overlap is also not a safe veto because a
real owner may repeat the assistant verbatim.

## Consequences

- Empty or one-to-three-word generic residuals cannot invoke speaker inference,
  cut playback, seed normal/offline ASR, or trigger a post-barge response, even if
  a speaker backend would return a perfect similarity score.
- Four novel words plus speaker acceptance remain sufficient; canonical and
  attested STOP-class controls retain their short, self-echo-aware path.
- Real talk-over whose playback ASR stays below four words now fails closed. This
  deliberately trades availability for the silent-control safety the live run
  disproved; better short-interrupt evidence needs a separately validated KWS or
  acoustic authority, not another global threshold.
- The floor is necessary, not sufficient provenance. Live run `20260711-200747`
  made zero active cuts and rejected a two-word playback trace, but that trace
  grew to five echo-like words during post-playback probation and weakly passed
  speaker-ID at `0.31`. Tail lexical/acoustic binding needs a separate decision;
  this branch must not be represented as live-green barge-in.
- Headless tests cover zero through four words, stale local floor values, STOP
  ambiguity, active/tail boundaries, PCM promotion, and offline recovery. Bare-
  speaker silent/STOP/generic-override A/B validation remains required.
