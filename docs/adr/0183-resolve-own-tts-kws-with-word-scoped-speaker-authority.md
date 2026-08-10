# ADR-0183: Resolve own-TTS KWS ambiguity with word-scoped speaker authority

Date: 2026-08-11
Status: accepted

## Decision

Permit one narrow speaker-authorized resolution for a playback-time KWS hit
that ADR-0182's exact shipped phrase binding identifies but its current/recent
own-TTS alias check makes ambiguous. Preserve ADR-0182's typed STOP-only
callback, closed labels, exact phrase binding, route guard, generation claim,
playback-start serialization, admission rejection, and admitted-only
terminalization. A non-ambiguous exact shipped hit follows that existing path;
an ambiguous hit may follow it only after every word in the resolved phrase has
independently received current enrolled-speaker `ACCEPT` authority. Every
failure or uncertainty abstains.

Maintain exact native sample-clock and feed-adjacency state for every KWS
stream. When reader-time authority has no compatible, enrolled, warmed speaker
receipt, retain only that clock/adjacency state: retain no KWS PCM and neither
request nor parse native timestamps. When such a receipt is available, enable
a dedicated, bounded ledger containing exactly the post-front-end PCM fed to
that native stream. Bind the stream and PCM as one continuity object to the full
reader-captured capture epoch/media generation, capture source and device,
authority source and device, OS/virtual route proof, speaking state, speak
generation, and playback generation. Bind the available receipt additionally
to the exact speaker engine/gate incarnation, model or embedding-function
generation, enrollment generation, threshold-policy generation, and successful
warm result for that same gate incarnation. Implementation field names may
differ, but the speaker receipt and final KWS claim must recheck all of those
semantics.

The PCM retention limit is configuration-derived, not one universal numeric
ceiling. Under the committed defaults it is 32,000 float32 samples (128,000
payload bytes), equal to twenty normal 100 ms capture chunks. A
speaker-unavailable scope retains zero PCM. This decision does not claim that
an arbitrarily larger finite configured window has the default numeric bound.

Restart KWS conservatively across every speaker-authority transition, including
availability loss/gain or a change to gate, model, enrollment, policy, or warm
identity, even when the current hit would be idle or novel and would never run
speaker inference. Any other mismatch, discontinuity, reset, native
fault/exhaustion, malformed feed, or scope transition also rotates or dirties
the native stream and, when enabled, its PCM ledger together. No prefix,
timestamp, PCM, or speaker result may cross a boundary.

A publication-ineligible capture block is not KWS input. This includes the
one-shot guarded first block after a capture gap. Mark native continuity and any
enabled PCM ledger dirty without feeding that block, then create a fresh stream
and, when applicable, ledger for the next eligible block. Thus a prefix from the
guarded block cannot combine with a later suffix and acquire control authority.

Only for an own-TTS-ambiguous exact hit in a speaker-receipt/PCM-enabled scope,
pin timestamp interpretation to sherpa-onnx v1.13.3. Use only the public Python
`tokens(stream)` and `timestamps(stream)` lists; do not reach through the
private `keyword_spotter` object for `start_time`, and do not invent keyword
onset or duration. Require one finite, non-negative, strictly increasing token
timestamp per exact decoded phone token on the pinned 40 ms lattice. The
code-owned phrase table also pins each word's exact phone-token count. Build
half-open word spans from the first token of each word to the first token of the
next word; end the last word exactly one 40 ms frame after its final token.
Every span must map wholly into the same ledger. Missing, malformed, reordered,
off-lattice, out-of-ledger, or otherwise inconsistent token/timestamp evidence
abstains. Speaker-unavailable, idle, and novel scopes do no timestamp work.

For the one or two words in that ambiguous shipped phrase, acquire the shared
speaker-inference slot once without blocking, then make at most two synchronous
native embedding calls while holding it, one independently bounded word clip
per call. A busy slot abstains before either call. Require a compatible enrolled
gate, a current same-incarnation warm receipt, the unchanged 0.10-second voiced
minimum for each word, and `ACCEPT` under the unchanged speaker threshold policy
for every word. Busy, cold, unenrolled, short/unvoiced, rejected, indeterminate,
malformed, raising, stale, or changed authority abstains for the entire hit. Do
not average or pool words, score the whole phrase as a substitute, reuse a
word-cut/final-ASR buffer, retry on the capture owner, or turn the result into
owner verification. The mailbox and retained PCM remain bounded by their
configured caps, but an already-entered native embedding call is not
wall-clock-preemptible.

This is word-level speaker checking derived from phone-token timestamps, not
phone-level diarization. It creates no aggregate speaker provenance and grants
no identity, continuation, private-context, tool, or device-action authority.
Within a stable authority scope, idle generic KWS and novel playback KWS retain
their existing label/effect eligibility and run no speaker inference. Their
native stream still restarts at a speaker-authority transition. The independent
word-cut path and its four-word/STOP rules, command/tool routing, thresholds,
boosts, model selection, configuration, and defaults remain unchanged.
ADR-0171's prohibition on raw-KWS effects and ADR-0172's returned native-work
bounds also remain unchanged.

This ADR supersedes only decision 4 of ADR-0082, where every own-TTS-ambiguous
playback KWS hit was unconditionally suppressed, and ADR-0182's clause that
speaker-authorized KWS ambiguity resolution was deferred. It carries every
other ADR-0082 and ADR-0182 fence forward and preserves ADR-0042, ADR-0072,
ADR-0137, ADR-0152, ADR-0171, and ADR-0172.

## Context / why

ADR-0082 deliberately preferred a missed simultaneous user STOP over a
self-cut, and ADR-0182 first closed playback KWS to an exact shipped phrase and
generation-scoped STOP-only effect. That made a speaker tie-breaker safe to
consider, but whole-phrase similarity would still allow one accepted speaker
to launder another speaker or the assistant's own words inside a two-word hit.
Requiring an independent decision for every code-pinned word closes that mixed-
speaker seam while retaining the conservative abstention for all incomplete
evidence.

Pinned upstream source inspection establishes the only timestamp semantics
used here: the [Python wrapper exposes token and timestamp lists](https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.3/sherpa-onnx/python/sherpa_onnx/keyword_spotter.py),
the [public result requires one timestamp per token](https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.3/sherpa-onnx/csrc/keyword-spotter.h),
the [v1.13.3 transducer converts 10 ms frames with subsampling factor four](https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.3/sherpa-onnx/csrc/keyword-spotter-transducer-impl.h),
and the [decoder appends each emitted token's frame plus the stream frame offset](https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.3/sherpa-onnx/csrc/transducer-keyword-decoder.cc).
This supports a strict 40 ms token-frame contract, not inferred phone duration,
word alignment, diarization, or physical speaker provenance.

Keeping an exact KWS-fed PCM ledger when speaker resolution is available is
necessary because token times are relative to native stream continuity. A
convenient recent-audio ring, ASR segment, word-cut candidate, or reconstructed
window could contain audio the KWS stream never consumed or span a gap/reset.
Retaining it while no speaker resolution exists would create unused biometric
audio and invite accidental timestamp work, so that mode keeps only the native
clock/adjacency needed for future continuity. Generation checks made only at
detection time are also insufficient: gate replacement, enrollment reload,
threshold mutation, or a stale warm result can otherwise create an ABA
acceptance before the final STOP claim.

## Consequences

- A simultaneous enrolled owner may recover an exact shipped STOP/WAIT phrase
  that overlaps current/recent own TTS, but only when every word independently
  clears the same current speaker policy. A two-word mixed-speaker phrase never
  gains authority from an aggregate score.
- A speaker-receipt/PCM-enabled KWS stream owns a bounded post-front-end PCM
  ledger and strict token-frame mapping. A speaker-unavailable stream owns only
  its native clock/adjacency and performs no timestamp parsing. Every KWS stream
  restarts across speaker-authority transitions; conservative misses are
  expected when evidence is short, busy, stale, or malformed.
- The post-gap first-block guard can no longer seed a hidden native prefix.
  The following eligible block always begins fresh native continuity and, when
  speaker resolution is available, a fresh PCM ledger.
- The added path makes one nonblocking inference-slot acquisition and at most
  two synchronous native embedding calls for a shipped two-word ambiguous
  phrase. No retry, blocking lock wait, whole-phrase fallback, or authority
  downgrade is allowed; an entered model call itself has no wall-clock
  preemption guarantee and can delay media-processor shutdown. The reader then
  advances only until bounded mailbox gaps, and the retained word clips remain
  bounded stack-local objects.
- Headless fakes can prove continuity, timestamp parsing, word slicing,
  generation/receipt freshness, inference bounds, and fail-closed effects.
  They cannot prove diarization, model accuracy, microphone behavior, physical
  echo separation, latency, or open-speaker barge quality.
- Owner acceptance still requires a paired bare-speaker A/B covering an
  ambiguous owner talk-over and matched own-TTS/no-talk control. That live gate
  is pending; no live result or aggregate speaker-provenance claim follows.

## Verification

- The exact focused ten-file command passed **607 tests in 6.07s**:
  `SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 19
  ionice -c 3 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p
  no:cacheprovider tests/test_kws_contract.py tests/test_speaker_gate.py
  tests/test_setup_kws.py tests/test_virtual_audio_engine.py
  tests/test_sherpa_media_session.py tests/test_barge_word_cut.py
  tests/test_sherpa_playback.py tests/test_barge_confirm.py
  tests/test_semantic_hold_capture.py tests/test_core_runtime.py -q`.
- The exact adjacent command passed **280 tests with two SWIG deprecation
  warnings in 4.81s**: `SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1 nice -n 19 ionice -c 3
  /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider
  tests/test_media_session.py tests/test_sherpa_streaming_decode_owner.py
  tests/test_sherpa_streaming_decode_session.py tests/test_speaker_input_gate.py
  tests/test_setup_doctor.py -q`.
- The standard exact complete local-Sherpa ten-file command passed **516 tests
  in 5.30s**: `SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 19
  ionice -c 3 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p
  no:cacheprovider tests/test_sherpa_streaming_decode_owner.py
  tests/test_sherpa_streaming_decode_session.py tests/test_media_session.py
  tests/test_sherpa_media_session.py tests/test_asr_postprocess.py
  tests/test_barge_confirm.py tests/test_barge_word_cut.py
  tests/test_sherpa_vad_final_gate.py tests/test_virtual_audio_engine.py
  tests/test_sherpa_duplex_runtime.py -q`.
- `/home/dobo/work/speaker/.venv/bin/python -m pytest
  tests/test_apm_double_talk.py -q` passed **6 tests in 1.15s**.
- Exact full Ruff lint is green on the three non-Sherpa production files and
  seven changed test files; Sherpa F821/F822/F823 is green. AST parsing is green
  on all 11 implementation/test files, `git diff --check` is clean, and STATUS
  is exactly 100 lines. Whole-file Ruff format remains red on exactly the same
  seven filenames as `main`; an independent audit of all 32 changed Sherpa
  hunks is green.
- The frozen four-file production diff SHA-256 is
  `8b611dabc6a42712f4e112281558ad344702d9066db117cc7386ccae3e1cec06`;
  `core/engines/sherpa.py` SHA-256 is
  `b4300779d6016016eff89b27a13a00b3a81d5acb4c6c387c674f529bcc80784d`.
- Pinned sherpa-onnx v1.13.3 source inspection only; no model, microphone,
  audio device, GPU, open-speaker, latency, quality, or live path ran. The
  paired owner bare-speaker A/B remains pending.
