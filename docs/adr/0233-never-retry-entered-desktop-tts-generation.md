# ADR-0233: Never retry an entered desktop TTS generation

Date: 2026-10-09
Status: accepted

## Decision

Select the callback-less whole-waveform delivery path only when inspection of the
TTS callable conclusively rejects the intended callback keyword call and accepts
the same call without that keyword. Cache this check once per owned TTS model
under the existing TTS lock as one atomic `(id(model), support-or-unknown)` value,
without retaining a model, bound method or signature for caching. Model rebuild
publishes the new ID with unknown support. An old in-flight preflight can publish
its old ID afterward, but the successor rechecks instead of inheriting it. Opaque native
signatures use the callback API supplied by the pinned Sherpa runtime. Once any
`generate` invocation is entered, propagate errors from inference, callback DSP,
or the sink without another invocation and without changing callback support.
Preserve output DSP, per-generation cancellation, speaker/markup settings and
terminal-receipt ownership. Keep Python orchestration and the existing native
inference library; no framework or model change is needed for this defect.

## Context / why

The old broad `except TypeError` around callback generation treated every such
failure as an unsupported keyword. A device-free fake sink that accepted two
samples and then raised TypeError caused two `generate` calls, writes of two then
three samples, and a permanent switch to whole-clip delivery. The already emitted
prefix could therefore replay; the failed sentence also paid a second inference.
Even an inference TypeError before the first callback is insufficient evidence
that the keyword is unsupported. Tracking only whether a callback ran would
still repeat that native work. Signature preflight removes this ambiguity before
inference, and an unrelated argument mismatch cannot authorize the fallback.

Installed Sherpa 1.13.3 exposes an opaque pybind signature with a documented
callback API. Its model-dependent callback timing remains separate from actual
within-sentence generation: the existing English comparison found callbacks for
short phrases usually near full-clip return. This change makes no streaming,
first-audio, CPU-percent, RAM, naturalness or live barge-in improvement claim.

## Consequences

A known callback-less Python adapter still generates once and chunks the finished
waveform. An old opaque native API with no callback support now fails closed
instead of retrying a sentence on ambiguous TypeError. A native/DSP/sink failure
retains the existing failed-playback lifecycle, including its worker shutdown;
it cannot produce a completed/safe-text receipt or silently switch to a slower
path. Recovery from a broken playback worker is a separate behavior change.

Twenty-two new deterministic cases cover errors before/after callback entry,
opaque signatures, variadic keywords, malformed argument contracts, sink/DSP
failure, once-only inspection without a bound-method cache, decorated adapters, deferred RMS seed,
failure before/after fake-driver rendering, model replacement in both directions,
and obsolete preflight publication after rebuild. The focused regression command
below passed **657 tests, 2 optional-model skips** in 18.72 seconds on Linux:

```sh
SPEAKER_TEST_LOG=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ionice -c 3 nice -n 19 /home/dobo/work/speaker/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_sherpa_playback.py tests/test_streaming_tts.py tests/test_tts_markup.py tests/test_tts_backend.py tests/test_audio_frontend.py tests/test_apm_double_talk.py tests/test_barge_word_cut.py tests/test_barge_confirm.py tests/test_playback_receipts.py tests/test_engine_playback_receipts.py tests/test_playback_history.py tests/test_speaker_input_gate.py tests/test_output_leveler.py -q
```

The synthetic sink fault now makes one inference attempt and one two-sample write;
the original error propagates. The resource evidence is an avoided duplicate call
and duplicate PCM prefix in that fault case, not a native performance benchmark.
No model execution, audio device, microphone, private recording or Windows/macOS
runtime ran. Owner live tests remain stopped. The full integration gate and
physical validation are separate from this scoped headless receipt. The fake
rebuild races qualify cache identity only; general live restart while old native
inference or startup warm is unsettled remains inherited and unqualified. This
change adds no build-time lock wait or forced native preemption.

Desktop portability remains incomplete independently of this fix. Windows
cannot import `tools/promote_enrollment.py` because it unconditionally imports
`fcntl`; `core/enroll.py` atomic publication and `tools/prepare_enrollment.py` use
`os.fchmod`. Promotion also requires POSIX UID/mode/ownership and `flock` checks.
Replacing them with omitted checks or a chmod fallback would weaken protections.
A separate qualified Windows handle/ACL/reparse-point/atomic-lock abstraction and
Windows runner are required before claiming secure enrollment persistence there.
The engine can load a compatible persisted reference, but enrollment preparation,
saving and promotion are not thereby proven portable.
