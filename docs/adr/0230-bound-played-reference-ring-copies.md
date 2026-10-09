# ADR-0230: Preserve the played-reference timeline with bounded copies

Date: 2026-10-09
Status: accepted

## Decision

Keep `FarEndRing` as the shared played-reference timeline and preserve each retained
sample's absolute position modulo capacity for every write. Read each delayed
window using at most two contiguous NumPy copies under the existing single
snapshot lock, with detached zero-padded outputs allocated before that lock.
Do not change an AEC backend, route, calibration, enrollment, authority policy,
default, or physical qualification through this change.

## Context / why

The capture reader obtains its zero-delay and operating-delay windows through
one `read_windows` call. Its original indexed gather allocated an integer index
array, validity masks, modulo indices and gathered samples while holding the
same lock used by playback writes. The retained interval is contiguous in
absolute sample time, so two array slices suffice even when storage wraps.

An oversized write also had a correctness defect. It copied its retained tail
to storage index zero, although the absolute write head could have a nonzero
remainder modulo capacity. A deterministic baseline example with capacity 8,
3 initial samples and a following 10-sample write returned 8 incorrect samples
out of the 8 retained samples. The old test's exactly twice-capacity write hid
this case. The corrected write starts at the retained tail's absolute position.
There is no evidence that oversized writes caused the stopped owner's live
failures; ordinary playback callbacks are much smaller than the default ring.

A new Rust/C++ module is unwarranted for this operation: NumPy already performs
the copies in native code, and the useful change removes unnecessary work and
repairs indexing. These microsecond savings cannot explain seconds of agent
latency or establish acoustic quality.

## Verification and performance evidence

Base source: `734713dde33b0f5bdf35430258fb066535f76985`.
The standalone diagnostic is `python -m tools.reference_ring_benchmark`.
It opens no device, loads no model or recording, invokes no network/API and
uses only generated timeline indices. Its gather comparator preserves the
previous read algorithm while sharing the corrected writer and lock, so the
timing comparison isolates reads, not the writer repair or an entire runtime.
The comparator's read-method AST, excluding its docstring and source positions,
matches the base commit's read method exactly. All three timed cases produce
exactly equal output arrays.

Measured on Linux x86-64, Intel Core i9-13980HX, CPython 3.12.3 and NumPy 2.4.6,
with two-CPU caller affinity, `nice -n 19`, `ionice -c 3`, and the testing guide's
single-thread environment. Each path receives 100 warm-up reads followed by
7 blocks of 5,000 reads. The table reports the median of the seven wall-time
block means; it is not a distribution of individual callback tail latency.
Tracemalloc runs separately for ten reads; its peak is not RSS or whole-agent
memory. The source hashes are checked before/after the measurement, and the
report separately binds the loaded reader code objects.

| Synthetic paired read | Gather median | Bounded-copy median | Gather / copy | Traced peak, gather / copy |
| --- | ---: | ---: | ---: | ---: |
| 160 samples/window (10 ms) | 20.56 us | 6.43 us | 3.20x | 6,480 / 2,216 bytes |
| 1,600 samples/window, wrapped | 49.04 us | 6.20 us | 7.91x | 54,000 / 13,896 bytes |
| 1,600 samples/window, partial eviction | 40.99 us | 8.32 us | 4.93x | 54,000 / 13,896 bytes |

Aggregate report SHA-256:
`8d26689dab34ea09ff01fd7c48e9b7a2958371d82fedacef827bbd497007889b`.
Benchmark tool source SHA-256:
`5533611ddf7f5a1d752645f90de4b36277440e7882a64acb705bb63a1806b0fd`.
AEC source SHA-256:
`3d36d59550af1837d143d67c8db45a56e12fae9f43147a3c2567848eb699cf3a`.
The generated aggregate report remains task-local; it contains no audio or text.

Headless gates used the low-priority prefix in `docs/agent-testing.md`:

- `tests/test_far_end_ring_timeline.py tests/test_aec_seam.py tests/test_apm.py
  tests/test_apm_double_talk.py tests/test_reference_recording.py`: 91 passed,
  2 optional-model skips in 8.16 s.
- `tests/test_reference_ring_benchmark.py tests/test_sherpa_playback.py
  tests/test_echo_coherence.py tests/test_echo_probe.py tests/test_denoise.py`:
  223 passed in 8.25 s.
- `tests/test_microsoft_aec_apm_dtd_eval.py tests/test_capture_integration.py
  tests/test_sherpa_media_session.py`: 123 passed in 20.61 s.
- Scoped E9/F63/F7/F82 Ruff checks pass; the three new Python files pass format
  checking. Existing AEC-file formatting debt is not reformatted by this task.
- Documentation check: 38 files, no dead links, stale terms, retired verbs or
  orphan documents; `git diff --check` passes and `STATUS.md` remains 120 lines.

The timeline tests cover whole-ring replacement at arbitrary head positions,
seeded arbitrary writes and delayed windows, wrap/eviction/zero padding,
independent returned arrays, reset, and consistent paired snapshots during
concurrent wrapping writes. Diagnostic tests check bounded parameters,
source-change refusal and affinity restoration on success and failure.

## Consequences

Playback and capture retain the same ring capacity, delay coordinates, atomic
paired snapshot, zero padding and reset contract with less transient work.
The lock remains; this is not a lock-free or hard-real-time implementation.
The benchmark does not measure lock contention/tail latency, drift tracking,
echo suppression, double talk, ASR quality, admission, STOP, enrollment, an
entire assistant, a physical phone, or thermal behavior.

The owner's microphone/doctor/live stop remains in force. The protected live
candidate worktree and private recordings are untouched. Physical route A/B
and acoustic acceptance remain open. LocalVQE evaluation is a separate future
tool/candidate task; this change downloads no weights and supplies no new AEC
or live-promotion authority.
