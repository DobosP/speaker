# Session 2026-06-02 — recovered + landed the PC-interrupted playback rewrite (callback `OutputStream` + `PlaybackFIFO`)

Cross-machine handoff (ran on the **desktop**: i9-13980HX, RTX 4090 Laptop).
Written in-repo because per-user Claude memory does NOT travel between machines.
Goal of this session (user's words): *"pc shut down during a session, continue
working on that session."*

## What I found (the interrupted work)

The bootstrap showed the *previous committed* session (the docs-unification
landing, `e05d14c`) was clean and done. But the working tree held **uncommitted,
never-landed in-flight work** — that was the session the shutdown interrupted:

- `core/engines/_aec.py` — a new **`PlaybackFIFO`** class (bounded single-producer
  / single-consumer float32 ring with backpressure + an abort escape).
- `core/engines/sherpa.py` — the TTS playback path rewritten from a blocking
  `out.write()` push to a **callback-driven `sounddevice.OutputStream`**.
- `tests/test_aec_seam.py` + `tests/test_sherpa_playback.py` — FIFO unit tests +
  reworked barge-in/shutdown tests.
- (Plus ~59 incidental `logs/runs/*` churn files — **not** part of the code work;
  left untouched, kept out of the commit.)

**The change & why it matters.** The far-end AEC reference must align to *true*
acoustic playback (DTLN tolerates only ±60 ms). The old blocking `out.write()`
handed a whole multi-second TTS chunk to the device instantly, so the far-ring
write head raced ahead of real playback by a swinging 50–1200 ms — far outside
tolerance, so the deep canceller couldn't work live. The rewrite routes audio
through the FIFO; the PortAudio **callback drains it and tees the EXACT
just-played block** into the far ring (+ coherence ref + level EWMA) and stamps
`TTS_FIRST_AUDIO`. Backpressure on a full FIFO paces synthesis to real time.
Barge-in/shutdown now **flush the FIFO** (stream stays open playing silence /
clean `stop()`+`close()` on teardown) instead of `abort()`ing the stream.

## What I did

1. **Verified the interrupted work**: all files parse; affected tests pass; full
   logic suite green (1269 → after my additions 1273); import smoke PASS.
2. **Ran an adversarial review before landing** (subtle real-time concurrency on
   `main` warranted more than green tests). 11 candidate findings → **6 confirmed,
   all minor; 0 blockers.** The 5 scariest candidates (AEC drift accumulation,
   mis-gating the utterance tail as barge-in, the callback lock as a *correctness*
   bug, an unguarded `outdata[:,0]`) were **refuted** with concrete reasoning.
3. **Hardened the confirmed gaps** (see below), re-verified green, and **landed**.

## What landed (`main` → `554b300`)

- **`d53a1f5`** `feat(aec): align far-end reference to true playback via callback
  OutputStream + PlaybackFIFO` — the interrupted rewrite **plus** the review
  hardening, as one coherent feature commit.
- **`554b300`** the `--no-ff` merge into `main` (pushed to origin).

### Review hardening folded into the commit
- **`PlaybackFIFO.write()` abort contract — documented, NOT "rolled back".** The
  review's option-1 (roll back a partial enqueue on mid-chunk abort) is **unsound**:
  I traced that under the production path (`stop()` flushes, then the producer
  aborts) it does `count -= i` after `flush()` already zeroed `count` → **negative
  count → buffer overrun on FIFO reuse**; and a true rollback is impossible anyway
  once the callback has already *played* part of the chunk. The correct,
  achievable contract (now in the docstring + a new test): `should_abort` halts
  further enqueue; the **paired `flush()`** (which `stop_speaking`/`stop` always
  call) drops the queue.
- **`_audio_cb` docstring made honest.** It claimed "the only lock taken is the
  FIFO's" — false. It also takes `FarEndRing`'s lock and, via `note_playback`,
  `EchoCoherenceDetector`'s lock (the one lock contended with the capture thread's
  `_snapshot_ref` concat). Bounded + harmless at the default `coherence_ring_ms`
  (~38 KB / sub-100 µs), so **not redesigned** (moving it back regresses the
  alignment the rewrite fixed) — documented + filed as a backlog item.
- **Tests:** added direct device-free `_audio_cb` coverage (drain + underrun
  zero-fill + tee **only** the real `view[:n]` + stamp-once); a mid-chunk-abort +
  flush contract test; **pinned the consumer→producer notify path** (the 0.1 s
  `wait()` self-wake backstop was masking a dropped `notify_all` — mutation-checked
  that the pinned test now fails if `notify_all` is removed); replaced a vacuous
  `assert _fifo.flush` truthiness check with an identity + emptied-count assertion.

## Verification

- **Full logic suite: 1273 passed, 13 skipped, 0 failed** (`python -m pytest tests -q`).
- **Whole-tree import smoke: PASS** (120 modules, `python tools/run_tests.py imports`).
- 2 pre-existing numpy divide-by-zero RuntimeWarnings in `core/endpointing.py`
  (mel filter) — not failures.

## Environment on this desktop
- `python` is the right interpreter here (anaconda 3.12 with sherpa/onnxruntime);
  the full suite runs directly. Push to `origin/main` worked (not read-only).
- The ~59 `logs/runs/*` working-tree changes (old runs deleted, today's added) are
  **still uncommitted by design** — incidental run-log churn, not code. Decide
  separately whether to commit/restore them; they're git-tracked, so nothing is lost.

## Next steps (pick up here)
1. **AEC on real hardware (P1, needs the mic) — now unblocked by this landing.**
   The far-end reference is finally aligned to true playback, so this is the
   moment to: enable `aec` in `config.local.json` (`aec_enabled=true`, start
   `aec_backend="nlms"`), calibrate `aec_ref_delay_ms` and **measure post-AEC
   ERLE** with `tools/echo_probe.py`, then try `aec_backend="dtln"`. Confirm no
   self-interrupt AND a real interrupt still cuts through.
2. **Smart Turn v3 live floor A/B** (the other P1 hardware item; detector +
   `tools/turn_detect_check` are on main, needs real-speech recordings).
3. **Latent real-time item (backlog):** before raising `coherence_ring_ms`, move
   `EchoCoherenceDetector.note_playback` off the audio callback (feed it from a
   lock-free SPSC stage drained on the capture/worker thread, like `FarEndRing`).
