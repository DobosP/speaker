# ADR-0249: Compare literal resource changes in recorded response experiments

Date: 2026-10-28
Status: accepted

## Decision

Extend the ADR-0246 development evaluator with three literal variants against
its verified VITS control: `vits-asr2` changes only `sherpa.asr_num_threads`
from 4 to 2, `vits-asr8` changes that field from 4 to 8, and `vits-tts4` changes
only `sherpa.tts_num_threads` from 2 to 4. Refuse a control whose declared field
has a different value/type. Bind the control selection SHA, variant name and
one exact previous/requested field delta in the existing selection SHA.

Preserve model files, streaming/final decoding, hotwords, verifier, context and
reply budgets, sampling, turn holds, expression/DSP, routing, capabilities,
privacy and authority. No production/default/local-config change, generic
`--set`, new model, relaxed pairing hash or report schema is introduced.

## Context / why

The separately retained 2026-10-28 cold development triad completed 12/12
recorded turns per condition with identical corpus, source, ASR, LLM and local
model bindings. VITS observed complete-PCM → first post-DSP answer PCM at
1.982 s median; VITS-fast at 1.969 s. This does not establish a material gain
from the fragment policy. VITS stage medians were approximately 0.579 s to the
selected final, 0.662 s model-request → first token and 0.284 s TTS-admission →
first PCM; these are separate distributions, not an additive median model.
First/subsequent cohorts and variable reply lengths remain explicit.

The code already passes `resolved_asr_threads` to streaming and offline Sherpa
recognizers, and `resolved_tts_threads` to the TTS model. The ASR knob therefore
changes both those native pool requests, not only final recognition or the
whole process's CPU budget. More threads can trade resources for latency or
increase contention; neither direction is a prior performance win. The
configured CUDA Faster Whisper verifier is retained unchanged.

A repeated approximately 1.212-second pre-model delay on one ordinal matches
the existing incomplete-turn hold. Its ASR/input seam is being investigated
separately. These resource experiments do not disable or shorten that safeguard.
The prior configured startup spent 119.440 seconds of its 120-second first-case
budget before response work; these prospective experiments continue the declared
warm-disabled protocol instead of changing deadlines or forcing short answers.

## Consequences

- Tests recursively compare complete resolved configurations and require exactly
  the one named field difference. All original settings and inputs stay unchanged.
- Full ASR settings SHA remains different for ASR-thread variants. Existing
  `compare_paired` continues to refuse those pairs; no relaxed behavior hash
  makes them equivalent. The orchestrator compares these named resource
  ablations only after retaining an external compact full-config delta witness
  and checking all non-resource bindings. TTS4 retains the exact ASR identity
  and remains eligible for the existing pairing helper.
- Obtain a fresh VITS baseline on the same newly committed source before any
  candidate cell. Run one condition at a time on the original Mic6 corpus with
  two repeats, unchanged 120-second case/1800-second cell limits, normal
  priority, unchanged CPU affinity and environment. Keep first/subsequent rows,
  quality/final/outcome counts, output lengths, worker CPU/RSS and host-load
  evidence. Do not turn a faster but lower-quality selection into a win.
- These remain after-complete-PCM, unpaced recording/null-sink observations.
  Capture, AEC, real-time endpointing, DAC/acoustics and physical-device
  validation are excluded. Existing private recordings are not copied/relabelled,
  and owner/human-held-out or default-promotion authority is not created.
- Keep one rolling reproducible setup and compact receipts. Preserve original
  baselines, interrupted/failed receipts, source branches and other owners'
  work; this change prunes no historical artifact tree.

## Verification

`tests/test_recording_response_benchmark.py`: 61 passed in 2.64 seconds with
native-free public fakes and a fresh owned scratch directory; scoped Ruff and
whitespace checks pass. The 13 new regressions verify literal-name/value
bounds, exact one-field deep differences, unchanged model/LLM/privacy/hold
settings and the existing strict ASR-pair refusal. No native/model job,
microphone, speaker, doctor, download, merge or default change ran for this
source receipt. Actual performance remains pending a serialized native window.

## Completed native resource observations (2026-10-28)

All four cells ran source `66442ad50c3f27ec0ec6cf0dab3af018db3a5982` and
completed 12/12 responses with source/model/config/corpus checks after execution.
The compact comparison and original cells remain under
`logs/runs/english-recorded-quality-latency-20261028/resource-pass/`. Independent
review reproduced each full resolved configuration, verified exactly its named
thread delta and checked all unaffected bindings. Full ASR hashes remain
different for ASR2/ASR8; no strict-pairing rule was relaxed. Each repeat retained
3 legacy errors / 25 reference words and 4 English-normalized errors; no task
or TTS-fragment failure was recorded.

The following latency distributions are the **11 subsequent cases**, excluding
the separate first cold response. They measure complete PCM availability to
first finite nonzero post-DSP answer PCM at the null sink.

| Condition | Subsequent p50 / p95 (s) | First cold response (s) | Whole-cell worker CPU (s) | Peak worker RSS (decimal GB) | Total provider characters / reply fragments |
|---|---:|---:|---:|---:|---:|
| VITS control, ASR4 / TTS2 | 3.189 / 4.206 | 10.970 | 306.1 | 2.505 | 6543 / 65 |
| VITS ASR2 | 3.035 / 4.516 | 11.570 | 211.1 | 2.451 | 5314 / 61 |
| VITS ASR8 | 4.306 / 6.324 | 13.929 | 553.8 | 2.662 | 5618 / 56 |
| VITS TTS4 | 3.035 / 5.003 | 11.583 | 421.2 | 2.510 | 7264 / 77 |

Observed selected-final medians were 1.246 / 1.057 / 2.308 / 1.247 seconds in
the table order. TTS4's subsequent TTS-admission-to-PCM median was 0.222 seconds
versus control 0.477, but its reply lengths, fragment counts and host load
differed. ASR2 used fewer observed worker CPU seconds; ASR8 was unhelpful in
this cell. These are resource leads, not causal per-knob gains or an accepted
default. Host CPU snapshots ranged from 75.2% to 99.3%, with sustained unrelated
work and changing load; no other owner's process was modified. Normal priority
and 32-CPU availability were retained. Worker CPU/RSS exclude the Ollama server.

The earlier approximately 1.98-second VITS result used a lighter host and a
different source campaign. It is not a comparable regression baseline for this
contended pass. All-12 p95 includes the cold first case and must not be confused
with the subsequent-only p95 above. Capture/DAC/acoustics, live readiness,
held-out quality, capability/default promotion and another platform remain
unqualified. No production/local configuration or original recording changed.
