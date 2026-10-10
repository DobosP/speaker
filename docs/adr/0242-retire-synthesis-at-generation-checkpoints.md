# ADR-0242: Retire TTS synthesis at generation checkpoints

Date: 2026-10-10
Status: accepted

## Decision

Reject a known stopped or superseded TTS generation before model-lock admission,
after the model lock is acquired, and immediately before native generate. A
callback for an already retired generation returns stop before reading/converting
samples or starting DSP. Recheck at callback DSP/write boundaries and after a
successful native return before whole-clip materialization, processing or carry
publication. Do not retry entered native generation or swallow its errors.

Keep the model lock across entered native generation and callbacks until their
true return. The existing producer FIFO independently rejects superseded samples;
its exact receipt, fade, resampling and output-cleanup ownership are unchanged.
Healthy model parameters, expression/speaker-lock policy, PCM samples, queue
bounds, inference threads and the batch/callback capability selection are retained.

## Context / why

On 1bed9e2 a synthetic lock/Event probe showed that a request cancelled while
waiting for the model lock still entered native generate after lock release.
Callbacks invoked after stop still converted/processed/offered a chunk before
returning stop; a cancelled batch return still entered whole-clip DSP. Although
the production FIFO rejects stale audio independently, this unnecessary work can
hold the single synthesis worker ahead of a successor for an entire native render.
It can also spend waveform conversion/filter/quality work after native completion.

The same probe now reports retired native entry, cancelled callback DSP/direct
sink offers and cancelled whole-clip DSP each reduced from one to zero. No model
or audio device was loaded. A deterministic successor test progresses without
releasing the fake obsolete render, because that render never entered. Healthy
PCM is bitwise identical, including locked/unlocked voice selection and emotion
speed. These are resource-work/ownership results, not native RTF or audible latency.

## Consequences

- Revocation known before native entry skips that work. Revocation after native
  entry remains cooperative/model-dependent; no C++ kernel preemption, forced
  model teardown or hard native cleanup deadline is added.
- Cancellation does not lend the TTS lock to a successor while an entered call
  is still running. Callback stop and true native return are separate events.
- Successful retired batch returns do not materialize their native samples.
  Already-entered DSP may finish its current operation; later work/carry is
  withheld after the next checkpoint. No state rollback of audio already offered
  or heard is claimed.
- Entered native failures still propagate once through existing output-failure
  handling. Caller/FIFO interruption and terminal receipts remain authoritative;
  no completed or audible receipt is inferred merely from an early return.
- Two older tests encoded the defective one-stale-chunk behavior; they now assert
  zero native entry/write for known stop or generation mismatch. Other healthy
  synthesis/receipt/markup/DSP regression contracts remain intact.
- One compact synthetic experiment is retained with its baseline and source
  bindings. No private recordings, native benchmark, downloads, microphone,
  doctor, route change or live validation accompanies this decision. Actual
  desktop/other-OS latency and physical cancellation quality remain open.
