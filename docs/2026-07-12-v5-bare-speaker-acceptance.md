Valid until: a newer enrollment or live-audio ADR supersedes ADR-0056 — then treat as history.

# V5 bare-speaker acceptance

This is the manual gate after the device-free prep, a successful three-pass v5
enrollment, and a green doctor. It makes sound and uses the microphone. Do not
start it without the operator at the laptop and explicitly ready.

Launch the production hybrid from the prepared feature worktree:

```bash
SPEAKER_KEEP_RUNS=0 /home/dobo/work/speaker/.venv/bin/python -m core \
  --engine sherpa --device desktop_gpu_4090 \
  --model gemma3:12b --fast-model minicpm5-1b:q8
```

Stay silent through the 1.5-second calibration and any one retry. Startup is red
unless it reports normal-final speech evidence armed, an enrolled speaker-ID
gate, and completed speaker-ID warm-up without a frontend mismatch, fail-open,
or unstable calibration.

Run on the bare laptop speakers, without headphones:

1. Say a soft exact `Yes`: one ordinary-evidence final, no drop.
2. Say `What is the ...`, pause about one second, then `capital of France?`:
   no reply during the pause and exactly one Paris answer.
3. Ask for a detailed troubleshooting explanation, then remain silent through
   the reply plus five seconds: zero cuts or self-generated user turns; every
   fragment has one unchanged voice and `sid=0`.
4. On two fresh long replies, wait more than 0.4 seconds after first audio and
   say `Actually, tell me a short joke instead`: both old replies audibly stop,
   no stale sentence plays, and the new request is answered. Each log path needs
   at least four novel words, accepted speaker authority, word-cut confirmation,
   and a non-null barge-to-FIFO-stop receipt.
5. During a long reply containing the word “stop,” say exact `Stop`: prompt cut,
   no follow-on speech; then a fresh Paris question proves recovery.
6. Begin `Actually add one practical example please` over the last 1–3 words and
   continue past playback end: exactly one reply and one staged/tail handoff.

Across the run require a CLEAN self-interrupt summary, no self-echo confirmation,
no speaker cold/unavailable/error event, and no `-9999`, restart, or corruption.
End with one Ctrl-C and wait for the log paths. Spoken Stop only stops speech.

Low sensitivity is a separate restart so calibration observes it: record the
current `echo-cancel-source` volume, set the historical canary 13%, relaunch,
repeat soft Yes, one normal question, and one seven-word override, then Ctrl-C and
restore the exact original volume. Failure at 13% is red; never leave the source
volume changed.

Only after every item passes and the operator explicitly accepts the run, stop
the runtime and use ADR-0066's device-free promotion command. The accepted name
must derive from the candidate, for example:

```bash
/home/dobo/work/speaker/.venv/bin/python -m tools.promote_enrollment \
  --worktree /absolute/feature/worktree \
  --primary-config /absolute/primary/config.local.json \
  --expected-candidate /absolute/feature/worktree/pretrained_models/sherpa/speaker/enrollment.v5-UNIQUE-ID.json \
  --expected-source-enrollment /absolute/primary/pretrained_models/sherpa/speaker/enrollment.json \
  --expected-backup /absolute/primary/pretrained_models/sherpa/speaker/enrollment.pre-v5-UNIQUE-ID.json \
  --accepted-enrollment /absolute/primary/pretrained_models/sherpa/speaker/enrollment.v5-UNIQUE-ID-accepted.json \
  --accept-live-gate
```

Exit 0 activates the new pointer. Exit 2 refuses before this invocation commits
accepted/config state. Exit 3 proves verified accepted bytes are staged while the
old pointer remains inactive; rerun the identical command to adopt that exact
orphan. Exit 4 is ambiguous: inspect every supplied path before deciding whether
to retry. Never replace or delete v4, the preparation backup, or the isolated
candidate during this workflow.

Logs can prove detector-to-FIFO causality. Physical onset, voice continuity, and
ear quality require the operator's explicit grade unless WAV recording is
separately opted in.
