Valid until: this branch is landed or superseded — then treat as history.

# Task result

## Outcome

The public-v2 draft now fails closed around provenance and binds each decoder
smoke report to one exact corpus/model/decoder tuple.

- The accepted manifest contains only hash-pinned Google Speech Commands and
  AMI inputs with bounded SPDX identifiers and cache-only policy. Unresolved
  CMU/Festvox and Microsoft AEC inputs were removed.
- Strict loading rejects duplicate keys, non-finite values, unknown fields and
  tags, unsafe annotation/member/group relationships, hidden or case-fold
  duplicate filenames, redirected downloads, and sources changed immediately
  before extraction. Extraction uses private hash-verified source snapshots,
  so source change/restore cannot alter attributed outputs.
- The evaluator validates the exact manifest and generated metadata, complete
  source set, every case (including excluded assertions), complete recursive
  model tree, implementation hash, package versions, and fixed decoder config.
  It rejects duplicate/malformed/missing/extra cases and rechecks corpus/model
  snapshots after decode. Eligible PCM comes from verified in-memory bytes and
  the recognizer loads only from a private read-only on-disk model snapshot.
- Reports include a versioned evaluator binding with exact SHA-256 identities
  for the public evaluator, aggregate metrics, and WER implementation.
- Empty-reference raw stripped text owns `nonempty_transcripts`/`exact_empty`;
  normalized text owns WER/CER. Excluded assertion counts are explicit.
- AMI laughter is event-only, annotation speaker identities are unique, and
  expected text uses token-subsequence proof.
- Product scenarios use current runtime semantics and identifiers. `wait`,
  `hold on`, and bystander handling are explicitly unsupported future
  taxonomy.
- FSDD adapted arrays now have named creator/project attribution, license link,
  change notice, and CC BY-SA 4.0 adapted-material terms. The unrecorded source
  revision remains explicitly unknown.

## Files changed

- `NOTICE`
- `STATUS.md`
- `TASK_RESULT.md`
- `docs/adr/0085-harden-public-voice-regression-provenance.md`
- `docs/public_voice_regression.md`
- `tests/fixture_audio/FSDD-NOTICE.md`
- `tests/fixture_audio/real_usage_full/metadata.json`
- `tests/fixture_audio/virtual_real_world/metadata.json`
- `tests/fixtures/public_voice_manifest.v2.json`
- `tests/test_public_stt_eval.py`
- `tests/test_public_voice_fixtures.py`
- `tools/public_stt_eval.py`
- `tools/public_voice_fixtures.py`

## Verification

No corpus, package, or model was downloaded. No model decode, GPU job, audio
device, or live test ran.

Focused deterministic gate:

```text
PYTHONDONTWRITEBYTECODE=1 SPEAKER_TEST_LOG=0 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
/home/dobo/work/speaker/.venv/bin/python -m pytest -p no:cacheprovider \
tests/test_public_voice_fixtures.py tests/test_public_stt_eval.py -q
................................................................         [100%]
64 passed in 0.78s
```

Fresh repository landing gate, pinned to four CPUs at nice 19 with numerical
libraries limited to one thread:

```text
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
taskset -c 28-31 nice -n 19 \
/home/dobo/work/speaker/.venv/bin/python -m pytest -p no:cacheprovider \
-m "not real_model" -q
5765 passed, 13 skipped, 20 deselected, 9 warnings in 111.05s
```

Static check:

```text
/home/dobo/.local/bin/ruff check --no-cache \
tools/public_voice_fixtures.py tools/public_stt_eval.py \
tests/test_public_voice_fixtures.py tests/test_public_stt_eval.py
All checks passed!
```

Manifest inventory:

```text
commands: 12 fixtures, up to 119.2 MiB of pinned sources
conversation: 3 fixtures, up to 60.7 MiB of pinned sources
development/regression only; not final held-out or live evidence
```

`STATUS.md` is 99 lines. `git diff --check` and untracked-file whitespace
checks completed without diagnostics.

## Remaining evidence

This branch establishes deterministic data governance and whole-clip decoder
smoke plumbing only. It does not establish production STT selection, streaming
latency, endpointing, AEC, target/bystander audio behavior, agent-tool
execution, owner-voice performance, or physical barge-in.

A later branch still needs timestamped frame replay through the real session
and action boundary, licensed AEC pairs, consented/disjoint owner data, and a
fresh `./live.sh` A/B on the current microphone, room, and speaker route.

## Handoff

The focused gate and static checks are green. Root should independently review
the exact-binding logic and run the repository-wide landing gate before merge.
