# ADR-0063: Fail fresh-install readiness closed

Date: 2026-07-12
Status: accepted

## Decision

Split the native desktop install into two explicit readiness stages. Stage one
installs the Python/audio base, including SciPy and soxr, then downloads the
streaming ASR/VAD files plus the selected SenseVoice, GTCRN, Kokoro, and default
speaker-ID artifacts. Invoke `tools.setup_models --require-selected`; publish
`config.local.json` with one atomic replacement only after every required path
exists. Propagate model-setup and base-doctor exit codes without printing a
success handoff. `--skip-models` is dependency-only and exits 2 as incomplete.

Run the stage-one doctor with `--defer-ollama`. That mode may report `BASE READY`
only for an Ollama-backed selected profile whose non-LLM imports, artifacts,
audio devices, and frontend route all pass; it never imports or contacts Ollama
and never reports full `READY`. Stage two provisions Gemma and the pinned
MiniCPM alias, then requires an ordinary `python -m tools.doctor` `READY` result
before launching Sherpa.

## Context / why

The committed desktop profile selects SenseVoice and GTCRN, and the adopted
desktop TTS is Kokoro, but the one-command installer fetched only the baseline
streaming/VITS set. Its lean dependency list also omitted SciPy and soxr even
though production coherence and resampling depend on them. Optional-download
failures could still publish a partial local config, and the installer ignored
both setup and doctor return codes before saying it was done.

Ollama and its model weights are intentionally provisioned outside the Python
installer. Treating their absence as a stage-one failure made the base install
impossible to attest; ignoring that absence and printing `READY` would instead
authorize an unrunnable assistant. A distinct `BASE READY` result preserves
both facts.

## Consequences

- A failed selected download or config replacement leaves the prior local
  config unchanged and makes the installer nonzero.
- Fresh installs download more speech data because their result matches the
  selected production profile rather than a degraded fallback.
- Dependency-only installs are still supported, but automation must treat exit
  code 2 as incomplete rather than success.
- `BASE READY` proves no Ollama/model state. The normal doctor remains the sole
  full-runtime readiness verdict, and live microphone/speaker quality still
  requires separate hardware validation.
