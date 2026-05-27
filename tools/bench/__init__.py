"""Real-model latency benchmark for the voice runtime.

Runs the actual ASR -> LLM -> TTS pipeline (sherpa-onnx + llama.cpp) headlessly
over recorded audio fixtures, records per-turn latencies via ``core.metrics``,
and reports the measured numbers against the ``tools.specsim`` budgets.

  python -m tools.bench --fake                 # plumbing smoke test, no models
  python -m tools.bench --profile phone         # real models over fixtures

See ``docs`` and the project plan for the cloud (GitHub Actions) perf job.
"""
