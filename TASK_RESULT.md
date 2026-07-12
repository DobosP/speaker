# Task Result — canonical MiniCPM Q8 provisioning identity

Valid until: this branch lands or its implementation changes — then treat as history.

Branch: `fix/minicpm-provision-identity`

Status: implemented and verified headlessly; intentionally uncommitted.

## Outcome

- Added one self-contained production Q8 contract and pure injected verifier in
  `core/minicpm_identity.py`.
- Setup now rejects noncanonical source/alias/Modelfile inputs, validates the
  shipped Modelfile, verifies alias plus source after create, and never prints
  `ready` on an identity failure.
- Native startup, doctor, and live-session readiness inspect the configured
  Ollama host with the same list/show client and deeply verify the canonical
  MiniCPM alias. Other Ollama models retain presence checks.
- Conversation evaluation consumes the production verifier while retaining its
  evaluation-only transport headers and generic alias evidence.
- Docker provisioning has a read-only `--verify-only` command that works from
  the speaker image without requiring the Ollama CLI or a copied `deploy/`
  directory.
- Phone llama.cpp Q4 behavior is unchanged and no Q4 digest was introduced.

## Verification

- `pytest tests -q`
  → 3714 passed, 31 skipped, 9 warnings in 75.76s.
- `pytest tests/test_setup_minicpm.py tests/test_setup_doctor.py tests/test_conversation_eval.py tests/test_conversation_eval_provenance.py -q`
  → 196 passed.
- `pytest tests/test_live_session.py tests/test_capture_integration.py tests/test_imports_smoke.py tests/test_docker_entrypoint.py -q`
  → 321 passed.
- `pytest tests/test_apm_double_talk.py -q`
  → 6 passed in 0.65s.
- `python -m tools.conversation_eval --runs 3`
  → candidate 42/42, all fourteen scenarios 3/3, `pass@1=True`,
  `pass^3=True`; semantic, coverage, A/B, provenance, and warmup gates passed.
- `git diff --check` → clean.
- Independent read-only review found duplicate-`FROM`, exact-template,
  `LICENSE`, client-construction, and Docker-test gaps; all five were fixed and
  the reviewer confirmed the code findings resolved.
- All external list/show/process/audio/model seams in those runs were fake or
  disabled. No microphone, speaker, Ollama daemon, model, download, or network
  access was used.

## Remaining validation and risks

- The Docker image was not built and the documented compose verification
  command was not run against a real container/daemon.
- A real installed alias was not inspected. The exact parameter rule may expose
  previously tolerated extra Ollama parameters; that is intentional fail-closed
  behavior, but requires a real-daemon smoke before claiming host readiness.
- An earlier consumer-test invocation included a pre-existing default-preflight
  test that could make read-only local Ollama/device discovery calls. No service
  was started and no model/audio operation was requested. That test was made
  explicitly hermetic and the complete 321-test consumer command then passed
  again with echo mode, fake imports/files, and physical checks disabled.
- `--no-pull` still requires the canonical source tag to remain locally
  inspectable because source-to-alias provenance is part of the contract.
- ADR number `0062` and `STATUS.md` may conflict with concurrently landed task
  branches and must be renumbered/reconciled during integration.

## Landing recommendation

The full logic, focused identity/readiness, APM/DTD, deterministic conversation,
and whitespace gates are green. Reconcile the concurrent ADR/status numbering
before landing. No live audio gate is required for this control-plane change,
but do not claim Docker or real-Ollama verification until those explicit smoke
checks actually run.
