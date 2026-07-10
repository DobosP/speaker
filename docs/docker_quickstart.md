# Docker quickstart — cloud-streaming console on a laptop

A focused recipe for running the **console engine** in a container with
**HedgeLLM streaming tokens from a cloud provider**. Fits a 4090-laptop
class machine (16 GB VRAM + 32 GB RAM): Ollama lives in one container
with GPU passthrough, the speaker app lives in another and just speaks
HTTP to it.

This is the developer flow for verifying PR-1's cloud middle layer
end-to-end without setting up audio hardware or sherpa-onnx models.
For the on-device desktop runtime (with audio), skip Docker and run
`python -m core --engine sherpa --device desktop_gpu_4090` directly.

## What you'll have running

```
  ┌───────────────────────────────────┐      ┌───────────────────┐
  │ container: speaker-console (CPU)  │      │ Cerebras / Groq /  │
  │   python -m core --engine console │ ───► │ DeepSeek / Moon-   │
  │   --llm ollama --device …4090     │      │ shot (HTTPS)       │
  │   HedgeLLM(local, [cloud chain])  │      └───────────────────┘
  └─────────────┬─────────────────────┘
                │ HTTP :11434
                ▼
  ┌───────────────────────────────────┐
  │ container: speaker-ollama (GPU)   │
  │   MiniCPM5-1B Q8 (fast/text tier) │
  │   gemma3:12b (main, hedged)       │
  └───────────────────────────────────┘
```

Two services, one bridge network, one persistent volume for the Ollama
models. Total disk: ~9 GB models + ~200 MB image. Total VRAM when both
models are warm: ~9 GB (fits comfortably in 16 GB).

## Prerequisites

- Docker + Docker Compose v2 (`docker compose version` should say `v2.x`).
- **NVIDIA Container Toolkit** installed on the host so the `ollama`
  container can see your GPU. Linux: `apt install nvidia-container-toolkit`.
  Windows: WSL2 with NVIDIA drivers + the toolkit installed in WSL2.
  Verify with `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`.
- A cloud LLM key for at least one provider. **`OPENROUTER_API_KEY` is the
  recommended US default** (P3): one key fronts the default `public` chain
  (`gpt-oss-120b` / `llama-3.3-70b`, US-hosted). The PR-1 keys (Cerebras / Groq
  / DeepSeek / Moonshot) still work alongside it. Signup links are in
  `docker/.env.example`. **DeepSeek and Moonshot are PRC-hosted (host=CN) and
  stay disabled** unless you opt in with `ALLOW_PRC=1` in your `.env` (default
  OFF keeps the cloud chain US-only); see the cloud section below.

## Setup (one time)

```sh
# 1. Clone + cd into the repo (assumed done).

# 2. Copy the env template and paste in any cloud keys you have.
#    Missing keys silently skip that provider; the chain falls through.
cp docker/.env.example docker/.env
$EDITOR docker/.env

# 3. Build the speaker image (uses the existing lean docker/Dockerfile;
#    the only new dep vs the remote/LiveKit image is `openai`).
docker compose -f docker/docker-compose.yml build

# 4. Start ollama and wait for healthcheck.
docker compose -f docker/docker-compose.yml up -d ollama

# 5. Pull the vision/main model and provision the template-pinned MiniCPM alias
#    (one-time; cached in the named volume across runs).
docker compose -f docker/docker-compose.yml exec ollama ollama pull gemma3:12b
docker compose -f docker/docker-compose.yml exec ollama \
  ollama pull hf.co/openbmb/MiniCPM5-1B-GGUF:Q8_0
docker compose -f docker/docker-compose.yml cp \
  deploy/ollama/Modelfile.minicpm5-1b-q8 ollama:/tmp/Modelfile.minicpm5
docker compose -f docker/docker-compose.yml exec ollama \
  ollama create minicpm5-1b:q8 -f /tmp/Modelfile.minicpm5
```

Do not skip the MiniCPM alias creation: the committed fast tier references that
exact identity so Ollama uses the validated ChatML template. The 12b model is
also required by this profile for local complex/vision turns even when cloud
hedging is enabled.

## Run an interactive session

```sh
docker compose -f docker/docker-compose.yml run --rm speaker
```

You'll see something like:

```
[entrypoint] /models/sherpa_paths.json not found; sherpa STT/TTS disabled
[entrypoint] merged config overlay from /app/config.overlay.json
[entrypoint] llm.host -> http://ollama:11434
[entrypoint] llm.keep_alive -> 30m
[console] mode=assistant. Type to talk; Ctrl-D to quit.
[console] try: 'research mode', 'assistant please help', 'stop'
```

Then type questions. Each turn the brain classifies sensitivity
(`core/sensitivity.py`), picks the right chain (`private` / `code` /
`public`), and `HedgeLLM` races local (Ollama via the sibling container)
against the cloud chain. Whichever produces a token first wins.

The default `public` chain leads with **OpenRouter** (`OPENROUTER_API_KEY`,
US-hosted) — set that one key and you have a working US-only cloud tier.
**DeepSeek and Moonshot are PRC-hosted (host=CN)** and are dropped from the
chain unless you opt in with `ALLOW_PRC=1` in `docker/.env` (Locked Decision 2:
US-hosted chains are the default). With `ALLOW_PRC` unset the entrypoint leaves
`llm.cloud.allow_prc` at its config default (`false`); set `ALLOW_PRC=1` and the
entrypoint flips it on so those PRC presets join the chain. Either way the §9.7
boundary holds: raw audio + PII never leave the device — only post-ASR text the
assistant was given crosses to the cloud.

The streaming you'll see in the console output is the live token feed
from whichever path won. `logs/runs/run-<id>.summary.json` (mirrored to
the host via the `../logs:/app/logs` mount) records which chain ran +
per-call TTFT + reasoning chars per turn -- inspect it after the
session to see how often cloud won versus local.

### Useful one-shot variants

```sh
# Smoke-test every configured provider with a real API call (counts
# tokens, ttft, total). Skips providers without a key.
docker compose -f docker/docker-compose.yml run --rm \
    --entrypoint "python tools/llm_sanity.py" speaker --all

# Pure local (no cloud), for a baseline TTFT comparison.
docker compose -f docker/docker-compose.yml run --rm \
    -e SPEAKER_CONFIG_OVERLAY=/nonexistent speaker

# Drop into the container shell to poke around.
docker compose -f docker/docker-compose.yml run --rm \
    --entrypoint /bin/bash speaker

# Run a different device profile (e.g. force fallback strategy).
docker compose -f docker/docker-compose.yml run --rm \
    speaker --engine console --llm ollama --device cpu_laptop
```

## Teardown

```sh
docker compose -f docker/docker-compose.yml down            # stop containers
docker compose -f docker/docker-compose.yml down --volumes  # also delete pulled models
```

## Troubleshooting

- **`docker: Error response from daemon: could not select device driver "nvidia"`**:
  the NVIDIA Container Toolkit isn't installed on the host. See the
  prerequisites section -- on Linux it's `apt install nvidia-container-toolkit`
  + `systemctl restart docker`.

- **All providers SKIP in the smoke test**: none of the cloud API key
  env vars are set in your `.env`. Add at least one (`OPENROUTER_API_KEY`
  is the US default) and re-run. Note DeepSeek/Moonshot also need
  `ALLOW_PRC=1` to be exercised — without it those PRC presets are dropped
  before the key is even checked.

- **Console echoes nothing**: check `logs/runs/run-*.txt` -- if the
  Ollama call timed out (`OLLAMA_HOST` mis-set, model not pulled), the
  cloud chain still runs but local can't fall back. If every cloud is
  also unset/down, the turn returns empty. The new
  `tools/llm_sanity.py --all` reproduces this in isolation.

- **`cloud: enabled=false` in the merged config**: the overlay didn't
  load. Confirm `docker/config.overlay.json` exists and the
  `[entrypoint] merged config overlay` line appeared in startup logs.
  The `SPEAKER_CONFIG_OVERLAY` env var defaults to `/app/config.overlay.json`
  and the compose file mounts the host file there.

- **Container exits immediately**: you used `docker compose up speaker`
  instead of `docker compose run --rm speaker`. The console engine
  needs an interactive TTY; `up` runs detached so stdin is closed.

## What's NOT in this flow

- **Audio hardware passthrough**. Containers + microphones on Linux is
  doable (`--device /dev/snd`) but ugly + platform-specific. For audio,
  run the desktop runtime directly: `python -m core --engine sherpa`.
- **The LiveKit remote/host path**. That's a separate docker-compose
  setup that reuses the same `Dockerfile` with `python -m remote.worker`
  as the command. See the legacy `docker/.env.example` Flow B section.
- **The memory layer (PR-2)**. The cloud-streaming console doesn't use
  Postgres; memory is in-process only. To exercise the memory layer in
  Docker, add `psycopg[binary,pool]` to `requirements-docker.txt` and a
  third service for Postgres + pgvector.
