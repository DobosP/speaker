# Deployment & Device Profiles

How the runtime is sized per machine and deployed per platform. Config lives in
`config.json`; the loader + profile merge are in `core/app.py`.

> **Note:** the pre-refactor knobs `profile` / `runtime_profile` /
> `transport_mode` and the `python main.py --profile …` CLI are **gone** (they
> belonged to the deleted `main.py`). Sizing is now `device_profiles`; transport
> is the choice of **engine** (`sherpa` local vs. `livekit` remote).

## Device profiles (`device` + `device_profiles`)

`config.device` picks the active profile; `--device <name>` overrides it.
`device_profiles[<name>]` is **shallow-merged over the base config per section**,
so a profile only states what differs.

| Profile  | LLM backend | Models | Notes |
|----------|-------------|--------|-------|
| `desktop` | `ollama` (GPU) | `gemma3:12b` main + `gemma3:4b` fast | Ollama is desktop-only |
| `phone`   | `llamacpp` (GGUF) | `gemma-3-4b` main + `gemma-3-1b` fast | `n_gpu_layers: 0`, `n_ctx: 2048`, STT/TTS threads dialed down |

The `phone` profile runs the **Python core** under phone-like limits (for
simulation / low-power desktops). The **shipped Flutter app** (`mobile/`) is a
separate shell and uses `flutter_gemma` (MediaPipe/LiteRT), not these profiles —
see [`../mobile/README.md`](../mobile/README.md).

## The `llm` block

```jsonc
"llm": {
  "backend": "ollama",        // or "llamacpp" (on-device GGUF)
  "main_model": "gemma3:12b",  // large / multimodal
  "fast_model": "gemma3:4b",   // snappy replies
  "router": { "backend": "heuristic", "threshold": 0.5 }  // or "learned"
}
```

The router picks fast-vs-main per turn (`core/routing.py`): `heuristic` is
dependency-light (phone-safe); `learned` lazy-imports torch (desktop).

## Engines (the transport choice)

`--engine` selects the `AudioEngine` (`core/engine.py`):

- `console` — typed I/O, no audio/models (tests + demo).
- `sherpa` — on-device mic/speaker via sherpa-onnx (the local product path).
- `replay` — the real pipeline over recorded `.npy`/`.wav` fixtures, headless (no
  sound card); used for latency benchmarks and CI.
- `livekit` — audio over a LiveKit/WebRTC room (the remote host+thin-client path).

## Deployment topologies

- **On-device (desktop):** `python -m core --engine sherpa --device desktop`.
- **On-device (Android):** the `mobile/` Flutter APK (built by
  `.github/workflows/android-apk.yml`).
- **Host + thin clients:** `python -m remote.worker` runs the brain in a LiveKit
  room; `uvicorn remote.token_server:app --port 8080` mints tokens and serves the
  `web/` client. Browser/phone become mic+speaker endpoints. Needs
  `requirements-remote.txt` + `LIVEKIT_URL/API_KEY/API_SECRET`.

## Example commands

```bash
python -m core --engine console --llm echo                 # logic only, no deps
python -m core --engine sherpa --device desktop            # local desktop
python -m core --engine sherpa --device phone              # python core, phone limits
python -m remote.worker --llm echo                          # remote brain, offline smoke
```

## Guardrails

`local_only: true` keeps STT/LLM/TTS fully local (a hard product requirement; no
cloud by default). Lazy imports mean a profile only pulls the heavy deps it
actually uses (Ollama, llama.cpp, torch, sherpa-onnx, livekit) — the base test
suite and the `console` engine need none of them.
