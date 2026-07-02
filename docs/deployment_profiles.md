# Deployment & Device Profiles

How the runtime is sized per machine and deployed per platform. Config lives in
`config.json`; the loader + profile merge are in `core/app.py`.

> **Note:** the pre-refactor knobs `profile` / `runtime_profile` /
> `transport_mode` and the `python main.py --profile …` CLI are **gone** (they
> belonged to the deleted `main.py`). Sizing is now `device_profiles`; transport
> is the choice of **engine** (`sherpa` local vs. `livekit` remote).

## Device profiles (`device` + `device_profiles`)

`config.device` picks the active profile — the committed default is `"auto"`,
which probes the host (cores/RAM/GPU/mobile, via `tools.recommend_profile`) and
applies the matching profile at launch; `--device <name>` overrides it.
`device_profiles[<name>]` is **shallow-merged over the base config per section**,
so a profile only states what differs. Profiles shipped in `config.json`
(2026-07-02):

| Profile  | LLM backend | Models | Notes |
|----------|-------------|--------|-------|
| `desktop` | `ollama` (GPU) | `gemma3:12b` main + `gemma3:4b` fast | Ollama is desktop-only |
| `desktop_gpu_4090` | `ollama` (GPU) | `gemma3:12b` + `gemma3:4b` | input gate + cleanup + capability router all on |
| `macbook_m_series` | `ollama` (Metal) | `gemma3:4b` + `gemma3:1b` | US-only cloud chains available, **default off** |
| `cpu_laptop` | `ollama` (CPU) | `gemma3:4b` + `gemma3:1b` | gates off (no fast-tier headroom); cloud default off |
| `open_speaker` | `ollama` | `gemma3:4b` + `gemma3:1b` | **explicit, never auto-picked**: WebRTC APM AEC for no-headphones barge-in (`docs/adr/0006`) |
| `phone`   | `llamacpp` (GGUF) | `gemma-3-4b` + `gemma-3-1b` | `n_gpu_layers: 0`, `n_ctx: 2048`, STT/TTS threads dialed down |
| `phone_lite` | `llamacpp` (GGUF) | `gemma-3-1b` single-tier | streaming-only ASR finals + greedy decode |

Newer model tiers (e.g. `gemma4:12b`) are pinned per machine in the gitignored
`config.local.json`, not in the committed profiles.

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
  "router": { "backend": "heuristic", "threshold": 0.3 }  // or "learned"
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

The always-on loop is fully local and raw audio never leaves the device; the
optional cloud *thinking tier* is a deliberate opt-in (`cloud.enabled=true`) and
never activates silently on API-key presence — the boundary is
`docs/target_architecture.md` §9.7 (it supersedes the older "no cloud
STT/LLM/TTS by default" stance). An invariant test keeps every committed
profile cloud-off (`tests/test_device_profile_invariants.py`). Lazy imports
mean a profile only pulls the heavy deps it
actually uses (Ollama, llama.cpp, torch, sherpa-onnx, livekit) — the base test
suite and the `console` engine need none of them.
