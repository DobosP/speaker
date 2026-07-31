# Deployment & Device Profiles

How the runtime is sized per machine and deployed per platform. Config lives in
`config.json`; the loader + profile merge are in `core/config.py`.

> **Note:** the pre-refactor knobs `profile` / `runtime_profile` /
> `transport_mode` and the `python main.py --profile …` CLI are **gone** (they
> belonged to the deleted `main.py`). Sizing is now `device_profiles`; transport
> is the choice of **engine** (`sherpa` local vs. `livekit` remote).

## Device profiles (`device` + `device_profiles`)

`config.device` picks the active profile — the committed default is `"auto"`,
which probes the host (cores/RAM/GPU/mobile, via `tools.recommend_profile`) and
applies the matching profile at launch; `--device <name>` overrides it.
`device_profiles[<name>]` is recursively merged over the base config (with
backend-specific option bags replaced wholesale), so a profile only states what
differs. Profiles shipped in `config.json` (2026-07-10):

| Profile  | LLM backend | Models | Notes |
|----------|-------------|--------|-------|
| `desktop` | `ollama` (GPU) | `gemma3:12b` main + MiniCPM5-1B Q8 fast | MiniCPM is the text/answering tier; Gemma keeps vision/complex turns |
| `desktop_gpu_4090` | `ollama` (GPU) | `gemma3:12b` + MiniCPM5-1B Q8 | input gate + cleanup + capability router all on |
| `macbook_m_series` | `ollama` (Metal) | `gemma3:4b` + MiniCPM5-1B Q8 | cloud chains available, **default off** |
| `cpu_laptop` | `ollama` (CPU) | `gemma3:4b` + MiniCPM5-1B Q8 | gates off when CPU headroom is limited; cloud default off |
| `open_speaker` | `ollama` | `gemma3:4b` + MiniCPM5-1B Q8 | **explicit, never auto-picked**: WebRTC APM fallback for no-headphones barge-in (`docs/adr/0006`) |
| `phone`   | `llamacpp` (GGUF) | MiniCPM5-1B Q4 shared single tier | one context for main/fast; `n_gpu_layers: 0`, `n_ctx: 2048` |
| `phone_lite` | `llamacpp` (GGUF) | MiniCPM5-1B Q4 shared single tier | streaming-only ASR finals + greedy decode |

Newer model tiers (e.g. `gemma4:12b`) are pinned per machine in the gitignored
`config.local.json`, not in the committed profiles.

Provision the supported Ollama alias with `python -m tools.setup_minicpm`; the
helper installs the committed ChatML template rather than relying on Ollama's
auto-generated template for a direct Hugging Face import.

The `phone` profile runs the **Python core** under phone-like limits (for
simulation / low-power desktops). The **shipped Flutter app** (`mobile/`) is a
separate shell and uses `flutter_gemma` (MediaPipe/LiteRT), not these profiles —
see [`../mobile/README.md`](../mobile/README.md).

## The `llm` block

```jsonc
"llm": {
  "backend": "ollama",        // or "llamacpp" (on-device GGUF)
  "main_model": "gemma3:12b",      // complex / multimodal
  "fast_model": "minicpm5-1b:q8",  // local text / spoken replies
  "router": { "backend": "heuristic", "threshold": 0.3 }  // or "learned"
}
```

The router picks fast-vs-main per turn (`core/routing.py`): `heuristic` is
dependency-light (phone-safe); `learned` lazy-imports torch (desktop).

## Engines (the transport choice)

`--session` selects the user-facing session (`core/voice_session.py`):

- `console` — typed I/O, no audio/models (tests + demo).
- `local` — on-device mic/speaker via the selected local media adapter.
- `replay` — the real pipeline over recorded `.npy`/`.wav` fixtures, headless (no
  sound card); used for latency benchmarks and CI.
- `trusted-lan` — see ADR-0096 and ADR-0097.

## Deployment topologies

- **On-device (desktop):** `python -m core --session local --device desktop`.
- **On-device (Android):** the `mobile/` Flutter APK (built by
  `.github/workflows/android-apk.yml`).
- **Host + thin clients:** see ADR-0096 and ADR-0097.

## Example commands

```bash
python -m core --session console --llm echo                 # logic only, no deps
python -m core --session local --device desktop             # local desktop
python -m core --session local --device phone               # python core, phone limits
```

## Guardrails

Audio topology policy is recorded in ADR-0097; the text/file/screen boundary is
in `docs/target_architecture.md` §9.7. An invariant test keeps every committed
profile cloud-off (`tests/test_device_profile_invariants.py`). Lazy imports
mean a profile only pulls the heavy deps it
actually uses (Ollama, llama.cpp, torch, sherpa-onnx, livekit) — the base test
suite and the `console` session need none of them.
