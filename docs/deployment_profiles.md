# Deployment and Compute Profiles

## Deployment Profiles (`profile`)

- `low`
  - conservative model defaults for lower-resource machines
- `mid`
  - default laptop profile
- `high`
  - higher quality defaults

## Runtime Profiles (`runtime_profile`)

- `edge`
  - STT path prefers `whispercpp`
  - lower thread count and conservative generation profile
- `balanced`
  - default local quality/speed tradeoff
- `max_quality`
  - larger model and longer generation budgets

## Transport Modes (`transport_mode`)

- `local_lan`
  - local and LAN session handling only
- `webrtc`
  - WebRTC-oriented sessions
- `hybrid`
  - both transports active

## Recommended Combinations

- Laptop daily usage:
  - `profile=mid`, `runtime_profile=balanced`, `transport_mode=local_lan`
- Shared home devices:
  - `profile=low`, `runtime_profile=edge`, `transport_mode=hybrid`
- Quality-focused workstation:
  - `profile=high`, `runtime_profile=max_quality`, `transport_mode=webrtc`

## Latency tuning (defaults per `runtime_profile`)

When `config.json` does not set `silence_duration` or `llm_min_phrase_words`, the assistant uses profile-based defaults (see `RUNTIME_CONVERSATION_DEFAULTS` in `main.py`):

| `runtime_profile` | Default `silence_duration` (s) | Default `llm_min_phrase_words` |
|---------------------|-------------------------------|--------------------------------|
| `edge`              | 1.2                           | 3                              |
| `balanced`          | 1.5                           | 5                              |
| `max_quality`       | 1.5                           | 6                              |

Override at any time with `silence_duration` / `llm_min_phrase_words` in `config.json`, or `--silence-duration` / `--llm-min-phrase-words` on the CLI (CLI and explicit config beat profile defaults).

**Measuring**

- Synthetic batch timings: `python benchmarks/benchmark_realtime.py` (STT + LLM first chunk where available).
- Live backend call counts and ordering: run `python main.py --trace-backends` and optionally `--diagnostics-log-path logs/trace.jsonl` to append structured `backend_trace` JSON lines alongside other diagnostics.

Shorter silence endpointing yields snappier turns but more false cuts; lower `llm_min_phrase_words` sends the first TTS chunk sooner but may sound choppier.

## Guardrails

At startup, the runtime validates:

- known profile names
- known transport mode
- known runtime profile
- risky combinations (e.g. `edge` + `webrtc`) are flagged

## Example Commands

```bash
python main.py --profile mid --runtime-profile balanced --transport-mode local_lan
python main.py --profile low --runtime-profile edge --transport-mode hybrid
python main.py --profile high --runtime-profile max_quality --transport-mode webrtc
python main.py --wakeword-enabled --wakeword "hey_jarvis" --wakeword-service-mode process
python main.py --wakeword-enabled --wakeword-policy hybrid_recovery --wakeword-miss-limit 80 --wakeword-recovery-window-sec 3.0
```
