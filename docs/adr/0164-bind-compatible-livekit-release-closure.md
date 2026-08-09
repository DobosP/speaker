# ADR-0164: Bind the compatible LiveKit 1.6.8 release closure

Date: 2026-08-09
Status: accepted

## Decision

Supersede ADR-0104's invalid dependency selection while retaining its trusted
publisher-session architecture and every admission, route, disposal, authority,
and promotion fence. Pin the optional host requirements to this exact
LiveKit-family release closure:

- `livekit==1.1.14`;
- `livekit-agents==1.6.8`;
- `livekit-api==1.2.0`;
- `livekit-protocol==1.1.21`.

Load the trusted publisher SDK lazily and fail before construction unless all
four installed distribution versions match. Keep the raw SDK objects behind
`TrustedLiveKitPublisherSession`; preserve manual publisher-scoped `RoomIO`,
explicit local STT/VAD/TTS objects, `llm=None`, empty tools and MCP, disabled
recording/preemptive generation/transcript transforms, roomless session start,
exact output leases, completed-turn ownership, and retryable disposal. Keep
trusted-LAN unselectable until a later installed-package and self-hosted live
A/B decision.

Model the additive 1.6.8 RoomIO participant-link callback in the fake SDK and
require the wrapper's explicit `aec_warmup_duration=None` to remain inert. Keep
the hard rejection of `replace_audio_tail`: 1.6.8 still falls back to replacing
the whole audio head when direct manual RoomIO provides no sink proxy. Do not
weaken the wrapper because 1.6.8 improves pause/resume, false-interruption, VAD,
or disconnect timing.

Make remote access-token TTL assignment fail closed. The reviewed API supports
the exact `AccessToken` → identity → name → `VideoGrants` → `with_ttl` → JWT
chain. If the requested one-hour TTL cannot be applied, publish no token instead
of silently falling back to the SDK's six-hour default.

The four pins are a deterministic LiveKit-family closure, not a complete Python
lock: unrelated transitive packages still use ranges. The legacy Docker remote
worker does not construct this Agents wrapper and its separate floating
requirements are outside this decision. Pinning or rebuilding that rollback
image requires its own compatibility gate.

## Context / why

ADR-0104 selected `livekit==1.1.14` with `livekit-agents==1.6.7`. Official
[`livekit-agents==1.6.7` metadata](https://pypi.org/pypi/livekit-agents/1.6.7/json)
requires `livekit==1.1.13`, so the committed pair cannot resolve. Official
[`livekit-agents==1.6.8` metadata](https://pypi.org/pypi/livekit-agents/1.6.8/json)
requires RTC 1.1.14, API at least 1.2.0 below 2, and protocol at least 1.1.21
below 2. API 1.2.0 accepts protocol 1.1.19 through 1.x. The
[upstream 1.6.8 lock](https://github.com/livekit/agents/blob/livekit-agents%401.6.8/uv.lock)
resolves exactly the four versions selected here; leaving API/protocol floating
would permit future 1.x execution drift without review.

The official
[Agents 1.6.7→1.6.8 comparison](https://github.com/livekit/agents/compare/livekit-agents%401.6.7...livekit-agents%401.6.8)
retains Speaker's Agent, session start/say/shutdown, SpeechHandle, output, console,
and manual RoomIO surfaces. Relevant changes add an optional transcription
timeout and participant-link hook, make playback-start one-shot across resume,
defer interim-STT interruption to supplied VAD, and tighten paused/false-resume
settlement. These are compatible with the stricter local wrapper but alter
timing that still needs real installed-package and live validation. The
[1.6.8 output source](https://github.com/livekit/agents/blob/livekit-agents%401.6.8/livekit-agents/livekit/agents/voice/io.py#L677)
retains the unsafe direct-RoomIO tail-replacement fallback.

Official [API 1.2.0 source](https://github.com/livekit/python-sdks/blob/api-v1.2.0/livekit-api/livekit/api/access_token.py)
preserves Speaker's token-mint chain. The old broad TTL exception handler was a
local fail-open bug, not an upstream regression.

## Consequences

- A fresh optional-host installation can now resolve one release-matched
  LiveKit-family set, and wrapper construction detects drift in any of its four
  distribution versions.
- RTC remains exactly 1.1.14, so the bounded Microsoft AEC evaluator's runtime
  requirement is unchanged. Because it binds `requirements-remote.txt`, its
  execution-closure digest changes even though no production AEC report exists.
- Default device-only topology, local/cloud boundaries, raw-audio grants,
  publisher identity semantics, action authority, tools, models, the legacy
  rollback path, and live selectability do not change.
- Fake/source-audit evidence cannot prove wheel installation, resolver health,
  network behavior, capture, playback, interruption timing, AEC, or audibility.
  Before promotion, provision the exact closure in isolation, run an installed
  headless surface/lifecycle contract and `pip check`, then pass the encrypted
  self-hosted publisher/device and open-speaker live A/B.

This ADR supersedes ADR-0104's package decision and refines ADR-0096 and
ADR-0163. Their historical evidence and the AEC component design remain valid.

## Verification

- Release commits: Agents 1.6.8
  `c4c303a34d06c64d06851cc2762265f98eb29ced`, RTC 1.1.14
  `23cbb0f33f91be9360f970980b445bb692b19b52`, API 1.2.0
  `8d979d6d35e591244f0158b267c497103b7c5094`, and protocol 1.1.21
  `0ef63ce92c510f3494d07aaae047284a2caebae7`.
- PyPI wheel SHA-256 values: Agents
  `fefe45142f398895f3fcb37f381569c2f9ce98b7eb0d9dfe6698aa9774d0f355`,
  Linux x86-64 RTC
  `80962c4a22ddbf0e0ebd3563fc090fce42df66b39b90de68b161b7db01970f68`,
  API `307f8e5cfb0358c3ca091814ab768af55896022151bcd7f951954ccefa036a24`,
  and protocol
  `ce0bb763327c91349ee8831843c4f8bd72132c4d06ac572171f2ae5c0217211d`.
- Headless gates pass 78 focused publisher-session tests; 6 token-server tests
  with 2 optional endpoint skips; 138 closure-sensitive Microsoft AEC/APM
  tests; 143 requirements/legacy-LiveKit tests with 2 optional skips; 149
  authority/default tests; 123 playback/runtime tests; and 332 device-profile
  and import-safety tests.
- The refreshed AEC execution closure binds 53 files, 1,784,000 bytes, and
  SHA-256
  `894fe05a7008755e260ecf68f4f8d8022b8ec02cb3547de67cc8aa3ccfd0dc54`.
- Scoped Ruff lint/format and whitespace checks are green; independent
  settled-byte landing review is GO. No package artifact was downloaded or
  installed; no network service, model, GPU, microphone, speaker, device, AEC
  replay, or live run occurred. The current environment's RTC 1.1.10 and absent
  Agents/API/protocol distributions correctly remain ineligible.
