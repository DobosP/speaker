"""Run the assistant against a LiveKit room: a thin client (browser/phone) talks
to one running brain.

This is the remote sibling of ``python -m core --engine sherpa`` -- same runtime
and brain, but the :class:`core.engines.livekit.LiveKitEngine` uses a LiveKit
(WebRTC) room for audio instead of the local mic/speaker. This entrypoint just
mints the agent's room token (LiveKit minting belongs to the remote layer) and
forwards to ``core.app`` with ``--engine livekit``; all runtime wiring is reused.

Setup:
    pip install -r requirements-remote.txt
    # start a server (dev): livekit-server --dev
    # set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET in the env / .env
    python -m remote.worker            # join room "assistant" with the config LLM
    python -m remote.worker --llm echo # offline smoke (no Ollama)

Then serve the web client + token endpoint for callers:
    uvicorn remote.token_server:app --host 0.0.0.0 --port 8080

This live path needs a running LiveKit server and a connected client, so it
cannot be verified headless; treat it as a working starting point to tune with
your setup. The pure helpers in core.engines.livekit / remote.token_server are
unit-tested.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from always_on_agent.events import Mode


def _parse(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LiveKit remote-voice worker")
    p.add_argument("--room", default=None, help="room to join (default: config remote.room)")
    p.add_argument("--identity", default=None, help="agent identity in the room")
    # Forwarded to core.app:
    p.add_argument("--llm", choices=["echo", "ollama"], default="ollama")
    p.add_argument("--model", default=None, help="main Ollama model")
    p.add_argument("--fast-model", dest="fast_model", default=None, help="small Ollama model")
    p.add_argument("--device", default=None, help="device profile from config 'device_profiles'")
    p.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.ASSISTANT.value)
    p.add_argument(
        "--agent",
        action="store_true",
        help="enable the Open Interpreter action brain for command-mode",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse(argv)

    from core.app import main as app_main
    from core.config import apply_device_profile, load_config, resolve_device

    from .token_server import create_access_token

    config = load_config()
    # device-adapt-1 / cross-platform-8: 'auto' (default) auto-detects the profile;
    # an unknown explicit --device fails fast (strict) instead of silently running
    # the heavy base config.
    requested_device = args.device or config.get("device", "auto")
    device, rationale = resolve_device(config, requested_device)
    if rationale:
        print(f"[device] auto-selected profile: {device} ({rationale})", file=sys.stderr)
    try:
        config = apply_device_profile(config, device, strict=True)
    except ValueError as exc:
        raise SystemExit(f"[device] {exc}")
    remote_cfg = config.get("remote", {}) or {}
    room = args.room or remote_cfg.get("room", "assistant")
    identity = args.identity or remote_cfg.get("agent_identity", "assistant-agent")

    if not os.environ.get("LIVEKIT_URL"):
        raise SystemExit(
            "LIVEKIT_URL is not set. Install requirements-remote.txt, start a "
            "LiveKit server, and set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET."
        )
    try:
        os.environ["LIVEKIT_TOKEN"] = create_access_token(identity, room)
    except Exception as exc:
        raise SystemExit(f"Could not mint LiveKit token: {exc}")

    forwarded = ["--engine", "livekit", "--llm", args.llm, "--mode", args.mode]
    if args.model:
        forwarded += ["--model", args.model]
    if args.fast_model:
        forwarded += ["--fast-model", args.fast_model]
    # Forward the RESOLVED concrete profile (not args.device) so the child runs
    # the exact tier the worker picked instead of re-probing / defaulting.
    forwarded += ["--device", device]
    if args.agent:
        forwarded += ["--agent"]

    print(f"[remote] joining room '{room}' as '{identity}'. Ctrl-C to quit.")
    return app_main(forwarded)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
