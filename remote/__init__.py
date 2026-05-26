"""Remote access layer: web app, phone-style live voice, and mobile.

Modules here let clients reach the assistant over LiveKit (WebRTC) and a small
FastAPI server, reusing the existing STT/LLM/TTS/agent pipeline rather than
rewriting it. All heavy/optional dependencies (livekit, fastapi) are imported
lazily inside functions so importing this package never breaks the base
install or the test suite.

Components:
  token_server.py   - FastAPI: mints LiveKit JWTs, serves the web client, /chat.
  livekit_agent.py  - LiveKit Agents worker: bridges room audio <-> pipeline.
  pipeline_bridge.py - reuses STT/router/LLM/agent/TTS for a remote session.

See each module's docstring for setup and how to run.
"""
