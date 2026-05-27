"""Remote access layer: web app, phone-style live voice, and mobile.

Modules here let clients reach the assistant over LiveKit (WebRTC) and a small
FastAPI server, reusing the new ``core`` runtime rather than rewriting it. All
heavy/optional dependencies (livekit, fastapi) are imported lazily inside
functions so importing this package never breaks the base install or the test
suite.

Components:
  token_server.py - FastAPI: mints LiveKit JWTs, serves the web client, /chat.
  worker.py       - runs a ``VoiceRuntime`` whose engine is the LiveKitEngine,
                    i.e. the same brain as the local app but with a room for I/O.

The room audio bridge itself is :class:`core.engines.livekit.LiveKitEngine`.
See each module's docstring for setup and how to run.
"""
