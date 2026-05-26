"""Reuse the local STT / router / LLM / agent / TTS pipeline for a remote session.

The LiveKit worker (live voice) and the token server (/chat text) both drive a
RemoteSession instead of rewriting the pipeline. Heavy deps (utils.stt/llm/audio,
numpy, soundfile) are imported lazily so this module imports cleanly without the
full stack installed.

Safety note: in a remote session the spoken confirmation gate isn't wired (there
is no local mic loop), so the action brain runs with on_confirm=None. With
confirm_mode=auto_safe that means only allowlisted commands run; anything else is
refused. Do NOT expose a remote agent with auto_run/always-allow to untrusted
clients.
"""
from __future__ import annotations

import os
from typing import Iterator, Optional


class RemoteSession:
    """Stateless-ish wrapper around the pipeline components for one logical session."""

    def __init__(self, config: dict):
        self.config = config or {}
        self._ready = False
        self._transcribe = None
        self._stt_model = None
        self._stt_model_type = None
        self._stt_threads = None
        self._llm = None
        self._router = None
        self._caps = None
        self._agent_brain = None
        self._player = None

    # -- lazy construction ----------------------------------------------------
    def _ensure(self):
        if self._ready:
            return
        from utils.stt import resolve_stt_runtime, transcribe_audio
        from utils.llm import get_llm
        from utils.conversation_router import ConversationRouter

        cfg = self.config
        self._stt_model = cfg.get("stt_model", "base")
        runtime = resolve_stt_runtime(
            runtime_profile=cfg.get("runtime_profile", "balanced"),
            model_id=self._stt_model,
        )
        self._stt_model_type = runtime["model_type"]
        self._stt_threads = runtime["n_threads"]
        self._transcribe = transcribe_audio

        self._llm = get_llm(
            llm_type="local",
            model=cfg.get("llm_model", "llama2"),
            stream_mode=cfg.get("llm_stream_mode", "phrase"),
        )

        agent_triggers = tuple(cfg.get("agent_trigger_phrases", []) or ())
        self._setup_agent(cfg)
        self._router = ConversationRouter(
            stop_phrases=tuple(cfg.get("stop_phrases", ["stop", "quit", "exit"])),
            stop_mode=cfg.get("stop_mode", "exact"),
            agent_trigger_phrases=agent_triggers if self._agent_brain else (),
        )
        self._ready = True

    def _setup_agent(self, cfg: dict):
        if not cfg.get("agent_enabled"):
            return
        try:
            import dataclasses

            from utils.agent_brain import AgentBrain, AgentBrainConfig
            from utils.agent_capability import create_agent_provider
            from utils.capabilities import create_default_registry

            brain_cfg = dict(cfg.get("agent_brain") or {})
            brain_cfg.setdefault("local_only", bool(cfg.get("local_only", True)))
            brain_cfg.setdefault("local_fallback_model", f"ollama/{cfg.get('llm_model', 'llama2')}")
            brain_cfg["allowlist"] = tuple(brain_cfg.get("allowlist") or ())
            brain_cfg["denylist"] = tuple(brain_cfg.get("denylist") or ())
            valid = {f.name for f in dataclasses.fields(AgentBrainConfig)}
            brain_cfg = {k: v for k, v in brain_cfg.items() if k in valid}

            self._agent_brain = AgentBrain(AgentBrainConfig(**brain_cfg))
            self._caps = create_default_registry()
            # Remote sessions speak via the returned text stream, so the provider
            # collects rather than calls a speak callback.
            self._caps.register(
                "agent.execute",
                create_agent_provider(self._agent_brain, speak_cb=lambda _t: None),
                overwrite=True,
            )
        except Exception as exc:
            print(f"[remote] agent disabled: {exc}")
            self._agent_brain = None

    # -- STT ------------------------------------------------------------------
    def transcribe(self, audio_float32_16k) -> str:
        """Transcribe mono float32 PCM at 16 kHz to text."""
        self._ensure()
        try:
            return self._transcribe(
                audio_float32_16k,
                model_id=self._stt_model,
                model_type=self._stt_model_type,
                n_threads=self._stt_threads,
            ) or ""
        except Exception as exc:
            print(f"[remote] transcribe error: {exc}")
            return ""

    # -- respond (text in -> text chunks out) ---------------------------------
    def respond(self, text: str, should_cancel: Optional[callable] = None) -> Iterator[str]:
        """Route the text and yield response phrases (agent action or LLM chat)."""
        self._ensure()
        from utils.conversation_router import RouteAction, RouteContext

        text = (text or "").strip()
        if not text:
            return

        available = tuple(self._caps.list_capabilities()) if self._caps else ()
        decision = self._router.route(RouteContext(transcript=text, available_capabilities=available))

        if decision.action == RouteAction.SHUTDOWN:
            yield "Goodbye."
            return
        if decision.action in (RouteAction.IGNORE, RouteAction.STOP_OUTPUT):
            return
        if decision.action == RouteAction.CAPABILITY:
            if decision.capability == "agent.execute" and self._agent_brain:
                instruction = (decision.payload or {}).get("instruction", text)
                for event in self._agent_brain.stream_run(instruction, should_cancel=should_cancel):
                    if event.kind in ("speak", "result") and event.text:
                        yield event.text
                return
            # other deterministic capabilities (system.time, etc.)
            if self._caps:
                from utils.capabilities import CapabilityRequest

                resp = self._caps.invoke(
                    CapabilityRequest(name=decision.capability, payload=decision.payload)
                )
                if resp.ok and not (isinstance(resp.data, dict) and resp.data.get("streamed")):
                    import json

                    yield json.dumps(resp.data, sort_keys=True)
                return

        # default: local LLM chat
        for chunk in self._llm.get_streaming_response(text, "", []):
            if should_cancel and should_cancel():
                return
            if chunk:
                yield chunk

    # -- TTS (text -> PCM) ----------------------------------------------------
    def synthesize(self, text: str):
        """Synthesize text to (mono int16 PCM ndarray, sample_rate) by reusing
        the local TTS backends. Returns (None, 0) on failure."""
        self._ensure()
        text = (text or "").strip()
        if not text:
            return None, 0
        if self._player is None:
            from utils.audio import AudioPlayer

            self._player = AudioPlayer(
                voice=self.config.get("tts_voice", "en-US"),
                tts_backend=self.config.get("tts_backend"),
                tts_model=self.config.get("tts_model"),
                prefer_local=bool(self.config.get("local_only", True)),
                playback_backend="sounddevice",
            )
        path = None
        try:
            import numpy as np
            import soundfile as sf

            path = self._player.prepare_speech_file(text)
            audio, sr = sf.read(path, dtype="int16")
            if getattr(audio, "ndim", 1) > 1:
                audio = audio[:, 0]
            return np.asarray(audio, dtype=np.int16), int(sr)
        except Exception as exc:
            print(f"[remote] synthesize error: {exc}")
            return None, 0
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
