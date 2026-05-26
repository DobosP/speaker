"""LiveKit Agents worker bridging room audio <-> the local pipeline.

Run (after `pip install -r requirements-remote.txt`, LIVEKIT_* in the env, and a
running LiveKit server such as `livekit-server --dev`):

    python -m remote.livekit_agent dev      # connect to a dev server
    python -m remote.livekit_agent start    # production worker

It subscribes to a caller's mic track, segments speech with a simple
energy+silence VAD, transcribes + responds via RemoteSession (chat or action),
synthesizes TTS with the local backends, and publishes the audio back. User and
assistant text go over the room data channel for the web/mobile UI.

This path needs a real LiveKit server + audio and cannot be verified headless --
treat it as a working starting point to tune with your setup (VAD threshold,
sample rates, turn-taking). The pure audio helpers below are unit-tested.
"""
from __future__ import annotations

import asyncio
import json

OUT_SR = 48000   # LiveKit-friendly output sample rate
STT_SR = 16000   # pipeline STT sample rate
FRAME_MS = 20


# -- pure audio helpers (unit-tested) ---------------------------------------
def pcm_int16_to_float32(pcm):
    import numpy as np

    return np.asarray(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm_int16(samples):
    import numpy as np

    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def resample_linear(samples, src_sr: int, dst_sr: int):
    """Linear-interpolate float32 mono audio from src_sr to dst_sr."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32)
    if src_sr == dst_sr or x.size == 0:
        return x
    n_out = int(round(x.shape[0] * float(dst_sr) / float(src_sr)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0.0, x.shape[0] - 1, num=n_out)
    return np.interp(src_idx, np.arange(x.shape[0]), x).astype(np.float32)


def _load_config() -> dict:
    try:
        with open("config.json") as f:
            return json.load(f)
    except Exception:
        return {}


# -- worker -----------------------------------------------------------------
async def entrypoint(ctx):  # pragma: no cover - needs a live LiveKit server
    import numpy as np
    from livekit import rtc

    from remote.pipeline_bridge import RemoteSession

    config = _load_config()
    session = RemoteSession(config)

    await ctx.connect()
    room = ctx.room

    source = rtc.AudioSource(OUT_SR, 1)
    track = rtc.LocalAudioTrack.create_audio_track("assistant-voice", source)
    await room.local_participant.publish_track(track)

    async def publish_data(event_type: str, payload: dict):
        try:
            await room.local_participant.publish_data(
                json.dumps({"event_type": event_type, "payload": payload}).encode("utf-8"),
                reliable=True,
            )
        except Exception:
            pass

    async def speak(text: str):
        await publish_data("assistant_sentence", {"text": text})
        pcm, sr = await asyncio.to_thread(session.synthesize, text)
        if pcm is None:
            return
        out = float32_to_pcm_int16(resample_linear(pcm_int16_to_float32(pcm), sr, OUT_SR))
        chunk = int(OUT_SR * FRAME_MS / 1000)
        for i in range(0, len(out), chunk):
            seg = out[i : i + chunk]
            await source.capture_frame(
                rtc.AudioFrame(
                    data=seg.tobytes(),
                    sample_rate=OUT_SR,
                    num_channels=1,
                    samples_per_channel=len(seg),
                )
            )

    async def handle_utterance(audio_16k):
        text = await asyncio.to_thread(session.transcribe, audio_16k)
        if not text.strip():
            return
        await publish_data("user_transcript", {"text": text})
        phrases = await asyncio.to_thread(lambda: list(session.respond(text)))
        for phrase in phrases:
            await speak(phrase)

    async def process_track(audio_track):
        stream = rtc.AudioStream(audio_track)
        seg, silence, speaking = [], 0.0, False
        sil_limit = float(config.get("silence_duration", 1.5))
        threshold = float(config.get("vad_threshold", 0.01))
        async for event in stream:
            frame = event.frame
            data = np.frombuffer(frame.data, dtype=np.int16)
            if frame.num_channels > 1:
                data = data.reshape(-1, frame.num_channels)[:, 0]
            mono = resample_linear(pcm_int16_to_float32(data), frame.sample_rate, STT_SR)
            energy = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
            if energy >= threshold:
                speaking, silence = True, 0.0
                seg.append(mono)
            elif speaking:
                silence += mono.size / STT_SR
                seg.append(mono)
                if silence >= sil_limit:
                    utt = np.concatenate(seg) if seg else np.zeros(0, dtype=np.float32)
                    seg, speaking, silence = [], False, 0.0
                    asyncio.create_task(handle_utterance(utt))

    @room.on("track_subscribed")
    def _on_track(t, publication, participant):
        if t.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(process_track(t))


def main():  # pragma: no cover - needs livekit-agents installed
    from livekit.agents import WorkerOptions, cli

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":  # pragma: no cover
    main()
