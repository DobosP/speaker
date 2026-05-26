#!/usr/bin/env python3
"""
Synthesize one phrase and play it (no microphone). Use to verify TTS + speakers.

Examples:
  python scripts/test_tts_playback.py
  python scripts/test_tts_playback.py --tts-backend piper --output-device 3
  python scripts/test_tts_playback.py --write-wav /tmp/tts_test.wav --no-play
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio import AudioPlayer, list_audio_devices, resolve_output_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="TTS synthesis + playback smoke test")
    parser.add_argument("--text", default="Hello. This is a speaker audio test.")
    parser.add_argument("--tts-backend", default=None, choices=["kokoro", "piper", "melotts", "supertonic"])
    parser.add_argument("--tts-voice", default="en-US")
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument(
        "--playback-backend",
        default="auto",
        choices=["auto", "sounddevice", "pygame"],
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--write-wav", default=None, help="Save WAV and skip playback")
    parser.add_argument("--no-play", action="store_true", help="Synthesize only (implies --write-wav if set)")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return 0

    output_device = resolve_output_device(None, args.output_device)
    if output_device is not None and args.output_device is None:
        print(f"Using output device {output_device} (matched to default microphone)")

    kwargs = {
        "voice": args.tts_voice,
        "output_device": output_device,
        "playback_backend": args.playback_backend,
    }
    if args.tts_backend:
        kwargs["tts_backend"] = args.tts_backend
    player = AudioPlayer(**kwargs)
    if player.tts_backend is None:
        print("No TTS backend available. Install kokoro-onnx or piper-tts.")
        return 1

    path = player.prepare_speech_file(args.text)
    out = args.write_wav
    if out:
        import shutil
        shutil.copy(path, out)
        print(f"Wrote {out}")
    if args.no_play or out:
        try:
            os.remove(path)
        except OSError:
            pass
        return 0

    ok = player.play_prepared_file(path)
    player.cleanup()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
