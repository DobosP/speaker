"""Combine two (or more) audio files into ONE buffer and play it as a single
output stream — or save the mix to a WAV.

**Why this exists.** This machine's PortAudio is exclusive ALSA with no software
mixing layer, so opening two *concurrent* ``sd.OutputStream``s fails with
"Device unavailable" (this is what blocked the two-speaker acoustic barge-in /
competing-voice tests). The fix is to *mix the clips into one buffer* and play a
single stream — which works everywhere and is the correct way to drive a
competing-voice / babble acoustic test on this box: target voice + intruder voice
(+ noise) summed, played out one speaker, heard back by the one mic.

It reuses the primitives already in the tree rather than reinventing them:
  * ``core.engines.file_replay.load_waveform`` — WAV/npy -> (float32 mono, sr)
  * ``tools.live_session.synthetic_user._resample`` — linear resample
  * ``tools.live_session.synthetic_user.save_wav`` — float32 -> 16-bit PCM WAV

It touches **no** mixer/mute/device state — pure audio I/O. (Opening an output
stream never changes a USB mic's hardware mute button.)

CLI::

    # combine two clips and play them as one, out the laptop speaker (device 4)
    python -m tools.audio_mix a.wav b.wav --play --output-device 4

    # start the second clip 1.5s in, 80% level, and also save the mix
    python -m tools.audio_mix assistant.wav stop.wav \\
        --offset 0 1.5 --gain 1.0 0.8 --out logs/mix.wav --play

    # mix a target voice + an intruder at a target SNR (intruder = clip #2)
    python -m tools.audio_mix target.wav intruder.wav --snr 5 --play
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from core.engines.file_replay import load_waveform
from tools.live_session.synthetic_user import _resample, save_wav


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype("float64") ** 2)))


def mix(
    clips: Sequence[tuple[np.ndarray, int]],
    *,
    target_sr: int | None = None,
    gains: Sequence[float] | None = None,
    offsets_sec: Sequence[float] | None = None,
    snr_db: float | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, int]:
    """Overlay ``clips`` (each ``(float32 mono, sample_rate)``) into one buffer.

    * ``target_sr`` — rate to mix at (default: the highest input rate). Every clip
      is resampled to it so the sum is sample-aligned.
    * ``gains`` — per-clip linear multiplier (default 1.0 each).
    * ``offsets_sec`` — per-clip start time; the clip is zero-padded on the left so
      it begins that many seconds into the timeline (default 0 — all start together,
      i.e. they play *at the same time*).
    * ``snr_db`` — convenience for the 2-clip case: scale clip #2 so it sits
      ``snr_db`` dB below clip #1's RMS (the competing-voice knob). Overrides
      ``gains`` for clip #2. Higher dB => the intruder is quieter.
    * ``normalize`` — if the summed peak exceeds 1.0, scale the whole mix down so
      the peak is 0.99 (preserves the relative levels; no distortion). Set False to
      leave it raw.

    Returns ``(mixed float32 mono, target_sr)``.
    """
    clips = list(clips)
    if not clips:
        return np.zeros(0, dtype="float32"), target_sr or 16000
    sr = target_sr or max(int(c[1]) for c in clips)

    n = len(clips)
    gains = list(gains) if gains is not None else [1.0] * n
    offsets_sec = list(offsets_sec) if offsets_sec is not None else [0.0] * n
    if len(gains) != n or len(offsets_sec) != n:
        raise ValueError("gains/offsets_sec length must match the number of clips")

    # Resample every clip to the common rate up front.
    resampled = [_resample(np.asarray(s, dtype="float32"), int(csr), sr) for s, csr in clips]

    # SNR convenience: override clip #2's gain so it sits snr_db below clip #1.
    if snr_db is not None and n >= 2:
        s_rms = _rms(resampled[0] * gains[0])
        i_rms = _rms(resampled[1])
        if s_rms > 0.0 and i_rms > 0.0:
            target_i_rms = s_rms / (10.0 ** (snr_db / 20.0))
            gains[1] = target_i_rms / i_rms

    # Total length = furthest (offset + clip) end.
    placed = []
    total = 0
    for clip, off in zip(resampled, offsets_sec):
        start = max(0, int(round(off * sr)))
        placed.append((start, clip))
        total = max(total, start + clip.shape[0])

    out = np.zeros(total, dtype="float64")
    for (start, clip), g in zip(placed, gains):
        out[start : start + clip.shape[0]] += clip.astype("float64") * float(g)

    if normalize:
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.0:
            out *= 0.99 / peak

    return out.astype("float32"), sr


def mix_files(
    paths: Sequence[str],
    *,
    target_sr: int | None = None,
    gains: Sequence[float] | None = None,
    offsets_sec: Sequence[float] | None = None,
    snr_db: float | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, int]:
    """Load each path via :func:`load_waveform` then :func:`mix` them."""
    clips = [load_waveform(p) for p in paths]
    clips = [(np.asarray(s, dtype="float32"), int(csr)) for s, csr in clips]
    return mix(
        clips,
        target_sr=target_sr,
        gains=gains,
        offsets_sec=offsets_sec,
        snr_db=snr_db,
        normalize=normalize,
    )


def play(samples: np.ndarray, sr: int, device=None) -> None:
    """Play ``samples`` (float32 mono) through a SINGLE output stream.

    Mirrors the robust rate-fallback used by the synthetic user: if the device
    rejects ``sr``, resample to a rate it accepts and retry. One stream only — no
    concurrent streams, so it works on exclusive-ALSA hardware.
    """
    import sounddevice as sd

    x = np.asarray(samples, dtype="float32").reshape(-1)
    try:
        dev_info = sd.query_devices(device, "output") if device is not None else sd.query_devices(kind="output")
        dev_sr = int(dev_info.get("default_samplerate") or sr)
    except Exception:
        dev_sr = sr

    last_err = None
    for rate in dict.fromkeys([sr, dev_sr, 48000, 44100]):  # de-duped, in order
        try:
            play_buf = x if rate == sr else _resample(x, sr, rate)
            sd.play(play_buf, rate, device=device)
            sd.wait()
            return
        except Exception as e:  # PortAudioError etc. — try the next rate
            last_err = e
    raise RuntimeError(f"could not play mixed audio on device {device!r}: {last_err}")


def _norm_device(dev: str | None):
    if dev is None or dev == "":
        return None
    return int(dev) if dev.isdigit() else dev


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Combine two+ audio files into one buffer and play/save it as a single stream."
    )
    ap.add_argument("files", nargs="+", help="audio files (.wav 16-bit PCM, or .npy) to overlay")
    ap.add_argument("--out", help="write the mix to this WAV path")
    ap.add_argument("--play", action="store_true", help="play the mix out a single output stream")
    ap.add_argument("--output-device", default=None, help="output device index or name (e.g. 4)")
    ap.add_argument("--gain", type=float, nargs="+", help="per-file linear gain (default 1.0 each)")
    ap.add_argument(
        "--offset",
        type=float,
        nargs="+",
        help="per-file start time in seconds (default 0 — all play at the same time)",
    )
    ap.add_argument(
        "--snr",
        type=float,
        default=None,
        help="2-file convenience: put file #2 this many dB below file #1 (competing-voice knob)",
    )
    ap.add_argument("--rate", type=int, default=None, help="mix sample rate (default: highest input rate)")
    ap.add_argument("--no-normalize", action="store_true", help="don't peak-limit if the sum clips")
    args = ap.parse_args(argv)

    mixed, sr = mix_files(
        args.files,
        target_sr=args.rate,
        gains=args.gain,
        offsets_sec=args.offset,
        snr_db=args.snr,
        normalize=not args.no_normalize,
    )
    dur = mixed.shape[0] / sr if sr else 0.0
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    print(
        f"mixed {len(args.files)} file(s) -> {dur:.2f}s @ {sr} Hz  peak={peak:.3f}  rms={_rms(mixed):.4f}"
    )

    if args.out:
        out_path = save_wav(mixed, sr, Path(args.out))
        print(f"saved: {out_path}")
    if args.play:
        print(f"playing on device {args.output_device!r} ...")
        play(mixed, sr, device=_norm_device(args.output_device))
        print("done.")
    if not args.out and not args.play:
        print("(nothing to do — pass --play and/or --out)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
