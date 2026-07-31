# ADR-0091: Add and reject exact Nemotron streaming candidate

Date: 2026-07-31
Status: accepted

## Decision

Land NVIDIA Nemotron 3.5 Streaming 0.6B as an isolated, benchmark-only
English/CUDA adapter for ADR-0089, and reject every measured lookahead mode as
a production recognizer or normal-runtime choice. Keep LA6 only as the best
tradeoff within the rejected modes; LA13 has the best measured accuracy but its
1.12-second native latency is unsuitable for the target conversation loop.
Require Python 3.12, Torch 2.12.1+cu126, Transformers 5.13.1, Librosa 0.11.0,
model revision `f3d333391852ba876df169dcc9ba902d25b6ab0b`,
the exact 74-wheel lock at SHA-256
`df268a2e268221428256b3ec525a3ad49da65b526b2e09b88df3802533b5af01`,
and runtime content SHA-256
`ebcf9cab82c93ab5503bb438fbc9585475fa138c4e18aaea73d9ab34fc48bb8f`
(24,273 files; 6,752,927,292 bytes). Run it only through no-network Bubblewrap
with the model and runtime read-only and all mutable caches in private scratch.
Leave production `.venv`, recognition defaults, and `./live.sh` unchanged.

## Context / why

Nemotron offers native streaming at four lookahead settings, but the exact
public 14-case burst evidence does not beat the rejected controls. Results were:
LA0 WER 0.7065, CER 0.6606, RTF 0.4331, 80 ms native latency, and three exact
finals; LA3 0.7065/0.6592/0.1395, 320 ms, and four exact finals; LA6
0.6957/0.6551/0.0859, 560 ms, and four exact finals; and LA13
0.6902/0.6467/0.0556, 1,120 ms, and five exact finals. An LA6 real-time replay
produced WER 0.6957, RTF 0.0992, first-nonempty-partial p50 1.666 seconds, one
deadline miss, and 10.439 ms maximum backlog. LA6 is therefore the least-bad
latency/accuracy compromise, not an adoption result.

The exact one-case CUDA smoke passed through Bubblewrap, including close-time
rehashing of the bound source, manifest, model, runtime, wheel lock, and corpus.
An earlier direct host import caused Numba to add empty Librosa `__pycache__`
directories to an otherwise byte-exact runtime. That runtime remains evidence;
the repaired worker redirects `NUMBA_CACHE_DIR` to scratch and uses a fresh,
untouched read-only rebuild.

The measurements begin after PCM is available and do not cover capture, device
I/O, controller IPC, VAD, endpointing, AEC, room echo, or physical barge-in.
They are development-corpus results, not held-out, owner-voice, multilingual,
multi-device, conversational, or live-hardware validity. A fast RTF therefore
does not establish low perceived latency or acceptable STT quality.

## Consequences

Nemotron remains available as a reproducible comparator behind explicit
prepare, provision, smoke, and aggregate-evaluation commands. It cannot be
selected by setup or the application entry point, and its packages cannot enter
the production environment. Future reconsideration requires disjoint recorded
speech, endpoint-to-final timing, error analysis on target commands, and fresh
`./live.sh` bare-speaker A/B evidence; multi-device resource and thermal
evidence is also required. Moonshine and Faster-Whisper controls remain
unchanged.
