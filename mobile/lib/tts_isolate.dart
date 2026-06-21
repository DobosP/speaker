// Stage 1 of moving the heavy on-device work off the UI thread.
//
// sherpa's TTS synthesis is a *synchronous* native call — running it on the main
// isolate froze the event loop for the whole synth, which (with the always-on
// mic) starved ASR + UI and made the app feel deadlocked. Here synthesis runs on
// a long-lived worker isolate: the main isolate resolves the model paths (plugin
// + asset access only works there), hands them over once, and from then on sends
// text and gets back a finished .wav path to play.
import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './utils.dart';

class TtsService {
  TtsService._();
  static final TtsService instance = TtsService._();

  Isolate? _isolate;
  SendPort? _toWorker;
  Completer<void> _ready = Completer<void>();
  bool _starting = false;
  int _nextId = 0;
  final Map<int, Completer<String?>> _pending = {};
  _TtsInit? _initData;

  // Spawn + initialize the worker once. Safe to call repeatedly/concurrently.
  Future<void> ensureReady() {
    if (_toWorker != null && _ready.isCompleted) return _ready.future;
    if (_starting) return _ready.future;
    _starting = true;
    return _spawn();
  }

  Future<void> _spawn() async {
    // Resolve everything that needs plugins/assets HERE on the main isolate.
    await copyAllAssetFiles();
    final dir = (await getApplicationSupportDirectory()).path;
    const modelDir = 'vits-piper-en_US-amy-low';
    _initData = _TtsInit(
      model: p.join(dir, modelDir, 'en_US-amy-low.onnx'),
      tokens: p.join(dir, modelDir, 'tokens.txt'),
      dataDir: p.join(dir, modelDir, 'espeak-ng-data'),
    );

    final fromWorker = ReceivePort();
    fromWorker.listen(_onWorkerMessage);
    _isolate = await Isolate.spawn(_ttsWorkerMain, fromWorker.sendPort);
    return _ready.future;
  }

  void _onWorkerMessage(dynamic msg) {
    if (msg is SendPort) {
      _toWorker = msg;
      _toWorker!.send(_initData);
    } else if (msg == 'ready') {
      if (!_ready.isCompleted) _ready.complete();
    } else if (msg is _TtsResult) {
      _pending.remove(msg.id)?.complete(msg.outPath);
    }
  }

  // Synthesize [text] into [outPath] on the worker; resolves with the path once
  // the file is written, or null on failure/timeout (so callers never hang).
  Future<String?> synthesize(String text, String outPath) async {
    try {
      await ensureReady().timeout(const Duration(seconds: 30));
    } catch (_) {
      return null;
    }
    if (_toWorker == null) return null;
    final id = _nextId++;
    final completer = Completer<String?>();
    _pending[id] = completer;
    _toWorker!.send(_TtsRequest(id, text, outPath));
    return completer.future.timeout(
      const Duration(seconds: 20),
      onTimeout: () {
        _pending.remove(id);
        return null;
      },
    );
  }

  Future<void> dispose() async {
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _toWorker = null;
    _pending.clear();
    _ready = Completer<void>();
    _starting = false;
  }
}

// --- cross-isolate messages (primitive fields only, so they're sendable) ---

class _TtsInit {
  final String model;
  final String tokens;
  final String dataDir;
  _TtsInit({required this.model, required this.tokens, required this.dataDir});
}

class _TtsRequest {
  final int id;
  final String text;
  final String outPath;
  _TtsRequest(this.id, this.text, this.outPath);
}

class _TtsResult {
  final int id;
  final String? outPath; // null = synthesis failed
  _TtsResult(this.id, this.outPath);
}

// --- worker isolate ---

void _ttsWorkerMain(SendPort toMain) {
  final fromMain = ReceivePort();
  toMain.send(fromMain.sendPort);
  sherpa_onnx.OfflineTts? tts;

  fromMain.listen((msg) {
    if (msg is _TtsInit) {
      sherpa_onnx.initBindings();
      final vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model: msg.model,
        tokens: msg.tokens,
        dataDir: msg.dataDir,
      );
      final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
        vits: vits,
        numThreads: 2,
        provider: 'cpu',
      );
      tts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(model: modelConfig, maxNumSenetences: 1),
      );
      toMain.send('ready');
    } else if (msg is _TtsRequest) {
      try {
        final audio = tts!.generateWithConfig(
          text: msg.text,
          config: sherpa_onnx.OfflineTtsGenerationConfig(sid: 0, speed: 1.0),
        );
        // Parity with the desktop core's synth post-processing (core/audio_frontend):
        // repair the VITS impulse spikes that click on an open speaker, then even
        // out the per-sentence loudness so playback isn't choppy. Runs on the worker
        // isolate, off the UI/ASR thread, exactly like the Python producer thread.
        final samples = _postProcessTts(audio.samples);
        final ok = sherpa_onnx.writeWave(
          filename: msg.outPath,
          samples: samples,
          sampleRate: audio.sampleRate,
        );
        toMain.send(_TtsResult(msg.id, ok ? msg.outPath : null));
      } catch (_) {
        toMain.send(_TtsResult(msg.id, null));
      }
    }
  });
}

// --- TTS post-processing (Dart port of core/audio_frontend.py) ----------------
//
// The on-device VITS voice (same family as the desktop core) emits deterministic
// sample-level impulse SPIKES on some text -> audible clicks/crackle on an open
// speaker, and emits a DIFFERENT absolute amplitude per sentence -> uneven, choppy
// playback. ``declick`` repairs the isolated impulses (3-point median test +
// linear interpolation across short runs); ``normalize_rms`` scales each sentence
// to a steady RMS with a soft-knee limiter on the peaks. Both are no-ops on
// already-clean / already-leveled audio.
Float32List _postProcessTts(
  Float32List x, {
  double declickThreshold = 0.22,
  int maxRun = 8,
  double targetRms = 0.12,
  double maxGain = 20.0,
}) {
  final n = x.length;
  if (n < 3) return x;
  final y = Float32List.fromList(x);

  // De-click: flag samples whose deviation from the 3-point median exceeds the
  // threshold (a real spike), then interpolate across runs up to maxRun long.
  if (declickThreshold > 0) {
    final bad = List<bool>.filled(n, false);
    for (var i = 0; i < n; i++) {
      final a = x[i == 0 ? 0 : i - 1];
      final b = x[i];
      final c = x[i == n - 1 ? n - 1 : i + 1];
      // median of three = sum - max - min
      final med = a + b + c -
          math.max(a, math.max(b, c)) -
          math.min(a, math.min(b, c));
      if ((b - med).abs() > declickThreshold) bad[i] = true;
    }
    var i = 0;
    while (i < n) {
      if (bad[i]) {
        var j = i;
        while (j < n && bad[j] && (j - i) < maxRun) {
          j++;
        }
        final lo = i - 1;
        final hi = j;
        if (lo >= 0 && hi < n) {
          final denom = (j - i) + 1; // matches numpy linspace interior spacing
          for (var k = i; k < j; k++) {
            final t = (k - i + 1) / denom;
            y[k] = y[lo] + (y[hi] - y[lo]) * t;
          }
        }
        i = j;
      } else {
        i++;
      }
    }
  }

  // Per-sentence loudness normalization with a soft-knee limiter (boost capped).
  if (targetRms > 0) {
    var sum = 0.0;
    for (var i = 0; i < n; i++) {
      sum += y[i] * y[i];
    }
    final rms = math.sqrt(sum / n);
    if (rms > 1e-6) {
      final gain = math.min(targetRms / rms, maxGain);
      const knee = 0.8;
      for (var i = 0; i < n; i++) {
        var v = y[i] * gain;
        final mag = v.abs();
        if (mag > knee) {
          final sign = v < 0 ? -1.0 : 1.0;
          v = sign * (knee + (1.0 - knee) * _tanh((mag - knee) / (1.0 - knee)));
        }
        y[i] = v;
      }
    }
  }
  return y;
}

double _tanh(double x) {
  if (x > 20.0) return 1.0;
  if (x < -20.0) return -1.0;
  final e2 = math.exp(2.0 * x);
  return (e2 - 1.0) / (e2 + 1.0);
}
