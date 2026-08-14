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

import './tts_isolate_lifecycle.dart';
import './tts_process_owner.dart';
import './utils.dart';

class TtsService {
  TtsService._()
      : _lifecycle = TtsIsolateLifecycle(driver: _DartTtsWorkerDriver());
  static final TtsService instance = TtsService._();

  final TtsIsolateLifecycle _lifecycle;
  TtsProcessLease? _lease;
  Future<_TtsInit>? _initFuture;
  Future<bool>? _readyFuture;

  /// Spawn and initialize the exact worker owned by [lease].
  ///
  /// Concurrent calls for that lease share one asset-resolution and spawn
  /// future. A foreign, revoked, or replacement lease cannot reuse the worker.
  Future<bool> ensureReady(TtsProcessLease lease) {
    if (!ttsProcessOwnerRegistry.ownsExact(lease)) {
      return Future<bool>.value(false);
    }
    final current = _lease;
    if (current != null && !identical(current, lease)) {
      return Future<bool>.value(false);
    }
    final existing = _readyFuture;
    if (existing != null) return existing;

    _lease = lease;
    final ready = _prepareAndStart(lease);
    _readyFuture = ready;
    return ready;
  }

  Future<bool> _prepareAndStart(TtsProcessLease lease) async {
    final init = await (_initFuture ??= _resolveInit());
    if (!lease.admitsWork || !identical(_lease, lease)) return false;
    return _lifecycle.ensureReady(lease, init);
  }

  Future<_TtsInit> _resolveInit() async {
    await copyAllAssetFiles();
    final dir = (await getApplicationSupportDirectory()).path;
    const modelDir = 'vits-piper-en_US-amy-low';
    return _TtsInit(
      model: p.join(dir, modelDir, 'en_US-amy-low.onnx'),
      tokens: p.join(dir, modelDir, 'tokens.txt'),
      dataDir: p.join(dir, modelDir, 'espeak-ng-data'),
    );
  }

  // Synthesize [text] into [outPath] on the worker; resolves with the path once
  // the file is written, or null for modeled readiness/request failure and
  // request timeout. Asset/plugin preparation, spawn, or entered native work
  // may remain pending, so owners retain authority until the exact Future returns.
  Future<String?> synthesize(
    TtsProcessLease lease,
    String text,
    String outPath,
  ) async {
    try {
      if (!await ensureReady(lease)) return null;
    } catch (_) {
      return null;
    }
    return _lifecycle.request(lease, text, outPath);
  }

  /// Return true only after the exact worker reports that its `OfflineTts.free`
  /// wrapper returned and the main-side shutdown/receive-subscription cleanup
  /// calls complete. This is not proof of native destruction or isolate exit.
  Future<bool> dispose(TtsProcessLease lease) async {
    if (!ttsProcessOwnerRegistry.holdsExact(lease)) return false;
    final current = _lease;
    if (current != null && !identical(current, lease)) return false;
    // Mark an already-started lifecycle closing before waiting for asset/plugin
    // preparation. A late spawn can then only enter the lifecycle's stale
    // cleanup path, never initialize or publish readiness after revocation.
    final lifecycleClose = _lifecycle.dispose();
    final preparing = _readyFuture;
    if (preparing != null) {
      try {
        await preparing;
      } catch (_) {
        // The lifecycle records spawn/send uncertainty. Continue into its exact
        // cleanup receipt; never clear or release merely because prep failed.
      }
    }
    final clean = await lifecycleClose;
    if (clean) {
      _lease = null;
      _initFuture = null;
      _readyFuture = null;
    }
    return clean;
  }
}

// --- cross-isolate messages (primitive fields only, so they're sendable) ---

class _TtsInit {
  final String model;
  final String tokens;
  final String dataDir;
  _TtsInit({required this.model, required this.tokens, required this.dataDir});
}

final class _TtsWorkerBootstrap {
  const _TtsWorkerBootstrap(this.epoch, this.toMain);

  final int epoch;
  final SendPort toMain;
}

final class _TtsWorkerPort {
  const _TtsWorkerPort(this.epoch, this.port);

  final int epoch;
  final SendPort port;
}

final class _DartTtsWorkerDriver implements TtsWorkerDriver {
  @override
  Future<TtsWorkerHandle> spawn(
    int epoch,
    void Function(TtsWorkerEvent event) emit,
  ) async {
    final events = ReceivePort();
    final handle = _DartTtsWorkerHandle(epoch, events, emit);
    try {
      final isolate = await Isolate.spawn(
        _ttsWorkerMain,
        _TtsWorkerBootstrap(epoch, events.sendPort),
      );
      handle.attach(isolate);
      return handle;
    } catch (_) {
      await handle.closeEvents();
      rethrow;
    }
  }
}

final class _DartTtsWorkerHandle implements TtsWorkerHandle {
  _DartTtsWorkerHandle(this.epoch, this._events, this._emit) {
    _subscription = _events.listen(_onMessage);
  }

  final int epoch;
  final ReceivePort _events;
  final void Function(TtsWorkerEvent event) _emit;
  late final StreamSubscription<dynamic> _subscription;
  Isolate? _isolate;
  SendPort? _worker;
  bool _killed = false;
  bool _eventsClosed = false;

  void attach(Isolate isolate) {
    if (_killed) {
      isolate.kill(priority: Isolate.immediate);
      return;
    }
    _isolate = isolate;
  }

  void _onMessage(dynamic message) {
    if (_eventsClosed) return;
    if (message is _TtsWorkerPort && message.epoch == epoch) {
      _worker = message.port;
      _emit(TtsWorkerSendPort(epoch));
    } else if (message is TtsWorkerEvent && message.epoch == epoch) {
      _emit(message);
    }
  }

  @override
  void send(TtsWorkerCommand command) {
    if (_killed || command.epoch != epoch) {
      throw StateError('stale mobile TTS worker command');
    }
    final worker = _worker;
    if (worker == null) throw StateError('mobile TTS worker port unavailable');
    worker.send(command);
  }

  @override
  void kill() {
    if (_killed) return;
    _killed = true;
    _worker = null;
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
  }

  @override
  Future<void> closeEvents() async {
    if (_eventsClosed) return;
    _eventsClosed = true;
    try {
      await _subscription.cancel();
    } finally {
      _events.close();
    }
  }
}

// --- worker isolate ---

void _ttsWorkerMain(_TtsWorkerBootstrap bootstrap) {
  final epoch = bootstrap.epoch;
  final toMain = bootstrap.toMain;
  final fromMain = ReceivePort();
  toMain.send(_TtsWorkerPort(epoch, fromMain.sendPort));
  sherpa_onnx.OfflineTts? tts;

  fromMain.listen((msg) {
    if (msg is TtsWorkerInit && msg.epoch == epoch) {
      final init = msg.payload;
      if (init is! _TtsInit) return;
      sherpa_onnx.initBindings();
      final vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model: init.model,
        tokens: init.tokens,
        dataDir: init.dataDir,
      );
      final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
        vits: vits,
        numThreads: 2,
        provider: 'cpu',
      );
      tts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(model: modelConfig, maxNumSenetences: 1),
      );
      toMain.send(TtsWorkerReady(epoch));
    } else if (msg is TtsWorkerRequest && msg.epoch == epoch) {
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
        toMain.send(TtsWorkerResult(epoch, msg.id, ok ? msg.outPath : null));
      } catch (_) {
        toMain.send(TtsWorkerResult(epoch, msg.id, null));
      }
    } else if (msg is TtsWorkerShutdown && msg.epoch == epoch) {
      try {
        tts?.free();
        tts = null;
        fromMain.close();
        toMain.send(TtsWorkerShutdownComplete(epoch));
      } finally {
        fromMain.close();
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

  // ORDER MATCHES THE DESKTOP CORE (core/engines/sherpa.py::_synthesize):
  // normalize_rms FIRST, then declick. The declick threshold is an ABSOLUTE
  // amplitude tuned for the post-normalization level, so declicking the raw
  // (pre-boost) signal would compare against the wrong scale on a quiet sentence.

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

  // De-click: flag samples whose deviation from the 3-point median exceeds the
  // threshold (a real spike), then interpolate across runs up to maxRun long.
  // Reads the normalized signal (y), repairs in place; bad[] is fully computed
  // before any interpolation so a repair can't contaminate a later median test.
  if (declickThreshold > 0) {
    final bad = List<bool>.filled(n, false);
    for (var i = 0; i < n; i++) {
      final a = y[i == 0 ? 0 : i - 1];
      final b = y[i];
      final c = y[i == n - 1 ? n - 1 : i + 1];
      // median of three = sum - max - min
      final med =
          a + b + c - math.max(a, math.max(b, c)) - math.min(a, math.min(b, c));
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
  return y;
}

double _tanh(double x) {
  if (x > 20.0) return 1.0;
  if (x < -20.0) return -1.0;
  final e2 = math.exp(2.0 * x);
  return (e2 - 1.0) / (e2 + 1.0);
}
