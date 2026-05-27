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
        final ok = sherpa_onnx.writeWave(
          filename: msg.outPath,
          samples: audio.samples,
          sampleRate: audio.sampleRate,
        );
        toMain.send(_TtsResult(msg.id, ok ? msg.outPath : null));
      } catch (_) {
        toMain.send(_TtsResult(msg.id, null));
      }
    }
  });
}
