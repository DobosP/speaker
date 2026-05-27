// Stage 2 of moving heavy work off the UI thread: streaming ASR.
//
// The recognizer's decode loop is synchronous native work. Running it on the
// main isolate for every audio chunk — including the backlog that builds up
// while the assistant is replying — froze the UI in a burst after each turn.
// Here the recognizer lives on a worker isolate: the main isolate only captures
// mic bytes (the `record` plugin is main-only) and forwards them; the worker
// decodes and sends back partial transcripts + endpoint events.
import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './asr_model.dart';

class AsrService {
  AsrService._();
  static final AsrService instance = AsrService._();

  // Endpoint trailing-silence (seconds) before an utterance is finalized.
  // sherpa's default 1.2s feels laggy; 0.8 trims it while tolerating pauses.
  static const endpointSilenceSec = 0.8;

  Isolate? _isolate;
  SendPort? _toWorker;
  Completer<void> _ready = Completer<void>();
  bool _starting = false;
  _AsrInit? _initData;

  // Set by the UI before listening: live partials and finalized utterances.
  void Function(String partial)? onPartial;
  void Function(String text)? onEndpoint;

  Future<void> ensureReady() {
    if (_toWorker != null && _ready.isCompleted) return _ready.future;
    if (_starting) return _ready.future;
    _starting = true;
    return _spawn();
  }

  Future<void> _spawn() async {
    // Resolve model file paths on the main isolate (asset copy needs plugins).
    final model = await getOnlineModelConfig();
    _initData = _AsrInit(
      encoder: model.transducer.encoder,
      decoder: model.transducer.decoder,
      joiner: model.transducer.joiner,
      tokens: model.tokens,
      modelType: model.modelType,
      silence: endpointSilenceSec,
    );
    final fromWorker = ReceivePort();
    fromWorker.listen(_onWorkerMessage);
    _isolate = await Isolate.spawn(_asrWorkerMain, fromWorker.sendPort);
    return _ready.future;
  }

  void _onWorkerMessage(dynamic msg) {
    if (msg is SendPort) {
      _toWorker = msg;
      _toWorker!.send(_initData);
    } else if (msg == 'ready') {
      if (!_ready.isCompleted) _ready.complete();
    } else if (msg is _AsrPartial) {
      onPartial?.call(msg.text);
    } else if (msg is _AsrEndpoint) {
      onEndpoint?.call(msg.text);
    }
  }

  // Forward a raw PCM16 mic chunk to the worker (cheap; no decoding on main).
  void feed(Uint8List bytes) {
    _toWorker?.send(_AsrAudio(bytes));
  }

  // Start a fresh recognizer stream (call when (re)starting a listen session).
  void reset() {
    _toWorker?.send('reset');
  }

  Future<void> stop() async {
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _toWorker = null;
    _ready = Completer<void>();
    _starting = false;
  }
}

// --- cross-isolate messages (sendable: primitives + Uint8List) ---

class _AsrInit {
  final String encoder;
  final String decoder;
  final String joiner;
  final String tokens;
  final String modelType;
  final double silence;
  _AsrInit({
    required this.encoder,
    required this.decoder,
    required this.joiner,
    required this.tokens,
    required this.modelType,
    required this.silence,
  });
}

class _AsrAudio {
  final Uint8List bytes;
  _AsrAudio(this.bytes);
}

class _AsrPartial {
  final String text;
  _AsrPartial(this.text);
}

class _AsrEndpoint {
  final String text;
  _AsrEndpoint(this.text);
}

// --- worker isolate ---

Float32List _toFloat32(Uint8List bytes) {
  final values = Float32List(bytes.length ~/ 2);
  final data = ByteData.sublistView(bytes);
  for (var i = 0; i + 1 < bytes.length; i += 2) {
    values[i ~/ 2] = data.getInt16(i, Endian.little) / 32768.0;
  }
  return values;
}

void _asrWorkerMain(SendPort toMain) {
  final fromMain = ReceivePort();
  toMain.send(fromMain.sendPort);
  sherpa_onnx.OnlineRecognizer? rec;
  sherpa_onnx.OnlineStream? stream;
  var lastPartial = '';

  fromMain.listen((msg) {
    if (msg is _AsrInit) {
      sherpa_onnx.initBindings();
      final model = sherpa_onnx.OnlineModelConfig(
        transducer: sherpa_onnx.OnlineTransducerModelConfig(
          encoder: msg.encoder,
          decoder: msg.decoder,
          joiner: msg.joiner,
        ),
        tokens: msg.tokens,
        modelType: msg.modelType,
      );
      rec = sherpa_onnx.OnlineRecognizer(
        sherpa_onnx.OnlineRecognizerConfig(
          model: model,
          ruleFsts: '',
          enableEndpoint: true,
          rule2MinTrailingSilence: msg.silence,
        ),
      );
      stream = rec!.createStream();
      toMain.send('ready');
    } else if (msg is _AsrAudio) {
      final r = rec;
      final s = stream;
      if (r == null || s == null) return;
      s.acceptWaveform(samples: _toFloat32(msg.bytes), sampleRate: 16000);
      while (r.isReady(s)) {
        r.decode(s);
      }
      final partial = r.getResult(s).text.trim();
      if (r.isEndpoint(s)) {
        final text = r.getResult(s).text.trim();
        r.reset(s);
        lastPartial = '';
        toMain.send(_AsrEndpoint(text));
      } else if (partial != lastPartial) {
        lastPartial = partial;
        toMain.send(_AsrPartial(partial));
      }
    } else if (msg == 'reset') {
      stream = rec?.createStream();
      lastPartial = '';
    }
  });
}
