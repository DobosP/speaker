// On-device streaming speech recognition screen.
// Adapted from the official sherpa-onnx streaming_asr Flutter example.
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './asr_model.dart';
import './utils.dart';

Future<sherpa_onnx.OnlineRecognizer> _createRecognizer() async {
  final modelConfig = await getOnlineModelConfig();
  final config = sherpa_onnx.OnlineRecognizerConfig(model: modelConfig, ruleFsts: '');
  return sherpa_onnx.OnlineRecognizer(config);
}

class AsrScreen extends StatefulWidget {
  const AsrScreen({super.key});

  @override
  State<AsrScreen> createState() => _AsrScreenState();
}

class _AsrScreenState extends State<AsrScreen> {
  final _controller = TextEditingController();
  final _audioRecorder = AudioRecorder();

  bool _isInitialized = false;
  String _last = '';
  int _index = 0;
  final int _sampleRate = 16000;

  sherpa_onnx.OnlineRecognizer? _recognizer;
  sherpa_onnx.OnlineStream? _stream;

  StreamSubscription<RecordState>? _recordSub;
  RecordState _recordState = RecordState.stop;

  @override
  void initState() {
    super.initState();
    _recordSub = _audioRecorder
        .onStateChanged()
        .listen((s) => setState(() => _recordState = s));
  }

  Future<void> _start() async {
    if (!_isInitialized) {
      sherpa_onnx.initBindings();
      _recognizer = await _createRecognizer();
      _stream = _recognizer!.createStream();
      _isInitialized = true;
    }

    if (!await _audioRecorder.hasPermission()) {
      return;
    }

    const config = RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: 16000,
      numChannels: 1,
    );
    final audioStream = await _audioRecorder.startStream(config);

    audioStream.listen((data) {
      final samples = convertBytesToFloat32(data);
      _stream!.acceptWaveform(samples: samples, sampleRate: _sampleRate);
      while (_recognizer!.isReady(_stream!)) {
        _recognizer!.decode(_stream!);
      }
      final text = _recognizer!.getResult(_stream!).text;
      var display = _last;
      if (text != '') {
        display = _last == '' ? '$_index: $text' : '$_index: $text\n$_last';
      }
      if (_recognizer!.isEndpoint(_stream!)) {
        _recognizer!.reset(_stream!);
        if (text != '') {
          _last = display;
          _index += 1;
        }
      }
      _controller.value = TextEditingValue(
        text: display,
        selection: TextSelection.collapsed(offset: display.length),
      );
    });
  }

  Future<void> _stop() async {
    await _audioRecorder.stop();
    _stream?.free();
    _stream = _recognizer?.createStream();
  }

  @override
  void dispose() {
    _recordSub?.cancel();
    _audioRecorder.dispose();
    _stream?.free();
    _recognizer?.free();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final listening = _recordState != RecordState.stop;
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          const SizedBox(height: 24),
          Text(listening ? 'Listening…' : 'Tap the mic and speak',
              style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 16),
          TextField(
            maxLines: 8,
            controller: _controller,
            readOnly: true,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              hintText: 'Recognized text appears here.\n'
                  'First run is slower while the model loads.',
            ),
          ),
          const SizedBox(height: 24),
          FloatingActionButton.large(
            backgroundColor: listening ? Colors.red : null,
            onPressed: () => listening ? _stop() : _start(),
            child: Icon(listening ? Icons.stop : Icons.mic),
          ),
        ],
      ),
    );
  }
}
