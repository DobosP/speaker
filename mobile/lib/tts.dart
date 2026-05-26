// On-device text-to-speech screen.
// Adapted from the official sherpa-onnx tts Flutter example.
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './tts_model.dart';
import './utils.dart';

class TtsScreen extends StatefulWidget {
  const TtsScreen({super.key});

  @override
  State<TtsScreen> createState() => _TtsScreenState();
}

class _TtsScreenState extends State<TtsScreen> {
  final _textController = TextEditingController(text: 'Hello from your on-device assistant.');
  final _hintController = TextEditingController();
  final _player = AudioPlayer();

  bool _isInitialized = false;
  bool _busy = false;
  double _speed = 1.0;
  String _lastFilename = '';

  sherpa_onnx.OfflineTts? _tts;

  Future<void> _ensureInit() async {
    if (_isInitialized) return;
    _tts = await createOfflineTts();
    _isInitialized = true;
  }

  Future<void> _generate() async {
    final text = _textController.text.trim();
    if (text.isEmpty) {
      _hintController.text = 'Please enter some text first.';
      return;
    }
    setState(() => _busy = true);
    try {
      await _ensureInit();
      await _player.stop();

      final sw = Stopwatch()..start();
      final audio = _tts!.generateWithConfig(
        text: text,
        config: sherpa_onnx.OfflineTtsGenerationConfig(sid: 0, speed: _speed),
      );
      final filename = await generateWaveFilename();
      final ok = sherpa_onnx.writeWave(
        filename: filename,
        samples: audio.samples,
        sampleRate: audio.sampleRate,
      );
      sw.stop();

      if (ok) {
        _lastFilename = filename;
        final waveDur = audio.samples.length / audio.sampleRate;
        final elapsed = sw.elapsed.inMilliseconds / 1000.0;
        _hintController.text =
            'Synthesized ${waveDur.toStringAsFixed(2)}s of audio in '
            '${elapsed.toStringAsFixed(2)}s (RTF ${(elapsed / waveDur).toStringAsFixed(2)}).';
        await _player.play(DeviceFileSource(filename));
      } else {
        _hintController.text = 'Failed to write audio.';
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  void dispose() {
    _tts?.free();
    _player.dispose();
    _textController.dispose();
    _hintController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          TextField(
            controller: _textController,
            maxLines: 4,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              labelText: 'Text to speak',
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              const Text('Speed'),
              Expanded(
                child: Slider(
                  min: 0.5,
                  max: 2.0,
                  divisions: 15,
                  label: _speed.toStringAsFixed(2),
                  value: _speed,
                  onChanged: (v) => setState(() => _speed = v),
                ),
              ),
            ],
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              FilledButton.icon(
                onPressed: _busy ? null : _generate,
                icon: _busy
                    ? const SizedBox(
                        width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.volume_up),
                label: const Text('Speak'),
              ),
              const SizedBox(width: 12),
              OutlinedButton.icon(
                onPressed: () async {
                  if (_lastFilename.isNotEmpty) {
                    await _player.stop();
                    await _player.play(DeviceFileSource(_lastFilename));
                  }
                },
                icon: const Icon(Icons.replay),
                label: const Text('Replay'),
              ),
            ],
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _hintController,
            maxLines: 4,
            readOnly: true,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              hintText: 'Status / timing shown here.\n'
                  'First run is slower while the model loads.',
            ),
          ),
        ],
      ),
    );
  }
}
