// On-device text-to-speech screen.
// Adapted from the official sherpa-onnx tts Flutter example.
import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './tts_model.dart';
import './tts_process_owner.dart';
import './utils.dart';

class TtsScreen extends StatefulWidget {
  const TtsScreen({super.key});

  @override
  State<TtsScreen> createState() => _TtsScreenState();
}

class _TtsScreenState extends State<TtsScreen> {
  final _textController =
      TextEditingController(text: 'Hello from your on-device assistant.');
  final _hintController = TextEditingController();
  TtsProcessLease? _speechLease;
  AudioPlayer? _player;

  bool _isInitialized = false;
  bool _busy = false;
  bool _constructionUncertain = false;
  bool _operationUncertain = false;
  double _speed = 1.0;
  String _lastFilename = '';

  sherpa_onnx.OfflineTts? _tts;
  Future<void> _operation = Future<void>.value();

  void _observeOperation(Future<void> operation) {
    // Keep the original future in [_operation] so disposal can inspect its
    // exact result, while preventing an already-recorded operation failure
    // from escaping as an unhandled Zone error before disposal runs.
    operation.ignore();
  }

  void _retainConstructionUncertainty() {
    _constructionUncertain = true;
    _speechLease?.revoke();
  }

  void _retainOperationUncertainty() {
    _operationUncertain = true;
    _speechLease?.revoke();
  }

  bool _ensureSpeechOwner() {
    if (_constructionUncertain || _operationUncertain) {
      _hintController.text = 'Speech output is unavailable until restart.';
      return false;
    }
    final existing = _speechLease;
    if (existing != null && existing.admitsWork) return true;
    final lease = ttsProcessOwnerRegistry.tryAcquire();
    if (lease == null) {
      _hintController.text =
          'Speech output is still shutting down — try again later.';
      return false;
    }
    _speechLease = lease;
    return true;
  }

  Future<void> _ensureInit() async {
    if (_isInitialized) return;
    if (!(_speechLease?.admitsWork ?? false)) {
      throw StateError('mobile speech ownership was revoked');
    }
    late final sherpa_onnx.OfflineTts created;
    try {
      created = await createOfflineTts();
    } catch (_) {
      _retainConstructionUncertainty();
      rethrow;
    }
    if (!(_speechLease?.admitsWork ?? false)) {
      try {
        created.free();
      } catch (_) {
        _retainConstructionUncertainty();
        rethrow;
      }
      throw StateError('mobile speech ownership was revoked');
    }
    _tts = created;
    _isInitialized = true;
  }

  AudioPlayer _ensurePlayer() {
    final existing = _player;
    if (existing != null) return existing;
    try {
      return _player = AudioPlayer();
    } catch (_) {
      _retainConstructionUncertainty();
      rethrow;
    }
  }

  void _generate() {
    if (_textController.text.trim().isEmpty) {
      _hintController.text = 'Please enter some text first.';
      return;
    }
    if (_busy || !_ensureSpeechOwner()) return;
    _busy = true;
    final operation = _runGenerate();
    _operation = operation;
    _observeOperation(operation);
  }

  Future<void> _runGenerate() async {
    final text = _textController.text.trim();
    try {
      if (mounted) setState(() {});
      await _ensureInit();
      if (!(_speechLease?.admitsWork ?? false)) return;
      final player = _ensurePlayer();
      await player.stop();
      if (!(_speechLease?.admitsWork ?? false)) return;

      final sw = Stopwatch()..start();
      final audio = _tts!.generateWithConfig(
        text: text,
        config: sherpa_onnx.OfflineTtsGenerationConfig(sid: 0, speed: _speed),
      );
      if (!(_speechLease?.admitsWork ?? false)) return;
      final filename = await generateWaveFilename();
      if (!(_speechLease?.admitsWork ?? false)) return;
      final ok = sherpa_onnx.writeWave(
        filename: filename,
        samples: audio.samples,
        sampleRate: audio.sampleRate,
      );
      sw.stop();

      if (ok && (_speechLease?.admitsWork ?? false)) {
        _lastFilename = filename;
        final waveDur = audio.samples.length / audio.sampleRate;
        final elapsed = sw.elapsed.inMilliseconds / 1000.0;
        _hintController.text =
            'Synthesized ${waveDur.toStringAsFixed(2)}s of audio in '
            '${elapsed.toStringAsFixed(2)}s (RTF ${(elapsed / waveDur).toStringAsFixed(2)}).';
        if (_speechLease?.admitsWork ?? false) {
          await player.play(DeviceFileSource(filename));
        }
      } else {
        _hintController.text = 'Failed to write audio.';
      }
    } catch (_) {
      _retainOperationUncertainty();
      rethrow;
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _runReplay() async {
    try {
      if (mounted) setState(() {});
      if (!(_speechLease?.admitsWork ?? false)) return;
      final player = _ensurePlayer();
      await player.stop();
      if (!(_speechLease?.admitsWork ?? false)) return;
      await player.play(DeviceFileSource(_lastFilename));
    } catch (_) {
      _retainOperationUncertainty();
      rethrow;
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  void dispose() {
    final lease = _speechLease;
    lease?.revoke();
    if (lease != null) {
      unawaited(lease.close(() async {
        var exact = !_constructionUncertain && !_operationUncertain;
        try {
          await _operation;
        } catch (_) {
          exact = false;
        }
        final player = _player;
        if (player != null) {
          try {
            await player.stop();
          } catch (_) {
            exact = false;
          }
          try {
            await player.dispose();
          } catch (_) {
            exact = false;
          }
        }
        try {
          _tts?.free();
          _tts = null;
        } catch (_) {
          exact = false;
        }
        return exact;
      }));
    }
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
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.volume_up),
                label: const Text('Speak'),
              ),
              const SizedBox(width: 12),
              OutlinedButton.icon(
                onPressed: _busy
                    ? null
                    : () {
                        if (_busy) return;
                        if (_lastFilename.isEmpty || !_ensureSpeechOwner())
                          return;
                        _busy = true;
                        _operation = _runReplay();
                        _observeOperation(_operation);
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
