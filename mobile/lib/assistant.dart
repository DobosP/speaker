// Assistant screen: the full on-device loop.
// Speak or type -> small Gemma 3 generates locally (GPU) -> reply is spoken.
import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './asr_model.dart';
import './llm.dart';
import './tts_model.dart';
import './utils.dart';

class AssistantScreen extends StatefulWidget {
  const AssistantScreen({super.key});

  @override
  State<AssistantScreen> createState() => _AssistantScreenState();
}

class _AssistantScreenState extends State<AssistantScreen> {
  final _promptController = TextEditingController();
  final _player = AudioPlayer();
  final _recorder = AudioRecorder();

  // Model lifecycle.
  bool _downloading = false;
  double _downloadPct = 0;
  String _status = '';

  // Generation.
  bool _thinking = false;
  String _answer = '';

  // Voice input.
  sherpa_onnx.OnlineRecognizer? _recognizer;
  sherpa_onnx.OnlineStream? _stream;
  StreamSubscription<List<int>>? _audioSub;
  bool _listening = false;

  // TTS.
  sherpa_onnx.OfflineTts? _tts;

  Future<void> _downloadModel() async {
    setState(() {
      _downloading = true;
      _downloadPct = 0;
      _status = 'Downloading Gemma 3 1B (one time, ~550 MB)…';
    });
    try {
      await GemmaService.instance.ensureReady(
        onProgress: (p) => setState(() => _downloadPct = p),
      );
      setState(() => _status = 'Model ready — ask me anything.');
    } catch (e) {
      setState(() => _status = 'Model download/init failed: $e');
    } finally {
      if (mounted) setState(() => _downloading = false);
    }
  }

  Future<void> _ask() async {
    final prompt = _promptController.text.trim();
    if (prompt.isEmpty || !GemmaService.instance.isReady) return;
    setState(() {
      _thinking = true;
      _answer = '';
    });
    try {
      await for (final token in GemmaService.instance.reply(prompt)) {
        setState(() => _answer += token);
      }
      await _speak(_answer);
    } catch (e) {
      setState(() => _answer = 'Generation failed: $e');
    } finally {
      if (mounted) setState(() => _thinking = false);
    }
  }

  Future<void> _speak(String text) async {
    if (text.trim().isEmpty) return;
    _tts ??= await createOfflineTts();
    final audio = _tts!.generateWithConfig(
      text: text,
      config: sherpa_onnx.OfflineTtsGenerationConfig(sid: 0, speed: 1.0),
    );
    final filename = await generateWaveFilename();
    final ok = sherpa_onnx.writeWave(
      filename: filename,
      samples: audio.samples,
      sampleRate: audio.sampleRate,
    );
    if (ok) {
      await _player.stop();
      await _player.play(DeviceFileSource(filename));
    }
  }

  Future<void> _toggleMic() async {
    if (_listening) {
      await _stopListening();
      if (_promptController.text.trim().isNotEmpty) await _ask();
      return;
    }
    if (!await _recorder.hasPermission()) return;

    if (_recognizer == null) {
      sherpa_onnx.initBindings();
      final modelConfig = await getOnlineModelConfig();
      _recognizer = sherpa_onnx.OnlineRecognizer(
        sherpa_onnx.OnlineRecognizerConfig(model: modelConfig, ruleFsts: ''),
      );
    }
    _stream = _recognizer!.createStream();

    const config = RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: 16000,
      numChannels: 1,
    );
    final audioStream = await _recorder.startStream(config);
    setState(() {
      _listening = true;
      _promptController.clear();
    });

    _audioSub = audioStream.listen((data) {
      final samples = convertBytesToFloat32(data);
      _stream!.acceptWaveform(samples: samples, sampleRate: 16000);
      while (_recognizer!.isReady(_stream!)) {
        _recognizer!.decode(_stream!);
      }
      final text = _recognizer!.getResult(_stream!).text;
      if (text.isNotEmpty) {
        _promptController.text = text;
      }
    });
  }

  Future<void> _stopListening() async {
    await _audioSub?.cancel();
    _audioSub = null;
    await _recorder.stop();
    _stream?.free();
    _stream = null;
    setState(() => _listening = false);
  }

  @override
  void dispose() {
    _audioSub?.cancel();
    _recorder.dispose();
    _stream?.free();
    _recognizer?.free();
    _tts?.free();
    _player.dispose();
    _promptController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ready = GemmaService.instance.isReady;
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (!ready) ...[
            const SizedBox(height: 8),
            Text('On-device Gemma 3',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            if (_downloading)
              Column(
                children: [
                  LinearProgressIndicator(value: _downloadPct / 100.0),
                  const SizedBox(height: 8),
                  Text('${_downloadPct.toStringAsFixed(0)}%'),
                ],
              )
            else
              FilledButton.icon(
                onPressed: _downloadModel,
                icon: const Icon(Icons.download),
                label: const Text('Download model (one time)'),
              ),
          ],
          if (_status.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(_status, style: Theme.of(context).textTheme.bodySmall),
          ],
          if (ready) ...[
            const SizedBox(height: 8),
            TextField(
              controller: _promptController,
              maxLines: 3,
              decoration: InputDecoration(
                border: const OutlineInputBorder(),
                labelText: _listening ? 'Listening…' : 'Ask something',
              ),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                FilledButton.icon(
                  onPressed: _thinking ? null : _ask,
                  icon: _thinking
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.send),
                  label: const Text('Ask'),
                ),
                const SizedBox(width: 12),
                IconButton.filledTonal(
                  onPressed: _thinking ? null : _toggleMic,
                  icon: Icon(_listening ? Icons.stop : Icons.mic),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Expanded(
              child: SingleChildScrollView(
                child: Text(
                  _answer.isEmpty ? 'The reply appears here and is spoken aloud.' : _answer,
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
