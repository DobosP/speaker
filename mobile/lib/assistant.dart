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

  // Streaming TTS pipeline: completed sentences are queued and played back in
  // order while later sentences are still being generated, so the first words
  // are spoken almost immediately instead of after the whole answer.
  final List<String> _speechQueue = [];
  bool _draining = false;
  String _ttsBuffer = '';

  // Command fast-path: control phrases act locally without invoking the LLM.
  // On mobile the meaningful action is interrupting playback ("stop").
  static const Map<String, String> _commands = {
    'stop': 'stop',
    'cancel': 'stop',
    'quiet': 'stop',
    'stop talking': 'stop',
    'be quiet': 'stop',
  };

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

    // Command fast-path: a control phrase acts immediately, skipping the LLM.
    if (_handleCommand(prompt)) {
      _promptController.clear();
      return;
    }

    setState(() {
      _thinking = true;
      _answer = '';
    });
    _ttsBuffer = '';
    try {
      await for (final token in GemmaService.instance.reply(prompt)) {
        setState(() => _answer += token);
        _ttsBuffer += token;
        _flushSentences();
      }
      _flushSentences(flushAll: true);
    } catch (e) {
      setState(() => _answer = 'Generation failed: $e');
    } finally {
      if (mounted) setState(() => _thinking = false);
    }
  }

  // Map a recognized control phrase to a local action. Returns true if handled.
  bool _handleCommand(String prompt) {
    final key = prompt.toLowerCase().replaceAll(RegExp(r'[^a-z ]'), '').trim();
    final action = _commands[key];
    if (action == null) return false;
    if (action == 'stop') _stopSpeaking();
    return true;
  }

  Future<void> _stopSpeaking() async {
    _speechQueue.clear();
    _ttsBuffer = '';
    await _player.stop();
    if (mounted) setState(() => _status = 'Stopped.');
  }

  // Pull every completed sentence out of the rolling buffer and queue it for
  // speech; with flushAll, also speak whatever remains at end of generation.
  void _flushSentences({bool flushAll = false}) {
    var idx = _firstSentenceBoundary(_ttsBuffer);
    while (idx != -1) {
      final sentence = _ttsBuffer.substring(0, idx).trim();
      _ttsBuffer = _ttsBuffer.substring(idx);
      if (sentence.isNotEmpty) unawaited(_enqueueSpeech(sentence));
      idx = _firstSentenceBoundary(_ttsBuffer);
    }
    if (flushAll) {
      final rest = _ttsBuffer.trim();
      _ttsBuffer = '';
      if (rest.isNotEmpty) unawaited(_enqueueSpeech(rest));
    }
  }

  int _firstSentenceBoundary(String s) {
    for (var i = 0; i < s.length; i++) {
      final c = s[i];
      if (c == '.' || c == '!' || c == '?' || c == '\n') return i + 1;
    }
    return -1;
  }

  // Sequential player: synthesize + play one sentence at a time so audio never
  // overlaps, while generation of later sentences continues in parallel.
  Future<void> _enqueueSpeech(String sentence) async {
    _speechQueue.add(sentence);
    if (_draining) return;
    _draining = true;
    try {
      while (_speechQueue.isNotEmpty) {
        await _synthesizeAndPlay(_speechQueue.removeAt(0));
      }
    } finally {
      _draining = false;
    }
  }

  Future<void> _synthesizeAndPlay(String text) async {
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
    if (!ok) return;
    // Subscribe before playing so a fast/short clip can't complete before we
    // start awaiting (which would otherwise stall the queue).
    final done = _player.onPlayerComplete.first;
    await _player.play(DeviceFileSource(filename));
    await done;
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
