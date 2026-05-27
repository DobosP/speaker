// Assistant screen: the full on-device loop, always-on.
// Tap the mic once to start continuous listening: the streaming recognizer
// shows words live, and when it detects the end of an utterance (endpoint) the
// phrase is sent to Gemma automatically and the reply is spoken. The mic is
// muted while the assistant talks (no echo cancellation on-device) and resumes
// on its own afterwards. Tap again to stop.
import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './asr_model.dart';
import './contract.dart';
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

  // Always-on voice input: a streaming recognizer runs continuously and the
  // utterance is submitted automatically when an endpoint is detected.
  sherpa_onnx.OnlineRecognizer? _recognizer;
  sherpa_onnx.OnlineStream? _stream;
  StreamSubscription<List<int>>? _audioSub;
  bool _alwaysOn = false; // user has enabled continuous listening
  bool _listening = false; // mic stream currently active
  bool _busy = false; // answering an utterance; pauses mic intake

  // TTS.
  sherpa_onnx.OfflineTts? _tts;

  // Streaming TTS pipeline: completed sentences are queued and played back in
  // order while later sentences are still being generated, so the first words
  // are spoken almost immediately instead of after the whole answer.
  final List<String> _speechQueue = [];
  bool _draining = false;
  String _ttsBuffer = '';
  Completer<void>? _speechDone;

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
      setState(() => _status = 'Model ready — tap the mic to start.');
    } catch (e) {
      setState(() => _status = 'Model download/init failed: $e');
    } finally {
      if (mounted) setState(() => _downloading = false);
    }
  }

  // --- always-on listening ---

  Future<void> _toggleAlwaysOn() async {
    if (_alwaysOn) {
      _alwaysOn = false;
      await _stopListening();
      if (mounted) setState(() => _status = 'Stopped.');
      return;
    }
    if (!GemmaService.instance.isReady) return;
    _alwaysOn = true;
    await _startListening();
  }

  Future<void> _startListening() async {
    if (!await _recorder.hasPermission()) {
      _alwaysOn = false;
      setState(() => _status =
          'Microphone permission denied — enable it in system settings.');
      return;
    }
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
      _status = 'Listening…';
      _promptController.clear();
    });

    // Synchronous handler: each chunk is decoded to completion before the next,
    // and `_busy` blocks re-entry while an utterance is being answered.
    _audioSub = audioStream.listen((data) {
      if (_busy || _stream == null) return;
      final samples = convertBytesToFloat32(data);
      _stream!.acceptWaveform(samples: samples, sampleRate: 16000);
      while (_recognizer!.isReady(_stream!)) {
        _recognizer!.decode(_stream!);
      }
      final partial = _recognizer!.getResult(_stream!).text;
      if (mounted && partial.isNotEmpty) {
        _promptController.value = TextEditingValue(
          text: partial,
          selection: TextSelection.collapsed(offset: partial.length),
        );
      }
      if (_recognizer!.isEndpoint(_stream!)) {
        final utterance = _recognizer!.getResult(_stream!).text.trim();
        _recognizer!.reset(_stream!);
        if (utterance.isNotEmpty) {
          _busy = true; // set synchronously so queued chunks are ignored
          unawaited(_answerAndResume(utterance));
        }
      }
    });
  }

  Future<void> _stopListening() async {
    await _audioSub?.cancel();
    _audioSub = null;
    await _recorder.stop();
    _stream?.free();
    _stream = null;
    if (mounted) setState(() => _listening = false);
  }

  // One completed utterance: a control phrase acts immediately and keeps the
  // mic on; anything else is answered with the mic muted (so the assistant does
  // not transcribe its own speech), then listening resumes.
  Future<void> _answerAndResume(String utterance) async {
    if (_handleCommand(utterance)) {
      _busy = false;
      return;
    }
    await _stopListening();
    await _ask(utterance);
    await _awaitSpeechDone();
    _busy = false;
    if (_alwaysOn && mounted) {
      await _startListening();
    }
  }

  // --- generation ---

  // Generate a reply for [prompt] and stream it to TTS. Pure: callers handle
  // command phrases and clearing the input field.
  Future<void> _ask(String prompt) async {
    if (prompt.isEmpty || !GemmaService.instance.isReady) return;
    setState(() {
      _thinking = true;
      _answer = '';
      _status = 'Thinking…';
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

  // Send whatever is typed in the field (the manual fallback to voice).
  Future<void> _submitTyped() async {
    final text = _promptController.text.trim();
    if (text.isEmpty) return;
    _promptController.clear();
    if (_handleCommand(text)) return;
    await _ask(text);
  }

  // Map a recognized control phrase to a local action. Returns true if handled.
  // "stop" is the only action the mobile shell takes (it has no modes/supervisor);
  // the shared contract (contract.dart) decides what counts as "stop".
  bool _handleCommand(String prompt) {
    if (!isStopCommand(prompt)) return false;
    _stopSpeaking();
    return true;
  }

  Future<void> _stopSpeaking() async {
    _speechQueue.clear();
    _ttsBuffer = '';
    await _player.stop();
    // Unblock anyone awaiting the (now interrupted) speech.
    if (!(_speechDone?.isCompleted ?? true)) _speechDone!.complete();
    _draining = false;
    if (mounted) setState(() => _status = 'Stopped.');
  }

  // --- streaming TTS ---

  // Pull every completed sentence out of the rolling buffer and queue it for
  // speech; with flushAll, also speak whatever remains at end of generation.
  // Sentence boundaries follow the shared contract (contract.dart) so mobile and
  // the Python core split identically.
  void _flushSentences({bool flushAll = false}) {
    final (sentences, rest) = drainCompleteSentences(_ttsBuffer);
    _ttsBuffer = rest;
    for (final sentence in sentences) {
      unawaited(_enqueueSpeech(sentence));
    }
    if (flushAll) {
      final tail = _ttsBuffer.trim();
      _ttsBuffer = '';
      if (tail.isNotEmpty) unawaited(_enqueueSpeech(tail));
    }
  }

  // Sequential player: synthesize + play one sentence at a time so audio never
  // overlaps, while generation of later sentences continues in parallel.
  Future<void> _enqueueSpeech(String sentence) async {
    _speechQueue.add(sentence);
    if (_draining) return;
    _draining = true;
    _speechDone = Completer<void>();
    if (mounted) setState(() => _status = 'Speaking…');
    try {
      while (_speechQueue.isNotEmpty) {
        await _synthesizeAndPlay(_speechQueue.removeAt(0));
      }
    } finally {
      _draining = false;
      if (!(_speechDone?.isCompleted ?? true)) _speechDone!.complete();
    }
  }

  // Resolve once the current run of speech has finished playing.
  Future<void> _awaitSpeechDone() async {
    if (_draining) await _speechDone?.future;
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

  @override
  void dispose() {
    _alwaysOn = false;
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
                  onPressed: _thinking ? null : _submitTyped,
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
                  onPressed: _toggleAlwaysOn,
                  isSelected: _alwaysOn,
                  icon: Icon(_alwaysOn ? Icons.stop : Icons.mic),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Expanded(
              child: SingleChildScrollView(
                child: Text(
                  _answer.isEmpty
                      ? 'Tap the mic to start always-on listening. Replies appear here and are spoken aloud.'
                      : _answer,
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
