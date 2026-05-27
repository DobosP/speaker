// Assistant screen: the full on-device loop.
// Speak or type -> small Gemma 3 generates locally (GPU) -> reply is spoken.
import 'dart:async';
import 'dart:typed_data';

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

  // Voice input. Two passes: a fast streaming recognizer shows words live as
  // they are spoken; when the user stops, the buffered audio is re-decoded by
  // a heavier offline model (_offline) to produce a more accurate transcript.
  sherpa_onnx.OnlineRecognizer? _recognizer;
  sherpa_onnx.OnlineStream? _stream;
  sherpa_onnx.OfflineRecognizer? _offline;
  StreamSubscription<List<int>>? _audioSub;
  bool _listening = false;
  bool _revising = false;
  // Raw mic samples for the current utterance, fed to the revision pass.
  final List<double> _utterance = [];
  static const int _maxUtteranceSamples = 16000 * 30; // Whisper handles ~30s.

  // TTS.
  sherpa_onnx.OfflineTts? _tts;

  // Streaming TTS pipeline: completed sentences are queued and played back in
  // order while later sentences are still being generated, so the first words
  // are spoken almost immediately instead of after the whole answer.
  final List<String> _speechQueue = [];
  bool _draining = false;
  String _ttsBuffer = '';

  // Command fast-path: control phrases act locally without the LLM. The shared
  // contract (contract.dart) decides what counts as "stop" so desktop and mobile
  // recognize the same phrases.

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
  // "stop" is the only action the mobile shell takes (it has no modes/supervisor).
  bool _handleCommand(String prompt) {
    if (!isStopCommand(prompt)) return false;
    _stopSpeaking();
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
      return;
    }
    try {
      if (!await _recorder.hasPermission()) {
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
      _utterance.clear();

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

      _audioSub = audioStream.listen((data) {
        final samples = convertBytesToFloat32(data);
        // Buffer the raw audio for the offline revision pass (cap to ~30s).
        _utterance.addAll(samples);
        if (_utterance.length > _maxUtteranceSamples) {
          _utterance.removeRange(0, _utterance.length - _maxUtteranceSamples);
        }
        _stream!.acceptWaveform(samples: samples, sampleRate: 16000);
        while (_recognizer!.isReady(_stream!)) {
          _recognizer!.decode(_stream!);
        }
        final text = _recognizer!.getResult(_stream!).text;
        if (text.isNotEmpty && mounted) {
          _promptController.value = TextEditingValue(
            text: text,
            selection: TextSelection.collapsed(offset: text.length),
          );
        }
      });
    } catch (e) {
      setState(() {
        _listening = false;
        _status = 'Could not start the microphone: $e';
      });
    }
  }

  Future<void> _stopListening() async {
    await _audioSub?.cancel();
    _audioSub = null;
    await _recorder.stop();
    _stream?.free();
    _stream = null;
    setState(() => _listening = false);

    // Second pass: re-decode the full utterance with the heavier offline model
    // and replace the live transcript with the more accurate result.
    final revised = await _reviseTranscript();
    if (revised != null && revised.isNotEmpty) {
      _promptController.value = TextEditingValue(
        text: revised,
        selection: TextSelection.collapsed(offset: revised.length),
      );
    }
    if (_promptController.text.trim().isNotEmpty) await _ask();
  }

  // Run the buffered utterance through offline Whisper for a corrected
  // transcript. Returns null (and leaves the live text in place) on failure so
  // a missing/incompatible model never breaks voice input.
  Future<String?> _reviseTranscript() async {
    if (_utterance.isEmpty) return null;
    setState(() {
      _revising = true;
      _status = 'Revising transcript…';
    });
    sherpa_onnx.OfflineStream? s;
    try {
      _offline ??= await _createOfflineRecognizer();
      final samples = Float32List.fromList(_utterance);
      s = _offline!.createStream();
      s.acceptWaveform(samples: samples, sampleRate: 16000);
      _offline!.decode(s);
      final text = _offline!.getResult(s).text.trim();
      setState(() => _status = text.isEmpty ? '' : 'Revised transcript ready.');
      return text;
    } catch (_) {
      setState(() => _status = 'Used live transcript (revision unavailable).');
      return null;
    } finally {
      s?.free();
      if (mounted) setState(() => _revising = false);
    }
  }

  Future<sherpa_onnx.OfflineRecognizer> _createOfflineRecognizer() async {
    final model = await getOfflineWhisperConfig();
    return sherpa_onnx.OfflineRecognizer(
      sherpa_onnx.OfflineRecognizerConfig(model: model),
    );
  }

  @override
  void dispose() {
    _audioSub?.cancel();
    _recorder.dispose();
    _stream?.free();
    _recognizer?.free();
    _offline?.free();
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
                labelText: _listening
                    ? 'Listening…'
                    : _revising
                        ? 'Revising…'
                        : 'Ask something',
              ),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                FilledButton.icon(
                  onPressed: (_thinking || _revising) ? null : _ask,
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
                  onPressed: (_thinking || _revising) ? null : _toggleMic,
                  icon: _revising
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2))
                      : Icon(_listening ? Icons.stop : Icons.mic),
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
