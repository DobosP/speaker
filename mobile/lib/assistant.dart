// Assistant screen: the full on-device loop, always-on with barge-in.
// Tap the mic once to start continuous listening: the streaming recognizer shows
// words live, and when it detects the end of an utterance (endpoint) the phrase
// is sent to Gemma automatically and the reply is spoken. The mic stays LIVE
// while the assistant talks — with acoustic echo cancellation so it doesn't hear
// its own voice — so you can interrupt (barge-in) just by talking, or say "stop".
// Tap again to stop. A per-turn latency readout + "Copy logs" export are wired in
// for on-device profiling.
import 'dart:async';
import 'dart:io' show Platform;
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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

  // Always-on voice input: a streaming recognizer runs continuously. The mic
  // stays live even while the assistant speaks (echo cancellation keeps its own
  // voice out) so the user can interrupt — barge-in — by talking or saying stop.
  sherpa_onnx.OnlineRecognizer? _recognizer;
  sherpa_onnx.OnlineStream? _stream;
  StreamSubscription<List<int>>? _audioSub;
  bool _alwaysOn = false; // user has enabled continuous listening
  bool _listening = false; // mic stream currently active
  bool _speaking = false; // TTS is playing (the barge-in target)
  int _turn = 0; // increments per utterance; a newer turn supersedes older gen

  // Barge-in sensitivity: how many recognized characters during playback count
  // as "the user is talking" and should cut the assistant off. Tune per device:
  // lower = snappier but more prone to false trips from residual echo.
  static const _bargeInChars = 2;

  // TTS.
  sherpa_onnx.OfflineTts? _tts;

  // Streaming TTS pipeline: completed sentences are queued and played back in
  // order while later sentences are still being generated, so the first words
  // are spoken almost immediately instead of after the whole answer.
  final List<String> _speechQueue = [];
  bool _draining = false;
  String _ttsBuffer = '';
  Completer<void>? _playInterrupt; // completes to cut the current clip short

  // Latency profiling, shown on screen (stages mirror core/metrics.py):
  //   silence wait : last speech -> endpoint (the trailing-silence timeout)
  //   -> 1st token : endpoint -> first LLM token (time-to-first-token)
  //   -> speaking  : endpoint -> first TTS audio (the felt round-trip)
  //   gen          : token count and tokens/sec
  // Lets us see on-device where each turn's seconds actually go.
  DateTime? _tLastVoice; // last chunk with a non-empty partial (~ user talking)
  DateTime? _tHeard; // endpoint fired / utterance submitted
  DateTime? _tFirstToken;
  DateTime? _tFirstAudio;
  DateTime? _tGenDone;
  int _genTokens = 0;
  String _metrics = '';

  // Rolling per-turn log (timestamp + utterance + the metrics above, or an
  // error). Exported via the "Copy logs" button — there is no embedded
  // credential, so the only way logs leave the device is the user sharing them.
  final List<String> _log = [];
  String? _lastUtterance;

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

  // Route playback through the Android voice-communication path so the hardware
  // echo-canceller (engaged by the voiceCommunication record source) actually
  // cancels the assistant's own TTS — without this the mic hears the speaker and
  // barges in on itself. Android-only here; iOS keeps the audioplayers default.
  Future<void> _configureAudioSession() async {
    await _player.setAudioContext(
      AudioContext(
        android: const AudioContextAndroid(
          isSpeakerphoneOn: true,
          contentType: AndroidContentType.speech,
          usageType: AndroidUsageType.voiceCommunication,
          audioFocus: AndroidAudioFocus.gainTransientMayDuck,
        ),
      ),
    );
  }

  Future<void> _startListening() async {
    if (!await _recorder.hasPermission()) {
      _alwaysOn = false;
      setState(() => _status =
          'Microphone permission denied — enable it in system settings.');
      return;
    }
    await _configureAudioSession();
    if (_recognizer == null) {
      sherpa_onnx.initBindings();
      final modelConfig = await getOnlineModelConfig();
      _recognizer = sherpa_onnx.OnlineRecognizer(
        sherpa_onnx.OnlineRecognizerConfig(model: modelConfig, ruleFsts: ''),
      );
    }
    _stream = _recognizer!.createStream();

    // voiceCommunication + echoCancel let the mic stay open during playback
    // without the recognizer transcribing the assistant's own TTS.
    final config = RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: 16000,
      numChannels: 1,
      echoCancel: true,
      noiseSuppress: true,
      autoGain: true,
      androidConfig: const AndroidRecordConfig(
        audioSource: AndroidAudioSource.voiceCommunication,
      ),
    );
    final audioStream = await _recorder.startStream(config);
    setState(() {
      _listening = true;
      _status = 'Listening…';
      _promptController.clear();
    });

    _audioSub = audioStream.listen(_onAudio);
  }

  // Each audio chunk is decoded to completion in order. The mic never pauses, so
  // this also runs while the assistant is speaking — that is what enables
  // barge-in.
  void _onAudio(Uint8List data) {
    if (_stream == null) return;
    final samples = convertBytesToFloat32(data);
    _stream!.acceptWaveform(samples: samples, sampleRate: 16000);
    while (_recognizer!.isReady(_stream!)) {
      _recognizer!.decode(_stream!);
    }
    final partial = _recognizer!.getResult(_stream!).text.trim();

    // Barge-in: the user starts talking while the assistant speaks -> cut the
    // playback immediately so we stop talking over them.
    if (_speaking && partial.length >= _bargeInChars) {
      unawaited(_stopSpeaking());
    }

    if (partial.isNotEmpty) {
      _tLastVoice = DateTime.now();
      if (mounted) {
        _promptController.value = TextEditingValue(
          text: partial,
          selection: TextSelection.collapsed(offset: partial.length),
        );
      }
    }

    if (_recognizer!.isEndpoint(_stream!)) {
      final utterance = _recognizer!.getResult(_stream!).text.trim();
      _recognizer!.reset(_stream!);
      if (utterance.isEmpty) return;
      // A completed utterance supersedes any in-flight reply (its tokens stop
      // feeding TTS) and silences whatever is still playing.
      _turn++;
      unawaited(_stopSpeaking());
      if (isStopCommand(utterance)) {
        if (mounted) {
          setState(() {
            _status = 'Stopped.';
            _promptController.clear();
          });
        }
        return;
      }
      _tHeard = DateTime.now();
      unawaited(_answerUtterance(utterance, _turn));
    }
  }

  Future<void> _stopListening() async {
    await _audioSub?.cancel();
    _audioSub = null;
    await _recorder.stop();
    await _stopSpeaking();
    _stream?.free();
    _stream = null;
    if (mounted) setState(() => _listening = false);
  }

  // --- generation ---

  // Generate a reply for [prompt] and stream it to TTS. [myTurn] guards against
  // a newer utterance arriving (barge-in): once the turn advances, this reply
  // stops emitting tokens and queuing speech.
  Future<void> _answerUtterance(String prompt, int myTurn) async {
    if (prompt.isEmpty || !GemmaService.instance.isReady) return;
    _lastUtterance = prompt;
    setState(() {
      _thinking = true;
      _answer = '';
      _status = 'Thinking…';
    });
    _ttsBuffer = '';
    _tFirstToken = null;
    _tFirstAudio = null;
    _tGenDone = null;
    _genTokens = 0;
    try {
      await for (final token in GemmaService.instance.reply(prompt)) {
        if (myTurn != _turn) return; // superseded by a newer utterance
        if (_tFirstToken == null) {
          _tFirstToken = DateTime.now();
          _updateMetrics();
        }
        _genTokens++;
        setState(() => _answer += token);
        _ttsBuffer += token;
        _flushSentences();
      }
      if (myTurn == _turn) {
        _tGenDone = DateTime.now();
        _flushSentences(flushAll: true);
        _updateMetrics();
        _appendLog();
      }
    } catch (e) {
      if (myTurn == _turn) {
        setState(() => _answer = 'Generation failed: $e');
        _appendLog(error: '$e');
      }
    } finally {
      if (mounted && myTurn == _turn) setState(() => _thinking = false);
    }
  }

  // Send whatever is typed in the field (the manual fallback to voice).
  Future<void> _submitTyped() async {
    final text = _promptController.text.trim();
    if (text.isEmpty) return;
    _promptController.clear();
    _turn++;
    await _stopSpeaking();
    if (isStopCommand(text)) {
      if (mounted) setState(() => _status = 'Stopped.');
      return;
    }
    _tLastVoice = null; // typed input has no silence-wait stage
    _tHeard = DateTime.now();
    await _answerUtterance(text, _turn);
  }

  // Cut all speech now: drop the queue, stop the current clip, and release any
  // coroutine awaiting playback. Idempotent.
  Future<void> _stopSpeaking() async {
    _speechQueue.clear();
    _ttsBuffer = '';
    _speaking = false;
    _draining = false;
    if (!(_playInterrupt?.isCompleted ?? true)) _playInterrupt!.complete();
    await _player.stop();
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
    _speaking = true;
    if (mounted) setState(() => _status = 'Speaking…');
    try {
      while (_speechQueue.isNotEmpty) {
        await _synthesizeAndPlay(_speechQueue.removeAt(0));
      }
    } finally {
      _draining = false;
      _speaking = false;
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
    _playInterrupt = Completer<void>();
    await _player.play(DeviceFileSource(filename));
    if (_tFirstAudio == null) {
      _tFirstAudio = DateTime.now();
      _updateMetrics();
    }
    // Whichever happens first: the clip finishes, or barge-in cuts it short.
    await Future.any([done, _playInterrupt!.future]);
  }

  // --- latency profiling ---

  int _ms(DateTime a, DateTime b) => b.difference(a).inMilliseconds;

  void _updateMetrics() {
    if (_tHeard == null) return;
    final lines = <String>[];
    if (_tLastVoice != null && _tLastVoice!.isBefore(_tHeard!)) {
      lines.add('silence wait : ${_ms(_tLastVoice!, _tHeard!)} ms');
    }
    if (_tFirstToken != null) {
      lines.add('-> 1st token : ${_ms(_tHeard!, _tFirstToken!)} ms');
    }
    if (_tFirstAudio != null) {
      lines.add('-> speaking  : ${_ms(_tHeard!, _tFirstAudio!)} ms');
    }
    if (_tFirstToken != null && _tGenDone != null && _genTokens > 1) {
      final secs = _ms(_tFirstToken!, _tGenDone!) / 1000.0;
      final rate = secs > 0 ? _genTokens / secs : 0.0;
      lines.add('gen          : $_genTokens tok @ ${rate.toStringAsFixed(1)}/s');
    }
    if (mounted) setState(() => _metrics = lines.join('\n'));
  }

  void _appendLog({String? error}) {
    final ts = DateTime.now().toIso8601String();
    final header = '[$ts] "${_lastUtterance ?? ''}"';
    _log.add(error != null ? '$header ERROR: $error' : '$header\n$_metrics');
    if (_log.length > 100) _log.removeRange(0, _log.length - 100);
    if (mounted) setState(() {});
  }

  Future<void> _copyLogs() async {
    final head = 'speaker mobile — ${Platform.operatingSystem} '
        '${Platform.operatingSystemVersion} — Gemma 3 1B';
    await Clipboard.setData(ClipboardData(text: '$head\n\n${_log.join('\n')}'));
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Logs copied — paste them to Claude.')),
      );
    }
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
          if (_metrics.isNotEmpty) ...[
            const SizedBox(height: 6),
            Text(_metrics,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
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
            if (_log.isNotEmpty)
              Align(
                alignment: Alignment.center,
                child: TextButton.icon(
                  onPressed: _copyLogs,
                  icon: const Icon(Icons.copy_all, size: 18),
                  label: Text('Copy logs (${_log.length} turns)'),
                ),
              ),
            const SizedBox(height: 16),
            Expanded(
              child: SingleChildScrollView(
                child: Text(
                  _answer.isEmpty
                      ? 'Tap the mic for always-on listening. It replies aloud — '
                          'just start talking (or say "stop") to interrupt.'
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
