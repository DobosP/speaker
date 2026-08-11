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
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:record/record.dart';

import './asr_isolate.dart';
import './audio_capture_config.dart';
import './barge_calibrator.dart';
import './contract.dart';
import './llm.dart';
import './tts_isolate.dart';
import './tts_playback_owner.dart';
import './utils.dart';

AudioContext _voicePlaybackContext() => AudioContext(
      android: const AudioContextAndroid(
        isSpeakerphoneOn: true,
        contentType: AndroidContentType.speech,
        usageType: AndroidUsageType.voiceCommunication,
        audioFocus: AndroidAudioFocus.gainTransientMayDuck,
      ),
    );

final class _AssistantTtsClipState {
  _AssistantTtsClipState({
    required this.player,
    required this.started,
    required this.terminal,
  });

  final AudioPlayer player;
  final Completer<TtsPlaybackResult> started;
  final Completer<TtsPlaybackTerminal> terminal;
  StreamSubscription<void>? completionSubscription;
  late TtsPlaybackClip handle;
  Future<TtsPlaybackResult>? cleanup;
  bool cleanupStarted = false;
}

/// Owns the exact AudioPlayer instance behind each sentence clip.
///
/// `AudioPlayer.onPlayerComplete` is not tagged with a play generation. A fresh
/// player per clip makes its event stream an exact identity boundary instead of
/// trying to infer which play produced an event on one shared stream.
final class _AssistantTtsPlayer {
  _AssistantTtsClipState? _current;
  bool _closed = false;

  Future<void> configureGlobalRoute() =>
      AudioPlayer.global.setAudioContext(_voicePlaybackContext());

  TtsPlaybackClip createClip(String path) {
    if (_closed) throw StateError('mobile TTS player is closed');
    if (_current != null) {
      throw StateError('previous mobile TTS clip is not released');
    }

    final started = Completer<TtsPlaybackResult>();
    final terminal = Completer<TtsPlaybackTerminal>();
    final state = _AssistantTtsClipState(
      player: AudioPlayer(),
      started: started,
      terminal: terminal,
    );
    final handle = TtsPlaybackClip(
      started: started.future,
      terminal: terminal.future,
      stopAndRelease: () => _stopAndRelease(state),
    );
    state.handle = handle;
    _current = state;

    try {
      state.completionSubscription = state.player.onPlayerComplete.listen(
        (_) {
          if (!terminal.isCompleted) {
            terminal.complete(const TtsPlaybackTerminal.completed());
          }
        },
        onError: (Object error, StackTrace stackTrace) {
          if (!terminal.isCompleted) {
            terminal.complete(
              TtsPlaybackTerminal.failed(error, stackTrace),
            );
          }
        },
        onDone: () {
          if (terminal.isCompleted) return;
          if (state.cleanupStarted) {
            terminal.complete(const TtsPlaybackTerminal.interrupted());
          } else {
            terminal.complete(
              TtsPlaybackTerminal.failed(
                StateError('mobile TTS completion stream closed early'),
                StackTrace.current,
              ),
            );
          }
        },
      );
      // Let the owner receive and install the exact handle before native
      // route/play admission. AudioPlayer construction may already have begun
      // plugin setup, but it cannot play this path before the scheduled start.
      scheduleMicrotask(() {
        unawaited(_start(state, path));
      });
    } catch (error, stackTrace) {
      final failed = TtsPlaybackResult.failure(error, stackTrace);
      if (!started.isCompleted) started.complete(failed);
      if (!terminal.isCompleted) {
        terminal.complete(TtsPlaybackTerminal.failed(error, stackTrace));
      }
    }
    return handle;
  }

  Future<void> _start(_AssistantTtsClipState state, String path) async {
    try {
      await state.player.setAudioContext(_voicePlaybackContext());
      await state.player.play(DeviceFileSource(path));
      if (!state.started.isCompleted) {
        state.started.complete(const TtsPlaybackResult.success());
      }
    } catch (error, stackTrace) {
      if (!state.started.isCompleted) {
        state.started.complete(TtsPlaybackResult.failure(error, stackTrace));
      }
      if (!state.terminal.isCompleted) {
        state.terminal.complete(
          TtsPlaybackTerminal.failed(error, stackTrace),
        );
      }
    }
  }

  Future<TtsPlaybackResult> _stopAndRelease(
    _AssistantTtsClipState state,
  ) {
    final existing = state.cleanup;
    if (existing != null) return existing;

    final completed = Completer<TtsPlaybackResult>();
    state.cleanup = completed.future;
    state.cleanupStarted = true;
    unawaited(
      _runCleanup(state).then(
        completed.complete,
        onError: (Object error, StackTrace stackTrace) {
          completed.complete(TtsPlaybackResult.failure(error, stackTrace));
        },
      ),
    );
    return completed.future;
  }

  Future<TtsPlaybackResult> _runCleanup(
    _AssistantTtsClipState state,
  ) async {
    Object? stopError;
    StackTrace? stopStackTrace;
    try {
      await state.player.stop();
    } catch (error, stackTrace) {
      stopError = error;
      stopStackTrace = stackTrace;
    }

    Object? disposeError;
    StackTrace? disposeStackTrace;
    try {
      await state.player.dispose();
    } catch (error, stackTrace) {
      disposeError = error;
      disposeStackTrace = stackTrace;
    }
    if (disposeError != null) {
      final error = stopError == null
          ? disposeError
          : StateError(
              'mobile TTS stop and dispose both failed: '
              '$stopError; $disposeError',
            );
      return TtsPlaybackResult.failure(
        error,
        disposeStackTrace ?? stopStackTrace ?? StackTrace.current,
      );
    }

    Object? subscriptionError;
    StackTrace? subscriptionStackTrace;
    try {
      await state.completionSubscription?.cancel();
    } catch (error, stackTrace) {
      subscriptionError = error;
      subscriptionStackTrace = stackTrace;
    }
    if (identical(_current, state)) _current = null;
    if (!state.terminal.isCompleted) {
      state.terminal.complete(const TtsPlaybackTerminal.interrupted());
    }

    final diagnostic = subscriptionError ?? stopError;
    return TtsPlaybackResult.success(
      error: diagnostic,
      stackTrace: subscriptionStackTrace ?? stopStackTrace,
    );
  }

  Future<bool> close() async {
    _closed = true;
    final current = _current;
    if (current == null) return true;
    return (await current.handle.stopAndRelease()).succeeded;
  }
}

class AssistantScreen extends StatefulWidget {
  const AssistantScreen({super.key});

  @override
  State<AssistantScreen> createState() => _AssistantScreenState();
}

class _AssistantScreenState extends State<AssistantScreen> {
  final _promptController = TextEditingController();
  final _ttsPlayer = _AssistantTtsPlayer();
  final _recorder = AudioRecorder();
  late final TtsPlaybackOwner _ttsPlayback;
  Future<bool>? _speechStopInFlight;
  TtsPlaybackGeneration? _speechStopGeneration;
  bool _disposing = false;

  // Model lifecycle.
  bool _downloading = false;
  double _downloadPct = 0;
  bool _modelPresent = false; // weights already on disk -> load, don't download
  String _status = '';

  // Generation.
  bool _thinking = false;
  String _answer = '';

  // Always-on voice input: the mic stays live even while the assistant speaks
  // (echo cancellation keeps its own voice out) so the user can interrupt —
  // barge-in. Mic bytes are forwarded to the ASR worker isolate (asr_isolate),
  // which decodes off the main thread and calls back with partials/endpoints.
  StreamSubscription<List<int>>? _audioSub;
  bool _alwaysOn = false; // user has enabled continuous listening
  bool _listening = false; // mic stream currently active
  bool _speaking = false; // TTS work/physical uncertainty (barge-in target)
  int _turn = 0; // increments per utterance; a newer turn supersedes older gen

  // Barge-in sensitivity: how many recognized characters during playback count
  // as "the user is talking" and should cut the assistant off. Tune per device:
  // lower = snappier but more prone to false trips from residual echo.
  static const _bargeInChars = 2;

  // Energy barge-in: talking over the assistant should cut it off even when the
  // recognizer can't transcribe the interruption (echo, or a short word). We
  // watch near-end loudness on the mic stream during playback and stop on
  // sustained sound. Thresholds are device-dependent — calibrate from the
  // "spoke: … peakRms" line in the exported logs.
  static const _bargeInRms =
      0.08; // legacy fixed floor; now the calibrator's min
  // Adaptive barge-in threshold: learns the room's ambient floor from the
  // echo-free (not-speaking) windows and raises the bar to a margin above it, so
  // background noise can't self-interrupt. Floored at _bargeInRms -> never less
  // sensitive than the old constant in a quiet room (no default regression).
  final BargeCalibrator _bargeCal = BargeCalibrator(absoluteMin: _bargeInRms);
  final QuietObservationGate _quietGate = QuietObservationGate();
  static const _bargeInMs = 150; // sustained loud audio (ms) before cutting off
  int _loudMs = 0; // accumulated loud-audio time in the current window
  // Per speaking-window diagnostics (exported via Copy logs) so we can see
  // whether the mic is even live during playback and how loud near-end is.
  int _spkChunks = 0; // mic chunks observed while speaking
  double _spkPeakRms = 0; // loudest near-end chunk while speaking
  int _spkPartials = 0; // partials received while speaking

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
  int? _logIndex; // index of the current turn's log entry (updated in place)

  // Rolling per-turn log (timestamp + utterance + the metrics above, or an
  // error). Exported via the "Copy logs" button — there is no embedded
  // credential, so the only way logs leave the device is the user sharing them.
  final List<String> _log = [];
  String? _lastUtterance;

  @override
  void initState() {
    super.initState();
    _ttsPlayback = TtsPlaybackOwner(
      synthesize: (text) async {
        final filename = await generateWaveFilename();
        return TtsService.instance.synthesize(text, filename);
      },
      createPlaybackClip: _ttsPlayer.createClip,
      onActivityChanged: _onSpeechActivityChanged,
      onPlaybackStarted: _onPlaybackStarted,
      onError: _onSpeechPlaybackError,
    );
    // Probe disk so the button/status reflect whether a network download is
    // actually needed (the model persists across reinstalls).
    GemmaService.instance.isModelPresent().then((present) {
      if (mounted) setState(() => _modelPresent = present);
    });
  }

  Future<void> _prepareModel() async {
    final present = await GemmaService.instance.isModelPresent();
    setState(() {
      _downloading = true;
      _modelPresent = present;
      _downloadPct = present ? 100 : 0;
      _status = present
          ? 'Loading model from device…'
          : 'Downloading Gemma 3 1B (one time, ~550 MB)…';
    });
    try {
      await GemmaService.instance.ensureReady(
        onProgress: (p) => setState(() => _downloadPct = p),
      );
      setState(() => _status = 'Model ready — tap the mic to start.');
    } catch (e) {
      setState(() => _status = 'Model load failed: $e');
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
    // Keep the Android voice-communication route global for the capture/playback
    // pair, and also apply it to each fresh exact clip player before play.
    await _ttsPlayer.configureGlobalRoute();
    unawaited(TtsService.instance.ensureReady()); // warm the TTS worker isolate

    // The recognizer lives on a worker isolate; wire its callbacks and start it.
    AsrService.instance.onPartial = _onPartial;
    AsrService.instance.onEndpoint = _onEndpoint;
    await AsrService.instance.ensureReady();
    AsrService.instance.reset(); // fresh recognizer stream for this session
    _quietGate.resetAsr();

    // voiceCommunication + echoCancel let the mic stay open during playback
    // without the recognizer transcribing the assistant's own TTS. The config is
    // a regression-tested factory (see audio_capture_config.dart) so a revert to
    // AudioSource.mic -- which silently drops the OS AEC/NS/AGC -- fails CI.
    final audioStream = await _recorder.startStream(buildCaptureConfig());
    setState(() {
      _listening = true;
      _status = 'Listening…';
      _promptController.clear();
    });

    // Forward mic bytes to the worker (no decoding here) and watch loudness for
    // energy barge-in while the assistant speaks.
    _audioSub = audioStream.listen(_onMicChunk);
  }

  // Live partial from the ASR worker. The worker emits only on change, so this
  // is where "last voice" advances (drives the silence-wait metric) and where
  // barge-in fires.
  void _onPartial(String partial) {
    if (_disposing || partial.isEmpty) return;
    _tLastVoice = DateTime.now();
    // TODO(recovered): ASR isolate has no explicit speech-start callback; a
    // non-empty partial is the earliest reliable signal that an utterance is
    // in flight, so calibration stays blocked until the endpoint callback.
    _quietGate.noteAsrStarted(_tLastVoice!);
    // Barge-in via transcription (complements the energy path): the user talking
    // — or saying a stop word — while the assistant speaks cuts it off.
    if (_speaking) {
      _spkPartials++;
      if (partial.length >= _bargeInChars || _looksLikeStop(partial)) {
        _appendBargeLog('partial "$partial"');
        unawaited(_stopSpeaking());
      }
    }
    if (mounted) {
      _promptController.value = TextEditingValue(
        text: partial,
        selection: TextSelection.collapsed(offset: partial.length),
      );
    }
  }

  // A finished utterance from the ASR worker.
  void _onEndpoint(String utterance) {
    if (_disposing) return;
    _quietGate.noteAsrFinished(DateTime.now(), hadSpeech: utterance.isNotEmpty);
    if (utterance.isEmpty) return;
    // A completed utterance supersedes any in-flight reply (its tokens stop
    // feeding TTS) and silences whatever is still playing.
    _turn++;
    final playbackGeneration = _ttsPlayback.supersede();
    if (isStopCommand(utterance) || _looksLikeStop(utterance)) {
      if (mounted) {
        setState(() {
          _thinking = false;
          _status = 'Stopped.';
          _promptController.clear();
        });
      }
      return;
    }
    _tHeard = DateTime.now();
    unawaited(
      _answerUtterance(utterance, _turn, playbackGeneration),
    );
  }

  Future<void> _stopListening() async {
    await _audioSub?.cancel();
    _audioSub = null;
    await _recorder.stop();
    _quietGate.resetAsr();
    await _stopSpeaking();
    if (mounted) setState(() => _listening = false);
  }

  // Mic bytes (PCM16) on the main isolate: forward to the ASR worker, and while
  // the assistant is speaking, watch near-end loudness so the user can cut it
  // off by talking even when the words aren't cleanly transcribed (energy
  // barge-in, independent of the recognizer).
  void _onMicChunk(Uint8List chunk) {
    if (_disposing) return;
    AsrService.instance.feed(chunk);
    if (!_speaking) {
      // Genuine-idle window: learn the room's ambient floor only after recent
      // user speech and playback tails have cooled down. `_speaking == false`
      // by itself is not enough: it is also true while ASR is endpointing the
      // user's own request and immediately after the player stops.
      if (_quietGate.canObserveQuiet(DateTime.now())) {
        _bargeCal.observeQuiet(_rms(chunk));
      }
      _loudMs = 0;
      return;
    }
    _spkChunks++;
    final rms = _rms(chunk);
    if (rms > _spkPeakRms) _spkPeakRms = rms;
    if (rms >= _bargeCal.threshold) {
      _loudMs += 1000 * (chunk.length ~/ 2) ~/ 16000;
      if (_loudMs >= _bargeInMs) {
        _appendBargeLog('energy ${rms.toStringAsFixed(3)}');
        unawaited(_stopSpeaking());
      }
    } else {
      _loudMs = 0;
    }
  }

  // Root-mean-square loudness of a PCM16 chunk, normalized to 0..1.
  double _rms(Uint8List bytes) {
    final n = bytes.length ~/ 2;
    if (n == 0) return 0;
    final data = ByteData.sublistView(bytes);
    var sumSq = 0.0;
    for (var i = 0; i + 1 < bytes.length; i += 2) {
      final s = data.getInt16(i, Endian.little) / 32768.0;
      sumSq += s * s;
    }
    return math.sqrt(sumSq / n);
  }

  // Lenient, mobile-local stop check for cutting off playback — broader than the
  // shared exact-match contract (which mobile and Python pin via golden tests),
  // so "stop", "stop speaking", "be quiet", etc. all interrupt.
  bool _looksLikeStop(String text) {
    final t = text.toLowerCase();
    return t.contains('stop') || t.contains('quiet') || t.contains('cancel');
  }

  // --- generation ---

  // Generate a reply for [prompt] and stream it to TTS. [myTurn] guards against
  // a newer utterance arriving (barge-in): once the turn advances, this reply
  // stops emitting tokens and queuing speech.
  Future<void> _answerUtterance(
    String prompt,
    int myTurn,
    TtsPlaybackGeneration playbackGeneration,
  ) async {
    if (prompt.isEmpty ||
        !GemmaService.instance.isReady ||
        !_replyIsCurrent(myTurn, playbackGeneration)) {
      return;
    }
    _lastUtterance = prompt;
    setState(() {
      _thinking = true;
      _answer = '';
      _status = 'Thinking…';
    });
    var ttsBuffer = '';
    _tFirstToken = null;
    _tFirstAudio = null;
    _tGenDone = null;
    _genTokens = 0;
    _logIndex = null; // a new turn gets a fresh log entry
    try {
      await for (final token in GemmaService.instance.reply(prompt)) {
        if (!_replyIsCurrent(myTurn, playbackGeneration)) return;
        if (_tFirstToken == null) {
          _tFirstToken = DateTime.now();
          _updateMetrics();
        }
        _genTokens++;
        setState(() => _answer += token);
        ttsBuffer += token;
        ttsBuffer = _flushSentences(
          ttsBuffer,
          playbackGeneration,
        );
      }
      if (_replyIsCurrent(myTurn, playbackGeneration)) {
        _tGenDone = DateTime.now();
        _flushSentences(
          ttsBuffer,
          playbackGeneration,
          flushAll: true,
        );
        _updateMetrics();
      }
    } catch (e) {
      if (_replyIsCurrent(myTurn, playbackGeneration)) {
        setState(() => _answer = 'Generation failed: $e');
        _appendLog(error: '$e');
      }
    } finally {
      if (_replyIsCurrent(myTurn, playbackGeneration)) {
        setState(() => _thinking = false);
      }
    }
  }

  bool _replyIsCurrent(
    int myTurn,
    TtsPlaybackGeneration playbackGeneration,
  ) =>
      !_disposing &&
      mounted &&
      myTurn == _turn &&
      _ttsPlayback.isCurrent(playbackGeneration);

  // Send whatever is typed in the field (the manual fallback to voice).
  Future<void> _submitTyped() async {
    if (_disposing || !mounted) return;
    final text = _promptController.text.trim();
    if (text.isEmpty) return;
    _promptController.clear();
    _turn++;
    final myTurn = _turn;
    final playbackGeneration = _ttsPlayback.supersede();
    await _ttsPlayback.waitForStop(playbackGeneration);
    if (!_replyIsCurrent(myTurn, playbackGeneration)) return;
    if (isStopCommand(text)) {
      if (mounted) {
        setState(() {
          _thinking = false;
          _status = 'Stopped.';
        });
      }
      return;
    }
    _tLastVoice = null; // typed input has no silence-wait stage
    _tHeard = DateTime.now();
    await _answerUtterance(text, myTurn, playbackGeneration);
  }

  // Cut all speech now: drop the queue, stop the current clip, and release any
  // coroutine awaiting playback. Idempotent.
  Future<bool> _stopSpeaking() {
    final existing = _speechStopInFlight;
    if (existing != null &&
        identical(_speechStopGeneration, _ttsPlayback.generation)) {
      return existing;
    }

    // Barge-in also revokes the reply stream. Without this turn fence, the same
    // LLM coroutine can enqueue later sentences after its current clip stops.
    _turn++;
    if (mounted && !_disposing) {
      setState(() => _thinking = false);
    }
    final stopped = _ttsPlayback.interrupt();
    final stopGeneration = _ttsPlayback.generation;
    _speechStopInFlight = stopped;
    _speechStopGeneration = stopGeneration;
    unawaited(stopped.whenComplete(() {
      if (identical(_speechStopInFlight, stopped) &&
          identical(_speechStopGeneration, stopGeneration)) {
        _speechStopInFlight = null;
        _speechStopGeneration = null;
      }
    }));
    return stopped;
  }

  // --- streaming TTS ---

  // Pull every completed sentence out of the rolling buffer and queue it for
  // speech; with flushAll, also speak whatever remains at end of generation.
  // Sentence boundaries follow the shared contract (contract.dart) so mobile and
  // the Python core split identically.
  String _flushSentences(
    String buffer,
    TtsPlaybackGeneration playbackGeneration, {
    bool flushAll = false,
  }) {
    final (sentences, rest) = drainCompleteSentences(buffer);
    var remaining = rest;
    for (final sentence in sentences) {
      _ttsPlayback.enqueue(playbackGeneration, sentence);
    }
    if (flushAll) {
      final tail = remaining.trim();
      remaining = '';
      if (tail.isNotEmpty) {
        _ttsPlayback.enqueue(playbackGeneration, tail);
      }
    }
    return remaining;
  }

  void _onSpeechActivityChanged(
    TtsPlaybackGeneration generation,
    bool speaking,
  ) {
    if (!_ttsPlayback.isCurrent(generation)) return;
    final wasSpeaking = _speaking;
    _speaking = speaking;
    if (speaking) {
      if (!wasSpeaking) {
        _spkChunks = 0;
        _spkPeakRms = 0;
        _spkPartials = 0;
        _loudMs = 0;
      }
      if (mounted && !_disposing) {
        setState(() => _status = 'Speaking…');
      }
      return;
    }
    if (!wasSpeaking) return;

    _quietGate.notePlaybackStopped(DateTime.now());
    if (_disposing) return;
    // Window summary: if chunks==0 the mic was starved during playback; if
    // peakRms stayed below _bargeInRms, near-end speech isn't reaching us.
    _logLine('spoke: chunks=$_spkChunks '
        'peakRms=${_spkPeakRms.toStringAsFixed(3)} partials=$_spkPartials');
  }

  void _onPlaybackStarted(TtsPlaybackGeneration generation) {
    if (!_ttsPlayback.isCurrent(generation) || _disposing) return;
    if (_tFirstAudio == null) {
      _tFirstAudio = DateTime.now();
      _updateMetrics();
    }
  }

  void _onSpeechPlaybackError(
    TtsPlaybackGeneration generation,
    Object error,
    StackTrace stackTrace,
  ) {
    if (_disposing) return;
    _logLine(
      'speech playback error (generation ${generation.ordinal}): $error',
    );
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
    final text = lines.join('\n');
    if (mounted) setState(() => _metrics = text);
    _logTurn(text);
  }

  // Keep the current turn's log entry in sync with the latest metrics, so the
  // exported log includes the speaking + gen lines that land after first token.
  void _logTurn(String metrics) {
    if (_tHeard == null) return;
    final entry =
        '[${_tHeard!.toIso8601String()}] "${_lastUtterance ?? ''}"\n$metrics';
    if (_logIndex == null || _logIndex! < 0 || _logIndex! >= _log.length) {
      _log.add(entry);
      if (_log.length > 100) _log.removeRange(0, _log.length - 100);
      _logIndex = _log.length - 1;
    } else {
      _log[_logIndex!] = entry;
    }
  }

  void _appendLog({required String error}) {
    final ts = (_tHeard ?? DateTime.now()).toIso8601String();
    _log.add('[$ts] "${_lastUtterance ?? ''}" ERROR: $error');
    if (_log.length > 100) _log.removeRange(0, _log.length - 100);
    if (mounted) setState(() {});
  }

  // Append a free-standing diagnostic line (barge-in events, per-window near-end
  // loudness) to the exported log — separate from the per-turn metrics entries.
  void _logLine(String s) {
    _log.add('[${DateTime.now().toIso8601String()}] $s');
    if (_log.length > 200) _log.removeRange(0, _log.length - 200);
    if (mounted) setState(() {});
  }

  void _appendBargeLog(String reason) => _logLine('BARGE-IN ($reason)');

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
    _disposing = true;
    _turn++;
    _thinking = false;
    _alwaysOn = false;
    _audioSub?.cancel();
    _recorder.dispose();
    unawaited(AsrService.instance.stop());
    unawaited(TtsService.instance.dispose());
    unawaited(() async {
      await _ttsPlayback.close();
      await _ttsPlayer.close();
    }());
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
                  // No byte-accurate percent when loading an on-disk model, so
                  // show an indeterminate bar in that case.
                  LinearProgressIndicator(
                      value: _modelPresent ? null : _downloadPct / 100.0),
                  if (!_modelPresent) ...[
                    const SizedBox(height: 8),
                    Text('${_downloadPct.toStringAsFixed(0)}%'),
                  ],
                ],
              )
            else
              FilledButton.icon(
                onPressed: _prepareModel,
                icon: Icon(_modelPresent ? Icons.play_arrow : Icons.download),
                label: Text(
                    _modelPresent ? 'Load model' : 'Download model (one time)'),
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
