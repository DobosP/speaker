// Assistant screen: the full on-device loop, always-on with barge-in.
// Tap the mic once to start continuous listening: the streaming recognizer shows
// words live, and when it detects the end of an utterance (endpoint) the phrase
// is sent to Gemma automatically and the reply is spoken. The mic stays LIVE
// while the assistant talks — with acoustic echo cancellation so it doesn't hear
// its own voice — so you can interrupt (barge-in) just by talking, or say "stop".
// Tap again to stop. A per-turn latency readout + "Copy logs" export are wired in
// for on-device profiling.
import 'dart:async';
import 'dart:convert' show utf8;
import 'dart:io' show Platform;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:record/record.dart';

import './asr_isolate.dart';
import './assistant_listening_owner.dart';
import './assistant_reply_owner.dart';
import './audio_capture_config.dart';
import './barge_calibrator.dart';
import './contract.dart';
import './llm.dart';
import './llm_generation_owner.dart';
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

bool _assistantPromptFitsBound(String text) {
  if (text.length > assistantReplyPromptMaximumUtf8Bytes) {
    return false;
  }
  return utf8.encode(text).length <= assistantReplyPromptMaximumUtf8Bytes;
}

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

typedef _AssistantListeningGeneration
    = AssistantListeningGeneration<AsrSession, _AssistantListeningCapture>;

/// One exact recorder stream and its immutable ASR authority.
///
/// The lane retains no transcript or widget callback. Cancellation, recorder
/// stop, and ambiguous-start recovery are prepublished and memoized before
/// invoking plugin code so reentrant cleanup always joins the same operation.
final class _AssistantListeningCapture {
  _AssistantListeningCapture({
    required this.generation,
    required this.session,
    required AudioRecorder recorder,
  }) : _recorder = recorder;

  final _AssistantListeningGeneration generation;
  final AsrSession session;
  final AudioRecorder _recorder;
  final Completer<AssistantListeningCaptureTerminal> _terminal =
      Completer<AssistantListeningCaptureTerminal>();
  final Completer<StreamSubscription<Uint8List>?> _subscription =
      Completer<StreamSubscription<Uint8List>?>();

  Future<bool>? _cancelFuture;
  Future<bool>? _stopFuture;
  Future<bool>? _recoveryFuture;
  bool _subscriptionUncertain = false;
  bool _failureRequested = false;
  bool _cancelSucceeded = false;
  bool _stopSucceeded = false;
  bool _retirementWatchInstalled = false;

  Future<AssistantListeningCaptureTerminal> get terminal => _terminal.future;
  Future<bool>? get cancelFuture => _cancelFuture;
  Future<bool>? get stopFuture => _stopFuture;
  bool get exactCleanupSucceeded => _cancelSucceeded && _stopSucceeded;

  void publishSubscription(StreamSubscription<Uint8List> subscription) {
    if (_subscription.isCompleted) return;
    _subscription.complete(subscription);
  }

  void publishNoSubscription({required bool exact}) {
    if (_subscription.isCompleted) return;
    _subscriptionUncertain = !exact;
    _subscription.complete(null);
  }

  void publishEnded() {
    if (_failureRequested || _terminal.isCompleted) return;
    _terminal.complete(AssistantListeningCaptureTerminal.ended);
  }

  void failAfterExactCancellation() {
    if (_terminal.isCompleted || _failureRequested) return;
    _failureRequested = true;
    final cancellation = cancelSource();
    unawaited(cancellation.then<void>((succeeded) {
      if (succeeded && !_terminal.isCompleted) {
        _terminal.complete(AssistantListeningCaptureTerminal.failed);
      }
    }));
  }

  Future<bool> cancelSource() {
    final existing = _cancelFuture;
    if (existing != null) return existing;
    final completer = Completer<bool>();
    final result = completer.future;
    _cancelFuture = result;

    unawaited(
      _subscription.future.then<void>((subscription) {
        if (subscription == null) {
          completer.complete(!_subscriptionUncertain);
          return;
        }
        Future<void>.sync(subscription.cancel).then<void>(
          (_) => completer.complete(true),
          onError: (_error, _stackTrace) => completer.complete(false),
        );
      }, onError: (_error, _stackTrace) {
        completer.complete(false);
      }),
    );
    unawaited(result.then<void>((succeeded) {
      _cancelSucceeded = succeeded;
    }));
    return result;
  }

  Future<bool> stopRecorder() {
    final existing = _stopFuture;
    if (existing != null) return existing;
    final completer = Completer<bool>();
    final result = completer.future;
    _stopFuture = result;

    try {
      _recorder.stop().then<void>(
            (_) => completer.complete(true),
            onError: (_error, _stackTrace) => completer.complete(false),
          );
    } catch (_) {
      completer.complete(false);
    }
    unawaited(result.then<void>((succeeded) {
      _stopSucceeded = succeeded;
    }));
    return result;
  }

  Future<bool> recoverAmbiguousStart() {
    final existing = _recoveryFuture;
    if (existing != null) return existing;
    final completer = Completer<bool>();
    final result = completer.future;
    _recoveryFuture = result;

    final cancellation = cancelSource();
    final stop = stopRecorder();
    Future.wait<bool>(<Future<bool>>[cancellation, stop]).then<void>(
      (receipts) => completer.complete(receipts.every((value) => value)),
      onError: (_error, _stackTrace) => completer.complete(false),
    );
    return result;
  }
}

/// Plugin adapter for one Assistant listening owner.
///
/// It owns the recorder lane, borrows the widget's playback owner only to
/// coordinate its audio route, and keeps only a weak reference back to widget
/// State. Every plugin or ASR callback closes over its immutable generation,
/// session, and capture lane; no callback consults a mutable current session or
/// stores transcript text here.
final class _AssistantListeningPluginLifecycle
    implements
        AssistantListeningLifecycle<AsrSession, _AssistantListeningCapture> {
  _AssistantListeningPluginLifecycle({
    required AudioRecorder recorder,
    required _AssistantTtsPlayer ttsPlayer,
    required AsrService asrService,
    required WeakReference<_AssistantScreenState> state,
  })  : _recorder = recorder,
        _ttsPlayer = ttsPlayer,
        _asr = asrService,
        _state = state;

  final AudioRecorder _recorder;
  final _AssistantTtsPlayer _ttsPlayer;
  final AsrService _asr;
  final WeakReference<_AssistantScreenState> _state;
  final Map<_AssistantListeningGeneration, AsrSession> _sessions =
      Map<_AssistantListeningGeneration, AsrSession>.identity();
  final Map<_AssistantListeningGeneration, _AssistantListeningCapture>
      _captureLanes =
      Map<_AssistantListeningGeneration, _AssistantListeningCapture>.identity();
  Future<bool>? _recorderDisposeFuture;

  bool ownsExactSession(
    _AssistantListeningGeneration generation,
    AsrSession session,
  ) =>
      identical(_sessions[generation], session);

  @override
  Future<bool> requestPermission(_AssistantListeningGeneration generation) =>
      _recorder.hasPermission();

  @override
  Future<bool> configureRoute(
    _AssistantListeningGeneration generation,
  ) async {
    await _ttsPlayer.configureGlobalRoute();
    return true;
  }

  @override
  AsrSession beginSession(_AssistantListeningGeneration generation) {
    late final AsrSession exactSession;
    exactSession = _asr.beginSession(
      onPartial: (partial) {
        _state.target?._onListeningPartial(
          generation,
          exactSession,
          partial,
        );
      },
      onEndpoint: (utterance) {
        _state.target?._onListeningEndpoint(
          generation,
          exactSession,
          utterance,
        );
      },
    );
    _sessions[generation] = exactSession;
    return exactSession;
  }

  @override
  Future<bool> waitSessionEnded(AsrSession session) async {
    try {
      return await session.cleanup;
    } finally {
      _sessions.removeWhere(
        (_generation, candidate) => identical(candidate, session),
      );
    }
  }

  @override
  Future<void> waitSessionReady(AsrSession session) => session.ready;

  @override
  Future<AssistantListeningCaptureStartResult<_AssistantListeningCapture>>
      startCapture(
    _AssistantListeningGeneration generation,
    AsrSession session,
  ) async {
    final capture = _AssistantListeningCapture(
      generation: generation,
      session: session,
      recorder: _recorder,
    );
    // Publish the exact recovery lane before native recorder admission.
    _captureLanes[generation] = capture;

    late final Stream<Uint8List> audioStream;
    try {
      audioStream = await _recorder.startStream(buildCaptureConfig());
    } catch (_) {
      capture.publishNoSubscription(exact: true);
      rethrow;
    }

    final exactGeneration = generation;
    final exactSession = session;
    try {
      final subscription = audioStream.listen(
        (chunk) => _onCaptureData(
          capture,
          exactGeneration,
          exactSession,
          chunk,
        ),
        onError: (_error, _stackTrace) => _onCaptureError(
          capture,
          exactGeneration,
          exactSession,
        ),
        onDone: capture.publishEnded,
        cancelOnError: false,
      );
      capture.publishSubscription(subscription);
    } catch (_) {
      // A throwing listen call did not return an exact cancellation token.
      capture.publishNoSubscription(exact: false);
      rethrow;
    }
    return AssistantListeningCaptureStartResult<
        _AssistantListeningCapture>.started(capture);
  }

  void _onCaptureData(
    _AssistantListeningCapture capture,
    _AssistantListeningGeneration exactGeneration,
    AsrSession exactSession,
    Uint8List chunk,
  ) {
    if (!_isExactCapture(capture, exactGeneration, exactSession)) return;
    final state = _state.target;
    if (state == null ||
        !state._listeningCallbackIsCurrent(
          exactGeneration,
          exactSession,
        )) {
      return;
    }
    if (!_asr.feed(exactSession, chunk)) {
      capture.failAfterExactCancellation();
      state._fenceListeningSource(exactGeneration, exactSession);
      return;
    }
    state._onListeningMicChunk(exactGeneration, exactSession, chunk);
  }

  void _onCaptureError(
    _AssistantListeningCapture capture,
    _AssistantListeningGeneration exactGeneration,
    AsrSession exactSession,
  ) {
    if (!_isExactCapture(capture, exactGeneration, exactSession)) return;
    capture.failAfterExactCancellation();
    _state.target?._fenceListeningSource(exactGeneration, exactSession);
  }

  bool _isExactCapture(
    _AssistantListeningCapture capture,
    _AssistantListeningGeneration generation,
    AsrSession session,
  ) =>
      identical(capture.generation, generation) &&
      identical(capture.session, session) &&
      identical(_captureLanes[generation], capture) &&
      ownsExactSession(generation, session);

  @override
  Future<AssistantListeningCaptureTerminal> waitCaptureTerminal(
    _AssistantListeningCapture capture,
  ) =>
      capture.terminal;

  @override
  Future<bool> cancelCaptureSource(_AssistantListeningCapture capture) {
    final result = capture.cancelSource();
    _retireCaptureWhenExact(capture);
    return result;
  }

  @override
  Future<bool> stopCapture(_AssistantListeningCapture capture) {
    final result = capture.stopRecorder();
    _retireCaptureWhenExact(capture);
    return result;
  }

  @override
  Future<bool> recoverAmbiguousCaptureStart(
    _AssistantListeningGeneration generation,
    AsrSession session,
  ) {
    final capture = _captureLanes[generation];
    if (capture == null || !identical(capture.session, session)) {
      return Future<bool>.value(false);
    }
    final result = capture.recoverAmbiguousStart();
    _retireCaptureWhenExact(capture);
    return result;
  }

  void _retireCaptureWhenExact(_AssistantListeningCapture capture) {
    final cancellation = capture.cancelFuture;
    final stop = capture.stopFuture;
    if (cancellation == null ||
        stop == null ||
        capture._retirementWatchInstalled) {
      return;
    }
    capture._retirementWatchInstalled = true;
    unawaited(Future.wait<bool>(<Future<bool>>[cancellation, stop]).then<void>(
      (_) {
        if (capture.exactCleanupSucceeded &&
            identical(_captureLanes[capture.generation], capture)) {
          _captureLanes.remove(capture.generation);
        }
      },
    ));
  }

  @override
  bool endSession(AsrSession session) => _asr.endSession(session);

  /// Dispose the recorder only behind the exact owner's successful close.
  /// A false receipt deliberately retains the plugin resource for diagnosis.
  Future<bool> disposeRecorderAfterExactClose(
    AssistantListeningCloseReceipt receipt,
  ) {
    if (!receipt.exactResourcesSettled) return Future<bool>.value(false);
    final existing = _recorderDisposeFuture;
    if (existing != null) return existing;
    final completer = Completer<bool>();
    final result = completer.future;
    _recorderDisposeFuture = result;
    try {
      _recorder.dispose().then<void>(
            (_) => completer.complete(true),
            onError: (_error, _stackTrace) => completer.complete(false),
          );
    } catch (_) {
      completer.complete(false);
    }
    return result;
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
  late final _AssistantListeningPluginLifecycle _listeningLifecycle;
  late final AssistantListeningOwner<AsrSession, _AssistantListeningCapture>
      _listeningOwner;
  _AssistantListeningGeneration? _listeningGeneration;
  late final AssistantReplyOwner _replyOwner;
  AssistantReplyGeneration? _replyGeneration;
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

  // Always-on voice input. The exact listening owner serializes one active plus
  // one latest pending desired state; `_alwaysOn` is that latest UI desire,
  // while `_listening` means its exact recorder lane reached admission.
  bool _alwaysOn = false;
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
    final state = WeakReference<_AssistantScreenState>(this);
    _listeningLifecycle = _AssistantListeningPluginLifecycle(
      recorder: AudioRecorder(),
      ttsPlayer: _ttsPlayer,
      asrService: AsrService.instance,
      state: state,
    );
    _listeningOwner =
        AssistantListeningOwner<AsrSession, _AssistantListeningCapture>(
      lifecycle: _listeningLifecycle,
      onRevoke: (generation, outcome) {
        state.target?._onListeningRevoked(generation, outcome);
      },
      onListening: (generation) {
        state.target?._onListeningStarted(generation);
      },
      onStopped: (generation) {
        state.target?._onListeningStopped(generation);
      },
    );
    _replyOwner = AssistantReplyOwner(
      openReply: GemmaService.instance.reply,
    );
    _ttsPlayback = TtsPlaybackOwner(
      synthesize: (text) async {
        final filename = await generateWaveFilename();
        return TtsService.instance.synthesize(text, filename);
      },
      createPlaybackClip: _ttsPlayer.createClip,
      onActivityChanged: (generation, speaking) {
        state.target?._onSpeechActivityChanged(generation, speaking);
      },
      onPlaybackStarted: (generation) {
        state.target?._onPlaybackStarted(generation);
      },
      onError: (generation, error, stackTrace) {
        state.target?._onSpeechPlaybackError(
          generation,
          error,
          stackTrace,
        );
      },
    );
    // Probe disk so the button/status reflect whether a network download is
    // actually needed (the model persists across reinstalls).
    GemmaService.instance.isModelPresent().then<void>(
      (present) {
        final target = state.target;
        if (target == null || target._disposing || !target.mounted) return;
        target.setState(() => target._modelPresent = present);
      },
      onError: (_error, _stackTrace) {},
    );
  }

  Future<void> _prepareModel() async {
    try {
      final present = await GemmaService.instance.isModelPresent();
      if (_disposing || !mounted) return;
      setState(() {
        _downloading = true;
        _modelPresent = present;
        _downloadPct = present ? 100 : 0;
        _status = present
            ? 'Loading model from device…'
            : 'Downloading Gemma 3 1B (one time, ~550 MB)…';
      });
      await GemmaService.instance.ensureReady(
        onProgress: (p) {
          if (_disposing || !mounted) return;
          setState(() => _downloadPct = p);
        },
      );
      if (_disposing || !mounted) return;
      setState(() => _status = 'Model ready — tap the mic to start.');
    } on GemmaGenerationFailure catch (failure) {
      if (_disposing || !mounted) return;
      setState(() => _status = 'Model load failed: ${failure.code}');
    } catch (_) {
      if (_disposing || !mounted) return;
      setState(() => _status = 'Model load failed: model_prepare_failed');
    } finally {
      if (!_disposing && mounted) setState(() => _downloading = false);
    }
  }

  // --- always-on listening ---

  Future<void> _toggleAlwaysOn() async {
    if (_disposing || !mounted) return;
    final enable = !_alwaysOn;
    if (enable && !GemmaService.instance.isReady) return;

    // A listening shutdown fences reply/playback and the prior UI generation
    // synchronously. Owner replacement then fences the exact recorder/ASR lane;
    // every plugin wait begins only after those authority changes.
    final stopped = enable ? null : _stopSpeaking();
    _listeningGeneration = null;
    setState(() {
      _alwaysOn = enable;
      _status = enable ? 'Starting listener…' : 'Stopping…';
    });

    late final _AssistantListeningGeneration generation;
    try {
      generation =
          enable ? _listeningOwner.enable() : _listeningOwner.disable();
      _listeningGeneration = generation;
    } on AssistantListeningFailure catch (failure) {
      if (mounted && !_disposing) {
        setState(() {
          _alwaysOn = false;
          _listening = false;
          _status = 'Listening unavailable: ${failure.code}';
        });
      }
      if (stopped != null) await stopped;
      return;
    } catch (_) {
      if (mounted && !_disposing) {
        setState(() {
          _alwaysOn = false;
          _listening = false;
          _status = 'Listening unavailable: listening_integration_failed';
        });
      }
      if (stopped != null) await stopped;
      return;
    }

    final done = await generation.done;
    if (stopped != null) await stopped;
    if (!_listeningUiGenerationIsCurrent(generation)) return;
    final failure = done.failure;
    if (failure != null) {
      setState(() {
        _alwaysOn = false;
        _listening = false;
        _status = 'Listening unavailable: ${failure.code}';
      });
    }
  }

  bool _listeningUiGenerationIsCurrent(
    _AssistantListeningGeneration generation,
  ) =>
      !_disposing && mounted && identical(_listeningGeneration, generation);

  bool _listeningCallbackIsCurrent(
    _AssistantListeningGeneration generation,
    AsrSession session,
  ) =>
      _listeningUiGenerationIsCurrent(generation) &&
      _listeningOwner.isAuthoritative(generation) &&
      _listeningLifecycle.ownsExactSession(generation, session);

  void _onListeningStarted(_AssistantListeningGeneration generation) {
    if (!_listeningUiGenerationIsCurrent(generation) ||
        !_listeningOwner.isAuthoritative(generation)) {
      return;
    }
    // This is app-global best-effort warming, not a resource owned or disposed
    // by the widget-local listening lifecycle.
    unawaited(
      TtsService.instance.ensureReady().then<void>(
            (_) {},
            onError: (_error, _stackTrace) {},
          ),
    );
    _quietGate.resetAsr();
    setState(() {
      _alwaysOn = true;
      _listening = true;
      _status = 'Listening…';
      _promptController.clear();
    });
  }

  void _onListeningRevoked(
    _AssistantListeningGeneration generation,
    AssistantListeningOutcome outcome,
  ) {
    // Owner serialization guarantees this revoke precedes successor admission.
    // Fence reply/playback even when UI identity was already cleared by the
    // replacing call, then gate only the state mutation below.
    unawaited(_stopSpeaking());
    if (!_listeningUiGenerationIsCurrent(generation)) return;
    _listeningGeneration = null;
    _quietGate.resetAsr();
    final status = switch (outcome) {
      AssistantListeningOutcome.permissionDenied =>
        'Microphone permission denied — enable it in system settings.',
      AssistantListeningOutcome.captureEnded => 'Microphone stream ended.',
      AssistantListeningOutcome.cancelled ||
      AssistantListeningOutcome.superseded ||
      AssistantListeningOutcome.ownerClosed =>
        'Stopped.',
      _ => 'Listening stopped: ${outcome.name}',
    };
    setState(() {
      _alwaysOn = false;
      _listening = false;
      _status = status;
    });
  }

  void _onListeningStopped(_AssistantListeningGeneration generation) {
    if (!_listeningUiGenerationIsCurrent(generation)) return;
    _quietGate.resetAsr();
    setState(() {
      _listening = false;
      if (generation.intent == AssistantListeningIntent.off) {
        _alwaysOn = false;
        _status = 'Stopped.';
      }
    });
  }

  void _fenceListeningSource(
    _AssistantListeningGeneration generation,
    AsrSession session,
  ) {
    if (!_listeningCallbackIsCurrent(generation, session)) return;
    _listeningGeneration = null;
    _alwaysOn = false;
    _listening = false;
    _quietGate.resetAsr();
    unawaited(_listeningOwner.revokeExact(generation));
    if (mounted && !_disposing) {
      setState(() => _status = 'Listening stopped: capture_source_failed');
    }
  }

  // Live partial from the ASR worker. The worker emits only on change, so this
  // is where "last voice" advances (drives the silence-wait metric) and where
  // barge-in fires.
  void _onListeningPartial(
    _AssistantListeningGeneration generation,
    AsrSession session,
    String partial,
  ) {
    if (!_listeningCallbackIsCurrent(generation, session) || partial.isEmpty) {
      return;
    }
    _tLastVoice = DateTime.now();
    // TODO(recovered): ASR isolate has no explicit speech-start callback; a
    // non-empty partial is the earliest reliable signal that an utterance is
    // in flight, so calibration stays blocked until the endpoint callback.
    _quietGate.noteAsrStarted(_tLastVoice!);
    // Any nontrivial live partial remains the generic transcription barge path.
    // Control-command semantics apply only to completed/typed exact phrases.
    if (_speaking) {
      _spkPartials++;
      if (partial.length >= _bargeInChars) {
        _appendBargeLog('partial "$partial"');
        unawaited(_stopSpeaking());
      }
    }
    _promptController.value = TextEditingValue(
      text: partial,
      selection: TextSelection.collapsed(offset: partial.length),
    );
  }

  // A finished utterance from the ASR worker.
  void _onListeningEndpoint(
    _AssistantListeningGeneration generation,
    AsrSession session,
    String utterance,
  ) {
    if (!_listeningCallbackIsCurrent(generation, session)) return;
    _quietGate.noteAsrFinished(DateTime.now(), hadSpeech: utterance.isNotEmpty);
    if (utterance.isEmpty) return;
    // A completed utterance supersedes any in-flight reply (its tokens stop
    // feeding TTS) and silences whatever is still playing.
    final isStop = isStopCommand(utterance);
    _turn++;
    _replyGeneration = null;
    unawaited(
      _replyOwner.cancelCurrent(
        reason: isStop
            ? AssistantReplyCancelReason.cancelled
            : AssistantReplyCancelReason.superseded,
      ),
    );
    final playbackGeneration = _ttsPlayback.supersede();
    if (isStop) {
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

  // Exact-lane mic bytes have already been accepted by AsrService.feed. The UI
  // sees them only for energy barge-in and quiet-room calibration.
  void _onListeningMicChunk(
    _AssistantListeningGeneration generation,
    AsrSession session,
    Uint8List chunk,
  ) {
    if (!_listeningCallbackIsCurrent(generation, session)) return;
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

  // --- generation ---

  // Generate a reply for [prompt] and stream it to TTS. [myTurn] guards against
  // a newer utterance arriving (barge-in): once the turn advances, this reply
  // stops emitting tokens and queuing speech.
  Future<void> _answerUtterance(
    String prompt,
    int myTurn,
    TtsPlaybackGeneration playbackGeneration,
  ) async {
    if (!_assistantPromptFitsBound(prompt)) {
      if (!_disposing && mounted) {
        setState(() => _status = 'Input rejected: input_too_large');
      }
      return;
    }
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
    AssistantReplyGeneration? replyGeneration;
    var finishThinking = false;
    try {
      final admittedGeneration = _replyOwner.start(
        prompt: prompt,
        onToken: (callbackGeneration, token) {
          if (!identical(callbackGeneration, replyGeneration) ||
              !identical(_replyGeneration, callbackGeneration) ||
              !_replyOwner.isAuthoritative(callbackGeneration) ||
              !_replyIsCurrent(myTurn, playbackGeneration)) {
            return;
          }
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
        },
      );
      replyGeneration = admittedGeneration;
      _replyGeneration = admittedGeneration;
      final done = await admittedGeneration.done;
      final isExactReply = identical(_replyGeneration, admittedGeneration);
      final remainingTts = ttsBuffer;
      ttsBuffer = '';
      if (!isExactReply || !_replyIsCurrent(myTurn, playbackGeneration)) {
        return;
      }
      finishThinking = true;
      if (done.outcome == AssistantReplyOutcome.completed) {
        _tGenDone = DateTime.now();
        _flushSentences(
          remainingTts,
          playbackGeneration,
          flushAll: true,
        );
        _updateMetrics();
      } else if (done.failure case final failure?) {
        setState(() => _answer = 'Generation failed: ${failure.code}');
        _appendLog(error: failure.code);
      }
    } on AssistantReplyFailure catch (failure) {
      final mayMutate = replyGeneration == null
          ? _replyGeneration == null
          : identical(_replyGeneration, replyGeneration);
      if (mayMutate && _replyIsCurrent(myTurn, playbackGeneration)) {
        finishThinking = true;
        setState(() => _answer = 'Generation failed: ${failure.code}');
        _appendLog(error: failure.code);
      }
    } catch (_) {
      final mayMutate = replyGeneration == null
          ? _replyGeneration == null
          : identical(_replyGeneration, replyGeneration);
      if (mayMutate && _replyIsCurrent(myTurn, playbackGeneration)) {
        finishThinking = true;
        const code = 'reply_integration_failed';
        setState(() => _answer = 'Generation failed: $code');
        _appendLog(error: code);
      }
    } finally {
      final isExactReply = replyGeneration == null
          ? _replyGeneration == null
          : identical(_replyGeneration, replyGeneration);
      if (finishThinking &&
          isExactReply &&
          _replyIsCurrent(myTurn, playbackGeneration)) {
        setState(() => _thinking = false);
      }
      if (replyGeneration != null && isExactReply) {
        _replyGeneration = null;
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
    final rawText = _promptController.text;
    if (!_assistantPromptFitsBound(rawText)) {
      _promptController.clear();
      setState(() => _status = 'Input rejected: input_too_large');
      return;
    }
    final text = rawText.trim();
    if (text.isEmpty) return;
    _promptController.clear();
    final isStop = isStopCommand(text);
    _turn++;
    final myTurn = _turn;
    _replyGeneration = null;
    unawaited(
      _replyOwner.cancelCurrent(
        reason: isStop
            ? AssistantReplyCancelReason.cancelled
            : AssistantReplyCancelReason.superseded,
      ),
    );
    final playbackGeneration = _ttsPlayback.supersede();
    await _ttsPlayback.waitForStop(playbackGeneration);
    if (!_replyIsCurrent(myTurn, playbackGeneration)) return;
    if (isStop) {
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
    // Fence reply authority even when exact playback cleanup is already in
    // flight; playback deduplication must not bypass LLM cancellation.
    _turn++;
    _replyGeneration = null;
    unawaited(_replyOwner.cancelCurrent());
    if (mounted && !_disposing) {
      setState(() => _thinking = false);
    }

    final existing = _speechStopInFlight;
    if (existing != null &&
        identical(_speechStopGeneration, _ttsPlayback.generation)) {
      return existing;
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
    Object _error,
    StackTrace _stackTrace,
  ) {
    if (_disposing) return;
    _logLine(
      'speech playback failed (generation ${generation.ordinal}): '
      'speech_playback_failed',
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
      lines
          .add('gen          : $_genTokens tok @ ${rate.toStringAsFixed(1)}/s');
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
    _replyGeneration = null;
    _listeningGeneration = null;
    // Each owner synchronously revokes its callback authority before returning
    // a cleanup Future. Start every fence before any asynchronous wait.
    final replyClose = _replyOwner.close();
    final listeningClose = _listeningOwner.close();
    final playbackClose = _ttsPlayback.close();
    final listeningLifecycle = _listeningLifecycle;
    final ttsPlayer = _ttsPlayer;
    _thinking = false;
    _alwaysOn = false;
    _listening = false;
    unawaited(() async {
      await replyClose;
      final listeningReceipt = await listeningClose;
      await listeningLifecycle.disposeRecorderAfterExactClose(
        listeningReceipt,
      );
      await playbackClose;
      await ttsPlayer.close();
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
