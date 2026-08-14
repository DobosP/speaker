// Source-level guard for the plugin-bound Assistant widget seam.
//
// Pure owner behavior is covered by the reply/listening owner tests. This test
// keeps the widget and plugin adapter wired to those owners without loading a
// model, Flutter plugin, audio device, emulator, or phone.
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  late String source;

  setUpAll(() {
    source = File('lib/assistant.dart').readAsStringSync();
  });

  String between(String start, String end) {
    final startIndex = source.indexOf(start);
    final endIndex = source.indexOf(end, startIndex + start.length);
    expect(startIndex, isNonNegative, reason: 'missing start marker: $start');
    expect(endIndex, greaterThan(startIndex),
        reason: 'missing end marker: $end');
    return source.substring(startIndex, endIndex);
  }

  void expectOrdered(String body, List<String> markers) {
    var previous = -1;
    for (final marker in markers) {
      final current = body.indexOf(marker, previous + 1);
      expect(current, greaterThan(previous), reason: 'out of order: $marker');
      previous = current;
    }
  }

  test('Assistant owns Gemma through one exact reply subscription seam', () {
    expect(source, contains("import './assistant_reply_owner.dart';"));
    expect(source, contains('late final AssistantReplyOwner _replyOwner;'));
    expect(
      source,
      contains('AssistantReplyGeneration? _replyGeneration;'),
    );
    expect(
      source,
      contains('openReply: GemmaService.instance.reply,'),
    );
    expect(
      RegExp(r'GemmaService\.instance\.reply').allMatches(source).length,
      1,
    );
    expect(source, isNot(contains('GemmaService.instance.cancelCurrent')));
    expect(source, isNot(contains('await for (')));

    final answer = between(
      'Future<void> _answerUtterance(',
      'bool _replyIsCurrent(',
    );
    expect(answer, contains('_replyOwner.start('));
    expectOrdered(answer, [
      'final admittedGeneration = _replyOwner.start(',
      'replyGeneration = admittedGeneration;',
      '_replyGeneration = admittedGeneration;',
      'final done = await admittedGeneration.done;',
      'final isExactReply = identical(_replyGeneration, admittedGeneration);',
    ]);
    expect(
      answer,
      contains('identical(callbackGeneration, replyGeneration)'),
    );
    expect(
      answer,
      contains('identical(_replyGeneration, callbackGeneration)'),
    );
    expect(
      answer,
      contains('_replyOwner.isAuthoritative(callbackGeneration)'),
    );
    expect(answer, contains('_replyIsCurrent(myTurn, playbackGeneration)'));
    expectOrdered(answer, [
      'final remainingTts = ttsBuffer;',
      "ttsBuffer = '';",
      'done.outcome == AssistantReplyOutcome.completed',
      'remainingTts,',
      'flushAll: true',
      'setState(() => _thinking = false);',
      '_replyGeneration = null;',
    ]);
    expect(answer, contains("const code = 'reply_integration_failed';"));
    expect(answer, isNot(contains(r'$e')));
  });

  test('typed prompts are bounded before normalization or retention', () {
    expect(source, contains("import 'dart:convert' show utf8;"));
    final preflight = between(
      'bool _assistantPromptFitsBound(',
      'final class _AssistantTtsClipState',
    );
    expectOrdered(preflight, [
      'text.length > assistantReplyPromptMaximumUtf8Bytes',
      'utf8.encode(text).length <= assistantReplyPromptMaximumUtf8Bytes',
    ]);

    final typed = between(
      'Future<void> _submitTyped()',
      'Future<bool> _stopSpeaking()',
    );
    expectOrdered(typed, [
      'final rawText = _promptController.text;',
      'if (!_assistantPromptFitsBound(rawText))',
      '_promptController.clear();',
      'input_too_large',
      'return;',
      'final text = rawText.trim();',
      'isStopCommand(text)',
    ]);

    final answer = between(
      'Future<void> _answerUtterance(',
      'bool _replyIsCurrent(',
    );
    expectOrdered(answer, [
      'if (!_assistantPromptFitsBound(prompt))',
      'input_too_large',
      'return;',
      '_lastUtterance = prompt;',
    ]);
  });

  test('every replacement path fences reply before playback or waits', () {
    final endpoint = between(
      'void _onListeningEndpoint(',
      'void _onListeningMicChunk(',
    );
    expectOrdered(endpoint, [
      '_turn++;',
      '_replyGeneration = null;',
      '_replyOwner.cancelCurrent(',
      '_ttsPlayback.supersede()',
      'if (isStop)',
      '_answerUtterance(',
    ]);

    final listeningToggle = between(
      'Future<void> _toggleAlwaysOn()',
      'bool _listeningUiGenerationIsCurrent(',
    );
    expectOrdered(listeningToggle, [
      'final stopped = enable ? null : _stopSpeaking();',
      '_listeningGeneration = null;',
      '_listeningOwner.enable()',
      '_listeningGeneration = generation;',
      'final done = await generation.done;',
      'if (stopped != null) await stopped;',
    ]);

    final typed = between(
      'Future<void> _submitTyped()',
      'Future<bool> _stopSpeaking()',
    );
    expectOrdered(typed, [
      '_turn++;',
      '_replyGeneration = null;',
      '_replyOwner.cancelCurrent(',
      '_ttsPlayback.supersede()',
      '_ttsPlayback.waitForStop(playbackGeneration)',
      'if (isStop)',
      '_answerUtterance(',
    ]);

    final stop = between(
      'Future<bool> _stopSpeaking()',
      '// --- streaming TTS ---',
    );
    expectOrdered(stop, [
      '_turn++;',
      '_replyGeneration = null;',
      '_replyOwner.cancelCurrent()',
      'final existing = _speechStopInFlight;',
      '_ttsPlayback.interrupt()',
    ]);
  });

  test('listening adapter retains exact tokens and only weak widget state', () {
    expect(source, contains("import './assistant_listening_owner.dart';"));
    expect(
      source,
      contains(
        'AssistantListeningOwner<AsrSession, _AssistantListeningCapture>',
      ),
    );

    final adapter = between(
      'final class _AssistantListeningPluginLifecycle',
      'class AssistantScreen extends StatefulWidget',
    );
    expect(
      adapter,
      contains('final WeakReference<_AssistantScreenState> _state;'),
    );
    expect(adapter, isNot(matches(RegExp(r'final _AssistantScreenState\b'))));
    expect(adapter, isNot(contains('AsrTextCallback')));
    expect(adapter, isNot(contains('_promptController')));
    expect(adapter, isNot(contains('_answer')));

    final init = between(
      'void initState()',
      'Future<void> _prepareModel()',
    );
    expectOrdered(init, [
      'final state = WeakReference<_AssistantScreenState>(this);',
      '_listeningLifecycle = _AssistantListeningPluginLifecycle(',
      'recorder: AudioRecorder(),',
      'ttsPlayer: _ttsPlayer,',
      'asrService: AsrService.instance,',
      '_listeningOwner =',
      'lifecycle: _listeningLifecycle,',
    ]);
    expect(
      RegExp(r'state\.target\?\._onListening').allMatches(init).length,
      3,
    );
    expectOrdered(init, [
      '_ttsPlayback = TtsPlaybackOwner(',
      'onActivityChanged: (generation, speaking)',
      'state.target?._onSpeechActivityChanged(generation, speaking);',
      'onPlaybackStarted: (generation)',
      'state.target?._onPlaybackStarted(generation);',
      'onError: (generation, error, stackTrace)',
      'state.target?._onSpeechPlaybackError(',
    ]);
    for (final strongCallback in <String>[
      'onActivityChanged: _onSpeechActivityChanged',
      'onPlaybackStarted: _onPlaybackStarted',
      'onError: _onSpeechPlaybackError',
    ]) {
      expect(init, isNot(contains(strongCallback)));
    }

    expectOrdered(init, [
      'GemmaService.instance.isModelPresent().then<void>(',
      '(present)',
      'final target = state.target;',
      'if (target == null || target._disposing || !target.mounted) return;',
      'target.setState(() => target._modelPresent = present);',
      'onError: (_error, _stackTrace) {},',
    ]);

    final begin = between(
      'AsrSession beginSession(',
      'Future<bool> waitSessionEnded(',
    );
    expectOrdered(begin, [
      'late final AsrSession exactSession;',
      'exactSession = _asr.beginSession(',
      'onPartial: (partial)',
      'generation,',
      'exactSession,',
      'onEndpoint: (utterance)',
      '_sessions[generation] = exactSession;',
      'return exactSession;',
    ]);
    expect(adapter, contains('return await session.cleanup;'));
    expect(adapter, contains('session.ready;'));
    expect(adapter, contains('_asr.endSession(session)'));
    expect(adapter, isNot(contains('TtsService')));

    for (final banned in <String>[
      'AsrService.instance.onPartial',
      'AsrService.instance.onEndpoint',
      'AsrService.instance.ensureReady',
      'AsrService.instance.reset',
      'AsrService.instance.feed',
      'AsrService.instance.stop',
      'AsrService.instance.close',
    ]) {
      expect(source, isNot(contains(banned)),
          reason: 'legacy ASR API: $banned');
    }
  });

  test('model preparation guards lifecycle and exports bounded failures', () {
    final prepare = between(
      'Future<void> _prepareModel()',
      '// --- always-on listening ---',
    );
    expectOrdered(prepare, [
      'final present = await GemmaService.instance.isModelPresent();',
      'if (_disposing || !mounted) return;',
      'setState(()',
      'await GemmaService.instance.ensureReady(',
      'onProgress: (p)',
      'if (_disposing || !mounted) return;',
      'setState(() => _downloadPct = p);',
      'if (_disposing || !mounted) return;',
      "setState(() => _status = 'Model ready — tap the mic to start.');",
      'on GemmaGenerationFailure catch (failure)',
      'if (_disposing || !mounted) return;',
      "'Model load failed: \${failure.code}'",
      'catch (_)',
      'if (_disposing || !mounted) return;',
      "'Model load failed: model_prepare_failed'",
      'finally',
      'if (!_disposing && mounted)',
    ]);
    expect(prepare, isNot(contains(r'$e')));
    expect(source, isNot(contains('GemmaService.instance.dispose')));
  });

  test('capture callbacks cannot redirect and fail only after exact cancel',
      () {
    final capture = between(
      'final class _AssistantListeningCapture',
      'final class _AssistantListeningPluginLifecycle',
    );
    final cancellation = between(
      'Future<bool> cancelSource()',
      'Future<bool> stopRecorder()',
    );
    expectOrdered(cancellation, [
      'final existing = _cancelFuture;',
      '_cancelFuture = result;',
      '_subscription.future',
      'Future<void>.sync(subscription.cancel)',
      'completer.complete(true)',
    ]);
    final failure = between(
      'void failAfterExactCancellation()',
      'Future<bool> cancelSource()',
    );
    expectOrdered(failure, [
      '_failureRequested = true;',
      'final cancellation = cancelSource();',
      'if (succeeded && !_terminal.isCompleted)',
      'AssistantListeningCaptureTerminal.failed',
    ]);
    final ended = between(
      'void publishEnded()',
      'void failAfterExactCancellation()',
    );
    expectOrdered(ended, [
      'if (_failureRequested || _terminal.isCompleted) return;',
      'AssistantListeningCaptureTerminal.ended',
    ]);
    expect(capture, contains('final AudioRecorder _recorder;'));
    expect(capture, isNot(matches(RegExp(r'final String\??\s'))));

    final start = between(
      'startCapture(',
      'void _onCaptureData(',
    );
    expectOrdered(start, [
      '_captureLanes[generation] = capture;',
      '_recorder.startStream(buildCaptureConfig())',
      'final exactGeneration = generation;',
      'final exactSession = session;',
      'audioStream.listen(',
      'capture,',
      'exactGeneration,',
      'exactSession,',
      'onError:',
      'onDone: capture.publishEnded,',
      'capture.publishSubscription(subscription);',
    ]);

    final onData = between(
      'void _onCaptureData(',
      'void _onCaptureError(',
    );
    expectOrdered(onData, [
      '_isExactCapture(capture, exactGeneration, exactSession)',
      '_listeningCallbackIsCurrent(',
      '_asr.feed(exactSession, chunk)',
      'capture.failAfterExactCancellation();',
      '_fenceListeningSource(exactGeneration, exactSession);',
    ]);
    final onError = between(
      'void _onCaptureError(',
      'bool _isExactCapture(',
    );
    expectOrdered(onError, [
      '_isExactCapture(capture, exactGeneration, exactSession)',
      'capture.failAfterExactCancellation();',
      '_fenceListeningSource(exactGeneration, exactSession);',
    ]);
    expect(
      onData,
      isNot(contains('_listeningGeneration')),
      reason: 'plugin callbacks must not look up mutable widget authority',
    );
    expect(
      onError,
      isNot(contains('_listeningGeneration')),
      reason: 'plugin callbacks must not look up mutable widget authority',
    );

    for (final method in <String>[
      'Future<bool> cancelSource()',
      'Future<bool> stopRecorder()',
      'Future<bool> recoverAmbiguousStart()',
    ]) {
      final body = between(
        method,
        method == 'Future<bool> cancelSource()'
            ? 'Future<bool> stopRecorder()'
            : method == 'Future<bool> stopRecorder()'
                ? 'Future<bool> recoverAmbiguousStart()'
                : '/// Plugin adapter for one Assistant listening owner.',
      );
      expect(body, contains('final existing = _'));
      expect(body, contains('return existing;'));
      expect(body, isNot(contains('partial')));
      expect(body, isNot(contains('utterance')));
    }
  });

  test('listening desired state and UI callbacks are exact and serialized', () {
    final toggle = between(
      'Future<void> _toggleAlwaysOn()',
      'bool _listeningUiGenerationIsCurrent(',
    );
    expect(toggle, contains('final enable = !_alwaysOn;'));
    expect(toggle, contains('_listeningOwner.enable()'));
    expect(toggle, contains('_listeningOwner.disable()'));
    expectOrdered(toggle, [
      '_listeningGeneration = null;',
      'setState(()',
      '_listeningOwner.enable()',
      '_listeningGeneration = generation;',
      'await generation.done;',
    ]);

    final uiGuard = between(
      'bool _listeningUiGenerationIsCurrent(',
      'bool _listeningCallbackIsCurrent(',
    );
    expect(uiGuard, contains('mounted'));
    expect(uiGuard, contains('identical(_listeningGeneration, generation)'));

    final callbackGuard = between(
      'bool _listeningCallbackIsCurrent(',
      'void _onListeningStarted(',
    );
    expectOrdered(callbackGuard, [
      '_listeningUiGenerationIsCurrent(generation)',
      '_listeningOwner.isAuthoritative(generation)',
      '_listeningLifecycle.ownsExactSession(generation, session)',
    ]);
    expect(source, isNot(contains('_currentSession')));
    expect(source, isNot(contains('_currentCapture')));

    final revoked = between(
      'void _onListeningRevoked(',
      'void _onListeningStopped(',
    );
    expectOrdered(revoked, [
      'unawaited(_stopSpeaking());',
      'if (!_listeningUiGenerationIsCurrent(generation)) return;',
      '_listeningGeneration = null;',
    ]);
  });

  test('STOP uses only the shared exact typed command contract', () {
    final endpoint = between(
      'void _onListeningEndpoint(',
      'void _onListeningMicChunk(',
    );
    expect(endpoint, contains('final isStop = isStopCommand(utterance);'));

    final typed = between(
      'Future<void> _submitTyped()',
      'Future<bool> _stopSpeaking()',
    );
    expect(typed, contains('final isStop = isStopCommand(text);'));

    final partial = between(
      'void _onListeningPartial(',
      'void _onListeningEndpoint(',
    );
    expect(partial, contains('partial.length >= _bargeInChars'));
    expect(partial, isNot(contains('isStopCommand')));
    expect(RegExp(r'isStopCommand\(').allMatches(source).length, 2);
    expect(source, isNot(contains('_looksLikeStop')));
    expect(source, isNot(contains("contains('stop')")));
    expect(source, isNot(contains("contains('quiet')")));
    expect(source, isNot(contains("contains('cancel')")));
  });

  test('dispose fences all owners and gates recorder disposal on exact close',
      () {
    final dispose = between(
      'void dispose()',
      'Widget build(BuildContext context)',
    );
    expectOrdered(dispose, [
      '_disposing = true;',
      '_turn++;',
      '_replyGeneration = null;',
      '_listeningGeneration = null;',
      'final replyClose = _replyOwner.close();',
      'final listeningClose = _listeningOwner.close();',
      'final playbackClose = _ttsPlayback.close();',
      'final listeningLifecycle = _listeningLifecycle;',
      'final ttsPlayer = _ttsPlayer;',
      'unawaited(() async {',
      'await replyClose;',
      'final listeningReceipt = await listeningClose;',
      'listeningLifecycle.disposeRecorderAfterExactClose(',
      'await playbackClose;',
      'await ttsPlayer.close();',
    ]);
    final asyncStart = dispose.indexOf('unawaited(() async {');
    final asyncEnd = dispose.indexOf('}());', asyncStart);
    expect(asyncStart, isNonNegative);
    expect(asyncEnd, greaterThan(asyncStart));
    final asyncDispose = dispose.substring(asyncStart, asyncEnd);
    expect(asyncDispose, isNot(contains('_listeningLifecycle')));
    expect(asyncDispose, isNot(contains('_ttsPlayer')));

    final recorderDispose = between(
      'Future<bool> disposeRecorderAfterExactClose(',
      'class AssistantScreen extends StatefulWidget',
    );
    expectOrdered(recorderDispose, [
      'if (!receipt.exactResourcesSettled)',
      'return Future<bool>.value(false);',
      '_recorderDisposeFuture = result;',
      '_recorder.dispose()',
    ]);
    expect(RegExp(r'_recorder\.dispose\(\)').allMatches(source).length, 1);
    expect(source, isNot(contains('AsrService.instance.close')));
    expect(source, isNot(contains('TtsService.instance.dispose')));
  });

  test('playback error logs expose only a bounded code', () {
    final playbackError = between(
      'void _onSpeechPlaybackError(',
      '// --- latency profiling ---',
    );
    expect(playbackError, contains('Object _error'));
    expect(playbackError, contains('StackTrace _stackTrace'));
    expect(playbackError, contains('speech_playback_failed'));
    expect(playbackError, isNot(contains(r'$error')));
    expect(playbackError, isNot(contains(r'$stackTrace')));
  });
}
