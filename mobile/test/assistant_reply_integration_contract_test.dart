// Source-level guard for the plugin-bound Assistant composition seam.
//
// Pure owner/session behavior has deterministic unit tests. This test binds the
// Flutter widget to those owners without loading a model, plugin, audio device,
// emulator, or phone.
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/agent_decision.dart';
import 'package:speaker_mobile/agent_event.dart';
import 'package:speaker_mobile/agent_session.dart';

void main() {
  late String source;

  setUpAll(() {
    source = File('lib/assistant.dart').readAsStringSync();
  });

  String between(String start, String end) {
    final startIndex = source.indexOf(start);
    final endIndex = source.indexOf(end, startIndex + start.length);
    expect(startIndex, isNonNegative, reason: 'missing start marker: $start');
    expect(
      endIndex,
      greaterThan(startIndex),
      reason: 'missing end marker: $end',
    );
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

  test('reply prompt consumes the session decision payload', () {
    final session = AgentSession(initialMode: AgentMode.passive);
    final transition = session.acceptFinal('assistant please help me');

    expect(transition.decision.kind, AgentIntentKind.assistant);
    expect(transition.decision.text, 'please help me');
    expect(transition.turn, isNotNull);

    final finalPath = between(
      '_AssistantReplyAdmission? _linearizeFinal(',
      'void _publishFinalStatus(',
    );
    expect(finalPath, contains('prompt: transition.decision.text.trim(),'));
    expect(finalPath, isNot(contains('prompt: rawText.trim(),')));
  });

  test('reply admission retains exact session, reply, playback, and lease', () {
    expect(source, contains("import './agent_session.dart';"));
    expect(source, contains("import './assistant_reply_owner.dart';"));
    expect(source, contains("import './tts_process_owner.dart';"));
    expect(source, contains('final AgentSession session;'));
    expect(source, contains('final AgentSessionTransition transition;'));
    expect(source, contains('AgentTurn? _agentTurn;'));
    expect(source, contains('TtsProcessLease? _speechLease;'));
    expect(source, contains('TtsPlaybackOwner? _ttsPlayback;'));
    expect(
      source,
      contains('AssistantReplyOwner(openReply: GemmaService.instance.reply)'),
    );
    expect(
      RegExp(r'GemmaService\.instance\.reply').allMatches(source).length,
      1,
    );
    expect(source, isNot(contains('GemmaService.instance.cancelCurrent')));
    expect(source, isNot(contains('GemmaService.instance.dispose')));
    expect(source, isNot(contains('await for (')));

    final answer = between(
      'Future<void> _answerUtterance(',
      'bool _replyIsCurrent(',
    );
    expectOrdered(answer, [
      'if (prompt.isEmpty ||',
      '_replyIsCurrent(admission)',
      'final admittedGeneration = _replyOwner.start(',
      'replyGeneration = admittedGeneration;',
      '_replyGeneration = admittedGeneration;',
      'final done = await admittedGeneration.done;',
      'final isExactReply = identical(_replyGeneration, admittedGeneration);',
    ]);
    expect(answer, contains('identical(callbackGeneration, replyGeneration)'));
    expect(answer, contains('identical(_replyGeneration, callbackGeneration)'));
    expect(answer, contains('_replyOwner.isAuthoritative(callbackGeneration)'));
    expect(answer, contains('_replyIsCurrent(admission)'));
    expectOrdered(answer, [
      'final remainingTts = ttsBuffer;',
      "ttsBuffer = '';",
      'done.outcome == AssistantReplyOutcome.completed',
      'remainingTts,',
      'admission.playback,',
      'admission.playbackGeneration,',
      'flushAll: true',
      'setState(() => _thinking = false);',
      '_replyGeneration = null;',
    ]);

    final current = between(
      'bool _replyIsCurrent(',
      '// Send whatever is typed',
    );
    expectOrdered(current, [
      'identical(_agentTurn, admission.turn)',
      'widget.session.isCurrent(admission.turn)',
      '_speechOwnerIsCurrent(admission.lease, admission.playback)',
      'admission.playback.isCurrent(admission.playbackGeneration)',
    ]);

    final admission = between(
      'Future<void> _runReplyAdmission(',
      '// --- generation ---',
    );
    expectOrdered(admission, [
      'admission.playback.waitForStop(',
      '_replyIsCurrent(admission)',
      'widget.session.isCurrentInput(admission.transition)',
      'await _answerUtterance(admission);',
    ]);
  });

  test('raw input is bounded before trim and every partial is observed', () {
    final preflight = between(
      'bool _assistantPromptFitsBound(',
      'final class _AssistantTtsClipState',
    );
    expectOrdered(preflight, [
      'text.length > assistantReplyPromptMaximumUtf8Bytes',
      'utf8.encode(text).length <= assistantReplyPromptMaximumUtf8Bytes',
    ]);

    final partial = between(
      'void _onListeningPartial(',
      'void _onListeningEndpoint(',
    );
    expectOrdered(partial, [
      '_listeningCallbackIsCurrent(generation, session)',
      '_assistantPromptFitsBound(partial)',
      'widget.session.acceptPartial(partial);',
      'if (partial.isEmpty) return;',
      'partial.length >= _bargeInChars',
      '_stopSpeaking()',
    ]);
    expect(partial, isNot(contains('mobileEffect')));
    expect(partial, isNot(contains('isStopCommand')));

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
      'if (text.isEmpty) return;',
      '_linearizeFinal(rawText)',
    ]);

    final endpoint = between(
      'void _onListeningEndpoint(',
      'void _onListeningMicChunk(',
    );
    expectOrdered(endpoint, [
      'if (utterance.isEmpty) return;',
      'final admission = _linearizeFinal(utterance);',
      '_promptController.clear();',
      'if (admission == null) return;',
      '_runReplyAdmission(admission)',
    ]);
    expect(source, isNot(contains('isStopCommand(')));
    expect(source, isNot(contains('_looksLikeStop')));
    expect(source, isNot(contains("contains('stop')")));
    expect(source, isNot(contains("contains('quiet')")));
    expect(source, isNot(contains("contains('cancel')")));
  });

  test('session final linearizes before every upper and lower fence', () {
    final finalPath = between(
      '_AssistantReplyAdmission? _linearizeFinal(',
      'void _publishFinalStatus(',
    );
    expect(RegExp(r'^\s*await ', multiLine: true).hasMatch(finalPath), isFalse);
    expectOrdered(finalPath, [
      'widget.session.acceptFinal(rawText);',
      '_agentTurn = transition.turn;',
      '_fenceExactReply(',
      'final fencedPlayback = _ttsPlayback;',
      'fencedPlayback.supersede();',
      'switch (transition.mobileEffect)',
      'case AgentMobileEffectKind.stop:',
      "_publishFinalStatus('Stopped.');",
      'case AgentMobileEffectKind.switchMode:',
      'case AgentMobileEffectKind.ignore:',
      'case AgentMobileEffectKind.unavailable:',
      'case AgentMobileEffectKind.reply:',
      'widget.session.isCurrent(turn)',
      '_tryAcquireSpeechOwner()',
      'final lease = _liveSpeechLease;',
      'final playback = _ttsPlayback;',
      'return _AssistantReplyAdmission(',
      'prompt: transition.decision.text.trim(),',
      'transition: transition,',
    ]);
    final tryAcquire = finalPath.indexOf('_tryAcquireSpeechOwner()');
    for (final authority in <String>[
      'case AgentMobileEffectKind.stop:',
      'case AgentMobileEffectKind.switchMode:',
      'case AgentMobileEffectKind.ignore:',
      'case AgentMobileEffectKind.unavailable:',
    ]) {
      expect(finalPath.indexOf(authority), lessThan(tryAcquire));
    }

    final fence = between(
      'void _fenceExactReply(',
      'void _interruptAgentTurn(',
    );
    expectOrdered(fence, [
      'final exactReply = _replyGeneration;',
      '_replyGeneration = null;',
      '_replyOwner.cancelExact(exactReply, reason: reason)',
      '_replyOwner.cancelCurrent(reason: reason)',
    ]);

    final admission = between(
      'Future<void> _runReplyAdmission(',
      '// --- generation ---',
    );
    expectOrdered(admission, [
      'admission.playback.waitForStop(',
      'if (!stopped)',
      'final wasCurrent = _replyIsCurrent(admission);',
      'admission.lease.revoke();',
      'if (wasCurrent)',
      'if (!_replyIsCurrent(admission) ||',
      'widget.session.isCurrentInput(admission.transition)',
      'await _answerUtterance(admission);',
    ]);
  });

  test('speech lease is nonblocking, nullable, and guards all TTS work', () {
    final acquire = between(
      'bool _tryAcquireSpeechOwner()',
      'TtsPlaybackOwner _buildTtsPlaybackOwner(',
    );
    expect(acquire, isNot(contains('await ')));
    expectOrdered(acquire, [
      'ttsProcessOwnerRegistry.tryAcquire();',
      'if (lease == null) return false;',
      '_speechLease = lease;',
      '_ttsPlayback = playback;',
    ]);

    final playback = between(
      'TtsPlaybackOwner _buildTtsPlaybackOwner(',
      'void _onSpeechUnavailable(',
    );
    expectOrdered(playback, [
      'final state = _stateReference;',
      'final ttsPlayer = _ttsPlayer;',
      'state.target?._speechOwnerIsCurrent(lease, playback)',
      'generateWaveFilename()',
      'state.target?._speechOwnerIsCurrent(lease, playback)',
      'TtsService.instance.synthesize(lease, text, filename)',
      'createPlaybackClip: (path)',
      'ttsPlayer.createClip(lease, path)',
    ]);
    expect(
      RegExp(r'state\.target\?\._speechOwnerIsCurrent')
          .allMatches(playback)
          .length,
      3,
    );
    expect(playback, isNot(contains('if (!_speechOwnerIsCurrent')));
    expect(playback, isNot(contains('_ttsPlayer.createClip')));

    final clip = between('TtsPlaybackClip createClip(', 'Future<void> _start(');
    expectOrdered(clip, [
      'ttsProcessOwnerRegistry.ownsExact(lease)',
      'player: AudioPlayer()',
      'onPlayerComplete.listen',
      '_start(state, path)',
    ]);

    final route = between(
      'Future<bool> configureRoute(',
      'AsrSession beginSession(',
    );
    expectOrdered(route, [
      'final lease = _speechLease();',
      'if (lease == null || !lease.admitsWork) return true;',
      '_ttsPlayer.configureGlobalRoute(lease)',
      'lease.revoke();',
      "_onSpeechUnavailable('speech_route_failed')",
      'return true;',
    ]);

    final playerClose = between(
      'Future<bool> close() async',
      'typedef _AssistantListeningGeneration',
    );
    expectOrdered(playerClose, [
      'revoke();',
      'await _routeTail;',
      'final current = _current;',
      'current.handle.stopAndRelease()',
      'return _routeExact && playerExact;',
    ]);

    final started = between(
      'void _onListeningStarted(',
      'void _onListeningRevoked(',
    );
    expectOrdered(started, [
      'final lease = _liveSpeechLease;',
      'if (lease != null)',
      '.ensureReady(lease)',
      "_onSpeechUnavailable('speech_warm_failed')",
      "'Listening… speech output is busy or unavailable.'",
    ]);
    expect(
      RegExp(r'TtsService\.instance\.synthesize\(').allMatches(source).length,
      1,
    );
    expect(
      RegExp(r'TtsService\.instance\s*\.ensureReady\(')
          .allMatches(source)
          .length,
      1,
    );
  });

  test('barge and dispose fence session plus upper owners before playback', () {
    final stop = between(
      'Future<bool> _stopSpeaking()',
      '// --- streaming TTS ---',
    );
    expectOrdered(stop, [
      "_interruptAgentTurn(reason: 'barge in');",
      '_fenceExactReply(reason: AssistantReplyCancelReason.cancelled);',
      'final playback = _ttsPlayback;',
      'final existing = _speechStopInFlight;',
      'playback.interrupt();',
    ]);

    final dispose = between(
      'void dispose()',
      'Widget build(BuildContext context)',
    );
    expectOrdered(dispose, [
      '_disposing = true;',
      "_interruptAgentTurn(reason: 'assistant disposed');",
      '_listeningGeneration = null;',
      'final replyClose = _replyOwner.close();',
      'final listeningClose = _listeningOwner.close();',
      'lease?.revoke();',
      'ttsPlayer.revoke();',
      '_ttsPlayback?.close()',
      'await replyClose;',
      'await listeningClose;',
      'disposeRecorderAfterExactClose(',
      'await playbackClose;',
      'await ttsPlayer.close();',
      'lease.close(() async {',
      'await closeUpperAndPlayback();',
      'TtsService.instance.dispose(lease)',
    ]);
    expect(dispose, isNot(contains('widget.session.close')));
    expect(source, isNot(contains('AsrService.instance.close')));
    expect(
      RegExp(r'TtsService\.instance\.dispose\(').allMatches(source).length,
      1,
    );
  });

  test('listening callbacks retain exact tokens and weak widget state', () {
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

    final onData = between('void _onCaptureData(', 'void _onCaptureError(');
    expectOrdered(onData, [
      '_isExactCapture(capture, exactGeneration, exactSession)',
      '_listeningCallbackIsCurrent(',
      '_asr.feed(exactSession, chunk)',
      'capture.failAfterExactCancellation();',
      '_fenceListeningSource(exactGeneration, exactSession);',
    ]);
    expect(onData, isNot(contains('_listeningGeneration')));

    final init = between(
      'void initState()',
      'TtsProcessLease? get _liveSpeechLease',
    );
    expectOrdered(init, [
      '_stateReference = WeakReference<_AssistantScreenState>(this);',
      '_tryAcquireSpeechOwner();',
      '_AssistantListeningPluginLifecycle(',
      'speechLease: () => state.target?._liveSpeechLease,',
      'AssistantListeningOwner<AsrSession, _AssistantListeningCapture>',
      'AssistantReplyOwner(openReply: GemmaService.instance.reply)',
    ]);
    expect(RegExp(r'state\.target\?\._onListening').allMatches(init).length, 3);
  });

  test('recorder cleanup and diagnostics remain bounded', () {
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

    final playbackError = between(
      'void _onSpeechPlaybackError(',
      '// --- latency profiling ---',
    );
    expectOrdered(playbackError, [
      'TtsProcessLease lease,',
      'TtsPlaybackOwner playback,',
      '_speechOwnerIsCurrent(lease, playback)',
      'playback.isCurrent(generation)',
      'return;',
      '_logLine(',
    ]);
    expect(playbackError, contains('Object _error'));
    expect(playbackError, contains('StackTrace _stackTrace'));
    expect(playbackError, contains('speech_playback_failed'));
    expect(playbackError, isNot(contains(r'$error')));
    expect(playbackError, isNot(contains(r'$stackTrace')));

    final callbacks = between(
      'void _onSpeechActivityChanged(',
      '// --- latency profiling ---',
    );
    expect(callbacks, contains('_speechOwnerIsCurrent(lease, playback)'));
    expect(callbacks, contains('playback.isCurrent(generation)'));
    expect(callbacks, contains('if (!speaking && _speaking)'));
    expect(
        callbacks, contains('stale authority cannot publish status or logs'));
  });
}
