import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/agent_decision.dart';
import 'package:speaker_mobile/agent_event.dart';
import 'package:speaker_mobile/agent_session.dart';
import 'package:speaker_mobile/contract.dart';

void main() {
  const commandsPath = '../tests/golden/commands.json';

  AgentEvent sttEvent(String text, {required bool isFinal}) => isFinal
      ? AgentEvent.sttFinal(
          text: text,
          inputGeneration: text.isEmpty ? 0 : 1,
          sequence: 1,
        )
      : AgentEvent.sttPartial(
          text: text,
          inputGeneration: 0,
          sequence: 1,
        );

  test('mobile observation and final STOP use commands.json exactly', () {
    final data = json.decode(File(commandsPath).readAsStringSync())
        as Map<String, dynamic>;
    final spine = AgentDecisionSpine();

    for (final raw in data['normalize'] as List<dynamic>) {
      final testCase = raw as Map<String, dynamic>;
      final text = testCase['in'] as String;
      final transition = spine.accept(
        sttEvent(text, isFinal: true),
        AgentMode.assistant,
      );
      expect(
        transition.observation.normalized,
        testCase['out'],
        reason: 'normalize: "$text"',
      );
      expect(
        transition.observation.normalized,
        normalizeCommand(text),
        reason: 'shared normalizer: "$text"',
      );
    }

    for (final raw in data['is_stop'] as List<dynamic>) {
      final testCase = raw as Map<String, dynamic>;
      final text = testCase['in'] as String;
      final expected = testCase['expect'] as bool;
      final event = sttEvent(text, isFinal: true);
      final transition = spine.accept(event, AgentMode.assistant);
      final trace = AgentDecisionTrace.project(
        input: event,
        observation: transition.observation,
        decision: transition.decision,
        inputGeneration: text.isEmpty ? 0 : 1,
      );
      expect(
        transition.decision.kind == AgentIntentKind.stop,
        expected,
        reason: 'final STOP: "$text"',
      );
      expect(
        trace.any((item) => item.kind == AgentEventKind.controlStop),
        expected,
        reason: 'final STOP trace: "$text"',
      );
      expect(isStopCommand(text), expected, reason: 'shared STOP: "$text"');

      final partialEvent = sttEvent(text, isFinal: false);
      final partial = spine.accept(partialEvent, AgentMode.assistant);
      final partialTrace = AgentDecisionTrace.project(
        input: partialEvent,
        observation: partial.observation,
        decision: partial.decision,
        inputGeneration: 0,
      );
      expect(
        partial.decision.kind,
        AgentIntentKind.ignore,
        reason: 'partial: "$text"',
      );
      expect(
        partialTrace.any((item) => item.kind == AgentEventKind.controlStop),
        isFalse,
        reason: 'partial trace: "$text"',
      );
    }
  });

  test('all partial grammar classes are semantic-ignore', () {
    final spine = AgentDecisionSpine();
    for (final text in const [
      'stop',
      'research mode',
      'search for pipecat',
      'run backup',
      'yes',
      'no',
    ]) {
      final event = sttEvent(text, isFinal: false);
      final transition = spine.accept(
        event,
        AgentMode.assistant,
        hasPendingConfirmation: true,
      );
      final trace = AgentDecisionTrace.project(
        input: event,
        observation: transition.observation,
        decision: transition.decision,
        inputGeneration: 0,
      );
      expect(transition.decision.kind, AgentIntentKind.ignore, reason: text);
      expect(transition.decision.startsTask, isFalse, reason: text);
      expect(transition.resultingMode, AgentMode.assistant, reason: text);
      expect(
        trace.where((item) => item.kind.name.startsWith('control')),
        isEmpty,
        reason: text,
      );
    }
  });

  test('Romanian, full-width, and digit-only text gains no control kind', () {
    final spine = AgentDecisionSpine();
    for (final text in const [
      'oprește!',
      'anulează',
      'mod cercetare',
      'ＳＴＯＰ',
      'ｒｅｓｅａｒｃｈ ｍｏｄｅ',
      '12345',
    ]) {
      final transition = spine.accept(
        sttEvent(text, isFinal: true),
        AgentMode.assistant,
      );
      expect(
        const {
          AgentIntentKind.stop,
          AgentIntentKind.modeSwitch,
          AgentIntentKind.confirm,
          AgentIntentKind.deny,
          AgentIntentKind.search,
          AgentIntentKind.research,
          AgentIntentKind.command,
          AgentIntentKind.dictation,
          AgentIntentKind.meetingNote,
        }.contains(transition.decision.kind),
        isFalse,
        reason: text,
      );
      expect(
        transition.observation.normalized,
        normalizeCommand(text),
        reason: text,
      );
      expect(transition.observation.keywords, isNot(contains('12345')));
    }
  });

  test('bounded final modes and typed unavailable decisions stay explicit', () {
    final spine = AgentDecisionSpine();
    const aliases = {
      'passive mode': AgentMode.passive,
      'assistant mode': AgentMode.assistant,
      'command mode': AgentMode.command,
      'search mode': AgentMode.search,
      'research mode': AgentMode.research,
      'dictation mode': AgentMode.dictation,
      'meeting mode': AgentMode.meeting,
    };
    for (final entry in aliases.entries) {
      final transition = spine.accept(
        sttEvent(entry.key, isFinal: true),
        AgentMode.assistant,
      );
      expect(transition.decision.kind, AgentIntentKind.modeSwitch);
      expect(transition.decision.targetMode, entry.value);
      expect(transition.resultingMode, entry.value);
    }

    for (final entry in const {
      'search for pipecat': AgentIntentKind.search,
      'research audio ownership': AgentIntentKind.research,
      'run backup': AgentIntentKind.command,
      'dictate release notes': AgentIntentKind.dictation,
    }.entries) {
      final session = AgentSession();
      final transition = session.acceptFinal(entry.key);
      expect(transition.decision.kind, entry.value, reason: entry.key);
      expect(transition.mobileEffect, AgentMobileEffectKind.unavailable);
      expect(transition.turn, isNull);
      expect(session.mode, AgentMode.assistant);
    }

    expect(() => AgentMode.parse('invented'), throwsFormatException);
  });

  test('partial control projection is rejected even for a hostile caller', () {
    final event = sttEvent('stop', isFinal: false);
    final projected = AgentDecisionSpine().accept(event, AgentMode.assistant);
    expect(
      () => AgentDecisionTrace.project(
        input: event,
        observation: projected.observation,
        decision: const AgentIntentDecision(
          kind: AgentIntentKind.stop,
          confidence: 1,
          text: 'stop',
          reason: 'hostile',
        ),
        inputGeneration: 0,
      ),
      throwsFormatException,
    );
  });
}
