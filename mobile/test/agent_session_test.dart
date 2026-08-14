import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/agent_decision.dart';
import 'package:speaker_mobile/agent_event.dart';
import 'package:speaker_mobile/agent_session.dart';

void main() {
  test('partials fence publication but cannot revoke a current turn', () {
    final session = AgentSession();
    final first = session.acceptFinal('hello');
    final turn = first.turn!;
    expect(session.isCurrent(turn), isTrue);
    expect(session.isCurrentInput(first), isTrue);

    for (final text in const [
      'stop',
      'research mode',
      'search for pipecat',
      'run backup',
    ]) {
      final partial = session.acceptPartial(text);
      expect(partial.decision.kind, AgentIntentKind.ignore, reason: text);
      expect(partial.mobileEffect, AgentMobileEffectKind.ignore, reason: text);
      expect(session.mode, AgentMode.assistant, reason: text);
      expect(session.isCurrent(turn), isTrue, reason: text);
      expect(
        partial.trace.any((event) => event.kind == AgentEventKind.controlStop),
        isFalse,
        reason: text,
      );
    }
    expect(session.isCurrentInput(first), isFalse);

    final stop = session.acceptFinal('Stop!');
    expect(stop.decision.kind, AgentIntentKind.stop);
    expect(stop.mobileEffect, AgentMobileEffectKind.stop);
    expect(stop.turn, isNull);
    expect(session.isCurrent(turn), isFalse);
  });

  test('new assistant final revokes only the exact prior reply turn', () {
    final session = AgentSession();
    final first = session.acceptFinal('hello');
    final oldTurn = first.turn!;
    final second = session.acceptFinal('another question');
    final newTurn = second.turn!;

    expect(session.isCurrent(oldTurn), isFalse);
    expect(session.isCurrent(newTurn), isTrue);
    expect(session.interruptIfCurrent(oldTurn, reason: 'stale'), isFalse);
    expect(session.isCurrent(newTurn), isTrue);
    expect(session.interruptIfCurrent(newTurn, reason: 'barge'), isTrue);
    expect(session.isCurrent(newTurn), isFalse);
  });

  test('final mode state is bounded and non-assistant work is unavailable', () {
    final session = AgentSession();
    final mode = session.acceptFinal('research mode');
    expect(mode.decision.kind, AgentIntentKind.modeSwitch);
    expect(mode.mobileEffect, AgentMobileEffectKind.switchMode);
    expect(mode.turn, isNull);
    expect(session.mode, AgentMode.research);

    final research = session.acceptFinal('compare ownership designs');
    expect(research.decision.kind, AgentIntentKind.research);
    expect(research.mobileEffect, AgentMobileEffectKind.unavailable);
    expect(research.turn, isNull);
    expect(session.mode, AgentMode.research);

    session.acceptFinal('assistant mode');
    final reply = session.acceptFinal('compare ownership designs');
    expect(reply.decision.kind, AgentIntentKind.assistant);
    expect(reply.mobileEffect, AgentMobileEffectKind.reply);
    expect(reply.turn, isNotNull);
    expect(session.mode, AgentMode.assistant);
  });

  test('foreign ownership and immutable trace fail closed', () {
    final left = AgentSession();
    final right = AgentSession();
    final leftTransition = left.acceptFinal('hello');
    final turn = leftTransition.turn!;

    expect(right.isCurrent(turn), isFalse);
    expect(right.isCurrentInput(leftTransition), isFalse);
    expect(right.interruptIfCurrent(turn, reason: 'foreign'), isFalse);
    expect(() => leftTransition.trace.add(leftTransition.input),
        throwsUnsupportedError);
    expect(
      () => leftTransition.input.payload['text'] = 'mutated',
      throwsUnsupportedError,
    );

    left.close();
    expect(left.isCurrent(turn), isFalse);
    expect(
      () => left.acceptFinal('late'),
      throwsA(isA<AgentSessionContractException>()),
    );
  });

  test('oversize and invalid interrupt inputs leave state atomic', () {
    final session = AgentSession();
    final accepted = session.acceptFinal('hello');
    final turn = accepted.turn!;
    final generation = session.inputGeneration;
    final oversized = 'x' * (AgentEvent.maxTextBytes + 1);

    expect(() => session.acceptFinal(oversized), throwsRangeError);
    expect(session.inputGeneration, generation);
    expect(session.mode, AgentMode.assistant);
    expect(session.isCurrentInput(accepted), isTrue);
    expect(session.isCurrent(turn), isTrue);

    expect(
      () => session.interruptIfCurrent(turn, reason: ''),
      throwsRangeError,
    );
    expect(session.isCurrent(turn), isTrue);
  });

  test('empty final preserves a turn while raw punctuation supersedes it', () {
    final session = AgentSession();
    final accepted = session.acceptFinal('hello');
    final turn = accepted.turn!;
    final generation = session.inputGeneration;

    final empty = session.acceptFinal('');
    expect(empty.decision.kind, AgentIntentKind.ignore);
    expect(session.inputGeneration, generation);
    expect(session.isCurrent(turn), isTrue);

    final punctuation = session.acceptFinal('!!!');
    expect(punctuation.decision.kind, AgentIntentKind.ignore);
    expect(session.inputGeneration, generation + 1);
    expect(session.isCurrent(turn), isFalse);
  });

  test('closed factories and inconsistent projections reject hostile values',
      () {
    expect(
      () => AgentEvent.sttFinal(text: 'x', inputGeneration: -1, sequence: 1),
      throwsRangeError,
    );
    expect(
      () => AgentEvent.intentDecision(
        kind: 'invented',
        confidence: 1,
        text: 'x',
        reason: 'x',
        sequence: 1,
      ),
      throwsFormatException,
    );
    expect(
      () => AgentEvent.intentDecision(
        kind: 'assistant',
        confidence: double.nan,
        text: 'x',
        reason: 'x',
        sequence: 1,
      ),
      throwsRangeError,
    );

    final input = AgentEvent.sttFinal(
      text: 'hello',
      inputGeneration: 1,
      sequence: 1,
    );
    final projected = AgentDecisionSpine().accept(input, AgentMode.assistant);
    expect(
      () => AgentDecisionTrace.project(
        input: AgentEvent.controlStop(inputGeneration: 1, sequence: 1),
        observation: projected.observation,
        decision: projected.decision,
        inputGeneration: 1,
      ),
      throwsFormatException,
    );
    expect(
      () => AgentDecisionTrace.project(
        input: input,
        observation: projected.observation,
        decision: const AgentIntentDecision(
          kind: AgentIntentKind.modeSwitch,
          confidence: 1,
          text: 'hello',
          reason: 'hostile',
        ),
        inputGeneration: 1,
      ),
      throwsFormatException,
    );
  });
}
