import 'dart:collection';

import 'agent_decision.dart';
import 'agent_event.dart';

final class AgentSessionContractException implements Exception {
  const AgentSessionContractException(this.code);

  final String code;
}

/// Opaque exact ownership token for one admitted mobile assistant reply.
final class AgentTurn {
  const AgentTurn._(this._owner, this.generation);

  final Object _owner;
  final int generation;
}

enum AgentMobileEffectKind {
  ignore,
  stop,
  switchMode,
  reply,
  unavailable,
}

/// Pure projection from one analyzer result to ordered public event kinds.
///
/// This models the bounded mobile order: STT, intent, then optional control.
/// It does not claim Python analyzer or priority-bus parity. A normalized-empty
/// STT event produces no intent publication.
final class AgentDecisionTrace {
  static List<AgentEvent> project({
    required AgentEvent input,
    required AgentSpeechObservation observation,
    required AgentIntentDecision decision,
    required int inputGeneration,
  }) {
    if (input.kind != AgentEventKind.sttPartial &&
        input.kind != AgentEventKind.sttFinal) {
      throw const FormatException('trace input must be an STT event');
    }
    final inputText = input.payload['text'];
    final inputIsFinal = input.payload['is_final'];
    final kindIsFinal = input.kind == AgentEventKind.sttFinal;
    if (inputText is! String ||
        inputIsFinal is! bool ||
        inputText != observation.text ||
        inputIsFinal != observation.isFinal ||
        inputIsFinal != kindIsFinal) {
      throw const FormatException('trace input/observation mismatch');
    }
    if ((decision.kind == AgentIntentKind.modeSwitch) !=
        (decision.targetMode != null)) {
      throw const FormatException('invalid mode-switch projection');
    }
    if (!observation.isFinal && decision.kind != AgentIntentKind.ignore) {
      throw const FormatException('partial decisions must be ignore');
    }
    final eventGeneration = input.payload['input_generation'];
    if (eventGeneration is! int || eventGeneration != inputGeneration) {
      throw const FormatException('input generation mismatch');
    }
    if (observation.normalized.isEmpty) {
      return List.unmodifiable([input]);
    }

    final intentSequence = _nextSequence(input.sequence);
    final events = <AgentEvent>[
      input,
      AgentEvent.intentDecision(
        kind: decision.kind.wireName,
        confidence: decision.confidence,
        text: decision.text,
        reason: decision.reason,
        sequence: intentSequence,
      ),
    ];
    if (_controlEventRequired(decision.kind)) {
      events.add(
        _projectControl(
          decision,
          inputGeneration,
          _nextSequence(intentSequence),
        ),
      );
    }
    return List.unmodifiable(events);
  }

  static bool _controlEventRequired(AgentIntentKind kind) => const {
        AgentIntentKind.stop,
        AgentIntentKind.modeSwitch,
        AgentIntentKind.confirm,
        AgentIntentKind.deny,
      }.contains(kind);

  static AgentEvent _projectControl(
    AgentIntentDecision decision,
    int inputGeneration,
    int sequence,
  ) {
    switch (decision.kind) {
      case AgentIntentKind.stop:
        return AgentEvent.controlStop(
          inputGeneration: inputGeneration,
          sequence: sequence,
          reason: decision.reason,
        );
      case AgentIntentKind.modeSwitch:
        return AgentEvent.controlMode(
          mode: decision.targetMode!,
          inputGeneration: inputGeneration,
          sequence: sequence,
        );
      case AgentIntentKind.confirm:
        return AgentEvent.controlConfirm(
          inputGeneration: inputGeneration,
          sequence: sequence,
        );
      case AgentIntentKind.deny:
        return AgentEvent.controlDeny(
          inputGeneration: inputGeneration,
          sequence: sequence,
        );
      case AgentIntentKind.ignore:
      case AgentIntentKind.assistant:
      case AgentIntentKind.search:
      case AgentIntentKind.research:
      case AgentIntentKind.command:
      case AgentIntentKind.dictation:
      case AgentIntentKind.meetingNote:
        throw StateError('decision has no control projection');
    }
  }

  static int _nextSequence(int value) {
    if (value >= AgentEvent.maxSafeSequence) {
      throw const AgentSessionContractException('event sequence exhausted');
    }
    return value + 1;
  }
}

final class AgentSessionTransition {
  AgentSessionTransition({
    required Object owner,
    required this.input,
    required this.observation,
    required this.decision,
    required this.resultingMode,
    required this.turn,
    required List<AgentEvent> trace,
  })  : _owner = owner,
        trace = UnmodifiableListView(List<AgentEvent>.of(trace));

  final Object _owner;
  final AgentEvent input;
  final AgentSpeechObservation observation;
  final AgentIntentDecision decision;
  final AgentMode resultingMode;
  final AgentTurn? turn;
  final List<AgentEvent> trace;

  int get inputGeneration => input.payload['input_generation']! as int;
  int get terminalSequence => trace.last.sequence;

  AgentMobileEffectKind get mobileEffect => switch (decision.kind) {
        AgentIntentKind.ignore => AgentMobileEffectKind.ignore,
        AgentIntentKind.stop => AgentMobileEffectKind.stop,
        AgentIntentKind.modeSwitch => AgentMobileEffectKind.switchMode,
        AgentIntentKind.assistant => AgentMobileEffectKind.reply,
        AgentIntentKind.confirm ||
        AgentIntentKind.deny ||
        AgentIntentKind.search ||
        AgentIntentKind.research ||
        AgentIntentKind.command ||
        AgentIntentKind.dictation ||
        AgentIntentKind.meetingNote =>
          AgentMobileEffectKind.unavailable,
      };
}

/// App-owned mobile decision/session spine.
///
/// It retains only the current mode and one exact assistant-turn token. No
/// transcript/event history is retained. Unsupported mobile capabilities are
/// represented by typed decisions but never receive a reply-capable turn.
final class AgentSession {
  AgentSession({
    AgentMode initialMode = AgentMode.assistant,
    AgentDecisionSpine? decisionSpine,
  })  : _mode = initialMode,
        _decisionSpine = decisionSpine ?? AgentDecisionSpine();

  final Object _owner = Object();
  final AgentDecisionSpine _decisionSpine;
  AgentMode _mode;
  AgentTurn? _currentTurn;
  int _inputGeneration = 0;
  int _sequence = 0;
  bool _closed = false;

  AgentMode get mode => _mode;
  int get inputGeneration => _inputGeneration;
  bool get isClosed => _closed;

  AgentSessionTransition acceptPartial(String text) =>
      _accept(text: text, isFinal: false);

  AgentSessionTransition acceptFinal(String text) =>
      _accept(text: text, isFinal: true);

  AgentSessionTransition _accept({
    required String text,
    required bool isFinal,
  }) {
    _requireOpen();
    final textBytes = AgentEvent.utf8Length(text);
    if (textBytes > AgentEvent.maxTextBytes) {
      throw RangeError.range(
        textBytes,
        0,
        AgentEvent.maxTextBytes,
        'text UTF-8 bytes',
      );
    }

    final hasRawFinal = isFinal && text.isNotEmpty;
    final candidateGeneration = hasRawFinal
        ? _checkedAdd(_inputGeneration, 1, 'input generation')
        : _inputGeneration;
    final inputSequence = _checkedAdd(_sequence, 1, 'event sequence');
    final input = isFinal
        ? AgentEvent.sttFinal(
            text: text,
            inputGeneration: candidateGeneration,
            sequence: inputSequence,
          )
        : AgentEvent.sttPartial(
            text: text,
            inputGeneration: candidateGeneration,
            sequence: inputSequence,
          );
    final projected = _decisionSpine.accept(input, _mode);
    final trace = AgentDecisionTrace.project(
      input: input,
      observation: projected.observation,
      decision: projected.decision,
      inputGeneration: candidateGeneration,
    );

    final candidateMode = projected.decision.kind == AgentIntentKind.modeSwitch
        ? projected.resultingMode
        : _mode;
    AgentTurn? candidateCurrentTurn = _currentTurn;
    AgentTurn? admittedTurn;
    if (hasRawFinal || projected.decision.kind == AgentIntentKind.stop) {
      candidateCurrentTurn = null;
    }
    // Gemma answering is the only mobile capability this slice admits. Search,
    // research, command, dictation, and meeting remain typed-but-unavailable.
    if (hasRawFinal && projected.decision.kind == AgentIntentKind.assistant) {
      admittedTurn = AgentTurn._(_owner, candidateGeneration);
      candidateCurrentTurn = admittedTurn;
    }

    final transition = AgentSessionTransition(
      owner: _owner,
      input: input,
      observation: projected.observation,
      decision: projected.decision,
      resultingMode: candidateMode,
      turn: admittedTurn,
      trace: trace,
    );

    // One linearization point: nothing above mutates session state.
    _sequence = trace.last.sequence;
    _inputGeneration = candidateGeneration;
    _mode = candidateMode;
    _currentTurn = candidateCurrentTurn;
    return transition;
  }

  bool isCurrent(AgentTurn turn) =>
      !_closed &&
      identical(turn._owner, _owner) &&
      identical(turn, _currentTurn);

  bool isCurrentInput(AgentSessionTransition transition) =>
      !_closed &&
      identical(transition._owner, _owner) &&
      transition.inputGeneration == _inputGeneration &&
      transition.terminalSequence == _sequence;

  bool interruptIfCurrent(AgentTurn turn, {required String reason}) {
    _requireOpen();
    final reasonBytes = AgentEvent.utf8Length(reason);
    if (reasonBytes < 1 || reasonBytes > 64) {
      throw RangeError.range(reasonBytes, 1, 64, 'reason UTF-8 bytes');
    }
    if (!isCurrent(turn)) return false;
    _currentTurn = null;
    return true;
  }

  void close() {
    if (_closed) return;
    _currentTurn = null;
    _closed = true;
  }

  void _requireOpen() {
    if (_closed) throw const AgentSessionContractException('closed');
  }

  static int _checkedAdd(int value, int increment, String label) {
    if (value > AgentEvent.maxSafeSequence - increment) {
      throw AgentSessionContractException('$label exhausted');
    }
    return value + increment;
  }
}
