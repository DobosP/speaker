import 'dart:collection';

/// Portable modes shared with `always_on_agent.events.Mode`.
enum AgentMode {
  passive('passive'),
  assistant('assistant'),
  command('command'),
  search('search'),
  research('research'),
  dictation('dictation'),
  meeting('meeting');

  const AgentMode(this.wireName);

  final String wireName;

  static AgentMode parse(String value) => values.firstWhere(
        (mode) => mode.wireName == value,
        orElse: () => throw FormatException('unknown agent mode'),
      );
}

/// The closed decision/event projection currently consumed by mobile.
enum AgentEventKind {
  sttPartial('stt.partial'),
  sttFinal('stt.final'),
  intentDecision('intent.decision'),
  controlStop('control.stop'),
  controlMode('control.mode'),
  controlConfirm('control.confirm'),
  controlDeny('control.deny');

  const AgentEventKind(this.wireName);

  final String wireName;
}

/// Immutable, bounded mobile AgentEvent projection.
///
/// This is deliberately not the complete or identical Python runtime bus. It
/// uses portable event/mode names for the bounded mobile decision seam while
/// each runtime tests its own partial-transcript authority policy.
final class AgentEvent {
  AgentEvent._({
    required this.kind,
    required Map<String, Object?> payload,
    required this.priority,
    required this.sequence,
  }) : payload = UnmodifiableMapView(Map<String, Object?>.of(payload)) {
    if (priority < 0 || priority > 100) {
      throw RangeError.range(priority, 0, 100, 'priority');
    }
    if (sequence < 0 || sequence > maxSafeSequence) {
      throw RangeError.range(sequence, 0, maxSafeSequence, 'sequence');
    }
    if (payload.length > maxPayloadEntries) {
      throw RangeError.range(
        payload.length,
        0,
        maxPayloadEntries,
        'payload.length',
      );
    }
    for (final entry in payload.entries) {
      if (entry.key.isEmpty || utf8Length(entry.key) > maxPayloadKeyBytes) {
        throw FormatException('invalid AgentEvent payload key');
      }
      final value = entry.value;
      if (value is! String &&
          value is! int &&
          value is! double &&
          value is! bool &&
          value != null) {
        throw FormatException('AgentEvent payload values must be scalar');
      }
      if (value is String && utf8Length(value) > maxTextBytes) {
        throw RangeError.range(
          utf8Length(value),
          0,
          maxTextBytes,
          'payload value UTF-8 bytes',
        );
      }
      if (value is int &&
          (value < -maxSafeSequence || value > maxSafeSequence)) {
        throw RangeError.range(
          value,
          -maxSafeSequence,
          maxSafeSequence,
          'payload integer',
        );
      }
      if (value is double && !value.isFinite) {
        throw FormatException('AgentEvent payload numbers must be finite');
      }
    }
  }

  static const int maxTextBytes = 16384;
  static const int maxPayloadEntries = 12;
  static const int maxPayloadKeyBytes = 64;
  static const int maxSafeSequence = 9007199254740991;

  final AgentEventKind kind;
  final Map<String, Object?> payload;
  final int priority;
  final int sequence;

  factory AgentEvent.sttPartial({
    required String text,
    required int inputGeneration,
    required int sequence,
  }) =>
      AgentEvent._(
        kind: AgentEventKind.sttPartial,
        payload: {
          'text': text,
          'is_final': false,
          'input_generation': _requireGeneration(inputGeneration),
        },
        priority: 90,
        sequence: sequence,
      );

  factory AgentEvent.sttFinal({
    required String text,
    required int inputGeneration,
    required int sequence,
  }) =>
      AgentEvent._(
        kind: AgentEventKind.sttFinal,
        payload: {
          'text': text,
          'is_final': true,
          'owner_verified': false,
          'origin': 'unknown',
          'input_generation': _requireGeneration(inputGeneration),
        },
        priority: 50,
        sequence: sequence,
      );

  factory AgentEvent.intentDecision({
    required String kind,
    required double confidence,
    required String text,
    required String reason,
    required int sequence,
  }) =>
      AgentEvent._(
        kind: AgentEventKind.intentDecision,
        payload: {
          'kind': _requireIntentKind(kind),
          'confidence': _requireConfidence(confidence),
          'text': text,
          'reason': reason,
        },
        priority: 55,
        sequence: sequence,
      );

  factory AgentEvent.controlStop({
    required int inputGeneration,
    required int sequence,
    String reason = 'voice',
  }) =>
      AgentEvent._(
        kind: AgentEventKind.controlStop,
        payload: {
          'reason': reason,
          'already_cancelled': false,
          'input_generation': _requireGeneration(inputGeneration),
        },
        priority: 0,
        sequence: sequence,
      );

  factory AgentEvent.controlMode({
    required AgentMode mode,
    required int inputGeneration,
    required int sequence,
  }) =>
      AgentEvent._(
        kind: AgentEventKind.controlMode,
        payload: {
          'mode': mode.wireName,
          'source': 'voice',
          'input_generation': _requireGeneration(inputGeneration),
        },
        priority: 10,
        sequence: sequence,
      );

  factory AgentEvent.controlConfirm({
    required int inputGeneration,
    required int sequence,
  }) =>
      AgentEvent._(
        kind: AgentEventKind.controlConfirm,
        payload: {
          'source': 'voice',
          'owner_verified': false,
          'origin': 'unknown',
          'direct_user_instruction': false,
          'input_generation': _requireGeneration(inputGeneration),
        },
        priority: 5,
        sequence: sequence,
      );

  factory AgentEvent.controlDeny({
    required int inputGeneration,
    required int sequence,
  }) =>
      AgentEvent._(
        kind: AgentEventKind.controlDeny,
        payload: {
          'source': 'voice',
          'input_generation': _requireGeneration(inputGeneration),
        },
        priority: 5,
        sequence: sequence,
      );

  Map<String, Object?> toJson() => {
        'kind': kind.wireName,
        'payload': payload,
        'priority': priority,
        'sequence': sequence,
      };

  static int utf8Length(String value) {
    var bytes = 0;
    for (final rune in value.runes) {
      if (rune <= 0x7f) {
        bytes += 1;
      } else if (rune <= 0x7ff) {
        bytes += 2;
      } else if (rune <= 0xffff) {
        bytes += 3;
      } else {
        bytes += 4;
      }
    }
    return bytes;
  }

  static int _requireGeneration(int value) {
    if (value < 0 || value > maxSafeSequence) {
      throw RangeError.range(value, 0, maxSafeSequence, 'inputGeneration');
    }
    return value;
  }

  static String _requireIntentKind(String value) {
    const kinds = {
      'ignore',
      'stop',
      'confirm',
      'deny',
      'mode_switch',
      'assistant',
      'search',
      'research',
      'command',
      'dictation',
      'meeting_note',
    };
    if (!kinds.contains(value)) {
      throw FormatException('unknown intent decision kind');
    }
    return value;
  }

  static double _requireConfidence(double value) {
    if (!value.isFinite || value < 0 || value > 1) {
      throw RangeError.range(value, 0, 1, 'confidence');
    }
    return value;
  }
}
