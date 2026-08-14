import 'agent_event.dart';
import 'contract.dart';

enum AgentIntentKind {
  ignore('ignore'),
  stop('stop'),
  confirm('confirm'),
  deny('deny'),
  modeSwitch('mode_switch'),
  assistant('assistant'),
  search('search'),
  research('research'),
  command('command'),
  dictation('dictation'),
  meetingNote('meeting_note');

  const AgentIntentKind(this.wireName);

  final String wireName;
}

final class AgentSpeechObservation {
  AgentSpeechObservation({
    required this.text,
    required this.normalized,
    required this.isFinal,
    required this.language,
    required this.activationScore,
    required List<String> keywords,
  }) : keywords = List.unmodifiable(keywords) {
    if (!activationScore.isFinite ||
        activationScore < 0 ||
        activationScore > 1) {
      throw RangeError.range(activationScore, 0, 1, 'activationScore');
    }
    if (language != 'en' && language != 'unknown') {
      throw const FormatException('unknown mobile observation language');
    }
    if (keywords.length > 8) {
      throw RangeError.range(keywords.length, 0, 8, 'keywords.length');
    }
  }

  final String text;
  final String normalized;
  final bool isFinal;
  final String language;
  final double activationScore;
  final List<String> keywords;

  Map<String, Object?> toGoldenJson() => {
        'normalized': normalized,
        'is_final': isFinal,
        'language': language,
        'activation_score': activationScore,
        'keywords': keywords,
      };
}

final class AgentIntentDecision {
  const AgentIntentDecision({
    required this.kind,
    required this.confidence,
    required this.text,
    required this.reason,
    this.mode,
    this.targetMode,
    this.requiresConfirmation = false,
    this.speak = true,
  });

  final AgentIntentKind kind;
  final double confidence;
  final String text;
  final String reason;
  final AgentMode? mode;
  final AgentMode? targetMode;
  final bool requiresConfirmation;
  final bool speak;

  bool get startsTask => const {
        AgentIntentKind.assistant,
        AgentIntentKind.search,
        AgentIntentKind.research,
        AgentIntentKind.command,
        AgentIntentKind.dictation,
        AgentIntentKind.meetingNote,
      }.contains(kind);

  Map<String, Object?> toGoldenJson() => {
        'kind': kind.wireName,
        'confidence': confidence,
        'text': text,
        'reason': reason,
        'mode': mode?.wireName,
        'target_mode': targetMode?.wireName,
        'requires_confirmation': requiresConfirmation,
        'speak': speak,
        'starts_task': startsTask,
      };
}

final class AgentDecisionTransition {
  const AgentDecisionTransition({
    required this.input,
    required this.observation,
    required this.decision,
    required this.resultingMode,
  });

  final AgentEvent input;
  final AgentSpeechObservation observation;
  final AgentIntentDecision decision;
  final AgentMode resultingMode;
}

/// Deterministic mobile-only speech-decision projection.
///
/// This intentionally is not a port of the Python `LiveSpeechAnalyzer`.
/// Mobile control text uses the shared, narrow `normalizeCommand` and
/// `isStopCommand` contract. Every partial is semantic-ignore; the Assistant
/// widget's separate acoustic barge path remains outside this projection.
final class AgentDecisionSpine {
  static const _confirmPhrases = {
    'yes',
    'confirm',
    'approve',
    'do it',
    'ok do it',
  };
  static const _denyPhrases = {
    'no',
    'deny',
    'cancel command',
    'do not',
    'dont',
  };
  static const _wakeTerms = {'assistant', 'computer', 'jarvis'};
  static const _modeAliases = {
    'passive mode': AgentMode.passive,
    'assistant mode': AgentMode.assistant,
    'command mode': AgentMode.command,
    'search mode': AgentMode.search,
    'research mode': AgentMode.research,
    'dictation mode': AgentMode.dictation,
    'meeting mode': AgentMode.meeting,
  };
  static const _stopwords = {
    'a',
    'an',
    'and',
    'are',
    'for',
    'from',
    'how',
    'i',
    'in',
    'is',
    'me',
    'of',
    'on',
    'or',
    'please',
    'the',
    'to',
    'what',
    'with',
    'you',
  };
  static const _englishMarkers = {
    'assistant',
    'computer',
    'jarvis',
    'search',
    'research',
    'stop',
    'cancel',
    'dictate',
    'meeting',
    'mode',
  };

  AgentDecisionTransition accept(
    AgentEvent input,
    AgentMode currentMode, {
    bool hasPendingConfirmation = false,
  }) {
    if (input.kind != AgentEventKind.sttPartial &&
        input.kind != AgentEventKind.sttFinal) {
      throw ArgumentError.value(input.kind, 'input.kind');
    }
    final rawText = input.payload['text'];
    final rawFinal = input.payload['is_final'];
    if (rawText is! String || rawFinal is! bool) {
      throw const FormatException('invalid STT AgentEvent payload');
    }
    if (rawFinal != (input.kind == AgentEventKind.sttFinal)) {
      throw const FormatException('STT kind/finality mismatch');
    }
    final utf8Bytes = AgentEvent.utf8Length(rawText);
    if (utf8Bytes > AgentEvent.maxTextBytes) {
      throw RangeError.range(
        utf8Bytes,
        0,
        AgentEvent.maxTextBytes,
        'text UTF-8 bytes',
      );
    }

    final observation = _observe(rawText, isFinal: rawFinal);
    final decision = _decide(
      observation,
      currentMode,
      hasPendingConfirmation: hasPendingConfirmation,
    );
    final resultingMode = decision.kind == AgentIntentKind.modeSwitch
        ? decision.targetMode!
        : currentMode;
    return AgentDecisionTransition(
      input: input,
      observation: observation,
      decision: decision,
      resultingMode: resultingMode,
    );
  }

  AgentSpeechObservation _observe(String text, {required bool isFinal}) {
    final normalized = normalizeCommand(text);
    return AgentSpeechObservation(
      text: text,
      normalized: normalized,
      isFinal: isFinal,
      language: _detectLanguage(normalized),
      activationScore: _activationScore(normalized),
      keywords: _keywords(normalized),
    );
  }

  AgentIntentDecision _decide(
    AgentSpeechObservation observation,
    AgentMode currentMode, {
    bool hasPendingConfirmation = false,
  }) {
    // Transient ASR text never owns mobile STOP, mode, confirmation, or work.
    if (!observation.isFinal) {
      return AgentIntentDecision(
        kind: AgentIntentKind.ignore,
        confidence: 1,
        text: observation.text,
        reason: 'partial_non_control',
      );
    }

    final text = observation.normalized;
    if (text.isEmpty) {
      return const AgentIntentDecision(
        kind: AgentIntentKind.ignore,
        confidence: 1,
        text: '',
        reason: 'empty',
      );
    }
    if (isStopCommand(observation.text)) {
      return AgentIntentDecision(
        kind: AgentIntentKind.stop,
        confidence: 1,
        text: observation.text,
        reason: 'stop_phrase',
      );
    }
    if (hasPendingConfirmation) {
      if (_confirmPhrases.contains(text)) {
        return AgentIntentDecision(
          kind: AgentIntentKind.confirm,
          confidence: 0.98,
          text: observation.text,
          reason: 'confirm_phrase',
        );
      }
      if (_denyPhrases.contains(text)) {
        return AgentIntentDecision(
          kind: AgentIntentKind.deny,
          confidence: 0.98,
          text: observation.text,
          reason: 'deny_phrase',
        );
      }
    }

    final targetMode = _modeAliases[text];
    if (targetMode != null) {
      return AgentIntentDecision(
        kind: AgentIntentKind.modeSwitch,
        confidence: 0.98,
        text: observation.text,
        reason: 'mode_phrase',
        targetMode: targetMode,
      );
    }

    final explicit = _explicitIntent(text, observation.text);
    if (explicit != null) return explicit;

    switch (currentMode) {
      case AgentMode.passive:
        if (observation.activationScore < 0.65) {
          return AgentIntentDecision(
            kind: AgentIntentKind.ignore,
            confidence: 0.9,
            text: observation.text,
            reason: 'passive_no_activation',
          );
        }
        return AgentIntentDecision(
          kind: AgentIntentKind.assistant,
          confidence: observation.activationScore,
          text: _stripWakeWord(observation.text),
          reason: 'wake_word_activation',
          mode: AgentMode.assistant,
        );
      case AgentMode.search:
        return AgentIntentDecision(
          kind: AgentIntentKind.search,
          confidence: 0.82,
          text: observation.text,
          reason: 'search_mode',
          mode: currentMode,
        );
      case AgentMode.research:
        return AgentIntentDecision(
          kind: AgentIntentKind.research,
          confidence: 0.82,
          text: observation.text,
          reason: 'research_mode',
          mode: currentMode,
        );
      case AgentMode.dictation:
        return AgentIntentDecision(
          kind: AgentIntentKind.dictation,
          confidence: 0.9,
          text: observation.text,
          reason: 'dictation_mode',
          speak: false,
        );
      case AgentMode.meeting:
        return AgentIntentDecision(
          kind: AgentIntentKind.meetingNote,
          confidence: 0.85,
          text: observation.text,
          reason: 'meeting_mode',
          speak: false,
        );
      case AgentMode.command:
        return AgentIntentDecision(
          kind: AgentIntentKind.command,
          confidence: 0.82,
          text: observation.text,
          reason: 'command_mode',
          mode: currentMode,
          requiresConfirmation: true,
        );
      case AgentMode.assistant:
        return AgentIntentDecision(
          kind: AgentIntentKind.assistant,
          confidence: 0.75,
          text: observation.text,
          reason: 'assistant_mode',
          mode: currentMode,
        );
    }
  }

  AgentIntentDecision? _explicitIntent(String normalized, String original) {
    if (normalized.startsWith('please ')) {
      final courteous = normalized.substring('please '.length);
      if (courteous.startsWith('research ') ||
          courteous.startsWith('search ')) {
        normalized = courteous;
        original = _afterFirstWord(original);
      }
    }
    if (normalized.startsWith('research ')) {
      return AgentIntentDecision(
        kind: AgentIntentKind.research,
        confidence: 0.95,
        text: _afterFirstWord(original),
        reason: 'research_prefix',
        mode: AgentMode.research,
      );
    }
    if (normalized.startsWith('search ')) {
      var query = _afterFirstWord(original);
      if (normalizeCommand(query).startsWith('for ')) {
        query = _afterFirstWord(query);
      }
      return AgentIntentDecision(
        kind: AgentIntentKind.search,
        confidence: 0.95,
        text: query,
        reason: 'search_prefix',
        mode: AgentMode.search,
      );
    }
    if (normalized.startsWith('dictate ')) {
      return AgentIntentDecision(
        kind: AgentIntentKind.dictation,
        confidence: 0.95,
        text: _afterFirstWord(original),
        reason: 'dictation_prefix',
        mode: AgentMode.dictation,
        speak: false,
      );
    }
    if (normalized.startsWith('run ') ||
        normalized.startsWith('open ') ||
        normalized.startsWith('execute ')) {
      return AgentIntentDecision(
        kind: AgentIntentKind.command,
        confidence: 0.9,
        text: original,
        reason: 'command_prefix',
        mode: AgentMode.command,
        requiresConfirmation: true,
      );
    }
    return null;
  }

  static List<String> _keywords(String normalized) {
    final output = <String>[];
    for (final word in normalized.split(' ')) {
      if (word.isEmpty || _stopwords.contains(word) || output.contains(word)) {
        continue;
      }
      output.add(word);
      if (output.length == 8) break;
    }
    return List.unmodifiable(output);
  }

  static String _detectLanguage(String normalized) {
    final words = normalized.split(' ').toSet();
    return words.any(_englishMarkers.contains) ? 'en' : 'unknown';
  }

  static double _activationScore(String normalized) {
    final words = normalized.split(' ').toSet();
    var score = 0.0;
    if (words.any(_wakeTerms.contains)) score += 0.55;
    if (words.any(const {'can', 'could', 'please', 'help'}.contains)) {
      score += 0.2;
    }
    if (words.any(const {'search', 'research'}.contains)) score += 0.3;
    return score > 1 ? 1 : score;
  }

  static String _stripWakeWord(String text) {
    final words = text.split(RegExp(r'\s+'));
    if (words.isNotEmpty &&
        _wakeTerms.contains(normalizeCommand(words.first))) {
      final tail = words.skip(1).join(' ').trim();
      return tail.isEmpty ? text : tail;
    }
    return text;
  }

  static String _afterFirstWord(String text) {
    final match = RegExp(r'^\S+\s+(.*)$', dotAll: true).firstMatch(text.trim());
    return match?.group(1)?.trim() ?? '';
  }
}
