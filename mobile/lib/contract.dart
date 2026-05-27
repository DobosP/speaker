/// The cross-language behavior contract every shell shares — the Dart port.
///
/// These pure functions MUST match the Python source of truth in
/// `core/contract.py` exactly; both are pinned by the shared fixtures in
/// `tests/golden/` (see `docs/target_architecture.md` §9). The Python half runs
/// in `tests/test_golden_contract.py`; the Dart half runs over the same JSON in
/// `mobile/test/golden_contract_test.dart`.

// --- streaming-TTS sentence splitting ---

const String _whitespace = ' \t\n\r';
const String _terminators = '.!?';

bool _isSpace(String ch) => _whitespace.contains(ch);

/// `(emitEnd, resume)` for the first complete boundary, or null. `emitEnd` is
/// the slice end of the spoken text (terminator included, newline excluded);
/// `resume` is where the remaining buffer continues (boundary chars consumed).
(int, int)? _nextCut(String buffer) {
  for (var i = 0; i < buffer.length; i++) {
    final ch = buffer[i];
    if (ch == '\n') return (i, i + 1);
    if (_terminators.contains(ch) &&
        i + 1 < buffer.length &&
        _isSpace(buffer[i + 1])) {
      return (i + 1, i + 2);
    }
  }
  return null;
}

/// Pull every complete sentence out of [buffer]; returns `(sentences, remaining)`.
/// Each sentence is trimmed and empties are dropped.
(List<String>, String) drainCompleteSentences(String buffer) {
  final out = <String>[];
  while (true) {
    final cut = _nextCut(buffer);
    if (cut == null) break;
    final (emitEnd, resume) = cut;
    final sentence = buffer.substring(0, emitEnd).trim();
    buffer = buffer.substring(resume);
    if (sentence.isNotEmpty) out.add(sentence);
  }
  return (out, buffer);
}

/// The full streaming-TTS emission for a token stream: complete sentences are
/// emitted as boundaries arrive and the trailing remainder is flushed at the end.
List<String> streamSentences(Iterable<String> tokens) {
  final out = <String>[];
  var buffer = '';
  for (final token in tokens) {
    buffer += token;
    final (sentences, rest) = drainCompleteSentences(buffer);
    out.addAll(sentences);
    buffer = rest;
  }
  final tail = buffer.trim();
  if (tail.isNotEmpty) out.add(tail);
  return out;
}

// --- control-command normalization + stop recognition ---

final RegExp _nonCommand = RegExp(r'[^a-z ]');
final RegExp _spaceRun = RegExp(r' +');

/// The stop-class control phrases both runtimes recognize. Mode/confirm/deny
/// commands are config-driven and desktop-only, so they are not listed here.
const Set<String> stopCommands = {
  'stop',
  'cancel',
  'quiet',
  'stop talking',
  'be quiet',
};

/// Lowercase, drop anything that isn't a-z or space, collapse spaces, trim.
String normalizeCommand(String text) {
  final lowered = text.toLowerCase().replaceAll(_nonCommand, '');
  return lowered.replaceAll(_spaceRun, ' ').trim();
}

bool isStopCommand(String text) => stopCommands.contains(normalizeCommand(text));
