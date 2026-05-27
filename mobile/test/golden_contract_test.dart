// Cross-language behavior contract — the Dart half.
//
// Runs the shared fixtures in ../tests/golden over mobile/lib/contract.dart and
// asserts the Dart port produces exactly the results the Python core does (the
// Python half is tests/test_golden_contract.py). Keeps the two shells from
// drifting — see docs/target_architecture.md §9. Pure Dart: no device/emulator,
// no plugins.
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/contract.dart';

void main() {
  // `flutter test` runs with the package root (mobile/) as the working dir.
  const goldenDir = '../tests/golden';

  test('sentence_split.json matches the shared contract', () {
    final data = json.decode(
      File('$goldenDir/sentence_split.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    for (final raw in data['cases'] as List) {
      final c = raw as Map<String, dynamic>;
      final tokens = (c['tokens'] as List).cast<String>();
      final expected = (c['expect'] as List).cast<String>();
      expect(streamSentences(tokens), expected, reason: c['name'] as String);
    }
  });

  test('commands.json matches the shared contract', () {
    final data = json.decode(
      File('$goldenDir/commands.json').readAsStringSync(),
    ) as Map<String, dynamic>;
    for (final raw in data['normalize'] as List) {
      final c = raw as Map<String, dynamic>;
      expect(normalizeCommand(c['in'] as String), c['out'] as String,
          reason: 'normalize: "${c['in']}"');
    }
    for (final raw in data['is_stop'] as List) {
      final c = raw as Map<String, dynamic>;
      expect(isStopCommand(c['in'] as String), c['expect'] as bool,
          reason: 'is_stop: "${c['in']}"');
    }
  });
}
