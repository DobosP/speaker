// Pure-Dart tests for the pre-partial ASR speech-start detector.
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/asr_isolate.dart';

Uint8List _pcm16(List<double> samples) {
  final bytes = Uint8List(samples.length * 2);
  final data = ByteData.sublistView(bytes);
  for (var i = 0; i < samples.length; i++) {
    final clamped = samples[i].clamp(-1.0, 1.0);
    data.setInt16(i * 2, (clamped * 32767).round(), Endian.little);
  }
  return bytes;
}

void main() {
  test('silence is not speech start', () {
    expect(pcm16LikelySpeechStart(_pcm16(List.filled(320, 0.0))), isFalse);
  });

  test('steady room tone below floor is not speech start', () {
    expect(pcm16LikelySpeechStart(_pcm16(List.filled(320, 0.006))), isFalse);
  });

  test('speech-like chunk fires before any ASR partial is available', () {
    final samples = List<double>.generate(320, (i) {
      final sign = i.isEven ? 1.0 : -1.0;
      return sign * 0.035;
    });

    expect(pcm16LikelySpeechStart(_pcm16(samples)), isTrue);
  });

  test('short transient peak also blocks quiet-floor training', () {
    final samples = List<double>.filled(320, 0.0);
    samples[12] = 0.06;

    expect(pcm16LikelySpeechStart(_pcm16(samples)), isTrue);
  });
}
