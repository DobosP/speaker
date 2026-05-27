// Guards the LLM fix: every reply() must create a FRESH chat (no reused session)
// with non-greedy sampling (topK > 1), so the model can't regress to the canned,
// looping output it produced before. Uses a fake engine injected via the test
// seam — no real model, no device, runs in the fast `flutter test`.
import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/llm.dart';

class _FakeChat {
  Future<void> addQueryChunk(dynamic message) async {}
  Stream<dynamic> generateChatResponseAsync() async* {} // no tokens needed
}

class _RecordingModel {
  final List<Map<String, dynamic>> createChatCalls = [];
  Future<dynamic> createChat({
    String? systemInstruction,
    double? temperature,
    int? topK,
    int? randomSeed,
  }) async {
    createChatCalls.add({
      'temperature': temperature,
      'topK': topK,
      'randomSeed': randomSeed,
    });
    return _FakeChat();
  }
}

void main() {
  test('reply uses a fresh chat per turn with non-greedy sampling', () async {
    final model = _RecordingModel();
    GemmaService.instance.debugModel = model;

    await GemmaService.instance.reply('what is the capital of France').toList();
    await GemmaService.instance.reply('tell me a short joke').toList();

    // A fresh chat per turn — not one reused session that accumulates state.
    expect(model.createChatCalls.length, 2);
    for (final call in model.createChatCalls) {
      expect(call['topK'], greaterThan(1)); // not greedy (topK=1) decoding
      expect(call['temperature'], greaterThan(0.0));
    }
  });
}
