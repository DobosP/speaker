// On-device LLM: small Gemma 3 (1B, int4) via flutter_gemma / MediaPipe.
//
// The model is NOT bundled in the APK (it would balloon it past a gigabyte).
// Instead it is downloaded once on first use from a public GitHub release
// asset and cached on device by flutter_gemma; every run after that is fully
// offline. We host the weights on our own release so the app needs no
// HuggingFace token — the gated download happens in CI (see
// .github/workflows/publish-model.yml), not on the phone.
import 'package:flutter_gemma/flutter_gemma.dart';

class GemmaService {
  GemmaService._();
  static final GemmaService instance = GemmaService._();

  // Public, ungated mirror of litert-community/Gemma3-1B-IT (q4, GPU-friendly),
  // republished by CI to our own release tag.
  static const modelUrl =
      'https://github.com/DobosP/speaker/releases/download/gemma-model/'
      'Gemma3-1B-IT-q4.litertlm';

  static const _systemInstruction =
      'You are a concise, friendly on-device voice assistant. '
      'Answer in one or two short, natural, speakable sentences.';

  dynamic _model;
  dynamic _chat;

  bool get isReady => _model != null;

  // Download (first run only) + initialize the GPU inference engine.
  // [onProgress] receives 0..100 during the one-time download.
  Future<void> ensureReady({void Function(double percent)? onProgress}) async {
    if (_model != null) return;

    await FlutterGemma.installModel(modelType: ModelType.gemmaIt)
        .fromNetwork(modelUrl)
        .withProgress((p) => onProgress?.call((p as num).toDouble()))
        .install();

    _model = await FlutterGemma.getActiveModel(
      maxTokens: 1024,
      preferredBackend: PreferredBackend.gpu,
    );
    _chat = await _model.createChat(systemInstruction: _systemInstruction);
  }

  // Stream the assistant's reply token-by-token.
  Stream<String> reply(String prompt) async* {
    await _chat.addQueryChunk(Message.text(text: prompt, isUser: true));
    await for (final response in _chat.generateChatResponseAsync()) {
      if (response is TextResponse) {
        yield response.token;
      }
    }
  }

  Future<void> dispose() async {
    await _model?.close();
    _model = null;
    _chat = null;
  }
}
