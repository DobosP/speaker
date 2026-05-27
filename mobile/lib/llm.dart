// On-device LLM: small Gemma 3 (1B, int4) via flutter_gemma / MediaPipe.
//
// The model is NOT bundled in the APK (it would balloon it past a gigabyte).
// Instead it is downloaded once on first use from a public GitHub release
// asset and cached on device by flutter_gemma; every run after that is fully
// offline. We host the weights on our own release so the app needs no
// HuggingFace token — the gated download happens in CI (see
// .github/workflows/publish-model.yml), not on the phone.
import 'package:flutter/foundation.dart' show visibleForTesting;
import 'package:flutter_gemma/flutter_gemma.dart';

import './model_store.dart';

class GemmaService {
  GemmaService._();
  static final GemmaService instance = GemmaService._();

  // MediaPipe .task bundle (q4) of litert-community/Gemma3-1B-IT, republished by
  // CI to our own release tag. We use .task (not .litertlm) because on Android
  // flutter_gemma loads .task via the stable MediaPipe engine factory; .litertlm
  // routes through a fragile FFI path that mis-routes and yields a null engine.
  static const modelUrl =
      'https://github.com/DobosP/speaker/releases/download/gemma-model/'
      'Gemma3-1B-IT-q4.task';

  static const _systemInstruction =
      'You are a friendly on-device voice assistant. Answer in a few clear, '
      'natural, speakable sentences — concise but complete, not clipped.';

  dynamic _model;

  bool get isReady => _model != null;

  // Whether the weights are already on the device (sideloaded via adb push or
  // kept from a previous run). Lets the UI say "loading" instead of
  // "downloading" when no network fetch is needed.
  Future<bool> isModelPresent() => ModelStore.hasModel();

  // Test seam: inject a fake engine so the chat-per-turn + sampling behavior can
  // be unit-tested without a real model (see test/llm_glue_test.dart).
  @visibleForTesting
  set debugModel(dynamic model) => _model = model;

  // Make the model ready, then initialize the GPU inference engine.
  // [onProgress] receives 0..100 during the one-time download.
  //
  // The weights are loaded from a fixed on-disk path (ModelStore) and only
  // downloaded when that file is genuinely absent — so a reinstall reuses the
  // sideloaded/cached model instead of re-fetching ~550 MB every time.
  Future<void> ensureReady({void Function(double percent)? onProgress}) async {
    if (_model != null) return;

    if (!await ModelStore.hasModel()) {
      await ModelStore.download(modelUrl, onProgress: onProgress);
    }
    final file = await ModelStore.modelFile();
    await FlutterGemma.installModel(modelType: ModelType.gemmaIt)
        .fromFile(file.path)
        .install();

    // Prefer the GPU engine; fall back to CPU if the device can't create it
    // (some GPUs/drivers return a null engine instead of throwing).
    _model = await _activate(PreferredBackend.gpu);
    _model ??= await _activate(PreferredBackend.cpu);
    if (_model == null) {
      throw Exception('Failed to initialize the on-device model engine.');
    }
  }

  Future<dynamic> _activate(PreferredBackend backend) async {
    try {
      return await FlutterGemma.getActiveModel(
        maxTokens: 1024,
        preferredBackend: backend,
      );
    } catch (_) {
      return null;
    }
  }

  // Stream the assistant's reply token-by-token. A FRESH chat per turn keeps
  // each request independent: reusing one session let the tiny 1B model
  // accumulate state and degenerate into the same looping reply. Explicit
  // sampling (topK/temperature) also stops the greedy (topK=1) repetition that
  // made it answer the same thing regardless of input.
  Stream<String> reply(String prompt) async* {
    final chat = await _model.createChat(
      systemInstruction: _systemInstruction,
      temperature: 0.8,
      topK: 40,
      randomSeed: DateTime.now().millisecondsSinceEpoch & 0x7fffffff,
    );
    await chat.addQueryChunk(Message.text(text: prompt, isUser: true));
    await for (final response in chat.generateChatResponseAsync()) {
      if (response is TextResponse) {
        yield response.token;
      }
    }
  }

  Future<void> dispose() async {
    await _model?.close();
    _model = null;
  }
}
