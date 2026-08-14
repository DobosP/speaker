// On-device LLM: small Gemma 3 (1B, int4) via flutter_gemma / MediaPipe.
//
// The model is NOT bundled in the APK (it would balloon it past a gigabyte).
// Instead it is downloaded once on first use from a public GitHub release
// asset and cached on device by flutter_gemma; every run after that is fully
// offline. We host the weights on our own release so the app needs no
// HuggingFace token — the gated download happens in CI (see
// .github/workflows/publish-model.yml), not on the phone.
import 'dart:async';

import 'package:flutter/foundation.dart' show visibleForTesting;
import 'package:flutter_gemma/flutter_gemma.dart';

import './llm_generation_owner.dart';
import './model_store.dart';

abstract interface class GemmaModelBootstrapPort {
  Future<bool> hasModel();

  Future<void> download(
    String url, {
    void Function(double percent)? onProgress,
  });

  Future<String> modelPath();

  Future<void> install(String path);

  Future<dynamic> activate(PreferredBackend backend);
}

final class _FlutterGemmaModelBootstrap implements GemmaModelBootstrapPort {
  const _FlutterGemmaModelBootstrap();

  @override
  Future<bool> hasModel() => ModelStore.hasModel();

  @override
  Future<void> download(
    String url, {
    void Function(double percent)? onProgress,
  }) {
    return ModelStore.download(url, onProgress: onProgress);
  }

  @override
  Future<String> modelPath() async => (await ModelStore.modelFile()).path;

  @override
  Future<void> install(String path) {
    return FlutterGemma.installModel(modelType: ModelType.gemmaIt)
        .fromFile(path)
        .install();
  }

  @override
  Future<dynamic> activate(PreferredBackend backend) {
    return FlutterGemma.getActiveModel(
      maxTokens: 1024,
      preferredBackend: backend,
    );
  }
}

class GemmaService {
  GemmaService._({
    GemmaGenerationOwner? generationOwner,
    GemmaModelBootstrapPort? bootstrap,
    dynamic initialModel,
  })  : _model = initialModel,
        _generationOwner = generationOwner ?? GemmaGenerationOwner.shared,
        _bootstrap = bootstrap ?? const _FlutterGemmaModelBootstrap();
  static final GemmaService instance = GemmaService._();

  @visibleForTesting
  factory GemmaService.forTesting({
    required GemmaGenerationOwner generationOwner,
    required GemmaModelBootstrapPort bootstrap,
    dynamic initialModel,
  }) =>
      GemmaService._(
        generationOwner: generationOwner,
        bootstrap: bootstrap,
        initialModel: initialModel,
      );

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
  final GemmaGenerationOwner _generationOwner;
  final GemmaModelBootstrapPort _bootstrap;
  Future<void>? _readyFuture;
  Future<void>? _disposeFuture;
  bool _modelActivationUncertain = false;
  bool _lateModelCloseUncertain = false;

  bool get isReady =>
      _model != null &&
      !_generationOwner.isClosed &&
      !_generationOwner.isPoisoned &&
      !_modelActivationUncertain &&
      !_lateModelCloseUncertain &&
      _disposeFuture == null;

  // Whether the weights are already on the device (sideloaded via adb push or
  // kept from a previous run). Lets the UI say "loading" instead of
  // "downloading" when no network fetch is needed.
  Future<bool> isModelPresent() => _bootstrap.hasModel();

  // Make the model ready, then initialize the GPU inference engine.
  // [onProgress] receives 0..100 during the one-time download.
  //
  // The weights are loaded from a fixed on-disk path (ModelStore) and only
  // downloaded when that file is genuinely absent — so a reinstall reuses the
  // sideloaded/cached model instead of re-fetching ~550 MB every time.
  Future<void> ensureReady({void Function(double percent)? onProgress}) {
    _requireOpenOwner();
    if (_model != null) {
      return Future<void>.value();
    }
    final existing = _readyFuture;
    if (existing != null) {
      return existing;
    }
    final completer = Completer<void>();
    _readyFuture = completer.future;
    unawaited(_runEnsureReady(onProgress, completer));
    return completer.future;
  }

  Future<void> _runEnsureReady(
    void Function(double percent)? onProgress,
    Completer<void> completer,
  ) async {
    Object? failure;
    StackTrace? failureStack;
    try {
      await _ensureReadyOwned(onProgress);
    } catch (error, stackTrace) {
      failure = error;
      failureStack = stackTrace;
    } finally {
      _readyFuture = null;
    }
    if (completer.isCompleted) {
      return;
    }
    if (failure == null) {
      completer.complete();
    } else {
      completer.completeError(failure, failureStack!);
    }
  }

  Future<void> _ensureReadyOwned(
    void Function(double percent)? onProgress,
  ) async {
    _requireOpenOwner();
    final present = await _bootstrap.hasModel();
    _requireOpenOwner();
    if (!present) {
      await _bootstrap.download(modelUrl, onProgress: onProgress);
      _requireOpenOwner();
    }
    final path = await _bootstrap.modelPath();
    _requireOpenOwner();
    await _bootstrap.install(path);
    _requireOpenOwner();

    // Prefer the GPU engine; fall back to CPU if the device can't create it
    // (some GPUs/drivers return a null engine instead of throwing).
    dynamic candidate = await _activate(PreferredBackend.gpu);
    if (!_ownerAcceptsModel) {
      if (candidate != null) {
        await _retireLateModel(candidate);
      }
      _requireOpenOwner();
    }
    candidate ??= await _activate(PreferredBackend.cpu);
    if (candidate == null) {
      throw Exception('Failed to initialize the on-device model engine.');
    }
    if (!_ownerAcceptsModel) {
      await _retireLateModel(candidate);
      _requireOpenOwner();
    }
    _model = candidate;
  }

  Future<dynamic> _activate(PreferredBackend backend) async {
    try {
      return await _bootstrap.activate(backend);
    } catch (_, stackTrace) {
      // Resolved flutter_gemma 0.16.5 can throw after allocating a native
      // candidate without returning a closable Dart model. Do not classify
      // that as a clean null or construct a fallback beside uncertain state.
      _modelActivationUncertain = true;
      Error.throwWithStackTrace(
        const GemmaGenerationFailure('model_activation_ambiguous'),
        stackTrace,
      );
    }
  }

  // Stream the assistant's reply token-by-token. A FRESH chat per turn keeps
  // each request independent: reusing one session let the tiny 1B model
  // accumulate state and degenerate into the same looping reply. Explicit
  // sampling (topK/temperature) also stops the greedy (topK=1) repetition that
  // made it answer the same thing regardless of input.
  Stream<String> reply(String prompt) {
    if (!isReady) {
      throw StateError('GemmaService is not ready.');
    }
    final model = _model;
    return _generationOwner.generate(
      prompt: prompt,
      createChat: () async {
        final chat = await model.createChat(
          systemInstruction: _systemInstruction,
          temperature: 0.8,
          topK: 40,
          randomSeed: DateTime.now().millisecondsSinceEpoch & 0x7fffffff,
        );
        return _FlutterGemmaChatPort(chat);
      },
    );
  }

  /// Fences the current reply immediately and begins exact chat cleanup.
  ///
  /// This method does not claim that the platform stopped synchronously. A
  /// replacement cannot construct a chat until the old response stream reaches
  /// its real terminal and the exact old chat closes.
  void cancelCurrent() {
    _generationOwner.cancelCurrent();
  }

  Future<void> dispose() {
    final existing = _disposeFuture;
    if (existing != null) {
      return existing;
    }
    final completer = Completer<void>();
    _disposeFuture = completer.future;
    unawaited(_runDispose(completer));
    return completer.future;
  }

  Future<void> _runDispose(Completer<void> completer) async {
    Object? failure;
    StackTrace? failureStack;
    try {
      await _disposeOwned();
    } catch (error, stackTrace) {
      failure = error;
      failureStack = stackTrace;
    }
    if (completer.isCompleted) {
      return;
    }
    if (failure == null) {
      completer.complete();
    } else {
      completer.completeError(failure, failureStack!);
    }
  }

  Future<void> _disposeOwned() async {
    final released = await _generationOwner.close();
    final readiness = _readyFuture;
    if (readiness != null) {
      try {
        await readiness;
      } catch (_) {
        // The owner-close fence makes a late readiness result unusable. Any
        // candidate returned after that fence retires itself below.
      }
    }
    if (!released) {
      throw const GemmaGenerationFailure('generation_cleanup_unproven');
    }
    if (_modelActivationUncertain) {
      throw const GemmaGenerationFailure('model_activation_cleanup_unproven');
    }
    if (_lateModelCloseUncertain) {
      throw const GemmaGenerationFailure('late_model_cleanup_unproven');
    }
    final model = _model;
    if (model != null) {
      await model.close();
      if (identical(_model, model)) {
        _model = null;
      }
    }
  }

  bool get _ownerAcceptsModel =>
      !_generationOwner.isClosed &&
      !_generationOwner.isPoisoned &&
      !_modelActivationUncertain &&
      !_lateModelCloseUncertain &&
      _disposeFuture == null;

  void _requireOpenOwner() {
    if (!_ownerAcceptsModel) {
      throw StateError('GemmaService is disposed or unavailable.');
    }
  }

  Future<void> _retireLateModel(dynamic candidate) async {
    _model ??= candidate;
    try {
      await candidate.close();
    } catch (_) {
      _lateModelCloseUncertain = true;
      rethrow;
    }
    if (identical(_model, candidate)) {
      _model = null;
    }
  }
}

final class _FlutterGemmaChatPort implements GemmaChatPort {
  _FlutterGemmaChatPort(this._chat);

  final dynamic _chat;

  @override
  Future<void> addPrompt(String prompt) async {
    await _chat.addQueryChunk(Message.text(text: prompt, isUser: true));
  }

  @override
  Stream<GemmaGenerationEvent> generate() {
    final Stream<dynamic> source = _chat.generateChatResponseAsync();
    return source.map<GemmaGenerationEvent>((response) {
      if (response is TextResponse) {
        return GemmaGenerationEvent.text(response.token);
      }
      return const GemmaGenerationEvent.nonText();
    });
  }

  @override
  Future<void> stopGeneration() async {
    await _chat.stopGeneration();
  }

  @override
  Future<void> close() async {
    await _chat.close();
  }
}
