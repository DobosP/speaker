// Guards the plugin adapter below the pure-Dart Gemma generation owner. Fakes
// execute no real model, plugin, network, GPU, audio, or device path.
import 'dart:async';

import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/llm.dart';
import 'package:speaker_mobile/llm_generation_owner.dart';

final class _FakeChat {
  _FakeChat({
    Stream<dynamic>? source,
    this.closeFails = false,
  }) : source = source ?? const Stream<dynamic>.empty();

  final Stream<dynamic> source;
  final bool closeFails;
  int addCalls = 0;
  int generateCalls = 0;
  int stopCalls = 0;
  int closeCalls = 0;
  dynamic prompt;

  Future<void> addQueryChunk(dynamic message) async {
    addCalls += 1;
    prompt = message;
  }

  Stream<dynamic> generateChatResponseAsync() {
    generateCalls += 1;
    return source;
  }

  Future<void> stopGeneration() async {
    stopCalls += 1;
  }

  Future<void> close() async {
    closeCalls += 1;
    if (closeFails) {
      throw StateError('chat close failed');
    }
  }
}

final class _RecordingModel {
  _RecordingModel(
    Iterable<_FakeChat> chats, {
    this.closeGate,
    this.closeFails = false,
  }) : _chats = List.of(chats);

  final List<_FakeChat> _chats;
  final Completer<void>? closeGate;
  final bool closeFails;
  final List<Map<String, dynamic>> createChatCalls = [];
  int closeCalls = 0;

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
    if (_chats.isEmpty) {
      throw StateError('no fake chat');
    }
    return _chats.removeAt(0);
  }

  Future<void> close() async {
    closeCalls += 1;
    await closeGate?.future;
    if (closeFails) {
      throw StateError('model close failed');
    }
  }
}

final class _FakeBootstrap implements GemmaModelBootstrapPort {
  _FakeBootstrap({
    this.present = true,
    this.hasModelGate,
    this.downloadGate,
    this.modelPathGate,
    this.installGate,
    this.activation,
  });

  final bool present;
  final Completer<bool>? hasModelGate;
  final Completer<void>? downloadGate;
  final Completer<String>? modelPathGate;
  final Completer<void>? installGate;
  final Future<dynamic> Function(PreferredBackend backend)? activation;
  final List<PreferredBackend> activationBackends = <PreferredBackend>[];
  final Completer<void> hasModelStarted = Completer<void>();
  final Completer<void> downloadStarted = Completer<void>();
  final Completer<void> modelPathStarted = Completer<void>();
  final Completer<void> installStarted = Completer<void>();
  final Completer<void> activationStarted = Completer<void>();
  int hasModelCalls = 0;
  int downloadCalls = 0;
  int modelPathCalls = 0;
  int installCalls = 0;

  @override
  Future<bool> hasModel() async {
    hasModelCalls += 1;
    if (!hasModelStarted.isCompleted) {
      hasModelStarted.complete();
    }
    final gate = hasModelGate;
    return gate == null ? present : gate.future;
  }

  @override
  Future<void> download(
    String url, {
    void Function(double percent)? onProgress,
  }) async {
    downloadCalls += 1;
    if (!downloadStarted.isCompleted) {
      downloadStarted.complete();
    }
    await downloadGate?.future;
  }

  @override
  Future<String> modelPath() async {
    modelPathCalls += 1;
    if (!modelPathStarted.isCompleted) {
      modelPathStarted.complete();
    }
    final gate = modelPathGate;
    return gate == null ? '/fake/model.task' : gate.future;
  }

  @override
  Future<void> install(String path) async {
    installCalls += 1;
    if (!installStarted.isCompleted) {
      installStarted.complete();
    }
    await installGate?.future;
  }

  @override
  Future<dynamic> activate(PreferredBackend backend) async {
    activationBackends.add(backend);
    if (!activationStarted.isCompleted) {
      activationStarted.complete();
    }
    final callback = activation;
    return callback == null ? null : callback(backend);
  }
}

GemmaService _service(_RecordingModel model) {
  return GemmaService.forTesting(
    generationOwner: GemmaGenerationOwner.forTesting(),
    bootstrap: _FakeBootstrap(),
    initialModel: model,
  );
}

void main() {
  test('reply creates and closes one fresh non-greedy chat per turn', () async {
    final first = _FakeChat(
      source: Stream<dynamic>.fromIterable(const [
        TextResponse('Paris'),
        ThinkingResponse('not user-visible'),
      ]),
    );
    final second = _FakeChat(
      source: Stream<dynamic>.fromIterable(const [TextResponse('A joke')]),
    );
    final model = _RecordingModel([first, second]);
    final service = _service(model);

    expect(
      await service.reply('what is the capital of France').toList(),
      ['Paris'],
    );
    expect(await service.reply('tell me a short joke').toList(), ['A joke']);

    expect(model.createChatCalls, hasLength(2));
    for (final call in model.createChatCalls) {
      expect(call['topK'], greaterThan(1));
      expect(call['temperature'], greaterThan(0.0));
    }
    expect(first.addCalls, 1);
    expect(first.generateCalls, 1);
    expect(first.stopCalls, 0);
    expect(first.closeCalls, 1);
    expect(second.closeCalls, 1);
    expect((first.prompt as Message).isUser, isTrue);

    await service.dispose();
    expect(model.closeCalls, 1);
    expect(service.isReady, isFalse);
  });

  test('cancelCurrent invokes adapter stop but waits for source terminal',
      () async {
    final source = StreamController<dynamic>(sync: true);
    final chat = _FakeChat(source: source.stream);
    final model = _RecordingModel([chat]);
    final service = _service(model);
    final subscription = service.reply('keep generating').listen((_) {});
    await Future<void>.delayed(Duration.zero);

    service.cancelCurrent();
    await Future<void>.delayed(Duration.zero);
    expect(chat.stopCalls, 1);
    expect(chat.closeCalls, 0);

    await source.close();
    await Future<void>.delayed(Duration.zero);
    expect(chat.closeCalls, 1);
    await subscription.cancel();
    await service.dispose();
    expect(model.closeCalls, 1);
  });

  test('dispose retains the model when exact chat close is unproved', () async {
    final chat = _FakeChat(closeFails: true);
    final model = _RecordingModel([chat]);
    final service = _service(model);

    await expectLater(
      service.reply('close failure').toList(),
      throwsA(
        isA<GemmaGenerationFailure>().having(
          (error) => error.code,
          'code',
          'chat_close_failed',
        ),
      ),
    );
    await expectLater(
      service.dispose(),
      throwsA(
        isA<GemmaGenerationFailure>().having(
          (error) => error.code,
          'code',
          'generation_cleanup_unproven',
        ),
      ),
    );
    expect(chat.closeCalls, 1);
    expect(model.closeCalls, 0);
    expect(service.isReady, isFalse);
  });

  test('concurrent dispose callers share one model-close transaction',
      () async {
    final closeGate = Completer<void>();
    final model = _RecordingModel(const [], closeGate: closeGate);
    final service = _service(model);

    final first = service.dispose();
    final second = service.dispose();
    expect(identical(first, second), isTrue);
    await Future<void>.delayed(Duration.zero);
    expect(model.closeCalls, 1);
    expect(service.isReady, isFalse);

    closeGate.complete();
    await Future.wait([first, second]);
    expect(model.closeCalls, 1);
  });

  test('concurrent readiness callers share one bootstrap transaction',
      () async {
    final activationGate = Completer<dynamic>();
    final model = _RecordingModel(const []);
    final bootstrap = _FakeBootstrap(
      activation: (_) => activationGate.future,
    );
    final service = GemmaService.forTesting(
      generationOwner: GemmaGenerationOwner.forTesting(),
      bootstrap: bootstrap,
    );

    final first = service.ensureReady();
    final second = service.ensureReady();
    expect(identical(first, second), isTrue);
    await bootstrap.activationStarted.future;
    expect(bootstrap.hasModelCalls, 1);
    expect(bootstrap.modelPathCalls, 1);
    expect(bootstrap.installCalls, 1);
    expect(bootstrap.activationBackends, <PreferredBackend>[
      PreferredBackend.gpu,
    ]);

    activationGate.complete(model);
    await Future.wait([first, second]);
    expect(service.isReady, isTrue);
    await service.dispose();
    expect(model.closeCalls, 1);
  });

  test('only a clean null GPU result permits one CPU fallback', () async {
    final model = _RecordingModel(const []);
    final bootstrap = _FakeBootstrap(
      activation: (backend) async =>
          backend == PreferredBackend.gpu ? null : model,
    );
    final service = GemmaService.forTesting(
      generationOwner: GemmaGenerationOwner.forTesting(),
      bootstrap: bootstrap,
    );

    await service.ensureReady();
    expect(service.isReady, isTrue);
    expect(bootstrap.activationBackends, <PreferredBackend>[
      PreferredBackend.gpu,
      PreferredBackend.cpu,
    ]);
    await service.dispose();
    expect(model.closeCalls, 1);
  });

  for (final failingBackend in <PreferredBackend>[
    PreferredBackend.gpu,
    PreferredBackend.cpu,
  ]) {
    test('$failingBackend activation failure latches ownership uncertainty',
        () async {
      final bootstrap = _FakeBootstrap(
        activation: (backend) async {
          if (backend == failingBackend) {
            throw StateError('activation failed ambiguously');
          }
          return null;
        },
      );
      final service = GemmaService.forTesting(
        generationOwner: GemmaGenerationOwner.forTesting(),
        bootstrap: bootstrap,
      );

      await expectLater(
        service.ensureReady(),
        throwsA(
          isA<GemmaGenerationFailure>().having(
            (error) => error.code,
            'code',
            'model_activation_ambiguous',
          ),
        ),
      );
      final expectedBackends = failingBackend == PreferredBackend.gpu
          ? <PreferredBackend>[PreferredBackend.gpu]
          : <PreferredBackend>[
              PreferredBackend.gpu,
              PreferredBackend.cpu,
            ];
      expect(bootstrap.activationBackends, expectedBackends);
      expect(service.isReady, isFalse);

      expect(service.ensureReady, throwsA(isA<StateError>()));
      expect(bootstrap.activationBackends, expectedBackends);
      await expectLater(
        service.dispose(),
        throwsA(
          isA<GemmaGenerationFailure>().having(
            (error) => error.code,
            'code',
            'model_activation_cleanup_unproven',
          ),
        ),
      );
    });
  }

  test('dispose during held model check prevents download and activation',
      () async {
    final hasModelGate = Completer<bool>();
    final bootstrap = _FakeBootstrap(
      present: false,
      hasModelGate: hasModelGate,
    );
    final service = GemmaService.forTesting(
      generationOwner: GemmaGenerationOwner.forTesting(),
      bootstrap: bootstrap,
    );

    final readiness = service.ensureReady();
    await bootstrap.hasModelStarted.future;
    expect(bootstrap.hasModelCalls, 1);
    final disposal = service.dispose();
    hasModelGate.complete(false);

    await expectLater(readiness, throwsA(isA<StateError>()));
    await disposal;
    expect(bootstrap.downloadCalls, 0);
    expect(bootstrap.modelPathCalls, 0);
    expect(bootstrap.installCalls, 0);
    expect(bootstrap.activationBackends, isEmpty);
  });

  test('dispose during held download prevents later bootstrap stages',
      () async {
    final downloadGate = Completer<void>();
    final bootstrap = _FakeBootstrap(
      present: false,
      downloadGate: downloadGate,
    );
    final service = GemmaService.forTesting(
      generationOwner: GemmaGenerationOwner.forTesting(),
      bootstrap: bootstrap,
    );

    final readiness = service.ensureReady();
    await bootstrap.downloadStarted.future;
    final disposal = service.dispose();
    downloadGate.complete();

    await expectLater(readiness, throwsA(isA<StateError>()));
    await disposal;
    expect(bootstrap.downloadCalls, 1);
    expect(bootstrap.modelPathCalls, 0);
    expect(bootstrap.installCalls, 0);
    expect(bootstrap.activationBackends, isEmpty);
  });

  test('dispose during held model path prevents install and activation',
      () async {
    final modelPathGate = Completer<String>();
    final bootstrap = _FakeBootstrap(modelPathGate: modelPathGate);
    final service = GemmaService.forTesting(
      generationOwner: GemmaGenerationOwner.forTesting(),
      bootstrap: bootstrap,
    );

    final readiness = service.ensureReady();
    await bootstrap.modelPathStarted.future;
    final disposal = service.dispose();
    modelPathGate.complete('/fake/late-model.task');

    await expectLater(readiness, throwsA(isA<StateError>()));
    await disposal;
    expect(bootstrap.modelPathCalls, 1);
    expect(bootstrap.installCalls, 0);
    expect(bootstrap.activationBackends, isEmpty);
  });

  test('dispose during held install prevents activation', () async {
    final installGate = Completer<void>();
    final bootstrap = _FakeBootstrap(installGate: installGate);
    final service = GemmaService.forTesting(
      generationOwner: GemmaGenerationOwner.forTesting(),
      bootstrap: bootstrap,
    );

    final readiness = service.ensureReady();
    await bootstrap.installStarted.future;
    final disposal = service.dispose();
    installGate.complete();

    await expectLater(readiness, throwsA(isA<StateError>()));
    await disposal;
    expect(bootstrap.installCalls, 1);
    expect(bootstrap.activationBackends, isEmpty);
  });

  test('dispose retires a late activation candidate exactly once', () async {
    final activationGate = Completer<dynamic>();
    final candidate = _RecordingModel(const []);
    final bootstrap = _FakeBootstrap(
      activation: (_) => activationGate.future,
    );
    final owner = GemmaGenerationOwner.forTesting();
    final service = GemmaService.forTesting(
      generationOwner: owner,
      bootstrap: bootstrap,
    );

    final readiness = service.ensureReady();
    await bootstrap.activationStarted.future;
    final disposal = service.dispose();
    activationGate.complete(candidate);
    await expectLater(
      readiness,
      throwsA(isA<StateError>()),
    );
    await disposal;
    expect(candidate.closeCalls, 1);
    expect(service.isReady, isFalse);
    expect(bootstrap.activationBackends, <PreferredBackend>[
      PreferredBackend.gpu,
    ]);
  });

  test('uncertain late activation close is retained and never retried',
      () async {
    final activationGate = Completer<dynamic>();
    final candidate = _RecordingModel(const [], closeFails: true);
    final bootstrap = _FakeBootstrap(
      activation: (_) => activationGate.future,
    );
    final service = GemmaService.forTesting(
      generationOwner: GemmaGenerationOwner.forTesting(),
      bootstrap: bootstrap,
    );

    final readiness = service.ensureReady();
    await bootstrap.activationStarted.future;
    final disposal = service.dispose();
    activationGate.complete(candidate);
    await expectLater(readiness, throwsA(isA<StateError>()));
    await expectLater(
      disposal,
      throwsA(
        isA<GemmaGenerationFailure>().having(
          (error) => error.code,
          'code',
          'late_model_cleanup_unproven',
        ),
      ),
    );
    expect(candidate.closeCalls, 1);
  });

  test('failed model close is retained and one-shot across dispose callers',
      () async {
    final model = _RecordingModel(const [], closeFails: true);
    final service = _service(model);

    final first = service.dispose();
    final second = service.dispose();
    expect(identical(first, second), isTrue);
    await expectLater(first, throwsA(isA<StateError>()));
    await expectLater(second, throwsA(isA<StateError>()));
    expect(model.closeCalls, 1);
    expect(service.isReady, isFalse);
  });
}
