import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;
import 'package:speaker_mobile/asr_isolate.dart';
import 'package:speaker_mobile/asr_model.dart';
import 'package:speaker_mobile/performance_budget.dart';
import 'package:speaker_mobile/tts_isolate.dart';
import 'package:speaker_mobile/tts_model.dart';

final class _NativeFactory {
  final List<Map<String, dynamic>> asr = [];
  final List<Map<String, dynamic>> tts = [];

  Object createAsr(sherpa.OnlineRecognizerConfig config) {
    asr.add(config.toJson());
    return Object();
  }

  Object createTts(sherpa.OfflineTtsConfig config) {
    tts.add(config.toJson());
    return Object();
  }
}

Map<String, dynamic> _withoutThreadRequest(Map<String, dynamic> config) {
  final model = Map<String, dynamic>.from(config['model'] as Map);
  model.remove('numThreads');
  return {...config, 'model': model};
}

sherpa.OnlineModelConfig _model(int threads, {String provider = 'cpu'}) =>
    sherpa.OnlineModelConfig(
      transducer: const sherpa.OnlineTransducerModelConfig(
        encoder: '/public/encoder.int8.onnx',
        decoder: '/public/decoder.onnx',
        joiner: '/public/joiner.onnx',
      ),
      tokens: '/public/tokens.txt',
      modelType: 'zipformer2',
      numThreads: threads,
      provider: provider,
    );

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('current and responsive preserve existing thread requests', () {
    for (final name in ['current', 'responsive']) {
      final budget = MobilePerformanceBudget.parse(name);
      expect(budget.mode, name);
      expect(budget.asrThreads, 1);
      expect(budget.ttsThreads, 2);
    }
  });

  test('compact lowers only TTS thread request', () {
    final budget = MobilePerformanceBudget.parse('compact');
    expect(budget.asrThreads, 1);
    expect(budget.ttsThreads, 1);
  });

  test('compiled startup budget is immutable and matches selected mode', () {
    final first = getStartupPerformanceBudget();
    expect(identical(first, getStartupPerformanceBudget()), isTrue);
    expect(
      identical(first, MobilePerformanceBudget.parse(mobilePerformanceMode)),
      isTrue,
    );
  });

  for (final name in [
    '',
    'quality',
    'COMPACT',
    ' compact',
    'compact ',
    'unknown',
  ]) {
    test('unknown mode prevents app and service construction: $name', () {
      var nativeOrAppStarts = 0;
      expect(
        () => startWithValidatedPerformance(name, () => nativeOrAppStarts++),
        throwsArgumentError,
      );
      expect(nativeOrAppStarts, 0);
    });
  }

  test('validated startup invokes app exactly once', () {
    var starts = 0;
    startWithValidatedPerformance('compact', () => starts++);
    expect(starts, 1);
  });

  test(
    'ASR payload reconstruction preserves requested threads and provider',
    () {
      final original = _model(3, provider: 'synthetic-provider');
      final config = rebuildAsrWorkerConfigForTesting(original);
      final factory = _NativeFactory();
      factory.createAsr(config);
      expect(config.model.toJson(), original.toJson());
      expect(config.model.numThreads, 3);
      expect(config.model.provider, 'synthetic-provider');
      expect(config.enableEndpoint, isTrue);
      expect(config.rule2MinTrailingSilence, 0.8);
      expect(config.rule1MinTrailingSilence, 2.4);
      expect(config.rule3MinUtteranceLength, 20);
      expect(config.ruleFsts, '');
      expect(config.decodingMethod, 'greedy_search');
      expect(factory.asr.single['model'], original.toJson());
    },
  );

  test(
    'every mode reaches ASR/TTS fake constructors with fixed other settings',
    () {
      final factory = _NativeFactory();
      final baselineAsr = rebuildAsrWorkerConfigForTesting(_model(1));
      final baselineTts = buildPiperTtsConfig(
        model: '/public/vits-piper-en_US-amy-low/en_US-amy-low.onnx',
        tokens: '/public/vits-piper-en_US-amy-low/tokens.txt',
        dataDir: '/public/vits-piper-en_US-amy-low/espeak-ng-data',
        numThreads: 2,
      );
      for (final name in ['current', 'responsive', 'compact']) {
        final budget = MobilePerformanceBudget.parse(name);
        final asr = rebuildAsrWorkerConfigForTesting(_model(budget.asrThreads));
        final directTts = buildPiperTtsConfig(
          model: baselineTts.model.vits.model,
          tokens: baselineTts.model.vits.tokens,
          dataDir: baselineTts.model.vits.dataDir,
          numThreads: budget.ttsThreads,
        );
        final workerTts = rebuildTtsWorkerConfigForTesting(
          model: baselineTts.model.vits.model,
          tokens: baselineTts.model.vits.tokens,
          dataDir: baselineTts.model.vits.dataDir,
          numThreads: budget.ttsThreads,
        );
        factory.createAsr(asr);
        factory.createTts(directTts);
        factory.createTts(workerTts);
        expect(asr.model.numThreads, budget.asrThreads);
        expect(directTts.model.numThreads, budget.ttsThreads);
        expect(workerTts.model.numThreads, budget.ttsThreads);
        expect(workerTts.toJson(), directTts.toJson());
        expect(
          _withoutThreadRequest(asr.toJson()),
          _withoutThreadRequest(baselineAsr.toJson()),
        );
        expect(
          _withoutThreadRequest(directTts.toJson()),
          _withoutThreadRequest(baselineTts.toJson()),
        );
      }
      expect(factory.asr, hasLength(3));
      expect(factory.tts, hasLength(6));
    },
  );

  test('production ASR asset loader uses compiled startup budget', () async {
    final support = await Directory.systemTemp.createTemp(
      'speaker-budget-assets-',
    );
    final messenger =
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
    const channel = MethodChannel('plugins.flutter.io/path_provider');
    final loaded = <String>[];
    messenger.setMockMethodCallHandler(channel, (call) async => support.path);
    messenger.setMockMessageHandler('flutter/assets', (message) async {
      final source = utf8.decode(
        message!.buffer.asUint8List(
          message.offsetInBytes,
          message.lengthInBytes,
        ),
      );
      loaded.add(source);
      return ByteData.sublistView(Uint8List.fromList([1, 2, 3]));
    });
    addTearDown(() async {
      messenger.setMockMethodCallHandler(channel, null);
      messenger.setMockMessageHandler('flutter/assets', null);
      await support.delete(recursive: true);
    });
    final model = await getOnlineModelConfig();
    expect(model.numThreads, getStartupPerformanceBudget().asrThreads);
    expect(model.provider, 'cpu');
    expect(model.modelType, 'zipformer2');
    expect(loaded, [
      'assets/sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
      'assets/sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx',
      'assets/sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx',
      'assets/sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt',
    ]);
    final config = rebuildAsrWorkerConfigForTesting(model);
    expect(config.model.toJson(), model.toJson());
  });
}
