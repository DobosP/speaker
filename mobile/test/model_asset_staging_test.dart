import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/asr_model.dart';
import 'package:speaker_mobile/tts_model.dart';
import 'package:speaker_mobile/utils.dart';

final class _Manifest implements AssetManifest {
  _Manifest(this.assets);
  final List<String> assets;

  @override
  List<String> listAssets() => assets;

  @override
  List<AssetMetadata>? getAssetVariants(String key) => null;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const tts = 'assets/vits-piper-en_US-amy-low';
  const asr = 'assets/sherpa-onnx-streaming-zipformer-en-2023-06-26';
  const whisper = 'assets/sherpa-onnx-whisper-base.en';
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  const pathChannel = MethodChannel('plugins.flutter.io/path_provider');
  late Directory support;
  late List<String> loaded;
  late Map<String, List<int>> payloads;

  setUp(() async {
    support = await Directory.systemTemp.createTemp('speaker-model-assets-');
    loaded = [];
    payloads = {
      '$tts/en_US-amy-low.onnx': [1, 2, 3],
      '$tts/tokens.txt': [4, 5],
      '$tts/espeak-ng-data/phontab': [6, 7],
      '$tts/espeak-ng-data/lang/gmw/en': [8, 9],
      '$asr/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx': [10],
      '$asr/decoder-epoch-99-avg-1-chunk-16-left-128.onnx': [11],
      '$asr/joiner-epoch-99-avg-1-chunk-16-left-128.onnx': [12],
      '$asr/tokens.txt': [13],
      '$whisper/base.en-encoder.int8.onnx': [14],
      '$whisper/base.en-decoder.int8.onnx': [15],
      '$whisper/base.en-tokens.txt': [16],
      '$tts-other/model.onnx': [17],
      'assets/sample.wav': [18],
      'fonts/sample.ttf': [19],
    };
    messenger.setMockMethodCallHandler(pathChannel, (call) async {
      if (call.method == 'getApplicationSupportDirectory') return support.path;
      throw StateError('unexpected directory request');
    });
    messenger.setMockMessageHandler('flutter/assets', (message) async {
      final name = utf8.decode(
        message!.buffer.asUint8List(
          message.offsetInBytes,
          message.lengthInBytes,
        ),
      );
      loaded.add(name);
      final bytes = payloads[name];
      if (bytes == null) return null;
      return ByteData.sublistView(Uint8List.fromList(bytes));
    });
  });

  tearDown(() async {
    messenger.setMockMethodCallHandler(pathChannel, null);
    messenger.setMockMessageHandler('flutter/assets', null);
    await support.delete(recursive: true);
  });

  test('TTS stages only its voice and preserves nested espeak paths', () async {
    final paths = await resolveTtsModelPaths(
      manifest: _Manifest(payloads.keys.toList()),
    );
    final expected = payloads.keys
        .where((source) => source.startsWith('$tts/'))
        .toList();
    expect(loaded, expected);
    expect(
      paths.model,
      '${support.path}/vits-piper-en_US-amy-low/en_US-amy-low.onnx',
    );
    expect(paths.tokens, '${support.path}/vits-piper-en_US-amy-low/tokens.txt');
    expect(
      paths.dataDir,
      '${support.path}/vits-piper-en_US-amy-low/espeak-ng-data',
    );
    for (final source in expected) {
      final file = File(
        '${support.path}/${source.substring('assets/'.length)}',
      );
      expect(await file.readAsBytes(), payloads[source]);
    }
    expect(
      await Directory('${support.path}/sherpa-onnx-whisper-base.en').exists(),
      isFalse,
    );
    expect(
      await Directory(
        '${support.path}/sherpa-onnx-streaming-zipformer-en-2023-06-26',
      ).exists(),
      isFalse,
    );
  });

  test(
    'ASR keeps the exact existing hybrid tuple and no optional assets',
    () async {
      final config = await getOnlineModelConfig();
      expect(loaded, [
        '$asr/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
        '$asr/decoder-epoch-99-avg-1-chunk-16-left-128.onnx',
        '$asr/joiner-epoch-99-avg-1-chunk-16-left-128.onnx',
        '$asr/tokens.txt',
      ]);
      expect(config.modelType, 'zipformer2');
      expect(
        config.transducer.encoder,
        '${support.path}/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
      );
      expect(
        config.transducer.decoder,
        '${support.path}/decoder-epoch-99-avg-1-chunk-16-left-128.onnx',
      );
      expect(
        config.transducer.joiner,
        '${support.path}/joiner-epoch-99-avg-1-chunk-16-left-128.onnx',
      );
      expect(config.tokens, '${support.path}/tokens.txt');
      for (final path in [
        config.transducer.encoder,
        config.transducer.decoder,
        config.transducer.joiner,
        config.tokens,
      ]) {
        expect(await File(path).exists(), isTrue);
      }
    },
  );

  test('optional Whisper configuration still resolves explicitly', () async {
    final config = await getOfflineWhisperConfig();
    expect(loaded, [
      '$whisper/base.en-encoder.int8.onnx',
      '$whisper/base.en-decoder.int8.onnx',
      '$whisper/base.en-tokens.txt',
    ]);
    expect(config.modelType, 'whisper');
    expect(await File(config.whisper.encoder).exists(), isTrue);
    expect(await File(config.whisper.decoder).exists(), isTrue);
    expect(await File(config.tokens).exists(), isTrue);
  });

  test('missing selected subtree fails before any asset load', () async {
    await expectLater(
      resolveTtsModelPaths(manifest: _Manifest(['$asr/tokens.txt'])),
      throwsStateError,
    );
    expect(loaded, isEmpty);
  });

  for (final directory in [
    'assets',
    '/assets/voice',
    'assets/../voice',
    'assets/voice/',
    r'assets\voice',
  ]) {
    test(
      'invalid selected directory fails before asset writes: $directory',
      () async {
        await expectLater(
          copyAssetFilesInDirectory(
            directory,
            manifest: _Manifest(payloads.keys.toList()),
          ),
          throwsArgumentError,
        );
        expect(loaded, isEmpty);
      },
    );
  }

  test('unsafe selected manifest path fails before all asset writes', () async {
    await expectLater(
      copyAssetFilesInDirectory(
        tts,
        manifest: _Manifest([
          '$tts/en_US-amy-low.onnx',
          '$tts/../outside.onnx',
        ]),
      ),
      throwsStateError,
    );
    expect(loaded, isEmpty);
  });
}
