// On-device streaming ASR model config.
// Model: sherpa-onnx-streaming-zipformer-en-2023-06-26 (English, int8).
// Downloaded into ./assets/ at build time by tool/download-models.sh.
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './utils.dart';
import './performance_budget.dart';

Future<sherpa_onnx.OnlineModelConfig> getOnlineModelConfig() async {
  final budget = getStartupPerformanceBudget();
  const modelDir = 'assets/sherpa-onnx-streaming-zipformer-en-2023-06-26';
  return sherpa_onnx.OnlineModelConfig(
    transducer: sherpa_onnx.OnlineTransducerModelConfig(
      encoder: await copyAssetFile(
        '$modelDir/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
      ),
      decoder: await copyAssetFile(
        '$modelDir/decoder-epoch-99-avg-1-chunk-16-left-128.onnx',
      ),
      joiner: await copyAssetFile(
        '$modelDir/joiner-epoch-99-avg-1-chunk-16-left-128.onnx',
      ),
    ),
    tokens: await copyAssetFile('$modelDir/tokens.txt'),
    modelType: 'zipformer2',
    numThreads: budget.asrThreads,
    provider: 'cpu',
  );
}

// Optional offline Whisper base.en configuration for explicit future/evidence use.
// The shipped assistant uses streaming Zipformer; this helper has no live caller.
Future<sherpa_onnx.OfflineModelConfig> getOfflineWhisperConfig() async {
  const modelDir = 'assets/sherpa-onnx-whisper-base.en';
  return sherpa_onnx.OfflineModelConfig(
    whisper: sherpa_onnx.OfflineWhisperModelConfig(
      encoder: await copyAssetFile('$modelDir/base.en-encoder.int8.onnx'),
      decoder: await copyAssetFile('$modelDir/base.en-decoder.int8.onnx'),
    ),
    tokens: await copyAssetFile('$modelDir/base.en-tokens.txt'),
    modelType: 'whisper',
  );
}
