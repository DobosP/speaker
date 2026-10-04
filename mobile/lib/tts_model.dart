// On-device TTS model config.
// Model: vits-piper-en_US-amy-low (English).
// Downloaded into ./assets/ at build time by tool/download-models.sh.
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './utils.dart';
import './performance_budget.dart';

// Pure config shared by the direct and worker-native constructors.
sherpa_onnx.OfflineTtsConfig buildPiperTtsConfig({
  required String model,
  required String tokens,
  required String dataDir,
  required int numThreads,
}) => sherpa_onnx.OfflineTtsConfig(
  model: sherpa_onnx.OfflineTtsModelConfig(
    vits: sherpa_onnx.OfflineTtsVitsModelConfig(
      model: model,
      tokens: tokens,
      dataDir: dataDir,
    ),
    numThreads: numThreads,
    provider: 'cpu',
  ),
  maxNumSenetences: 1,
);

Future<sherpa_onnx.OfflineTts> createOfflineTts() async {
  final budget = getStartupPerformanceBudget();
  // sherpa-onnx needs files on disk; copy everything bundled in the APK.
  await copyAllAssetFiles();
  sherpa_onnx.initBindings();

  const modelDir = 'vits-piper-en_US-amy-low';
  final dir = (await getApplicationSupportDirectory()).path;

  return sherpa_onnx.OfflineTts(
    buildPiperTtsConfig(
      model: p.join(dir, modelDir, 'en_US-amy-low.onnx'),
      tokens: p.join(dir, modelDir, 'tokens.txt'),
      dataDir: p.join(dir, modelDir, 'espeak-ng-data'),
      numThreads: budget.ttsThreads,
    ),
  );
}
