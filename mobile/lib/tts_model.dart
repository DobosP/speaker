// On-device TTS model config.
// Model: vits-piper-en_US-amy-low (English).
// Downloaded into ./assets/ at build time by tool/download-models.sh.
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './utils.dart';

Future<sherpa_onnx.OfflineTts> createOfflineTts() async {
  // sherpa-onnx needs files on disk; copy everything bundled in the APK.
  await copyAllAssetFiles();
  sherpa_onnx.initBindings();

  const modelDir = 'vits-piper-en_US-amy-low';
  final dir = (await getApplicationSupportDirectory()).path;

  final vits = sherpa_onnx.OfflineTtsVitsModelConfig(
    model: p.join(dir, modelDir, 'en_US-amy-low.onnx'),
    tokens: p.join(dir, modelDir, 'tokens.txt'),
    dataDir: p.join(dir, modelDir, 'espeak-ng-data'),
  );

  final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
    vits: vits,
    numThreads: 2,
    provider: 'cpu',
  );

  final config = sherpa_onnx.OfflineTtsConfig(model: modelConfig, maxNumSenetences: 1);
  return sherpa_onnx.OfflineTts(config);
}
