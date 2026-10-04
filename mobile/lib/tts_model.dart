// On-device TTS model config.
// Model: vits-piper-en_US-amy-low (English).
// Downloaded into ./assets/ at build time by tool/download-models.sh.
import 'dart:io';

import 'package:flutter/services.dart' show AssetManifest;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './utils.dart';

const piperModelDirectory = 'vits-piper-en_US-amy-low';

final class TtsModelPaths {
  const TtsModelPaths({
    required this.model,
    required this.tokens,
    required this.dataDir,
  });

  final String model;
  final String tokens;
  final String dataDir;
}

// Both shipped TTS paths stage only their selected model before native startup.
Future<TtsModelPaths> resolveTtsModelPaths({
  AssetManifest? manifest,
  AssetFileCopier? copyFile,
  Future<Directory> Function()? supportDirectory,
}) async {
  await copyAssetFilesInDirectory(
    'assets/$piperModelDirectory',
    manifest: manifest,
    copyFile: copyFile,
  );
  final dir =
      (await (supportDirectory ?? getApplicationSupportDirectory)()).path;
  return TtsModelPaths(
    model: p.join(dir, piperModelDirectory, 'en_US-amy-low.onnx'),
    tokens: p.join(dir, piperModelDirectory, 'tokens.txt'),
    dataDir: p.join(dir, piperModelDirectory, 'espeak-ng-data'),
  );
}

Future<sherpa_onnx.OfflineTts> createOfflineTts() async {
  final paths = await resolveTtsModelPaths();
  sherpa_onnx.initBindings();
  final vits = sherpa_onnx.OfflineTtsVitsModelConfig(
    model: paths.model,
    tokens: paths.tokens,
    dataDir: paths.dataDir,
  );

  final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
    vits: vits,
    numThreads: 2,
    provider: 'cpu',
  );

  final config = sherpa_onnx.OfflineTtsConfig(
    model: modelConfig,
    maxNumSenetences: 1,
  );
  return sherpa_onnx.OfflineTts(config);
}
