// Adapted from the official sherpa-onnx Flutter examples (Apache-2.0).
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle, AssetManifest;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

// Copy an asset bundled in the APK to a real file on disk, because
// sherpa-onnx needs filesystem paths. Returns the absolute path written.
Future<String> copyAssetFile(String src, [String? dst]) async {
  final Directory directory = await getApplicationSupportDirectory();
  dst ??= p.basename(src);
  final target = p.join(directory.path, dst);
  final exists = await File(target).exists();

  final data = await rootBundle.load(src);
  if (!exists || File(target).lengthSync() != data.lengthInBytes) {
    final bytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );
    await (await File(target).create(recursive: true)).writeAsBytes(bytes);
  }
  return target;
}

typedef AssetFileCopier =
    Future<String> Function(String source, String destination);

// Stage one model subtree, preserving nested paths such as espeak-ng-data.
// The manifest may also contain unrelated models, fonts and sample audio.
Future<void> copyAssetFilesInDirectory(
  String assetDirectory, {
  AssetManifest? manifest,
  AssetFileCopier? copyFile,
}) async {
  final parts = assetDirectory.split('/');
  if (parts.length < 2 ||
      parts.first != 'assets' ||
      parts.any((part) => part.isEmpty || part == '.' || part == '..') ||
      assetDirectory.contains('\\')) {
    throw ArgumentError.value(assetDirectory, 'assetDirectory');
  }
  final selectedManifest =
      manifest ?? await AssetManifest.loadFromAssetBundle(rootBundle);
  final prefix = '$assetDirectory/';
  final selected = selectedManifest
      .listAssets()
      .where((source) => source.startsWith(prefix))
      .toList(growable: false);
  if (selected.isEmpty) throw StateError('model_assets_unavailable');
  // Validate all destinations before the first write.
  for (final source in selected) {
    if (source.contains('\\') ||
        source
            .split('/')
            .any((part) => part.isEmpty || part == '.' || part == '..')) {
      throw StateError('model_asset_path_invalid');
    }
  }
  final copier =
      copyFile ?? (source, destination) => copyAssetFile(source, destination);
  for (final source in selected) {
    await copier(source, p.joinAll(source.split('/').skip(1)));
  }
}

Float32List convertBytesToFloat32(
  Uint8List bytes, [
  Endian endian = Endian.little,
]) {
  final values = Float32List(bytes.length ~/ 2);
  // sublistView honors this Uint8List's own offset/length. `record` hands back
  // chunks that are often *views* into a larger reused buffer, so the old
  // `ByteData.view(bytes.buffer)` read from the wrong offset and corrupted the
  // audio (a full utterance decoded down to a word or two of garbage).
  final data = ByteData.sublistView(bytes);
  for (var i = 0; i + 1 < bytes.length; i += 2) {
    values[i ~/ 2] = data.getInt16(i, endian) / 32768.0;
  }
  return values;
}

int _waveFileSequence = 0;

Future<String> generateWaveFilename([String suffix = '']) async {
  final Directory directory = await getApplicationSupportDirectory();
  final now = DateTime.now();
  String two(int v) => v.toString().padLeft(2, '0');
  final filename =
      '${now.year}-${two(now.month)}-${two(now.day)}-${two(now.hour)}-${two(now.minute)}-${two(now.second)}-${now.microsecondsSinceEpoch}-${_waveFileSequence++}$suffix.wav';
  return p.join(directory.path, filename);
}
