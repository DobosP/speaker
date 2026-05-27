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
    final bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await (await File(target).create(recursive: true)).writeAsBytes(bytes);
  }
  return target;
}

Future<List<String>> _getAllAssetFiles() async {
  final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
  return manifest.listAssets();
}

String _stripLeadingDirectory(String src, {int n = 1}) {
  return p.joinAll(p.split(src).sublist(n));
}

// Copy every bundled asset to disk, preserving its directory layout minus the
// leading "assets/" segment. Used by the TTS model loader.
Future<void> copyAllAssetFiles() async {
  for (final src in await _getAllAssetFiles()) {
    await copyAssetFile(src, _stripLeadingDirectory(src));
  }
}

Float32List convertBytesToFloat32(Uint8List bytes, [Endian endian = Endian.little]) {
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

Future<String> generateWaveFilename([String suffix = '']) async {
  final Directory directory = await getApplicationSupportDirectory();
  final now = DateTime.now();
  String two(int v) => v.toString().padLeft(2, '0');
  final filename =
      '${now.year}-${two(now.month)}-${two(now.day)}-${two(now.hour)}-${two(now.minute)}-${two(now.second)}$suffix.wav';
  return p.join(directory.path, filename);
}
