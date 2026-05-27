// Where the on-device Gemma weights live, and how they get there.
//
// The whole point of this file is to make the ~550 MB model survive app
// reinstalls so it is downloaded from the internet at most ONCE. We keep the
// weights at a fixed, predictable path that:
//   * the app can read/write with no runtime permission, and
//   * `adb push` can write to from a dev machine (see tool/push-model.sh),
// then load the model from that file instead of re-downloading every run.
//
// Android resolves this to the app's external files dir
// (/sdcard/Android/data/<applicationId>/files/models/<file>). That directory
// survives `adb install -r` (install-in-place) and is reachable over adb, so
// the fast test loop is: push the model once, then reinstall the APK freely.
import 'dart:io';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class ModelStore {
  ModelStore._();

  // On-disk filename of the Gemma .task bundle. tool/push-model.sh pushes the
  // file under this exact name, so keep the two in sync.
  static const fileName = 'Gemma3-1B-IT-q4.task';

  // A file smaller than this is treated as truncated/incomplete (a half-done
  // download or a botched push) and re-fetched rather than handed to the
  // engine. The real file is ~550 MB; this is just a sanity floor that tolerates
  // a future model swap while rejecting an obviously-partial file.
  static const _minValidBytes = 100 * 1024 * 1024; // 100 MB

  static Directory? _resolvedDir;

  // The directory the weights live in. Prefers the app's external files dir on
  // Android (adb-pushable, no permission, survives install -r); falls back to
  // app-support storage where there is no external dir (e.g. iOS).
  static Future<Directory> _modelDir() async {
    if (_resolvedDir != null) return _resolvedDir!;
    Directory base;
    try {
      base = (await getExternalStorageDirectory()) ??
          await getApplicationSupportDirectory();
    } catch (_) {
      base = await getApplicationSupportDirectory();
    }
    final dir = Directory(p.join(base.path, 'models'));
    await dir.create(recursive: true);
    debugPrint('ModelStore: model directory is ${dir.path}');
    _resolvedDir = dir;
    return dir;
  }

  // Absolute path the engine loads from and the dev tooling pushes to.
  static Future<File> modelFile() async {
    final dir = await _modelDir();
    return File(p.join(dir.path, fileName));
  }

  // True when a plausibly-complete model is already on disk — the check that
  // lets us skip the network entirely on reinstall.
  static Future<bool> hasModel() async {
    final f = await modelFile();
    if (!await f.exists()) return false;
    return await f.length() >= _minValidBytes;
  }

  // Download [url] to the stable path, reporting 0..100 progress. Streams to a
  // .part file and renames on success, so an interrupted download is never
  // mistaken for a complete model on the next launch.
  static Future<File> download(
    String url, {
    void Function(double percent)? onProgress,
  }) async {
    final dest = await modelFile();
    final part = File('${dest.path}.part');
    if (await part.exists()) await part.delete();

    final client = HttpClient();
    try {
      final request = await client.getUrl(Uri.parse(url));
      final response = await request.close(); // follows redirects by default
      if (response.statusCode != HttpStatus.ok) {
        throw HttpException('GET $url failed: HTTP ${response.statusCode}');
      }
      final total = response.contentLength; // -1 when the server omits it
      var received = 0;
      final sink = part.openWrite();
      try {
        await for (final chunk in response) {
          sink.add(chunk);
          received += chunk.length;
          if (total > 0) onProgress?.call(received * 100.0 / total);
        }
      } finally {
        await sink.flush();
        await sink.close();
      }
    } finally {
      client.close();
    }

    if (await dest.exists()) await dest.delete();
    await part.rename(dest.path);
    return dest;
  }
}
