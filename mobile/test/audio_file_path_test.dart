import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/utils.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('overlapping synthesis requests receive distinct audio paths', () async {
    const channel = MethodChannel('plugins.flutter.io/path_provider');
    final messenger =
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(channel, (call) async {
      if (call.method == 'getApplicationSupportDirectory') {
        return '/speaker-mobile-test';
      }
      throw StateError('unexpected directory request');
    });
    addTearDown(() => messenger.setMockMethodCallHandler(channel, null));

    final paths = await Future.wait(
      List.generate(64, (_) => generateWaveFilename()),
    );
    expect(paths.toSet(), hasLength(paths.length));
    expect(paths.every((path) => path.startsWith('/speaker-mobile-test/')),
        isTrue);
    expect(paths.every((path) => path.endsWith('.wav')), isTrue);
  });
}
