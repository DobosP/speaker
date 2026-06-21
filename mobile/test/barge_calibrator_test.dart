// Pure-Dart unit tests for the adaptive barge-in threshold. No device/plugins.
import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/barge_calibrator.dart';

void main() {
  test('quiet room keeps the legacy absolute floor (no regression)', () {
    final c = BargeCalibrator(absoluteMin: 0.08, margin: 2.0, alpha: 0.5);
    for (var i = 0; i < 50; i++) {
      c.observeQuiet(0.01); // very quiet ambient
    }
    // ambient ~0.01 -> adaptive 0.02 < 0.08 -> floored at the old constant.
    expect(c.threshold, closeTo(0.08, 1e-9));
  });

  test('noisy room raises the threshold above ambient', () {
    final c = BargeCalibrator(absoluteMin: 0.08, margin: 2.0, alpha: 0.5);
    for (var i = 0; i < 50; i++) {
      c.observeQuiet(0.06); // loud room
    }
    // ambient ~0.06 -> adaptive 0.12 > 0.08 -> threshold rises so noise can't fire.
    expect(c.threshold, greaterThan(0.10));
    expect(c.ambientFloor, closeTo(0.06, 1e-3));
  });

  test('threshold can only rise above the legacy constant, never below', () {
    final c = BargeCalibrator(absoluteMin: 0.08);
    expect(c.threshold, greaterThanOrEqualTo(0.08)); // before any observation
    c.observeQuiet(0.0);
    expect(c.threshold, greaterThanOrEqualTo(0.08));
  });

  test('EWMA is robust to a single loud transient (a word)', () {
    final c = BargeCalibrator(absoluteMin: 0.0, margin: 1.0, alpha: 0.02);
    for (var i = 0; i < 100; i++) {
      c.observeQuiet(0.01);
    }
    final before = c.ambientFloor;
    c.observeQuiet(0.5); // one loud chunk
    expect(c.ambientFloor, lessThan(before + 0.02)); // floor barely moves
  });

  test('ignores NaN / negative rms', () {
    final c = BargeCalibrator();
    c.observeQuiet(double.nan);
    c.observeQuiet(-1.0);
    expect(c.ambientFloor, 0.0); // unseeded, unchanged
  });
}
