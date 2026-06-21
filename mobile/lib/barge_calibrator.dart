// Adaptive barge-in threshold for the mobile energy gate.
//
// The energy barge-in used a FIXED near-end RMS threshold (0.08): too sensitive
// in a noisy room (false self-interrupts on background noise) and not adaptive to
// the device/room at all. This learns the room's ambient floor from the
// echo-FREE windows (while the assistant is NOT speaking) and raises the bar to a
// margin above it -- so a real talk-over still clears it but room noise does not.
//
// Pure Dart (no plugins/device): unit-tested in mobile/test/barge_calibrator_test.dart.
// The `absoluteMin` floor guarantees the threshold can only become LESS sensitive
// than today's constant, so the default (quiet-room) behavior never regresses.
import 'dart:math' as math;

class BargeCalibrator {
  BargeCalibrator({
    this.absoluteMin = 0.08, // the legacy fixed threshold -> a hard floor
    this.margin = 2.0, // ~6 dB above the learned ambient
    this.alpha = 0.02, // EWMA rate: slow, so a stray word doesn't spike it
  });

  /// The threshold never drops below this (no regression vs the old constant).
  final double absoluteMin;

  /// Adaptive threshold = ambient floor * margin.
  final double margin;

  /// EWMA smoothing factor for the ambient estimate (0..1; smaller = slower).
  final double alpha;

  double _floor = 0.0;
  bool _seeded = false;

  /// Feed a near-end RMS measured while the assistant is NOT speaking (so the mic
  /// hears the room, not the assistant's echo). EWMA so a brief sound doesn't move
  /// the floor much.
  void observeQuiet(double rms) {
    if (rms.isNaN || rms < 0) return;
    if (!_seeded) {
      _floor = rms;
      _seeded = true;
    } else {
      _floor = (1.0 - alpha) * _floor + alpha * rms;
    }
  }

  /// The current barge-in RMS threshold: the larger of the absolute floor and a
  /// margin above the learned ambient. A real talk-over clears it; steady room
  /// noise (which the ambient estimate has absorbed) does not.
  double get threshold => math.max(absoluteMin, _floor * margin);

  /// The learned ambient floor (for logging / tests).
  double get ambientFloor => _floor;
}
