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
    this.maxAmbientFloor = 0.12, // contamination/drift cap for mobile mics
  });

  /// The threshold never drops below this (no regression vs the old constant).
  final double absoluteMin;

  /// Adaptive threshold = ambient floor * margin.
  final double margin;

  /// EWMA smoothing factor for the ambient estimate (0..1; smaller = slower).
  final double alpha;

  /// Upper bound for the learned ambient floor. This keeps accidental training
  /// on speech or playback tails from making barge-in unreachable for the rest
  /// of the session. Set to null to disable the cap in tests/experiments.
  final double? maxAmbientFloor;

  double _floor = 0.0;
  bool _seeded = false;

  /// Feed a near-end RMS measured while the assistant is NOT speaking (so the mic
  /// hears the room, not the assistant's echo). EWMA so a brief sound doesn't move
  /// the floor much.
  void observeQuiet(double rms) {
    if (!rms.isFinite || rms < 0) return;
    final capped = maxAmbientFloor == null ? rms : math.min(rms, maxAmbientFloor!);
    if (!_seeded) {
      _floor = capped;
      _seeded = true;
    } else {
      final next = (1.0 - alpha) * _floor + alpha * capped;
      _floor = maxAmbientFloor == null ? next : math.min(next, maxAmbientFloor!);
    }
  }

  /// The current barge-in RMS threshold: the larger of the absolute floor and a
  /// margin above the learned ambient. A real talk-over clears it; steady room
  /// noise (which the ambient estimate has absorbed) does not.
  double get threshold => math.max(absoluteMin, _floor * margin);

  /// The learned ambient floor (for logging / tests).
  double get ambientFloor => _floor;
}

/// Pure-Dart guard for deciding when a mic chunk is safe to use as ambient
/// training data.
///
/// The assistant can have `_speaking == false` while the user is still talking
/// before endpoint, or while the player's acoustic tail is still draining after
/// stop/complete. Those chunks are not room tone; learning from them raises the
/// barge threshold and makes later talk-over harder to detect.
class QuietObservationGate {
  QuietObservationGate({
    this.speechCooldown = const Duration(milliseconds: 500),
    this.playbackCooldown = const Duration(milliseconds: 500),
  });

  final Duration speechCooldown;
  final Duration playbackCooldown;

  DateTime? _lastVoiceAt;
  DateTime? _playbackQuietAfter;
  bool _asrInFlight = false;

  void noteVoice(DateTime now) {
    _lastVoiceAt = now;
  }

  void noteAsrStarted(DateTime now) {
    _asrInFlight = true;
    noteVoice(now);
  }

  void noteAsrFinished(DateTime now, {bool hadSpeech = true}) {
    final wasInFlight = _asrInFlight;
    _asrInFlight = false;
    if (wasInFlight || hadSpeech) noteVoice(now);
  }

  void resetAsr() {
    _asrInFlight = false;
  }

  void notePlaybackStopped(DateTime now) {
    _playbackQuietAfter = now.add(playbackCooldown);
  }

  bool canObserveQuiet(DateTime now) {
    if (_asrInFlight) return false;
    final lastVoice = _lastVoiceAt;
    if (lastVoice != null && now.difference(lastVoice) < speechCooldown) {
      return false;
    }
    final quietAfter = _playbackQuietAfter;
    if (quietAfter != null && now.isBefore(quietAfter)) {
      return false;
    }
    return true;
  }

  bool get asrInFlight => _asrInFlight;
}
