// Immutable per-engine native thread requests, selected before app startup.
// These are pool settings, not measured latency or an OS-wide CPU cap.
final class MobilePerformanceBudget {
  const MobilePerformanceBudget._(this.mode, this.asrThreads, this.ttsThreads);

  final String mode;
  final int asrThreads;
  final int ttsThreads;

  static const current = MobilePerformanceBudget._('current', 1, 2);
  static const responsive = MobilePerformanceBudget._('responsive', 1, 2);
  static const compact = MobilePerformanceBudget._('compact', 1, 1);

  static MobilePerformanceBudget parse(String mode) => switch (mode) {
    'current' => current,
    'responsive' => responsive,
    'compact' => compact,
    _ => throw ArgumentError('unsupported_mobile_performance'),
  };
}

const mobilePerformanceMode = String.fromEnvironment(
  'SPEAKER_PERFORMANCE',
  defaultValue: 'current',
);

final _startupBudget = MobilePerformanceBudget.parse(mobilePerformanceMode);

MobilePerformanceBudget getStartupPerformanceBudget() => _startupBudget;

// The entry point calls this before bindings, plugins, widgets or services.
void startWithValidatedPerformance(String mode, void Function() start) {
  MobilePerformanceBudget.parse(mode);
  start();
}
