import 'dart:async';

/// App-lifetime speech-output admission for the canonical Flutter UI isolate.
///
/// Every shipped UI speech path shares [ttsProcessOwnerRegistry]. This is not
/// an operating-system-wide lock and does not coordinate a second Flutter
/// engine or an independently spawned Dart isolate.
final TtsProcessOwnerRegistry ttsProcessOwnerRegistry =
    TtsProcessOwnerRegistry();

/// Owns the single speech-output lease for one app/UI-isolate lifetime.
///
/// Admission is deliberately non-blocking. Once cleanup is uncertain, the
/// registry stays poisoned until the app process is restarted; a replacement
/// must never overlap native synthesis or playback whose return is unproved.
final class TtsProcessOwnerRegistry {
  TtsProcessLease? _current;
  bool _poisoned = false;

  bool get poisoned => _poisoned;
  bool get busy => _current != null;

  /// Whether [lease] is still the exact current holder, including after revoke.
  bool holdsExact(TtsProcessLease lease) => identical(_current, lease);

  /// Whether [lease] is the exact current authority of this registry.
  bool ownsExact(TtsProcessLease lease) =>
      !_poisoned && holdsExact(lease) && lease.admitsWork;

  TtsProcessLease? tryAcquire() {
    if (_poisoned || _current != null) return null;
    final lease = TtsProcessLease._(this);
    _current = lease;
    return lease;
  }

  bool _releaseExact(TtsProcessLease lease) {
    if (!identical(_current, lease) || _poisoned) return false;
    _current = null;
    return true;
  }

  void _poisonExact(TtsProcessLease lease) {
    if (!identical(_current, lease)) return;
    _poisoned = true;
  }
}

/// Opaque exact authority for one admitted speech-output lifetime.
final class TtsProcessLease {
  TtsProcessLease._(this._registry);

  final TtsProcessOwnerRegistry _registry;
  bool _revoked = false;
  Future<bool>? _closeResult;

  bool get revoked => _revoked;

  bool get admitsWork =>
      !_revoked && !_registry._poisoned && identical(_registry._current, this);

  /// Remove new-work authority synchronously without releasing ownership.
  void revoke() {
    _revoked = true;
  }

  /// Revoke admission, run exact cleanup once, and memoize its receipt.
  ///
  /// Only an exact `true` cleanup receipt releases this exact lease. A false
  /// receipt or thrown cleanup error permanently poisons the registry. A
  /// cleanup future that never completes keeps the lease busy indefinitely.
  Future<bool> close(Future<bool> Function() cleanup) {
    final existing = _closeResult;
    if (existing != null) return existing;

    final completed = Completer<bool>();
    _closeResult = completed.future;
    revoke();
    unawaited(_runCleanup(cleanup, completed));
    return completed.future;
  }

  Future<void> _runCleanup(
    Future<bool> Function() cleanup,
    Completer<bool> completed,
  ) async {
    var succeeded = false;
    try {
      succeeded = await cleanup();
    } catch (_) {
      succeeded = false;
    }

    if (succeeded && _registry._releaseExact(this)) {
      completed.complete(true);
      return;
    }
    _registry._poisonExact(this);
    completed.complete(false);
  }
}
