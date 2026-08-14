// Generation-scoped ownership for the mobile streaming-TTS queue.
//
// This file is deliberately pure Dart. The Assistant screen supplies exact
// native clip handles, while deterministic tests control every await boundary
// without loading plugins, models, or audio devices.
import 'dart:async';
import 'dart:collection';

typedef TtsSynthesize = Future<String?> Function(String text);
typedef TtsCreatePlaybackClip = TtsPlaybackClip Function(String path);
typedef TtsPlaybackActivityChanged = void Function(
  TtsPlaybackGeneration generation,
  bool speaking,
);
typedef TtsPlaybackStarted = void Function(
  TtsPlaybackGeneration generation,
);
typedef TtsPlaybackError = void Function(
  TtsPlaybackGeneration generation,
  Object error,
  StackTrace stackTrace,
);

/// Opaque identity for one admitted mobile speech generation.
///
/// Callers may retain and pass this token back to [TtsPlaybackOwner], but only
/// the owner that created the exact object can accept it.
final class TtsPlaybackGeneration {
  TtsPlaybackGeneration._(this._ownerKey, this.ordinal, this._stopFence);

  final Object _ownerKey;
  final Future<bool> _stopFence;

  /// Monotonic diagnostic value. Authority is object identity, not this number.
  final int ordinal;

  @override
  String toString() => 'TtsPlaybackGeneration($ordinal)';
}

/// A native operation result represented as data, never an erroring Future.
final class TtsPlaybackResult {
  const TtsPlaybackResult.success({this.error, this.stackTrace})
      : succeeded = true;

  const TtsPlaybackResult.failure(this.error, this.stackTrace)
      : succeeded = false;

  final bool succeeded;

  /// Optional diagnostic. A successful cleanup may still report a recovered
  /// stop error when a later exact dispose proved the clip quiescent.
  final Object? error;
  final StackTrace? stackTrace;
}

enum TtsPlaybackTerminalKind { completed, interrupted, failed }

/// Exact terminal receipt for one native clip.
final class TtsPlaybackTerminal {
  const TtsPlaybackTerminal.completed()
      : kind = TtsPlaybackTerminalKind.completed,
        error = null,
        stackTrace = null;

  const TtsPlaybackTerminal.interrupted()
      : kind = TtsPlaybackTerminalKind.interrupted,
        error = null,
        stackTrace = null;

  const TtsPlaybackTerminal.failed(this.error, this.stackTrace)
      : kind = TtsPlaybackTerminalKind.failed;

  final TtsPlaybackTerminalKind kind;
  final Object? error;
  final StackTrace? stackTrace;
}

/// Synchronously published ownership handle for one native player instance.
///
/// The adapter must install and return this object before native route/play
/// admission. Player construction may already have started plugin setup. Both
/// [started] and [terminal] are normalized to data. [stopAndRelease] is
/// memoized before its callback runs, so natural completion, interruption, and
/// close share one exact cleanup transaction.
final class TtsPlaybackClip {
  TtsPlaybackClip({
    required Future<TtsPlaybackResult> started,
    required Future<TtsPlaybackTerminal> terminal,
    required Future<TtsPlaybackResult> Function() stopAndRelease,
  })  : started = _guardResult(started),
        terminal = _guardTerminal(terminal),
        _stopAndRelease = stopAndRelease;

  final Future<TtsPlaybackResult> started;
  final Future<TtsPlaybackTerminal> terminal;
  final Future<TtsPlaybackResult> Function() _stopAndRelease;
  Future<TtsPlaybackResult>? _cleanupResult;

  Future<TtsPlaybackResult> stopAndRelease() {
    final existing = _cleanupResult;
    if (existing != null) return existing;

    final completed = Completer<TtsPlaybackResult>();
    _cleanupResult = completed.future;
    unawaited(
      Future<TtsPlaybackResult>.sync(_stopAndRelease).then(
        completed.complete,
        onError: (Object error, StackTrace stackTrace) {
          completed.complete(TtsPlaybackResult.failure(error, stackTrace));
        },
      ),
    );
    return completed.future;
  }

  static Future<TtsPlaybackResult> _guardResult(
    Future<TtsPlaybackResult> result,
  ) =>
      result.then(
        (value) => value,
        onError: (Object error, StackTrace stackTrace) =>
            TtsPlaybackResult.failure(error, stackTrace),
      );

  static Future<TtsPlaybackTerminal> _guardTerminal(
    Future<TtsPlaybackTerminal> terminal,
  ) =>
      terminal.then(
        (value) => value,
        onError: (Object error, StackTrace stackTrace) =>
            TtsPlaybackTerminal.failed(error, stackTrace),
      );
}

/// Immutable diagnostics for tests and UI-independent lifecycle inspection.
final class TtsPlaybackSnapshot {
  const TtsPlaybackSnapshot({
    required this.generation,
    required this.speaking,
    required this.logicalWork,
    required this.physicalActive,
    required this.poisoned,
    required this.pumpRunning,
    required this.queued,
    required this.closed,
  });

  final int generation;
  final bool speaking;
  final bool logicalWork;
  final bool physicalActive;
  final bool poisoned;
  final bool pumpRunning;
  final int queued;
  final bool closed;
}

final class _QueuedSpeech {
  const _QueuedSpeech(this.generation, this.text);

  final TtsPlaybackGeneration generation;
  final String text;
}

final class _ClipInterrupted {
  const _ClipInterrupted(this.stopFence);

  final Future<bool> stopFence;
}

final class _ClipTerminal {
  const _ClipTerminal(this.terminal);

  final TtsPlaybackTerminal terminal;
}

final class _ActiveClip {
  _ActiveClip(this.generation);

  final TtsPlaybackGeneration generation;
  final Completer<_ClipInterrupted> interrupt = Completer<_ClipInterrupted>();
  TtsPlaybackClip? clip;
  bool cleanupReported = false;
}

enum _PlaybackStartDisposition { started, stale, failed }

final class _PlaybackStartOutcome {
  const _PlaybackStartOutcome(this.disposition);

  final _PlaybackStartDisposition disposition;
}

/// Owns one sequential synth/play pump across every mobile reply generation.
///
/// Superseding speech invalidates old queue entries synchronously. Native
/// synthesis already in flight is not cancelled, but its result is rechecked
/// and discarded before player admission. Replacement playback also waits for
/// the exact prior clip cleanup fence, so a late stop cannot kill new audio.
final class TtsPlaybackOwner {
  TtsPlaybackOwner({
    required TtsSynthesize synthesize,
    required TtsCreatePlaybackClip createPlaybackClip,
    TtsPlaybackActivityChanged? onActivityChanged,
    TtsPlaybackStarted? onPlaybackStarted,
    TtsPlaybackError? onError,
  })  : _synthesize = synthesize,
        _createPlaybackClip = createPlaybackClip,
        _onActivityChanged = onActivityChanged,
        _onPlaybackStarted = onPlaybackStarted,
        _onError = onError {
    _current = TtsPlaybackGeneration._(
      _ownerKey,
      _nextOrdinal,
      Future<bool>.value(true),
    );
  }

  final TtsSynthesize _synthesize;
  final TtsCreatePlaybackClip _createPlaybackClip;
  final TtsPlaybackActivityChanged? _onActivityChanged;
  final TtsPlaybackStarted? _onPlaybackStarted;
  final TtsPlaybackError? _onError;

  final Object _ownerKey = Object();
  final Queue<_QueuedSpeech> _queue = Queue<_QueuedSpeech>();
  late TtsPlaybackGeneration _current;
  int _nextOrdinal = 0;
  bool _logicalWork = false;
  bool _physicalActive = false;
  bool _poisoned = false;
  bool _speaking = false;
  bool _closed = false;
  bool _pumpRunning = false;
  Future<void>? _pumpFuture;
  Future<bool>? _closeResult;
  Future<void> _playerOperationTail = Future<void>.value();
  _ActiveClip? _activeClip;
  TtsPlaybackGeneration? _publishedGeneration;
  bool? _publishedSpeaking;

  TtsPlaybackGeneration get generation => _current;
  bool get speaking => _speaking;

  TtsPlaybackSnapshot get snapshot => TtsPlaybackSnapshot(
        generation: _current.ordinal,
        speaking: _speaking,
        logicalWork: _logicalWork,
        physicalActive: _physicalActive,
        poisoned: _poisoned,
        pumpRunning: _pumpRunning,
        queued: _queue.length,
        closed: _closed,
      );

  bool isCurrent(TtsPlaybackGeneration generation) =>
      identical(generation._ownerKey, _ownerKey) &&
      identical(generation, _current) &&
      !_closed;

  /// Invalidate all older speech immediately and return the replacement token.
  TtsPlaybackGeneration supersede() {
    if (_closed) throw StateError('TTS playback owner is closed');
    return _advanceGeneration();
  }

  /// Invalidate current speech and await its exact physical cleanup fence.
  Future<bool> interrupt() {
    if (_closed) return Future<bool>.value(false);
    return _advanceGeneration()._stopFence;
  }

  /// Wait for the clip-cleanup operation associated with [generation].
  Future<bool> waitForStop(TtsPlaybackGeneration generation) {
    if (!identical(generation._ownerKey, _ownerKey)) {
      return Future<bool>.value(false);
    }
    return generation._stopFence;
  }

  /// Queue exact text only while [generation] is still the current owner.
  bool enqueue(TtsPlaybackGeneration generation, String text) {
    if (!isCurrent(generation) || _poisoned || text.trim().isEmpty) {
      return false;
    }
    _queue.addLast(_QueuedSpeech(generation, text));
    _logicalWork = true;
    _publishActivity();
    _ensurePump();
    return true;
  }

  /// Wait until all currently reachable cleanup and pump work has settled.
  ///
  /// Tests and orderly shutdown may use this after releasing their controlled
  /// futures. It intentionally has no timeout and cannot cancel native synth
  /// or a native player operation that has already been entered.
  Future<void> whenIdle() async {
    while (true) {
      final playerOperation = _playerOperationTail;
      final pump = _pumpFuture;
      await playerOperation;
      if (pump != null) await pump;
      if (identical(playerOperation, _playerOperationTail) && !_pumpRunning) {
        return;
      }
    }
  }

  /// Permanently reject new work and clean up the current generation.
  Future<bool> close() {
    final existing = _closeResult;
    if (existing != null) return existing;
    final closed = Completer<bool>();
    _closeResult = closed.future;
    _closed = true;
    final stopFence = _advanceGeneration(allowClosed: true)._stopFence;
    unawaited(_runClose(stopFence, closed));
    return closed.future;
  }

  Future<void> _runClose(
    Future<bool> stopFence,
    Completer<bool> closed,
  ) async {
    var stopped = false;
    try {
      stopped = await stopFence;
      // `_advanceGeneration` synchronously drops queued work, but a synthesis
      // already admitted by the pump may still be inside native code. Keep the
      // close receipt pending until that exact call returns and every resulting
      // stale-path/clip-cleanup step has drained.
      await whenIdle();
    } catch (_) {
      stopped = false;
    }
    closed.complete(
      stopped && !_poisoned && !_physicalActive && !_pumpRunning,
    );
  }

  TtsPlaybackGeneration _advanceGeneration({bool allowClosed = false}) {
    if (_closed && !allowClosed) {
      throw StateError('TTS playback owner is closed');
    }

    final previousOperation = _playerOperationTail;
    final stopped = Completer<bool>();
    final next = TtsPlaybackGeneration._(
      _ownerKey,
      ++_nextOrdinal,
      stopped.future,
    );

    // Everything through this point is synchronous: old enqueue authority and
    // the exact clip interrupt are gone before callers can submit new work.
    _current = next;
    _queue.clear();
    _logicalWork = false;
    final active = _activeClip;
    if (active != null && !active.interrupt.isCompleted) {
      active.interrupt.complete(_ClipInterrupted(stopped.future));
    }

    final operation = _performStop(
      previousOperation,
      active,
      stopped,
    );
    _playerOperationTail = operation;
    _publishActivity();
    return next;
  }

  Future<void> _performStop(
    Future<void> previousOperation,
    _ActiveClip? active,
    Completer<bool> stopped,
  ) async {
    await previousOperation;
    final clip = active?.clip;
    if (clip == null) {
      if (active != null && identical(_activeClip, active)) {
        _activeClip = null;
      }
      stopped.complete(true);
      return;
    }

    final result = await clip.stopAndRelease();
    _reportCleanupResult(active!, result);
    if (result.succeeded) {
      if (identical(_activeClip, active)) {
        _activeClip = null;
        _physicalActive = false;
      }
    } else {
      _poisonPlayback();
    }
    _publishActivity();
    stopped.complete(result.succeeded);
  }

  void _ensurePump() {
    if (_pumpRunning) return;
    _pumpRunning = true;
    final pump = _pump();
    _pumpFuture = pump;
    unawaited(pump.then((_) {
      if (!identical(_pumpFuture, pump)) return;
      _pumpFuture = null;
      _pumpRunning = false;
      if (_queue.isNotEmpty && !_closed && !_poisoned) {
        _ensurePump();
      } else {
        _logicalWork = false;
        _publishActivity();
      }
    }));
  }

  Future<void> _pump() async {
    while (_queue.isNotEmpty && !_poisoned) {
      final item = _queue.removeFirst();
      if (!isCurrent(item.generation)) continue;

      String? path;
      try {
        path = await _synthesize(item.text);
      } catch (error, stackTrace) {
        if (isCurrent(item.generation)) {
          _reportError(item.generation, error, stackTrace);
        }
        continue;
      }
      if (path == null || !isCurrent(item.generation) || _poisoned) continue;

      final stopped = await item.generation._stopFence;
      if (!stopped || !isCurrent(item.generation) || _poisoned) continue;

      await _playClip(item.generation, path);
    }
  }

  Future<void> _playClip(
    TtsPlaybackGeneration generation,
    String path,
  ) async {
    if (!isCurrent(generation) || _poisoned) return;

    final active = _ActiveClip(generation);
    _activeClip = active;
    final startOutcome = await _admitPlaybackStart(active, generation, path);
    if (startOutcome.disposition == _PlaybackStartDisposition.stale) return;
    if (startOutcome.disposition == _PlaybackStartDisposition.failed) {
      if (isCurrent(generation)) _advanceGeneration();
      return;
    }

    final clip = active.clip;
    if (clip == null) {
      if (isCurrent(generation)) _advanceGeneration();
      return;
    }
    if (isCurrent(generation)) _notifyPlaybackStarted(generation);

    final winner = await Future.any<Object>([
      clip.terminal.then<Object>(_ClipTerminal.new),
      active.interrupt.future.then<Object>((value) => value),
    ]);
    if (winner is _ClipInterrupted) {
      await winner.stopFence;
      return;
    }

    final terminal = (winner as _ClipTerminal).terminal;
    if (terminal.error != null) {
      _reportError(
        generation,
        terminal.error!,
        terminal.stackTrace ?? StackTrace.current,
      );
    }
    if (terminal.kind == TtsPlaybackTerminalKind.completed) {
      if (identical(_activeClip, active)) {
        // An exact natural terminal proves that audible playback ended even if
        // subsequent native resource disposal fails.
        _physicalActive = false;
        _publishActivity();
      }
      final result = await _queueNaturalCleanup(active);
      if (!result.succeeded) _poisonPlayback();
      return;
    }

    if (isCurrent(generation)) _advanceGeneration();
  }

  Future<_PlaybackStartOutcome> _admitPlaybackStart(
    _ActiveClip active,
    TtsPlaybackGeneration generation,
    String path,
  ) {
    final previousOperation = _playerOperationTail;
    final outcome = Completer<_PlaybackStartOutcome>();
    final operation = _performPlaybackStart(
      previousOperation,
      active,
      generation,
      path,
      outcome,
    );
    // Reserve the single player-operation lane before the async body can pass
    // its first await. A concurrent supersede therefore queues cleanup after an
    // already-admitted start, never before it.
    _playerOperationTail = operation;
    return outcome.future;
  }

  Future<void> _performPlaybackStart(
    Future<void> previousOperation,
    _ActiveClip active,
    TtsPlaybackGeneration generation,
    String path,
    Completer<_PlaybackStartOutcome> outcome,
  ) async {
    await previousOperation;
    if (!isCurrent(generation) || _poisoned) {
      if (identical(_activeClip, active)) _activeClip = null;
      outcome.complete(
        const _PlaybackStartOutcome(_PlaybackStartDisposition.stale),
      );
      return;
    }

    TtsPlaybackClip clip;
    try {
      clip = _createPlaybackClip(path);
    } catch (error, stackTrace) {
      // Construction may already have entered plugin/native work before
      // throwing without returning an exact cleanup handle. Retain poison.
      _poisonPlayback();
      _reportError(generation, error, stackTrace);
      outcome.complete(
        const _PlaybackStartOutcome(_PlaybackStartDisposition.failed),
      );
      return;
    }
    active.clip = clip;
    _physicalActive = true;
    _publishActivity();

    final result = await clip.started;
    _reportResult(generation, result);
    outcome.complete(
      _PlaybackStartOutcome(
        result.succeeded
            ? _PlaybackStartDisposition.started
            : _PlaybackStartDisposition.failed,
      ),
    );
  }

  Future<TtsPlaybackResult> _queueNaturalCleanup(
    _ActiveClip active,
  ) {
    final previousOperation = _playerOperationTail;
    final result = Completer<TtsPlaybackResult>();
    final operation = _performNaturalCleanup(
      previousOperation,
      active,
      result,
    );
    _playerOperationTail = operation;
    return result.future;
  }

  Future<void> _performNaturalCleanup(
    Future<void> previousOperation,
    _ActiveClip active,
    Completer<TtsPlaybackResult> completed,
  ) async {
    await previousOperation;
    final clip = active.clip;
    final result = clip == null
        ? const TtsPlaybackResult.success()
        : await clip.stopAndRelease();
    _reportCleanupResult(active, result);
    if (result.succeeded && identical(_activeClip, active)) {
      _activeClip = null;
      _physicalActive = false;
    }
    if (!result.succeeded) _poisonPlayback();
    _publishActivity();
    completed.complete(result);
  }

  void _poisonPlayback() {
    _poisoned = true;
    _queue.clear();
    _logicalWork = false;
  }

  void _reportResult(
    TtsPlaybackGeneration generation,
    TtsPlaybackResult result,
  ) {
    final error = result.error;
    if (error == null) return;
    _reportError(
      generation,
      error,
      result.stackTrace ?? StackTrace.current,
    );
  }

  void _reportCleanupResult(
    _ActiveClip active,
    TtsPlaybackResult result,
  ) {
    if (active.cleanupReported) return;
    active.cleanupReported = true;
    _reportResult(active.generation, result);
  }

  void _publishActivity() {
    final speaking = _logicalWork || _physicalActive;
    _speaking = speaking;
    if (identical(_publishedGeneration, _current) &&
        _publishedSpeaking == speaking) {
      return;
    }
    _publishedGeneration = _current;
    _publishedSpeaking = speaking;
    final callback = _onActivityChanged;
    if (callback == null) return;
    try {
      callback(_current, speaking);
    } catch (error, stackTrace) {
      _reportError(_current, error, stackTrace);
    }
  }

  void _notifyPlaybackStarted(TtsPlaybackGeneration generation) {
    final callback = _onPlaybackStarted;
    if (callback == null || !isCurrent(generation)) return;
    try {
      callback(generation);
    } catch (error, stackTrace) {
      _reportError(generation, error, stackTrace);
    }
  }

  void _reportError(
    TtsPlaybackGeneration generation,
    Object error,
    StackTrace stackTrace,
  ) {
    try {
      _onError?.call(generation, error, stackTrace);
    } catch (_) {
      // Diagnostic callbacks never own the pump or its cleanup.
    }
  }
}
