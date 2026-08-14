// Serialized, exact-resource ownership for one Assistant listening widget.
//
// This file is deliberately pure Dart. Adapters own plugin-specific tokens;
// this owner only sequences them and records bounded, content-free evidence.
import 'dart:async';

const int assistantListeningMaximumSafeOrdinal = 0x1fffffffffffff;
const Duration assistantListeningAdmissionMaximumLifetime = Duration(
  seconds: 30,
);
const Duration assistantListeningCleanupMaximumLifetime = Duration(seconds: 10);

typedef AssistantListeningTimerFactory = Timer Function(
    Duration duration, void Function() callback);
typedef AssistantListeningRevokeCallback<S extends Object, C extends Object>
    = void Function(
  AssistantListeningGeneration<S, C> generation,
  AssistantListeningOutcome outcome,
);
typedef AssistantListeningStateCallback<S extends Object, C extends Object>
    = void Function(AssistantListeningGeneration<S, C> generation);

enum AssistantListeningIntent { on, off }

enum AssistantListeningCaptureTerminal { ended, failed }

enum AssistantListeningCaptureStartDisposition {
  notAttempted,
  started,
  failedClean,
  failedRecoverable,
  failedAmbiguous,
  threwAmbiguous,
}

enum AssistantListeningOutcome {
  listening,
  stopped,
  cancelled,
  superseded,
  permissionDenied,
  permissionFailed,
  routeRejected,
  routeFailed,
  sessionAcquireAmbiguous,
  sessionCleanupWatchAmbiguous,
  sessionReadyFailed,
  sessionSourceFailed,
  captureStartFailed,
  captureStartAmbiguous,
  captureTerminalWatchAmbiguous,
  captureEnded,
  captureSourceFailed,
  callbackFailed,
  admissionDeadlineExceeded,
  cleanupDeadlineExceeded,
  ownerClosed,
  ownerPoisoned,
}

enum AssistantListeningCancelReason { cancelled, superseded }

/// A fixed-code error. Provider/plugin exception text is never retained.
final class AssistantListeningFailure implements Exception {
  const AssistantListeningFailure(this.code);

  final String code;

  @override
  String toString() => 'AssistantListeningFailure($code)';
}

/// A typed capture-start result. Untyped throws are always ambiguous.
sealed class AssistantListeningCaptureStartResult<C extends Object> {
  const AssistantListeningCaptureStartResult._();

  const factory AssistantListeningCaptureStartResult.started(C capture) =
      AssistantListeningCaptureStarted<C>;
  const factory AssistantListeningCaptureStartResult.failedClean() =
      AssistantListeningCaptureFailedClean<C>;
  const factory AssistantListeningCaptureStartResult.failedRecoverable(
    C capture,
  ) = AssistantListeningCaptureFailedRecoverable<C>;
  const factory AssistantListeningCaptureStartResult.failedAmbiguous() =
      AssistantListeningCaptureFailedAmbiguous<C>;
}

final class AssistantListeningCaptureStarted<C extends Object>
    extends AssistantListeningCaptureStartResult<C> {
  const AssistantListeningCaptureStarted(this.capture) : super._();

  final C capture;
}

final class AssistantListeningCaptureFailedClean<C extends Object>
    extends AssistantListeningCaptureStartResult<C> {
  const AssistantListeningCaptureFailedClean() : super._();
}

final class AssistantListeningCaptureFailedRecoverable<C extends Object>
    extends AssistantListeningCaptureStartResult<C> {
  const AssistantListeningCaptureFailedRecoverable(this.capture) : super._();

  final C capture;
}

final class AssistantListeningCaptureFailedAmbiguous<C extends Object>
    extends AssistantListeningCaptureStartResult<C> {
  const AssistantListeningCaptureFailedAmbiguous() : super._();
}

/// Cleanup-capable adapter retained until every entered lifecycle call unwinds.
///
/// Implementations must not retain widget State, transcript text, or other UI
/// callbacks. In particular, [startCapture] must prepublish its exact recorder
/// lane before native start, so a thrown start can be recovered through
/// [recoverAmbiguousCaptureStart]. [endSession] only synchronously revokes the
/// exact ASR token; [waitSessionEnded] supplies the later worker-release proof.
/// A rejected feed or source error must synchronously fence its immutable
/// generation and request the memoized exact source cancellation. It must
/// complete only that capture's [waitCaptureTerminal] result as failed,
/// exactly once, after cancellation succeeds; a failed terminal is a failure
/// trigger, not independent source-release proof. A natural done may complete
/// as ended immediately. Adapters must never redirect stale capture work to a
/// mutable current session.
abstract interface class AssistantListeningLifecycle<S extends Object,
    C extends Object> {
  Future<bool> requestPermission(AssistantListeningGeneration<S, C> generation);

  Future<bool> configureRoute(AssistantListeningGeneration<S, C> generation);

  S beginSession(AssistantListeningGeneration<S, C> generation);

  Future<bool> waitSessionEnded(S session);

  Future<void> waitSessionReady(S session);

  Future<AssistantListeningCaptureStartResult<C>> startCapture(
    AssistantListeningGeneration<S, C> generation,
    S session,
  );

  Future<AssistantListeningCaptureTerminal> waitCaptureTerminal(C capture);

  Future<bool> cancelCaptureSource(C capture);

  Future<bool> stopCapture(C capture);

  Future<bool> recoverAmbiguousCaptureStart(
    AssistantListeningGeneration<S, C> generation,
    S session,
  );

  bool endSession(S session);
}

/// The first admission result. A listening result does not prove later cleanup.
final class AssistantListeningDone {
  const AssistantListeningDone({required this.outcome, this.failure});

  final AssistantListeningOutcome outcome;
  final AssistantListeningFailure? failure;
}

/// Immutable aggregate evidence for one exact generation.
final class AssistantListeningCleanupReceipt {
  const AssistantListeningCleanupReceipt({
    required this.ordinal,
    required this.intent,
    required this.outcome,
    required this.started,
    required this.admissionDeadlineExpired,
    required this.cleanupDeadlineExpired,
    required this.permissionAttempted,
    required this.permissionGranted,
    required this.routeAttempted,
    required this.routeConfigured,
    required this.sessionAcquireAttempted,
    required this.sessionAcquireReturned,
    required this.sessionCleanupWatchAttempted,
    required this.sessionCleanupWatchReturned,
    required this.sessionReadyAttempted,
    required this.sessionReadySucceeded,
    required this.captureStartAttempted,
    required this.captureStartDisposition,
    required this.captureRecoveryAttempted,
    required this.captureRecoverySucceeded,
    required this.captureTerminalWatchAttempted,
    required this.captureTerminalWatchReturned,
    required this.captureTerminalObserved,
    required this.captureSourceErrorObserved,
    required this.captureCancelAttempted,
    required this.captureCancelSucceeded,
    required this.captureStopAttempted,
    required this.captureStopSucceeded,
    required this.sessionEndAttempted,
    required this.sessionEndAccepted,
    required this.sessionTerminalObserved,
    required this.sessionTerminalSucceeded,
    required this.revokeCallbackAttempts,
    required this.revokeCallbackFailures,
    required this.listeningCallbackAttempts,
    required this.listeningCallbackFailures,
    required this.stoppedCallbackAttempts,
    required this.stoppedCallbackFailures,
    required this.exactCaptureSettled,
    required this.exactSessionSettled,
    required this.exactResourcesSettled,
    required this.ownerPoisonedAtSettlement,
  });

  final int ordinal;
  final AssistantListeningIntent intent;
  final AssistantListeningOutcome outcome;
  final bool started;
  final bool admissionDeadlineExpired;
  final bool cleanupDeadlineExpired;
  final bool permissionAttempted;
  final bool permissionGranted;
  final bool routeAttempted;
  final bool routeConfigured;
  final bool sessionAcquireAttempted;
  final bool sessionAcquireReturned;
  final bool sessionCleanupWatchAttempted;
  final bool sessionCleanupWatchReturned;
  final bool sessionReadyAttempted;
  final bool sessionReadySucceeded;
  final bool captureStartAttempted;
  final AssistantListeningCaptureStartDisposition captureStartDisposition;
  final bool captureRecoveryAttempted;
  final bool captureRecoverySucceeded;
  final bool captureTerminalWatchAttempted;
  final bool captureTerminalWatchReturned;
  final bool captureTerminalObserved;
  final bool captureSourceErrorObserved;
  final bool captureCancelAttempted;
  final bool captureCancelSucceeded;
  final bool captureStopAttempted;
  final bool captureStopSucceeded;
  final bool sessionEndAttempted;
  final bool sessionEndAccepted;
  final bool sessionTerminalObserved;
  final bool sessionTerminalSucceeded;
  final int revokeCallbackAttempts;
  final int revokeCallbackFailures;
  final int listeningCallbackAttempts;
  final int listeningCallbackFailures;
  final int stoppedCallbackAttempts;
  final int stoppedCallbackFailures;
  final bool exactCaptureSettled;
  final bool exactSessionSettled;
  final bool exactResourcesSettled;
  final bool ownerPoisonedAtSettlement;
}

final class AssistantListeningCloseReceipt {
  const AssistantListeningCloseReceipt({
    required this.exactResourcesSettled,
    required this.poisoned,
    required this.retainsUncertainResources,
    required this.lastOrdinal,
  });

  final bool exactResourcesSettled;
  final bool poisoned;
  final bool retainsUncertainResources;
  final int lastOrdinal;
}

final class AssistantListeningSnapshot {
  const AssistantListeningSnapshot({
    required this.active,
    required this.pending,
    required this.listening,
    required this.poisoned,
    required this.closed,
    required this.retainsUncertainResources,
    required this.lastOrdinal,
  });

  final bool active;
  final bool pending;
  final bool listening;
  final bool poisoned;
  final bool closed;
  final bool retainsUncertainResources;
  final int lastOrdinal;
}

/// Opaque owner-keyed authority for one desired listening state.
final class AssistantListeningGeneration<S extends Object, C extends Object> {
  AssistantListeningGeneration._(
    this._ownerKey,
    this.ordinal,
    this.intent,
    this._run,
  );

  final Object _ownerKey;
  final _AssistantListeningRun<S, C> _run;
  final int ordinal;
  final AssistantListeningIntent intent;

  Future<AssistantListeningDone> get done => _run.done;
  Future<AssistantListeningCleanupReceipt> get cleanup => _run.cleanup;
  bool get isRevoked => _run.isRevoked;
  bool get isListening => _run._listening;
  bool get isCleanupDone => _run.isCleanupPublished;

  @override
  String toString() => 'AssistantListeningGeneration($ordinal, $intent)';
}

/// Owns at most one running reconciliation and one latest pending intent.
final class AssistantListeningOwner<S extends Object, C extends Object> {
  factory AssistantListeningOwner({
    required AssistantListeningLifecycle<S, C> lifecycle,
    AssistantListeningRevokeCallback<S, C>? onRevoke,
    AssistantListeningStateCallback<S, C>? onListening,
    AssistantListeningStateCallback<S, C>? onStopped,
    Duration admissionMaximumLifetime =
        assistantListeningAdmissionMaximumLifetime,
    Duration cleanupMaximumLifetime = assistantListeningCleanupMaximumLifetime,
    AssistantListeningTimerFactory timerFactory = Timer.new,
  }) =>
      AssistantListeningOwner._(
        lifecycle: lifecycle,
        onRevoke: onRevoke,
        onListening: onListening,
        onStopped: onStopped,
        admissionMaximumLifetime: admissionMaximumLifetime,
        cleanupMaximumLifetime: cleanupMaximumLifetime,
        timerFactory: timerFactory,
        initialOrdinal: 0,
      );

  factory AssistantListeningOwner.forTesting({
    required AssistantListeningLifecycle<S, C> lifecycle,
    AssistantListeningRevokeCallback<S, C>? onRevoke,
    AssistantListeningStateCallback<S, C>? onListening,
    AssistantListeningStateCallback<S, C>? onStopped,
    Duration admissionMaximumLifetime =
        assistantListeningAdmissionMaximumLifetime,
    Duration cleanupMaximumLifetime = assistantListeningCleanupMaximumLifetime,
    AssistantListeningTimerFactory timerFactory = Timer.new,
    int initialOrdinal = 0,
  }) =>
      AssistantListeningOwner._(
        lifecycle: lifecycle,
        onRevoke: onRevoke,
        onListening: onListening,
        onStopped: onStopped,
        admissionMaximumLifetime: admissionMaximumLifetime,
        cleanupMaximumLifetime: cleanupMaximumLifetime,
        timerFactory: timerFactory,
        initialOrdinal: initialOrdinal,
      );

  AssistantListeningOwner._({
    required AssistantListeningLifecycle<S, C> lifecycle,
    required AssistantListeningRevokeCallback<S, C>? onRevoke,
    required AssistantListeningStateCallback<S, C>? onListening,
    required AssistantListeningStateCallback<S, C>? onStopped,
    required Duration admissionMaximumLifetime,
    required Duration cleanupMaximumLifetime,
    required AssistantListeningTimerFactory timerFactory,
    required int initialOrdinal,
  })  : _lifecycle = lifecycle,
        _onRevoke = onRevoke,
        _onListening = onListening,
        _onStopped = onStopped,
        _admissionMaximumLifetime = admissionMaximumLifetime,
        _cleanupMaximumLifetime = cleanupMaximumLifetime,
        _timerFactory = timerFactory,
        _nextOrdinal = initialOrdinal {
    if (admissionMaximumLifetime <= Duration.zero) {
      throw ArgumentError.value(
        admissionMaximumLifetime,
        'admissionMaximumLifetime',
      );
    }
    if (cleanupMaximumLifetime <= Duration.zero) {
      throw ArgumentError.value(
        cleanupMaximumLifetime,
        'cleanupMaximumLifetime',
      );
    }
    if (initialOrdinal < 0 ||
        initialOrdinal > assistantListeningMaximumSafeOrdinal) {
      throw ArgumentError.value(initialOrdinal, 'initialOrdinal');
    }
  }

  final AssistantListeningLifecycle<S, C> _lifecycle;
  AssistantListeningRevokeCallback<S, C>? _onRevoke;
  AssistantListeningStateCallback<S, C>? _onListening;
  AssistantListeningStateCallback<S, C>? _onStopped;
  final Duration _admissionMaximumLifetime;
  final Duration _cleanupMaximumLifetime;
  final AssistantListeningTimerFactory _timerFactory;
  final Object _ownerKey = Object();

  _AssistantListeningRun<S, C>? _active;
  _AssistantListeningRun<S, C>? _pending;
  _AssistantListeningRun<S, C>? _latest;
  _AssistantListeningRun<S, C>? _retainedUncertainRun;
  AssistantListeningCleanupReceipt? _lastReceipt;
  Completer<AssistantListeningCloseReceipt>? _closeCompleter;
  Future<AssistantListeningCloseReceipt>? _closeFuture;
  int _nextOrdinal;
  bool _closed = false;
  bool _poisoned = false;
  bool _allExactResourcesSettled = true;

  bool get isClosed => _closed;
  bool get isPoisoned => _poisoned;
  bool get hasActive => _active != null;
  bool get hasPending => _pending != null;
  AssistantListeningCleanupReceipt? get lastReceipt => _lastReceipt;

  AssistantListeningSnapshot get snapshot => AssistantListeningSnapshot(
        active: _active != null,
        pending: _pending != null,
        listening: _active?._listening ?? false,
        poisoned: _poisoned,
        closed: _closed,
        retainsUncertainResources: _retainedUncertainRun != null,
        lastOrdinal: _nextOrdinal,
      );

  AssistantListeningGeneration<S, C> enable() =>
      replace(AssistantListeningIntent.on);

  AssistantListeningGeneration<S, C> disable() =>
      replace(AssistantListeningIntent.off);

  AssistantListeningGeneration<S, C> replace(AssistantListeningIntent intent) {
    if (_closed) throw const AssistantListeningFailure('owner_closed');
    if (_poisoned) throw const AssistantListeningFailure('owner_poisoned');
    if (_nextOrdinal >= assistantListeningMaximumSafeOrdinal) {
      _poisoned = true;
      final pending = _pending;
      _pending = null;
      if (identical(_latest, pending)) _latest = null;
      pending?._finishUnstarted(
        AssistantListeningOutcome.ownerPoisoned,
        const AssistantListeningFailure('ordinal_exhausted'),
      );
      _active?._revoke(
        AssistantListeningOutcome.ownerPoisoned,
        const AssistantListeningFailure('ordinal_exhausted'),
        poison: true,
      );
      throw const AssistantListeningFailure('ordinal_exhausted');
    }

    final run = _AssistantListeningRun<S, C>(
      owner: this,
      ordinal: ++_nextOrdinal,
      intent: intent,
    );
    final active = _active;
    if (active == null) {
      _active = run;
      _latest = run;
      run._armAdmissionDeadline();
      if (identical(_active, run) && !run.isRevoked) run._scheduleStart();
      return run.generation;
    }

    final displaced = _pending;
    _pending = run;
    _latest = run;
    // Publish the desired state, then synchronously fence the old exact lane
    // before constructing a timer or invoking a displaced-pending callback.
    active._revoke(AssistantListeningOutcome.superseded, null);
    displaced?._finishUnstarted(AssistantListeningOutcome.superseded, null);
    if (identical(_pending, run) &&
        identical(_latest, run) &&
        !run.isRevoked &&
        !run.isCleanupPublished) {
      run._armAdmissionDeadline();
    }
    return run.generation;
  }

  bool isAuthoritative(AssistantListeningGeneration<S, C> generation) =>
      identical(generation._ownerKey, _ownerKey) &&
      identical(_latest, generation._run) &&
      !generation._run.isRevoked &&
      !_closed &&
      !_poisoned;

  Future<AssistantListeningCleanupReceipt> revokeExact(
    AssistantListeningGeneration<S, C> generation, {
    AssistantListeningCancelReason reason =
        AssistantListeningCancelReason.cancelled,
  }) {
    if (!identical(generation._ownerKey, _ownerKey)) {
      throw ArgumentError('foreign AssistantListeningGeneration');
    }
    final run = generation._run;
    final outcome = reason == AssistantListeningCancelReason.cancelled
        ? AssistantListeningOutcome.cancelled
        : AssistantListeningOutcome.superseded;
    if (identical(_pending, run)) {
      _pending = null;
      if (identical(_latest, run)) _latest = null;
      run._finishUnstarted(outcome, null);
    } else if (identical(_active, run)) {
      if (identical(_latest, run)) _latest = null;
      run._revoke(outcome, null);
    }
    return run.cleanup;
  }

  /// Permanently closes the owner. The same prepublished Future is returned.
  Future<AssistantListeningCloseReceipt> close() {
    final existing = _closeFuture;
    if (existing != null) return existing;

    final completer = Completer<AssistantListeningCloseReceipt>();
    _closeCompleter = completer;
    _closeFuture = completer.future;
    _closed = true;
    _latest = null;

    final pending = _pending;
    _pending = null;
    final active = _active;
    // Fence and enqueue every exact cleanup operation before a pending
    // generation's revoke callback is allowed to reenter.
    active?._revoke(AssistantListeningOutcome.ownerClosed, null);
    pending?._finishUnstarted(AssistantListeningOutcome.ownerClosed, null);
    if (active == null) {
      _clearCallbacks();
      _completeClose();
    } else {
      _clearCallbacks();
      unawaited(
        active.cleanup.then<void>(
          (_) => _completeClose(),
          onError: (_error, _stackTrace) {
            _poisoned = true;
            _allExactResourcesSettled = false;
            _completeClose();
          },
        ),
      );
    }
    return completer.future;
  }

  void _clearCallbacks() {
    _onRevoke = null;
    _onListening = null;
    _onStopped = null;
  }

  void _completeClose() {
    final completer = _closeCompleter;
    if (completer == null || completer.isCompleted) return;
    completer.complete(
      AssistantListeningCloseReceipt(
        exactResourcesSettled:
            _allExactResourcesSettled && _retainedUncertainRun == null,
        poisoned: _poisoned,
        retainsUncertainResources: _retainedUncertainRun != null,
        lastOrdinal: _nextOrdinal,
      ),
    );
  }

  bool _isActive(_AssistantListeningRun<S, C> run) => identical(_active, run);

  bool _isAuthoritativeRun(_AssistantListeningRun<S, C> run) =>
      identical(_active, run) &&
      identical(_latest, run) &&
      !run.isRevoked &&
      !_closed &&
      !_poisoned;

  Timer _newTimer(Duration duration, void Function() callback) =>
      _timerFactory(duration, callback);

  void _pendingAdmissionExpired(_AssistantListeningRun<S, C> run) {
    if (!identical(_pending, run) || run._started || run.isCleanupPublished) {
      return;
    }
    _pending = null;
    if (identical(_latest, run)) _latest = null;
    run._admissionDeadlineExpired = true;
    run._finishUnstarted(
      AssistantListeningOutcome.admissionDeadlineExceeded,
      const AssistantListeningFailure('admission_deadline_exceeded'),
    );
  }

  void _activeAdmissionExpired(_AssistantListeningRun<S, C> run) {
    if (!identical(_active, run) || run.isCleanupPublished) return;
    if (!run._started) {
      if (identical(_latest, run)) _latest = null;
      run._admissionDeadlineExpired = true;
      run._finishUnstarted(
        AssistantListeningOutcome.admissionDeadlineExceeded,
        const AssistantListeningFailure('admission_deadline_exceeded'),
      );
      return;
    }
    run._admissionDeadlineExpired = true;
    run._revoke(
      AssistantListeningOutcome.admissionDeadlineExceeded,
      const AssistantListeningFailure('admission_deadline_exceeded'),
      poison: true,
    );
  }

  void _timerConstructionFailed(_AssistantListeningRun<S, C> run) {
    _poisoned = true;
    if (!run._started) {
      if (identical(_pending, run)) _pending = null;
      if (identical(_latest, run)) _latest = null;
      final active = _active;
      if (active != null && !identical(active, run)) {
        active._revoke(
          AssistantListeningOutcome.ownerPoisoned,
          const AssistantListeningFailure('deadline_timer_failed'),
          poison: true,
        );
      }
      run._finishUnstarted(
        AssistantListeningOutcome.ownerPoisoned,
        const AssistantListeningFailure('deadline_timer_failed'),
        exactResourcesSettled: false,
      );
      _finishPendingPoisoned(except: run);
      return;
    }
    run._revoke(
      AssistantListeningOutcome.ownerPoisoned,
      const AssistantListeningFailure('deadline_timer_failed'),
      poison: true,
    );
    _finishPendingPoisoned(except: run);
  }

  void _poison(
    _AssistantListeningRun<S, C> run, {
    bool clearCallbacks = false,
  }) {
    _poisoned = true;
    if (clearCallbacks) _clearCallbacks();
    _finishPendingPoisoned(except: run);
  }

  void _finishPendingPoisoned({required _AssistantListeningRun<S, C> except}) {
    final pending = _pending;
    if (pending == null || identical(pending, except)) return;
    _pending = null;
    if (identical(_latest, pending)) _latest = null;
    pending._finishUnstarted(
      AssistantListeningOutcome.ownerPoisoned,
      const AssistantListeningFailure('owner_poisoned'),
    );
  }

  void _unstartedFinished(
    _AssistantListeningRun<S, C> run,
    AssistantListeningCleanupReceipt receipt,
  ) {
    _lastReceipt = receipt;
    if (identical(_active, run)) {
      _active = null;
      if (identical(_latest, run)) _latest = null;
      _startPendingIfAllowed();
    }
  }

  void _activeFinished(
    _AssistantListeningRun<S, C> run,
    AssistantListeningCleanupReceipt receipt,
  ) {
    _lastReceipt = receipt;
    if (!identical(_active, run)) {
      _poisoned = true;
      _allExactResourcesSettled = false;
      _retainedUncertainRun ??= run;
      _finishPendingPoisoned(except: run);
      return;
    }
    if (!receipt.exactResourcesSettled) {
      _poisoned = true;
      _allExactResourcesSettled = false;
      _retainedUncertainRun ??= run;
      _finishPendingPoisoned(except: run);
      return;
    }

    _active = null;
    if (identical(_latest, run)) _latest = null;
    _startPendingIfAllowed();
  }

  void _lateUnwindFinished(
    _AssistantListeningRun<S, C> run, {
    required bool exactResourcesSettled,
  }) {
    if (exactResourcesSettled && identical(_retainedUncertainRun, run)) {
      _retainedUncertainRun = null;
    }
    if (identical(_active, run)) _active = null;
  }

  void _startPendingIfAllowed() {
    if (_closed || _poisoned) {
      final pending = _pending;
      _pending = null;
      if (pending != null) {
        if (identical(_latest, pending)) _latest = null;
        pending._finishUnstarted(
          _closed
              ? AssistantListeningOutcome.ownerClosed
              : AssistantListeningOutcome.ownerPoisoned,
          _poisoned ? const AssistantListeningFailure('owner_poisoned') : null,
        );
      }
      return;
    }
    final next = _pending;
    _pending = null;
    if (next == null || next.isCleanupPublished) return;
    _active = next;
    _latest = next;
    next._scheduleStart();
  }

  void _callbackFailed(_AssistantListeningRun<S, C> run) {
    _poison(run);
  }
}

final class _AssistantListeningRun<S extends Object, C extends Object> {
  _AssistantListeningRun({
    required AssistantListeningOwner<S, C> owner,
    required int ordinal,
    required AssistantListeningIntent intent,
  })  : _owner = owner,
        _intent = intent {
    generation = AssistantListeningGeneration<S, C>._(
      owner._ownerKey,
      ordinal,
      intent,
      this,
    );
  }

  final AssistantListeningOwner<S, C> _owner;
  final AssistantListeningIntent _intent;
  late final AssistantListeningGeneration<S, C> generation;
  final Completer<AssistantListeningDone> _done =
      Completer<AssistantListeningDone>();
  final Completer<AssistantListeningCleanupReceipt> _cleanup =
      Completer<AssistantListeningCleanupReceipt>();
  final Completer<void> _revokeSignal = Completer<void>();
  final Completer<AssistantListeningCaptureTerminal> _captureTerminalSignal =
      Completer<AssistantListeningCaptureTerminal>();

  Timer? _admissionTimer;
  Timer? _cleanupTimer;
  S? _session;
  C? _capture;
  Future<bool>? _sessionTerminalFuture;
  Future<bool>? _captureRecoveryFuture;
  Future<bool>? _captureCancelFuture;
  Future<bool>? _captureStopFuture;
  AssistantListeningOutcome _outcome = AssistantListeningOutcome.stopped;
  AssistantListeningCaptureStartDisposition _captureStartDisposition =
      AssistantListeningCaptureStartDisposition.notAttempted;

  bool _startScheduled = false;
  bool _started = false;
  bool _revoked = false;
  bool _listening = false;
  bool _cleanupStarted = false;
  bool _cleanupPublished = false;
  bool _admissionDeadlineExpired = false;
  bool _cleanupDeadlineExpired = false;
  bool _permissionAttempted = false;
  bool _permissionGranted = false;
  bool _routeAttempted = false;
  bool _routeConfigured = false;
  bool _sessionAcquireAttempted = false;
  bool _sessionAcquireReturned = false;
  bool _sessionCleanupWatchAttempted = false;
  bool _sessionCleanupWatchReturned = false;
  bool _sessionReadyAttempted = false;
  bool _sessionReadySucceeded = false;
  bool _captureStartAttempted = false;
  bool _captureRecoveryAttempted = false;
  bool _captureRecoverySucceeded = false;
  bool _captureTerminalWatchAttempted = false;
  bool _captureTerminalWatchReturned = false;
  bool _captureTerminalObserved = false;
  bool _captureSourceErrorObserved = false;
  bool _captureCancelAttempted = false;
  bool _captureCancelSucceeded = false;
  bool _captureStopAttempted = false;
  bool _captureStopSucceeded = false;
  bool _sessionEndAttempted = false;
  bool _sessionEndAccepted = false;
  bool _sessionTerminalObserved = false;
  bool _sessionTerminalSucceeded = false;
  bool _captureKnownClean = true;
  bool _sessionKnownClean = true;
  bool _listeningCallbackEnabled = true;
  bool _synchronousCleanupInitiationFailed = false;
  int _revokeCallbackAttempts = 0;
  int _revokeCallbackFailures = 0;
  int _listeningCallbackAttempts = 0;
  int _listeningCallbackFailures = 0;
  int _stoppedCallbackAttempts = 0;
  int _stoppedCallbackFailures = 0;

  Future<AssistantListeningDone> get done => _done.future;
  Future<AssistantListeningCleanupReceipt> get cleanup => _cleanup.future;
  bool get isRevoked => _revoked;
  bool get isCleanupPublished => _cleanupPublished;

  void _scheduleStart() {
    if (_startScheduled || _started || _revoked || _cleanupPublished) return;
    _startScheduled = true;
    scheduleMicrotask(() {
      _startScheduled = false;
      if (_started || _cleanupPublished) return;
      if (_revoked) {
        _finishWithoutResources();
        return;
      }
      _started = true;
      unawaited(
        _run().catchError((_error, _stackTrace) {
          _revoke(
            AssistantListeningOutcome.ownerPoisoned,
            const AssistantListeningFailure('owner_internal_failure'),
            poison: true,
          );
          _owner._clearCallbacks();
          _publishCleanup(forceExactResourcesSettled: false);
        }),
      );
    });
  }

  bool _cancelTimer(Timer? timer) {
    if (timer == null) return true;
    try {
      timer.cancel();
      return true;
    } catch (_) {
      return false;
    }
  }

  void _armAdmissionDeadline() {
    if (_admissionTimer != null || _cleanupPublished) return;
    var firedSynchronously = false;
    Timer timer;
    try {
      timer = _owner._newTimer(_owner._admissionMaximumLifetime, () {
        firedSynchronously = true;
        _admissionTimer = null;
        if (_cleanupPublished || _listening) return;
        if (_owner._isActive(this)) {
          _owner._activeAdmissionExpired(this);
        } else {
          _owner._pendingAdmissionExpired(this);
        }
      });
    } catch (_) {
      _owner._timerConstructionFailed(this);
      return;
    }
    if (firedSynchronously || _cleanupPublished || _listening) {
      if (!_cancelTimer(timer)) _owner._poison(this);
    } else {
      _admissionTimer = timer;
    }
  }

  void _armCleanupDeadline() {
    if (_cleanupTimer != null || _cleanupPublished) return;
    var firedSynchronously = false;
    Timer timer;
    try {
      timer = _owner._newTimer(_owner._cleanupMaximumLifetime, () {
        firedSynchronously = true;
        _cleanupTimer = null;
        _onCleanupDeadline();
      });
    } catch (_) {
      _owner._poison(this, clearCallbacks: true);
      _onCleanupDeadline();
      return;
    }
    if (firedSynchronously || _cleanupPublished) {
      if (!_cancelTimer(timer)) _owner._poison(this);
    } else {
      _cleanupTimer = timer;
    }
  }

  void _onCleanupDeadline() {
    if (_cleanupPublished) return;
    _cleanupDeadlineExpired = true;
    _outcome = AssistantListeningOutcome.cleanupDeadlineExceeded;
    _owner._poison(this, clearCallbacks: true);
    _publishCleanup(forceExactResourcesSettled: false);
  }

  Future<void> _run() async {
    if (_intent == AssistantListeningIntent.off) {
      _outcome = AssistantListeningOutcome.stopped;
      _completeDone(_outcome, null);
      _invokeStopped();
      _publishCleanup(forceExactResourcesSettled: true);
      return;
    }

    _permissionAttempted = true;
    bool permission;
    try {
      permission = await _owner._lifecycle.requestPermission(generation);
    } catch (_) {
      permission = false;
      if (!_revoked) {
        _revoke(
          AssistantListeningOutcome.permissionFailed,
          const AssistantListeningFailure('permission_failed'),
        );
      }
      await _cleanupResources();
      return;
    }
    if (!_owner._isAuthoritativeRun(this)) {
      await _cleanupResources();
      return;
    }
    _permissionGranted = permission;
    if (!permission) {
      _revoke(
        AssistantListeningOutcome.permissionDenied,
        const AssistantListeningFailure('permission_denied'),
      );
      await _cleanupResources();
      return;
    }

    _routeAttempted = true;
    bool route;
    try {
      route = await _owner._lifecycle.configureRoute(generation);
    } catch (_) {
      route = false;
      if (!_revoked) {
        _revoke(
          AssistantListeningOutcome.routeFailed,
          const AssistantListeningFailure('route_failed'),
          poison: true,
        );
      }
      await _cleanupResources();
      return;
    }
    if (!_owner._isAuthoritativeRun(this)) {
      await _cleanupResources();
      return;
    }
    _routeConfigured = route;
    if (!route) {
      _revoke(
        AssistantListeningOutcome.routeRejected,
        const AssistantListeningFailure('route_rejected'),
      );
      await _cleanupResources();
      return;
    }

    _sessionAcquireAttempted = true;
    S session;
    try {
      session = _owner._lifecycle.beginSession(generation);
    } catch (_) {
      _sessionKnownClean = false;
      _revoke(
        AssistantListeningOutcome.sessionAcquireAmbiguous,
        const AssistantListeningFailure('session_acquire_ambiguous'),
        poison: true,
      );
      await _cleanupResources();
      return;
    }
    _session = session;
    _sessionAcquireReturned = true;

    _sessionCleanupWatchAttempted = true;
    try {
      final terminal = _owner._lifecycle.waitSessionEnded(session);
      _sessionTerminalFuture = _observeSessionTerminal(terminal);
      _sessionCleanupWatchReturned = true;
    } catch (_) {
      _sessionKnownClean = false;
      _revoke(
        AssistantListeningOutcome.sessionCleanupWatchAmbiguous,
        const AssistantListeningFailure('session_cleanup_watch_ambiguous'),
        poison: true,
      );
      await _cleanupResources();
      return;
    }
    if (!_owner._isAuthoritativeRun(this)) {
      await _cleanupResources();
      return;
    }

    _sessionReadyAttempted = true;
    try {
      await _owner._lifecycle.waitSessionReady(session);
      _sessionReadySucceeded = true;
    } catch (_) {
      if (!_revoked) {
        _revoke(
          AssistantListeningOutcome.sessionReadyFailed,
          const AssistantListeningFailure('session_ready_failed'),
        );
      }
      await _cleanupResources();
      return;
    }
    if (!_owner._isAuthoritativeRun(this)) {
      await _cleanupResources();
      return;
    }

    _captureStartAttempted = true;
    AssistantListeningCaptureStartResult<C> startResult;
    try {
      startResult = await _owner._lifecycle.startCapture(generation, session);
    } catch (_) {
      _captureStartDisposition =
          AssistantListeningCaptureStartDisposition.threwAmbiguous;
      _captureKnownClean = false;
      _revoke(
        AssistantListeningOutcome.captureStartAmbiguous,
        const AssistantListeningFailure('capture_start_ambiguous'),
      );
      await _recoverAmbiguousCapture(session);
      await _cleanupResources();
      return;
    }

    if (startResult is AssistantListeningCaptureStarted<C>) {
      _capture = startResult.capture;
      _captureKnownClean = false;
      _captureStartDisposition =
          AssistantListeningCaptureStartDisposition.started;
    } else if (startResult is AssistantListeningCaptureFailedRecoverable<C>) {
      _capture = startResult.capture;
      _captureKnownClean = false;
      _captureStartDisposition =
          AssistantListeningCaptureStartDisposition.failedRecoverable;
    } else if (startResult is AssistantListeningCaptureFailedClean<C>) {
      _captureStartDisposition =
          AssistantListeningCaptureStartDisposition.failedClean;
    } else {
      _captureKnownClean = false;
      _captureStartDisposition =
          AssistantListeningCaptureStartDisposition.failedAmbiguous;
    }

    if (startResult is AssistantListeningCaptureFailedAmbiguous<C>) {
      _revoke(
        AssistantListeningOutcome.captureStartAmbiguous,
        const AssistantListeningFailure('capture_start_ambiguous'),
      );
      await _recoverAmbiguousCapture(session);
      await _cleanupResources();
      return;
    }
    if (startResult is AssistantListeningCaptureFailedClean<C>) {
      if (!_revoked) {
        _revoke(
          AssistantListeningOutcome.captureStartFailed,
          const AssistantListeningFailure('capture_start_failed'),
        );
      }
      await _cleanupResources();
      return;
    }

    final capture = _capture as C;
    _installCaptureTerminalWatch(capture);
    if (startResult is AssistantListeningCaptureFailedRecoverable<C>) {
      if (!_revoked) {
        _revoke(
          AssistantListeningOutcome.captureStartFailed,
          const AssistantListeningFailure('capture_start_failed'),
        );
      }
      await _cleanupResources();
      return;
    }
    if (!_captureTerminalWatchReturned) {
      if (!_revoked) {
        _revoke(
          AssistantListeningOutcome.captureTerminalWatchAmbiguous,
          const AssistantListeningFailure('capture_terminal_watch_ambiguous'),
          poison: true,
        );
      }
      await _cleanupResources();
      return;
    }
    // A Future that was already complete queues its terminal callback. Drain
    // that callback before listening can be published, then recheck exact
    // authority like every other awaited admission boundary.
    await Future<void>.value();
    if (!_owner._isAuthoritativeRun(this)) {
      await _cleanupResources();
      return;
    }

    _listening = true;
    final admissionTimer = _admissionTimer;
    _admissionTimer = null;
    if (!_cancelTimer(admissionTimer)) {
      _revoke(
        AssistantListeningOutcome.ownerPoisoned,
        const AssistantListeningFailure('deadline_timer_cancel_failed'),
        poison: true,
      );
      await _cleanupResources();
      return;
    }
    _completeDone(AssistantListeningOutcome.listening, null);
    _invokeListening();
    if (!_owner._isAuthoritativeRun(this)) {
      await _cleanupResources();
      return;
    }

    await Future.any<void>(<Future<void>>[
      _revokeSignal.future,
      _captureTerminalSignal.future.then<void>((_) {}),
    ]);
    await _cleanupResources();
  }

  void _installCaptureTerminalWatch(C capture) {
    _captureTerminalWatchAttempted = true;
    Future<AssistantListeningCaptureTerminal> terminal;
    try {
      terminal = _owner._lifecycle.waitCaptureTerminal(capture);
      _captureTerminalWatchReturned = true;
    } catch (_) {
      return;
    }
    unawaited(
      terminal.then<void>(
        _captureTerminalArrived,
        onError: (_error, _stackTrace) {
          _captureTerminalArrived(AssistantListeningCaptureTerminal.failed);
        },
      ),
    );
  }

  void _captureTerminalArrived(AssistantListeningCaptureTerminal terminal) {
    if (_captureTerminalObserved) return;
    _captureTerminalObserved = true;
    _captureSourceErrorObserved =
        terminal == AssistantListeningCaptureTerminal.failed;
    if (!_captureTerminalSignal.isCompleted) {
      _captureTerminalSignal.complete(terminal);
    }
    if (!_revoked) {
      _revoke(
        _captureSourceErrorObserved
            ? AssistantListeningOutcome.captureSourceFailed
            : AssistantListeningOutcome.captureEnded,
        _captureSourceErrorObserved
            ? const AssistantListeningFailure('capture_source_failed')
            : null,
      );
    }
  }

  Future<bool> _observeSessionTerminal(Future<bool> terminal) async {
    try {
      _sessionTerminalSucceeded = await terminal;
    } catch (_) {
      _sessionTerminalSucceeded = false;
    }
    _sessionTerminalObserved = true;
    final endedBeforeRequested = !_sessionEndAttempted;
    if (endedBeforeRequested && !_revoked) {
      _revoke(
        AssistantListeningOutcome.sessionSourceFailed,
        const AssistantListeningFailure('session_source_failed'),
        poison: !_sessionTerminalSucceeded,
      );
    } else if (!_sessionTerminalSucceeded) {
      _owner._poison(this);
    }
    return _sessionTerminalSucceeded;
  }

  Future<bool> _observeRecovery(Future<bool> recovery) async {
    try {
      _captureRecoverySucceeded = await recovery;
    } catch (_) {
      _captureRecoverySucceeded = false;
    }
    if (_captureRecoverySucceeded) {
      _captureKnownClean = true;
    } else {
      _owner._poison(this);
    }
    return _captureRecoverySucceeded;
  }

  Future<bool> _observeCaptureCancel(Future<bool> cancellation) async {
    try {
      _captureCancelSucceeded = await cancellation;
    } catch (_) {
      _captureCancelSucceeded = false;
    }
    if (!_captureCancelSucceeded) _owner._poison(this);
    return _captureCancelSucceeded;
  }

  Future<bool> _observeCaptureStop(Future<bool> stop) async {
    try {
      _captureStopSucceeded = await stop;
    } catch (_) {
      _captureStopSucceeded = false;
    }
    if (!_captureStopSucceeded) _owner._poison(this);
    return _captureStopSucceeded;
  }

  void _initiateSessionEnd() {
    final session = _session;
    if (session == null || _sessionEndAttempted) return;
    _sessionEndAttempted = true;
    try {
      _sessionEndAccepted = _owner._lifecycle.endSession(session);
    } catch (_) {
      _sessionEndAccepted = false;
    }
    if (!_sessionEndAccepted) {
      _sessionKnownClean = false;
      _synchronousCleanupInitiationFailed = true;
    }
  }

  void _initiateCaptureCleanup(C capture) {
    // record 6.2.1: enqueue exact subscription cancellation and recorder stop,
    // in that order, before an await. The record package keeps its platform
    // stream until stop and serializes stop/start globally.
    if (!_captureCancelAttempted) {
      _captureCancelAttempted = true;
      try {
        _captureCancelFuture = _observeCaptureCancel(
          _owner._lifecycle.cancelCaptureSource(capture),
        );
      } catch (_) {
        _captureCancelSucceeded = false;
        _synchronousCleanupInitiationFailed = true;
      }
    }
    if (!_captureStopAttempted) {
      _captureStopAttempted = true;
      try {
        _captureStopFuture = _observeCaptureStop(
          _owner._lifecycle.stopCapture(capture),
        );
      } catch (_) {
        _captureStopSucceeded = false;
        _synchronousCleanupInitiationFailed = true;
      }
    }
  }

  void _initiateAmbiguousCaptureRecovery() {
    final session = _session;
    if (session == null || _captureRecoveryAttempted) return;
    _captureRecoveryAttempted = true;
    _captureKnownClean = false;
    try {
      _captureRecoveryFuture = _observeRecovery(
        _owner._lifecycle.recoverAmbiguousCaptureStart(generation, session),
      );
    } catch (_) {
      _captureRecoverySucceeded = false;
      _synchronousCleanupInitiationFailed = true;
    }
  }

  void _initiateCleanupOperations() {
    // Revoke exact ASR callback authority before touching recorder/plugin work.
    _initiateSessionEnd();
    final capture = _capture;
    if (capture != null) {
      _initiateCaptureCleanup(capture);
    } else if (_captureStartAttempted &&
        _captureStartDisposition !=
            AssistantListeningCaptureStartDisposition.failedClean) {
      _initiateAmbiguousCaptureRecovery();
    }
    if (_synchronousCleanupInitiationFailed) _owner._poison(this);
  }

  Future<void> _recoverAmbiguousCapture(S _session) async {
    _initiateAmbiguousCaptureRecovery();
    final recovery = _captureRecoveryFuture;
    if (recovery != null) await recovery;
  }

  Future<void> _cleanupResources() async {
    if (_cleanupStarted) return;
    _cleanupStarted = true;
    _initiateCleanupOperations();
    if (_revoked) _armCleanupDeadline();

    final recovery = _captureRecoveryFuture;
    if (recovery != null) await recovery;

    final capture = _capture;
    if (capture != null) {
      _initiateCaptureCleanup(capture);
      final cancellation = _captureCancelFuture;
      final stop = _captureStopFuture;
      if (cancellation != null) await cancellation;
      if (stop != null) await stop;
      final sourceSettled =
          (_captureTerminalObserved && !_captureSourceErrorObserved) ||
              _captureCancelSucceeded;
      _captureKnownClean = sourceSettled && _captureStopSucceeded;
    }

    final session = _session;
    if (session != null) {
      _initiateSessionEnd();
      final terminal = _sessionTerminalFuture;
      if (_sessionEndAccepted && terminal != null) {
        await terminal;
      } else {
        _sessionKnownClean = false;
      }
      _sessionKnownClean = _sessionEndAccepted && _sessionTerminalSucceeded;
    }

    final exact = _captureKnownClean && _sessionKnownClean;
    if (exact) {
      _capture = null;
      _session = null;
      _sessionTerminalFuture = null;
      _captureRecoveryFuture = null;
      _captureCancelFuture = null;
      _captureStopFuture = null;
    }
    _invokeStopped();
    if (_cleanupPublished) {
      _owner._lateUnwindFinished(this, exactResourcesSettled: exact);
      return;
    }
    _publishCleanup(forceExactResourcesSettled: exact);
  }

  void _revoke(
    AssistantListeningOutcome outcome,
    AssistantListeningFailure? failure, {
    bool poison = false,
  }) {
    if (_revoked) {
      if (poison) _owner._poison(this);
      return;
    }
    _revoked = true;
    _listening = false;
    _listeningCallbackEnabled = false;
    _outcome = outcome;
    if (!_revokeSignal.isCompleted) _revokeSignal.complete();
    _completeDone(outcome, failure);
    _initiateCleanupOperations();
    final admissionTimer = _admissionTimer;
    _admissionTimer = null;
    if (!_cancelTimer(admissionTimer)) poison = true;
    final callback = _owner._onRevoke;
    if (callback != null) {
      _revokeCallbackAttempts += 1;
      try {
        callback(generation, outcome);
      } catch (_) {
        _revokeCallbackFailures += 1;
        poison = true;
        _owner._clearCallbacks();
      }
    }
    if (poison) _owner._poison(this);
    if (_started) _armCleanupDeadline();
  }

  void _completeDone(
    AssistantListeningOutcome outcome,
    AssistantListeningFailure? failure,
  ) {
    if (_done.isCompleted) return;
    _done.complete(AssistantListeningDone(outcome: outcome, failure: failure));
  }

  void _invokeListening() {
    final callback = _owner._onListening;
    if (callback == null || _revoked || !_listeningCallbackEnabled) return;
    _listeningCallbackAttempts += 1;
    try {
      callback(generation);
    } catch (_) {
      _listeningCallbackFailures += 1;
      _revoke(
        AssistantListeningOutcome.callbackFailed,
        const AssistantListeningFailure('listening_callback_failed'),
        poison: true,
      );
    }
  }

  void _invokeStopped() {
    final callback = _owner._onStopped;
    if (callback == null) return;
    _stoppedCallbackAttempts += 1;
    try {
      callback(generation);
    } catch (_) {
      _stoppedCallbackFailures += 1;
      _owner._callbackFailed(this);
    }
  }

  void _finishWithoutResources() {
    _captureKnownClean = true;
    _sessionKnownClean = true;
    _publishCleanup(forceExactResourcesSettled: true);
  }

  void _finishUnstarted(
    AssistantListeningOutcome outcome,
    AssistantListeningFailure? failure, {
    bool exactResourcesSettled = true,
  }) {
    if (_cleanupPublished) return;
    _outcome = outcome;
    _revoked = true;
    final timerCancelSucceeded = _cancelTimer(_admissionTimer);
    _admissionTimer = null;
    _completeDone(outcome, failure);
    if (!timerCancelSucceeded) _owner._poison(this);
    final callback = _owner._onRevoke;
    if (callback != null) {
      _revokeCallbackAttempts += 1;
      try {
        callback(generation, outcome);
      } catch (_) {
        _revokeCallbackFailures += 1;
        _owner._callbackFailed(this);
      }
    }
    _publishCleanup(forceExactResourcesSettled: exactResourcesSettled);
  }

  void _publishCleanup({required bool forceExactResourcesSettled}) {
    if (_cleanupPublished) return;
    _cleanupPublished = true;
    final admissionTimerCancelled = _cancelTimer(_admissionTimer);
    final cleanupTimerCancelled = _cancelTimer(_cleanupTimer);
    _admissionTimer = null;
    _cleanupTimer = null;
    if (!admissionTimerCancelled || !cleanupTimerCancelled) {
      _owner._poison(this);
    }

    final exactCapture = forceExactResourcesSettled && _captureKnownClean;
    final exactSession = forceExactResourcesSettled && _sessionKnownClean;
    final exactResources = exactCapture && exactSession;
    final receipt = AssistantListeningCleanupReceipt(
      ordinal: generation.ordinal,
      intent: _intent,
      outcome: _outcome,
      started: _started,
      admissionDeadlineExpired: _admissionDeadlineExpired,
      cleanupDeadlineExpired: _cleanupDeadlineExpired,
      permissionAttempted: _permissionAttempted,
      permissionGranted: _permissionGranted,
      routeAttempted: _routeAttempted,
      routeConfigured: _routeConfigured,
      sessionAcquireAttempted: _sessionAcquireAttempted,
      sessionAcquireReturned: _sessionAcquireReturned,
      sessionCleanupWatchAttempted: _sessionCleanupWatchAttempted,
      sessionCleanupWatchReturned: _sessionCleanupWatchReturned,
      sessionReadyAttempted: _sessionReadyAttempted,
      sessionReadySucceeded: _sessionReadySucceeded,
      captureStartAttempted: _captureStartAttempted,
      captureStartDisposition: _captureStartDisposition,
      captureRecoveryAttempted: _captureRecoveryAttempted,
      captureRecoverySucceeded: _captureRecoverySucceeded,
      captureTerminalWatchAttempted: _captureTerminalWatchAttempted,
      captureTerminalWatchReturned: _captureTerminalWatchReturned,
      captureTerminalObserved: _captureTerminalObserved,
      captureSourceErrorObserved: _captureSourceErrorObserved,
      captureCancelAttempted: _captureCancelAttempted,
      captureCancelSucceeded: _captureCancelSucceeded,
      captureStopAttempted: _captureStopAttempted,
      captureStopSucceeded: _captureStopSucceeded,
      sessionEndAttempted: _sessionEndAttempted,
      sessionEndAccepted: _sessionEndAccepted,
      sessionTerminalObserved: _sessionTerminalObserved,
      sessionTerminalSucceeded: _sessionTerminalSucceeded,
      revokeCallbackAttempts: _revokeCallbackAttempts,
      revokeCallbackFailures: _revokeCallbackFailures,
      listeningCallbackAttempts: _listeningCallbackAttempts,
      listeningCallbackFailures: _listeningCallbackFailures,
      stoppedCallbackAttempts: _stoppedCallbackAttempts,
      stoppedCallbackFailures: _stoppedCallbackFailures,
      exactCaptureSettled: exactCapture,
      exactSessionSettled: exactSession,
      exactResourcesSettled: exactResources,
      ownerPoisonedAtSettlement: _owner._poisoned,
    );
    if (!exactResources) {
      _owner._allExactResourcesSettled = false;
      _owner._retainedUncertainRun ??= this;
    }
    _cleanup.complete(receipt);
    if (_started) {
      _owner._activeFinished(this, receipt);
    } else {
      _owner._unstartedFinished(this, receipt);
    }
  }
}
