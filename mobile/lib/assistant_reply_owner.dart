// Exact StreamSubscription ownership for one widget-bound Assistant reply.
//
// This file is deliberately pure Dart. The widget supplies a backend adapter;
// tests can therefore control every callback, cancel Future, and deadline
// without loading Flutter, a model, a plugin, or an audio device.
import 'dart:async';
import 'dart:collection';
import 'dart:convert';

const int assistantReplyPromptMaximumUtf8Bytes = 16 * 1024;
const int assistantReplyChunkMaximumUtf8Bytes = 4 * 1024;
const int assistantReplyMaximumUtf8Bytes = 64 * 1024;
const int assistantReplyMaximumSourceDataEvents = 2048;
const int assistantReplyMaximumSafeOrdinal = 0x1fffffffffffff;
const Duration assistantReplyMaximumLifetime = Duration(seconds: 120);
const Duration assistantReplyCancelMaximumLifetime = Duration(seconds: 10);

typedef AssistantReplyTimerFactory = Timer Function(
  Duration duration,
  void Function() callback,
);

/// Opens the widget's lower reply stream.
///
/// This function must be side-effect-free until its returned stream is
/// listened to. A factory throw is therefore a clean startup failure; a listen
/// throw is ambiguous and poisons this owner.
typedef AssistantReplyOpen = Stream<String> Function(String prompt);
typedef AssistantReplyTokenCallback = void Function(
  AssistantReplyGeneration generation,
  String token,
);
typedef AssistantReplyFailureCallback = void Function(
  AssistantReplyGeneration generation,
  AssistantReplyFailure failure,
);

enum AssistantReplyOutcome {
  completed,
  cancelled,
  superseded,
  deadlineExceeded,
  outputLimitExceeded,
  sourceFailed,
  callbackFailed,
  startupFailed,
  listenAmbiguous,
  ownerClosed,
  ownerPoisoned,
}

/// The only outcomes a public caller may choose while fencing a generation.
enum AssistantReplyCancelReason { cancelled, superseded }

/// A bounded public failure; no provider exception text is retained.
final class AssistantReplyFailure implements Exception {
  const AssistantReplyFailure(this.code);

  final String code;

  @override
  String toString() => 'AssistantReplyFailure($code)';
}

/// Immediate UI-authority result, distinct from later source cleanup proof.
final class AssistantReplyDone {
  const AssistantReplyDone({required this.outcome, this.failure});

  final AssistantReplyOutcome outcome;
  final AssistantReplyFailure? failure;
}

/// Aggregate-only evidence that the exact Dart subscription settled.
final class AssistantReplyCleanupReceipt {
  const AssistantReplyCleanupReceipt({
    required this.ordinal,
    required this.outcome,
    required this.openAttempted,
    required this.openReturned,
    required this.sourceListenAttempted,
    required this.sourceListenReturned,
    required this.subscriptionCancelAttempted,
    required this.subscriptionCancelSucceeded,
    required this.sourceErrorObserved,
    required this.sourceDoneObserved,
    required this.observedSourceDataEvents,
    required this.callbackAttempts,
    required this.callbackAttemptUtf8Bytes,
    required this.exactSubscriptionSettled,
  });

  final int ordinal;
  final AssistantReplyOutcome outcome;
  final bool openAttempted;
  final bool openReturned;
  final bool sourceListenAttempted;
  final bool sourceListenReturned;
  final bool subscriptionCancelAttempted;
  final bool subscriptionCancelSucceeded;
  final bool sourceErrorObserved;
  final bool sourceDoneObserved;
  final int observedSourceDataEvents;

  /// Number of bounded token callbacks actually entered.
  final int callbackAttempts;

  /// UTF-8 bytes passed to those callback attempts.
  final int callbackAttemptUtf8Bytes;

  /// True only for this exact Dart subscription. It does not prove native
  /// generation stop, source terminal, chat close, or model cleanup.
  final bool exactSubscriptionSettled;
}

final class AssistantReplyCloseReceipt {
  const AssistantReplyCloseReceipt({
    required this.exactSubscriptionsSettled,
    required this.poisoned,
    required this.lastOrdinal,
  });

  /// Widget-local Dart subscription proof only, never native cleanup proof.
  final bool exactSubscriptionsSettled;
  final bool poisoned;
  final int lastOrdinal;
}

/// Opaque owner-keyed identity for one admitted reply.
///
/// [done] completes as soon as UI authority is fenced. [cleanup] completes
/// later, only after exact subscription cancellation or a true source onDone.
final class AssistantReplyGeneration {
  AssistantReplyGeneration._(this._ownerKey, this.ordinal, this._reply);

  final Object _ownerKey;
  final _OwnedAssistantReply _reply;

  /// Monotonic diagnostic only. Authority is exact object identity.
  final int ordinal;

  Future<AssistantReplyDone> get done => _reply.done;
  Future<AssistantReplyCleanupReceipt> get cleanup => _reply.cleanup;
  bool get isDone => _reply.isAuthorityDone;

  @override
  String toString() => 'AssistantReplyGeneration($ordinal)';
}

final class AssistantReplySnapshot {
  const AssistantReplySnapshot({
    required this.active,
    required this.pending,
    required this.poisoned,
    required this.closed,
    required this.retainsUncertainReply,
    required this.lastOrdinal,
  });

  final bool active;
  final bool pending;
  final bool poisoned;
  final bool closed;
  final bool retainsUncertainReply;
  final int lastOrdinal;
}

/// One active reply plus one latest-wins request that has done no backend work.
final class AssistantReplyOwner {
  factory AssistantReplyOwner({
    required AssistantReplyOpen openReply,
    Duration maximumLifetime = assistantReplyMaximumLifetime,
    Duration cancelMaximumLifetime = assistantReplyCancelMaximumLifetime,
    AssistantReplyTimerFactory timerFactory = Timer.new,
  }) =>
      AssistantReplyOwner._(
        openReply: openReply,
        maximumLifetime: maximumLifetime,
        cancelMaximumLifetime: cancelMaximumLifetime,
        timerFactory: timerFactory,
        initialOrdinal: 0,
      );

  factory AssistantReplyOwner.forTesting({
    required AssistantReplyOpen openReply,
    Duration maximumLifetime = assistantReplyMaximumLifetime,
    Duration cancelMaximumLifetime = assistantReplyCancelMaximumLifetime,
    AssistantReplyTimerFactory timerFactory = Timer.new,
    int initialOrdinal = 0,
  }) =>
      AssistantReplyOwner._(
        openReply: openReply,
        maximumLifetime: maximumLifetime,
        cancelMaximumLifetime: cancelMaximumLifetime,
        timerFactory: timerFactory,
        initialOrdinal: initialOrdinal,
      );

  AssistantReplyOwner._({
    required AssistantReplyOpen openReply,
    required Duration maximumLifetime,
    required Duration cancelMaximumLifetime,
    required AssistantReplyTimerFactory timerFactory,
    required int initialOrdinal,
  })  : _openReply = openReply,
        _maximumLifetime = maximumLifetime,
        _cancelMaximumLifetime = cancelMaximumLifetime,
        _timerFactory = timerFactory,
        _nextOrdinal = initialOrdinal {
    if (maximumLifetime <= Duration.zero) {
      throw ArgumentError.value(maximumLifetime, 'maximumLifetime');
    }
    if (cancelMaximumLifetime <= Duration.zero) {
      throw ArgumentError.value(
        cancelMaximumLifetime,
        'cancelMaximumLifetime',
      );
    }
    if (initialOrdinal < 0 ||
        initialOrdinal > assistantReplyMaximumSafeOrdinal) {
      throw ArgumentError.value(initialOrdinal, 'initialOrdinal');
    }
  }

  final AssistantReplyOpen _openReply;
  final Duration _maximumLifetime;
  final Duration _cancelMaximumLifetime;
  final AssistantReplyTimerFactory _timerFactory;
  final Object _ownerKey = Object();

  _OwnedAssistantReply? _active;
  _OwnedAssistantReply? _pending;
  _OwnedAssistantReply? _latest;
  _OwnedAssistantReply? _retainedUncertainReply;
  AssistantReplyCleanupReceipt? _lastReceipt;
  Future<AssistantReplyCloseReceipt>? _closeFuture;
  int _nextOrdinal;
  bool _closed = false;
  bool _poisoned = false;
  bool _allExactSubscriptionsSettled = true;

  bool get isClosed => _closed;
  bool get isPoisoned => _poisoned;
  bool get hasActive => _active != null;
  bool get hasPending => _pending != null;
  AssistantReplyCleanupReceipt? get lastReceipt => _lastReceipt;

  AssistantReplySnapshot get snapshot => AssistantReplySnapshot(
        active: _active != null,
        pending: _pending != null,
        poisoned: _poisoned,
        closed: _closed,
        retainsUncertainReply: _retainedUncertainReply != null,
        lastOrdinal: _nextOrdinal,
      );

  /// Admit a reply and return its opaque identity before backend callbacks run.
  AssistantReplyGeneration start({
    required String prompt,
    required AssistantReplyTokenCallback onToken,
    AssistantReplyFailureCallback? onFailure,
  }) {
    _validatePrompt(prompt);
    if (_closed) throw const AssistantReplyFailure('owner_closed');
    if (_poisoned) throw const AssistantReplyFailure('owner_poisoned');
    if (_nextOrdinal >= assistantReplyMaximumSafeOrdinal) {
      _poisoned = true;
      final pending = _pending;
      _pending = null;
      if (pending != null) {
        if (identical(_latest, pending)) _latest = null;
        pending._finishUnstarted(
          AssistantReplyOutcome.ownerPoisoned,
          failure: const AssistantReplyFailure('ordinal_exhausted'),
        );
      }
      final active = _active;
      if (active != null) {
        if (identical(_latest, active)) _latest = null;
        active._revoke(AssistantReplyOutcome.ownerPoisoned);
      }
      throw const AssistantReplyFailure('ordinal_exhausted');
    }

    final reply = _OwnedAssistantReply(
      owner: this,
      openReply: _openReply,
      ordinal: ++_nextOrdinal,
      prompt: prompt,
      onToken: onToken,
      onFailure: onFailure,
    );
    // Even a hostile Stream may emit synchronously from listen(). Public
    // callbacks open in a microtask, after this start() stack has returned.
    reply._scheduleDeliveryOpen();
    final active = _active;
    if (active == null) {
      if (!reply._armLifetimeDeadline()) {
        _poisoned = true;
        reply._finishUnstarted(
          AssistantReplyOutcome.ownerPoisoned,
          failure: const AssistantReplyFailure('deadline_timer_failed'),
        );
        return reply.generation;
      }
      _active = reply;
      _latest = reply;
      reply._start();
      return reply.generation;
    }

    final displaced = _pending;
    // Publish the newcomer first. Reentrant completion observers must see the
    // latest identity rather than being able to resurrect the displaced one.
    _pending = reply;
    _latest = reply;
    displaced?._finishUnstarted(AssistantReplyOutcome.superseded);
    if (!reply._armLifetimeDeadline()) {
      _poisonActive(active);
      if (identical(_pending, reply)) _pending = null;
      reply._finishUnstarted(
        AssistantReplyOutcome.ownerPoisoned,
        failure: const AssistantReplyFailure('deadline_timer_failed'),
      );
      active._revoke(AssistantReplyOutcome.ownerPoisoned);
      return reply.generation;
    }
    active._revoke(AssistantReplyOutcome.superseded);
    return reply.generation;
  }

  bool isAuthoritative(AssistantReplyGeneration generation) =>
      identical(generation._ownerKey, _ownerKey) &&
      identical(_latest, generation._reply) &&
      !generation._reply.isAuthorityDone &&
      !_closed;

  /// Fence one same-owner generation and return its exact cleanup receipt.
  Future<AssistantReplyCleanupReceipt> cancelExact(
    AssistantReplyGeneration generation, {
    AssistantReplyCancelReason reason = AssistantReplyCancelReason.cancelled,
  }) {
    if (!identical(generation._ownerKey, _ownerKey)) {
      throw ArgumentError('foreign AssistantReplyGeneration');
    }
    final reply = generation._reply;
    final outcome = _cancelOutcome(reason);
    if (identical(_pending, reply)) {
      _pending = null;
      if (identical(_latest, reply)) _latest = null;
      reply._finishUnstarted(outcome);
    } else if (identical(_active, reply)) {
      if (identical(_latest, reply)) _latest = null;
      reply._revoke(outcome);
    }
    return reply.cleanup;
  }

  /// Fence the current UI generation now; cleanup can finish later.
  Future<AssistantReplyCleanupReceipt?> cancelCurrent({
    AssistantReplyCancelReason reason = AssistantReplyCancelReason.cancelled,
  }) {
    final outcome = _cancelOutcome(reason);
    final latest = _latest;
    if (latest == null) return Future.value(null);
    if (identical(_pending, latest)) {
      _pending = null;
      _latest = null;
      latest._finishUnstarted(outcome);
      return latest.cleanup
          .then<AssistantReplyCleanupReceipt?>((value) => value);
    }
    if (identical(_active, latest)) {
      _latest = null;
      latest._revoke(outcome);
      return latest.cleanup
          .then<AssistantReplyCleanupReceipt?>((value) => value);
    }
    _latest = null;
    return latest.cleanup.then<AssistantReplyCleanupReceipt?>((value) => value);
  }

  /// Permanently close the widget owner. The exact Future is returned forever.
  Future<AssistantReplyCloseReceipt> close() {
    final existing = _closeFuture;
    if (existing != null) return existing;

    final completed = Completer<AssistantReplyCloseReceipt>();
    _closeFuture = completed.future;
    _closed = true;
    _latest = null;

    final pending = _pending;
    _pending = null;
    pending?._finishUnstarted(AssistantReplyOutcome.ownerClosed);

    final active = _active;
    if (active == null) {
      completed.complete(
        _closeReceipt(
          exactSubscriptionsSettled: _allExactSubscriptionsSettled,
        ),
      );
    } else {
      active._revoke(AssistantReplyOutcome.ownerClosed);
      unawaited(
        active.cleanup.then(
          (receipt) => completed.complete(
            _closeReceipt(
              exactSubscriptionsSettled: receipt.exactSubscriptionSettled &&
                  _allExactSubscriptionsSettled,
            ),
          ),
          onError: (_error, _stackTrace) {
            _poisoned = true;
            _allExactSubscriptionsSettled = false;
            completed.complete(
              _closeReceipt(exactSubscriptionsSettled: false),
            );
          },
        ),
      );
    }
    return completed.future;
  }

  AssistantReplyCloseReceipt _closeReceipt({
    required bool exactSubscriptionsSettled,
  }) =>
      AssistantReplyCloseReceipt(
        exactSubscriptionsSettled: exactSubscriptionsSettled,
        poisoned: _poisoned,
        lastOrdinal: _nextOrdinal,
      );

  Timer _newTimer(Duration duration, void Function() callback) =>
      _timerFactory(duration, callback);

  static AssistantReplyOutcome _cancelOutcome(
    AssistantReplyCancelReason reason,
  ) =>
      switch (reason) {
        AssistantReplyCancelReason.cancelled => AssistantReplyOutcome.cancelled,
        AssistantReplyCancelReason.superseded =>
          AssistantReplyOutcome.superseded,
      };

  bool _isActive(_OwnedAssistantReply reply) => identical(_active, reply);

  void _pendingDeadlineExpired(_OwnedAssistantReply reply) {
    if (!identical(_pending, reply) || reply.isCleanupDone) return;
    _pending = null;
    if (identical(_latest, reply)) _latest = null;
    reply._finishUnstarted(
      AssistantReplyOutcome.deadlineExceeded,
      failure: const AssistantReplyFailure('reply_deadline_exceeded'),
    );
  }

  void _poisonActive(_OwnedAssistantReply reply) {
    _poisoned = true;
    final pending = _pending;
    _pending = null;
    if (pending != null && !identical(pending, reply)) {
      if (identical(_latest, pending)) _latest = null;
      pending._finishUnstarted(
        AssistantReplyOutcome.ownerPoisoned,
        failure: const AssistantReplyFailure('owner_poisoned'),
      );
    }
  }

  void _unstartedFinished(AssistantReplyCleanupReceipt receipt) {
    _lastReceipt = receipt;
  }

  void _activeFinished(
    _OwnedAssistantReply reply,
    AssistantReplyCleanupReceipt receipt,
  ) {
    if (!identical(_active, reply)) {
      _poisoned = true;
      _allExactSubscriptionsSettled = false;
      _retainedUncertainReply ??= reply;
      _poisonActive(reply);
      return;
    }
    _lastReceipt = receipt;
    if (!receipt.exactSubscriptionSettled) {
      _allExactSubscriptionsSettled = false;
      _retainedUncertainReply ??= reply;
      _poisonActive(reply);
      return;
    }

    _active = null;
    if (identical(_latest, reply)) _latest = null;
    if (_closed || _poisoned) {
      final pending = _pending;
      _pending = null;
      pending?._finishUnstarted(
        _closed
            ? AssistantReplyOutcome.ownerClosed
            : AssistantReplyOutcome.ownerPoisoned,
        failure:
            _poisoned ? const AssistantReplyFailure('owner_poisoned') : null,
      );
      return;
    }

    final next = _pending;
    _pending = null;
    if (next != null && !next.isCleanupDone) {
      _active = next;
      _latest = next;
      next._start();
    }
  }

  static void _validatePrompt(String prompt) {
    if (prompt.isEmpty ||
        prompt.length > assistantReplyPromptMaximumUtf8Bytes) {
      throw const AssistantReplyFailure('invalid_prompt');
    }
    final bytes = utf8.encode(prompt).length;
    if (bytes == 0 || bytes > assistantReplyPromptMaximumUtf8Bytes) {
      throw const AssistantReplyFailure('invalid_prompt');
    }
  }
}

final class _OwnedAssistantReply {
  _OwnedAssistantReply({
    required AssistantReplyOwner owner,
    required AssistantReplyOpen openReply,
    required int ordinal,
    required String prompt,
    required AssistantReplyTokenCallback onToken,
    required AssistantReplyFailureCallback? onFailure,
  })  : _owner = owner,
        _openReply = openReply,
        _prompt = prompt,
        _onToken = onToken,
        _onFailure = onFailure {
    generation = AssistantReplyGeneration._(
      owner._ownerKey,
      ordinal,
      this,
    );
  }

  final AssistantReplyOwner _owner;
  final AssistantReplyOpen _openReply;
  late final AssistantReplyGeneration generation;
  final Completer<AssistantReplyDone> _done = Completer<AssistantReplyDone>();
  final Completer<AssistantReplyCleanupReceipt> _cleanup =
      Completer<AssistantReplyCleanupReceipt>();
  final ListQueue<String> _deferredTokens = ListQueue<String>();

  String? _prompt;
  AssistantReplyTokenCallback? _onToken;
  AssistantReplyFailureCallback? _onFailure;
  AssistantReplyFailureCallback? _deferredFailureCallback;
  AssistantReplyFailure? _deferredFailure;
  AssistantReplyOutcome? _startupFailureOutcome;
  AssistantReplyFailure? _startupFailure;
  StreamSubscription<String>? _subscription;
  Timer? _lifetimeDeadline;
  Timer? _cancelDeadline;

  AssistantReplyOutcome _outcome = AssistantReplyOutcome.completed;
  int _observedSourceDataEvents = 0;
  int _acceptedUtf8Bytes = 0;
  int _callbackAttempts = 0;
  int _callbackAttemptUtf8Bytes = 0;
  bool _started = false;
  bool _listenInProgress = false;
  bool _deliveryOpen = false;
  bool _deliveryOpenScheduled = false;
  bool _deliveryFlushScheduled = false;
  bool _naturalCompletionPending = false;
  bool _openAttempted = false;
  bool _openReturned = false;
  bool _sourceListenAttempted = false;
  bool _sourceListenReturned = false;
  bool _subscriptionCancelAttempted = false;
  bool _subscriptionCancelSucceeded = false;
  bool _sourceErrorObserved = false;
  bool _sourceDoneObserved = false;
  bool _authorityDone = false;
  bool _cancellationStarted = false;
  bool _cleanupDone = false;

  Future<AssistantReplyDone> get done => _done.future;
  Future<AssistantReplyCleanupReceipt> get cleanup => _cleanup.future;
  bool get isAuthorityDone => _authorityDone;
  bool get isCleanupDone => _cleanupDone;

  void _scheduleDeliveryOpen() {
    if (_deliveryOpen || _deliveryOpenScheduled) return;
    _deliveryOpenScheduled = true;
    scheduleMicrotask(() {
      _deliveryOpenScheduled = false;
      _deliveryOpen = true;
      _flushDeferredDelivery();
    });
  }

  void _scheduleDeferredFlush() {
    if (_deliveryFlushScheduled) return;
    _deliveryFlushScheduled = true;
    scheduleMicrotask(() {
      _deliveryFlushScheduled = false;
      _flushDeferredDelivery();
    });
  }

  void _flushDeferredDelivery() {
    if (!_deliveryOpen || _listenInProgress) return;
    final deferredFailure = _deferredFailure;
    final failureCallback = _deferredFailureCallback;
    _deferredFailure = null;
    _deferredFailureCallback = null;
    if (deferredFailure != null && failureCallback != null) {
      try {
        failureCallback(generation, deferredFailure);
      } catch (_) {
        // A diagnostic callback has no lifecycle authority.
      }
    }
    if (_authorityDone) {
      _deferredTokens.clear();
      return;
    }
    while (!_authorityDone && _deferredTokens.isNotEmpty) {
      _deliverToken(_deferredTokens.removeFirst());
    }
    if (!_authorityDone &&
        _naturalCompletionPending &&
        _deferredTokens.isEmpty) {
      _naturalCompletionPending = false;
      _completeNatural();
    }
  }

  bool _armLifetimeDeadline() {
    try {
      _lifetimeDeadline = _owner._newTimer(
        _owner._maximumLifetime,
        _onLifetimeDeadline,
      );
      return true;
    } catch (_) {
      return false;
    }
  }

  void _onLifetimeDeadline() {
    _lifetimeDeadline = null;
    if (_started) {
      _onActiveDeadline();
    } else {
      _owner._pendingDeadlineExpired(this);
    }
  }

  void _start() {
    if (_started || _cleanupDone) return;
    _started = true;
    if (_authorityDone) {
      _settleCleanup(exactSubscriptionSettled: true);
      return;
    }

    final prompt = _prompt;
    if (prompt == null) {
      _finishCleanStartupFailure('prompt_unavailable');
      return;
    }

    Stream<String> source;
    // This barrier spans both factory return and listen return. A hostile
    // synchronous stream may revoke this run, but cleanup waits for the exact
    // subscription handle before attempting exact subscription cancellation.
    _listenInProgress = true;
    _openAttempted = true;
    try {
      source = _openReply(prompt);
      _openReturned = true;
    } catch (_) {
      _listenInProgress = false;
      _prompt = null;
      _finishCleanStartupFailure('reply_start_failed');
      return;
    }
    _prompt = null;

    StreamSubscription<String> subscription;
    _sourceListenAttempted = true;
    try {
      subscription = source.listen(
        _onSourceData,
        onError: _onSourceError,
        onDone: _onSourceDone,
        cancelOnError: false,
      );
      _sourceListenReturned = true;
    } catch (_) {
      _listenInProgress = false;
      _finishAmbiguousListen();
      return;
    }
    _listenInProgress = false;
    _subscription = subscription;
    _scheduleDeferredFlush();

    if (_authorityDone) {
      if (_sourceDoneObserved && !_cancellationStarted) {
        _settleCleanup(exactSubscriptionSettled: true);
      } else {
        _beginCancellation();
      }
      return;
    }
    final startupFailureOutcome = _startupFailureOutcome;
    final startupFailure = _startupFailure;
    _startupFailureOutcome = null;
    _startupFailure = null;
    if (startupFailureOutcome != null && startupFailure != null) {
      _failAndCancel(startupFailureOutcome, startupFailure);
      return;
    }
    if (_sourceDoneObserved) {
      _naturalCompletionPending = true;
      _scheduleDeferredFlush();
      return;
    }
  }

  void _finishCleanStartupFailure(String code) {
    _completeAuthority(
      AssistantReplyOutcome.startupFailed,
      AssistantReplyFailure(code),
    );
    _settleCleanup(exactSubscriptionSettled: true);
  }

  void _finishAmbiguousListen() {
    _owner._poisonActive(this);
    _completeAuthority(
      AssistantReplyOutcome.listenAmbiguous,
      const AssistantReplyFailure('reply_listen_ambiguous'),
    );
    _settleCleanup(exactSubscriptionSettled: false);
  }

  void _onSourceData(String token) {
    if (_cleanupDone ||
        _authorityDone ||
        _startupFailure != null ||
        _sourceDoneObserved ||
        !_owner._isActive(this)) {
      return;
    }
    _observedSourceDataEvents += 1;
    if (_observedSourceDataEvents > assistantReplyMaximumSourceDataEvents) {
      _failAndCancel(
        AssistantReplyOutcome.outputLimitExceeded,
        const AssistantReplyFailure('reply_event_limit_exceeded'),
      );
      return;
    }
    if (token.length > assistantReplyChunkMaximumUtf8Bytes) {
      _failAndCancel(
        AssistantReplyOutcome.outputLimitExceeded,
        const AssistantReplyFailure('reply_chunk_limit_exceeded'),
      );
      return;
    }
    final tokenBytes = utf8.encode(token).length;
    if (tokenBytes > assistantReplyChunkMaximumUtf8Bytes) {
      _failAndCancel(
        AssistantReplyOutcome.outputLimitExceeded,
        const AssistantReplyFailure('reply_chunk_limit_exceeded'),
      );
      return;
    }
    if (_acceptedUtf8Bytes > assistantReplyMaximumUtf8Bytes - tokenBytes) {
      _failAndCancel(
        AssistantReplyOutcome.outputLimitExceeded,
        const AssistantReplyFailure('reply_aggregate_limit_exceeded'),
      );
      return;
    }

    _acceptedUtf8Bytes += tokenBytes;
    if (_listenInProgress || !_deliveryOpen) {
      _deferredTokens.addLast(token);
      return;
    }
    _deliverToken(token);
  }

  void _deliverToken(String token) {
    if (_cleanupDone || _authorityDone || !_owner._isActive(this)) return;
    final callback = _onToken;
    if (callback == null) return;
    _callbackAttempts += 1;
    _callbackAttemptUtf8Bytes += utf8.encode(token).length;
    try {
      callback(generation, token);
    } catch (_) {
      _failAndCancel(
        AssistantReplyOutcome.callbackFailed,
        const AssistantReplyFailure('reply_callback_failed'),
      );
    }
  }

  void _onSourceError(Object _error, StackTrace _stackTrace) {
    if (_cleanupDone ||
        _authorityDone ||
        _startupFailure != null ||
        _sourceDoneObserved) {
      return;
    }
    _sourceErrorObserved = true;
    _failAndCancel(
      AssistantReplyOutcome.sourceFailed,
      const AssistantReplyFailure('reply_source_failed'),
    );
  }

  void _onSourceDone() {
    if (_cleanupDone || _sourceDoneObserved) return;
    _sourceDoneObserved = true;
    if (_authorityDone) return;
    _naturalCompletionPending = true;
    if (_listenInProgress || !_deliveryOpen || _deferredTokens.isNotEmpty) {
      _scheduleDeferredFlush();
    } else {
      _naturalCompletionPending = false;
      _completeNatural();
    }
  }

  void _completeNatural() {
    if (_cleanupDone || _authorityDone) return;
    _completeAuthority(AssistantReplyOutcome.completed, null);
    _settleCleanup(exactSubscriptionSettled: true);
  }

  void _onActiveDeadline() {
    if (_cleanupDone || _authorityDone || !_owner._isActive(this)) return;
    _owner._poisonActive(this);
    _failAndCancel(
      AssistantReplyOutcome.deadlineExceeded,
      const AssistantReplyFailure('reply_deadline_exceeded'),
    );
  }

  void _failAndCancel(
    AssistantReplyOutcome outcome,
    AssistantReplyFailure failure,
  ) {
    if (_cleanupDone || _authorityDone) return;
    if (_listenInProgress) {
      _startupFailureOutcome ??= outcome;
      _startupFailure ??= failure;
      return;
    }
    _completeAuthority(outcome, failure);
    _beginCancellation();
  }

  void _revoke(AssistantReplyOutcome outcome) {
    if (_cleanupDone || _authorityDone) return;
    _completeAuthority(outcome, null);
    if (_sourceDoneObserved && !_cancellationStarted) {
      _settleCleanup(exactSubscriptionSettled: true);
      return;
    }
    if (!_started) {
      _settleCleanup(exactSubscriptionSettled: true);
      return;
    }
    _beginCancellation();
  }

  void _completeAuthority(
    AssistantReplyOutcome outcome,
    AssistantReplyFailure? failure,
  ) {
    if (_authorityDone) return;
    _authorityDone = true;
    _outcome = outcome;
    // Clear all content callbacks before publishing done or cancelling the
    // exact subscription. No late callback can regain UI authority.
    _onToken = null;
    _deferredTokens.clear();
    _naturalCompletionPending = false;
    final failureCallback = _onFailure;
    _onFailure = null;
    _prompt = null;
    _done.complete(AssistantReplyDone(outcome: outcome, failure: failure));
    if (failure != null && failureCallback != null) {
      if (_deliveryOpen && !_listenInProgress) {
        try {
          failureCallback(generation, failure);
        } catch (_) {
          // A diagnostic callback has no lifecycle authority.
        }
      } else {
        _deferredFailure = failure;
        _deferredFailureCallback = failureCallback;
      }
    }
  }

  void _beginCancellation() {
    if (_cleanupDone || _cancellationStarted || _listenInProgress) return;
    if (_sourceDoneObserved) {
      _settleCleanup(exactSubscriptionSettled: true);
      return;
    }
    if (!_owner._isActive(this)) {
      _owner._poisonActive(this);
      _settleCleanup(exactSubscriptionSettled: false);
      return;
    }
    final subscription = _subscription;
    if (subscription == null) {
      _owner._poisonActive(this);
      _settleCleanup(exactSubscriptionSettled: false);
      return;
    }

    _cancellationStarted = true;
    _lifetimeDeadline?.cancel();
    _lifetimeDeadline = null;
    var deadlineArmed = true;
    try {
      _cancelDeadline = _owner._newTimer(
        _owner._cancelMaximumLifetime,
        _onCancelDeadline,
      );
    } catch (_) {
      deadlineArmed = false;
      _owner._poisonActive(this);
    }

    // This retained subscription is the only upper-layer cancellation
    // authority. Its onCancel synchronously revokes the exact lower owner; no
    // wider cancellation call can cross widget-owner boundaries.
    _subscriptionCancelAttempted = true;

    Future<void> cancelled;
    try {
      cancelled = subscription.cancel();
    } catch (_) {
      _subscriptionCancelSucceeded = false;
      _owner._poisonActive(this);
      _cancelDeadline?.cancel();
      _cancelDeadline = null;
      _settleCleanup(exactSubscriptionSettled: false);
      return;
    }
    unawaited(
      cancelled.then<void>(
        (_) => _subscriptionCancellationFinished(succeeded: true),
        onError: (_error, _stackTrace) {
          _subscriptionCancellationFinished(succeeded: false);
        },
      ),
    );
    if (!deadlineArmed && !_cleanupDone) {
      _settleCleanup(exactSubscriptionSettled: false);
    }
  }

  void _subscriptionCancellationFinished({required bool succeeded}) {
    if (_cleanupDone) return;
    _cancelDeadline?.cancel();
    _cancelDeadline = null;
    _subscriptionCancelSucceeded = succeeded;
    if (!succeeded) {
      _owner._poisonActive(this);
    }
    _settleCleanup(
      exactSubscriptionSettled: succeeded && _owner._isActive(this),
    );
  }

  void _onCancelDeadline() {
    _cancelDeadline = null;
    if (_cleanupDone || !_cancellationStarted) return;
    _owner._poisonActive(this);
    _settleCleanup(exactSubscriptionSettled: false);
  }

  void _finishUnstarted(
    AssistantReplyOutcome outcome, {
    AssistantReplyFailure? failure,
  }) {
    if (_cleanupDone) return;
    _lifetimeDeadline?.cancel();
    _lifetimeDeadline = null;
    _completeAuthority(outcome, failure);
    _settleCleanup(exactSubscriptionSettled: true);
  }

  void _settleCleanup({required bool exactSubscriptionSettled}) {
    if (_cleanupDone) return;
    _cleanupDone = true;
    _lifetimeDeadline?.cancel();
    _cancelDeadline?.cancel();
    _lifetimeDeadline = null;
    _cancelDeadline = null;
    _prompt = null;
    _onToken = null;
    _onFailure = null;
    _startupFailureOutcome = null;
    _startupFailure = null;

    final receipt = AssistantReplyCleanupReceipt(
      ordinal: generation.ordinal,
      outcome: _outcome,
      openAttempted: _openAttempted,
      openReturned: _openReturned,
      sourceListenAttempted: _sourceListenAttempted,
      sourceListenReturned: _sourceListenReturned,
      subscriptionCancelAttempted: _subscriptionCancelAttempted,
      subscriptionCancelSucceeded: _subscriptionCancelSucceeded,
      sourceErrorObserved: _sourceErrorObserved,
      sourceDoneObserved: _sourceDoneObserved,
      observedSourceDataEvents: _observedSourceDataEvents,
      callbackAttempts: _callbackAttempts,
      callbackAttemptUtf8Bytes: _callbackAttemptUtf8Bytes,
      exactSubscriptionSettled: exactSubscriptionSettled,
    );
    if (exactSubscriptionSettled) _subscription = null;
    _cleanup.complete(receipt);
    if (_started) {
      _owner._activeFinished(this, receipt);
    } else {
      _owner._unstartedFinished(receipt);
    }
  }
}
