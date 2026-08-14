// Streaming ASR with exact, bounded session ownership.
//
// The app isolate owns admission and callback authority. The worker isolate
// owns sherpa-onnx objects. A reset acknowledgement means only that the worker
// accepted the tagged session after its Dart cleanup calls returned; it is not
// evidence that native work or native destruction has finished.
import 'dart:async';
import 'dart:convert';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './asr_model.dart';

typedef AsrTextCallback = void Function(String text);
typedef AsrWorkerEventSink = void Function(AsrWorkerEvent event);
typedef AsrWorkerStartupFactory = Future<AsrWorkerStartup> Function(
  AsrWorkerEventSink onEvent,
  AsrWorkerLaunchFence launchFence,
);
typedef AsrTimerFactory = AsrTimerHandle Function(
  Duration duration,
  void Function() callback,
);

abstract interface class AsrTimerHandle {
  void cancel();
}

/// One-way launch permit. Only AsrService can revoke it; startup factories
/// observe it after every asynchronous construction boundary.
final class AsrWorkerLaunchFence {
  bool _current = true;

  bool get isCurrent => _current;

  void _revoke() => _current = false;
}

final class _DartTimerHandle implements AsrTimerHandle {
  _DartTimerHandle(this._timer);

  final Timer _timer;

  @override
  void cancel() => _timer.cancel();
}

/// Opaque capability for one exact ASR listen session.
///
/// [ordinal] is safe to log as a bounded diagnostic. It is not authority;
/// methods also require object identity and the service-private owner key.
final class AsrSession {
  AsrSession._(this._owner, this._state);

  static const int maxSafeOrdinal = 0x1fffffffffffff;

  final Object _owner;
  final _AsrSessionState _state;

  int get ordinal => _state.ordinal;
  Future<void> get ready => _state.ready.future;
  Future<bool> get cleanup => _state.cleanup.future;
}

/// A worker transport is retained even after a failed close. Its close future
/// is prepublished and memoized so reentrant/concurrent callers get the exact
/// same receipt and a failed close is never silently retried.
final class AsrWorkerTransport {
  AsrWorkerTransport({
    required void Function(AsrWorkerCommand command) send,
    required Future<bool> Function() close,
  })  : _send = send,
        _close = close;

  final void Function(AsrWorkerCommand command) _send;
  final Future<bool> Function() _close;
  Future<bool>? _closeFuture;

  void send(AsrWorkerCommand command) => _send(command);

  Future<bool> close() {
    final existing = _closeFuture;
    if (existing != null) return existing;
    final completer = Completer<bool>();
    final result = completer.future;
    _closeFuture = result;
    Future<bool>.sync(_close).then(
      completer.complete,
      onError: (Object error, StackTrace stackTrace) {
        completer.complete(false);
      },
    );
    return result;
  }
}

final class AsrWorkerStartup {
  AsrWorkerStartup({required this.ready, required this.transport}) {
    // Observe failures immediately so a revoked startup cannot report an
    // unhandled Zone error before the owning session awaits this same future.
    unawaited(
      ready.then<void>(
        (_) {},
        onError: (Object error, StackTrace stackTrace) {},
      ),
    );
  }

  final Future<void> ready;
  final AsrWorkerTransport transport;

  Future<bool> close() => transport.close();
}

sealed class AsrWorkerCommand {
  const AsrWorkerCommand();
}

final class AsrWorkerReset extends AsrWorkerCommand {
  const AsrWorkerReset(this.ordinal);
  final int ordinal;
}

final class AsrWorkerAudio extends AsrWorkerCommand {
  const AsrWorkerAudio(this.ordinal, this.sequence, this.bytes);
  final int ordinal;
  final int sequence;
  final Uint8List bytes;
}

final class AsrWorkerEnd extends AsrWorkerCommand {
  const AsrWorkerEnd(this.ordinal);
  final int ordinal;
}

final class AsrWorkerEndAck extends AsrWorkerEvent {
  const AsrWorkerEndAck(this.ordinal, {required this.released});
  final int ordinal;
  final bool released;
}

final class AsrWorkerShutdown extends AsrWorkerCommand {
  const AsrWorkerShutdown(this.shutdownOrdinal);
  final int shutdownOrdinal;
}

sealed class AsrWorkerEvent {
  const AsrWorkerEvent();
}

final class AsrWorkerResetAck extends AsrWorkerEvent {
  const AsrWorkerResetAck(this.ordinal);
  final int ordinal;
}

final class AsrWorkerAudioAck extends AsrWorkerEvent {
  const AsrWorkerAudioAck(this.ordinal, this.sequence);
  final int ordinal;
  final int sequence;
}

final class AsrWorkerPartial extends AsrWorkerEvent {
  const AsrWorkerPartial(this.ordinal, this.text);
  final int ordinal;
  final String text;
}

final class AsrWorkerEndpoint extends AsrWorkerEvent {
  const AsrWorkerEndpoint(this.ordinal, this.text);
  final int ordinal;
  final String text;
}

final class AsrWorkerSessionFailure extends AsrWorkerEvent {
  const AsrWorkerSessionFailure(
    this.ordinal,
    this.code, {
    required this.sessionReleased,
    required this.workerHealthy,
  });
  final int ordinal;
  final String code;
  final bool sessionReleased;
  final bool workerHealthy;
}

final class AsrWorkerLifecycleFailure extends AsrWorkerEvent {
  const AsrWorkerLifecycleFailure(this.code);
  final String code;
}

final class AsrWorkerShutdownAck extends AsrWorkerEvent {
  const AsrWorkerShutdownAck(this.shutdownOrdinal);
  final int shutdownOrdinal;
}

enum _AsrSessionDisposition { active, ending, released, uncertain }

final class _AsrSessionState {
  _AsrSessionState({
    required this.ordinal,
    required this.onPartial,
    required this.onEndpoint,
  });

  final int ordinal;
  final Completer<void> ready = Completer<void>();
  final Completer<bool> cleanup = Completer<bool>();
  AsrTextCallback? onPartial;
  AsrTextCallback? onEndpoint;
  AsrTimerHandle? timer;
  AsrTimerHandle? cleanupTimer;
  _AsrSessionDisposition disposition = _AsrSessionDisposition.active;
  bool resetSent = false;
  bool resetAcknowledged = false;
  bool endSent = false;
  int nextAudioSequence = 1;
  final Set<int> outstandingAudio = <int>{};
  late Future<bool> admissionBarrier;
}

class AsrService {
  AsrService._production()
      : _startupFactory = _startProductionWorker,
        _admissionTimeout = admissionTimeout,
        _cleanupTimeout = sessionCleanupTimeout,
        _closeTimeout = workerCloseTimeout,
        _timerFactory = _defaultTimerFactory,
        _lastOrdinal = 0;

  AsrService.forTesting({
    required AsrWorkerStartupFactory startupFactory,
    Duration admissionTimeout = const Duration(seconds: 30),
    Duration cleanupTimeout = const Duration(seconds: 10),
    Duration closeTimeout = const Duration(seconds: 10),
    AsrTimerFactory timerFactory = _defaultTimerFactory,
    int initialOrdinal = 0,
  })  : _startupFactory = startupFactory,
        _admissionTimeout = admissionTimeout,
        _cleanupTimeout = cleanupTimeout,
        _closeTimeout = closeTimeout,
        _timerFactory = timerFactory,
        _lastOrdinal = initialOrdinal {
    if (initialOrdinal < 0 || initialOrdinal > AsrSession.maxSafeOrdinal) {
      throw ArgumentError.value(initialOrdinal, 'initialOrdinal');
    }
    if (admissionTimeout <= Duration.zero ||
        cleanupTimeout <= Duration.zero ||
        closeTimeout <= Duration.zero) {
      throw ArgumentError('ASR timeouts must be positive');
    }
  }

  static final AsrService instance = AsrService._production();

  static const endpointSilenceSec = 0.8;
  static const admissionTimeout = Duration(seconds: 30);
  static const sessionCleanupTimeout = Duration(seconds: 10);
  static const workerCloseTimeout = Duration(seconds: 10);
  static const int maxPcmBytesPerChunk = 32768;
  static const int maxOutstandingAudioChunks = 4;
  static const int maxResultUtf8Bytes = 16384;
  static const int maxDecodeStepsPerChunk = 2048;

  final Object _owner = Object();
  final AsrWorkerStartupFactory _startupFactory;
  final Duration _admissionTimeout;
  final Duration _cleanupTimeout;
  final Duration _closeTimeout;
  final AsrTimerFactory _timerFactory;
  int _lastOrdinal;
  int _launchGeneration = 0;
  int? _closingEventGeneration;
  _AsrSessionState? _active;
  final Map<int, _AsrSessionState> _releasing = <int, _AsrSessionState>{};
  Future<bool> _admissionTail = Future<bool>.value(true);
  Future<AsrWorkerStartup>? _startupFuture;
  AsrWorkerLaunchFence? _startupFence;
  AsrWorkerStartup? _startup;
  Future<bool>? _closeFuture;
  String? _retainedCloseFailureCode;
  bool _closing = false;
  bool _closed = false;
  bool _transportUncertain = false;

  /// Bounded diagnostic only; never contains an exception or platform text.
  String? get closeFailureCode => _retainedCloseFailureCode;

  static AsrTimerHandle _defaultTimerFactory(
    Duration duration,
    void Function() callback,
  ) =>
      _DartTimerHandle(Timer(duration, callback));

  /// Starts admission synchronously and returns exact authority immediately.
  /// [ready] spans app-lifetime worker startup and this session's reset ACK.
  AsrSession beginSession({
    required AsrTextCallback onPartial,
    required AsrTextCallback onEndpoint,
  }) {
    if (_closing || _closed) throw StateError('asr_service_closed');
    if (_lastOrdinal >= AsrSession.maxSafeOrdinal) {
      throw StateError('asr_session_ordinal_exhausted');
    }

    final predecessor = _active;
    if (predecessor != null) {
      _beginEnd(predecessor, readyCode: 'asr_superseded');
    }

    final state = _AsrSessionState(
      ordinal: ++_lastOrdinal,
      onPartial: onPartial,
      onEndpoint: onEndpoint,
    );
    final token = AsrSession._(_owner, state);
    state.admissionBarrier = _admissionTail;
    _admissionTail = _chainReceipts(
      state.admissionBarrier,
      state.cleanup.future,
    );
    // Observe internal errors immediately. Consumers still receive the same
    // future/error, but end-before-await cannot create an unhandled Zone error.
    unawaited(
      state.ready.future.then<void>(
        (_) {},
        onError: (Object error, StackTrace stackTrace) {},
      ),
    );
    _active = state;

    try {
      final timer = _timerFactory(
        _admissionTimeout,
        () => _onAdmissionTimeout(state),
      );
      if (state.disposition == _AsrSessionDisposition.active) {
        state.timer = timer;
      } else {
        _cancelTimer(timer);
      }
    } catch (_) {
      _failAdmission(state, 'asr_admission_timer_failed');
      return token;
    }

    if (_transportUncertain) {
      _failAdmission(state, 'asr_transport_uncertain');
      return token;
    }
    unawaited(_admit(state));
    return token;
  }

  /// Copies and enqueues one bounded PCM16 chunk only for the exact admitted
  /// session. Four unacknowledged chunks is a hard mailbox backpressure cap.
  bool feed(AsrSession session, Uint8List bytes) {
    final state = _ownedState(session);
    if (state == null ||
        !identical(_active, state) ||
        state.disposition != _AsrSessionDisposition.active ||
        !state.resetAcknowledged ||
        _closing ||
        _closed ||
        _transportUncertain) {
      return false;
    }
    if (bytes.isEmpty ||
        bytes.length.isOdd ||
        bytes.length > maxPcmBytesPerChunk ||
        state.outstandingAudio.length >= maxOutstandingAudioChunks ||
        state.nextAudioSequence > AsrSession.maxSafeOrdinal) {
      return false;
    }

    final sequence = state.nextAudioSequence++;
    final copy = Uint8List.fromList(bytes);
    state.outstandingAudio.add(sequence);
    try {
      _startup!.transport.send(
        AsrWorkerAudio(state.ordinal, sequence, copy),
      );
      return true;
    } catch (_) {
      _transportUncertain = true;
      _beginEnd(
        state,
        readyCode: 'asr_audio_send_failed',
        forceUncertain: true,
      );
      return false;
    }
  }

  /// Synchronously revokes callbacks. True is only local accepted/idempotent
  /// revocation. Await [AsrSession.cleanup] for the exact worker release
  /// receipt; neither value proves native destruction or termination.
  bool endSession(AsrSession session) {
    final state = _ownedState(session);
    if (state == null) return false;
    if (state.disposition == _AsrSessionDisposition.uncertain) return false;
    if (state.disposition == _AsrSessionDisposition.ending ||
        state.disposition == _AsrSessionDisposition.released) {
      return true;
    }
    if (!identical(_active, state)) return false;
    return _beginEnd(state, readyCode: 'asr_session_ended');
  }

  /// App-lifetime hard close. The exact future is memoized before any work,
  /// including reentrant transport callbacks. Failure is bounded and retained;
  /// close is never retried and is not a native-cleanup receipt.
  Future<bool> close() {
    final existing = _closeFuture;
    if (existing != null) return existing;
    final completer = Completer<bool>();
    final result = completer.future;
    _closeFuture = result;
    _closing = true;
    _closingEventGeneration = _launchGeneration;
    _startupFence?._revoke();
    _launchGeneration++;
    final active = _active;
    if (active != null) {
      _beginEnd(active, readyCode: 'asr_service_closed');
    }
    final cleanup = active?.cleanup.future ?? Future<bool>.value(true);
    final work = Future<bool>.sync(() => _closeImpl(cleanup));
    AsrTimerHandle? deadline;
    void freeze(bool value) {
      if (completer.isCompleted) return;
      if (!value) _retainedCloseFailureCode = 'asr_close_failed';
      completer.complete(value);
      final timer = deadline;
      if (timer != null) _cancelTimer(timer);
    }

    try {
      deadline = _timerFactory(_closeTimeout, () => freeze(false));
      if (completer.isCompleted) _cancelTimer(deadline);
    } catch (_) {
      freeze(false);
    }
    work.then(
      (value) {
        _closed = true;
        _closingEventGeneration = null;
        freeze(value);
      },
      onError: (Object error, StackTrace stackTrace) {
        _closed = true;
        _closingEventGeneration = null;
        freeze(false);
      },
    );
    return result;
  }

  _AsrSessionState? _ownedState(AsrSession session) {
    if (!identical(session._owner, _owner)) return null;
    return session._state;
  }

  Future<void> _admit(_AsrSessionState state) async {
    final predecessorClean = await state.admissionBarrier;
    if (!_hasAuthority(state)) return;
    if (!predecessorClean || _transportUncertain) {
      _transportUncertain = true;
      _failAdmission(state, 'asr_predecessor_cleanup_failed');
      return;
    }

    late final AsrWorkerStartup startup;
    try {
      startup = await _ensureStartup();
    } catch (error) {
      if (error is AsrWorkerStartupUncertain) {
        _transportUncertain = true;
        if (_hasAuthority(state)) {
          _beginEnd(
            state,
            readyCode: 'asr_worker_start_uncertain',
            forceUncertain: true,
          );
        }
        return;
      }
      if (_hasAuthority(state)) {
        _failAdmission(state, 'asr_worker_start_failed');
      }
      return;
    }
    try {
      await startup.ready;
    } catch (_) {
      _transportUncertain = true;
      if (_hasAuthority(state)) {
        _failAdmission(state, 'asr_worker_ready_failed');
      }
      return;
    }
    if (!_hasAuthority(state)) return;
    state.resetSent = true;
    try {
      startup.transport.send(AsrWorkerReset(state.ordinal));
    } catch (_) {
      _transportUncertain = true;
      _beginEnd(
        state,
        readyCode: 'asr_reset_send_failed',
        forceUncertain: true,
      );
    }
  }

  Future<AsrWorkerStartup> _ensureStartup() {
    final existing = _startupFuture;
    if (existing != null) return existing;
    final generation = ++_launchGeneration;
    final launchFence = AsrWorkerLaunchFence();
    _startupFence = launchFence;
    final completer = Completer<AsrWorkerStartup>();
    final future = completer.future;
    // Publish before invoking caller-controlled factory code: the factory may
    // synchronously reenter close(), which must retain and await this launch.
    _startupFuture = future;
    future.then(
      (startup) {
        _startup = startup;
        if (_closing || _closed || generation != _launchGeneration) {
          unawaited(
            startup.close().then((value) {
              if (!value) {
                _retainedCloseFailureCode ??= 'asr_late_close_failed';
              }
            }),
          );
        }
      },
      onError: (Object error, StackTrace stackTrace) {
        if (error is AsrWorkerStartupUncertain) {
          _transportUncertain = true;
          return;
        }
        if (!_closing &&
            !_closed &&
            generation == _launchGeneration &&
            identical(_startupFuture, future)) {
          // A factory failure publishes no transport and is cleanly retryable.
          _startupFuture = null;
          if (identical(_startupFence, launchFence)) _startupFence = null;
        }
      },
    );
    unawaited(
      Future<AsrWorkerStartup>.sync(
        () => _startupFactory(
          (event) => _onWorkerEvent(generation, event),
          launchFence,
        ),
      ).then<void>(
        (startup) => completer.complete(startup),
        onError: (Object error, StackTrace stackTrace) {
          completer.completeError(error, stackTrace);
        },
      ),
    );
    return future;
  }

  void _onWorkerEvent(int generation, AsrWorkerEvent event) {
    if (_closed) return;
    final currentGeneration = generation == _launchGeneration;
    final closingGeneration = _closing &&
        generation == _closingEventGeneration &&
        (event is AsrWorkerEndAck ||
            event is AsrWorkerSessionFailure ||
            event is AsrWorkerLifecycleFailure);
    if (!currentGeneration && !closingGeneration) return;
    if (event is AsrWorkerLifecycleFailure) {
      _transportUncertain = true;
      _failAllSessionsUncertain('asr_worker_failed');
      return;
    }
    if (event is AsrWorkerEndAck) {
      final ending = _releasing[event.ordinal];
      if (!event.released) {
        _transportUncertain = true;
        if (ending != null) _finishCleanup(ending, false);
        _failActiveForPoison('asr_worker_release_failed');
      } else if (ending != null) {
        _finishCleanup(ending, true);
      }
      return;
    }
    if (event is AsrWorkerSessionFailure) {
      final state = _stateForOrdinal(event.ordinal);
      if (state != null) {
        _finishWorkerFailure(state, event);
      }
      if (!event.workerHealthy) {
        _transportUncertain = true;
        _failActiveForPoison('asr_worker_poisoned', except: state);
      }
      return;
    }

    final active = _active;
    if (active == null || active.disposition != _AsrSessionDisposition.active) {
      return;
    }
    if (event is AsrWorkerResetAck) {
      if (event.ordinal != active.ordinal || !active.resetSent) return;
      active.resetAcknowledged = true;
      final timer = active.timer;
      active.timer = null;
      if (timer != null) _cancelTimer(timer);
      if (!active.ready.isCompleted) active.ready.complete();
      return;
    }
    if (event is AsrWorkerAudioAck) {
      if (event.ordinal == active.ordinal && event.sequence > 0) {
        active.outstandingAudio.remove(event.sequence);
      }
      return;
    }
    if (!active.resetAcknowledged) return;
    if (event is AsrWorkerPartial && event.ordinal == active.ordinal) {
      _deliver(active, event.text, partial: true);
    } else if (event is AsrWorkerEndpoint && event.ordinal == active.ordinal) {
      _deliver(active, event.text, partial: false);
    }
  }

  void _deliver(_AsrSessionState state, String text, {required bool partial}) {
    if (!_hasAuthority(state) || !_boundedResult(text)) {
      if (_hasAuthority(state)) {
        _beginEnd(state, readyCode: 'asr_result_invalid');
      }
      return;
    }
    final callback = partial ? state.onPartial : state.onEndpoint;
    if (callback == null) return;
    try {
      callback(text);
    } catch (_) {
      if (_hasAuthority(state)) {
        _beginEnd(state, readyCode: 'asr_callback_failed');
      }
    }
  }

  bool _hasAuthority(_AsrSessionState state) =>
      !_closing &&
      !_closed &&
      identical(_active, state) &&
      state.disposition == _AsrSessionDisposition.active;

  void _onAdmissionTimeout(_AsrSessionState state) {
    if (!_hasAuthority(state) || state.resetAcknowledged) return;
    _beginEnd(
      state,
      readyCode: 'asr_admission_timeout',
      readyTimeout: true,
    );
  }

  void _failAdmission(_AsrSessionState state, String code) {
    if (!_hasAuthority(state)) return;
    _beginEnd(state, readyCode: code);
  }

  bool _beginEnd(
    _AsrSessionState state, {
    required String readyCode,
    bool forceUncertain = false,
    bool readyTimeout = false,
  }) {
    if (state.disposition != _AsrSessionDisposition.active) {
      return state.disposition == _AsrSessionDisposition.ending ||
          state.disposition == _AsrSessionDisposition.released;
    }
    if (identical(_active, state)) _active = null;
    state.onPartial = null;
    state.onEndpoint = null;
    state.outstandingAudio.clear();
    final timer = state.timer;
    state.timer = null;
    if (timer != null) _cancelTimer(timer);
    if (!state.ready.isCompleted) {
      if (readyTimeout) {
        state.ready.completeError(
          TimeoutException(readyCode, _admissionTimeout),
        );
      } else {
        state.ready.completeError(StateError(readyCode));
      }
    }
    if (forceUncertain) {
      state.disposition = _AsrSessionDisposition.uncertain;
      _finishCleanup(state, false);
      return false;
    }
    if (!state.resetSent || _startup == null) {
      state.disposition = _AsrSessionDisposition.released;
      _finishCleanup(state, true);
      return true;
    }

    state.disposition = _AsrSessionDisposition.ending;
    _releasing[state.ordinal] = state;
    try {
      final timer = _timerFactory(
        _cleanupTimeout,
        () => _onCleanupTimeout(state),
      );
      if (state.disposition == _AsrSessionDisposition.ending) {
        state.cleanupTimer = timer;
      } else {
        _cancelTimer(timer);
      }
    } catch (_) {
      _transportUncertain = true;
      _finishCleanup(state, false);
    }
    state.endSent = true;
    try {
      _startup!.transport.send(AsrWorkerEnd(state.ordinal));
    } catch (_) {
      _transportUncertain = true;
      _finishCleanup(state, false);
    }
    return state.disposition != _AsrSessionDisposition.uncertain;
  }

  void _onCleanupTimeout(_AsrSessionState state) {
    if (state.disposition != _AsrSessionDisposition.ending) return;
    _transportUncertain = true;
    _finishCleanup(state, false);
    _failActiveForPoison('asr_cleanup_timeout');
  }

  void _finishCleanup(_AsrSessionState state, bool released) {
    if (state.cleanup.isCompleted) return;
    _releasing.remove(state.ordinal);
    final timer = state.cleanupTimer;
    state.cleanupTimer = null;
    if (timer != null) _cancelTimer(timer);
    state.disposition = released
        ? _AsrSessionDisposition.released
        : _AsrSessionDisposition.uncertain;
    state.cleanup.complete(released);
  }

  _AsrSessionState? _stateForOrdinal(int ordinal) {
    final active = _active;
    if (active != null && active.ordinal == ordinal) return active;
    return _releasing[ordinal];
  }

  void _finishWorkerFailure(
    _AsrSessionState state,
    AsrWorkerSessionFailure event,
  ) {
    if (identical(_active, state)) _active = null;
    state.onPartial = null;
    state.onEndpoint = null;
    state.outstandingAudio.clear();
    final timer = state.timer;
    state.timer = null;
    if (timer != null) _cancelTimer(timer);
    if (!state.ready.isCompleted) {
      state.ready.completeError(StateError('asr_worker_session_failed'));
    }
    _finishCleanup(state, event.sessionReleased);
  }

  void _failActiveForPoison(
    String code, {
    _AsrSessionState? except,
  }) {
    final active = _active;
    if (active != null && !identical(active, except)) {
      _beginEnd(active, readyCode: code, forceUncertain: true);
    }
  }

  void _failAllSessionsUncertain(String code) {
    final active = _active;
    if (active != null) {
      _beginEnd(active, readyCode: code, forceUncertain: true);
    }
    for (final state in _releasing.values.toList(growable: false)) {
      _finishCleanup(state, false);
    }
  }

  static void _cancelTimer(AsrTimerHandle timer) {
    try {
      timer.cancel();
    } catch (_) {
      // Authority checks make a hostile timer callback inert after release.
    }
  }

  Future<bool> _closeImpl(Future<bool> sessionCleanup) async {
    final startup = _startup;
    if (startup != null) {
      final workerClose = startup.close();
      final sessionReleased = await sessionCleanup;
      final workerClosed = await workerClose;
      return sessionReleased && workerClosed;
    }
    final pending = _startupFuture;
    if (pending == null) return sessionCleanup;
    late final AsrWorkerStartup lateStartup;
    try {
      lateStartup = await pending;
    } catch (error) {
      if (error is AsrWorkerStartupUncertain) return false;
      return sessionCleanup;
    }
    _startup = lateStartup;
    final workerClose = lateStartup.close();
    final sessionReleased = await sessionCleanup;
    final workerClosed = await workerClose;
    return sessionReleased && workerClosed;
  }

  static Future<bool> _chainReceipts(
    Future<bool> predecessor,
    Future<bool> current,
  ) async {
    if (!await predecessor) return false;
    return current;
  }

  static bool _boundedResult(String text) {
    if (text.length > maxResultUtf8Bytes) return false;
    return utf8.encode(text).length <= maxResultUtf8Bytes;
  }
}

/// Pure worker-side ordinal/sequence gate. Reset only moves forward; ending a
/// session never permits an older reset or audio message to become current.
final class AsrWorkerSessionGate {
  int _highestOrdinal = 0;
  int? _currentOrdinal;
  int _lastAudioSequence = 0;
  bool _poisoned = false;

  int get highestOrdinal => _highestOrdinal;
  int? get currentOrdinal => _currentOrdinal;
  bool get isPoisoned => _poisoned;

  bool canReset(int ordinal) =>
      !_poisoned &&
      ordinal > _highestOrdinal &&
      ordinal <= AsrSession.maxSafeOrdinal;

  void commitReset(int ordinal) {
    if (!canReset(ordinal)) throw StateError('asr_worker_reset_not_admitted');
    _highestOrdinal = ordinal;
    _currentOrdinal = ordinal;
    _lastAudioSequence = 0;
  }

  void commitReleasedReset(int ordinal) {
    if (!canReset(ordinal)) throw StateError('asr_worker_reset_not_admitted');
    _highestOrdinal = ordinal;
    _currentOrdinal = null;
    _lastAudioSequence = 0;
  }

  bool acceptAudio(int ordinal, int sequence) {
    if (_poisoned ||
        ordinal != _currentOrdinal ||
        sequence != _lastAudioSequence + 1 ||
        sequence <= 0 ||
        sequence > AsrSession.maxSafeOrdinal) {
      return false;
    }
    _lastAudioSequence = sequence;
    return true;
  }

  bool end(int ordinal) {
    if (_poisoned || ordinal != _currentOrdinal) return false;
    _currentOrdinal = null;
    _lastAudioSequence = 0;
    return true;
  }

  void poison() {
    _poisoned = true;
    _currentOrdinal = null;
    _lastAudioSequence = 0;
  }
}

abstract interface class AsrWorkerStreamAdapter {
  void acceptPcm16(Float32List samples);
  void free();
}

abstract interface class AsrWorkerRecognizerAdapter {
  AsrWorkerStreamAdapter createStream();
  bool isReady(AsrWorkerStreamAdapter stream);
  void decode(AsrWorkerStreamAdapter stream);
  String resultText(AsrWorkerStreamAdapter stream);
  bool isEndpoint(AsrWorkerStreamAdapter stream);
  void reset(AsrWorkerStreamAdapter stream);
  void free();
}

/// Purely injectable worker command executor. It makes native-resource order
/// testable without loading a model or plugin.
final class AsrWorkerCore {
  AsrWorkerCore({
    required AsrWorkerRecognizerAdapter recognizer,
    required AsrWorkerEventSink emit,
    required void Function() terminate,
  })  : _recognizer = recognizer,
        _emit = emit,
        _terminate = terminate;

  final AsrWorkerRecognizerAdapter _recognizer;
  final AsrWorkerEventSink _emit;
  final void Function() _terminate;
  final AsrWorkerSessionGate gate = AsrWorkerSessionGate();
  final List<AsrWorkerStreamAdapter> _uncertainStreams =
      <AsrWorkerStreamAdapter>[];
  AsrWorkerStreamAdapter? _stream;
  String _lastPartial = '';
  bool _recognizerUncertain = false;
  bool _resourceUncertain = false;
  bool _terminated = false;

  int get retainedUncertainStreamCount => _uncertainStreams.length;
  bool get recognizerUncertain => _recognizerUncertain;

  void handle(AsrWorkerCommand command) {
    if (_terminated) return;
    if (command is AsrWorkerReset) {
      _reset(command);
    } else if (command is AsrWorkerAudio) {
      _audio(command);
    } else if (command is AsrWorkerEnd) {
      _end(command);
    } else if (command is AsrWorkerShutdown) {
      _shutdown(command);
    }
  }

  void _reset(AsrWorkerReset command) {
    if (!gate.canReset(command.ordinal)) return;
    final predecessor = _stream;
    if (predecessor != null) {
      final predecessorOrdinal = gate.currentOrdinal!;
      _stream = null;
      _lastPartial = '';
      gate.end(predecessorOrdinal);
      try {
        predecessor.free();
      } catch (_) {
        _uncertainStreams.add(predecessor);
        _resourceUncertain = true;
        gate.poison();
        _safeEmit(
          AsrWorkerEndAck(predecessorOrdinal, released: false),
        );
        _safeEmit(
          AsrWorkerSessionFailure(
            command.ordinal,
            'asr_predecessor_release_failed',
            sessionReleased: true,
            workerHealthy: false,
          ),
        );
        return;
      }
      if (!_safeEmit(
        AsrWorkerEndAck(predecessorOrdinal, released: true),
      )) {
        gate.poison();
        return;
      }
    }

    late final AsrWorkerStreamAdapter replacement;
    try {
      replacement = _recognizer.createStream();
    } catch (_) {
      gate.commitReleasedReset(command.ordinal);
      _safeEmit(
        AsrWorkerSessionFailure(
          command.ordinal,
          'asr_stream_create_failed',
          sessionReleased: true,
          workerHealthy: true,
        ),
      );
      return;
    }

    _stream = replacement;
    _lastPartial = '';
    gate.commitReset(command.ordinal);
    try {
      _emit(AsrWorkerResetAck(command.ordinal));
    } catch (_) {
      _poisonCurrentAfterEmitFailure();
    }
  }

  void _audio(AsrWorkerAudio command) {
    if (command.ordinal != gate.currentOrdinal) return;
    if (command.bytes.isEmpty ||
        command.bytes.length.isOdd ||
        command.bytes.length > AsrService.maxPcmBytesPerChunk ||
        !gate.acceptAudio(command.ordinal, command.sequence)) {
      _failCurrent('asr_worker_audio_invalid');
      return;
    }
    final stream = _stream;
    if (stream == null) {
      _failCurrent('asr_worker_stream_missing');
      return;
    }

    try {
      stream.acceptPcm16(_toFloat32(command.bytes));
      var steps = 0;
      while (_recognizer.isReady(stream)) {
        if (steps++ >= AsrService.maxDecodeStepsPerChunk) {
          throw StateError('asr_decode_step_bound');
        }
        _recognizer.decode(stream);
      }
      final rawText = _recognizer.resultText(stream);
      if (rawText.length > AsrService.maxResultUtf8Bytes) {
        throw StateError('asr_result_bound');
      }
      final text = rawText.trim();
      if (!AsrService._boundedResult(text)) {
        throw StateError('asr_result_bound');
      }
      if (_recognizer.isEndpoint(stream)) {
        _recognizer.reset(stream);
        _lastPartial = '';
        _emit(AsrWorkerEndpoint(command.ordinal, text));
      } else if (text != _lastPartial) {
        _lastPartial = text;
        _emit(AsrWorkerPartial(command.ordinal, text));
      }
      // Output is deliberately emitted before credit is returned.
      _emit(AsrWorkerAudioAck(command.ordinal, command.sequence));
    } catch (_) {
      _failCurrent('asr_worker_decode_failed');
    }
  }

  void _end(AsrWorkerEnd command) {
    if (command.ordinal != gate.currentOrdinal) return;
    final stream = _stream;
    gate.end(command.ordinal);
    _stream = null;
    _lastPartial = '';
    if (stream == null) {
      _safeEmit(AsrWorkerEndAck(command.ordinal, released: true));
      return;
    }
    try {
      stream.free();
      if (!_safeEmit(AsrWorkerEndAck(command.ordinal, released: true))) {
        gate.poison();
      }
    } catch (_) {
      _uncertainStreams.add(stream);
      _resourceUncertain = true;
      gate.poison();
      _safeEmit(
        AsrWorkerEndAck(command.ordinal, released: false),
      );
    }
  }

  void _shutdown(AsrWorkerShutdown command) {
    _terminated = true;
    final stream = _stream;
    _stream = null;
    _lastPartial = '';
    if (_resourceUncertain || _uncertainStreams.isNotEmpty) {
      gate.poison();
      _terminate();
      return;
    }
    if (stream != null) {
      try {
        stream.free();
      } catch (_) {
        _uncertainStreams.add(stream);
        _resourceUncertain = true;
        gate.poison();
        _terminate();
        return;
      }
    }
    try {
      _recognizer.free();
    } catch (_) {
      _recognizerUncertain = true;
      gate.poison();
      _terminate();
      return;
    }
    try {
      _emit(AsrWorkerShutdownAck(command.shutdownOrdinal));
    } finally {
      _terminate();
    }
  }

  void _failCurrent(String code) {
    final ordinal = gate.currentOrdinal;
    if (ordinal == null) return;
    final stream = _stream;
    _stream = null;
    _lastPartial = '';
    gate.end(ordinal);
    var released = true;
    if (stream != null) {
      try {
        stream.free();
      } catch (_) {
        _uncertainStreams.add(stream);
        _resourceUncertain = true;
        released = false;
      }
    }
    gate.poison();
    _safeEmit(
      AsrWorkerSessionFailure(
        ordinal,
        code,
        sessionReleased: released,
        workerHealthy: false,
      ),
    );
  }

  void _poisonCurrentAfterEmitFailure() {
    final stream = _stream;
    _stream = null;
    _lastPartial = '';
    gate.poison();
    if (stream != null) {
      try {
        stream.free();
      } catch (_) {
        _uncertainStreams.add(stream);
        _resourceUncertain = true;
      }
    }
  }

  bool _safeEmit(AsrWorkerEvent event) {
    try {
      _emit(event);
      return true;
    } catch (_) {
      // There is no second trustworthy channel after event delivery fails.
      return false;
    }
  }
}

final class _AsrInit {
  const _AsrInit({
    required this.encoder,
    required this.decoder,
    required this.joiner,
    required this.tokens,
    required this.modelType,
    required this.silence,
  });

  final String encoder;
  final String decoder;
  final String joiner;
  final String tokens;
  final String modelType;
  final double silence;
}

final class _AsrWorkerHello {
  const _AsrWorkerHello(this.port);
  final SendPort port;
}

final class _AsrWorkerInit {
  const _AsrWorkerInit(this.config);
  final _AsrInit config;
}

final class _AsrWorkerReady {
  const _AsrWorkerReady();
}

/// Content-free factory failure used only when startup may retain Dart/worker
/// resources. The service freezes false and never retries this launch.
final class AsrWorkerStartupUncertain implements Exception {
  const AsrWorkerStartupUncertain();
}

final class _SherpaStreamAdapter implements AsrWorkerStreamAdapter {
  _SherpaStreamAdapter(this.stream);
  final sherpa_onnx.OnlineStream stream;

  @override
  void acceptPcm16(Float32List samples) {
    stream.acceptWaveform(samples: samples, sampleRate: 16000);
  }

  @override
  void free() => stream.free();
}

final class _SherpaRecognizerAdapter implements AsrWorkerRecognizerAdapter {
  _SherpaRecognizerAdapter(this.recognizer);
  final sherpa_onnx.OnlineRecognizer recognizer;

  @override
  AsrWorkerStreamAdapter createStream() =>
      _SherpaStreamAdapter(recognizer.createStream());

  _SherpaStreamAdapter _cast(AsrWorkerStreamAdapter stream) =>
      stream as _SherpaStreamAdapter;

  @override
  void decode(AsrWorkerStreamAdapter stream) =>
      recognizer.decode(_cast(stream).stream);

  @override
  void free() => recognizer.free();

  @override
  bool isEndpoint(AsrWorkerStreamAdapter stream) =>
      recognizer.isEndpoint(_cast(stream).stream);

  @override
  bool isReady(AsrWorkerStreamAdapter stream) =>
      recognizer.isReady(_cast(stream).stream);

  @override
  String resultText(AsrWorkerStreamAdapter stream) =>
      recognizer.getResult(_cast(stream).stream).text;

  @override
  void reset(AsrWorkerStreamAdapter stream) =>
      recognizer.reset(_cast(stream).stream);
}

Future<AsrWorkerStartup> _startProductionWorker(
  AsrWorkerEventSink onEvent,
  AsrWorkerLaunchFence launchFence,
) async {
  final model = await getOnlineModelConfig();
  if (!launchFence.isCurrent) {
    throw StateError('asr_worker_launch_revoked_before_spawn');
  }
  final init = _AsrInit(
    encoder: model.transducer.encoder,
    decoder: model.transducer.decoder,
    joiner: model.transducer.joiner,
    tokens: model.tokens,
    modelType: model.modelType,
    silence: AsrService.endpointSilenceSec,
  );

  final messages = ReceivePort();
  final errors = ReceivePort();
  final exits = ReceivePort();
  final ready = Completer<void>();
  final portReady = Completer<SendPort>();
  final exited = Completer<void>();
  final shutdownAcks = <int, Completer<void>>{};
  Isolate? isolate;
  SendPort? workerPort;
  late final StreamSubscription<dynamic> messageSub;
  late final StreamSubscription<dynamic> errorSub;
  late final StreamSubscription<dynamic> exitSub;
  var intentionallyClosing = false;
  var transportClosing = false;
  Future<bool>? portsCloseFuture;

  void failReady(String code) {
    if (!ready.isCompleted) ready.completeError(StateError(code));
  }

  messageSub = messages.listen((dynamic message) {
    if (message is _AsrWorkerHello) {
      workerPort = message.port;
      if (!portReady.isCompleted) portReady.complete(message.port);
      if (!launchFence.isCurrent) {
        intentionallyClosing = true;
        transportClosing = true;
      }
      if (!transportClosing && launchFence.isCurrent) {
        try {
          message.port.send(_AsrWorkerInit(init));
        } catch (_) {
          failReady('asr_worker_init_send_failed');
          onEvent(
            const AsrWorkerLifecycleFailure('asr_worker_init_send_failed'),
          );
        }
      }
    } else if (message is _AsrWorkerReady) {
      if (!ready.isCompleted) ready.complete();
    } else if (message is AsrWorkerShutdownAck) {
      final ack = shutdownAcks.remove(message.shutdownOrdinal);
      if (ack != null && !ack.isCompleted) ack.complete();
    } else if (message is AsrWorkerEvent) {
      onEvent(message);
    }
  });
  errorSub = errors.listen((dynamic message) {
    failReady('asr_worker_isolate_error');
    if (!intentionallyClosing) {
      onEvent(const AsrWorkerLifecycleFailure('asr_worker_isolate_error'));
    }
  });
  exitSub = exits.listen((dynamic message) {
    if (!exited.isCompleted) exited.complete();
    failReady('asr_worker_exited_before_ready');
    if (!intentionallyClosing) {
      onEvent(const AsrWorkerLifecycleFailure('asr_worker_exited'));
    }
  });

  try {
    isolate = await Isolate.spawn<SendPort>(
      _asrWorkerMain,
      messages.sendPort,
      onError: errors.sendPort,
      onExit: exits.sendPort,
      errorsAreFatal: true,
    );
  } catch (_) {
    messages.close();
    errors.close();
    exits.close();
    var portsClosed = true;
    try {
      await Future.wait<void>(<Future<void>>[
        messageSub.cancel(),
        errorSub.cancel(),
        exitSub.cancel(),
      ]);
    } catch (_) {
      portsClosed = false;
    }
    if (!portsClosed) throw const AsrWorkerStartupUncertain();
    rethrow;
  }

  Future<bool> closePorts() {
    final existing = portsCloseFuture;
    if (existing != null) return existing;
    final completer = Completer<bool>();
    final result = completer.future;
    portsCloseFuture = result;
    messages.close();
    errors.close();
    exits.close();
    Future.wait<void>(<Future<void>>[
      messageSub.cancel(),
      errorSub.cancel(),
      exitSub.cancel(),
    ]).then(
      (_) => completer.complete(true),
      onError: (Object error, StackTrace stackTrace) {
        completer.complete(false);
      },
    );
    return result;
  }

  var shutdownOrdinal = 0;
  Future<bool> closeWorkerOperation() async {
    intentionallyClosing = true;
    transportClosing = true;
    var cooperative = false;
    try {
      final port = workerPort ?? await portReady.future;
      if (shutdownOrdinal >= AsrSession.maxSafeOrdinal) {
        throw StateError('asr_shutdown_ordinal_exhausted');
      }
      final ordinal = ++shutdownOrdinal;
      final ack = Completer<void>();
      shutdownAcks[ordinal] = ack;
      port.send(AsrWorkerShutdown(ordinal));
      await ack.future;
      await exited.future;
      cooperative = true;
    } catch (_) {
      isolate?.kill(priority: Isolate.immediate);
    }
    final portsClosed = await closePorts();
    shutdownAcks.clear();
    return cooperative && portsClosed;
  }

  Future<bool> closeWorker() {
    transportClosing = true;
    intentionallyClosing = true;
    final completer = Completer<bool>();
    final operation = closeWorkerOperation();
    final timer = Timer(AsrService.workerCloseTimeout, () {
      if (completer.isCompleted) return;
      isolate?.kill(priority: Isolate.immediate);
      unawaited(closePorts());
      completer.complete(false);
    });
    operation.then(
      (value) {
        if (completer.isCompleted) return;
        timer.cancel();
        completer.complete(value);
      },
      onError: (Object error, StackTrace stackTrace) {
        if (completer.isCompleted) return;
        timer.cancel();
        isolate?.kill(priority: Isolate.immediate);
        unawaited(closePorts());
        completer.complete(false);
      },
    );
    return completer.future;
  }

  final startup = AsrWorkerStartup(
    ready: ready.future,
    transport: AsrWorkerTransport(
      send: (command) {
        if (transportClosing) throw StateError('asr_worker_transport_closing');
        final port = workerPort;
        if (port == null) throw StateError('asr_worker_port_not_ready');
        port.send(command);
      },
      close: closeWorker,
    ),
  );
  if (!launchFence.isCurrent) {
    // Spawn is not cancellable. Retire this exact isolate and every port
    // before allowing the revoked factory result to escape. A false receipt
    // remains false; kill fallback is not native-cleanup evidence.
    await startup.close();
  }
  return startup;
}

Float32List _toFloat32(Uint8List bytes) {
  final values = Float32List(bytes.length ~/ 2);
  final data = ByteData.sublistView(bytes);
  for (var i = 0; i < bytes.length; i += 2) {
    values[i ~/ 2] = data.getInt16(i, Endian.little) / 32768.0;
  }
  return values;
}

void _asrWorkerMain(SendPort toMain) {
  final fromMain = ReceivePort();
  toMain.send(_AsrWorkerHello(fromMain.sendPort));
  AsrWorkerCore? core;

  fromMain.listen((dynamic message) {
    if (message is _AsrWorkerInit) {
      if (core != null) return;
      final config = message.config;
      sherpa_onnx.OnlineRecognizer? recognizer;
      try {
        sherpa_onnx.initBindings();
        final model = sherpa_onnx.OnlineModelConfig(
          transducer: sherpa_onnx.OnlineTransducerModelConfig(
            encoder: config.encoder,
            decoder: config.decoder,
            joiner: config.joiner,
          ),
          tokens: config.tokens,
          modelType: config.modelType,
        );
        recognizer = sherpa_onnx.OnlineRecognizer(
          sherpa_onnx.OnlineRecognizerConfig(
            model: model,
            ruleFsts: '',
            enableEndpoint: true,
            rule2MinTrailingSilence: config.silence,
          ),
        );
        core = AsrWorkerCore(
          recognizer: _SherpaRecognizerAdapter(recognizer),
          emit: toMain.send,
          terminate: fromMain.close,
        );
        toMain.send(const _AsrWorkerReady());
      } catch (_) {
        try {
          recognizer?.free();
        } catch (_) {
          // The isolate exits; no cleanup receipt is published.
        }
        fromMain.close();
        rethrow;
      }
    } else if (message is AsrWorkerCommand) {
      final worker = core;
      if (worker != null) {
        worker.handle(message);
      } else if (message is AsrWorkerShutdown) {
        try {
          toMain.send(AsrWorkerShutdownAck(message.shutdownOrdinal));
        } finally {
          fromMain.close();
        }
      }
    }
  });
}
