// Deterministic hostile-fake tests for AssistantListeningOwner.
//
// No Flutter widget, plugin, model, network, microphone, or audio device is
// loaded. The fakes expose every stage Future and every exact cleanup proof.
import 'dart:async';
import 'dart:collection';

import 'package:flutter_test/flutter_test.dart';

import '../lib/assistant_listening_owner.dart';

const _admissionLifetime = Duration(seconds: 101);
const _cleanupLifetime = Duration(seconds: 7);

Future<void> _flush([int turns = 24]) async {
  for (var index = 0; index < turns; index += 1) {
    await Future<void>.delayed(Duration.zero);
  }
}

final class _ManualTimer implements Timer {
  _ManualTimer(this.duration, this._callback, {required this.throwOnCancel});

  final Duration duration;
  final void Function() _callback;
  final bool throwOnCancel;
  bool _active = true;
  int _tick = 0;

  @override
  bool get isActive => _active;

  @override
  int get tick => _tick;

  @override
  void cancel() {
    _active = false;
    if (throwOnCancel) throw StateError('private timer cancel failure');
  }

  void fire() {
    if (!_active) return;
    _active = false;
    _tick += 1;
    _callback();
  }
}

final class _TimerHarness {
  final List<_ManualTimer> timers = <_ManualTimer>[];
  final Set<int> throwCalls = <int>{};
  final Set<int> synchronousFireCalls = <int>{};
  final Set<int> throwCancelCalls = <int>{};
  int calls = 0;

  Timer call(Duration duration, void Function() callback) {
    calls += 1;
    if (throwCalls.contains(calls)) {
      throw StateError('private timer failure');
    }
    final timer = _ManualTimer(
      duration,
      callback,
      throwOnCancel: throwCancelCalls.contains(calls),
    );
    timers.add(timer);
    if (synchronousFireCalls.contains(calls)) timer.fire();
    return timer;
  }

  _ManualTimer active(Duration duration) => timers.singleWhere(
        (timer) => timer.duration == duration && timer.isActive,
      );

  List<_ManualTimer> activeAll(Duration duration) => timers
      .where((timer) => timer.duration == duration && timer.isActive)
      .toList(growable: false);
}

final class _Session {
  _Session(this.id, {required bool readyHeld, required bool terminalHeld}) {
    if (!readyHeld) ready.complete();
    autoCompleteTerminal = !terminalHeld;
  }

  final int id;
  final Completer<void> ready = Completer<void>();
  final Completer<bool> terminal = Completer<bool>();
  late final bool autoCompleteTerminal;
  bool terminalCounted = false;
}

final class _Capture {
  _Capture(this.id);

  final int id;
  final Completer<AssistantListeningCaptureTerminal> terminal =
      Completer<AssistantListeningCaptureTerminal>();
  Completer<bool>? cancelGate;
  Completer<bool>? stopGate;
  bool cancelResult = true;
  bool stopResult = true;
  bool throwCancel = false;
  bool throwStop = false;
  bool counted = false;
  bool stopCounted = false;
}

final class _StartPlan {
  _StartPlan.immediate(this.result) : gate = null;

  _StartPlan.held()
      : gate = Completer<AssistantListeningCaptureStartResult<_Capture>>(),
        result = null;

  final Completer<AssistantListeningCaptureStartResult<_Capture>>? gate;
  final AssistantListeningCaptureStartResult<_Capture>? result;
  bool throwSynchronously = false;
  bool throwAsynchronously = false;
  AssistantListeningCaptureTerminal? terminalBeforeReturn;

  Future<AssistantListeningCaptureStartResult<_Capture>> future() {
    if (throwSynchronously)
      throw StateError('private synchronous start failure');
    if (throwAsynchronously) {
      return Future<AssistantListeningCaptureStartResult<_Capture>>.error(
        StateError('private asynchronous start failure'),
        StackTrace.current,
      );
    }
    final held = gate;
    if (held != null) return held.future;
    return Future<AssistantListeningCaptureStartResult<_Capture>>.value(result);
  }
}

final class _Backend
    implements AssistantListeningLifecycle<_Session, _Capture> {
  _Backend(this.log);

  final List<String> log;
  final Queue<_StartPlan> startPlans = Queue<_StartPlan>();
  final List<_Session> sessions = <_Session>[];
  final List<_Capture> captures = <_Capture>[];

  Completer<bool>? permissionGate;
  Completer<bool>? routeGate;
  Completer<bool>? recoveryGate;
  bool permissionResult = true;
  bool routeResult = true;
  bool recoveryResult = false;
  bool holdReady = false;
  bool holdSessionTerminal = false;
  bool endAccepted = true;
  bool throwPermission = false;
  bool throwRoute = false;
  bool throwBegin = false;
  bool throwSessionWatch = false;
  bool throwReady = false;
  bool throwRecovery = false;
  bool throwEnd = false;
  int nextSessionId = 0;
  int nextCaptureId = 0;
  int activeSessions = 0;
  int activeCaptures = 0;
  int maxActiveSessions = 0;
  int maxActiveCaptures = 0;
  int permissionCalls = 0;
  int routeCalls = 0;
  int beginCalls = 0;
  int startCalls = 0;
  int recoveryCalls = 0;
  int cancelCalls = 0;
  int stopCalls = 0;
  int endCalls = 0;

  _Capture capture() {
    final value = _Capture(++nextCaptureId);
    captures.add(value);
    return value;
  }

  Future<bool> requestPermission(
    AssistantListeningGeneration<_Session, _Capture> generation,
  ) {
    permissionCalls += 1;
    log.add('permission:${generation.ordinal}');
    if (throwPermission) throw StateError('private permission failure');
    return permissionGate?.future ?? Future<bool>.value(permissionResult);
  }

  Future<bool> configureRoute(
    AssistantListeningGeneration<_Session, _Capture> generation,
  ) {
    routeCalls += 1;
    log.add('route:${generation.ordinal}');
    if (throwRoute) throw StateError('private route failure');
    return routeGate?.future ?? Future<bool>.value(routeResult);
  }

  _Session beginSession(
    AssistantListeningGeneration<_Session, _Capture> generation,
  ) {
    beginCalls += 1;
    log.add('begin:${generation.ordinal}');
    if (throwBegin) throw StateError('private begin failure');
    final session = _Session(
      ++nextSessionId,
      readyHeld: holdReady,
      terminalHeld: holdSessionTerminal,
    );
    sessions.add(session);
    activeSessions += 1;
    if (activeSessions > maxActiveSessions) maxActiveSessions = activeSessions;
    return session;
  }

  Future<bool> waitSessionEnded(_Session session) {
    log.add('watch-session:${session.id}');
    if (throwSessionWatch) throw StateError('private watch failure');
    return session.terminal.future.then<bool>((value) {
      if (!session.terminalCounted) {
        session.terminalCounted = true;
        activeSessions -= 1;
      }
      return value;
    });
  }

  Future<void> waitSessionReady(_Session session) {
    log.add('ready:${session.id}');
    if (throwReady) throw StateError('private ready failure');
    return session.ready.future;
  }

  Future<AssistantListeningCaptureStartResult<_Capture>> startCapture(
    AssistantListeningGeneration<_Session, _Capture> generation,
    _Session session,
  ) {
    startCalls += 1;
    log.add('start:${generation.ordinal}:${session.id}');
    final plan = startPlans.isEmpty
        ? _StartPlan.immediate(
            AssistantListeningCaptureStartResult<_Capture>.started(capture()),
          )
        : startPlans.removeFirst();
    final terminalBeforeReturn = plan.terminalBeforeReturn;
    final plannedResult = plan.result;
    if (terminalBeforeReturn != null &&
        plannedResult is AssistantListeningCaptureStarted<_Capture>) {
      plannedResult.capture.terminal.complete(terminalBeforeReturn);
    }
    return plan.future().then((result) {
      final _Capture? capture = switch (result) {
        AssistantListeningCaptureStarted<_Capture>(:final capture) => capture,
        AssistantListeningCaptureFailedRecoverable<_Capture>(:final capture) =>
          capture,
        _ => null,
      };
      if (capture != null && !capture.counted) {
        capture.counted = true;
        activeCaptures += 1;
        if (activeCaptures > maxActiveCaptures) {
          maxActiveCaptures = activeCaptures;
        }
      }
      return result;
    });
  }

  Future<AssistantListeningCaptureTerminal> waitCaptureTerminal(
    _Capture capture,
  ) {
    log.add('watch-capture:${capture.id}');
    return capture.terminal.future;
  }

  Future<bool> cancelCaptureSource(_Capture capture) {
    cancelCalls += 1;
    log.add('cancel-call:${capture.id}');
    if (capture.throwCancel) throw StateError('private cancel failure');
    return capture.cancelGate?.future ??
        Future<bool>.value(capture.cancelResult);
  }

  Future<bool> stopCapture(_Capture capture) {
    stopCalls += 1;
    log.add('stop-call:${capture.id}');
    if (capture.throwStop) throw StateError('private stop failure');
    final result =
        capture.stopGate?.future ?? Future<bool>.value(capture.stopResult);
    return result.then<bool>((value) {
      if (value && capture.counted && !capture.stopCounted) {
        capture.stopCounted = true;
        activeCaptures -= 1;
      }
      return value;
    });
  }

  Future<bool> recoverAmbiguousCaptureStart(
    AssistantListeningGeneration<_Session, _Capture> generation,
    _Session session,
  ) {
    recoveryCalls += 1;
    log.add('recover:${generation.ordinal}:${session.id}');
    if (throwRecovery) throw StateError('private recovery failure');
    return recoveryGate?.future ?? Future<bool>.value(recoveryResult);
  }

  bool endSession(_Session session) {
    endCalls += 1;
    log.add('end-call:${session.id}');
    if (throwEnd) throw StateError('private end failure');
    if (endAccepted &&
        session.autoCompleteTerminal &&
        !session.terminal.isCompleted) {
      session.terminal.complete(true);
    }
    return endAccepted;
  }
}

AssistantListeningOwner<_Session, _Capture> _owner(
  _Backend backend,
  _TimerHarness timers, {
  AssistantListeningRevokeCallback<_Session, _Capture>? onRevoke,
  AssistantListeningStateCallback<_Session, _Capture>? onListening,
  AssistantListeningStateCallback<_Session, _Capture>? onStopped,
  int initialOrdinal = 0,
}) =>
    AssistantListeningOwner<_Session, _Capture>.forTesting(
      lifecycle: backend,
      onRevoke: onRevoke,
      onListening: onListening,
      onStopped: onStopped,
      admissionMaximumLifetime: _admissionLifetime,
      cleanupMaximumLifetime: _cleanupLifetime,
      timerFactory: timers.call,
      initialOrdinal: initialOrdinal,
    );

void main() {
  test('validates positive deadlines and safe initial ordinal', () {
    final backend = _Backend(<String>[]);
    final timers = _TimerHarness();

    expect(
      () => AssistantListeningOwner<_Session, _Capture>.forTesting(
        lifecycle: backend,
        admissionMaximumLifetime: Duration.zero,
        timerFactory: timers.call,
      ),
      throwsArgumentError,
    );
    expect(
      () => _owner(
        backend,
        timers,
        initialOrdinal: assistantListeningMaximumSafeOrdinal + 1,
      ),
      throwsArgumentError,
    );
  });

  test(
    'runs ordered stages and natural terminal performs exact cleanup',
    () async {
      final log = <String>[];
      final backend = _Backend(log);
      final timers = _TimerHarness();
      final revoked = <AssistantListeningOutcome>[];
      final owner = _owner(
        backend,
        timers,
        onRevoke: (_generation, outcome) {
          revoked.add(outcome);
          log.add('revoke:$outcome');
        },
        onListening: (generation) => log.add('listening:${generation.ordinal}'),
        onStopped: (generation) => log.add('stopped:${generation.ordinal}'),
      );

      final generation = owner.enable();
      final done = await generation.done;
      expect(done.outcome, AssistantListeningOutcome.listening);
      expect(generation.isListening, isTrue);
      expect(owner.isAuthoritative(generation), isTrue);
      expect(log.take(6), <String>[
        'permission:1',
        'route:1',
        'begin:1',
        'watch-session:1',
        'ready:1',
        'start:1:1',
      ]);

      backend.captures.single.terminal.complete(
        AssistantListeningCaptureTerminal.ended,
      );
      final receipt = await generation.cleanup;
      expect(revoked, <AssistantListeningOutcome>[
        AssistantListeningOutcome.captureEnded,
      ]);
      expect(receipt.captureTerminalObserved, isTrue);
      expect(receipt.captureSourceErrorObserved, isFalse);
      expect(receipt.captureCancelAttempted, isTrue);
      expect(receipt.captureStopAttempted, isTrue);
      expect(receipt.sessionEndAccepted, isTrue);
      expect(receipt.sessionTerminalSucceeded, isTrue);
      expect(receipt.exactResourcesSettled, isTrue);
      expect(backend.activeSessions, 0);
      expect(backend.activeCaptures, 0);
    },
  );

  for (final terminal in AssistantListeningCaptureTerminal.values) {
    test('capture $terminal revokes once and cleans once', () async {
      final backend = _Backend(<String>[]);
      final timers = _TimerHarness();
      var revokeCalls = 0;
      final owner = _owner(
        backend,
        timers,
        onRevoke: (_generation, _outcome) => revokeCalls += 1,
      );
      final generation = owner.enable();
      await generation.done;
      backend.captures.single.terminal.complete(terminal);
      final receipt = await generation.cleanup;

      expect(revokeCalls, 1);
      expect(backend.cancelCalls, 1);
      expect(backend.stopCalls, 1);
      expect(backend.endCalls, 1);
      expect(
        receipt.captureSourceErrorObserved,
        terminal == AssistantListeningCaptureTerminal.failed,
      );
      expect(receipt.exactResourcesSettled, isTrue);
    });
  }

  test('failed terminal plus cancel false remains exactly unsettled', () async {
    final backend = _Backend(<String>[]);
    final owner = _owner(backend, _TimerHarness());
    final generation = owner.enable();
    await generation.done;
    final capture = backend.captures.single
      ..cancelResult = false
      ..stopResult = true;

    capture.terminal.complete(AssistantListeningCaptureTerminal.failed);
    final receipt = await generation.cleanup;

    expect(receipt.captureSourceErrorObserved, isTrue);
    expect(receipt.captureCancelSucceeded, isFalse);
    expect(receipt.captureStopSucceeded, isTrue);
    expect(receipt.exactCaptureSettled, isFalse);
    expect(receipt.exactResourcesSettled, isFalse);
    expect(owner.isPoisoned, isTrue);
    expect(owner.snapshot.retainsUncertainResources, isTrue);
  });

  for (final terminal in AssistantListeningCaptureTerminal.values) {
    test('already-complete $terminal never publishes listening', () async {
      final backend = _Backend(<String>[]);
      final capture = backend.capture();
      final plan = _StartPlan.immediate(
        AssistantListeningCaptureStartResult<_Capture>.started(capture),
      )..terminalBeforeReturn = terminal;
      backend.startPlans.add(plan);
      var listeningCalls = 0;
      final owner = _owner(
        backend,
        _TimerHarness(),
        onListening: (_generation) => listeningCalls += 1,
      );

      final generation = owner.enable();
      final done = await generation.done;
      final receipt = await generation.cleanup;

      expect(
        done.outcome,
        terminal == AssistantListeningCaptureTerminal.ended
            ? AssistantListeningOutcome.captureEnded
            : AssistantListeningOutcome.captureSourceFailed,
      );
      expect(listeningCalls, 0);
      expect(generation.isListening, isFalse);
      expect(receipt.captureTerminalObserved, isTrue);
      expect(receipt.exactResourcesSettled, isTrue);
      expect(backend.stopCalls, 1);
    });
  }

  test('held permission then disable never reaches a later stage', () async {
    final backend = _Backend(<String>[]);
    backend.permissionGate = Completer<bool>();
    final owner = _owner(backend, _TimerHarness());

    final first = owner.enable();
    await _flush();
    final off = owner.disable();
    expect((await first.done).outcome, AssistantListeningOutcome.superseded);
    backend.permissionGate!.complete(true);
    expect((await first.cleanup).exactResourcesSettled, isTrue);
    expect((await off.done).outcome, AssistantListeningOutcome.stopped);
    expect(backend.routeCalls, 0);
    expect(backend.beginCalls, 0);
  });

  test('held route then disable never acquires a session', () async {
    final backend = _Backend(<String>[]);
    backend.routeGate = Completer<bool>();
    final owner = _owner(backend, _TimerHarness());

    final first = owner.enable();
    await _flush();
    final off = owner.disable();
    backend.routeGate!.complete(true);
    await first.cleanup;
    await off.done;

    expect(backend.beginCalls, 0);
    expect(backend.startCalls, 0);
  });

  test('held ready then disable ends exact session before OFF', () async {
    final backend = _Backend(<String>[])
      ..holdReady = true
      ..holdSessionTerminal = true;
    final owner = _owner(backend, _TimerHarness());

    final first = owner.enable();
    await _flush();
    final off = owner.disable();
    backend.sessions.single.ready.complete();
    await _flush();
    expect(backend.endCalls, 1);
    expect(backend.startCalls, 0);
    var offCompleted = false;
    unawaited(off.done.then((_) => offCompleted = true));
    await _flush();
    expect(offCompleted, isFalse);
    backend.sessions.single.terminal.complete(true);
    expect((await first.cleanup).exactResourcesSettled, isTrue);
    expect((await off.done).outcome, AssistantListeningOutcome.stopped);
  });

  test(
    'start OFF ON waits exact old cleanup and never overlaps resources',
    () async {
      final backend = _Backend(<String>[])
        ..holdSessionTerminal = true
        ..recoveryResult = true;
      final heldStart = _StartPlan.held();
      backend.startPlans.add(heldStart);
      final owner = _owner(backend, _TimerHarness());

      final first = owner.enable();
      await _flush();
      final off = owner.disable();
      final latest = owner.enable();
      expect((await off.done).outcome, AssistantListeningOutcome.superseded);

      final capture = backend.capture();
      heldStart.gate!.complete(
        AssistantListeningCaptureStartResult<_Capture>.started(capture),
      );
      await _flush();
      expect(backend.cancelCalls, 1);
      expect(backend.stopCalls, 1);
      expect(backend.endCalls, 1);
      expect(backend.permissionCalls, 1);
      var latestCompleted = false;
      unawaited(latest.done.then((_) => latestCompleted = true));
      await _flush();
      expect(latestCompleted, isFalse);

      backend.sessions.single.terminal.complete(true);
      await first.cleanup;
      expect((await latest.done).outcome, AssistantListeningOutcome.listening);
      expect(backend.maxActiveSessions, 1);
      expect(backend.maxActiveCaptures, 1);
      final close = owner.close();
      backend.sessions.last.terminal.complete(true);
      await close;
    },
  );

  test(
    'cancel and recorder stop are called in order before either await',
    () async {
      final log = <String>[];
      final backend = _Backend(log);
      final owner = _owner(
        backend,
        _TimerHarness(),
        onRevoke: (_generation, _outcome) => log.add('revoke-callback'),
      );
      final first = owner.enable();
      await first.done;
      final capture = backend.captures.single
        ..cancelGate = Completer<bool>()
        ..stopGate = Completer<bool>();

      final off = owner.disable();
      await _flush();
      final endIndex = log.indexOf('end-call:1');
      final cancelIndex = log.indexOf('cancel-call:${capture.id}');
      final stopIndex = log.indexOf('stop-call:${capture.id}');
      final revokeIndex = log.indexOf('revoke-callback');
      expect(endIndex, greaterThanOrEqualTo(0));
      expect(cancelIndex, endIndex + 1);
      expect(stopIndex, cancelIndex + 1);
      expect(revokeIndex, stopIndex + 1);
      expect(backend.endCalls, 1);

      capture.cancelGate!.complete(true);
      await _flush();
      expect(backend.endCalls, 1);
      capture.stopGate!.complete(true);
      await first.cleanup;
      expect(backend.endCalls, 1);
      expect((await off.done).outcome, AssistantListeningOutcome.stopped);
    },
  );

  test(
    'session ready failure waits exact same-token terminal before retry',
    () async {
      final backend = _Backend(<String>[])
        ..throwReady = true
        ..holdSessionTerminal = true;
      final owner = _owner(backend, _TimerHarness());
      final first = owner.enable();
      await _flush();
      backend.throwReady = false;
      backend.holdSessionTerminal = false;
      final retry = owner.enable();
      await _flush();

      expect(backend.permissionCalls, 1);
      var retryCompleted = false;
      unawaited(retry.done.then((_) => retryCompleted = true));
      await _flush();
      expect(retryCompleted, isFalse);
      backend.sessions.single.terminal.complete(true);
      expect((await first.cleanup).exactSessionSettled, isTrue);
      expect((await retry.done).outcome, AssistantListeningOutcome.listening);
      await owner.close();
    },
  );

  test('synchronous cancel throw still enqueues recorder stop before callbacks',
      () async {
    final log = <String>[];
    final backend = _Backend(log);
    late AssistantListeningOwner<_Session, _Capture> owner;
    owner = _owner(
      backend,
      _TimerHarness(),
      onRevoke: (revoked, _outcome) {
        expect(owner.isAuthoritative(revoked), isFalse);
        final capture = backend.captures.single;
        expect(
          log.indexOf('stop-call:${capture.id}'),
          log.indexOf('cancel-call:${capture.id}') + 1,
        );
      },
    );
    final generation = owner.enable();
    await generation.done;
    final capture = backend.captures.single..throwCancel = true;

    final pending = owner.disable();
    final receipt = await generation.cleanup;
    final endIndex = log.indexOf('end-call:1');
    final cancelIndex = log.indexOf('cancel-call:${capture.id}');
    final stopIndex = log.indexOf('stop-call:${capture.id}');

    expect(endIndex, greaterThanOrEqualTo(0));
    expect(cancelIndex, endIndex + 1);
    expect(stopIndex, cancelIndex + 1);
    expect(backend.stopCalls, 1);
    expect(receipt.exactResourcesSettled, isFalse);
    expect(owner.isPoisoned, isTrue);
    expect(
      (await pending.done).outcome,
      AssistantListeningOutcome.ownerPoisoned,
    );
    expect(backend.permissionCalls, 1);
  });

  test(
    'endSession false retains, poisons, and rejects pending retry',
    () async {
      final backend = _Backend(<String>[])..endAccepted = false;
      final owner = _owner(backend, _TimerHarness());
      final first = owner.enable();
      await first.done;
      final retry = owner.enable();

      final receipt = await first.cleanup;
      expect(receipt.sessionEndAccepted, isFalse);
      expect(receipt.exactSessionSettled, isFalse);
      expect(receipt.exactResourcesSettled, isFalse);
      expect(owner.isPoisoned, isTrue);
      expect(owner.snapshot.retainsUncertainResources, isTrue);
      expect(
        (await retry.done).outcome,
        AssistantListeningOutcome.ownerPoisoned,
      );
      expect(() => owner.enable(), throwsA(isA<AssistantListeningFailure>()));
      final close = await owner.close();
      expect(close.exactResourcesSettled, isFalse);
      expect(close.retainsUncertainResources, isTrue);
    },
  );

  test('clean capture-start failure permits a later generation', () async {
    final backend = _Backend(<String>[]);
    backend.startPlans.add(
      _StartPlan.immediate(
        const AssistantListeningCaptureStartResult<_Capture>.failedClean(),
      ),
    );
    final owner = _owner(backend, _TimerHarness());

    final first = owner.enable();
    final receipt = await first.cleanup;
    expect(
      receipt.captureStartDisposition,
      AssistantListeningCaptureStartDisposition.failedClean,
    );
    expect(receipt.captureRecoveryAttempted, isFalse);
    expect(receipt.captureCancelAttempted, isFalse);
    expect(receipt.captureStopAttempted, isFalse);
    expect(receipt.exactResourcesSettled, isTrue);
    expect(owner.isPoisoned, isFalse);

    final retry = owner.enable();
    expect((await retry.done).outcome, AssistantListeningOutcome.listening);
    await owner.close();
  });

  test('recoverable capture-start failure cleans exact capture', () async {
    final backend = _Backend(<String>[]);
    final capture = backend.capture();
    backend.startPlans.add(
      _StartPlan.immediate(
        AssistantListeningCaptureStartResult<_Capture>.failedRecoverable(
          capture,
        ),
      ),
    );
    final owner = _owner(backend, _TimerHarness());

    final receipt = await owner.enable().cleanup;
    expect(
      receipt.captureStartDisposition,
      AssistantListeningCaptureStartDisposition.failedRecoverable,
    );
    expect(receipt.captureCancelSucceeded, isTrue);
    expect(receipt.captureStopSucceeded, isTrue);
    expect(receipt.exactResourcesSettled, isTrue);
    expect(owner.isPoisoned, isFalse);
  });

  for (final throws in <bool>[false, true]) {
    test(
      'ambiguous capture start (throws=$throws) recovers only on exact true',
      () async {
        final backend = _Backend(<String>[])..recoveryResult = true;
        final plan = throws
            ? (_StartPlan.immediate(
                const AssistantListeningCaptureStartResult<
                    _Capture>.failedClean(),
              )..throwAsynchronously = true)
            : _StartPlan.immediate(
                const AssistantListeningCaptureStartResult<
                    _Capture>.failedAmbiguous(),
              );
        backend.startPlans.add(plan);
        final owner = _owner(backend, _TimerHarness());

        final receipt = await owner.enable().cleanup;
        expect(receipt.captureRecoveryAttempted, isTrue);
        expect(receipt.captureRecoverySucceeded, isTrue);
        expect(receipt.exactResourcesSettled, isTrue);
        expect(owner.isPoisoned, isFalse);
        expect(
          receipt.captureStartDisposition,
          throws
              ? AssistantListeningCaptureStartDisposition.threwAmbiguous
              : AssistantListeningCaptureStartDisposition.failedAmbiguous,
        );
      },
    );
  }

  test('failed ambiguous recovery publishes uncertainty and poisons', () async {
    final backend = _Backend(<String>[])..recoveryResult = false;
    backend.startPlans.add(
      _StartPlan.immediate(
        const AssistantListeningCaptureStartResult<_Capture>.failedAmbiguous(),
      ),
    );
    final owner = _owner(backend, _TimerHarness());

    final receipt = await owner.enable().cleanup;
    expect(receipt.captureRecoverySucceeded, isFalse);
    expect(receipt.exactCaptureSettled, isFalse);
    expect(owner.isPoisoned, isTrue);
    expect(owner.snapshot.retainsUncertainResources, isTrue);
  });

  test(
    'active admission expiry fences late start and cleans returned handle',
    () async {
      final backend = _Backend(<String>[]);
      final held = _StartPlan.held();
      backend.startPlans.add(held);
      final timers = _TimerHarness();
      final owner = _owner(backend, timers);
      final generation = owner.enable();
      await _flush();

      timers.active(_admissionLifetime).fire();
      expect(
        (await generation.done).outcome,
        AssistantListeningOutcome.admissionDeadlineExceeded,
      );
      expect(owner.isPoisoned, isTrue);
      expect(backend.recoveryCalls, 1);
      final capture = backend.capture();
      held.gate!.complete(
        AssistantListeningCaptureStartResult<_Capture>.started(capture),
      );
      final receipt = await generation.cleanup;

      expect(receipt.admissionDeadlineExpired, isTrue);
      expect(receipt.captureCancelSucceeded, isTrue);
      expect(receipt.captureStopSucceeded, isTrue);
      expect(receipt.exactResourcesSettled, isTrue);
      expect(generation.isListening, isFalse);
    },
  );

  test(
    'cleanup deadline freezes false receipt and late unwind cannot upgrade it',
    () async {
      final backend = _Backend(<String>[])..holdSessionTerminal = true;
      final timers = _TimerHarness();
      final owner = _owner(backend, timers);
      final generation = owner.enable();
      await generation.done;
      final capture = backend.captures.single
        ..cancelGate = Completer<bool>()
        ..stopGate = Completer<bool>();
      final pending = owner.disable();
      await _flush();

      timers.active(_cleanupLifetime).fire();
      final frozen = await generation.cleanup;
      expect(frozen.cleanupDeadlineExpired, isTrue);
      expect(frozen.exactResourcesSettled, isFalse);
      expect(
        (await pending.done).outcome,
        AssistantListeningOutcome.ownerPoisoned,
      );

      capture.cancelGate!.complete(true);
      capture.stopGate!.complete(true);
      await _flush();
      backend.sessions.single.terminal.complete(true);
      await _flush();
      expect(identical(await generation.cleanup, frozen), isTrue);
      expect((await owner.close()).exactResourcesSettled, isFalse);
    },
  );

  test(
    'terminal during held stop deduplicates revoke and cleanup calls',
    () async {
      final backend = _Backend(<String>[]);
      var revokes = 0;
      final owner = _owner(
        backend,
        _TimerHarness(),
        onRevoke: (_generation, _outcome) => revokes += 1,
      );
      final generation = owner.enable();
      await generation.done;
      final capture = backend.captures.single
        ..cancelGate = Completer<bool>()
        ..stopGate = Completer<bool>();
      owner.disable();
      await _flush();
      capture.terminal.complete(AssistantListeningCaptureTerminal.ended);
      capture.cancelGate!.complete(true);
      capture.stopGate!.complete(true);
      final receipt = await generation.cleanup;

      expect(revokes, 1);
      expect(backend.cancelCalls, 1);
      expect(backend.stopCalls, 1);
      expect(backend.endCalls, 1);
      expect(receipt.exactResourcesSettled, isTrue);
    },
  );

  test('pending expiry is clean and does not poison the owner', () async {
    final backend = _Backend(<String>[]);
    backend.permissionGate = Completer<bool>();
    final timers = _TimerHarness();
    final owner = _owner(backend, timers);
    final first = owner.enable();
    await _flush();
    final pending = owner.disable();

    timers.active(_admissionLifetime).fire();
    final receipt = await pending.cleanup;
    expect(receipt.started, isFalse);
    expect(receipt.exactResourcesSettled, isTrue);
    expect(owner.isPoisoned, isFalse);

    backend.permissionGate!.complete(true);
    await first.cleanup;
    expect(owner.isPoisoned, isFalse);
  });

  test(
    'timer construction throw and synchronous fire are fail closed',
    () async {
      final backend1 = _Backend(<String>[]);
      final throwing = _TimerHarness()..throwCalls.add(1);
      final owner1 = _owner(backend1, throwing);
      final failed = owner1.enable();
      expect(
        (await failed.done).outcome,
        AssistantListeningOutcome.ownerPoisoned,
      );
      expect((await failed.cleanup).exactResourcesSettled, isFalse);
      expect(backend1.permissionCalls, 0);
      expect(owner1.isPoisoned, isTrue);

      final backend2 = _Backend(<String>[]);
      final synchronous = _TimerHarness()..synchronousFireCalls.add(1);
      final owner2 = _owner(backend2, synchronous);
      final expired = owner2.enable();
      expect(
        (await expired.done).outcome,
        AssistantListeningOutcome.admissionDeadlineExceeded,
      );
      expect((await expired.cleanup).exactResourcesSettled, isTrue);
      expect(owner2.isPoisoned, isFalse);
      expect(backend2.permissionCalls, 0);
    },
  );

  test(
    'cleanup timer construction throw publishes bounded false receipt',
    () async {
      final backend = _Backend(<String>[]);
      final timers = _TimerHarness()..throwCalls.add(2);
      final owner = _owner(backend, timers);
      final generation = owner.enable();
      await generation.done;

      owner.disable();
      final receipt = await generation.cleanup;
      expect(receipt.cleanupDeadlineExpired, isTrue);
      expect(receipt.exactResourcesSettled, isFalse);
      expect(owner.isPoisoned, isTrue);
    },
  );

  test('timer cancel throw is contained before listening publication',
      () async {
    final backend = _Backend(<String>[]);
    final timers = _TimerHarness()..throwCancelCalls.add(1);
    var listeningCalls = 0;
    final owner = _owner(
      backend,
      timers,
      onListening: (_generation) => listeningCalls += 1,
    );

    final generation = owner.enable();
    final done = await generation.done;
    final receipt = await generation.cleanup;

    expect(done.outcome, AssistantListeningOutcome.ownerPoisoned);
    expect(listeningCalls, 0);
    expect(owner.isPoisoned, isTrue);
    expect(receipt.exactResourcesSettled, isTrue);
  });

  for (final failedOperation in <String>['cancel', 'stop', 'end']) {
    test(
      '$failedOperation false/throw poisons without false settlement',
      () async {
        final backend = _Backend(<String>[]);
        final owner = _owner(backend, _TimerHarness());
        final generation = owner.enable();
        await generation.done;
        final capture = backend.captures.single;
        switch (failedOperation) {
          case 'cancel':
            capture.cancelResult = false;
          case 'stop':
            capture.throwStop = true;
          case 'end':
            backend.endAccepted = false;
        }
        owner.disable();
        final receipt = await generation.cleanup;
        expect(receipt.exactResourcesSettled, isFalse);
        expect(owner.isPoisoned, isTrue);
        expect(owner.snapshot.retainsUncertainResources, isTrue);
      },
    );
  }

  test(
    'callback failure poisons but exact resources remain truthfully settled',
    () async {
      final backend = _Backend(<String>[]);
      final owner = _owner(
        backend,
        _TimerHarness(),
        onListening: (_generation) => throw StateError('private UI failure'),
      );
      final generation = owner.enable();
      final receipt = await generation.cleanup;

      expect(receipt.listeningCallbackFailures, 1);
      expect(receipt.exactResourcesSettled, isTrue);
      expect(owner.isPoisoned, isTrue);
      expect(owner.snapshot.retainsUncertainResources, isFalse);
    },
  );

  test('revoke callback throw cannot interrupt exact cleanup', () async {
    final backend = _Backend(<String>[]);
    final owner = _owner(
      backend,
      _TimerHarness(),
      onRevoke: (_generation, _outcome) {
        throw StateError('private revoke callback failure');
      },
    );
    final generation = owner.enable();
    await generation.done;

    owner.disable();
    final receipt = await generation.cleanup;
    expect(receipt.revokeCallbackFailures, 1);
    expect(receipt.exactResourcesSettled, isTrue);
    expect(owner.isPoisoned, isTrue);
    expect(owner.snapshot.retainsUncertainResources, isFalse);
  });

  test('stopped callback throw does not rewrite exact resource proof',
      () async {
    final backend = _Backend(<String>[]);
    final owner = _owner(
      backend,
      _TimerHarness(),
      onStopped: (_generation) {
        throw StateError('private stopped callback failure');
      },
    );
    final generation = owner.enable();
    await generation.done;

    owner.disable();
    final receipt = await generation.cleanup;
    expect(receipt.stoppedCallbackFailures, 1);
    expect(receipt.exactResourcesSettled, isTrue);
    expect(owner.isPoisoned, isTrue);
    expect(owner.snapshot.retainsUncertainResources, isFalse);
  });

  test('early session terminal revokes before late ready can start capture',
      () async {
    final log = <String>[];
    final backend = _Backend(log)
      ..holdReady = true
      ..holdSessionTerminal = true;
    final owner = _owner(backend, _TimerHarness());
    final generation = owner.enable();
    await _flush();

    backend.sessions.single.terminal.complete(false);
    await _flush();
    expect(backend.endCalls, 1);
    expect(backend.startCalls, 0);
    backend.sessions.single.ready.complete();
    final receipt = await generation.cleanup;

    expect(receipt.outcome, AssistantListeningOutcome.sessionSourceFailed);
    expect(receipt.exactSessionSettled, isFalse);
    expect(backend.startCalls, 0);
    expect(log.indexOf('end-call:1'), greaterThan(log.indexOf('ready:1')));
  });

  test(
      'session failure fences and starts stop before reentrant pending callback',
      () async {
    final log = <String>[];
    final backend = _Backend(log)..holdSessionTerminal = true;
    late AssistantListeningOwner<_Session, _Capture> owner;
    AssistantListeningGeneration<_Session, _Capture>? pending;
    owner = _owner(
      backend,
      _TimerHarness(),
      onRevoke: (generation, _outcome) {
        expect(owner.isAuthoritative(generation), isFalse);
        final capture = backend.captures.single;
        expect(log.indexOf('end-call:1'), greaterThanOrEqualTo(0));
        expect(
          log.indexOf('cancel-call:${capture.id}'),
          log.indexOf('end-call:1') + 1,
        );
        expect(
          log.indexOf('stop-call:${capture.id}'),
          log.indexOf('cancel-call:${capture.id}') + 1,
        );
        pending = owner.enable();
      },
    );
    final generation = owner.enable();
    await generation.done;

    backend.sessions.single.terminal.complete(false);
    final receipt = await generation.cleanup;

    expect(receipt.outcome, AssistantListeningOutcome.sessionSourceFailed);
    expect(receipt.captureStopAttempted, isTrue);
    expect(receipt.exactSessionSettled, isFalse);
    expect(
        (await pending!.done).outcome, AssistantListeningOutcome.ownerPoisoned);
    expect(backend.permissionCalls, 1);
  });

  test(
    'onListening can reentrantly disable without reopening capture',
    () async {
      final backend = _Backend(<String>[]);
      late AssistantListeningOwner<_Session, _Capture> owner;
      AssistantListeningGeneration<_Session, _Capture>? off;
      owner = _owner(
        backend,
        _TimerHarness(),
        onListening: (_generation) => off = owner.disable(),
      );

      final first = owner.enable();
      expect((await first.done).outcome, AssistantListeningOutcome.listening);
      expect((await first.cleanup).exactResourcesSettled, isTrue);
      expect((await off!.done).outcome, AssistantListeningOutcome.stopped);
      expect(backend.startCalls, 1);
    },
  );

  test('onRevoke can reentrantly enable and latest generation wins', () async {
    final backend = _Backend(<String>[]);
    backend.permissionGate = Completer<bool>();
    late AssistantListeningOwner<_Session, _Capture> owner;
    AssistantListeningGeneration<_Session, _Capture>? reentrant;
    var didReenter = false;
    owner = _owner(
      backend,
      _TimerHarness(),
      onRevoke: (_generation, _outcome) {
        if (!didReenter) {
          didReenter = true;
          reentrant = owner.enable();
        }
      },
    );
    final first = owner.enable();
    await _flush();
    final displaced = owner.disable();
    expect(
      (await displaced.done).outcome,
      AssistantListeningOutcome.superseded,
    );
    backend.permissionGate!.complete(true);
    await first.cleanup;
    expect(owner.isAuthoritative(reentrant!), isTrue);
    await owner.close();
  });

  test(
    'stopped callback can prepublish a memoized reentrant close',
    () async {
      final backend = _Backend(<String>[]);
      late AssistantListeningOwner<_Session, _Capture> owner;
      Future<AssistantListeningCloseReceipt>? nestedClose;
      owner = _owner(
        backend,
        _TimerHarness(),
        onStopped: (_generation) => nestedClose ??= owner.close(),
      );
      final generation = owner.enable();
      await generation.done;
      owner.disable();
      await generation.cleanup;
      final firstClose = nestedClose!;
      final secondClose = owner.close();
      expect(identical(firstClose, secondClose), isTrue);
      final receipt = await firstClose;
      expect(identical(firstClose, nestedClose), isTrue);
      expect(receipt.exactResourcesSettled, isTrue);
      expect(() => owner.enable(), throwsA(isA<AssistantListeningFailure>()));
    },
  );

  test(
    'foreign exact revoke and stale exact revoke cannot touch current',
    () async {
      final firstBackend = _Backend(<String>[]);
      final secondBackend = _Backend(<String>[]);
      final firstOwner = _owner(firstBackend, _TimerHarness());
      final secondOwner = _owner(secondBackend, _TimerHarness());
      final foreign = firstOwner.enable();
      final current = secondOwner.enable();
      await Future.wait(<Future<AssistantListeningDone>>[
        foreign.done,
        current.done,
      ]);

      expect(() => secondOwner.revokeExact(foreign), throwsArgumentError);
      expect(secondOwner.isAuthoritative(current), isTrue);
      await secondOwner.revokeExact(current);
      expect(secondOwner.isAuthoritative(current), isFalse);
      expect(secondBackend.cancelCalls, 1);
      expect(firstBackend.cancelCalls, 0);
      await firstOwner.close();
    },
  );

  test(
    'safe ordinal exhaustion poisons and revokes the last generation',
    () async {
      final backend = _Backend(<String>[]);
      final owner = _owner(
        backend,
        _TimerHarness(),
        initialOrdinal: assistantListeningMaximumSafeOrdinal - 1,
      );
      final last = owner.enable();
      expect(last.ordinal, assistantListeningMaximumSafeOrdinal);
      await last.done;

      expect(() => owner.disable(), throwsA(isA<AssistantListeningFailure>()));
      final receipt = await last.cleanup;
      expect(receipt.exactResourcesSettled, isTrue);
      expect(owner.isPoisoned, isTrue);
    },
  );

  test('receipt and failures retain no provider error content', () async {
    final backend = _Backend(<String>[])..throwPermission = true;
    final owner = _owner(backend, _TimerHarness());
    final generation = owner.enable();
    final done = await generation.done;
    final receipt = await generation.cleanup;

    expect(done.failure?.code, 'permission_failed');
    expect(done.toString(), isNot(contains('private permission failure')));
    expect(receipt.toString(), isNot(contains('private permission failure')));
  });
}
