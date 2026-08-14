// Deterministic tests for the pure-Dart TTS isolate-generation coordinator.
// Fakes drive every spawn, message, send, timeout, and cleanup edge. No real
// isolate, plugin, model, filesystem asset, network, audio, or device is used.
import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/tts_isolate_lifecycle.dart';
import 'package:speaker_mobile/tts_process_owner.dart';

final class _SpawnCall {
  _SpawnCall(this.epoch, this.emit);

  final int epoch;
  final void Function(TtsWorkerEvent) emit;
  final Completer<TtsWorkerHandle> result = Completer<TtsWorkerHandle>();
}

final class _FakeDriver implements TtsWorkerDriver {
  final List<_SpawnCall> spawns = <_SpawnCall>[];

  @override
  Future<TtsWorkerHandle> spawn(
    int epoch,
    void Function(TtsWorkerEvent) emit,
  ) {
    final call = _SpawnCall(epoch, emit);
    spawns.add(call);
    return call.result.future;
  }
}

final class _FakeHandle implements TtsWorkerHandle {
  final List<TtsWorkerCommand> commands = <TtsWorkerCommand>[];
  Object? sendError;
  Object? killError;
  Object? closeError;
  int sendCalls = 0;
  int killCalls = 0;
  int closeCalls = 0;

  @override
  void send(TtsWorkerCommand command) {
    sendCalls++;
    final error = sendError;
    if (error != null) throw error;
    commands.add(command);
  }

  @override
  void kill() {
    killCalls++;
    final error = killError;
    if (error != null) throw error;
  }

  @override
  Future<void> closeEvents() async {
    closeCalls++;
    final error = closeError;
    if (error != null) throw error;
  }
}

final class _ControlledDelay {
  final List<Completer<void>> waits = <Completer<void>>[];

  Future<void> call(Duration duration) {
    final wait = Completer<void>();
    waits.add(wait);
    return wait.future;
  }

  void fire(int index) => waits[index].complete();
}

TtsProcessLease _lease() => TtsProcessOwnerRegistry().tryAcquire()!;

TtsIsolateLifecycle _lifecycle(_FakeDriver driver) => TtsIsolateLifecycle(
      driver: driver,
      delay: _ControlledDelay().call,
    );

Future<_FakeHandle> _makeReady(
  _FakeDriver driver,
  TtsIsolateLifecycle lifecycle,
  TtsProcessLease lease, {
  Object init = 'init',
}) async {
  final ready = lifecycle.ensureReady(lease, init);
  expect(driver.spawns, hasLength(1));
  final spawn = driver.spawns.single;
  final handle = _FakeHandle();
  spawn.result.complete(handle);
  await Future<void>.delayed(Duration.zero);
  spawn.emit(TtsWorkerSendPort(spawn.epoch));
  await Future<void>.delayed(Duration.zero);
  expect(handle.commands, hasLength(1));
  final command = handle.commands.single;
  expect(command, isA<TtsWorkerInit>());
  final initCommand = command as TtsWorkerInit;
  expect(initCommand.epoch, spawn.epoch);
  expect(initCommand.initPayload, same(init));
  spawn.emit(TtsWorkerReady(spawn.epoch));
  expect(await ready, isTrue);
  return handle;
}

void main() {
  test('concurrent ensure shares exactly one spawn and one initialization',
      () async {
    final driver = _FakeDriver();
    final delay = _ControlledDelay();
    final lifecycle = TtsIsolateLifecycle(
      driver: driver,
      readyTimeout: const Duration(seconds: 30),
      resultTimeout: const Duration(seconds: 20),
      shutdownTimeout: const Duration(seconds: 5),
      delay: delay.call,
    );
    final lease = _lease();
    final init = Object();

    final first = lifecycle.ensureReady(lease, init);
    final second = lifecycle.ensureReady(lease, init);
    expect(driver.spawns, hasLength(1));

    final spawn = driver.spawns.single;
    final handle = _FakeHandle();
    spawn.result.complete(handle);
    await Future<void>.delayed(Duration.zero);
    spawn.emit(TtsWorkerSendPort(spawn.epoch));
    await Future<void>.delayed(Duration.zero);
    expect(handle.commands.whereType<TtsWorkerInit>(), hasLength(1));
    spawn.emit(TtsWorkerReady(spawn.epoch));

    expect(await Future.wait([first, second]), [true, true]);
    expect(lifecycle.snapshot.ready, isTrue);
    expect(lifecycle.snapshot.spawning, isFalse);
  });

  test('handshake and ready emitted before spawn return replay exactly once',
      () async {
    final driver = _FakeDriver();
    final lifecycle = _lifecycle(driver);
    final lease = _lease();
    final init = Object();

    final ready = lifecycle.ensureReady(lease, init);
    final spawn = driver.spawns.single;
    spawn.emit(TtsWorkerSendPort(spawn.epoch));
    spawn.emit(TtsWorkerReady(spawn.epoch));
    expect(lifecycle.snapshot.spawning, isTrue);

    final handle = _FakeHandle();
    spawn.result.complete(handle);
    expect(await ready, isTrue);
    final initCommands = handle.commands.whereType<TtsWorkerInit>().toList();
    expect(initCommands, hasLength(1));
    expect(initCommands.single.epoch, spawn.epoch);
    expect(initCommands.single.initPayload, same(init));
    expect(lifecycle.snapshot.ready, isTrue);
    expect(lifecycle.snapshot.spawning, isFalse);
  });

  test('early ready before handshake fails closed and cannot later recover',
      () async {
    final driver = _FakeDriver();
    final lifecycle = _lifecycle(driver);
    final lease = _lease();

    final ready = lifecycle.ensureReady(lease, Object());
    final spawn = driver.spawns.single;
    spawn.emit(TtsWorkerReady(spawn.epoch));
    spawn.emit(TtsWorkerSendPort(spawn.epoch));
    final handle = _FakeHandle();
    spawn.result.complete(handle);

    expect(await ready, isFalse);
    expect(lifecycle.snapshot.ready, isFalse);
    expect(lifecycle.snapshot.uncertain, isTrue);
    expect(handle.commands, isEmpty);
    expect(await lifecycle.request(lease, 'no', '/no.wav'), isNull);
    expect(await lease.close(lifecycle.dispose), isFalse);
    expect(handle.killCalls, 1);
    expect(handle.closeCalls, 1);
  });

  test('dispose completes pending request null before exact shutdown receipt',
      () async {
    final driver = _FakeDriver();
    final lifecycle = _lifecycle(driver);
    final lease = _lease();
    final handle = await _makeReady(driver, lifecycle, lease);
    final spawn = driver.spawns.single;

    final pending = lifecycle.request(lease, 'hello', '/out.wav');
    await Future<void>.delayed(Duration.zero);
    final request = handle.commands.whereType<TtsWorkerRequest>().single;
    expect(request.epoch, spawn.epoch);
    expect(request.text, 'hello');
    expect(request.outPath, '/out.wav');
    expect(lifecycle.snapshot.pending, 1);

    final disposing = lifecycle.dispose();
    expect(await pending, isNull);
    expect(lifecycle.snapshot.pending, 0);
    expect(lifecycle.snapshot.closing, isTrue);
    expect(handle.commands.whereType<TtsWorkerShutdown>(), hasLength(1));
    expect(handle.killCalls, 0);
    expect(handle.closeCalls, 0);

    // An exact result after synchronous disposal revocation is stale even
    // before the shutdown receipt. It cannot resurrect the pending caller.
    spawn.emit(TtsWorkerResult(spawn.epoch, request.id, '/late.wav'));
    expect(await pending, isNull);
    expect(lifecycle.snapshot.pending, 0);

    spawn.emit(TtsWorkerShutdownComplete(spawn.epoch));
    expect(await disposing, isTrue);
    expect(handle.killCalls, 1);
    expect(handle.closeCalls, 1);
  });

  test('late old result and ready cannot mutate a fresh generation', () async {
    final registry = TtsProcessOwnerRegistry();
    final firstLease = registry.tryAcquire()!;
    final driver = _FakeDriver();
    final lifecycle = _lifecycle(driver);
    final firstHandle = await _makeReady(driver, lifecycle, firstLease);
    final firstSpawn = driver.spawns.single;

    final firstPending = lifecycle.request(firstLease, 'old', '/old.wav');
    await Future<void>.delayed(Duration.zero);
    final firstRequest =
        firstHandle.commands.whereType<TtsWorkerRequest>().single;
    final firstClose = firstLease.close(lifecycle.dispose);
    expect(await firstPending, isNull);
    firstSpawn.emit(TtsWorkerShutdownComplete(firstSpawn.epoch));
    expect(await firstClose, isTrue);

    final secondLease = registry.tryAcquire()!;
    final secondReady = lifecycle.ensureReady(secondLease, Object());
    expect(driver.spawns, hasLength(2));
    final secondSpawn = driver.spawns.last;
    final secondHandle = _FakeHandle();
    secondSpawn.result.complete(secondHandle);
    await Future<void>.delayed(Duration.zero);
    secondSpawn.emit(TtsWorkerSendPort(secondSpawn.epoch));
    await Future<void>.delayed(Duration.zero);
    secondSpawn.emit(TtsWorkerReady(secondSpawn.epoch));
    expect(await secondReady, isTrue);

    final secondPending = lifecycle.request(secondLease, 'new', '/new.wav');
    await Future<void>.delayed(Duration.zero);
    final secondRequest =
        secondHandle.commands.whereType<TtsWorkerRequest>().single;
    expect(secondRequest.id, isNot(firstRequest.id));

    firstSpawn.emit(
      TtsWorkerResult(
        firstSpawn.epoch,
        firstRequest.id,
        '/forged-old.wav',
      ),
    );
    firstSpawn.emit(TtsWorkerReady(firstSpawn.epoch));
    firstSpawn.emit(TtsWorkerSendPort(secondSpawn.epoch));
    firstSpawn.emit(TtsWorkerReady(secondSpawn.epoch));
    firstSpawn.emit(
      TtsWorkerResult(
        secondSpawn.epoch,
        secondRequest.id,
        '/forged-current.wav',
      ),
    );
    firstSpawn.emit(TtsWorkerShutdownComplete(secondSpawn.epoch));
    await Future<void>.delayed(Duration.zero);
    expect(lifecycle.snapshot.epoch, secondSpawn.epoch);
    expect(lifecycle.snapshot.ready, isTrue);
    expect(lifecycle.snapshot.pending, 1);

    secondSpawn.emit(
      TtsWorkerResult(
        secondSpawn.epoch,
        secondRequest.id,
        '/new.wav',
      ),
    );
    expect(await secondPending, '/new.wav');
  });

  test('dispose during spawn invalidates late handle and never resurrects',
      () async {
    final driver = _FakeDriver();
    final lifecycle = TtsIsolateLifecycle(
      driver: driver,
      readyTimeout: Duration.zero,
      delay: _ControlledDelay().call,
    );
    final lease = _lease();

    final ready = lifecycle.ensureReady(lease, Object());
    final spawn = driver.spawns.single;
    final disposing = lifecycle.dispose();
    var disposeCompleted = false;
    unawaited(disposing.then((_) => disposeCompleted = true));
    expect(lifecycle.snapshot.closing, isTrue);
    expect(await ready, isFalse);
    await Future<void>.delayed(Duration.zero);
    expect(disposeCompleted, isFalse);

    final lateHandle = _FakeHandle();
    spawn.result.complete(lateHandle);
    expect(await disposing, isFalse);
    await Future<void>.delayed(Duration.zero);
    spawn.emit(TtsWorkerSendPort(spawn.epoch));
    spawn.emit(TtsWorkerReady(spawn.epoch));
    await Future<void>.delayed(Duration.zero);
    expect(lifecycle.snapshot.ready, isFalse);
    expect(lifecycle.snapshot.closing, isTrue);
    expect(lateHandle.commands, isEmpty);
    expect(lateHandle.killCalls, 1);
    expect(lateHandle.closeCalls, 1);
  });

  test('foreign and revoked leases cannot spawn or send', () async {
    final driver = _FakeDriver();
    final lifecycle = _lifecycle(driver);
    final admitted = _lease();
    final foreign = _lease();

    final ready = lifecycle.ensureReady(admitted, Object());
    expect(driver.spawns, hasLength(1));
    expect(await lifecycle.ensureReady(foreign, Object()), isFalse);
    expect(driver.spawns, hasLength(1));

    final spawn = driver.spawns.single;
    final handle = _FakeHandle();
    spawn.result.complete(handle);
    await Future<void>.delayed(Duration.zero);
    spawn.emit(TtsWorkerSendPort(spawn.epoch));
    await Future<void>.delayed(Duration.zero);
    spawn.emit(TtsWorkerReady(spawn.epoch));
    expect(await ready, isTrue);

    admitted.revoke();
    expect(await lifecycle.request(admitted, 'revoked', '/no.wav'), isNull);
    expect(await lifecycle.request(foreign, 'foreign', '/no.wav'), isNull);
    expect(handle.commands.whereType<TtsWorkerRequest>(), isEmpty);
  });

  test('spawn and init-send failures are conservative and close once',
      () async {
    final spawnDriver = _FakeDriver();
    final spawnLifecycle = _lifecycle(spawnDriver);
    final spawnLease = _lease();
    final ready = spawnLifecycle.ensureReady(spawnLease, Object());
    spawnDriver.spawns.single.result.completeError(StateError('spawn failed'));
    expect(await ready, isFalse);
    expect(spawnLifecycle.snapshot.uncertain, isTrue);
    expect(await spawnLease.close(spawnLifecycle.dispose), isFalse);

    final sendDriver = _FakeDriver();
    final sendLifecycle = _lifecycle(sendDriver);
    final sendLease = _lease();
    final sendReady = sendLifecycle.ensureReady(sendLease, Object());
    final sendSpawn = sendDriver.spawns.single;
    final handle = _FakeHandle()..sendError = StateError('send failed');
    sendSpawn.result.complete(handle);
    await Future<void>.delayed(Duration.zero);
    sendSpawn.emit(TtsWorkerSendPort(sendSpawn.epoch));
    expect(await sendReady, isFalse);
    expect(sendLifecycle.snapshot.uncertain, isTrue);
    final sendClose = sendLease.close(sendLifecycle.dispose);
    await Future<void>.delayed(Duration.zero);
    sendSpawn.emit(TtsWorkerShutdownComplete(sendSpawn.epoch));
    expect(await sendClose, isFalse);
    expect(handle.killCalls, 1);
    expect(handle.closeCalls, 1);
  });

  test('late ready after failed initialization cannot reopen generation',
      () async {
    final driver = _FakeDriver();
    final lifecycle = _lifecycle(driver);
    final lease = _lease();
    final ready = lifecycle.ensureReady(lease, Object());
    final spawn = driver.spawns.single;
    final handle = _FakeHandle()..sendError = StateError('init failed');
    spawn.result.complete(handle);
    await Future<void>.delayed(Duration.zero);
    spawn.emit(TtsWorkerSendPort(spawn.epoch));
    expect(await ready, isFalse);
    expect(lifecycle.snapshot.uncertain, isTrue);

    handle.sendError = null;
    spawn.emit(TtsWorkerReady(spawn.epoch));
    final rejected = lifecycle.request(lease, 'must not run', '/no.wav');
    await Future<void>.delayed(Duration.zero);
    final requestCount = handle.commands.whereType<TtsWorkerRequest>().length;
    final closing = lease.close(lifecycle.dispose);

    expect(await rejected, isNull);
    expect(await closing, isFalse);
    expect(lifecycle.snapshot.ready, isFalse);
    expect(requestCount, 0);
    expect(handle.killCalls, 1);
    expect(handle.closeCalls, 1);
  });

  test('request send failure fences every later request', () async {
    final driver = _FakeDriver();
    final lifecycle = _lifecycle(driver);
    final lease = _lease();
    final handle = await _makeReady(driver, lifecycle, lease);

    handle.sendError = StateError('request send failed');
    expect(await lifecycle.request(lease, 'first', '/first.wav'), isNull);
    expect(lifecycle.snapshot.uncertain, isTrue);
    handle.sendError = null;

    final rejected = lifecycle.request(lease, 'second', '/second.wav');
    await Future<void>.delayed(Duration.zero);
    final requestCount = handle.commands.whereType<TtsWorkerRequest>().length;
    final closing = lease.close(lifecycle.dispose);

    expect(await rejected, isNull);
    expect(await closing, isFalse);
    expect(requestCount, 0);
    expect(handle.killCalls, 1);
    expect(handle.closeCalls, 1);
  });

  test('result timeout returns null and makes later cleanup ambiguous',
      () async {
    final driver = _FakeDriver();
    final delay = _ControlledDelay();
    final lifecycle = TtsIsolateLifecycle(
      driver: driver,
      readyTimeout: const Duration(seconds: 30),
      resultTimeout: const Duration(seconds: 20),
      shutdownTimeout: const Duration(seconds: 5),
      delay: delay.call,
    );
    final lease = _lease();
    final handle = await _makeReady(driver, lifecycle, lease);

    final request = lifecycle.request(lease, 'blocked', '/blocked.wav');
    await Future<void>.delayed(Duration.zero);
    expect(delay.waits, hasLength(2));
    delay.fire(1);
    expect(await request, isNull);
    expect(lifecycle.snapshot.uncertain, isTrue);

    final rejected = lifecycle.request(lease, 'later', '/later.wav');
    await Future<void>.delayed(Duration.zero);
    final requestCount = handle.commands.whereType<TtsWorkerRequest>().length;
    final close = lease.close(lifecycle.dispose);
    expect(await rejected, isNull);
    final spawn = driver.spawns.single;
    spawn.emit(TtsWorkerShutdownComplete(spawn.epoch));
    expect(await close, isFalse);
    expect(requestCount, 1);
  });

  test('shutdown timeout poisons and still closes exact resources once',
      () async {
    final registry = TtsProcessOwnerRegistry();
    final lease = registry.tryAcquire()!;
    final driver = _FakeDriver();
    final delay = _ControlledDelay();
    final lifecycle = TtsIsolateLifecycle(
      driver: driver,
      readyTimeout: const Duration(seconds: 30),
      resultTimeout: const Duration(seconds: 20),
      shutdownTimeout: const Duration(seconds: 5),
      delay: delay.call,
    );
    final handle = await _makeReady(driver, lifecycle, lease);
    final spawn = driver.spawns.single;

    final closing = lease.close(lifecycle.dispose);
    await Future<void>.delayed(Duration.zero);
    expect(delay.waits, hasLength(2));
    expect(handle.commands.whereType<TtsWorkerShutdown>(), hasLength(1));
    delay.fire(1);

    expect(await closing, isFalse);
    expect(lifecycle.snapshot.uncertain, isTrue);
    expect(lifecycle.snapshot.poisoned, isTrue);
    expect(registry.poisoned, isTrue);
    expect(registry.tryAcquire(), isNull);
    expect(handle.killCalls, 1);
    expect(handle.closeCalls, 1);

    spawn.emit(TtsWorkerShutdownComplete(spawn.epoch));
    await Future<void>.delayed(Duration.zero);
    expect(handle.killCalls, 1);
    expect(handle.closeCalls, 1);
  });

  test('kill and event-close failures each retain poison after both attempts',
      () async {
    for (final failure in ['kill', 'close']) {
      final registry = TtsProcessOwnerRegistry();
      final lease = registry.tryAcquire()!;
      final driver = _FakeDriver();
      final lifecycle = _lifecycle(driver);
      final handle = await _makeReady(driver, lifecycle, lease);
      final spawn = driver.spawns.single;
      if (failure == 'kill') {
        handle.killError = StateError('kill failed');
      } else {
        handle.closeError = StateError('event close failed');
      }

      final closing = lease.close(lifecycle.dispose);
      spawn.emit(TtsWorkerShutdownComplete(spawn.epoch));
      expect(await closing, isFalse, reason: failure);
      expect(lifecycle.snapshot.poisoned, isTrue, reason: failure);
      expect(registry.poisoned, isTrue, reason: failure);
      expect(registry.tryAcquire(), isNull, reason: failure);
      expect(handle.killCalls, 1, reason: failure);
      expect(handle.closeCalls, 1, reason: failure);
    }
  });

  test('dispose is memoized and closes exact resources only once', () async {
    final driver = _FakeDriver();
    final lifecycle = _lifecycle(driver);
    final lease = _lease();
    final handle = await _makeReady(driver, lifecycle, lease);
    final spawn = driver.spawns.single;

    final first = lifecycle.dispose();
    final second = lifecycle.dispose();
    expect(identical(first, second), isTrue);
    expect(handle.commands.whereType<TtsWorkerShutdown>(), hasLength(1));
    spawn.emit(TtsWorkerShutdownComplete(spawn.epoch));
    expect(await Future.wait([first, second]), [true, true]);
    expect(handle.killCalls, 1);
    expect(handle.closeCalls, 1);
    expect(await lifecycle.dispose(), isTrue);
    expect(handle.killCalls, 1);
    expect(handle.closeCalls, 1);
  });
}
