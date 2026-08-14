import 'dart:async';

import './tts_process_owner.dart';

sealed class TtsWorkerEvent {
  const TtsWorkerEvent(this.epoch);

  final int epoch;
}

final class TtsWorkerSendPort extends TtsWorkerEvent {
  const TtsWorkerSendPort(super.epoch);
}

final class TtsWorkerReady extends TtsWorkerEvent {
  const TtsWorkerReady(super.epoch);
}

final class TtsWorkerResult extends TtsWorkerEvent {
  const TtsWorkerResult(super.epoch, this.id, this.outPath);

  final int id;
  final String? outPath;
}

final class TtsWorkerShutdownComplete extends TtsWorkerEvent {
  const TtsWorkerShutdownComplete(super.epoch);
}

sealed class TtsWorkerCommand {
  const TtsWorkerCommand(this.epoch);

  final int epoch;
}

final class TtsWorkerInit extends TtsWorkerCommand {
  const TtsWorkerInit(super.epoch, this.initPayload);

  final Object initPayload;

  Object get payload => initPayload;
}

final class TtsWorkerRequest extends TtsWorkerCommand {
  const TtsWorkerRequest(
    super.epoch,
    this.id,
    this.text,
    this.outPath,
  );

  final int id;
  final String text;
  final String outPath;
}

final class TtsWorkerShutdown extends TtsWorkerCommand {
  const TtsWorkerShutdown(super.epoch);
}

abstract interface class TtsWorkerHandle {
  void send(TtsWorkerCommand command);

  void kill();

  Future<void> closeEvents();
}

abstract interface class TtsWorkerDriver {
  Future<TtsWorkerHandle> spawn(
    int epoch,
    void Function(TtsWorkerEvent event) emit,
  );
}

final class TtsIsolateLifecycleSnapshot {
  const TtsIsolateLifecycleSnapshot({
    required this.epoch,
    required this.spawning,
    required this.ready,
    required this.closing,
    required this.pending,
    required this.uncertain,
    required this.poisoned,
    required this.disposed,
  });

  final int epoch;
  final bool spawning;
  final bool ready;
  final bool closing;
  final int pending;
  final bool uncertain;
  final bool poisoned;
  final bool disposed;

  bool get ambiguous => uncertain;
}

/// Pure lifecycle coordinator for one exact TTS worker generation.
///
/// The driver owns concrete Dart isolate/port objects. This coordinator owns
/// generation admission and never lets an old callback mutate a replacement.
/// Any request or shutdown timeout is conservative: it reports no result and
/// prevents an exact-clean shutdown receipt.
final class TtsIsolateLifecycle {
  TtsIsolateLifecycle({
    required TtsWorkerDriver driver,
    this.readyTimeout = const Duration(seconds: 30),
    this.resultTimeout = const Duration(seconds: 20),
    this.shutdownTimeout = const Duration(seconds: 5),
    Future<void> Function(Duration)? delay,
  })  : _driver = driver,
        _delay = delay ?? Future<void>.delayed;

  final TtsWorkerDriver _driver;
  final Duration readyTimeout;
  final Duration resultTimeout;
  final Duration shutdownTimeout;
  final Future<void> Function(Duration) _delay;

  int _nextEpoch = 0;
  int _nextRequestId = 0;
  _WorkerGeneration? _active;
  bool _poisoned = false;

  TtsIsolateLifecycleSnapshot get snapshot {
    final active = _active;
    return TtsIsolateLifecycleSnapshot(
      epoch: active?.epoch ?? _nextEpoch,
      spawning: active?.spawning ?? false,
      ready: active?.ready ?? false,
      closing: active?.closing ?? false,
      pending: active?.pending.length ?? 0,
      uncertain: active?.uncertain ?? false,
      poisoned: _poisoned,
      disposed: active == null || active.closing,
    );
  }

  Future<bool> ensureReady(TtsProcessLease lease, Object init) {
    if (!lease.admitsWork || _poisoned) return Future<bool>.value(false);

    final current = _active;
    if (current != null) {
      if (!identical(current.lease, lease) ||
          current.closing ||
          current.uncertain) {
        return Future<bool>.value(false);
      }
      if (current.ready) return Future<bool>.value(true);
      return current.readyResult.future;
    }

    final generation = _WorkerGeneration(
      epoch: ++_nextEpoch,
      lease: lease,
      init: init,
    );
    _active = generation;
    generation.spawning = true;
    unawaited(_spawn(generation));
    return generation.readyResult.future;
  }

  Future<void> _spawn(_WorkerGeneration generation) async {
    try {
      final handle = await _driver.spawn(
        generation.epoch,
        (event) {
          if (generation.spawning && generation.handle == null) {
            generation.earlyEvents.add(event);
          } else {
            _onEvent(generation, event);
          }
        },
      );
      generation.spawning = false;
      generation.handle = handle;
      if (!generation.spawnSettled.isCompleted) {
        generation.spawnSettled.complete();
      }
      if (!identical(_active, generation) || generation.closing) {
        generation.uncertain = true;
        _completeReady(generation, false);
        if (!identical(_active, generation)) {
          await _disposeLateHandle(generation, handle);
        }
        return;
      }
      final earlyEvents = List<TtsWorkerEvent>.of(generation.earlyEvents);
      generation.earlyEvents.clear();
      for (final event in earlyEvents) {
        _onEvent(generation, event);
      }
      unawaited(_readyDeadline(generation));
    } catch (_) {
      generation.spawning = false;
      if (!generation.spawnSettled.isCompleted) {
        generation.spawnSettled.complete();
      }
      _markUncertain(generation);
      _completeReady(generation, false);
    }
  }

  Future<void> _readyDeadline(_WorkerGeneration generation) async {
    await _delay(readyTimeout);
    if (!identical(_active, generation) || generation.readyResult.isCompleted) {
      return;
    }
    _markUncertain(generation);
    _completeReady(generation, false);
  }

  Future<void> _disposeLateHandle(
    _WorkerGeneration generation,
    TtsWorkerHandle handle,
  ) async {
    try {
      handle.kill();
    } catch (_) {
      generation.uncertain = true;
    }
    try {
      await handle.closeEvents();
    } catch (_) {
      generation.uncertain = true;
    }
    _completeReady(generation, false);
  }

  void _onEvent(_WorkerGeneration generation, TtsWorkerEvent event) {
    if (!identical(_active, generation) || event.epoch != generation.epoch) {
      return;
    }
    if (event is TtsWorkerShutdownComplete) {
      if (generation.closing && !generation.shutdownReturned.isCompleted) {
        generation.shutdownReturned.complete();
      }
      return;
    }
    if (generation.closing) return;
    if (generation.uncertain) return;

    if (event is TtsWorkerSendPort) {
      if (generation.handshake || generation.handle == null) return;
      generation.handshake = true;
      try {
        generation.handle!.send(
          TtsWorkerInit(generation.epoch, generation.init),
        );
      } catch (_) {
        _markUncertain(generation);
        _completeReady(generation, false);
      }
      return;
    }
    if (event is TtsWorkerReady) {
      if (!generation.handshake) {
        _markUncertain(generation);
        _completeReady(generation, false);
        return;
      }
      generation.ready = true;
      _completeReady(generation, true);
      return;
    }
    if (event is TtsWorkerResult) {
      generation.pending.remove(event.id)?.complete(event.outPath);
    }
  }

  Future<String?> request(
    TtsProcessLease lease,
    String text,
    String outPath,
  ) async {
    if (!lease.admitsWork || _poisoned) return null;
    final generation = _active;
    if (generation == null ||
        !identical(generation.lease, lease) ||
        !generation.ready ||
        generation.closing ||
        generation.uncertain) {
      return null;
    }

    final id = _nextRequestId++;
    final result = Completer<String?>();
    generation.pending[id] = result;
    try {
      generation.handle!.send(
        TtsWorkerRequest(generation.epoch, id, text, outPath),
      );
    } catch (_) {
      generation.pending.remove(id);
      _markUncertain(generation);
      return null;
    }

    unawaited(_requestDeadline(generation, id));
    return result.future;
  }

  Future<void> _requestDeadline(_WorkerGeneration generation, int id) async {
    await _delay(resultTimeout);
    final pending = generation.pending.remove(id);
    if (pending == null) return;
    _markUncertain(generation);
    if (!pending.isCompleted) pending.complete(null);
    for (final other in generation.pending.values) {
      if (!other.isCompleted) other.complete(null);
    }
    generation.pending.clear();
  }

  /// Stop the exact generation and return whether all entered work quiesced.
  ///
  /// A spawn future that never returns keeps this future pending. That is the
  /// honest result when the existence of a worker cannot yet be determined.
  Future<bool> dispose() {
    final generation = _active;
    if (generation == null) return Future<bool>.value(!_poisoned);
    final existing = generation.disposeResult;
    if (existing != null) return existing;

    final completed = Completer<bool>();
    generation.disposeResult = completed.future;
    generation.closing = true;
    if (generation.spawning) generation.uncertain = true;
    generation.lease.revoke();
    _completeReady(generation, false);
    for (final pending in generation.pending.values) {
      if (!pending.isCompleted) pending.complete(null);
    }
    generation.pending.clear();
    if (!generation.spawning &&
        generation.handle != null &&
        !generation.uncertain) {
      try {
        generation.handle!.send(TtsWorkerShutdown(generation.epoch));
        generation.shutdownSent = true;
      } catch (_) {
        generation.uncertain = true;
      }
    }
    unawaited(_runDispose(generation, completed));
    return completed.future;
  }

  Future<void> _runDispose(
    _WorkerGeneration generation,
    Completer<bool> completed,
  ) async {
    await generation.spawnSettled.future;

    final handle = generation.handle;
    if (handle == null) {
      _finishDispose(generation, completed, false);
      return;
    }

    var exact = !generation.uncertain;
    if (exact) {
      try {
        if (!generation.shutdownSent) {
          handle.send(TtsWorkerShutdown(generation.epoch));
          generation.shutdownSent = true;
        }
        exact = await _waitForShutdown(generation);
        if (!exact) generation.uncertain = true;
      } catch (_) {
        exact = false;
        generation.uncertain = true;
      }
    }
    try {
      handle.kill();
    } catch (_) {
      exact = false;
      generation.uncertain = true;
    }
    try {
      await handle.closeEvents();
    } catch (_) {
      exact = false;
      generation.uncertain = true;
    }
    _finishDispose(generation, completed, exact && !generation.uncertain);
  }

  Future<bool> _waitForShutdown(_WorkerGeneration generation) async {
    final timeout = _delay(shutdownTimeout).then((_) => false);
    final returned = generation.shutdownReturned.future.then((_) => true);
    return Future.any<bool>(<Future<bool>>[returned, timeout]);
  }

  void _finishDispose(
    _WorkerGeneration generation,
    Completer<bool> completed,
    bool exact,
  ) {
    if (exact && identical(_active, generation)) {
      _active = null;
      completed.complete(true);
      return;
    }
    if (identical(_active, generation)) _poisoned = true;
    completed.complete(false);
  }

  void _markUncertain(_WorkerGeneration generation) {
    generation.uncertain = true;
  }

  void _completeReady(_WorkerGeneration generation, bool ready) {
    if (!generation.readyResult.isCompleted) {
      generation.readyResult.complete(ready);
    }
  }
}

final class _WorkerGeneration {
  _WorkerGeneration({
    required this.epoch,
    required this.lease,
    required this.init,
  });

  final int epoch;
  final TtsProcessLease lease;
  final Object init;
  final Completer<bool> readyResult = Completer<bool>();
  final Completer<void> spawnSettled = Completer<void>();
  final Completer<void> shutdownReturned = Completer<void>();
  final Map<int, Completer<String?>> pending = {};
  final List<TtsWorkerEvent> earlyEvents = <TtsWorkerEvent>[];
  TtsWorkerHandle? handle;
  Future<bool>? disposeResult;
  bool spawning = false;
  bool handshake = false;
  bool ready = false;
  bool closing = false;
  bool shutdownSent = false;
  bool uncertain = false;
}
