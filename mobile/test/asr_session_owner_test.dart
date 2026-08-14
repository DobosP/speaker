import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/asr_isolate.dart';

Future<void> _settle([int turns = 6]) async {
  for (var i = 0; i < turns; i++) {
    await Future<void>.delayed(Duration.zero);
  }
}

final class _ManualTimer implements AsrTimerHandle {
  _ManualTimer(this.callback);
  final void Function() callback;
  bool canceled = false;

  void fire({bool ignoreCancel = false}) {
    if (ignoreCancel || !canceled) callback();
  }

  @override
  void cancel() => canceled = true;
}

final class _TimerBank {
  final List<_ManualTimer> timers = <_ManualTimer>[];
  bool throwOnCreate = false;
  bool fireSynchronously = false;
  int? fireSynchronouslyOnCall;

  AsrTimerHandle create(Duration duration, void Function() callback) {
    if (throwOnCreate) throw StateError('hostile_timer_factory');
    final timer = _ManualTimer(callback);
    timers.add(timer);
    if (fireSynchronously || fireSynchronouslyOnCall == timers.length) {
      callback();
    }
    return timer;
  }
}

final class _Harness {
  _Harness({
    Completer<AsrWorkerStartup>? heldStartup,
    Completer<void>? heldReady,
    _TimerBank? timerBank,
    Duration closeTimeout = const Duration(milliseconds: 20),
    int initialOrdinal = 0,
    void Function(AsrWorkerCommand command)? onSend,
    Future<bool> Function()? onClose,
  })  : timers = timerBank ?? _TimerBank(),
        startupCompleter = heldStartup,
        readyCompleter = heldReady ?? Completer<void>() {
    if (heldReady == null) readyCompleter.complete();
    transport = AsrWorkerTransport(
      send: (command) {
        commands.add(command);
        onSend?.call(command);
      },
      close: () async {
        closeCalls++;
        return onClose == null ? true : await onClose();
      },
    );
    startup = AsrWorkerStartup(
      ready: readyCompleter.future,
      transport: transport,
    );
    service = AsrService.forTesting(
      startupFactory: (onEvent, launchFence) {
        starts++;
        sink = onEvent;
        this.launchFence = launchFence;
        return startupCompleter?.future ?? Future.value(startup);
      },
      timerFactory: timers.create,
      closeTimeout: closeTimeout,
      initialOrdinal: initialOrdinal,
    );
  }

  final _TimerBank timers;
  final Completer<AsrWorkerStartup>? startupCompleter;
  final Completer<void> readyCompleter;
  final List<AsrWorkerCommand> commands = <AsrWorkerCommand>[];
  late final AsrWorkerTransport transport;
  late final AsrWorkerStartup startup;
  late final AsrService service;
  AsrWorkerEventSink? sink;
  AsrWorkerLaunchFence? launchFence;
  int starts = 0;
  int closeCalls = 0;

  Iterable<T> commandsOf<T extends AsrWorkerCommand>() =>
      commands.whereType<T>();

  void event(AsrWorkerEvent event) => sink!(event);
}

void main() {
  group('AsrService exact session owner', () {
    test('begin is synchronous and readiness spans startup plus reset ACK',
        () async {
      final startup = Completer<AsrWorkerStartup>();
      final harness = _Harness(heldStartup: startup);
      final session = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );

      expect(session.ordinal, 1);
      expect(harness.starts, 0);
      expect(harness.commands, isEmpty);
      var becameReady = false;
      session.ready.then((_) => becameReady = true);

      await _settle();
      expect(harness.starts, 1);
      startup.complete(harness.startup);
      await _settle();
      expect(harness.commandsOf<AsrWorkerReset>().single.ordinal, 1);
      expect(becameReady, isFalse);

      harness.event(const AsrWorkerResetAck(1));
      await session.ready;
      expect(becameReady, isTrue);
      expect(harness.timers.timers.single.canceled, isTrue);
    });

    test('feed waits for exact ACK and enforces four exact sequence credits',
        () async {
      final harness = _Harness();
      final session = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await _settle();
      final bytes = Uint8List.fromList(<int>[1, 2]);
      expect(harness.service.feed(session, bytes), isFalse);
      harness.event(const AsrWorkerResetAck(1));
      await session.ready;

      for (var i = 0; i < AsrService.maxOutstandingAudioChunks; i++) {
        expect(harness.service.feed(session, bytes), isTrue);
      }
      expect(harness.service.feed(session, bytes), isFalse);
      final audio = harness.commandsOf<AsrWorkerAudio>().toList();
      expect(audio.map((item) => item.sequence), <int>[1, 2, 3, 4]);

      harness.event(const AsrWorkerAudioAck(999, 1));
      harness.event(const AsrWorkerAudioAck(1, 999));
      expect(harness.service.feed(session, bytes), isFalse);
      harness.event(const AsrWorkerAudioAck(1, 1));
      expect(harness.service.feed(session, bytes), isTrue);
      harness.event(const AsrWorkerAudioAck(1, 1));
      expect(harness.service.feed(session, bytes), isFalse);
    });

    test('feed copies only after validation and never aliases caller PCM',
        () async {
      final harness = _Harness();
      final session = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await _settle();
      harness.event(const AsrWorkerResetAck(1));
      await session.ready;
      final bytes = Uint8List.fromList(<int>[1, 2, 3, 4]);
      expect(harness.service.feed(session, bytes), isTrue);
      bytes[0] = 99;
      expect(harness.commandsOf<AsrWorkerAudio>().single.bytes[0], 1);
    });

    test('empty odd and oversized PCM are rejected without sends', () async {
      final harness = _Harness();
      final session = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await _settle();
      harness.event(const AsrWorkerResetAck(1));
      await session.ready;
      final before = harness.commands.length;
      expect(harness.service.feed(session, Uint8List(0)), isFalse);
      expect(harness.service.feed(session, Uint8List(3)), isFalse);
      expect(
        harness.service.feed(
          session,
          Uint8List(AsrService.maxPcmBytesPerChunk + 2),
        ),
        isFalse,
      );
      expect(harness.commands.length, before);
    });

    test('end releases callbacks immediately and old output stays inert',
        () async {
      final harness = _Harness();
      final seen = <String>[];
      final session = harness.service.beginSession(
        onPartial: (text) => seen.add('p:$text'),
        onEndpoint: (text) => seen.add('e:$text'),
      );
      await _settle();
      harness.event(const AsrWorkerResetAck(1));
      await session.ready;
      expect(harness.service.endSession(session), isTrue);
      expect(harness.service.endSession(session), isTrue);
      var cleanupSettled = false;
      session.cleanup.then((_) => cleanupSettled = true);
      harness.event(const AsrWorkerPartial(1, 'old'));
      harness.event(const AsrWorkerEndpoint(1, 'old'));
      expect(seen, isEmpty);
      expect(harness.commandsOf<AsrWorkerEnd>().length, 1);
      expect(cleanupSettled, isFalse);
      harness.event(const AsrWorkerEndAck(1, released: true));
      expect(await session.cleanup, isTrue);
    });

    test('stale end reset ACK output and audio ACK never touch replacement',
        () async {
      final harness = _Harness();
      final seen = <String>[];
      final first = harness.service.beginSession(
        onPartial: (text) => seen.add('a:$text'),
        onEndpoint: (_) {},
      );
      await _settle();
      harness.event(const AsrWorkerResetAck(1));
      await first.ready;
      final second = harness.service.beginSession(
        onPartial: (text) => seen.add('b:$text'),
        onEndpoint: (_) {},
      );
      await _settle();
      expect(
          harness.commandsOf<AsrWorkerReset>().map((e) => e.ordinal), <int>[1]);
      expect(harness.service.endSession(first), isTrue);
      harness.event(const AsrWorkerResetAck(1));
      harness.event(const AsrWorkerPartial(1, 'old'));
      harness.event(const AsrWorkerAudioAck(1, 1));
      expect(harness.service.feed(second, Uint8List.fromList(<int>[0, 0])),
          isFalse);
      harness.event(const AsrWorkerEndAck(1, released: true));
      expect(await first.cleanup, isTrue);
      await _settle();
      expect(harness.commandsOf<AsrWorkerReset>().last.ordinal, 2);
      harness.event(const AsrWorkerResetAck(2));
      await second.ready;
      harness.event(const AsrWorkerPartial(1, 'old-again'));
      harness.event(const AsrWorkerPartial(2, 'new'));
      expect(seen, <String>['b:new']);
      final bytes = Uint8List.fromList(<int>[0, 0]);
      for (var i = 0; i < AsrService.maxOutstandingAudioChunks; i++) {
        expect(harness.service.feed(second, bytes), isTrue);
      }
      harness.event(const AsrWorkerAudioAck(1, 1));
      expect(harness.service.feed(second, bytes), isFalse);
      harness.event(const AsrWorkerAudioAck(2, 1));
      expect(harness.service.feed(second, bytes), isTrue);
    });

    test('cleanup timeout freezes false and late End ACK cannot upgrade it',
        () async {
      final harness = _Harness();
      final session = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await _settle();
      harness.event(const AsrWorkerResetAck(1));
      await session.ready;
      expect(harness.service.endSession(session), isTrue);
      harness.timers.timers.last.fire();
      expect(await session.cleanup, isFalse);
      harness.event(const AsrWorkerEndAck(1, released: true));
      expect(await session.cleanup, isFalse);
      expect(harness.service.endSession(session), isFalse);
    });

    test('synchronous cleanup deadline freezes false and denies successor',
        () async {
      final timers = _TimerBank()..fireSynchronouslyOnCall = 2;
      final harness = _Harness(timerBank: timers);
      final first = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await _settle();
      harness.event(const AsrWorkerResetAck(1));
      await first.ready;
      expect(harness.service.endSession(first), isFalse);
      expect(await first.cleanup, isFalse);
      expect(timers.timers[1].canceled, isTrue);
      timers.timers[1].fire(ignoreCancel: true);
      harness.event(const AsrWorkerEndAck(1, released: true));
      expect(await first.cleanup, isFalse);
      expect(harness.service.endSession(first), isFalse);

      final second = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await expectLater(second.ready, throwsA(isA<StateError>()));
      expect(
          harness.commandsOf<AsrWorkerReset>().map((e) => e.ordinal), <int>[1]);
    });

    test('failed predecessor cleanup denies successor reset immediately',
        () async {
      final harness = _Harness();
      final first = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await _settle();
      harness.event(const AsrWorkerResetAck(1));
      await first.ready;
      final second = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      final secondFailed =
          expectLater(second.ready, throwsA(isA<StateError>()));
      harness.event(const AsrWorkerEndAck(1, released: false));
      expect(await first.cleanup, isFalse);
      await secondFailed;
      expect(
          harness.commandsOf<AsrWorkerReset>().map((e) => e.ordinal), <int>[1]);
      expect(await second.cleanup, isFalse);
    });

    test('foreign token is rejected without touching active session', () async {
      final first = _Harness();
      final second = _Harness();
      final foreign = first.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      final local = second.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await _settle();
      second.event(const AsrWorkerResetAck(1));
      await local.ready;
      expect(second.service.endSession(foreign), isFalse);
      expect(second.service.feed(foreign, Uint8List.fromList(<int>[0, 0])),
          isFalse);
      expect(
          second.service.feed(local, Uint8List.fromList(<int>[0, 0])), isTrue);
    });

    test('admission timer spans held startup and makes late startup inert',
        () async {
      final startup = Completer<AsrWorkerStartup>();
      final harness = _Harness(heldStartup: startup);
      var calls = 0;
      final session = harness.service.beginSession(
        onPartial: (_) => calls++,
        onEndpoint: (_) => calls++,
      );
      await _settle();
      expect(harness.starts, 1);
      final failed =
          expectLater(session.ready, throwsA(isA<TimeoutException>()));
      harness.timers.timers.single.fire();
      await failed;
      startup.complete(harness.startup);
      await _settle();
      expect(harness.commandsOf<AsrWorkerReset>(), isEmpty);
      harness.event(const AsrWorkerPartial(1, 'late'));
      expect(calls, 0);
      expect(harness.service.endSession(session), isTrue);
    });

    test('timer factory throw and synchronous callback both fail closed',
        () async {
      final throwingTimers = _TimerBank()..throwOnCreate = true;
      final throwing = _Harness(timerBank: throwingTimers);
      final first = throwing.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await expectLater(first.ready, throwsA(isA<StateError>()));
      expect(throwing.starts, 0);

      final synchronousTimers = _TimerBank()..fireSynchronously = true;
      final synchronous = _Harness(timerBank: synchronousTimers);
      final second = synchronous.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await expectLater(second.ready, throwsA(isA<TimeoutException>()));
      expect(synchronous.starts, 0);
      expect(synchronousTimers.timers.single.canceled, isTrue);
    });

    test('reset failure releases exact callbacks and a higher ordinal retries',
        () async {
      final harness = _Harness();
      final seen = <String>[];
      final first = harness.service.beginSession(
        onPartial: seen.add,
        onEndpoint: seen.add,
      );
      await _settle();
      final failed = expectLater(first.ready, throwsA(isA<StateError>()));
      harness.event(const AsrWorkerSessionFailure(
        1,
        'bounded',
        sessionReleased: true,
        workerHealthy: true,
      ));
      await failed;
      expect(await first.cleanup, isTrue);
      final second = harness.service.beginSession(
        onPartial: seen.add,
        onEndpoint: seen.add,
      );
      await _settle();
      expect(harness.commandsOf<AsrWorkerReset>().last.ordinal, 2);
      harness.event(const AsrWorkerResetAck(2));
      await second.ready;
      harness.event(const AsrWorkerPartial(1, 'old'));
      harness.event(const AsrWorkerPartial(2, 'new'));
      expect(seen, <String>['new']);
    });

    test('stale poisoned failure fails a waiting successor immediately',
        () async {
      final harness = _Harness();
      final first = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await _settle();
      harness.event(const AsrWorkerResetAck(1));
      await first.ready;
      final second = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      final failed = expectLater(second.ready, throwsA(isA<StateError>()));
      harness.event(const AsrWorkerSessionFailure(
        1,
        'poisoned',
        sessionReleased: false,
        workerHealthy: false,
      ));
      await failed;
      expect(await first.cleanup, isFalse);
      expect(await second.cleanup, isFalse);
      expect(
          harness.commandsOf<AsrWorkerReset>().map((e) => e.ordinal), <int>[1]);
    });

    test('clean startup factory failure is retryable at a higher ordinal',
        () async {
      final timers = _TimerBank();
      final commands = <AsrWorkerCommand>[];
      final transport = AsrWorkerTransport(
        send: commands.add,
        close: () async => true,
      );
      final startup = AsrWorkerStartup(
        ready: Future<void>.value(),
        transport: transport,
      );
      var attempts = 0;
      AsrWorkerEventSink? sink;
      final service = AsrService.forTesting(
        startupFactory: (onEvent, _) {
          sink = onEvent;
          if (attempts++ == 0) {
            return Future<AsrWorkerStartup>.error(
              StateError('clean_factory_failure'),
            );
          }
          return Future<AsrWorkerStartup>.value(startup);
        },
        timerFactory: timers.create,
      );
      final first = service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await expectLater(first.ready, throwsA(isA<StateError>()));
      expect(await first.cleanup, isTrue);
      final second = service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await _settle();
      expect(attempts, 2);
      expect(commands.whereType<AsrWorkerReset>().single.ordinal, 2);
      sink!(const AsrWorkerResetAck(2));
      await second.ready;
    });

    test('typed uncertain startup freezes false and is never retried',
        () async {
      final timers = _TimerBank();
      var attempts = 0;
      final service = AsrService.forTesting(
        startupFactory: (_, __) {
          attempts++;
          return Future<AsrWorkerStartup>.error(
            const AsrWorkerStartupUncertain(),
          );
        },
        timerFactory: timers.create,
      );
      final first = service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await expectLater(first.ready, throwsA(isA<StateError>()));
      expect(await first.cleanup, isFalse);
      expect(service.endSession(first), isFalse);

      final second = service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await expectLater(second.ready, throwsA(isA<StateError>()));
      expect(attempts, 1);
      expect(await service.close(), isFalse);
      expect(identical(service.close(), service.close()), isTrue);
    });

    test('oversized UTF-8 output fails exact session without delivery',
        () async {
      final harness = _Harness();
      final seen = <String>[];
      final session = harness.service.beginSession(
        onPartial: seen.add,
        onEndpoint: seen.add,
      );
      await _settle();
      harness.event(const AsrWorkerResetAck(1));
      await session.ready;
      harness.event(
        AsrWorkerPartial(1, 'é' * (AsrService.maxResultUtf8Bytes ~/ 2 + 1)),
      );
      expect(seen, isEmpty);
      expect(harness.service.feed(session, Uint8List.fromList(<int>[0, 0])),
          isFalse);
      expect(harness.service.endSession(session), isTrue);
    });

    test('throwing and reentrant callbacks are contained and revoke authority',
        () async {
      final throwing = _Harness();
      final first = throwing.service.beginSession(
        onPartial: (_) => throw StateError('ui_callback'),
        onEndpoint: (_) {},
      );
      await _settle();
      throwing.event(const AsrWorkerResetAck(1));
      await first.ready;
      expect(
        () => throwing.event(const AsrWorkerPartial(1, 'x')),
        returnsNormally,
      );
      expect(throwing.service.endSession(first), isTrue);

      final reentrant = _Harness();
      late AsrSession second;
      second = reentrant.service.beginSession(
        onPartial: (_) => reentrant.service.endSession(second),
        onEndpoint: (_) {},
      );
      await _settle();
      reentrant.event(const AsrWorkerResetAck(1));
      await second.ready;
      reentrant.event(const AsrWorkerPartial(1, 'x'));
      expect(reentrant.service.feed(second, Uint8List.fromList(<int>[0, 0])),
          isFalse);
    });

    test('send uncertainty is sticky and end returns false', () async {
      var resetSends = 0;
      final harness = _Harness(onSend: (command) {
        if (command is AsrWorkerReset && resetSends++ == 0) {
          throw StateError('uncertain_send');
        }
      });
      final first = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await expectLater(first.ready, throwsA(isA<StateError>()));
      expect(harness.service.endSession(first), isFalse);
      final second = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await expectLater(second.ready, throwsA(isA<StateError>()));
      expect(resetSends, 1);
    });

    test('safe ordinal exhaustion is synchronous and starts no work', () {
      final harness = _Harness(initialOrdinal: AsrSession.maxSafeOrdinal);
      expect(
        () => harness.service.beginSession(
          onPartial: (_) {},
          onEndpoint: (_) {},
        ),
        throwsA(isA<StateError>()),
      );
      expect(harness.starts, 0);
    });
  });

  group('AsrService hard lifecycle', () {
    test('close is prepublished and identical under transport reentry',
        () async {
      late _Harness harness;
      Future<bool>? reentrant;
      final seen = <String>[];
      harness = _Harness(onClose: () async {
        reentrant = harness.service.close();
        return true;
      });
      final session = harness.service.beginSession(
        onPartial: seen.add,
        onEndpoint: (_) {},
      );
      await _settle();
      harness.event(const AsrWorkerResetAck(1));
      await session.ready;
      final first = harness.service.close();
      final second = harness.service.close();
      expect(identical(first, second), isTrue);
      expect(harness.launchFence!.isCurrent, isFalse);
      harness.event(const AsrWorkerPartial(1, 'after-close'));
      expect(seen, isEmpty);
      harness.event(const AsrWorkerEndAck(1, released: true));
      expect(await first, isTrue);
      expect(identical(first, reentrant), isTrue);
      expect(harness.closeCalls, 1);
    });

    test('startup is prepublished before synchronous close reentry', () async {
      final timers = _TimerBank();
      late AsrService service;
      late Future<bool> reentrantClose;
      var closeCalls = 0;
      var fenceWasRevoked = false;
      final startup = AsrWorkerStartup(
        ready: Future<void>.value(),
        transport: AsrWorkerTransport(
          send: (_) {},
          close: () async {
            closeCalls++;
            return false;
          },
        ),
      );
      service = AsrService.forTesting(
        startupFactory: (_, launchFence) {
          reentrantClose = service.close();
          fenceWasRevoked = !launchFence.isCurrent;
          return Future<AsrWorkerStartup>.value(startup);
        },
        timerFactory: timers.create,
      );
      final session = service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      final readyFailure =
          expectLater(session.ready, throwsA(isA<StateError>()));

      await _settle();
      await readyFailure;
      expect(fenceWasRevoked, isTrue);
      expect(await session.cleanup, isTrue);
      expect(await reentrantClose, isFalse);
      expect(identical(reentrantClose, service.close()), isTrue);
      expect(closeCalls, 1);
    });

    test('transport close itself prepublishes one exact future', () async {
      late AsrWorkerTransport transport;
      Future<bool>? reentrant;
      var calls = 0;
      transport = AsrWorkerTransport(
        send: (_) {},
        close: () async {
          calls++;
          reentrant = transport.close();
          return true;
        },
      );
      final first = transport.close();
      expect(identical(first, transport.close()), isTrue);
      expect(await first, isTrue);
      expect(identical(first, reentrant), isTrue);
      expect(calls, 1);
    });

    test('close failure is value-only retained and never retried', () async {
      final harness = _Harness(onClose: () async {
        throw StateError('close_failed');
      });
      final session = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await _settle();
      harness.event(const AsrWorkerResetAck(1));
      await session.ready;
      final first = harness.service.close();
      harness.event(const AsrWorkerEndAck(1, released: true));
      expect(await first, isFalse);
      final second = harness.service.close();
      expect(identical(first, second), isTrue);
      expect(await second, isFalse);
      expect(harness.closeCalls, 1);
      expect(harness.service.closeFailureCode, 'asr_close_failed');
    });

    test('close revokes held pre-spawn launch before construction', () async {
      final heldModel = Completer<void>();
      final timers = _TimerBank();
      AsrWorkerLaunchFence? fence;
      var constructed = 0;
      final service = AsrService.forTesting(
        startupFactory: (_, launchFence) async {
          fence = launchFence;
          await heldModel.future;
          if (!launchFence.isCurrent) {
            throw StateError('revoked_before_spawn');
          }
          constructed++;
          return AsrWorkerStartup(
            ready: Future<void>.value(),
            transport: AsrWorkerTransport(
              send: (_) {},
              close: () async => true,
            ),
          );
        },
        timerFactory: timers.create,
      );
      final session = service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      final readyFailure =
          expectLater(session.ready, throwsA(isA<StateError>()));
      await _settle();
      expect(fence!.isCurrent, isTrue);

      final close = service.close();
      expect(fence!.isCurrent, isFalse);
      heldModel.complete();

      await readyFailure;
      expect(await session.cleanup, isTrue);
      expect(await close, isTrue);
      expect(constructed, 0);
      expect(identical(close, service.close()), isTrue);
    });

    test('close bounds a held spawn and retires its late exact startup once',
        () async {
      final heldSpawn = Completer<void>();
      final timers = _TimerBank();
      AsrWorkerLaunchFence? fence;
      var constructed = 0;
      var closeCalls = 0;
      final service = AsrService.forTesting(
        startupFactory: (_, launchFence) async {
          fence = launchFence;
          await heldSpawn.future;
          constructed++;
          final startup = AsrWorkerStartup(
            ready: Future<void>.value(),
            transport: AsrWorkerTransport(
              send: (_) {},
              close: () async {
                closeCalls++;
                return true;
              },
            ),
          );
          if (!launchFence.isCurrent) await startup.close();
          return startup;
        },
        timerFactory: timers.create,
      );
      final session = service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      final readyFailure =
          expectLater(session.ready, throwsA(isA<StateError>()));
      await _settle();
      expect(fence!.isCurrent, isTrue);

      final close = service.close();
      expect(fence!.isCurrent, isFalse);
      timers.timers.last.fire();
      expect(await close, isFalse);
      await readyFailure;
      expect(await session.cleanup, isTrue);

      heldSpawn.complete();
      await _settle(12);
      expect(constructed, 1);
      expect(closeCalls, 1);
      expect(await service.close(), isFalse);
    });

    test('close during held startup is bounded and closes late startup once',
        () async {
      final held = Completer<AsrWorkerStartup>();
      final harness = _Harness(heldStartup: held);
      final session = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      await _settle();
      expect(harness.starts, 1);
      final readyFailure =
          expectLater(session.ready, throwsA(isA<StateError>()));
      final close = harness.service.close();
      await readyFailure;
      harness.timers.timers.last.fire();
      expect(await close, isFalse);
      held.complete(harness.startup);
      await _settle(12);
      expect(harness.closeCalls, 1);
      expect(identical(close, harness.service.close()), isTrue);
    });

    test('close during held handshake does not await session readiness',
        () async {
      final heldReady = Completer<void>();
      final harness = _Harness(heldReady: heldReady);
      final session = harness.service.beginSession(
        onPartial: (_) {},
        onEndpoint: (_) {},
      );
      final readyFailure =
          expectLater(session.ready, throwsA(isA<StateError>()));
      await _settle();
      expect(await harness.service.close(), isTrue);
      await readyFailure;
      expect(harness.closeCalls, 1);
      heldReady.complete();
      await _settle();
      expect(harness.commandsOf<AsrWorkerReset>(), isEmpty);
    });

    test('synchronous close deadline callback freezes false', () async {
      final timers = _TimerBank()..fireSynchronously = true;
      final harness = _Harness(timerBank: timers);
      final close = harness.service.close();
      expect(await close, isFalse);
      expect(identical(close, harness.service.close()), isTrue);
    });
  });

  group('AsrWorkerSessionGate', () {
    test('only higher reset and exact increasing audio can advance', () {
      final gate = AsrWorkerSessionGate();
      expect(gate.canReset(1), isTrue);
      gate.commitReset(1);
      expect(gate.acceptAudio(1, 1), isTrue);
      expect(gate.acceptAudio(1, 1), isFalse);
      expect(gate.acceptAudio(1, 3), isFalse);
      expect(gate.acceptAudio(1, 2), isTrue);
      expect(gate.acceptAudio(0, 2), isFalse);
      expect(gate.end(0), isFalse);
      expect(gate.currentOrdinal, 1);
      expect(gate.end(1), isTrue);
      expect(gate.canReset(1), isFalse);
      expect(gate.canReset(2), isTrue);
    });
  });

  group('AsrWorkerCore native wrapper ordering', () {
    test('successor ACK follows predecessor stream free', () {
      final log = <String>[];
      final recognizer = _FakeRecognizer(log);
      final events = <AsrWorkerEvent>[];
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: (event) {
          log.add('emit:${event.runtimeType}');
          events.add(event);
        },
        terminate: () => log.add('terminate'),
      );
      core.handle(const AsrWorkerReset(1));
      core.handle(const AsrWorkerReset(2));
      expect(
        log,
        containsAllInOrder(<String>[
          'create:1',
          'emit:AsrWorkerResetAck',
          'stream-free:1',
          'emit:AsrWorkerEndAck',
          'create:2',
          'emit:AsrWorkerResetAck',
        ]),
      );
      expect(events.whereType<AsrWorkerResetAck>().map((e) => e.ordinal),
          <int>[1, 2]);
    });

    test('predecessor free failure poisons and withholds successor ACK', () {
      final log = <String>[];
      final recognizer = _FakeRecognizer(log);
      final events = <AsrWorkerEvent>[];
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: events.add,
        terminate: () {},
      );
      core.handle(const AsrWorkerReset(1));
      recognizer.streams.first.throwOnFree = true;
      core.handle(const AsrWorkerReset(2));
      expect(core.gate.isPoisoned, isTrue);
      expect(events.whereType<AsrWorkerResetAck>().map((e) => e.ordinal),
          <int>[1]);
      expect(recognizer.streams, hasLength(1));
      expect(
        events.whereType<AsrWorkerEndAck>().single.released,
        isFalse,
      );
      final failure = events.whereType<AsrWorkerSessionFailure>().single;
      expect(failure.sessionReleased, isTrue);
      expect(failure.workerHealthy, isFalse);
      expect(core.retainedUncertainStreamCount, 1);
      core.handle(const AsrWorkerShutdown(1));
      expect(recognizer.streams.single.freeCalls, 1);
      expect(recognizer.freeCalls, 0);
    });

    test('stale end and audio never touch replacement stream', () {
      final recognizer = _FakeRecognizer(<String>[]);
      final events = <AsrWorkerEvent>[];
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: events.add,
        terminate: () {},
      );
      core.handle(const AsrWorkerReset(1));
      core.handle(const AsrWorkerReset(2));
      final replacement = recognizer.streams.last;
      core.handle(const AsrWorkerEnd(1));
      core.handle(AsrWorkerAudio(1, 1, Uint8List.fromList(<int>[0, 0])));
      expect(replacement.freeCalls, 0);
      expect(replacement.acceptCalls, 0);
      expect(core.gate.currentOrdinal, 2);
    });

    test('partial or endpoint output precedes exact audio ACK', () {
      final log = <String>[];
      final recognizer = _FakeRecognizer(log)..text = 'hello';
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: (event) => log.add('emit:${event.runtimeType}'),
        terminate: () {},
      );
      core.handle(const AsrWorkerReset(1));
      log.clear();
      core.handle(AsrWorkerAudio(1, 1, Uint8List.fromList(<int>[0, 0])));
      expect(
        log,
        containsAllInOrder(<String>[
          'emit:AsrWorkerPartial',
          'emit:AsrWorkerAudioAck',
        ]),
      );

      recognizer.endpoint = true;
      recognizer.text = 'done';
      log.clear();
      core.handle(AsrWorkerAudio(1, 2, Uint8List.fromList(<int>[0, 0])));
      expect(
        log,
        containsAllInOrder(<String>[
          'recognizer-reset',
          'emit:AsrWorkerEndpoint',
          'emit:AsrWorkerAudioAck',
        ]),
      );
    });

    test('worker repeats PCM and UTF-8 bounds and does not ACK failures', () {
      final recognizer = _FakeRecognizer(<String>[]);
      final events = <AsrWorkerEvent>[];
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: events.add,
        terminate: () {},
      );
      core.handle(const AsrWorkerReset(1));
      core.handle(AsrWorkerAudio(1, 1, Uint8List(3)));
      expect(core.gate.isPoisoned, isTrue);
      expect(events.whereType<AsrWorkerAudioAck>(), isEmpty);

      final recognizer2 = _FakeRecognizer(<String>[])
        ..text = 'é' * (AsrService.maxResultUtf8Bytes ~/ 2 + 1);
      final events2 = <AsrWorkerEvent>[];
      final core2 = AsrWorkerCore(
        recognizer: recognizer2,
        emit: events2.add,
        terminate: () {},
      );
      core2.handle(const AsrWorkerReset(1));
      core2.handle(AsrWorkerAudio(1, 1, Uint8List.fromList(<int>[0, 0])));
      expect(core2.gate.isPoisoned, isTrue);
      expect(events2.whereType<AsrWorkerAudioAck>(), isEmpty);
      expect(events2.whereType<AsrWorkerPartial>(), isEmpty);
    });

    test('decode loop bound poisons and frees exact stream', () {
      final recognizer = _FakeRecognizer(<String>[])..alwaysReady = true;
      final events = <AsrWorkerEvent>[];
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: events.add,
        terminate: () {},
      );
      core.handle(const AsrWorkerReset(1));
      core.handle(AsrWorkerAudio(1, 1, Uint8List.fromList(<int>[0, 0])));
      expect(recognizer.decodeCalls, AsrService.maxDecodeStepsPerChunk);
      expect(recognizer.streams.single.freeCalls, 1);
      expect(core.gate.isPoisoned, isTrue);
      expect(events.whereType<AsrWorkerAudioAck>(), isEmpty);
    });

    test('reset ACK emission failure frees replacement and poisons', () {
      final recognizer = _FakeRecognizer(<String>[]);
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: (_) => throw StateError('send_failed'),
        terminate: () {},
      );
      core.handle(const AsrWorkerReset(1));
      expect(core.gate.isPoisoned, isTrue);
      expect(recognizer.streams.single.freeCalls, 1);
    });

    test('exact end ACK follows stream free and stale end is inert', () {
      final log = <String>[];
      final recognizer = _FakeRecognizer(log);
      final events = <AsrWorkerEvent>[];
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: (event) {
          log.add('emit:${event.runtimeType}');
          events.add(event);
        },
        terminate: () {},
      );
      core.handle(const AsrWorkerReset(1));
      log.clear();
      core.handle(const AsrWorkerEnd(1));
      core.handle(const AsrWorkerEnd(1));
      expect(
        log,
        <String>['stream-free:1', 'emit:AsrWorkerEndAck'],
      );
      expect(events.whereType<AsrWorkerEndAck>().single.released, isTrue);
      expect(recognizer.streams.single.freeCalls, 1);
    });

    test('end free failure ACKs false and shutdown never double-frees child',
        () {
      final recognizer = _FakeRecognizer(<String>[]);
      final events = <AsrWorkerEvent>[];
      var terminated = false;
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: events.add,
        terminate: () => terminated = true,
      );
      core.handle(const AsrWorkerReset(1));
      recognizer.streams.single.throwOnFree = true;
      core.handle(const AsrWorkerEnd(1));
      expect(events.whereType<AsrWorkerEndAck>().single.released, isFalse);
      core.handle(const AsrWorkerShutdown(1));
      expect(terminated, isTrue);
      expect(recognizer.streams.single.freeCalls, 1);
      expect(recognizer.freeCalls, 0);
      expect(events.whereType<AsrWorkerShutdownAck>(), isEmpty);
    });

    test('clean stream creation failure advances ordinal and remains reusable',
        () {
      final recognizer = _FakeRecognizer(<String>[])..createFailures = 1;
      final events = <AsrWorkerEvent>[];
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: events.add,
        terminate: () {},
      );
      core.handle(const AsrWorkerReset(1));
      final failure = events.whereType<AsrWorkerSessionFailure>().single;
      expect(failure.sessionReleased, isTrue);
      expect(failure.workerHealthy, isTrue);
      expect(core.gate.highestOrdinal, 1);
      core.handle(const AsrWorkerReset(2));
      expect(events.whereType<AsrWorkerResetAck>().single.ordinal, 2);
    });

    test('cooperative shutdown frees stream then recognizer then ACKs', () {
      final log = <String>[];
      final recognizer = _FakeRecognizer(log);
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: (event) => log.add('emit:${event.runtimeType}'),
        terminate: () => log.add('terminate'),
      );
      core.handle(const AsrWorkerReset(1));
      log.clear();
      core.handle(const AsrWorkerShutdown(9));
      expect(
        log,
        <String>[
          'stream-free:1',
          'recognizer-free',
          'emit:AsrWorkerShutdownAck',
          'terminate',
        ],
      );
    });

    test('uncertain stream release withholds shutdown ACK and recognizer free',
        () {
      final log = <String>[];
      final recognizer = _FakeRecognizer(log);
      final events = <AsrWorkerEvent>[];
      var terminated = false;
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: events.add,
        terminate: () => terminated = true,
      );
      core.handle(const AsrWorkerReset(1));
      recognizer.streams.single.throwOnFree = true;
      core.handle(const AsrWorkerShutdown(1));
      expect(terminated, isTrue);
      expect(recognizer.freeCalls, 0);
      expect(events.whereType<AsrWorkerShutdownAck>(), isEmpty);
      expect(core.retainedUncertainStreamCount, 1);
    });

    test('recognizer free failure withholds shutdown ACK', () {
      final recognizer = _FakeRecognizer(<String>[])..throwOnFree = true;
      final events = <AsrWorkerEvent>[];
      var terminated = false;
      final core = AsrWorkerCore(
        recognizer: recognizer,
        emit: events.add,
        terminate: () => terminated = true,
      );
      core.handle(const AsrWorkerShutdown(1));
      expect(terminated, isTrue);
      expect(recognizer.freeCalls, 1);
      expect(core.recognizerUncertain, isTrue);
      expect(events.whereType<AsrWorkerShutdownAck>(), isEmpty);
    });
  });
}

final class _FakeStream implements AsrWorkerStreamAdapter {
  _FakeStream(this.id, this.log);
  final int id;
  final List<String> log;
  bool throwOnFree = false;
  int freeCalls = 0;
  int acceptCalls = 0;

  @override
  void acceptPcm16(Float32List samples) {
    acceptCalls++;
    log.add('accept:$id');
  }

  @override
  void free() {
    freeCalls++;
    log.add('stream-free:$id');
    if (throwOnFree) throw StateError('stream_free_failed');
  }
}

final class _FakeRecognizer implements AsrWorkerRecognizerAdapter {
  _FakeRecognizer(this.log);
  final List<String> log;
  final List<_FakeStream> streams = <_FakeStream>[];
  String text = '';
  bool endpoint = false;
  bool alwaysReady = false;
  int readySteps = 0;
  int decodeCalls = 0;
  int freeCalls = 0;
  bool throwOnFree = false;
  int createFailures = 0;

  @override
  AsrWorkerStreamAdapter createStream() {
    if (createFailures > 0) {
      createFailures--;
      throw StateError('create_failed');
    }
    final stream = _FakeStream(streams.length + 1, log);
    streams.add(stream);
    log.add('create:${stream.id}');
    return stream;
  }

  @override
  void decode(AsrWorkerStreamAdapter stream) {
    decodeCalls++;
    if (readySteps > 0) readySteps--;
  }

  @override
  void free() {
    freeCalls++;
    log.add('recognizer-free');
    if (throwOnFree) throw StateError('recognizer_free_failed');
  }

  @override
  bool isEndpoint(AsrWorkerStreamAdapter stream) => endpoint;

  @override
  bool isReady(AsrWorkerStreamAdapter stream) => alwaysReady || readySteps > 0;

  @override
  String resultText(AsrWorkerStreamAdapter stream) => text;

  @override
  void reset(AsrWorkerStreamAdapter stream) {
    log.add('recognizer-reset');
  }
}
