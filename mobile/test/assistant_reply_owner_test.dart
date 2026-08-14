// Deterministic tests for the pure-Dart widget-bound reply owner.
//
// The hostile stream deliberately keeps invoking callbacks after cancel and
// can hold or fail its exact cancellation Future. No model, plugin, network,
// timer clock, Flutter widget, or audio device is loaded by this suite.
import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/assistant_reply_owner.dart';

const _activeLifetime = Duration(seconds: 101);
const _cancelLifetime = Duration(seconds: 7);

final class _ManualTimer implements Timer {
  _ManualTimer(this.duration, this._callback);

  final Duration duration;
  final void Function() _callback;
  bool _active = true;
  int _tick = 0;

  @override
  bool get isActive => _active;

  @override
  int get tick => _tick;

  @override
  void cancel() {
    _active = false;
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
  int? throwOnCall;
  int _calls = 0;

  Timer call(Duration duration, void Function() callback) {
    _calls += 1;
    if (_calls == throwOnCall) {
      throw StateError('fake timer construction failure');
    }
    final timer = _ManualTimer(duration, callback);
    timers.add(timer);
    return timer;
  }

  _ManualTimer onlyActive(Duration duration) => timers.singleWhere(
        (timer) => timer.duration == duration && timer.isActive,
      );
}

final class _HostileSource {
  _HostileSource(
    this.name,
    this.log, {
    this.holdCancel = false,
    this.cancelError,
    this.throwCancelSynchronously = false,
    this.throwListen = false,
  }) : _cancelGate = holdCancel ? Completer<void>() : null;

  final String name;
  final List<String> log;
  final bool holdCancel;
  final Object? cancelError;
  final bool throwCancelSynchronously;
  final bool throwListen;
  final Completer<void>? _cancelGate;

  void Function(String)? _onData;
  Function? _onError;
  void Function()? _onDone;
  void Function()? duringListen;
  _HostileSubscription? subscription;
  int listenCalls = 0;

  Stream<String> get stream => _HostileStream(this);

  StreamSubscription<String> listen(
    void Function(String)? onData, {
    Function? onError,
    void Function()? onDone,
    bool? cancelOnError,
  }) {
    listenCalls += 1;
    log.add('listen:$name');
    _onData = onData;
    _onError = onError;
    _onDone = onDone;
    final exact = _HostileSubscription(this);
    subscription = exact;
    duringListen?.call();
    if (throwListen) throw StateError('private listen failure for $name');
    return exact;
  }

  void emitData(String value) {
    _onData?.call(value);
  }

  void emitError(Object error) {
    final handler = _onError;
    if (handler == null) return;
    final stackTrace = StackTrace.current;
    if (handler is void Function(Object, StackTrace)) {
      handler(error, stackTrace);
    } else if (handler is void Function(Object)) {
      handler(error);
    } else {
      Function.apply(handler, <Object>[error, stackTrace]);
    }
  }

  void emitDone() {
    _onDone?.call();
  }

  void releaseCancel() {
    final gate = _cancelGate;
    if (gate != null && !gate.isCompleted) gate.complete();
  }
}

final class _HostileStream extends Stream<String> {
  const _HostileStream(this.source);

  final _HostileSource source;

  @override
  StreamSubscription<String> listen(
    void Function(String)? onData, {
    Function? onError,
    void Function()? onDone,
    bool? cancelOnError,
  }) =>
      source.listen(
        onData,
        onError: onError,
        onDone: onDone,
        cancelOnError: cancelOnError,
      );
}

final class _HostileSubscription implements StreamSubscription<String> {
  _HostileSubscription(this.source);

  final _HostileSource source;
  Future<void>? _cancelFuture;
  bool _paused = false;
  int cancelCalls = 0;
  int pauseCalls = 0;
  int resumeCalls = 0;

  @override
  Future<void> cancel() {
    cancelCalls += 1;
    source.log.add('subscription.cancel:${source.name}');
    if (source.throwCancelSynchronously) {
      throw StateError('private synchronous cancel failure');
    }
    final existing = _cancelFuture;
    if (existing != null) return existing;
    final Object? error = source.cancelError;
    if (error != null) {
      return _cancelFuture = Future<void>.error(error, StackTrace.current);
    }
    return _cancelFuture = source._cancelGate?.future ?? Future<void>.value();
  }

  @override
  void onData(void Function(String)? handleData) {
    source._onData = handleData;
  }

  @override
  void onError(Function? handleError) {
    source._onError = handleError;
  }

  @override
  void onDone(void Function()? handleDone) {
    source._onDone = handleDone;
  }

  @override
  void pause([Future<void>? resumeSignal]) {
    pauseCalls += 1;
    _paused = true;
    if (resumeSignal != null) {
      unawaited(resumeSignal.whenComplete(resume));
    }
  }

  @override
  void resume() {
    resumeCalls += 1;
    _paused = false;
  }

  @override
  bool get isPaused => _paused;

  @override
  Future<E> asFuture<E>([E? futureValue]) => Completer<E>().future;
}

final class _FakeBackend {
  _FakeBackend(this.log);

  final List<String> log;
  final Map<String, _HostileSource> sources = <String, _HostileSource>{};
  final Set<String> throwingPrompts = <String>{};
  final List<String> replyCalls = <String>[];

  void add(String prompt, _HostileSource source) {
    sources[prompt] = source;
  }

  Stream<String> open(String prompt) {
    replyCalls.add(prompt);
    if (throwingPrompts.contains(prompt)) {
      throw StateError('private factory failure for $prompt');
    }
    final source = sources[prompt];
    if (source == null) throw StateError('missing fake source');
    return source.stream;
  }
}

AssistantReplyOwner _owner(
  _FakeBackend backend,
  _TimerHarness timers, {
  int initialOrdinal = 0,
}) =>
    AssistantReplyOwner.forTesting(
      openReply: backend.open,
      maximumLifetime: _activeLifetime,
      cancelMaximumLifetime: _cancelLifetime,
      timerFactory: timers.call,
      initialOrdinal: initialOrdinal,
    );

Future<void> _drain() async {
  await Future<void>.delayed(Duration.zero);
  await Future<void>.delayed(Duration.zero);
}

Future<void> _closeClean(AssistantReplyOwner owner) async {
  final receipt = await owner.close();
  expect(receipt.exactSubscriptionsSettled, isTrue);
  expect(receipt.poisoned, isFalse);
}

void main() {
  test('sync listen data waits for handle publication and natural done',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('sync', log);
    backend.add('prompt', source);
    source.duringListen = () {
      source.emitData('early-private-token');
      source.emitDone();
    };
    final owner = _owner(backend, _TimerHarness());
    final tokens = <String>[];
    var startReturned = false;

    final generation = owner.start(
      prompt: 'prompt',
      onToken: (exact, token) {
        expect(startReturned, isTrue);
        expect(owner.isAuthoritative(exact), isTrue);
        tokens.add(token);
      },
    );
    startReturned = true;
    expect(tokens, isEmpty);
    expect(generation.isDone, isFalse);

    await _drain();
    expect(tokens, <String>['early-private-token']);
    expect((await generation.done).outcome, AssistantReplyOutcome.completed);
    final receipt = await generation.cleanup;
    expect(receipt.exactSubscriptionSettled, isTrue);
    expect(receipt.sourceDoneObserved, isTrue);
    expect(receipt.subscriptionCancelAttempted, isFalse);
    expect(source.subscription!.pauseCalls, 0);
    expect(source.subscription!.resumeCalls, 0);
    await _closeClean(owner);
  });

  test('hostile callbacks after true done are inert before deferred finish',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('terminal', log);
    source.duringListen = () {
      source.emitData('terminal-prefix');
      source.emitDone();
      source.emitData('post-terminal-private-token');
      source.emitError(StateError('post-terminal private error'));
    };
    backend.add('prompt', source);
    final owner = _owner(backend, _TimerHarness());
    final tokens = <String>[];
    final failures = <String>[];
    final generation = owner.start(
      prompt: 'prompt',
      onToken: (_, token) => tokens.add(token),
      onFailure: (_, failure) => failures.add(failure.code),
    );

    await _drain();
    expect(tokens, <String>['terminal-prefix']);
    expect(failures, isEmpty);
    expect((await generation.done).outcome, AssistantReplyOutcome.completed);
    final receipt = await generation.cleanup;
    expect(receipt.sourceDoneObserved, isTrue);
    expect(receipt.sourceErrorObserved, isFalse);
    expect(receipt.observedSourceDataEvents, 1);
    expect(receipt.callbackAttempts, 1);
    expect(receipt.subscriptionCancelAttempted, isFalse);
    expect(receipt.exactSubscriptionSettled, isTrue);
    expect(source.subscription!.cancelCalls, 0);
    await _closeClean(owner);
  });

  test('one active and one latest pending wait for exact held cancellation',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final old = _HostileSource('old', log, holdCancel: true);
    final middle = _HostileSource('middle', log);
    final newest = _HostileSource('newest', log);
    backend
      ..add('old', old)
      ..add('middle', middle)
      ..add('newest', newest);
    final owner = _owner(backend, _TimerHarness());
    final oldTokens = <String>[];
    final newestTokens = <String>[];

    final oldGeneration = owner.start(
      prompt: 'old',
      onToken: (_, token) => oldTokens.add(token),
    );
    await _drain();
    final middleGeneration = owner.start(
      prompt: 'middle',
      onToken: (_, __) {},
    );
    expect(oldGeneration.isDone, isTrue);
    expect(
        (await oldGeneration.done).outcome, AssistantReplyOutcome.superseded);
    expect(backend.replyCalls, <String>['old']);
    expect(log, <String>[
      'listen:old',
      'subscription.cancel:old',
    ]);

    final newestGeneration = owner.start(
      prompt: 'newest',
      onToken: (_, token) => newestTokens.add(token),
    );
    expect((await middleGeneration.done).outcome,
        AssistantReplyOutcome.superseded);
    final middleReceipt = await middleGeneration.cleanup;
    expect(middleReceipt.openAttempted, isFalse);
    expect(backend.replyCalls, <String>['old']);

    old.releaseCancel();
    await _drain();
    expect(backend.replyCalls, <String>['old', 'newest']);
    expect(owner.isAuthoritative(newestGeneration), isTrue);

    // A hostile stale subscription keeps calling back after cancel. None of
    // these callbacks may reach UI or disturb the new exact subscription.
    old.emitData('secret-stale-token');
    old.emitError(StateError('secret stale error'));
    old.emitDone();
    newest.emitData('fresh');
    newest.emitDone();
    await _drain();
    expect(oldTokens, isEmpty);
    expect(newestTokens, <String>['fresh']);
    expect(old.subscription!.pauseCalls, 0);
    expect(old.subscription!.cancelCalls, 1);
    expect(
        (await newestGeneration.done).outcome, AssistantReplyOutcome.completed);
    await _closeClean(owner);
  });

  test('pending exact cancel does no backend work and stale cancel is inert',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final old = _HostileSource('old', log, holdCancel: true);
    final pending = _HostileSource('pending', log);
    backend
      ..add('old', old)
      ..add('pending', pending);
    final owner = _owner(backend, _TimerHarness());
    final oldGeneration = owner.start(prompt: 'old', onToken: (_, __) {});
    final pendingGeneration =
        owner.start(prompt: 'pending', onToken: (_, __) {});

    final pendingReceipt = await owner.cancelExact(pendingGeneration);
    expect(pendingReceipt.outcome, AssistantReplyOutcome.cancelled);
    expect(pendingReceipt.openAttempted, isFalse);
    expect(backend.replyCalls, <String>['old']);

    old.releaseCancel();
    final oldReceipt = await oldGeneration.cleanup;
    expect(oldReceipt.exactSubscriptionSettled, isTrue);
    final sameReceipt = await owner.cancelExact(oldGeneration);
    expect(identical(oldReceipt, sameReceipt), isTrue);
    await _closeClean(owner);
  });

  test('foreign generation is rejected without either exact cancellation',
      () async {
    final log = <String>[];
    final firstBackend = _FakeBackend(log);
    final secondBackend = _FakeBackend(log);
    final firstSource = _HostileSource('first', log);
    final secondSource = _HostileSource('second', log);
    firstBackend.add('first', firstSource);
    secondBackend.add('second', secondSource);
    final firstOwner = _owner(firstBackend, _TimerHarness());
    final secondOwner = _owner(secondBackend, _TimerHarness());
    firstOwner.start(prompt: 'first', onToken: (_, __) {});
    final foreign = secondOwner.start(prompt: 'second', onToken: (_, __) {});

    expect(
      () => firstOwner.cancelExact(foreign),
      throwsA(isA<ArgumentError>()),
    );
    expect(firstSource.subscription!.cancelCalls, 0);
    expect(secondSource.subscription!.cancelCalls, 0);
    firstSource.emitDone();
    secondSource.emitDone();
    await _drain();
    await _closeClean(firstOwner);
    await _closeClean(secondOwner);
  });

  test('source error reports bounded failure before exact cancel ordering',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('error', log);
    backend.add('prompt', source);
    final owner = _owner(backend, _TimerHarness());
    final failures = <String>[];
    final tokens = <String>[];
    final generation = owner.start(
      prompt: 'prompt',
      onToken: (_, token) => tokens.add(token),
      onFailure: (_, failure) {
        failures.add(failure.code);
        log.add('failure:${failure.code}');
      },
    );
    await _drain();

    source.emitError(StateError('private raw provider error'));
    expect(generation.isDone, isTrue);
    expect(log, <String>[
      'listen:error',
      'failure:reply_source_failed',
      'subscription.cancel:error',
    ]);
    final done = await generation.done;
    expect(done.outcome, AssistantReplyOutcome.sourceFailed);
    expect(done.failure?.code, 'reply_source_failed');
    final receipt = await generation.cleanup;
    expect(receipt.sourceErrorObserved, isTrue);
    expect(receipt.sourceDoneObserved, isFalse);
    expect(receipt.exactSubscriptionSettled, isTrue);

    source.emitData('late-private-token');
    source.emitError(StateError('late private error'));
    source.emitDone();
    expect(tokens, isEmpty);
    expect(failures, <String>['reply_source_failed']);
    expect(source.subscription!.cancelCalls, 1);
    await _closeClean(owner);
  });

  test('old widget owner cancels only its subscription on shared opener',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final oldSource = _HostileSource('old-widget', log);
    final newSource = _HostileSource('new-widget', log);
    backend
      ..add('old', oldSource)
      ..add('new', newSource);
    final oldOwner = _owner(backend, _TimerHarness());
    final newOwner = _owner(backend, _TimerHarness());
    final oldGeneration = oldOwner.start(prompt: 'old', onToken: (_, __) {});
    final newTokens = <String>[];
    final newGeneration = newOwner.start(
      prompt: 'new',
      onToken: (_, token) => newTokens.add(token),
    );
    await _drain();

    final oldReceipt = await oldOwner.cancelExact(oldGeneration);
    expect(oldReceipt.exactSubscriptionSettled, isTrue);
    expect(log, <String>[
      'listen:old-widget',
      'listen:new-widget',
      'subscription.cancel:old-widget',
    ]);
    oldSource.emitData('stale-private-token');
    oldSource.emitError(StateError('stale private error'));
    oldSource.emitDone();
    newSource.emitData('still-current');
    newSource.emitDone();
    await _drain();
    expect(newTokens, <String>['still-current']);
    expect((await newGeneration.done).outcome, AssistantReplyOutcome.completed);
    expect(newSource.subscription!.cancelCalls, 0);
    expect(oldSource.subscription!.cancelCalls, 1);
    await _closeClean(oldOwner);
    await _closeClean(newOwner);
  });

  test('subscription cancel Future error poisons exact widget owner', () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource(
      'active',
      log,
      cancelError: StateError('private cancellation failure'),
    );
    backend.add('prompt', source);
    final owner = _owner(backend, _TimerHarness());
    final generation = owner.start(prompt: 'prompt', onToken: (_, __) {});
    final receipt = await owner.cancelExact(generation);

    expect(receipt.subscriptionCancelAttempted, isTrue);
    expect(receipt.subscriptionCancelSucceeded, isFalse);
    expect(receipt.exactSubscriptionSettled, isFalse);
    expect(source.subscription!.cancelCalls, 1);
    expect(owner.isPoisoned, isTrue);
    expect((await owner.close()).exactSubscriptionsSettled, isFalse);
  });

  test('listen throw is ambiguous and never fake-settled', () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('ambiguous', log, throwListen: true);
    backend.add('prompt', source);
    final owner = _owner(backend, _TimerHarness());
    final failures = <String>[];
    var startReturned = false;
    final generation = owner.start(
      prompt: 'prompt',
      onToken: (_, __) => fail('no token expected'),
      onFailure: (_, failure) {
        expect(startReturned, isTrue);
        failures.add(failure.code);
      },
    );
    startReturned = true;
    expect(generation.isDone, isTrue);
    final receipt = await generation.cleanup;
    await _drain();

    expect(
        (await generation.done).outcome, AssistantReplyOutcome.listenAmbiguous);
    expect(failures, <String>['reply_listen_ambiguous']);
    expect(receipt.sourceListenAttempted, isTrue);
    expect(receipt.sourceListenReturned, isFalse);
    expect(receipt.subscriptionCancelAttempted, isFalse);
    expect(receipt.exactSubscriptionSettled, isFalse);
    expect(source.subscription!.cancelCalls, 0);
    expect(owner.isPoisoned, isTrue);
    expect((await owner.close()).exactSubscriptionsSettled, isFalse);
  });

  test('sync source error followed by listen throw is listen ambiguity',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('ambiguous', log, throwListen: true);
    source.duringListen = () {
      source.emitError(StateError('private pre-throw source error'));
    };
    backend.add('prompt', source);
    final owner = _owner(backend, _TimerHarness());
    final failures = <String>[];
    final generation = owner.start(
      prompt: 'prompt',
      onToken: (_, __) {},
      onFailure: (_, failure) => failures.add(failure.code),
    );
    final done = await generation.done;
    final receipt = await generation.cleanup;
    await _drain();

    expect(done.outcome, AssistantReplyOutcome.listenAmbiguous);
    expect(done.failure?.code, 'reply_listen_ambiguous');
    expect(failures, <String>['reply_listen_ambiguous']);
    expect(receipt.sourceErrorObserved, isTrue);
    expect(receipt.sourceListenReturned, isFalse);
    expect(receipt.subscriptionCancelAttempted, isFalse);
    expect(receipt.exactSubscriptionSettled, isFalse);
    expect(owner.isPoisoned, isTrue);
    expect((await owner.close()).exactSubscriptionsSettled, isFalse);
  });

  test('sync source error plus true done uses terminal subscription proof',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('error-done', log);
    source.duringListen = () {
      source.emitError(StateError('private synchronous source error'));
      source.emitDone();
    };
    backend.add('prompt', source);
    final owner = _owner(backend, _TimerHarness());
    final failures = <String>[];
    final generation = owner.start(
      prompt: 'prompt',
      onToken: (_, __) {},
      onFailure: (_, failure) => failures.add(failure.code),
    );
    final done = await generation.done;
    final receipt = await generation.cleanup;
    await _drain();

    expect(done.outcome, AssistantReplyOutcome.sourceFailed);
    expect(failures, <String>['reply_source_failed']);
    expect(receipt.sourceErrorObserved, isTrue);
    expect(receipt.sourceDoneObserved, isTrue);
    expect(receipt.subscriptionCancelAttempted, isFalse);
    expect(receipt.exactSubscriptionSettled, isTrue);
    expect(source.subscription!.cancelCalls, 0);
    expect(owner.isPoisoned, isFalse);
    await _closeClean(owner);
  });

  test('side-effect-free reply factory throw is a clean startup failure',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log)..throwingPrompts.add('bad');
    final good = _HostileSource('good', log);
    backend.add('good', good);
    final owner = _owner(backend, _TimerHarness());
    final failures = <String>[];
    final failed = owner.start(
      prompt: 'bad',
      onToken: (_, __) {},
      onFailure: (_, failure) => failures.add(failure.code),
    );
    final failedReceipt = await failed.cleanup;
    expect(failedReceipt.exactSubscriptionSettled, isTrue);
    expect(failedReceipt.openAttempted, isTrue);
    expect(failedReceipt.openReturned, isFalse);
    expect(owner.isPoisoned, isFalse);

    final successor = owner.start(prompt: 'good', onToken: (_, __) {});
    good.emitDone();
    await _drain();
    expect((await successor.done).outcome, AssistantReplyOutcome.completed);
    expect(failures, <String>['reply_start_failed']);
    await _closeClean(owner);
  });

  test('throwing token callback is bounded failure and exact cancellation',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('callback', log);
    backend.add('prompt', source);
    final owner = _owner(backend, _TimerHarness());
    final failures = <String>[];
    final generation = owner.start(
      prompt: 'prompt',
      onToken: (_, __) => throw StateError('private callback exception'),
      onFailure: (_, failure) => failures.add(failure.code),
    );
    await _drain();
    source.emitData('private-token');

    final done = await generation.done;
    final receipt = await generation.cleanup;
    expect(done.outcome, AssistantReplyOutcome.callbackFailed);
    expect(done.failure?.code, 'reply_callback_failed');
    expect(failures, <String>['reply_callback_failed']);
    expect(receipt.callbackAttempts, 1);
    expect(receipt.callbackAttemptUtf8Bytes, 13);
    expect(receipt.exactSubscriptionSettled, isTrue);
    expect(source.subscription!.cancelCalls, 1);
    await _closeClean(owner);
  });

  test('UTF-8 prompt and per-chunk bounds fail without retaining content',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('chunk', log);
    backend.add('valid', source);
    final owner = _owner(backend, _TimerHarness());

    expect(
      () => owner.start(prompt: '', onToken: (_, __) {}),
      throwsA(isA<AssistantReplyFailure>()),
    );
    expect(
      () => owner.start(
        prompt: List<String>.filled(4097, '😀').join(),
        onToken: (_, __) {},
      ),
      throwsA(
        isA<AssistantReplyFailure>()
            .having((failure) => failure.code, 'code', 'invalid_prompt'),
      ),
    );
    expect(backend.replyCalls, isEmpty);

    final failures = <String>[];
    final generation = owner.start(
      prompt: 'valid',
      onToken: (_, __) => fail('oversized token must not escape'),
      onFailure: (_, failure) => failures.add(failure.code),
    );
    await _drain();
    source.emitData(List<String>.filled(1025, '😀').join());
    final receipt = await generation.cleanup;
    expect(failures, <String>['reply_chunk_limit_exceeded']);
    expect(receipt.observedSourceDataEvents, 1);
    expect(receipt.callbackAttempts, 0);
    expect(receipt.callbackAttemptUtf8Bytes, 0);
    expect(receipt.exactSubscriptionSettled, isTrue);
    await _closeClean(owner);
  });

  test('aggregate UTF-8 bound accepts the edge and rejects one more byte',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('aggregate', log);
    backend.add('prompt', source);
    final owner = _owner(backend, _TimerHarness());
    var delivered = 0;
    final failures = <String>[];
    final generation = owner.start(
      prompt: 'prompt',
      onToken: (_, __) => delivered += 1,
      onFailure: (_, failure) => failures.add(failure.code),
    );
    await _drain();
    final chunk = List<String>.filled(4096, 'a').join();
    for (var index = 0; index < 16; index++) {
      source.emitData(chunk);
    }
    source.emitData('x');

    final receipt = await generation.cleanup;
    expect(delivered, 16);
    expect(failures, <String>['reply_aggregate_limit_exceeded']);
    expect(receipt.observedSourceDataEvents, 17);
    expect(receipt.callbackAttempts, 16);
    expect(
      receipt.callbackAttemptUtf8Bytes,
      assistantReplyMaximumUtf8Bytes,
    );
    expect(receipt.exactSubscriptionSettled, isTrue);
    await _closeClean(owner);
  });

  test('active deadline is sticky even when exact cancel later succeeds',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('active', log);
    backend.add('prompt', source);
    final timers = _TimerHarness();
    final owner = _owner(backend, timers);
    final generation = owner.start(prompt: 'prompt', onToken: (_, __) {});
    await _drain();

    timers.onlyActive(_activeLifetime).fire();
    expect(generation.isDone, isTrue);
    final done = await generation.done;
    final receipt = await generation.cleanup;
    expect(done.outcome, AssistantReplyOutcome.deadlineExceeded);
    expect(done.failure?.code, 'reply_deadline_exceeded');
    expect(receipt.subscriptionCancelSucceeded, isTrue);
    expect(receipt.exactSubscriptionSettled, isTrue);
    expect(owner.isPoisoned, isTrue);
    expect(owner.snapshot.retainsUncertainReply, isFalse);
    final closeReceipt = await owner.close();
    expect(closeReceipt.exactSubscriptionsSettled, isTrue);
    expect(closeReceipt.poisoned, isTrue);
  });

  test('pending deadline is clean and performs zero backend work', () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final active = _HostileSource('active', log, holdCancel: true);
    final pending = _HostileSource('pending', log);
    backend
      ..add('active', active)
      ..add('pending', pending);
    final timers = _TimerHarness();
    final owner = _owner(backend, timers);
    final activeGeneration = owner.start(prompt: 'active', onToken: (_, __) {});
    final pendingGeneration =
        owner.start(prompt: 'pending', onToken: (_, __) {});

    timers.onlyActive(_activeLifetime).fire();
    final pendingDone = await pendingGeneration.done;
    final pendingReceipt = await pendingGeneration.cleanup;
    expect(pendingDone.outcome, AssistantReplyOutcome.deadlineExceeded);
    expect(pendingDone.failure?.code, 'reply_deadline_exceeded');
    expect(pendingReceipt.openAttempted, isFalse);
    expect(pendingReceipt.exactSubscriptionSettled, isTrue);
    expect(backend.replyCalls, <String>['active']);
    expect(owner.isPoisoned, isFalse);

    active.releaseCancel();
    expect(
      (await activeGeneration.cleanup).exactSubscriptionSettled,
      isTrue,
    );
    await _closeClean(owner);
  });

  test('pending promotion keeps its original admission lifetime timer',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final active = _HostileSource('active', log, holdCancel: true);
    final pending = _HostileSource('pending', log);
    backend
      ..add('active', active)
      ..add('pending', pending);
    final timers = _TimerHarness();
    final owner = _owner(backend, timers);
    owner.start(prompt: 'active', onToken: (_, __) {});
    final pendingGeneration =
        owner.start(prompt: 'pending', onToken: (_, __) {});
    final admissionTimer = timers.onlyActive(_activeLifetime);

    active.releaseCancel();
    await _drain();
    expect(backend.replyCalls, <String>['active', 'pending']);
    expect(
        identical(timers.onlyActive(_activeLifetime), admissionTimer), isTrue);
    expect(
      timers.timers.where((timer) => timer.duration == _activeLifetime).length,
      2,
    );

    admissionTimer.fire();
    final done = await pendingGeneration.done;
    final receipt = await pendingGeneration.cleanup;
    expect(done.outcome, AssistantReplyOutcome.deadlineExceeded);
    expect(receipt.exactSubscriptionSettled, isTrue);
    expect(owner.isPoisoned, isTrue);
    final closeReceipt = await owner.close();
    expect(closeReceipt.exactSubscriptionsSettled, isTrue);
    expect(closeReceipt.poisoned, isTrue);
  });

  test('cancel deadline poisons, drops pending, and ignores late completion',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final active = _HostileSource('active', log, holdCancel: true);
    final pending = _HostileSource('pending', log);
    backend
      ..add('active', active)
      ..add('pending', pending);
    final timers = _TimerHarness();
    final owner = _owner(backend, timers);
    final activeGeneration = owner.start(prompt: 'active', onToken: (_, __) {});
    final pendingGeneration =
        owner.start(prompt: 'pending', onToken: (_, __) {});

    timers.onlyActive(_cancelLifetime).fire();
    final activeReceipt = await activeGeneration.cleanup;
    final pendingDone = await pendingGeneration.done;
    expect(activeReceipt.exactSubscriptionSettled, isFalse);
    expect(pendingDone.outcome, AssistantReplyOutcome.ownerPoisoned);
    expect((await pendingGeneration.cleanup).openAttempted, isFalse);
    expect(owner.isPoisoned, isTrue);
    expect(backend.replyCalls, <String>['active']);

    active.releaseCancel();
    await _drain();
    expect(backend.replyCalls, <String>['active']);
    expect((await owner.close()).exactSubscriptionsSettled, isFalse);
  });

  test('promoted successor flushes sync token before later async token',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final active = _HostileSource('active', log, holdCancel: true);
    final successor = _HostileSource('successor', log);
    successor.duringListen = () => successor.emitData('sync-first');
    backend
      ..add('active', active)
      ..add('successor', successor);
    final owner = _owner(backend, _TimerHarness());
    owner.start(prompt: 'active', onToken: (_, __) {});
    final tokens = <String>[];
    final generation = owner.start(
      prompt: 'successor',
      onToken: (_, token) => tokens.add(token),
    );
    await _drain();
    expect(backend.replyCalls, <String>['active']);

    active.releaseCancel();
    await _drain();
    expect(tokens, <String>['sync-first']);
    successor.emitData('async-second');
    expect(tokens, <String>['sync-first', 'async-second']);
    successor.emitDone();
    await _drain();
    expect((await generation.done).outcome, AssistantReplyOutcome.completed);
    await _closeClean(owner);
  });

  test('promoted sync error without done reports once and cancels exact sub',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final active = _HostileSource('active', log, holdCancel: true);
    final successor = _HostileSource('successor', log);
    successor.duringListen = () {
      successor.emitError(StateError('private synchronous source failure'));
    };
    backend
      ..add('active', active)
      ..add('successor', successor);
    final owner = _owner(backend, _TimerHarness());
    owner.start(prompt: 'active', onToken: (_, __) {});
    final failures = <String>[];
    final generation = owner.start(
      prompt: 'successor',
      onToken: (_, __) => fail('no token expected'),
      onFailure: (_, failure) => failures.add(failure.code),
    );
    await _drain();
    active.releaseCancel();
    await _drain();

    expect(failures, <String>['reply_source_failed']);
    expect((await generation.done).outcome, AssistantReplyOutcome.sourceFailed);
    final receipt = await generation.cleanup;
    expect(receipt.sourceErrorObserved, isTrue);
    expect(receipt.sourceDoneObserved, isFalse);
    expect(receipt.subscriptionCancelSucceeded, isTrue);
    expect(receipt.exactSubscriptionSettled, isTrue);
    expect(successor.subscription!.cancelCalls, 1);
    successor.emitError(StateError('late private failure'));
    await _drain();
    expect(failures, <String>['reply_source_failed']);
    await _closeClean(owner);
  });

  test('sync done then immediate successor uses terminal proof without cancel',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final old = _HostileSource('old', log);
    old.duringListen = () {
      old.emitData('must-be-dropped');
      old.emitDone();
    };
    final successor = _HostileSource('successor', log);
    backend
      ..add('old', old)
      ..add('successor', successor);
    final owner = _owner(backend, _TimerHarness());
    final oldTokens = <String>[];
    final oldGeneration = owner.start(
      prompt: 'old',
      onToken: (_, token) => oldTokens.add(token),
    );
    final successorGeneration =
        owner.start(prompt: 'successor', onToken: (_, __) {});

    expect(old.subscription!.cancelCalls, 0);
    expect(backend.replyCalls, <String>['old', 'successor']);
    expect(
        (await oldGeneration.done).outcome, AssistantReplyOutcome.superseded);
    expect(
      (await oldGeneration.cleanup).exactSubscriptionSettled,
      isTrue,
    );
    await _drain();
    expect(oldTokens, isEmpty);
    successor.emitDone();
    await _drain();
    expect((await successorGeneration.done).outcome,
        AssistantReplyOutcome.completed);
    await _closeClean(owner);
  });

  test('pending timer construction failure poisons and revokes active',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final active = _HostileSource('active', log);
    final pending = _HostileSource('pending', log);
    backend
      ..add('active', active)
      ..add('pending', pending);
    final timers = _TimerHarness()..throwOnCall = 2;
    final owner = _owner(backend, timers);
    final tokens = <String>[];
    final activeGeneration = owner.start(
      prompt: 'active',
      onToken: (_, token) => tokens.add(token),
    );
    await _drain();
    active.emitData('before-failure');
    final pendingGeneration =
        owner.start(prompt: 'pending', onToken: (_, __) {});
    active.emitData('late-private-token');

    expect((await activeGeneration.done).outcome,
        AssistantReplyOutcome.ownerPoisoned);
    expect((await pendingGeneration.done).outcome,
        AssistantReplyOutcome.ownerPoisoned);
    expect(tokens, <String>['before-failure']);
    expect(backend.replyCalls, <String>['active']);
    expect(active.subscription!.cancelCalls, 1);
    expect(
      (await activeGeneration.cleanup).exactSubscriptionSettled,
      isTrue,
    );
    expect(owner.isPoisoned, isTrue);
    final closeReceipt = await owner.close();
    expect(closeReceipt.exactSubscriptionsSettled, isTrue);
    expect(closeReceipt.poisoned, isTrue);
  });

  test('cancelCurrent covers active, pending, and empty owner states',
      () async {
    final activeLog = <String>[];
    final activeBackend = _FakeBackend(activeLog);
    final activeSource = _HostileSource('active', activeLog);
    activeBackend.add('active', activeSource);
    final activeOwner = _owner(activeBackend, _TimerHarness());
    final activeGeneration =
        activeOwner.start(prompt: 'active', onToken: (_, __) {});
    final activeReceipt = await activeOwner.cancelCurrent();
    expect(activeGeneration.isDone, isTrue);
    expect(activeReceipt?.outcome, AssistantReplyOutcome.cancelled);
    expect(activeSource.subscription!.cancelCalls, 1);
    expect(await activeOwner.cancelCurrent(), isNull);
    await _closeClean(activeOwner);

    final pendingLog = <String>[];
    final pendingBackend = _FakeBackend(pendingLog);
    final old = _HostileSource('old', pendingLog, holdCancel: true);
    final pending = _HostileSource('pending', pendingLog);
    pendingBackend
      ..add('old', old)
      ..add('pending', pending);
    final pendingOwner = _owner(pendingBackend, _TimerHarness());
    pendingOwner.start(prompt: 'old', onToken: (_, __) {});
    final pendingGeneration =
        pendingOwner.start(prompt: 'pending', onToken: (_, __) {});
    final pendingReceipt = await pendingOwner.cancelCurrent();
    expect(pendingReceipt?.outcome, AssistantReplyOutcome.cancelled);
    expect(pendingGeneration.isDone, isTrue);
    expect(pendingReceipt?.openAttempted, isFalse);
    expect(await pendingOwner.cancelCurrent(), isNull);
    old.releaseCancel();
    await _drain();
    await _closeClean(pendingOwner);
  });

  test('synchronous subscription cancel throw poisons and retains handle',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource(
      'active',
      log,
      throwCancelSynchronously: true,
    );
    backend.add('prompt', source);
    final owner = _owner(backend, _TimerHarness());
    final generation = owner.start(prompt: 'prompt', onToken: (_, __) {});
    final receipt = await owner.cancelExact(generation);

    expect(receipt.subscriptionCancelAttempted, isTrue);
    expect(receipt.subscriptionCancelSucceeded, isFalse);
    expect(receipt.exactSubscriptionSettled, isFalse);
    expect(source.subscription!.cancelCalls, 1);
    expect(owner.isPoisoned, isTrue);
    expect(owner.snapshot.retainsUncertainReply, isTrue);
    expect((await owner.close()).exactSubscriptionsSettled, isFalse);
  });

  test('source event limit receipt counts the 2049th causing event', () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('events', log);
    backend.add('prompt', source);
    final owner = _owner(backend, _TimerHarness());
    var callbackAttempts = 0;
    final generation = owner.start(
      prompt: 'prompt',
      onToken: (_, __) => callbackAttempts += 1,
    );
    await _drain();
    for (var index = 0;
        index < assistantReplyMaximumSourceDataEvents + 1;
        index++) {
      source.emitData('');
    }

    final done = await generation.done;
    final receipt = await generation.cleanup;
    expect(done.failure?.code, 'reply_event_limit_exceeded');
    expect(
      receipt.observedSourceDataEvents,
      assistantReplyMaximumSourceDataEvents + 1,
    );
    expect(callbackAttempts, assistantReplyMaximumSourceDataEvents);
    expect(receipt.callbackAttempts, assistantReplyMaximumSourceDataEvents);
    expect(receipt.callbackAttemptUtf8Bytes, 0);
    expect(receipt.exactSubscriptionSettled, isTrue);
    await _closeClean(owner);
  });

  test('failure callback reentrantly admits successor after exact cleanup',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final old = _HostileSource('old', log);
    final next = _HostileSource('next', log);
    backend
      ..add('old', old)
      ..add('next', next);
    final owner = _owner(backend, _TimerHarness());
    AssistantReplyGeneration? replacement;
    final oldGeneration = owner.start(
      prompt: 'old',
      onToken: (_, __) {},
      onFailure: (_, __) {
        replacement = owner.start(prompt: 'next', onToken: (_, __) {});
      },
    );
    await _drain();
    old.emitError(StateError('private error'));
    expect(replacement, isNotNull);
    expect(backend.replyCalls, <String>['old']);

    await oldGeneration.cleanup;
    await _drain();
    expect(backend.replyCalls, <String>['old', 'next']);
    expect(owner.isAuthoritative(replacement!), isTrue);
    next.emitDone();
    await _drain();
    await _closeClean(owner);
  });

  test('reentrant pending completion preserves the newest pending winner',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final active = _HostileSource('active', log, holdCancel: true);
    final middle = _HostileSource('middle', log);
    final third = _HostileSource('third', log);
    final reentrant = _HostileSource('reentrant', log);
    backend
      ..add('active', active)
      ..add('middle', middle)
      ..add('third', third)
      ..add('reentrant', reentrant);
    final owner = _owner(backend, _TimerHarness());
    owner.start(prompt: 'active', onToken: (_, __) {});
    final middleGeneration = owner.start(prompt: 'middle', onToken: (_, __) {});
    AssistantReplyGeneration? reentrantGeneration;
    unawaited(
      middleGeneration.done.then((_) {
        reentrantGeneration =
            owner.start(prompt: 'reentrant', onToken: (_, __) {});
      }),
    );
    final thirdGeneration = owner.start(prompt: 'third', onToken: (_, __) {});
    await _drain();

    expect(
        (await thirdGeneration.done).outcome, AssistantReplyOutcome.superseded);
    expect(reentrantGeneration, isNotNull);
    expect(backend.replyCalls, <String>['active']);
    active.releaseCancel();
    await _drain();
    expect(backend.replyCalls, <String>['active', 'reentrant']);
    expect(owner.isAuthoritative(reentrantGeneration!), isTrue);
    reentrant.emitDone();
    await _drain();
    await _closeClean(owner);
  });

  test('close is memoized, fences pending, and waits exact active cancel',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final active = _HostileSource('active', log, holdCancel: true);
    final pending = _HostileSource('pending', log);
    backend
      ..add('active', active)
      ..add('pending', pending);
    final owner = _owner(backend, _TimerHarness());
    final activeGeneration = owner.start(prompt: 'active', onToken: (_, __) {});
    final pendingGeneration =
        owner.start(prompt: 'pending', onToken: (_, __) {});
    final firstClose = owner.close();
    expect(identical(firstClose, owner.close()), isTrue);
    expect((await pendingGeneration.done).outcome,
        AssistantReplyOutcome.ownerClosed);
    expect((await pendingGeneration.cleanup).openAttempted, isFalse);
    expect(activeGeneration.isDone, isTrue);
    expect(owner.isClosed, isTrue);
    expect(
      () => owner.start(prompt: 'late', onToken: (_, __) {}),
      throwsA(
        isA<AssistantReplyFailure>()
            .having((failure) => failure.code, 'code', 'owner_closed'),
      ),
    );

    active.releaseCancel();
    final closeReceipt = await firstClose;
    expect(closeReceipt.exactSubscriptionsSettled, isTrue);
    expect(closeReceipt.poisoned, isFalse);
    expect(backend.replyCalls, <String>['active']);
    expect(active.subscription!.cancelCalls, 1);
  });

  test('safe ordinal exhaustion fences existing active and poisons owner',
      () async {
    final log = <String>[];
    final backend = _FakeBackend(log);
    final source = _HostileSource('last', log);
    backend.add('last', source);
    final owner = _owner(
      backend,
      _TimerHarness(),
      initialOrdinal: assistantReplyMaximumSafeOrdinal - 1,
    );
    final last = owner.start(prompt: 'last', onToken: (_, __) {});
    expect(last.ordinal, assistantReplyMaximumSafeOrdinal);
    expect(
      () => owner.start(prompt: 'overflow', onToken: (_, __) {}),
      throwsA(
        isA<AssistantReplyFailure>()
            .having((failure) => failure.code, 'code', 'ordinal_exhausted'),
      ),
    );
    expect(owner.isPoisoned, isTrue);
    expect((await last.done).outcome, AssistantReplyOutcome.ownerPoisoned);
    expect((await last.cleanup).exactSubscriptionSettled, isTrue);
    expect(source.subscription!.cancelCalls, 1);
    final closeReceipt = await owner.close();
    expect(closeReceipt.exactSubscriptionsSettled, isTrue);
    expect(closeReceipt.poisoned, isTrue);
  });
}
