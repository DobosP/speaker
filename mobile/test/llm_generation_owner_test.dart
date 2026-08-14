// Deterministic lifecycle tests for the pure-Dart Gemma generation owner.
// Fakes control every async boundary; no plugin, model, network, timer, GPU,
// audio device, or native session is loaded.
import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/llm_generation_owner.dart';

Future<void> _flush([int turns = 12]) async {
  for (var i = 0; i < turns; i++) {
    await Future<void>.value();
  }
}

final class _ManualTimer implements Timer {
  _ManualTimer(this.callback);

  final void Function() callback;
  bool _active = true;
  int _tick = 0;

  void fire() {
    if (!_active) return;
    _active = false;
    _tick = 1;
    callback();
  }

  @override
  void cancel() => _active = false;

  @override
  bool get isActive => _active;

  @override
  int get tick => _tick;
}

final class _TimerHarness {
  final List<_ManualTimer> timers = <_ManualTimer>[];
  final List<Duration> durations = <Duration>[];

  Timer call(Duration duration, void Function() callback) {
    durations.add(duration);
    final timer = _ManualTimer(callback);
    timers.add(timer);
    return timer;
  }
}

final class _ThrowingListenStream extends Stream<GemmaGenerationEvent> {
  @override
  StreamSubscription<GemmaGenerationEvent> listen(
    void Function(GemmaGenerationEvent event)? onData, {
    Function? onError,
    void Function()? onDone,
    bool? cancelOnError,
  }) {
    throw StateError('listen failed after generation entry');
  }
}

final class _FakeChat implements GemmaChatPort {
  _FakeChat({
    this.generateThrows = false,
    this.listenThrows = false,
    this.holdAdd = false,
    this.holdStop = false,
    this.holdClose = false,
    this.onStop,
    List<String>? log,
  }) : log = log ?? <String>[] {
    if (!holdAdd) addGate.complete();
    if (!holdStop) stopGate.complete();
    if (!holdClose) closeGate.complete();
  }

  final bool generateThrows;
  final bool listenThrows;
  final bool holdAdd;
  final bool holdStop;
  final bool holdClose;
  final void Function()? onStop;
  final List<String> log;
  final StreamController<GemmaGenerationEvent> source =
      StreamController<GemmaGenerationEvent>(sync: true);
  final Completer<void> addGate = Completer<void>();
  final Completer<void> stopGate = Completer<void>();
  final Completer<void> closeGate = Completer<void>();

  int addCalls = 0;
  int generateCalls = 0;
  int stopCalls = 0;
  int closeCalls = 0;
  String? prompt;

  @override
  Future<void> addPrompt(String prompt) {
    addCalls++;
    this.prompt = prompt;
    log.add('add');
    return addGate.future;
  }

  @override
  Stream<GemmaGenerationEvent> generate() {
    generateCalls++;
    log.add('generate');
    if (generateThrows) {
      throw StateError('generate failed ambiguously');
    }
    if (listenThrows) return _ThrowingListenStream();
    return source.stream;
  }

  void addText(String text) => source.add(GemmaGenerationEvent.text(text));

  void addNonText() => source.add(const GemmaGenerationEvent.nonText());

  @override
  Future<void> stopGeneration() {
    stopCalls++;
    log.add('stop');
    onStop?.call();
    return stopGate.future;
  }

  @override
  Future<void> close() {
    closeCalls++;
    log.add('close');
    return closeGate.future;
  }
}

final class _FakeFactory {
  _FakeFactory(this.chat, {this.error, this.gate});

  final _FakeChat chat;
  final Object? error;
  final Completer<GemmaChatPort>? gate;
  int calls = 0;

  Future<GemmaChatPort> call() async {
    calls++;
    final failure = error;
    if (failure != null) throw failure;
    final held = gate;
    if (held != null) return held.future;
    return chat;
  }
}

final class _ObservedStream {
  _ObservedStream(
    Stream<String> stream, {
    void Function(String)? onData,
    void Function()? onDone,
  }) {
    subscription = stream.listen(
      (value) {
        values.add(value);
        onData?.call(value);
      },
      onError: (Object error, StackTrace stackTrace) {
        errors.add(error);
      },
      onDone: () {
        onDone?.call();
        done.complete();
      },
      cancelOnError: false,
    );
  }

  final List<String> values = <String>[];
  final List<Object> errors = <Object>[];
  final Completer<void> done = Completer<void>();
  late final StreamSubscription<String> subscription;
}

GemmaGenerationOwner _owner({_TimerHarness? timers}) =>
    GemmaGenerationOwner.forTesting(
      maximumLifetime: const Duration(seconds: 1),
      timerFactory:
          timers?.call ?? ((duration, callback) => _ManualTimer(callback)),
    );

Future<void> _finishNaturally(_FakeChat chat) async {
  await chat.source.close();
  await _flush();
}

void main() {
  test('natural terminal closes once before exact release', () async {
    final owner = _owner();
    final chat = _FakeChat(holdClose: true);
    final factory = _FakeFactory(chat);
    final observed = _ObservedStream(
      owner.generate(prompt: 'hello', createChat: factory.call),
    );
    var outwardDone = false;
    unawaited(observed.done.future.then((_) => outwardDone = true));
    await _flush();

    expect(factory.calls, 1);
    expect(chat.addCalls, 1);
    expect(chat.generateCalls, 1);
    chat.addText('one');
    chat.addText(' two');
    await chat.source.close();
    await _flush();

    expect(observed.values, <String>['one', ' two']);
    expect(chat.stopCalls, 0);
    expect(chat.closeCalls, 1);
    expect(owner.hasActiveGeneration, isTrue);
    expect(outwardDone, isFalse);
    chat.closeGate.complete();
    await _flush();

    expect(owner.hasActiveGeneration, isFalse);
    expect(owner.isPoisoned, isFalse);
    expect(owner.lastReceipt?.sourceTerminalObserved, isTrue);
    expect(owner.lastReceipt?.chatCloseSucceeded, isTrue);
    expect(owner.lastReceipt?.exactlyReleased, isTrue);
    expect(chat.closeCalls, 1);
    await observed.done.future;
    expect(outwardDone, isTrue);
  });

  test(
    'cancelCurrent after source terminal closes output but not the chat',
    () async {
      final owner = _owner();
      final chat = _FakeChat(holdClose: true);
      final observed = _ObservedStream(
        owner.generate(
          prompt: 'terminal cancel',
          createChat: _FakeFactory(chat).call,
        ),
      );
      var outwardDone = false;
      unawaited(observed.done.future.then((_) => outwardDone = true));
      await _flush();

      await chat.source.close();
      await _flush();
      expect(chat.closeCalls, 1);
      expect(outwardDone, isFalse);
      expect(owner.hasActiveGeneration, isTrue);

      owner.cancelCurrent();
      await observed.done.future;
      expect(outwardDone, isTrue);
      expect(chat.stopCalls, 0);
      expect(owner.hasActiveGeneration, isTrue);

      chat.closeGate.complete();
      await _flush();
      expect(owner.hasActiveGeneration, isFalse);
      expect(owner.isPoisoned, isFalse);
      expect(owner.lastReceipt?.outcome, GemmaGenerationOutcome.completed);
      expect(owner.lastReceipt?.sourceTerminalObserved, isTrue);
      expect(owner.lastReceipt?.chatCloseSucceeded, isTrue);
      expect(owner.lastReceipt?.exactlyReleased, isTrue);
    },
  );

  test(
    'real await-for forwards multiple chunks without implicit cancellation',
    () async {
      final owner = _owner();
      final chat = _FakeChat();
      final values = <String>[];
      final consumed = () async {
        await for (final value in owner.generate(
          prompt: 'await for',
          createChat: _FakeFactory(chat).call,
        )) {
          values.add(value);
        }
      }();
      await _flush();

      chat.addText('one');
      await _flush();
      chat.addText(' two');
      await _flush();
      expect(values, <String>['one', ' two']);
      expect(chat.stopCalls, 0);

      await chat.source.close();
      await consumed;
      await _flush();
      expect(owner.isPoisoned, isFalse);
    },
  );

  test('consumer cancel future settles while explicit stop is held', () async {
    final owner = _owner();
    final chat = _FakeChat(holdStop: true, holdClose: true);
    final observed = _ObservedStream(
      owner.generate(prompt: 'cancel me', createChat: _FakeFactory(chat).call),
    );
    await _flush();

    var cancelSettled = false;
    final cancelFuture = observed.subscription.cancel()
      ..then((_) => cancelSettled = true);
    await _flush();

    expect(cancelSettled, isTrue);
    expect(chat.stopCalls, 1);
    expect(chat.closeCalls, 0);
    expect(owner.hasActiveGeneration, isTrue);

    chat.stopGate.complete();
    await chat.source.close();
    await _flush();
    expect(chat.closeCalls, 1);
    chat.closeGate.complete();
    await cancelFuture;
    await _flush();
    expect(owner.hasActiveGeneration, isFalse);
  });

  test(
    'cancelling a pending prompt settles while predecessor stop is held',
    () async {
      final owner = _owner();
      final active = _FakeChat(holdStop: true, holdClose: true);
      final activeObserved = _ObservedStream(
        owner.generate(prompt: 'active', createChat: _FakeFactory(active).call),
      );
      await _flush();

      final pendingFactory = _FakeFactory(_FakeChat());
      final pending = _ObservedStream(
        owner.generate(prompt: 'pending', createChat: pendingFactory.call),
      );
      await _flush();
      expect(active.stopCalls, 1);
      expect(pendingFactory.calls, 0);

      var cancelSettled = false;
      final cancelled = pending.subscription.cancel()
        ..then((_) => cancelSettled = true);
      await _flush();
      expect(cancelSettled, isTrue);
      expect(owner.hasPendingGeneration, isFalse);
      expect(pendingFactory.calls, 0);

      active.stopGate.complete();
      await active.source.close();
      await _flush();
      active.closeGate.complete();
      await cancelled;
      await _flush();
      expect(owner.hasActiveGeneration, isFalse);
      await activeObserved.done.future;
    },
  );

  test(
    'cleanup orders explicit stop, true source terminal, then close',
    () async {
      final log = <String>[];
      final owner = _owner();
      final chat = _FakeChat(holdStop: true, holdClose: true, log: log);
      final observed = _ObservedStream(
        owner.generate(prompt: 'ordered', createChat: _FakeFactory(chat).call),
      );
      await _flush();
      await observed.subscription.cancel();
      await _flush();

      expect(log, containsAllInOrder(<String>['generate', 'stop']));
      expect(log, isNot(contains('close')));
      chat.stopGate.complete();
      await _flush();
      expect(log, isNot(contains('close')));
      await chat.source.close();
      await _flush();
      expect(log, containsAllInOrder(<String>['stop', 'close']));
      chat.closeGate.complete();
      await _flush();
      expect(chat.stopCalls, 1);
      expect(chat.closeCalls, 1);
    },
  );

  test(
    'synchronous terminal from stop still waits for the stop future',
    () async {
      final owner = _owner();
      late final _FakeChat chat;
      chat = _FakeChat(
        holdStop: true,
        holdClose: true,
        onStop: () {
          unawaited(chat.source.close());
        },
      );
      final observed = _ObservedStream(
        owner.generate(
          prompt: 'reentrant stop',
          createChat: _FakeFactory(chat).call,
        ),
      );
      await _flush();
      await observed.subscription.cancel();
      await _flush();

      expect(chat.stopCalls, 1);
      expect(chat.closeCalls, 0);
      expect(owner.hasActiveGeneration, isTrue);
      chat.stopGate.complete();
      await _flush();
      expect(chat.closeCalls, 1);
      chat.closeGate.complete();
      await _flush();
      expect(owner.hasActiveGeneration, isFalse);
    },
  );

  test(
    'stop error still drains true terminal and closes exactly once',
    () async {
      final owner = _owner();
      final chat = _FakeChat(holdStop: true, holdClose: true);
      final observed = _ObservedStream(
        owner.generate(
          prompt: 'stop error',
          createChat: _FakeFactory(chat).call,
        ),
      );
      await _flush();
      await observed.subscription.cancel();
      await _flush();
      chat.stopGate.completeError(StateError('cooperative stop failed'));
      await _flush();

      expect(chat.closeCalls, 0);
      expect(owner.hasActiveGeneration, isTrue);
      await chat.source.close();
      await _flush();
      expect(chat.closeCalls, 1);
      chat.closeGate.complete();
      await _flush();

      expect(owner.isPoisoned, isFalse);
      expect(owner.lastReceipt?.explicitStopAttempted, isTrue);
      expect(owner.lastReceipt?.explicitStopSucceeded, isFalse);
      expect(owner.lastReceipt?.sourceTerminalObserved, isTrue);
    },
  );

  test(
    'source error is not a terminal receipt and late tokens stay fenced',
    () async {
      final owner = _owner();
      final chat = _FakeChat(holdStop: true, holdClose: true);
      final observed = _ObservedStream(
        owner.generate(
          prompt: 'source error',
          createChat: _FakeFactory(chat).call,
        ),
      );
      await _flush();
      chat.addText('before');
      chat.source.addError(
        StateError('native stream error'),
        StackTrace.current,
      );
      chat.addText('late');
      await _flush();

      expect(observed.values, <String>['before']);
      expect(observed.errors, hasLength(1));
      expect(chat.stopCalls, 1);
      expect(chat.closeCalls, 0);
      expect(owner.hasActiveGeneration, isTrue);

      chat.stopGate.complete();
      await _flush();
      expect(chat.closeCalls, 0);
      await chat.source.close();
      await _flush();
      expect(chat.closeCalls, 1);
      chat.closeGate.complete();
      await _flush();
      expect(owner.lastReceipt?.sourceTerminalObserved, isTrue);
    },
  );

  test('explicit pause keeps draining with bounded retained output', () async {
    final owner = _owner();
    final chat = _FakeChat(holdStop: true, holdClose: true);
    final observed = _ObservedStream(
      owner.generate(prompt: 'pause', createChat: _FakeFactory(chat).call),
    );
    await _flush();
    chat.addText('a');
    observed.subscription.pause();
    await _flush();
    final fullChunk = List<String>.filled(
      gemmaChunkMaximumUtf8Bytes,
      'b',
    ).join();
    for (var i = 0; i < 15; i++) {
      chat.addText(fullChunk);
    }
    chat.addText(List<String>.filled(4095, 'c').join());
    await _flush();

    expect(observed.values, <String>['a']);
    expect(chat.stopCalls, 0);
    expect(chat.source.hasListener, isTrue);
    chat.addText('excess');
    await _flush();
    expect(chat.stopCalls, 1);

    chat.stopGate.complete();
    await chat.source.close();
    await _flush();
    expect(chat.closeCalls, 1);
    chat.closeGate.complete();
    observed.subscription.resume();
    await observed.done.future;
    await _flush();
    expect(observed.values, <String>['a']);
    expect(observed.errors, hasLength(1));
    expect(owner.hasActiveGeneration, isFalse);
    expect(owner.lastReceipt?.forwardedUtf8Bytes, gemmaReplyMaximumUtf8Bytes);
  });

  test('cancelCurrent drops paused text before exact cleanup', () async {
    final owner = _owner();
    final chat = _FakeChat(holdStop: true, holdClose: true);
    final observed = _ObservedStream(
      owner.generate(
        prompt: 'paused cancellation',
        createChat: _FakeFactory(chat).call,
      ),
    );
    await _flush();

    chat.addText('already delivered');
    observed.subscription.pause();
    chat.addText('queued private text');
    await _flush();

    owner.cancelCurrent();
    await _flush();
    expect(observed.values, <String>['already delivered']);
    expect(observed.errors, isEmpty);
    expect(chat.stopCalls, 1);
    expect(chat.source.hasListener, isTrue);

    chat.stopGate.complete();
    await chat.source.close();
    await _flush();
    expect(chat.closeCalls, 1);
    chat.closeGate.complete();
    await _flush();
    expect(owner.hasActiveGeneration, isFalse);

    observed.subscription.resume();
    await observed.done.future;
    expect(observed.values, <String>['already delivered']);
    expect(observed.errors, isEmpty);
    expect(chat.stopCalls, 1);
    expect(owner.lastReceipt?.outcome, GemmaGenerationOutcome.cancelled);
    expect(owner.lastReceipt?.exactlyReleased, isTrue);
  });

  test(
    'successor drops naturally released predecessor paused output',
    () async {
      final owner = _owner();
      final oldChat = _FakeChat();
      final oldObserved = _ObservedStream(
        owner.generate(prompt: 'old', createChat: _FakeFactory(oldChat).call),
      );
      var oldOutwardDone = false;
      unawaited(oldObserved.done.future.then((_) => oldOutwardDone = true));
      await _flush();
      oldChat.addText('delivered');
      oldObserved.subscription.pause();
      oldChat.addText('queued stale');
      await oldChat.source.close();
      await _flush();

      expect(oldObserved.values, <String>['delivered']);
      expect(oldOutwardDone, isFalse);
      expect(owner.hasActiveGeneration, isFalse);
      expect(oldChat.closeCalls, 1);

      final successorChat = _FakeChat();
      final successorFactory = _FakeFactory(successorChat);
      final successorObserved = _ObservedStream(
        owner.generate(prompt: 'new', createChat: successorFactory.call),
      );
      await _flush();
      expect(successorFactory.calls, 1);
      expect(successorChat.generateCalls, 1);

      oldObserved.subscription.resume();
      await oldObserved.done.future;
      expect(oldObserved.values, <String>['delivered']);
      expect(oldOutwardDone, isTrue);

      successorChat.addText('successor');
      await successorChat.source.close();
      await successorObserved.done.future;
      expect(successorObserved.values, <String>['successor']);
      expect(owner.isPoisoned, isFalse);
    },
  );

  test(
    'latest wins: third supersedes pending second with zero factory work',
    () async {
      final owner = _owner();
      final first = _FakeChat(holdStop: true, holdClose: true);
      final second = _FakeChat();
      final third = _FakeChat();
      final firstFactory = _FakeFactory(first);
      final secondFactory = _FakeFactory(second);
      final thirdFactory = _FakeFactory(third);

      final one = _ObservedStream(
        owner.generate(prompt: 'one', createChat: firstFactory.call),
      );
      await _flush();
      final two = _ObservedStream(
        owner.generate(prompt: 'two', createChat: secondFactory.call),
      );
      final three = _ObservedStream(
        owner.generate(prompt: 'three', createChat: thirdFactory.call),
      );
      await _flush();

      expect(first.stopCalls, 1);
      expect(secondFactory.calls, 0);
      expect(thirdFactory.calls, 0);
      expect(owner.hasPendingGeneration, isTrue);
      await two.done.future;

      first.stopGate.complete();
      await first.source.close();
      await _flush();
      expect(thirdFactory.calls, 0);
      first.closeGate.complete();
      await _flush();

      expect(secondFactory.calls, 0);
      expect(thirdFactory.calls, 1);
      expect(third.prompt, 'three');
      await _finishNaturally(third);
      await three.done.future;
      expect(owner.isPoisoned, isFalse);
      await one.done.future;
    },
  );

  test(
    'reentrant displaced onDone publishes the fourth pending winner',
    () async {
      final owner = _owner();
      final first = _FakeChat(holdStop: true, holdClose: true);
      final secondFactory = _FakeFactory(_FakeChat());
      final thirdFactory = _FakeFactory(_FakeChat());
      final fourth = _FakeChat();
      final fourthFactory = _FakeFactory(fourth);
      _ObservedStream? fourthObserved;

      final firstObserved = _ObservedStream(
        owner.generate(prompt: 'one', createChat: _FakeFactory(first).call),
      );
      await _flush();
      final secondObserved = _ObservedStream(
        owner.generate(prompt: 'two', createChat: secondFactory.call),
        onDone: () {
          fourthObserved = _ObservedStream(
            owner.generate(prompt: 'four', createChat: fourthFactory.call),
          );
        },
      );
      final thirdObserved = _ObservedStream(
        owner.generate(prompt: 'three', createChat: thirdFactory.call),
      );
      await _flush();

      expect(secondFactory.calls, 0);
      expect(thirdFactory.calls, 0);
      expect(fourthFactory.calls, 0);
      expect(owner.hasPendingGeneration, isTrue);
      await secondObserved.done.future;
      await thirdObserved.done.future;

      first.stopGate.complete();
      await first.source.close();
      await _flush();
      expect(fourthFactory.calls, 0);
      first.closeGate.complete();
      await _flush();

      expect(secondFactory.calls, 0);
      expect(thirdFactory.calls, 0);
      expect(fourthFactory.calls, 1);
      expect(fourth.prompt, 'four');
      await _finishNaturally(fourth);
      await fourthObserved?.done.future;
      await firstObserved.done.future;
      expect(owner.isPoisoned, isFalse);
    },
  );

  test(
    'cancelCurrent fences active plus pending and pending does zero work',
    () async {
      final owner = _owner();
      final active = _FakeChat(holdStop: true, holdClose: true);
      final pendingFactory = _FakeFactory(_FakeChat());
      final activeObserved = _ObservedStream(
        owner.generate(prompt: 'active', createChat: _FakeFactory(active).call),
      );
      await _flush();
      final pendingObserved = _ObservedStream(
        owner.generate(prompt: 'pending', createChat: pendingFactory.call),
      );
      await _flush();

      owner.cancelCurrent();
      await _flush();
      expect(pendingFactory.calls, 0);
      expect(owner.hasPendingGeneration, isFalse);
      expect(active.stopCalls, 1);
      await pendingObserved.done.future;

      active.stopGate.complete();
      await active.source.close();
      await _flush();
      active.closeGate.complete();
      await _flush();
      expect(owner.hasActiveGeneration, isFalse);
      await activeObserved.done.future;
    },
  );

  test(
    'cancel while factory is held fences output and blocks model successors',
    () async {
      final owner = _owner();
      final heldFactory = Completer<GemmaChatPort>();
      final activeChat = _FakeChat(holdClose: true);
      final activeFactory = _FakeFactory(activeChat, gate: heldFactory);
      final activeObserved = _ObservedStream(
        owner.generate(prompt: 'held factory', createChat: activeFactory.call),
      );
      await _flush();
      expect(activeFactory.calls, 1);

      owner.cancelCurrent();
      await activeObserved.done.future;
      final successorFactory = _FakeFactory(_FakeChat());
      final successorObserved = _ObservedStream(
        owner.generate(prompt: 'successor', createChat: successorFactory.call),
      );
      await _flush();
      expect(successorFactory.calls, 0);

      var ownerCloseSettled = false;
      final ownerClose = owner.close()..then((_) => ownerCloseSettled = true);
      await _flush();
      expect(ownerCloseSettled, isFalse);
      expect(successorFactory.calls, 0);
      await successorObserved.done.future;

      heldFactory.complete(activeChat);
      await _flush();
      expect(activeChat.addCalls, 0);
      expect(activeChat.generateCalls, 0);
      expect(activeChat.closeCalls, 1);
      activeChat.closeGate.complete();
      expect(await ownerClose, isTrue);
    },
  );

  test(
    'synchronous onData replacement fences reentrant old emissions',
    () async {
      final owner = _owner();
      final first = _FakeChat(holdStop: true, holdClose: true);
      final second = _FakeChat();
      final secondFactory = _FakeFactory(second);
      _ObservedStream? replacement;
      final firstObserved = _ObservedStream(
        owner.generate(prompt: 'first', createChat: _FakeFactory(first).call),
        onData: (_) {
          replacement = _ObservedStream(
            owner.generate(
              prompt: 'replacement',
              createChat: secondFactory.call,
            ),
          );
        },
      );
      await _flush();

      first.addText('trigger');
      first.addText('must-not-forward');
      await _flush();
      expect(firstObserved.values, <String>['trigger']);
      expect(secondFactory.calls, 0);
      expect(first.stopCalls, 1);

      first.stopGate.complete();
      await first.source.close();
      first.closeGate.complete();
      await _flush();
      expect(secondFactory.calls, 1);
      second.addText('new');
      await _finishNaturally(second);
      expect(replacement?.values, <String>['new']);
    },
  );

  test('prompt UTF-8 bound is checked before listening or factory work', () {
    final owner = _owner();
    final factory = _FakeFactory(_FakeChat());

    expect(
      () => owner.generate(prompt: '', createChat: factory.call),
      throwsA(isA<GemmaGenerationFailure>()),
    );
    expect(
      () => owner.generate(
        prompt: List<String>.filled(8193, 'é').join(),
        createChat: factory.call,
      ),
      throwsA(isA<GemmaGenerationFailure>()),
    );
    expect(factory.calls, 0);

    final allowed = owner.generate(
      prompt: List<String>.filled(gemmaPromptMaximumUtf8Bytes, 'a').join(),
      createChat: factory.call,
    );
    expect(allowed, isA<Stream<String>>());
    expect(factory.calls, 0);
  });

  test(
    'one oversized UTF-8 chunk is never forwarded and triggers cleanup',
    () async {
      final owner = _owner();
      final chat = _FakeChat(holdStop: true, holdClose: true);
      final observed = _ObservedStream(
        owner.generate(prompt: 'bounded', createChat: _FakeFactory(chat).call),
      );
      await _flush();
      chat.addText(List<String>.filled(2049, 'é').join());
      await _flush();

      expect(observed.values, isEmpty);
      expect(observed.errors, hasLength(1));
      expect(chat.stopCalls, 1);
      chat.stopGate.complete();
      await chat.source.close();
      await _flush();
      chat.closeGate.complete();
      await _flush();
      expect(owner.lastReceipt?.forwardedChunks, 0);
      expect(owner.lastReceipt?.forwardedUtf8Bytes, 0);
    },
  );

  test(
    'source-event and aggregate-byte limits fence the first excess event',
    () async {
      final countOwner = _owner();
      final countChat = _FakeChat();
      final countObserved = _ObservedStream(
        countOwner.generate(
          prompt: 'count',
          createChat: _FakeFactory(countChat).call,
        ),
      );
      await _flush();
      for (var i = 0; i < gemmaReplyMaximumSourceEvents; i++) {
        countChat.addNonText();
      }
      countChat.addText('excess');
      await _flush();
      expect(countObserved.values, isEmpty);
      expect(countObserved.errors, hasLength(1));
      await countChat.source.close();
      await _flush();
      expect(
        countOwner.lastReceipt?.observedSourceEvents,
        gemmaReplyMaximumSourceEvents,
      );
      expect(countOwner.lastReceipt?.forwardedChunks, 0);

      final bytesOwner = _owner();
      final bytesChat = _FakeChat();
      final bytesObserved = _ObservedStream(
        bytesOwner.generate(
          prompt: 'bytes',
          createChat: _FakeFactory(bytesChat).call,
        ),
      );
      await _flush();
      final fullChunk = List<String>.filled(
        gemmaChunkMaximumUtf8Bytes,
        'a',
      ).join();
      for (var i = 0; i < 16; i++) {
        bytesChat.addText(fullChunk);
      }
      bytesChat.addText('excess');
      await _flush();
      expect(bytesObserved.values, hasLength(16));
      expect(bytesObserved.errors, hasLength(1));
      await bytesChat.source.close();
      await _flush();
      expect(
        bytesOwner.lastReceipt?.forwardedUtf8Bytes,
        gemmaReplyMaximumUtf8Bytes,
      );
    },
  );

  test('factory failure is retained as same-isolate owner poison', () async {
    final owner = _owner();
    final factory = _FakeFactory(
      _FakeChat(),
      error: StateError('factory failed'),
    );
    final observed = _ObservedStream(
      owner.generate(prompt: 'factory', createChat: factory.call),
    );
    await observed.done.future;
    await _flush();

    expect(factory.calls, 1);
    expect(observed.errors, hasLength(1));
    expect(owner.isPoisoned, isTrue);
    expect(owner.hasActiveGeneration, isFalse);
    final successor = _FakeFactory(_FakeChat());
    final rejected = _ObservedStream(
      owner.generate(prompt: 'blocked', createChat: successor.call),
    );
    await rejected.done.future;
    expect(successor.calls, 0);
    expect(rejected.errors, hasLength(1));
  });

  for (final mode in <String>['generate', 'listen']) {
    test(
      '$mode ambiguity stops/closes but retains poisoned ownership',
      () async {
        final owner = _owner();
        final chat = _FakeChat(
          generateThrows: mode == 'generate',
          listenThrows: mode == 'listen',
        );
        final observed = _ObservedStream(
          owner.generate(prompt: mode, createChat: _FakeFactory(chat).call),
        );
        await observed.done.future;
        await _flush();

        expect(observed.errors, hasLength(1));
        expect(chat.stopCalls, 1);
        expect(chat.closeCalls, 1);
        expect(owner.isPoisoned, isTrue);
        expect(owner.lastReceipt?.generationEntered, isTrue);
        expect(owner.lastReceipt?.exactlyReleased, isFalse);
      },
    );
  }

  test('failed chat close retains exact poisoned ownership', () async {
    final owner = _owner();
    final chat = _FakeChat(holdClose: true);
    final observed = _ObservedStream(
      owner.generate(
        prompt: 'close fails',
        createChat: _FakeFactory(chat).call,
      ),
    );
    await chat.source.close();
    await _flush();
    chat.closeGate.completeError(StateError('close failed'));
    await observed.done.future;
    await _flush();

    expect(owner.isPoisoned, isTrue);
    expect(owner.hasActiveGeneration, isFalse);
    expect(owner.lastReceipt?.chatCloseAttempted, isTrue);
    expect(owner.lastReceipt?.chatCloseSucceeded, isFalse);
    expect(owner.lastReceipt?.exactlyReleased, isFalse);
  });

  test(
    'deadline fences output and poisons the owner even after cleanup',
    () async {
      final timers = _TimerHarness();
      final owner = _owner(timers: timers);
      final chat = _FakeChat(holdStop: true, holdClose: true);
      final observed = _ObservedStream(
        owner.generate(prompt: 'deadline', createChat: _FakeFactory(chat).call),
      );
      await _flush();
      expect(timers.timers, hasLength(1));
      expect(timers.durations, <Duration>[const Duration(seconds: 1)]);
      timers.timers.single.fire();
      chat.addText('late');
      await _flush();

      expect(observed.values, isEmpty);
      expect(observed.errors, hasLength(1));
      expect(owner.isPoisoned, isTrue);
      expect(chat.stopCalls, 1);
      chat.stopGate.complete();
      await chat.source.close();
      await _flush();
      chat.closeGate.complete();
      await observed.done.future;
      await _flush();
      expect(
        owner.lastReceipt?.outcome,
        GemmaGenerationOutcome.deadlineExceeded,
      );
      expect(owner.isPoisoned, isTrue);
    },
  );

  test(
    'deadline while chat factory is held retains the late exact chat',
    () async {
      final timers = _TimerHarness();
      final owner = _owner(timers: timers);
      final factoryGate = Completer<GemmaChatPort>();
      final chat = _FakeChat(holdClose: true);
      final factory = _FakeFactory(chat, gate: factoryGate);
      final observed = _ObservedStream(
        owner.generate(
          prompt: 'held factory deadline',
          createChat: factory.call,
        ),
      );
      await _flush();
      expect(factory.calls, 1);

      timers.timers.single.fire();
      await observed.done.future;
      expect(observed.errors, hasLength(1));
      expect(owner.isPoisoned, isTrue);
      expect(owner.retainsUncertainGeneration, isTrue);
      expect(owner.lastReceipt?.exactlyReleased, isFalse);

      factoryGate.complete(chat);
      await _flush();
      expect(chat.addCalls, 0);
      expect(chat.generateCalls, 0);
      expect(chat.stopCalls, 0);
      expect(chat.closeCalls, 1);
      chat.closeGate.complete();
      await _flush();
      expect(owner.isPoisoned, isTrue);
      expect(owner.retainsUncertainGeneration, isTrue);
    },
  );

  test(
    'deadline while natural chat close is held poisons permanently',
    () async {
      final timers = _TimerHarness();
      final owner = _owner(timers: timers);
      final chat = _FakeChat(holdClose: true);
      final observed = _ObservedStream(
        owner.generate(
          prompt: 'held close deadline',
          createChat: _FakeFactory(chat).call,
        ),
      );
      var outwardDone = false;
      unawaited(observed.done.future.then((_) => outwardDone = true));
      await _flush();

      await chat.source.close();
      await _flush();
      expect(chat.closeCalls, 1);
      expect(outwardDone, isFalse);
      expect(observed.errors, isEmpty);
      expect(chat.stopCalls, 0);

      timers.timers.single.fire();
      await observed.done.future;
      expect(observed.errors, hasLength(1));
      expect(outwardDone, isTrue);
      expect(owner.isPoisoned, isTrue);
      expect(
        owner.lastReceipt?.outcome,
        GemmaGenerationOutcome.deadlineExceeded,
      );
      expect(owner.lastReceipt?.sourceTerminalObserved, isTrue);
      expect(owner.lastReceipt?.chatCloseSucceeded, isFalse);
      expect(owner.lastReceipt?.exactlyReleased, isFalse);
      expect(chat.stopCalls, 0);

      chat.closeGate.complete();
      await _flush();
      expect(owner.isPoisoned, isTrue);
      expect(owner.retainsUncertainGeneration, isTrue);
      expect(owner.lastReceipt?.exactlyReleased, isFalse);
    },
  );

  test(
    'deadline during held prompt admission does not enter generation',
    () async {
      final timers = _TimerHarness();
      final owner = _owner(timers: timers);
      final chat = _FakeChat(holdAdd: true, holdClose: true);
      final observed = _ObservedStream(
        owner.generate(prompt: 'held add', createChat: _FakeFactory(chat).call),
      );
      await _flush();
      expect(chat.addCalls, 1);
      expect(chat.generateCalls, 0);

      timers.timers.single.fire();
      await observed.done.future;
      expect(observed.errors, hasLength(1));
      expect(owner.isPoisoned, isTrue);
      expect(chat.stopCalls, 0);
      final successor = _FakeFactory(_FakeChat());
      final rejected = _ObservedStream(
        owner.generate(prompt: 'blocked', createChat: successor.call),
      );
      await rejected.done.future;
      expect(successor.calls, 0);

      chat.addGate.complete();
      await _flush();
      expect(chat.generateCalls, 0);
      expect(chat.closeCalls, 1);
      chat.closeGate.complete();
      await _flush();
      expect(owner.retainsUncertainGeneration, isTrue);
    },
  );

  test('owner close fences admission and waits active exact cleanup', () async {
    final owner = _owner();
    final active = _FakeChat(holdStop: true, holdClose: true);
    final activeObserved = _ObservedStream(
      owner.generate(prompt: 'active', createChat: _FakeFactory(active).call),
    );
    await _flush();
    final pendingFactory = _FakeFactory(_FakeChat());
    final pendingObserved = _ObservedStream(
      owner.generate(prompt: 'pending', createChat: pendingFactory.call),
    );
    await _flush();

    var closeSettled = false;
    final firstClose = owner.close()..then((_) => closeSettled = true);
    expect(identical(firstClose, owner.close()), isTrue);
    await _flush();
    expect(closeSettled, isFalse);
    expect(pendingFactory.calls, 0);
    await pendingObserved.done.future;

    active.stopGate.complete();
    await active.source.close();
    await _flush();
    expect(closeSettled, isFalse);
    active.closeGate.complete();
    expect(await firstClose, isTrue);
    expect(owner.isClosed, isTrue);

    final rejectedFactory = _FakeFactory(_FakeChat());
    final rejected = _ObservedStream(
      owner.generate(prompt: 'after close', createChat: rejectedFactory.call),
    );
    await rejected.done.future;
    expect(rejectedFactory.calls, 0);
    expect(rejected.errors, hasLength(1));
    await activeObserved.done.future;
  });
}
