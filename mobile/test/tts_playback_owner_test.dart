// Deterministic lifecycle tests for the pure-Dart mobile TTS playback owner.
// Every asynchronous boundary is driven by a Completer. These tests load no
// player plugin, model, device, timer, or network resource.
import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/tts_playback_owner.dart';

final class _FakeClip {
  _FakeClip({this.onCleanup}) {
    handle = TtsPlaybackClip(
      started: started.future,
      terminal: terminal.future,
      stopAndRelease: _stopAndRelease,
    );
  }

  final void Function()? onCleanup;
  final Completer<TtsPlaybackResult> started = Completer<TtsPlaybackResult>();
  final Completer<TtsPlaybackTerminal> terminal =
      Completer<TtsPlaybackTerminal>();
  final Completer<TtsPlaybackResult> cleanup = Completer<TtsPlaybackResult>();
  final Completer<String> created = Completer<String>();
  final Completer<void> cleanupEntered = Completer<void>();
  late final TtsPlaybackClip handle;
  int cleanupCalls = 0;

  TtsPlaybackClip create(String path) {
    if (created.isCompleted) {
      throw StateError('fake clip was created more than once');
    }
    created.complete(path);
    return handle;
  }

  Future<TtsPlaybackResult> _stopAndRelease() {
    cleanupCalls++;
    onCleanup?.call();
    if (!cleanupEntered.isCompleted) cleanupEntered.complete();
    return cleanup.future;
  }

  void completeStartSuccess({Object? diagnostic}) {
    started.complete(
      TtsPlaybackResult.success(
        error: diagnostic,
        stackTrace: diagnostic == null ? null : StackTrace.current,
      ),
    );
  }

  void completeStartFailure(Object error) {
    started.complete(TtsPlaybackResult.failure(error, StackTrace.current));
  }

  void completeTerminal() {
    terminal.complete(const TtsPlaybackTerminal.completed());
  }

  void completeTerminalFailure(Object error) {
    terminal.complete(TtsPlaybackTerminal.failed(error, StackTrace.current));
  }

  void completeTerminalFutureError(Object error) {
    terminal.completeError(error, StackTrace.current);
  }

  void completeCleanupSuccess({Object? diagnostic}) {
    cleanup.complete(
      TtsPlaybackResult.success(
        error: diagnostic,
        stackTrace: diagnostic == null ? null : StackTrace.current,
      ),
    );
  }

  void completeCleanupFailure(Object error) {
    cleanup.complete(TtsPlaybackResult.failure(error, StackTrace.current));
  }
}

final class _FakeClipFactory {
  _FakeClipFactory(this.clips);

  final List<_FakeClip> clips;
  final List<String> paths = <String>[];
  int _next = 0;

  TtsPlaybackClip call(String path) {
    paths.add(path);
    if (_next >= clips.length) {
      throw StateError('unexpected clip creation for $path');
    }
    return clips[_next++].create(path);
  }
}

_FakeClip _readyClip({void Function()? onCleanup}) {
  final clip = _FakeClip(onCleanup: onCleanup);
  clip.completeStartSuccess();
  clip.completeCleanupSuccess();
  return clip;
}

Future<void> _closeClean(TtsPlaybackOwner owner) async {
  expect(await owner.close(), isTrue);
  await owner.whenIdle();
}

void main() {
  test('activity distinguishes an idle generation edge from true to false',
      () async {
    final synthEntered = Completer<void>();
    final synthRelease = Completer<String?>();
    final activity = <(TtsPlaybackGeneration, bool)>[];
    final factory = _FakeClipFactory([]);

    final owner = TtsPlaybackOwner(
      synthesize: (text) {
        synthEntered.complete();
        return synthRelease.future;
      },
      createPlaybackClip: factory.call,
      onActivityChanged: (generation, speaking) {
        activity.add((generation, speaking));
      },
    );
    final foreignOwner = TtsPlaybackOwner(
      synthesize: (text) async => null,
      createPlaybackClip: _FakeClipFactory([]).call,
    );

    expect(owner.enqueue(foreignOwner.generation, 'foreign'), isFalse);
    expect(await owner.waitForStop(foreignOwner.generation), isFalse);

    final idleGeneration = owner.supersede();
    expect(activity, [(idleGeneration, false)]);
    expect(owner.speaking, isFalse);

    expect(owner.enqueue(idleGeneration, 'blocked'), isTrue);
    await synthEntered.future;
    expect(activity.last, (idleGeneration, true));

    final replacement = owner.supersede();
    expect(activity.last, (replacement, false));
    synthRelease.complete('/stale.wav');
    await owner.whenIdle();
    expect(factory.paths, isEmpty);

    await _closeClean(owner);
    await _closeClean(foreignOwner);
  });

  test('one pump discards blocked old synthesis and drains the new generation',
      () async {
    final oldSynthEntered = Completer<void>();
    final oldSynthResult = Completer<String?>();
    final newClip = _readyClip();
    final factory = _FakeClipFactory([newClip]);
    final playbackStarted = Completer<void>();
    final synthCalls = <String>[];
    final activity = <(TtsPlaybackGeneration, bool)>[];

    late final TtsPlaybackOwner owner;
    owner = TtsPlaybackOwner(
      synthesize: (text) {
        synthCalls.add(text);
        if (text == 'old sentence') {
          oldSynthEntered.complete();
          return oldSynthResult.future;
        }
        if (text == 'new sentence') return Future.value('/new.wav');
        throw StateError('unexpected synthesis: $text');
      },
      createPlaybackClip: factory.call,
      onActivityChanged: (generation, speaking) {
        activity.add((generation, speaking));
      },
      onPlaybackStarted: (generation) => playbackStarted.complete(),
    );

    final oldGeneration = owner.generation;
    expect(owner.enqueue(oldGeneration, 'old sentence'), isTrue);
    await oldSynthEntered.future;

    final newGeneration = owner.supersede();
    expect(owner.enqueue(newGeneration, 'new sentence'), isTrue);
    expect(owner.snapshot.pumpRunning, isTrue);
    expect(owner.snapshot.queued, 1);

    oldSynthResult.complete('/old.wav');
    expect(await newClip.created.future, '/new.wav');
    await playbackStarted.future;

    expect(synthCalls, ['old sentence', 'new sentence']);
    expect(factory.paths, ['/new.wav']);
    expect(owner.speaking, isTrue);
    expect(owner.snapshot.pumpRunning, isTrue);
    expect(activity.last, (newGeneration, true));

    newClip.completeTerminal();
    await newClip.cleanupEntered.future;
    await owner.whenIdle();
    expect(owner.speaking, isFalse);
    expect(owner.snapshot.pumpRunning, isFalse);

    await _closeClean(owner);
  });

  test('replacement play waits for the exact delayed clip-cleanup fence',
      () async {
    final oldClip = _FakeClip()..completeStartSuccess();
    final newClip = _readyClip();
    final factory = _FakeClipFactory([oldClip, newClip]);
    final playbackSignals = [Completer<void>(), Completer<void>()];
    var playbackCount = 0;

    final owner = TtsPlaybackOwner(
      synthesize: (text) async => '/$text.wav',
      createPlaybackClip: factory.call,
      onPlaybackStarted: (generation) {
        playbackSignals[playbackCount++].complete();
      },
    );

    expect(owner.enqueue(owner.generation, 'old'), isTrue);
    await playbackSignals[0].future;

    final replacement = owner.supersede();
    expect(owner.enqueue(replacement, 'new'), isTrue);
    await oldClip.cleanupEntered.future;
    expect(newClip.created.isCompleted, isFalse);
    expect(owner.snapshot.physicalActive, isTrue);
    expect(owner.speaking, isTrue);

    oldClip.completeCleanupSuccess();
    expect(await owner.waitForStop(replacement), isTrue);
    expect(await newClip.created.future, '/new.wav');
    await playbackSignals[1].future;

    newClip.completeTerminal();
    await newClip.cleanupEntered.future;
    await owner.whenIdle();
    await _closeClean(owner);
  });

  test('failed cleanup stays conservative, poisons, and blocks replacement',
      () async {
    final oldClip = _FakeClip()..completeStartSuccess();
    final replacementClip = _readyClip();
    final factory = _FakeClipFactory([oldClip, replacementClip]);
    final playbackStarted = Completer<void>();
    final errors = <Object>[];
    final synthCalls = <String>[];

    final owner = TtsPlaybackOwner(
      synthesize: (text) async {
        synthCalls.add(text);
        return '/$text.wav';
      },
      createPlaybackClip: factory.call,
      onPlaybackStarted: (generation) => playbackStarted.complete(),
      onError: (generation, error, stackTrace) => errors.add(error),
    );

    expect(owner.enqueue(owner.generation, 'old'), isTrue);
    await playbackStarted.future;

    final interrupted = owner.interrupt();
    final replacement = owner.generation;
    expect(owner.enqueue(replacement, 'replacement'), isTrue);
    await oldClip.cleanupEntered.future;
    expect(owner.snapshot.physicalActive, isTrue);
    expect(owner.speaking, isTrue);

    oldClip.completeCleanupFailure(StateError('release failed'));
    expect(await interrupted, isFalse);
    await owner.whenIdle();

    expect(owner.snapshot.poisoned, isTrue);
    expect(owner.snapshot.physicalActive, isTrue);
    expect(owner.speaking, isTrue);
    expect(owner.enqueue(owner.generation, 'late'), isFalse);
    expect(replacementClip.created.isCompleted, isFalse);
    expect(synthCalls, ['old']);

    // Close reuses the exact failed cleanup receipt; it cannot pretend that a
    // second in-process stop proved silence.
    expect(await owner.close(), isFalse);
    expect(await owner.close(), isFalse);
    await owner.whenIdle();
    expect(oldClip.cleanupCalls, 1);
    expect(errors, isNotEmpty);
  });

  test('interrupt during blocked start queues exact cleanup after start',
      () async {
    final clip = _FakeClip();
    final factory = _FakeClipFactory([clip]);
    final playbackNotifications = <TtsPlaybackGeneration>[];

    final owner = TtsPlaybackOwner(
      synthesize: (text) async => '/blocked.wav',
      createPlaybackClip: factory.call,
      onPlaybackStarted: playbackNotifications.add,
    );

    final oldGeneration = owner.generation;
    expect(owner.enqueue(oldGeneration, 'sentence'), isTrue);
    expect(await clip.created.future, '/blocked.wav');

    // The synchronous handle already owns possible physical activity even
    // though its native start Future has not returned.
    expect(owner.snapshot.physicalActive, isTrue);
    final interrupted = owner.interrupt();
    expect(clip.cleanupEntered.isCompleted, isFalse);

    clip.completeStartSuccess();
    await clip.cleanupEntered.future;
    expect(playbackNotifications, isEmpty);
    expect(owner.speaking, isTrue);

    clip.completeCleanupSuccess();
    expect(await interrupted, isTrue);
    await owner.whenIdle();
    expect(clip.cleanupCalls, 1);
    expect(owner.snapshot.physicalActive, isFalse);
    expect(owner.speaking, isFalse);

    await _closeClean(owner);
  });

  test('late terminal from an interrupted clip cannot finish a fresh clip',
      () async {
    final oldClip = _readyClip();
    final newFirstClip = _readyClip();
    final newSecondClip = _readyClip();
    final factory = _FakeClipFactory([
      oldClip,
      newFirstClip,
      newSecondClip,
    ]);
    final playbackSignals = [
      Completer<void>(),
      Completer<void>(),
      Completer<void>(),
    ];
    var playbackCount = 0;

    final owner = TtsPlaybackOwner(
      synthesize: (text) async => '/$text.wav',
      createPlaybackClip: factory.call,
      onPlaybackStarted: (generation) {
        playbackSignals[playbackCount++].complete();
      },
    );

    expect(owner.enqueue(owner.generation, 'old'), isTrue);
    await playbackSignals[0].future;
    expect(await owner.interrupt(), isTrue);

    final replacement = owner.generation;
    expect(owner.enqueue(replacement, 'new-first'), isTrue);
    expect(owner.enqueue(replacement, 'new-second'), isTrue);
    await playbackSignals[1].future;

    oldClip.completeTerminal();
    expect(
      (await oldClip.handle.terminal).kind,
      TtsPlaybackTerminalKind.completed,
    );
    expect(newSecondClip.created.isCompleted, isFalse);
    expect(owner.snapshot.queued, 1);

    newFirstClip.completeTerminal();
    await newFirstClip.cleanupEntered.future;
    await playbackSignals[2].future;
    expect(await newSecondClip.created.future, '/new-second.wav');

    newSecondClip.completeTerminal();
    await newSecondClip.cleanupEntered.future;
    await owner.whenIdle();
    expect(factory.paths, ['/old.wav', '/new-first.wav', '/new-second.wav']);

    await _closeClean(owner);
  });

  test('terminal Future error before start returns is normalized and closed',
      () async {
    final clip = _FakeClip()..completeCleanupSuccess();
    final unusedClip = _readyClip();
    final factory = _FakeClipFactory([clip, unusedClip]);
    final errors = <Object>[];
    final synthCalls = <String>[];

    final owner = TtsPlaybackOwner(
      synthesize: (text) async {
        synthCalls.add(text);
        return '/$text.wav';
      },
      createPlaybackClip: factory.call,
      onError: (generation, error, stackTrace) => errors.add(error),
    );

    final failedGeneration = owner.generation;
    expect(owner.enqueue(failedGeneration, 'broken'), isTrue);
    expect(owner.enqueue(failedGeneration, 'must-drop'), isTrue);
    await clip.created.future;

    final terminalError = StateError('terminal failed before start returned');
    clip.completeTerminalFutureError(terminalError);
    final normalized = await clip.handle.terminal;
    expect(normalized.kind, TtsPlaybackTerminalKind.failed);
    expect(normalized.error, same(terminalError));

    clip.completeStartSuccess();
    await clip.cleanupEntered.future;
    await owner.whenIdle();

    expect(owner.isCurrent(failedGeneration), isFalse);
    expect(owner.snapshot.physicalActive, isFalse);
    expect(owner.speaking, isFalse);
    expect(synthCalls, ['broken']);
    expect(unusedClip.created.isCompleted, isFalse);
    expect(
        errors.where((error) => identical(error, terminalError)), hasLength(1));
    expect(clip.cleanupCalls, 1);

    await _closeClean(owner);
  });

  test('stale enqueue and post-barge synthesis cannot resume old speech',
      () async {
    final synthEntered = Completer<void>();
    final synthRelease = Completer<String?>();
    final synthCalls = <String>[];
    final factory = _FakeClipFactory([]);

    final owner = TtsPlaybackOwner(
      synthesize: (text) {
        synthCalls.add(text);
        synthEntered.complete();
        return synthRelease.future;
      },
      createPlaybackClip: factory.call,
    );

    final staleGeneration = owner.generation;
    expect(owner.enqueue(staleGeneration, 'in-flight'), isTrue);
    expect(owner.enqueue(staleGeneration, 'queued-behind-it'), isTrue);
    await synthEntered.future;
    expect(owner.snapshot.queued, 1);

    final currentGeneration = owner.supersede();
    expect(owner.snapshot.queued, 0);
    expect(owner.enqueue(staleGeneration, 'late-token-sentence'), isFalse);
    expect(owner.isCurrent(currentGeneration), isTrue);

    synthRelease.complete('/stale.wav');
    await owner.whenIdle();

    expect(synthCalls, ['in-flight']);
    expect(factory.paths, isEmpty);
    expect(owner.snapshot.queued, 0);
    expect(owner.snapshot.pumpRunning, isFalse);
    expect(owner.speaking, isFalse);

    await _closeClean(owner);
  });

  test('FIFO pump admits at most one synthesis and one exact clip at a time',
      () async {
    final texts = ['one', 'two', 'three'];
    final synthEntered = <String, Completer<void>>{
      for (final text in texts) text: Completer<void>(),
    };
    final synthRelease = <String, Completer<String?>>{
      for (final text in texts) text: Completer<String?>(),
    };
    final playbackSignals = <Completer<void>>[
      Completer<void>(),
      Completer<void>(),
      Completer<void>(),
    ];
    final synthCalls = <String>[];
    var activeSyntheses = 0;
    var maxActiveSyntheses = 0;
    var activeClips = 0;
    var maxActiveClips = 0;
    var playbackCount = 0;

    void clipCleaned() => activeClips--;

    final clips = [
      _readyClip(onCleanup: clipCleaned),
      _readyClip(onCleanup: clipCleaned),
      _readyClip(onCleanup: clipCleaned),
    ];
    final factory = _FakeClipFactory(clips);

    final owner = TtsPlaybackOwner(
      synthesize: (text) async {
        synthCalls.add(text);
        activeSyntheses++;
        if (activeSyntheses > maxActiveSyntheses) {
          maxActiveSyntheses = activeSyntheses;
        }
        synthEntered[text]!.complete();
        try {
          return await synthRelease[text]!.future;
        } finally {
          activeSyntheses--;
        }
      },
      createPlaybackClip: (path) {
        activeClips++;
        if (activeClips > maxActiveClips) maxActiveClips = activeClips;
        return factory.call(path);
      },
      onPlaybackStarted: (generation) {
        playbackSignals[playbackCount++].complete();
      },
    );

    final generation = owner.generation;
    for (final text in texts) {
      expect(owner.enqueue(generation, text), isTrue);
    }
    await synthEntered['one']!.future;
    expect(synthCalls, ['one']);

    synthRelease['one']!.complete('/one.wav');
    await playbackSignals[0].future;
    expect(synthCalls, ['one']);
    clips[0].completeTerminal();
    await clips[0].cleanupEntered.future;

    await synthEntered['two']!.future;
    expect(synthCalls, ['one', 'two']);
    synthRelease['two']!.complete('/two.wav');
    await playbackSignals[1].future;
    clips[1].completeTerminal();
    await clips[1].cleanupEntered.future;

    await synthEntered['three']!.future;
    expect(synthCalls, ['one', 'two', 'three']);
    synthRelease['three']!.complete('/three.wav');
    await playbackSignals[2].future;
    clips[2].completeTerminal();
    await clips[2].cleanupEntered.future;

    await owner.whenIdle();
    expect(factory.paths, ['/one.wav', '/two.wav', '/three.wav']);
    expect(maxActiveSyntheses, 1);
    expect(maxActiveClips, 1);
    expect(activeSyntheses, 0);
    expect(activeClips, 0);

    await _closeClean(owner);
  });

  test('null and throwing synthesis continue to the next FIFO item', () async {
    final goodClip = _readyClip();
    final factory = _FakeClipFactory([goodClip]);
    final playbackStarted = Completer<void>();
    final synthCalls = <String>[];
    final errors = <Object>[];

    final owner = TtsPlaybackOwner(
      synthesize: (text) async {
        synthCalls.add(text);
        if (text == 'null-result') return null;
        if (text == 'throws') throw StateError('synthesis failed');
        return '/good.wav';
      },
      createPlaybackClip: factory.call,
      onPlaybackStarted: (generation) => playbackStarted.complete(),
      onError: (generation, error, stackTrace) => errors.add(error),
    );

    final generation = owner.generation;
    expect(owner.enqueue(generation, 'null-result'), isTrue);
    expect(owner.enqueue(generation, 'throws'), isTrue);
    expect(owner.enqueue(generation, 'good'), isTrue);

    await playbackStarted.future;
    expect(synthCalls, ['null-result', 'throws', 'good']);
    expect(factory.paths, ['/good.wav']);
    expect(errors, hasLength(1));
    expect(errors.single, isA<StateError>());

    goodClip.completeTerminal();
    await goodClip.cleanupEntered.future;
    await owner.whenIdle();
    await _closeClean(owner);
  });

  test('terminal failure revokes and cleans the exact generation', () async {
    final brokenClip = _readyClip();
    final unusedClip = _readyClip();
    final factory = _FakeClipFactory([brokenClip, unusedClip]);
    final playbackStarted = Completer<void>();
    final synthCalls = <String>[];
    final errors = <Object>[];

    final owner = TtsPlaybackOwner(
      synthesize: (text) async {
        synthCalls.add(text);
        return '/$text.wav';
      },
      createPlaybackClip: factory.call,
      onPlaybackStarted: (generation) => playbackStarted.complete(),
      onError: (generation, error, stackTrace) => errors.add(error),
    );

    final failedGeneration = owner.generation;
    expect(owner.enqueue(failedGeneration, 'broken'), isTrue);
    expect(owner.enqueue(failedGeneration, 'must-drop'), isTrue);
    await playbackStarted.future;

    final terminalError = StateError('decoder failed');
    brokenClip.completeTerminalFailure(terminalError);
    await brokenClip.cleanupEntered.future;
    await owner.whenIdle();

    expect(owner.isCurrent(failedGeneration), isFalse);
    expect(owner.generation.ordinal, failedGeneration.ordinal + 1);
    expect(owner.enqueue(failedGeneration, 'late'), isFalse);
    expect(synthCalls, ['broken']);
    expect(unusedClip.created.isCompleted, isFalse);
    expect(
        errors.where((error) => identical(error, terminalError)), hasLength(1));
    expect(brokenClip.cleanupCalls, 1);
    expect(owner.speaking, isFalse);

    await _closeClean(owner);
  });

  test('failed start remains physical until exact cleanup succeeds', () async {
    final clip = _FakeClip();
    final unusedClip = _readyClip();
    final factory = _FakeClipFactory([clip, unusedClip]);
    final synthCalls = <String>[];
    final errors = <Object>[];

    final owner = TtsPlaybackOwner(
      synthesize: (text) async {
        synthCalls.add(text);
        return '/$text.wav';
      },
      createPlaybackClip: factory.call,
      onError: (generation, error, stackTrace) => errors.add(error),
    );

    final failedGeneration = owner.generation;
    expect(owner.enqueue(failedGeneration, 'broken'), isTrue);
    expect(owner.enqueue(failedGeneration, 'must-drop'), isTrue);
    await clip.created.future;

    final startError = StateError('native start failed');
    clip.completeStartFailure(startError);
    await clip.cleanupEntered.future;
    expect(owner.snapshot.physicalActive, isTrue);
    expect(owner.speaking, isTrue);

    clip.completeCleanupSuccess();
    await owner.whenIdle();

    expect(owner.isCurrent(failedGeneration), isFalse);
    expect(owner.snapshot.physicalActive, isFalse);
    expect(owner.speaking, isFalse);
    expect(synthCalls, ['broken']);
    expect(unusedClip.created.isCompleted, isFalse);
    expect(errors.where((error) => identical(error, startError)), hasLength(1));
    expect(clip.cleanupCalls, 1);

    await _closeClean(owner);
  });

  test('failed start plus failed cleanup poisons retained physical state',
      () async {
    final clip = _FakeClip();
    final factory = _FakeClipFactory([clip]);

    final owner = TtsPlaybackOwner(
      synthesize: (text) async => '/broken.wav',
      createPlaybackClip: factory.call,
    );

    expect(owner.enqueue(owner.generation, 'broken'), isTrue);
    await clip.created.future;
    clip.completeStartFailure(StateError('start failed'));
    await clip.cleanupEntered.future;
    clip.completeCleanupFailure(StateError('cleanup failed'));
    await owner.whenIdle();

    expect(owner.snapshot.poisoned, isTrue);
    expect(owner.snapshot.physicalActive, isTrue);
    expect(owner.speaking, isTrue);
    expect(owner.enqueue(owner.generation, 'rejected'), isFalse);
    expect(await owner.close(), isFalse);
    expect(clip.cleanupCalls, 1);
  });

  test('successful cleanup can carry a recovered stop diagnostic', () async {
    final oldClip = _FakeClip()..completeStartSuccess();
    final newClip = _readyClip();
    final factory = _FakeClipFactory([oldClip, newClip]);
    final playbackSignals = [Completer<void>(), Completer<void>()];
    final errors = <Object>[];
    var playbackCount = 0;

    final owner = TtsPlaybackOwner(
      synthesize: (text) async => '/$text.wav',
      createPlaybackClip: factory.call,
      onPlaybackStarted: (generation) {
        playbackSignals[playbackCount++].complete();
      },
      onError: (generation, error, stackTrace) => errors.add(error),
    );

    expect(owner.enqueue(owner.generation, 'old'), isTrue);
    await playbackSignals[0].future;
    final interrupted = owner.interrupt();
    await oldClip.cleanupEntered.future;

    final stopDiagnostic = StateError('stop failed; dispose succeeded');
    oldClip.completeCleanupSuccess(diagnostic: stopDiagnostic);
    expect(await interrupted, isTrue);
    expect(owner.snapshot.poisoned, isFalse);
    expect(owner.snapshot.physicalActive, isFalse);
    expect(errors.where((error) => identical(error, stopDiagnostic)),
        hasLength(1));

    expect(owner.enqueue(owner.generation, 'new'), isTrue);
    await playbackSignals[1].future;
    newClip.completeTerminal();
    await newClip.cleanupEntered.future;
    await owner.whenIdle();

    await _closeClean(owner);
  });

  test('repeated supersedes share one cleanup and one active-generation report',
      () async {
    final clip = _FakeClip()..completeStartSuccess();
    final factory = _FakeClipFactory([clip]);
    final playbackStarted = Completer<void>();
    final reports = <(TtsPlaybackGeneration, Object)>[];

    final owner = TtsPlaybackOwner(
      synthesize: (text) async => '/shared.wav',
      createPlaybackClip: factory.call,
      onPlaybackStarted: (generation) => playbackStarted.complete(),
      onError: (generation, error, stackTrace) {
        reports.add((generation, error));
      },
    );

    final activeGeneration = owner.generation;
    expect(owner.enqueue(activeGeneration, 'shared'), isTrue);
    await playbackStarted.future;

    final firstReplacement = owner.supersede();
    await clip.cleanupEntered.future;
    final secondReplacement = owner.supersede();
    final diagnostic = StateError('stop failed; exact dispose succeeded');
    clip.completeCleanupSuccess(diagnostic: diagnostic);

    expect(await owner.waitForStop(firstReplacement), isTrue);
    expect(await owner.waitForStop(secondReplacement), isTrue);
    await owner.whenIdle();

    expect(clip.cleanupCalls, 1);
    expect(reports, [(activeGeneration, diagnostic)]);
    expect(owner.snapshot.poisoned, isFalse);
    expect(owner.snapshot.physicalActive, isFalse);

    await _closeClean(owner);
  });

  test('natural-terminal and interrupt race share exactly one cleanup',
      () async {
    final clip = _FakeClip()..completeStartSuccess();
    final factory = _FakeClipFactory([clip]);
    final playbackStarted = Completer<void>();

    final owner = TtsPlaybackOwner(
      synthesize: (text) async => '/race.wav',
      createPlaybackClip: factory.call,
      onPlaybackStarted: (generation) => playbackStarted.complete(),
    );

    expect(owner.enqueue(owner.generation, 'race'), isTrue);
    await playbackStarted.future;

    clip.completeTerminal();
    final interrupted = owner.interrupt();
    await clip.cleanupEntered.future;
    expect(clip.cleanupCalls, 1);

    clip.completeCleanupSuccess();
    expect(await interrupted, isTrue);
    await owner.whenIdle();
    expect(clip.cleanupCalls, 1);

    await _closeClean(owner);
    expect(clip.cleanupCalls, 1);
  });

  test('natural completion proves silence even when disposal poisons owner',
      () async {
    final clip = _FakeClip()..completeStartSuccess();
    final factory = _FakeClipFactory([clip]);
    final playbackStarted = Completer<void>();

    final owner = TtsPlaybackOwner(
      synthesize: (text) async => '/natural.wav',
      createPlaybackClip: factory.call,
      onPlaybackStarted: (generation) => playbackStarted.complete(),
    );

    expect(owner.enqueue(owner.generation, 'natural'), isTrue);
    await playbackStarted.future;
    clip.completeTerminal();
    await clip.cleanupEntered.future;

    // The exact natural terminal has already proved audible silence while the
    // still-pending disposal keeps the pump logically occupied.
    expect(owner.snapshot.physicalActive, isFalse);
    clip.completeCleanupFailure(StateError('dispose failed'));
    await owner.whenIdle();

    expect(owner.snapshot.poisoned, isTrue);
    expect(owner.snapshot.physicalActive, isFalse);
    expect(owner.speaking, isFalse);
    expect(owner.enqueue(owner.generation, 'rejected'), isFalse);
    expect(await owner.close(), isFalse);
    expect(clip.cleanupCalls, 1);
  });

  test('close revokes authority and discards synthesis that returns late',
      () async {
    final synthEntered = Completer<void>();
    final synthRelease = Completer<String?>();
    final factory = _FakeClipFactory([]);

    final owner = TtsPlaybackOwner(
      synthesize: (text) {
        synthEntered.complete();
        return synthRelease.future;
      },
      createPlaybackClip: factory.call,
    );

    final staleGeneration = owner.generation;
    expect(owner.enqueue(staleGeneration, 'in-flight'), isTrue);
    await synthEntered.future;

    expect(await owner.close(), isTrue);
    expect(owner.snapshot.closed, isTrue);
    expect(owner.isCurrent(staleGeneration), isFalse);
    expect(owner.isCurrent(owner.generation), isFalse);
    expect(owner.enqueue(owner.generation, 'after-close'), isFalse);
    expect(await owner.interrupt(), isFalse);
    expect(owner.supersede, throwsStateError);
    expect(await owner.close(), isTrue);

    synthRelease.complete('/late.wav');
    await owner.whenIdle();
    expect(factory.paths, isEmpty);
    expect(owner.snapshot.pumpRunning, isFalse);
    expect(owner.speaking, isFalse);
  });
}
