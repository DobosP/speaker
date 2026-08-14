// Deterministic UI-isolate/app-lifetime admission tests for mobile speech
// output. The production singleton covers every shipped UI speech path; it is
// not an operating-system-wide or arbitrary multi-engine/multi-isolate lock.
//
// These tests use only the pure-Dart registry. They construct no Flutter
// plugin, model, isolate, file, network, audio, or device resource.
import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:speaker_mobile/tts_process_owner.dart';

void main() {
  test('active owner makes a successor immediately busy with zero work',
      () async {
    final registry = TtsProcessOwnerRegistry();
    final first = registry.tryAcquire();
    expect(first, isNotNull);
    expect(first!.admitsWork, isTrue);

    var successorWork = 0;
    final second = registry.tryAcquire();
    if (second != null) successorWork++;

    expect(second, isNull);
    expect(successorWork, 0);
    expect(await first.close(() async => true), isTrue);
    expect(first.admitsWork, isFalse);
  });

  test('exact registry check rejects a foreign lease', () async {
    final canonical = TtsProcessOwnerRegistry();
    final foreignRegistry = TtsProcessOwnerRegistry();
    final canonicalLease = canonical.tryAcquire()!;
    final foreignLease = foreignRegistry.tryAcquire()!;

    expect(canonical.holdsExact(canonicalLease), isTrue);
    expect(canonical.ownsExact(canonicalLease), isTrue);
    expect(canonical.holdsExact(foreignLease), isFalse);
    expect(canonical.ownsExact(foreignLease), isFalse);
    expect(foreignRegistry.ownsExact(canonicalLease), isFalse);

    canonicalLease.revoke();
    expect(canonical.holdsExact(canonicalLease), isTrue);
    expect(canonical.ownsExact(canonicalLease), isFalse);
    expect(await canonicalLease.close(() async => true), isTrue);
    final replacement = canonical.tryAcquire()!;
    expect(canonical.holdsExact(canonicalLease), isFalse);
    expect(canonical.holdsExact(replacement), isTrue);
    expect(await replacement.close(() async => true), isTrue);
    expect(await foreignLease.close(() async => true), isTrue);
  });

  test('close revokes synchronously and cleanup pending remains busy',
      () async {
    final registry = TtsProcessOwnerRegistry();
    final owner = registry.tryAcquire()!;
    final cleanupEntered = Completer<void>();
    final cleanup = Completer<bool>();

    final closing = owner.close(() {
      cleanupEntered.complete();
      return cleanup.future;
    });

    expect(owner.revoked, isTrue);
    expect(owner.admitsWork, isFalse);
    expect(registry.tryAcquire(), isNull);
    await cleanupEntered.future;
    expect(registry.tryAcquire(), isNull);

    cleanup.complete(true);
    expect(await closing, isTrue);
    final successor = registry.tryAcquire();
    expect(successor, isNotNull);
    expect(await successor!.close(() async => true), isTrue);
  });

  test('widget-style revoke closes the pre-cleanup scheduling gap', () async {
    final registry = TtsProcessOwnerRegistry();
    final owner = registry.tryAcquire()!;

    owner.revoke();
    owner.revoke();
    expect(owner.revoked, isTrue);
    expect(owner.admitsWork, isFalse);
    expect(registry.tryAcquire(), isNull);

    expect(await owner.close(() async => true), isTrue);
    final replacement = registry.tryAcquire();
    expect(replacement, isNotNull);
    expect(await replacement!.close(() async => true), isTrue);
  });

  test('failed, thrown, and ambiguous cleanup retain process poison', () async {
    Future<void> expectRetained(
      Future<bool> Function() cleanup, {
      required bool completes,
    }) async {
      final registry = TtsProcessOwnerRegistry();
      final owner = registry.tryAcquire()!;
      final closing = owner.close(cleanup);

      if (completes) expect(await closing, isFalse);
      await Future<void>.delayed(Duration.zero);
      expect(owner.revoked, isTrue);
      expect(registry.poisoned, isTrue);
      expect(registry.tryAcquire(), isNull);
    }

    await expectRetained(() async => false, completes: true);
    await expectRetained(
      () => throw StateError('cleanup threw synchronously'),
      completes: true,
    );
    await expectRetained(
      () => Future<bool>.error(StateError('cleanup future failed')),
      completes: true,
    );

    final never = Completer<bool>();
    final registry = TtsProcessOwnerRegistry();
    final owner = registry.tryAcquire()!;
    final closing = owner.close(() => never.future);
    var completed = false;
    unawaited(closing.then((_) => completed = true));
    await Future<void>.delayed(Duration.zero);
    expect(owner.revoked, isTrue);
    expect(registry.tryAcquire(), isNull);
    expect(completed, isFalse);
  });

  test('close is memoized before a reentrant cleanup callback runs', () async {
    final registry = TtsProcessOwnerRegistry();
    final owner = registry.tryAcquire()!;
    late Future<bool> reentrant;
    var cleanupCalls = 0;

    final first = owner.close(() {
      cleanupCalls++;
      reentrant = owner.close(() async {
        cleanupCalls++;
        return false;
      });
      return Future<bool>.value(true);
    });
    final second = owner.close(() async {
      cleanupCalls++;
      return false;
    });

    expect(identical(first, second), isTrue);
    expect(identical(first, reentrant), isTrue);
    expect(await first, isTrue);
    expect(cleanupCalls, 1);
  });

  test('acquire cannot win until exact successful release completes', () async {
    final registry = TtsProcessOwnerRegistry();
    final owner = registry.tryAcquire()!;
    final release = Completer<bool>();
    final closing = owner.close(() => release.future);

    final attempts = <TtsProcessLease?>[];
    scheduleMicrotask(() => attempts.add(registry.tryAcquire()));
    await Future<void>.delayed(Duration.zero);
    attempts.add(registry.tryAcquire());
    expect(attempts, everyElement(isNull));

    release.complete(true);
    expect(await closing, isTrue);
    attempts.add(registry.tryAcquire());
    expect(attempts.last, isNotNull);
    expect(await attempts.last!.close(() async => true), isTrue);
  });

  test('foreign and stale exact leases cannot release the current owner',
      () async {
    final firstRegistry = TtsProcessOwnerRegistry();
    final foreignRegistry = TtsProcessOwnerRegistry();
    final first = firstRegistry.tryAcquire()!;
    final foreign = foreignRegistry.tryAcquire()!;

    expect(await foreign.close(() async => true), isTrue);
    expect(firstRegistry.tryAcquire(), isNull);

    expect(await first.close(() async => true), isTrue);
    final current = firstRegistry.tryAcquire()!;
    // A stale lease can only observe its memoized receipt. It cannot target or
    // release the newer exact owner in the registry it used to own.
    expect(await first.close(() async => false), isTrue);
    expect(firstRegistry.tryAcquire(), isNull);

    expect(await current.close(() async => true), isTrue);
  });

  test('manual path runs no construction while another lease is retained',
      () async {
    final registry = TtsProcessOwnerRegistry();
    final assistant = registry.tryAcquire()!;
    final cleanup = Completer<bool>();
    final closing = assistant.close(() => cleanup.future);

    var modelConstructions = 0;
    var playerConstructions = 0;
    var synthCalls = 0;
    var playCalls = 0;
    final manual = registry.tryAcquire();
    if (manual != null) {
      modelConstructions++;
      playerConstructions++;
      synthCalls++;
      playCalls++;
    }

    expect(manual, isNull);
    expect(
      [modelConstructions, playerConstructions, synthCalls, playCalls],
      everyElement(0),
    );

    cleanup.complete(true);
    expect(await closing, isTrue);
    final admitted = registry.tryAcquire();
    expect(admitted, isNotNull);
    expect(await admitted!.close(() async => true), isTrue);
  });

  test('entered synthesis keeps process ownership after widget revocation',
      () async {
    final registry = TtsProcessOwnerRegistry();
    final assistant = registry.tryAcquire()!;
    final synthReturned = Completer<void>();

    // Revocation is synchronous, but exact cleanup must include settlement of
    // entered synthesis rather than relying only on an empty clip stop fence.
    final closing = assistant.close(() async {
      await synthReturned.future;
      return true;
    });
    expect(assistant.revoked, isTrue);
    expect(registry.tryAcquire(), isNull);

    await Future<void>.delayed(Duration.zero);
    expect(registry.tryAcquire(), isNull);
    synthReturned.complete();
    expect(await closing, isTrue);

    final replacement = registry.tryAcquire();
    expect(replacement, isNotNull);
    expect(await replacement!.close(() async => true), isTrue);
  });

  test('synthesis timeout or unknown native return retains ownership',
      () async {
    final registry = TtsProcessOwnerRegistry();
    final assistant = registry.tryAcquire()!;

    // `false` is the conservative receipt for a request whose caller timed out
    // without proof that the worker/native synthesis actually returned.
    expect(await assistant.close(() async => false), isFalse);
    expect(assistant.revoked, isTrue);
    expect(registry.poisoned, isTrue);
    expect(registry.tryAcquire(), isNull);
  });
}
