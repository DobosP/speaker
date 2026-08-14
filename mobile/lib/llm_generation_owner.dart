import 'dart:async';
import 'dart:collection';
import 'dart:convert';

const int gemmaPromptMaximumUtf8Bytes = 16 * 1024;
const int gemmaChunkMaximumUtf8Bytes = 4 * 1024;
const int gemmaReplyMaximumUtf8Bytes = 64 * 1024;
const int gemmaReplyMaximumSourceEvents = 2048;
const Duration gemmaGenerationMaximumLifetime = Duration(seconds: 120);

const int _maximumSafeOrdinal = 0x1fffffffffffff;

typedef GemmaChatFactory = Future<GemmaChatPort> Function();
typedef GemmaTimerFactory = Timer Function(
  Duration duration,
  void Function() callback,
);

/// One raw response observed from the plugin-owned generation stream.
///
/// Non-text responses are retained as events so the owner can enforce one
/// bound across every response exposed by its adapter rather than only across
/// displayed text. The adapter may itself filter native/plugin callbacks.
final class GemmaGenerationEvent {
  const GemmaGenerationEvent.text(String value) : text = value;
  const GemmaGenerationEvent.nonText() : text = null;

  final String? text;
}

abstract interface class GemmaChatPort {
  Future<void> addPrompt(String prompt);

  Stream<GemmaGenerationEvent> generate();

  Future<void> stopGeneration();

  Future<void> close();
}

enum GemmaGenerationOutcome {
  completed,
  cancelled,
  superseded,
  deadlineExceeded,
  outputLimitExceeded,
  generationFailed,
  startupAmbiguous,
  ownerClosed,
}

final class GemmaGenerationFailure implements Exception {
  const GemmaGenerationFailure(this.code);

  final String code;

  @override
  String toString() => 'GemmaGenerationFailure($code)';
}

final class GemmaGenerationReceipt {
  const GemmaGenerationReceipt({
    required this.ordinal,
    required this.outcome,
    required this.generationEntered,
    required this.explicitStopAttempted,
    required this.explicitStopSucceeded,
    required this.sourceTerminalObserved,
    required this.chatCloseAttempted,
    required this.chatCloseSucceeded,
    required this.observedSourceEvents,
    required this.forwardedChunks,
    required this.forwardedUtf8Bytes,
    required this.exactlyReleased,
  });

  final int ordinal;
  final GemmaGenerationOutcome outcome;
  final bool generationEntered;
  final bool explicitStopAttempted;
  final bool explicitStopSucceeded;
  final bool sourceTerminalObserved;
  final bool chatCloseAttempted;
  final bool chatCloseSucceeded;
  final int observedSourceEvents;
  final int forwardedChunks;
  final int forwardedUtf8Bytes;
  final bool exactlyReleased;
}

/// Owns every shipped Gemma reply in the canonical Flutter UI isolate.
///
/// The owner deliberately permits one active native chat and one latest-wins
/// pending request. A newer listener revokes the active generation and replaces
/// any older pending request, but it cannot construct a chat until the active
/// source reaches its real terminal and that exact chat closes successfully.
final class GemmaGenerationOwner {
  GemmaGenerationOwner._({
    required Duration maximumLifetime,
    required GemmaTimerFactory timerFactory,
  })  : _maximumLifetime = maximumLifetime,
        _timerFactory = timerFactory;

  static final GemmaGenerationOwner shared = GemmaGenerationOwner._(
    maximumLifetime: gemmaGenerationMaximumLifetime,
    timerFactory: Timer.new,
  );

  /// Creates an isolated owner for deterministic fake-only tests.
  factory GemmaGenerationOwner.forTesting({
    Duration maximumLifetime = gemmaGenerationMaximumLifetime,
    GemmaTimerFactory timerFactory = Timer.new,
  }) {
    if (maximumLifetime <= Duration.zero) {
      throw ArgumentError.value(maximumLifetime, 'maximumLifetime');
    }
    return GemmaGenerationOwner._(
      maximumLifetime: maximumLifetime,
      timerFactory: timerFactory,
    );
  }

  final Duration _maximumLifetime;
  final GemmaTimerFactory _timerFactory;

  _OwnedGemmaGeneration? _active;
  _OwnedGemmaGeneration? _pending;
  _OwnedGemmaGeneration? _latestOutward;
  _OwnedGemmaGeneration? _retainedPoison;
  GemmaGenerationReceipt? _lastReceipt;
  Future<bool>? _closeFuture;
  int _nextOrdinal = 0;
  bool _closed = false;
  bool _poisoned = false;

  bool get isClosed => _closed;
  bool get isPoisoned => _poisoned;
  bool get hasActiveGeneration => _active != null;
  bool get hasPendingGeneration => _pending != null;
  bool get retainsUncertainGeneration => _retainedPoison != null;
  GemmaGenerationReceipt? get lastReceipt => _lastReceipt;

  Stream<String> generate({
    required String prompt,
    required GemmaChatFactory createChat,
  }) {
    _validatePrompt(prompt);
    return _OwnedGemmaGeneration(
      owner: this,
      prompt: prompt,
      createChat: createChat,
    ).stream;
  }

  /// Synchronously fences the active and pending reply, if any.
  ///
  /// This is an admission fence only. Exact release still waits for the active
  /// source terminal and exact chat close.
  void cancelCurrent() {
    _latestOutward?._cancelOutward();
    final pending = _pending;
    if (pending != null) {
      _pending = null;
      pending._finishWithoutNative(GemmaGenerationOutcome.cancelled);
    }
    _active?._revoke(GemmaGenerationOutcome.cancelled);
  }

  Future<bool> close() {
    final existing = _closeFuture;
    if (existing != null) {
      return existing;
    }
    final completer = Completer<bool>();
    _closeFuture = completer.future;
    unawaited(_runClose(completer));
    return completer.future;
  }

  Future<void> _runClose(Completer<bool> completer) async {
    var released = false;
    try {
      released = await _closeOwned();
    } catch (_) {
      _poisoned = true;
    }
    if (!completer.isCompleted) {
      completer.complete(released);
    }
  }

  Future<bool> _closeOwned() async {
    _closed = true;
    _latestOutward?._cancelOutward();
    final pending = _pending;
    if (pending != null) {
      _pending = null;
      pending._finishWithoutNative(GemmaGenerationOutcome.ownerClosed);
    }
    final active = _active;
    if (active != null) {
      active._revoke(GemmaGenerationOutcome.ownerClosed);
      await active.done;
    }
    return !_poisoned && _active == null;
  }

  void _listen(_OwnedGemmaGeneration generation) {
    if (_closed) {
      _replaceLatestOutward(generation);
      generation._rejectBeforeAdmission('owner_closed');
      return;
    }
    if (_poisoned) {
      _replaceLatestOutward(generation);
      generation._rejectBeforeAdmission('owner_poisoned');
      return;
    }
    if (_nextOrdinal >= _maximumSafeOrdinal) {
      _poisoned = true;
      _replaceLatestOutward(generation);
      generation._rejectBeforeAdmission('ordinal_exhausted');
      return;
    }
    generation._admit(
      ++_nextOrdinal,
      _timerFactory(_maximumLifetime, generation._deadlineExpired),
    );

    final active = _active;
    if (active == null) {
      _active = generation;
      _replaceLatestOutward(generation);
      generation._start();
      return;
    }

    final displaced = _pending;
    // Publish the newcomer before closing the displaced stream. Its synchronous
    // onDone callback may listen to an even newer request; that reentrant
    // request must become the one pending winner rather than be overwritten by
    // this older stack frame.
    _pending = generation;
    _replaceLatestOutward(generation);
    if (displaced != null) {
      displaced._finishWithoutNative(GemmaGenerationOutcome.superseded);
    }
    active._revoke(GemmaGenerationOutcome.superseded);
  }

  void _replaceLatestOutward(_OwnedGemmaGeneration generation) {
    final previous = _latestOutward;
    _latestOutward = generation;
    if (previous != null && !identical(previous, generation)) {
      previous._supersedeOutward();
    }
  }

  void _outwardFinished(_OwnedGemmaGeneration generation) {
    if (identical(_latestOutward, generation)) {
      _latestOutward = null;
    }
  }

  bool _isActive(_OwnedGemmaGeneration generation) =>
      identical(_active, generation);

  void _pendingRevoked(
    _OwnedGemmaGeneration generation,
    GemmaGenerationOutcome outcome,
  ) {
    if (!identical(_pending, generation)) {
      return;
    }
    _pending = null;
    generation._finishWithoutNative(outcome);
  }

  void _generationFinished(
    _OwnedGemmaGeneration generation,
    GemmaGenerationReceipt receipt,
  ) {
    if (!identical(_active, generation)) {
      _poisoned = true;
      _retainedPoison = generation;
      final pending = _pending;
      _pending = null;
      pending?._rejectBeforeAdmission('owner_poisoned');
      return;
    }

    _lastReceipt = receipt;
    _active = null;
    if (!receipt.exactlyReleased) {
      _poisoned = true;
      _retainedPoison = generation;
      final pending = _pending;
      _pending = null;
      pending?._rejectBeforeAdmission('owner_poisoned');
      return;
    }

    if (_closed) {
      final pending = _pending;
      _pending = null;
      pending?._finishWithoutNative(GemmaGenerationOutcome.ownerClosed);
      return;
    }

    final next = _pending;
    _pending = null;
    if (next != null && !next._isSettled) {
      _active = next;
      next._start();
    }
  }

  static void _validatePrompt(String prompt) {
    if (prompt.isEmpty || prompt.length > gemmaPromptMaximumUtf8Bytes) {
      throw const GemmaGenerationFailure('invalid_prompt');
    }
    final bytes = utf8.encode(prompt).length;
    if (bytes == 0 || bytes > gemmaPromptMaximumUtf8Bytes) {
      throw const GemmaGenerationFailure('invalid_prompt');
    }
  }
}

final class _OwnedGemmaGeneration {
  _OwnedGemmaGeneration({
    required GemmaGenerationOwner owner,
    required String prompt,
    required GemmaChatFactory createChat,
  })  : _owner = owner,
        _prompt = prompt,
        _createChat = createChat {
    late final StreamController<String> controller;
    controller = StreamController<String>(
      sync: true,
      onListen: _onListen,
      onPause: _onPause,
      onResume: _onResume,
      onCancel: _onCancel,
    );
    _controller = controller;
  }

  final GemmaGenerationOwner _owner;
  late final StreamController<String> _controller;
  final Completer<bool> _done = Completer<bool>();
  final Completer<void> _sourceTerminal = Completer<void>();
  final ListQueue<String> _pausedOutput = ListQueue<String>();

  String? _prompt;
  GemmaChatFactory? _createChat;
  GemmaChatPort? _chat;
  String? _queuedFailureCode;
  StreamSubscription<GemmaGenerationEvent>? _sourceSubscription;
  Timer? _deadline;
  Future<void>? _cleanupFuture;
  Future<void>? _stopFuture;
  Future<bool>? _chatCloseFuture;

  int _ordinal = 0;
  int _observedSourceEvents = 0;
  int _forwardedChunks = 0;
  int _forwardedUtf8Bytes = 0;
  bool _admitted = false;
  bool _started = false;
  bool _listened = false;
  bool _revoked = false;
  bool _outwardFenced = false;
  bool _generationEntered = false;
  bool _stopRequired = false;
  bool _explicitStopAttempted = false;
  bool _explicitStopSucceeded = false;
  bool _sourceTerminalObserved = false;
  bool _chatCloseAttempted = false;
  bool _chatCloseSucceeded = false;
  bool _outwardAddInProgress = false;
  bool _outwardCloseStarted = false;
  bool _consumerPaused = false;
  bool _naturalOutputComplete = false;
  bool _settled = false;
  GemmaGenerationOutcome _outcome = GemmaGenerationOutcome.completed;

  Stream<String> get stream => _controller.stream;
  Future<bool> get done => _done.future;
  bool get _isSettled => _settled;

  void _onListen() {
    if (_listened) {
      return;
    }
    _listened = true;
    _owner._listen(this);
  }

  void _onPause() {
    // `await for` may pause its subscription while executing the loop body.
    // Do not translate ordinary backpressure into native cancellation. The
    // upstream remains independently subscribed and the controller's retained
    // output is bounded by the source-event and UTF-8 envelopes above.
    _consumerPaused = true;
  }

  void _onResume() {
    _consumerPaused = false;
    if (_outwardFenced) {
      _deliverQueuedFailureAndClose();
      return;
    }
    _flushPausedOutput();
  }

  void _onCancel() {
    _cancelOutward();
    _revoke(GemmaGenerationOutcome.cancelled);
  }

  void _cancelOutward() {
    _fenceOutward(
      dropBufferedOutput: true,
      dropQueuedFailure: true,
    );
  }

  void _supersedeOutward() {
    _fenceOutward(
      dropBufferedOutput: true,
      dropQueuedFailure: true,
    );
  }

  void _admit(int ordinal, Timer deadline) {
    _ordinal = ordinal;
    _deadline = deadline;
    _admitted = true;
  }

  void _start() {
    if (_started || _settled) {
      return;
    }
    _started = true;
    unawaited(_startOwned());
  }

  Future<void> _startOwned() async {
    final factory = _createChat;
    if (factory == null) {
      _finishPoisoned(GemmaGenerationOutcome.startupAmbiguous);
      return;
    }

    GemmaChatPort chat;
    try {
      chat = await factory();
      _chat = chat;
      _createChat = null;
    } catch (_) {
      _emitFailure('chat_factory_ambiguous');
      _finishPoisoned(GemmaGenerationOutcome.startupAmbiguous);
      return;
    }

    if (_revoked) {
      await _closeBeforeGeneration();
      return;
    }

    final prompt = _prompt;
    if (prompt == null) {
      _emitFailure('prompt_unavailable');
      await _closeBeforeGeneration(
        outcome: GemmaGenerationOutcome.generationFailed,
      );
      return;
    }
    try {
      await chat.addPrompt(prompt);
    } catch (_) {
      _emitFailure('prompt_admission_failed');
      await _closeBeforeGeneration(
        outcome: GemmaGenerationOutcome.generationFailed,
      );
      return;
    }

    if (_revoked) {
      await _closeBeforeGeneration();
      return;
    }

    _generationEntered = true;
    Stream<GemmaGenerationEvent> source;
    try {
      source = chat.generate();
    } catch (_) {
      _emitFailure('generation_start_ambiguous');
      await _closeAmbiguousGeneration();
      return;
    }
    try {
      _sourceSubscription = source.listen(
        _onSourceData,
        onError: _onSourceError,
        onDone: _onSourceDone,
        cancelOnError: false,
      );
    } catch (_) {
      _emitFailure('generation_listen_ambiguous');
      await _closeAmbiguousGeneration();
      return;
    }

    if (_revoked) {
      _stopRequired = !_sourceTerminalObserved;
      _scheduleGenerationCleanup();
    }
  }

  void _onSourceData(GemmaGenerationEvent event) {
    if (_settled) {
      return;
    }

    if (_observedSourceEvents >= gemmaReplyMaximumSourceEvents) {
      if (!_revoked) {
        _failAndRevoke(
          'reply_event_limit_exceeded',
          GemmaGenerationOutcome.outputLimitExceeded,
        );
      }
      return;
    }
    _observedSourceEvents += 1;

    final chunk = event.text;
    if (chunk == null || _revoked || _outwardFenced) {
      return;
    }
    final nextChunks = _forwardedChunks + 1;
    if (chunk.length > gemmaChunkMaximumUtf8Bytes) {
      _failAndRevoke(
        'reply_limit_exceeded',
        GemmaGenerationOutcome.outputLimitExceeded,
      );
      return;
    }

    final chunkBytes = utf8.encode(chunk).length;
    if (chunkBytes > gemmaChunkMaximumUtf8Bytes ||
        _forwardedUtf8Bytes > gemmaReplyMaximumUtf8Bytes - chunkBytes) {
      _failAndRevoke(
        'reply_limit_exceeded',
        GemmaGenerationOutcome.outputLimitExceeded,
      );
      return;
    }

    _forwardedChunks = nextChunks;
    _forwardedUtf8Bytes += chunkBytes;
    if (_consumerPaused) {
      _pausedOutput.addLast(chunk);
      return;
    }
    _addOutward(chunk);
  }

  void _addOutward(String chunk) {
    if (_outwardFenced) {
      return;
    }
    _outwardAddInProgress = true;
    try {
      _controller.add(chunk);
    } catch (_) {
      _revoke(GemmaGenerationOutcome.cancelled);
    } finally {
      _outwardAddInProgress = false;
      if (_outwardFenced) {
        _closeOutward();
      }
    }
  }

  void _flushPausedOutput() {
    while (!_consumerPaused && !_outwardFenced && _pausedOutput.isNotEmpty) {
      _addOutward(_pausedOutput.removeFirst());
    }
    if (!_consumerPaused &&
        !_outwardFenced &&
        _naturalOutputComplete &&
        _pausedOutput.isEmpty) {
      _fenceOutward();
    }
  }

  void _onSourceError(Object _error, StackTrace _stackTrace) {
    if (_settled) {
      return;
    }
    _failAndRevoke(
      'generation_failed',
      GemmaGenerationOutcome.generationFailed,
    );
  }

  void _onSourceDone() {
    if (_sourceTerminalObserved) {
      return;
    }
    _sourceTerminalObserved = true;
    if (!_sourceTerminal.isCompleted) {
      _sourceTerminal.complete();
    }
    if (!_revoked) {
      _outcome = GemmaGenerationOutcome.completed;
    }
    _scheduleGenerationCleanup();
  }

  void _deadlineExpired() {
    if (_settled) {
      return;
    }
    if (_sourceTerminalObserved) {
      _revoked = true;
      _outcome = GemmaGenerationOutcome.deadlineExceeded;
      _emitFailure('generation_deadline_exceeded');
      _fenceOutward();
    } else {
      _failAndRevoke(
        'generation_deadline_exceeded',
        GemmaGenerationOutcome.deadlineExceeded,
      );
    }
    if (!_settled) {
      // A code-owned deadline cannot fabricate source-terminal or close proof.
      // Retain the exact generation and refuse replacement even if its
      // best-effort cleanup later returns.
      _finishPoisoned(GemmaGenerationOutcome.deadlineExceeded);
    }
  }

  void _revoke(GemmaGenerationOutcome outcome) {
    if (_settled || _sourceTerminalObserved) {
      return;
    }
    _latchRevocation(outcome);
    _fenceOutward();
    _continueRevocation();
  }

  void _failAndRevoke(String code, GemmaGenerationOutcome outcome) {
    if (_settled || _sourceTerminalObserved) {
      return;
    }
    _latchRevocation(outcome);
    // Latch the outcome before the synchronous error callback can start a
    // replacement. The callback may re-enter this owner, but it cannot rewrite
    // the failure that already won this generation.
    _emitFailure(code);
    _fenceOutward();
    _continueRevocation();
  }

  void _latchRevocation(GemmaGenerationOutcome outcome) {
    if (!_revoked) {
      _revoked = true;
      _outcome = outcome;
    } else if (_outcome == GemmaGenerationOutcome.cancelled &&
        outcome != GemmaGenerationOutcome.cancelled) {
      _outcome = outcome;
    }
  }

  void _continueRevocation() {
    if (!_started) {
      if (_owner._isActive(this)) {
        _settle(exactlyReleased: true);
      } else {
        _owner._pendingRevoked(this, _outcome);
      }
      return;
    }
    if (_generationEntered && _sourceSubscription != null) {
      _stopRequired = true;
      _scheduleGenerationCleanup();
    }
  }

  void _scheduleGenerationCleanup() {
    if (_cleanupFuture != null) {
      return;
    }
    final completer = Completer<void>();
    _cleanupFuture = completer.future;
    unawaited(_runGenerationCleanup(completer));
  }

  Future<void> _runGenerationCleanup(Completer<void> completer) async {
    try {
      await _cleanupGeneration();
    } catch (_) {
      _finishPoisoned(GemmaGenerationOutcome.generationFailed);
    } finally {
      if (!completer.isCompleted) {
        completer.complete();
      }
    }
  }

  Future<void> _cleanupGeneration() async {
    final chat = _chat;
    if (chat == null) {
      _finishPoisoned(GemmaGenerationOutcome.startupAmbiguous);
      return;
    }

    if (_stopRequired) {
      await _attemptExplicitStop(chat);
    }

    await _sourceTerminal.future;
    final closed = await _closeChat(chat);
    if (!closed && !_revoked) {
      _outcome = GemmaGenerationOutcome.generationFailed;
      _emitFailure('chat_close_failed');
    }
    _settle(exactlyReleased: closed);
  }

  Future<void> _closeBeforeGeneration({
    GemmaGenerationOutcome? outcome,
  }) async {
    if (outcome != null) {
      _outcome = outcome;
    }
    final chat = _chat;
    if (chat == null) {
      _finishPoisoned(GemmaGenerationOutcome.startupAmbiguous);
      return;
    }
    final closed = await _closeChat(chat);
    _settle(exactlyReleased: closed);
  }

  Future<void> _closeAmbiguousGeneration() async {
    final chat = _chat;
    if (chat != null) {
      await _attemptExplicitStop(chat);
      await _closeChat(chat);
    }
    _finishPoisoned(GemmaGenerationOutcome.startupAmbiguous);
  }

  Future<void> _attemptExplicitStop(GemmaChatPort chat) {
    final existing = _stopFuture;
    if (existing != null) {
      return existing;
    }
    final completer = Completer<void>();
    _stopFuture = completer.future;
    _explicitStopAttempted = true;
    unawaited(_runExplicitStop(chat, completer));
    return completer.future;
  }

  Future<void> _runExplicitStop(
    GemmaChatPort chat,
    Completer<void> completer,
  ) async {
    try {
      await chat.stopGeneration();
      _explicitStopSucceeded = true;
    } catch (_) {
      _explicitStopSucceeded = false;
    } finally {
      if (!completer.isCompleted) {
        completer.complete();
      }
    }
  }

  Future<bool> _closeChat(GemmaChatPort chat) {
    final existing = _chatCloseFuture;
    if (existing != null) {
      return existing;
    }
    final completer = Completer<bool>();
    _chatCloseFuture = completer.future;
    _chatCloseAttempted = true;
    unawaited(_runChatClose(chat, completer));
    return completer.future;
  }

  Future<void> _runChatClose(
    GemmaChatPort chat,
    Completer<bool> completer,
  ) async {
    try {
      await chat.close();
      _chatCloseSucceeded = true;
    } catch (_) {
      _chatCloseSucceeded = false;
    } finally {
      if (!completer.isCompleted) {
        completer.complete(_chatCloseSucceeded);
      }
    }
  }

  void _finishWithoutNative(GemmaGenerationOutcome outcome) {
    if (_settled) {
      return;
    }
    if (!_revoked) {
      _outcome = outcome;
    }
    _fenceOutward();
    _deadline?.cancel();
    _deadline = null;
    _prompt = null;
    _createChat = null;
    _settled = true;
    if (!_done.isCompleted) {
      _done.complete(true);
    }
  }

  void _rejectBeforeAdmission(String code) {
    if (_settled) {
      return;
    }
    scheduleMicrotask(() {
      if (_settled) {
        return;
      }
      _emitFailure(code);
      _finishWithoutNative(GemmaGenerationOutcome.ownerClosed);
    });
  }

  void _finishPoisoned(GemmaGenerationOutcome outcome) {
    if (_settled) {
      return;
    }
    _outcome = outcome;
    _settle(exactlyReleased: false);
  }

  void _settle({required bool exactlyReleased}) {
    if (_settled) {
      return;
    }
    _settled = true;
    _deadline?.cancel();
    _deadline = null;
    final completeOutwardNaturally = exactlyReleased &&
        !_revoked &&
        _outcome == GemmaGenerationOutcome.completed;
    if (completeOutwardNaturally) {
      _naturalOutputComplete = true;
    } else {
      _fenceOutward();
    }
    _prompt = null;
    _createChat = null;
    if (exactlyReleased) {
      _chat = null;
      _sourceSubscription = null;
    }
    final receipt = GemmaGenerationReceipt(
      ordinal: _ordinal,
      outcome: _outcome,
      generationEntered: _generationEntered,
      explicitStopAttempted: _explicitStopAttempted,
      explicitStopSucceeded: _explicitStopSucceeded,
      sourceTerminalObserved: _sourceTerminalObserved,
      chatCloseAttempted: _chatCloseAttempted,
      chatCloseSucceeded: _chatCloseSucceeded,
      observedSourceEvents: _observedSourceEvents,
      forwardedChunks: _forwardedChunks,
      forwardedUtf8Bytes: _forwardedUtf8Bytes,
      exactlyReleased: exactlyReleased,
    );
    if (!_done.isCompleted) {
      _done.complete(exactlyReleased);
    }
    if (_admitted) {
      _owner._generationFinished(this, receipt);
    }
    if (completeOutwardNaturally) {
      _flushPausedOutput();
    }
  }

  void _emitFailure(String code) {
    if (_outwardFenced || _controller.isClosed) {
      return;
    }
    if (_consumerPaused) {
      _queuedFailureCode ??= code;
      return;
    }
    _addOutwardFailure(code);
  }

  void _addOutwardFailure(String code) {
    _outwardAddInProgress = true;
    try {
      _controller.addError(GemmaGenerationFailure(code));
    } catch (_) {
      // The consumer may have cancelled between the fence check and addError.
    } finally {
      _outwardAddInProgress = false;
      if (_outwardFenced) {
        _closeOutward();
      }
    }
  }

  void _fenceOutward({
    bool dropBufferedOutput = false,
    bool dropQueuedFailure = false,
  }) {
    if (dropBufferedOutput || _pausedOutput.isNotEmpty) {
      _pausedOutput.clear();
    }
    if (dropQueuedFailure) {
      _queuedFailureCode = null;
    }
    if (_outwardFenced) {
      _deliverQueuedFailureAndClose();
      return;
    }
    _outwardFenced = true;
    _deliverQueuedFailureAndClose();
  }

  void _deliverQueuedFailureAndClose() {
    if (_consumerPaused && _queuedFailureCode != null) {
      return;
    }
    final code = _queuedFailureCode;
    _queuedFailureCode = null;
    if (code != null) {
      _addOutwardFailure(code);
    }
    if (_outwardAddInProgress) {
      scheduleMicrotask(_closeOutward);
      return;
    }
    _closeOutward();
  }

  void _closeOutward() {
    if (_outwardCloseStarted || _outwardAddInProgress) {
      return;
    }
    _outwardCloseStarted = true;
    try {
      final closed = _controller.close();
      unawaited(
        closed.then<void>(
          (_) => _owner._outwardFinished(this),
          onError: (_error, _stackTrace) {
            _owner._outwardFinished(this);
          },
        ),
      );
    } catch (_) {
      // A synchronous controller can reject close only while dispatching. The
      // dispatch finally block schedules one safe retry without blocking the
      // consumer's cancellation Future.
      _outwardCloseStarted = false;
      scheduleMicrotask(_closeOutward);
    }
  }
}
