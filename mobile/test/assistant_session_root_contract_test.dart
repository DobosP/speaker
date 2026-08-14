// Static app-root ownership guard. No Flutter widget/plugin/model/device is
// constructed; pure AgentSession behavior is covered by agent_session_test.
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  late String mainSource;
  late String assistantSource;

  setUpAll(() {
    mainSource = File('lib/main.dart').readAsStringSync();
    assistantSource = File('lib/assistant.dart').readAsStringSync();
  });

  String between(String source, String start, String end) {
    final startIndex = source.indexOf(start);
    final endIndex = source.indexOf(end, startIndex + start.length);
    expect(startIndex, isNonNegative, reason: 'missing start marker: $start');
    expect(
      endIndex,
      greaterThan(startIndex),
      reason: 'missing end marker: $end',
    );
    return source.substring(startIndex, endIndex);
  }

  void expectOrdered(String body, List<String> markers) {
    var previous = -1;
    for (final marker in markers) {
      final current = body.indexOf(marker, previous + 1);
      expect(current, greaterThan(previous), reason: 'out of order: $marker');
      previous = current;
    }
  }

  test('one app-root AgentSession is injected into the Assistant screen', () {
    expect(mainSource, contains("import './agent_session.dart';"));
    expect(mainSource, contains('class SpeakerApp extends StatefulWidget'));
    final root = between(
      mainSource,
      'class _SpeakerAppState',
      'class HomePage',
    );
    expectOrdered(root, [
      'late final AgentSession _session;',
      'void initState()',
      '_session = AgentSession();',
      'void dispose()',
      '_session.close();',
      'Widget build(BuildContext context)',
      'HomePage(session: _session)',
    ]);
    expect(RegExp(r'AgentSession\(\)').allMatches(mainSource).length, 1);

    final home = between(mainSource, 'class HomePage', 'class _HomePageState');
    expect(home, contains('final AgentSession session;'));
    final screens = between(
      mainSource,
      'class _HomePageState',
      'Widget build(BuildContext context)',
    );
    expect(screens, contains('AssistantScreen(session: widget.session)'));
  });

  test('screen consumes but never closes the app-root session', () {
    final widget = between(
      assistantSource,
      'class AssistantScreen extends StatefulWidget',
      'class _AssistantScreenState',
    );
    expect(widget, contains('const AssistantScreen({required this.session'));
    expect(widget, contains('final AgentSession session;'));
    expect(assistantSource, contains('widget.session.acceptPartial(partial)'));
    expect(assistantSource, contains('widget.session.acceptFinal(rawText)'));
    expect(
      assistantSource,
      contains('widget.session.isCurrent(admission.turn)'),
    );
    expect(assistantSource, isNot(contains('widget.session.close()')));
    expect(RegExp(r'AgentSession\(\)').allMatches(assistantSource).length, 0);
  });
}
