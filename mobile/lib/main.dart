import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';

import './asr.dart';
import './assistant.dart';
import './tts.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // We download the model from a public release, so no token is needed; the
  // default empty value is fine. maxDownloadRetries guards flaky first-run pulls.
  FlutterGemma.initialize(
    huggingFaceToken: const String.fromEnvironment('HUGGINGFACE_TOKEN'),
    maxDownloadRetries: 10,
  );
  runApp(const SpeakerApp());
}

class SpeakerApp extends StatelessWidget {
  const SpeakerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Speaker (on-device)',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _index = 0;
  static const _screens = [AssistantScreen(), AsrScreen(), TtsScreen()];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Speaker — on-device test')),
      body: _screens[_index],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.auto_awesome), label: 'Assistant'),
          NavigationDestination(icon: Icon(Icons.mic), label: 'Listen'),
          NavigationDestination(icon: Icon(Icons.record_voice_over), label: 'Speak'),
        ],
      ),
    );
  }
}
