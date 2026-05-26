import 'package:flutter/material.dart';

import './asr.dart';
import './tts.dart';

void main() => runApp(const SpeakerApp());

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
  static const _screens = [AsrScreen(), TtsScreen()];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Speaker — on-device test')),
      body: _screens[_index],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.mic), label: 'Listen'),
          NavigationDestination(icon: Icon(Icons.record_voice_over), label: 'Speak'),
        ],
      ),
    );
  }
}
