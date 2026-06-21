// Pins the capture config that keeps the assistant from transcribing its own TTS.
// A revert to AudioSource.mic silently drops the OS AEC/NS/AGC -> this fails CI.
// Pure data types from package:record; no device/plugin needed.
import 'package:flutter_test/flutter_test.dart';
import 'package:record/record.dart';
import 'package:speaker_mobile/audio_capture_config.dart';

void main() {
  test('capture keeps the voice-comm path + hardware AEC/NS/AGC', () {
    final c = buildCaptureConfig();
    expect(c.encoder, AudioEncoder.pcm16bits);
    expect(c.sampleRate, 16000);
    expect(c.numChannels, 1);
    expect(c.echoCancel, isTrue);
    expect(c.noiseSuppress, isTrue);
    expect(c.autoGain, isTrue);
    // The load-bearing field: voiceCommunication enables Android's hardware
    // echo-canceller; .mic would drop all of it.
    expect(c.androidConfig.audioSource, AndroidAudioSource.voiceCommunication);
  });
}
