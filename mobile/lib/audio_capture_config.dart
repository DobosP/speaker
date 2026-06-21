// The mic capture configuration, extracted so it can be regression-tested.
//
// voiceCommunication + echoCancel/noiseSuppress/autoGain route capture through
// Android's hardware voice-processing path (the OS AEC/NS/AGC) -- the only thing
// stopping the recognizer from transcribing the assistant's own TTS during
// playback. A regression to AudioSource.mic silently drops ALL of that, so
// capture_config_test.dart pins these field values.
import 'package:record/record.dart';

RecordConfig buildCaptureConfig() => RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: 16000,
      numChannels: 1,
      echoCancel: true,
      noiseSuppress: true,
      autoGain: true,
      androidConfig: const AndroidRecordConfig(
        audioSource: AndroidAudioSource.voiceCommunication,
      ),
    );
