# Test Stage: full

## Result

- Return code: `1`
- Duration: `86.272s`
- Passed: `255`
- Failed: `5`
- Errors: `0`
- Skipped: `0`

## Failures

### tests.test_noise_robustness.TestBabbleNoise.test_babble_does_not_fire_without_calibration

- Area: `general`
- Message: AssertionError: 1 != 0 : Babble noise (3 FSDD speakers, RMS≈0.06) must NOT trigger barge-in. Got 1 false interrupt(s).  This is background TV speech — the user is not speaking.  BUG: Silero detects voices in the babble mix → score ≥ 2.0 → fires.  FIX: Require calibrated noise_floor or a pre-TTS energy baseline.
- Duration: `0.277s`

```text
ckground, not just an absolute threshold.
        """
        babble = babble_noise(3.0, amplitude=0.06, num_speakers=3)
    
        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
    
        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(babble)
            h.drain(timeout=5.0)
    
>       self.assertEqual(
            len(interrupts), 0,
            f"Babble noise (3 FSDD speakers, RMS≈0.06) must NOT trigger barge-in. "
            f"Got {len(interrupts)} false interrupt(s).  "
            f"This is background TV speech — the user is not speaking.  "
            f"BUG: Silero detects voices in the babble mix → score ≥ 2.0 → fires.  "
            f"FIX: Require calibrated noise_floor or a pre-TTS energy baseline.",
        )
E       AssertionError: 1 != 0 : Babble noise (3 FSDD speakers, RMS≈0.06) must NOT trigger barge-in. Got 1 false interrupt(s).  This is background TV speech — the user is not speaking.  BUG: Silero detects voices in the babble mix → score ≥ 2.0 → fires.  FIX: Require calibrated noise_floor or a pre-TTS energy baseline.

tests/test_noise_robustness.py:121: AssertionError
```

### tests.test_noise_robustness.TestNonstationaryNoise.test_environment_spike_with_babble_causes_false_positive

- Area: `general`
- Message: AssertionError: 1 != 0 : Environment change (TV turns on mid-stream) must NOT fire barge-in. Got 1 false interrupt(s).  BUG: noise_floor=0.01 is stale; TV speech at 0.08 RMS passes the gate.  FIX: Dynamically update noise_floor during idle / non-speaking periods.
- Duration: `0.27s`

```text
peech to trigger Silero
        babble = babble_noise(3.0, amplitude=0.08, num_speakers=3)
        transition = int(0.4 * 3.0 * SR)
        noise[transition:] = babble[transition:]
    
        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
        rec._noise_floor = 0.01  # calibrated to quiet
    
        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(noise)
            h.drain(timeout=6.0)
    
>       self.assertEqual(
            len(interrupts), 0,
            f"Environment change (TV turns on mid-stream) must NOT fire barge-in. "
            f"Got {len(interrupts)} false interrupt(s).  "
            f"BUG: noise_floor=0.01 is stale; TV speech at 0.08 RMS passes the gate.  "
            f"FIX: Dynamically update noise_floor during idle / non-speaking periods.",
        )
E       AssertionError: 1 != 0 : Environment change (TV turns on mid-stream) must NOT fire barge-in. Got 1 false interrupt(s).  BUG: noise_floor=0.01 is stale; TV speech at 0.08 RMS passes the gate.  FIX: Dynamically update noise_floor during idle / non-speaking periods.

tests/test_noise_robustness.py:252: AssertionError
```

### tests.test_voice_variability.TestMicDistance.test_whispered_speech_fires_barge_in

- Area: `audio`
- Message: AssertionError: 0 not greater than 0 : Whispered speech (amplitude=0.03) must trigger barge-in.  Got 0 interrupts.  BUG: Whispers are below the noise gate AND below Silero's 0.80 confidence threshold in barge-in mode.  FIX: Lower barge-in Silero threshold to 0.60, or implement AGC so weak signals are amplified before gating.
- Duration: `0.285s`

```text
 for whisper mode, or implement AGC.
        """
        audio = human_voice_concat(1.5, amplitude=0.03)
    
        interrupts = []
        rec = make_recorder(on_interrupt=lambda info=None: interrupts.append(info))
    
        with AudioHarness(rec) as h:
            h.set_tts_speaking()
            h.inject(audio)
            h.drain(timeout=4.0)
    
>       self.assertGreater(
            len(interrupts), 0,
            f"Whispered speech (amplitude=0.03) must trigger barge-in.  "
            f"Got 0 interrupts.  "
            f"BUG: Whispers are below the noise gate AND below Silero's 0.80 "
            f"confidence threshold in barge-in mode.  "
            f"FIX: Lower barge-in Silero threshold to 0.60, or implement AGC "
            f"so weak signals are amplified before gating.",
        )
E       AssertionError: 0 not greater than 0 : Whispered speech (amplitude=0.03) must trigger barge-in.  Got 0 interrupts.  BUG: Whispers are below the noise gate AND below Silero's 0.80 confidence threshold in barge-in mode.  FIX: Lower barge-in Silero threshold to 0.60, or implement AGC so weak signals are amplified before gating.

tests/test_voice_variability.py:161: AssertionError
```

### tests.recorded.test_session_20260406_165119.test_turn_003_no_self_interrupt

- Area: `recorded`
- Message: AssertionError: Turn 3: self-interruption regression detected. The pipeline fired barge-in 1 time(s) on TTS echo alone. Last interrupt: {'rms': 0.03155454993247986, 'threshold': 0.01, 'voiced': True, 'duration_sec': 0.256, 'timestamp': 1777724670.2848904, 'echo': False}
assert 1 == 0
 +  where 1 = len([{'duration_sec': 0.256, 'echo': False, 'rms': 0.03155454993247986, 'threshold': 0.01, ...}])
- Duration: `2.423s`

```text
rupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
            barge_in_min_delay_after_ref_sec=GATE_SEC,
            aec_enabled=False,
        )
        rec._noise_floor = 0.005
    
        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts, zero_delays=False)
            h.inject(mic, inter_chunk_delay=CHUNK_SEC)
            h.drain(timeout=10.0)
    
>       assert len(interrupts) == 0, (
            f"Turn 3: self-interruption regression detected. "
            f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
            f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
        )
E       AssertionError: Turn 3: self-interruption regression detected. The pipeline fired barge-in 1 time(s) on TTS echo alone. Last interrupt: {'rms': 0.03155454993247986, 'threshold': 0.01, 'voiced': True, 'duration_sec': 0.256, 'timestamp': 1777724670.2848904, 'echo': False}
E       assert 1 == 0
E        +  where 1 = len([{'duration_sec': 0.256, 'echo': False, 'rms': 0.03155454993247986, 'threshold': 0.01, ...}])

tests/recorded/test_session_20260406_165119.py:172: AssertionError
```

### tests.recorded.test_session_20260406_165119.test_turn_004_no_self_interrupt

- Area: `recorded`
- Message: AssertionError: Turn 4: self-interruption regression detected. The pipeline fired barge-in 1 time(s) on TTS echo alone. Last interrupt: {'rms': 0.02916882000863552, 'threshold': 0.01, 'voiced': True, 'duration_sec': 0.224, 'timestamp': 1777724672.647219, 'echo': False}
assert 1 == 0
 +  where 1 = len([{'duration_sec': 0.224, 'echo': False, 'rms': 0.02916882000863552, 'threshold': 0.01, ...}])
- Duration: `2.617s`

```text
rrupts = []
        rec = make_recorder(
            on_interrupt=lambda info=None: interrupts.append(info),
            barge_in_min_delay_after_ref_sec=GATE_SEC,
            aec_enabled=False,
        )
        rec._noise_floor = 0.005
    
        with AudioHarness(rec) as h:
            h.set_tts_speaking(audio_ref=tts, zero_delays=False)
            h.inject(mic, inter_chunk_delay=CHUNK_SEC)
            h.drain(timeout=10.0)
    
>       assert len(interrupts) == 0, (
            f"Turn 4: self-interruption regression detected. "
            f"The pipeline fired barge-in {len(interrupts)} time(s) on TTS echo alone. "
            f"Last interrupt: {interrupts[-1] if interrupts else 'n/a'}"
        )
E       AssertionError: Turn 4: self-interruption regression detected. The pipeline fired barge-in 1 time(s) on TTS echo alone. Last interrupt: {'rms': 0.02916882000863552, 'threshold': 0.01, 'voiced': True, 'duration_sec': 0.224, 'timestamp': 1777724672.647219, 'echo': False}
E       assert 1 == 0
E        +  where 1 = len([{'duration_sec': 0.224, 'echo': False, 'rms': 0.02916882000863552, 'threshold': 0.01, ...}])

tests/recorded/test_session_20260406_165119.py:210: AssertionError
```
