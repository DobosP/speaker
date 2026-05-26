# Test Stage: full

## Result

- Return code: `1`
- Duration: `284.508s`
- Passed: `392`
- Failed: `216`
- Errors: `0`
- Skipped: `0`

## Failures

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_phone_table_30cm]

- Area: `audio`
- Message: AssertionError: Phone lying beside laptop speaker. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.20700345933437347, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.283s`

```text
if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Phone lying beside laptop speaker. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.20700345933437347, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_laptop_mic_70cm]

- Area: `audio`
- Message: AssertionError: Laptop speaker to built-in microphone path. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.17284372448921204, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.277s`

```text
ation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Laptop speaker to built-in microphone path. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.17284372448921204, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_smart_speaker_2m]

- Area: `audio`
- Message: AssertionError: Smart speaker two metres from user microphone. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1532914936542511, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.271s`

```text
ion == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Smart speaker two metres from user microphone. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1532914936542511, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_kitchen_counter_tile]

- Area: `audio`
- Message: AssertionError: Kitchen counter with hard tile reflections. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.07538808882236481, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.282s`

```text
ation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Kitchen counter with hard tile reflections. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.07538808882236481, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_bathroom_tile]

- Area: `audio`
- Message: AssertionError: Bathroom tile reverberation. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.06440234184265137, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.27s`

```text
      if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Bathroom tile reverberation. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.06440234184265137, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_hallway_far_wall]

- Area: `audio`
- Message: AssertionError: Hallway far-wall assistant reflection. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08435460925102234, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.273s`

```text
xpectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Hallway far-wall assistant reflection. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08435460925102234, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_open_living_room_wall]

- Area: `audio`
- Message: AssertionError: Living-room first wall reflection. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.12791404128074646, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.28s`

```text
if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Living-room first wall reflection. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.12791404128074646, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_car_cabin_dashboard]

- Area: `audio`
- Message: AssertionError: Small car cabin dashboard reflection. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08567164093255997, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.257s`

```text
expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Small car cabin dashboard reflection. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08567164093255997, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_projector_speaker_ceiling]

- Area: `audio`
- Message: AssertionError: Ceiling projector speaker reflection. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.09706397354602814, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.263s`

```text
expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Ceiling projector speaker reflection. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.09706397354602814, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_bluetooth_speaker_shelf]

- Area: `audio`
- Message: AssertionError: Bluetooth speaker on bookshelf. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.051547810435295105, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.274s`

```text
  if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Bluetooth speaker on bookshelf. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.051547810435295105, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_tv_soundbar_living_room]

- Area: `audio`
- Message: AssertionError: TV soundbar leaking assistant speech. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.05531902611255646, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.27s`

```text
expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: TV soundbar leaking assistant speech. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.05531902611255646, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_conference_table]

- Area: `audio`
- Message: AssertionError: Conference table speakerphone echo. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.061116158962249756, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.269s`

```text
 expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Conference table speakerphone echo. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.061116158962249756, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_corner_room_bass_buildup]

- Area: `audio`
- Message: AssertionError: Speaker in room corner with gain buildup. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.0917760357260704, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.294s`

```text
ectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Speaker in room corner with gain buildup. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.0917760357260704, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_curtained_room_soft_reverb]

- Area: `audio`
- Message: AssertionError: Curtained room with soft reverb. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.16978241503238678, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.264s`

```text
  if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Curtained room with soft reverb. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.16978241503238678, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_open_door_secondary_room]

- Area: `audio`
- Message: AssertionError: Assistant audio heard from another room. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.03907366842031479, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.268s`

```text
ectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Assistant audio heard from another room. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.03907366842031479, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_high_volume_near_field]

- Area: `audio`
- Message: AssertionError: High-volume near-field assistant playback. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.3066733479499817, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.27s`

```text
ctation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: High-volume near-field assistant playback. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.3066733479499817, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_low_volume_far_field]

- Area: `audio`
- Message: AssertionError: Low-volume far-field assistant leakage. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.051804881542921066, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.272s`

```text
ectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Low-volume far-field assistant leakage. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.051804881542921066, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_sample_rate_artifact]

- Area: `audio`
- Message: AssertionError: Echo after device sample-rate conversion. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.13466860353946686, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.274s`

```text
ctation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Echo after device sample-rate conversion. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.13466860353946686, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_clipped_speaker_output]

- Area: `audio`
- Message: AssertionError: Clipped speaker output echo. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.15810346603393555, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.261s`

```text
      if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Clipped speaker output echo. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.15810346603393555, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_tts_with_background_tv]

- Area: `audio`
- Message: AssertionError: Assistant echo mixed with TV bed. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1376277059316635, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.265s`

```text
  if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Assistant echo mixed with TV bed. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1376277059316635, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_tts_with_music_bed]

- Area: `audio`
- Message: AssertionError: Assistant echo mixed with music. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.05337503179907799, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.266s`

```text
  if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Assistant echo mixed with music. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.05337503179907799, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_tts_with_babble_bed]

- Area: `audio`
- Message: AssertionError: Assistant echo mixed with distant people. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.048496633768081665, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.269s`

```text
tation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Assistant echo mixed with distant people. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.048496633768081665, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_multi_path_two_reflections]

- Area: `audio`
- Message: AssertionError: Two dominant echo paths. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1729833334684372, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.278s`

```text
  
        if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Two dominant echo paths. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1729833334684372, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_multi_path_three_reflections]

- Area: `audio`
- Message: AssertionError: Three dominant echo paths. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.19788946211338043, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.279s`

```text
        if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Three dominant echo paths. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.19788946211338043, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_reverb_long_phrase_tail]

- Area: `audio`
- Message: AssertionError: Long TTS phrase with heavy tail. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08119175583124161, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.261s`

```text
  if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks == [], case["description"]
            return
    
        ref = SAMPLES[case["ref"]]
        if case["kind"] == "echo_control":
            interrupts = _interrupts_for_tts_leak(
                audio,
                ref,
                gate_sec=0.20,
                inter_chunk_delay=1024 / SR,
                zero_delays=False,
            )
        else:
            interrupts = _interrupts_for_tts_leak(audio, ref)
>       assert interrupts == [], (
            f"{case['description']} Fired {len(interrupts)} false interrupt(s)."
        )
E       AssertionError: Long TTS phrase with heavy tail. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08119175583124161, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:630: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[background_tv_news_anchor]

- Area: `audio`
- Message: AssertionError: TV news anchor-like broadband audio.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00971873, -0.03316876,  0.02393442, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.687s`

```text
case = {'description': 'TV news anchor-like broadband audio.', 'expectation': 'no_callback', 'importance': 'TV speech must not become a user command.', 'kind': 'background', ...}

    @pytest.mark.parametrize("case", ACTION_CASES, ids=lambda c: c["name"])
    def test_failure_discovery_corpus(case):
        audio = SAMPLES[case["name"]]
        expectation = case["expectation"]
    
        if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
>           assert callbacks == [], case["description"]
E           AssertionError: TV news anchor-like broadband audio.
E           assert [array([ 0.00...type=float32)] == []
E             Left contains one more item: array([ 0.00971873, -0.03316876,  0.02393442, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_failure_discovery_audio.py:616: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[background_tv_loud_commercial]

- Area: `audio`
- Message: AssertionError: Loud TV commercial burst.
assert [array([ 0.01...type=float32)] == []
  Left contains one more item: array([ 0.01518664, -0.05182773,  0.03739867, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.691s`

```text
case = {'description': 'Loud TV commercial burst.', 'expectation': 'no_callback', 'importance': 'Commercials are sudden and speech-like.', 'kind': 'background', ...}

    @pytest.mark.parametrize("case", ACTION_CASES, ids=lambda c: c["name"])
    def test_failure_discovery_corpus(case):
        audio = SAMPLES[case["name"]]
        expectation = case["expectation"]
    
        if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
>           assert callbacks == [], case["description"]
E           AssertionError: Loud TV commercial burst.
E           assert [array([ 0.01...type=float32)] == []
E             Left contains one more item: array([ 0.01518664, -0.05182773,  0.03739867, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_failure_discovery_audio.py:616: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[background_podcast_far_field]

- Area: `audio`
- Message: AssertionError: Podcast playing across the room.
assert [array([0.018...type=float32)] == []
  Left contains one more item: array([0.01867158, 0.01719079, 0.01517614, ..., 0.        , 0.        ,\n       0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.692s`

```text
case = {'description': 'Podcast playing across the room.', 'expectation': 'no_callback', 'importance': 'Single/few-speaker media can look voiced.', 'kind': 'background', ...}

    @pytest.mark.parametrize("case", ACTION_CASES, ids=lambda c: c["name"])
    def test_failure_discovery_corpus(case):
        audio = SAMPLES[case["name"]]
        expectation = case["expectation"]
    
        if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
>           assert callbacks == [], case["description"]
E           AssertionError: Podcast playing across the room.
E           assert [array([0.018...type=float32)] == []
E             Left contains one more item: array([0.01867158, 0.01719079, 0.01517614, ..., 0.        , 0.        ,\n       0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_failure_discovery_audio.py:616: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[background_party_babble]

- Area: `audio`
- Message: AssertionError: Party babble without direct user intent.
assert [array([ 0.01...type=float32)] == []
  Left contains one more item: array([ 0.01082264,  0.0038078 , -0.0036503 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.702s`

```text
case = {'description': 'Party babble without direct user intent.', 'expectation': 'no_callback', 'importance': 'Multi-talker noise is common in homes.', 'kind': 'background', ...}

    @pytest.mark.parametrize("case", ACTION_CASES, ids=lambda c: c["name"])
    def test_failure_discovery_corpus(case):
        audio = SAMPLES[case["name"]]
        expectation = case["expectation"]
    
        if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
>           assert callbacks == [], case["description"]
E           AssertionError: Party babble without direct user intent.
E           assert [array([ 0.01...type=float32)] == []
E             Left contains one more item: array([ 0.01082264,  0.0038078 , -0.0036503 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_failure_discovery_audio.py:616: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[background_kids_room_babble]

- Area: `audio`
- Message: AssertionError: Children talking in another room.
assert [array([0.025...type=float32)] == []
  Left contains one more item: array([0.02595172, 0.01864822, 0.01055624, ..., 0.        , 0.        ,\n       0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.704s`

```text
case = {'description': 'Children talking in another room.', 'expectation': 'no_callback', 'importance': 'Distant family speech should not trigger callbacks.', 'kind': 'background', ...}

    @pytest.mark.parametrize("case", ACTION_CASES, ids=lambda c: c["name"])
    def test_failure_discovery_corpus(case):
        audio = SAMPLES[case["name"]]
        expectation = case["expectation"]
    
        if expectation == "callback":
            callbacks = _callbacks_for_user_audio(audio)
            assert callbacks, case["description"]
            return
    
        if expectation == "no_callback":
            callbacks = _callbacks_for_user_audio(audio)
>           assert callbacks == [], case["description"]
E           AssertionError: Children talking in another room.
E           assert [array([0.025...type=float32)] == []
E             Left contains one more item: array([0.02595172, 0.01864822, 0.01055624, ..., 0.        , 0.        ,\n       0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_failure_discovery_audio.py:616: AssertionError
```

... 186 more failures omitted. See failures.json.