# Test Stage: discovery

## Result

- Return code: `1`
- Duration: `106.777s`
- Passed: `84`
- Failed: `62`
- Errors: `0`
- Skipped: `0`

## Failures

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_phone_table_30cm]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Phone lying beside laptop speaker. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.2070034295320511, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.337s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.2070034295320511, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_laptop_mic_70cm]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Laptop speaker to built-in microphone path. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.17284372448921204, 'threshold': 0.01, ...}
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
E       AssertionError: Laptop speaker to built-in microphone path. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.17284372448921204, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_smart_speaker_2m]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Smart speaker two metres from user microphone. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1532914638519287, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.281s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1532914638519287, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_kitchen_counter_tile]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Kitchen counter with hard tile reflections. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.0753880962729454, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.279s`

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
E       AssertionError: Kitchen counter with hard tile reflections. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.0753880962729454, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_bathroom_tile]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Bathroom tile reverberation. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.06440238654613495, 'threshold': 0.01, ...}
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
E       AssertionError: Bathroom tile reverberation. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.06440238654613495, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_hallway_far_wall]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Hallway far-wall assistant reflection. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08435456454753876, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.276s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08435456454753876, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_open_living_room_wall]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Living-room first wall reflection. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.12791402637958527, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.303s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.12791402637958527, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_car_cabin_dashboard]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Small car cabin dashboard reflection. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08567162603139877, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.276s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08567162603139877, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_projector_speaker_ceiling]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Ceiling projector speaker reflection. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.09706400334835052, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.276s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.09706400334835052, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_bluetooth_speaker_shelf]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Bluetooth speaker on bookshelf. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.05154785141348839, 'threshold': 0.01, ...}
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
E       AssertionError: Bluetooth speaker on bookshelf. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.05154785141348839, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_tv_soundbar_living_room]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: TV soundbar leaking assistant speech. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.055319078266620636, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.274s`

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
E       AssertionError: TV soundbar leaking assistant speech. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.055319078266620636, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_conference_table]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Conference table speakerphone echo. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.06111621484160423, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.276s`

```text
f expectation == "callback":
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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.06111621484160423, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_corner_room_bass_buildup]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Speaker in room corner with gain buildup. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.09177596867084503, 'threshold': 0.01, ...}
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
E       AssertionError: Speaker in room corner with gain buildup. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.09177596867084503, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_curtained_room_soft_reverb]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Curtained room with soft reverb. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.16978240013122559, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.277s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.16978240013122559, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_open_door_secondary_room]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Assistant audio heard from another room. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.039073649793863297, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.272s`

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
E       AssertionError: Assistant audio heard from another room. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.039073649793863297, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_high_volume_near_field]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: High-volume near-field assistant playback. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.3066733777523041, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.271s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.3066733777523041, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_low_volume_far_field]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Low-volume far-field assistant leakage. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.051804907619953156, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.279s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.051804907619953156, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_sample_rate_artifact]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Echo after device sample-rate conversion. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.13466854393482208, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.287s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.13466854393482208, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_clipped_speaker_output]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Clipped speaker output echo. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.15810346603393555, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.276s`

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

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_tts_with_background_tv]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Assistant echo mixed with TV bed. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1376277208328247, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.288s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1376277208328247, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_tts_with_music_bed]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Assistant echo mixed with music. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.053375061601400375, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.276s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.053375061601400375, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_tts_with_babble_bed]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Assistant echo mixed with distant people. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.04849665239453316, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.279s`

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
E       AssertionError: Assistant echo mixed with distant people. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.04849665239453316, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_multi_path_two_reflections]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Two dominant echo paths. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.17298339307308197, 'threshold': 0.01, ...}
  Use -v to get more diff
- Duration: `0.281s`

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
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.17298339307308197, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_multi_path_three_reflections]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Three dominant echo paths. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1978895366191864, 'threshold': 0.01, ...}
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
E       AssertionError: Three dominant echo paths. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.1978895366191864, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[echo_reverb_long_phrase_tail]

- Area: `audio`
- Category: `echo_alignment_gap`
- Solvable: `True`
- Cause: Assistant audio leaked back into the microphone with delay, reverb, clipping, or sample-rate artifacts that the current echo similarity gate did not recognize as echo.
- Suggested fix: Use lag-aware cross-correlation/echo alignment, room echo floor tracking, or stronger AEC reference handling.
- Message: AssertionError: Long TTS phrase with heavy tail. Fired 1 false interrupt(s).
assert [{'duration_s...': 0.01, ...}] == []
  Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08119171857833862, 'threshold': 0.01, ...}
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
E       AssertionError: Long TTS phrase with heavy tail. Fired 1 false interrupt(s).
E       assert [{'duration_s...': 0.01, ...}] == []
E         Left contains one more item: {'duration_sec': 0.128, 'echo': False, 'rms': 0.08119171857833862, 'threshold': 0.01, ...}
E         Use -v to get more diff

tests/test_failure_discovery_audio.py:374: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[background_tv_news_anchor]

- Area: `audio`
- Category: `background_noise_gate_gap`
- Solvable: `True`
- Cause: Speech-like background noise passes VAD/noise gates, often because the noise floor is missing or stale.
- Suggested fix: Require calibrated baseline, update noise floor dynamically, and reject signals already present before assistant speech.
- Message: AssertionError: TV news anchor-like broadband audio.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00971879, -0.03316975,  0.02393525, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.797s`

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
E             Left contains one more item: array([ 0.00971879, -0.03316975,  0.02393525, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_failure_discovery_audio.py:360: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[background_tv_loud_commercial]

- Area: `audio`
- Category: `unknown`
- Solvable: `True`
- Cause: No automatic cause matched; inspect traceback and test metadata.
- Suggested fix: Inspect the failing test and implementation.
- Message: AssertionError: Loud TV commercial burst.
assert [array([ 0.01...type=float32)] == []
  Left contains one more item: array([ 0.01518561, -0.05182773,  0.03739883, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.693s`

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
E             Left contains one more item: array([ 0.01518561, -0.05182773,  0.03739883, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_failure_discovery_audio.py:360: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[background_podcast_far_field]

- Area: `audio`
- Category: `unknown`
- Solvable: `True`
- Cause: No automatic cause matched; inspect traceback and test metadata.
- Suggested fix: Inspect the failing test and implementation.
- Message: AssertionError: Podcast playing across the room.
assert [array([0.018...type=float32)] == []
  Left contains one more item: array([0.0186704 , 0.01719079, 0.01517375, ..., 0.        , 0.        ,\n       0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.694s`

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
E             Left contains one more item: array([0.0186704 , 0.01719079, 0.01517375, ..., 0.        , 0.        ,\n       0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_failure_discovery_audio.py:360: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[background_party_babble]

- Area: `audio`
- Category: `background_noise_gate_gap`
- Solvable: `True`
- Cause: Speech-like background noise passes VAD/noise gates, often because the noise floor is missing or stale.
- Suggested fix: Require calibrated baseline, update noise floor dynamically, and reject signals already present before assistant speech.
- Message: AssertionError: Party babble without direct user intent.
assert [array([ 0.01...type=float32)] == []
  Left contains one more item: array([ 0.01082398,  0.00380801, -0.0036492 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
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
E             Left contains one more item: array([ 0.01082398,  0.00380801, -0.0036492 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_failure_discovery_audio.py:360: AssertionError
```

### tests.test_failure_discovery_audio.test_failure_discovery_corpus[background_kids_room_babble]

- Area: `audio`
- Category: `background_noise_gate_gap`
- Solvable: `True`
- Cause: Speech-like background noise passes VAD/noise gates, often because the noise floor is missing or stale.
- Suggested fix: Require calibrated baseline, update noise floor dynamically, and reject signals already present before assistant speech.
- Message: AssertionError: Children talking in another room.
assert [array([0.025...type=float32)] == []
  Left contains one more item: array([0.025953  , 0.01864869, 0.01055665, ..., 0.        , 0.        ,\n       0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.706s`

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
E             Left contains one more item: array([0.025953  , 0.01864869, 0.01055665, ..., 0.        , 0.        ,\n       0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_failure_discovery_audio.py:360: AssertionError
```

... 32 more failures omitted. See failures.json.