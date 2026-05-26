# Test Stage: full

## Result

- Return code: `1`
- Duration: `38.989s`
- Passed: `181`
- Failed: `20`
- Errors: `0`
- Skipped: `0`

## Failures

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00444025, -0.00551854, -0.00542578, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.158s`

```text
nt_speech', kind='background_open_data', expectation='no_callback', audi...distant speech.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00444025, -0.00551854, -0.00542578, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_tv]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech tv. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00251711,  0.01493473, -0.00977169, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.16s`

```text
speech_tv', kind='background_open_data', expectation='no_callback', a...tant speech tv.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech tv. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00251711,  0.01493473, -0.00977169, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_music]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech music. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.04...type=float32)] == []
  Left contains one more item: array([-0.04164777, -0.02648615, -0.04670862, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.156s`

```text
ech_music', kind='background_open_data', expectation='no_callback'...t speech music.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech music. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.04...type=float32)] == []
E             Left contains one more item: array([-0.04164777, -0.02648615, -0.04670862, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_hvac]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech hvac. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00934496, -0.05422829, -0.01041808, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.155s`

```text
eech_hvac', kind='background_open_data', expectation='no_callback',...nt speech hvac.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech hvac. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([ 0.00...type=float32)] == []
E             Left contains one more item: array([ 0.00934496, -0.05422829, -0.01041808, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_clipped_speaker]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech clipped speaker. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.01...type=float32)] == []
  Left contains one more item: array([-0.01115342, -0.01386194, -0.01362895, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.165s`

```text
d_speaker', kind='background_open_data', expectation='no...lipped speaker.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech clipped speaker. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.01...type=float32)] == []
E             Left contains one more item: array([-0.01115342, -0.01386194, -0.01362895, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_resampled]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech resampled. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00444025, -0.00551854, -0.00542578, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.161s`

```text
resampled', kind='background_open_data', expectation='no_callb...eech resampled.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech resampled. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00444025, -0.00551854, -0.00542578, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_babble]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech babble. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00047826,  0.00616404,  0.01122743, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.159s`

```text
ch_babble', kind='background_open_data', expectation='no_callback... speech babble.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech babble. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00047826,  0.00616404,  0.01122743, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_low_rumble]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech low rumble. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00097694,  0.02646559, -0.06335241, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.163s`

```text
ow_rumble', kind='background_open_data', expectation='no_call...ech low rumble.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech low rumble. Source=FSDD:0_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([ 0.00...type=float32)] == []
E             Left contains one more item: array([ 0.00097694,  0.02646559, -0.06335241, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00746868, -0.00912053, -0.00899993, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.164s`

```text
nt_speech', kind='background_open_data', expectation='no_callback', audi...distant speech.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00746868, -0.00912053, -0.00899993, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_tv]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech tv. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.04...type=float32)] == []
  Left contains one more item: array([-0.04925969, -0.0102992 ,  0.01539566, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.158s`

```text
speech_tv', kind='background_open_data', expectation='no_callback', a...tant speech tv.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech tv. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.04...type=float32)] == []
E             Left contains one more item: array([-0.04925969, -0.0102992 ,  0.01539566, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_music]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech music. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.04...type=float32)] == []
  Left contains one more item: array([-0.04651056, -0.03112186, -0.05231804, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.166s`

```text
ech_music', kind='background_open_data', expectation='no_callback'...t speech music.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech music. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.04...type=float32)] == []
E             Left contains one more item: array([-0.04651056, -0.03112186, -0.05231804, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_hvac]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech hvac. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00713267, -0.06071408, -0.01428779, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.161s`

```text
eech_hvac', kind='background_open_data', expectation='no_callback',...nt speech hvac.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech hvac. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([ 0.00...type=float32)] == []
E             Left contains one more item: array([ 0.00713267, -0.06071408, -0.01428779, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_clipped_speaker]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech clipped speaker. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.01...type=float32)] == []
  Left contains one more item: array([-0.01876048, -0.02290973, -0.02260679, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.161s`

```text
d_speaker', kind='background_open_data', expectation='no...lipped speaker.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech clipped speaker. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.01...type=float32)] == []
E             Left contains one more item: array([-0.01876048, -0.02290973, -0.02260679, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_resampled]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech resampled. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00746868, -0.00912053, -0.00899993, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.16s`

```text
resampled', kind='background_open_data', expectation='no_callb...eech resampled.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech resampled. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00746868, -0.00912053, -0.00899993, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_babble]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech babble. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00327212,  0.0032537 ,  0.00863922, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.159s`

```text
ch_babble', kind='background_open_data', expectation='no_callback... speech babble.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech babble. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00327212,  0.0032537 ,  0.00863922, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_low_rumble]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech low rumble. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.12...type=float32)] == []
  Left contains one more item: array([-0.12100255, -0.05612217, -0.07169521, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.158s`

```text
ow_rumble', kind='background_open_data', expectation='no_call...ech low rumble.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech low rumble. Source=FSDD:1_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.12...type=float32)] == []
E             Left contains one more item: array([-0.12100255, -0.05612217, -0.07169521, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech. Source=FSDD:2_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.01...type=float32)] == []
  Left contains one more item: array([-0.01160683, -0.02153094, -0.0131548 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.158s`

```text
nt_speech', kind='background_open_data', expectation='no_callback', audi...distant speech.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech. Source=FSDD:2_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.01...type=float32)] == []
E             Left contains one more item: array([-0.01160683, -0.02153094, -0.0131548 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech_tv]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech tv. Source=FSDD:2_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.05...type=float32)] == []
  Left contains one more item: array([-0.05625723,  0.01024601, -0.02873969, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.161s`

```text
speech_tv', kind='background_open_data', expectation='no_callback', a...tant speech tv.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech tv. Source=FSDD:2_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.05...type=float32)] == []
E             Left contains one more item: array([-0.05625723,  0.01024601, -0.02873969, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech_music]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech music. Source=FSDD:2_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([-0.04...type=float32)] == []
  Left contains one more item: array([-0.04925577, -0.04274731, -0.05492738, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.165s`

```text
ech_music', kind='background_open_data', expectation='no_callback'...t speech music.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech music. Source=FSDD:2_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([-0.04...type=float32)] == []
E             Left contains one more item: array([-0.04925577, -0.04274731, -0.05492738, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech_hvac]

- Area: `general`
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech hvac. Source=FSDD:2_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00272636, -0.07217696, -0.01834555, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.169s`

```text
eech_hvac', kind='background_open_data', expectation='no_callback',...nt speech hvac.', importance='Distant open-source human speech and media should not start a user turn without intent.')

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_real_usage_callback_expectations(case: RealUsageCase):
        callbacks = _finish_recording_callbacks(case.audio)
        if case.expectation == "callback":
            assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], f"{case.description} Source={case.source}. {case.importance}"
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech hvac. Source=FSDD:2_jackson_0.wav. Distant open-source human speech and media should not start a user turn without intent.
E           assert [array([ 0.00...type=float32)] == []
E             Left contains one more item: array([ 0.00272636, -0.07217696, -0.01834555, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:200: AssertionError
```
