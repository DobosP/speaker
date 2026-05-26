# Test Stage: full

## Result

- Return code: `1`
- Duration: `144.05s`
- Passed: `586`
- Failed: `160`
- Errors: `0`
- Skipped: `0`

## Quality Metrics

- Failure categories: `{'unsolvable_without_intent_gate': 160}`
- Failure solvability: `{'unsolvable': 160}`
- False accept failures: `160`
- False reject failures: `0`
- False accept failure rate: `0.2145`
- False reject failure rate: `0.0`

## Failures

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00443938, -0.00551872, -0.00542672, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.167s`

```text
  assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00443938, -0.00551872, -0.00542672, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_tv]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech tv. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_tv' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00251688,  0.01493537, -0.0097715 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.161s`

```text
rt callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech tv. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_tv' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00251688,  0.01493537, -0.0097715 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_music]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech music. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_music' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.04...type=float32)] == []
  Left contains one more item: array([-0.0416458 , -0.02648572, -0.04670901, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.159s`

```text
lbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech music. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_music' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.04...type=float32)] == []
E             Left contains one more item: array([-0.0416458 , -0.02648572, -0.04670901, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_hvac]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech hvac. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_hvac' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00934406, -0.05422705, -0.01041681, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.162s`

```text
allbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech hvac. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_hvac' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([ 0.00...type=float32)] == []
E             Left contains one more item: array([ 0.00934406, -0.05422705, -0.01041681, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_clipped_speaker]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech clipped speaker. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_clipped_speaker' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.01...type=float32)] == []
  Left contains one more item: array([-0.01115352, -0.01386222, -0.01362964, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.161s`

```text
ription} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech clipped speaker. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_clipped_speaker' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.01...type=float32)] == []
E             Left contains one more item: array([-0.01115352, -0.01386222, -0.01362964, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_resampled]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech resampled. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_resampled' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00444159, -0.00551849, -0.00542619, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.165s`

```text
f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech resampled. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_resampled' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00444159, -0.00551849, -0.00542619, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_babble]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech babble. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_babble' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00047911,  0.00616352,  0.01122724, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.162s`

```text
acks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech babble. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_babble' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00047911,  0.00616352,  0.01122724, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_0_distant_speech_low_rumble]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 0 with distant speech low rumble. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_low_rumble' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00097772,  0.026464  , -0.06335104, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.164s`

```text
{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 0 with distant speech low rumble. Source=FSDD:0_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_low_rumble' for speaker 'jackson' digit 0. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([ 0.00...type=float32)] == []
E             Left contains one more item: array([ 0.00097772,  0.026464  , -0.06335104, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00746937, -0.0091225 , -0.00900048, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.166s`

```text
  assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00746937, -0.0091225 , -0.00900048, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_tv]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech tv. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_tv' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.04...type=float32)] == []
  Left contains one more item: array([-0.04925915, -0.01029904,  0.01539566, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.161s`

```text
rt callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech tv. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_tv' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.04...type=float32)] == []
E             Left contains one more item: array([-0.04925915, -0.01029904,  0.01539566, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_music]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech music. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_music' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.04...type=float32)] == []
  Left contains one more item: array([-0.04651193, -0.03112038, -0.05231738, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.166s`

```text
lbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech music. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_music' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.04...type=float32)] == []
E             Left contains one more item: array([-0.04651193, -0.03112038, -0.05231738, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_hvac]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech hvac. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_hvac' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00713269, -0.06071435, -0.0142862 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.179s`

```text
allbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech hvac. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_hvac' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([ 0.00...type=float32)] == []
E             Left contains one more item: array([ 0.00713269, -0.06071435, -0.0142862 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_clipped_speaker]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech clipped speaker. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_clipped_speaker' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.01...type=float32)] == []
  Left contains one more item: array([-0.01876106, -0.02290962, -0.02260645, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.169s`

```text
ription} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech clipped speaker. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_clipped_speaker' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.01...type=float32)] == []
E             Left contains one more item: array([-0.01876106, -0.02290962, -0.02260645, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_resampled]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech resampled. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_resampled' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00746865, -0.00912169, -0.00899946, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.162s`

```text
f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech resampled. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_resampled' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00746865, -0.00912169, -0.00899946, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_babble]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech babble. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_babble' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.0032719 ,  0.00325518,  0.00864023, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.162s`

```text
acks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech babble. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_babble' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.0032719 ,  0.00325518,  0.00864023, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_1_distant_speech_low_rumble]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 1 with distant speech low rumble. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_low_rumble' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.12...type=float32)] == []
  Left contains one more item: array([-0.12100139, -0.0561235 , -0.07169528, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.165s`

```text
{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 1 with distant speech low rumble. Source=FSDD:1_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_low_rumble' for speaker 'jackson' digit 1. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.12...type=float32)] == []
E             Left contains one more item: array([-0.12100139, -0.0561235 , -0.07169528, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.01...type=float32)] == []
  Left contains one more item: array([-0.01160707, -0.0215311 , -0.01315458, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.164s`

```text
  assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.01...type=float32)] == []
E             Left contains one more item: array([-0.01160707, -0.0215311 , -0.01315458, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech_tv]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech tv. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_tv' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.05...type=float32)] == []
  Left contains one more item: array([-0.05625657,  0.01024732, -0.02874099, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.162s`

```text
rt callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech tv. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_tv' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.05...type=float32)] == []
E             Left contains one more item: array([-0.05625657,  0.01024732, -0.02874099, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech_music]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech music. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_music' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.04...type=float32)] == []
  Left contains one more item: array([-0.04925624, -0.04274739, -0.05492637, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.164s`

```text
lbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech music. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_music' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.04...type=float32)] == []
E             Left contains one more item: array([-0.04925624, -0.04274739, -0.05492637, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech_hvac]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech hvac. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_hvac' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00272518, -0.07217569, -0.01834618, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.163s`

```text
allbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech hvac. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_hvac' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([ 0.00...type=float32)] == []
E             Left contains one more item: array([ 0.00272518, -0.07217569, -0.01834618, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech_clipped_speaker]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech clipped speaker. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_clipped_speaker' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.02...type=float32)] == []
  Left contains one more item: array([-0.02915442, -0.05408331, -0.03304401, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.163s`

```text
ription} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech clipped speaker. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_clipped_speaker' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.02...type=float32)] == []
E             Left contains one more item: array([-0.02915442, -0.05408331, -0.03304401, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech_resampled]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech resampled. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_resampled' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.01...type=float32)] == []
  Left contains one more item: array([-0.01160595, -0.02153188, -0.0131555 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.164s`

```text
f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech resampled. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_resampled' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.01...type=float32)] == []
E             Left contains one more item: array([-0.01160595, -0.02153188, -0.0131555 , ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech_babble]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech babble. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_babble' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00748707, -0.00938489,  0.00416233, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.163s`

```text
acks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech babble. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_babble' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00748707, -0.00938489,  0.00416233, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_2_distant_speech_low_rumble]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 2 with distant speech low rumble. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_low_rumble' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([ 0.03...type=float32)] == []
  Left contains one more item: array([ 0.03901066, -0.05867141,  0.04072711, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.162s`

```text
{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 2 with distant speech low rumble. Source=FSDD:2_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_low_rumble' for speaker 'jackson' digit 2. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([ 0.03...type=float32)] == []
E             Left contains one more item: array([ 0.03901066, -0.05867141,  0.04072711, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_3_distant_speech]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 3 with distant speech. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00884857, -0.01139922, -0.00615325, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.162s`

```text
  assert callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 3 with distant speech. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00884857, -0.01139922, -0.00615325, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_3_distant_speech_tv]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 3 with distant speech tv. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_tv' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00108608, -0.0030776 ,  0.02354525, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.164s`

```text
rt callbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 3 with distant speech tv. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_tv' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([ 0.00...type=float32)] == []
E             Left contains one more item: array([ 0.00108608, -0.0030776 ,  0.02354525, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_3_distant_speech_music]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 3 with distant speech music. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_music' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.04...type=float32)] == []
  Left contains one more item: array([-0.04608474, -0.03238357, -0.04746581, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.162s`

```text
lbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 3 with distant speech music. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_music' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.04...type=float32)] == []
E             Left contains one more item: array([-0.04608474, -0.03238357, -0.04746581, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_3_distant_speech_hvac]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 3 with distant speech hvac. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_hvac' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([ 0.00...type=float32)] == []
  Left contains one more item: array([ 0.00535805, -0.06160095, -0.01129592, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.159s`

```text
allbacks, f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 3 with distant speech hvac. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_hvac' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([ 0.00...type=float32)] == []
E             Left contains one more item: array([ 0.00535805, -0.06160095, -0.01129592, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_3_distant_speech_clipped_speaker]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 3 with distant speech clipped speaker. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_clipped_speaker' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.02...type=float32)] == []
  Left contains one more item: array([-0.02222531, -0.02863576, -0.01545123, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.158s`

```text
ription} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 3 with distant speech clipped speaker. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_clipped_speaker' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.02...type=float32)] == []
E             Left contains one more item: array([-0.02222531, -0.02863576, -0.01545123, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

### tests.test_real_usage_open_data.test_real_usage_callback_expectations[background_jackson_3_distant_speech_resampled]

- Area: `real_usage`
- Category: `unsolvable_without_intent_gate`
- Solvable: `False`
- Cause: The input is real human speech-like background audio. In ungated listening mode the recorder has no wakeword, speaker, or intent signal, so raw audio alone cannot prove whether speech is directed at the assistant.
- Suggested fix: Use wakeword/speaker verification/intent gating for background speech rejection, or change the product requirement for ungated mode.
- Message: AssertionError: Non-command background speech from jackson digit 3 with distant speech resampled. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_resampled' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
assert [array([-0.00...type=float32)] == []
  Left contains one more item: array([-0.00884739, -0.01140011, -0.00615289, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
  Use -v to get more diff
- Duration: `0.163s`

```text
f"{case.description} Source={case.source}. {case.importance}"
        elif case.expectation == "no_callback":
>           assert callbacks == [], (
                f"{case.description} Source={case.source}. {case.importance} "
                "Raw speech-like background audio cannot be reliably rejected in "
                "ungated mode because there is no intent signal."
            )
E           AssertionError: Non-command background speech from jackson digit 3 with distant speech resampled. Source=FSDD:3_jackson_0.wav. Open-data background speech should stay ignored in scenario 'distant_speech_resampled' for speaker 'jackson' digit 3. CAUSE=ungated_audio_has_no_intent_signal. CATEGORY=unsolvable_without_intent_gate. MITIGATION=enable_wakeword_or_speaker_verification. Raw speech-like background audio cannot be reliably rejected in ungated mode because there is no intent signal.
E           assert [array([-0.00...type=float32)] == []
E             Left contains one more item: array([-0.00884739, -0.01140011, -0.00615289, ...,  0.        ,\n        0.        ,  0.        ], dtype=float32)
E             Use -v to get more diff

tests/test_real_usage_open_data.py:250: AssertionError
```

... 130 more failures omitted. See failures.json.