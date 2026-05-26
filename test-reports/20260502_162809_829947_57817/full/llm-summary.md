# Test Stage: full

## Result

- Return code: `1`
- Duration: `39.595s`
- Passed: `181`
- Failed: `20`
- Errors: `0`
- Skipped: `0`

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
- Duration: `0.158s`

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
- Duration: `0.156s`

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
- Duration: `0.157s`

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
- Duration: `0.154s`

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
- Duration: `0.158s`

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
- Duration: `0.157s`

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
- Duration: `0.157s`

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
- Duration: `0.158s`

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
- Duration: `0.157s`

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
- Duration: `0.158s`

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
- Duration: `0.157s`

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
- Duration: `0.158s`

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
- Duration: `0.155s`

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
- Duration: `0.159s`

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
- Duration: `0.159s`

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
- Duration: `0.157s`

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
- Duration: `0.158s`

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
- Duration: `0.158s`

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
- Duration: `0.155s`

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
- Duration: `0.157s`

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
