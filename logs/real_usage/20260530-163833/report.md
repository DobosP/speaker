# Real-usage validation: 20260530-163833

**1/5 PASS** (FAILURES PRESENT)


## Results

| fixture | asr_finals (heard) | spoken_response | first_audio | barge_ins | playback | shutdown | VERDICT |
|---|---|---|---|---|---|---|---|
| run-20260530-181513.wav | Poop / I'm conjuring saundering | (silent) | - | 0 | ok | 0.03s | **FAIL** |
| run-20260530-181600.wav | Came hired into me / To hear me | Okay. | 1.26s | 0 | ok | 0.08s | **PASS** |
| run-20260530-182911.wav | (none) | (silent) | - | 0 | ok | 0.18s | **FAIL** |
| run-20260530-183432.wav | He / He / How are you | (silent) | - | 0 | ok | 0.24s | **FAIL** |
| run-20260530-185756.wav | I listening to me | (silent) | - | 0 | ok | 0.15s | **FAIL** |

## Detail

### run-20260530-181513.wav -- FAIL

- **heard (ASR finals):** ['Poop', "I'm conjuring saundering"]
- **spoken response:** (silent)
- **barge-in fires:** 0
- **shutdown:** ok=True seconds=0.028 timeout=8.0
- **reasons:**
    - FAIL: WENT SILENT: no non-control spoken response

### run-20260530-181600.wav -- PASS

- **heard (ASR finals):** ['Came hired into me', 'To hear me']
- **spoken response:** Okay.
- **first-audio latency (s):** [1.2582]
- **barge-in fires:** 0
- **shutdown:** ok=True seconds=0.081 timeout=8.0

### run-20260530-182911.wav -- FAIL

- **heard (ASR finals):** (none -- went deaf)
- **spoken response:** (silent)
- **barge-in fires:** 0
- **shutdown:** ok=True seconds=0.178 timeout=8.0
- **reasons:**
    - FAIL: WENT DEAF: no ASR finals (recognizer produced nothing)
    - FAIL: WENT SILENT: no non-control spoken response

### run-20260530-183432.wav -- FAIL

- **heard (ASR finals):** ['He', 'He', 'How are you']
- **spoken response:** (silent)
- **barge-in fires:** 0
- **shutdown:** ok=True seconds=0.238 timeout=8.0
- **reasons:**
    - FAIL: WENT SILENT: no non-control spoken response

### run-20260530-185756.wav -- FAIL

- **heard (ASR finals):** ['I listening to me']
- **spoken response:** (silent)
- **barge-in fires:** 0
- **shutdown:** ok=True seconds=0.153 timeout=8.0
- **reasons:**
    - FAIL: WENT SILENT: no non-control spoken response

