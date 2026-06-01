# Real-usage validation: 20260530-163032

**1/5 PASS** (FAILURES PRESENT)


## Results

| fixture | asr_finals (heard) | spoken_response | first_audio | barge_ins | playback | shutdown | VERDICT |
|---|---|---|---|---|---|---|---|
| run-20260530-181513.wav | Our countrying sometim | (silent) | - | 0 | ok | 0.09s | **FAIL** |
| run-20260530-181600.wav | Came hired into me / To hear me | Okay. | 1.29s | 0 | ok | 0.10s | **PASS** |
| run-20260530-182911.wav | (none) | (silent) | - | 0 | ok | 0.06s | **FAIL** |
| run-20260530-183432.wav | He / He | (silent) | - | 0 | ok | 0.08s | **FAIL** |
| run-20260530-185756.wav | I listening to me | (silent) | - | 0 | ok | 0.15s | **FAIL** |

## Detail

### run-20260530-181513.wav -- FAIL

- **heard (ASR finals):** ['Our countrying sometim']
- **spoken response:** (silent)
- **barge-in fires:** 0
- **shutdown:** ok=True seconds=0.085 timeout=8.0
- **reasons:**
    - FAIL: WENT SILENT: no non-control spoken response

### run-20260530-181600.wav -- PASS

- **heard (ASR finals):** ['Came hired into me', 'To hear me']
- **spoken response:** Okay.
- **first-audio latency (s):** [1.2858]
- **barge-in fires:** 0
- **shutdown:** ok=True seconds=0.102 timeout=8.0

### run-20260530-182911.wav -- FAIL

- **heard (ASR finals):** (none -- went deaf)
- **spoken response:** (silent)
- **barge-in fires:** 0
- **shutdown:** ok=True seconds=0.056 timeout=8.0
- **reasons:**
    - FAIL: WENT DEAF: no ASR finals (recognizer produced nothing)
    - FAIL: WENT SILENT: no non-control spoken response

### run-20260530-183432.wav -- FAIL

- **heard (ASR finals):** ['He', 'He']
- **spoken response:** (silent)
- **barge-in fires:** 0
- **shutdown:** ok=True seconds=0.084 timeout=8.0
- **reasons:**
    - FAIL: WENT SILENT: no non-control spoken response

### run-20260530-185756.wav -- FAIL

- **heard (ASR finals):** ['I listening to me']
- **spoken response:** (silent)
- **barge-in fires:** 0
- **shutdown:** ok=True seconds=0.153 timeout=8.0
- **reasons:**
    - FAIL: WENT SILENT: no non-control spoken response

