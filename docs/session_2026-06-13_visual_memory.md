# Session 2026-06-13 (pt2) — Persistent visual memory (caption + OCR)

**Headline:** The assistant could *see* the current screen (ephemeral per-turn
multimodal context) but never *remembered* it. This adds optional, **default-OFF**
persistent visual memory: each new screen frame becomes — off the hot path — a
short **local** multimodal caption **+ an OCR snippet**, stored as a recallable
`vision` memory and surfaced through the **same** token-budgeted smart-recall layer
shipped earlier today.

**Branch:** `feat/visual-memory` → merged to `main`.
**Commits:** `b6952c8` (feat) + `d94ec3c` (adversarial-review fixes).
**Verdict:** `.venv/bin/python -m pytest tests` → **1614 passed / 13 skipped / 0 failed**.
(Use `.venv/bin/python` — system anaconda python lacks `psycopg`.)

## What changed

| File | Change |
|---|---|
| `core/visual_memory.py` **(NEW)** | `VisualMemorizer`: throttled (min-interval) + on-change (sha1 fingerprint) background worker (bounded drop-oldest queue) so captioning never runs per-frame or on the bus/answer thread. `caption_fn`/`ocr_fn`/`ingest` injectable; defaults `llm.generate(images=)` + `pytesseract`, degrade to skip when absent. `compose()` guards each component. `build_visual_memorizer` wires it. |
| `utils/memory.py` | `add_observation()` persists `role='observation'`/`source='vision'` **bypassing `_extract_profile`**; `clear_observations()` purge; `_search_observations()` (separate query); `get_context_for_llm` is now **3 reserved-floor sub-passes** (recall / vision / profile) sharing one `max_tokens`. |
| `always_on_agent/recall.py` | `Candidate kind='vision'` → `Screen:` label; `VISION_LABEL`. |
| `always_on_agent/memory.py` | `MemoryManagerAdapter.add` routes the `vision` tag to `add_observation`; `SessionMemory._candidates` maps it to `kind='vision'`. |
| `core/capabilities.py` | a turn recalling any `Screen:` line floats sensitivity **PRIVATE**. |
| `core/llm_factory.py` | `_tag_local_main` stamps the cloud-wrapped main with `.local_main` (the bare local handle). |
| `core/app.py` | builds the memorizer with `getattr(llm,'local_main',llm)`; wires it as the screen-feed observer; start/stop. |
| `core/screen_capture.py` | `ScreenFrameFeed` gains an optional `observer`. |
| `core/runtime.py` | addressing-classifier + transcript-cleaner recent windows now exclude `vision` items. |
| `config.json` | `screen_capture.memorize*` knobs, all default OFF. |
| `tests/test_visual_memory.py` **(NEW)** | ~24 Tier-0 tests (no display/model/tesseract/DB). |

## Adversarial review (20 agents) — what it caught + fixed

- **BLOCKER §9.7 leak:** captioning used the MAIN llm, which is **cloud-wrapped**
  when `llm.cloud.enabled` + keys are set → raw screen frame **bytes** went to the
  cloud (default runs were safe; the supported cloud config was not). **Fixed:**
  caption on the bare local handle only; if a caller can't supply one and the
  client is cloud-capable, captioning hard-disables (OCR-only).
- **MAJOR:** vision leaked into the addressing/cleaner recent windows → tag-filtered.
- **MAJOR:** vision could starve user-message recall → separate query + reserved
  sub-budget (combined still ≤ `max_tokens`).
- **MINOR:** `stop()`/`start()` orphaned-worker race → fixed.
- Plus degradation/empty-trace tests + docstrings.

## How to use it

```jsonc
// config.json (or a per-machine config.local.json)
"screen_capture": { "enabled": true, "memorize": true }
```
Needs `mss` (capture), and `pytesseract`+`Pillow`+tesseract for OCR (degrades to
caption-only without them). Captioning needs a local multimodal model (Gemma 3).
Purge with `MemoryManager.clear_observations()`.

## Next steps (pick up here)

1. **Live validation on a real screen** (needs a display + a local multimodal model):
   enable, browse, ask "what was I looking at?" — confirm the caption/OCR trace is
   recalled and reads well; tune `memorize_min_interval_sec` / `memorize_on_change`.
2. **Optional packaging:** add `mss`/`pytesseract`/`Pillow` to an extras group; they
   are intentionally optional today (feature degrades without them).
3. Same recall follow-ups as `docs/session_2026-06-13_smart_memory_recall.md`
   (bench + enable recall; the optional `005` salience/hybrid migration; recall on
   the ReAct/escalated path) — visual memory rides the same recall layer, so those
   apply to it too.
