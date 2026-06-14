# Session 2026-06-14 (pt7) — Computer-use: safety foundation + read-only

**Headline:** Started the **computer-use** capability (let the assistant take
actions — type/click/press/identify) by landing the **security foundation + the
read-only tier**, after a 9-agent research+audit+design workflow and a 7-agent
adversarial security review. The **actuator (click/type/press) is a deliberate
HARD NO-GO** until its preconditions land (owner speaker-ID + a §9.7 local pin +
bound-confirm + live validation). All default-OFF. Full logic suite **1754 passed,
10 skipped**.

## Why only the foundation + read-only

Letting the model drive the keyboard/mouse is the *lethal trifecta* (private data
+ untrusted content + the ability to act). The security review's blockers for any
actuator:
1. **"Came through the mic" ≠ "the owner."** The open-speaker self-echo guard is a
   fuzzy heuristic — ambient/leaked audio ("a video plays *click delete, confirm*")
   could drive a real action **and** approve its own confirmation. The actuator
   must gate on **speaker-ID** (`speaker_gate.py`/`enroll.py`) — and enrollment is
   an owner step still owed.
2. **`ACT` can route to a cloud tier**; actions + screen content must be pinned
   local + forbidden on the `remote/` path.
3. **Confirm is blind-FIFO + spoofable** — must bind to a specific action token +
   owner-verified voice.

So this slice ships the parts that are **safe, high-value, and fully testable now**.

## What landed (commit map)

- `<feat>` — origin cut + read-only screen.identify + wiring + tests
- `<docs/merge>` — this handoff + status

### `always_on_agent/origin.py` (the load-bearing cut — keystone)
Stdlib, Dart-portable. The action-trust chokepoint: `Origin {LIVE_AUDIO trusted;
SCREEN/WEB/MEMORY/FILE/UNKNOWN untrusted}`; `is_action_allowed(origin, *,
owner_verified)` = origin is `LIVE_AUDIO` **and** `owner_verified is True` (strict —
a truthy sentinel does not verify; bakes in "mic ≠ owner"); `combine()` =
most-untrusted-wins lineage (so "owner asks about this screen text" can't launder
it); `should_block_action()` fail-closed; `enforce_action()` raises `ActionBlocked`;
`origin_for_tags()` maps the content-tag vocabulary. The actuator slice wires
`enforce_action` (with real speaker-ID `owner_verified`) in front of every GUI
primitive. Verified fail-closed against every adversarial input by the review.

### `core/ui_grounding.py` (read-only "identify on screen")
`find_targets(words, query)` ranks OCR word boxes (pure, stopword-filtered);
`ocr_words()` via pytesseract `image_to_data` (optional, degrades to `[]`);
`render_elements()` **PII-redacts + `wrap_untrusted()`-fences** screen labels +
reports centers. `attach_computer_use_capability()` registers **`screen.identify`**
(read-only, `side_effecting=False`, `egress=local`, `planner_tool=False`) only when
enabled; result is marked `sensitivity=private`, `origin=screen`. **No actuator** is
registered — verified: zero pyautogui/pynput/ydotool/click/type in shipped code; no
Open Interpreter import. `planner_tool=False` keeps screen text off the
cloud-routable planner until the §9.7 float is enforced.

### Wiring (all default-OFF)
`config.json` `gui_actions` block (`enabled:false`); `--gui-actions` flag;
`core/runtime.py` attaches the read-only capability gated on
`computer_use_config.enabled`; `core/app.py` builds it. Separate opt-in from
`--agent` (the Open Interpreter brain — which the research corrected is **AGPL-3.0,
not MIT**; keep it lazy/optional/unbundled + off `remote/`).

## Tests
`tests/test_origin.py` (adversarial: untrusted blocked, mic≠owner, fail-closed,
lineage, strict `owner_verified`, no-audio attack fixtures → zero actions) +
`tests/test_ui_grounding.py` (ranking, stopword filter, read-only, fenced+redacted,
default-off, planner_tool=False, capture-degrade). 16 tests.

## Reuse decided (from the research)
- **pyautogui (BSD)** as the future actuator backend — NOT Open Interpreter (AGPL);
  **ydotool** (shell-out) for Wayland; **pytesseract/Tesseract (Apache)** for OCR
  grounding (already half-present); a11y trees (pywinauto/pyatspi/pyobjc) as the
  preferred grounding for the actuator slice; **Anthropic computer-use** action
  schema as the primitive contract; **Set-of-Mark** (MIT, pattern only) for VLM
  disambiguation (owner picks, never the VLM).

## Next steps (the actuator — gated, owner-involved)
1. **Owner:** enroll speaker-ID (`python -m core --enroll`) — the precondition that
   makes "owner-verified live audio" real.
2. **Slice 2 (actuator, dry-run default):** `core/gui_control.py` discrete
   primitives (pyautogui backend, Anthropic schema, bounds-checked) registered
   `side_effecting=True`, `planner_tool=False`, behind `enforce_action(origin,
   owner_verified=<speaker-ID verdict>)`; route through the **existing**
   `pending_confirmations` with a **bound** spoken read-back (fix the blind-FIFO
   `_confirm_next`); kill-switch (PAUSE/DISARM latch checked before every event +
   KWS stop + FAILSAFE); per-session rate budget; audit record in the run bundle;
   `dry_run` default-ON. Pin `ACT`/screen turns to a **local** tier + forbid on
   `remote/`. Then live-validate together before flipping `dry_run` off.
3. **Then:** make `screen.identify` a planner tool once the §9.7 float is wired;
   cross-platform reach (ydotool Wayland, a11y grounding); `SECURITY.md` OWASP
   LLM01/LLM06 checklist + CI gate; optional local Prompt-Guard-2 classifier.

Full design + the 7-agent security critique are in the workflow result; the
**first invariant** to keep: *a keyboard/mouse/Enter action may be derived ONLY
from a turn whose entire lineage is owner-verified live audio, at a single
fail-closed chokepoint below planner/router/KWS.*
