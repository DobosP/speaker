"""Watch / monitor a granted application for a described event (v1).

The owner asks the assistant to watch ONE explicitly-granted app window for a
condition; a single shared poller observes only that window, checks the condition
locally, and -- when it flips false->true -- speaks one alert. Observe + speak only;
a watch can NEVER actuate (no keyboard/mouse/code), and the model can never arm one.

Security model (fail-closed; see the invariants on each enforcement point):
- DEFAULT-DENY: nothing is watchable. ``config.json`` ships ``grants: []``; grants
  live ONLY in machine-local ``config.local.json`` (gitignored) so they never travel.
- OWNER-VERIFIED: ``watch.grant`` / ``watch.start`` / ``watch.stop`` route through the
  always_on_agent.origin chokepoint (owner-verified LIVE_AUDIO) -- the SAME gate the
  computer-use actuator uses. ``planner_tool=False`` keeps every watch capability off
  the (cloud-routable) ReAct planner, so an LLM/injection can never arm/egress one.
- SCOPED + EPHEMERAL: capture is one granted window (core/watch_source.py, never
  full-screen); frames are discarded inside ``tick`` and never persisted.
- LOCAL: v1 condition-checking is a pure text match -- no LLM, no network. Any text
  that ever reaches a spoken alert is ``redact_pii(force=True)`` + spotlight-fenced.
- BOUNDED: a hard cap on active watches + a minimum poll interval (no tight loop).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from always_on_agent.capabilities import CapabilityRegistry, CapabilityResult, CapabilitySpec
from always_on_agent.events import AgentEvent, EventKind
from always_on_agent.origin import Origin, should_block_action
from always_on_agent.untrusted import redact_pii, wrap_untrusted

from .watch_source import Observation, WatchSource, make_watch_source, safe_search

log = logging.getLogger("speaker.watch")

_EVIDENCE_CAP = 160  # chars of matched context surfaced in an alert (pre-redaction)


# --- data ------------------------------------------------------------------

@dataclass(frozen=True)
class WatchGrant:
    """A per-app ALLOWLIST entry: what the owner has permitted to be watched."""
    id: str
    label: str
    app: dict
    min_poll_sec: float = 5.0
    granted_at: str = ""
    granted_by: str = "owner_verified"

    @classmethod
    def from_dict(cls, d: dict) -> "WatchGrant":
        return cls(
            id=str(d.get("id", "")),
            label=str(d.get("label", "") or d.get("id", "")),
            app=dict(d.get("app", {}) or {}),
            min_poll_sec=max(0.0, float(d.get("min_poll_sec", 5.0) or 5.0)),
            granted_at=str(d.get("granted_at", "")),
            granted_by=str(d.get("granted_by", "owner_verified")),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label, "app": self.app,
            "min_poll_sec": self.min_poll_sec, "granted_at": self.granted_at,
            "granted_by": self.granted_by,
        }


@dataclass
class ActiveWatch:
    """A LIVE watch over a granted app (session-only -- not persisted)."""
    watch_id: str
    grant_id: str
    condition: str
    interval_sec: float
    one_shot: bool = True
    last_met: bool = False
    cancelled: bool = False
    next_due: float = 0.0


# --- evaluator (pure, local) -----------------------------------------------

class TextMatchEvaluator:
    """Pure, local condition check over a window's OCR text. v1 grammar:
    ``/regex/`` -> case-insensitive regex search; otherwise a case-insensitive
    substring. Returns ``(met, evidence)`` where evidence is the raw matched snippet
    (the caller redacts+fences it before it can be spoken)."""

    def evaluate(self, obs: Observation, condition: str) -> tuple[bool, str]:
        text = (obs.text or "")
        cond = (condition or "").strip()
        if not cond or not text:
            return False, ""
        if len(cond) >= 2 and cond.startswith("/") and cond.endswith("/"):
            # ReDoS-guarded + length-bounded: an owner /regex/ runs on the shared
            # poller thread against attacker-influenceable OCR text.
            m = safe_search(cond[1:-1], text)
            return (bool(m), self._snippet(text, m.start(), m.end())) if m else (False, "")
        idx = text.lower().find(cond.lower())
        if idx < 0:
            return False, ""
        return True, self._snippet(text, idx, idx + len(cond))

    @staticmethod
    def _snippet(text: str, start: int, end: int) -> str:
        pad = max(0, (_EVIDENCE_CAP - (end - start)) // 2)
        return text[max(0, start - pad): end + pad].strip()[:_EVIDENCE_CAP]


# --- grant store (machine-local persistence) -------------------------------

class GrantStore:
    """Holds the in-memory grant allowlist; persists changes to ``config.local.json``
    ONLY (never the committed config.json). Default-deny: an empty base means nothing
    is watchable until the owner grants."""

    def __init__(self, grants: Optional[list] = None, *, local_path: str = "config.local.json"):
        self._grants: list[WatchGrant] = [WatchGrant.from_dict(g) for g in (grants or []) if g.get("id")]
        self._local_path = local_path
        self._lock = threading.Lock()

    def list(self) -> tuple[WatchGrant, ...]:
        with self._lock:
            return tuple(self._grants)

    def get(self, grant_id: str) -> Optional[WatchGrant]:
        gid = str(grant_id or "")
        with self._lock:
            return next((g for g in self._grants if g.id == gid), None)

    def add(self, grant: WatchGrant) -> None:
        with self._lock:
            self._grants = [g for g in self._grants if g.id != grant.id] + [grant]
            self._persist()

    def remove(self, grant_id: str) -> bool:
        gid = str(grant_id or "")
        with self._lock:
            before = len(self._grants)
            self._grants = [g for g in self._grants if g.id != gid]
            if len(self._grants) == before:
                return False
            self._persist()
            return True

    def _persist(self) -> None:
        """Atomically write the grants into config.local.json['watch']['grants'],
        preserving every other machine-local key."""
        data: dict = {}
        if os.path.exists(self._local_path):
            try:
                with open(self._local_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or {}
            except Exception as exc:  # noqa: BLE001
                # An unreadable local file must NOT be silently overwritten with {} --
                # that would destroy every other machine-local key (model paths, device,
                # cloud settings). Back it up and abort the grant loudly instead.
                backup = f"{self._local_path}.corrupt"
                try:
                    os.replace(self._local_path, backup)
                except OSError:
                    backup = self._local_path
                raise RuntimeError(
                    f"{self._local_path} is unreadable (backed up to {backup}); "
                    f"refusing to overwrite it -- fix or remove it, then re-grant."
                ) from exc
        watch = dict(data.get("watch", {}) or {})
        watch["enabled"] = True  # a grant means the owner wants watching enabled on THIS machine
        watch["grants"] = [g.to_dict() for g in self._grants]
        data["watch"] = watch
        tmp = f"{self._local_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, self._local_path)


# --- manager (single shared poller) ----------------------------------------

class WatchManager:
    """Tracks active watches and runs them on ONE shared poller thread (so N watches
    cost one thread, never N of the 6-task pool). ``tick`` is the unit-test seam: it
    runs exactly one poll pass; the poller loop just calls it on a cadence."""

    def __init__(
        self,
        store: GrantStore,
        source: Optional[WatchSource] = None,
        *,
        publish: Callable[[AgentEvent], None],
        current_epoch: Callable[[], int],
        evaluator: Optional[TextMatchEvaluator] = None,
        max_active: int = 2,
        min_poll_sec: float = 5.0,
        autostart: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._store = store
        self._source = source if source is not None else make_watch_source()
        self._publish = publish
        self._current_epoch = current_epoch
        self._evaluator = evaluator or TextMatchEvaluator()
        self._max_active = max(1, int(max_active))
        self._min_poll_sec = max(1.0, float(min_poll_sec))
        self._autostart = autostart
        self._clock = clock
        self._sleep = sleep
        self._active: dict[str, ActiveWatch] = {}
        self._lock = threading.Lock()
        self._poller: Optional[threading.Thread] = None
        self._shutdown = False
        self._seq = 0

    # -- arming / disarming --
    def start_watch(self, grant_id: str, condition: str, *, interval_sec: float, one_shot: bool = True) -> str:
        grant = self._store.get(grant_id)
        if grant is None:
            raise ValueError(f"no grant named {grant_id!r}")  # not in the allowlist (INV-7)
        with self._lock:
            if len([w for w in self._active.values() if not w.cancelled]) >= self._max_active:
                raise RuntimeError(f"watch limit reached ({self._max_active})")  # INV-5
            # Clamp the poll interval UP only -- never below the floor (INV-5).
            interval = max(self._min_poll_sec, float(grant.min_poll_sec), float(interval_sec))
            self._seq += 1
            watch_id = f"w{self._seq}"
            self._active[watch_id] = ActiveWatch(
                watch_id=watch_id, grant_id=grant.id, condition=str(condition or ""),
                interval_sec=interval, one_shot=bool(one_shot), next_due=self._clock(),
            )
        self._maybe_start_poller()
        return watch_id

    def stop_watch(self, watch_id: Optional[str]) -> int:
        with self._lock:
            if watch_id is None:
                n = len([w for w in self._active.values() if not w.cancelled])
                self._active.clear()
                return n
            w = self._active.pop(str(watch_id), None)
            return 1 if w is not None else 0

    def active(self) -> tuple[ActiveWatch, ...]:
        with self._lock:
            return tuple(w for w in self._active.values() if not w.cancelled)

    # -- one poll pass (the test seam) --
    def tick(self) -> None:
        now = self._clock()
        with self._lock:
            due = [w for w in self._active.values() if not w.cancelled and now >= w.next_due]
        for w in due:
            w.next_due = now + w.interval_sec
            grant = self._store.get(w.grant_id)
            if grant is None:
                # Grant revoked mid-loop -> stop watching immediately (INV-2/7).
                self.stop_watch(w.watch_id)
                continue
            obs = self._source.observe(grant.app)
            if obs is None:
                continue  # window not resolvable this pass; nothing captured/leaked
            try:
                met, evidence = self._evaluator.evaluate(obs, w.condition)
            except Exception:  # noqa: BLE001 - a bad condition must not kill the poller
                log.debug("watch evaluate failed", exc_info=True)
                met, evidence = False, ""
            if met and not w.last_met:
                self._fire(w, grant.label, evidence)
                if w.one_shot:
                    self.stop_watch(w.watch_id)
            w.last_met = met
            # ``obs`` (and any frame-derived text) goes out of scope here -- ephemeral.

    def _fire(self, w: ActiveWatch, label: str, evidence: str) -> None:
        # Single egress chokepoint: redact (force=True, ignores the local-record
        # kill-switch) then spotlight-fence any captured text before it can be spoken
        # (INV-4). The alert is system-initiated -- epoch-stamped, never carrying the
        # granting turn's owner trust (INV-6), so watched screen text can shape a
        # spoken heads-up but can never originate an action. The owner-supplied label
        # AND condition are ALSO redacted: they can carry PII ("watch Mom's bank
        # +1-415-...") and the alert text lands in the git-committed run bundle (§9.7).
        safe_label = redact_pii(label or "", force=True)
        safe_cond = redact_pii(w.condition or "", force=True)
        safe = redact_pii(evidence or "", force=True)
        fenced = wrap_untrusted(safe, source="screen") if safe else ""
        alert = f"Heads up: {safe_label} -- {safe_cond}." + (f"\n{fenced}" if fenced else "")
        try:
            epoch = int(self._current_epoch())
        except Exception:  # noqa: BLE001
            epoch = 0
        self._publish(AgentEvent(
            EventKind.TTS_REQUEST,
            {"task_id": w.watch_id, "text": alert, "epoch": epoch, "origin": Origin.SCREEN.value},
        ))
        log.info("watch fired: id=%s grant=%s", w.watch_id, w.grant_id)  # metadata only -- no frame/raw OCR

    # -- poller lifecycle --
    def _maybe_start_poller(self) -> None:
        if not self._autostart:
            return
        with self._lock:
            if self._poller is not None or self._shutdown:
                return
            self._poller = threading.Thread(target=self._run, name="watch-poller", daemon=True)
            self._poller.start()

    def _run(self) -> None:
        while not self._shutdown:
            self._sleep(self._min_poll_sec)
            if self._shutdown:
                break
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the poller must never die on one bad pass
                log.exception("watch poller tick failed")

    def shutdown(self) -> None:
        self._shutdown = True
        with self._lock:
            self._active.clear()


# --- capability registration -----------------------------------------------

def _redact_for_log(text: str) -> str:
    """Scrub a condition phrase before it lands in a committed run bundle -- the
    phrase itself can carry PII ("alert when <person> messages")."""
    return redact_pii(str(text or "")[:120], force=True)


def attach_watch_capability(
    registry: CapabilityRegistry,
    watch_cfg: dict,
    *,
    manager: WatchManager,
) -> CapabilityRegistry:
    """Register ``watch.grant`` / ``watch.start`` / ``watch.stop`` (owner-verified,
    side-effecting) and ``watch.list`` (read-only). Default-OFF: registers nothing
    unless ``watch_cfg['enabled']``. All four are ``planner_tool=False`` and
    ``user_facing=False`` -- kept off the answering model's self-description AND off
    the cloud-routable ReAct planner, so the model can never arm/list/egress a
    watcher. ``manager`` owns the poller + the grant store."""
    if not watch_cfg or not watch_cfg.get("enabled"):
        return registry

    store = manager._store
    default_poll = max(manager._min_poll_sec, float(watch_cfg.get("default_poll_sec", 15.0) or 15.0))

    def _blocked(context: dict) -> Optional[CapabilityResult]:
        """The owner-verified chokepoint, copied from core/agent.py:390-398. Returns
        a refusal CapabilityResult when the turn is not owner-verified LIVE_AUDIO, else
        None (allowed)."""
        origin = context.get("origin", Origin.UNKNOWN)
        owner_verified = context.get("owner_verified", False)
        if should_block_action(origin, owner_verified=owner_verified):
            return CapabilityResult(
                True,
                "I can't change what I watch without verified-owner authorization.",
                data={"executed": False, "blocked": "owner_verification"},
            )
        return None

    def watch_grant(query: str, context: dict) -> CapabilityResult:
        blocked = _blocked(context)
        if blocked is not None:
            return blocked
        grant_spec = context.get("grant")
        if not isinstance(grant_spec, dict) or not grant_spec.get("id") or not (grant_spec.get("app") or {}).get("wm_class"):
            return CapabilityResult(False, "I need an application id and window identity to grant a watch.",
                                    error="missing grant id/app.wm_class")
        grant = WatchGrant.from_dict({**grant_spec, "granted_by": "owner_verified"})
        store.add(grant)
        log.info("watch grant added: id=%s", grant.id)
        return CapabilityResult(True, f"You can now ask me to watch {grant.label}.",
                                data={"executed": True, "grant_id": grant.id})

    def watch_start(query: str, context: dict) -> CapabilityResult:
        blocked = _blocked(context)
        if blocked is not None:
            return blocked
        grant_id = str(context.get("grant_id", "") or "")
        condition = str(context.get("condition", "") or query or "")
        if not grant_id or not condition.strip():
            return CapabilityResult(False, "Tell me which granted app to watch and what to watch for.",
                                    error="missing grant_id/condition")
        grant = store.get(grant_id)
        if grant is None:
            return CapabilityResult(True, f"I don't have permission to watch {grant_id}. Grant it first.",
                                    data={"executed": False, "blocked": "not_granted"})
        try:
            interval = float(context.get("interval_sec", default_poll) or default_poll)
            one_shot = bool(context.get("one_shot", True))
            watch_id = manager.start_watch(grant_id, condition, interval_sec=interval, one_shot=one_shot)
        except (ValueError, RuntimeError) as exc:
            return CapabilityResult(True, f"I can't start that watch: {exc}",
                                    data={"executed": False, "blocked": str(exc)})
        log.info("watch started: id=%s grant=%s condition=%s", watch_id, grant_id, _redact_for_log(condition))
        return CapabilityResult(True, f"Watching {grant.label}. I'll let you know.",
                                data={"executed": True, "watch_id": watch_id})

    def watch_stop(query: str, context: dict) -> CapabilityResult:
        blocked = _blocked(context)
        if blocked is not None:
            return blocked
        watch_id = context.get("watch_id")
        n = manager.stop_watch(str(watch_id) if watch_id else None)
        return CapabilityResult(True, ("Stopped watching." if n else "I wasn't watching anything."),
                                data={"executed": True, "stopped": n})

    def watch_list(query: str, context: dict) -> CapabilityResult:
        # Read-only: no owner-verify needed (no action, no laundering risk).
        grants = store.list()
        active = manager.active()
        if not grants and not active:
            return CapabilityResult(True, "I'm not watching anything, and nothing is granted.",
                                    data={"grants": 0, "active": 0})
        parts = []
        if active:
            parts.append("Currently watching: " + ", ".join(
                f"{(store.get(w.grant_id).label if store.get(w.grant_id) else w.grant_id)}" for w in active))
        if grants:
            parts.append("Granted apps: " + ", ".join(g.label for g in grants))
        return CapabilityResult(True, " ".join(parts), data={"grants": len(grants), "active": len(active)})

    _LOCAL = "local"
    registry.register("watch.grant", watch_grant, spec=CapabilitySpec(
        name="watch.grant",
        summary="Allow watching a specific application (owner-verified).",
        when_to_use="When the owner explicitly asks to let me watch/monitor a specific application.",
        egress=_LOCAL, speaks=True, side_effecting=True, planner_tool=False, user_facing=False))
    registry.register("watch.start", watch_start, spec=CapabilitySpec(
        name="watch.start",
        summary="Start watching an already-granted app for a described event.",
        when_to_use="When the owner asks me to watch an already-granted app for an event.",
        egress=_LOCAL, speaks=True, side_effecting=True, planner_tool=False, user_facing=False))
    registry.register("watch.stop", watch_stop, spec=CapabilitySpec(
        name="watch.stop",
        summary="Stop watching an app.",
        when_to_use="When the owner asks me to stop watching.",
        egress=_LOCAL, speaks=True, side_effecting=True, planner_tool=False, user_facing=False))
    registry.register("watch.list", watch_list, spec=CapabilitySpec(
        name="watch.list",
        summary="Say what I'm currently watching and which apps are granted.",
        when_to_use="When the owner asks what I'm watching.",
        egress=_LOCAL, speaks=True, side_effecting=False, planner_tool=False, user_facing=False))
    log.info("watch: grant/start/stop/list registered (default-deny, owner-verified)")
    return registry
