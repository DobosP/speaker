"""Exact optional endpoint-model retirement; never free an admitted native owner."""
from __future__ import annotations

import threading

_REGISTRY_LOCK = threading.Lock()
_RETIRED_LEASES = {}
_UNCERTAIN_LEASES = []


def _require_certain_cleanup():
    with _REGISTRY_LOCK:
        if _UNCERTAIN_LEASES:
            raise RuntimeError("endpoint model cleanup is uncertain; restart the process")
        if _RETIRED_LEASES:
            raise RuntimeError("previous endpoint model cleanup is pending")


def _track_retirement(lease):
    with _REGISTRY_LOCK:
        _RETIRED_LEASES[id(lease)] = lease


def _quarantine(lease):
    with _REGISTRY_LOCK:
        if not any(item is lease for item in _UNCERTAIN_LEASES):
            _UNCERTAIN_LEASES.append(lease)


def _forget_retirement(lease):
    with _REGISTRY_LOCK:
        _RETIRED_LEASES.pop(id(lease), None)


class EndpointModelRetired(RuntimeError):
    pass


class EndpointModelLease:
    """Serialize warm/decode, with one nonblocking retirement request.

    Load/operation/release callbacks are fixed adapter code, never configuration.
    Operations return copied Python data, not native handles. A failed or
    ambiguous release retains its exact object and never authorizes replacement.
    The process-local retirement registry also covers temporary readiness probes.
    """

    def __init__(self, load, release):
        _require_certain_cleanup()
        self._load = load
        self._release = release
        self._state = threading.Lock()
        self._operation = threading.Lock()
        self._native = None
        self._active = False
        self._retired = False
        self._cleanup_started = False
        self._cleanup_failed = False
        self._release_succeeded = False
        self._launch_confirmed = False
        self._failed_native = None
        self._closed = threading.Event()
        self._cleanup_thread = None

    @property
    def closed(self):
        with self._state:
            return self._closed.is_set() and not self._cleanup_failed

    def wait_closed(self, timeout):
        self._closed.wait(timeout)
        return self.closed

    def ensure_open(self):
        with self._state:
            if self._retired:
                raise EndpointModelRetired("endpoint model is retired")
        _require_certain_cleanup()

    def run(self, operation):
        self.ensure_open()
        with self._operation:
            self.ensure_open()
            with self._state:
                if self._retired:
                    raise EndpointModelRetired("endpoint model is retired")
                self._active = True
                native = self._native
            try:
                if native is None:
                    native = self._load()
                    if native is None:
                        raise RuntimeError("endpoint model construction returned no model")
                    with self._state:
                        self._native = native
                with self._state:
                    if self._retired:
                        raise EndpointModelRetired("endpoint model is retired")
                return operation(native)
            finally:
                # Relinquish the borrowed reference before publishing idle;
                # otherwise an idle cleanup could falsely attest RAII release.
                native = None
                with self._state:
                    if self._retired and not self._cleanup_started:
                        self._cleanup_started = True
                        self._launch_confirmed = True  # Already on its admitted native owner.
                        holder = [self._native]
                        self._native = None
                    else:
                        holder = None
                        self._active = False
                if holder is not None:
                    self._finish_release(holder)

    def _publish_closed_locked(self):
        if self._release_succeeded and self._launch_confirmed and not self._cleanup_failed:
            _forget_retirement(self)
            self._closed.set()

    def _finish_release(self, holder):
        native = holder.pop()
        try:
            if native is not None and self._release(native) is not None:
                raise RuntimeError("endpoint model release outcome is invalid")
            native = None  # Drop the last lease-owned RAII reference on this owner.
        except BaseException:
            with self._state:
                self._failed_native = native
                self._cleanup_failed = True
                self._active = False
                _quarantine(self)
            return
        with self._state:
            self._active = False
            self._release_succeeded = True
            self._publish_closed_locked()

    def retire(self):
        with self._state:
            self._retired = True
            if self._closed.is_set() and not self._cleanup_failed:
                return True
            _track_retirement(self)
            if self._cleanup_failed or self._active or self._cleanup_started:
                return False
            self._cleanup_started = True
            holder = [self._native]
            self._native = None
            if holder[0] is None:
                self._release_succeeded = self._launch_confirmed = True
                self._publish_closed_locked()
                return True
        # Idle native destruction can block/re-enter: never run it on stop()
        # or under an engine lock. There is one exact cleanup launch, no retry.
        try:
            thread = threading.Thread(target=self._finish_release, args=(holder,),
                                      name="endpoint-model-retirement", daemon=True)
            self._cleanup_thread = thread
            thread.start()
        except BaseException:
            with self._state:
                self._cleanup_failed = True
                self._closed.clear()
                if self._failed_native is None:
                    self._failed_native = holder
                _quarantine(self)
        else:
            with self._state:
                # A target can finish before Thread.start returns. Its result
                # cannot publish closed until launch certainty is established.
                self._launch_confirmed = True
                self._publish_closed_locked()
        return self.closed


def retire_endpoint_models(owner, *models):
    """Retain only explicitly leased adapters; legacy model lifecycle is inert."""
    lock = owner.__dict__.setdefault("_endpoint_retirement_lock", threading.Lock())
    with lock:
        retained = [m for m in getattr(owner, "_retired_endpoint_models", ())
                    if not m._endpoint_lease.closed]
        for model in models:
            lease = getattr(model, "_endpoint_lease", None)
            if not isinstance(lease, EndpointModelLease):
                continue
            if not any(m is model for m in retained):
                retained.append(model)
            lease.retire()
        owner._retired_endpoint_models = [m for m in retained if not m._endpoint_lease.closed]


def require_endpoint_models_retired(owner):
    _require_certain_cleanup()
    retire_endpoint_models(owner)
    if owner._retired_endpoint_models:
        raise RuntimeError("previous endpoint model cleanup is pending or uncertain")
