"""
OVOS-style capability registry and adapters.

This module provides a small plugin-like capability bus that can be extended
with tool providers (memory search, device control, utilities, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import time


@dataclass
class CapabilityRequest:
    name: str
    payload: Dict[str, Any]
    session_id: Optional[str] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp <= 0:
            self.timestamp = time.time()


@dataclass
class CapabilityResponse:
    ok: bool
    data: Dict[str, Any]
    error: str = ""


class CapabilityRegistry:
    """Simple capability registry with function providers."""

    def __init__(self):
        self._providers: Dict[str, Callable[[CapabilityRequest], CapabilityResponse]] = {}

    def register(
        self,
        name: str,
        provider: Callable[[CapabilityRequest], CapabilityResponse],
        overwrite: bool = False,
    ):
        if not overwrite and name in self._providers:
            raise ValueError(f"Capability already registered: {name}")
        self._providers[name] = provider

    def unregister(self, name: str):
        self._providers.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._providers

    def list_capabilities(self) -> list[str]:
        return sorted(self._providers.keys())

    def invoke(self, req: CapabilityRequest) -> CapabilityResponse:
        provider = self._providers.get(req.name)
        if provider is None:
            return CapabilityResponse(
                ok=False,
                data={},
                error=f"Capability not found: {req.name}",
            )
        try:
            return provider(req)
        except Exception as exc:  # defensive boundary
            return CapabilityResponse(ok=False, data={}, error=str(exc))


def create_default_registry() -> CapabilityRegistry:
    """Create a registry with a few built-in utility capabilities."""
    registry = CapabilityRegistry()

    def _cap_system_time(req: CapabilityRequest) -> CapabilityResponse:
        return CapabilityResponse(
            ok=True,
            data={"unix_time": time.time(), "session_id": req.session_id},
        )

    def _cap_echo(req: CapabilityRequest) -> CapabilityResponse:
        return CapabilityResponse(ok=True, data={"echo": req.payload})

    registry.register("system.time", _cap_system_time)
    registry.register("debug.echo", _cap_echo)
    return registry
