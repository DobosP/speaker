"""Bridge the Open Interpreter action brain into the capability registry.

The capability contract (utils/capabilities.py) is synchronous and returns a
single CapabilityResponse, but the action brain *streams* output over many
seconds. We resolve this by injecting callbacks: the provider speaks each phrase
as it arrives via ``speak_cb`` and returns ``data={"streamed": True}`` so the
caller (main._execute_route_decision) knows the turn was already voiced and must
not speak a second time.
"""
from __future__ import annotations

from typing import Callable, Optional

from utils.capabilities import CapabilityRequest, CapabilityResponse


def create_agent_provider(
    brain,
    speak_cb: Callable[[str], None],
    confirm_cb: Optional[Callable[[str, str], bool]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Callable[[CapabilityRequest], CapabilityResponse]:
    """Return a capability provider that runs an instruction on the action brain."""

    def provider(req: CapabilityRequest) -> CapabilityResponse:
        payload = req.payload if isinstance(req.payload, dict) else {}
        instruction = str(payload.get("instruction") or "").strip()
        if not instruction:
            return CapabilityResponse(ok=False, data={}, error="empty instruction")

        spoke = False
        last_error = ""
        stream = brain.stream_run(
            instruction,
            should_cancel=cancel_cb,
            on_confirm=confirm_cb,
        )
        try:
            for event in stream:
                if cancel_cb and cancel_cb():
                    break
                if event.kind in ("speak", "result"):
                    if event.text:
                        speak_cb(event.text)
                        spoke = True
                elif event.kind == "error":
                    last_error = event.text
        except Exception as exc:  # brain/OI failure (e.g. not installed)
            last_error = str(exc)
        finally:
            # Close the generator so agent_brain's stdin shim is always restored,
            # even when we break out early on cancellation.
            stream.close()

        if last_error and not spoke:
            speak_cb("Sorry, I couldn't complete that action.")
        return CapabilityResponse(
            ok=True,
            data={"streamed": True, "spoke": spoke, "error": last_error},
        )

    return provider
